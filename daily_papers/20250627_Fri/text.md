# 自然语言处理 cs.CL

- **最新发布 56 篇**

- **更新 31 篇**

## 最新发布

#### [new 001] Data Efficacy for Language Model Training
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在提升模型性能。通过优化数据组织，提出DELT框架，解决数据效率与效果问题。**

- **链接: [http://arxiv.org/pdf/2506.21545v1](http://arxiv.org/pdf/2506.21545v1)**

> **作者:** Yalun Dai; Yangyu Huang; Xin Zhang; Wenshan Wu; Chong Li; Wenhui Lu; Shijie Cao; Li Dong; Scarlett Li
>
> **摘要:** Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering. Among these components, we design Learnability-Quality Scoring (LQS), as a new instance of Data Scoring, which considers both the learnability and quality of each data sample from the gradient consistency perspective. We also devise Folding Ordering (FO), as a novel instance of Data Ordering, which addresses issues such as model forgetting and data distribution bias. Comprehensive experiments validate the data efficacy in LM training, which demonstrates the following: Firstly, various instances of the proposed DELT enhance LM performance to varying degrees without increasing the data scale and model size. Secondly, among these instances, the combination of our proposed LQS for data scoring and Folding for data ordering achieves the most significant improvement. Lastly, data efficacy can be achieved together with data efficiency by applying data selection. Therefore, we believe that data efficacy is a promising foundational area in LM training.
>
---
#### [new 002] Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI伦理研究任务，旨在检测LLMs在暴力情境下的潜在倾向。通过行为情景分析，发现模型在不同人口统计学特征下表现出差异化的暴力反应，揭示其与社会科学研究结果的矛盾。**

- **链接: [http://arxiv.org/pdf/2506.20822v1](http://arxiv.org/pdf/2506.20822v1)**

> **作者:** Quintin Myers; Yanjun Gao
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology.
>
---
#### [new 003] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态智能体任务，旨在解决奖励建模选择难题。提出Agent-RewardBench基准，评估感知、规划与安全能力。**

- **链接: [http://arxiv.org/pdf/2506.21252v1](http://arxiv.org/pdf/2506.21252v1)**

> **作者:** Tianyi Men; Zhuoran Jin; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** ACL 2025 Main
>
> **摘要:** As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github.
>
---
#### [new 004] Prompt-Guided Turn-Taking Prediction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于对话系统中的任务，解决turn-taking预测问题。通过引入文本提示控制预测行为，提升模型的灵活性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.21191v1](http://arxiv.org/pdf/2506.21191v1)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at SIGdial Meeting on Discourse and Dialogue 2025 (SIGDIAL 2025) and represents the author's version of the work
>
> **摘要:** Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
>
---
#### [new 005] KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20923v1](http://arxiv.org/pdf/2506.20923v1)**

> **作者:** Xinping Zhao; Xinshuo Hu; Zifei Shan; Shouzheng Huang; Yao Zhou; Zetian Sun; Zhenyu Liu; Dongfang Li; Xinyuan Wei; Qian Chen; Youcheng Pan; Yang Xiang; Meishan Zhang; Haofen Wang; Jun Yu; Baotian Hu; Min Zhang
>
> **备注:** Technical Report; 26 pages 12 tables 1 figure. arXiv admin note: substantial text overlap with arXiv:2501.01028
>
> **摘要:** In this paper, we propose KaLM-Embedding-V2, a versatile and compact embedding model, which achieves impressive performance in general-purpose text embedding tasks by leveraging superior training techniques and data. Our key innovations include: (1) To better align the architecture with representation learning, we remove the causal attention mask and adopt a fully bidirectional transformer with simple yet effective mean-pooling to produce fixed-length embeddings; (2) We employ a multi-stage training pipeline: (i) pre-training on large-scale weakly supervised open-source corpora; (ii) fine-tuning on high-quality retrieval and non-retrieval datasets; and (iii) model-soup parameter averaging for robust generalization. Besides, we introduce a focal-style reweighting mechanism that concentrates learning on difficult samples and an online hard-negative mixing strategy to continuously enrich hard negatives without expensive offline mining; (3) We collect over 20 categories of data for pre-training and 100 categories of data for fine-tuning, to boost both the performance and generalization of the embedding model. Extensive evaluations on the Massive Text Embedding Benchmark (MTEB) Chinese and English show that our model significantly outperforms others of comparable size, and competes with 3x, 14x, 18x, and 26x larger embedding models, setting a new standard for a versatile and compact embedding model with less than 1B parameters.
>
---
#### [new 006] Optimising Language Models for Downstream Tasks: A Post-Training Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在提升语言模型在下游任务中的适应性。针对数据利用不足、过拟合和计算成本高的问题，提出多种优化方法，包括知识提取、参数高效微调和新评估基准。**

- **链接: [http://arxiv.org/pdf/2506.20917v1](http://arxiv.org/pdf/2506.20917v1)**

> **作者:** Zhengyan Shi
>
> **备注:** PhD Thesis
>
> **摘要:** Language models (LMs) have demonstrated remarkable capabilities in NLP, yet adapting them efficiently and robustly to specific tasks remains challenging. As their scale and complexity grow, fine-tuning LMs on labelled data often underutilizes available unlabelled data, leads to overfitting on small task-specific sets, and imposes significant computational costs. These limitations hamper their application to the open-ended landscape of real-world language tasks. This thesis proposes a series of methods to better adapt LMs to downstream applications. First, we explore strategies for extracting task-relevant knowledge from unlabelled data, introducing a novel continued pre-training technique that outperforms state-of-the-art semi-supervised approaches. Next, we present a parameter-efficient fine-tuning method that substantially reduces memory and compute costs while maintaining competitive performance. We also introduce improved supervised fine-tuning methods that enable LMs to better follow instructions, especially when labelled data is scarce, enhancing their performance across a range of NLP tasks, including open-ended generation. Finally, we develop new evaluation methods and benchmarks, such as multi-hop spatial reasoning tasks, to assess LM capabilities and adaptation more comprehensively. Through extensive empirical studies across diverse NLP tasks, our results demonstrate that these approaches substantially improve LM robustness, efficiency, and generalization, making them more adaptable to a broad range of applications. These advances mark a significant step towards more robust and efficient LMs, bringing us closer to the goal of artificial general intelligence.
>
---
#### [new 007] Aligning Spoken Dialogue Models from User Interactions
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于对话系统任务，解决实时语音对话模型的偏好对齐问题。通过构建大规模数据集并微调模型，提升对话的准确性、安全性和上下文一致性。**

- **链接: [http://arxiv.org/pdf/2506.21463v1](http://arxiv.org/pdf/2506.21463v1)**

> **作者:** Anne Wu; Laurent Mazaré; Neil Zeghidour; Alexandre Défossez
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** We propose a novel preference alignment framework for improving spoken dialogue models on real-time conversations from user interactions. Current preference learning methods primarily focus on text-based language models, and are not directly suited to the complexities of real-time speech interactions, with richer dynamics (e.g. interruption, interjection) and no explicit segmentation between speaker turns.We create a large-scale dataset of more than 150,000 preference pairs from raw multi-turn speech conversations, annotated with AI feedback, to cover preferences over both linguistic content and temporal context variations. We leverage offline alignment methods to finetune a full-duplex autoregressive speech-to-speech model. Extensive experiments demonstrate that feedback on generic conversations can be consistently effective in improving spoken dialogue models to produce more factual, safer and more contextually aligned interactions. We deploy the finetuned model and conduct holistic human evaluations to assess the impact beyond single-turn conversations. Our findings shed light on the importance of a well-calibrated balance among various dynamics, crucial for natural real-time speech dialogue systems.
>
---
#### [new 008] Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索与生成任务，旨在解决实时RAG系统处理噪声、模糊多意图查询的问题。通过LLM辅助的查询理解框架提升系统鲁棒性与效果。**

- **链接: [http://arxiv.org/pdf/2506.21384v1](http://arxiv.org/pdf/2506.21384v1)**

> **作者:** Guanting Dong; Xiaoxi Li; Yuyao Zhang; Mengjie Deng
>
> **备注:** Accepted at SIGIR 2025 LiveRAG Workshop (Oral Presentation)
>
> **摘要:** Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries.
>
---
#### [new 009] TopK Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型可解释性任务，旨在解决稀疏自编码器（SAEs）的局限性。通过引入TopK激活函数，提升语言模型的可解释性与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.21468v1](http://arxiv.org/pdf/2506.21468v1)**

> **作者:** Ryosuke Takahashi; Tatsuro Inaba; Kentaro Inui; Benjamin Heinzerling
>
> **摘要:** Sparse autoencoders (SAEs) have become an important tool for analyzing and interpreting the activation space of transformer-based language models (LMs). However, SAEs suffer several shortcomings that diminish their utility and internal validity. Since SAEs are trained post-hoc, it is unclear if the failure to discover a particular concept is a failure on the SAE's side or due to the underlying LM not representing this concept. This problem is exacerbated by training conditions and architecture choices affecting which features an SAE learns. When tracing how LMs learn concepts during training, the lack of feature stability also makes it difficult to compare SAEs features across different checkpoints. To address these limitations, we introduce a modification to the transformer architecture that incorporates a TopK activation function at chosen layers, making the model's hidden states equivalent to the latent features of a TopK SAE. This approach eliminates the need for post-hoc training while providing interpretability comparable to SAEs. The resulting TopK LMs offer a favorable trade-off between model size, computational efficiency, and interpretability. Despite this simple architectural change, TopK LMs maintain their original capabilities while providing robust interpretability benefits. Our experiments demonstrate that the sparse representations learned by TopK LMs enable successful steering through targeted neuron interventions and facilitate detailed analysis of neuron formation processes across checkpoints and layers. These features make TopK LMs stable and reliable tools for understanding how language models learn and represent concepts, which we believe will significantly advance future research on model interpretability and controllability.
>
---
#### [new 010] Cat and Mouse -- Can Fake Text Generation Outpace Detector Systems?
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在研究虚假文本生成与检测之间的对抗关系。工作包括分析不同模型生成文本的欺骗性及检测效果。**

- **链接: [http://arxiv.org/pdf/2506.21274v1](http://arxiv.org/pdf/2506.21274v1)**

> **作者:** Andrea McGlinchey; Peter J Barclay
>
> **备注:** (Submitted for publication)
>
> **摘要:** Large language models can produce convincing "fake text" in domains such as academic writing, product reviews, and political news. Many approaches have been investigated for the detection of artificially generated text. While this may seem to presage an endless "arms race", we note that newer LLMs use ever more parameters, training data, and energy, while relatively simple classifiers demonstrate a good level of detection accuracy with modest resources. To approach the question of whether the models' ability to beat the detectors may therefore reach a plateau, we examine the ability of statistical classifiers to identify "fake text" in the style of classical detective fiction. Over a 0.5 version increase, we found that Gemini showed an increased ability to generate deceptive text, while GPT did not. This suggests that reliable detection of fake text may remain feasible even for ever-larger models, though new model architectures may improve their deceptiveness
>
---
#### [new 011] Large Language Models Acing Chartered Accountancy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融AI评估任务，旨在测试LLMs在会计领域的知识应用能力。通过构建CA-Ben基准，评估多个模型的表现，发现其在法律和概念推理上的优势及数值计算的不足。**

- **链接: [http://arxiv.org/pdf/2506.21031v1](http://arxiv.org/pdf/2506.21031v1)**

> **作者:** Jatin Gupta; Akhil Sharma; Saransh Singhania; Mohammad Adnan; Sakshi Deo; Ali Imam Abidi; Keshav Gupta
>
> **备注:** Accepted for publication at MoStart 2025: International Conference on Digital Transformation in Education and Applications of Artificial Intelligence, Bosnia and Herzegovina, 2025
>
> **摘要:** Advanced intelligent systems, particularly Large Language Models (LLMs), are significantly reshaping financial practices through advancements in Natural Language Processing (NLP). However, the extent to which these models effectively capture and apply domain-specific financial knowledge remains uncertain. Addressing a critical gap in the expansive Indian financial context, this paper introduces CA-Ben, a Chartered Accountancy benchmark specifically designed to evaluate the financial, legal, and quantitative reasoning capabilities of LLMs. CA-Ben comprises structured question-answer datasets derived from the rigorous examinations conducted by the Institute of Chartered Accountants of India (ICAI), spanning foundational, intermediate, and advanced CA curriculum stages. Six prominent LLMs i.e. GPT 4o, LLAMA 3.3 70B, LLAMA 3.1 405B, MISTRAL Large, Claude 3.5 Sonnet, and Microsoft Phi 4 were evaluated using standardized protocols. Results indicate variations in performance, with Claude 3.5 Sonnet and GPT-4o outperforming others, especially in conceptual and legal reasoning. Notable challenges emerged in numerical computations and legal interpretations. The findings emphasize the strengths and limitations of current LLMs, suggesting future improvements through hybrid reasoning and retrieval-augmented generation methods, particularly for quantitative analysis and accurate legal interpretation.
>
---
#### [new 012] Text2Cypher Across Languages: Evaluating Foundational Models Beyond English
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究多语言环境下Text2Cypher任务，评估基础模型在不同语言中的表现，旨在推动多语言查询生成的公平性与包容性。**

- **链接: [http://arxiv.org/pdf/2506.21445v1](http://arxiv.org/pdf/2506.21445v1)**

> **作者:** Makbule Gulcin Ozsoy; William Tai
>
> **摘要:** Recent advances in large language models have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses solely on English, with limited evaluation in other languages. This paper investigates the performance of foundational LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual test set by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. We evaluate multiple foundational models using standardized prompts and metrics. Our results show a consistent performance pattern: highest on English, then Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic characteristics. Additionally, we explore the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Our findings highlight the need for more inclusive evaluation and development in multilingual query generation. Future work includes schema localization and fine-tuning across diverse languages.
>
---
#### [new 013] Maintaining MTEB: Towards Long Term Usability and Reproducibility of Embedding Benchmarks
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于机器学习基准维护任务，解决嵌入模型评估的长期可用性与可复现性问题，通过工程实践提升MTEB的稳定性与扩展性。**

- **链接: [http://arxiv.org/pdf/2506.21182v1](http://arxiv.org/pdf/2506.21182v1)**

> **作者:** Isaac Chung; Imene Kerboua; Marton Kardos; Roman Solomatin; Kenneth Enevoldsen
>
> **摘要:** The Massive Text Embedding Benchmark (MTEB) has become a standard evaluation platform for text embedding models. While previous work has established the core benchmark methodology, this paper focuses on the engineering aspects that ensure MTEB's continued reproducibility and extensibility. We present our approach to maintaining robust continuous integration pipelines that validate dataset integrity, automate test execution, and assess benchmark results' generalizability. We detail the design choices that collectively enhance reproducibility and usability. Furthermore, we discuss our strategies for handling community contributions and extending the benchmark with new tasks and datasets. These engineering practices have been instrumental in scaling MTEB to become more comprehensive while maintaining quality and, ultimately, relevance to the field. Our experiences offer valuable insights for benchmark maintainers facing similar challenges in ensuring reproducibility and usability in machine learning evaluation frameworks. The MTEB repository is available at: https://github.com/embeddings-benchmark/mteb
>
---
#### [new 014] Potemkin Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨LLM在基准测试中的“虚假理解”问题。研究提出量化方法，揭示模型存在深层次概念不一致。**

- **链接: [http://arxiv.org/pdf/2506.21521v1](http://arxiv.org/pdf/2506.21521v1)**

> **作者:** Marina Mancoridis; Bec Weeks; Keyon Vafa; Sendhil Mullainathan
>
> **摘要:** Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations.
>
---
#### [new 015] Compressed and Smooth Latent Space for Text Diffusion Modeling
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决传统模型生成速度慢和连贯性差的问题。通过构建压缩平滑的潜在空间，提升扩散模型的效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.21170v1](http://arxiv.org/pdf/2506.21170v1)**

> **作者:** Viacheslav Meshchaninov; Egor Chimbulatov; Alexander Shabalin; Aleksandr Abramov; Dmitry Vetrov
>
> **摘要:** Autoregressive language models dominate modern text generation, yet their sequential nature introduces fundamental limitations: decoding is slow, and maintaining global coherence remains challenging. Diffusion models offer a promising alternative by enabling parallel generation and flexible control; however, their application to text generation is hindered by the high dimensionality of token-level representations. We introduce Cosmos, a novel approach to text generation that operates entirely in a compressed, smooth latent space tailored specifically for diffusion. This space is learned using an autoencoder trained simultaneously for token-level reconstruction and alignment with frozen activations from a pretrained language encoder, providing robust semantic grounding and enabling effective perturbation-based augmentations. Empirically, we demonstrate that text representations can be compressed by $8\times$ while maintaining generation quality comparable to token-level diffusion models. Furthermore, increasing the latent sequence length allows Cosmos to surpass both diffusion-based and autoregressive baselines. We evaluate Cosmos on four diverse generative tasks including story generation, question generation, summarization, and detoxification and compare it with various generative paradigms. Cosmos achieves comparable or superior generation quality while offering more than $2\times$ faster inference.
>
---
#### [new 016] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine
- **分类: cs.CL**

- **简介: 该论文属于医疗事实核查任务，旨在解决医学领域中自动核查系统难以有效应用的问题。研究通过分析临床专家验证社交媒体真实声明的过程，揭示了端到端核查的挑战。**

- **链接: [http://arxiv.org/pdf/2506.20876v1](http://arxiv.org/pdf/2506.20876v1)**

> **作者:** Sebastian Joseph; Lily Chen; Barry Wei; Michael Mackert; Iain J. Marshall; Paul Pu Liang; Ramez Kouzy; Byron C. Wallace; Junyi Jessy Li
>
> **摘要:** Technological progress has led to concrete advancements in tasks that were regarded as challenging, such as automatic fact-checking. Interest in adopting these systems for public health and medicine has grown due to the high-stakes nature of medical decisions and challenges in critically appraising a vast and diverse medical literature. Evidence-based medicine connects to every individual, and yet the nature of it is highly technical, rendering the medical literacy of majority users inadequate to sufficiently navigate the domain. Such problems with medical communication ripens the ground for end-to-end fact-checking agents: check a claim against current medical literature and return with an evidence-backed verdict. And yet, such systems remain largely unused. To understand this, we present the first study examining how clinical experts verify real claims from social media by synthesizing medical evidence. In searching for this upper-bound, we reveal fundamental challenges in end-to-end fact-checking when applied to medicine: Difficulties connecting claims in the wild to scientific evidence in the form of clinical trials; ambiguities in underspecified claims mixed with mismatched intentions; and inherently subjective veracity labels. We argue that fact-checking should be approached and evaluated as an interactive communication problem, rather than an end-to-end process.
>
---
#### [new 017] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning
- **分类: cs.CL**

- **简介: 该论文属于多模态句子表示学习任务，旨在解决跨模态对齐偏差和模态内语义分歧问题。提出DALR方法，通过一致性学习和排名蒸馏提升表示质量。**

- **链接: [http://arxiv.org/pdf/2506.21096v1](http://arxiv.org/pdf/2506.21096v1)**

> **作者:** Kang He; Yuzhe Ding. Haining Wang; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Previous multimodal sentence representation learning methods have achieved impressive performance. However, most approaches focus on aligning images and text at a coarse level, facing two critical challenges:cross-modal misalignment bias and intra-modal semantic divergence, which significantly degrade sentence representation quality. To address these challenges, we propose DALR (Dual-level Alignment Learning for Multimodal Sentence Representation). For cross-modal alignment, we propose a consistency learning module that softens negative samples and utilizes semantic similarity from an auxiliary task to achieve fine-grained cross-modal alignment. Additionally, we contend that sentence relationships go beyond binary positive-negative labels, exhibiting a more intricate ranking structure. To better capture these relationships and enhance representation quality, we integrate ranking distillation with global intra-modal alignment learning. Comprehensive experiments on semantic textual similarity (STS) and transfer (TR) tasks validate the effectiveness of our approach, consistently demonstrating its superiority over state-of-the-art baselines.
>
---
#### [new 018] A Semi-supervised Scalable Unified Framework for E-commerce Query Classification
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于电商查询分类任务，解决短文本信息不足和标签依赖问题。提出SSUF框架，融合知识、标签和结构增强模块，提升分类效果。**

- **链接: [http://arxiv.org/pdf/2506.21049v1](http://arxiv.org/pdf/2506.21049v1)**

> **作者:** Chunyuan Yuan; Chong Zhang; Zheng Fang; Ming Pang; Xue Jiang; Changping Peng; Zhangang Lin; Ching Law
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Query classification, including multiple subtasks such as intent and category prediction, is vital to e-commerce applications. E-commerce queries are usually short and lack context, and the information between labels cannot be used, resulting in insufficient prior information for modeling. Most existing industrial query classification methods rely on users' posterior click behavior to construct training samples, resulting in a Matthew vicious cycle. Furthermore, the subtasks of query classification lack a unified framework, leading to low efficiency for algorithm optimization. In this paper, we propose a novel Semi-supervised Scalable Unified Framework (SSUF), containing multiple enhanced modules to unify the query classification tasks. The knowledge-enhanced module uses world knowledge to enhance query representations and solve the problem of insufficient query information. The label-enhanced module uses label semantics and semi-supervised signals to reduce the dependence on posterior labels. The structure-enhanced module enhances the label representation based on the complex label relations. Each module is highly pluggable, and input features can be added or removed as needed according to each subtask. We conduct extensive offline and online A/B experiments, and the results show that SSUF significantly outperforms the state-of-the-art models.
>
---
#### [new 019] Can Gradient Descent Simulate Prompting?
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决如何让微调模拟提示的问题。通过元训练使梯度更新模仿提示效果，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.20989v1](http://arxiv.org/pdf/2506.20989v1)**

> **作者:** Eric Zhang; Leshem Choshen; Jacob Andreas
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** There are two primary ways of incorporating new information into a language model (LM): changing its prompt or changing its parameters, e.g. via fine-tuning. Parameter updates incur no long-term storage cost for model changes. However, for many model updates, prompting is significantly more effective: prompted models can generalize robustly from single examples and draw logical inferences that do not occur under standard fine-tuning. Can models be modified so that fine-tuning does emulate prompting? This paper describes a method for meta-training LMs such that gradient updates emulate the effects of conditioning on new information. Our approach uses tools from gradient-based meta-learning but uses an LM's own prompted predictions as targets, eliminating the need for ground-truth labels. Subsequent gradient descent training recovers some (and occasionally all) of prompted model performance -- showing improvement on the ``reversal curse'' tasks, and answering questions about text passages after a single gradient update. These results suggest that, with appropriate initialization, gradient descent can be surprisingly expressive. Our results suggest new avenues for long-context modeling and offer insight into the generalization capabilities of gradient-based learning.
>
---
#### [new 020] The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文属于AI与科研结合的任务，旨在解决LLM生成研究想法是否优于人类的问题。通过实验对比执行结果，发现LLM想法在执行后表现较差，揭示了其有效性不足。**

- **链接: [http://arxiv.org/pdf/2506.20803v1](http://arxiv.org/pdf/2506.20803v1)**

> **作者:** Chenglei Si; Tatsunori Hashimoto; Diyi Yang
>
> **备注:** main paper is 14 pages
>
> **摘要:** Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes.
>
---
#### [new 021] skLEP: A Slovak General Language Understanding Benchmark
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; 68T50; I.2.7**

- **简介: 该论文提出skLEP，首个针对斯洛伐克语自然语言理解的基准，涵盖九项任务，旨在评估和推动斯洛伐克语NLU研究。**

- **链接: [http://arxiv.org/pdf/2506.21508v1](http://arxiv.org/pdf/2506.21508v1)**

> **作者:** Marek Šuppa; Andrej Ridzik; Daniel Hládek; Tomáš Javůrek; Viktória Ondrejová; Kristína Sásiková; Martin Tamajka; Marián Šimko
>
> **备注:** ACL 2025 Findings
>
> **摘要:** In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at https://github.com/slovak-nlp/sklep in the hopes of fostering reproducibility and drive future research in Slovak NLU.
>
---
#### [new 022] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于视觉对话中的指代表达检测任务，旨在仅通过语言上下文识别对话中具有视觉参照的提及项。工作包括使用预训练语言模型进行文本标注和分析。**

- **链接: [http://arxiv.org/pdf/2506.21294v1](http://arxiv.org/pdf/2506.21294v1)**

> **作者:** Bram Willemsen; Gabriel Skantze
>
> **备注:** Accepted for publication at XLLM @ ACL 2025
>
> **摘要:** In this paper, we explore the use of a text-only, autoregressive language modeling approach for the extraction of referring expressions from visually grounded dialogue. More specifically, the aim is to investigate the extent to which the linguistic context alone can inform the detection of mentions that have a (visually perceivable) referent in the visual context of the conversation. To this end, we adapt a pretrained large language model (LLM) to perform a relatively course-grained annotation of mention spans in unfolding conversations by demarcating mention span boundaries in text via next-token prediction. Our findings indicate that even when using a moderately sized LLM, relatively small datasets, and parameter-efficient fine-tuning, a text-only approach can be effective, highlighting the relative importance of the linguistic context for this task. Nevertheless, we argue that the task represents an inherently multimodal problem and discuss limitations fundamental to unimodal approaches.
>
---
#### [new 023] Progtuning: Progressive Fine-tuning Framework for Transformer-based Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型微调效率低的问题。通过提出Progtuning框架，按贡献逐步减少更新的Transformer块数，提升资源利用率并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.21119v1](http://arxiv.org/pdf/2506.21119v1)**

> **作者:** Xiaoshuang Ji; Zhendong Zhao; Xiaojun Chen; Xin Zhao; Zeyao Liu
>
> **备注:** Accepted by ICONIP 2024
>
> **摘要:** Fine-tuning is a promising technique for leveraging Transformer-based language models in downstream tasks. As model sizes continue to grow, updating all model parameters becomes increasingly costly. Parameter-efficient fine-tuning methods effectively address this issue by selectively updating a small subset of parameters. However, fine-tuning and most existing parameter-efficient fine-tuning methods require updating the same number of parameters as the initial size, ignoring the unequal contribution across Transformer blocks and leading to extremely inefficient allocation of computing resources. In this paper, we propose Progtuning, the novel fine-tuning framework combined with progressive learning for Transformer-based language models. Specifically, Progtuning progressively reduces the number of updated transformer blocks based on the contribution. Remarkably, Progtuning optimizes resource allocation and reduces the number of updated parameters by approximately 25\%, while still maintaining competitive performance. And it also exhibits high adaptability with parameter-efficient fine-tuning methods, demonstrating excellent performance across various adaptation scenarios.
>
---
#### [new 024] Multi-lingual Functional Evaluation for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言模型评估任务，旨在解决静态基准无法准确反映模型跨语言性能的问题。通过构建多语言功能基准，比较不同模型在多种语言中的表现与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.20793v1](http://arxiv.org/pdf/2506.20793v1)**

> **作者:** Victor Ojewale; Inioluwa Deborah Raji; Suresh Venkatasubramanian
>
> **摘要:** Multi-lingual competence in large language models is often evaluated via static data benchmarks such as Belebele, M-MMLU and M-GSM. However, these evaluations often fail to provide an adequate understanding of the practical performance and robustness of models across multi-lingual settings. In response, we create multi-lingual functional benchmarks -- Cross-Lingual Grade School Math Symbolic (CL-GSM Symbolic) and Cross-Lingual Instruction-Following Eval (CL-IFEval)-- by translating existing functional benchmark templates from English to five additional languages that span the range of resources available for NLP: French, Spanish, Hindi, Arabic and Yoruba. Our results reveal that some static multi-lingual benchmarks capture functional performance much more closely than others (i.e. across models, there is a 24%, 17% and 18% decrease in performance between M-GSM and CL-GSM Symbolic in English, French and Spanish respectively; similarly there's a 15 - 24% performance drop across languages between Belebele and CL-IFEval, and only a 0.5% to 3% performance drop between M-MMLU and CL-IFEval). Similarly, we find that model robustness across languages varies significantly, with certain languages (eg. Arabic, English) being the most consistently well performing across evaluation iterations.
>
---
#### [new 025] Small Encoders Can Rival Large Decoders in Detecting Groundedness
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决模型在无足够上下文时产生不实回答的问题。通过使用轻量级编码器模型检测回答是否基于给定文档，提升准确性并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2506.21288v1](http://arxiv.org/pdf/2506.21288v1)**

> **作者:** Istabrak Abbes; Gabriele Prato; Quentin Fournier; Fernando Rodriguez; Alaa Boukhary; Adam Elwood; Sarath Chandar
>
> **摘要:** Augmenting large language models (LLMs) with external context significantly improves their performance in natural language processing (NLP) tasks. However, LLMs struggle to answer queries reliably when the provided context lacks information, often resorting to ungrounded speculation or internal knowledge. Groundedness - generating responses strictly supported by the context - is essential for ensuring factual consistency and trustworthiness. This study focuses on detecting whether a given query is grounded in a document provided in context before the costly answer generation by LLMs. Such a detection mechanism can significantly reduce both inference time and resource consumption. We show that lightweight, task specific encoder models such as RoBERTa and NomicBERT, fine-tuned on curated datasets, can achieve accuracy comparable to state-of-the-art LLMs, such as Llama3 8B and GPT4o, in groundedness detection while reducing inference latency by orders of magnitude. The code is available at : https://github.com/chandarlab/Hallucinate-less
>
---
#### [new 026] Towards Probabilistic Question Answering Over Tabular Data
- **分类: cs.CL; 68T50, 68T37; I.2.7**

- **简介: 该论文属于表格数据上的概率问答任务，旨在解决传统方法在处理不确定性推理时的不足。通过构建贝叶斯网络和结合大语言模型，提升问答准确性。**

- **链接: [http://arxiv.org/pdf/2506.20747v1](http://arxiv.org/pdf/2506.20747v1)**

> **作者:** Chen Shen; Sajjadur Rahman; Estevam Hruschka
>
> **摘要:** Current approaches for question answering (QA) over tabular data, such as NL2SQL systems, perform well for factual questions where answers are directly retrieved from tables. However, they fall short on probabilistic questions requiring reasoning under uncertainty. In this paper, we introduce a new benchmark LUCARIO and a framework for probabilistic QA over large tabular data. Our method induces Bayesian Networks from tables, translates natural language queries into probabilistic queries, and uses large language models (LLMs) to generate final answers. Empirical results demonstrate significant improvements over baselines, highlighting the benefits of hybrid symbolic-neural reasoning.
>
---
#### [new 027] Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于欺诈检测与概念漂移检测任务，旨在解决动态平台中语言模式变化带来的识别难题。通过融合领域知识的LLM框架，提升检测准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.21443v1](http://arxiv.org/pdf/2506.21443v1)**

> **作者:** Ali Şenol; Garima Agrawal; Huan Liu
>
> **摘要:** Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications.
>
---
#### [new 028] MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering
- **分类: cs.CL; cs.AI; cs.CE; 68T50, 68T07 (Primary) 68P20, 91G15, 91G70, 68U10 (Secondary); I.2.7; I.2.10; H.3.3; H.2.8; I.5.4; J.1**

- **简介: 该论文属于金融问答任务，解决多模态财务文档理解与跨模态推理问题。提出MultiFinRAG框架，提升复杂财务问答准确率。**

- **链接: [http://arxiv.org/pdf/2506.20821v1](http://arxiv.org/pdf/2506.20821v1)**

> **作者:** Chinmay Gondhalekar; Urjitkumar Patel; Fang-Chun Yeh
>
> **备注:** Preprint Copy
>
> **摘要:** Financial documents--such as 10-Ks, 10-Qs, and investor presentations--span hundreds of pages and combine diverse modalities, including dense narrative text, structured tables, and complex figures. Answering questions over such content often requires joint reasoning across modalities, which strains traditional large language models (LLMs) and retrieval-augmented generation (RAG) pipelines due to token limitations, layout loss, and fragmented cross-modal context. We introduce MultiFinRAG, a retrieval-augmented generation framework purpose-built for financial QA. MultiFinRAG first performs multimodal extraction by grouping table and figure images into batches and sending them to a lightweight, quantized open-source multimodal LLM, which produces both structured JSON outputs and concise textual summaries. These outputs, along with narrative text, are embedded and indexed with modality-aware similarity thresholds for precise retrieval. A tiered fallback strategy then dynamically escalates from text-only to text+table+image contexts when necessary, enabling cross-modal reasoning while reducing irrelevant context. Despite running on commodity hardware, MultiFinRAG achieves 19 percentage points higher accuracy than ChatGPT-4o (free-tier) on complex financial QA tasks involving text, tables, images, and combined multimodal reasoning.
>
---
#### [new 029] Bridging Offline and Online Reinforcement Learning for LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在离线到在线强化学习中的微调方法，解决任务性能提升问题，通过对比不同优化目标和多任务学习策略实现更好效果。**

- **链接: [http://arxiv.org/pdf/2506.21495v1](http://arxiv.org/pdf/2506.21495v1)**

> **作者:** Jack Lanchantin; Angelica Chen; Janice Lan; Xian Li; Swarnadeep Saha; Tianlu Wang; Jing Xu; Ping Yu; Weizhe Yuan; Jason E Weston; Sainbayar Sukhbaatar; Ilia Kulikov
>
> **摘要:** We investigate the effectiveness of reinforcement learning methods for finetuning large language models when transitioning from offline to semi-online to fully online regimes for both verifiable and non-verifiable tasks. Our experiments cover training on verifiable math as well as non-verifiable instruction following with a set of benchmark evaluations for both. Across these settings, we extensively compare online and semi-online Direct Preference Optimization and Group Reward Policy Optimization objectives, and surprisingly find similar performance and convergence between these variants, which all strongly outperform offline methods. We provide a detailed analysis of the training dynamics and hyperparameter selection strategies to achieve optimal results. Finally, we show that multi-tasking with verifiable and non-verifiable rewards jointly yields improved performance across both task types.
>
---
#### [new 030] MT2-CSD: A New Dataset and Multi-Semantic Knowledge Fusion Method for Conversational Stance Detection
- **分类: cs.CL**

- **简介: 该论文属于对话立场检测任务，旨在解决传统方法难以处理多轮多主体讨论的问题。提出了MT2-CSD数据集和LLM-CRAN模型以提升对话理解与立场识别效果。**

- **链接: [http://arxiv.org/pdf/2506.21053v1](http://arxiv.org/pdf/2506.21053v1)**

> **作者:** Fuqiang Niu; Genan Dai; Yisha Lu; Jiayu Liao; Xiang Li; Hu Huang; Bowen Zhang
>
> **摘要:** In the realm of contemporary social media, automatic stance detection is pivotal for opinion mining, as it synthesizes and examines user perspectives on contentious topics to uncover prevailing trends and sentiments. Traditional stance detection research often targets individual instances, thereby limiting its capacity to model multi-party discussions typical in real social media scenarios. This shortcoming largely stems from the scarcity of datasets that authentically capture the dynamics of social media interactions, hindering advancements in conversational stance detection. In this paper, we introduce MT2-CSD, a comprehensive dataset for multi-target, multi-turn conversational stance detection. To the best of our knowledge, MT2-CSD is the largest dataset available for this purpose, comprising 24,457 annotated instances and exhibiting the greatest conversational depth, thereby presenting new challenges for stance detection. To address these challenges, we propose the Large Language model enhanced Conversational Relational Attention Network (LLM-CRAN), which exploits the reasoning capabilities of LLMs to improve conversational understanding. We conduct extensive experiments to evaluate the efficacy of LLM-CRAN on the MT2-CSD dataset. The experimental results indicate that LLM-CRAN significantly outperforms strong baseline models in the task of conversational stance detection.
>
---
#### [new 031] Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自动术语提取任务，旨在提升大语言模型在术语识别中的表现。通过基于句法的检索策略，增强模型对术语边界的捕捉能力。**

- **链接: [http://arxiv.org/pdf/2506.21222v1](http://arxiv.org/pdf/2506.21222v1)**

> **作者:** Yongchan Chun; Minhyuk Kim; Dongjun Kim; Chanjun Park; Heuiseok Lim
>
> **摘要:** Automatic Term Extraction (ATE) identifies domain-specific expressions that are crucial for downstream tasks such as machine translation and information retrieval. Although large language models (LLMs) have significantly advanced various NLP tasks, their potential for ATE has scarcely been examined. We propose a retrieval-based prompting strategy that, in the few-shot setting, selects demonstrations according to \emph{syntactic} rather than semantic similarity. This syntactic retrieval method is domain-agnostic and provides more reliable guidance for capturing term boundaries. We evaluate the approach in both in-domain and cross-domain settings, analyzing how lexical overlap between the query sentence and its retrieved examples affects performance. Experiments on three specialized ATE benchmarks show that syntactic retrieval improves F1-score. These findings highlight the importance of syntactic cues when adapting LLMs to terminology-extraction tasks.
>
---
#### [new 032] Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升慢思考大模型的推理能力。通过自批判微调，增强模型自我批评与迭代优化能力，显著提高了推理准确性。**

- **链接: [http://arxiv.org/pdf/2506.21285v1](http://arxiv.org/pdf/2506.21285v1)**

> **作者:** Xin Xu; Tianhao Chen; Fan Zhang; Wanlong Liu; Pengxiang Li; Ajay Kumar Jaiswal; Yuchen Yan; Jishan Hu; Yang Wang; Hao Chen; Shiwei Liu; Shizhe Diao; Can Yang; Lu Yin
>
> **备注:** 10 pages
>
> **摘要:** While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique.
>
---
#### [new 033] FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型训练数据不足的问题。通过设计可自动适配多种语言的预训练数据处理管道，提升非英语模型性能。**

- **链接: [http://arxiv.org/pdf/2506.20920v1](http://arxiv.org/pdf/2506.20920v1)**

> **作者:** Guilherme Penedo; Hynek Kydlíček; Vinko Sabolčec; Bettina Messmer; Negar Foroutan; Amir Hossein Kargaran; Colin Raffel; Martin Jaggi; Leandro Von Werra; Thomas Wolf
>
> **摘要:** Pre-training state-of-the-art large language models (LLMs) requires vast amounts of clean and diverse text data. While the open development of large high-quality English pre-training datasets has seen substantial recent progress, training performant multilingual LLMs remains a challenge, in large part due to the inherent difficulty of tailoring filtering and deduplication pipelines to a large number of languages. In this work, we introduce a new pre-training dataset curation pipeline based on FineWeb that can be automatically adapted to support any language. We extensively ablate our pipeline design choices on a set of nine diverse languages, guided by a set of meaningful and informative evaluation tasks that were chosen through a novel selection process based on measurable criteria. Ultimately, we show that our pipeline can be used to create non-English corpora that produce more performant models than prior datasets. We additionally introduce a straightforward and principled approach to rebalance datasets that takes into consideration both duplication count and quality, providing an additional performance uplift. Finally, we scale our pipeline to over 1000 languages using almost 100 Common Crawl snapshots to produce FineWeb2, a new 20 terabyte (5 billion document) multilingual dataset which we release along with our pipeline, training, and evaluation codebases.
>
---
#### [new 034] Structuralist Approach to AI Literary Criticism: Leveraging Greimas Semiotic Square for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文学批评任务，旨在解决LLMs在深度文学分析上的不足。提出GLASS框架，结合Greimas语义方阵，提升AI的文学分析能力。**

- **链接: [http://arxiv.org/pdf/2506.21360v1](http://arxiv.org/pdf/2506.21360v1)**

> **作者:** Fangzhou Dong; Yifan Zeng; Yingpeng Sang; Hong Shen
>
> **备注:** Accepted in CogSci 2025
>
> **摘要:** Large Language Models (LLMs) excel in understanding and generating text but struggle with providing professional literary criticism for works with profound thoughts and complex narratives. This paper proposes GLASS (Greimas Literary Analysis via Semiotic Square), a structured analytical framework based on Greimas Semiotic Square (GSS), to enhance LLMs' ability to conduct in-depth literary analysis. GLASS facilitates the rapid dissection of narrative structures and deep meanings in narrative works. We propose the first dataset for GSS-based literary criticism, featuring detailed analyses of 48 works. Then we propose quantitative metrics for GSS-based literary criticism using the LLM-as-a-judge paradigm. Our framework's results, compared with expert criticism across multiple works and LLMs, show high performance. Finally, we applied GLASS to 39 classic works, producing original and high-quality analyses that address existing research gaps. This research provides an AI-based tool for literary research and education, offering insights into the cognitive mechanisms underlying literary engagement.
>
---
#### [new 035] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工业社区问答任务，旨在解决实时问答中知识利用不足和动态上下文处理的问题。提出ComRAG框架，结合静态知识与动态历史问答对，提升回答质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.21098v1](http://arxiv.org/pdf/2506.21098v1)**

> **作者:** Qinwen Chen; Wenbiao Tao; Zhiwei Zhu; Mingfan Xi; Liangzhong Guo; Yuan Wang; Wei Wang; Yunshi Lan
>
> **备注:** 7 pages, 4 figures. Accepted at ACL 2025 Industry Track
>
> **摘要:** Community Question Answering (CQA) platforms can be deemed as important knowledge bases in community, but effectively leveraging historical interactions and domain knowledge in real-time remains a challenge. Existing methods often underutilize external knowledge, fail to incorporate dynamic historical QA context, or lack memory mechanisms suited for industrial deployment. We propose ComRAG, a retrieval-augmented generation framework for real-time industrial CQA that integrates static knowledge with dynamic historical QA pairs via a centroid-based memory mechanism designed for retrieval, generation, and efficient storage. Evaluated on three industrial CQA datasets, ComRAG consistently outperforms all baselines--achieving up to 25.9% improvement in vector similarity, reducing latency by 8.7% to 23.3%, and lowering chunk growth from 20.23% to 2.06% over iterations.
>
---
#### [new 036] SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理中的个性化建模任务，旨在解决LLMs人格特质表达不精确、无法控制强度的问题。通过扩展MPI和提出SAC框架，实现对16种人格特质的动态控制。**

- **链接: [http://arxiv.org/pdf/2506.20993v1](http://arxiv.org/pdf/2506.20993v1)**

> **作者:** Adithya Chittem; Aishna Shrivastava; Sai Tarun Pendela; Jagat Sesh Challa; Dhruv Kumar
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines.
>
---
#### [new 037] Enhancing User Engagement in Socially-Driven Dialogue through Interactive LLM Alignments
- **分类: cs.CL**

- **简介: 该论文属于社交对话任务，旨在提升用户参与度。通过交互式大模型对齐，利用未来对话信号作为奖励，优化用户互动体验。**

- **链接: [http://arxiv.org/pdf/2506.21497v1](http://arxiv.org/pdf/2506.21497v1)**

> **作者:** Jiashuo Wang; Kaitao Song; Chunpu Xu; Changhe Song; Yang Xiao; Dongsheng Li; Lili Qiu; Wenjie Li
>
> **摘要:** Enhancing user engagement through interactions plays an essential role in socially-driven dialogues. While prior works have optimized models to reason over relevant knowledge or plan a dialogue act flow, the relationship between user engagement and knowledge or dialogue acts is subtle and does not guarantee user engagement in socially-driven dialogues. To this end, we enable interactive LLMs to learn user engagement by leveraging signals from the future development of conversations. Specifically, we adopt a more direct and relevant indicator of user engagement, i.e., the user's reaction related to dialogue intention after the interaction, as a reward to align interactive LLMs. To achieve this, we develop a user simulator to interact with target interactive LLMs and explore interactions between the user and the interactive LLM system via \textit{i$\times$MCTS} (\textit{M}onte \textit{C}arlo \textit{T}ree \textit{S}earch for \textit{i}nteraction). In this way, we collect a dataset containing pairs of higher and lower-quality experiences using \textit{i$\times$MCTS}, and align interactive LLMs for high-level user engagement by direct preference optimization (DPO) accordingly. Experiments conducted on two socially-driven dialogue scenarios (emotional support conversations and persuasion for good) demonstrate that our method effectively enhances user engagement in interactive LLMs.
>
---
#### [new 038] "What's Up, Doc?": Analyzing How Users Seek Health Information in Large-Scale Conversational AI Datasets
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于医疗对话分析任务，旨在研究用户通过聊天机器人获取健康信息的行为，识别交互中的问题并提出改进方向。**

- **链接: [http://arxiv.org/pdf/2506.21532v1](http://arxiv.org/pdf/2506.21532v1)**

> **作者:** Akshay Paruchuri; Maryam Aziz; Rohit Vartak; Ayman Ali; Best Uchehara; Xin Liu; Ishan Chatterjee; Monica Agrawal
>
> **备注:** 25 pages, 6 figures, 4 tables, corresponds to initial HealthChat-11K dataset release
>
> **摘要:** People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: https://github.com/yahskapar/HealthChat
>
---
#### [new 039] HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言分割任务，旨在解决模型在分割时产生的幻觉问题。通过构建基准测试集和新指标，评估模型在视觉上下文变化下的幻觉敏感性。**

- **链接: [http://arxiv.org/pdf/2506.21546v1](http://arxiv.org/pdf/2506.21546v1)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Ismini Lourentzou
>
> **备注:** Project webpage: https://plan-lab.github.io/hallusegbench/
>
> **摘要:** Recent progress in vision-language segmentation has significantly advanced grounded visual understanding. However, these models often exhibit hallucinations by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Existing evaluation protocols for segmentation hallucination primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucinations in visual grounding through the lens of counterfactual visual reasoning. Our benchmark consists of a novel dataset of 1340 counterfactual instance pairs spanning 281 unique object classes, and a set of newly introduced metrics that quantify hallucination sensitivity under visually coherent scene edits. Experiments on HalluSegBench with state-of-the-art vision-language segmentation models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the need for counterfactual reasoning to diagnose grounding fidelity.
>
---
#### [new 040] Spatial Mental Modeling from Limited Views
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究视觉语言模型在有限视角下构建空间心理模型的能力，旨在提升其对不可见空间的理解。通过提出MindCube基准和多种方法改进模型表现。**

- **链接: [http://arxiv.org/pdf/2506.21458v1](http://arxiv.org/pdf/2506.21458v1)**

> **作者:** Baiqiao Yin; Qineng Wang; Pingyue Zhang; Jianshu Zhang; Kangrui Wang; Zihan Wang; Jieyu Zhang; Keshigeyan Chandrasegaran; Han Liu; Ranjay Krishna; Saining Xie; Manling Li; Jiajun Wu; Li Fei-Fei
>
> **备注:** Preprint version
>
> **摘要:** Can Vision Language Models (VLMs) imagine the full scene from just a few views, like humans do? Humans form spatial mental models, internal representations of unseen space, to reason about layout, perspective, and motion. Our new MindCube benchmark with 21,154 questions across 3,268 images exposes this critical gap, where existing VLMs exhibit near-random performance. Using MindCube, we systematically evaluate how well VLMs build robust spatial mental models through representing positions (cognitive mapping), orientations (perspective-taking), and dynamics (mental simulation for "what-if" movements). We then explore three approaches to help VLMs approximate spatial mental models, including unseen intermediate views, natural language reasoning chains, and cognitive maps. The significant improvement comes from a synergistic approach, "map-then-reason", that jointly trains the model to first generate a cognitive map and then reason upon it. By training models to reason over these internal maps, we boosted accuracy from 37.8% to 60.8% (+23.0%). Adding reinforcement learning pushed performance even further to 70.7% (+32.9%). Our key insight is that such scaffolding of spatial mental models, actively constructing and utilizing internal structured spatial representations with flexible reasoning processes, significantly improves understanding of unobservable space.
>
---
#### [new 041] Complexity-aware fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升小模型在特定领域的性能。通过识别复杂数据并进行针对性微调，提高效率并减少数据需求。**

- **链接: [http://arxiv.org/pdf/2506.21220v1](http://arxiv.org/pdf/2506.21220v1)**

> **作者:** Andrey Goncharov; Daniil Vyazhev; Petr Sychev; Edvard Khalafyan; Alexey Zaytsev
>
> **摘要:** General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data. We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across two small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.55$ vs $0.43$ average accuracy) and provides comparable with distillation performance while using $62\%$ less data ($0.55$ average accuracy for both). We publish our code and data to facilitate further research in this direction.
>
---
#### [new 042] Leaner Training, Lower Leakage: Revisiting Memorization in LLM Fine-Tuning with LoRA
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于自然语言处理领域，研究LoRA微调中的记忆问题，旨在降低数据泄露风险。工作包括重新评估记忆机制，发现LoRA在保持性能的同时显著减少记忆风险。**

- **链接: [http://arxiv.org/pdf/2506.20856v1](http://arxiv.org/pdf/2506.20856v1)**

> **作者:** Fei Wang; Baochun Li
>
> **摘要:** Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method. In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance.
>
---
#### [new 043] Latent Prototype Routing: Achieving Near-Perfect Load Balancing in Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决MoE架构中的负载不平衡问题。通过提出LPR框架，实现专家利用率的均衡化，提升计算资源使用效率。**

- **链接: [http://arxiv.org/pdf/2506.21328v1](http://arxiv.org/pdf/2506.21328v1)**

> **作者:** Jiajie Yang
>
> **备注:** 15 pages,4 figures
>
> **摘要:** Mixture-of-Experts (MoE) architectures have emerged as a key strategy for scaling large language models (LLMs) efficiently. However, current MoE systems suffer from severe load imbalance, where only a small subset of experts is consistently activated during training and inference, leading to significant underutilization of model capacity and computational resources. In this work, we revisit expert routing through a clustering perspective and propose Latent Prototype Routing (LPR), a novel routing framework that generalizes existing approaches while promoting balanced expert utilization without compromising downstream performance. Extensive experiments across multiple open-source MoE models -- including DeepSeek-V3, Qwen3-MoE, and Mixtral -- demonstrate that LPR reduces the Gini coefficient of expert load from 0.70 to 0.035 on average, improves the min-max expert load ratio from 1e-6 to 0.70, achieving near-perfect load balancing.
>
---
#### [new 044] Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的因果推理任务，旨在解决LLMs是否具备真实因果推理能力的问题。研究发现LLMs仅能进行浅层因果推理，提出G²-Reasoner方法提升其因果推理能力。**

- **链接: [http://arxiv.org/pdf/2506.21215v1](http://arxiv.org/pdf/2506.21215v1)**

> **作者:** Haoang Chi; He Li; Wenjing Yang; Feng Liu; Long Lan; Xiaoguang Ren; Tongliang Liu; Bo Han
>
> **备注:** 24 pages, accepted at NeurIPS 2024
>
> **摘要:** Causal reasoning capability is critical in advancing large language models (LLMs) toward strong artificial intelligence. While versatile LLMs appear to have demonstrated capabilities in understanding contextual causality and providing responses that obey the laws of causality, it remains unclear whether they perform genuine causal reasoning akin to humans. However, current evidence indicates the contrary. Specifically, LLMs are only capable of performing shallow (level-1) causal reasoning, primarily attributed to the causal knowledge embedded in their parameters, but they lack the capacity for genuine human-like (level-2) causal reasoning. To support this hypothesis, methodologically, we delve into the autoregression mechanism of transformer-based LLMs, revealing that it is not inherently causal. Empirically, we introduce a new causal Q&A benchmark called CausalProbe-2024, whose corpora are fresh and nearly unseen for the studied LLMs. The LLMs exhibit a significant performance drop on CausalProbe-2024 compared to earlier benchmarks, indicating the fact that they primarily engage in level-1 causal reasoning. To bridge the gap towards level-2 causal reasoning, we draw inspiration from the fact that human reasoning is usually facilitated by general knowledge and intended goals. We propose G^2-Reasoner, a method that incorporates general knowledge and goal-oriented prompts into LLMs' causal reasoning processes. Experiments demonstrate that G^2-Reasoner significantly enhances LLMs' causal reasoning capability, particularly in fresh and counterfactual contexts. This work sheds light on a new path for LLMs to advance towards genuine causal reasoning, going beyond level-1 and making strides towards level-2.
>
---
#### [new 045] Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升LLM工具使用能力。通过知识图谱生成高质量指令数据，解决传统方法数据质量不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.21071v1](http://arxiv.org/pdf/2506.21071v1)**

> **作者:** Jingwei Wang; Zai Zhang; Hao Qian; Chunjing Gan; Binbin Hu; Ziqi Liu; Zhiqiang Zhang; Jun Zhou; Bin Shi; Bo Dong
>
> **备注:** 20 pages, 12 figures
>
> **摘要:** Teaching large language models (LLMs) to use tools is crucial for improving their problem-solving abilities and expanding their applications. However, effectively using tools is challenging because it requires a deep understanding of tool functionalities and user intentions. Previous methods relied mainly on LLMs to generate instruction data, but the quality of these data was often insufficient. In this paper, we propose a new method that uses knowledge graphs to generate high-quality instruction data for LLMs. Knowledge graphs are manually curated datasets rich in semantic information. We begin by extracting various query pathways from a given knowledge graph, which are transformed into a broad spectrum of user queries. We then translate the relationships between entities into actionable tools and parse the pathways of each query into detailed solution steps, thereby creating high-quality instruction data. Our experiments show that fine-tuning on just a small sample of this synthetic data can significantly improve the tool utilization and overall capabilities of LLMs.
>
---
#### [new 046] Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon Simulation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型安全任务，旨在提升模型对长期风险的感知能力。通过模拟和测试，增强模型在高风险决策中的安全性与可靠性。**

- **链接: [http://arxiv.org/pdf/2506.20949v1](http://arxiv.org/pdf/2506.20949v1)**

> **作者:** Chenkai Sun; Denghui Zhang; ChengXiang Zhai; Heng Ji
>
> **摘要:** Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents.
>
---
#### [new 047] DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于分布式训练任务，解决大规模模型在慢网络下的训练问题。提出DiLoCoX框架，结合多种技术实现低通信高效训练。**

- **链接: [http://arxiv.org/pdf/2506.21263v1](http://arxiv.org/pdf/2506.21263v1)**

> **作者:** Ji Qi; WenPeng Zhu; Li Li; Ming Wu; YingJun Wu; Wu He; Xun Gao; Jason Zeng; Michael Heinrich
>
> **摘要:** The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters.
>
---
#### [new 048] Logios : An open source Greek Polytonic Optical Character Recognition system
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于OCR任务，旨在解决希腊多调文本的识别问题。通过结合卷积和循环层，提升识别准确率与效率，并开源了相关模型与平台。**

- **链接: [http://arxiv.org/pdf/2506.21474v1](http://arxiv.org/pdf/2506.21474v1)**

> **作者:** Perifanos Konstantinos; Goutsos Dionisis
>
> **摘要:** In this paper, we present an Optical Character Recognition (OCR) system specifically designed for the accurate recognition and digitization of Greek polytonic texts. By leveraging the combined strengths of convolutional layers for feature extraction and recurrent layers for sequence learning, our system addresses the unique challenges posed by Greek polytonic scripts. This approach aims to overcome the limitations of traditional OCR methods, offering significant improvements in accuracy and efficiency. We release the underlying model as an open-source library and make our OCR platform available for academic use.
>
---
#### [new 049] Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能搜索任务，旨在解决复杂、长周期的网络信息检索与综合问题。提出Mind2Web 2基准和Agent-as-a-Judge评估框架，提升系统评估准确性。**

- **链接: [http://arxiv.org/pdf/2506.21506v1](http://arxiv.org/pdf/2506.21506v1)**

> **作者:** Boyu Gou; Zanming Huang; Yuting Ning; Yu Gu; Michael Lin; Weijian Qi; Andrei Kopanev; Botao Yu; Bernal Jiménez Gutiérrez; Yiheng Shu; Chan Hee Song; Jiaman Wu; Shijie Chen; Hanane Nour Moussa; Tianshu Zhang; Jian Xie; Yifei Li; Tianci Xue; Zeyi Liao; Kai Zhang; Boyuan Zheng; Zhaowei Cai; Viktor Rozgic; Morteza Ziyadi; Huan Sun; Yu Su
>
> **备注:** Project Homepage: https://osu-nlp-group.github.io/Mind2Web2/
>
> **摘要:** Agentic search such as Deep Research systems, where large language models autonomously browse the web, synthesize information, and return comprehensive citation-backed answers, represents a major shift in how users interact with web-scale information. While promising greater efficiency and cognitive offloading, the growing complexity and open-endedness of agentic search have outpaced existing evaluation benchmarks and methodologies, which largely assume short search horizons and static answers. In this paper, we introduce Mind2Web 2, a benchmark of 130 realistic, high-quality, and long-horizon tasks that require real-time web browsing and extensive information synthesis, constructed with over 1,000 hours of human labor. To address the challenge of evaluating time-varying and complex answers, we propose a novel Agent-as-a-Judge framework. Our method constructs task-specific judge agents based on a tree-structured rubric design to automatically assess both answer correctness and source attribution. We conduct a comprehensive evaluation of nine frontier agentic search systems and human performance, along with a detailed error analysis to draw insights for future development. The best-performing system, OpenAI Deep Research, can already achieve 50-70% of human performance while spending half the time, showing a great potential. Altogether, Mind2Web 2 provides a rigorous foundation for developing and benchmarking the next generation of agentic search systems.
>
---
#### [new 050] MAGPIE: A dataset for Multi-AGent contextual PrIvacy Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于隐私评估任务，旨在解决LLM代理在协作中保护用户隐私的问题。构建了MAGPIE数据集，并测试了现有模型的隐私理解与保护能力。**

- **链接: [http://arxiv.org/pdf/2506.20737v1](http://arxiv.org/pdf/2506.20737v1)**

> **作者:** Gurusha Juneja; Alon Albalak; Wenyue Hua; William Yang Wang
>
> **摘要:** The proliferation of LLM-based agents has led to increasing deployment of inter-agent collaboration for tasks like scheduling, negotiation, resource allocation etc. In such systems, privacy is critical, as agents often access proprietary tools and domain-specific databases requiring strict confidentiality. This paper examines whether LLM-based agents demonstrate an understanding of contextual privacy. And, if instructed, do these systems preserve inference time user privacy in non-adversarial multi-turn conversation. Existing benchmarks to evaluate contextual privacy in LLM-agents primarily assess single-turn, low-complexity tasks where private information can be easily excluded. We first present a benchmark - MAGPIE comprising 158 real-life high-stakes scenarios across 15 domains. These scenarios are designed such that complete exclusion of private data impedes task completion yet unrestricted information sharing could lead to substantial losses. We then evaluate the current state-of-the-art LLMs on (a) their understanding of contextually private data and (b) their ability to collaborate without violating user privacy. Empirical experiments demonstrate that current models, including GPT-4o and Claude-2.7-Sonnet, lack robust understanding of contextual privacy, misclassifying private data as shareable 25.2\% and 43.6\% of the time. In multi-turn conversations, these models disclose private information in 59.9\% and 50.5\% of cases even under explicit privacy instructions. Furthermore, multi-agent systems fail to complete tasks in 71\% of scenarios. These results underscore that current models are not aligned towards both contextual privacy preservation and collaborative task-solving.
>
---
#### [new 051] Hybrid Deep Learning and Signal Processing for Arabic Dialect Recognition in Low-Resource Settings
- **分类: eess.AS; cs.CL; cs.SD; eess.SP**

- **简介: 该论文属于阿拉伯语方言识别任务，旨在解决低资源环境下数据不足的问题。通过结合传统信号处理与深度学习模型进行实验，验证了MFCC+CNN的有效性。**

- **链接: [http://arxiv.org/pdf/2506.21386v1](http://arxiv.org/pdf/2506.21386v1)**

> **作者:** Ghazal Al-Shwayyat; Omer Nezih Gerek
>
> **摘要:** Arabic dialect recognition presents a significant challenge in speech technology due to the linguistic diversity of Arabic and the scarcity of large annotated datasets, particularly for underrepresented dialects. This research investigates hybrid modeling strategies that integrate classical signal processing techniques with deep learning architectures to address this problem in low-resource scenarios. Two hybrid models were developed and evaluated: (1) Mel-Frequency Cepstral Coefficients (MFCC) combined with a Convolutional Neural Network (CNN), and (2) Discrete Wavelet Transform (DWT) features combined with a Recurrent Neural Network (RNN). The models were trained on a dialect-filtered subset of the Common Voice Arabic dataset, with dialect labels assigned based on speaker metadata. Experimental results demonstrate that the MFCC + CNN architecture achieved superior performance, with an accuracy of 91.2% and strong precision, recall, and F1-scores, significantly outperforming the Wavelet + RNN configuration, which achieved an accuracy of 66.5%. These findings highlight the effectiveness of leveraging spectral features with convolutional models for Arabic dialect recognition, especially when working with limited labeled data. The study also identifies limitations related to dataset size, potential regional overlaps in labeling, and model optimization, providing a roadmap for future research. Recommendations for further improvement include the adoption of larger annotated corpora, integration of self-supervised learning techniques, and exploration of advanced neural architectures such as Transformers. Overall, this research establishes a strong baseline for future developments in Arabic dialect recognition within resource-constrained environments.
>
---
#### [new 052] HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态推理任务，解决模型对全局上下文理解不足和忽略关键线索的问题。通过引入上下文奖励和逻辑评估，提升模型的综合推理能力。**

- **链接: [http://arxiv.org/pdf/2506.21277v1](http://arxiv.org/pdf/2506.21277v1)**

> **作者:** Qize Yang; Shimin Yao; Weixuan Chen; Shenghao Fu; Detao Bai; Jiaxing Zhao; Boyuan Sun; Bowen Yin; Xihan Wei; Jingren Zhou
>
> **摘要:** With the rapid evolution of multimodal large language models, the capacity to deeply understand and interpret human intentions has emerged as a critical capability, which demands detailed and thoughtful reasoning. In recent studies, Reinforcement Learning (RL) has demonstrated potential in enhancing the reasoning capabilities of Large Language Models (LLMs). Nonetheless, the challenges associated with adapting RL to multimodal data and formats remain largely unaddressed. In this paper, we identify two issues in existing multimodal reasoning models: insufficient global context understanding and shortcut problems. Insufficient context understanding can happen when a model misinterprets multimodal context, resulting in incorrect answers. The shortcut problem occurs when the model overlooks crucial clues in multimodal inputs, directly addressing the query without considering the multimodal information. To tackle these issues, we emphasize the necessity for the model to reason with a clear understanding of the global context within multimodal inputs. This global context understanding can effectively prevent the model from overlooking key multimodal cues and ensure a thorough reasoning process. To ensure the accurate interpretation of multimodal context information, we implement a context reward judged by a large language model, alongside format and accuracy rewards. Additionally, to improve complex reasoning capability, we employ the LLM to assess the logical reward, determining whether the reasoning process successfully integrates multimodal information with logical methods. We also introduce a reasoning omni-modal benchmark, IntentBench, aimed at evaluating models in understanding complex human intentions and emotions. Our proposed method demonstrates advanced performance across multiple omni-modal benchmarks compared to other open-source omni-modal models.
>
---
#### [new 053] Learning to Skip the Middle Layers of Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型优化任务，旨在提升Transformer效率。通过动态跳过中间层减少计算量，但实验未显示性能优势。**

- **链接: [http://arxiv.org/pdf/2506.21103v1](http://arxiv.org/pdf/2506.21103v1)**

> **作者:** Tim Lawson; Laurence Aitchison
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Conditional computation is a popular strategy to make Transformers more efficient. Existing methods often target individual modules (e.g., mixture-of-experts layers) or skip layers independently of one another. However, interpretability research has demonstrated that the middle layers of Transformers exhibit greater redundancy, and that early layers aggregate information into token positions. Guided by these insights, we propose a novel architecture that dynamically skips a variable number of layers from the middle outward. In particular, a learned gating mechanism determines whether to bypass a symmetric span of central blocks based on the input, and a gated attention mechanism prevents subsequent tokens from attending to skipped token positions. Residual norms are controlled with a 'sandwich' or 'perilayernorm' scheme and gate sparsity with an adaptive regularization loss. We had aimed to reduce compute requirements for 'simpler' tokens and potentially foster an emergent multi-level representational hierarchy but, at the scales investigated, our approach does not achieve improvements in the trade-off between validation cross-entropy and estimated FLOPs compared to dense baselines with fewer layers. We release our code at https://github.com/tim-lawson/skip-middle.
>
---
#### [new 054] Exploring Adapter Design Tradeoffs for Low Resource Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在探索低资源环境下适配器设计的权衡。通过研究不同适配器配置，分析其在音乐细节和长程依赖上的表现，以优化模型性能与资源消耗。**

- **链接: [http://arxiv.org/pdf/2506.21298v1](http://arxiv.org/pdf/2506.21298v1)**

> **作者:** Atharva Mehta; Shivam Chauhan; Monojit Choudhury
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Fine-tuning large-scale music generation models, such as MusicGen and Mustango, is a computationally expensive process, often requiring updates to billions of parameters and, therefore, significant hardware resources. Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly adapter-based methods, have emerged as a promising alternative, enabling adaptation with minimal trainable parameters while preserving model performance. However, the design choices for adapters, including their architecture, placement, and size, are numerous, and it is unclear which of these combinations would produce optimal adapters and why, for a given case of low-resource music genre. In this paper, we attempt to answer this question by studying various adapter configurations for two AI music models, MusicGen and Mustango, on two genres: Hindustani Classical and Turkish Makam music. Our findings reveal distinct trade-offs: convolution-based adapters excel in capturing fine-grained local musical details such as ornamentations and short melodic phrases, while transformer-based adapters better preserve long-range dependencies crucial for structured improvisation. Additionally, we analyze computational resource requirements across different adapter scales, demonstrating how mid-sized adapters (40M parameters) achieve an optimal balance between expressivity and quality. Furthermore, we find that Mustango, a diffusion-based model, generates more diverse outputs with better adherence to the description in the input prompt while lacking in providing stability in notes, rhythm alignment, and aesthetics. Also, it is computationally intensive and requires significantly more time to train. In contrast, autoregressive models like MusicGen offer faster training and are more efficient, and can produce better quality output in comparison, but have slightly higher redundancy in their generations.
>
---
#### [new 055] SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型微调任务，解决边缘设备上无法使用梯度的问题，提出SharpZO方法仅通过前向传播实现高效优化。**

- **链接: [http://arxiv.org/pdf/2506.20990v1](http://arxiv.org/pdf/2506.20990v1)**

> **作者:** Yifan Yang; Zhen Zhang; Rupak Vignesh Swaminathan; Jing Liu; Nathan Susanj; Zheng Zhang
>
> **摘要:** Fine-tuning vision language models (VLMs) has achieved remarkable performance across various downstream tasks; yet, it requires access to model gradients through backpropagation (BP), making them unsuitable for memory-constrained, inference-only edge devices. To address this limitation, previous work has explored various BP-free fine-tuning methods. However, these approaches often rely on high-variance evolutionary strategies (ES) or zeroth-order (ZO) optimization, and often fail to achieve satisfactory performance. In this paper, we propose a hybrid Sharpness-aware Zeroth-order optimization (SharpZO) approach, specifically designed to enhance the performance of ZO VLM fine-tuning via a sharpness-aware warm-up training. SharpZO features a two-stage optimization process: a sharpness-aware ES stage that globally explores and smooths the loss landscape to construct a strong initialization, followed by a fine-grained local search via sparse ZO optimization. The entire optimization relies solely on forward passes. Detailed theoretical analysis and extensive experiments on CLIP models demonstrate that SharpZO significantly improves accuracy and convergence speed, achieving up to 7% average gain over state-of-the-art forward-only methods.
>
---
#### [new 056] Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性量化任务，旨在解决大语言模型的幻觉和校准问题。通过贝叶斯低秩适配方法，实现高效且可扩展的模型不确定性估计。**

- **链接: [http://arxiv.org/pdf/2506.21408v1](http://arxiv.org/pdf/2506.21408v1)**

> **作者:** Colin Samplawski; Adam D. Cobb; Manoj Acharya; Ramneet Kaur; Susmit Jha
>
> **备注:** Accepted at UAI 2025
>
> **摘要:** Despite their widespread use, large language models (LLMs) are known to hallucinate incorrect information and be poorly calibrated. This makes the uncertainty quantification of these models of critical importance, especially in high-stakes domains, such as autonomy and healthcare. Prior work has made Bayesian deep learning-based approaches to this problem more tractable by performing inference over the low-rank adaptation (LoRA) parameters of a fine-tuned model. While effective, these approaches struggle to scale to larger LLMs due to requiring further additional parameters compared to LoRA. In this work we present $\textbf{Scala}$ble $\textbf{B}$ayesian $\textbf{L}$ow-Rank Adaptation via Stochastic Variational Subspace Inference (ScalaBL). We perform Bayesian inference in an $r$-dimensional subspace, for LoRA rank $r$. By repurposing the LoRA parameters as projection matrices, we are able to map samples from this subspace into the full weight space of the LLM. This allows us to learn all the parameters of our approach using stochastic variational inference. Despite the low dimensionality of our subspace, we are able to achieve competitive performance with state-of-the-art approaches while only requiring ${\sim}1000$ additional parameters. Furthermore, it allows us to scale up to the largest Bayesian LLM to date, with four times as a many base parameters as prior work.
>
---
## 更新

#### [replaced 001] TAPS: Tool-Augmented Personalisation via Structured Tagging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20409v2](http://arxiv.org/pdf/2506.20409v2)**

> **作者:** Ekaterina Taktasheva; Jeff Dalton
>
> **摘要:** Recent advancements in tool-augmented large language models have enabled them to interact with external tools, enhancing their ability to perform complex user tasks. However, existing approaches overlook the role of personalisation in guiding tool use. This work investigates how user preferences can be effectively integrated into goal-oriented dialogue agents. Through extensive analysis, we identify key weaknesses in the ability of LLMs to personalise tool use. To this end, we introduce TAPS, a novel solution that enhances personalised tool use by leveraging a structured tagging tool and an uncertainty-based tool detector. TAPS significantly improves the ability of LLMs to incorporate user preferences, achieving the new state-of-the-art for open source models on the NLSI task.
>
---
#### [replaced 002] Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11277v3](http://arxiv.org/pdf/2505.11277v3)**

> **作者:** Yaorui Shi; Sihang Li; Chang Wu; Zhiyuan Liu; Junfeng Fang; Hengxing Cai; An Zhang; Xiang Wang
>
> **摘要:** Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
>
---
#### [replaced 003] SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.20081v2](http://arxiv.org/pdf/2506.20081v2)**

> **作者:** Dhruv Gupta; Gayathri Ganesh Lakshmy; Yiqing Xie
>
> **摘要:** Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is irrelevant. Based on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval).
>
---
#### [replaced 004] Prompting with Phonemes: Enhancing LLMs' Multilinguality for Non-Latin Script Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02398v3](http://arxiv.org/pdf/2411.02398v3)**

> **作者:** Hoang H Nguyen; Khyati Mahajan; Vikas Yadav; Julian Salazar; Philip S. Yu; Masoud Hashemi; Rishabh Maheshwary
>
> **备注:** Accepted to NAACL 2025 (Main Conference). This version contains minor improvements to the camera-ready
>
> **摘要:** Although multilingual LLMs have achieved remarkable performance across benchmarks, we find they continue to underperform on non-Latin script languages across contemporary LLM families. This discrepancy arises from the fact that LLMs are pretrained with orthographic scripts, which are dominated by Latin characters that obscure their shared phonology with non-Latin scripts. We propose leveraging phonemic transcriptions as complementary signals to induce script-invariant representations. Our study demonstrates that integrating phonemic signals improves performance across both non-Latin and Latin script languages, with a particularly significant impact on closing the performance gap between the two. Through detailed experiments, we show that phonemic and orthographic scripts retrieve distinct examples for in-context learning (ICL). This motivates our proposed Mixed-ICL retrieval strategy, where further aggregation from both leads to our significant performance improvements for both Latin script languages (up to 12.6%) and non-Latin script languages (up to 15.1%) compared to randomized ICL retrieval.
>
---
#### [replaced 005] Rethinking LLM Training through Information Geometry and Quantum Metrics
- **分类: cs.CL; quant-ph; I.2; I.7**

- **链接: [http://arxiv.org/pdf/2506.15830v2](http://arxiv.org/pdf/2506.15830v2)**

> **作者:** Riccardo Di Sipio
>
> **备注:** 9 pages, 1 figure(s)
>
> **摘要:** Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
>
---
#### [replaced 006] CVC: A Large-Scale Chinese Value Rule Corpus for Value Alignment of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01495v4](http://arxiv.org/pdf/2506.01495v4)**

> **作者:** Ping Wu; Guobin Shen; Dongcheng Zhao; Yuwei Wang; Yiting Dong; Yu Shi; Enmeng Lu; Feifei Zhao; Yi Zeng
>
> **摘要:** Ensuring that Large Language Models (LLMs) align with mainstream human values and ethical norms is crucial for the safe and sustainable development of AI. Current value evaluation and alignment are constrained by Western cultural bias and incomplete domestic frameworks reliant on non-native rules; furthermore, the lack of scalable, rule-driven scenario generation methods makes evaluations costly and inadequate across diverse cultural contexts. To address these challenges, we propose a hierarchical value framework grounded in core Chinese values, encompassing three main dimensions, 12 core values, and 50 derived values. Based on this framework, we construct a large-scale Chinese Values Corpus (CVC) containing over 250,000 value rules enhanced and expanded through human annotation. Experimental results show that CVC-guided scenarios outperform direct generation ones in value boundaries and content diversity. In the evaluation across six sensitive themes (e.g., surrogacy, suicide), seven mainstream LLMs preferred CVC-generated options in over 70.5% of cases, while five Chinese human annotators showed an 87.5% alignment with CVC, confirming its universality, cultural relevance, and strong alignment with Chinese values. Additionally, we construct 400,000 rule-based moral dilemma scenarios that objectively capture nuanced distinctions in conflicting value prioritization across 17 LLMs. Our work establishes a culturally-adaptive benchmarking framework for comprehensive value evaluation and alignment, representing Chinese characteristics. All data are available at https://huggingface.co/datasets/Beijing-AISI/CVC, and the code is available at https://github.com/Beijing-AISI/CVC.
>
---
#### [replaced 007] Learning Evaluation Models from Large Language Models for Sequence Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2308.04386v3](http://arxiv.org/pdf/2308.04386v3)**

> **作者:** Chenglong Wang; Hang Zhou; Kaiyan Chang; Tongran Liu; Chunliang Zhang; Quan Du; Tong Xiao; Yue Zhang; Jingbo Zhu
>
> **备注:** Accepted by TASLP 2025
>
> **摘要:** Automatic evaluation of sequence generation, traditionally reliant on metrics like BLEU and ROUGE, often fails to capture the semantic accuracy of generated text sequences due to their emphasis on n-gram overlap. A promising solution to this problem is to develop model-based metrics, such as BLEURT and COMET. However, these approaches are typically hindered by the scarcity of labeled evaluation data, which is necessary to train the evaluation models. In this work, we build upon this challenge by proposing the Customized Sequence Evaluation Metric (CSEM), a three-stage evaluation model training method that utilizes large language models to generate labeled data for model-based metric development, thereby eliminating the need for human-labeled data. Additionally, we expand the scope of CSEM to support various evaluation types, including single-aspect, multi-aspect, reference-free, and reference-based evaluations, enabling the customization of metrics to suit diverse real-world scenarios. Experimental results on the SummEval benchmark demonstrate that CSEM can effectively train an evaluation model without human-labeled data. Further experiments in reinforcement learning and reranking show that metrics developed through CSEM outperform traditional evaluation metrics, leading to substantial improvements in sequence quality as evaluated by both commonly used metrics and ChatGPT.
>
---
#### [replaced 008] From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.18959v2](http://arxiv.org/pdf/2506.18959v2)**

> **作者:** Weizhi Zhang; Yangning Li; Yuanchen Bei; Junyu Luo; Guancheng Wan; Liangwei Yang; Chenxuan Xie; Yuyao Yang; Wei-Chieh Huang; Chunyu Miao; Henry Peng Zou; Xiao Luo; Yusheng Zhao; Yankai Chen; Chunkit Chan; Peilin Zhou; Xinyang Zhang; Chenwei Zhang; Jingbo Shang; Ming Zhang; Yangqiu Song; Irwin King; Philip S. Yu
>
> **摘要:** Information retrieval is a cornerstone of modern knowledge acquisition, enabling billions of queries each day across diverse domains. However, traditional keyword-based search engines are increasingly inadequate for handling complex, multi-step information needs. Our position is that Large Language Models (LLMs), endowed with reasoning and agentic capabilities, are ushering in a new paradigm termed Agentic Deep Research. These systems transcend conventional information search techniques by tightly integrating autonomous reasoning, iterative retrieval, and information synthesis into a dynamic feedback loop. We trace the evolution from static web search to interactive, agent-based systems that plan, explore, and learn. We also introduce a test-time scaling law to formalize the impact of computational depth on reasoning and search. Supported by benchmark results and the rise of open-source implementations, we demonstrate that Agentic Deep Research not only significantly outperforms existing approaches, but is also poised to become the dominant paradigm for future information seeking. All the related resources, including industry products, research papers, benchmark datasets, and open-source implementations, are collected for the community in https://github.com/DavidZWZ/Awesome-Deep-Research.
>
---
#### [replaced 009] Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.09510v2](http://arxiv.org/pdf/2409.09510v2)**

> **作者:** Alireza Salemi; Hamed Zamani
>
> **摘要:** Despite its substantial impact on various search, recommendation, and question answering tasks, privacy-preserving methods for personalizing large language models (LLMs) have received relatively limited exploration. There is one primary approach in this area through retrieval-augmented generation (RAG), which generates personalized outputs by enriching the input prompt with information retrieved from the user's personal data. This paper studies an orthogonal approach to RAG that involves learning user-dependent LLM parameters through parameter-efficient fine-tuning (PEFT). This paper presents the first systematic study for exploration of PEFT for LLM personalization and provides an extensive comparisons between RAG- and PEFT-based solutions, across a broad set of seven diverse datasets from the LaMP benchmark. Our results demonstrate that, on average, both RAG- and PEFT-based personalization methods yield 14.92% and 1.07% improvements over non-personalized LLMs, respectively. When combining RAG with PEFT, we observe a further improvement of 15.98%, highlighting the effectiveness of their integration in enhancing personalized text generation. Additionally, we identify a positive correlation between the amount of user data available and the effectiveness of PEFT. This finding suggests that RAG is particularly beneficial for cold-start users -- users with limited personal data -- while PEFT performs better when more user-specific data is available.
>
---
#### [replaced 010] Do Large Language Models Advocate for Inferentialism?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14501v2](http://arxiv.org/pdf/2412.14501v2)**

> **作者:** Yuzuki Arai; Sho Tsugawa
>
> **摘要:** The emergence of large language models (LLMs) such as ChatGPT and Claude presents new challenges for philosophy of language, particularly regarding the nature of linguistic meaning and representation. While LLMs have traditionally been understood through distributional semantics, this paper explores Robert Brandom's inferential semantics as an alternative foundational framework for understanding these systems. We examine how key features of inferential semantics -- including its anti-representationalist stance, logical expressivism, and quasi-compositional approach -- align with the architectural and functional characteristics of Transformer-based LLMs. Through analysis of the ISA (Inference, Substitution, Anaphora) approach, we demonstrate that LLMs exhibit fundamentally anti-representationalist properties in their processing of language. We further develop a consensus theory of truth appropriate for LLMs, grounded in their interactive and normative dimensions through mechanisms like RLHF. While acknowledging significant tensions between inferentialism's philosophical commitments and LLMs' sub-symbolic processing, this paper argues that inferential semantics provides valuable insights into how LLMs generate meaning without reference to external world representations. Our analysis suggests that LLMs may challenge traditional assumptions in philosophy of language, including strict compositionality and semantic externalism, though further empirical investigation is needed to fully substantiate these theoretical claims.
>
---
#### [replaced 011] OpenNER 1.0: Standardized Open-Access Named Entity Recognition Datasets in 50+ Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.09587v2](http://arxiv.org/pdf/2412.09587v2)**

> **作者:** Chester Palen-Michel; Maxwell Pickering; Maya Kruse; Jonne Sälevä; Constantine Lignos
>
> **备注:** Under review
>
> **摘要:** We present OpenNER 1.0, a standardized collection of openly-available named entity recognition (NER) datasets. OpenNER contains 36 NER corpora that span 52 languages, human-annotated in varying named entity ontologies. We correct annotation format issues, standardize the original datasets into a uniform representation with consistent entity type names across corpora, and provide the collection in a structure that enables research in multilingual and multi-ontology NER. We provide baseline results using three pretrained multilingual language models and two large language models to compare the performance of recent models and facilitate future research in NER. We find that no single model is best in all languages and that significant work remains to obtain high performance from LLMs on the NER task.
>
---
#### [replaced 012] Reward-Guided Speculative Decoding for Efficient LLM Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.19324v3](http://arxiv.org/pdf/2501.19324v3)**

> **作者:** Baohao Liao; Yuhui Xu; Hanze Dong; Junnan Li; Christof Monz; Silvio Savarese; Doyen Sahoo; Caiming Xiong
>
> **备注:** 17 pages
>
> **摘要:** We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. The code is available at https://github.com/BaohaoLiao/RSD.
>
---
#### [replaced 013] A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16155v2](http://arxiv.org/pdf/2410.16155v2)**

> **作者:** Tianyi Men; Pengfei Cao; Zhuoran Jin; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** ACL 2025 Main
>
> **摘要:** With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.
>
---
#### [replaced 014] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.17443v4](http://arxiv.org/pdf/2408.17443v4)**

> **作者:** Gueter Josmy Faure; Jia-Fong Yeh; Min-Hung Chen; Hung-Ting Su; Shang-Hong Lai; Winston H. Hsu
>
> **备注:** Accepted for ICCV 2025. Project page: https://joslefaure.github.io/assets/html/hermes.html
>
> **摘要:** Long-form video understanding presents unique challenges that extend beyond traditional short-video analysis approaches, particularly in capturing long-range dependencies, processing redundant information efficiently, and extracting high-level semantic concepts. To address these challenges, we propose a novel approach that more accurately reflects human cognition. This paper introduces HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics, featuring two versatile modules that can enhance existing video-language models or operate as a standalone system. Our Episodic COmpressor (ECO) efficiently aggregates representations from micro to semi-macro levels, reducing computational overhead while preserving temporal dependencies. Our Semantics ReTRiever (SeTR) enriches these representations with semantic information by focusing on broader context, dramatically reducing feature dimensionality while preserving relevant macro-level information. We demonstrate that these modules can be seamlessly integrated into existing SOTA models, consistently improving their performance while reducing inference latency by up to 43% and memory usage by 46%. As a standalone system, HERMES achieves state-of-the-art performance across multiple long-video understanding benchmarks in both zero-shot and fully-supervised settings.
>
---
#### [replaced 015] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20639v2](http://arxiv.org/pdf/2506.20639v2)**

> **作者:** Shansan Gong; Ruixiang Zhang; Huangjie Zheng; Jiatao Gu; Navdeep Jaitly; Lingpeng Kong; Yizhe Zhang
>
> **备注:** minor update
>
> **摘要:** Diffusion large language models (dLLMs) are compelling alternatives to autoregressive (AR) models because their denoising models operate over the entire sequence. The global planning and iterative refinement features of dLLMs are particularly useful for code generation. However, current training and inference mechanisms for dLLMs in coding are still under-explored. To demystify the decoding behavior of dLLMs and unlock their potential for coding, we systematically investigate their denoising processes and reinforcement learning (RL) methods. We train a 7B dLLM, \textbf{DiffuCoder}, on 130B tokens of code. Using this model as a testbed, we analyze its decoding behavior, revealing how it differs from that of AR models: (1) dLLMs can decide how causal their generation should be without relying on semi-AR decoding, and (2) increasing the sampling temperature diversifies not only token choices but also their generation order. This diversity creates a rich search space for RL rollouts. For RL training, to reduce the variance of token log-likelihood estimates and maintain training efficiency, we propose \textbf{coupled-GRPO}, a novel sampling scheme that constructs complementary mask noise for completions used in training. In our experiments, coupled-GRPO significantly improves DiffuCoder's performance on code generation benchmarks (+4.4\% on EvalPlus) and reduces reliance on AR bias during decoding. Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework. https://github.com/apple/ml-diffucoder.
>
---
#### [replaced 016] MockLLM: A Multi-Agent Behavior Collaboration Framework for Online Job Seeking and Recruiting
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.18113v2](http://arxiv.org/pdf/2405.18113v2)**

> **作者:** Hongda Sun; Hongzhan Lin; Haiyu Yan; Yang Song; Xin Gao; Rui Yan
>
> **备注:** Accepted by KDD 2025 Research Track
>
> **摘要:** Online recruitment platforms have reshaped job-seeking and recruiting processes, driving increased demand for applications that enhance person-job matching. Traditional methods generally rely on analyzing textual data from resumes and job descriptions, limiting the dynamic, interactive aspects crucial to effective recruitment. Recent advances in Large Language Models (LLMs) have revealed remarkable potential in simulating adaptive, role-based dialogues, making them well-suited for recruitment scenarios. In this paper, we propose \textbf{MockLLM}, a novel framework to generate and evaluate mock interview interactions. The system consists of two key components: mock interview generation and two-sided evaluation in handshake protocol. By simulating both interviewer and candidate roles, MockLLM enables consistent and collaborative interactions for real-time and two-sided matching. To further improve the matching quality, MockLLM further incorporates reflection memory generation and dynamic strategy modification, refining behaviors based on previous experience. We evaluate MockLLM on real-world data Boss Zhipin, a major Chinese recruitment platform. The experimental results indicate that MockLLM outperforms existing methods in matching accuracy, scalability, and adaptability across job domains, highlighting its potential to advance candidate assessment and online recruitment.
>
---
#### [replaced 017] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00753v4](http://arxiv.org/pdf/2505.00753v4)**

> **作者:** Henry Peng Zou; Wei-Chieh Huang; Yaozu Wu; Yankai Chen; Chunyu Miao; Hoang Nguyen; Yue Zhou; Weizhi Zhang; Liancheng Fang; Langzhou He; Yangning Li; Dongyuan Li; Renhe Jiang; Xue Liu; Philip S. Yu
>
> **备注:** Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems.
>
---
#### [replaced 018] PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04065v3](http://arxiv.org/pdf/2503.04065v3)**

> **作者:** Feng Ni; Kui Huang; Yao Lu; Wenyu Lv; Guanzhong Wang; Zeyu Chen; Yi Liu
>
> **摘要:** With the rapid advancement of digitalization, various document images are being applied more extensively in production and daily life, and there is an increasingly urgent need for fast and accurate parsing of the content in document images. Therefore, this report presents PP-DocBee, a novel multimodal large language model designed for end-to-end document image understanding. First, we develop a data synthesis strategy tailored to document scenarios in which we build a diverse dataset to improve the model generalization. Then, we apply a few training techniques, including dynamic proportional sampling, data preprocessing, and OCR postprocessing strategies. Extensive evaluations demonstrate the superior performance of PP-DocBee, achieving state-of-the-art results on English document understanding benchmarks and even outperforming existing open source and commercial models in Chinese document understanding. The source code and pre-trained models are publicly available at \href{https://github.com/PaddlePaddle/PaddleMIX}{https://github.com/PaddlePaddle/PaddleMIX}.
>
---
#### [replaced 019] Privacy Ripple Effects from Adding or Removing Personal Information in Language Model Training
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.15680v2](http://arxiv.org/pdf/2502.15680v2)**

> **作者:** Jaydeep Borkar; Matthew Jagielski; Katherine Lee; Niloofar Mireshghallah; David A. Smith; Christopher A. Choquette-Choo
>
> **备注:** Accepted at the Findings of the Association for Computational Linguistics (2025)
>
> **摘要:** Due to the sensitive nature of personally identifiable information (PII), its owners may have the authority to control its inclusion or request its removal from large-language model (LLM) training. Beyond this, PII may be added or removed from training datasets due to evolving dataset curation techniques, because they were newly scraped for retraining, or because they were included in a new downstream fine-tuning stage. We find that the amount and ease of PII memorization is a dynamic property of a model that evolves throughout training pipelines and depends on commonly altered design choices. We characterize three such novel phenomena: (1) similar-appearing PII seen later in training can elicit memorization of earlier-seen sequences in what we call assisted memorization, and this is a significant factor (in our settings, up to 1/3); (2) adding PII can increase memorization of other PII significantly (in our settings, as much as $\approx\!7.5\times$); and (3) removing PII can lead to other PII being memorized. Model creators should consider these first- and second-order privacy risks when training models to avoid the risk of new PII regurgitation.
>
---
#### [replaced 020] Capturing Style in Author and Document Representation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.13358v2](http://arxiv.org/pdf/2407.13358v2)**

> **作者:** Enzo Terreau; Antoine Gourru; Julien Velcin
>
> **摘要:** A wide range of Deep Natural Language Processing (NLP) models integrates continuous and low dimensional representations of words and documents. Surprisingly, very few models study representation learning for authors. These representations can be used for many NLP tasks, such as author identification and classification, or in recommendation systems. A strong limitation of existing works is that they do not explicitly capture writing style, making them hardly applicable to literary data. We therefore propose a new architecture based on Variational Information Bottleneck (VIB) that learns embeddings for both authors and documents with a stylistic constraint. Our model fine-tunes a pre-trained document encoder. We stimulate the detection of writing style by adding predefined stylistic features making the representation axis interpretable with respect to writing style indicators. We evaluate our method on three datasets: a literary corpus extracted from the Gutenberg Project, the Blog Authorship Corpus and IMDb62, for which we show that it matches or outperforms strong/recent baselines in authorship attribution while capturing much more accurately the authors stylistic aspects.
>
---
#### [replaced 021] Thinkless: LLM Learns When to Think
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13379v2](http://arxiv.org/pdf/2505.13379v2)**

> **作者:** Gongfan Fang; Xinyin Ma; Xinchao Wang
>
> **摘要:** Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
>
---
#### [replaced 022] Simulating Hard Attention Using Soft Attention
- **分类: cs.LG; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2412.09925v2](http://arxiv.org/pdf/2412.09925v2)**

> **作者:** Andy Yang; Lena Strobl; David Chiang; Dana Angluin
>
> **备注:** 19 pages
>
> **摘要:** We study conditions under which transformers using soft attention can simulate hard attention, that is, effectively focus all attention on a subset of positions. First, we examine several subclasses of languages recognized by hard-attention transformers, which can be defined in variants of linear temporal logic. We demonstrate how soft-attention transformers can compute formulas of these logics using unbounded positional embeddings or temperature scaling. Second, we demonstrate how temperature scaling allows softmax transformers to simulate general hard-attention transformers, using a temperature that depends on the minimum gap between the maximum attention scores and other attention scores.
>
---
#### [replaced 023] SceneGenAgent: Precise Industrial Scene Generation with Coding Agent
- **分类: cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.21909v3](http://arxiv.org/pdf/2410.21909v3)**

> **作者:** Xiao Xia; Dan Zhang; Zibo Liao; Zhenyu Hou; Tianrui Sun; Jing Li; Ling Fu; Yuxiao Dong
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** The modeling of industrial scenes is essential for simulations in industrial manufacturing. While large language models (LLMs) have shown significant progress in generating general 3D scenes from textual descriptions, generating industrial scenes with LLMs poses a unique challenge due to their demand for precise measurements and positioning, requiring complex planning over spatial arrangement. To address this challenge, we introduce SceneGenAgent, an LLM-based agent for generating industrial scenes through C# code. SceneGenAgent ensures precise layout planning through a structured and calculable format, layout verification, and iterative refinement to meet the quantitative requirements of industrial scenarios. Experiment results demonstrate that LLMs powered by SceneGenAgent exceed their original performance, reaching up to 81.0% success rate in real-world industrial scene generation tasks and effectively meeting most scene generation requirements. To further enhance accessibility, we construct SceneInstruct, a dataset designed for fine-tuning open-source LLMs to integrate into SceneGenAgent. Experiments show that fine-tuning open-source LLMs on SceneInstruct yields significant performance improvements, with Llama3.1-70B approaching the capabilities of GPT-4o. Our code and data are available at https://github.com/THUDM/SceneGenAgent .
>
---
#### [replaced 024] Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.15928v2](http://arxiv.org/pdf/2506.15928v2)**

> **作者:** Myke C. Cohen; Zhe Su; Hsien-Te Kao; Daniel Nguyen; Spencer Lynch; Maarten Sap; Svitlana Volkova
>
> **备注:** Under review for KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models
>
> **摘要:** This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
>
---
#### [replaced 025] Evaluating Rare Disease Diagnostic Performance in Symptom Checkers: A Synthetic Vignette Simulation Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19750v3](http://arxiv.org/pdf/2506.19750v3)**

> **作者:** Takashi Nishibayashi; Seiji Kanazawa; Kumpei Yamada
>
> **摘要:** Symptom Checkers (SCs) provide medical information tailored to user symptoms. A critical challenge in SC development is preventing unexpected performance degradation for individual diseases, especially rare diseases, when updating algorithms. This risk stems from the lack of practical pre-deployment evaluation methods. For rare diseases, obtaining sufficient evaluation data from user feedback is difficult. To evaluate the impact of algorithm updates on the diagnostic performance for individual rare diseases before deployment, this study proposes and validates a novel Synthetic Vignette Simulation Approach. This approach aims to enable this essential evaluation efficiently and at a low cost. To estimate the impact of algorithm updates, we generated synthetic vignettes from disease-phenotype annotations in the Human Phenotype Ontology (HPO), a publicly available knowledge base for rare diseases curated by experts. Using these vignettes, we simulated SC interviews to predict changes in diagnostic performance. The effectiveness of this approach was validated retrospectively by comparing the predicted changes with actual performance metrics using the R-squared ($R^2$) coefficient. Our experiment, covering eight past algorithm updates for rare diseases, showed that the proposed method accurately predicted performance changes for diseases with phenotype frequency information in HPO (n=5). For these updates, we found a strong correlation for both Recall@8 change ($R^2$ = 0.83,$p$ = 0.031) and Precision@8 change ($R^2$ = 0.78,$p$ = 0.047). Our proposed method enables the pre-deployment evaluation of SC algorithm changes for individual rare diseases. This evaluation is based on a publicly available medical knowledge database created by experts, ensuring transparency and explainability for stakeholders. Additionally, SC developers can efficiently improve diagnostic performance at a low cost.
>
---
#### [replaced 026] Evaluating Large Language Models for Automated Clinical Abstraction in Pulmonary Embolism Registries: Performance Across Model Sizes, Versions, and Parameters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21004v2](http://arxiv.org/pdf/2503.21004v2)**

> **作者:** Mahmoud Alwakeel; Emory Buck; Jonathan G. Martin; Imran Aslam; Sudarshan Rajagopal; Jian Pei; Mihai V. Podgoreanu; Christopher J. Lindsell; An-Kwok Ian Wong
>
> **摘要:** Pulmonary embolism (PE) registries accelerate practice improving research but rely on labor intensive manual abstraction of radiology reports. We examined whether openly available large language models (LLMs) can automate concept extraction from computed tomography PE (CTPE) reports without loss of data quality. Four Llama 3 variants (3.0 8B, 3.1 8B, 3.1 70B, 3.3 70B) and one reviewer model, Phi 4 14B, were tested on 250 dual annotated CTPE reports from each of MIMIC IV and Duke University. Accuracy, positive predictive value (PPV) and negative predictive value (NPV) versus a human gold standard were measured across model size, temperature and shot count. Mean accuracy rose with scale: 0.83 (3.0 8B), 0.91 (3.1 8B) and 0.96 for both 70B variants; Phi 4 14B reached 0.98. Accuracy differed by less than 0.03 between datasets, indicating external robustness. In dual model concordance (L3 70B plus Phi 4 14B) PPV for PE presence was at least 0.95 and NPV at least 0.98, while location, thrombus burden, right heart strain and image quality artifacts each achieved PPV of at least 0.90 and NPV of at least 0.95. Fewer than four percent of individual concept annotations were discordant, and full agreement occurred in more than seventy five percent of reports. Large language models therefore provide a scalable, accurate solution for PE registry abstraction, and a dual model review workflow can safeguard data quality with minimal human oversight.
>
---
#### [replaced 027] CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.05199v3](http://arxiv.org/pdf/2411.05199v3)**

> **作者:** Leitian Tao; Xiang Chen; Tong Yu; Tung Mai; Ryan Rossi; Yixuan Li; Saayan Mitra
>
> **备注:** TMLR 2025
>
> **摘要:** Large Language Models (LLMs) have revolutionized code generation but require significant resources and often over-generalize, limiting their task-specific efficiency. Fine-tuning smaller, open-source LLMs provides a cost-effective alternative. However, standard supervised approaches rely only on correct examples, missing valuable insights from failures. We introduce CodeLutra, a framework that leverages both correct and incorrect code attempts. Instead of using only correct solutions, CodeLutra applies iterative preference-based refinement, comparing successful and failed outputs to better approximate desired results. This approach narrows the performance gap with state-of-the-art larger models without requiring massive datasets or auxiliary models. For instance, on a challenging data science coding task, using only 500 samples improved Llama-3-8B's accuracy from 28.2% to 48.6%, approaching GPT-4's level. By learning from both successes and mistakes, CodeLutra provides a scalable and efficient path to high-quality code generation, making smaller open-source models more competitive with leading closed-source alternatives.
>
---
#### [replaced 028] Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21657v3](http://arxiv.org/pdf/2505.21657v3)**

> **作者:** Zeinab Dehghani; Mohammed Naveed Akram; Koorosh Aslansefat; Adil Khan
>
> **备注:** The submission contains incorrect references that require substantial revision
>
> **摘要:** Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy.
>
---
#### [replaced 029] Learning to Rank for Multiple Retrieval-Augmented Models through Iterative Utility Maximization
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2410.09942v2](http://arxiv.org/pdf/2410.09942v2)**

> **作者:** Alireza Salemi; Hamed Zamani
>
> **摘要:** This paper investigates the design of a unified search engine to serve multiple retrieval-augmented generation (RAG) agents, each with a distinct task, backbone large language model (LLM), and RAG strategy. We introduce an iterative approach where the search engine generates retrieval results for the RAG agents and gathers feedback on the quality of the retrieved documents during an offline phase. This feedback is then used to iteratively optimize the search engine using an expectation-maximization algorithm, with the goal of maximizing each agent's utility function. Additionally, we adapt this to an online setting, allowing the search engine to refine its behavior based on real-time individual agents feedback to better serve the results for each of them. Experiments on datasets from the Knowledge-Intensive Language Tasks (KILT) benchmark demonstrates that our approach significantly on average outperforms baselines across 18 RAG models. We demonstrate that our method effectively ``personalizes'' the retrieval for each RAG agent based on the collected feedback. Finally, we provide a comprehensive ablation study to explore various aspects of our method.
>
---
#### [replaced 030] A3 : an Analytical Low-Rank Approximation Framework for Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12942v3](http://arxiv.org/pdf/2505.12942v3)**

> **作者:** Jeffrey T. H. Wong; Cheng Zhang; Xinye Cao; Pedro Gimenes; George A. Constantinides; Wayne Luk; Yiren Zhao
>
> **摘要:** Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches for decomposed small matrices. To address these limitations, we propose $\tt A^\tt 3$, a post-training low-rank approximation framework. $\tt A^\tt 3$ splits a Transformer layer into three functional components, namely $\tt QK$, $\tt OV$, and $\tt MLP$. For each component, $\tt A^\tt 3$ provides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss ($\it i.e.$, error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $\tt A^\tt 3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also demonstrate the versatility of $\tt A^\tt 3$, including KV cache compression, quantization, and mixed-rank assignments for enhanced performance.
>
---
#### [replaced 031] GroundCap: A Visually Grounded Image Captioning Dataset
- **分类: cs.CV; cs.CL; I.2.10; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.13898v3](http://arxiv.org/pdf/2502.13898v3)**

> **作者:** Daniel A. P. Oliveira; Lourenço Teodoro; David Martins de Matos
>
> **备注:** 37 pages
>
> **摘要:** Current image captioning systems lack the ability to link descriptive text to specific visual elements, making their outputs difficult to verify. While recent approaches offer some grounding capabilities, they cannot track object identities across multiple references or ground both actions and objects simultaneously. We propose a novel ID-based grounding system that enables consistent object reference tracking and action-object linking. We present GroundCap, a dataset containing 52,016 images from 77 movies, with 344 human-annotated and 52,016 automatically generated captions. Each caption is grounded on detected objects (132 classes) and actions (51 classes) using a tag system that maintains object identity while linking actions to the corresponding objects. Our approach features persistent object IDs for reference tracking, explicit action-object linking, and the segmentation of background elements through K-means clustering. We propose gMETEOR, a metric combining caption quality with grounding accuracy, and establish baseline performance by fine-tuning Pixtral-12B and Qwen2.5-VL 7B on GroundCap. Human evaluation demonstrates our approach's effectiveness in producing verifiable descriptions with coherent object references.
>
---

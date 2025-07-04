# 自然语言处理 cs.CL

- **最新发布 50 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] A Cookbook for Community-driven Data Collection of Impaired Speech in LowResource Languages
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言中受损语音数据收集难题。通过构建“指南手册”和开放数据集，促进社区参与的ASR模型开发。**

- **链接: [http://arxiv.org/pdf/2507.02428v1](http://arxiv.org/pdf/2507.02428v1)**

> **作者:** Sumaya Ahmed Salihs; Isaac Wiafe; Jamal-Deen Abdulai; Elikem Doe Atsakpo; Gifty Ayoka; Richard Cave; Akon Obu Ekpezu; Catherine Holloway; Katrin Tomanek; Fiifi Baffoe Payin Winful
>
> **备注:** This version has been reviewed and accepted for presentation at the InterSpeech 2025 conference to be held in Rotterdam from 17 to 21 August. 5 pages and 3 tables
>
> **摘要:** This study presents an approach for collecting speech samples to build Automatic Speech Recognition (ASR) models for impaired speech, particularly, low-resource languages. It aims to democratize ASR technology and data collection by developing a "cookbook" of best practices and training for community-driven data collection and ASR model building. As a proof-of-concept, this study curated the first open-source dataset of impaired speech in Akan: a widely spoken indigenous language in Ghana. The study involved participants from diverse backgrounds with speech impairments. The resulting dataset, along with the cookbook and open-source tools, are publicly available to enable researchers and practitioners to create inclusive ASR technologies tailored to the unique needs of speech impaired individuals. In addition, this study presents the initial results of fine-tuning open-source ASR models to better recognize impaired speech in Akan.
>
---
#### [new 002] Revisiting Active Learning under (Human) Label Variation
- **分类: cs.CL; cs.HC; cs.LG; stat.ML**

- **简介: 该论文属于主动学习任务，解决标注差异问题。针对人类标注变异，提出融合其信息的AL框架，提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2507.02593v1](http://arxiv.org/pdf/2507.02593v1)**

> **作者:** Cornelia Gruber; Helen Alber; Bernd Bischl; Göran Kauermann; Barbara Plank; Matthias Aßenmacher
>
> **摘要:** Access to high-quality labeled data remains a limiting factor in applied supervised learning. While label variation (LV), i.e., differing labels for the same instance, is common, especially in natural language processing, annotation frameworks often still rest on the assumption of a single ground truth. This overlooks human label variation (HLV), the occurrence of plausible differences in annotations, as an informative signal. Similarly, active learning (AL), a popular approach to optimizing the use of limited annotation budgets in training ML models, often relies on at least one of several simplifying assumptions, which rarely hold in practice when acknowledging HLV. In this paper, we examine foundational assumptions about truth and label nature, highlighting the need to decompose observed LV into signal (e.g., HLV) and noise (e.g., annotation error). We survey how the AL and (H)LV communities have addressed -- or neglected -- these distinctions and propose a conceptual framework for incorporating HLV throughout the AL loop, including instance selection, annotator choice, and label representation. We further discuss the integration of large language models (LLM) as annotators. Our work aims to lay a conceptual foundation for HLV-aware active learning, better reflecting the complexities of real-world annotation.
>
---
#### [new 003] MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; cs.SY; eess.SY; math.IT**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在有限上下文长度下的推理问题。通过提出MOTIF方法，实现多轮模块化思考，提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.02851v1](http://arxiv.org/pdf/2507.02851v1)**

> **作者:** Purbesh Mitra; Sennur Ulukus
>
> **摘要:** Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at https://github.com/purbeshmitra/MOTIF and https://huggingface.co/purbeshmitra/MOTIF, respectively.
>
---
#### [new 004] Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在提升大语言模型训练效率。通过优化数据选择策略，确保数据分布一致性和多样性，以减少计算成本并提高模型性能。**

- **链接: [http://arxiv.org/pdf/2507.02378v1](http://arxiv.org/pdf/2507.02378v1)**

> **作者:** Weijie Lyu; Sheng-Jun Huang; Xuan Xia
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. Experimental results demonstrate that using only 10K samples, our method achieves gains of 2.4% (HumanEval) and 2.3% (MBPP) over 92K full-sampled baseline, outperforming other sampling approaches in both performance and efficiency. This underscores that our method effectively boosts model performance while significantly reducing computational costs.
>
---
#### [new 005] Multimodal Mathematical Reasoning with Diverse Solving Perspective
- **分类: cs.CL**

- **简介: 该论文属于多模态数学推理任务，旨在解决现有模型依赖单一解法、缺乏多样性的问题。通过构建新数据集和改进模型，提升推理准确性和多样性。**

- **链接: [http://arxiv.org/pdf/2507.02804v1](http://arxiv.org/pdf/2507.02804v1)**

> **作者:** Wenhao Shi; Zhiqiang Hu; Yi Bin; Yang Yang; See-Kiong Ng; Heng Tao Shen
>
> **备注:** 8 pages
>
> **摘要:** Recent progress in large-scale reinforcement learning (RL) has notably enhanced the reasoning capabilities of large language models (LLMs), especially in mathematical domains. However, current multimodal LLMs (MLLMs) for mathematical reasoning often rely on one-to-one image-text pairs and single-solution supervision, overlooking the diversity of valid reasoning perspectives and internal reflections. In this work, we introduce MathV-DP, a novel dataset that captures multiple diverse solution trajectories for each image-question pair, fostering richer reasoning supervision. We further propose Qwen-VL-DP, a model built upon Qwen-VL, fine-tuned with supervised learning and enhanced via group relative policy optimization (GRPO), a rule-based RL approach that integrates correctness discrimination and diversity-aware reward functions. Our method emphasizes learning from varied reasoning perspectives and distinguishing between correct yet distinct solutions. Extensive experiments on the MathVista's minitest and Math-V benchmarks demonstrate that Qwen-VL-DP significantly outperforms prior base MLLMs in both accuracy and generative diversity, highlighting the importance of incorporating diverse perspectives and reflective reasoning in multimodal mathematical reasoning.
>
---
#### [new 006] SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型优化任务，旨在解决LLM在准确性和成本间的平衡问题。通过动态路由框架SynapseRoute，将查询分配到合适模式，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.02822v1](http://arxiv.org/pdf/2507.02822v1)**

> **作者:** Wencheng Zhang; Shiqin Qiao; Lingjie Luo; Yinfeng Li; Chuanyang Zheng; Qian Xu; Meng Li; Yong Gui; Yijun He; Jianing Qiu; Jindong Hong; Jiankai Sun
>
> **摘要:** With the widespread adoption of large language models (LLMs) in practical applications, selecting an appropriate model requires balancing not only performance but also operational cost. The emergence of reasoning-capable models has further widened the cost gap between "thinking" (high reasoning) and "non-thinking" (fast, low-cost) modes. In this work, we reveal that approximately 58% of medical questions can be accurately answered by the non-thinking mode alone, without requiring the high-cost reasoning process. This highlights a clear dichotomy in problem complexity and suggests that dynamically routing queries to the appropriate mode based on complexity could optimize accuracy, cost-efficiency, and overall user experience. Based on this, we further propose SynapseRoute, a machine learning-based dynamic routing framework that intelligently assigns input queries to either thinking or non-thinking modes. Experimental results on several medical datasets demonstrate that SynapseRoute not only improves overall accuracy (0.8390 vs. 0.8272) compared to the thinking mode alone but also reduces inference time by 36.8% and token consumption by 39.66%. Importantly, qualitative analysis indicates that over-reasoning on simpler queries can lead to unnecessary delays and even decreased accuracy, a pitfall avoided by our adaptive routing. Finally, this work further introduces the Accuracy-Inference-Token (AIT) index to comprehensively evaluate the trade-offs among accuracy, latency, and token cost.
>
---
#### [new 007] Generalizing Verifiable Instruction Following
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型难以准确遵循复杂输出约束的问题。通过构建新基准IFBench和引入RLVR方法提升模型的指令遵循能力。**

- **链接: [http://arxiv.org/pdf/2507.02833v1](http://arxiv.org/pdf/2507.02833v1)**

> **作者:** Valentina Pyatkin; Saumya Malik; Victoria Graf; Hamish Ivison; Shengyi Huang; Pradeep Dasigi; Nathan Lambert; Hannaneh Hajishirzi
>
> **备注:** 11 pages
>
> **摘要:** A crucial factor for successful human and AI interaction is the ability of language models or chatbots to follow human instructions precisely. A common feature of instructions are output constraints like ``only answer with yes or no" or ``mention the word `abrakadabra' at least 3 times" that the user adds to craft a more useful answer. Even today's strongest models struggle with fulfilling such constraints. We find that most models strongly overfit on a small set of verifiable constraints from the benchmarks that test these abilities, a skill called precise instruction following, and are not able to generalize well to unseen output constraints. We introduce a new benchmark, IFBench, to evaluate precise instruction following generalization on 58 new, diverse, and challenging verifiable out-of-domain constraints. In addition, we perform an extensive analysis of how and on what data models can be trained to improve precise instruction following generalization. Specifically, we carefully design constraint verification modules and show that reinforcement learning with verifiable rewards (RLVR) significantly improves instruction following. In addition to IFBench, we release 29 additional new hand-annotated training constraints and verification functions, RLVR training prompts, and code.
>
---
#### [new 008] Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究深度循环Transformer模型是否具备隐式链式思维能力，通过探针技术分析其内部行为，旨在理解隐式推理的有效性与局限性。**

- **链接: [http://arxiv.org/pdf/2507.02199v1](http://arxiv.org/pdf/2507.02199v1)**

> **作者:** Wenquan Lu; Yuechuan Yang; Kyle Lee; Yanshu Li; Enqi Liu
>
> **摘要:** Chain-of-thought (CoT) reasoning has enabled transformer-based language models to excel at complex mathematics and multi-step planning. However, in standard decoder-only architectures, these reasoning steps are externalized in natural language, improving interpretability at the cost of efficiency. To capture reasoning that is not easily represented in words, many works have explored recurrent architectures that aim to internalize reasoning in latent space, potentially supporting latent CoT. In this paper, we investigate whether such reasoning structures emerge in Huginn-3.5B, a depth-recurrent Transformer that reuses layers at inference time without increasing parameter count. We examine the model's internal behavior on arithmetic tasks using a suite of probing techniques including the Logit Lens and Coda Lens. Our findings reveal limited evidence of interpretable latent CoT by tracking rank trajectories of final and intermediate result tokens. Furthermore, we uncover significant probing inconsistencies across recurrent blocks, where the interpretability of hidden states depends heavily on both the layer index and the decoding method. Finally, we empirically show that increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps. The code is available at https://github.com/wenquanlu/huginn-latent-cot.
>
---
#### [new 009] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见评估任务，旨在解决中文语境下多维度偏见评估不足的问题。研究构建了McBE基准，涵盖多种偏见类别和评估任务。**

- **链接: [http://arxiv.org/pdf/2507.02088v1](http://arxiv.org/pdf/2507.02088v1)**

> **作者:** Tian Lan; Xiangdong Su; Xu Liu; Ruirui Wang; Ke Chang; Jiang Li; Guanglai Gao
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** As large language models (LLMs) are increasingly applied to various NLP tasks, their inherent biases are gradually disclosed. Therefore, measuring biases in LLMs is crucial to mitigate its ethical risks. However, most existing bias evaluation datasets focus on English and North American culture, and their bias categories are not fully applicable to other cultures. The datasets grounded in the Chinese language and culture are scarce. More importantly, these datasets usually only support single evaluation tasks and cannot evaluate the bias from multiple aspects in LLMs. To address these issues, we present a Multi-task Chinese Bias Evaluation Benchmark (McBE) that includes 4,077 bias evaluation instances, covering 12 single bias categories, 82 subcategories and introducing 5 evaluation tasks, providing extensive category coverage, content diversity, and measuring comprehensiveness. Additionally, we evaluate several popular LLMs from different series and with parameter sizes. In general, all these LLMs demonstrated varying degrees of bias. We conduct an in-depth analysis of results, offering novel insights into bias in LLMs.
>
---
#### [new 010] QFFN-BERT: An Empirical Study of Depth, Performance, and Data Efficiency in Hybrid Quantum-Classical Transformers
- **分类: cs.CL; quant-ph**

- **简介: 该论文属于自然语言处理任务，旨在解决传统Transformer中FFN参数多、数据效率低的问题。通过引入量子电路替代FFN，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.02364v1](http://arxiv.org/pdf/2507.02364v1)**

> **作者:** Pilsung Kang
>
> **摘要:** Parameterized quantum circuits (PQCs) have recently emerged as promising components for enhancing the expressibility of neural architectures. In this work, we introduce QFFN-BERT, a hybrid quantum-classical transformer where the feedforward network (FFN) modules of a compact BERT variant are replaced by PQC-based layers. This design is motivated by the dominant parameter contribution of FFNs, which account for approximately two-thirds of the parameters within standard Transformer encoder blocks. While prior studies have primarily integrated PQCs into self-attention modules, our work focuses on the FFN and systematically investigates the trade-offs between PQC depth, expressibility, and trainability. Our final PQC architecture incorporates a residual connection, both $R_Y$ and $R_Z$ rotations, and an alternating entanglement strategy to ensure stable training and high expressibility. Our experiments, conducted on a classical simulator, on the SST-2 and DBpedia benchmarks demonstrate two key findings. First, a carefully configured QFFN-BERT achieves up to 102.0% of the baseline accuracy, surpassing its classical counterpart in a full-data setting while reducing FFN-specific parameters by over 99%. Second, our model exhibits a consistent and competitive edge in few-shot learning scenarios, confirming its potential for superior data efficiency. These results, supported by an ablation study on a non-optimized PQC that failed to learn, confirm that PQCs can serve as powerful and parameter-efficient alternatives to classical FFNs when co-designed with foundational deep learning principles.
>
---
#### [new 011] Measurement of the Granularity of Vowel Production Space By Just Producible Different (JPD) Limens
- **分类: cs.CL**

- **简介: 该论文属于语音研究任务，旨在测量人类对元音听觉空间差异的感知阈值（JPD），通过实验确定两个元音需多远才能被可靠区分。**

- **链接: [http://arxiv.org/pdf/2507.02744v1](http://arxiv.org/pdf/2507.02744v1)**

> **作者:** Peter Viechnicki
>
> **摘要:** A body of work over the past several decades has demonstrated that the complex and coordinated articulatory movements of human vowel production are governed (at least in part)by control mechanisms whose targets are regions of auditory space. Within the target region control at the sub-phonemic level has also been demonstrated. But the degree of accuracy of that control is unknown. The current work investigates this question by asking how far apart must two vowel stimuli lie in auditory space in order to yield reliably different imitations? This distance is termed 'Just Producible Difference' (JPD). The current study uses a vowel mimicry paradigm to derive the first measurement of JPD among two sets of English speakers during front vowel production. JPD is estimated at between 14 and 51 mels in F1 X F2 space. This finding has implications for episodic theories of speech production. It also clarifies the possible structures of human vowel systems, by setting a theoretical lower bound for how close two vowel phonemes may be in a speaker's formant space, and hence a psychophysical explanation of observed trends in number and patterns of possible vowel phonemes.
>
---
#### [new 012] Answer Matching Outperforms Multiple Choice for Language Model Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型评估任务，旨在解决多选题评估不准确的问题。通过生成式答案匹配提升评估有效性，实验表明其更接近人工评分。**

- **链接: [http://arxiv.org/pdf/2507.02856v1](http://arxiv.org/pdf/2507.02856v1)**

> **作者:** Nikhil Chandak; Shashwat Goel; Ameya Prabhu; Moritz Hardt; Jonas Geiping
>
> **备注:** 34 pages, Code is available at https://github.com/nikhilchandak/answer-matching
>
> **摘要:** Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice--but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we annotate MMLU-Pro and GPQA-Diamond to obtain human grading data, and measure the agreement of each evaluation approach. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improving evaluations via answer matching is not merely a conceptual concern: the rankings of several models change significantly when evaluating their free-form responses with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching.
>
---
#### [new 013] GDC Cohort Copilot: An AI Copilot for Curating Cohorts from the Genomic Data Commons
- **分类: cs.CL**

- **简介: 该论文提出GDC Cohort Copilot，解决用户难以通过复杂字段创建特定队列的问题。通过自然语言生成过滤器，简化队列构建过程。**

- **链接: [http://arxiv.org/pdf/2507.02221v1](http://arxiv.org/pdf/2507.02221v1)**

> **作者:** Steven Song; Anirudh Subramanyam; Zhenyu Zhang; Aarti Venkat; Robert L. Grossman
>
> **备注:** 11 pages, 1 figure, 7 tables
>
> **摘要:** Motivation: The Genomic Data Commons (GDC) provides access to high quality, harmonized cancer genomics data through a unified curation and analysis platform centered around patient cohorts. While GDC users can interactively create complex cohorts through the graphical Cohort Builder, users (especially new ones) may struggle to find specific cohort descriptors across hundreds of possible fields and properties. However, users may be better able to describe their desired cohort in free-text natural language. Results: We introduce GDC Cohort Copilot, an open-source copilot tool for curating cohorts from the GDC. GDC Cohort Copilot automatically generates the GDC cohort filter corresponding to a user-input natural language description of their desired cohort, before exporting the cohort back to the GDC for further analysis. An interactive user interface allows users to further refine the generated cohort. We develop and evaluate multiple large language models (LLMs) for GDC Cohort Copilot and demonstrate that our locally-served, open-source GDC Cohort LLM achieves better results than GPT-4o prompting in generating GDC cohorts. Availability and implementation: The standalone docker image for GDC Cohort Copilot is available at https://quay.io/repository/cdis/gdc-cohort-copilot. Source code is available at https://github.com/uc-cdis/gdc-cohort-copilot. GDC Cohort LLM weights are available at https://huggingface.co/uc-ctds.
>
---
#### [new 014] LLM Hypnosis: Exploiting User Feedback for Unauthorized Knowledge Injection to All Users
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于语言模型安全研究，揭示用户反馈可被利用进行知识注入攻击，解决模型行为被恶意操控的问题，通过实验验证了攻击的有效性。**

- **链接: [http://arxiv.org/pdf/2507.02850v1](http://arxiv.org/pdf/2507.02850v1)**

> **作者:** Almog Hilel; Idan Shenfeld; Leshem Choshen; Jacob Andreas
>
> **摘要:** We describe a vulnerability in language models (LMs) trained with user feedback, whereby a single user can persistently alter LM knowledge and behavior given only the ability to provide prompts and upvote / downvote feedback on LM outputs. To implement the attack, the attacker prompts the LM to stochastically output either a "poisoned" or benign response, then upvotes the poisoned response or downvotes the benign one. When feedback signals are used in a subsequent preference tuning behavior, LMs exhibit increased probability of producing poisoned responses even in contexts without malicious prompts. We show that this attack can be used to (1) insert factual knowledge the model did not previously possess, (2) modify code generation patterns in ways that introduce exploitable security flaws, and (3) inject fake financial news. Our finding both identifies a new qualitative feature of language model preference tuning (showing that it even highly restricted forms of preference data can be used to exert fine-grained control over behavior), and a new attack mechanism for LMs trained with user feedback (extending work on pretraining-time data poisoning and deployment-time prompt injection).
>
---
#### [new 015] Coling-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对科学视觉问答任务，提出结合多模态大模型和少量示例检索的集成方法，以提高回答准确性。**

- **链接: [http://arxiv.org/pdf/2507.02357v1](http://arxiv.org/pdf/2507.02357v1)**

> **作者:** Christian Jaumann; Annemarie Friedrich; Rainer Lienhart
>
> **备注:** Accepted at 5th Workshop on Scholarly Document Processing @ ACL 2025
>
> **摘要:** This paper describes our system for the SciVQA 2025 Shared Task on Scientific Visual Question Answering. Our system employs an ensemble of two Multimodal Large Language Models and various few-shot example retrieval strategies. The model and few-shot setting are selected based on the figure and question type. We also select answers based on the models' confidence levels. On the blind test data, our system ranks third out of seven with an average F1 score of 85.12 across ROUGE-1, ROUGE-L, and BERTS. Our code is publicly available.
>
---
#### [new 016] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型微调任务，旨在解决持续领域自适应预训练中的计算成本高、数据顺序敏感及模型泛化性差的问题。提出DoMIX框架，利用LoRA实现高效、并行的领域适配。**

- **链接: [http://arxiv.org/pdf/2507.02302v1](http://arxiv.org/pdf/2507.02302v1)**

> **作者:** Dohoon Kim; Donghun Kang; Taesup Moon
>
> **备注:** 22 pages, 5 figures, ACL 2025 Main
>
> **摘要:** Domain-Adaptive Pre-training (DAP) has recently gained attention for its effectiveness in fine-tuning pre-trained models. Building on this, continual DAP has been explored to develop pre-trained models capable of incrementally incorporating different domain datasets. However, existing continual DAP methods face several limitations: (1) high computational cost and GPU memory usage during training; (2) sensitivity to incremental data order; and (3) providing a single, generalized model for all end tasks, which contradicts the essence of DAP. In this paper, we propose DoMIX, a novel approach that addresses these challenges by leveraging LoRA modules, a representative parameter-efficient fine-tuning (PEFT) method. Our approach enables efficient and parallel domain-adaptive pre-training that is robust to domain order and effectively utilizes accumulated knowledge to provide tailored pre-trained models for specific tasks. We also demonstrate that our method can be extended beyond the DAP setting to standard LLM fine-tuning scenarios. Code is available at https://github.com/dohoonkim-ai/DoMIX.
>
---
#### [new 017] IndianBailJudgments-1200: A Multi-Attribute Dataset for Legal NLP on Indian Bail Orders
- **分类: cs.CL; cs.AI; cs.LG; 91B14, 68T50; I.2.7; K.4.1; K.5.2**

- **简介: 该论文提出IndianBailJudgments-1200数据集，解决印度法律NLP数据不足问题，用于判决预测、摘要等任务。**

- **链接: [http://arxiv.org/pdf/2507.02506v1](http://arxiv.org/pdf/2507.02506v1)**

> **作者:** Sneha Deshmukh; Prathmesh Kamble
>
> **备注:** 9 pages, 9 figures, 2 tables. Dataset available at Hugging Face and GitHub. Submitted to arXiv for open access
>
> **摘要:** Legal NLP remains underdeveloped in regions like India due to the scarcity of structured datasets. We introduce IndianBailJudgments-1200, a new benchmark dataset comprising 1200 Indian court judgments on bail decisions, annotated across 20+ attributes including bail outcome, IPC sections, crime type, and legal reasoning. Annotations were generated using a prompt-engineered GPT-4o pipeline and verified for consistency. This resource supports a wide range of legal NLP tasks such as outcome prediction, summarization, and fairness analysis, and is the first publicly available dataset focused specifically on Indian bail jurisprudence.
>
---
#### [new 018] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐与偏见缓解任务，旨在解决部署后模型偏见问题。通过多视角融合框架MPF，对齐模型输出与人类基准，减少偏差。**

- **链接: [http://arxiv.org/pdf/2507.02595v1](http://arxiv.org/pdf/2507.02595v1)**

> **作者:** Xin Guan; PeiHsin Lin; Zekun Wu; Ze Wang; Ruibo Zhang; Emre Kazim; Adriano Koshiyama
>
> **备注:** Accepted at ICML 2025 AIW Workshop
>
> **摘要:** Multiperspective Fusion (MPF) is a novel posttraining alignment framework for large language models (LLMs) developed in response to the growing need for easy bias mitigation. Built on top of the SAGED pipeline, an automated system for constructing bias benchmarks and extracting interpretable baseline distributions, MPF leverages multiperspective generations to expose and align biases in LLM outputs with nuanced, humanlike baselines. By decomposing baseline, such as sentiment distributions from HR professionals, into interpretable perspective components, MPF guides generation through sampling and balancing of responses, weighted by the probabilities obtained in the decomposition. Empirically, we demonstrate its ability to align LLM sentiment distributions with both counterfactual baselines (absolute equality) and the HR baseline (biased for Top Univeristy), resulting in small KL divergence, reduction of calibration error and generalization to unseen questions. This shows that MPF offers a scalable and interpretable method for alignment and bias mitigation, compatible with deployed LLMs and requiring no extensive prompt engineering or finetuning.
>
---
#### [new 019] Benchmarking Akan ASR Models Across Domain-Specific Datasets: A Comparative Evaluation of Performance, Scalability, and Adaptability
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在评估Akan语言模型在不同领域的泛化能力，通过对比性能、可扩展性和适应性，揭示模型在跨领域场景下的表现差异。**

- **链接: [http://arxiv.org/pdf/2507.02407v1](http://arxiv.org/pdf/2507.02407v1)**

> **作者:** Mark Atta Mensah; Isaac Wiafe; Akon Ekpezu; Justice Kwame Appati; Jamal-Deen Abdulai; Akosua Nyarkoa Wiafe-Akenten; Frank Ernest Yeboah; Gifty Odame
>
> **备注:** This version has been reviewed and accepted for presentation at the Future Technologies Conference (FTC) 2025, to be held on 6 & 7 November 2025 in Munich, Germany. 17 pages, 4 figures, 1 table
>
> **摘要:** Most existing automatic speech recognition (ASR) research evaluate models using in-domain datasets. However, they seldom evaluate how they generalize across diverse speech contexts. This study addresses this gap by benchmarking seven Akan ASR models built on transformer architectures, such as Whisper and Wav2Vec2, using four Akan speech corpora to determine their performance. These datasets encompass various domains, including culturally relevant image descriptions, informal conversations, biblical scripture readings, and spontaneous financial dialogues. A comparison of the word error rate and character error rate highlighted domain dependency, with models performing optimally only within their training domains while showing marked accuracy degradation in mismatched scenarios. This study also identified distinct error behaviors between the Whisper and Wav2Vec2 architectures. Whereas fine-tuned Whisper Akan models led to more fluent but potentially misleading transcription errors, Wav2Vec2 produced more obvious yet less interpretable outputs when encountering unfamiliar inputs. This trade-off between readability and transparency in ASR errors should be considered when selecting architectures for low-resource language (LRL) applications. These findings highlight the need for targeted domain adaptation techniques, adaptive routing strategies, and multilingual training frameworks for Akan and other LRLs.
>
---
#### [new 020] MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本处理任务，旨在解决长文档处理中线性复杂度与性能保持的问题。提出MemAgent框架，通过分段读取和记忆更新策略实现高效长上下文处理。**

- **链接: [http://arxiv.org/pdf/2507.02259v1](http://arxiv.org/pdf/2507.02259v1)**

> **作者:** Hongli Yu; Tinghong Chen; Jiangtao Feng; Jiangjie Chen; Weinan Dai; Qiying Yu; Ya-Qin Zhang; Wei-Ying Ma; Jingjing Liu; Mingxuan Wang; Hao Zhou
>
> **备注:** Project Page: https://memagent-sialab.github.io/
>
> **摘要:** Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test.
>
---
#### [new 021] Exploring Gender Bias Beyond Occupational Titles
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的性别偏见分析任务，旨在解决性别偏见识别与解释问题。研究构建了GenderLexicon数据集，提出框架以评估和解释上下文中的性别偏见。**

- **链接: [http://arxiv.org/pdf/2507.02679v1](http://arxiv.org/pdf/2507.02679v1)**

> **作者:** Ahmed Sabir; Rajesh Sharama
>
> **备注:** Work in progress
>
> **摘要:** In this work, we investigate the correlation between gender and contextual biases, focusing on elements such as action verbs, object nouns, and particularly on occupations. We introduce a novel dataset, GenderLexicon, and a framework that can estimate contextual bias and its related gender bias. Our model can interpret the bias with a score and thus improve the explainability of gender bias. Also, our findings confirm the existence of gender biases beyond occupational stereotypes. To validate our approach and demonstrate its effectiveness, we conduct evaluations on five diverse datasets, including a Japanese dataset.
>
---
#### [new 022] Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话摘要任务，旨在评估推理型与非推理型大语言模型的效果。研究发现，推理模型在对话摘要中表现不佳，存在冗长和不一致问题。**

- **链接: [http://arxiv.org/pdf/2507.02145v1](http://arxiv.org/pdf/2507.02145v1)**

> **作者:** Keyan Jin; Yapeng Wang; Leonel Santos; Tao Fang; Xu Yang; Sio Kei Im; Hugo Gonçalo Oliveira
>
> **摘要:** Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization.
>
---
#### [new 023] Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型安全研究，旨在探讨推理语言模型在社会偏见下的鲁棒性。通过实验分析不同推理机制对模型公平性和安全的影响。**

- **链接: [http://arxiv.org/pdf/2507.02799v1](http://arxiv.org/pdf/2507.02799v1)**

> **作者:** Riccardo Cantini; Nicola Gabriele; Alessio Orsino; Domenico Talia
>
> **摘要:** Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.
>
---
#### [new 024] Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers
- **分类: cs.CL**

- **简介: 该论文属于AI辅助科研任务，旨在解决LLMs识别科学论文局限性的问题。研究构建了LimitGen基准，包含合成与真实数据，提升LLMs在早期反馈中的能力。**

- **链接: [http://arxiv.org/pdf/2507.02694v1](http://arxiv.org/pdf/2507.02694v1)**

> **作者:** Zhijian Xu; Yilun Zhao; Manasi Patwardhan; Lovekesh Vig; Arman Cohan
>
> **摘要:** Peer review is fundamental to scientific research, but the growing volume of publications has intensified the challenges of this expertise-intensive process. While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. We first present a comprehensive taxonomy of limitation types in scientific research, with a focus on AI. Guided by this taxonomy, for studying limitations, we present LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of high-quality papers, and LimitGen-Human, a collection of real human-written limitations. To improve the ability of LLM systems to identify limitations, we augment them with literature retrieval, which is essential for grounding identifying limitations in prior scientific findings. Our approach enhances the capabilities of LLM systems to generate limitations in research papers, enabling them to provide more concrete and constructive feedback.
>
---
#### [new 025] WebSailor: Navigating Super-human Reasoning for Web Agent
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决开放源代码模型在复杂搜索任务中的能力不足。通过引入WebSailor方法，提升模型处理高不确定性信息的能力。**

- **链接: [http://arxiv.org/pdf/2507.02592v1](http://arxiv.org/pdf/2507.02592v1)**

> **作者:** Kuan Li; Zhongwang Zhang; Huifeng Yin; Liwen Zhang; Litu Ou; Jialong Wu; Wenbiao Yin; Baixuan Li; Zhengwei Tao; Xinyu Wang; Weizhou Shen; Junkai Zhang; Dingchu Zhang; Xixi Wu; Yong Jiang; Ming Yan; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **摘要:** Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.
>
---
#### [new 026] Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在自我纠错中的盲点问题。通过构建基准测试框架，分析错误修正能力，并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2507.02778v1](http://arxiv.org/pdf/2507.02778v1)**

> **作者:** Ken Tsui
>
> **备注:** 31 pages, 18 figures
>
> **摘要:** Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness.
>
---
#### [new 027] Energy-Based Transformers are Scalable Learners and Thinkers
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Energy-Based Transformers（EBTs），解决模型学习与推理能力提升问题。通过无监督学习实现系统2思维，提升语言和图像任务性能。**

- **链接: [http://arxiv.org/pdf/2507.02092v1](http://arxiv.org/pdf/2507.02092v1)**

> **作者:** Alexi Gladstone; Ganesh Nanduru; Md Mofijul Islam; Peixuan Han; Hyeonjeong Ha; Aman Chadha; Yilun Du; Heng Ji; Jundong Li; Tariq Iqbal
>
> **摘要:** Inference-time computation techniques, analogous to human System 2 Thinking, have recently become popular for improving model performances. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" Interestingly, we find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs) -- a new class of Energy-Based Models (EBMs) -- to assign an energy value to every input and candidate-prediction pair, enabling predictions through gradient descent-based energy minimization until convergence. Across both discrete (text) and continuous (visual) modalities, we find EBTs scale faster than the dominant Transformer++ approach during training, achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth. During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks, and EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes. Further, we find that EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches. Consequently, EBTs are a promising new paradigm for scaling both the learning and thinking capabilities of models.
>
---
#### [new 028] Legal Requirements Translation from Law
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于法律文本处理任务，旨在解决法律要求自动提取问题。通过文本蕴含和上下文学习生成可执行的法律元数据表示，提升合规性检查效率。**

- **链接: [http://arxiv.org/pdf/2507.02846v1](http://arxiv.org/pdf/2507.02846v1)**

> **作者:** Anmol Singhal; Travis Breaux
>
> **备注:** 13 pages, 7 figures, Accepted at the 33rd IEEE International Requirements Engineering 2025
>
> **摘要:** Software systems must comply with legal regulations, which is a resource-intensive task, particularly for small organizations and startups lacking dedicated legal expertise. Extracting metadata from regulations to elicit legal requirements for software is a critical step to ensure compliance. However, it is a cumbersome task due to the length and complex nature of legal text. Although prior work has pursued automated methods for extracting structural and semantic metadata from legal text, key limitations remain: they do not consider the interplay and interrelationships among attributes associated with these metadata types, and they rely on manual labeling or heuristic-driven machine learning, which does not generalize well to new documents. In this paper, we introduce an approach based on textual entailment and in-context learning for automatically generating a canonical representation of legal text, encodable and executable as Python code. Our representation is instantiated from a manually designed Python class structure that serves as a domain-specific metamodel, capturing both structural and semantic legal metadata and their interrelationships. This design choice reduces the need for large, manually labeled datasets and enhances applicability to unseen legislation. We evaluate our approach on 13 U.S. state data breach notification laws, demonstrating that our generated representations pass approximately 89.4% of test cases and achieve a precision and recall of 82.2 and 88.7, respectively.
>
---
#### [new 029] ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频自监督表示学习任务，旨在解决Transformer注意力机制分配无效注意力的问题，提出ASDA模型通过差分注意力机制提升性能。**

- **链接: [http://arxiv.org/pdf/2507.02666v1](http://arxiv.org/pdf/2507.02666v1)**

> **作者:** Junyu Wang; Tianrui Wang; Meng Ge; Longbiao Wang; Jianwu Dang
>
> **备注:** Accepted at Interspeech2025
>
> **摘要:** In recent advancements in audio self-supervised representation learning, the standard Transformer architecture has emerged as the predominant approach, yet its attention mechanism often allocates a portion of attention weights to irrelevant information, potentially impairing the model's discriminative ability. To address this, we introduce a differential attention mechanism, which effectively mitigates ineffective attention allocation through the integration of dual-softmax operations and appropriately tuned differential coefficients. Experimental results demonstrate that our ASDA model achieves state-of-the-art (SOTA) performance across multiple benchmarks, including audio classification (49.0% mAP on AS-2M, 41.5% mAP on AS20K), keyword spotting (98.3% accuracy on SPC-2), and environmental sound classification (96.1% accuracy on ESC-50). These results highlight ASDA's effectiveness in audio tasks, paving the way for broader applications.
>
---
#### [new 030] Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency
- **分类: cs.OS; cs.CL**

- **简介: 该论文属于移动设备上大语言模型优化任务，解决LLM推理中能量效率与延迟问题。通过分析DVFS调节器的协同作用，设计FUSE框架提升能效。**

- **链接: [http://arxiv.org/pdf/2507.02135v1](http://arxiv.org/pdf/2507.02135v1)**

> **作者:** Zongpu Zhang; Pranab Dash; Y. Charlie Hu; Qiang Xu; Jian Li; Haibing Guan
>
> **备注:** equal contribution between Zhang and Dash
>
> **摘要:** Large Language Models (LLMs) are increasingly being integrated into various applications and services running on billions of mobile devices. However, deploying LLMs on resource-limited mobile devices faces a significant challenge due to their high demand for computation, memory, and ultimately energy. While current LLM frameworks for mobile use three power-hungry components-CPU, GPU, and Memory-even when running primarily-GPU LLM models, optimized DVFS governors for CPU, GPU, and memory featured in modern mobile devices operate independently and are oblivious of each other. Motivated by the above observation, in this work, we first measure the energy-efficiency of a SOTA LLM framework consisting of various LLM models on mobile phones which showed the triplet mobile governors result in up to 40.4% longer prefilling and decoding latency compared to optimal combinations of CPU, GPU, and memory frequencies with the same energy consumption for sampled prefill and decode lengths. Second, we conduct an in-depth measurement study to uncover how the intricate interplay (or lack of) among the mobile governors cause the above inefficiency in LLM inference. Finally, based on these insights, we design FUSE - a unified energy-aware governor for optimizing the energy efficiency of LLM inference on mobile devices. Our evaluation using a ShareGPT dataset shows FUSE reduces the time-to-first-token and time-per-output-token latencies by 7.0%-16.9% and 25.4%-36.8% on average with the same energy-per-token for various mobile LLM models.
>
---
#### [new 031] FinAI-BERT: A Transformer-Based Model for Sentence-Level Detection of AI Disclosures in Financial Reports
- **分类: q-fin.CP; cs.CL; econ.GN; q-fin.EC; q-fin.GN**

- **简介: 该论文属于金融文本分类任务，旨在精准识别财务报告中的AI披露内容。通过构建FinAI-BERT模型，实现句子级别的AI相关声明检测。**

- **链接: [http://arxiv.org/pdf/2507.01991v1](http://arxiv.org/pdf/2507.01991v1)**

> **作者:** Muhammad Bilal Zafar
>
> **备注:** The FinAI-BERT model can be directly loaded via Hugging Face Transformers (https://huggingface.co/bilalzafar/FinAI-BERT) for sentence-level AI disclosure classification
>
> **摘要:** The proliferation of artificial intelligence (AI) in financial services has prompted growing demand for tools that can systematically detect AI-related disclosures in corporate filings. While prior approaches often rely on keyword expansion or document-level classification, they fall short in granularity, interpretability, and robustness. This study introduces FinAI-BERT, a domain-adapted transformer-based language model designed to classify AI-related content at the sentence level within financial texts. The model was fine-tuned on a manually curated and balanced dataset of 1,586 sentences drawn from 669 annual reports of U.S. banks (2015 to 2023). FinAI-BERT achieved near-perfect classification performance (accuracy of 99.37 percent, F1 score of 0.993), outperforming traditional baselines such as Logistic Regression, Naive Bayes, Random Forest, and XGBoost. Interpretability was ensured through SHAP-based token attribution, while bias analysis and robustness checks confirmed the model's stability across sentence lengths, adversarial inputs, and temporal samples. Theoretically, the study advances financial NLP by operationalizing fine-grained, theme-specific classification using transformer architectures. Practically, it offers a scalable, transparent solution for analysts, regulators, and scholars seeking to monitor the diffusion and framing of AI across financial institutions.
>
---
#### [new 032] StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中的近似奖励问题和探索停滞问题。提出StepHint算法，通过多级提示提升模型推理能力与训练效率。**

- **链接: [http://arxiv.org/pdf/2507.02841v1](http://arxiv.org/pdf/2507.02841v1)**

> **作者:** Kaiyi Zhang; Ang Lv; Jinpeng Li; Yongbo Wang; Feng Wang; Haoyuan Hu; Rui Yan
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving the complex reasoning abilities of large language models (LLMs). However, current RLVR methods face two significant challenges: the near-miss reward problem, where a small mistake can invalidate an otherwise correct reasoning process, greatly hindering training efficiency; and exploration stagnation, where models tend to focus on solutions within their ``comfort zone,'' lacking the motivation to explore potentially more effective alternatives. To address these challenges, we propose StepHint, a novel RLVR algorithm that utilizes multi-level stepwise hints to help models explore the solution space more effectively. StepHint generates valid reasoning chains from stronger models and partitions these chains into reasoning steps using our proposed adaptive partitioning method. The initial few steps are used as hints, and simultaneously, multiple-level hints (each comprising a different number of steps) are provided to the model. This approach directs the model's exploration toward a promising solution subspace while preserving its flexibility for independent exploration. By providing hints, StepHint mitigates the near-miss reward problem, thereby improving training efficiency. Additionally, the external reasoning pathways help the model develop better reasoning abilities, enabling it to move beyond its ``comfort zone'' and mitigate exploration stagnation. StepHint outperforms competitive RLVR enhancement methods across six mathematical benchmarks, while also demonstrating superior generalization and excelling over baselines on out-of-domain benchmarks.
>
---
#### [new 033] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升模型在复杂推理任务中的表现。针对传统方法依赖初始样本的问题，提出ExPO框架，通过生成高质量样本提升学习效率和性能。**

- **链接: [http://arxiv.org/pdf/2507.02834v1](http://arxiv.org/pdf/2507.02834v1)**

> **作者:** Ruiyang Zhou; Shuozhe Li; Amy Zhang; Liu Leqi
>
> **摘要:** Recent advances in large language models have been driven by reinforcement learning (RL)-style post-training, which improves reasoning by optimizing model outputs based on reward or preference signals. GRPO-style approaches implement this by using self-generated samples labeled by an outcome-based verifier. However, these methods depend heavily on the model's initial ability to produce positive samples. They primarily refine what the model already knows (distribution sharpening) rather than enabling the model to solve problems where it initially fails. This limitation is especially problematic in early-stage RL training and on challenging reasoning tasks, where positive samples are unlikely to be generated. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most.
>
---
#### [new 034] OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理任务，解决在线部署中draft模型与目标模型不兼容及延迟问题。提出OmniDraft框架，实现跨词汇适配和动态优化，提升解码速度。**

- **链接: [http://arxiv.org/pdf/2507.02659v1](http://arxiv.org/pdf/2507.02659v1)**

> **作者:** Ramchalam Kinattinkara Ramakrishnan; Zhaocong Yuan; Shaojie Zhuo; Chen Feng; Yicheng Lin; Chenzheng Su; Xiaopeng Zhang
>
> **摘要:** Speculative decoding generally dictates having a small, efficient draft model that is either pretrained or distilled offline to a particular target model series, for instance, Llama or Qwen models. However, within online deployment settings, there are two major challenges: 1) usage of a target model that is incompatible with the draft model; 2) expectation of latency improvements over usage and time. In this work, we propose OmniDraft, a unified framework that enables a single draft model to operate with any target model and adapt dynamically to user data. We introduce an online n-gram cache with hybrid distillation fine-tuning to address the cross-vocabulary mismatch across draft and target models; and further improve decoding speed by leveraging adaptive drafting techniques. OmniDraft is particularly suitable for on-device LLM applications where model cost, efficiency and user customization are the major points of contention. This further highlights the need to tackle the above challenges and motivates the \textit{``one drafter for all''} paradigm. We showcase the proficiency of the OmniDraft framework by performing online learning on math reasoning, coding and text generation tasks. Notably, OmniDraft enables a single Llama-68M model to pair with various target models including Vicuna-7B, Qwen2-7B and Llama3-8B models for speculative decoding; and additionally provides up to 1.5-2x speedup.
>
---
#### [new 035] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于招聘领域的AI应用研究，旨在解决LLMs在招聘中准确性与公平性问题。通过对比不同模型，验证了定制模型在准确性和公平性上的优势。**

- **链接: [http://arxiv.org/pdf/2507.02087v1](http://arxiv.org/pdf/2507.02087v1)**

> **作者:** Eitan Anzenberg; Arunava Samajpati; Sivasankaran Chandrasekar; Varun Kacholia
>
> **备注:** 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
>
> **摘要:** The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
>
---
#### [new 036] Why Multi-Interest Fairness Matters: Hypergraph Contrastive Multi-Interest Learning for Fair Conversational Recommender System
- **分类: cs.IR; cs.CL; cs.MM**

- **简介: 该论文属于推荐系统任务，旨在解决对话推荐系统中的公平性问题。通过构建多兴趣图模型，提升推荐多样性与公平性。**

- **链接: [http://arxiv.org/pdf/2507.02000v1](http://arxiv.org/pdf/2507.02000v1)**

> **作者:** Yongsen Zheng; Zongxuan Xie; Guohua Wang; Ziyao Liu; Liang Lin; Kwok-Yan Lam
>
> **摘要:** Unfairness is a well-known challenge in Recommender Systems (RSs), often resulting in biased outcomes that disadvantage users or items based on attributes such as gender, race, age, or popularity. Although some approaches have started to improve fairness recommendation in offline or static contexts, the issue of unfairness often exacerbates over time, leading to significant problems like the Matthew effect, filter bubbles, and echo chambers. To address these challenges, we proposed a novel framework, Hypergraph Contrastive Multi-Interest Learning for Fair Conversational Recommender System (HyFairCRS), aiming to promote multi-interest diversity fairness in dynamic and interactive Conversational Recommender Systems (CRSs). HyFairCRS first captures a wide range of user interests by establishing diverse hypergraphs through contrastive learning. These interests are then utilized in conversations to generate informative responses and ensure fair item predictions within the dynamic user-system feedback loop. Experiments on two CRS-based datasets show that HyFairCRS achieves a new state-of-the-art performance while effectively alleviating unfairness. Our code is available at https://github.com/zysensmile/HyFairCRS.
>
---
#### [new 037] SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出SciGA-145k数据集，用于支持图形摘要的设计与推荐，解决可视化科学传播难题。**

- **链接: [http://arxiv.org/pdf/2507.02212v1](http://arxiv.org/pdf/2507.02212v1)**

> **作者:** Takuro Kawada; Shunsuke Kitada; Sota Nemoto; Hitoshi Iyatomi
>
> **备注:** 21 pages, 15 figures, 4 tables. Project Page: https://iyatomilab.github.io/SciGA/
>
> **摘要:** Graphical Abstracts (GAs) play a crucial role in visually conveying the key findings of scientific papers. While recent research has increasingly incorporated visual materials such as Figure 1 as de facto GAs, their potential to enhance scientific communication remains largely unexplored. Moreover, designing effective GAs requires advanced visualization skills, creating a barrier to their widespread adoption. To tackle these challenges, we introduce SciGA-145k, a large-scale dataset comprising approximately 145,000 scientific papers and 1.14 million figures, explicitly designed for supporting GA selection and recommendation as well as facilitating research in automated GA generation. As a preliminary step toward GA design support, we define two tasks: 1) Intra-GA recommendation, which identifies figures within a given paper that are well-suited to serve as GAs, and 2) Inter-GA recommendation, which retrieves GAs from other papers to inspire the creation of new GAs. We provide reasonable baseline models for these tasks. Furthermore, we propose Confidence Adjusted top-1 ground truth Ratio (CAR), a novel recommendation metric that offers a fine-grained analysis of model behavior. CAR addresses limitations in traditional ranking-based metrics by considering cases where multiple figures within a paper, beyond the explicitly labeled GA, may also serve as GAs. By unifying these tasks and metrics, our SciGA-145k establishes a foundation for advancing visual scientific communication while contributing to the development of AI for Science.
>
---
#### [new 038] Requirements Elicitation Follow-Up Question Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于需求获取任务，旨在解决访谈中生成有效问题的难题。研究利用GPT-4o生成跟进问题，通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.02858v1](http://arxiv.org/pdf/2507.02858v1)**

> **作者:** Yuchen Shen; Anmol Singhal; Travis Breaux
>
> **备注:** 13 pages, 2 figures, accepted at the 33rd IEEE International Requirements Engineering 2025
>
> **摘要:** Interviews are a widely used technique in eliciting requirements to gather stakeholder needs, preferences, and expectations for a software system. Effective interviewing requires skilled interviewers to formulate appropriate interview questions in real time while facing multiple challenges, including lack of familiarity with the domain, excessive cognitive load, and information overload that hinders how humans process stakeholders' speech. Recently, large language models (LLMs) have exhibited state-of-the-art performance in multiple natural language processing tasks, including text summarization and entailment. To support interviewers, we investigate the application of GPT-4o to generate follow-up interview questions during requirements elicitation by building on a framework of common interviewer mistake types. In addition, we describe methods to generate questions based on interviewee speech. We report a controlled experiment to evaluate LLM-generated and human-authored questions with minimal guidance, and a second controlled experiment to evaluate the LLM-generated questions when generation is guided by interviewer mistake types. Our findings demonstrate that, for both experiments, the LLM-generated questions are no worse than the human-authored questions with respect to clarity, relevancy, and informativeness. In addition, LLM-generated questions outperform human-authored questions when guided by common mistakes types. This highlights the potential of using LLMs to help interviewers improve the quality and ease of requirements elicitation interviews in real time.
>
---
#### [new 039] Seeing Through Green: Text-Based Classification and the Firm's Returns from Green Patents
- **分类: econ.GN; cs.CL; q-fin.EC**

- **简介: 该论文属于文本分类任务，旨在识别真正的绿色专利。通过NLP技术验证绿色专利有效性，并分析其对企业绩效的影响。**

- **链接: [http://arxiv.org/pdf/2507.02287v1](http://arxiv.org/pdf/2507.02287v1)**

> **作者:** Lapo Santarlasci; Armando Rungi; Antonio Zinilli
>
> **摘要:** This paper introduces Natural Language Processing for identifying ``true'' green patents from official supporting documents. We start our training on about 12.4 million patents that had been classified as green from previous literature. Thus, we train a simple neural network to enlarge a baseline dictionary through vector representations of expressions related to environmental technologies. After testing, we find that ``true'' green patents represent about 20\% of the total of patents classified as green from previous literature. We show heterogeneity by technological classes, and then check that `true' green patents are about 1\% less cited by following inventions. In the second part of the paper, we test the relationship between patenting and a dashboard of firm-level financial accounts in the European Union. After controlling for reverse causality, we show that holding at least one ``true'' green patent raises sales, market shares, and productivity. If we restrict the analysis to high-novelty ``true'' green patents, we find that they also yield higher profits. Our findings underscore the importance of using text analyses to gauge finer-grained patent classifications that are useful for policymaking in different domains.
>
---
#### [new 040] Strategic Intelligence in Large Language Models: Evidence from evolutionary Game Theory
- **分类: cs.AI; cs.CL; cs.GT**

- **简介: 该论文属于AI战略决策研究，探讨LLMs在博弈中的表现。通过演化IPD实验，分析其策略与推理能力，揭示其在不确定环境下的决策机制。**

- **链接: [http://arxiv.org/pdf/2507.02618v1](http://arxiv.org/pdf/2507.02618v1)**

> **作者:** Kenneth Payne; Baptiste Alloui-Cros
>
> **备注:** 29 pages, 27 tables, 4 figures
>
> **摘要:** Are Large Language Models (LLMs) a new form of strategic intelligence, able to reason about goals in competitive settings? We present compelling supporting evidence. The Iterated Prisoner's Dilemma (IPD) has long served as a model for studying decision-making. We conduct the first ever series of evolutionary IPD tournaments, pitting canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from the leading frontier AI companies OpenAI, Google, and Anthropic. By varying the termination probability in each tournament (the "shadow of the future"), we introduce complexity and chance, confounding memorisation. Our results show that LLMs are highly competitive, consistently surviving and sometimes even proliferating in these complex ecosystems. Furthermore, they exhibit distinctive and persistent "strategic fingerprints": Google's Gemini models proved strategically ruthless, exploiting cooperative opponents and retaliating against defectors, while OpenAI's models remained highly cooperative, a trait that proved catastrophic in hostile environments. Anthropic's Claude emerged as the most forgiving reciprocator, showing remarkable willingness to restore cooperation even after being exploited or successfully defecting. Analysis of nearly 32,000 prose rationales provided by the models reveals that they actively reason about both the time horizon and their opponent's likely strategy, and we demonstrate that this reasoning is instrumental to their decisions. This work connects classic game theory with machine psychology, offering a rich and granular view of algorithmic decision-making under uncertainty.
>
---
#### [new 041] Early Signs of Steganographic Capabilities in Frontier LLMs
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全风险研究任务，旨在评估大语言模型的隐写能力。研究发现当前模型在特定条件下可进行简单隐写和编码推理，但尚不足以绕过有效监控。**

- **链接: [http://arxiv.org/pdf/2507.02737v1](http://arxiv.org/pdf/2507.02737v1)**

> **作者:** Artur Zolkowski; Kei Nishimura-Gasparian; Robert McCarthy; Roland S. Zimmermann; David Lindner
>
> **摘要:** Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future.
>
---
#### [new 042] From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频编辑任务，旨在解决自动剪辑缺乏连贯性的问题。提出HIVE框架，结合多模态理解与场景分割，提升剪辑质量。**

- **链接: [http://arxiv.org/pdf/2507.02790v1](http://arxiv.org/pdf/2507.02790v1)**

> **作者:** Xiangfeng Wang; Xiao Li; Yadong Wei; Xueyu Song; Yang Song; Xiaoqiang Xia; Fangrui Zeng; Zaiyi Chen; Liu Liu; Gu Xu; Tong Xu
>
> **摘要:** The rapid growth of online video content, especially on short video platforms, has created a growing demand for efficient video editing techniques that can condense long-form videos into concise and engaging clips. Existing automatic editing methods predominantly rely on textual cues from ASR transcripts and end-to-end segment selection, often neglecting the rich visual context and leading to incoherent outputs. In this paper, we propose a human-inspired automatic video editing framework (HIVE) that leverages multimodal narrative understanding to address these limitations. Our approach incorporates character extraction, dialogue analysis, and narrative summarization through multimodal large language models, enabling a holistic understanding of the video content. To further enhance coherence, we apply scene-level segmentation and decompose the editing process into three subtasks: highlight detection, opening/ending selection, and pruning of irrelevant content. To facilitate research in this area, we introduce DramaAD, a novel benchmark dataset comprising over 800 short drama episodes and 500 professionally edited advertisement clips. Experimental results demonstrate that our framework consistently outperforms existing baselines across both general and advertisement-oriented editing tasks, significantly narrowing the quality gap between automatic and human-edited videos.
>
---
#### [new 043] JoyTTS: LLM-based Spoken Chatbot With Voice Cloning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音对话任务，旨在实现基于大语言模型的语音聊天机器人，并解决语音克隆问题。通过结合LLM与TTS技术，构建了JoyTTS系统。**

- **链接: [http://arxiv.org/pdf/2507.02380v1](http://arxiv.org/pdf/2507.02380v1)**

> **作者:** Fangru Zhou; Jun Zhao; Guoxin Wang
>
> **摘要:** JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at https://github.com/jdh-algo/JoyTTS.git.
>
---
#### [new 044] STELLA: Self-Evolving LLM Agent for Biomedical Research
- **分类: cs.AI; cs.CL; q-bio.BM**

- **简介: 该论文提出STELLA，一种自进化AI代理，用于解决生物医学研究中工具和数据碎片化问题。通过动态扩展工具库和优化推理策略，提升研究效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.02004v1](http://arxiv.org/pdf/2507.02004v1)**

> **作者:** Ruofan Jin; Zaixi Zhang; Mengdi Wang; Le Cong
>
> **摘要:** The rapid growth of biomedical data, tools, and literature has created a fragmented research landscape that outpaces human expertise. While AI agents offer a solution, they typically rely on static, manually curated toolsets, limiting their ability to adapt and scale. Here, we introduce STELLA, a self-evolving AI agent designed to overcome these limitations. STELLA employs a multi-agent architecture that autonomously improves its own capabilities through two core mechanisms: an evolving Template Library for reasoning strategies and a dynamic Tool Ocean that expands as a Tool Creation Agent automatically discovers and integrates new bioinformatics tools. This allows STELLA to learn from experience. We demonstrate that STELLA achieves state-of-the-art accuracy on a suite of biomedical benchmarks, scoring approximately 26\% on Humanity's Last Exam: Biomedicine, 54\% on LAB-Bench: DBQA, and 63\% on LAB-Bench: LitQA, outperforming leading models by up to 6 percentage points. More importantly, we show that its performance systematically improves with experience; for instance, its accuracy on the Humanity's Last Exam benchmark almost doubles with increased trials. STELLA represents a significant advance towards AI Agent systems that can learn and grow, dynamically scaling their expertise to accelerate the pace of biomedical discovery.
>
---
#### [new 045] Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在通过视觉上下文注入破解多模态大语言模型，提出VisCo攻击方法提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.02844v1](http://arxiv.org/pdf/2507.02844v1)**

> **作者:** Ziqi Miao; Yi Ding; Lijun Li; Jing Shao
>
> **备注:** 16 pages
>
> **摘要:** With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack.
>
---
#### [new 046] DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出DeSTA2.5-Audio，一种通用大音频语言模型，解决音频感知与指令跟随问题，通过自生成跨模态对齐策略提升性能。**

- **链接: [http://arxiv.org/pdf/2507.02768v1](http://arxiv.org/pdf/2507.02768v1)**

> **作者:** Ke-Han Lu; Zhehuai Chen; Szu-Wei Fu; Chao-Han Huck Yang; Sung-Feng Huang; Chih-Kai Yang; Chee-En Yu; Chun-Wei Chen; Wei-Chih Chen; Chien-yu Huang; Yi-Cheng Lin; Yu-Xiang Lin; Chi-An Fu; Chun-Yi Kuan; Wenze Ren; Xuanjun Chen; Wei-Ping Huang; En-Pei Hu; Tzu-Quan Lin; Yuan-Kuei Wu; Kuan-Po Huang; Hsiao-Ying Huang; Huang-Cheng Chou; Kai-Wei Chang; Cheng-Han Chiang; Boris Ginsburg; Yu-Chiang Frank Wang; Hung-yi Lee
>
> **备注:** Model and code available at: https://github.com/kehanlu/DeSTA2.5-Audio
>
> **摘要:** We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following, without requiring task-specific audio instruction-tuning. Recent LALMs typically augment Large Language Models (LLMs) with auditory capabilities by training on large-scale, manually curated or LLM-synthesized audio-instruction datasets. However, these approaches have often suffered from the catastrophic forgetting of the LLM's original language abilities. To address this, we revisit the data construction pipeline and propose DeSTA, a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets. This approach preserves the LLM's native language proficiency while establishing effective audio-text alignment, thereby enabling zero-shot generalization without task-specific tuning. Using DeSTA, we construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms widely adopted data construction and training strategies in both auditory perception and instruction-following capabilities. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs.
>
---
#### [new 047] ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于事件流场景文本识别任务，旨在解决识别准确性和可解释性不足的问题。提出ESTR-CoT框架，结合视觉编码与语言模型，实现端到端优化和逻辑推理。**

- **链接: [http://arxiv.org/pdf/2507.02200v1](http://arxiv.org/pdf/2507.02200v1)**

> **作者:** Xiao Wang; Jingtao Jiang; Qiang Chen; Lan Chen; Lin Zhu; Yaowei Wang; Yonghong Tian; Jin Tang
>
> **备注:** A Strong Baseline for Reasoning based Event Stream Scene Text Recognition
>
> **摘要:** Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on https://github.com/Event-AHU/ESTR-CoT.
>
---
#### [new 048] Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音合成中的说话人相似性评估任务，旨在解决ASV嵌入忽略动态特征的问题，提出U3D度量动态节奏模式。**

- **链接: [http://arxiv.org/pdf/2507.02176v1](http://arxiv.org/pdf/2507.02176v1)**

> **作者:** Marc-André Carbonneau; Benjamin van Niekerk; Hugo Seuté; Jean-Philippe Letendre; Herman Kamper; Julian Zaïdi
>
> **备注:** Accepted at SSW13 - Interspeech 2025 Speech Synthesis Workshop
>
> **摘要:** Modeling voice identity is challenging due to its multifaceted nature. In generative speech systems, identity is often assessed using automatic speaker verification (ASV) embeddings, designed for discrimination rather than characterizing identity. This paper investigates which aspects of a voice are captured in such representations. We find that widely used ASV embeddings focus mainly on static features like timbre and pitch range, while neglecting dynamic elements such as rhythm. We also identify confounding factors that compromise speaker similarity measurements and suggest mitigation strategies. To address these gaps, we propose U3D, a metric that evaluates speakers' dynamic rhythm patterns. This work contributes to the ongoing challenge of assessing speaker identity consistency in the context of ever-better voice cloning systems. We publicly release our code.
>
---
#### [new 049] Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决传统系统在处理复杂搜索时推理效率低的问题。提出HiRA框架，分离规划与执行，提升搜索效果和效率。**

- **链接: [http://arxiv.org/pdf/2507.02652v1](http://arxiv.org/pdf/2507.02652v1)**

> **作者:** Jiajie Jin; Xiaoxi Li; Guanting Dong; Yuyao Zhang; Yutao Zhu; Yang Zhao; Hongjin Qian; Zhicheng Dou
>
> **备注:** 9 pages
>
> **摘要:** Complex information needs in real-world search scenarios demand deep reasoning and knowledge synthesis across diverse sources, which traditional retrieval-augmented generation (RAG) pipelines struggle to address effectively. Current reasoning-based approaches suffer from a fundamental limitation: they use a single model to handle both high-level planning and detailed execution, leading to inefficient reasoning and limited scalability. In this paper, we introduce HiRA, a hierarchical framework that separates strategic planning from specialized execution. Our approach decomposes complex search tasks into focused subtasks, assigns each subtask to domain-specific agents equipped with external tools and reasoning capabilities, and coordinates the results through a structured integration mechanism. This separation prevents execution details from disrupting high-level reasoning while enabling the system to leverage specialized expertise for different types of information processing. Experiments on four complex, cross-modal deep search benchmarks demonstrate that HiRA significantly outperforms state-of-the-art RAG and agent-based systems. Our results show improvements in both answer quality and system efficiency, highlighting the effectiveness of decoupled planning and execution for multi-step information seeking tasks. Our code is available at https://github.com/ignorejjj/HiRA.
>
---
#### [new 050] Multimodal Misinformation Detection Using Early Fusion of Linguistic, Visual, and Social Features
- **分类: cs.LG; cs.CL; cs.SI**

- **简介: 该论文属于多模态信息检测任务，旨在解决社交媒体中虚假信息识别问题。通过融合文本、图像和社会特征，提升分类模型性能。**

- **链接: [http://arxiv.org/pdf/2507.01984v1](http://arxiv.org/pdf/2507.01984v1)**

> **作者:** Gautam Kishore Shahi
>
> **摘要:** Amid a tidal wave of misinformation flooding social media during elections and crises, extensive research has been conducted on misinformation detection, primarily focusing on text-based or image-based approaches. However, only a few studies have explored multimodal feature combinations, such as integrating text and images for building a classification model to detect misinformation. This study investigates the effectiveness of different multimodal feature combinations, incorporating text, images, and social features using an early fusion approach for the classification model. This study analyzed 1,529 tweets containing both text and images during the COVID-19 pandemic and election periods collected from Twitter (now X). A data enrichment process was applied to extract additional social features, as well as visual features, through techniques such as object detection and optical character recognition (OCR). The results show that combining unsupervised and supervised machine learning models improves classification performance by 15% compared to unimodal models and by 5% compared to bimodal models. Additionally, the study analyzes the propagation patterns of misinformation based on the characteristics of misinformation tweets and the users who disseminate them.
>
---
## 更新

#### [replaced 001] Delving into LLM-assisted writing in biomedical publications through excess vocabulary
- **分类: cs.CL; cs.AI; cs.CY; cs.DL; cs.SI**

- **链接: [http://arxiv.org/pdf/2406.07016v5](http://arxiv.org/pdf/2406.07016v5)**

> **作者:** Dmitry Kobak; Rita González-Márquez; Emőke-Ágnes Horvát; Jan Lause
>
> **备注:** v5: Reverting to v3
>
> **摘要:** Large language models (LLMs) like ChatGPT can generate and revise text with human-level performance. These models come with clear limitations: they can produce inaccurate information, reinforce existing biases, and be easily misused. Yet, many scientists use them for their scholarly writing. But how wide-spread is such LLM usage in the academic literature? To answer this question for the field of biomedical research, we present an unbiased, large-scale approach: we study vocabulary changes in over 15 million biomedical abstracts from 2010--2024 indexed by PubMed, and show how the appearance of LLMs led to an abrupt increase in the frequency of certain style words. This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs. This lower bound differed across disciplines, countries, and journals, reaching 40% for some subcorpora. We show that LLMs have had an unprecedented impact on scientific writing in biomedical research, surpassing the effect of major world events such as the Covid pandemic.
>
---
#### [replaced 002] Improved Unbiased Watermark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11268v2](http://arxiv.org/pdf/2502.11268v2)**

> **作者:** Ruibo Chen; Yihan Wu; Junfeng Guo; Heng Huang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** As artificial intelligence surpasses human capabilities in text generation, the necessity to authenticate the origins of AI-generated content has become paramount. Unbiased watermarks offer a powerful solution by embedding statistical signals into language model-generated text without distorting the quality. In this paper, we introduce MCmark, a family of unbiased, Multi-Channel-based watermarks. MCmark works by partitioning the model's vocabulary into segments and promoting token probabilities within a selected segment based on a watermark key. We demonstrate that MCmark not only preserves the original distribution of the language model but also offers significant improvements in detectability and robustness over existing unbiased watermarks. Our experiments with widely-used language models demonstrate an improvement in detectability of over 10% using MCmark, compared to existing state-of-the-art unbiased watermarks. This advancement underscores MCmark's potential in enhancing the practical application of watermarking in AI-generated texts.
>
---
#### [replaced 003] Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.00606v2](http://arxiv.org/pdf/2507.00606v2)**

> **作者:** Tao Xiong; Xavier Hu; Wenyan Fan; Shengyu Zhang
>
> **摘要:** Large language models (LLMs) excel in complex tasks through advanced prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), but their reliance on manually crafted, task-specific prompts limits adaptability and efficiency. We introduce Mixture of Reasoning (MoR), a training framework that embeds diverse reasoning strategies into LLMs for autonomous, task-adaptive reasoning without external prompt engineering. MoR has two phases: Thought Generation, creating reasoning chain templates with models like GPT-4o, and SFT Dataset Construction, pairing templates with benchmark datasets for supervised fine-tuning. Our experiments show that MoR significantly enhances performance, with MoR150 achieving 0.730 (2.2% improvement) using CoT prompting and 0.734 (13.5% improvement) compared to baselines. MoR eliminates the need for task-specific prompts, offering a generalizable solution for robust reasoning across diverse tasks.
>
---
#### [replaced 004] Self-Guided Process Reward Optimization with Redefined Step-wise Advantage for Process Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01551v2](http://arxiv.org/pdf/2507.01551v2)**

> **作者:** Wu Fei; Hao Kong; Shuxian Liang; Yang Lin; Yibo Yang; Jing Tang; Lei Chen; Xiansheng Hua
>
> **摘要:** Process Reinforcement Learning~(PRL) has demonstrated considerable potential in enhancing the reasoning capabilities of Large Language Models~(LLMs). However, introducing additional process reward models incurs substantial computational overhead, and there is no unified theoretical framework for process-level advantage estimation. To bridge this gap, we propose \textbf{S}elf-Guided \textbf{P}rocess \textbf{R}eward \textbf{O}ptimization~(\textbf{SPRO}), a novel framework that enables process-aware RL through two key innovations: (1) we first theoretically demonstrate that process rewards can be derived intrinsically from the policy model itself, and (2) we introduce well-defined cumulative process rewards and \textbf{M}asked \textbf{S}tep \textbf{A}dvantage (\textbf{MSA}), which facilitates rigorous step-wise action advantage estimation within shared-prompt sampling groups. Our experimental results demonstrate that SPRO outperforms vaniila GRPO with 3.4x higher training efficiency and a 17.5\% test accuracy improvement. Furthermore, SPRO maintains a stable and elevated policy entropy throughout training while reducing the average response length by approximately $1/3$, evidencing sufficient exploration and prevention of reward hacking. Notably, SPRO incurs no additional computational overhead compared to outcome-supervised RL methods such as GRPO, which benefit industrial implementation.
>
---
#### [replaced 005] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15075v2](http://arxiv.org/pdf/2505.15075v2)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
#### [replaced 006] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2406.10576v3](http://arxiv.org/pdf/2406.10576v3)**

> **作者:** Yuan Gao; Zujing Liu; Weizhong Zhang; Bo Du; Gui-Song Xia
>
> **备注:** ACL2025 Main Accepted
>
> **摘要:** Recent Large-Language Models (LLMs) pruning methods typically operate at the post-training phase without the expensive weight finetuning, however, their pruning criteria often rely on heuristically hand-crafted metrics, potentially leading to suboptimal performance. We instead propose a novel optimization-based structural pruning that learns the pruning masks in a probabilistic space directly by optimizing the loss of the pruned model. To preserve efficiency, our method eliminates the back-propagation through the LLM per se during optimization, requiring only the forward pass of the LLM. We achieve this by learning an underlying Bernoulli distribution to sample binary pruning masks, where we decouple the Bernoulli parameters from LLM loss, facilitating efficient optimization via policy gradient estimator without back-propagation. Thus, our method can 1) support global and heterogeneous pruning (i.e., automatically determine different redundancy for different layers), and 2) optionally initialize with a metric-based method (for our Bernoulli distributions). Extensive experiments conducted on LLaMA, LLaMA-2, LLaMA-3, Vicuna, and Mistral models using the C4 and WikiText2 datasets demonstrate the promising performance of our method in efficiency and effectiveness. Code is available at https://github.com/ethanygao/backprop-free_LLM_pruning.
>
---
#### [replaced 007] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01352v2](http://arxiv.org/pdf/2507.01352v2)**

> **作者:** Chris Yuhao Liu; Liang Zeng; Yuzhen Xiao; Jujie He; Jiacai Liu; Chaojie Wang; Rui Yan; Wei Shen; Fuxiang Zhang; Jiacheng Xu; Yang Liu; Yahui Zhou
>
> **摘要:** Despite the critical role of reward models (RMs) in reinforcement learning from human feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches that incorporate advanced training techniques have not yielded meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while large language models perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling, achieving state-of-the-art performance across seven major reward model benchmarks. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality.
>
---
#### [replaced 008] Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation
- **分类: cs.MA; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.21880v2](http://arxiv.org/pdf/2505.21880v2)**

> **作者:** Yu-Lun Song; Chung-En Tsern; Che-Cheng Wu; Yu-Ming Chang; Syuan-Bo Huang; Wei-Chu Chen; Michael Chia-Liang Lin; Yu-Ta Lin
>
> **备注:** 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
>
> **摘要:** This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
>
---
#### [replaced 009] Crafting Hanzi as Narrative Bridges: An AI Co-Creation Workshop for Elderly Migrants
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01548v2](http://arxiv.org/pdf/2507.01548v2)**

> **作者:** Wen Zhan; Ziqun Hua; Peiyue Lin; Yunfei Chen
>
> **备注:** A version of this manuscript has been submitted to the [IASDR 2025 Conference](https://iasdr2025.org/) and is currently under review
>
> **摘要:** This paper explores how older adults, particularly aging migrants in urban China, can engage AI-assisted co-creation to express personal narratives that are often fragmented, underrepresented, or difficult to verbalize. Through a pilot workshop combining oral storytelling and the symbolic reconstruction of Hanzi, participants shared memories of migration and recreated new character forms using Xiaozhuan glyphs, suggested by the Large Language Model (LLM), together with physical materials. Supported by human facilitation and a soft AI presence, participants transformed lived experience into visual and tactile expressions without requiring digital literacy. This approach offers new perspectives on human-AI collaboration and aging by repositioning AI not as a content producer but as a supportive mechanism, and by supporting narrative agency within sociotechnical systems.
>
---
#### [replaced 010] GPAS: Accelerating Convergence of LLM Pretraining via Gradient-Preserving Activation Scaling
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22049v2](http://arxiv.org/pdf/2506.22049v2)**

> **作者:** Tianhao Chen; Xin Xu; Zijing Liu; Pengxiang Li; Xinyuan Song; Ajay Kumar Jaiswal; Fan Zhang; Jishan Hu; Yang Wang; Hao Chen; Shizhe Diao; Shiwei Liu; Yu Li; Lu Yin; Can Yang
>
> **摘要:** Modern Large Language Models, such as the LLaMA, Qwen and DeepSeek series, predominantly adopt the Pre-LayerNorm (Pre-LN) Transformer architecture. While being stable during pretraining and scalable to large model sizes, Pre-LN suffers from an exponential growth in activation variance across layers, causing the shortcut to dominate over sub-layer outputs in the residual connection and limiting the learning capacity of deeper layers. To mitigate this issue, we propose Gradient-Preserving Activation Scaling (GPAS), a simple technique that can be used in combination with existing approaches. GPAS works by scaling down the intermediate activations while keeping their gradients unchanged. This leaves information in the activations intact, and avoids the gradient vanishing problem associated with gradient downscaling. Extensive experiments across various model sizes from 71M to 1B show that GPAS achieves consistent performance gains. Beyond enhancing Pre-LN Transformers, GPAS also shows promise in improving alternative architectures such as Sandwich-LN and DeepNorm, demonstrating its versatility and potential for improving training dynamics in a wide range of settings. Our code is available at https://github.com/dandingsky/GPAS.
>
---
#### [replaced 011] REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03262v4](http://arxiv.org/pdf/2501.03262v4)**

> **作者:** Jian Hu; Xibin Wu; Wei Shen; Jason Klein Liu; Zilin Zhu; Weixun Wang; Songlin Jiang; Haoran Wang; Hao Chen; Bin Chen; Weikai Fang; Xianyu; Yu Cao; Haotian Xu
>
> **备注:** fix typo
>
> **摘要:** Large Language Models (LLMs) fine-tuned via Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) significantly improve the alignment of human-AI values and further raise the upper bound of AI capabilities, particularly in reasoning-intensive, long-context Chain-of-Thought (long-CoT) tasks. However, existing RLHF (or RLVR) frameworks commonly face challenges such as inference bottlenecks and complexity barriers, restricting their accessibility for newcomers. To bridge this gap, we introduce \textbf{OpenRLHF}, a user-friendly, scalable, and easy-to-learn open-source RLHF framework built upon Ray, vLLM, DeepSpeed, and HuggingFace Transformers, featuring a simplified design, clear code structure, and comprehensive documentation to facilitate entry for researchers and practitioners. Experimental results show that OpenRLHF achieves superior training efficiency with speedups ranging from 1.22x to 1.68x across different model sizes compared to state-of-the-art frameworks, while requiring significantly fewer lines of code for implementation. OpenRLHF is publicly available at https://github.com/OpenRLHF/OpenRLHF, and has already been adopted by leading institutions to accelerate RLHF research and learning.
>
---
#### [replaced 012] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22618v3](http://arxiv.org/pdf/2505.22618v3)**

> **作者:** Chengyue Wu; Hao Zhang; Shuchen Xue; Zhijian Liu; Shizhe Diao; Ligeng Zhu; Ping Luo; Song Han; Enze Xie
>
> **摘要:** Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
>
---
#### [replaced 013] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16765v3](http://arxiv.org/pdf/2411.16765v3)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu; Alexander H. Liu
>
> **备注:** Fixed Figure 1. ACL 2025
>
> **摘要:** Sign language processing has traditionally relied on task-specific models, limiting the potential for transfer learning across tasks. Pre-training methods for sign language have typically focused on either supervised pre-training, which cannot take advantage of unlabeled data, or context-independent (frame or video segment) representations, which ignore the effects of relationships across time in sign language. We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.
>
---
#### [replaced 014] Commander-GPT: Fully Unleashing the Sarcasm Detection Capability of Multi-Modal Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.18681v3](http://arxiv.org/pdf/2503.18681v3)**

> **作者:** Yazhou Zhang; Chunwang Zou; Bo Wang; Jing Qin
>
> **备注:** Our original goal was to use Commander-GPT: Dividing and Routing for Multimodal Sarcasm Detection (arXiv:2506.19420) to replace Commander-GPT: Fully Unleashing the Sarcasm Detection Capability of Multi-Modal Large Language Models (arXiv:2503.18681). Due to various reasons, both versions were released, so we would like to withdraw the latter
>
> **摘要:** Sarcasm detection, as a crucial research direction in the field of Natural Language Processing (NLP), has attracted widespread attention. Traditional sarcasm detection tasks have typically focused on single-modal approaches (e.g., text), but due to the implicit and subtle nature of sarcasm, such methods often fail to yield satisfactory results. In recent years, researchers have shifted the focus of sarcasm detection to multi-modal approaches. However, effectively leveraging multi-modal information to accurately identify sarcastic content remains a challenge that warrants further exploration. Leveraging the powerful integrated processing capabilities of Multi-Modal Large Language Models (MLLMs) for various information sources, we propose an innovative multi-modal Commander-GPT framework. Inspired by military strategy, we first decompose the sarcasm detection task into six distinct sub-tasks. A central commander (decision-maker) then assigns the best-suited large language model to address each specific sub-task. Ultimately, the detection results from each model are aggregated to identify sarcasm. We conducted extensive experiments on MMSD and MMSD 2.0, utilizing four multi-modal large language models and six prompting strategies. Our experiments demonstrate that our approach achieves state-of-the-art performance, with a 19.3% improvement in F1 score, without necessitating fine-tuning or ground-truth rationales.
>
---
#### [replaced 015] AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.14634v3](http://arxiv.org/pdf/2506.14634v3)**

> **作者:** Leah von der Heyde; Anna-Carolina Haensch; Bernd Weiß; Jessica Daikeler
>
> **备注:** to appear in Survey Research Methods
>
> **摘要:** The recent development and wider accessibility of LLMs have spurred discussions about how they can be used in survey research, including classifying open-ended survey responses. Due to their linguistic capacities, it is possible that LLMs are an efficient alternative to time-consuming manual coding and the pre-training of supervised machine learning models. As most existing research on this topic has focused on English-language responses relating to non-complex topics or on single LLMs, it is unclear whether its findings generalize and how the quality of these classifications compares to established methods. In this study, we investigate to what extent different LLMs can be used to code open-ended survey responses in other contexts, using German data on reasons for survey participation as an example. We compare several state-of-the-art LLMs and several prompting approaches, and evaluate the LLMs' performance by using human expert codings. Overall performance differs greatly between LLMs, and only a fine-tuned LLM achieves satisfactory levels of predictive performance. Performance differences between prompting approaches are conditional on the LLM used. Finally, LLMs' unequal classification performance across different categories of reasons for survey participation results in different categorical distributions when not using fine-tuning. We discuss the implications of these findings, both for methodological research on coding open-ended responses and for their substantive analysis, and for practitioners processing or substantively analyzing such data. Finally, we highlight the many trade-offs researchers need to consider when choosing automated methods for open-ended response classification in the age of LLMs. In doing so, our study contributes to the growing body of research about the conditions under which LLMs can be efficiently, accurately, and reliably leveraged in survey research.
>
---
#### [replaced 016] Task Prompt Vectors: Effective Initialization through Multi-Task Soft-Prompt Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.01119v3](http://arxiv.org/pdf/2408.01119v3)**

> **作者:** Robert Belanec; Simon Ostermann; Ivan Srba; Maria Bielikova
>
> **摘要:** Prompt tuning is an efficient solution for training large language models (LLMs). However, current soft-prompt-based methods often sacrifice multi-task modularity, requiring the training process to be fully or partially repeated for each newly added task. While recent work on task vectors applied arithmetic operations on full model weights to achieve the desired multi-task performance, a similar approach for soft-prompts is still missing. To this end, we introduce Task Prompt Vectors, created by element-wise difference between weights of tuned soft-prompts and their random initialization. Experimental results on 12 NLU datasets show that task prompt vectors can be used in low-resource settings to effectively initialize prompt tuning on similar tasks. In addition, we show that task prompt vectors are independent of the random initialization of prompt tuning on 2 different language model architectures. This allows prompt arithmetics with the pre-trained vectors from different tasks. In this way, we provide a competitive alternative to state-of-the-art baselines by arithmetic addition of task prompt vectors from multiple tasks.
>
---
#### [replaced 017] Direct Preference Optimization Using Sparse Feature-Level Constraints
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07618v2](http://arxiv.org/pdf/2411.07618v2)**

> **作者:** Qingyu Yin; Chak Tou Leong; Hongbo Zhang; Minjun Zhu; Hanqi Yan; Qiang Zhang; Yulan He; Wenjie Li; Jun Wang; Yue Zhang; Linyi Yang
>
> **摘要:** The alignment of large language models (LLMs) with human preferences remains a key challenge. While post-training techniques like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have achieved notable success, they often introduce computational inefficiencies and training instability. In this paper, we propose Feature-level constrained Preference Optimization (FPO), a novel method designed to simplify the alignment process while ensuring stability. FPO leverages pre-trained Sparse Autoencoders (SAEs) and introduces feature-level constraints, allowing for efficient, sparsity-enforced alignment. Our approach enjoys efficiency by using sparse features activated in a well-trained sparse autoencoder and the quality of sequential KL divergence by using the feature-level offline reference. Experimental results on benchmark datasets demonstrate that FPO achieves a 5.08% absolute improvement in win rate with much lower computational cost compared to state-of-the-art baselines, making it a promising solution for efficient and controllable LLM alignments.
>
---
#### [replaced 018] From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.18959v3](http://arxiv.org/pdf/2506.18959v3)**

> **作者:** Weizhi Zhang; Yangning Li; Yuanchen Bei; Junyu Luo; Guancheng Wan; Liangwei Yang; Chenxuan Xie; Yuyao Yang; Wei-Chieh Huang; Chunyu Miao; Henry Peng Zou; Xiao Luo; Yusheng Zhao; Yankai Chen; Chunkit Chan; Peilin Zhou; Xinyang Zhang; Chenwei Zhang; Jingbo Shang; Ming Zhang; Yangqiu Song; Irwin King; Philip S. Yu
>
> **摘要:** Information retrieval is a cornerstone of modern knowledge acquisition, enabling billions of queries each day across diverse domains. However, traditional keyword-based search engines are increasingly inadequate for handling complex, multi-step information needs. Our position is that Large Language Models (LLMs), endowed with reasoning and agentic capabilities, are ushering in a new paradigm termed Agentic Deep Research. These systems transcend conventional information search techniques by tightly integrating autonomous reasoning, iterative retrieval, and information synthesis into a dynamic feedback loop. We trace the evolution from static web search to interactive, agent-based systems that plan, explore, and learn. We also introduce a test-time scaling law to formalize the impact of computational depth on reasoning and search. Supported by benchmark results and the rise of open-source implementations, we demonstrate that Agentic Deep Research not only significantly outperforms existing approaches, but is also poised to become the dominant paradigm for future information seeking. All the related resources, including industry products, research papers, benchmark datasets, and open-source implementations, are collected for the community in https://github.com/DavidZWZ/Awesome-Deep-Research.
>
---
#### [replaced 019] Optimal strategies to perform multilingual analysis of social content for a novel dataset in the tourism domain
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.14727v2](http://arxiv.org/pdf/2311.14727v2)**

> **作者:** Maxime Masson; Rodrigo Agerri; Christian Sallaberry; Marie-Noelle Bessagnet; Annig Le Parc Lacayrelle; Philippe Roose
>
> **摘要:** The rising influence of social media platforms in various domains, including tourism, has highlighted the growing need for efficient and automated Natural Language Processing (NLP) strategies to take advantage of this valuable resource. However, the transformation of multilingual, unstructured, and informal texts into structured knowledge still poses significant challenges, most notably the never-ending requirement for manually annotated data to train deep learning classifiers. In this work, we study different NLP techniques to establish the best ones to obtain competitive performances while keeping the need for training annotated data to a minimum. To do so, we built the first publicly available multilingual dataset (French, English, and Spanish) for the tourism domain, composed of tourism-related tweets. The dataset includes multilayered, manually revised annotations for Named Entity Recognition (NER) for Locations and Fine-grained Thematic Concepts Extraction mapped to the Thesaurus of Tourism and Leisure Activities of the World Tourism Organization, as well as for Sentiment Analysis at the tweet level. Extensive experimentation comparing various few-shot and fine-tuning techniques with modern language models demonstrate that modern few-shot techniques allow us to obtain competitive results for all three tasks with very little annotation data: 5 tweets per label (15 in total) for Sentiment Analysis, 30 tweets for Named Entity Recognition of Locations and 1K tweets annotated with fine-grained thematic concepts, a highly fine-grained sequence labeling task based on an inventory of 315 classes. We believe that our results, grounded in a novel dataset, pave the way for applying NLP to new domain-specific applications, reducing the need for manual annotations and circumventing the complexities of rule-based, ad-hoc solutions.
>
---
#### [replaced 020] Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17828v2](http://arxiv.org/pdf/2506.17828v2)**

> **作者:** Xinnan Zhang; Chenliang Li; Siliang Zeng; Jiaxiang Li; Zhongruo Wang; Kaixiang Lin; Songtao Lu; Alfredo Garcia; Mingyi Hong
>
> **摘要:** Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights.
>
---
#### [replaced 021] Next-Token Prediction Task Assumes Optimal Data Ordering for LLM Training in Proof Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.00863v2](http://arxiv.org/pdf/2411.00863v2)**

> **作者:** Chenyang An; Shima Imani; Feng Yao; Chengyu Dong; Ali Abbasi; Harsh Shrivastava; Samuel Buss; Jingbo Shang; Gayathri Mahalingam; Pramod Sharma; Maurice Diesendruck
>
> **摘要:** In the field of large language model (LLM)-based proof generation, despite extensive training on large datasets such as ArXiv, LLMs still exhibit only modest performance on proving tasks of moderate difficulty. We believe that this is partly due to the widespread presence of suboptimal ordering within the data for each proof used in training. For example, published proofs often follow a purely logical order, where each step logically proceeds from the previous steps based on the deductive rules. This order is designed to facilitate the verification of the proof's soundness, rather than to help people and models learn the discovery process of the proof. In proof generation, we argue that the optimal order for one training data sample occurs when the relevant intermediate supervision for a particular proof step in the proof is always positioned to the left of that proof step. We call such order the intuitively sequential order. We validate our claims using two tasks: intuitionistic propositional logic theorem-proving and digit multiplication. Our experiments verify the order effect and provide support for our explanations. We demonstrate that training is most effective when the proof is in the intuitively sequential order. Moreover, the order effect and the performance gap between models trained on different data orders can be substantial -- with an 11 percent improvement in proof success rate observed in the propositional logic theorem-proving task, between models trained on the optimal order compared to the worst order. Lastly, we define a common type of order issue in advanced math proofs and find that 17.3 percent of theorems with nontrivial proofs in the first two chapters of a widely used graduate-level mathematics textbook suffer from this issue. A detailed list of those proofs is provided in the appendix.
>
---
#### [replaced 022] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.11556v2](http://arxiv.org/pdf/2412.11556v2)**

> **作者:** Yuchen Fu; Zifeng Cheng; Zhiwei Jiang; Zhonghui Wang; Yafeng Yin; Zhengliang Li; Qing Gu
>
> **备注:** Accept to ACL 2025 (Oral). Code are available on https://github.com/fuyuchenIfyw/token_prepending.git
>
> **摘要:** Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. To this end, we propose a novel Token Prepending (TP) technique that prepends each layer's decoded sentence embedding to the beginning of the sentence in the next layer's input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs. Extensive experiments on various Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.
>
---
#### [replaced 023] De-mark: Watermark Removal in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13808v2](http://arxiv.org/pdf/2410.13808v2)**

> **作者:** Ruibo Chen; Yihan Wu; Junfeng Guo; Heng Huang
>
> **备注:** ICML 2025
>
> **摘要:** Watermarking techniques offer a promising way to identify machine-generated content via embedding covert information into the contents generated from language models (LMs). However, the robustness of the watermarking schemes has not been well explored. In this paper, we present De-mark, an advanced framework designed to remove n-gram-based watermarks effectively. Our method utilizes a novel querying strategy, termed random selection probing, which aids in assessing the strength of the watermark and identifying the red-green list within the n-gram watermark. Experiments on popular LMs, such as Llama3 and ChatGPT, demonstrate the efficiency and effectiveness of De-mark in watermark removal and exploitation tasks.
>
---
#### [replaced 024] SMARTe: Slot-based Method for Accountable Relational Triple extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12816v3](http://arxiv.org/pdf/2504.12816v3)**

> **作者:** Xue Wen Tan; Stanley Kok
>
> **摘要:** Relational Triple Extraction (RTE) is a fundamental task in Natural Language Processing (NLP). However, prior research has primarily focused on optimizing model performance, with limited efforts to understand the internal mechanisms driving these models. Many existing methods rely on complex preprocessing to induce specific interactions, often resulting in opaque systems that may not fully align with their theoretical foundations. To address these limitations, we propose SMARTe: a Slot-based Method for Accountable Relational Triple extraction. SMARTe introduces intrinsic interpretability through a slot attention mechanism and frames the task as a set prediction problem. Slot attention consolidates relevant information into distinct slots, ensuring all predictions can be explicitly traced to learned slot representations and the tokens contributing to each predicted relational triple. While emphasizing interpretability, SMARTe achieves performance comparable to state-of-the-art models. Evaluations on the NYT and WebNLG datasets demonstrate that adding interpretability does not compromise performance. Furthermore, we conducted qualitative assessments to showcase the explanations provided by SMARTe, using attention heatmaps that map to their respective tokens. We conclude with a discussion of our findings and propose directions for future research. Our code is available at https://github.com/Chen-XueWen/SMARTe.
>
---
#### [replaced 025] Quantifying the Importance of Data Alignment in Downstream Model Performance
- **分类: cs.CL; cs.AI; cs.LG; cs.PL**

- **链接: [http://arxiv.org/pdf/2501.08496v3](http://arxiv.org/pdf/2501.08496v3)**

> **作者:** Krrish Chawla; Aryan Sahai; Mario DePavia; Sudharsan Sundar; Brando Miranda; Elyas Obbad; Sanmi Koyejo
>
> **摘要:** Contrary to the conventional emphasis on dataset size, we explore the role of data alignment -- an often overlooked aspect of data quality -- in training capable Large Language Models (LLMs). To do so, we use the Task2Vec-based alignment coefficient, a quantitative measure of the similarity between two datasets, to quantify the impact of alignment between training data and evaluation data on downstream performance. In particular, we conduct controlled \textit{interventional} experiments for two settings: 1. the impact of increased alignment coefficients between various pre-training (pt) against evaluation datasets, and 2. the impact of increased alignment coefficients between domain specific fine-tuning (ft) against domain specific evaluation. The domain specific task we explore is Autoformalization -- the machine translation task between natural language and code for formal verification. In both settings, we find a strong, predictable negative correlation between the alignment coefficient of a model's training and evaluation data and the model's loss/perplexity on the respective downstream task. These findings suggest a re-evaluation of LLM training approaches, demonstrating the relevance of data alignment compared to data quantity, especially in specialized downstream tasks such as Autoformalization.
>
---
#### [replaced 026] Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments
- **分类: stat.AP; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.00903v3](http://arxiv.org/pdf/2410.00903v3)**

> **作者:** Kosuke Imai; Kentaro Nakamura
>
> **摘要:** In this paper, we demonstrate how to enhance the validity of causal inference with unstructured high-dimensional treatments like texts, by leveraging the power of generative Artificial Intelligence (GenAI). Specifically, we propose to use a deep generative model such as large language models (LLMs) to efficiently generate treatments and use their internal representation for subsequent causal effect estimation. We show that the knowledge of this true internal representation helps disentangle the treatment features of interest, such as specific sentiments and certain topics, from other possibly unknown confounding features. Unlike existing methods, the proposed GenAI-Powered Inference (GPI) methodology eliminates the need to learn causal representation from the data, and hence produces more accurate and efficient estimates. We formally establish the conditions required for the nonparametric identification of the average treatment effect, propose an estimation strategy that avoids the violation of the overlap assumption, and derive the asymptotic properties of the proposed estimator through the application of double machine learning. Finally, using an instrumental variables approach, we extend the proposed methodology to the settings in which the treatment feature is based on human perception. The proposed GPI methodology is also applicable to text reuse where an LLM is used to regenerate existing texts. We conduct simulation and empirical studies, using the generated text data from an open-source LLM, Llama 3, to illustrate the advantages of our estimator over state-of-the-art causal representation learning algorithms.
>
---
#### [replaced 027] On Characterizations for Language Generation: Interplay of Hallucinations, Breadth, and Stability
- **分类: cs.LG; cs.AI; cs.CL; cs.DS; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.18530v2](http://arxiv.org/pdf/2412.18530v2)**

> **作者:** Alkis Kalavasis; Anay Mehrotra; Grigoris Velegkas
>
> **备注:** v2 improves exposition and simplifies proofs
>
> **摘要:** We study language generation in the limit - introduced by Kleinberg and Mullainathan [KM24] - building on classical works of Gold [Gol67] and Angluin [Ang79]. [KM24]'s main result is an algorithm for generating from any countable language collection in the limit. While their algorithm eventually generates unseen strings from the target language $K$, it sacrifices coverage or breadth, i.e., its ability to generate a rich set of strings. Recent work introduces different notions of breadth and explores when generation with breadth is possible, leaving a full characterization of these notions open. Our first set of results settles this by characterizing generation for existing notions of breadth and their natural extensions. Interestingly, our lower bounds are very flexible and hold for many performance metrics beyond breadth - for instance, showing that, in general, it is impossible to train generators which achieve a higher perplexity or lower hallucination rate for $K$ compared to other languages. Next, we study language generation with breadth and stable generators - algorithms that eventually stop changing after seeing an arbitrary but finite number of strings - and prove unconditional lower bounds for such generators, strengthening the results of [KMV25] and demonstrating that generation with many existing notions of breadth becomes equally hard, when stability is required. This gives a separation for generation with approximate breadth, between stable and unstable generators, highlighting the rich interplay between breadth, stability, and consistency in language generation.
>
---
#### [replaced 028] Explainable Compliance Detection with Multi-Hop Natural Language Inference on Assurance Case Structure
- **分类: cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.08713v2](http://arxiv.org/pdf/2506.08713v2)**

> **作者:** Fariz Ikhwantri; Dusica Marijan
>
> **摘要:** Ensuring complex systems meet regulations typically requires checking the validity of assurance cases through a claim-argument-evidence framework. Some challenges in this process include the complicated nature of legal and technical texts, the need for model explanations, and limited access to assurance case data. We propose a compliance detection approach based on Natural Language Inference (NLI): EXplainable CompLiance detection with Argumentative Inference of Multi-hop reasoning (EXCLAIM). We formulate the claim-argument-evidence structure of an assurance case as a multi-hop inference for explainable and traceable compliance detection. We address the limited number of assurance cases by generating them using large language models (LLMs). We introduce metrics that measure the coverage and structural consistency. We demonstrate the effectiveness of the generated assurance case from GDPR requirements in a multi-hop inference task as a case study. Our results highlight the potential of NLI-based approaches in automating the regulatory compliance process.
>
---
#### [replaced 029] Rethinking LLM Training through Information Geometry and Quantum Metrics
- **分类: cs.CL; quant-ph; I.2; I.7**

- **链接: [http://arxiv.org/pdf/2506.15830v3](http://arxiv.org/pdf/2506.15830v3)**

> **作者:** Riccardo Di Sipio
>
> **备注:** 9 pages, 1 figure(s)
>
> **摘要:** Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
>
---
#### [replaced 030] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v2](http://arxiv.org/pdf/2506.12479v2)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 031] MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12532v3](http://arxiv.org/pdf/2410.12532v3)**

> **作者:** Dingkang Yang; Jinjie Wei; Mingcheng Li; Jiyao Liu; Lihao Liu; Ming Hu; Junjun He; Yakun Ju; Wei Zhou; Yang Liu; Lihua Zhang
>
> **备注:** LLM-based Multi-Agent Collaboration for Medical Applications
>
> **摘要:** In healthcare intelligence, the ability to fuse heterogeneous, multi-intent information from diverse clinical sources is fundamental to building reliable decision-making systems. Large Language Model (LLM)-driven information interaction systems currently showing potential promise in the healthcare domain. Nevertheless, they often suffer from information redundancy and coupling when dealing with complex medical intents, leading to severe hallucinations and performance bottlenecks. To this end, we propose MedAide, an LLM-based medical multi-agent collaboration framework designed to enable intent-aware information fusion and coordinated reasoning across specialized healthcare domains. Specifically, we introduce a regularization-guided module that combines syntactic constraints with retrieval augmented generation to decompose complex queries into structured representations, facilitating fine-grained clinical information fusion and intent resolution. Additionally, a dynamic intent prototype matching module is proposed to utilize dynamic prototype representation with a semantic similarity matching mechanism to achieve adaptive recognition and updating of the agent's intent in multi-round healthcare dialogues. Ultimately, we design a rotation agent collaboration mechanism that introduces dynamic role rotation and decision-level information fusion across specialized medical agents. Extensive experiments are conducted on four medical benchmarks with composite intents. Experimental results from automated metrics and expert doctor evaluations show that MedAide outperforms current LLMs and improves their medical proficiency and strategic reasoning.
>
---
#### [replaced 032] Enhancing Clinical Multiple-Choice Questions Benchmarks with Knowledge Graph Guided Distractor Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00612v3](http://arxiv.org/pdf/2506.00612v3)**

> **作者:** Running Yang; Wenlong Deng; Minghui Chen; Yuyin Zhou; Xiaoxiao Li
>
> **摘要:** Clinical tasks such as diagnosis and treatment require strong decision-making abilities, highlighting the importance of rigorous evaluation benchmarks to assess the reliability of large language models (LLMs). In this work, we introduce a knowledge-guided data augmentation framework that enhances the difficulty of clinical multiple-choice question (MCQ) datasets by generating distractors (i.e., incorrect choices that are similar to the correct one and may confuse existing LLMs). Using our KG-based pipeline, the generated choices are both clinically plausible and deliberately misleading. Our approach involves multi-step, semantically informed walks on a medical knowledge graph to identify distractor paths-associations that are medically relevant but factually incorrect-which then guide the LLM in crafting more deceptive distractors. We apply the designed knowledge graph guided distractor generation (KGGDG) pipline, to six widely used medical QA benchmarks and show that it consistently reduces the accuracy of state-of-the-art LLMs. These findings establish KGGDG as a powerful tool for enabling more robust and diagnostic evaluations of medical LLMs.
>
---
#### [replaced 033] Batch-Max: Higher LLM Throughput using Larger Batch Sizes and KV Cache Compression
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.05693v3](http://arxiv.org/pdf/2412.05693v3)**

> **作者:** Michael R. Metel; Boxing Chen; Mehdi Rezagholizadeh
>
> **摘要:** Several works have developed eviction policies to remove key-value (KV) pairs from the KV cache for more efficient inference. The focus has been on compressing the KV cache after the input prompt has been processed for faster token generation. In settings with limited GPU memory, and when the input context is longer than the generation length, we show that by also compressing the KV cache during the input processing phase, larger batch sizes can be used resulting in significantly higher throughput while still maintaining the original model's accuracy.
>
---
#### [replaced 034] Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.23661v2](http://arxiv.org/pdf/2506.23661v2)**

> **作者:** Arnisa Fazla; Lucas Krauter; David Guzman Piedrahita; Andrianos Michail
>
> **备注:** 12 pages main text, 27 pages total including references and appendices. 13 figures, 10 tables. Accepted for publication in the LNCS proceedings of CLEF 2025 (Best-of-Labs track)
>
> **摘要:** We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack
>
---
#### [replaced 035] Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2306.13840v4](http://arxiv.org/pdf/2306.13840v4)**

> **作者:** Brando Miranda; Alycia Lee; Sudharsan Sundar; Allison Casasola; Rylan Schaeffer; Elyas Obbad; Sanmi Koyejo
>
> **摘要:** Current trends in pre-training Large Language Models (LLMs) primarily focus on the scaling of model and dataset size. While the quality of pre-training data is considered an important factor for training powerful LLMs, it remains a nebulous concept that has not been rigorously characterized. To this end, we propose a formalization of one key aspect of data quality -- measuring the variability of natural language data -- specifically via a measure we call the diversity coefficient. Our empirical analysis shows that the proposed diversity coefficient aligns with the intuitive properties of diversity and variability, e.g., it increases as the number of latent concepts increases. Then, we measure the diversity coefficient of publicly available pre-training datasets and demonstrate that their formal diversity is high compared to theoretical lower and upper bounds. Finally, we conduct a comprehensive set of controlled interventional experiments with GPT-2 and LLaMAv2 that demonstrate the diversity coefficient of pre-training data characterizes useful aspects of downstream model evaluation performance -- totaling 44 models of various sizes (51M to 7B parameters). We conclude that our formal notion of diversity is an important aspect of data quality that captures variability and causally leads to improved evaluation performance.
>
---
#### [replaced 036] Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01334v2](http://arxiv.org/pdf/2507.01334v2)**

> **作者:** Nifu Dan; Yujun Cai; Yiwei Wang
>
> **摘要:** Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains.
>
---
#### [replaced 037] Circuit-tuning: A Mechanistic Approach for Identifying Parameter Redundancy and Fine-tuning Neural Networks
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06106v2](http://arxiv.org/pdf/2502.06106v2)**

> **作者:** Yueyan Li; Wenhao Gao; Caixia Yuan; Xiaojie Wang
>
> **摘要:** The study of mechanistic interpretability aims to reverse-engineer a model to explain its behaviors. While recent studies have focused on the static mechanism of a certain behavior, the learning dynamics inside a model remain to be explored. In this work, we develop an interpretable fine-tuning method for analyzing the mechanism behind learning. We first introduce the concept of node-level intrinsic dimensionality to describe the learning process of a model in a computational graph. Based on our theory, we propose circuit-tuning, a two-stage algorithm that iteratively builds the minimal subgraph for a specific task and updates the key parameters in a heuristic way. Experimental results confirm the existence of the intrinsic dimensionality at the node level and demonstrate the effectiveness of our method for transparent and interpretable fine-tuning. We visualize and analyze the circuits before, during, and after fine-tuning, providing new insights into the self-organization mechanism of a neural network in the learning process.
>
---
#### [replaced 038] Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21506v2](http://arxiv.org/pdf/2506.21506v2)**

> **作者:** Boyu Gou; Zanming Huang; Yuting Ning; Yu Gu; Michael Lin; Weijian Qi; Andrei Kopanev; Botao Yu; Bernal Jiménez Gutiérrez; Yiheng Shu; Chan Hee Song; Jiaman Wu; Shijie Chen; Hanane Nour Moussa; Tianshu Zhang; Jian Xie; Yifei Li; Tianci Xue; Zeyi Liao; Kai Zhang; Boyuan Zheng; Zhaowei Cai; Viktor Rozgic; Morteza Ziyadi; Huan Sun; Yu Su
>
> **备注:** Project Homepage: https://osu-nlp-group.github.io/Mind2Web-2/
>
> **摘要:** Agentic search such as Deep Research systems-where agents autonomously browse the web, synthesize information, and return comprehensive citation-backed answers-represents a major shift in how users interact with web-scale information. While promising greater efficiency and cognitive offloading, the growing complexity and open-endedness of agentic search have outpaced existing evaluation benchmarks and methodologies, which largely assume short search horizons and static answers. In this paper, we introduce Mind2Web 2, a benchmark of 130 realistic, high-quality, and long-horizon tasks that require real-time web browsing and extensive information synthesis, constructed with over 1000 hours of human labor. To address the challenge of evaluating time-varying and complex answers, we propose a novel Agent-as-a-Judge framework. Our method constructs task-specific judge agents based on a tree-structured rubric design to automatically assess both answer correctness and source attribution. We conduct a comprehensive evaluation of ten frontier agentic search systems and human performance, along with a detailed error analysis to draw insights for future development. The best-performing system, OpenAI Deep Research, can already achieve 50-70% of human performance while spending half the time, highlighting its great potential. Altogether, Mind2Web 2 provides a rigorous foundation for developing and benchmarking the next generation of agentic search systems.
>
---
#### [replaced 039] Prompt-Guided Turn-Taking Prediction
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21191v2](http://arxiv.org/pdf/2506.21191v2)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at SIGdial Meeting on Discourse and Dialogue 2025 (SIGDIAL 2025) and represents the author's version of the work
>
> **摘要:** Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
>
---
#### [replaced 040] Decision-Oriented Text Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01923v2](http://arxiv.org/pdf/2507.01923v2)**

> **作者:** Yu-Shiang Huang; Chuan-Ju Wang; Chung-Chi Chen
>
> **摘要:** Natural language generation (NLG) is increasingly deployed in high-stakes domains, yet common intrinsic evaluation methods, such as n-gram overlap or sentence plausibility, weakly correlate with actual decision-making efficacy. We propose a decision-oriented framework for evaluating generated text by directly measuring its influence on human and large language model (LLM) decision outcomes. Using market digest texts--including objective morning summaries and subjective closing-bell analyses--as test cases, we assess decision quality based on the financial performance of trades executed by human investors and autonomous LLM agents informed exclusively by these texts. Our findings reveal that neither humans nor LLM agents consistently surpass random performance when relying solely on summaries. However, richer analytical commentaries enable collaborative human-LLM teams to outperform individual human or agent baselines significantly. Our approach underscores the importance of evaluating generated text by its ability to facilitate synergistic decision-making between humans and LLMs, highlighting critical limitations of traditional intrinsic metrics.
>
---
#### [replaced 041] Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning
- **分类: cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.13886v2](http://arxiv.org/pdf/2505.13886v2)**

> **作者:** Jingqi Tong; Jixin Tang; Hangcheng Li; Yurong Mou; Ming Zhang; Jun Zhao; Yanbo Wen; Fan Song; Jiahao Zhan; Yuyang Lu; Chaoran Tao; Zhiyuan Guo; Jizhou Yu; Tianhao Cheng; Changhao Jiang; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Weifeng Ge; Guanhua Chen; Tao Gui; Xipeng Qiu; Qi Zhang; Xuanjing Huang
>
> **备注:** 63 pages, 23 figures, submitted to NeurIPS 2025
>
> **摘要:** Visual-language Chain-of-Thought (CoT) data resources are relatively scarce compared to text-only counterparts, limiting the improvement of reasoning capabilities in Vision Language Models (VLMs). However, high-quality vision-language reasoning data is expensive and labor-intensive to annotate. To address this issue, we leverage a promising resource: game code, which naturally contains logical structures and state transition processes. Therefore, we propose Code2Logic, a novel game-code-driven approach for multimodal reasoning data synthesis. Our approach leverages Large Language Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning processes and results through code execution. Using the Code2Logic approach, we developed the GameQA dataset to train and evaluate VLMs. GameQA is cost-effective and scalable, offers controllable difficulty gradation and is diverse with 30 games and 158 tasks. Surprisingly, despite training solely on game data, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33% across 7 diverse vision-language benchmarks. Our code, dataset and models are available at https://github.com/tongjingqi/Code2Logic.
>
---
#### [replaced 042] Layered Insights: Generalizable Analysis of Authorial Style by Leveraging All Transformer Layers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00958v2](http://arxiv.org/pdf/2503.00958v2)**

> **作者:** Milad Alshomary; Nikhil Reddy Varimalla; Vishal Anand; Smaranda Muresan; Kathleen McKeown
>
> **摘要:** We propose a new approach for the authorship attribution task that leverages the various linguistic representations learned at different layers of pre-trained transformer-based models. We evaluate our approach on three datasets, comparing it to a state-of-the-art baseline in in-domain and out-of-domain scenarios. We found that utilizing various transformer layers improves the robustness of authorship attribution models when tested on out-of-domain data, resulting in new state-of-the-art results. Our analysis gives further insights into how our model's different layers get specialized in representing certain stylistic features that benefit the model when tested out of the domain.
>
---
#### [replaced 043] Improving the Robustness of Distantly-Supervised Named Entity Recognition via Uncertainty-Aware Teacher Learning and Student-Student Collaborative Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2311.08010v3](http://arxiv.org/pdf/2311.08010v3)**

> **作者:** Shuzheng Si; Helan Hu; Haozhe Zhao; Shuang Zeng; Kaikai An; Zefan Cai; Baobao Chang
>
> **备注:** ACL 2024 (Findings)
>
> **摘要:** Distantly-Supervised Named Entity Recognition (DS-NER) is widely used in real-world scenarios. It can effectively alleviate the burden of annotation by matching entities in existing knowledge bases with snippets in the text but suffer from the label noise. Recent works attempt to adopt the teacher-student framework to gradually refine the training labels and improve the overall robustness. However, these teacher-student methods achieve limited performance because the poor calibration of the teacher network produces incorrectly pseudo-labeled samples, leading to error propagation. Therefore, we propose: (1) Uncertainty-Aware Teacher Learning that leverages the prediction uncertainty to reduce the number of incorrect pseudo labels in the self-training stage; (2) Student-Student Collaborative Learning that allows the transfer of reliable labels between two student networks instead of indiscriminately relying on all pseudo labels from its teacher, and further enables a full exploration of mislabeled samples rather than simply filtering unreliable pseudo-labeled samples. We evaluate our proposed method on five DS-NER datasets, demonstrating that our method is superior to the state-of-the-art DS-NER methods.
>
---

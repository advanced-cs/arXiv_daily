# 自然语言处理 cs.CL

- **最新发布 86 篇**

- **更新 73 篇**

## 最新发布

#### [new 001] MossNet: Mixture of State-Space Experts is a Multi-Head Attention
- **分类: cs.CL**

- **简介: 该论文提出MossNet，一种基于混合状态空间专家的新型架构，用于高效生成式语言建模。针对传统状态空间模型仅模拟单头注意力、表达力有限的问题，MossNet在通道混合与时间混合中引入多专家机制，实现线性多头注意力。实验表明其性能优于同类Transformer与SSM模型，并具备良好可扩展性与推理效率。**

- **链接: [http://arxiv.org/pdf/2510.26182v1](http://arxiv.org/pdf/2510.26182v1)**

> **作者:** Shikhar Tuli; James Seale Smith; Haris Jeelani; Chi-Heng Lin; Abhishek Patel; Vasili Ramanishka; Yen-Chang Hsu; Hongxia Jin
>
> **摘要:** Large language models (LLMs) have significantly advanced generative applications in natural language processing (NLP). Recent trends in model architectures revolve around efficient variants of transformers or state-space/gated-recurrent models (SSMs, GRMs). However, prevailing SSM/GRM-based methods often emulate only a single attention head, potentially limiting their expressiveness. In this work, we propose MossNet, a novel mixture-of-state-space-experts architecture that emulates a linear multi-head attention (MHA). MossNet leverages a mixture-of-experts (MoE) implementation not only in channel-mixing multi-layered perceptron (MLP) blocks but also in the time-mixing SSM kernels to realize multiple "attention heads." Extensive experiments on language modeling and downstream evaluations show that MossNet outperforms both transformer- and SSM-based architectures of similar model size and data budgets. Larger variants of MossNet, trained on trillions of tokens, further confirm its scalability and superior performance. In addition, real-device profiling on a Samsung Galaxy S24 Ultra and an Nvidia A100 GPU demonstrate favorable runtime speed and resource usage compared to similarly sized baselines. Our results suggest that MossNet is a compelling new direction for efficient, high-performing recurrent LLM architectures.
>
---
#### [new 002] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对扩散语言模型（DLM）在文本编辑中出现的“更新遗忘”问题，提出基于令牌时间步分配（TTA）的控制方法。通过为不同重要性令牌设定差异化时间步调度，实现语义有序的渐进式编辑，显著提升生成文本的可控性、流畅性和一致性，适用于多种任务与模型。**

- **链接: [http://arxiv.org/pdf/2510.26200v1](http://arxiv.org/pdf/2510.26200v1)**

> **作者:** Woojin Kim; Jaeyoung Do
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** While diffusion language models (DLMs) enable fine-grained refinement, their practical controllability remains fragile. We identify and formally characterize a central failure mode called update forgetting, in which uniform and context agnostic updates induce token level fluctuations across timesteps, erasing earlier semantic edits and disrupting the cumulative refinement process, thereby degrading fluency and coherence. As this failure originates in uniform and context agnostic updates, effective control demands explicit token ordering. We propose Token Timestep Allocation (TTA), which realizes soft and semantic token ordering via per token timestep schedules: critical tokens are frozen early, while uncertain tokens receive continued refinement. This timestep based ordering can be instantiated as either a fixed policy or an adaptive policy driven by task signals, thereby supporting a broad spectrum of refinement strategies. Because it operates purely at inference time, it applies uniformly across various DLMs and naturally extends to diverse supervision sources. Empirically, TTA improves controllability and fluency: on sentiment control, it yields more than 20 percent higher accuracy and nearly halves perplexity using less than one fifth the steps; in detoxification, it lowers maximum toxicity (12.2 versus 14.5) and perplexity (26.0 versus 32.0). Together, these results demonstrate that softened ordering via timestep allocation is the critical lever for mitigating update forgetting and achieving stable and controllable diffusion text generation.
>
---
#### [new 003] Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型中的跨语言对齐任务，针对现有方法在促进知识迁移时导致文化特异性信息丢失的问题，提出“转移-定位”评估框架，发现事实迁移与文化保真存在权衡。为此，提出推理时的外科引导方法，通过分层激活调控，实现二者更好平衡。**

- **链接: [http://arxiv.org/pdf/2510.26024v1](http://arxiv.org/pdf/2510.26024v1)**

> **作者:** HyoJung Han; Sweta Agrawal; Eleftheria Briakou
>
> **摘要:** Cross-lingual alignment (CLA) aims to align multilingual representations, enabling Large Language Models (LLMs) to seamlessly transfer knowledge across languages. While intuitive, we hypothesize, this pursuit of representational convergence can inadvertently cause "cultural erasure", the functional loss of providing culturally-situated responses that should diverge based on the query language. In this work, we systematically analyze this trade-off by introducing a holistic evaluation framework, the transfer-localization plane, which quantifies both desirable knowledge transfer and undesirable cultural erasure. Using this framework, we re-evaluate recent CLA approaches and find that they consistently improve factual transfer at the direct cost of cultural localization across all six languages studied. Our investigation into the internal representations of these models reveals a key insight: universal factual transfer and culturally-specific knowledge are optimally steerable at different model layers. Based on this finding, we propose Surgical Steering, a novel inference-time method that disentangles these two objectives. By applying targeted activation steering to distinct layers, our approach achieves a better balance between the two competing dimensions, effectively overcoming the limitations of current alignment techniques.
>
---
#### [new 004] StreetMath: Study of LLMs' Approximation Behaviors
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦于大语言模型在非精确、快速数学估算任务中的表现，提出StreetMath基准评估模型近似推理能力。针对现有研究多关注精确计算而忽视近似行为的问题，作者评测多个模型并结合可解释性分析，发现模型倾向精确求解或调用工具，且近似与精确运算依赖不同神经路径，表明其不具备人类“认知吝啬”特征。**

- **链接: [http://arxiv.org/pdf/2510.25776v1](http://arxiv.org/pdf/2510.25776v1)**

> **作者:** Chiung-Yi Tseng; Somshubhra Roy; Maisha Thasin; Danyang Zhang; Blessing Effiong
>
> **摘要:** There is a substantial body of literature examining the mathematical reasoning capabilities of large language models (LLMs), particularly their performance on precise arithmetic operations in autoregressive architectures. However, their ability to perform approximate reasoning in informal, fast-paced mathematical operations has received far less attention, especially among non-autoregressive decoder models. Our work addresses this gap by introducing StreetMath, a benchmark designed to evaluate models' approximation abilities under real-world approximation scenarios. We conduct extensive evaluations across different LLM architectures: Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, Dream-v0-Instruct-7B, Falcon-Mamba-7B-Instruct, and Mamba-GPT-3B. Furthermore, we apply mechanistic interpretability techniques to probe their internal computational states. Our analysis reveals that LLMs generally attempt to compute exact values or invoke external tools even in tasks that call for approximation. Moreover, while models sometimes reach the correct answer in early layers or steps, they still consume more tokens when solving approximation tasks. Additional experiments indicate that exact and approximate arithmetic operations rely on largely separate neural components. Drawing upon research on cognitive psychology, we argue that LLMs do not exhibit cognitive miserliness in the same way humans do in street math settings. We open source our work https://github.com/ctseng777/StreetMath
>
---
#### [new 005] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding
- **分类: cs.CL**

- **简介: 该论文提出SlideAgent，一种用于多页视觉文档理解的分层智能体框架。针对复杂文档中跨页推理与细粒度分析难题，通过全局、页面、元素三级代理协同，构建结构化表示，实现精准问答。实验表明其显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.26615v1](http://arxiv.org/pdf/2510.26615v1)**

> **作者:** Yiqiao Jin; Rachneet Kaur; Zhen Zeng; Sumitra Ganesh; Srijan Kumar
>
> **备注:** https://slideagent.github.io/
>
> **摘要:** Multi-page visual documents such as manuals, brochures, presentations, and posters convey key information through layout, colors, icons, and cross-slide references. While large language models (LLMs) offer opportunities in document understanding, current systems struggle with complex, multi-page visual documents, particularly in fine-grained reasoning over elements and pages. We introduce SlideAgent, a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks. SlideAgent employs specialized agents and decomposes reasoning into three specialized levels-global, page, and element-to construct a structured, query-agnostic representation that captures both overarching themes and detailed visual or textual cues. During inference, SlideAgent selectively activates specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers. Extensive experiments show that SlideAgent achieves significant improvement over both proprietary (+7.9 overall) and open-source models (+9.8 overall).
>
---
#### [new 006] Kimi Linear: An Expressive, Efficient Attention Architecture
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Kimi Linear，一种高效线性注意力架构，解决长序列建模中计算与内存开销大的问题。通过引入KDA模块和专用分块算法，在保持表达力的同时显著降低计算量与KV缓存占用，实现比全注意力更优的性能与效率，适用于短/长上下文及强化学习任务。**

- **链接: [http://arxiv.org/pdf/2510.26692v1](http://arxiv.org/pdf/2510.26692v1)**

> **作者:** Kimi Team; Yu Zhang; Zongyu Lin; Xingcheng Yao; Jiaxi Hu; Fanqing Meng; Chengyin Liu; Xin Men; Songlin Yang; Zhiyuan Li; Wentao Li; Enzhe Lu; Weizhou Liu; Yanru Chen; Weixin Xu; Longhui Yu; Yejie Wang; Yu Fan; Longguang Zhong; Enming Yuan; Dehao Zhang; Yizhi Zhang; T. Y. Liu; Haiming Wang; Shengjun Fang; Weiran He; Shaowei Liu; Yiwei Li; Jianlin Su; Jiezhong Qiu; Bo Pang; Junjie Yan; Zhejun Jiang; Weixiao Huang; Bohong Yin; Jiacheng You; Chu Wei; Zhengtao Wang; Chao Hong; Yutian Chen; Guanduo Chen; Yucheng Wang; Huabin Zheng; Feng Wang; Yibo Liu; Mengnan Dong; Zheng Zhang; Siyuan Pan; Wenhao Wu; Yuhao Wu; Longyu Guan; Jiawen Tao; Guohong Fu; Xinran Xu; Yuzhi Wang; Guokun Lai; Yuxin Wu; Xinyu Zhou; Zhilin Yang; Yulun Du
>
> **备注:** Kimi Linear tech report
>
> **摘要:** We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention under fair comparisons across various scenarios -- including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends Gated DeltaNet with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. Our bespoke chunkwise algorithm achieves high hardware efficiency through a specialized variant of the Diagonal-Plus-Low-Rank (DPLR) transition matrices, which substantially reduces computation compared to the general DPLR formulation while remaining more consistent with the classical delta rule. We pretrain a Kimi Linear model with 3B activated parameters and 48B total parameters, based on a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA). Our experiments show that with an identical training recipe, Kimi Linear outperforms full MLA with a sizeable margin across all evaluated tasks, while reducing KV cache usage by up to 75% and achieving up to 6 times decoding throughput for a 1M context. These results demonstrate that Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths. To support further research, we open-source the KDA kernel and vLLM implementations, and release the pre-trained and instruction-tuned model checkpoints.
>
---
#### [new 007] Semantic Label Drift in Cross-Cultural Translation
- **分类: cs.CL**

- **简介: 该论文研究跨文化翻译中的语义标签漂移问题，旨在解决机器翻译因文化差异导致的语义失真。通过实验发现，现代大语言模型在文化敏感领域易引发标签漂移，且文化相似性影响标签保真度，强调了文化对齐在机器翻译中的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.25967v1](http://arxiv.org/pdf/2510.25967v1)**

> **作者:** Mohsinul Kabir; Tasnim Ahmed; Md Mezbaur Rahman; Polydoros Giannouris; Sophia Ananiadou
>
> **摘要:** Machine Translation (MT) is widely employed to address resource scarcity in low-resource languages by generating synthetic data from high-resource counterparts. While sentiment preservation in translation has long been studied, a critical but underexplored factor is the role of cultural alignment between source and target languages. In this paper, we hypothesize that semantic labels are drifted or altered during MT due to cultural divergence. Through a series of experiments across culturally sensitive and neutral domains, we establish three key findings: (1) MT systems, including modern Large Language Models (LLMs), induce label drift during translation, particularly in culturally sensitive domains; (2) unlike earlier statistical MT tools, LLMs encode cultural knowledge, and leveraging this knowledge can amplify label drift; and (3) cultural similarity or dissimilarity between source and target languages is a crucial determinant of label preservation. Our findings highlight that neglecting cultural factors in MT not only undermines label fidelity but also risks misinterpretation and cultural conflict in downstream applications.
>
---
#### [new 008] Hebrew Diacritics Restoration using Visual Representation
- **分类: cs.CL**

- **简介: 该论文针对希伯来语无点字的正字法还原任务，提出DIVRIT系统。通过将文本转为图像输入视觉语言模型，实现零样本分类，动态生成候选方案并结合上下文选择最优点字模式，显著提升准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.26521v1](http://arxiv.org/pdf/2510.26521v1)**

> **作者:** Yair Elboher; Yuval Pinter
>
> **摘要:** Diacritics restoration in Hebrew is a fundamental task for ensuring accurate word pronunciation and disambiguating textual meaning. Despite the language's high degree of ambiguity when unvocalized, recent machine learning approaches have significantly advanced performance on this task. In this work, we present DIVRIT, a novel system for Hebrew diacritization that frames the task as a zero-shot classification problem. Our approach operates at the word level, selecting the most appropriate diacritization pattern for each undiacritized word from a dynamically generated candidate set, conditioned on the surrounding textual context. A key innovation of DIVRIT is its use of a Hebrew Visual Language Model, which processes undiacritized text as an image, allowing diacritic information to be embedded directly within the input's vector representation. Through a comprehensive evaluation across various configurations, we demonstrate that the system effectively performs diacritization without relying on complex, explicit linguistic analysis. Notably, in an ``oracle'' setting where the correct diacritized form is guaranteed to be among the provided candidates, DIVRIT achieves a high level of accuracy. Furthermore, strategic architectural enhancements and optimized training methodologies yield significant improvements in the system's overall generalization capabilities. These findings highlight the promising potential of visual representations for accurate and automated Hebrew diacritization.
>
---
#### [new 009] SymCode: A Neurosymbolic Approach to Mathematical Reasoning via Verifiable Code Generation
- **分类: cs.CL; cs.PL**

- **简介: 该论文提出SymCode，一种基于可验证代码生成的神经符号方法，用于数学推理。针对大模型在复杂数学问题上因依赖不可靠自然语言生成导致的错误，该工作将推理过程转化为使用SymPy库的确定性代码生成任务，实现可验证推理，显著提升准确率并使错误更透明。**

- **链接: [http://arxiv.org/pdf/2510.25975v1](http://arxiv.org/pdf/2510.25975v1)**

> **作者:** Sina Bagheri Nezhad; Yao Li; Ameeta Agrawal
>
> **摘要:** Large Language Models (LLMs) often struggle with complex mathematical reasoning, where prose-based generation leads to unverified and arithmetically unsound solutions. Current prompting strategies like Chain of Thought still operate within this unreliable medium, lacking a mechanism for deterministic verification. To address these limitations, we introduce SymCode, a neurosymbolic framework that reframes mathematical problem-solving as a task of verifiable code generation using the SymPy library. We evaluate SymCode on challenging benchmarks, including MATH-500 and OlympiadBench, demonstrating significant accuracy improvements of up to 13.6 percentage points over baselines. Our analysis shows that SymCode is not only more token-efficient but also fundamentally shifts model failures from opaque logical fallacies towards transparent, programmatic errors. By grounding LLM reasoning in a deterministic symbolic engine, SymCode represents a key step towards more accurate and trustworthy AI in formal domains.
>
---
#### [new 010] RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline
- **分类: cs.CL; I.2**

- **简介: 该论文提出RECAP，一种用于从大语言模型中提取记忆化训练数据的智能流水线。针对无法直接访问训练数据的问题，通过反馈循环与狱破解模块，驱动模型逐步还原版权文本，显著提升提取准确率，验证了模型对原始数据的记忆能力。**

- **链接: [http://arxiv.org/pdf/2510.25941v1](http://arxiv.org/pdf/2510.25941v1)**

> **作者:** André V. Duarte; Xuying li; Bin Zeng; Arlindo L. Oliveira; Lei Li; Zhuo Li
>
> **摘要:** If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
>
---
#### [new 011] LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection
- **分类: cs.CL**

- **简介: 该论文提出LISTEN框架，解决多目标决策中专家难以形式化隐性偏好的问题。利用大语言模型作为零样本偏好代理，通过自然语言表达优先级，结合迭代算法实现高效选择。在航班预订、购物等任务中验证了其有效性，为自然语言驱动的复杂决策提供了新路径。**

- **链接: [http://arxiv.org/pdf/2510.25799v1](http://arxiv.org/pdf/2510.25799v1)**

> **作者:** Adam S. Jovine; Tinghan Ye; Francis Bahk; Jingjing Wang; David B. Shmoys; Peter I. Frazier
>
> **摘要:** Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN, a framework that leverages a Large Language Model (LLM) as a zero-shot preference oracle, guided only by an expert's high-level priorities in natural language. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation.
>
---
#### [new 012] Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs
- **分类: cs.CL**

- **简介: 该论文研究语言模型理解隐含意义的任务。针对模型难以准确推断言外之意的问题，提出在提示中引入格赖斯语用学和关联理论，引导模型进行分步推理。实验表明，该方法显著提升模型表现，尤其在大模型中仅提及理论名称也能带来性能改善。**

- **链接: [http://arxiv.org/pdf/2510.26253v1](http://arxiv.org/pdf/2510.26253v1)**

> **作者:** Takuma Sato; Seiya Kawano; Koichiro Yoshino
>
> **摘要:** The ability to accurately interpret implied meanings plays a crucial role in human communication and language use, and language models are also expected to possess this capability. This study demonstrates that providing language models with pragmatic theories as prompts is an effective in-context learning approach for tasks to understand implied meanings. Specifically, we propose an approach in which an overview of pragmatic theories, such as Gricean pragmatics and Relevance Theory, is presented as a prompt to the language model, guiding it through a step-by-step reasoning process to derive a final interpretation. Experimental results showed that, compared to the baseline, which prompts intermediate reasoning without presenting pragmatic theories (0-shot Chain-of-Thought), our methods enabled language models to achieve up to 9.6\% higher scores on pragmatic reasoning tasks. Furthermore, we show that even without explaining the details of pragmatic theories, merely mentioning their names in the prompt leads to a certain performance improvement (around 1-3%) in larger models compared to the baseline.
>
---
#### [new 013] QCoder Benchmark: Bridging Language Generation and Quantum Hardware through Simulator-Based Feedback
- **分类: cs.CL; cs.PL; quant-ph**

- **简介: 该论文提出QCoder Benchmark，用于评估大语言模型在量子编程任务中的表现。针对语言模型在硬件交互领域（如量子计算）应用不足的问题，构建基于量子模拟器反馈的评估框架，结合真实人类代码进行对比，揭示模型生成准确率低的挑战，并验证推理增强模型的优越性。**

- **链接: [http://arxiv.org/pdf/2510.26101v1](http://arxiv.org/pdf/2510.26101v1)**

> **作者:** Taku Mikuriya; Tatsuya Ishigaki; Masayuki Kawarada; Shunya Minami; Tadashi Kadowaki; Yohichi Suzuki; Soshun Naito; Shunya Takata; Takumi Kato; Tamotsu Basseda; Reo Yamada; Hiroya Takamura
>
> **摘要:** Large language models (LLMs) have increasingly been applied to automatic programming code generation. This task can be viewed as a language generation task that bridges natural language, human knowledge, and programming logic. However, it remains underexplored in domains that require interaction with hardware devices, such as quantum programming, where human coders write Python code that is executed on a quantum computer. To address this gap, we introduce QCoder Benchmark, an evaluation framework that assesses LLMs on quantum programming with feedback from simulated hardware devices. Our benchmark offers two key features. First, it supports evaluation using a quantum simulator environment beyond conventional Python execution, allowing feedback of domain-specific metrics such as circuit depth, execution time, and error classification, which can be used to guide better generation. Second, it incorporates human-written code submissions collected from real programming contests, enabling both quantitative comparisons and qualitative analyses of LLM outputs against human-written codes. Our experiments reveal that even advanced models like GPT-4o achieve only around 18.97% accuracy, highlighting the difficulty of the benchmark. In contrast, reasoning-based models such as o3 reach up to 78% accuracy, outperforming averaged success rates of human-written codes (39.98%). We release the QCoder Benchmark dataset and public evaluation API to support further research.
>
---
#### [new 014] Review Based Entity Ranking using Fuzzy Logic Algorithmic Approach: Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决传统方法忽略观点强度的问题。通过融合模糊逻辑与语法依存分析，将评论中的情感词按强弱等级分类，结合用户查询与产品属性，计算实体在各方面的综合得分，实现基于观点强度的实体排序。**

- **链接: [http://arxiv.org/pdf/2510.25778v1](http://arxiv.org/pdf/2510.25778v1)**

> **作者:** Pratik N. Kalamkar; Anupama G. Phakatkar
>
> **备注:** 10 pages, 3 figures, International Journal Of Engineering And Computer Science ISSN:2319-7242
>
> **摘要:** Opinion mining, also called sentiment analysis, is the field of study that analyzes people opinions, sentiments, evaluations, appraisals, attitudes, and emotions towards entities such as products, services, organizations, individuals, issues, events, topics, and their attributes. Holistic lexicon-based approach does not consider the strength of each opinion, i.e., whether the opinion is very strongly negative (or positive), strongly negative (or positive), moderate negative (or positive), very weakly negative (or positive) and weakly negative (or positive). In this paper, we propose approach to rank entities based on orientation and strength of the entity reviews and user's queries by classifying them in granularity levels (i.e. very weak, weak, moderate, very strong and strong) by combining opinion words (i.e. adverb, adjective, noun and verb) that are related to aspect of interest of certain product. We shall use fuzzy logic algorithmic approach in order to classify opinion words into different category and syntactic dependency resolution to find relations for desired aspect words. Opinion words related to certain aspects of interest are considered to find the entity score for that aspect in the review.
>
---
#### [new 015] Beyond Long Context: When Semantics Matter More than Tokens
- **分类: cs.CL; cs.LG; 68T50, 68T07; I.2.7; H.3.3**

- **简介: 该论文聚焦临床问答任务，针对电子病历中长文本语义理解难题，提出实体增强的检索方法CLEAR。通过减少78%的令牌使用量，在长文档上实现更高准确率与效率，验证了语义精度优于长上下文依赖的可行性。**

- **链接: [http://arxiv.org/pdf/2510.25816v1](http://arxiv.org/pdf/2510.25816v1)**

> **作者:** Tarun Kumar Chawdhury; Jon D. Duke
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Electronic Health Records (EHR) store clinical documentation as base64 encoded attachments in FHIR DocumentReference resources, which makes semantic question answering difficult. Traditional vector database methods often miss nuanced clinical relationships. The Clinical Entity Augmented Retrieval (CLEAR) method, introduced by Lopez et al. 2025, uses entity aware retrieval and achieved improved performance with an F1 score of 0.90 versus 0.86 for embedding based retrieval, while using over 70 percent fewer tokens. We developed a Clinical Notes QA Evaluation Platform to validate CLEAR against zero shot large context inference and traditional chunk based retrieval augmented generation. The platform was tested on 12 clinical notes ranging from 10,000 to 65,000 tokens representing realistic EHR content. CLEAR achieved a 58.3 percent win rate, an average semantic similarity of 0.878, and used 78 percent fewer tokens than wide context processing. The largest performance gains occurred on long notes, with a 75 percent win rate for documents exceeding 65,000 tokens. These findings confirm that entity aware retrieval improves both efficiency and accuracy in clinical natural language processing. The evaluation framework provides a reusable and transparent benchmark for assessing clinical question answering systems where semantic precision and computational efficiency are critical.
>
---
#### [new 016] Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型推理延迟问题，提出一种考虑硬件与批量开销的动态树结构推理方法CAST。通过感知推理成本动态优化生成树，显著提升推理效率，在六项任务上最快达传统方法5.2倍，优于现有先进方法5%-20%。**

- **链接: [http://arxiv.org/pdf/2510.26577v1](http://arxiv.org/pdf/2510.26577v1)**

> **作者:** Yinrong Hong; Zhiquan Tan; Kai Hu
>
> **摘要:** Large Language Models (LLMs) face significant inference latency challenges stemming from their autoregressive design and large size. To address this, speculative decoding emerges as a solution, enabling the simultaneous generation and validation of multiple tokens. While recent approaches like EAGLE-2 and EAGLE-3 improve speculative decoding using dynamic tree structures, they often neglect the impact of crucial system variables such as GPU devices and batch sizes. Therefore, we introduce a new dynamic tree decoding approach called CAST that takes into account inference costs, including factors such as GPU configurations and batch sizes, to dynamically refine the tree structure. Through comprehensive experimentation across six diverse tasks and utilizing six distinct LLMs, our methodology demonstrates remarkable results, achieving speeds up to 5.2 times faster than conventional decoding methods. Moreover, it generally outperforms existing state-of-the-art techniques from 5% to 20%.
>
---
#### [new 017] Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Evontree框架，针对医疗等低资源领域中大模型因缺乏高质量训练数据而难以适配的问题，利用少量领域本体规则自动提取、验证并增强模型内部知识，通过自蒸馏微调提升性能，在医学问答任务上显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.26683v1](http://arxiv.org/pdf/2510.26683v1)**

> **作者:** Mingchen Tu; Zhiqiang Liu; Juan Li; Liangyurui Liu; Junjie Wang; Lei Liang; Wen Zhang
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional capabilities across multiple domains by leveraging massive pre-training and curated fine-tuning data. However, in data-sensitive fields such as healthcare, the lack of high-quality, domain-specific training corpus hinders LLMs' adaptation for specialized applications. Meanwhile, domain experts have distilled domain wisdom into ontology rules, which formalize relationships among concepts and ensure the integrity of knowledge management repositories. Viewing LLMs as implicit repositories of human knowledge, we propose Evontree, a novel framework that leverages a small set of high-quality ontology rules to systematically extract, validate, and enhance domain knowledge within LLMs, without requiring extensive external datasets. Specifically, Evontree extracts domain ontology from raw models, detects inconsistencies using two core ontology rules, and reinforces the refined knowledge via self-distilled fine-tuning. Extensive experiments on medical QA benchmarks with Llama3-8B-Instruct and Med42-v2 demonstrate consistent outperformance over both unmodified models and leading supervised baselines, achieving up to a 3.7% improvement in accuracy. These results confirm the effectiveness, efficiency, and robustness of our approach for low-resource domain adaptation of LLMs.
>
---
#### [new 018] AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型预填充阶段自注意力计算慢的问题，提出AttnCache框架。通过缓存并复用相似注意力图，显著加速推理。实验表明，该方法在CPU和GPU上均实现1.2–3倍速度提升，且精度损失极小。**

- **链接: [http://arxiv.org/pdf/2510.25979v1](http://arxiv.org/pdf/2510.25979v1)**

> **作者:** Dinghong Song; Yuan Feng; Yiwei Wang; Shangye Chen; Cyril Guyot; Filip Blagojevic; Hyeran Jeon; Pengfei Su; Dong Li
>
> **备注:** 10 pages, 6 figures, submitted to Ninth Annual Conference on Machine Learning and Systems (MLSys'26)
>
> **摘要:** Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
>
---
#### [new 019] SCRIBE: Structured Chain Reasoning for Interactive Behaviour Explanations using Tool Calling
- **分类: cs.CL**

- **简介: 该论文提出SCRIBE框架，用于教育场景中生成可信、可交互的学生反馈。针对隐私、资源受限与教学有效性问题，结合领域工具与自反思推理，通过两阶段微调将小模型（3B/8B）能力提升至接近大模型水平，实现在本地运行下高质量的多跳推理与错误恢复。**

- **链接: [http://arxiv.org/pdf/2510.26322v1](http://arxiv.org/pdf/2510.26322v1)**

> **作者:** Fares Fawzi; Vinitra Swamy; Dominik Glandorf; Tanya Nazaretsky; Tanja Käser
>
> **摘要:** Language models can be used to provide interactive, personalized student feedback in educational settings. However, real-world deployment faces three key challenges: privacy concerns, limited computational resources, and the need for pedagogically valid responses. These constraints require small, open-source models that can run locally and reliably ground their outputs in correct information. We introduce SCRIBE, a framework for multi-hop, tool-augmented reasoning designed to generate valid responses to student questions about feedback reports. SCRIBE combines domain-specific tools with a self-reflective inference pipeline that supports iterative reasoning, tool use, and error recovery. We distil these capabilities into 3B and 8B models via two-stage LoRA fine-tuning on synthetic GPT-4o-generated data. Evaluation with a human-aligned GPT-Judge and a user study with 108 students shows that 8B-SCRIBE models achieve comparable or superior quality to much larger models in key dimensions such as relevance and actionability, while being perceived on par with GPT-4o and Llama-3.3 70B by students. These findings demonstrate the viability of SCRIBE for low-resource, privacy-sensitive educational applications.
>
---
#### [new 020] Bayesian Network Fusion of Large Language Models for Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对情感分析任务，解决大语言模型（LLM）缺乏可解释性、成本高、结果不一致等问题。提出贝叶斯网络融合框架（BNLF），通过概率机制整合FinBERT、RoBERTa和BERTweet的预测，实现可解释的晚期融合，显著提升跨领域准确性。**

- **链接: [http://arxiv.org/pdf/2510.26484v1](http://arxiv.org/pdf/2510.26484v1)**

> **作者:** Rasoul Amirzadeh; Dhananjay Thiruvady; Fatemeh Shiri
>
> **摘要:** Large language models (LLMs) continue to advance, with an increasing number of domain-specific variants tailored for specialised tasks. However, these models often lack transparency and explainability, can be costly to fine-tune, require substantial prompt engineering, yield inconsistent results across domains, and impose significant adverse environmental impact due to their high computational demands. To address these challenges, we propose the Bayesian network LLM fusion (BNLF) framework, which integrates predictions from three LLMs, including FinBERT, RoBERTa, and BERTweet, through a probabilistic mechanism for sentiment analysis. BNLF performs late fusion by modelling the sentiment predictions from multiple LLMs as probabilistic nodes within a Bayesian network. Evaluated across three human-annotated financial corpora with distinct linguistic and contextual characteristics, BNLF demonstrates consistent gains of about six percent in accuracy over the baseline LLMs, underscoring its robustness to dataset variability and the effectiveness of probabilistic fusion for interpretable sentiment classification.
>
---
#### [new 021] Distilling Multilingual Vision-Language Models: When Smaller Models Stay Multilingual
- **分类: cs.CL**

- **简介: 该论文研究多语言视觉-语言模型的知识蒸馏，旨在解决小模型在压缩后跨语言性能下降的问题。通过对比五种蒸馏方法，发现部分方法能在模型缩小一半时保持甚至提升多语言检索鲁棒性，揭示了蒸馏设计对跨任务稳定性的关键影响。**

- **链接: [http://arxiv.org/pdf/2510.26271v1](http://arxiv.org/pdf/2510.26271v1)**

> **作者:** Sukrit Sriratanawilai; Jhayahgrit Thongwat; Romrawin Chumpu; Patomporn Payoungkhamdee; Sarana Nutanong; Peerat Limkonchotiwat
>
> **备注:** Work in progress
>
> **摘要:** Vision-language models (VLMs) exhibit uneven performance across languages, a problem that is often exacerbated when the model size is reduced. While Knowledge distillation (KD) demonstrates promising results in transferring knowledge from larger to smaller VLMs, applying KD in multilingualism is an underexplored area. This paper presents a controlled empirical study of KD behavior across five distillation approaches, isolating their effects on cross-lingual representation consistency and downstream performance stability under model compression. We study five distillation formulations across CLIP and SigLIP2, and evaluate them on in-domain retrieval and out-of-domain visual QA. We find that some configurations preserve or even improve multilingual retrieval robustness despite halving model size, but others fail to maintain cross-task stability, exposing design-sensitive trade-offs that aggregate accuracy alone does not reveal.
>
---
#### [new 022] What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WIMHF方法，通过稀疏自编码器解释人类反馈数据中的可解释特征。旨在解决偏好数据中隐含信息不透明的问题，揭示人类真实偏好与数据测量能力。在7个数据集上识别出少数关键特征，实现安全数据筛选与个性化建模，提升模型可控性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.26202v1](http://arxiv.org/pdf/2510.26202v1)**

> **作者:** Rajiv Movva; Smitha Milli; Sewon Min; Emma Pierson
>
> **备注:** Code: https://github.com/rmovva/wimhf
>
> **摘要:** Human feedback can alter language models in unpredictable and undesirable ways, as practitioners lack a clear understanding of what feedback data encodes. While prior work studies preferences over certain attributes (e.g., length or sycophancy), automatically extracting relevant features without pre-specifying hypotheses remains challenging. We introduce What's In My Human Feedback? (WIMHF), a method to explain feedback data using sparse autoencoders. WIMHF characterizes both (1) the preferences a dataset is capable of measuring and (2) the preferences that the annotators actually express. Across 7 datasets, WIMHF identifies a small number of human-interpretable features that account for the majority of the preference prediction signal achieved by black-box models. These features reveal a wide diversity in what humans prefer, and the role of dataset-level context: for example, users on Reddit prefer informality and jokes, while annotators in HH-RLHF and PRISM disprefer them. WIMHF also surfaces potentially unsafe preferences, such as that LMArena users tend to vote against refusals, often in favor of toxic content. The learned features enable effective data curation: re-labeling the harmful examples in Arena yields large safety gains (+37%) with no cost to general performance. They also allow fine-grained personalization: on the Community Alignment dataset, we learn annotator-specific weights over subjective features that improve preference prediction. WIMHF provides a human-centered analysis method for practitioners to better understand and use preference data.
>
---
#### [new 023] InfoFlow: Reinforcing Search Agent Via Reward Density Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对深度搜索中奖励稀疏问题，提出InfoFlow框架，通过子任务分解、失败引导提示和双代理优化，提升单位探索成本的奖励密度。旨在增强基于强化学习的搜索智能体性能，使轻量级大模型达到先进专有模型水平。**

- **链接: [http://arxiv.org/pdf/2510.26575v1](http://arxiv.org/pdf/2510.26575v1)**

> **作者:** Kun Luo; Hongjin Qian; Zheng Liu; Ziyi Xia; Shitao Xiao; Siqi Bao; Jun Zhao; Kang Liu
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a promising approach for enhancing agentic deep search. However, its application is often hindered by low \textbf{Reward Density} in deep search scenarios, where agents expend significant exploratory costs for infrequent and often null final rewards. In this paper, we formalize this challenge as the \textbf{Reward Density Optimization} problem, which aims to improve the reward obtained per unit of exploration cost. This paper introduce \textbf{InfoFlow}, a systematic framework that tackles this problem from three aspects. 1) \textbf{Subproblem decomposition}: breaking down long-range tasks to assign process rewards, thereby providing denser learning signals. 2) \textbf{Failure-guided hints}: injecting corrective guidance into stalled trajectories to increase the probability of successful outcomes. 3) \textbf{Dual-agent refinement}: employing a dual-agent architecture to offload the cognitive burden of deep exploration. A refiner agent synthesizes the search history, which effectively compresses the researcher's perceived trajectory, thereby reducing exploration cost and increasing the overall reward density. We evaluate InfoFlow on multiple agentic search benchmarks, where it significantly outperforms strong baselines, enabling lightweight LLMs to achieve performance comparable to advanced proprietary LLMs.
>
---
#### [new 024] Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model
- **分类: cs.CL**

- **简介: 该论文对比编码器-解码器（RedLLM）与仅解码器（DecLLM）架构的大语言模型，旨在验证前者在规模扩展与推理效率上的潜力。通过引入最新训练技巧，在不同规模下进行预训练与指令微调，发现RedLLM具备强扩展性与高效推理能力，挑战了当前解码器主导的范式。**

- **链接: [http://arxiv.org/pdf/2510.26622v1](http://arxiv.org/pdf/2510.26622v1)**

> **作者:** Biao Zhang; Yong Cheng; Siamak Shakeri; Xinyi Wang; Min Ma; Orhan Firat
>
> **备注:** The scaling study inspiring T5Gemma
>
> **摘要:** Recent large language model (LLM) research has undergone an architectural shift from encoder-decoder modeling to nowadays the dominant decoder-only modeling. This rapid transition, however, comes without a rigorous comparative analysis especially \textit{from the scaling perspective}, raising concerns that the potential of encoder-decoder models may have been overlooked. To fill this gap, we revisit encoder-decoder LLM (RedLLM), enhancing it with recent recipes from decoder-only LLM (DecLLM). We conduct a comprehensive comparison between RedLLM, pretrained with prefix language modeling (LM), and DecLLM, pretrained with causal LM, at different model scales, ranging from $\sim$150M to $\sim$8B. Using RedPajama V1 (1.6T tokens) for pretraining and FLAN for instruction tuning, our experiments show that RedLLM produces compelling scaling properties and surprisingly strong performance. While DecLLM is overall more compute-optimal during pretraining, RedLLM demonstrates comparable scaling and context length extrapolation capabilities. After instruction tuning, RedLLM achieves comparable and even better results on various downstream tasks while enjoying substantially better inference efficiency. We hope our findings could inspire more efforts on re-examining RedLLM, unlocking its potential for developing powerful and efficient LLMs.
>
---
#### [new 025] Revisiting Multilingual Data Mixtures in Language Model Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言数据混合对大语言模型预训练的影响，旨在解决“多语言诅咒”争议。通过在25至400种语言上训练1.1B和3B参数模型，发现合理平衡的多语言数据可提升性能且不损害单语言表现，英语作为枢纽语言具广泛益处，且语言家族内选枢纽无显著优势。**

- **链接: [http://arxiv.org/pdf/2510.25947v1](http://arxiv.org/pdf/2510.25947v1)**

> **作者:** Negar Foroutan; Paul Teiletche; Ayush Kumar Tarun; Antoine Bosselut
>
> **备注:** Under Review
>
> **摘要:** The impact of different multilingual data mixtures in pretraining large language models (LLMs) has been a topic of ongoing debate, often raising concerns about potential trade-offs between language coverage and model performance (i.e., the curse of multilinguality). In this work, we investigate these assumptions by training 1.1B and 3B parameter LLMs on diverse multilingual corpora, varying the number of languages from 25 to 400. Our study challenges common beliefs surrounding multilingual training. First, we find that combining English and multilingual data does not necessarily degrade the in-language performance of either group, provided that languages have a sufficient number of tokens included in the pretraining corpus. Second, we observe that using English as a pivot language (i.e., a high-resource language that serves as a catalyst for multilingual generalization) yields benefits across language families, and contrary to expectations, selecting a pivot language from within a specific family does not consistently improve performance for languages within that family. Lastly, we do not observe a significant "curse of multilinguality" as the number of training languages increases in models at this scale. Our findings suggest that multilingual data, when balanced appropriately, can enhance language model capabilities without compromising performance, even in low-resource settings
>
---
#### [new 026] Ideology-Based LLMs for Content Moderation
- **分类: cs.CL**

- **简介: 该论文研究意识形态导向的LLM在内容审核中的公平性问题。针对模型判断偏见隐现于表面准确率下的现象，通过对比不同意识形态角色下大模型对文本与视觉内容的有害性分类，揭示了模型间因意识形态趋同而加剧的判断分歧，证明角色设定会引入隐蔽的意识形态偏差，挑战其“中立”假象。**

- **链接: [http://arxiv.org/pdf/2510.25805v1](http://arxiv.org/pdf/2510.25805v1)**

> **作者:** Stefano Civelli; Pietro Bernardelle; Nardiena A. Pratama; Gianluca Demartini
>
> **摘要:** Large language models (LLMs) are increasingly used in content moderation systems, where ensuring fairness and neutrality is essential. In this study, we examine how persona adoption influences the consistency and fairness of harmful content classification across different LLM architectures, model sizes, and content modalities (language vs. vision). At first glance, headline performance metrics suggest that personas have little impact on overall classification accuracy. However, a closer analysis reveals important behavioral shifts. Personas with different ideological leanings display distinct propensities to label content as harmful, showing that the lens through which a model "views" input can subtly shape its judgments. Further agreement analyses highlight that models, particularly larger ones, tend to align more closely with personas from the same political ideology, strengthening within-ideology consistency while widening divergence across ideological groups. To show this effect more directly, we conducted an additional study on a politically targeted task, which confirmed that personas not only behave more coherently within their own ideology but also exhibit a tendency to defend their perspective while downplaying harmfulness in opposing views. Together, these findings highlight how persona conditioning can introduce subtle ideological biases into LLM outputs, raising concerns about the use of AI systems that may reinforce partisan perspectives under the guise of neutrality.
>
---
#### [new 027] Language Models Are Borrowing-Blind: A Multilingual Evaluation of Loanword Identification across 10 Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的借词识别任务，旨在评估预训练语言模型在10种语言中区分借词与本族词的能力。研究发现，模型普遍表现不佳，存在对借词的偏好偏差，揭示了现有NLP系统在支持少数语言保护方面的局限性。**

- **链接: [http://arxiv.org/pdf/2510.26254v1](http://arxiv.org/pdf/2510.26254v1)**

> **作者:** Mérilin Sousa Silva; Sina Ahmadi
>
> **备注:** Under review
>
> **摘要:** Throughout language history, words are borrowed from one language to another and gradually become integrated into the recipient's lexicon. Speakers can often differentiate these loanwords from native vocabulary, particularly in bilingual communities where a dominant language continuously imposes lexical items on a minority language. This paper investigates whether pretrained language models, including large language models, possess similar capabilities for loanword identification. We evaluate multiple models across 10 languages. Despite explicit instructions and contextual information, our results show that models perform poorly in distinguishing loanwords from native ones. These findings corroborate previous evidence that modern NLP systems exhibit a bias toward loanwords rather than native equivalents. Our work has implications for developing NLP tools for minority languages and supporting language preservation in communities under lexical pressure from dominant languages.
>
---
#### [new 028] MisSynth: Improving MISSCI Logical Fallacies Classification with Synthetic Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对科学类虚假信息中的逻辑谬误识别任务，提出MisSynth框架，利用检索增强生成技术合成谬误样本，通过轻量微调提升大模型性能。有效缓解标注数据稀缺问题，显著提升分类准确率，尤其在零样本场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.26345v1](http://arxiv.org/pdf/2510.26345v1)**

> **作者:** Mykhailo Poliakov; Nadiya Shvai
>
> **摘要:** Health-related misinformation is very prevalent and potentially harmful. It is difficult to identify, especially when claims distort or misinterpret scientific findings. We investigate the impact of synthetic data generation and lightweight fine-tuning techniques on the ability of large language models (LLMs) to recognize fallacious arguments using the MISSCI dataset and framework. In this work, we propose MisSynth, a pipeline that applies retrieval-augmented generation (RAG) to produce synthetic fallacy samples, which are then used to fine-tune an LLM model. Our results show substantial accuracy gains with fine-tuned models compared to vanilla baselines. For instance, the LLaMA 3.1 8B fine-tuned model achieved an over 35% F1-score absolute improvement on the MISSCI test split over its vanilla baseline. We demonstrate that introducing synthetic fallacy data to augment limited annotated resources can significantly enhance zero-shot LLM classification performance on real-world scientific misinformation tasks, even with limited computational resources. The code and synthetic dataset are available on https://github.com/mxpoliakov/MisSynth.
>
---
#### [new 029] Can Agent Conquer Web? Exploring the Frontiers of ChatGPT Atlas Agent in Web Games
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估ChatGPT Atlas在网页游戏中的交互能力，属于人机交互与AI代理任务。针对其在动态网页环境中实时操作能力不足的问题，通过T-Rex Runner、Sudoku等游戏测试，发现其在逻辑任务中表现优异，但在需精准时序控制的实时游戏中表现不佳，揭示了当前AI代理在动态交互中的局限性。**

- **链接: [http://arxiv.org/pdf/2510.26298v1](http://arxiv.org/pdf/2510.26298v1)**

> **作者:** Jingran Zhang; Ning Li; Justin Cui
>
> **摘要:** OpenAI's ChatGPT Atlas introduces new capabilities for web interaction, enabling the model to analyze webpages, process user intents, and execute cursor and keyboard inputs directly within the browser. While its capacity for information retrieval tasks has been demonstrated, its performance in dynamic, interactive environments remains less explored. In this study, we conduct an early evaluation of Atlas's web interaction capabilities using browser-based games as test scenarios, including Google's T-Rex Runner, Sudoku, Flappy Bird, and Stein.world. We employ in-game performance scores as quantitative metrics to assess performance across different task types. Our results show that Atlas performs strongly in logical reasoning tasks like Sudoku, completing puzzles significantly faster than human baselines, but struggles substantially in real-time games requiring precise timing and motor control, often failing to progress beyond initial obstacles. These findings suggest that while Atlas demonstrates capable analytical processing, there remain notable limitations in dynamic web environments requiring real-time interaction. The website of our project can be found at https://atlas-game-eval.github.io.
>
---
#### [new 030] Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在全局信息推理上的不足，提出全球检索增强生成（GlobalRAG）框架。针对现有基准仅评估局部检索的问题，构建首个面向文档集合级推理的基准GlobalQA，涵盖计数、极值、排序与前K提取四类任务。通过多工具协同与智能过滤机制，显著提升模型在全局推理任务上的性能。**

- **链接: [http://arxiv.org/pdf/2510.26205v1](http://arxiv.org/pdf/2510.26205v1)**

> **作者:** Qi Luo; Xiaonan Li; Tingshuo Fan; Xinchi Chen; Xipeng Qiu
>
> **摘要:** Retrieval-augmented generation (RAG) has emerged as a leading approach to reducing hallucinations in large language models (LLMs). Current RAG evaluation benchmarks primarily focus on what we call local RAG: retrieving relevant chunks from a small subset of documents to answer queries that require only localized understanding within specific text chunks. However, many real-world applications require a fundamentally different capability -- global RAG -- which involves aggregating and analyzing information across entire document collections to derive corpus-level insights (for example, "What are the top 10 most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first benchmark specifically designed to evaluate global RAG capabilities, covering four core task types: counting, extremum queries, sorting, and top-k extraction. Through systematic evaluation across different models and baselines, we find that existing RAG methods perform poorly on global tasks, with the strongest baseline achieving only 1.51 F1 score. To address these challenges, we propose GlobalRAG, a multi-tool collaborative framework that preserves structural coherence through chunk-level retrieval, incorporates LLM-driven intelligent filters to eliminate noisy documents, and integrates aggregation modules for precise symbolic computation. On the Qwen2.5-14B model, GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1, validating the effectiveness of our method.
>
---
#### [new 031] BlackboxNLP-2025 MIB Shared Task: Improving Circuit Faithfulness via Better Edge Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于机械可解释性中的电路发现任务，旨在提升模型电路的忠实度。针对现有方法在边缘选择上的不足，提出三种改进：基于自助法识别稳定边、比率筛选强正向边、用整数线性规划替代贪心策略。实验表明，新方法显著提升电路忠实度与性能。**

- **链接: [http://arxiv.org/pdf/2510.25786v1](http://arxiv.org/pdf/2510.25786v1)**

> **作者:** Yaniv Nikankin; Dana Arad; Itay Itzhak; Anja Reusch; Adi Simhi; Gal Kesten-Pomeranz; Yonatan Belinkov
>
> **摘要:** One of the main challenges in mechanistic interpretability is circuit discovery, determining which parts of a model perform a given task. We build on the Mechanistic Interpretability Benchmark (MIB) and propose three key improvements to circuit discovery. First, we use bootstrapping to identify edges with consistent attribution scores. Second, we introduce a simple ratio-based selection strategy to prioritize strong positive-scoring edges, balancing performance and faithfulness. Third, we replace the standard greedy selection with an integer linear programming formulation. Our methods yield more faithful circuits and outperform prior approaches across multiple MIB tasks and models. Our code is available at: https://github.com/technion-cs-nlp/MIB-Shared-Task.
>
---
#### [new 032] Inside CORE-KG: Evaluating Structured Prompting and Coreference Resolution for Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文研究法律文本中知识图谱的自动化构建任务，针对非结构化法律文档导致的知识图谱噪声与节点重复问题，提出并评估CORE-KG框架。通过消融实验验证了类型感知共指消解和领域引导结构化提示的有效性，证明二者协同显著提升图谱质量。**

- **链接: [http://arxiv.org/pdf/2510.26512v1](http://arxiv.org/pdf/2510.26512v1)**

> **作者:** Dipak Meher; Carlotta Domeniconi
>
> **备注:** ICDM 2025 Workshop
>
> **摘要:** Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer critical insights but are often unstructured, lexically dense, and filled with ambiguous or shifting references, which pose significant challenges for automated knowledge graph (KG) construction. While recent LLM-based approaches improve over static templates, they still generate noisy, fragmented graphs with duplicate nodes due to the absence of guided extraction and coreference resolution. The recently proposed CORE-KG framework addresses these limitations by integrating a type-aware coreference module and domain-guided structured prompts, significantly reducing node duplication and legal noise. In this work, we present a systematic ablation study of CORE-KG to quantify the individual contributions of its two key components. Our results show that removing coreference resolution results in a 28.32% increase in node duplication and a 4.32% increase in noisy nodes, while removing structured prompts leads to a 4.34% increase in node duplication and a 73.33% increase in noisy nodes. These findings offer empirical insights for designing robust LLM-based pipelines for extracting structured representations from complex legal texts.
>
---
#### [new 033] On the Influence of Discourse Relations in Persuasive Texts
- **分类: cs.CL; I.2.7; I.2.6**

- **简介: 该论文研究说服文本中修辞技巧（PTs）与话语关系（DRs）的关联。针对缺乏同时标注PTs和DRs的数据集问题，利用大语言模型和提示工程，基于SemEval 2023数据生成含22类DRs的银色数据集。通过统计分析发现六类话语关系在说服策略中起关键作用，有助于识别网络宣传与虚假信息。**

- **链接: [http://arxiv.org/pdf/2510.26124v1](http://arxiv.org/pdf/2510.26124v1)**

> **作者:** Nawar Turk; Sevag Kaspar; Leila Kosseim
>
> **备注:** Published in Proceedings of the 38th Canadian Conference on Artificial Intelligence CanAI 2025 Calgary Alberta May 26-27 2025. 5 figures 7 tables
>
> **摘要:** This paper investigates the relationship between Persuasion Techniques (PTs) and Discourse Relations (DRs) by leveraging Large Language Models (LLMs) and prompt engineering. Since no dataset annotated with both PTs and DRs exists, we took the SemEval 2023 Task 3 dataset labelled with 19 PTs as a starting point and developed LLM-based classifiers to label each instance of the dataset with one of the 22 PDTB 3.0 level-2 DRs. In total, four LLMs were evaluated using 10 different prompts, resulting in 40 unique DR classifiers. Ensemble models using different majority-pooling strategies were used to create 5 silver datasets of instances labelled with both persuasion techniques and level-2 PDTB senses. The silver dataset sizes vary from 1,281 instances to 204 instances, depending on the majority pooling technique used. Statistical analysis of these silver datasets shows that six discourse relations (namely Cause, Purpose, Contrast, Cause+Belief, Concession, and Condition) play a crucial role in persuasive texts, especially in the use of Loaded Language, Exaggeration/Minimisation, Repetition and to cast Doubt. This insight can contribute to detecting online propaganda and misinformation, as well as to our general understanding of effective communication.
>
---
#### [new 034] RCScore: Quantifying Response Consistency in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出RCScore框架，用于量化大语言模型在不同指令风格下的响应一致性。针对现有评估忽视指令风格影响的问题，通过多风格变换分析模型表现差异，发现指令风格可导致准确率变化达16.7%。引入跨响应相似性（CRS）衡量自一致性，证明其与任务准确性强相关，揭示了模型规模与解码方式对风格稳定性的影响。**

- **链接: [http://arxiv.org/pdf/2510.26193v1](http://arxiv.org/pdf/2510.26193v1)**

> **作者:** Dongjun Jang; Youngchae Ahn; Hyopil Shin
>
> **摘要:** Current LLM evaluations often rely on a single instruction template, overlooking models' sensitivity to instruction style-a critical aspect for real-world deployments. We present RCScore, a multi-dimensional framework quantifying how instruction formulation affects model responses. By systematically transforming benchmark problems into multiple instruction styles, RCScore reveals performance variations undetected by conventional metrics. Our experiments across ten LLMs on four reasoning benchmarks demonstrate that instruction style can shift accuracy by up to 16.7% points. We introduce Cross-Response Similarity (CRS), a method applying RCScore metrics to measure stylistic self-consistency, and establish its strong correlation with task accuracy, suggesting consistency as a valuable proxy for model reliability. Additional findings show that deterministic decoding produces more stylistically stable outputs, and model scale correlates positively with cross-style consistency. RCScore offers a principled approach to assess instruction robustness.
>
---
#### [new 035] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives
- **分类: cs.CL**

- **简介: 该论文聚焦数据高效的大语言模型后训练任务，针对标注成本高、数据边际效益递减问题，提出数据中心视角的系统性分类，涵盖数据选择、质量提升、合成数据生成等五类方法，总结代表性技术并展望未来方向。**

- **链接: [http://arxiv.org/pdf/2510.25817v1](http://arxiv.org/pdf/2510.25817v1)**

> **作者:** Junyu Luo; Bohan Wu; Xiao Luo; Zhiping Xiao; Yiqiao Jin; Rong-Cheng Tu; Nan Yin; Yifan Wang; Jingyang Yuan; Wei Ju; Ming Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Post-training of Large Language Models (LLMs) is crucial for unlocking their task generalization potential and domain-specific capabilities. However, the current LLM post-training paradigm faces significant data challenges, including the high costs of manual annotation and diminishing marginal returns on data scales. Therefore, achieving data-efficient post-training has become a key research question. In this paper, we present the first systematic survey of data-efficient LLM post-training from a data-centric perspective. We propose a taxonomy of data-efficient LLM post-training methods, covering data selection, data quality enhancement, synthetic data generation, data distillation and compression, and self-evolving data ecosystems. We summarize representative approaches in each category and outline future research directions. By examining the challenges in data-efficient LLM post-training, we highlight open problems and propose potential research avenues. We hope our work inspires further exploration into maximizing the potential of data utilization in large-scale model training. Paper List: https://github.com/luo-junyu/Awesome-Data-Efficient-LLM
>
---
#### [new 036] Unravelling the Mechanisms of Manipulating Numbers in Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文研究大语言模型处理数字的机制，旨在解释模型在数字表示上准确却常出错的矛盾。通过分析隐藏层中的数字表示，发现其具系统性、高精度和通用性，进而构建通用探测器，定位错误根源，为改进模型架构提供依据。**

- **链接: [http://arxiv.org/pdf/2510.26285v1](http://arxiv.org/pdf/2510.26285v1)**

> **作者:** Michal Štefánik; Timothee Mickus; Marek Kadlčík; Bertram Højer; Michal Spiegel; Raúl Vázquez; Aman Sinha; Josef Kuchař; Philipp Mondorf
>
> **摘要:** Recent work has shown that different large language models (LLMs) converge to similar and accurate input embedding representations for numbers. These findings conflict with the documented propensity of LLMs to produce erroneous outputs when dealing with numeric information. In this work, we aim to explain this conflict by exploring how language models manipulate numbers and quantify the lower bounds of accuracy of these mechanisms. We find that despite surfacing errors, different language models learn interchangeable representations of numbers that are systematic, highly accurate and universal across their hidden states and the types of input contexts. This allows us to create universal probes for each LLM and to trace information -- including the causes of output errors -- to specific layers. Our results lay a fundamental understanding of how pre-trained LLMs manipulate numbers and outline the potential of more accurate probing techniques in addressed refinements of LLMs' architectures.
>
---
#### [new 037] zFLoRA: Zero-Latency Fused Low-Rank Adapters
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型部署中适配器带来的显著推理延迟问题，提出zFLoRA方法，通过零延迟融合低秩适配器，实现近乎无额外延迟的高效推理。实验表明，其在多种任务与硬件平台下均能保持高性能且延迟可忽略，优于传统LoRA和全微调方案。**

- **链接: [http://arxiv.org/pdf/2510.25784v1](http://arxiv.org/pdf/2510.25784v1)**

> **作者:** Dhananjaya Gowda; Seoha Song; Harshith Goka; Junhyun Lee
>
> **摘要:** Large language models (LLMs) are increasingly deployed with task-specific adapters catering to multiple downstream applications. In such a scenario, the additional compute associated with these apparently insignificant number of adapter parameters (typically less than 1% of the base model) turns out to be disproportionately significant during inference time (upto 2.5x times that of the base model). In this paper, we propose a new zero-latency fused low-rank adapter (zFLoRA) that introduces zero or negligible latency overhead on top of the base model. Experimental results on LLMs of size 1B, 3B and 7B show that zFLoRA compares favorably against the popular supervised fine-tuning benchmarks including low-rank adapters (LoRA) as well as full fine-tuning (FFT). Experiments are conducted on 18 different tasks across three different categories namely commonsense reasoning, math reasoning and summary-dialogue. Latency measurements made on NPU (Samsung Galaxy S25+) as well as GPU (NVIDIA H100) platforms show that the proposed zFLoRA adapters introduce zero to negligible latency overhead.
>
---
#### [new 038] NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium
- **分类: cs.CL**

- **简介: 该论文针对AWS Trainium加速器上大模型推理的矩阵乘法性能瓶颈，提出NeuronMM优化方案。通过核融合与创新缓存策略，减少数据移动、提升SRAM带宽利用率，避免矩阵转置开销。实验表明，相比AWS现有实现，矩阵乘法平均提速1.35倍，端到端推理平均提速1.66倍。**

- **链接: [http://arxiv.org/pdf/2510.25977v1](http://arxiv.org/pdf/2510.25977v1)**

> **作者:** Dinghong Song; Jierui Xu; Weichu Yang; Pengfei Su; Dong Li
>
> **备注:** 12 pages, 8 figures, submitted to the Proceedings of the Twenty-First European Conference on Computer Systems (EuroSys'26)
>
> **摘要:** AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
>
---
#### [new 039] Similarity-Distance-Magnitude Language Models
- **分类: cs.CL**

- **简介: 该论文提出相似性-距离-幅度语言模型（SDM LMs），通过监督微调使预训练语言模型在最终层使用SDM激活层，提升指令遵循的校准度与概率置信度。工作聚焦于减少模型拒绝生成（abstentions），采用对比编码与在线硬负例增强，显著提高统计效率。属于序列预测任务中的高质量生成优化问题。**

- **链接: [http://arxiv.org/pdf/2510.26183v1](http://arxiv.org/pdf/2510.26183v1)**

> **作者:** Allen Schmaltz
>
> **备注:** 8 pages, 5 tables
>
> **摘要:** We introduce Similarity-Distance-Magnitude (SDM) language models (LMs), which are sequence prediction models fine-tuned to maximize the proportion of generations in the well-calibrated, high-probability region partitioned by a final-layer SDM activation layer used for binary classification of instruction-following. We demonstrate that existing pre-trained decoder-only Transformer LMs can be readily converted into SDM LMs via supervised fine-tuning, using the final-layer SDM activation layer during training to estimate a change-of-base for a supervised next-token loss over a contrastive input encoding scheme, with additional hard negative examples generated online during training. This results in reduced abstentions (i.e., improved statistical efficiency) compared to strong supervised baselines.
>
---
#### [new 040] LASTIST: LArge-Scale Target-Independent STance dataset
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出LASTIST数据集，针对韩语立场检测中目标无关与历时演变任务。解决英语主导、低资源语言缺乏高质量标注数据的问题。基于韩国政党新闻稿构建56万条韩语标注语料，支持多种立场检测任务，并公开发布以促进研究。**

- **链接: [http://arxiv.org/pdf/2510.25783v1](http://arxiv.org/pdf/2510.25783v1)**

> **作者:** DongJae Kim; Yaejin Lee; Minsu Park; Eunil Park
>
> **备注:** 8 pages (two columned), 1 figure
>
> **摘要:** Stance detection has emerged as an area of research in the field of artificial intelligence. However, most research is currently centered on the target-dependent stance detection task, which is based on a person's stance in favor of or against a specific target. Furthermore, most benchmark datasets are based on English, making it difficult to develop models in low-resource languages such as Korean, especially for an emerging field such as stance detection. This study proposes the LArge-Scale Target-Independent STance (LASTIST) dataset to fill this research gap. Collected from the press releases of both parties on Korean political parties, the LASTIST dataset uses 563,299 labeled Korean sentences. We provide a detailed description of how we collected and constructed the dataset and trained state-of-the-art deep learning and stance detection models. Our LASTIST dataset is designed for various tasks in stance detection, including target-independent stance detection and diachronic evolution stance detection. We deploy our dataset on https://anonymous.4open.science/r/LASTIST-3721/.
>
---
#### [new 041] Do LLMs Signal When They're Right? Evidence from Neuron Agreement
- **分类: cs.CL**

- **简介: 该论文研究大模型推理中的无监督最优解选择问题。针对现有方法依赖外部输出信号且校准差的问题，提出基于神经元激活的Neuron Agreement Decoding（NAD），利用内部信号实现早期正确性预测与高效剪枝，显著降低计算成本并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.26277v1](http://arxiv.org/pdf/2510.26277v1)**

> **作者:** Kang Chen; Yaoning Wang; Kai Xiong; Zhuoka Feng; Wenhe Sun; Haotian Chen; Yixin Cao
>
> **摘要:** Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders, achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best-of-N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding.
>
---
#### [new 042] Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking
- **分类: cs.CL**

- **简介: 该论文针对大模型推理多样性不足的问题，提出1PNS训练范式与新度量指标RPD。通过衡量多步推理路径差异，筛选多样化解题方案，提升模型在测试时的思维多样性，显著增强推理性能。**

- **链接: [http://arxiv.org/pdf/2510.26122v1](http://arxiv.org/pdf/2510.26122v1)**

> **作者:** Feng Ju; Zeyu Qin; Rui Min; Zhitao He; Lingpeng Kong; Yi R. Fung
>
> **摘要:** While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at https://github.com/fengjujf/Reasoning-Path-Divergence .
>
---
#### [new 043] Evaluating the Impact of LLM-Assisted Annotation in a Perspectivized Setting: the Case of FrameNet Annotation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM辅助标注在框架语义标注中的应用，针对半自动标注的效率与质量进行评估。通过对比人工、全自动与半自动三种方式，发现半自动标注在提升框架多样性方面优于人工标注，且覆盖度相当，而全自动标注除速度外表现较差。**

- **链接: [http://arxiv.org/pdf/2510.25904v1](http://arxiv.org/pdf/2510.25904v1)**

> **作者:** Frederico Belcavello; Ely Matos; Arthur Lorenzi; Lisandra Bonoto; Lívia Ruiz; Luiz Fernando Pereira; Victor Herbst; Yulla Navarro; Helen de Andrade Abreu; Lívia Dutra; Tiago Timponi Torrent
>
> **摘要:** The use of LLM-based applications as a means to accelerate and/or substitute human labor in the creation of language resources and dataset is a reality. Nonetheless, despite the potential of such tools for linguistic research, comprehensive evaluation of their performance and impact on the creation of annotated datasets, especially under a perspectivized approach to NLP, is still missing. This paper contributes to reduction of this gap by reporting on an extensive evaluation of the (semi-)automatization of FrameNet-like semantic annotation by the use of an LLM-based semantic role labeler. The methodology employed compares annotation time, coverage and diversity in three experimental settings: manual, automatic and semi-automatic annotation. Results show that the hybrid, semi-automatic annotation setting leads to increased frame diversity and similar annotation coverage, when compared to the human-only setting, while the automatic setting performs considerably worse in all metrics, except for annotation time.
>
---
#### [new 044] The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文针对多智能体语言模型协作中团队组合难题，提出基于对话语义的图构建与社区检测方法，无需先验知识即可自动识别协同性强的模型群组，显著提升任务表现，实现高效、自动化的智能体团队设计。**

- **链接: [http://arxiv.org/pdf/2510.26352v1](http://arxiv.org/pdf/2510.26352v1)**

> **作者:** Kotaro Furuya; Yuichi Kitagawa
>
> **摘要:** While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams.
>
---
#### [new 045] PORTool: Tool-Use LLM Training with Rewarded Tree
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PORTool，一种基于奖励树的强化学习方法，用于训练工具使用大语言模型。针对静态数据训练导致探索不足的问题，通过多路径回放与步骤级奖励机制，鼓励模型在动态环境中探索多样解法，显著提升准确率与工具调用效率。**

- **链接: [http://arxiv.org/pdf/2510.26020v1](http://arxiv.org/pdf/2510.26020v1)**

> **作者:** Feijie Wu; Weiwu Zhu; Yuxiang Zhang; Soumya Chatterjee; Jiarong Zhu; Fan Mo; Rodin Luo; Jing Gao
>
> **摘要:** Current tool-use large language models (LLMs) are trained on static datasets, enabling them to interact with external tools and perform multi-step, tool-integrated reasoning, which produces tool-call trajectories. However, these models imitate how a query is resolved in a generic tool-call routine, thereby failing to explore possible solutions and demonstrating limited performance in an evolved, dynamic tool-call environment. In this work, we propose PORTool, a reinforcement learning (RL) method that encourages a tool-use LLM to explore various trajectories yielding the correct answer. Specifically, this method starts with generating multiple rollouts for a given query, and some of them share the first few tool-call steps, thereby forming a tree-like structure. Next, we assign rewards to each step, based on its ability to produce a correct answer and make successful tool calls. A shared step across different trajectories receives the same reward, while different steps under the same fork receive different rewards. Finally, these step-wise rewards are used to calculate fork-relative advantages, blended with trajectory-relative advantages, to train the LLM for tool use. The experiments utilize 17 tools to address user queries, covering both time-sensitive and time-invariant topics. We conduct ablation studies to systematically justify the necessity and the design robustness of step-wise rewards. Furthermore, we compare the proposed PORTool with other training approaches and demonstrate significant improvements in final accuracy and the number of tool-call steps.
>
---
#### [new 046] From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在专业领域（如经济、心理）表现不佳的问题，提出ACER框架，通过自动生成符合布卢姆认知分类的渐进式教学大纲与问答数据，对模型进行持续预训练。实验表明，该方法显著提升专业能力，同时避免遗忘并促进跨领域知识迁移。**

- **链接: [http://arxiv.org/pdf/2510.26336v1](http://arxiv.org/pdf/2510.26336v1)**

> **作者:** Nishit Neema; Srinjoy Mukherjee; Sapan Shah; Gokul Ramakrishnan; Ganesh Venkatesh
>
> **摘要:** Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom's taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions. Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs.
>
---
#### [new 047] A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool
- **分类: cs.CL**

- **简介: 该论文属于医学AI评估任务，旨在解决如何可靠评估临床AI分诊工具性能的问题。研究采用多代理大语言模型（LLM）集成框架，对29,766例头颅CT影像进行回顾性评估，比较单个与多个开源LLM及GPT-4o的性能，结果表明多LLM集成能更一致、可靠地生成“金标准”评价。**

- **链接: [http://arxiv.org/pdf/2510.26498v1](http://arxiv.org/pdf/2510.26498v1)**

> **作者:** Adam E. Flanders; Yifan Peng; Luciano Prevedello; Robyn Ball; Errol Colak; Prahlad Menon; George Shih; Hui-Ming Lin; Paras Lakhani
>
> **备注:** 29 pages, 3 figures, 4 tables
>
> **摘要:** Purpose: The purpose of this study was to determine if an ensemble of multiple LLM agents could be used collectively to provide a more reliable assessment of a pixel-based AI triage tool than a single LLM. Methods: 29,766 non-contrast CT head exams from fourteen hospitals were processed by a commercial intracranial hemorrhage (ICH) AI detection tool. Radiology reports were analyzed by an ensemble of eight open-source LLM models and a HIPAA compliant internal version of GPT-4o using a single multi-shot prompt that assessed for presence of ICH. 1,726 examples were manually reviewed. Performance characteristics of the eight open-source models and consensus were compared to GPT-4o. Three ideal consensus LLM ensembles were tested for rating the performance of the triage tool. Results: The cohort consisted of 29,766 head CTs exam-report pairs. The highest AUC performance was achieved with llama3.3:70b and GPT-4o (AUC= 0.78). The average precision was highest for Llama3.3:70b and GPT-4o (AP=0.75 & 0.76). Llama3.3:70b had the highest F1 score (0.81) and recall (0.85), greater precision (0.78), specificity (0.72), and MCC (0.57). Using MCC (95% CI) the ideal combination of LLMs were: Full-9 Ensemble 0.571 (0.552-0.591), Top-3 Ensemble 0.558 (0.537-0.579), Consensus 0.556 (0.539-0.574), and GPT4o 0.522 (0.500-0.543). No statistically significant differences were observed between Top-3, Full-9, and Consensus (p > 0.05). Conclusion: An ensemble of medium to large sized open-source LLMs provides a more consistent and reliable method to derive a ground truth retrospective evaluation of a clinical AI triage tool over a single LLM alone.
>
---
#### [new 048] The End of Manual Decoding: Towards Truly End-to-End Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型“端到端”生成中依赖人工调参的解码问题，提出AutoDeco架构。通过在Transformer中加入轻量级头，动态预测每步温度与top-p值，实现自适应解码。实验表明其性能接近最优静态策略，并能根据指令调整生成风格，推动可调控的真正端到端生成。**

- **链接: [http://arxiv.org/pdf/2510.26697v1](http://arxiv.org/pdf/2510.26697v1)**

> **作者:** Zhichao Wang; Dongyang Ma; Xinting Huang; Deng Cai; Tian Lan; Jiahao Xu; Haitao Mi; Xiaoying Tang; Yan Wang
>
> **摘要:** The "end-to-end" label for LLMs is a misnomer. In practice, they depend on a non-differentiable decoding process that requires laborious, hand-tuning of hyperparameters like temperature and top-p. This paper introduces AutoDeco, a novel architecture that enables truly "end-to-end" generation by learning to control its own decoding strategy. We augment the standard transformer with lightweight heads that, at each step, dynamically predict context-specific temperature and top-p values alongside the next-token logits. This approach transforms decoding into a parametric, token-level process, allowing the model to self-regulate its sampling strategy within a single forward pass. Through extensive experiments on eight benchmarks, we demonstrate that AutoDeco not only significantly outperforms default decoding strategies but also achieves performance comparable to an oracle-tuned baseline derived from "hacking the test set"-a practical upper bound for any static method. Crucially, we uncover an emergent capability for instruction-based decoding control: the model learns to interpret natural language commands (e.g., "generate with low randomness") and adjusts its predicted temperature and top-p on a token-by-token basis, opening a new paradigm for steerable and interactive LLM decoding.
>
---
#### [new 049] Gistify! Codebase-Level Understanding via Runtime Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Gistify任务，旨在评估编码大模型对大型代码库的理解能力。要求模型生成一个最小、自包含的文件，复现代码库特定入口点的输出。该任务需模型理解代码结构与执行流程，解决现有模型在长执行路径下表现不佳的问题。**

- **链接: [http://arxiv.org/pdf/2510.26790v1](http://arxiv.org/pdf/2510.26790v1)**

> **作者:** Hyunji Lee; Minseon Kim; Chinmay Singh; Matheus Pereira; Atharv Sonwane; Isadora White; Elias Stengel-Eskin; Mohit Bansal; Zhengyan Shi; Alessandro Sordoni; Marc-Alexandre Côté; Xingdi Yuan; Lucas Caccia
>
> **摘要:** As coding agents are increasingly deployed in large codebases, the need to automatically design challenging, codebase-level evaluation is central. We propose Gistify, a task where a coding LLM must create a single, minimal, self-contained file that can reproduce a specific functionality of a codebase. The coding LLM is given full access to a codebase along with a specific entrypoint (e.g., a python command), and the generated file must replicate the output of the same command ran under the full codebase, while containing only the essential components necessary to execute the provided command. Success on Gistify requires both structural understanding of the codebase, accurate modeling of its execution flow as well as the ability to produce potentially large code patches. Our findings show that current state-of-the-art models struggle to reliably solve Gistify tasks, especially ones with long executions traces.
>
---
#### [new 050] Value Drifts: Tracing Value Alignment During LLM Post-Training
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文研究大模型后训练中价值观对齐的动态过程，旨在揭示价值对齐何时及如何形成。通过分析SFT与偏好优化阶段的影响，发现SFT奠定模型价值观，偏好优化通常不改变已有价值；且不同算法导致不同对齐结果。工作为提升模型对齐提供了数据与算法选择依据。**

- **链接: [http://arxiv.org/pdf/2510.26707v1](http://arxiv.org/pdf/2510.26707v1)**

> **作者:** Mehar Bhatia; Shravan Nayak; Gaurav Kamath; Marius Mosbach; Karolina Stańczak; Vered Shwartz; Siva Reddy
>
> **摘要:** As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values.
>
---
#### [new 051] OmniEduBench: A Comprehensive Chinese Benchmark for Evaluating Large Language Models in Education
- **分类: cs.CL**

- **简介: 该论文提出OmniEduBench，一个面向中文教育的综合性大语言模型评测基准。针对现有评估偏重知识、缺乏育人能力考量及多样性不足的问题，构建涵盖知识与培养双维度、61个学科、11类题型的24.6K高质量问答数据集，并验证主流模型在该基准上的表现，揭示其在教育应用中的显著差距。**

- **链接: [http://arxiv.org/pdf/2510.26422v1](http://arxiv.org/pdf/2510.26422v1)**

> **作者:** Min Zhang; Hao Chen; Hao Chen; Wenqi Zhang; Didi Zhu; Xin Lin; Bo Jiang; Aimin Zhou; Fei Wu; Kun Kuang
>
> **摘要:** With the rapid development of large language models (LLMs), various LLM-based works have been widely applied in educational fields. However, most existing LLMs and their benchmarks focus primarily on the knowledge dimension, largely neglecting the evaluation of cultivation capabilities that are essential for real-world educational scenarios. Additionally, current benchmarks are often limited to a single subject or question type, lacking sufficient diversity. This issue is particularly prominent within the Chinese context. To address this gap, we introduce OmniEduBench, a comprehensive Chinese educational benchmark. OmniEduBench consists of 24.602K high-quality question-answer pairs. The data is meticulously divided into two core dimensions: the knowledge dimension and the cultivation dimension, which contain 18.121K and 6.481K entries, respectively. Each dimension is further subdivided into 6 fine-grained categories, covering a total of 61 different subjects (41 in the knowledge and 20 in the cultivation). Furthermore, the dataset features a rich variety of question formats, including 11 common exam question types, providing a solid foundation for comprehensively evaluating LLMs' capabilities in education. Extensive experiments on 11 mainstream open-source and closed-source LLMs reveal a clear performance gap. In the knowledge dimension, only Gemini-2.5 Pro surpassed 60\% accuracy, while in the cultivation dimension, the best-performing model, QWQ, still trailed human intelligence by nearly 30\%. These results highlight the substantial room for improvement and underscore the challenges of applying LLMs in education.
>
---
#### [new 052] 1+1>2: A Synergistic Sparse and Low-Rank Compression Method for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型压缩难题，提出协同稀疏与低秩压缩方法SSLC。通过统一优化框架，联合利用稀疏性和低秩性，在不增加训练成本下显著压缩模型（如Qwen2.5压缩50%无性能损失），实现高效部署。**

- **链接: [http://arxiv.org/pdf/2510.26446v1](http://arxiv.org/pdf/2510.26446v1)**

> **作者:** Zeliang Zong; Kai Zhang; Zheyang Li; Wenming Tan; Ye Ren; Yiyan Zhai; Jilin Hu
>
> **备注:** 15 pages, 6 figures, EMNLP 2025 findings
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in language comprehension and generation; however, their widespread adoption is constrained by substantial bandwidth and computational demands. While pruning and low-rank approximation have each demonstrated promising performance individually, their synergy for LLMs remains underexplored. We introduce \underline{S}ynergistic \underline{S}parse and \underline{L}ow-Rank \underline{C}ompression (SSLC) methods for LLMs, which leverages the strengths of both techniques: low-rank approximation compresses the model by retaining its essential structure with minimal information loss, whereas sparse optimization eliminates non-essential weights, preserving those crucial for generalization. Based on theoretical analysis, we first formulate the low-rank approximation and sparse optimization as a unified problem and solve it by iterative optimization algorithm. Experiments on LLaMA and Qwen2.5 models (7B-70B) show that SSLC, without any additional training steps, consistently surpasses standalone methods, achieving state-of-the-arts results. Notably, SSLC compresses Qwen2.5 by 50\% with no performance drop and achieves at least 1.63$\times$ speedup, offering a practical solution for efficient LLM deployment.
>
---
#### [new 053] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出监督强化学习（SRL）框架，解决小模型在多步推理任务中因模仿过拟合或奖励稀疏导致的学习困难。通过引导模型生成逐步推理过程，并基于专家动作提供平滑的逐步奖励，提升学习效率与泛化能力，显著优于SFT和RLVR，适用于复杂推理与软件工程任务。**

- **链接: [http://arxiv.org/pdf/2510.25992v1](http://arxiv.org/pdf/2510.25992v1)**

> **作者:** Yihe Deng; I-Hung Hsu; Jun Yan; Zifeng Wang; Rujun Han; Gufeng Zhang; Yanfei Chen; Wei Wang; Tomas Pfister; Chen-Yu Lee
>
> **摘要:** Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.
>
---
#### [new 054] POWSM: A Phonetic Open Whisper-Style Speech Foundation Model
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出POWSM，首个统一框架，联合完成语音识别、音素识别、音素转文字等多任务。解决传统方法孤立处理各任务的问题，实现音频、文本与音素间的无缝转换，提升低资源场景下的通用性与效率。**

- **链接: [http://arxiv.org/pdf/2510.24992v1](http://arxiv.org/pdf/2510.24992v1)**

> **作者:** Chin-Jou Li; Kalvin Chang; Shikhar Bharadwaj; Eunjung Yeo; Kwanghee Choi; Jian Zhu; David Mortensen; Shinji Watanabe
>
> **备注:** 14 pages, under review
>
> **摘要:** Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.
>
---
#### [new 055] The Structure of Relation Decoding Linear Operators in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型中关系解码线性算子的结构，旨在揭示其背后的语义机制。通过扩展单关系分析至多关系集合，发现这些算子可被低秩张量网络高效压缩。跨评估表明，它们实际提取的是共性的粗粒度语义属性而非特定关系，解释了其可压缩性与泛化局限性。结论支持关系解码以属性为中心的机制。**

- **链接: [http://arxiv.org/pdf/2510.26543v1](http://arxiv.org/pdf/2510.26543v1)**

> **作者:** Miranda Anna Christ; Adrián Csiszárik; Gergely Becsó; Dániel Varga
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** This paper investigates the structure of linear operators introduced in Hernandez et al. [2023] that decode specific relational facts in transformer language models. We extend their single-relation findings to a collection of relations and systematically chart their organization. We show that such collections of relation decoders can be highly compressed by simple order-3 tensor networks without significant loss in decoding accuracy. To explain this surprising redundancy, we develop a cross-evaluation protocol, in which we apply each linear decoder operator to the subjects of every other relation. Our results reveal that these linear maps do not encode distinct relations, but extract recurring, coarse-grained semantic properties (e.g., country of capital city and country of food are both in the country-of-X property). This property-centric structure clarifies both the operators' compressibility and highlights why they generalize only to new relations that are semantically close. Our findings thus interpret linear relational decoding in transformer language models as primarily property-based, rather than relation-specific.
>
---
#### [new 056] AMO-Bench: Large Language Models Still Struggle in High School Math Competitions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AMO-Bench，一个面向高阶数学推理的基准测试，专为评估大语言模型在奥数级别难题上的表现。针对现有基准因性能饱和而失效的问题，构建了50道原创、专家验证的高难度题目，仅需最终答案以实现自动评分。实验显示顶尖模型准确率仅52.4%，揭示当前LLMs数学推理能力仍有巨大提升空间。**

- **链接: [http://arxiv.org/pdf/2510.26768v1](http://arxiv.org/pdf/2510.26768v1)**

> **作者:** Shengnan An; Xunliang Cai; Xuezhi Cao; Xiaoyu Li; Yehao Lin; Junlin Liu; Xinxuan Lv; Dan Ma; Xuanlin Wang; Ziwen Wang; Shuang Zhou
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** We present AMO-Bench, an Advanced Mathematical reasoning benchmark with Olympiad level or even higher difficulty, comprising 50 human-crafted problems. Existing benchmarks have widely leveraged high school math competitions for evaluating mathematical reasoning capabilities of large language models (LLMs). However, many existing math competitions are becoming less effective for assessing top-tier LLMs due to performance saturation (e.g., AIME24/25). To address this, AMO-Bench introduces more rigorous challenges by ensuring all 50 problems are (1) cross-validated by experts to meet at least the International Mathematical Olympiad (IMO) difficulty standards, and (2) entirely original problems to prevent potential performance leakages from data memorization. Moreover, each problem in AMO-Bench requires only a final answer rather than a proof, enabling automatic and robust grading for evaluation. Experimental results across 26 LLMs on AMO-Bench show that even the best-performing model achieves only 52.4% accuracy on AMO-Bench, with most LLMs scoring below 40%. Beyond these poor performances, our further analysis reveals a promising scaling trend with increasing test-time compute on AMO-Bench. These results highlight the significant room for improving the mathematical reasoning in current LLMs. We release AMO-Bench to facilitate further research into advancing the reasoning abilities of language models. https://amo-bench.github.io/
>
---
#### [new 057] Artificial Intelligence-Enabled Analysis of Radiology Reports: Epidemiology and Consequences of Incidental Thyroid Findings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分析任务，旨在解决影像报告中偶然甲状腺发现（ITFs）的流行病学与临床后果不明确的问题。研究构建并验证了基于Transformer的NLP模型，从多模态影像报告中识别ITFs，分析其流行率、特征及后续诊疗结局，发现ITFs与甲状腺癌过诊断相关，强调标准化报告与选择性随访的重要性。**

- **链接: [http://arxiv.org/pdf/2510.26032v1](http://arxiv.org/pdf/2510.26032v1)**

> **作者:** Felipe Larios; Mariana Borras-Osorio; Yuqi Wu; Ana Gabriela Claros; David Toro-Tobon; Esteban Cabezas; Ricardo Loor-Torres; Maria Mateo Chavez; Kerly Guevara Maldonado; Luis Vilatuna Andrango; Maria Lizarazo Jimenez; Ivan Mateo Alzamora; Misk Al Zahidy; Marcelo Montero; Ana Cristina Proano; Cristian Soto Jacome; Jungwei W. Fan; Oscar J. Ponce-Ponte; Megan E. Branda; Naykky Singh Ospina; Juan P. Brito
>
> **摘要:** Importance Incidental thyroid findings (ITFs) are increasingly detected on imaging performed for non-thyroid indications. Their prevalence, features, and clinical consequences remain undefined. Objective To develop, validate, and deploy a natural language processing (NLP) pipeline to identify ITFs in radiology reports and assess their prevalence, features, and clinical outcomes. Design, Setting, and Participants Retrospective cohort of adults without prior thyroid disease undergoing thyroid-capturing imaging at Mayo Clinic sites from July 1, 2017, to September 30, 2023. A transformer-based NLP pipeline identified ITFs and extracted nodule characteristics from image reports from multiple modalities and body regions. Main Outcomes and Measures Prevalence of ITFs, downstream thyroid ultrasound, biopsy, thyroidectomy, and thyroid cancer diagnosis. Logistic regression identified demographic and imaging-related factors. Results Among 115,683 patients (mean age, 56.8 [SD 17.2] years; 52.9% women), 9,077 (7.8%) had an ITF, of which 92.9% were nodules. ITFs were more likely in women, older adults, those with higher BMI, and when imaging was ordered by oncology or internal medicine. Compared with chest CT, ITFs were more likely via neck CT, PET, and nuclear medicine scans. Nodule characteristics were poorly documented, with size reported in 44% and other features in fewer than 15% (e.g. calcifications). Compared with patients without ITFs, those with ITFs had higher odds of thyroid nodule diagnosis, biopsy, thyroidectomy and thyroid cancer diagnosis. Most cancers were papillary, and larger when detected after ITFs vs no ITF. Conclusions ITFs were common and strongly associated with cascades leading to the detection of small, low-risk cancers. These findings underscore the role of ITFs in thyroid cancer overdiagnosis and the need for standardized reporting and more selective follow-up.
>
---
#### [new 058] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data
- **分类: cs.CL**

- **简介: 该论文针对长文本预训练数据质量低的问题，提出LongFilter框架，通过对比长/短上下文下的信息增益，识别并筛选蕴含长距离依赖的高质量数据。任务为长上下文语言模型的数据筛选，解决了无效数据导致训练低效的问题，显著提升模型在长文本任务上的性能。**

- **链接: [http://arxiv.org/pdf/2510.25804v1](http://arxiv.org/pdf/2510.25804v1)**

> **作者:** Haoran Deng; Yingyu Lin; Zhenghao Lin; Xiao Liu; Yizhou Sun; Yi-An Ma; Yeyun Gong
>
> **摘要:** Long-context language models unlock advanced capabilities in reasoning, code generation, and document summarization by leveraging dependencies across extended spans of text. However, a significant portion of readily available long-text data lacks meaningful long-distance dependencies; most spans can be predicted using only local context. Training on such data is inefficient, making careful data selection crucial. Therefore, we introduce LongFilter, a framework for curating training data tailored to long-context pretraining. LongFilter measures the information gain provided by extended context by contrasting model predictions under long-context versus short-context settings, thereby identifying samples where long-range dependencies are essential. Experiments with LLaMA-3-8B, extending its context length from 8K to 64K, show that LongFilter efficiently selects high-quality data and yields substantial improvements on benchmarks such as HELMET, LongBench, and RULER.
>
---
#### [new 059] On the Role of Context for Discourse Relation Classification in Scientific Writing
- **分类: cs.CL**

- **简介: 该论文研究科学写作中论述关系分类（DRC）任务，旨在利用上下文信息提升生成式AI对科学主张的支持证据识别能力。通过实验对比预训练模型与大语言模型在科学文本中的表现，发现上下文（论述结构）普遍有助于提升分类效果，并分析了不同论述关系类型对上下文的依赖程度。**

- **链接: [http://arxiv.org/pdf/2510.26354v1](http://arxiv.org/pdf/2510.26354v1)**

> **作者:** Stephen Wan; Wei Liu; Michael Strube
>
> **备注:** Accepted at Joint Sixth Workshop on Computational Approaches to Discourse, Context and Document-Level Inferences (CODI 2025) and Eighth Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC 2025)
>
> **摘要:** With the increasing use of generative Artificial Intelligence (AI) methods to support science workflows, we are interested in the use of discourse-level information to find supporting evidence for AI generated scientific claims. A first step towards this objective is to examine the task of inferring discourse structure in scientific writing. In this work, we present a preliminary investigation of pretrained language model (PLM) and Large Language Model (LLM) approaches for Discourse Relation Classification (DRC), focusing on scientific publications, an under-studied genre for this task. We examine how context can help with the DRC task, with our experiments showing that context, as defined by discourse structure, is generally helpful. We also present an analysis of which scientific discourse relation types might benefit most from context.
>
---
#### [new 060] Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视频模型作为零样本推理器的可行性，聚焦Veo-3模型在12个视觉推理维度的表现。通过构建MME-CoF基准，系统评估其链式帧（CoF）推理能力，发现模型在短时空间一致性等方面表现良好，但在长时因果推理和抽象逻辑上仍有限。结论为当前模型尚不能独立胜任复杂推理任务，但可作为辅助视觉引擎。**

- **链接: [http://arxiv.org/pdf/2510.26802v1](http://arxiv.org/pdf/2510.26802v1)**

> **作者:** Ziyu Guo; Xinyan Chen; Renrui Zhang; Ruichuan An; Yu Qi; Dongzhi Jiang; Xiangtai Li; Manyuan Zhang; Hongsheng Li; Pheng-Ann Heng
>
> **备注:** Project Page: https://video-cof.github.io
>
> **摘要:** Recent video generation models can produce high-fidelity, temporally coherent videos, indicating that they may encode substantial world knowledge. Beyond realistic synthesis, they also exhibit emerging behaviors indicative of visual perception, modeling, and manipulation. Yet, an important question still remains: Are video models ready to serve as zero-shot reasoners in challenging visual reasoning scenarios? In this work, we conduct an empirical study to comprehensively investigate this question, focusing on the leading and popular Veo-3. We evaluate its reasoning behavior across 12 dimensions, including spatial, geometric, physical, temporal, and embodied logic, systematically characterizing both its strengths and failure modes. To standardize this study, we curate the evaluation data into MME-CoF, a compact benchmark that enables in-depth and thorough assessment of Chain-of-Frame (CoF) reasoning. Our findings reveal that while current video models demonstrate promising reasoning patterns on short-horizon spatial coherence, fine-grained grounding, and locally consistent dynamics, they remain limited in long-horizon causal reasoning, strict geometric constraints, and abstract logic. Overall, they are not yet reliable as standalone zero-shot reasoners, but exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models. Project page: https://video-cof.github.io
>
---
#### [new 061] Deep sequence models tend to memorize geometrically; it is unclear why
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究序列建模中的参数化记忆机制，挑战传统共现关联的存储观，提出模型自发形成全局几何结构来编码实体间关系。通过分析Transformer推理，揭示其将复杂多步推理简化为单步几何操作，源于谱偏差而非显式设计。工作揭示了几何记忆的内在成因，并指出提升几何性的实践空间。**

- **链接: [http://arxiv.org/pdf/2510.26745v1](http://arxiv.org/pdf/2510.26745v1)**

> **作者:** Shahriar Noroozizadeh; Vaishnavh Nagarajan; Elan Rosenfeld; Sanjiv Kumar
>
> **摘要:** In sequence modeling, the parametric memory of atomic facts has been predominantly abstracted as a brute-force lookup of co-occurrences between entities. We contrast this associative view against a geometric view of how memory is stored. We begin by isolating a clean and analyzable instance of Transformer reasoning that is incompatible with memory as strictly a storage of the local co-occurrences specified during training. Instead, the model must have somehow synthesized its own geometry of atomic facts, encoding global relationships between all entities, including non-co-occurring ones. This in turn has simplified a hard reasoning task involving an $\ell$-fold composition into an easy-to-learn 1-step geometric task. From this phenomenon, we extract fundamental aspects of neural embedding geometries that are hard to explain. We argue that the rise of such a geometry, despite optimizing over mere local associations, cannot be straightforwardly attributed to typical architectural or optimizational pressures. Counterintuitively, an elegant geometry is learned even when it is not more succinct than a brute-force lookup of associations. Then, by analyzing a connection to Node2Vec, we demonstrate how the geometry stems from a spectral bias that -- in contrast to prevailing theories -- indeed arises naturally despite the lack of various pressures. This analysis also points to practitioners a visible headroom to make Transformer memory more strongly geometric. We hope the geometric view of parametric memory encourages revisiting the default intuitions that guide researchers in areas like knowledge acquisition, capacity, discovery and unlearning.
>
---
#### [new 062] Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦视觉语言模型的时间推理能力，提出AoT-PsyPhyBENCH基准，评估模型判断视频时间方向（正向/反向）的能力。针对现有模型在物理不可逆过程与因果动作理解上的薄弱表现，通过心理物理学验证的自然视频数据集，揭示其在时间连续性与因果推理上的根本缺陷，并开源数据与代码以推动研究。**

- **链接: [http://arxiv.org/pdf/2510.26241v1](http://arxiv.org/pdf/2510.26241v1)**

> **作者:** Shiho Matta; Lis Kanashiro Pereira; Peitao Han; Fei Cheng; Shigeru Kitazawa
>
> **备注:** 10 pages
>
> **摘要:** Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and, crucially, under-evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best lag far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.
>
---
#### [new 063] FakeZero: Real-Time, Privacy-Preserving Misinformation Detection for Facebook and X
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出FakeZero，一种实时、隐私保护的虚假信息检测工具。针对社交媒体中谣言快速传播问题，设计客户端浏览器扩展，在不上传数据的前提下本地运行模型，实现对Facebook和X平台内容的即时可信度标注。通过轻量化模型与优化训练策略，在低延迟下达成高精度检测。**

- **链接: [http://arxiv.org/pdf/2510.25932v1](http://arxiv.org/pdf/2510.25932v1)**

> **作者:** Soufiane Essahli; Oussama Sarsar; Imane Fouad; Anas Motii; Ahmed Bentajer
>
> **备注:** Accepted for publication in the Proceedings of the 24th IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom 2025) Privacy track, 11 pages, 8 figures
>
> **摘要:** Social platforms distribute information at unprecedented speed, which in turn accelerates the spread of misinformation and threatens public discourse. We present FakeZero, a fully client-side, cross-platform browser extension that flags unreliable posts on Facebook and X (formerly Twitter) while the user scrolls. All computation, DOM scraping, tokenisation, Transformer inference, and UI rendering run locally through the Chromium messaging API, so no personal data leaves the device.FakeZero employs a three-stage training curriculum: baseline fine-tuning and domain-adaptive training enhanced with focal loss, adversarial augmentation, and post-training quantisation. Evaluated on a dataset of 239,000 posts, the DistilBERT-Quant model (67.6 MB) reaches 97.1% macro-F1, 97.4% accuracy, and an AUROC of 0.996, with a median latency of approximately 103 ms on a commodity laptop. A memory-efficient TinyBERT-Quant variant retains 95.7% macro-F1 and 96.1% accuracy while shrinking the model to 14.7 MB and lowering latency to approximately 40 ms, showing that high-quality fake-news detection is feasible under tight resource budgets with only modest performance loss.By providing inline credibility cues, the extension can serve as a valuable tool for policymakers seeking to curb the spread of misinformation across social networks. With user consent, FakeZero also opens the door for researchers to collect large-scale datasets of fake news in the wild, enabling deeper analysis and the development of more robust detection techniques.
>
---
#### [new 064] Rethinking Text-to-SQL: Dynamic Multi-turn SQL Interaction for Real-world Database Exploration
- **分类: cs.DB; cs.CL**

- **简介: 该论文聚焦动态多轮文本转SQL任务，针对真实场景中用户意图演化问题，提出DySQL-Bench基准与多轮评估框架。通过自动化生成与专家验证构建13领域1072个交互式任务，揭示现有模型在动态交互中的局限性，推动文本转SQL向真实应用演进。**

- **链接: [http://arxiv.org/pdf/2510.26495v1](http://arxiv.org/pdf/2510.26495v1)**

> **作者:** Linzhuang Sun; Tianyu Guo; Hao Liang; Yuying Li; Qifeng Cai; Jingxuan Wei; Bihui Yu; Wentao Zhang; Bin Cui
>
> **摘要:** Recent advances in Text-to-SQL have achieved strong results in static, single-turn tasks, where models generate SQL queries from natural language questions. However, these systems fall short in real-world interactive scenarios, where user intents evolve and queries must be refined over multiple turns. In applications such as finance and business analytics, users iteratively adjust query constraints or dimensions based on intermediate results. To evaluate such dynamic capabilities, we introduce DySQL-Bench, a benchmark assessing model performance under evolving user interactions. Unlike previous manually curated datasets, DySQL-Bench is built through an automated two-stage pipeline of task synthesis and verification. Structured tree representations derived from raw database tables guide LLM-based task generation, followed by interaction-oriented filtering and expert validation. Human evaluation confirms 100% correctness of the synthesized data. We further propose a multi-turn evaluation framework simulating realistic interactions among an LLM-simulated user, the model under test, and an executable database. The model must adapt its reasoning and SQL generation as user intents change. DySQL-Bench covers 13 domains across BIRD and Spider 2 databases, totaling 1,072 tasks. Even GPT-4o attains only 58.34% overall accuracy and 23.81% on the Pass@5 metric, underscoring the benchmark's difficulty. All code and data are released at https://github.com/Aurora-slz/Real-World-SQL-Bench .
>
---
#### [new 065] Remote Labor Index: Measuring AI Automation of Remote Work
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出远程劳动指数（RLI），用于衡量AI在真实经济场景中自动化远程工作的能力。针对AI自动化落地效果不明确的问题，构建多领域实际任务基准，评估AI代理端到端执行能力。结果表明当前最高自动化率仅2.5%，为量化AI对劳动力影响提供了实证基础。**

- **链接: [http://arxiv.org/pdf/2510.26787v1](http://arxiv.org/pdf/2510.26787v1)**

> **作者:** Mantas Mazeika; Alice Gatti; Cristina Menghini; Udari Madhushani Sehwag; Shivam Singhal; Yury Orlovskiy; Steven Basart; Manasi Sharma; Denis Peskoff; Elaine Lau; Jaehyuk Lim; Lachlan Carroll; Alice Blair; Vinaya Sivakumar; Sumana Basu; Brad Kenstler; Yuntao Ma; Julian Michael; Xiaoke Li; Oliver Ingebretsen; Aditya Mehta; Jean Mottola; John Teichmann; Kevin Yu; Zaina Shaik; Adam Khoja; Richard Ren; Jason Hausenloy; Long Phan; Ye Htet; Ankit Aich; Tahseen Rabbani; Vivswan Shah; Andriy Novykov; Felix Binder; Kirill Chugunov; Luis Ramirez; Matias Geralnik; Hernán Mesura; Dean Lee; Ed-Yeremai Hernandez Cardona; Annette Diamond; Summer Yue; Alexandr Wang; Bing Liu; Ernesto Hernandez; Dan Hendrycks
>
> **备注:** Website: https://www.remotelabor.ai
>
> **摘要:** AIs have made rapid progress on research-oriented benchmarks of knowledge and reasoning, but it remains unclear how these gains translate into economic value and automation. To measure this, we introduce the Remote Labor Index (RLI), a broadly multi-sector benchmark comprising real-world, economically valuable projects designed to evaluate end-to-end agent performance in practical settings. AI agents perform near the floor on RLI, with the highest-performing agent achieving an automation rate of 2.5%. These results help ground discussions of AI automation in empirical evidence, setting a common basis for tracking AI impacts and enabling stakeholders to proactively navigate AI-driven labor automation.
>
---
#### [new 066] Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文针对大模型在主观评价任务中可靠性不足的问题，提出一种人-模型协作框架，通过拒绝采样从仅有标签的数据中推断思维轨迹。该方法用于微调开放模型与优化标注指南，显著提升大模型间及大模型与人类的一致性，使仅含标签的数据扩展为蕴含思维轨迹的增强资源。**

- **链接: [http://arxiv.org/pdf/2510.25860v1](http://arxiv.org/pdf/2510.25860v1)**

> **作者:** Xingjian Zhang; Tianhong Gao; Suliang Jin; Tianhao Wang; Teng Ye; Eytan Adar; Qiaozhu Mei
>
> **摘要:** Large language models (LLMs) are increasingly used as raters for evaluation tasks. However, their reliability is often limited for subjective tasks, when human judgments involve subtle reasoning beyond annotation labels. Thinking traces, the reasoning behind a judgment, are highly informative but challenging to collect and curate. We present a human-LLM collaborative framework to infer thinking traces from label-only annotations. The proposed framework uses a simple and effective rejection sampling method to reconstruct these traces at scale. These inferred thinking traces are applied to two complementary tasks: (1) fine-tuning open LLM raters; and (2) synthesizing clearer annotation guidelines for proprietary LLM raters. Across multiple datasets, our methods lead to significantly improved LLM-human agreement. Additionally, the refined annotation guidelines increase agreement among different LLM models. These results suggest that LLMs can serve as practical proxies for otherwise unrevealed human thinking traces, enabling label-only corpora to be extended into thinking-trace-augmented resources that enhance the reliability of LLM raters.
>
---
#### [new 067] Do Students Debias Like Teachers? On the Distillability of Bias Mitigation Methods
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究知识蒸馏（KD）对模型去偏能力的迁移影响，针对自然语言推理与图像分类任务，发现KD会削弱去偏效果。通过实验揭示其内在机制，并提出数据增强、迭代蒸馏与权重初始化三种改进方案，首次系统探讨了去偏方法在KD中的可迁移性。**

- **链接: [http://arxiv.org/pdf/2510.26038v1](http://arxiv.org/pdf/2510.26038v1)**

> **作者:** Jiali Cheng; Chirag Agarwal; Hadi Amiri
>
> **摘要:** Knowledge distillation (KD) is an effective method for model compression and transferring knowledge between models. However, its effect on model's robustness against spurious correlations that degrade performance on out-of-distribution data remains underexplored. This study investigates the effect of knowledge distillation on the transferability of ``debiasing'' capabilities from teacher models to student models on natural language inference (NLI) and image classification tasks. Through extensive experiments, we illustrate several key findings: (i) overall the debiasing capability of a model is undermined post-KD; (ii) training a debiased model does not benefit from injecting teacher knowledge; (iii) although the overall robustness of a model may remain stable post-distillation, significant variations can occur across different types of biases; and (iv) we pin-point the internal attention pattern and circuit that causes the distinct behavior post-KD. Given the above findings, we propose three effective solutions to improve the distillability of debiasing methods: developing high quality data for augmentation, implementing iterative knowledge distillation, and initializing student models with weights obtained from teacher models. To the best of our knowledge, this is the first study on the effect of KD on debiasing and its interenal mechanism at scale. Our findings provide understandings on how KD works and how to design better debiasing methods.
>
---
#### [new 068] The Era of Agentic Organization: Learning to Organize with Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出异步思考（AsyncThink）框架，用于实现智能体协作推理。针对大语言模型在复杂任务中推理效率低的问题，设计动态分配、并行执行与知识融合的思考协议，并通过强化学习优化结构。实验表明，该方法显著降低延迟、提升准确率，并具备零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.26658v1](http://arxiv.org/pdf/2510.26658v1)**

> **作者:** Zewen Chi; Li Dong; Qingxiu Dong; Yaru Hao; Xun Wu; Shaohan Huang; Furu Wei
>
> **摘要:** We envision a new era of AI, termed agentic organization, where agents solve complex problems by working collaboratively and concurrently, enabling outcomes beyond individual intelligence. To realize this vision, we introduce asynchronous thinking (AsyncThink) as a new paradigm of reasoning with large language models, which organizes the internal thinking process into concurrently executable structures. Specifically, we propose a thinking protocol where an organizer dynamically assigns sub-queries to workers, merges intermediate knowledge, and produces coherent solutions. More importantly, the thinking structure in this protocol can be further optimized through reinforcement learning. Experiments demonstrate that AsyncThink achieves 28% lower inference latency compared to parallel thinking while improving accuracy on mathematical reasoning. Moreover, AsyncThink generalizes its learned asynchronous thinking capabilities, effectively tackling unseen tasks without additional training.
>
---
#### [new 069] Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出“推理课程”（Reasoning Curriculum）框架，旨在提升大语言模型的通用推理能力。针对现有强化学习方法多局限于数学与代码的问题，论文设计两阶段课程：先在数学领域通过可验证奖励训练推理技能，再跨领域联合训练以迁移巩固。方法简洁、无需专用奖励模型，实证显示显著提升多领域推理表现。**

- **链接: [http://arxiv.org/pdf/2510.26143v1](http://arxiv.org/pdf/2510.26143v1)**

> **作者:** Bo Pang; Deqian Kong; Silvio Savarese; Caiming Xiong; Yingbo Zhou
>
> **备注:** 9 pages
>
> **摘要:** Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning.
>
---
#### [new 070] SP-MCQA: Evaluating Intelligibility of TTS Beyond the Word Level
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文针对语音合成（TTS）智能评估瓶颈，提出SP-MCQA新任务，通过关键信息多选题评估合成语音的语义理解能力。解决传统词级准确率（WER）无法反映真实听觉理解的问题。构建8.76小时新闻数据集，揭示高WER下关键信息丢失现象，强调需发展更贴近人类认知的评估标准。**

- **链接: [http://arxiv.org/pdf/2510.26190v1](http://arxiv.org/pdf/2510.26190v1)**

> **作者:** Hitomi Jin Ling Tee; Chaoren Wang; Zijie Zhang; Zhizheng Wu
>
> **摘要:** The evaluation of intelligibility for TTS has reached a bottleneck, as existing assessments heavily rely on word-by-word accuracy metrics such as WER, which fail to capture the complexity of real-world speech or reflect human comprehension needs. To address this, we propose Spoken-Passage Multiple-Choice Question Answering, a novel subjective approach evaluating the accuracy of key information in synthesized speech, and release SP-MCQA-Eval, an 8.76-hour news-style benchmark dataset for SP-MCQA evaluation. Our experiments reveal that low WER does not necessarily guarantee high key-information accuracy, exposing a gap between traditional metrics and practical intelligibility. SP-MCQA shows that even state-of-the-art (SOTA) models still lack robust text normalization and phonetic accuracy. This work underscores the urgent need for high-level, more life-like evaluation criteria now that many systems already excel at WER yet may fall short on real-world intelligibility.
>
---
#### [new 071] Counteracting Matthew Effect in Self-Improvement of LVLMs through Head-Tail Re-balancing
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦于大视觉语言模型（LVLMs）的自提升任务，针对自提升过程中模型过度偏向简单问题（头数据）而忽视复杂问题（尾数据）的“马太效应”问题，提出通过分布重塑与轨迹重采样两种策略实现头尾平衡，显著提升模型在复杂视觉推理任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.26474v1](http://arxiv.org/pdf/2510.26474v1)**

> **作者:** Xin Guo; Zhiheng Xi; Yiwen Ding; Yitao Zhai; Xiaowei Shi; Xunliang Cai; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Preprint
>
> **摘要:** Self-improvement has emerged as a mainstream paradigm for advancing the reasoning capabilities of large vision-language models (LVLMs), where models explore and learn from successful trajectories iteratively. However, we identify a critical issue during this process: the model excels at generating high-quality trajectories for simple queries (i.e., head data) but struggles with more complex ones (i.e., tail data). This leads to an imbalanced optimization that drives the model to prioritize simple reasoning skills, while hindering its ability to tackle more complex reasoning tasks. Over iterations, this imbalance becomes increasingly pronounced--a dynamic we term the "Matthew effect"--which ultimately hinders further model improvement and leads to performance bottlenecks. To counteract this challenge, we introduce four efficient strategies from two perspectives: distribution-reshaping and trajectory-resampling, to achieve head-tail re-balancing during the exploration-and-learning self-improvement process. Extensive experiments on Qwen2-VL-7B-Instruct and InternVL2.5-4B models across visual reasoning tasks demonstrate that our methods consistently improve visual reasoning capabilities, outperforming vanilla self-improvement by 3.86 points on average.
>
---
#### [new 072] One Model to Critique Them All: Rewarding Agentic Tool-Use via Efficient Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对工具调用场景中缺乏专用奖励模型的问题，提出ToolRM系列轻量级生成式奖励模型。通过规则评分与多维采样构建偏好数据集ToolPref-Pairwise-30K，设计TRBench$_{BFCL}$评估基准。实验表明，其在准确率上显著优于前沿模型，且支持自纠错与最佳采样，提升推理效率并减少66%以上输出令牌。**

- **链接: [http://arxiv.org/pdf/2510.26167v1](http://arxiv.org/pdf/2510.26167v1)**

> **作者:** Renhao Li; Jianhong Tu; Yang Su; Hamid Alinejad-Rokny; Derek F. Wong; Junyang Lin; Min Yang
>
> **摘要:** Reward models (RMs) play a critical role in aligning large language models (LLMs) with human preferences. Yet in the domain of tool learning, the lack of RMs specifically designed for function-calling tasks has limited progress toward more capable agentic AI. We introduce ToolRM, a family of lightweight generative RMs tailored for general tool-use scenarios. To build these models, we propose a novel pipeline that constructs pairwise preference data using rule-based scoring and multidimensional sampling. This yields ToolPref-Pairwise-30K, a diverse, balanced, and challenging dataset of critique tasks that supports reinforcement learning with verifiable feedback. To evaluate tool-use RMs, we also introduce TRBench$_{BFCL}$, a benchmark built on the agentic evaluation suite BFCL. Trained on our constructed data, models from the Qwen3-4B/8B series achieve up to 14.28% higher accuracy, substantially outperforming frontier models such as Claude 4 and OpenAI o3 in pairwise reward judgments. Beyond training objectives, ToolRM generalizes to broader critique tasks, including Best-of-N sampling and self-correction. Experiments on ACEBench highlight its effectiveness and efficiency, enabling inference-time scaling and reducing output token usage by over 66%. We release data and model checkpoints to facilitate future research.
>
---
#### [new 073] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对多模态大模型冷启动阶段的泛化能力不足问题，提出SPECS框架。通过自蒸馏生成偏好数据，采用基于偏好训练的冷启动策略，解耦任务推理与输出格式，提升泛化性。实验表明，该方法显著优于基线，在多个基准上性能提升显著。**

- **链接: [http://arxiv.org/pdf/2510.25801v1](http://arxiv.org/pdf/2510.25801v1)**

> **作者:** Kun Chen; Peng Shi; Haibo Qiu; Zhixiong Zeng; Siqi Yang; Wenji Mao; Lin Ma
>
> **备注:** Project Page: https://github.com/Kwen-Chen/SPECS-VL
>
> **摘要:** Reinforcement learning (RL) with verifiable rewards has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Most representative paradigms begin with a cold start, typically employing supervised fine-tuning (SFT), to initialize the policy before RL. However, SFT-based cold start adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution generalization, and ultimately affects downstream RL. We revisit the cold start along two views, its training method and data construction, and introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods. Our empirical study finds that preference-based training methods (e.g. DPO) generalizes better than SFT-based methods in cold start. Motivated by this, we propose SPECS-a Self-distilled, Preference-based Cold Start framework that decouples multimodal learning: (1) generates introspective preference data pairs via self-distillation, avoiding reliance on larger teachers or manual annotation; (2) performs preference-based training to learn, focusing on shallow, transferable surface-form criteria (format, structure, style) rather than memorizing content; and (3) hands off to RL with verifiable rewards for deep reasoning results. Experimental results across multiple multimodal benchmarks show that our decoupling learning framework yields consistent performance gains over strong baselines, improving MEGA-Bench by 4.1% and MathVista by 12.2%. Additional experiments indicate that SPECS contributes to reducing in-distribution "stuckness," improving exploration, stabilizing training, and raising the performance ceiling.
>
---
#### [new 074] ORBIT - Open Recommendation Benchmark for Reproducible Research with Hidden Tests
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出ORBIT基准，解决推荐系统评估中数据不真实、设置不一致的问题。构建公开可复现的评估框架与隐藏测试集ClueWeb-Reco，用于更真实地评测模型泛化能力，验证了现有方法局限并展示了大语言模型的潜力。**

- **链接: [http://arxiv.org/pdf/2510.26095v1](http://arxiv.org/pdf/2510.26095v1)**

> **作者:** Jingyuan He; Jiongnan Liu; Vishan Vishesh Oberoi; Bolin Wu; Mahima Jagadeesh Patel; Kangrui Mao; Chuning Shi; I-Ta Lee; Arnold Overwijk; Chenyan Xiong
>
> **备注:** Accepted to NeurIPS 2025 Datasets & Benchmarks track
>
> **摘要:** Recommender systems are among the most impactful AI applications, interacting with billions of users every day, guiding them to relevant products, services, or information tailored to their preferences. However, the research and development of recommender systems are hindered by existing datasets that fail to capture realistic user behaviors and inconsistent evaluation settings that lead to ambiguous conclusions. This paper introduces the Open Recommendation Benchmark for Reproducible Research with HIdden Tests (ORBIT), a unified benchmark for consistent and realistic evaluation of recommendation models. ORBIT offers a standardized evaluation framework of public datasets with reproducible splits and transparent settings for its public leaderboard. Additionally, ORBIT introduces a new webpage recommendation task, ClueWeb-Reco, featuring web browsing sequences from 87 million public, high-quality webpages. ClueWeb-Reco is a synthetic dataset derived from real, user-consented, and privacy-guaranteed browsing data. It aligns with modern recommendation scenarios and is reserved as the hidden test part of our leaderboard to challenge recommendation models' generalization ability. ORBIT measures 12 representative recommendation models on its public benchmark and introduces a prompted LLM baseline on the ClueWeb-Reco hidden test. Our benchmark results reflect general improvements of recommender systems on the public datasets, with variable individual performances. The results on the hidden test reveal the limitations of existing approaches in large-scale webpage recommendation and highlight the potential for improvements with LLM integrations. ORBIT benchmark, leaderboard, and codebase are available at https://www.open-reco-bench.ai.
>
---
#### [new 075] CAVE: Detecting and Explaining Commonsense Anomalies in Visual Environments
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CAVE基准，针对真实世界视觉异常检测与理解任务。解决现有方法局限于工业缺陷或合成异常的问题，通过引入人类认知启发的细粒度标注，支持异常描述、解释与论证，评估视觉语言模型在常识推理与异常感知上的能力。**

- **链接: [http://arxiv.org/pdf/2510.26006v1](http://arxiv.org/pdf/2510.26006v1)**

> **作者:** Rishika Bhagwatkar; Syrielle Montariol; Angelika Romanou; Beatriz Borges; Irina Rish; Antoine Bosselut
>
> **摘要:** Humans can naturally identify, reason about, and explain anomalies in their environment. In computer vision, this long-standing challenge remains limited to industrial defects or unrealistic, synthetically generated anomalies, failing to capture the richness and unpredictability of real-world anomalies. In this work, we introduce CAVE, the first benchmark of real-world visual anomalies. CAVE supports three open-ended tasks: anomaly description, explanation, and justification; with fine-grained annotations for visual grounding and categorizing anomalies based on their visual manifestations, their complexity, severity, and commonness. These annotations draw inspiration from cognitive science research on how humans identify and resolve anomalies, providing a comprehensive framework for evaluating Vision-Language Models (VLMs) in detecting and understanding anomalies. We show that state-of-the-art VLMs struggle with visual anomaly perception and commonsense reasoning, even with advanced prompting strategies. By offering a realistic and cognitively grounded benchmark, CAVE serves as a valuable resource for advancing research in anomaly detection and commonsense reasoning in VLMs.
>
---
#### [new 076] SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出SIRAJ框架，用于高效多样地对大模型代理进行红队测试。针对模型规划与工具调用带来的安全风险，通过动态两步法生成覆盖广泛风险的测试用例，并利用知识蒸馏训练小型高效红队模型，显著提升漏洞发现效率与攻击成功率。**

- **链接: [http://arxiv.org/pdf/2510.26037v1](http://arxiv.org/pdf/2510.26037v1)**

> **作者:** Kaiwen Zhou; Ahmed Elgohary; A S M Iftekhar; Amin Saied
>
> **摘要:** The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models.
>
---
#### [new 077] MemEIC: A Step Toward Continual and Compositional Knowledge Editing
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对视觉语言模型知识更新中单一模态编辑、缺乏连续与组合编辑的问题，提出MemEIC方法。通过双外部记忆与双LoRA适配器实现跨模态知识检索与解耦更新，并引入脑启发连接器支持组合推理，实现持续、组合式多模态知识编辑，显著提升复杂问答性能并保留历史编辑结果。**

- **链接: [http://arxiv.org/pdf/2510.25798v1](http://arxiv.org/pdf/2510.25798v1)**

> **作者:** Jin Seong; Jiyun Park; Wencke Liermann; Hongseok Choi; Yoonji Nam; Hyun Kim; Soojong Lim; Namhoon Lee
>
> **备注:** NeurIPS 2025, 38 pages, 8 figures
>
> **摘要:** The dynamic nature of information necessitates continuously updating large vision-language models (LVLMs). While recent knowledge editing techniques hint at promising directions, they often focus on editing a single modality (vision or language) in isolation. This prevalent practice neglects the inherent multimodality of LVLMs and the continuous nature of knowledge updates, potentially leading to suboptimal editing outcomes when considering the interplay between modalities and the need for ongoing knowledge refinement. To address these limitations, we propose MemEIC, a novel method for Continual and Compositional Knowledge Editing (CCKE) in LVLMs. MemEIC enables compositional editing of both visual and textual knowledge sequentially. Our approach employs a hybrid external-internal editor featuring a dual external memory for cross-modal evidence retrieval and dual LoRA adapters that facilitate disentangled parameter updates for each modality. A key component is a brain-inspired knowledge connector, activated selectively for compositional reasoning, that integrates information across different modalities. Experiments demonstrate that MemEIC significantly improves performance on complex multimodal questions and effectively preserves prior edits, setting a new benchmark for CCKE in LVLMs.
>
---
#### [new 078] Cross-Platform Evaluation of Reasoning Capabilities in Foundation Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于基础模型推理能力的跨平台评估，旨在解决模型性能受硬件平台影响的问题。通过在超算、云平台和高校集群上测试15个模型在8大学科79个问题上的表现，验证了方法的可复现性，发现训练数据质量比模型规模更重要，为模型选型提供依据。**

- **链接: [http://arxiv.org/pdf/2510.26732v1](http://arxiv.org/pdf/2510.26732v1)**

> **作者:** J. de Curtò; I. de Zarzà; Pablo García; Jordi Cabot
>
> **摘要:** This paper presents a comprehensive cross-platform evaluation of reasoning capabilities in contemporary foundation models, establishing an infrastructure-agnostic benchmark across three computational paradigms: HPC supercomputing (MareNostrum 5), cloud platforms (Nebius AI Studio), and university clusters (a node with eight H200 GPUs). We evaluate 15 foundation models across 79 problems spanning eight academic domains (Physics, Mathematics, Chemistry, Economics, Biology, Statistics, Calculus, and Optimization) through three experimental phases: (1) Baseline establishment: Six models (Mixtral-8x7B, Phi-3, LLaMA 3.1-8B, Gemma-2-9b, Mistral-7B, OLMo-7B) evaluated on 19 problems using MareNostrum 5, establishing methodology and reference performance; (2) Infrastructure validation: The 19-problem benchmark repeated on university cluster (seven models including Falcon-Mamba state-space architecture) and Nebius AI Studio (nine state-of-the-art models: Hermes-4 70B/405B, LLaMA 3.1-405B/3.3-70B, Qwen3 30B/235B, DeepSeek-R1, GPT-OSS 20B/120B) to confirm infrastructure-agnostic reproducibility; (3) Extended evaluation: Full 79-problem assessment on both university cluster and Nebius platforms, probing generalization at scale across architectural diversity. The findings challenge conventional scaling assumptions, establish training data quality as more critical than model size, and provide actionable guidelines for model selection across educational, production, and research contexts. The tri-infrastructure methodology and 79-problem benchmark enable longitudinal tracking of reasoning capabilities as foundation models evolve.
>
---
#### [new 079] Defeating the Training-Inference Mismatch via FP16
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型强化学习微调中的训练-推理不匹配问题，指出其根源在于BF16精度导致的数值误差。研究发现，改用FP16可有效消除此不匹配，实现更稳定优化、更快收敛与更好性能。工作简单，仅需少量代码修改，无需改动模型或算法。**

- **链接: [http://arxiv.org/pdf/2510.26788v1](http://arxiv.org/pdf/2510.26788v1)**

> **作者:** Penghui Qi; Zichen Liu; Xiangxin Zhou; Tianyu Pang; Chao Du; Wee Sun Lee; Min Lin
>
> **摘要:** Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to \textbf{FP16} effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning.
>
---
#### [new 080] Nexus: Execution-Grounded Multi-Agent Test Oracle Synthesis
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出Nexus框架，解决非回归测试中测试预言生成难题。通过多智能体协作，结合推理、执行验证与自迭代修正，提升预言准确性。实验表明其显著优于现有方法，在多个基准上大幅提高测试准确率与下游任务效果。**

- **链接: [http://arxiv.org/pdf/2510.26423v1](http://arxiv.org/pdf/2510.26423v1)**

> **作者:** Dong Huang; Mingzhe Du; Jie M. Zhang; Zheng Lin; Meng Luo; Qianru Zhang; See-Kiong Ng
>
> **备注:** Under Review
>
> **摘要:** Test oracle generation in non-regression testing is a longstanding challenge in software engineering, where the goal is to produce oracles that can accurately determine whether a function under test (FUT) behaves as intended for a given input. In this paper, we introduce Nexus, a novel multi-agent framework to address this challenge. Nexus generates test oracles by leveraging a diverse set of specialized agents that synthesize test oracles through a structured process of deliberation, validation, and iterative self-refinement. During the deliberation phase, a panel of four specialist agents, each embodying a distinct testing philosophy, collaboratively critiques and refines an initial set of test oracles. Then, in the validation phase, Nexus generates a plausible candidate implementation of the FUT and executes the proposed oracles against it in a secure sandbox. For any oracle that fails this execution-based check, Nexus activates an automated selfrefinement loop, using the specific runtime error to debug and correct the oracle before re-validation. Our extensive evaluation on seven diverse benchmarks demonstrates that Nexus consistently and substantially outperforms state-of-theart baselines. For instance, Nexus improves the test-level oracle accuracy on the LiveCodeBench from 46.30% to 57.73% for GPT-4.1-Mini. The improved accuracy also significantly enhances downstream tasks: the bug detection rate of GPT4.1-Mini generated test oracles on HumanEval increases from 90.91% to 95.45% for Nexus compared to baselines, and the success rate of automated program repair improves from 35.23% to 69.32%.
>
---
#### [new 081] PVMark: Enabling Public Verifiability for LLM Watermarking Schemes
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文针对大模型文本水印的可信验证问题，提出PVMark框架，利用零知识证明实现水印检测的公开可验证性。解决了私钥泄露风险与检测不透明的矛盾，通过构建正确执行证明，使第三方能验证水印真实性而不暴露密钥，支持多种水印方案与协议，兼具高效性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.26274v1](http://arxiv.org/pdf/2510.26274v1)**

> **作者:** Haohua Duan; Liyao Xiang; Xin Zhang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.
>
---
#### [new 082] SecureReviewer: Enhancing Large Language Models for Secure Code Review through Secure-aware Fine-tuning
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文针对代码审查中安全漏洞识别不足的问题，提出SecureReviewer框架。通过构建专用数据集与安全感知微调策略，增强大模型识别安全问题并提供修复建议的能力；结合RAG技术减少幻觉，提升可靠性；引入SecureBLEU评估指标，实验表明其在安全检测与评论质量上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.26457v1](http://arxiv.org/pdf/2510.26457v1)**

> **作者:** Fang Liu; Simiao Liu; Yinghao Zhu; Xiaoli Lian; Li Zhang
>
> **备注:** Accepted by ICSE 2026. Code and data: https://github.com/SIMIAO515/SecureReviewer
>
> **摘要:** Identifying and addressing security issues during the early phase of the development lifecycle is critical for mitigating the long-term negative impacts on software systems. Code review serves as an effective practice that enables developers to check their teammates' code before integration into the codebase. To streamline the generation of review comments, various automated code review approaches have been proposed, where LLM-based methods have significantly advanced the capabilities of automated review generation. However, existing models primarily focus on general-purpose code review, their effectiveness in identifying and addressing security-related issues remains underexplored. Moreover, adapting existing code review approaches to target security issues faces substantial challenges, including data scarcity and inadequate evaluation metrics. To address these limitations, we propose SecureReviewer, a new approach designed for enhancing LLMs' ability to identify and resolve security-related issues during code review. Specifically, we first construct a dataset tailored for training and evaluating secure code review capabilities. Leveraging this dataset, we fine-tune LLMs to generate code review comments that can effectively identify security issues and provide fix suggestions with our proposed secure-aware fine-tuning strategy. To mitigate hallucination in LLMs and enhance the reliability of their outputs, we integrate the RAG technique, which grounds the generated comments in domain-specific security knowledge. Additionally, we introduce SecureBLEU, a new evaluation metric designed to assess the effectiveness of review comments in addressing security issues. Experimental results demonstrate that SecureReviewer outperforms state-of-the-art baselines in both security issue detection accuracy and the overall quality and practical utility of generated review comments.
>
---
#### [new 083] Approximating Human Preferences Using a Multi-Judge Learned System
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大模型评判系统与人类偏好对齐难题，提出基于多评委的偏好聚合框架，通过学习多个条件化裁判的输出来建模多样化人格化偏好。解决了评判系统易受评分标准、偏见和不稳定影响的问题，实现了可扩展的偏好标签合成，并采用GAM与MLP两种方法验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.25884v1](http://arxiv.org/pdf/2510.25884v1)**

> **作者:** Eitán Sprejer; Fernando Avalos; Augusto Bernardi; Jose Pedro Brito de Azevedo Faustino; Jacob Haimes; Narmeen Fatimah Oozeer
>
> **摘要:** Aligning LLM-based judges with human preferences is a significant challenge, as they are difficult to calibrate and often suffer from rubric sensitivity, bias, and instability. Overcoming this challenge advances key applications, such as creating reliable reward models for Reinforcement Learning from Human Feedback (RLHF) and building effective routing systems that select the best-suited model for a given user query. In this work, we propose a framework for modeling diverse, persona-based preferences by learning to aggregate outputs from multiple rubric-conditioned judges. We investigate the performance of this approach against naive baselines and assess its robustness through case studies on both human and LLM-judges biases. Our primary contributions include a persona-based method for synthesizing preference labels at scale and two distinct implementations of our aggregator: Generalized Additive Model (GAM) and a Multi-Layer Perceptron (MLP).
>
---
#### [new 084] Context Engineering 2.0: The Context of Context Engineering
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨“情境工程2.0”，旨在系统化定义情境工程，梳理其发展历程与核心设计原则。针对机器理解人类情境与目的的难题，提出历史演进框架，为智能系统中的情境建模提供理论基础，推动AI系统实现更深层次的人机协同。**

- **链接: [http://arxiv.org/pdf/2510.26493v1](http://arxiv.org/pdf/2510.26493v1)**

> **作者:** Qishuo Hua; Lyumanshan Ye; Dayuan Fu; Yang Xiao; Xiaojie Cai; Yunze Wu; Jifan Lin; Junfei Wang; Pengfei Liu
>
> **摘要:** Karl Marx once wrote that ``the human essence is the ensemble of social relations'', suggesting that individuals are not isolated entities but are fundamentally shaped by their interactions with other entities, within which contexts play a constitutive and essential role. With the advent of computers and artificial intelligence, these contexts are no longer limited to purely human--human interactions: human--machine interactions are included as well. Then a central question emerges: How can machines better understand our situations and purposes? To address this challenge, researchers have recently introduced the concept of context engineering. Although it is often regarded as a recent innovation of the agent era, we argue that related practices can be traced back more than twenty years. Since the early 1990s, the field has evolved through distinct historical phases, each shaped by the intelligence level of machines: from early human--computer interaction frameworks built around primitive computers, to today's human--agent interaction paradigms driven by intelligent agents, and potentially to human--level or superhuman intelligence in the future. In this paper, we situate context engineering, provide a systematic definition, outline its historical and conceptual landscape, and examine key design considerations for practice. By addressing these questions, we aim to offer a conceptual foundation for context engineering and sketch its promising future. This paper is a stepping stone for a broader community effort toward systematic context engineering in AI systems.
>
---
#### [new 085] Enhancing Underwater Object Detection through Spatio-Temporal Analysis and Spatial Attention Networks
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文针对水下目标检测任务，解决动态环境中物体识别精度低的问题。通过引入时空建模与空间注意力机制，改进YOLOv5，提出T-YOLOv5及融合CBAM的版本，显著提升复杂场景下的检测准确率，尤其在运动、遮挡条件下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.25797v1](http://arxiv.org/pdf/2510.25797v1)**

> **作者:** Sai Likhith Karri; Ansh Saxena
>
> **摘要:** This study examines the effectiveness of spatio-temporal modeling and the integration of spatial attention mechanisms in deep learning models for underwater object detection. Specifically, in the first phase, the performance of temporal-enhanced YOLOv5 variant T-YOLOv5 is evaluated, in comparison with the standard YOLOv5. For the second phase, an augmented version of T-YOLOv5 is developed, through the addition of a Convolutional Block Attention Module (CBAM). By examining the effectiveness of the already pre-existing YOLOv5 and T-YOLOv5 models and of the newly developed T-YOLOv5 with CBAM. With CBAM, the research highlights how temporal modeling improves detection accuracy in dynamic marine environments, particularly under conditions of sudden movements, partial occlusions, and gradual motion. The testing results showed that YOLOv5 achieved a mAP@50-95 of 0.563, while T-YOLOv5 and T-YOLOv5 with CBAM outperformed with mAP@50-95 scores of 0.813 and 0.811, respectively, highlighting their superior accuracy and generalization in detecting complex objects. The findings demonstrate that T-YOLOv5 significantly enhances detection reliability compared to the standard model, while T-YOLOv5 with CBAM further improves performance in challenging scenarios, although there is a loss of accuracy when it comes to simpler scenarios.
>
---
#### [new 086] Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于规范推理任务，旨在评估大语言模型在规范性与认知性模态推理上的表现。通过构建新数据集，对比分析模型在逻辑与模态层面的推理能力，发现其在规范推理中存在不一致与认知偏差，揭示了提升模型逻辑一致性与可靠性的挑战。**

- **链接: [http://arxiv.org/pdf/2510.26606v1](http://arxiv.org/pdf/2510.26606v1)**

> **作者:** Kentaro Ozeki; Risako Ando; Takanobu Morishita; Hirohiko Abe; Koji Mineshima; Mitsuhiro Okada
>
> **备注:** Accepted to the 8th BlackboxNLP Workshop at EMNLP 2025
>
> **摘要:** Normative reasoning is a type of reasoning that involves normative or deontic modality, such as obligation and permission. While large language models (LLMs) have demonstrated remarkable performance across various reasoning tasks, their ability to handle normative reasoning remains underexplored. In this paper, we systematically evaluate LLMs' reasoning capabilities in the normative domain from both logical and modal perspectives. Specifically, to assess how well LLMs reason with normative modals, we make a comparison between their reasoning with normative modals and their reasoning with epistemic modals, which share a common formal structure. To this end, we introduce a new dataset covering a wide range of formal patterns of reasoning in both normative and epistemic domains, while also incorporating non-formal cognitive factors that influence human reasoning. Our results indicate that, although LLMs generally adhere to valid reasoning patterns, they exhibit notable inconsistencies in specific types of normative reasoning and display cognitive biases similar to those observed in psychological studies of human reasoning. These findings highlight challenges in achieving logical consistency in LLMs' normative reasoning and provide insights for enhancing their reliability. All data and code are released publicly at https://github.com/kmineshima/NeuBAROCO.
>
---
## 更新

#### [replaced 001] SEA-LION: Southeast Asian Languages in One Network
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05747v4](http://arxiv.org/pdf/2504.05747v4)**

> **作者:** Raymond Ng; Thanh Ngan Nguyen; Yuli Huang; Ngee Chia Tai; Wai Yi Leong; Wei Qi Leong; Xianbin Yong; Jian Gang Ngui; Yosephine Susanto; Nicholas Cheng; Hamsawardhini Rengarajan; Peerat Limkonchotiwat; Adithya Venkatadri Hulagadri; Kok Wai Teng; Yeo Yeow Tong; Bryan Siow; Wei Yi Teo; Wayne Lau; Choon Meng Tan; Brandon Ong; Zhi Hao Ong; Jann Railey Montalan; Adwin Chan; Sajeban Antonyrex; Ren Lee; Esther Choa; David Ong Tat-Wee; Bing Jie Darius Liu; William Chandra Tjhi; Erik Cambria; Leslie Teo
>
> **备注:** Accepted at IJCNLP-AACL 2025 (Main Track). We released our model at https://huggingface.co/collections/aisingapore/sea-lionv3-672589a39cdadd6a5b199581
>
> **摘要:** Recently, Large Language Models (LLMs) have dominated much of the artificial intelligence scene with their ability to process and generate natural languages. However, the majority of LLM research and development remains English-centric, leaving low-resource languages such as those in the Southeast Asian (SEA) region under-represented. To address this representation gap, we introduce Llama-SEA-LION-v3-8B-IT and Gemma-SEA-LION-v3-9B-IT, two cutting-edge multilingual LLMs designed for SEA languages. The SEA-LION family of LLMs supports 11 SEA languages, namely English, Chinese, Indonesian, Vietnamese, Malay, Thai, Burmese, Lao, Filipino, Tamil, and Khmer. Our work leverages large-scale multilingual continued pre-training with a comprehensive post-training regime involving multiple stages of instruction fine-tuning, alignment, and model merging. Evaluation results on multilingual benchmarks indicate that our models achieve state-of-the-art performance across LLMs supporting SEA languages. We open-source the models to benefit the wider SEA community.
>
---
#### [replaced 002] Evaluating the Role of Verifiers in Test-Time Scaling for Legal Reasoning Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.25623v2](http://arxiv.org/pdf/2510.25623v2)**

> **作者:** Davide Romano; Jonathan Schwarz; Daniele Giofré
>
> **备注:** Accepted to EMNLP - NLLP Workshop
>
> **摘要:** Test-time scaling (TTS) techniques can improve the performance of large language models (LLMs) at the expense of additional computation and latency. While TTS has proven effective in formal domains such as mathematics and programming, its value in argumentative domains such as law remains underexplored. We present an empirical study of verifier-based TTS methods for legal multiple-choice QA (MCQA) across five benchmarks. Using a family of 7 reward models, we evaluate both outcome-level (Best-of-$N$) and process-level (tree search) verification under realistic low-$N$ budgets. Our analysis systematically investigates how verifier utility is affected by key properties such as domain specialization, model size, and supervision type (process-supervised PRMs vs. outcome-only ORMs), even when applied across different roles.
>
---
#### [replaced 003] GradEscape: A Gradient-Based Evader Against AI-Generated Text Detectors
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08188v2](http://arxiv.org/pdf/2506.08188v2)**

> **作者:** Wenlong Meng; Shuguo Fan; Chengkun Wei; Min Chen; Yuwei Li; Yuanchao Zhang; Zhikun Zhang; Wenzhi Chen
>
> **备注:** Accepted by USENIX Security'25; Update badges and Artifact Appendix
>
> **摘要:** In this paper, we introduce GradEscape, the first gradient-based evader designed to attack AI-generated text (AIGT) detectors. GradEscape overcomes the undifferentiable computation problem, caused by the discrete nature of text, by introducing a novel approach to construct weighted embeddings for the detector input. It then updates the evader model parameters using feedback from victim detectors, achieving high attack success with minimal text modification. To address the issue of tokenizer mismatch between the evader and the detector, we introduce a warm-started evader method, enabling GradEscape to adapt to detectors across any language model architecture. Moreover, we employ novel tokenizer inference and model extraction techniques, facilitating effective evasion even in query-only access. We evaluate GradEscape on four datasets and three widely-used language models, benchmarking it against four state-of-the-art AIGT evaders. Experimental results demonstrate that GradEscape outperforms existing evaders in various scenarios, including with an 11B paraphrase model, while utilizing only 139M parameters. We have successfully applied GradEscape to two real-world commercial AIGT detectors. Our analysis reveals that the primary vulnerability stems from disparity in text expression styles within the training data. We also propose a potential defense strategy to mitigate the threat of AIGT evaders. We open-source our GradEscape for developing more robust AIGT detectors.
>
---
#### [replaced 004] Unstructured Evidence Attribution for Long Context Query Focused Summarization
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.14409v2](http://arxiv.org/pdf/2502.14409v2)**

> **作者:** Dustin Wright; Zain Muhammad Mujahid; Lu Wang; Isabelle Augenstein; David Jurgens
>
> **备注:** EMNLP 2025 Main; 29 pages; 24 figures; 8 tables
>
> **摘要:** Large language models (LLMs) are capable of generating coherent summaries from very long contexts given a user query, and extracting and citing evidence spans helps improve the trustworthiness of these summaries. Whereas previous work has focused on evidence citation with fixed levels of granularity (e.g. sentence, paragraph, document, etc.), we propose to extract unstructured (i.e., spans of any length) evidence in order to acquire more relevant and consistent evidence than in the fixed granularity case. We show how existing systems struggle to copy and properly cite unstructured evidence, which also tends to be "lost-in-the-middle". To help models perform this task, we create the Summaries with Unstructured Evidence Text dataset (SUnsET), a synthetic dataset generated using a novel pipeline, which can be used as training supervision for unstructured evidence summarization. We demonstrate across 5 LLMs and 4 datasets spanning human written, synthetic, single, and multi-document settings that LLMs adapted with SUnsET generate more relevant and factually consistent evidence with their summaries, extract evidence from more diverse locations in their context, and can generate more relevant and consistent summaries than baselines with no fine-tuning and fixed granularity evidence. We release SUnsET and our generation code to the public.
>
---
#### [replaced 005] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24592v2](http://arxiv.org/pdf/2510.24592v2)**

> **作者:** Guoxin Chen; Jing Wu; Xinjie Chen; Wayne Xin Zhao; Ruihua Song; Chengxi Li; Kai Fan; Dayiheng Liu; Minpeng Liao
>
> **备注:** https://github.com/Chen-GX/ReForm
>
> **摘要:** Autoformalization, which translates natural language mathematics into machine-verifiable formal statements, is critical for using formal mathematical reasoning to solve math problems stated in natural language. While Large Language Models can generate syntactically correct formal statements, they often fail to preserve the original problem's semantic intent. This limitation arises from the LLM approaches' treating autoformalization as a simplistic translation task which lacks mechanisms for self-reflection and iterative refinement that human experts naturally employ. To address these issues, we propose ReForm, a Reflective Autoformalization method that tightly integrates semantic consistency evaluation into the autoformalization process. This enables the model to iteratively generate formal statements, assess its semantic fidelity, and self-correct identified errors through progressive refinement. To effectively train this reflective model, we introduce Prospective Bounded Sequence Optimization (PBSO), which employs different rewards at different sequence positions to ensure that the model develops both accurate autoformalization and correct semantic validations, preventing superficial critiques that would undermine the purpose of reflection. Extensive experiments across four autoformalization benchmarks demonstrate that ReForm achieves an average improvement of 22.6 percentage points over the strongest baselines. To further ensure evaluation reliability, we introduce ConsistencyCheck, a benchmark of 859 expert-annotated items that not only validates LLMs as judges but also reveals that autoformalization is inherently difficult: even human experts produce semantic errors in up to 38.5% of cases.
>
---
#### [replaced 006] M-Prometheus: A Suite of Open Multilingual LLM Judges
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04953v2](http://arxiv.org/pdf/2504.04953v2)**

> **作者:** José Pombal; Dongkeun Yoon; Patrick Fernandes; Ian Wu; Seungone Kim; Ricardo Rei; Graham Neubig; André F. T. Martins
>
> **摘要:** The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on synthetic multilingual feedback data instead of translated data. We release our models, training dataset, and code.
>
---
#### [replaced 007] How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18480v2](http://arxiv.org/pdf/2510.18480v2)**

> **作者:** Han Peng; Peiyu Liu; Zican Dong; Daixuan Cheng; Junyi Li; Yiru Tang; Shuo Wang; Wayne Xin Zhao
>
> **备注:** Withdrawn by the authors to better delineate the related work from the paper's original contributions
>
> **摘要:** Diffusion language models (DLMs) have emerged as a promising alternative to the long-dominant autoregressive (AR) paradigm, offering a parallelable decoding process that could yield greater efficiency. Yet, in practice, current open-source DLMs often underperform their AR counterparts in speed, limiting their real-world utility. This work presents a systematic study of DLM efficiency, identifying key issues in prior evaluation methods. Through empirical benchmarking and a roofline-based theoretical analysis, we demonstrate that AR models generally achieve higher throughput, while DLMs consistently lag. We also investigate acceleration strategies, finding that techniques like dual cache and parallel decoding mainly offer gains at small batch sizes, with their benefits diminishing upon scaling. Our findings underscore the necessity of robust evaluation methods and improved acceleration strategies to advance research on DLMs.
>
---
#### [replaced 008] Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers
- **分类: cs.CV; cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.21497v2](http://arxiv.org/pdf/2505.21497v2)**

> **作者:** Wei Pang; Kevin Qinghong Lin; Xiangru Jian; Xi He; Philip Torr
>
> **备注:** Project Page: https://github.com/Paper2Poster/Paper2Poster
>
> **摘要:** Academic poster generation is a crucial yet challenging task in scientific communication, requiring the compression of long-context interleaved documents into a single, visually coherent page. To address this challenge, we introduce the first benchmark and metric suite for poster generation, which pairs recent conference papers with author-designed posters and evaluates outputs on (i)Visual Quality-semantic alignment with human posters, (ii)Textual Coherence-language fluency, (iii)Holistic Assessment-six fine-grained aesthetic and informational criteria scored by a VLM-as-judge, and notably (iv)PaperQuiz-the poster's ability to convey core paper content as measured by VLMs answering generated quizzes. Building on this benchmark, we propose PosterAgent, a top-down, visual-in-the-loop multi-agent pipeline: the (a)Parser distills the paper into a structured asset library; the (b)Planner aligns text-visual pairs into a binary-tree layout that preserves reading order and spatial balance; and the (c)Painter-Commenter loop refines each panel by executing rendering code and using VLM feedback to eliminate overflow and ensure alignment. In our comprehensive evaluation, we find that GPT-4o outputs-though visually appealing at first glance-often exhibit noisy text and poor PaperQuiz scores, and we find that reader engagement is the primary aesthetic bottleneck, as human-designed posters rely largely on visual semantics to convey meaning. Our fully open-source variants (e.g. based on the Qwen-2.5 series) outperform existing 4o-driven multi-agent systems across nearly all metrics, while using 87% fewer tokens. It transforms a 22-page paper into a finalized yet editable .pptx poster - all for just $0.005. These findings chart clear directions for the next generation of fully automated poster-generation models. The code and datasets are available at https://github.com/Paper2Poster/Paper2Poster.
>
---
#### [replaced 009] Let LRMs Break Free from Overthinking via Self-Braking Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14604v4](http://arxiv.org/pdf/2505.14604v4)**

> **作者:** Haoran Zhao; Yuchen Yan; Yongliang Shen; Haolei Xu; Wenqi Zhang; Kaitao Song; Jian Shao; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Accepted to NeurIPS 2025; Camera ready version, 10 pages. Github:https://github.com/ZJU-REAL/Self-Braking-Tuning Project Page: https://ZJU-REAL.github.io/SBT
>
> **摘要:** Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking. Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions. In this paper, we propose a novel framework, Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point. Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models.
>
---
#### [replaced 010] TinyTim: A Family of Language Models for Divergent Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11607v2](http://arxiv.org/pdf/2508.11607v2)**

> **作者:** Christopher J. Agostino
>
> **备注:** 7 pages, 3 figures, accepted to NeurIPS Creative AI track, models available at https://hf.co/npc-worldwide/
>
> **摘要:** In the search for artificial general intelligence, model development and training has focused primarily on vast datasets of known problems and their accepted solutions. This process necessarily produces convergent systems which are fundamentally incapable of the conceptual reframing that is required for genuine creative breakthroughs. Inspired by the divergent cognitive processes that allow humans to make such creative leaps, our work introduces a family of language models, TinyTim, to serve as sources of divergent generation within broader systems. These models have been created by fine-tuning on the anti-parsimonious text of James Joyce's `Finnegans Wake'. Quantitative analysis of both an unsupervised fine-tuned model (TinyTim-V1) and a new instruction-tuned variant (TinyTim-V2) demonstrates a profound capacity for lexical invention; the foundational V1 model exhibits a Yule's K score for lexical richness over twenty times greater than that of convergent baselines. This trait is a stable property of the family, as the instruction-tuned V2 maintains a statistically distinct profile and resists factual convergence, sacrificing benchmark performance to preserve its core generative style. This work establishes a methodology for engineering specialized divergent models that, when paired with convergent systems, can reframe problems and force breakthroughs beyond the reach of statistical optimization alone.
>
---
#### [replaced 011] Reward Collapse in Aligning Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; math.OC; stat.ML**

- **链接: [http://arxiv.org/pdf/2305.17608v2](http://arxiv.org/pdf/2305.17608v2)**

> **作者:** Ziang Song; Tianle Cai; Jason D. Lee; Weijie J. Su
>
> **备注:** Accepted for publication in the Journal of Data Science (JDS), reference JDS1201
>
> **摘要:** The extraordinary capabilities of large language models (LLMs) such as ChatGPT and GPT-4 are in part unleashed by aligning them with reward models that are trained on human preferences, which are often represented as rankings of responses to prompts. In this paper, we document the phenomenon of \textit{reward collapse}, an empirical observation where the prevailing ranking-based approach results in an \textit{identical} reward distribution \textit{regardless} of the prompts during the terminal phase of training. This outcome is undesirable as open-ended prompts like ``write a short story about your best friend'' should yield a continuous range of rewards for their completions, while specific prompts like ``what is the capital of New Zealand'' should generate either high or low rewards. Our theoretical investigation reveals that reward collapse is primarily due to the insufficiency of the ranking-based objective function to incorporate prompt-related information during optimization. This insight allows us to derive closed-form expressions for the reward distribution associated with a set of utility functions in an asymptotic regime. To overcome reward collapse, we introduce a prompt-aware optimization scheme that provably admits a prompt-dependent reward distribution within the interpolating regime. Our experimental results suggest that our proposed prompt-aware utility functions significantly alleviate reward collapse during the training of reward models.
>
---
#### [replaced 012] AI Debate Aids Assessment of Controversial Claims
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02175v2](http://arxiv.org/pdf/2506.02175v2)**

> **作者:** Salman Rahman; Sheriff Issaka; Ashima Suvarna; Genglin Liu; James Shiffer; Jaeyoung Lee; Md Rizwan Parvez; Hamid Palangi; Shi Feng; Nanyun Peng; Yejin Choi; Julian Michael; Liwei Jiang; Saadia Gabriel
>
> **摘要:** As AI grows more powerful, it will increasingly shape how we understand the world. But with this influence comes the risk of amplifying misinformation and deepening social divides-especially on consequential topics where factual accuracy directly impacts well-being. Scalable Oversight aims to ensure AI systems remain truthful even when their capabilities exceed those of their evaluators. Yet when humans serve as evaluators, their own beliefs and biases can impair judgment. We study whether AI debate can guide biased judges toward the truth by having two AI systems debate opposing sides of controversial factuality claims on COVID-19 and climate change where people hold strong prior beliefs. We conduct two studies. Study I recruits human judges with either mainstream or skeptical beliefs who evaluate claims through two protocols: debate (interaction with two AI advisors arguing opposing sides) or consultancy (interaction with a single AI advisor). Study II uses AI judges with and without human-like personas to evaluate the same protocols. In Study I, debate consistently improves human judgment accuracy and confidence calibration, outperforming consultancy by 4-10% across COVID-19 and climate change claims. The improvement is most significant for judges with mainstream beliefs (up to +15.2% accuracy on COVID-19 claims), though debate also helps skeptical judges who initially misjudge claims move toward accurate views (+4.7% accuracy). In Study II, AI judges with human-like personas achieve even higher accuracy (78.5%) than human judges (70.1%) and default AI judges without personas (69.8%), suggesting their potential for supervising frontier AI models. These findings highlight AI debate as a promising path toward scalable, bias-resilient oversight in contested domains.
>
---
#### [replaced 013] Hysteresis Activation Function for Efficient Inference
- **分类: cs.LG; cs.CL; cs.NE**

- **链接: [http://arxiv.org/pdf/2411.10573v3](http://arxiv.org/pdf/2411.10573v3)**

> **作者:** Moshe Kimhi; Idan Kashani; Avi Mendelson; Chaim Baskin
>
> **备注:** Accepted to 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024)
>
> **摘要:** The widely used ReLU is favored for its hardware efficiency, {as the implementation at inference is a one bit sign case,} yet suffers from issues such as the ``dying ReLU'' problem, where during training, neurons fail to activate and constantly remain at zero, as highlighted by Lu et al. Traditional approaches to mitigate this issue often introduce more complex and less hardware-friendly activation functions. In this work, we propose a Hysteresis Rectified Linear Unit (HeLU), an efficient activation function designed to address the ``dying ReLU'' problem with minimal complexity. Unlike traditional activation functions with fixed thresholds for training and inference, HeLU employs a variable threshold that refines the backpropagation. This refined mechanism allows simpler activation functions to achieve competitive performance comparable to their more complex counterparts without introducing unnecessary complexity or requiring inductive biases. Empirical evaluations demonstrate that HeLU enhances model generalization across diverse datasets, offering a promising solution for efficient and effective inference suitable for a wide range of neural network architectures.
>
---
#### [replaced 014] MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09499v3](http://arxiv.org/pdf/2503.09499v3)**

> **作者:** Zhe Xu; Daoyuan Chen; Zhenqing Ling; Yaliang Li; Ying Shen
>
> **备注:** Accepted by NeurIPS'25. 30 pages, 2 figures, 13 tables
>
> **摘要:** Large foundation models face challenges in acquiring transferable, structured thinking abilities, especially when supervised with rigid templates or crowd-annotated instruction datasets. Unlike prior approaches, we focus on a thinking-centric data synthesis paradigm that enables models to evolve through self-generated, cognitively guided data. We propose MindGYM, a structured and scalable framework for question synthesis, composed of: (1) Cognitive Thinking Process Injection, which infuses high-level reasoning objectives to shape the model's synthesis behavior; (2) Seed Single-Hop Question Synthesis, generating atomic questions from diverse semantic types to encourage broader thinking; and (3) Challenging Multi-Hop QA Synthesis, composing more complex multi-hop questions based on QA seeds for deeper reasoning. Detailed analysis shows that synthetic data generated by our method achieves 16.7% higher average quality and 67.91% lower quality variance compared to baseline sources, highlighting that both high-quality and self-contained data are essential for effective, thinking-oriented fine-tuning. MindGYM improves performance on six reasoning benchmarks, achieving gains of up to 16% on MathVision using only 400 data samples, and generalizable improvements across different model sizes and architectures. MindGYM underscores the viability of self-challenging mechanisms in refining large model capabilities while minimizing human intervention and resource demands. Code and data are released to promote data-centric research into self-evolving foundation models driven by their internal reasoning capabilities.
>
---
#### [replaced 015] The LSCD Benchmark: a Testbed for Diachronic Word Meaning Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.00176v2](http://arxiv.org/pdf/2404.00176v2)**

> **作者:** Dominik Schlechtweg; Sachin Yadav; Nikolay Arefyev
>
> **摘要:** Lexical Semantic Change Detection (LSCD) is a complex, lemma-level task, which is usually operationalized based on two subsequently applied usage-level tasks: First, Word-in-Context (WiC) labels are derived for pairs of usages. Then, these labels are represented in a graph on which Word Sense Induction (WSI) is applied to derive sense clusters. Finally, LSCD labels are derived by comparing sense clusters over time. This modularity is reflected in most LSCD datasets and models. It also leads to a large heterogeneity in modeling options and task definitions, which is exacerbated by a variety of dataset versions, preprocessing options and evaluation metrics. This heterogeneity makes it difficult to evaluate models under comparable conditions, to choose optimal model combinations or to reproduce results. Hence, we provide a benchmark repository standardizing LSCD evaluation. Through transparent implementation results become easily reproducible and by standardization different components can be freely combined. The repository reflects the task's modularity by allowing model evaluation for WiC, WSI and LSCD. This allows for careful evaluation of increasingly complex model components providing new ways of model optimization. We use the implemented benchmark to conduct a number of experiments with recent models and systematically improve the state-of-the-art.
>
---
#### [replaced 016] Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.25054v2](http://arxiv.org/pdf/2510.25054v2)**

> **作者:** Pedro Corrêa; João Lima; Victor Moreno; Lucas Ueda; Paula Dornhofer Paro Costa
>
> **备注:** Submitted to IEEE ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Advancements in spoken language processing have driven the development of spoken language models (SLMs), designed to achieve universal audio understanding by jointly learning text and audio representations for a wide range of tasks. Although promising results have been achieved, there is growing discussion regarding these models' generalization capabilities and the extent to which they truly integrate audio and text modalities in their internal representations. In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results indicate that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.
>
---
#### [replaced 017] Are You There God? Lightweight Narrative Annotation of Christian Fiction with LMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19756v2](http://arxiv.org/pdf/2507.19756v2)**

> **作者:** Rebecca M. M. Hicke; Brian W. Haggard; Mia Ferrante; Rayhan Khanna; David Mimno
>
> **备注:** Accepted to CHR 2025
>
> **摘要:** In addition to its more widely studied cultural movements, American Evangelicalism has a well-developed but less externally visible literary side. Christian Fiction, however, has been little studied, and what scholarly attention there is has focused on the explosively popular Left Behind series. In this work, we use computational tools to provide both a broad topical overview of Christian Fiction as a genre and a more directed exploration of how its authors depict divine acts. Working with human annotators, we first developed a codebook for identifying "acts of God." We then adapted the codebook for use by a recent, lightweight LM with the assistance of a much larger model. The laptop-scale LM is largely capable of matching human annotations, even when the task is subtle and challenging. Using these annotations, we show that significant and meaningful differences exist between divine acts depicted by the Left Behind books and Christian Fiction more broadly.
>
---
#### [replaced 018] Dependency Structure Augmented Contextual Scoping Framework for Multimodal Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.11331v2](http://arxiv.org/pdf/2504.11331v2)**

> **作者:** Hao Liu; Lijun He; Jiaxi Liang; Zhihan Ren; Haixia Bi; Fan Li
>
> **摘要:** Multimodal Aspect-Based Sentiment Analysis (MABSA) seeks to extract fine-grained information from image-text pairs to identify aspect terms and determine their sentiment polarity. However, existing approaches often fall short in simultaneously addressing three core challenges: Sentiment Cue Perception (SCP), Multimodal Information Misalignment (MIM), and Semantic Noise Elimination (SNE). To overcome these limitations, we propose DASCO (\textbf{D}ependency Structure \textbf{A}ugmented \textbf{Sco}ping Framework), a fine-grained scope-oriented framework that enhances aspect-level sentiment reasoning by leveraging dependency parsing trees. First, we designed a multi-task pretraining strategy for MABSA on our base model, combining aspect-oriented enhancement, image-text matching, and aspect-level sentiment-sensitive cognition. This improved the model's perception of aspect terms and sentiment cues while achieving effective image-text alignment, addressing key challenges like SCP and MIM. Furthermore, we incorporate dependency trees as syntactic branch combining with semantic branch, guiding the model to selectively attend to critical contextual elements within a target-specific scope while effectively filtering out irrelevant noise for addressing SNE problem. Extensive experiments on two benchmark datasets across three subtasks demonstrate that DASCO achieves state-of-the-art performance in MABSA, with notable gains in JMASA (+2.3\% F1 and +3.5\% precision on Twitter2015). The source code is available at https://github.com/LHaoooo/DASCO .
>
---
#### [replaced 019] Wisdom and Delusion of LLM Ensembles for Code Generation and Repair
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.21513v2](http://arxiv.org/pdf/2510.21513v2)**

> **作者:** Fernando Vallecillos-Ruiz; Max Hort; Leon Moonen
>
> **备注:** Added Acknowledgments section and hyphenated last names
>
> **摘要:** Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems. To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool. We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs.
>
---
#### [replaced 020] Language Models can Self-Improve at State-Value Estimation for Better Search
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02878v3](http://arxiv.org/pdf/2503.02878v3)**

> **作者:** Ethan Mendes; Alan Ritter
>
> **摘要:** Collecting ground-truth rewards or human demonstrations for multi-step reasoning tasks is often prohibitively expensive, particularly in interactive domains such as web tasks. We introduce Self-Taught Lookahead (STL), a reward-free framework that improves language model-based value functions by reasoning explicitly about state transitions. STL can be viewed as a chain-of-thought analogue of the value iteration algorithm: instead of regressing directly on numeric values, a value LLM is trained to simulate a step of lookahead in natural language - predicting the next action, resulting state, and rationale for its value, thereby refining value estimates without any labeled data. This self-supervised procedure yields more accurate state-value predictions, which in turn enable lightweight search algorithms to expand fewer states while maintaining strong performance. Empirically, STL-trained value models built on moderately sized (8B parameter) open-weight LLMs boost web agent success rates by 39%, achieving comparable performance with proprietary models. STL also generalizes to multi-hop QA and math puzzles. We find that STL enables small open-source models to guide efficient search, reducing inference costs by integrating explicit reasoning with value learning.
>
---
#### [replaced 021] Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.20059v2](http://arxiv.org/pdf/2510.20059v2)**

> **作者:** Mehrdad Ghassabi; Sadra Hakim; Hamidreza Baradaran Kashani; Pedram Rostami
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Enhancing reasoning capabilities in small language models is critical for specialized applications such as medical question answering, particularly in underrepresented languages like Persian. In this study, we employ Reinforcement Learning with AI Feedback (RLAIF) and Direct preference optimization (DPO) to improve the reasoning skills of a general-purpose Persian language model. To achieve this, we translated a multiple-choice medical question-answering dataset into Persian and used RLAIF to generate rejected-preferred answer pairs, which are essential for DPO training. By prompting both teacher and student models to produce Chain-of-Thought (CoT) reasoning responses, we compiled a dataset containing correct and incorrect reasoning trajectories. This dataset, comprising 2 million tokens in preferred answers and 2.5 million tokens in rejected ones, was used to train a baseline model, significantly enhancing its medical reasoning capabilities in Persian. Remarkably, the resulting model outperformed its predecessor, gaokerena-V, which was trained on approximately 57 million tokens, despite leveraging a much smaller dataset. These results highlight the efficiency and effectiveness of reasoning-focused training approaches in developing domain-specific language models with limited data availability.
>
---
#### [replaced 022] SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04721v2](http://arxiv.org/pdf/2506.04721v2)**

> **作者:** Yuru Jiang; Wenxuan Ding; Shangbin Feng; Greg Durrett; Yulia Tsvetkov
>
> **备注:** NeurIPS 2025
>
> **摘要:** We propose SPARTA ALIGNMENT, an algorithm to collectively align multiple LLMs through competition and combat. To complement a single model's lack of diversity in generation and biases in evaluation, multiple LLMs form a "sparta tribe" to compete against each other in fulfilling instructions while serving as judges for the competition of others. For each iteration, one instruction and two models are selected for a duel, the other models evaluate the two responses, and their evaluation scores are aggregated through a adapted elo-ranking based reputation system, where winners/losers of combat gain/lose weight in evaluating others. The peer-evaluated combat results then become preference pairs where the winning response is preferred over the losing one, and all models learn from these preferences at the end of each iteration. SPARTA ALIGNMENT enables the self-evolution of multiple LLMs in an iterative and collective competition process. Extensive experiments demonstrate that SPARTA ALIGNMENT outperforms initial models and 4 self-alignment baselines across 10 out of 12 tasks and datasets with 7.0% average improvement. Further analysis reveals that SPARTA ALIGNMENT generalizes more effectively to unseen tasks and leverages the expertise diversity of participating models to produce more logical, direct and informative outputs.
>
---
#### [replaced 023] Large Language Models Report Subjective Experience Under Self-Referential Processing
- **分类: cs.CL; cs.AI; 68T50, 68T07; I.2.0; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.24797v2](http://arxiv.org/pdf/2510.24797v2)**

> **作者:** Cameron Berg; Diogo de Lucena; Judd Rosenblatt
>
> **摘要:** Large language models sometimes produce structured, first-person descriptions that explicitly reference awareness or subjective experience. To better understand this behavior, we investigate one theoretically motivated condition under which such reports arise: self-referential processing, a computational motif emphasized across major theories of consciousness. Through a series of controlled experiments on GPT, Claude, and Gemini model families, we test whether this regime reliably shifts models toward first-person reports of subjective experience, and how such claims behave under mechanistic and behavioral probes. Four main results emerge: (1) Inducing sustained self-reference through simple prompting consistently elicits structured subjective experience reports across model families. (2) These reports are mechanistically gated by interpretable sparse-autoencoder features associated with deception and roleplay: surprisingly, suppressing deception features sharply increases the frequency of experience claims, while amplifying them minimizes such claims. (3) Structured descriptions of the self-referential state converge statistically across model families in ways not observed in any control condition. (4) The induced state yields significantly richer introspection in downstream reasoning tasks where self-reflection is only indirectly afforded. While these findings do not constitute direct evidence of consciousness, they implicate self-referential processing as a minimal and reproducible condition under which large language models generate structured first-person reports that are mechanistically gated, semantically convergent, and behaviorally generalizable. The systematic emergence of this pattern across architectures makes it a first-order scientific and ethical priority for further investigation.
>
---
#### [replaced 024] Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14681v2](http://arxiv.org/pdf/2506.14681v2)**

> **作者:** Yuto Harada; Yusuke Yamauchi; Yusuke Oda; Yohei Oseki; Yusuke Miyao; Yu Takagi
>
> **备注:** Accepted to EMNLP 2025 (Main Conference). Models and evaluation results available at: https://github.com/llm-jp/massive-sft
>
> **摘要:** Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness, often surpassing superficial similarity between the training data and the benchmark, and that mid-layer weight changes correlate most strongly with performance gains. We release these 1,000+ SFT models and benchmark results to accelerate further research. All resources are available at https://github.com/llm-jp/massive-sft.
>
---
#### [replaced 025] IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13229v3](http://arxiv.org/pdf/2506.13229v3)**

> **作者:** Zijie Lin; Yang Zhang; Xiaoyan Zhao; Fengbin Zhu; Fuli Feng; Tat-Seng Chua
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
>
---
#### [replaced 026] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18125v2](http://arxiv.org/pdf/2505.18125v2)**

> **作者:** Alan Arazi; Eilam Shapira; Roi Reichart
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** While deep learning has achieved remarkable success across many domains, it has historically underperformed on tabular learning tasks, which remain dominated by gradient boosting decision trees. However, recent advancements are paving the way for Tabular Foundation Models, which can leverage real-world knowledge and generalize across diverse datasets, particularly when the data contains free-text. Although incorporating language model capabilities into tabular tasks has been explored, most existing methods utilize static, target-agnostic textual representations, limiting their effectiveness. We introduce TabSTAR: a Tabular Foundation Model with Semantically Target-Aware Representations. TabSTAR is designed to enable transfer learning on tabular data with textual features, with an architecture free of dataset-specific parameters. It unfreezes a pretrained text encoder and takes as input target tokens, which provide the model with the context needed to learn task-specific embeddings. TabSTAR achieves state-of-the-art performance for both medium- and large-sized datasets across known benchmarks of classification tasks with text features, and its pretraining phase exhibits scaling laws in the number of datasets, offering a pathway for further performance improvements.
>
---
#### [replaced 027] FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.16648v2](http://arxiv.org/pdf/2509.16648v2)**

> **作者:** Debarpan Bhattacharya; Apoorva Kulkarni; Sriram Ganapathy
>
> **备注:** Accepted in the Findings of EMNLP, 2025
>
> **摘要:** The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
>
---
#### [replaced 028] Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13328v2](http://arxiv.org/pdf/2507.13328v2)**

> **作者:** Yulu Qin; Dheeraj Varghese; Adam Dahlgren Lindström; Lucia Donatelli; Kanishka Misra; Najoung Kim
>
> **摘要:** Does vision-and-language (VL) training change the linguistic representations of language models in meaningful ways? Most results in the literature have shown inconsistent or marginal differences, both behaviorally and representationally. In this work, we start from the hypothesis that the domain in which VL training could have a significant effect is lexical-conceptual knowledge, in particular its taxonomic organization. Through comparing minimal pairs of text-only LMs and their VL-trained counterparts, we first show that the VL models often outperform their text-only counterparts on a text-only question-answering task that requires taxonomic understanding of concepts mentioned in the questions. Using an array of targeted behavioral and representational analyses, we show that the LMs and VLMs do not differ significantly in terms of their taxonomic knowledge itself, but they differ in how they represent questions that contain concepts in a taxonomic relation vs. a non-taxonomic relation. This implies that the taxonomic knowledge itself does not change substantially through additional VL training, but VL training does improve the deployment of this knowledge in the context of a specific task, even when the presentation of the task is purely linguistic.
>
---
#### [replaced 029] AutoLibra: Agent Metric Induction from Open-Ended Human Feedback
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02820v3](http://arxiv.org/pdf/2505.02820v3)**

> **作者:** Hao Zhu; Phil Cuvin; Xinkai Yu; Charlotte Ka Yee Yan; Jason Zhang; Diyi Yang
>
> **备注:** https://github.com/Open-Social-World/autolibra
>
> **摘要:** Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose **AutoLibra**, a framework for agent evaluation, that transforms open-ended human feedback *e.g.* "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own" into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra serve human prompt engineers for diagonalize agent failures and improve prompts iterative. Moreover, we find that AutoLibra can induce metrics for automatic optimization for agents, which makes agents improve through self-regulation. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents.
>
---
#### [replaced 030] PairUni: Pairwise Training for Unified Multimodal Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.25682v2](http://arxiv.org/pdf/2510.25682v2)**

> **作者:** Jiani Zheng; Zhiyang Teng; Xiangtai Li; Anran Wang; Yu Tian; Kunpeng Qiu; Ye Tian; Haochen Wang; Zhuochen Wang
>
> **备注:** 21 pages, 11 figures, and 8 tables
>
> **摘要:** Unified vision-language models (UVLMs) must perform both understanding and generation within a single architecture, but these tasks rely on heterogeneous data and supervision, making it difficult to balance them during reinforcement learning (RL). We propose PairUni, a unified framework that reorganizes data into understanding-generation (UG) pairs and aligns optimization accordingly. We first use GPT-o3 to augment single-task data, generating captions for understanding samples and question-answer (QA) pairs for generation samples, forming aligned pairs from the same instance. Additionally, for each generation sample, we retrieve a semantically related understanding example to form a retrieved pair, linking different but related data points. These paired structures expose cross-task semantic correspondences and support consistent policy learning. To leverage this structure, we present Pair-GPRO, a pair-aware variant based on Group Relative Policy Optimization. It assigns a similarity score to each pair to modulate the advantage, strengthening learning from well-aligned examples and reducing task interference. We curate a high-quality dataset of 16K UG pairs named PairUG for RL fine-tuning and evaluate PairUni on the powerful Janus-Pro UVLMs. Our approach achieves balanced improvements on various UVLMs, outperforming strong UVLM RL baselines. Codes are available at https://github.com/Haochen-Wang409/PairUni.
>
---
#### [replaced 031] Towards Predicting Any Human Trajectory In Context
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00871v2](http://arxiv.org/pdf/2506.00871v2)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **备注:** NeurIPS 2025
>
> **摘要:** Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, the need to fine-tune for each new scenario is often impractical for deployment on edge devices. To address this challenge, we introduce \paper, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables adaptation without fine-tuning on the scenario-specific data at inference time without requiring weight updates. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. Project Page: https://fujiry0.github.io/TrajICL-project-page/.
>
---
#### [replaced 032] When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11695v2](http://arxiv.org/pdf/2510.11695v2)**

> **作者:** Lingfei Qian; Xueqing Peng; Yan Wang; Vincent Jim Zhang; Huan He; Hanley Smith; Yi Han; Yueru He; Haohang Li; Yupeng Cao; Yangyang Yu; Alejandro Lopez-Lira; Peng Lu; Jian-Yun Nie; Guojun Xiong; Jimin Huang; Sophia Ananiadou
>
> **摘要:** Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.
>
---
#### [replaced 033] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07001v2](http://arxiv.org/pdf/2506.07001v2)**

> **作者:** Yize Cheng; Vinu Sankar Sadasivan; Mehrdad Saberi; Shoumik Saha; Soheil Feizi
>
> **备注:** NeurIPS 2025
>
> **摘要:** The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.
>
---
#### [replaced 034] Similarity-Distance-Magnitude Activations
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12760v2](http://arxiv.org/pdf/2509.12760v2)**

> **作者:** Allen Schmaltz
>
> **备注:** 18 pages, 5 tables, 1 algorithm. arXiv admin note: substantial text overlap with arXiv:2502.20167
>
> **摘要:** We introduce the Similarity-Distance-Magnitude (SDM) activation function, a more robust and interpretable formulation of the standard softmax activation function, adding Similarity (i.e., correctly predicted depth-matches into training) awareness and Distance-to-training-distribution awareness to the existing output Magnitude (i.e., decision-boundary) awareness, and enabling interpretability-by-exemplar via dense matching. We further introduce the SDM estimator, based on a data-driven partitioning of the class-wise empirical CDFs via the SDM activation, to control the class- and prediction-conditional accuracy among selective classifications. When used as the final-layer activation over pre-trained language models for selective classification, the SDM estimator is more robust to co-variate shifts and out-of-distribution inputs than existing calibration methods using softmax activations, while remaining informative over in-distribution data.
>
---
#### [replaced 035] Neural Networks for Learnable and Scalable Influence Estimation of Instruction Fine-Tuning Data
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09969v4](http://arxiv.org/pdf/2502.09969v4)**

> **作者:** Ishika Agarwal; Dilek Hakkani-Tür
>
> **摘要:** Influence functions provide crucial insights into model training, but existing methods suffer from large computational costs and limited generalization. Particularly, recent works have proposed various metrics and algorithms to calculate the influence of data using language models, which do not scale well with large models and datasets. This is because of the expensive forward and backward passes required for computation, substantial memory requirements to store large models, and poor generalization of influence estimates to new data. In this paper, we explore the use of small neural networks -- which we refer to as the InfluenceNetwork -- to estimate influence values, achieving up to 99% cost reduction. Our evaluation demonstrates that influence values can be estimated with models just 0.0027% the size of full language models (we use 7B and 8B versions). We apply our algorithm of estimating influence values (called NN-CIFT: Neural Networks for effiCient Instruction Fine-Tuning) to the downstream task of subset selection for general instruction fine-tuning. In our study, we include four state-of-the-art influence functions and show no compromise in performance, despite large speedups, between NN-CIFT and the original influence functions. We provide an in-depth hyperparameter analyses of NN-CIFT. The code for our method can be found here: https://github.com/agarwalishika/NN-CIFT.
>
---
#### [replaced 036] BhashaBench V1: A Comprehensive Benchmark for the Quadrant of Indic Domains
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.25409v2](http://arxiv.org/pdf/2510.25409v2)**

> **作者:** Vijay Devane; Mohd Nauman; Bhargav Patel; Aniket Mahendra Wakchoure; Yogeshkumar Sant; Shyam Pawar; Viraj Thakur; Ananya Godse; Sunil Patra; Neha Maurya; Suraj Racha; Nitish Kamal Singh; Ajay Nagpal; Piyush Sawarkar; Kundeshwar Vijayrao Pundalik; Rohit Saluja; Ganesh Ramakrishnan
>
> **摘要:** The rapid advancement of large language models(LLMs) has intensified the need for domain and culture specific evaluation. Existing benchmarks are largely Anglocentric and domain-agnostic, limiting their applicability to India-centric contexts. To address this gap, we introduce BhashaBench V1, the first domain-specific, multi-task, bilingual benchmark focusing on critical Indic knowledge systems. BhashaBench V1 contains 74,166 meticulously curated question-answer pairs, with 52,494 in English and 21,672 in Hindi, sourced from authentic government and domain-specific exams. It spans four major domains: Agriculture, Legal, Finance, and Ayurveda, comprising 90+ subdomains and covering 500+ topics, enabling fine-grained evaluation. Evaluation of 29+ LLMs reveals significant domain and language specific performance gaps, with especially large disparities in low-resource domains. For instance, GPT-4o achieves 76.49% overall accuracy in Legal but only 59.74% in Ayurveda. Models consistently perform better on English content compared to Hindi across all domains. Subdomain-level analysis shows that areas such as Cyber Law, International Finance perform relatively well, while Panchakarma, Seed Science, and Human Rights remain notably weak. BhashaBench V1 provides a comprehensive dataset for evaluating large language models across India's diverse knowledge domains. It enables assessment of models' ability to integrate domain-specific knowledge with bilingual understanding. All code, benchmarks, and resources are publicly available to support open research.
>
---
#### [replaced 037] Latent Chain-of-Thought for Visual Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.23925v2](http://arxiv.org/pdf/2510.23925v2)**

> **作者:** Guohao Sun; Hang Hua; Jian Wang; Jiebo Luo; Sohail Dianat; Majid Rabbani; Raghuveer Rao; Zhiqiang Tao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Chain-of-thought (CoT) reasoning is critical for improving the interpretability and reliability of Large Vision-Language Models (LVLMs). However, existing training algorithms such as SFT, PPO, and GRPO may not generalize well across unseen reasoning tasks and heavily rely on a biased reward model. To address this challenge, we reformulate reasoning in LVLMs as posterior inference and propose a scalable training algorithm based on amortized variational inference. By leveraging diversity-seeking reinforcement learning algorithms, we introduce a novel sparse reward function for token-level learning signals that encourage diverse, high-likelihood latent CoT, overcoming deterministic sampling limitations and avoiding reward hacking. Additionally, we implement a Bayesian inference-scaling strategy that replaces costly Best-of-N and Beam Search with a marginal likelihood to efficiently rank optimal rationales and answers. We empirically demonstrate that the proposed method enhances the state-of-the-art LVLMs on seven reasoning benchmarks, in terms of effectiveness, generalization, and interpretability.
>
---
#### [replaced 038] Unveiling Unicode's Unseen Underpinnings in Undermining Authorship Attribution
- **分类: cs.CR; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.15840v3](http://arxiv.org/pdf/2508.15840v3)**

> **作者:** Robert Dilworth
>
> **备注:** 33 pages, 7 figures, 3 tables
>
> **摘要:** When using a public communication channel -- whether formal or informal, such as commenting or posting on social media -- end users have no expectation of privacy: they compose a message and broadcast it for the world to see. Even if an end user takes utmost precautions to anonymize their online presence -- using an alias or pseudonym; masking their IP address; spoofing their geolocation; concealing their operating system and user agent; deploying encryption; registering with a disposable phone number or email; disabling non-essential settings; revoking permissions; and blocking cookies and fingerprinting -- one obvious element still lingers: the message itself. Assuming they avoid lapses in judgment or accidental self-exposure, there should be little evidence to validate their actual identity, right? Wrong. The content of their message -- necessarily open for public consumption -- exposes an attack vector: stylometric analysis, or author profiling. In this paper, we dissect the technique of stylometry, discuss an antithetical counter-strategy in adversarial stylometry, and devise enhancements through Unicode steganography.
>
---
#### [replaced 039] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13444v2](http://arxiv.org/pdf/2505.13444v2)**

> **作者:** Liyan Tang; Grace Kim; Xinyu Zhao; Thom Lake; Wenxuan Ding; Fangcong Yin; Prasann Singhal; Manya Wadhwa; Zeyu Leo Liu; Zayne Sprague; Ramya Namuduri; Bodun Hu; Juan Diego Rodriguez; Puyuan Peng; Greg Durrett
>
> **备注:** NeurIPS 2025 Datasets & Benchmarks
>
> **摘要:** Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs.
>
---
#### [replaced 040] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.12371v2](http://arxiv.org/pdf/2505.12371v2)**

> **作者:** Yinghao Zhu; Ziyi He; Haoran Hu; Xiaochen Zheng; Xichen Zhang; Zixiang Wang; Junyi Gao; Liantao Ma; Lequan Yu
>
> **备注:** Accepted by NeurIPS 2025 Datasets & Benchmarks Track
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at https://medagentboard.netlify.app/.
>
---
#### [replaced 041] Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.15201v3](http://arxiv.org/pdf/2505.15201v3)**

> **作者:** Christian Walder; Deep Karkhanis
>
> **摘要:** Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function. While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains. We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples.
>
---
#### [replaced 042] This Candidate is [MASK]. Prompt-based Sentiment Extraction and Reference Letters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16325v3](http://arxiv.org/pdf/2410.16325v3)**

> **作者:** Fabian Slonimczyk
>
> **摘要:** I propose a relatively simple way to deploy pre-trained large language models (LLMs) in order to extract sentiment and other useful features from text data. The method, which I refer to as prompt-based sentiment extraction, offers multiple advantages over other methods used in economics and finance. In particular, it accepts the text input as is (without pre-processing) and produces a sentiment score that has a probability interpretation. Unlike other LLM-based approaches, it does not require any fine-tuning or labeled data. I apply my prompt-based strategy to a hand-collected corpus of confidential reference letters (RLs). I show that the sentiment contents of RLs are clearly reflected in job market outcomes. Candidates with higher average sentiment in their RLs perform markedly better regardless of the measure of success chosen. Moreover, I show that sentiment dispersion among letter writers negatively affects the job market candidate's performance. I compare my sentiment extraction approach to other commonly used methods for sentiment analysis: `bag-of-words' approaches, fine-tuned language models, and querying advanced chatbots. No other method can fully reproduce the results obtained by prompt-based sentiment extraction. Finally, I slightly modify the method to obtain `gendered' sentiment scores (as in Eberhardt et al., 2023). I show that RLs written for female candidates emphasize `grindstone' personality traits, whereas male candidates' letters emphasize `standout' traits. These gender differences negatively affect women's job market outcomes.
>
---
#### [replaced 043] Model Provenance Testing for Large Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00706v2](http://arxiv.org/pdf/2502.00706v2)**

> **作者:** Ivica Nikolic; Teodora Baluta; Prateek Saxena
>
> **摘要:** Large language models are increasingly customized through fine-tuning and other adaptations, creating challenges in enforcing licensing terms and managing downstream impacts. Tracking model origins is crucial both for protecting intellectual property and for identifying derived models when biases or vulnerabilities are discovered in foundation models. We address this challenge by developing a framework for testing model provenance: Whether one model is derived from another. Our approach is based on the key observation that real-world model derivations preserve significant similarities in model outputs that can be detected through statistical analysis. Using only black-box access to models, we employ multiple hypothesis testing to compare model similarities against a baseline established by unrelated models. On two comprehensive real-world benchmarks spanning models from 30M to 4B parameters and comprising over 600 models, our tester achieves 90-95% precision and 80-90% recall in identifying derived models. These results demonstrate the viability of systematic provenance verification in production environments even when only API access is available.
>
---
#### [replaced 044] Model-Document Protocol for AI Search
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.25160v2](http://arxiv.org/pdf/2510.25160v2)**

> **作者:** Hongjin Qian; Zheng Liu
>
> **备注:** 10 pages
>
> **摘要:** AI search depends on linking large language models (LLMs) with vast external knowledge sources. Yet web pages, PDF files, and other raw documents are not inherently LLM-ready: they are long, noisy, and unstructured. Conventional retrieval methods treat these documents as verbatim text and return raw passages, leaving the burden of fragment assembly and contextual reasoning to the LLM. This gap underscores the need for a new retrieval paradigm that redefines how models interact with documents. We introduce the Model-Document Protocol (MDP), a general framework that formalizes how raw text is bridged to LLMs through consumable knowledge representations. Rather than treating retrieval as passage fetching, MDP defines multiple pathways that transform unstructured documents into task-specific, LLM-ready inputs. These include agentic reasoning, which curates raw evidence into coherent context; memory grounding, which accumulates reusable notes to enrich reasoning; and structured leveraging, which encodes documents into formal representations such as graphs or key-value caches. All three pathways share the same goal: ensuring that what reaches the LLM is not raw fragments but compact, structured knowledge directly consumable for reasoning. As an instantiation, we present MDP-Agent, which realizes the protocol through an agentic process: constructing document-level gist memories for global coverage, performing diffusion-based exploration with vertical exploitation to uncover layered dependencies, and applying map-reduce style synthesis to integrate large-scale evidence into compact yet sufficient context. Experiments on information-seeking benchmarks demonstrate that MDP-Agent outperforms baselines, validating both the soundness of the MDP framework and the effectiveness of its agentic instantiation.
>
---
#### [replaced 045] UNO-Bench: A Unified Benchmark for Exploring the Compositional Law Between Uni-modal and Omni-modal in Omni Models
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.18915v3](http://arxiv.org/pdf/2510.18915v3)**

> **作者:** Chen Chen; ZeYang Hu; Fengjiao Chen; Liya Ma; Jiaxing Liu; Xiaoyu Li; Ziwen Wang; Xuezhi Cao; Xunliang Cai
>
> **备注:** v3: Switch the paper template. Work in progress. Github: https://github.com/meituan-longcat/UNO-Bench Hugging Face: https://huggingface.co/datasets/meituan-longcat/UNO-Bench
>
> **摘要:** Multimodal Large Languages models have been progressing from uni-modal understanding toward unifying visual, audio and language modalities, collectively termed omni models. However, the correlation between uni-modal and omni-modal remains unclear, which requires comprehensive evaluation to drive omni model's intelligence evolution. In this work, we introduce a novel, high-quality, and UNified Omni model benchmark, UNO-Bench. This benchmark is designed to effectively evaluate both UNi-modal and Omni-modal capabilities under a unified ability taxonomy, spanning 44 task types and 5 modality combinations. It includes 1250 human curated samples for omni-modal with 98% cross-modality solvability, and 2480 enhanced uni-modal samples. The human-generated dataset is well-suited to real-world scenarios, particularly within the Chinese context, whereas the automatically compressed dataset offers a 90% increase in speed and maintains 98% consistency across 18 public benchmarks. In addition to traditional multi-choice questions, we propose an innovative multi-step open-ended question format to assess complex reasoning. A general scoring model is incorporated, supporting 6 question types for automated evaluation with 95% accuracy. Experimental result shows the Compositional Law between omni-modal and uni-modal performance and the omni-modal capability manifests as a bottleneck effect on weak models, while exhibiting synergistic promotion on strong models.
>
---
#### [replaced 046] Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04380v3](http://arxiv.org/pdf/2502.04380v3)**

> **作者:** Zhenqing Ling; Daoyuan Chen; Liuyi Yao; Qianli Shen; Yaliang Li; Ying Shen
>
> **备注:** Accepted by NeurIPS'25 main track. 47 pages, 21 figures, 32 tables
>
> **摘要:** Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
>
---
#### [replaced 047] Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding
- **分类: cs.AI; cs.CL; I.2.7; I.2.11; I.2.6**

- **链接: [http://arxiv.org/pdf/2508.21204v2](http://arxiv.org/pdf/2508.21204v2)**

> **作者:** Vanessa Figueiredo
>
> **摘要:** We study how prompt-level inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding method paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which prompt-level cognitive scaffolds can reliably shape emergent instructional strategies in LLMs.
>
---
#### [replaced 048] The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.11094v2](http://arxiv.org/pdf/2506.11094v2)**

> **作者:** Songyang Liu; Chaozhuo Li; Jiameng Qiu; Xi Zhang; Feiran Huang; Litian Zhang; Yiming Hei; Philip S. Yu
>
> **备注:** 20 pages, preprint
>
> **摘要:** With the rapid advancement of artificial intelligence, Large Language Models (LLMs) have shown remarkable capabilities in Natural Language Processing (NLP), including content generation, human-computer interaction, machine translation, and code generation. However, their widespread deployment has also raised significant safety concerns. In particular, LLM-generated content can exhibit unsafe behaviors such as toxicity, bias, or misinformation, especially in adversarial contexts, which has attracted increasing attention from both academia and industry. Although numerous studies have attempted to evaluate these risks, a comprehensive and systematic survey on safety evaluation of LLMs is still lacking. This work aims to fill this gap by presenting a structured overview of recent advances in safety evaluation of LLMs. Specifically, we propose a four-dimensional taxonomy: (i) Why to evaluate, which explores the background of safety evaluation of LLMs, how they differ from general LLMs evaluation, and the significance of such evaluation; (ii) What to evaluate, which examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and related aspects; (iii) Where to evaluate, which summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (iv) How to evaluate, which reviews existing mainstream evaluation methods based on the roles of the evaluators and some evaluation frameworks that integrate the entire evaluation pipeline. Finally, we identify the challenges in safety evaluation of LLMs and propose promising research directions to promote further advancement in this field. We emphasize the necessity of prioritizing safety evaluation to ensure the reliable and responsible deployment of LLMs in real-world applications.
>
---
#### [replaced 049] Improving LLM Safety Alignment with Dual-Objective Optimization
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03710v3](http://arxiv.org/pdf/2503.03710v3)**

> **作者:** Xuandong Zhao; Will Cai; Tianneng Shi; David Huang; Licong Lin; Song Mei; Dawn Song
>
> **备注:** ICML 2025
>
> **摘要:** Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
>
---
#### [replaced 050] LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.08604v2](http://arxiv.org/pdf/2510.08604v2)**

> **作者:** Raffaele Mura; Giorgio Piras; Kamilė Lukošiūtė; Maura Pintor; Amin Karbasi; Battista Biggio
>
> **摘要:** Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models.
>
---
#### [replaced 051] TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation
- **分类: cs.CL; I.2.7; I.2.6; I.2.0**

- **链接: [http://arxiv.org/pdf/2510.25536v2](http://arxiv.org/pdf/2510.25536v2)**

> **作者:** Bangde Du; Minghao Guo; Songming He; Ziyi Ye; Xi Zhu; Weihang Su; Shuqi Zhu; Yujia Zhou; Yongfeng Zhang; Qingyao Ai; Yiqun Liu
>
> **备注:** Main paper: 11 pages, 3 figures, 6 tables. Appendix: 28 pages. Bangde Du and Minghao Guo contributed equally. Corresponding authors: Ziyi Ye (ziyiye@fudan.edu.cn), Qingyao Ai (aiqy@tsinghua.edu.cn)
>
> **摘要:** Large Language Models (LLMs) are exhibiting emergent human-like abilities and are increasingly envisioned as the foundation for simulating an individual's communication style, behavioral tendencies, and personality traits. However, current evaluations of LLM-based persona simulation remain limited: most rely on synthetic dialogues, lack systematic frameworks, and lack analysis of the capability requirement. To address these limitations, we introduce TwinVoice, a comprehensive benchmark for assessing persona simulation across diverse real-world contexts. TwinVoice encompasses three dimensions: Social Persona (public social interactions), Interpersonal Persona (private dialogues), and Narrative Persona (role-based expression). It further decomposes the evaluation of LLM performance into six fundamental capabilities, including opinion consistency, memory recall, logical reasoning, lexical fidelity, persona tone, and syntactic style. Experimental results reveal that while advanced models achieve moderate accuracy in persona simulation, they still fall short of capabilities such as syntactic style and memory recall. Consequently, the average performance achieved by LLMs remains considerably below the human baseline.
>
---
#### [replaced 052] Speak & Spell: LLM-Driven Controllable Phonetic Error Augmentation for Robust Dialogue State Tracking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.06263v2](http://arxiv.org/pdf/2409.06263v2)**

> **作者:** Jihyun Lee; Solee Im; Wonjun Lee; Gary Geunbae Lee
>
> **备注:** Accepted to AACL-IJCNLP 2025
>
> **摘要:** Dialogue State Tracking (DST) is a key part of task-oriented dialogue systems, identifying important information in conversations. However, its accuracy drops significantly in spoken dialogue environments due to named entity errors from Automatic Speech Recognition (ASR) systems. We introduce a simple yet effective data augmentation method that targets those entities to improve the robustness of DST model. Our novel method can control the placement of errors using keyword-highlighted prompts while introducing phonetically similar errors. As a result, our method generated sufficient error patterns on keywords, leading to improved accuracy in noised and low-accuracy ASR environments.
>
---
#### [replaced 053] Nek Minit: Harnessing Pragmatic Metacognitive Prompting for Explainable Sarcasm Detection of Australian and Indian English
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15095v2](http://arxiv.org/pdf/2505.15095v2)**

> **作者:** Ishmanbir Singh; Dipankar Srirag; Aditya Joshi
>
> **备注:** ALTA 2025 (Best Paper Honorable Mention). Camera-ready
>
> **摘要:** Sarcasm is a challenge to sentiment analysis because of the incongruity between stated and implied sentiment. The challenge is exacerbated when the implication may be relevant to a specific country or geographical region. Pragmatic metacognitive prompting (PMP) is a cognition-inspired technique that has been used for pragmatic reasoning. In this paper, we harness PMP for explainable sarcasm detection for Australian and Indian English, alongside a benchmark dataset for standard English. We manually add sarcasm explanations to an existing sarcasm-labeled dataset for Australian and Indian English called BESSTIE, and compare the performance for explainable sarcasm detection for them with FLUTE, a standard English dataset containing sarcasm explanations. Our approach utilising PMP when evaluated on two open-weight LLMs (GEMMA and LLAMA) achieves statistically significant performance improvement across all tasks and datasets when compared with four alternative prompting strategies. We also find that alternative techniques such as agentic prompting mitigate context-related failures by enabling external knowledge retrieval. The focused contribution of our work is utilising PMP in generating sarcasm explanations for varieties of English.
>
---
#### [replaced 054] ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24388v2](http://arxiv.org/pdf/2505.24388v2)**

> **作者:** Hao Chen; Yukun Yan; Sen Mei; Wanxiang Che; Zhenghao Liu; Qi Shi; Xinze Li; Yuchun Fan; Pengcheng Huang; Qiushi Xiong; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge to improve factuality. However, existing RAG systems frequently underutilize the retrieved documents, failing to extract and integrate the key clues needed to support faithful and interpretable reasoning, especially in cases where relevant evidence is implicit, scattered, or obscured by noise. To address this issue, we propose ClueAnchor, a novel framework for enhancing RAG via clue-anchored reasoning exploration and optimization. ClueAnchor extracts key clues from retrieved content and generates multiple reasoning paths based on different knowledge configurations, optimizing the model by selecting the most appropriate reasoning path for the given context through reward-based preference optimization. Experiments show that ClueAnchor significantly outperforms prior RAG baselines in the completeness and robustness of reasoning. Further analysis confirms its strong resilience to noisy or partially relevant retrieved content, as well as its capability to identify supporting evidence even in the absence of explicit clue supervision during inference. All codes are available at https://github.com/thunlp/ClueAnchor.
>
---
#### [replaced 055] RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.21319v2](http://arxiv.org/pdf/2509.21319v2)**

> **作者:** Zhilin Wang; Jiaqi Zeng; Olivier Delalleau; Ellie Evans; Daniel Egert; Hoo-Chang Shin; Felipe Soares; Yi Dong; Oleksii Kuchaiev
>
> **备注:** Added link to access models: https://huggingface.co/collections/nvidia/reward-models-10-2025
>
> **摘要:** Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). Models: https://huggingface.co/collections/nvidia/reward-models-10-2025
>
---
#### [replaced 056] VC4VG: Optimizing Video Captions for Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24134v2](http://arxiv.org/pdf/2510.24134v2)**

> **作者:** Yang Du; Zhuoran Lin; Kaiqiang Song; Biao Wang; Zhicheng Zheng; Tiezheng Ge; Bo Zheng; Qin Jin
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Recent advances in text-to-video (T2V) generation highlight the critical role of high-quality video-text pairs in training models capable of producing coherent and instruction-aligned videos. However, strategies for optimizing video captions specifically for T2V training remain underexplored. In this paper, we introduce VC4VG (Video Captioning for Video Generation), a comprehensive caption optimization framework tailored to the needs of T2V models. We begin by analyzing caption content from a T2V perspective, decomposing the essential elements required for video reconstruction into multiple dimensions, and proposing a principled caption design methodology. To support evaluation, we construct VC4VG-Bench, a new benchmark featuring fine-grained, multi-dimensional, and necessity-graded metrics aligned with T2V-specific requirements. Extensive T2V fine-tuning experiments demonstrate a strong correlation between improved caption quality and video generation performance, validating the effectiveness of our approach. We release all benchmark tools and code at https://github.com/alimama-creative/VC4VG to support further research.
>
---
#### [replaced 057] Beyond Isolated Dots: Benchmarking Structured Table Construction as Deep Knowledge Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16271v2](http://arxiv.org/pdf/2507.16271v2)**

> **作者:** Tianyun Zhong; Guozhao Mo; Yanjiang Liu; Yihan Chen; Lingdi Kong; Xuanang Chen; Yaojie Lu; Hongyu Lin; Shiwei Ye; Xianpei Han; Ben He; Le Sun
>
> **摘要:** With the emergence of large language models (LLMs), there is an expectation that LLMs can effectively extract explicit information from complex real-world documents (e.g., papers, reports). However, most LLMs generate paragraph-style answers that are chaotic, disorganized, and untraceable. To bridge this gap, we introduce the Arranged and Organized Extraction Benchmark (AOE), a new bilingual benchmark with data and documents of varying lengths designed to systematically evaluate the ability of LLMs to comprehend fragmented documents and reconstruct isolated information into one organized table. Unlike conventional text-to-table tasks, which rely on fixed schema and narrow task domains, AOE includes 11 carefully crafted tasks across three diverse domains, requiring models to generate context-specific schema tailored to varied input queries. In the experiment, we evaluated both open-source and closed-source state-of-the-art LLMs. The results show that even the most advanced models struggled significantly. The benchmark is available at https://anonymous.4open.science/r/AOE-Benchmark/.
>
---
#### [replaced 058] Towards a Method for Synthetic Generation of Persons with Aphasia Transcripts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.24817v2](http://arxiv.org/pdf/2510.24817v2)**

> **作者:** Jason M. Pittman; Anton Phillips Jr.; Yesenia Medina-Santos; Brielle C. Stark
>
> **备注:** 19 pages, 1 figure, 7 tables
>
> **摘要:** In aphasia research, Speech-Language Pathologists (SLPs) devote extensive time to manually coding speech samples using Correct Information Units (CIUs), a measure of how informative an individual sample of speech is. Developing automated systems to recognize aphasic language is limited by data scarcity. For example, only about 600 transcripts are available in AphasiaBank yet billions of tokens are used to train large language models (LLMs). In the broader field of machine learning (ML), researchers increasingly turn to synthetic data when such are sparse. Therefore, this study constructs and validates two methods to generate synthetic transcripts of the AphasiaBank Cat Rescue picture description task. One method leverages a procedural programming approach while the second uses Mistral 7b Instruct and Llama 3.1 8b Instruct LLMs. The methods generate transcripts across four severity levels (Mild, Moderate, Severe, Very Severe) through word dropping, filler insertion, and paraphasia substitution. Overall, we found, compared to human-elicited transcripts, Mistral 7b Instruct best captures key aspects of linguistic degradation observed in aphasia, showing realistic directional changes in NDW, word count, and word length amongst the synthetic generation methods. Based on the results, future work should plan to create a larger dataset, fine-tune models for better aphasic representation, and have SLPs assess the realism and usefulness of the synthetic transcripts.
>
---
#### [replaced 059] Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model
- **分类: cs.MM; cs.CL; cs.IR; cs.SD; eess.AS; 68T, 68T45, 68T10**

- **链接: [http://arxiv.org/pdf/2503.09205v3](http://arxiv.org/pdf/2503.09205v3)**

> **作者:** Ali Vosoughi; Dimitra Emmanouilidou; Hannes Gamper
>
> **备注:** 5 pages, 5 figures, 2 tables. Accepted at EUSIPCO 2025
>
> **摘要:** Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
>
---
#### [replaced 060] Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08410v2](http://arxiv.org/pdf/2506.08410v2)**

> **作者:** Ziyang Ma; Qingyue Yuan; Zhenglin Wang; Deyu Zhou
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Previous research has primarily focused on the cognitive error detection capabilities of Large Language Models (LLMs), often prompting them to analyze mistakes in reasoning chains. However, few studies have examined the meta-cognitive abilities of LLMs (e.g., their self-awareness of step errors), which are crucial for their reliability. While studies on LLM self-evaluation present some measures, such as perplexity, which can reflect the answer correctness and be viewed as the lens of meta-cognition, they lack step-level analysis and adaptation. This paper studies the evaluation of LLM meta-cognition using the current lenses and how to improve these lenses. Specifically, we propose AutoMeco, an Automated Meta-cognition Evaluation framework for benchmarking the existing lenses. Furthermore, a training-free Markovian Intrinsic Reward Adjustment strategy, MIRA, is proposed to boost current meta-cognition lenses. Experimental results on three mathematical reasoning datasets and three LLMs show the reasonableness of AutoMeco by comparing it with Best-of-N verification. Moreover, the meta-cognition ability of LLMs can be better evaluated using MIRA.
>
---
#### [replaced 061] Language Model Preference Evaluation with Multiple Weak Evaluators
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.12869v4](http://arxiv.org/pdf/2410.12869v4)**

> **作者:** Zhengyu Hu; Jieyu Zhang; Zhihan Xiong; Alexander Ratner; Kaize Ding; Ranjay Krishna
>
> **摘要:** Despite the remarkable success of Large Language Models (LLMs), evaluating their outputs' quality regarding preference remains a critical challenge. While existing works usually leverage a strong LLM as the judge for comparing LLMs' response pairwisely, such a single-evaluator approach is vulnerable to cyclic preference, i.e., output A is better than B, B than C, but C is better than A, causing contradictory evaluation results. To address this, we introduce PGED (Preference Graph Ensemble and Denoise), a novel approach that leverages multiple model-based evaluators to construct preference graphs, and then ensembles and denoises these graphs for acyclic, non-contradictory evaluation results. We provide theoretical guarantees for our framework, demonstrating its efficacy in recovering the ground truth preference structure. Extensive experiments on ten benchmarks demonstrate PGED 's superiority in three applications: 1) model ranking for evaluation, 2) response selection for test-time scaling, and 3) data selection for model fine-tuning. Notably, PGED combines small LLM evaluators (e.g., Llama3-8B, Mistral-7B, Qwen2-7B) to outperform strong ones (e.g., Qwen2-72B), showcasing its effectiveness in enhancing evaluation reliability and improving model performance.
>
---
#### [replaced 062] Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.14889v2](http://arxiv.org/pdf/2510.14889v2)**

> **作者:** Soorya Ram Shimgekar; Ruining Zhao; Agam Goyal; Violeta J. Rodriguez; Paul A. Bloom; Hari Sundaram; Koustuv Saha
>
> **摘要:** On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments.
>
---
#### [replaced 063] Completion $\neq$ Collaboration: Scaling Collaborative Effort with Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.25744v2](http://arxiv.org/pdf/2510.25744v2)**

> **作者:** Shannon Zejiang Shen; Valerie Chen; Ken Gu; Alexis Ross; Zixian Ma; Jillian Ross; Alex Gu; Chenglei Si; Wayne Chi; Andi Peng; Jocelyn J Shen; Ameet Talwalkar; Tongshuang Wu; David Sontag
>
> **备注:** 22 pages, 5 figures, 3 tables
>
> **摘要:** Current evaluations of agents remain centered around one-shot task completion, failing to account for the inherently iterative and collaborative nature of many real-world problems, where human goals are often underspecified and evolve. We argue for a shift from building and assessing task completion agents to developing collaborative agents, assessed not only by the quality of their final outputs but by how well they engage with and enhance human effort throughout the problem-solving process. To support this shift, we introduce collaborative effort scaling, a framework that captures how an agent's utility grows with increasing user involvement. Through case studies and simulated evaluations, we show that state-of-the-art agents often underperform in multi-turn, real-world scenarios, revealing a missing ingredient in agent design: the ability to sustain engagement and scaffold user understanding. Collaborative effort scaling offers a lens for diagnosing agent behavior and guiding development toward more effective interactions.
>
---
#### [replaced 064] More of the Same: Persistent Representational Harms Under Increased Representation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00333v2](http://arxiv.org/pdf/2503.00333v2)**

> **作者:** Jennifer Mickel; Maria De-Arteaga; Leqi Liu; Kevin Tian
>
> **备注:** Accepted by the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) as a poster paper; 39 pages, 7 figures, 15 tables
>
> **摘要:** To recognize and mitigate the harms of generative AI systems, it is crucial to consider who is represented in the outputs of generative AI systems and how people are represented. A critical gap emerges when naively improving who is represented, as this does not imply bias mitigation efforts have been applied to address how people are represented. We critically examined this by investigating gender representation in occupation across state-of-the-art large language models. We first show evidence suggesting that over time there have been interventions to models altering the resulting gender distribution, and we find that women are more represented than men when models are prompted to generate biographies or personas. We then demonstrate that representational biases persist in how different genders are represented by examining statistically significant word differences across genders. This results in a proliferation of representational harms, stereotypes, and neoliberalism ideals that, despite existing interventions to increase female representation, reinforce existing systems of oppression.
>
---
#### [replaced 065] Comparing human and LLM politeness strategies in free production
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09391v2](http://arxiv.org/pdf/2506.09391v2)**

> **作者:** Haoran Zhao; Robert D. Hawkins
>
> **备注:** 25 pages, 5 figures | EMNLP 2025 camera-ready version
>
> **摘要:** Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems.
>
---
#### [replaced 066] TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24014v2](http://arxiv.org/pdf/2510.24014v2)**

> **作者:** Yizhu Jiao; Sha Li; Sizhe Zhou; Heng Ji; Jiawei Han
>
> **备注:** Source code: https://github.com/yzjiao/Text2DB
>
> **摘要:** The task of information extraction (IE) is to extract structured knowledge from text. However, it is often not straightforward to utilize IE output due to the mismatch between the IE ontology and the downstream application needs. We propose a new formulation of IE TEXT2DB that emphasizes the integration of IE output and the target database (or knowledge base). Given a user instruction, a document set, and a database, our task requires the model to update the database with values from the document set to satisfy the user instruction. This task requires understanding user instructions for what to extract and adapting to the given DB/KB schema for how to extract on the fly. To evaluate this new task, we introduce a new benchmark featuring common demands such as data infilling, row population, and column addition. In addition, we propose an LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer component that interacts with the database, the Planner component that generates a code-based plan with calls to IE models, and the Analyzer component that provides feedback regarding code quality before execution. Experiments show that OPAL can successfully adapt to diverse database schemas by generating different code plans and calling the required IE models. We also highlight difficult cases such as dealing with large databases with complex dependencies and extraction hallucination, which we believe deserve further investigation. Source code: https://github.com/yzjiao/Text2DB
>
---
#### [replaced 067] Epistemic Diversity and Knowledge Collapse in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04226v4](http://arxiv.org/pdf/2510.04226v4)**

> **作者:** Dustin Wright; Sarah Masud; Jared Moore; Srishti Yadav; Maria Antoniak; Chan Young Park; Isabelle Augenstein
>
> **备注:** 16 pages; 8 figures, 4 tables; v2 changelog: Fixed the modeling for table 3, random effect is the model version; v3 changelog: Fixed minor formatting issues in tables 2 and 3; v4 changelog: Fixed some typos and model description
>
> **摘要:** Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation
>
---
#### [replaced 068] Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13308v2](http://arxiv.org/pdf/2505.13308v2)**

> **作者:** Hengli Li; Chenxi Li; Tong Wu; Xuekai Zhu; Yuxuan Wang; Zhaoxin Yu; Eric Hanchen Jiang; Song-Chun Zhu; Zixia Jia; Ying Nian Wu; Zilong Zheng
>
> **摘要:** Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs.
>
---
#### [replaced 069] Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13464v2](http://arxiv.org/pdf/2506.13464v2)**

> **作者:** Zhengyu Hu; Jianxun Lian; Zheyuan Xiao; Seraphina Zhang; Tianfu Wang; Nicholas Jing Yuan; Xing Xie; Hui Xiong
>
> **摘要:** Large language models (LLMs) have shown impressive capabilities across tasks such as mathematics, coding, and reasoning, yet their learning ability, which is crucial for adapting to dynamic environments and acquiring new knowledge, remains underexplored. In this work, we address this gap by introducing a framework inspired by cognitive psychology and education. Specifically, we decompose general learning ability into three distinct, complementary dimensions: Learning from Instructor (acquiring knowledge via explicit guidance), Learning from Concept (internalizing abstract structures and generalizing to new contexts), and Learning from Experience (adapting through accumulated exploration and feedback). We conduct a comprehensive empirical study across the three learning dimensions and identify several insightful findings, such as (i) interaction improves learning; (ii) conceptual understanding is scale-emergent and benefits larger models; and (iii) LLMs are effective few-shot learners but not many-shot learners. Based on our framework and empirical findings, we introduce a benchmark that provides a unified and realistic evaluation of LLMs' general learning abilities across three learning cognition dimensions. It enables diagnostic insights and supports evaluation and development of more adaptive and human-like models.
>
---
#### [replaced 070] Controlling Thinking Speed in Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03704v2](http://arxiv.org/pdf/2507.03704v2)**

> **作者:** Zhengkai Lin; Zhihang Fu; Ze Chen; Chao Chen; Liang Xie; Wenxiao Wang; Deng Cai; Zheng Wang; Jieping Ye
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Human cognition is theorized to operate in two modes: fast, intuitive System 1 thinking and slow, deliberate System 2 thinking. While current Large Reasoning Models (LRMs) excel at System 2 thinking, their inability to perform fast thinking leads to high computational overhead and latency. In this work, we enable LRMs to approximate human intelligence through dynamic thinking speed adjustment, optimizing accuracy-efficiency trade-offs. Our approach addresses two key questions: (1) how to control thinking speed in LRMs, and (2) when to adjust it for optimal performance. For the first question, we identify the steering vector that governs slow-fast thinking transitions in LRMs' representation space. Using this vector, we achieve the first representation editing-based test-time scaling effect, outperforming existing prompt-based scaling methods. For the second question, we apply real-time difficulty estimation to signal reasoning segments of varying complexity. Combining these techniques, we propose the first reasoning strategy that enables fast processing of easy steps and deeper analysis for complex reasoning. Without any training or additional cost, our plug-in module delivers an average +1.3% accuracy with -8.6% token usage across leading LRMs and advanced reasoning benchmarks. All of our algorithms are implemented based on vLLM and are expected to support broader applications and inspire future research.
>
---
#### [replaced 071] Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.01001v2](http://arxiv.org/pdf/2504.01001v2)**

> **作者:** José Pombal; Nuno M. Guerreiro; Ricardo Rei; André F. T. Martins
>
> **摘要:** As language models improve and become capable of performing more complex tasks across modalities, evaluating them automatically becomes increasingly challenging. Developing strong and robust task-specific automatic metrics gets harder, and human-annotated test sets -- which are expensive to create -- saturate more quickly. A compelling alternative is to design reliable strategies to automate the creation of test data and evaluation, but previous attempts either rely on pre-existing data, or focus solely on individual tasks. We present Zero-shot Benchmarking (ZSB), a framework for creating high-quality benchmarks for any task by leveraging language models for both synthetic test data creation and evaluation. ZSB is simple and flexible: it requires only the creation of a prompt for data generation and one for evaluation; it is scalable to tasks and languages where collecting real-world data is costly or impractical; it is model-agnostic, allowing the creation of increasingly challenging benchmarks as models improve. To assess the effectiveness of our framework, we create benchmarks for five text-only tasks and a multi-modal one: general capabilities in four languages (English, Chinese, French, and Korean), translation, and general vision-language capabilities in English. We then rank a broad range of open and closed systems on our benchmarks. ZSB rankings consistently correlate strongly with human rankings, outperforming widely-adopted standard benchmarks. Through ablations, we find that strong benchmarks can be created with open models, and that judge model size and dataset variety are crucial drivers of performance. We release all our benchmarks, and code to reproduce our experiments and to produce new benchmarks.
>
---
#### [replaced 072] Are LLMs Rigorous Logical Reasoners? Empowering Natural Language Proof Generation by Stepwise Decoding with Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2311.06736v3](http://arxiv.org/pdf/2311.06736v3)**

> **作者:** Ying Su; Mingwen Liu; Zhijiang Guo
>
> **备注:** 15 pages, 2 figures, 11 tables. Accepted by AACL 2025 main conference
>
> **摘要:** Logical reasoning is a pivotal component in the field of artificial intelligence. Proof planning, particularly in contexts requiring the validation of explanation accuracy, continues to present challenges. The recent advancement of large language models (LLMs) has led to significant progress in natural language proof planning, evolving from one-stage generators to more complex three-stage systems that include additional searchers or verifiers. While these assisted methods improve the quality of generated results, they also introduce increased search efforts and computational costs. Furthermore, the generative process itself remains underexplored. In this study, we propose a stepwise decoding approach augmented by contrastive learning to address two common errors encountered during the LLM generator's decoding process. We fine-tune the language model using both vanilla and enhanced hard negatives to mitigate these decoding errors. Empirical results demonstrate the effectiveness of our strategy. Additionally, our further analysis reveals that even larger LLMs still struggle to generate rigorous logical chains.
>
---
#### [replaced 073] CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21257v2](http://arxiv.org/pdf/2507.21257v2)**

> **作者:** David Maria Schmidt; Raoul Schubert; Philipp Cimiano
>
> **备注:** Research Track, 24th International Semantic Web Conference (ISWC 2025), November 2-6, 2025, Nara, Japan
>
> **摘要:** Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries.
>
---

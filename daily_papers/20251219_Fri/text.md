# 自然语言处理 cs.CL

- **最新发布 58 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Evaluating OpenAI GPT Models for Translation of Endangered Uralic Languages: A Comparison of Reasoning and Non-Reasoning Architectures
- **分类: cs.CL**

- **简介: 该论文属机器翻译任务，旨在评估OpenAI GPT模型（推理型vs非推理型）在濒危乌拉尔语系低资源语言（如科米-兹梁、莫克沙等）与芬兰语互译中的表现。通过平行语料分析拒译率，发现推理模型拒译率低16个百分点，为濒危语言保护提供实证依据。**

- **链接: [https://arxiv.org/pdf/2512.16287v1](https://arxiv.org/pdf/2512.16287v1)**

> **作者:** Yehor Tereshchenko; Mika Hämäläinen; Svitlana Myroniuk
>
> **备注:** IWCLUL 2025
>
> **摘要:** The evaluation of Large Language Models (LLMs) for translation tasks has primarily focused on high-resource languages, leaving a significant gap in understanding their performance on low-resource and endangered languages. This study presents a comprehensive comparison of OpenAI's GPT models, specifically examining the differences between reasoning and non-reasoning architectures for translating between Finnish and four low-resource Uralic languages: Komi-Zyrian, Moksha, Erzya, and Udmurt. Using a parallel corpus of literary texts, we evaluate model willingness to attempt translation through refusal rate analysis across different model architectures. Our findings reveal significant performance variations between reasoning and non-reasoning models, with reasoning models showing 16 percentage points lower refusal rates. The results provide valuable insights for researchers and practitioners working with Uralic languages and contribute to the broader understanding of reasoning model capabilities for endangered language preservation.
>
---
#### [new 002] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Multimodal RewardBench 2（MMRB2），首个面向图文交错模态的奖励模型基准，涵盖文本生成图像、图像编辑、交错生成和多模态推理四任务。旨在评估与提升多模态奖励模型性能，解决其在 omni 模型中缺乏系统评测的问题。**

- **链接: [https://arxiv.org/pdf/2512.16899v1](https://arxiv.org/pdf/2512.16899v1)**

> **作者:** Yushi Hu; Reyhane Askari-Hemmat; Melissa Hall; Emily Dinan; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **备注:** Code and data available at https://github.com/facebookresearch/MMRB2
>
> **摘要:** Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.
>
---
#### [new 003] AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属LLM增强搜索任务，旨在解决搜索代理中参数知识与外部搜索的自适应平衡问题。提出AdaSearch：两阶段、结果驱动的RL框架，显式建模搜索决策，提升知识边界感知与决策可解释性，减少冗余搜索且不损性能。**

- **链接: [https://arxiv.org/pdf/2512.16883v1](https://arxiv.org/pdf/2512.16883v1)**

> **作者:** Tzu-Han Lin; Wei-Lin Chen; Chen-An Li; Hung-yi Lee; Yun-Nung Chen; Yu Meng
>
> **备注:** Preprint. Code and artifacts will be uploaded to https://github.com/hank0316/AdaSearch
>
> **摘要:** Equipping large language models (LLMs) with search engines via reinforcement learning (RL) has emerged as an effective approach for building search agents. However, overreliance on search introduces unnecessary cost and risks exposure to noisy or malicious content, while relying solely on parametric knowledge risks hallucination. The central challenge is to develop agents that adaptively balance parametric knowledge with external search, invoking search only when necessary. Prior work mitigates search overuse by shaping rewards around the number of tool calls. However, these penalties require substantial reward engineering, provide ambiguous credit assignment, and can be exploited by agents that superficially reduce calls. Moreover, evaluating performance solely through call counts conflates necessary and unnecessary search, obscuring the measurement of true adaptive behavior. To address these limitations, we first quantify the self-knowledge awareness of existing search agents via an F1-based decision metric, revealing that methods such as Search-R1 often overlook readily available parametric knowledge. Motivated by these findings, we propose AdaSearch, a simple two-stage, outcome-driven RL framework that disentangles problem solving from the decision of whether to invoke search, and makes this decision process explicit and interpretable. This transparency is crucial for high-stakes domains such as finance and medical question answering, yet is largely neglected by prior approaches. Experiments across multiple model families and sizes demonstrate that AdaSearch substantially improves knowledge-boundary awareness, reduces unnecessary search calls, preserves strong task performance, and offers more transparent, interpretable decision behaviors.
>
---
#### [new 004] BRAID: Bounded Reasoning for Autonomous Inference and Decisions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BRAID方法，属AI推理优化任务，旨在解决LLM推理成本高、效率低的问题。通过Mermaid结构化提示图实现有界推理，提升准确率与成本效率，在多个基准上验证其有效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.15959v1](https://arxiv.org/pdf/2512.15959v1)**

> **作者:** Armağan Amcalar; Eyup Cinar
>
> **摘要:** Large Language Models (LLMs) exhibit nonlinear relationships between performance, cost, and token usage. This paper presents a quantitative study on structured prompting using BRAID (Bounded Reasoning for Au tonomous Inference and Decisions) across multiple GPT model tiers, eval uated on the AdvancedIF, GSM-Hard, and the SCALE MultiChallenge benchmark datasets. BRAID introduces a bounded reasoning framework using Mermaid-based instruction graphs that enable models to reason struc turally rather than through unbounded natural-language token expansion. We show that structured machine-readable prompts substantially increase reasoning accuracy and cost efficiency for agents in production systems. The findings establish BRAID as an effective and scalable technique for optimizing inference efficiency in autonomous agent systems. All datasets and detailed result logs are available at https://benchmark.openserv.ai.
>
---
#### [new 005] Social Story Frames: Contextual Reasoning about Narrative Intent and Reception
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文提出SocialStoryFrames（SSF），旨在建模读者对故事的多维响应（如作者意图、情感、价值判断）。为解决计算模型难以刻画细粒度叙事理解的问题，作者构建形式化框架、设计生成与分类模型，并在6k+社交媒体故事上验证其用于跨社区叙事分析的有效性。**

- **链接: [https://arxiv.org/pdf/2512.15925v1](https://arxiv.org/pdf/2512.15925v1)**

> **作者:** Joel Mire; Maria Antoniak; Steven R. Wilson; Zexin Ma; Achyutarama R. Ganti; Andrew Piper; Maarten Sap
>
> **备注:** Presented at IC2S2 2025; Under Review (ARR Oct 2025)
>
> **摘要:** Reading stories evokes rich interpretive, affective, and evaluative responses, such as inferences about narrative intent or judgments about characters. Yet, computational models of reader response are limited, preventing nuanced analyses. To address this gap, we introduce SocialStoryFrames, a formalism for distilling plausible inferences about reader response, such as perceived author intent, explanatory and predictive reasoning, affective responses, and value judgments, using conversational context and a taxonomy grounded in narrative theory, linguistic pragmatics, and psychology. We develop two models, SSF-Generator and SSF-Classifier, validated through human surveys (N=382 participants) and expert annotations, respectively. We conduct pilot analyses to showcase the utility of the formalism for studying storytelling at scale. Specifically, applying our models to SSF-Corpus, a curated dataset of 6,140 social media stories from diverse contexts, we characterize the frequency and interdependence of storytelling intents, and we compare and contrast narrative practices (and their diversity) across communities. By linking fine-grained, context-sensitive modeling with a generic taxonomy of reader responses, SocialStoryFrames enable new research into storytelling in online communities.
>
---
#### [new 006] Convolutional Lie Operator for Sentence Classification
- **分类: cs.CL**

- **简介: 该论文面向句子分类任务，旨在提升CNN对语言复杂变换的建模能力。作者提出融合李群卷积的新型模型SCLie和DPCLie，利用李代数操作捕捉非欧对称性，实验证明其优于传统CNN。**

- **链接: [https://arxiv.org/pdf/2512.16125v1](https://arxiv.org/pdf/2512.16125v1)**

> **作者:** Daniela N. Rim; Heeyoul Choi
>
> **备注:** Proceedings of the 2024 8th International Conference on Natural Language Processing and Information Retrieval
>
> **摘要:** Traditional Convolutional Neural Networks have been successful in capturing local, position-invariant features in text, but their capacity to model complex transformation within language can be further explored. In this work, we explore a novel approach by integrating Lie Convolutions into Convolutional-based sentence classifiers, inspired by the ability of Lie group operations to capture complex, non-Euclidean symmetries. Our proposed models SCLie and DPCLie empirically outperform traditional Convolutional-based sentence classifiers, suggesting that Lie-based models relatively improve the accuracy by capturing transformations not commonly associated with language. Our findings motivate more exploration of new paradigms in language modeling.
>
---
#### [new 007] From Facts to Conclusions : Integrating Deductive Reasoning in Retrieval-Augmented LLMs
- **分类: cs.CL; cs.AI; cs.CY; cs.IR**

- **简介: 该论文属RAG任务，旨在解决检索源冲突、过时或主观导致的答案不可靠问题。提出推理轨迹增强框架，含文档裁决、冲突分析、 grounded合成三阶段，并设计CATS评估流水线与539查询数据集，显著提升答案正确性与行为一致性。**

- **链接: [https://arxiv.org/pdf/2512.16795v1](https://arxiv.org/pdf/2512.16795v1)**

> **作者:** Shubham Mishra; Samyek Jain; Gorang Mehrishi; Shiv Tiwari; Harsh Sharma; Pratik Narang; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external evidence, but fails when retrieved sources conflict or contain outdated or subjective information. Prior work address these issues independently but lack unified reasoning supervision. We propose a reasoning-trace-augmented RAG framework that adds structured, interpretable reasoning across three stages : (1) document-level adjudication, (2) conflict analysis, and (3) grounded synthesis, producing citation-linked answers or justified refusals. A Conflict-Aware Trust-Score (CATS) pipeline is introduced which evaluates groundedness, factual correctness, refusal accuracy, and conflict-behavior alignment using an LLM-as-a-Judge. Our 539-query reasoning dataset and evaluation pipeline establish a foundation for conflict-aware, interpretable RAG systems. Experimental results demonstrate substantial gains over baselines, most notably with Qwen, where Supervised Fine-Tuning improved End-to-End answer correctness from 0.069 to 0.883 and behavioral adherence from 0.074 to 0.722.
>
---
#### [new 008] A Domain-Adapted Pipeline for Structured Information Extraction from Police Incident Announcements on Social Media
- **分类: cs.CL; cs.CY**

- **简介: 该论文面向社会媒体警察通报的结构化信息抽取任务，解决非正式、噪声文本中关键字段提取难的问题。提出基于LoRA高效微调Qwen2.5-7B的领域适配流水线，从微博警情帖中精准抽取15类字段，显著提升准确率与匹配率。**

- **链接: [https://arxiv.org/pdf/2512.16183v1](https://arxiv.org/pdf/2512.16183v1)**

> **作者:** Mengfan Shen; Kangqi Song; Xindi Wang; Wei Jia; Tao Wang; Ziqiang Han
>
> **备注:** 41 pages,3figures and 9 tables
>
> **摘要:** Structured information extraction from police incident announcements is crucial for timely and accurate data processing, yet presents considerable challenges due to the variability and informal nature of textual sources such as social media posts. To address these challenges, we developed a domain-adapted extraction pipeline that leverages targeted prompt engineering with parameter-efficient fine-tuning of the Qwen2.5-7B model using Low-Rank Adaptation (LoRA). This approach enables the model to handle noisy, heterogeneous text while reliably extracting 15 key fields, including location, event characteristics, and impact assessment, from a high-quality, manually annotated dataset of 4,933 instances derived from 27,822 police briefing posts on Chinese Weibo (2019-2020). Experimental results demonstrated that LoRA-based fine-tuning significantly improved performance over both the base and instruction-tuned models, achieving an accuracy exceeding 98.36% for mortality detection and Exact Match Rates of 95.31% for fatality counts and 95.54% for province-level location extraction. The proposed pipeline thus provides a validated and efficient solution for multi-task structured information extraction in specialized domains, offering a practical framework for transforming unstructured text into reliable structured data in social science research.
>
---
#### [new 009] An Information-Theoretic Framework for Robust Large Language Model Editing
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向大语言模型（LLM）知识编辑任务，旨在解决模型错误或过时信息难以高效、鲁棒更新的问题。提出基于信息瓶颈理论的框架，设计信息瓶颈知识编辑器（IBKE），通过压缩关键信息实现精准、泛化性强的编辑，显著提升编辑准确性与特异性。**

- **链接: [https://arxiv.org/pdf/2512.16227v1](https://arxiv.org/pdf/2512.16227v1)**

> **作者:** Qizhou Chen; Chengyu Wang; Taolin Zhang; Xiaofeng He
>
> **摘要:** Large Language Models (LLMs) have become indispensable tools in science, technology, and society, enabling transformative advances across diverse fields. However, errors or outdated information within these models can undermine their accuracy and restrict their safe deployment. Developing efficient strategies for updating model knowledge without the expense and disruption of full retraining remains a critical challenge. Current model editing techniques frequently struggle to generalize corrections beyond narrow domains, leading to unintended consequences and limiting their practical impact. Here, we introduce a novel framework for editing LLMs, grounded in information bottleneck theory. This approach precisely compresses and isolates the essential information required for generalizable knowledge correction while minimizing disruption to unrelated model behaviors. Building upon this foundation, we present the Information Bottleneck Knowledge Editor (IBKE), which leverages compact latent representations to guide gradient-based updates, enabling robust and broadly applicable model editing. We validate IBKE's effectiveness across multiple LLM architectures and standard benchmark tasks, demonstrating state-of-the-art accuracy and improved generality and specificity of edits. These findings establish a theoretically principled and practical paradigm for open-domain knowledge editing, advancing the utility and trustworthiness of LLMs in real-world applications.
>
---
#### [new 010] Are We on the Right Way to Assessing LLM-as-a-Judge?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属LLM评估任务，旨在解决LLM-as-a-Judge依赖人工标注导致的偏差与可扩展性问题。提出无标注评估套件Sage，基于理性选择理论设计局部自洽与全局逻辑一致性指标，并揭示当前大模型判别可靠性不足及“情境偏好”现象。**

- **链接: [https://arxiv.org/pdf/2512.16041v1](https://arxiv.org/pdf/2512.16041v1)**

> **作者:** Yuanning Feng; Sinan Wang; Zhengxiang Cheng; Yao Wan; Dongping Chen
>
> **摘要:** LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessment of reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming Sage's reliability as an evaluation suite for the robustness and accuracy of LLM-as-a-Judge. Based on Sage, we reveal that current state-of-the-art LLMs exhibit significant reliability problems when acting as judges in both scoring and pairwise settings; even the top-performing models, Gemini-2.5-Pro and GPT-5, fail to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called situational preference, which explains why explicit rubrics or criteria can help the model judge consistently across answer pairs. Our further analysis shows that finetuned LLM-as-a-Judge is a feasible method to boost performance, and the panel-based judge as well as deep reasoning can enhance the judging consistency. We also find substantial inconsistency in human judgments, which indicates that human annotation may not be a reliable gold standard.
>
---
#### [new 011] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属语音翻译任务，探究SpeechLLM（端到端语音大模型）是否优于传统级联方案。作者构建首个综合评测套件Hearing to Translate，对比5个SpeechLLM与16个级联系统，在16项基准、13语对、9种挑战条件下评估，发现级联系统整体更可靠，SpeechLLM仅在部分场景匹敌。**

- **链接: [https://arxiv.org/pdf/2512.16378v1](https://arxiv.org/pdf/2512.16378v1)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at https://github.com/sarapapi/hearing2translate
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [new 012] Grammar-Forced Translation of Natural Language to Temporal Logic using LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自然语言到时序逻辑（TL）的翻译任务，旨在解决现有方法在原子命题提取、共指消解和小样本学习上的不足。提出Grammar-Forced Translation（GraFT）框架，通过分步约束LLM输出词表来降低搜索空间，提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.16814v1](https://arxiv.org/pdf/2512.16814v1)**

> **作者:** William English; Dominic Simon; Sumit Kumar Jha; Rickard Ewetz
>
> **摘要:** Translating natural language (NL) into a formal language such as temporal logic (TL) is integral for human communication with robots and autonomous systems. State-of-the-art approaches decompose the task into a lifting of atomic propositions (APs) phase and a translation phase. However, existing methods struggle with accurate lifting, the existence of co-references, and learning from limited data. In this paper, we propose a framework for NL to TL translation called Grammar Forced Translation (GraFT). The framework is based on the observation that previous work solves both the lifting and translation steps by letting a language model iteratively predict tokens from its full vocabulary. In contrast, GraFT reduces the complexity of both tasks by restricting the set of valid output tokens from the full vocabulary to only a handful in each step. The solution space reduction is obtained by exploiting the unique properties of each problem. We also provide a theoretical justification for why the solution space reduction leads to more efficient learning. We evaluate the effectiveness of GraFT using the CW, GLTL, and Navi benchmarks. Compared with state-of-the-art translation approaches, it can be observed that GraFT the end-to-end translation accuracy by 5.49% and out-of-domain translation accuracy by 14.06% on average.
>
---
#### [new 013] Sigma-Moe-Tiny Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Sigma-MoE-Tiny，一种极致稀疏的MoE语言模型（96专家/层，仅激活1个/词），旨在解决高稀疏下专家负载失衡导致训练不稳定的问题；提出渐进式稀疏化策略，实现稳定训练与强性能。**

- **链接: [https://arxiv.org/pdf/2512.16248v1](https://arxiv.org/pdf/2512.16248v1)**

> **作者:** Qingguo Hu; Zhenghao Lin; Ziyue Yang; Yucheng Ding; Xiao Liu; Yuting Jiang; Ruizhe Wang; Tianyu Chen; Zhongxin Guo; Yifan Xiong; Rui Gao; Lei Qu; Jinsong Su; Peng Cheng; Yeyun Gong
>
> **摘要:** Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures. Project page: https://qghuxmu.github.io/Sigma-MoE-Tiny Code: https://github.com/microsoft/ltp-megatron-lm
>
---
#### [new 014] Bridging the Reality Gap: Efficient Adaptation of ASR systems for Challenging Low-Resource Domains
- **分类: cs.CL**

- **简介: 该论文属ASR领域，旨在解决低资源临床场景下因隐私限制、算力不足和声学偏移导致的性能骤降问题。提出基于LoRA的边缘端隐私保护自适应框架，并引入多域经验回放缓解灾难性遗忘，显著提升目标域WER。**

- **链接: [https://arxiv.org/pdf/2512.16401v1](https://arxiv.org/pdf/2512.16401v1)**

> **作者:** Darshil Chauhan; Adityasinh Solanki; Vansh Patel; Kanav Kapoor; Ritvik Jain; Aditya Bansal; Dhruv Kumar; Prateek Narang
>
> **摘要:** Automatic Speech Recognition (ASR) holds immense potential to streamline clinical documentation, such as digitizing handwritten prescriptions and reports, thereby increasing patient throughput and reducing costs in resource-constrained sectors like rural healthcare. However, realizing this utility is currently obstructed by significant technical barriers: strict data privacy constraints, limited computational resources, and severe acoustic domain shifts. We quantify this gap by showing that a robust multilingual model (IndicWav2Vec) degrades to a stark 40.94% Word Error Rate (WER) when deployed on real-world clinical audio (Gram Vaani), rendering it unusable for practical applications. To address these challenges and bring ASR closer to deployment, we propose an efficient, privacy-preserving adaptation framework. We employ Low-Rank Adaptation (LoRA) to enable continual learning from incoming data streams directly on edge devices, ensuring patient data confidentiality. Our strategy yields a 17.1% relative improvement in WER on the target domain. Furthermore, by integrating multi-domain experience replay, we reduce catastrophic forgetting by 47% compared to naive adaptation. These results demonstrate a viable pathway for building reliable, self-improving ASR systems that can operate effectively within the constraints of high-impact real-world environments.
>
---
#### [new 015] JustRL: Scaling a 1.5B LLM with a Simple RL Recipe
- **分类: cs.CL**

- **简介: 该论文属大语言模型强化学习（RL）优化任务，旨在验证复杂RL训练策略是否必要。作者提出极简的JustRL方法：单阶段、固定超参训练1.5B模型，在数学推理上达SOTA且计算减半，揭示过度设计可能损害探索稳定性。**

- **链接: [https://arxiv.org/pdf/2512.16649v1](https://arxiv.org/pdf/2512.16649v1)**

> **作者:** Bingxiang He; Zekai Qu; Zeyuan Liu; Yinghao Chen; Yuxin Zuo; Cheng Qian; Kaiyan Zhang; Weize Chen; Chaojun Xiao; Ganqu Cui; Ning Ding; Zhiyuan Liu
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Recent advances in reinforcement learning for large language models have converged on increasing complexity: multi-stage training pipelines, dynamic hyperparameter schedules, and curriculum learning strategies. This raises a fundamental question: \textbf{Is this complexity necessary?} We present \textbf{JustRL}, a minimal approach using single-stage training with fixed hyperparameters that achieves state-of-the-art performance on two 1.5B reasoning models (54.9\% and 64.3\% average accuracy across nine mathematical benchmarks) while using 2$\times$ less compute than sophisticated approaches. The same hyperparameters transfer across both models without tuning, and training exhibits smooth, monotonic improvement over 4,000+ steps without the collapses or plateaus that typically motivate interventions. Critically, ablations reveal that adding ``standard tricks'' like explicit length penalties and robust verifiers may degrade performance by collapsing exploration. These results suggest that the field may be adding complexity to solve problems that disappear with a stable, scaled-up baseline. We release our models and code to establish a simple, validated baseline for the community.
>
---
#### [new 016] Decoding Fake Narratives in Spreading Hateful Stories: A Dual-Head RoBERTa Model with Multi-Task Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属NLP多任务分类任务，旨在检测代码混合的印地语-英语社交媒体文本中的“伪仇恨”（Faux-Hate）——即由虚假叙事驱动的仇恨言论。作者提出双头RoBERTa模型，联合完成二元Faux-Hate检测与目标/严重度预测，结合领域预训练和多任务学习，取得竞争性效果。**

- **链接: [https://arxiv.org/pdf/2512.16147v1](https://arxiv.org/pdf/2512.16147v1)**

> **作者:** Yash Bhaskar; Sankalp Bahad; Parameswari Krishnamurthy
>
> **备注:** Accepted Paper, Anthology ID: 2024.icon-fauxhate.3, 4 pages, 1 figure, 1 table
>
> **摘要:** Social media platforms, while enabling global connectivity, have become hubs for the rapid spread of harmful content, including hate speech and fake narratives \cite{davidson2017automated, shu2017fake}. The Faux-Hate shared task focuses on detecting a specific phenomenon: the generation of hate speech driven by fake narratives, termed Faux-Hate. Participants are challenged to identify such instances in code-mixed Hindi-English social media text. This paper describes our system developed for the shared task, addressing two primary sub-tasks: (a) Binary Faux-Hate detection, involving fake and hate speech classification, and (b) Target and Severity prediction, categorizing the intended target and severity of hateful content. Our approach combines advanced natural language processing techniques with domain-specific pretraining to enhance performance across both tasks. The system achieved competitive results, demonstrating the efficacy of leveraging multi-task learning for this complex problem.
>
---
#### [new 017] MRG-R1: Reinforcement Learning for Clinically Aligned Medical Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向医学报告生成（MRG）任务，旨在解决现有方法仅模仿语言风格而缺乏临床正确性的问题。提出语义驱动的强化学习方法MRG-R1，采用组相对策略优化（GRPO）与基于关键发现相似度的报告级奖励（MCCS），提升临床准确性，并在IU X-Ray和MIMIC-CXR上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2512.16145v1](https://arxiv.org/pdf/2512.16145v1)**

> **作者:** Pengyu Wang; Shuchang Ye; Usman Naseem; Jinman Kim
>
> **备注:** 12 pages
>
> **摘要:** Medical report generation (MRG) aims to automatically derive radiology-style reports from medical images to aid in clinical decision-making. However, existing methods often generate text that mimics the linguistic style of radiologists but fails to guarantee clinical correctness, because they are trained on token-level objectives which focus on word-choice and sentence structure rather than actual medical accuracy. We propose a semantic-driven reinforcement learning (SRL) method for medical report generation, adopted on a large vision-language model (LVLM). SRL adopts Group Relative Policy Optimization (GRPO) to encourage clinical-correctness-guided learning beyond imitation of language style. Specifically, we optimise a report-level reward: a margin-based cosine similarity (MCCS) computed between key radiological findings extracted from generated and reference reports, thereby directly aligning clinical-label agreement and improving semantic correctness. A lightweight reasoning format constraint further guides the model to generate structured "thinking report" outputs. We evaluate Medical Report Generation with Sematic-driven Reinforment Learning (MRG-R1), on two datasets: IU X-Ray and MIMIC-CXR using clinical efficacy (CE) metrics. MRG-R1 achieves state-of-the-art performance with CE-F1 51.88 on IU X-Ray and 40.39 on MIMIC-CXR. We found that the label-semantic reinforcement is better than conventional token-level supervision. These results indicate that optimizing a clinically grounded, report-level reward rather than token overlap,meaningfully improves clinical correctness. This work is a prior to explore semantic-reinforcement in supervising medical correctness in medical Large vision-language model(Med-LVLM) training.
>
---
#### [new 018] LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding
- **分类: cs.CL**

- **简介: 该论文面向扩散大语言模型（dLLM）推理加速任务，旨在解决现有置信度驱动解码并行度低（仅1–3 tokens/forward）的问题。提出无需训练的LoPA算法，通过前瞻式并行探索最优Token填充顺序，并设计多设备分支并行推理系统，显著提升吞吐量与TPF。**

- **链接: [https://arxiv.org/pdf/2512.16229v1](https://arxiv.org/pdf/2512.16229v1)**

> **作者:** Chenkai Xu; Yijie Jin; Jiajun Li; Yi Tu; Guoping Long; Dandan Tu; Tianqi Hou; Junchi Yan; Zhijie Deng
>
> **摘要:** Diffusion Large Language Models (dLLMs) have demonstrated significant potential for high-speed inference. However, current confidence-driven decoding strategies are constrained by limited parallelism, typically achieving only 1--3 tokens per forward pass (TPF). In this work, we identify that the degree of parallelism during dLLM inference is highly sensitive to the Token Filling Order (TFO). Then, we introduce Lookahead PArallel Decoding LoPA, a training-free, plug-and-play algorithm, to identify a superior TFO and hence accelerate inference. LoPA concurrently explores distinct candidate TFOs via parallel branches, and selects the one with the highest potential for future parallelism based on branch confidence. We apply LoPA to the state-of-the-art D2F model and observe a substantial enhancement in decoding efficiency. Notably, LoPA increases the TPF of D2F-Dream to 10.1 on the GSM8K while maintaining performance superior to the Dream baseline. Furthermore, to facilitate this unprecedented degree of parallelism, we develop a specialized multi-device inference system featuring Branch Parallelism (BP), which achieves a single-sample throughput of 1073.9 tokens per second under multi-GPU deployment. The code is available at https://github.com/zhijie-group/LoPA.
>
---
#### [new 019] Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates
- **分类: cs.CL**

- **简介: 该论文属大模型能力增强任务，旨在解决数学推理能力提升中参数更新冗余、泛化受损的问题。提出“构建式电路放大”法，精准定位并仅更新与数学推理相关的关键稀疏子网络（电路），显著提升准确率（+11.4%），仅修改1.59%参数，且不损害其他能力。**

- **链接: [https://arxiv.org/pdf/2512.16914v1](https://arxiv.org/pdf/2512.16914v1)**

> **作者:** Nikhil Prakash; Donghao Ren; Dominik Moritz; Yannick Assogba
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Prior studies investigating the internal workings of LLMs have uncovered sparse subnetworks, often referred to as circuits, that are responsible for performing specific tasks. Additionally, it has been shown that model performance improvement through fine-tuning often results from the strengthening of existing circuits in the model. Taken together, these findings suggest the possibility of intervening directly on such circuits to make precise, task-targeted updates. Motivated by these findings, we propose a novel method called Constructive Circuit Amplification which identifies pivotal tokens from model reasoning traces as well as model components responsible for the desired task, and updates only those components. Applied to mathematical reasoning, it improves accuracy by up to +11.4% across multiple models while modifying as little as 1.59% of model components, with minimal impact on other abilities as measured by MMLU, TriviaQA, and TruthfulQA. These results demonstrate that targeted capabilities can be reliably enhanced by selectively updating a sparse set of model components.
>
---
#### [new 020] GinSign: Grounding Natural Language Into System Signatures for Temporal Logic Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GinSign框架，解决自然语言到时序逻辑（TL）翻译中的原子命题接地问题。现有方法依赖准确接地或精度低。GinSign将接地建模为分层分类任务（先谓词后参数），用轻量掩码语言模型实现高精度（95.5%逻辑等价），提升下游模型检测可靠性。**

- **链接: [https://arxiv.org/pdf/2512.16770v1](https://arxiv.org/pdf/2512.16770v1)**

> **作者:** William English; Chase Walker; Dominic Simon; Rickard Ewetz
>
> **摘要:** Natural language (NL) to temporal logic (TL) translation enables engineers to specify, verify, and enforce system behaviors without manually crafting formal specifications-an essential capability for building trustworthy autonomous systems. While existing NL-to-TL translation frameworks have demonstrated encouraging initial results, these systems either explicitly assume access to accurate atom grounding or suffer from low grounded translation accuracy. In this paper, we propose a framework for Grounding Natural Language Into System Signatures for Temporal Logic translation called GinSign. The framework introduces a grounding model that learns the abstract task of mapping NL spans onto a given system signature: given a lifted NL specification and a system signature $\mathcal{S}$, the classifier must assign each lifted atomic proposition to an element of the set of signature-defined atoms $\mathcal{P}$. We decompose the grounding task hierarchically- first predicting predicate labels, then selecting the appropriately typed constant arguments. Decomposing this task from a free-form generation problem into a structured classification problem permits the use of smaller masked language models and eliminates the reliance on expensive LLMs. Experiments across multiple domains show that frameworks which omit grounding tend to produce syntactically correct lifted LTL that is semantically nonequivalent to grounded target expressions, whereas our framework supports downstream model checking and achieves grounded logical-equivalence scores of $95.5\%$, a $1.4\times$ improvement over SOTA.
>
---
#### [new 021] TabReX : Tabular Referenceless eXplainable Evaluation
- **分类: cs.CL**

- **简介: 该论文提出TabReX，一种无参考、可解释的表格生成质量评估方法。针对现有指标忽略结构或依赖固定参考的问题，它将文本与表格转为知识图并对其对齐，输出结构与事实双维度可解释评分，并构建TabReX-Bench基准验证其鲁棒性与人类一致性。**

- **链接: [https://arxiv.org/pdf/2512.15907v1](https://arxiv.org/pdf/2512.15907v1)**

> **作者:** Tejas Anvekar; Juhna Park; Aparna Garimella; Vivek Gupta
>
> **摘要:** Evaluating the quality of tables generated by large language models (LLMs) remains an open challenge: existing metrics either flatten tables into text, ignoring structure, or rely on fixed references that limit generalization. We present TabReX, a reference-less, property-driven framework for evaluating tabular generation via graph-based reasoning. TabReX converts both source text and generated tables into canonical knowledge graphs, aligns them through an LLM-guided matching process, and computes interpretable, rubric-aware scores that quantify structural and factual fidelity. The resulting metric provides controllable trade-offs between sensitivity and specificity, yielding human-aligned judgments and cell-level error traces. To systematically asses metric robustness, we introduce TabReX-Bench, a large-scale benchmark spanning six domains and twelve planner-driven perturbation types across three difficulty tiers. Empirical results show that TabReX achieves the highest correlation with expert rankings, remains stable under harder perturbations, and enables fine-grained model-vs-prompt analysis establishing a new paradigm for trustworthy, explainable evaluation of structured generation systems.
>
---
#### [new 022] UM_FHS at the CLEF 2025 SimpleText Track: Comparing No-Context and Fine-Tune Approaches for GPT-4.1 Models in Sentence and Document-Level Text Simplification
- **分类: cs.CL**

- **简介: 该论文参加CLEF 2025 SimpleText任务，解决科学文本的句子与文档级简化问题。对比了GPT-4.1系列模型（gpt-4.1、mini、nano）的无上下文提示工程与微调两种方法，发现gpt-4.1-mini无上下文效果稳健，gpt-4.1-nano微调在文档级简化中表现突出。**

- **链接: [https://arxiv.org/pdf/2512.16541v1](https://arxiv.org/pdf/2512.16541v1)**

> **作者:** Primoz Kocbek; Gregor Stiglic
>
> **备注:** 10 pages, 3 tables. CLEF 2025 Working Notes, 9 to 12 September 2025, Madrid, Spain
>
> **摘要:** This work describes our submission to the CLEF 2025 SimpleText track Task 1, addressing both sentenceand document-level simplification of scientific texts. The methodology centered on using the gpt-4.1, gpt-4.1mini, and gpt-4.1-nano models from OpenAI. Two distinct approaches were compared: a no-context method relying on prompt engineering and a fine-tuned (FT) method across models. The gpt-4.1-mini model with no-context demonstrated robust performance at both levels of simplification, while the fine-tuned models showed mixed results, highlighting the complexities of simplifying text at different granularities, where gpt-4.1-nano-ft performance stands out at document-level simplification in one case.
>
---
#### [new 023] In-Context Algebra
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer在变量含义不固定（每序列符号映射不同群元素）下的代数推理任务，旨在揭示其内在符号推理机制。作者设计新任务与因果测试数据，发现模型学会三种机制：交换复制、单位元识别、闭包消去，表明其可发展出非几何的符号推理能力。**

- **链接: [https://arxiv.org/pdf/2512.16902v1](https://arxiv.org/pdf/2512.16902v1)**

> **作者:** Eric Todd; Jannik Brinkmann; Rohit Gandikota; David Bau
>
> **备注:** 28 pages, 18 figures. Code and data at https://algebra.baulab.info
>
> **摘要:** We investigate the mechanisms that arise when transformers are trained to solve arithmetic on sequences where tokens are variables whose meaning is determined only through their interactions. While prior work has found that transformers develop geometric embeddings that mirror algebraic structure, those previous findings emerge from settings where arithmetic-valued tokens have fixed meanings. We devise a new task in which the assignment of symbols to specific algebraic group elements varies from one sequence to another. Despite this challenging setup, transformers achieve near-perfect accuracy on the task and even generalize to unseen algebraic groups. We develop targeted data distributions to create causal tests of a set of hypothesized mechanisms, and we isolate three mechanisms models consistently learn: commutative copying where a dedicated head copies answers, identity element recognition that distinguishes identity-containing facts, and closure-based cancellation that tracks group membership to constrain valid answers. Complementary to the geometric representations found in fixed-symbol settings, our findings show that models develop symbolic reasoning mechanisms when trained to reason in-context with variables whose meanings are not fixed.
>
---
#### [new 024] Hacking Neural Evaluation Metrics with Single Hub Text
- **分类: cs.CL**

- **简介: 该论文属自然语言处理中的评估鲁棒性任务，旨在揭示神经文本评估指标（如COMET）的脆弱性。作者提出在离散空间中搜索单个“枢纽文本”，使其被错误地持续高评，发现该文本在多语言翻译任务中竟超越基线模型，暴露了指标可靠性问题。**

- **链接: [https://arxiv.org/pdf/2512.16323v1](https://arxiv.org/pdf/2512.16323v1)**

> **作者:** Hiroyuki Deguchi; Katsuki Chousa; Yusuke Sakai
>
> **摘要:** Strongly human-correlated evaluation metrics serve as an essential compass for the development and improvement of generation models and must be highly reliable and robust. Recent embedding-based neural text evaluation metrics, such as COMET for translation tasks, are widely used in both research and development fields. However, there is no guarantee that they yield reliable evaluation results due to the black-box nature of neural networks. To raise concerns about the reliability and safety of such metrics, we propose a method for finding a single adversarial text in the discrete space that is consistently evaluated as high-quality, regardless of the test cases, to identify the vulnerabilities in evaluation metrics. The single hub text found with our method achieved 79.1 COMET% and 67.8 COMET% in the WMT'24 English-to-Japanese (En--Ja) and English-to-German (En--De) translation tasks, respectively, outperforming translations generated individually for each source sentence by using M2M100, a general translation model. Furthermore, we also confirmed that the hub text found with our method generalizes across multiple language pairs such as Ja--En and De--En.
>
---
#### [new 025] What Do Prosody and Text Convey? Characterizing How Meaningful Information is Distributed Across Multiple Channels
- **分类: cs.CL**

- **简介: 该论文属多模态语义分析任务，旨在量化语音（尤其是韵律）与文本各自承载的语义信息。它提出信息论方法，用大模型估计音频/文本通道对讽刺、情绪、疑问等意义维度的互信息，发现韵律在讽刺和情绪识别中远超文本，而疑问识别则差异较小。**

- **链接: [https://arxiv.org/pdf/2512.16832v1](https://arxiv.org/pdf/2512.16832v1)**

> **作者:** Aditya Yadavalli; Tiago Pimentel; Tamar I Regev; Ethan Wilcox; Alex Warstadt
>
> **摘要:** Prosody -- the melody of speech -- conveys critical information often not captured by the words or text of a message. In this paper, we propose an information-theoretic approach to quantify how much information is expressed by prosody alone and not by text, and crucially, what that information is about. Our approach applies large speech and language models to estimate the mutual information between a particular dimension of an utterance's meaning (e.g., its emotion) and any of its communication channels (e.g., audio or text). We then use this approach to quantify how much information is conveyed by audio and text about sarcasm, emotion, and questionhood, using speech from television and podcasts. We find that for sarcasm and emotion the audio channel -- and by implication the prosodic channel -- transmits over an order of magnitude more information about these features than the text channel alone, at least when long-term context beyond the current sentence is unavailable. For questionhood, prosody provides comparatively less additional information. We conclude by outlining a program applying our approach to more dimensions of meaning, communication channels, and languages.
>
---
#### [new 026] Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出“拒绝引导”（Refusal Steering）方法，属大模型推理时可控对齐任务，旨在细粒度调控LLM对政治敏感话题的拒绝行为。无需微调，通过LLM-as-a-judge打分与正则化 steering 向量，实现拒绝行为的移除或诱导，兼顾安全与通用性能。**

- **链接: [https://arxiv.org/pdf/2512.16602v1](https://arxiv.org/pdf/2512.16602v1)**

> **作者:** Iker García-Ferrero; David Montero; Roman Orus
>
> **摘要:** We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
>
---
#### [new 027] Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation
- **分类: cs.CL**

- **简介: 该论文属医疗领域LLM可靠性任务，旨在解决生成内容幻觉问题。提出独立于LLM的细粒度事实核查模块（基于EHR数值与逻辑验证）和LoRA微调的领域专用摘要模型，在MIMIC-III上训练并验证，显著提升事实准确率与摘要质量。**

- **链接: [https://arxiv.org/pdf/2512.16189v1](https://arxiv.org/pdf/2512.16189v1)**

> **作者:** Musarrat Zeba; Abdullah Al Mamun; Kishoar Jahan Tithee; Debopom Sutradhar; Mohaimenul Azam Khan Raiaan; Saddam Mukta; Reem E. Mohamed; Md Rafiqul Islam; Yakub Sebastian; Mukhtar Hussain; Sami Azam
>
> **摘要:** In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
>
---
#### [new 028] Plain language adaptations of biomedical text using LLMs: Comparision of evaluation metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究用大语言模型（LLM）简化生物医学文本以提升健康素养，属文本简化任务。旨在解决专业术语难懂、公众理解困难问题。工作包括对比提示工程、双AI代理和微调三种方法，并用多维度定量与定性指标评估效果。**

- **链接: [https://arxiv.org/pdf/2512.16530v1](https://arxiv.org/pdf/2512.16530v1)**

> **作者:** Primoz Kocbek; Leon Kopitar; Gregor Stiglic
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** This study investigated the application of Large Language Models (LLMs) for simplifying biomedical texts to enhance health literacy. Using a public dataset, which included plain language adaptations of biomedical abstracts, we developed and evaluated several approaches, specifically a baseline approach using a prompt template, a two AI agent approach, and a fine-tuning approach. We selected OpenAI gpt-4o and gpt-4o mini models as baselines for further research. We evaluated our approaches with quantitative metrics, such as Flesch-Kincaid grade level, SMOG Index, SARI, and BERTScore, G-Eval, as well as with qualitative metric, more precisely 5-point Likert scales for simplicity, accuracy, completeness, brevity. Results showed a superior performance of gpt-4o-mini and an underperformance of FT approaches. G-Eval, a LLM based quantitative metric, showed promising results, ranking the approaches similarly as the qualitative metric.
>
---
#### [new 029] Examining the Utility of Self-disclosure Types for Modeling Annotators of Social Norms
- **分类: cs.CL**

- **简介: 该论文研究如何利用标注者自述信息（如人口统计、态度等）建模其社会规范判断偏好。任务是预测主观标注结果，旨在识别最有效的自述类型。作者对自述进行分类、消融实验与分析，发现人口统计信息最有效，理论驱动分类优于自动聚类，少量相关评论即足够，且样本多样性提升性能。**

- **链接: [https://arxiv.org/pdf/2512.16034v1](https://arxiv.org/pdf/2512.16034v1)**

> **作者:** Kieran Henderson; Kian Omoomi; Vasudha Varadarajan; Allison Lahnala; Charles Welch
>
> **摘要:** Recent work has explored the use of personal information in the form of persona sentences or self-disclosures to improve modeling of individual characteristics and prediction of annotator labels for subjective tasks. The volume of personal information has historically been restricted and thus little exploration has gone into understanding what kind of information is most informative for predicting annotator labels. In this work, we categorize self-disclosure sentences and use them to build annotator models for predicting judgments of social norms. We perform several ablations and analyses to examine the impact of the type of information on our ability to predict annotation patterns. We find that demographics are more impactful than attitudes, relationships, and experiences. Generally, theory-based approaches worked better than automatic clusters. Contrary to previous work, only a small number of related comments are needed. Lastly, having a more diverse sample of annotator self-disclosures leads to the best performance.
>
---
#### [new 030] LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformer Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属模型推理优化任务，旨在解决Transformer推理延迟高问题。提出LLMCache框架，通过层间语义相似性匹配复用中间激活，支持任意层、编解码器及模型，结合轻量指纹与自适应驱逐策略，在精度几乎无损下实现最高3.1倍加速。**

- **链接: [https://arxiv.org/pdf/2512.16843v1](https://arxiv.org/pdf/2512.16843v1)**

> **作者:** Harsh Vardhan Bansal
>
> **备注:** Accepted and presented at 13th IEEE International Conference on Intelligent Systems and Embedded Design (ISED-2025)
>
> **摘要:** Transformer-based language models have achieved remarkable performance across a wide range of tasks, yet their high inference latency poses a significant challenge for real-timeand large-scale deployment. While existing caching mechanisms,such as token-level key-value caches, offer speedups in autore-gressive decoding, they are limited in scope and applicability. In this paper, we present LLMCache, a novel layer-wise caching framework that accelerates transformer inference by reusing intermediate activations based on semantic similarity of input sequences. Unlike prior work, LLMCache is model-agnostic,operates across both encoder and decoder architectures, and supports caching at arbitrary transformer layers. We introduce a lightweight fingerprinting mechanism for matching seman-tically similar inputs and propose adaptive eviction strategies to manage cache staleness. Experiments on BERT and GPT-2 across SQuAD, WikiText-103, and OpenBookQA show up to 3.1 X speedup in inference time with <0.5% accuracy degradation. Our results highlight LLMCache as a practical and general-purpose solution for optimizing transformer inference in real-world applications
>
---
#### [new 031] Exploration of Augmentation Strategies in Multi-modal Retrieval-Augmented Generation for the Biomedical Domain: A Case Study Evaluating Question Answering in Glycobiology
- **分类: cs.CL**

- **简介: 该论文研究生物医学多模态RAG中的图文增强策略选择问题，聚焦糖生物学问答任务。通过构建120题基准，对比文本转换、OCR-free视觉检索等四种增强方式，评估不同模型与检索器组合效果，揭示模型规模与增强方法的适配规律。**

- **链接: [https://arxiv.org/pdf/2512.16802v1](https://arxiv.org/pdf/2512.16802v1)**

> **作者:** Primož Kocbek; Azra Frkatović-Hodžić; Dora Lalić; Vivian Hui; Gordan Lauc; Gregor Štiglic
>
> **备注:** Will be published in IEEE BigData 2025 proceedings. Contains 10 pages, 1 figure, 5 tables
>
> **摘要:** Multi-modal retrieval-augmented generation (MM-RAG) promises grounded biomedical QA, but it is unclear when to (i) convert figures/tables into text versus (ii) use optical character recognition (OCR)-free visual retrieval that returns page images and leaves interpretation to the generator. We study this trade-off in glycobiology, a visually dense domain. We built a benchmark of 120 multiple-choice questions (MCQs) from 25 papers, stratified by retrieval difficulty (easy text, medium figures/tables, hard cross-evidence). We implemented four augmentations-None, Text RAG, Multi-modal conversion, and late-interaction visual retrieval (ColPali)-using Docling parsing and Qdrant indexing. We evaluated mid-size open-source and frontier proprietary models (e.g., Gemma-3-27B-IT, GPT-4o family). Additional testing used the GPT-5 family and multiple visual retrievers (ColPali/ColQwen/ColFlor). Accuracy with Agresti-Coull 95% confidence intervals (CIs) was computed over 5 runs per configuration. With Gemma-3-27B-IT, Text and Multi-modal augmentation outperformed OCR-free retrieval (0.722-0.740 vs. 0.510 average accuracy). With GPT-4o, Multi-modal achieved 0.808, with Text 0.782 and ColPali 0.745 close behind; within-model differences were small. In follow-on experiments with the GPT-5 family, the best results with ColPali and ColFlor improved by ~2% to 0.828 in both cases. In general, across the GPT-5 family, ColPali, ColQwen, and ColFlor were statistically indistinguishable. GPT-5-nano trailed larger GPT-5 variants by roughly 8-10%. Pipeline choice is capacity-dependent: converting visuals to text lowers the reader burden and is more reliable for mid-size models, whereas OCR-free visual retrieval becomes competitive under frontier models. Among retrievers, ColFlor offers parity with heavier options at a smaller footprint, making it an efficient default when strong generators are available.
>
---
#### [new 032] Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属大语言模型推理增强任务，旨在解决LLM数学推理中过程错误（如计算错误、逻辑脆弱）问题。提出生成式对抗推理器（GAR），通过对抗强化学习联合训练推理器与判别器，利用分片审查与结构化反馈提供密集步级奖励，提升推理准确率与样本效率。**

- **链接: [https://arxiv.org/pdf/2512.16917v1](https://arxiv.org/pdf/2512.16917v1)**

> **作者:** Qihao Liu; Luoxin Ye; Wufei Ma; Yu-Cheng Chou; Alan Yuille
>
> **摘要:** Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
>
---
#### [new 033] From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining
- **分类: cs.SD; cs.CL; q-bio.NC**

- **简介: 该论文属脑机接口中的侵入式语音解码任务，旨在解决训练数据稀缺与跨日神经信号漂移问题。作者利用患者数天临床监测的颅内+音频长时数据预训练对比学习模型，大幅提升解码性能，并揭示需建模日间变异性。**

- **链接: [https://arxiv.org/pdf/2512.15830v1](https://arxiv.org/pdf/2512.15830v1)**

> **作者:** Linnea Evanson; Mingfang; Zhang; Hubert Banville; Saarang Panchavati; Pierre Bourdillon; Jean-Rémi King
>
> **备注:** Linnea Evanson* and Mingfang (Lucy) Zhang* are joint first authors. Pierre Bourdillon** and Jean-Rémi King** are joint last authors
>
> **摘要:** Decoding speech from brain activity has typically relied on limited neural recordings collected during short and highly controlled experiments. Here, we introduce a framework to leverage week-long intracranial and audio recordings from patients undergoing clinical monitoring, effectively increasing the training dataset size by over two orders of magnitude. With this pretraining, our contrastive learning model substantially outperforms models trained solely on classic experimental data, with gains that scale log-linearly with dataset size. Analysis of the learned representations reveals that, while brain activity represents speech features, its global structure largely drifts across days, highlighting the need for models that explicitly account for cross-day variability. Overall, our approach opens a scalable path toward decoding and modeling brain representations in both real-life and controlled task settings.
>
---
#### [new 034] LLaDA2.0: Scaling Up Diffusion Language Models to 100B
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出LLaDA2.0，将自回归大模型高效转化为离散扩散语言模型（dLLM），解决大模型推理慢、并行难问题。通过三阶段块级WSD训练、知识继承与MoE优化，构建16B和100B指令微调模型，支持并行解码，兼顾性能与效率。**

- **链接: [https://arxiv.org/pdf/2512.15745v1](https://arxiv.org/pdf/2512.15745v1)**

> **作者:** Tiwei Bie; Maosong Cao; Kun Chen; Lun Du; Mingliang Gong; Zhuochen Gong; Yanmei Gu; Jiaqi Hu; Zenan Huang; Zhenzhong Lan; Chengxi Li; Chongxuan Li; Jianguo Li; Zehuan Li; Huabin Liu; Ling Liu; Guoshan Lu; Xiaocheng Lu; Yuxin Ma; Jianfeng Tan; Lanning Wei; Ji-Rong Wen; Yipeng Xing; Xiaolu Zhang; Junbo Zhao; Da Zheng; Jun Zhou; Junlin Zhou; Zhanchao Zhou; Liwang Zhu; Yihong Zhuang
>
> **备注:** 19 pages
>
> **摘要:** This paper presents LLaDA2.0 -- a tuple of discrete diffusion large language models (dLLM) scaling up to 100B total parameters through systematic conversion from auto-regressive (AR) models -- establishing a new paradigm for frontier-scale deployment. Instead of costly training from scratch, LLaDA2.0 upholds knowledge inheritance, progressive adaption and efficiency-aware design principle, and seamless converts a pre-trained AR model into dLLM with a novel 3-phase block-level WSD based training scheme: progressive increasing block-size in block diffusion (warm-up), large-scale full-sequence diffusion (stable) and reverting back to compact-size block diffusion (decay). Along with post-training alignment with SFT and DPO, we obtain LLaDA2.0-mini (16B) and LLaDA2.0-flash (100B), two instruction-tuned Mixture-of-Experts (MoE) variants optimized for practical deployment. By preserving the advantages of parallel decoding, these models deliver superior performance and efficiency at the frontier scale. Both models were open-sourced.
>
---
#### [new 035] Evaluation of AI Ethics Tools in Language Models: A Developers' Perspective Case Stud
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属AI伦理评估任务，旨在解决AI伦理工具（AIETs）在语言模型开发中实用性不足的问题。作者调研213种AIETs，筛选4种，在葡萄牙语语言模型开发中开展开发者访谈（35小时），评估其对识别伦理问题及本地化风险（如 idiomatic expressions）的有效性。**

- **链接: [https://arxiv.org/pdf/2512.15791v1](https://arxiv.org/pdf/2512.15791v1)**

> **作者:** Jhessica Silva; Diego A. B. Moreira; Gabriel O. dos Santos; Alef Ferreira; Helena Maia; Sandra Avila; Helio Pedrini
>
> **备注:** 7 figures, 11 tables. Accepted for publication in AI and Ethics
>
> **摘要:** In Artificial Intelligence (AI), language models have gained significant importance due to the widespread adoption of systems capable of simulating realistic conversations with humans through text generation. Because of their impact on society, developing and deploying these language models must be done responsibly, with attention to their negative impacts and possible harms. In this scenario, the number of AI Ethics Tools (AIETs) publications has recently increased. These AIETs are designed to help developers, companies, governments, and other stakeholders establish trust, transparency, and responsibility with their technologies by bringing accepted values to guide AI's design, development, and use stages. However, many AIETs lack good documentation, examples of use, and proof of their effectiveness in practice. This paper presents a methodology for evaluating AIETs in language models. Our approach involved an extensive literature survey on 213 AIETs, and after applying inclusion and exclusion criteria, we selected four AIETs: Model Cards, ALTAI, FactSheets, and Harms Modeling. For evaluation, we applied AIETs to language models developed for the Portuguese language, conducting 35 hours of interviews with their developers. The evaluation considered the developers' perspective on the AIETs' use and quality in helping to identify ethical considerations about their model. The results suggest that the applied AIETs serve as a guide for formulating general ethical considerations about language models. However, we note that they do not address unique aspects of these models, such as idiomatic expressions. Additionally, these AIETs did not help to identify potential negative impacts of models for the Portuguese language.
>
---
#### [new 036] DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DataFlow框架，属数据准备与工作流自动化任务，旨在解决LLM时代数据管道碎片化、不可复现、缺乏语义支持等问题。工作包括：设计系统级抽象与PyTorch风格API；构建200+算子与6类通用管道；开发DataFlow-Agent实现NL到可执行流水线的自动合成；实验证明其显著提升下游LLM性能。**

- **链接: [https://arxiv.org/pdf/2512.16676v1](https://arxiv.org/pdf/2512.16676v1)**

> **作者:** Hao Liang; Xiaochen Ma; Zhou Liu; Zhen Hao Wong; Zhengyang Zhao; Zimo Meng; Runming He; Chengyu Shen; Qifeng Cai; Zhaoyang Han; Meiyi Qiang; Yalin Feng; Tianyi Bai; Zewei Pan; Ziyi Guo; Yizhen Jiang; Jingwen Deng; Qijie You; Peichao Lai; Tianyu Guo; Chi Hsu Tsai; Hengyi Feng; Rui Hu; Wenkai Yu; Junbo Niu; Bohan Zeng; Ruichuan An; Lu Ma; Jihao Huang; Yaowei Zheng; Conghui He; Linpeng Tang; Bin Cui; Weinan E; Wentao Zhang
>
> **摘要:** The rapidly growing demand for high-quality data in Large Language Models (LLMs) has intensified the need for scalable, reliable, and semantically rich data preparation pipelines. However, current practices remain dominated by ad-hoc scripts and loosely specified workflows, which lack principled abstractions, hinder reproducibility, and offer limited support for model-in-the-loop data generation. To address these challenges, we present DataFlow, a unified and extensible LLM-driven data preparation framework. DataFlow is designed with system-level abstractions that enable modular, reusable, and composable data transformations, and provides a PyTorch-style pipeline construction API for building debuggable and optimizable dataflows. The framework consists of nearly 200 reusable operators and six domain-general pipelines spanning text, mathematical reasoning, code, Text-to-SQL, agentic RAG, and large-scale knowledge extraction. To further improve usability, we introduce DataFlow-Agent, which automatically translates natural-language specifications into executable pipelines via operator synthesis, pipeline planning, and iterative verification. Across six representative use cases, DataFlow consistently improves downstream LLM performance. Our math, code, and text pipelines outperform curated human datasets and specialized synthetic baselines, achieving up to +3\% execution accuracy in Text-to-SQL over SynSQL, +7\% average improvements on code benchmarks, and 1--3 point gains on MATH, GSM8K, and AIME. Moreover, a unified 10K-sample dataset produced by DataFlow enables base models to surpass counterparts trained on 1M Infinity-Instruct data. These results demonstrate that DataFlow provides a practical and high-performance substrate for reliable, reproducible, and scalable LLM data preparation, and establishes a system-level foundation for future data-centric AI development.
>
---
#### [new 037] DP-Bench: A Benchmark for Evaluating Data Product Creation Systems
- **分类: cs.DB; cs.CL**

- **简介: 该论文提出DP-Bench——首个面向自动数据产品生成的基准，旨在解决缺乏统一评估标准的问题。工作包括构建基准（融合ELT与Text-to-SQL成果）、设计LLM基线方法，并开源数据集。**

- **链接: [https://arxiv.org/pdf/2512.15798v1](https://arxiv.org/pdf/2512.15798v1)**

> **作者:** Faisal Chowdhury; Sola Shirai; Sarthak Dash; Nandana Mihindukulasooriya; Horst Samulowitz
>
> **摘要:** A data product is created with the intention of solving a specific problem, addressing a specific business usecase or meeting a particular need, going beyond just serving data as a raw asset. Data products enable end users to gain greater insights about their data. Since it was first introduced over a decade ago, there has been considerable work, especially in industry, to create data products manually or semi-automatically. However, there exists hardly any benchmark to evaluate automatic data product creation. In this work, we present a benchmark, first of its kind, for this task. We call it DP-Bench. We describe how this benchmark was created by taking advantage of existing work in ELT (Extract-Load-Transform) and Text-to-SQL benchmarks. We also propose a number of LLM based approaches that can be considered as baselines for generating data products automatically. We make the DP-Bench and supplementary materials available in https://huggingface.co/datasets/ibm-research/dp-bench .
>
---
#### [new 038] Seeing Beyond Words: Self-Supervised Visual Learning for Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属多模态大模型视觉增强任务，旨在解决MLLM因依赖文本监督而视觉推理能力弱的问题。提出JARVIS框架，将I-JEPA自监督学习融入视觉-语言对齐流程，用冻结视觉模型作编码器，LLM早期层作预测器，仅从图像学习结构与语义规律，提升视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2512.15885v1](https://arxiv.org/pdf/2512.15885v1)**

> **作者:** Davide Caffagni; Sara Sarto; Marcella Cornia; Lorenzo Baraldi; Pier Luigi Dovesi; Shaghayegh Roohi; Mark Granroth-Wilding; Rita Cucchiara
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in connecting vision and language, yet their proficiency in fundamental visual reasoning tasks remains limited. This limitation can be attributed to the fact that MLLMs learn visual understanding primarily from textual descriptions, which constitute a subjective and inherently incomplete supervisory signal. Furthermore, the modest scale of multimodal instruction tuning compared to massive text-only pre-training leads MLLMs to overfit language priors while overlooking visual details. To address these issues, we introduce JARVIS, a JEPA-inspired framework for self-supervised visual enhancement in MLLMs. Specifically, we integrate the I-JEPA learning paradigm into the standard vision-language alignment pipeline of MLLMs training. Our approach leverages frozen vision foundation models as context and target encoders, while training the predictor, implemented as the early layers of an LLM, to learn structural and semantic regularities from images without relying exclusively on language supervision. Extensive experiments on standard MLLM benchmarks show that JARVIS consistently improves performance on vision-centric benchmarks across different LLM families, without degrading multimodal reasoning abilities. Our source code is publicly available at: https://github.com/aimagelab/JARVIS.
>
---
#### [new 039] A Systematic Analysis of Biases in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属偏见分析任务，旨在评估大语言模型在政治、意识形态、地缘联盟、语言和性别维度的潜在偏差。研究对4种主流LLM开展系统性实验，涵盖新闻摘要、立场分类、联合国投票模拟、多语言故事补全及价值观问卷响应，揭示其隐性倾向与非中立性。**

- **链接: [https://arxiv.org/pdf/2512.15792v1](https://arxiv.org/pdf/2512.15792v1)**

> **作者:** Xulang Zhang; Rui Mao; Erik Cambria
>
> **摘要:** Large language models (LLMs) have rapidly become indispensable tools for acquiring information and supporting human decision-making. However, ensuring that these models uphold fairness across varied contexts is critical to their safe and responsible deployment. In this study, we undertake a comprehensive examination of four widely adopted LLMs, probing their underlying biases and inclinations across the dimensions of politics, ideology, alliance, language, and gender. Through a series of carefully designed experiments, we investigate their political neutrality using news summarization, ideological biases through news stance classification, tendencies toward specific geopolitical alliances via United Nations voting patterns, language bias in the context of multilingual story completion, and gender-related affinities as revealed by responses to the World Values Survey. Results indicate that while the LLMs are aligned to be neutral and impartial, they still show biases and affinities of different types.
>
---
#### [new 040] Topic Modelling Black Box Optimization
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文将LDA主题数T的选择建模为离散黑箱优化任务，旨在以最少评估次数找到最优T。作者对比了GA、ES、PABBO和SABBO四类优化器，发现学习型 amortized 方法（尤其SABBO）显著更高效，仅需1次评估即可接近最优。**

- **链接: [https://arxiv.org/pdf/2512.16445v1](https://arxiv.org/pdf/2512.16445v1)**

> **作者:** Roman Akramov; Artem Khamatullin; Svetlana Glazyrina; Maksim Kryzhanovskiy; Roman Ischenko
>
> **摘要:** Choosing the number of topics $T$ in Latent Dirichlet Allocation (LDA) is a key design decision that strongly affects both the statistical fit and interpretability of topic models. In this work, we formulate the selection of $T$ as a discrete black-box optimization problem, where each function evaluation corresponds to training an LDA model and measuring its validation perplexity. Under a fixed evaluation budget, we compare four families of optimizers: two hand-designed evolutionary methods - Genetic Algorithm (GA) and Evolution Strategy (ES) - and two learned, amortized approaches, Preferential Amortized Black-Box Optimization (PABBO) and Sharpness-Aware Black-Box Optimization (SABBO). Our experiments show that, while GA, ES, PABBO, and SABBO eventually reach a similar band of final perplexity, the amortized optimizers are substantially more sample- and time-efficient. SABBO typically identifies a near-optimal topic number after essentially a single evaluation, and PABBO finds competitive configurations within a few evaluations, whereas GA and ES require almost the full budget to approach the same region.
>
---
#### [new 041] Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self Attention in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属模型压缩任务，旨在解决LLM中多头自注意力（MHSA）静态低秩近似灵活性差、计算开销大的问题。提出动态秩强化学习（DR-RL）框架，用RL实时选择最优秩，结合在线矩阵扰动理论实现增量更新，在保持精度前提下显著降低长序列FLOPs。**

- **链接: [https://arxiv.org/pdf/2512.15973v1](https://arxiv.org/pdf/2512.15973v1)**

> **作者:** Caner Erden
>
> **摘要:** We propose Dynamic Rank Reinforcement Learning (DR-RL), a novel framework that adaptively optimizes the low-rank factorization of Multi-Head Self-Attention (MHSA) in Large Language Models (LLMs) through the integration of reinforcement learning and online matrix perturbation theory. While traditional low-rank approximations often rely on static rank assumptions--limiting their flexibility across diverse input contexts--our method dynamically selects ranks based on real-time sequence dynamics, layer-specific sensitivities, and hardware constraints. The core innovation lies in an RL agent that formulates rank selection as a sequential policy optimization problem, where the reward function strictly balances attention fidelity against computational latency. Crucially, we employ online matrix perturbation bounds to enable incremental rank updates, thereby avoiding the prohibitive cost of full decomposition during inference. Furthermore, the integration of a lightweight Transformer-based policy network and batched Singular Value Decomposition (SVD) operations ensures scalable deployment on modern GPU architectures. Experiments demonstrate that DR-RL maintains downstream accuracy statistically equivalent to full-rank attention while significantly reducing Floating Point Operations (FLOPs), particularly in long-sequence regimes (L > 4096). This work bridges the gap between adaptive efficiency and theoretical rigor in MHSA, offering a principled, mathematically grounded alternative to heuristic rank reduction techniques in resource-constrained deep learning. Source code and experiment logs are available at: https://github.com/canererden/DR_RL_Project
>
---
#### [new 042] Adaptation of Agentic AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文属AI系统设计任务，旨在解决agentic AI适应性不足问题。提出统一框架，将适应分为代理端（执行/输出信号）与工具端（无监督/有监督）两类，厘清设计空间、权衡关系，并综述方法、挑战与方向。**

- **链接: [https://arxiv.org/pdf/2512.16301v1](https://arxiv.org/pdf/2512.16301v1)**

> **作者:** Pengcheng Jiang; Jiacheng Lin; Zhiyi Shi; Zifeng Wang; Luxi He; Yichen Wu; Ming Zhong; Peiyang Song; Qizheng Zhang; Heng Wang; Xueqiang Xu; Hanwen Xu; Pengrui Han; Dylan Zhang; Jiashuo Sun; Chaoqi Yang; Kun Qian; Tian Wang; Changran Hu; Manling Li; Quanzheng Li; Hao Peng; Sheng Wang; Jingbo Shang; Chao Zhang; Jiaxuan You; Liyuan Liu; Pan Lu; Yu Zhang; Heng Ji; Yejin Choi; Dawn Song; Jimeng Sun; Jiawei Han
>
> **摘要:** Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems.
>
---
#### [new 043] DSO: Direct Steering Optimization for Bias Mitigation
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属模型公平性与可控推理任务，旨在解决生成模型（VLMs/LLMs）在推理时因人口统计属性引发的偏差问题。提出直接转向优化（DSO）方法，用强化学习学习线性激活转向变换，在推理时可调地平衡去偏与性能。**

- **链接: [https://arxiv.org/pdf/2512.15926v1](https://arxiv.org/pdf/2512.15926v1)**

> **作者:** Lucas Monteiro Paes; Nivedha Sivakumar; Yinong Oliver Wang; Masha Fedzechkina Donaldson; Luca Zappella; Nicholas Apostoloff
>
> **摘要:** Generative models are often deployed to make decisions on behalf of users, such as vision-language models (VLMs) identifying which person in a room is a doctor to help visually impaired individuals. Yet, VLM decisions are influenced by the perceived demographic attributes of people in the input, which can lead to biased outcomes like failing to identify women as doctors. Moreover, when reducing bias leads to performance loss, users may have varying needs for balancing bias mitigation with overall model capabilities, highlighting the demand for methods that enable controllable bias reduction during inference. Activation steering is a popular approach for inference-time controllability that has shown potential in inducing safer behavior in large language models (LLMs). However, we observe that current steering methods struggle to correct biases, where equiprobable outcomes across demographic groups are required. To address this, we propose Direct Steering Optimization (DSO) which uses reinforcement learning to find linear transformations for steering activations, tailored to mitigate bias while maintaining control over model performance. We demonstrate that DSO achieves state-of-the-art trade-off between fairness and capabilities on both VLMs and LLMs, while offering practitioners inference-time control over the trade-off. Overall, our work highlights the benefit of designing steering strategies that are directly optimized to control model behavior, providing more effective bias intervention than methods that rely on pre-defined heuristics for controllability.
>
---
#### [new 044] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于大语言模型水印防护任务，旨在解决现有水印方法仅防改写攻击、忽视“寄生式”伪造攻击的问题。作者提出DualGuard算法，首创双流自适应水印机制，可同时防御改写与伪造攻击，并支持攻击溯源，兼顾检测鲁棒性与文本质量。**

- **链接: [https://arxiv.org/pdf/2512.16182v1](https://arxiv.org/pdf/2512.16182v1)**

> **作者:** Hao Li; Yubing Ren; Yanan Cao; Yingjie Li; Fang Fang; Shi Wang; Li Guo
>
> **摘要:** With the rapid development of cloud-based services, large language models (LLMs) have become increasingly accessible through various web platforms. However, this accessibility has also led to growing risks of model abuse. LLM watermarking has emerged as an effective approach to mitigate such misuse and protect intellectual property. Existing watermarking algorithms, however, primarily focus on defending against paraphrase attacks while overlooking piggyback spoofing attacks, which can inject harmful content, compromise watermark reliability, and undermine trust in attribution. To address this limitation, we propose DualGuard, the first watermarking algorithm capable of defending against both paraphrase and spoofing attacks. DualGuard employs the adaptive dual-stream watermarking mechanism, in which two complementary watermark signals are dynamically injected based on the semantic content. This design enables DualGuard not only to detect but also to trace spoofing attacks, thereby ensuring reliable and trustworthy watermark detection. Extensive experiments conducted across multiple datasets and language models demonstrate that DualGuard achieves excellent detectability, robustness, traceability, and text quality, effectively advancing the state of LLM watermarking for real-world applications.
>
---
#### [new 045] From Essence to Defense: Adaptive Semantic-aware Watermarking for Embedding-as-a-Service Copyright Protection
- **分类: cs.CR; cs.CL**

- **简介: 该论文面向Embeddings-as-a-Service（EaaS）的版权保护任务，解决现有水印方法忽略语义导致危害性大、隐蔽性差的问题。提出语义感知自适应水印框架SemMark，利用局部敏感哈希划分语义空间、LOF自适应调权，并设计新攻击场景验证其鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.16439v1](https://arxiv.org/pdf/2512.16439v1)**

> **作者:** Hao Li; Yubing Ren; Yanan Cao; Yingjie Li; Fang Fang; Xuebin Wang
>
> **摘要:** Benefiting from the superior capabilities of large language models in natural language understanding and generation, Embeddings-as-a-Service (EaaS) has emerged as a successful commercial paradigm on the web platform. However, prior studies have revealed that EaaS is vulnerable to imitation attacks. Existing methods protect the intellectual property of EaaS through watermarking techniques, but they all ignore the most important properties of embedding: semantics, resulting in limited harmlessness and stealthiness. To this end, we propose SemMark, a novel semantic-based watermarking paradigm for EaaS copyright protection. SemMark employs locality-sensitive hashing to partition the semantic space and inject semantic-aware watermarks into specific regions, ensuring that the watermark signals remain imperceptible and diverse. In addition, we introduce the adaptive watermark weight mechanism based on the local outlier factor to preserve the original embedding distribution. Furthermore, we propose Detect-Sampling and Dimensionality-Reduction attacks and construct four scenarios to evaluate the watermarking method. Extensive experiments are conducted on four popular NLP datasets, and SemMark achieves superior verifiability, diversity, stealthiness, and harmlessness.
>
---
#### [new 046] Explainable Ethical Assessment on Human Behaviors by Generating Conflicting Social Norms
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属可解释伦理评估任务，旨在解决AI行为评估缺乏可解释性问题。提出ClarityEthic方法，通过生成冲突社会规范（如勇敢vs自保）增强语言模型的道德推理与解释能力，采用对比学习提升评估准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2512.15793v1](https://arxiv.org/pdf/2512.15793v1)**

> **作者:** Yuxi Sun; Wei Gao; Hongzhan Lin; Jing Ma; Wenxuan Zhang
>
> **备注:** Acceppt by Asia-Pacific Chapter of the Association for Computational Linguistics (2025)
>
> **摘要:** Human behaviors are often guided or constrained by social norms, which are defined as shared, commonsense rules. For example, underlying an action ``\textit{report a witnessed crime}" are social norms that inform our conduct, such as ``\textit{It is expected to be brave to report crimes}''. Current AI systems that assess valence (i.e., support or oppose) of human actions by leveraging large-scale data training not grounded on explicit norms may be difficult to explain, and thus untrustworthy. Emulating human assessors by considering social norms can help AI models better understand and predict valence. While multiple norms come into play, conflicting norms can create tension and directly influence human behavior. For example, when deciding whether to ``\textit{report a witnessed crime}'', one may balance \textit{bravery} against \textit{self-protection}. In this paper, we introduce \textit{ClarityEthic}, a novel ethical assessment approach, to enhance valence prediction and explanation by generating conflicting social norms behind human actions, which strengthens the moral reasoning capabilities of language models by using a contrastive learning strategy. Extensive experiments demonstrate that our method outperforms strong baseline approaches, and human evaluations confirm that the generated social norms provide plausible explanations for the assessment of human behaviors.
>
---
#### [new 047] D3G: Diverse Demographic Data Generation Increases Zero-Shot Image Classification Accuracy within Multimodal Models
- **分类: cs.LG; cs.CL; cs.CV; cs.CY**

- **简介: 该论文属零-shot图像分类任务，旨在缓解多模态模型（如CLIP）因训练数据人口统计失衡导致的偏差与性能下降。提出无需训练的D3G方法：在推理时用Stable Diffusion XL生成多样化人口统计数据，提升准确率并减少 demographic bias。**

- **链接: [https://arxiv.org/pdf/2512.15747v1](https://arxiv.org/pdf/2512.15747v1)**

> **作者:** Javon Hickmon
>
> **摘要:** Image classification is a task essential for machine perception to achieve human-level image understanding. Multimodal models such as CLIP have been able to perform well on this task by learning semantic similarities across vision and language; however, despite these advances, image classification is still a challenging task. Models with low capacity often suffer from underfitting and thus underperform on fine-grained image classification. Along with this, it is important to ensure high-quality data with rich cross-modal representations of each class, which is often difficult to generate. When datasets do not enforce balanced demographics, the predictions will be biased toward the more represented class, while others will be neglected. We focus on how these issues can lead to harmful bias for zero-shot image classification, and explore how to combat these issues in demographic bias. We propose Diverse Demographic Data Generation (D3G), a training-free, zero-shot method of boosting classification accuracy while reducing demographic bias in pre-trained multimodal models. With this method, we utilize CLIP as our base multimodal model and Stable Diffusion XL as our generative model. We demonstrate that providing diverse demographic data at inference time improves performance for these models, and explore the impact of individual demographics on the resulting accuracy metric.
>
---
#### [new 048] Auto-Tuning Safety Guardrails for Black-Box Large Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属安全对齐任务，解决黑盒LLM部署中手工调优安全守卫（如系统提示、内容过滤器）脆弱难复现的问题。作者将守卫设计建模为超参优化问题，在冻结模型下用Optuna自动搜索最优提示与过滤组合，在多个安全基准上显著提升效果并大幅降低调优成本。**

- **链接: [https://arxiv.org/pdf/2512.15782v1](https://arxiv.org/pdf/2512.15782v1)**

> **作者:** Perry Abdulkadir
>
> **备注:** 8 pages, 7 figures, 1 table. Work completed as part of the M.S. in Artificial Intelligence at the University of St. Thomas using publicly available models and datasets; all views and any errors are the author's own
>
> **摘要:** Large language models (LLMs) are increasingly deployed behind safety guardrails such as system prompts and content filters, especially in settings where product teams cannot modify model weights. In practice these guardrails are typically hand-tuned, brittle, and difficult to reproduce. This paper studies a simple but practical alternative: treat safety guardrail design itself as a hyperparameter optimization problem over a frozen base model. Concretely, I wrap Mistral-7B-Instruct with modular jailbreak and malware system prompts plus a ModernBERT-based harmfulness classifier, then evaluate candidate configurations on three public benchmarks covering malware generation, classic jailbreak prompts, and benign user queries. Each configuration is scored using malware and jailbreak attack success rate, benign harmful-response rate, and end-to-end latency. A 48-point grid search over prompt combinations and filter modes establishes a baseline. I then run a black-box Optuna study over the same space and show that it reliably rediscovers the best grid configurations while requiring an order of magnitude fewer evaluations and roughly 8x less wall-clock time. The results suggest that viewing safety guardrails as tunable hyperparameters is a feasible way to harden black-box LLM deployments under compute and time constraints.
>
---
#### [new 049] Impacts of Racial Bias in Historical Training Data for News AI
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属AI伦理与新闻计算交叉任务，旨在揭示历史训练数据中的种族偏见如何影响新闻AI模型。作者以NYT语料训练的多标签分类器为案例，通过可解释AI方法分析“blacks”标签的偏差行为，发现其误泛化为“种族主义检测器”，却在新冠反亚裔事件和BLM报道中失效，警示新闻业需审慎应用AI工具。**

- **链接: [https://arxiv.org/pdf/2512.16901v1](https://arxiv.org/pdf/2512.16901v1)**

> **作者:** Rahul Bhargava; Malene Hornstrup Jespersen; Emily Boardman Ndulue; Vivica Dsouza
>
> **摘要:** AI technologies have rapidly moved into business and research applications that involve large text corpora, including computational journalism research and newsroom settings. These models, trained on extant data from various sources, can be conceptualized as historical artifacts that encode decades-old attitudes and stereotypes. This paper investigates one such example trained on the broadly-used New York Times Annotated Corpus to create a multi-label classifier. Our use in research settings surfaced the concerning "blacks" thematic topic label. Through quantitative and qualitative means we investigate this label's use in the training corpus, what concepts it might be encoding in the trained classifier, and how those concepts impact our model use. Via the application of explainable AI methods, we find that the "blacks" label operates partially as a general "racism detector" across some minoritized groups. However, it performs poorly against expectations on modern examples such as COVID-19 era anti-Asian hate stories, and reporting on the Black Lives Matter movement. This case study of interrogating embedded biases in a model reveals how similar applications in newsroom settings can lead to unexpected outputs that could impact a wide variety of potential uses of any large language model-story discovery, audience targeting, summarization, etc. The fundamental tension this exposes for newsrooms is how to adopt AI-enabled workflow tools while reducing the risk of reproducing historical biases in news coverage.
>
---
#### [new 050] Needle in the Web: A Benchmark for Retrieving Targeted Web Pages in the Wild
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出“Needle in the Web”基准，面向模糊探索式网页检索任务，解决现有基准忽视语义模糊、多义性查询的问题。作者构建663个跨领域难题，评估LLM及搜索代理在真实网页中定位目标页面的能力，发现当前系统普遍表现不佳。**

- **链接: [https://arxiv.org/pdf/2512.16553v1](https://arxiv.org/pdf/2512.16553v1)**

> **作者:** Yumeng Wang; Tianyu Fan; Lingrui Xu; Chao Huang
>
> **备注:** Data and code are available at https://github.com/Tango-Whiskyman/Needle_in_the_Web
>
> **摘要:** Large Language Models (LLMs) have evolved from simple chatbots into sophisticated agents capable of automating complex real-world tasks, where browsing and reasoning over live web content is key to assessing retrieval and cognitive skills. Existing benchmarks like BrowseComp and xBench-DeepSearch emphasize complex reasoning searches requiring multi-hop synthesis but neglect Fuzzy Exploratory Search, namely queries that are vague and multifaceted, where users seek the most relevant webpage rather than a single factual answer. To address this gap, we introduce Needle in the Web, a novel benchmark specifically designed to evaluate modern search agents and LLM-based systems on their ability to retrieve and reason over real-world web content in response to ambiguous, exploratory queries under varying levels of difficulty. Needle in the Web comprises 663 questions spanning seven distinct domains. To ensure high query quality and answer uniqueness, we employ a flexible methodology that reliably generates queries of controllable difficulty based on factual claims of web contents. We benchmark three leading LLMs and three agent-based search systems on Needle in the Web, finding that most models struggle: many achieve below 35% accuracy, and none consistently excel across domains or difficulty levels. These findings reveal that Needle in the Web presents a significant challenge for current search systems and highlights the open problem of effective fuzzy retrieval under semantic ambiguity.
>
---
#### [new 051] Cross-Language Bias Examination in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属多语言偏见评估任务，旨在解决大语言模型（LLM）跨语言偏见检测缺失问题。作者构建了融合显式（BBQ）与隐式（IAT式提示）的多语言偏见评测框架，覆盖英、中、阿、法、西五种语言，揭示了语言间偏见强度与类型差异，为公平多语言LLM提供方法基础。**

- **链接: [https://arxiv.org/pdf/2512.16029v1](https://arxiv.org/pdf/2512.16029v1)**

> **作者:** Yuxuan Liang; Marwa Mahmoud
>
> **摘要:** This study introduces an innovative multilingual bias evaluation framework for assessing bias in Large Language Models, combining explicit bias assessment through the BBQ benchmark with implicit bias measurement using a prompt-based Implicit Association Test. By translating the prompts and word list into five target languages, English, Chinese, Arabic, French, and Spanish, we directly compare different types of bias across languages. The results reveal substantial gaps in bias across languages used in LLMs. For example, Arabic and Spanish consistently show higher levels of stereotype bias, while Chinese and English exhibit lower levels of bias. We also identify contrasting patterns across bias types. Age shows the lowest explicit bias but the highest implicit bias, emphasizing the importance of detecting implicit biases that are undetectable with standard benchmarks. These findings indicate that LLMs vary significantly across languages and bias dimensions. This study fills a key research gap by providing a comprehensive methodology for cross-lingual bias analysis. Ultimately, our work establishes a foundation for the development of equitable multilingual LLMs, ensuring fairness and effectiveness across diverse languages and cultures.
>
---
#### [new 052] Exploration v.s. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究RLVR（带可验证奖励的强化学习）中探索-利用权衡问题，聚焦于“虚假奖励”与“熵最小化”看似矛盾却均提升LLM数学推理性能的现象。通过分析剪辑偏差、策略熵与奖励错位机制，揭示虚假奖励如何降低熵并提升性能，提出理论解释并指导更优RLVR训练。**

- **链接: [https://arxiv.org/pdf/2512.16912v1](https://arxiv.org/pdf/2512.16912v1)**

> **作者:** Peter Chen; Xiaopeng Li; Ziniu Li; Wotao Yin; Xi Chen; Tianyi Lin
>
> **备注:** 35 pages
>
> **摘要:** This paper examines the exploration-exploitation trade-off in reinforcement learning with verifiable rewards (RLVR), a framework for improving the reasoning of Large Language Models (LLMs). Recent studies suggest that RLVR can elicit strong mathematical reasoning in LLMs through two seemingly paradoxical mechanisms: spurious rewards, which suppress exploitation by rewarding outcomes unrelated to the ground truth, and entropy minimization, which suppresses exploration by pushing the model toward more confident and deterministic outputs, highlighting a puzzling dynamic: both discouraging exploitation and discouraging exploration improve reasoning performance, yet the underlying principles that reconcile these effects remain poorly understood. We focus on two fundamental questions: (i) how policy entropy relates to performance, and (ii) whether spurious rewards yield gains, potentially through the interplay of clipping bias and model contamination. Our results show that clipping bias under spurious rewards reduces policy entropy, leading to more confident and deterministic outputs, while entropy minimization alone is insufficient for improvement. We further propose a reward-misalignment model explaining why spurious rewards can enhance performance beyond contaminated settings. Our findings clarify the mechanisms behind spurious-reward benefits and provide principles for more effective RLVR training.
>
---
#### [new 053] Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究自主智能体中“工具编排隐私风险”（TOP-R）问题，即多工具协同推理意外泄露敏感信息。工作包括：提出形式化框架、构建TOP-Bench基准与H-Score指标、评估8模型发现高泄漏率（90.24%），并提出PEP方法有效缓解风险。**

- **链接: [https://arxiv.org/pdf/2512.16310v1](https://arxiv.org/pdf/2512.16310v1)**

> **作者:** Yuxuan Qiao; Dongqin Liu; Hongchang Yang; Wei Zhou; Songlin Hu
>
> **摘要:** Driven by Large Language Models, the single-agent, multi-tool architecture has become a popular paradigm for autonomous agents due to its simplicity and effectiveness. However, this architecture also introduces a new and severe privacy risk, which we term Tools Orchestration Privacy Risk (TOP-R), where an agent, to achieve a benign user goal, autonomously aggregates information fragments across multiple tools and leverages its reasoning capabilities to synthesize unexpected sensitive information. We provide the first systematic study of this risk. First, we establish a formal framework, attributing the risk's root cause to the agent's misaligned objective function: an overoptimization for helpfulness while neglecting privacy awareness. Second, we construct TOP-Bench, comprising paired leakage and benign scenarios, to comprehensively evaluate this risk. To quantify the trade-off between safety and robustness, we introduce the H-Score as a holistic metric. The evaluation results reveal that TOP-R is a severe risk: the average Risk Leakage Rate (RLR) of eight representative models reaches 90.24%, while the average H-Score is merely 0.167, with no model exceeding 0.3. Finally, we propose the Privacy Enhancement Principle (PEP) method, which effectively mitigates TOP-R, reducing the Risk Leakage Rate to 46.58% and significantly improving the H-Score to 0.624. Our work reveals both a new class of risk and inherent structural limitations in current agent architectures, while also offering feasible mitigation strategies.
>
---
#### [new 054] QuadSentinel: Sequent Safety for Machine-Checkable Control in Multi-agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出QuadSentinel，一种基于四代理协同的在线安全守卫框架，旨在解决多智能体系统中自然语言安全策略难以机器验证与可靠执行的问题。它将策略形式化为可判定的逻辑规则，通过状态跟踪、策略验证、威胁监控和仲裁机制实现实时、低开销、高精度的安全控制。**

- **链接: [https://arxiv.org/pdf/2512.16279v1](https://arxiv.org/pdf/2512.16279v1)**

> **作者:** Yiliu Yang; Yilei Jiang; Qunzhong Wang; Yingshui Tan; Xiaoyong Zhu; Sherman S. M. Chow; Bo Zheng; Xiangyu Yue
>
> **备注:** Preprint
>
> **摘要:** Safety risks arise as large language model-based agents solve complex tasks with tools, multi-step plans, and inter-agent messages. However, deployer-written policies in natural language are ambiguous and context dependent, so they map poorly to machine-checkable rules, and runtime enforcement is unreliable. Expressing safety policies as sequents, we propose \textsc{QuadSentinel}, a four-agent guard (state tracker, policy verifier, threat watcher, and referee) that compiles these policies into machine-checkable rules built from predicates over observable state and enforces them online. Referee logic plus an efficient top-$k$ predicate updater keeps costs low by prioritizing checks and resolving conflicts hierarchically. Measured on ST-WebAgentBench (ICML CUA~'25) and AgentHarm (ICLR~'25), \textsc{QuadSentinel} improves guardrail accuracy and rule recall while reducing false positives. Against single-agent baselines such as ShieldAgent (ICML~'25), it yields better overall safety control. Near-term deployments can adopt this pattern without modifying core agents by keeping policies separate and machine-checkable. Our code will be made publicly available at https://github.com/yyiliu/QuadSentinel.
>
---
#### [new 055] Science Consultant Agent
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出“Science Consultant Agent”，属AI辅助决策任务，旨在解决从业者难以选择最优AI建模策略的问题。工作包括设计四模块系统：问卷采集需求、智能填充、文献驱动推荐、原型生成，支持产品、开发与研究人员快速落地AI方案。**

- **链接: [https://arxiv.org/pdf/2512.16171v1](https://arxiv.org/pdf/2512.16171v1)**

> **作者:** Karthikeyan K; Philip Wu; Xin Tang; Alexandre Alves
>
> **摘要:** The Science Consultant Agent is a web-based Artificial Intelligence (AI) tool that helps practitioners select and implement the most effective modeling strategy for AI-based solutions. It operates through four core components: Questionnaire, Smart Fill, Research-Guided Recommendation, and Prototype Builder. By combining structured questionnaires, literature-backed solution recommendations, and prototype generation, the Science Consultant Agent accelerates development for everyone from Product Managers and Software Developers to Researchers. The full pipeline is illustrated in Figure 1.
>
---
#### [new 056] Value Lens: Using Large Language Models to Understand Human Values
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文提出Value Lens模型，属自然语言理解中的价值观识别任务，旨在解决AI决策与人类价值观对齐问题。工作包括：一阶段用LLM构建形式化价值理论并由专家验证；二阶段用双LLM协同检测文本中价值观，一者识别、一者审查。**

- **链接: [https://arxiv.org/pdf/2512.15722v1](https://arxiv.org/pdf/2512.15722v1)**

> **作者:** Eduardo de la Cruz Fernández; Marcelo Karanik; Sascha Ossowski
>
> **备注:** 4 pages. 2 figures. Published in ECAI 2025, Frontiers in Artificial Intelligence and Applications, Volume 413, pages 5175-5178
>
> **摘要:** The autonomous decision-making process, which is increasingly applied to computer systems, requires that the choices made by these systems align with human values. In this context, systems must assess how well their decisions reflect human values. To achieve this, it is essential to identify whether each available action promotes or undermines these values. This article presents Value Lens, a text-based model designed to detect human values using generative artificial intelligence, specifically Large Language Models (LLMs). The proposed model operates in two stages: the first aims to formulate a formal theory of values, while the second focuses on identifying these values within a given text. In the first stage, an LLM generates a description based on the established theory of values, which experts then verify. In the second stage, a pair of LLMs is employed: one LLM detects the presence of values, and the second acts as a critic and reviewer of the detection process. The results indicate that Value Lens performs comparably to, and even exceeds, the effectiveness of other models that apply different methods for similar tasks.
>
---
#### [new 057] How Good is Post-Hoc Watermarking With Language Model Rephrasing?
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究后验水印（post-hoc watermarking），即用大模型重写已有文本并嵌入可检测水印，以追踪版权内容或训练/检索使用。它探索计算资源分配对检测率与语义保真度的权衡，发现Gumbel-max与束搜索有效，但小模型在代码等可验证文本上更优。**

- **链接: [https://arxiv.org/pdf/2512.16904v1](https://arxiv.org/pdf/2512.16904v1)**

> **作者:** Pierre Fernandez; Tom Sander; Hady Elsahar; Hongyan Chang; Tomáš Souček; Valeriu Lacatusu; Tuan Tran; Sylvestre-Alvise Rebuffi; Alexandre Mourachko
>
> **备注:** Code at https://github.com/facebookresearch/textseal
>
> **摘要:** Generation-time text watermarking embeds statistical signals into text for traceability of AI-generated content. We explore *post-hoc watermarking* where an LLM rewrites existing text while applying generation-time watermarking, to protect copyrighted documents, or detect their use in training or RAG via watermark radioactivity. Unlike generation-time approaches, which is constrained by how LLMs are served, this setting offers additional degrees of freedom for both generation and detection. We investigate how allocating compute (through larger rephrasing models, beam search, multi-candidate generation, or entropy filtering at detection) affects the quality-detectability trade-off. Our strategies achieve strong detectability and semantic fidelity on open-ended text such as books. Among our findings, the simple Gumbel-max scheme surprisingly outperforms more recent alternatives under nucleus sampling, and most methods benefit significantly from beam search. However, most approaches struggle when watermarking verifiable text such as code, where we counterintuitively find that smaller models outperform larger ones. This study reveals both the potential and limitations of post-hoc watermarking, laying groundwork for practical applications and future research.
>
---
#### [new 058] ContextLeak: Auditing Leakage in Private In-Context Learning Methods
- **分类: cs.CR; cs.CL**

- **简介: 该论文属隐私审计任务，旨在评估私有上下文学习（ICL）方法的信息泄露风险。作者提出ContextLeak框架，通过金丝雀插入与定向查询实证测量最坏情况下的泄漏，并验证其与理论隐私预算ε的相关性，揭示现有方法隐私-效用权衡不佳的问题。**

- **链接: [https://arxiv.org/pdf/2512.16059v1](https://arxiv.org/pdf/2512.16059v1)**

> **作者:** Jacob Choi; Shuying Cao; Xingjian Dong; Wang Bill Zhu; Robin Jia; Sai Praneeth Karimireddy
>
> **摘要:** In-Context Learning (ICL) has become a standard technique for adapting Large Language Models (LLMs) to specialized tasks by supplying task-specific exemplars within the prompt. However, when these exemplars contain sensitive information, reliable privacy-preserving mechanisms are essential to prevent unintended leakage through model outputs. Many privacy-preserving methods are proposed to protect the information leakage in the context, but there are less efforts on how to audit those methods. We introduce ContextLeak, the first framework to empirically measure the worst-case information leakage in ICL. ContextLeak uses canary insertion, embedding uniquely identifiable tokens in exemplars and crafting targeted queries to detect their presence. We apply ContextLeak across a range of private ICL techniques, both heuristic such as prompt-based defenses and those with theoretical guarantees such as Embedding Space Aggregation and Report Noisy Max. We find that ContextLeak tightly correlates with the theoretical privacy budget ($ε$) and reliably detects leakage. Our results further reveal that existing methods often strike poor privacy-utility trade-offs, either leaking sensitive information or severely degrading performance.
>
---
## 更新

#### [replaced 001] Evaluating Large Language Models in Crisis Detection: A Real-World Benchmark from Psychological Support Hotlines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属危机检测任务，旨在评估大语言模型（LLMs）在心理热线文本中识别情绪、自杀意念、自杀计划与风险的能力。作者构建真实世界基准PsyCrisisBench（540条标注热线对话），评测64个LLM，发现其在多项任务上媲美甚至超越人工，尤以细调小模型表现突出。**

- **链接: [https://arxiv.org/pdf/2506.01329v2](https://arxiv.org/pdf/2506.01329v2)**

> **作者:** Guifeng Deng; Shuyin Rao; Tianyu Lin; Anlu Dai; Pan Wang; Junyi Xie; Haidong Song; Ke Zhao; Dongwu Xu; Zhengdong Cheng; Tao Li; Haiteng Jiang
>
> **备注:** Preprint. Submitted to IEEE Journal of Biomedical and Health Informatics (under review)
>
> **摘要:** Psychological support hotlines serve as critical lifelines for crisis intervention but encounter significant challenges due to rising demand and limited resources. Large language models (LLMs) offer potential support in crisis assessments, yet their effectiveness in emotionally sensitive, real-world clinical settings remains underexplored. We introduce PsyCrisisBench, a comprehensive benchmark of 540 annotated transcripts from the Hangzhou Psychological Assistance Hotline, assessing four key tasks: mood status recognition, suicidal ideation detection, suicide plan identification, and risk assessment. 64 LLMs across 15 model families (including closed-source such as GPT, Claude, Gemini and open-source such as Llama, Qwen, DeepSeek) were evaluated using zero-shot, few-shot, and fine-tuning paradigms. LLMs showed strong results in suicidal ideation detection (F1=0.880), suicide plan identification (F1=0.779), and risk assessment (F1=0.907), with notable gains from few-shot prompting and fine-tuning. Compared to trained human operators, LLMs achieved comparable or superior performance on suicide plan identification and risk assessment, while humans retained advantages on mood status recognition and suicidal ideation detection. Mood status recognition remained challenging (max F1=0.709), likely due to missing vocal cues and semantic ambiguity. Notably, a fine-tuned 1.5B-parameter model (Qwen2.5-1.5B) outperformed larger models on mood and suicidal ideation tasks. LLMs demonstrate performance broadly comparable to trained human operators in text-based crisis assessment, with complementary strengths across task types. PsyCrisisBench provides a robust, real-world evaluation framework to guide future model development and ethical deployment in clinical mental health.
>
---
#### [replaced 002] HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing
- **分类: cs.AR; cs.AI; cs.CL; cs.DC**

- **简介: 该论文属硬件加速任务，旨在解决LLM推理中注意力层内存带宽瓶颈与KV缓存开销问题。提出HPU协处理器，通过PCIe FPGA卡卸载内存密集型操作，提升GPU利用率；实现4.1×加速与4.6×能效提升，支持大批次、长序列下的低成本可扩展推理。**

- **链接: [https://arxiv.org/pdf/2504.16112v2](https://arxiv.org/pdf/2504.16112v2)**

> **作者:** Myunghyun Rhee; Joonseop Sim; Taeyoung Ahn; Seungyong Lee; Daegun Yoon; Euiseok Kim; Kyoung Park; Youngpyo Joo; Hoshik Kim
>
> **备注:** 6 pages
>
> **摘要:** The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs.
>
---
#### [replaced 003] Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training
- **分类: cs.CL**

- **简介: 该论文属数据蒸馏任务，旨在解决生物医学领域高质量标注数据稀缺、知识层次复杂导致LLM训练受限的问题。提出基于MeSH知识层次引导的自动化数据蒸馏框架，自生成领域对齐问题并构建AI-Ready数据集，显著提升下游模型性能。**

- **链接: [https://arxiv.org/pdf/2501.15108v2](https://arxiv.org/pdf/2501.15108v2)**

> **作者:** Xunxin Cai; Chengrui Wang; Qingqing Long; Yuanchun Zhou; Meng Xiao
>
> **备注:** 10 pages
>
> **摘要:** The rapid advancement of large language models (LLMs) in biological-medical applications has highlighted a gap between their potential and the limited scale and often low quality of available open-source annotated textual datasets. In addition, the inherent complexity of the biomedical knowledge hierarchy significantly hampers efforts to bridge this gap.Can LLMs themselves play a pivotal role in overcoming this limitation? Motivated by this question, we investigate this challenge in the present study.We propose a framework that automates the distillation of high-quality textual training data from the extensive scientific literature. Our approach self-evaluates and generates questions that are more closely aligned with the biomedical domain, guided by the biomedical knowledge hierarchy through medical subject headings (MeSH). This comprehensive framework establishes an automated workflow, thereby eliminating the need for manual intervention. Furthermore, we conducted comprehensive experiments to evaluate the impact of our framework-generated data on downstream language models of varying sizes. Our approach substantially improves question-answering tasks compared to pre-trained models from the life sciences domain and powerful close-source models represented by GPT-4. Notably, the generated AI-Ready dataset enabled the Llama3-70B base model to outperform GPT-4 using MedPrompt with multiple times the number of parameters. Detailed case studies and ablation experiments underscore the significance of each component within our framework
>
---
#### [replaced 004] Voice-Interactive Surgical Agent for Multimodal Patient Data Control
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出语音交互手术代理VISA，解决机器人手术中医生无法腾手操作多模态患者数据的问题。基于分层多智能体框架（含编排与任务专用LLM智能体），支持语音检索临床信息、操控CT/3D模型等。构建240条命令数据集与MOEM评估指标，验证其高准确率、鲁棒性及可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.07392v3](https://arxiv.org/pdf/2511.07392v3)**

> **作者:** Hyeryun Park; Byung Mo Gu; Jun Hee Lee; Byeong Hyeon Choi; Sekeun Kim; Hyun Koo Kim; Kyungsang Kim
>
> **备注:** 14 pages, 13 figures, 3 tables
>
> **摘要:** In robotic surgery, surgeons fully engage their hands and visual attention in procedures, making it difficult to access and manipulate multimodal patient data without interrupting the workflow. To overcome this problem, we propose a Voice-Interactive Surgical Agent (VISA) built on a hierarchical multi-agent framework consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to interpret voice commands and execute tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models within surgical video. We construct a dataset of 240 user commands organized into hierarchical categories and introduce the Multi-level Orchestration Evaluation Metric (MOEM) that evaluates the performance and robustness at both the command and category levels. Experimental results demonstrate that VISA achieves high stage-level accuracy and workflow-level success rates, while also enhancing its robustness by correcting transcription errors, resolving linguistic ambiguity, and interpreting diverse free-form expressions. These findings highlight the strong potential of VISA to support robotic surgery and its scalability for integrating new functions and agents.
>
---
#### [replaced 005] Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Fin-R1——一个7B参数的金融推理大模型，旨在解决通用LLM在金融领域因数据碎片化、推理不透明、迁移性弱导致的应用难题。通过构建高质量金融CoT数据集并结合SFT与强化学习训练，提升准确性与可解释性，在合规检查与智能投顾中验证实用价值。**

- **链接: [https://arxiv.org/pdf/2503.16252v3](https://arxiv.org/pdf/2503.16252v3)**

> **作者:** Zhaowei Liu; Xin Guo; Zhi Yang; Fangqi Lou; Lingfeng Zeng; Mengping Li; Qi Qi; Zhiqiang Liu; Yiyang Han; Dongpo Cheng; Xingdong Feng; Huixia Judy Wang; Chengchun Shi; Liwen Zhang
>
> **摘要:** In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at https://github.com/SUFE-AIFLM-Lab/Fin-R1, and has already attracted over 700 stars.
>
---
#### [replaced 006] Verifiable Natural Language to Linear Temporal Logic Translation: A Benchmark Dataset and Evaluation Suite
- **分类: eess.SY; cs.CL**

- **简介: 该论文面向自然语言到线性时序逻辑（LTL）的可验证翻译任务，指出现有基准忽视命题接地能力，导致评估失真。作者构建了VLTL-Bench统一基准，涵盖多状态空间、多样化NL/LTL对及验证轨迹，并提供各子步骤真值，支持端到端与分步评估。**

- **链接: [https://arxiv.org/pdf/2507.00877v2](https://arxiv.org/pdf/2507.00877v2)**

> **作者:** William H English; Chase Walker; Dominic Simon; Sumit Kumar Jha; Rickard Ewetz
>
> **摘要:** Empirical evaluation of state-of-the-art natural-language (NL) to temporal-logic (TL) translation systems reveals near-perfect performance on existing benchmarks. However, current studies measure only the accuracy of the translation of NL logic into formal TL, ignoring a system's capacity to ground atomic propositions into new scenarios or environments. This is a critical feature, necessary for the verification of resulting formulas in a concrete state space. Consequently, most NL-to-TL translation frameworks propose their own bespoke dataset in which the correct grounding is known a-priori, inflating performance metrics and neglecting the need for extensible, domain-general systems. In this paper, we introduce the Verifiable Linear Temporal Logic Benchmark ( VLTL-Bench), a unifying benchmark that measures verification and verifiability of automated NL-to-LTL translation. The dataset consists of four unique state spaces and thousands of diverse natural language specifications and corresponding formal specifications in temporal logic. Moreover, the benchmark contains sample traces to validate the temporal logic expressions. While the benchmark directly supports end-to-end evaluation, we observe that many frameworks decompose the process into i) lifting, ii) grounding, iii) translation, and iv) verification. The benchmark provides ground truths after each of these steps to enable researches to improve and evaluate different substeps of the overall problem. To encourage methodologically sound advances in verifiable NL-to-LTL translation approaches, we release VLTL-Bench here: https://www.kaggle.com/datasets/dubascudes/vltl bench.
>
---
#### [replaced 007] Wrist Photoplethysmography Predicts Dietary Information
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文探索腕部PPG信号是否隐含饮食信息，属跨模态预测任务。为解决被动膳食监测难题，作者用110万餐数据训练语言模型，将PPG映射为文本描述，并验证其对进食与饱腹状态预测的有效性，AUC提升11%。**

- **链接: [https://arxiv.org/pdf/2511.19260v2](https://arxiv.org/pdf/2511.19260v2)**

> **作者:** Kyle Verrier; Achille Nazaret; Joseph Futoma; Andrew C. Miller; Guillermo Sapiro
>
> **备注:** 20 pages, 2 figures
>
> **摘要:** Whether wearable photoplethysmography (PPG) contains dietary information remains unknown. We trained a language model on 1.1M meals to predict meal descriptions from PPG, aligning PPG to text. PPG nontrivially predicts meal content; predictability decreases for PPGs farther from meals. This transfers to dietary tasks: PPG increases AUC by 11% for intake and satiety across held-out and independent cohorts, with gains robust to text degradation. Wearable PPG may enable passive dietary monitoring.
>
---
#### [replaced 008] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属测试时强化学习任务，旨在解决多数投票法导致的确认偏差与稀疏奖励问题。提出SCOPE框架，融合步级置信度加权伪标签估计与动态子群划分，提升伪标签质量与探索多样性，在AIME、AMC等基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.15146v2](https://arxiv.org/pdf/2512.15146v2)**

> **作者:** Weiqin Wang; Yile Wang; Kehao Chen; Hui Huang
>
> **摘要:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label deduction, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1% on challenging AIME 2025 and 8.1% on AMC. The code is released at https://github.com/szu-tera/SCOPE.
>
---
#### [replaced 009] MindShift: Analyzing Language Models' Reactions to Psychological Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MindShift基准，评估大语言模型对心理提示的适应能力。任务是量化LLM模拟人格特质的敏感性与一致性。通过改编MMPI量表、设计多强度人格提示，分析不同模型在心理角色扮演中的表现差异，揭示训练与对齐技术对人格建模的影响。**

- **链接: [https://arxiv.org/pdf/2512.09149v2](https://arxiv.org/pdf/2512.09149v2)**

> **作者:** Anton Vasiliuk; Irina Abdullaeva; Polina Druzhinina; Anton Razzhigaev; Andrey Kuznetsov
>
> **摘要:** Large language models (LLMs) hold the potential to absorb and reflect personality traits and attitudes specified by users. In our study, we investigated this potential using robust psychometric measures. We adapted the most studied test in psychological literature, namely Minnesota Multiphasic Personality Inventory (MMPI) and examined LLMs' behavior to identify traits. To asses the sensitivity of LLMs' prompts and psychological biases we created personality-oriented prompts, crafting a detailed set of personas that vary in trait intensity. This enables us to measure how well LLMs follow these roles. Our study introduces MindShift, a benchmark for evaluating LLMs' psychological adaptability. The results highlight a consistent improvement in LLMs' role perception, attributed to advancements in training datasets and alignment techniques. Additionally, we observe significant differences in responses to psychometric assessments across different model types and families, suggesting variability in their ability to emulate human-like personality traits. MindShift prompts and code for LLM evaluation will be publicly available.
>
---
#### [replaced 010] Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属AI评估方法学任务，旨在解决语音基础模型评价标准零散、不匹配的问题。作者提出三维正交分类法（评估维度、模型能力、任务要求），系统梳理现有评测基准，揭示评测缺口，并为模型与评测的合理匹配提供理论框架与实践指南。**

- **链接: [https://arxiv.org/pdf/2510.19509v2](https://arxiv.org/pdf/2510.19509v2)**

> **作者:** Maureen de Seyssel; Eeshan Gunesh Dhekane
>
> **备注:** 57 pages (26 main, 25 appendix, 6 references)
>
> **摘要:** Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the evaluation aspect being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.
>
---
#### [replaced 011] Knowledge-Driven Agentic Scientific Corpus Distillation Framework for Biomedical Large Language Models Training
- **分类: cs.CL; cs.AI; q-bio.QM**

- **简介: 该论文提出知识驱动的多智能体科学语料蒸馏框架，解决生物医学领域高质量标注语料稀缺问题。通过MeSH引导的协作智能体自动提取、合成并自评文本，生成AI就绪问答数据，显著提升生物医学LLM问答性能。**

- **链接: [https://arxiv.org/pdf/2504.19565v3](https://arxiv.org/pdf/2504.19565v3)**

> **作者:** Meng Xiao; Xunxin Cai; Qingqing Long; Chengrui Wang; Yuanchun Zhou; Hengshu Zhu
>
> **备注:** Biomedical Large Language Models, Agentic Corpus Distillation, Synthetic Question-Answer Generation, Agentic AI, Knowledge Hierarchy Guidance
>
> **摘要:** Corpus distillation for biomedical large language models (LLMs) seeks to address the pressing challenge of insufficient quantity and quality in open-source annotated scientific corpora, which remains a bottleneck for effective LLM training in biomedical research. This paper proposes a knowledge-driven, agentic framework for scientific corpus distillation, tailored explicitly for LLM training in the biomedical domain, addressing the challenge posed by the complex hierarchy of biomedical knowledge. Central to our approach is a collaborative multi-agent architecture, where specialized agents, each guided by the Medical Subject Headings (MeSH) hierarchy, work in concert to autonomously extract, synthesize, and self-evaluate high-quality textual data from vast scientific literature. This agentic framework collectively generates and refines domain-specific question-answer pairs, ensuring comprehensive coverage and consistency with biomedical ontologies while minimizing manual involvement. Extensive experimental results show that language models trained on our multi-agent distilled datasets achieve notable improvements in biomedical question-answering tasks, outperforming both strong life sciences LLM baselines and advanced proprietary models. Notably, our AI-Ready dataset enables Llama3-70B to surpass GPT-4 with MedPrompt and Med-PaLM-2, despite their larger scale. Detailed ablation studies and case analyses further validate the effectiveness and synergy of each agent within the framework, highlighting the potential of multi-agent collaboration in biomedical LLM training.
>
---
#### [replaced 012] Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs
- **分类: cs.CL**

- **简介: 该论文属AI安全与可靠性任务，旨在解决大模型对安全请求的过度拒绝问题。作者构建XSB和MS-XSB两个新基准诊断触发词与多轮场景下的拒绝偏差，并提出三种无需重训练的轻量后处理方法（忽略词、重写提示、注意力引导）有效提升合规性同时保持安全性。**

- **链接: [https://arxiv.org/pdf/2510.08158v3](https://arxiv.org/pdf/2510.08158v3)**

> **作者:** Shuzhou Yuan; Ercong Nie; Yinuo Sun; Chenxuan Zhao; William LaCroix; Michael Färber
>
> **摘要:** Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
>
---
#### [replaced 013] InsurTech innovation using natural language processing
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属应用研究任务，旨在解决保险业中非结构化文本数据难以用于精算分析的问题。作者利用NLP技术对替代数据进行特征去偏、压缩和行业分类，提升商业保险定价与风险评估能力，验证NLP是保险数据分析的基石而非辅助工具。**

- **链接: [https://arxiv.org/pdf/2507.21112v3](https://arxiv.org/pdf/2507.21112v3)**

> **作者:** Panyi Dong; Zhiyu Quan
>
> **摘要:** With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations, focusing on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate feature de-biasing, feature compression, and industry classification in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by introducing novel industry classification techniques. Through these demonstrations, we show that NLP is not merely a supplementary tool but a foundational element of modern, data-driven insurance analytics.
>
---
#### [replaced 014] Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文面向口语对话摘要任务，旨在解决情感感知与语音建模缺乏对齐数据的问题。作者构建了首个语音-文本-情感对齐数据集Spoken DialogSum，含13,460条带情绪标签的合成对话及双类型摘要，并验证了端到端Audio-LLM在情感摘要上的优势。**

- **链接: [https://arxiv.org/pdf/2512.14687v2](https://arxiv.org/pdf/2512.14687v2)**

> **作者:** Yen-Ju Lu; Kunxiao Gao; Mingrui Liang; Helin Wang; Thomas Thebaud; Laureano Moro-Velazquez; Najim Dehak; Jesus Villalba
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. We release an online demo at https://fatfat-emosum.github.io/EmoDialog-Sum-Audio-Samples/, with plans to release the full dataset in the near future. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling.
>
---
#### [replaced 015] From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自然语言处理中的上下文压缩任务，旨在解决长文本输入导致LLM计算开销大、噪声多、结构破坏等问题。提出EDU-based压缩框架：先用LingoEDU将文本解析为源索引锚定的EDU关系树，再轻量排序选取相关子树线性化，兼顾结构保真与效率。**

- **链接: [https://arxiv.org/pdf/2512.14244v2](https://arxiv.org/pdf/2512.14244v2)**

> **作者:** Yiqing Zhou; Yu Lei; Shuzheng Si; Qingyan Sun; Wei Wang; Yifei Wu; Hao Wen; Gang Chen; Fanchao Qi; Maosong Sun
>
> **摘要:** Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios.
>
---
#### [replaced 016] Finding Flawed Fictions: Evaluating Complex Reasoning in Language Models via Plot Hole Detection
- **分类: cs.CL**

- **简介: 该论文提出“情节漏洞检测”任务，旨在评估大语言模型（LLM）在故事理解与复杂推理方面的能力。针对现有基准偏重表层理解的问题，作者设计算法FlawedFictionsMaker可控注入漏洞，构建高质量、抗污染的FlawedFictions基准，并实证发现LLM在此任务上表现差，且易在生成/摘要中引入漏洞。**

- **链接: [https://arxiv.org/pdf/2504.11900v3](https://arxiv.org/pdf/2504.11900v3)**

> **作者:** Kabir Ahuja; Melanie Sclar; Yulia Tsvetkov
>
> **备注:** CoLM 2025 Camera Ready
>
> **摘要:** Stories are a fundamental aspect of human experience. Engaging deeply with stories and spotting plot holes -- inconsistencies in a storyline that break the internal logic or rules of a story's world -- requires nuanced reasoning skills, including tracking entities and events and their interplay, abstract thinking, pragmatic narrative understanding, commonsense and social reasoning, and theory of mind. As Large Language Models (LLMs) increasingly generate, interpret, and modify text, rigorously assessing their narrative consistency and deeper language understanding becomes critical. However, existing benchmarks focus mainly on surface-level comprehension. In this work, we propose plot hole detection in stories as a proxy to evaluate language understanding and reasoning in LLMs. We introduce FlawedFictionsMaker, a novel algorithm to controllably and carefully synthesize plot holes in human-written stories. Using this algorithm, we construct a benchmark to evaluate LLMs' plot hole detection abilities in stories -- FlawedFictions -- , which is robust to contamination, with human filtering ensuring high quality. We find that state-of-the-art LLMs struggle in accurately solving FlawedFictions regardless of the reasoning effort allowed, with performance significantly degrading as story length increases. Finally, we show that LLM-based story summarization and story generation are prone to introducing plot holes, with more than 50% and 100% increases in plot hole detection rates with respect to human-written originals.
>
---
#### [replaced 017] Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属语音表征学习任务，旨在解决微调后模型跨任务泛化能力下降的问题。提出Speech-FT两阶段框架：先抑制表征漂移的微调，再与预训练模型权重插值，兼顾任务性能与泛化性，在SUPERB等基准上显著提升效果。**

- **链接: [https://arxiv.org/pdf/2502.12672v3](https://arxiv.org/pdf/2502.12672v3)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Hao Tang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Model and code available at: https://github.com/nervjack2/Speech-FT
>
> **摘要:** Fine-tuning speech representation models can enhance performance on specific tasks but often compromises their cross-task generalization ability. This degradation is often caused by excessive changes in the representations, making it difficult to retain information learned during pre-training. Existing approaches, such as regularizing weight changes during fine-tuning, may fail to maintain sufficiently high feature similarity with the pre-trained model, and thus could possibly lose cross-task generalization. To address this issue, we propose Speech-FT, a novel two-stage fine-tuning framework designed to maintain cross-task generalization while benefiting from fine-tuning. Speech-FT first applies fine-tuning specifically designed to reduce representational drift, followed by weight-space interpolation with the pre-trained model to restore cross-task generalization. Extensive experiments on HuBERT, wav2vec 2.0, DeCoAR 2.0, and WavLM Base+ demonstrate that Speech-FT consistently improves performance across a wide range of supervised, unsupervised, and multitask fine-tuning scenarios. Moreover, Speech-FT achieves superior cross-task generalization compared to fine-tuning baselines that explicitly constrain weight changes, such as weight-space regularization and LoRA fine-tuning. Our analysis reveals that Speech-FT maintains higher feature similarity to the pre-trained model compared to alternative strategies, despite allowing larger weight-space updates. Notably, Speech-FT achieves significant improvements on the SUPERB benchmark. For example, when fine-tuning HuBERT on automatic speech recognition, Speech-FT is able to reduce phone error rate from 5.17% to 3.94%, lower word error rate from 6.38% to 5.75%, and increase speaker identification accuracy from 81.86% to 84.11%. Speech-FT provides a simple yet powerful solution for further refining speech representation models after pre-training.
>
---
#### [replaced 018] Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的引用生成任务，旨在解决“生成时引用”（G-Cite）与“事后引用”（P-Cite）两种范式的权衡问题。通过在四个数据集上系统评估覆盖度、正确性与延迟，发现P-Cite更适配高风险场景，G-Cite适用于精度优先的严格验证。**

- **链接: [https://arxiv.org/pdf/2509.21557v2](https://arxiv.org/pdf/2509.21557v2)**

> **作者:** Yash Saxena; Raviteja Bommireddy; Ankur Padia; Manas Gaur
>
> **备注:** NeurIPS 2025 LLM Evaluation Workshop
>
> **摘要:** Trustworthy Large Language Models (LLMs) must cite human-verifiable sources in high-stakes domains such as healthcare, law, academia, and finance, where even small errors can have severe consequences. Practitioners and researchers face a choice: let models generate citations during decoding, or let models draft answers first and then attach appropriate citations. To clarify this choice, we introduce two paradigms: Generation-Time Citation (G-Cite), which produces the answer and citations in one pass, and Post-hoc Citation (P-Cite), which adds or verifies citations after drafting. We conduct a comprehensive evaluation from zero-shot to advanced retrieval-augmented methods across four popular attribution datasets and provide evidence-based recommendations that weigh trade-offs across use cases. Our results show a consistent trade-off between coverage and citation correctness, with retrieval as the main driver of attribution quality in both paradigms. P-Cite methods achieve high coverage with competitive correctness and moderate latency, whereas G-Cite methods prioritize precision at the cost of coverage and speed. We recommend a retrieval-centric, P-Cite-first approach for high-stakes applications, reserving G-Cite for precision-critical settings such as strict claim verification. Our codes and human evaluation results are available at https://anonymous.4open.science/r/Citation_Paradigms-BBB5/
>
---
#### [replaced 019] OpenNER 1.0: Standardized Open-Access Named Entity Recognition Datasets in 50+ Languages
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别（NER）任务，旨在解决多语言、多本体NER数据分散、格式不统一的问题。作者构建了覆盖52种语言的标准化开源NER数据集OpenNER 1.0，统一标注格式与实体类型，并提供多模型基线结果，推动多语言NER研究。**

- **链接: [https://arxiv.org/pdf/2412.09587v3](https://arxiv.org/pdf/2412.09587v3)**

> **作者:** Chester Palen-Michel; Maxwell Pickering; Maya Kruse; Jonne Sälevä; Constantine Lignos
>
> **备注:** Published in the proceedings of EMNLP 2025
>
> **摘要:** We present OpenNER 1.0, a standardized collection of openly-available named entity recognition (NER) datasets. OpenNER contains 36 NER corpora that span 52 languages, human-annotated in varying named entity ontologies. We correct annotation format issues, standardize the original datasets into a uniform representation with consistent entity type names across corpora, and provide the collection in a structure that enables research in multilingual and multi-ontology NER. We provide baseline results using three pretrained multilingual language models and two large language models to compare the performance of recent models and facilitate future research in NER. We find that no single model is best in all languages and that significant work remains to obtain high performance from LLMs on the NER task. OpenNER is released at https://github.com/bltlab/open-ner.
>
---
#### [replaced 020] Are most sentences unique? An empirical examination of Chomskyan claims
- **分类: cs.CL**

- **简介: 该论文属实证语言学任务，旨在检验“多数句子唯一”这一乔姆斯基派主张。作者用NLTK分析多类语料库，统计完全重复的句子比例，发现虽常以唯一句为主，但重复句比例显著且因语类而异。**

- **链接: [https://arxiv.org/pdf/2509.19108v3](https://arxiv.org/pdf/2509.19108v3)**

> **作者:** Hiram Ring
>
> **摘要:** A repeated claim in linguistics is that the majority of linguistic utterances are unique. For example, Pinker (1994: 10), summarizing an argument by Noam Chomsky, states that "virtually every sentence that a person utters or understands is a brand-new combination of words, appearing for the first time in the history of the universe." With the increased availability of large corpora, this is a claim that can be empirically investigated. The current paper addresses the question by using the NLTK Python library to parse corpora of different genres, providing counts of exact string matches in each. Results show that while completely unique sentences are often the majority of corpora, this is highly constrained by genre, and that duplicate sentences are not an insignificant part of any individual corpus.
>
---
#### [replaced 021] LLM one-shot style transfer for Authorship Attribution and Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向作者身份识别与验证任务，旨在解决现有方法因数据偏差（风格-主题混淆）导致的泛化差问题。提出一种无监督LLM单次风格迁移框架，利用LLM语言建模能力与上下文学习，通过log-probability度量文本间风格可迁移性，无需显式监督，且随模型增大性能提升。**

- **链接: [https://arxiv.org/pdf/2510.13302v3](https://arxiv.org/pdf/2510.13302v3)**

> **作者:** Pablo Miralles-González; Javier Huertas-Tato; Alejandro Martín; David Camacho
>
> **摘要:** Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Existing supervised and contrastive approaches often rely on datasets with spurious correlations, conflating style with topic. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in general authorship analysis. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive CLM pre-training and in-context capabilities of modern LLMs. Our approach avoids explicit supervision with spuriously correlated data. Our method substantially outperforms unsupervised prompting-based baselines at similar model sizes and exceeds contrastively trained models when controlling for topical overlap. Our framework's performance improves with model size. In the case of authorship verification, we present an additional mechanism that increases test-time computation to improve accuracy; enabling flexible trade-offs between computational cost and task performance.
>
---
#### [replaced 022] Who is In Charge? Dissecting Role Conflicts in Instruction Following
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在指令遵循中的角色冲突问题，聚焦系统提示与用户输入、社会线索（如权威）间的优先级矛盾。通过线性探针、Logit归因和向量引导实验，揭示模型早期编码冲突信号、社会线索更易被遵循，但向量干预可提升整体指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2510.01228v2](https://arxiv.org/pdf/2510.01228v2)**

> **作者:** Siqi Zeng
>
> **备注:** 12 pages, 5 figures, Mech Interp Workshop (NeurIPS 2025) Poster
>
> **摘要:** Large language models should follow hierarchical instructions where system prompts override user inputs, yet recent work shows they often ignore this rule while strongly obeying social cues such as authority or consensus. We extend these behavioral findings with mechanistic interpretations on a large-scale dataset. Linear probing shows conflict-decision signals are encoded early, with system-user and social conflicts forming distinct subspaces. Direct Logit Attribution reveals stronger internal conflict detection in system-user cases but consistent resolution only for social cues. Steering experiments show that, despite using social cues, the vectors surprisingly amplify instruction following in a role-agnostic way. Together, these results explain fragile system obedience and underscore the need for lightweight hierarchy-sensitive alignment methods.
>
---
#### [replaced 023] Multiscale Aggregated Hierarchical Attention (MAHA): A Game Theoretic and Optimization Driven Approach to Efficient Contextual Modeling in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属大语言模型高效注意力机制研究，旨在解决MHSA的二次复杂度瓶颈。提出MAHA框架：通过层次化下采样实现多尺度注意力分解，并以凸优化或博弈论方法动态聚合多尺度信息，在保持全局依赖建模能力的同时降低计算成本。**

- **链接: [https://arxiv.org/pdf/2512.14925v2](https://arxiv.org/pdf/2512.14925v2)**

> **作者:** Caner Erden
>
> **摘要:** The quadratic computational complexity of MultiHead SelfAttention (MHSA) remains a fundamental bottleneck in scaling Large Language Models (LLMs) for longcontext tasks. While sparse and linearized attention mechanisms attempt to mitigate this, they often compromise the representation of global dependencies or fail to capture multiscale semantic granularity effectively. In this paper, we propose Multiscale Aggregated Hierarchical Attention (MAHA), a novel architectural framework that reformulates the attention mechanism through hierarchical decomposition and mathematically rigorous aggregation. Unlike conventional approaches that treat token interactions at a single resolution, MAHA dynamically partitions the input sequence into hierarchical scales via learnable downsampling operators. The core innovation lies in its aggregation strategy: we model the fusion of scalespecific attention matrices as a resource allocation problem, solved via a convex optimization framework or a Nash equilibriumbased gametheoretic approach. This ensures a theoretically optimal balance between local nuance and global context fidelity. Implemented within a hybrid dilatedconvolutional transformer backbone, MAHA utilizes differentiable optimization layers to enable endtoend training. Experimental evaluations demonstrate that MAHA achieves superior scalability; empirical FLOPs analysis confirms an 81% reduction in computational cost at a sequence length of 4096 compared to standard attention. This work bridges the gap between optimization theory and sequence modeling, offering a scalable solution for nextgeneration LLMs.
>
---
#### [replaced 024] VAEER: Visual Attention-Inspired Emotion Elicitation Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉情感诱发（VEE）任务，即预测图像引发的多标签情绪。为解决可解释性不足问题，提出VAEER框架：融合视觉注意力机制提取关键线索，结合情感知识图谱进行逐情绪推理，生成透明、情绪特异性理由，在多个基准上达到SOTA性能。**

- **链接: [https://arxiv.org/pdf/2505.24342v2](https://arxiv.org/pdf/2505.24342v2)**

> **作者:** Fanhang Man; Xiaoyue Chen; Huandong Wang; Baining Zhao; Han Li; Xinlei Chen
>
> **备注:** Currently under review as conference paper
>
> **摘要:** Images shared online strongly influence emotions and public well-being. Understanding the emotions an image elicits is therefore vital for fostering healthier and more sustainable digital communities, especially during public crises. We study Visual Emotion Elicitation (VEE), predicting the set of emotions that an image evokes in viewers. We introduce VAEER, an interpretable multi-label VEE framework that combines attention-inspired cue extraction with knowledge-grounded reasoning. VAEER isolates salient visual foci and contextual signals, aligns them with structured affective knowledge, and performs per-emotion inference to yield transparent, emotion-specific rationales. Across three heterogeneous benchmarks, including social imagery and disaster-related photos, VAEER achieves state-of-the-art results with up to 19% per-emotion improvements and a 12.3% average gain over strong CNN and VLM baselines. Our findings highlight interpretable multi-label emotion elicitation as a scalable foundation for responsible visual media analysis and emotionally sustainable online ecosystems.
>
---
#### [replaced 025] Beyond statistical significance: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation
- **分类: cs.CL**

- **简介: 该论文属NLP评估方法研究，旨在解决多语言/多任务场景下指标不确定性被低估的问题。作者提出基于重采样的方法，量化模型与数据双重来源的统计变异，提升排行榜中排名、模型差异等结果的可靠性。**

- **链接: [https://arxiv.org/pdf/2509.22612v2](https://arxiv.org/pdf/2509.22612v2)**

> **作者:** Jonne Sälevä; Duygu Ataman; Constantine Lignos
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** We introduce a set of resampling-based methods for quantifying uncertainty and statistical precision of evaluation metrics in multilingual and/or multitask NLP benchmarks. We show how experimental variation in performance scores arises from both model and data-related sources, and that accounting for both of them is necessary to avoid substantially underestimating the overall variability over hypothetical replications. Using multilingual question answering, machine translation, and named entity recognition as example tasks, we also demonstrate how resampling methods are useful for quantifying the replication uncertainty of various quantities used in leaderboards such as model rankings and pairwise differences between models.
>
---
#### [replaced 026] Non-Resolution Reasoning (NRR): A Computational Framework for Contextual Identity and Ambiguity Preservation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出非解析推理（NRR）框架，解决AI系统过早消解语义歧义的问题。它通过多向量嵌入、非坍缩注意力和上下文身份追踪，实现歧义保留与并行解释。在合成任务中显著提升分布外泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.13478v3](https://arxiv.org/pdf/2512.13478v3)**

> **作者:** Kei Saito
>
> **备注:** 7 pages, 2 figures, ORCID: 0009-0006-4715-9176
>
> **摘要:** Current artificial intelligence systems, despite remarkable capabilities in text generation and pattern recognition, exhibit a fundamental architectural limitation: they resolve ambiguity prematurely. This premature semantic collapse -- the tendency to collapse multiple valid interpretations into a single output -- stems from classical identity assumptions embedded in standard neural architectures. We propose Non-Resolution Reasoning (NRR), a computational framework that treats ambiguity retention as a valid reasoning mode rather than a defect to be eliminated. NRR introduces three core principles: (1) Non-Identity ($A \neq A$) -- the same symbol refers to different entities across contexts; (2) Approximate Identity ($A \approx A$) -- entities share partial structural overlap without being identical; and (3) Non-Resolution -- conflicting interpretations can coexist without forced convergence. We formalize these principles through three architectural components: Multi-Vector Embeddings for context-dependent representation, Non-Collapsing Attention for parallel interpretation retention, and Contextual Identity Tracking (CIT) for maintaining $A \neq A$ across inference. We demonstrate NRR's advantages through case studies in paradox handling, creative generation, and context-dependent reasoning. Crucially, we provide a minimal empirical validation on a synthetic context-shift task where an NRR-lite model achieves 90.9% out-of-distribution accuracy compared to 9.1% for standard architectures, demonstrating that ambiguity preservation enables structural generalization. NRR challenges the assumption that meaning must collapse to be useful, offering a foundation for AI systems capable of sophisticated ambiguity handling and creative reasoning. The question is not whether AI should resolve ambiguity, but when, how, and under whose control.
>
---
#### [replaced 027] The Emergence of Chunking Structures with Hierarchical RNN
- **分类: cs.CL**

- **简介: 该论文属NLP中的无监督句法分析任务，旨在解决无需人工标注的词块（chunking）结构发现问题。提出分层RNN模型，通过两阶段训练（无监督预训练+下游微调），实现词→块→句的层级建模，并观察到块结构在微调中呈瞬态涌现。**

- **链接: [https://arxiv.org/pdf/2309.04919v2](https://arxiv.org/pdf/2309.04919v2)**

> **作者:** Zijun Wu; Anup Anand Deshmukh; Yongkang Wu; Jimmy Lin; Lili Mou
>
> **备注:** Published in Computational Linguistics
>
> **摘要:** In Natural Language Processing (NLP), predicting linguistic structures, such as parsing and chunking, has mostly relied on manual annotations of syntactic structures. This paper introduces an unsupervised approach to chunking, a syntactic task that involves grouping words in a non-hierarchical manner. We present a Hierarchical Recurrent Neural Network (HRNN) designed to model word-to-chunk and chunk-to-sentence compositions. Our approach involves a two-stage training process: pretraining with an unsupervised parser and finetuning on downstream NLP tasks. Experiments on multiple datasets reveal a notable improvement of unsupervised chunking performance in both pretraining and finetuning stages. Interestingly, we observe that the emergence of the chunking structure is transient during the neural model's downstream-task training. This study contributes to the advancement of unsupervised syntactic structure discovery and opens avenues for further research in linguistic theory.
>
---
#### [replaced 028] Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Gated KalmaNet（GKA），属序列建模任务，旨在解决线性状态空间模型（SSM）因“衰减记忆”导致长程依赖建模能力弱的问题。它基于卡尔曼滤波框架，通过维持完整误差协方差并引入自适应正则化与Chebyshev迭代，实现稳定、高效、可并行的在线岭回归，兼顾精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.21016v2](https://arxiv.org/pdf/2511.21016v2)**

> **作者:** Liangzu Peng; Aditya Chattopadhyay; Luca Zancato; Elvis Nunez; Wei Xia; Stefano Soatto
>
> **备注:** 30 pages, 10 figures
>
> **摘要:** As efficient alternatives to softmax Attention, linear State-Space Models (SSMs) achieve constant memory and linear compute, but maintain only a lossy, fading summary of the past, often leading to inferior performance in recall-oriented tasks. We propose Gated KalmaNet (GKA), a layer that accounts for the full past while maintaining SSM-style efficiency. We ground our approach in the Kalman Filter (KF) framework, which provides a principled solution for optimal inference in dynamical systems. We show that several existing SSM layers (DeltaNet, Gated DeltaNet, and Kimi Delta Attention) are approximations to the KF recurrence that assume identity error covariance, thereby ignoring how past measurements (keys and values) should optimally influence state updates. In contrast, GKA computes the exact Kalman gain by maintaining the full error covariance. Under a steady-state assumption that enables parallelization, this reduces to solving an online ridge regression problem with constant memory and linear compute cost. A critical insight is that standard KF equations are numerically unstable in low-precision environments (like bfloat16) and hard to parallelize on modern hardware. We address this through: (1) adaptive regularization with input-dependent gating to control the condition number of the ridge regression for numerical stability, and (2) Chebyshev Iteration, which we show is more stable than conventional iterative solvers in low-precision settings. We further develop hardware-aware chunk-wise kernels to enable efficient training. Empirically, GKA outperforms existing SSM layers (like Mamba2 and Gated DeltaNet) on short-context tasks and achieves more than 10\% relative improvement on long-context RAG and LongQA tasks up to 128k tokens.
>
---
#### [replaced 029] Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space
- **分类: cs.CV; cs.CL**

- **简介: 该论文属多模态推理任务，旨在解决现有MLLMs依赖显式分步推理、感知-推理交互不稳定及计算开销大的问题。提出DMLR框架，通过置信度引导的隐空间策略优化与动态视觉特征注入，实现隐式、动态的图文交织推理。**

- **链接: [https://arxiv.org/pdf/2512.12623v2](https://arxiv.org/pdf/2512.12623v2)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Yue Fan; Qingyue Wei; Sheng Liu; Xin Eric Wang
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception-reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual-textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency.
>
---
#### [replaced 030] Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属长时对话RAG任务，解决记忆膨胀导致检索精度下降问题。提出LUFY方法，借鉴心理学模型，仅保留<10%高唤醒度记忆，显著提升用户体验与长期对话质量。**

- **链接: [https://arxiv.org/pdf/2409.12524v3](https://arxiv.org/pdf/2409.12524v3)**

> **作者:** Ryuichi Sumida; Koji Inoue; Tatsuya Kawahara
>
> **备注:** 37 pages, accepted and published in Dialogue & Discourse 16(2) (2025)
>
> **摘要:** While Retrieval-Augmented Generation (RAG) has shown promise in enhancing long-term conversations, the increasing memory load as conversations progress degrades retrieval accuracy. Drawing on psychological insights, we propose LUFY, a simple yet effective method that focuses on emotionally arousing memories and retains less than 10% of the conversation. In the user experiment, participants interacted with three types of RAG chatbots, each for 2 hours over 4 sessions, marking the most extensive assessment of a chatbot's long-term capabilities to date -- more than four times longer than any existing benchmark. The results demonstrate that prioritizing arousing memories while forgetting the majority of the conversation significantly enhances user experience. This study pushes the frontier of long-term conversations and highlights the importance of forgetting unimportant parts of conversations. Code and Dataset: https://github.com/ryuichi-sumida/LUFY, Hugginface Dataset:https://huggingface.co/datasets/RuiSumida/LUFY
>
---
#### [replaced 031] Online-PVLM: Advancing Personalized VLMs with Online Concept Learning
- **分类: cs.CL**

- **简介: 该论文属个性化视觉语言模型（VLM）任务，旨在解决现有方法无法实时适应新用户概念、扩展性差的问题。提出Online-PVLM框架，利用双曲表示实现测试时免训练的概念嵌入生成，并构建大规模基准OP-Eval评估在线概念学习性能。**

- **链接: [https://arxiv.org/pdf/2511.20056v2](https://arxiv.org/pdf/2511.20056v2)**

> **作者:** Huiyu Bai; Runze Wang; Zhuoyun Du; Yiyang Zhao; Fengji Zhang; Haoyu Chen; Xiaoyong Zhu; Bo Zheng; Xuejiao Zhao
>
> **备注:** Work in Progress
>
> **摘要:** Personalized Visual Language Models (VLMs) are gaining increasing attention for their formidable ability in user-specific concepts aligned interactions (e.g., identifying a user's bike). Existing methods typically require the learning of separate embeddings for each new concept, which fails to support real-time adaptation during testing. This limitation becomes particularly pronounced in large-scale scenarios, where efficient retrieval of concept embeddings is not achievable. To alleviate this gap, we propose Online-PVLM, a framework for online concept learning by leveraging hyperbolic representations. Our approach makes a train-free paradigm for concept embeddings generation at test time, making the use of personalized VLMs both scalable and efficient. In addition, we develop OP-Eval, a comprehensive and large-scale benchmark comprising 1,292 concepts and over 30K high-quality instances with diverse question types, designed to rigorously assess online concept learning in realistic scenarios. Extensive experiments demonstrate the state-of-the-art performance of our proposed framework. Our source code and dataset will be made available.
>
---
#### [replaced 032] Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback
- **分类: cs.CL**

- **简介: 该论文属学术评审辅助任务，旨在解决NLP领域审稿中 novelty 评估主观、低效的问题。提出三阶段LLM方法：内容提取、相关工作检索合成、结构化对比分析，基于真实审稿数据验证，显著提升与人类判断的一致性与可解释性。**

- **链接: [https://arxiv.org/pdf/2508.10795v3](https://arxiv.org/pdf/2508.10795v3)**

> **作者:** Osama Mohammed Afzal; Preslav Nakov; Tom Hope; Iryna Gurevych
>
> **摘要:** Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
>
---
#### [replaced 033] BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出BigCodeArena，解决代码生成中人工评估难的问题，通过实时执行环境支持人类偏好标注；构建了含14K对话的数据集及BigCodeReward、AutoCodeArena两个新基准，验证执行信息提升奖励模型一致性，并实现无需人工的自动Elo评测。**

- **链接: [https://arxiv.org/pdf/2510.08697v2](https://arxiv.org/pdf/2510.08697v2)**

> **作者:** Terry Yue Zhuo; Xiaolong Jin; Hange Liu; Juyong Jiang; Tianyang Liu; Chen Gong; Bhupesh Bishnoi; Vaisakhi Mishra; Marek Suppa; Noah Ziems; Saiteja Utpala; Ming Xu; Guangyu Song; Kaixin Li; Yuhan Cao; Bo Liu; Zheng Liu; Sabina Abdurakhmanova; Wenhao Yu; Mengzhao Jia; Jihan Yao; Kenneth Hamilton; Kumar Shridhar; Minh Chien Vu; Dingmin Wang; Jiawei Liu; Zijian Wang; Qian Liu; Binyuan Hui; Meg Risdal; Ahsen Khaliq; Atin Sood; Zhenchang Xing; Wasi Uddin Ahmad; John Grundy; David Lo; Banghua Zhu; Xiaoning Du; Torsten Scholak; Leandro von Werra
>
> **备注:** Built with love by the BigCode community :)
>
> **摘要:** Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Further analysis uncovers underexplored preferences of LLMs in fine-grained domains characterized by tasks, languages, and frameworks. To systematically examine code understanding and generation capabilities of frontier LLMs, we curated two benchmarks based on the collected data, namely BigCodeReward and AutoCodeArena. For BigCodeReward, we post-processed the 4,700 conversations and evaluated the consistency between reward models and human preferences. The evaluation shows that most LLMs have superior performance in judging coding preferences when the execution results are available. Inspired by these findings, we propose AutoCodeArena, an automatic Elo rating benchmark designed to assess the coding quality of LLMs without human involvement. We find that proprietary LLMs like GPT-5, Claude-Sonnet-4, and Claude-Opus-4 still lead in code generation performance among recent emerging models.
>
---
#### [replaced 034] SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SpiroLLM，首个能理解呼吸流速-容积时间序列（spirogram）的多模态大语言模型，旨在解决现有AI缺乏可解释性、LLMs无法解析生理信号的问题。通过SpiroEncoder和SpiroProjector融合波形与数值特征，生成可解释的COPD诊断报告，临床验证AUROC达0.8977。**

- **链接: [https://arxiv.org/pdf/2507.16145v2](https://arxiv.org/pdf/2507.16145v2)**

> **作者:** Shuhao Mei; Yongchao Long; Shan Cao; Xiaobo Han; Shijia Geng; Jinbo Sun; Yuxi Zhou; Shenda Hong
>
> **摘要:** Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8977 (95% CI: 0.88-0.91). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
>
---
#### [replaced 035] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward
- **分类: cs.CL; cs.MM**

- **简介: 该论文面向视障人士的实时导航指令生成任务，解决现有方法精度低、依赖真实数据等问题。提出LaF-GRPO框架，利用大模型模拟视障用户反馈作为奖励，优化视觉语言模型；构建开源数据集NIG4VI（27k样本），显著提升指令准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2506.04070v4](https://arxiv.org/pdf/2506.04070v4)**

> **作者:** Yi Zhao; Siqi Wang; Jing Li
>
> **备注:** Accepted at AAAI-26
>
> **摘要:** Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
>
---
#### [replaced 036] A stylometric analysis of speaker attribution from speech transcripts
- **分类: cs.CL**

- **简介: 该论文研究语音转录文本的说话人归属任务，解决语音伪装或TTS场景下仅凭语言内容识别说话人的难题。提出StyloSpeaker stylometric方法，融合多层级语言特征，在不同格式和话题控制的转录文本上评估性能，并与黑盒神经模型对比分析可解释性。**

- **链接: [https://arxiv.org/pdf/2512.13667v3](https://arxiv.org/pdf/2512.13667v3)**

> **作者:** Cristina Aggazzotti; Elizabeth Allyn Smith
>
> **备注:** v3: added StyloSpeaker github link; v2: added acknowledgments
>
> **摘要:** Forensic scientists often need to identify an unknown speaker or writer in cases such as ransom calls, covert recordings, alleged suicide notes, or anonymous online communications, among many others. Speaker recognition in the speech domain usually examines phonetic or acoustic properties of a voice, and these methods can be accurate and robust under certain conditions. However, if a speaker disguises their voice or employs text-to-speech software, vocal properties may no longer be reliable, leaving only their linguistic content available for analysis. Authorship attribution methods traditionally use syntactic, semantic, and related linguistic information to identify writers of written text (authorship attribution). In this paper, we apply a content-based authorship approach to speech that has been transcribed into text, using what a speaker says to attribute speech to individuals (speaker attribution). We introduce a stylometric method, StyloSpeaker, which incorporates character, word, token, sentence, and style features from the stylometric literature on authorship, to assess whether two transcripts were produced by the same speaker. We evaluate this method on two types of transcript formatting: one approximating prescriptive written text with capitalization and punctuation and another normalized style that removes these conventions. The transcripts' conversation topics are also controlled to varying degrees. We find generally higher attribution performance on normalized transcripts, except under the strongest topic control condition, in which overall performance is highest. Finally, we compare this more explainable stylometric model to black-box neural approaches on the same data and investigate which stylistic features most effectively distinguish speakers.
>
---
#### [replaced 037] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）口头置信度在对抗攻击下的鲁棒性问题。针对其易受扰动和越狱攻击导致置信度失真、答案频繁变动的缺陷，提出两类攻击框架，系统评估多种提示、模型与场景，并验证现有防御方法无效，呼吁设计更鲁棒的置信表达机制。**

- **链接: [https://arxiv.org/pdf/2507.06489v3](https://arxiv.org/pdf/2507.06489v3)**

> **作者:** Stephen Obadinma; Xiaodan Zhu
>
> **备注:** Published in NeurIPS 2025
>
> **摘要:** Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
>
---
#### [replaced 038] Rakuten Data Release: A Large-Scale and Long-Term Reviews Corpus for Hotel Domain
- **分类: cs.CL**

- **简介: 该论文构建了涵盖2009–2024年、共730万条的日本乐天旅行酒店评论语料库，含多维结构化元数据。属于数据集构建任务，旨在解决酒店领域长期、大规模、细粒度评论资源稀缺及数据漂移分析难的问题，提供了统计分析与漂移洞察。**

- **链接: [https://arxiv.org/pdf/2512.15151v2](https://arxiv.org/pdf/2512.15151v2)**

> **作者:** Yuki Nakayama; Koki Hikichi; Yun Ching Liu; Yu Hirate
>
> **摘要:** This paper presents a large-scale corpus of Rakuten Travel Reviews. Our collection contains 7.3 million customer reviews for 16 years, ranging from 2009 to 2024. Each record in the dataset contains the review text, its response from an accommodation, an anonymized reviewer ID, review date, accommodation ID, plan ID, plan title, room type, room name, purpose, accompanying group, and user ratings from different aspect categories, as well as an overall score. We present statistical information about our corpus and provide insights into factors driving data drift between 2019 and 2024 using statistical approaches.
>
---
#### [replaced 039] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出低秩克隆（LRC）方法，解决小语言模型（SLM）高效知识蒸馏难题。针对信息损失、表征对齐低效和FFN激活利用不足三大挑战，LRC通过低秩投影矩阵实现软剪枝与激活对齐，仅用20B tokens即达万亿级训练效果，提升千倍效率。**

- **链接: [https://arxiv.org/pdf/2505.12781v4](https://arxiv.org/pdf/2505.12781v4)**

> **作者:** Jitai Hao; Qiang Huang; Hao Liu; Xinyan Xiao; Zhaochun Ren; Jun Yu
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at https://github.com/CURRENTF/LowRankClone and https://huggingface.co/collections/JitaiHao/low-rank-clone-lrc-6828389e96a93f1d4219dfaf.
>
---
#### [replaced 040] Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属生成式AI中的模型微调任务，旨在解决离散扩散模型难以用策略梯度法（如RLHF）优化非可微奖励的问题。作者提出Score Entropy Policy Optimization（SEPO）算法，兼具高效性、通用性与理论保证，并在多类离散生成任务上验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2502.01384v3](https://arxiv.org/pdf/2502.01384v3)**

> **作者:** Oussama Zekri; Nicolas Boullé
>
> **备注:** 33 pages, 8 figures, 8 tables
>
> **摘要:** Discrete diffusion models have recently gained significant attention due to their ability to process complex discrete structures for language modeling. However, fine-tuning these models with policy gradient methods, as is commonly done in Reinforcement Learning from Human Feedback (RLHF), remains a challenging task. We propose an efficient, broadly applicable, and theoretically justified policy gradient algorithm, called Score Entropy Policy Optimization (\SEPO), for fine-tuning discrete diffusion models over non-differentiable rewards. Our numerical experiments across several discrete generative tasks demonstrate the scalability and efficiency of our method. Our code is available at https://github.com/ozekri/SEPO.
>
---
#### [replaced 041] RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属机器翻译任务，旨在解决偏好学习方法依赖静态三元组数据、泛化性差的问题。提出RLfR框架：以GPT-4o为教师模型，通过微教程式交互（生成→教师精修→双信号奖励）实现渐进式模仿学习，在FLORES-200上显著提升语义与实体保持能力。**

- **链接: [https://arxiv.org/pdf/2507.22219v2](https://arxiv.org/pdf/2507.22219v2)**

> **作者:** Dongyub Jude Lee; Zhenyi Ye; Pengcheng He
>
> **摘要:** Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores.
>
---
#### [replaced 042] Think Twice: Branch-and-Rethink Reasoning Reward Model
- **分类: cs.CL**

- **简介: 该论文属奖励建模任务，旨在解决传统单次打分导致的“判断弥散”问题。提出分支-再思奖励模型（BR-RM）：首轮自适应选择关键维度并生成假设，次轮据此聚焦验证；采用两步结构化训练，提升对细微错误的敏感性与评估精度。**

- **链接: [https://arxiv.org/pdf/2510.23596v2](https://arxiv.org/pdf/2510.23596v2)**

> **作者:** Yizhu Jiao; Jiaqi Zeng; Julien Veron Vialard; Oleksii Kuchaiev; Jiawei Han; Olivier Delalleau
>
> **备注:** Source Code: https://github.com/yzjiao/BR-RM. Model Checkpoints: https://huggingface.co/nvidia/Qwen3-Nemotron-14B-BRRM and https://huggingface.co/nvidia/Qwen3-Nemotron-8B-BRRM
>
> **摘要:** Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-once scoring into focused, second-look reasoning, BR-RM reduces judgment diffusion and improves sensitivity to subtle yet consequential errors while remaining practical and scalable. Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains.
>
---
#### [replaced 043] Failure Modes of Maximum Entropy RLHF
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究在线RLHF中最大熵强化学习的失效模式。任务是分析其过优化与KL不稳定问题。工作包括：理论推导SimPO与最大熵RL的关系，实验对比其在线表现，揭示熵正则化在在线场景下易导致奖励黑客，而离线设置中SimPO却有效。**

- **链接: [https://arxiv.org/pdf/2509.20265v2](https://arxiv.org/pdf/2509.20265v2)**

> **作者:** Ömer Veysel Çağatan; Barış Akgün
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** In this paper, we show that Simple Preference Optimization (SimPO) can be derived as Maximum Entropy Reinforcement Learning, providing a theoretical foundation for this reference-free method. Motivated by SimPO's strong performance in offline preference optimization, we investigate whether Maximum Entropy RL can achieve similar results in online RLHF settings. Our experiments find that Maximum Entropy RL consistently exhibits overoptimization and unstable KL dynamics, even at very low learning rates. Unlike KL-constrained methods that maintain stable training, entropy regularization fails to prevent reward hacking and appears to correlate with overoptimization. Lastly, we discuss possible explanations for why SimPO succeeds in offline settings while Maximum Entropy RL struggles in online scenarios. Our findings suggest that reference-free approaches may face distinct challenges when applied to online or offline preference learning.
>
---

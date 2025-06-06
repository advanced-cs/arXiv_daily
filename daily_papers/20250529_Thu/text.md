# 自然语言处理 cs.CL

- **最新发布 148 篇**

- **更新 134 篇**

## 最新发布

#### [new 001] Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MM-UPT框架，通过GRPO算法实现多模态LLM的无监督后训练，解决依赖标注数据及方法复杂的问题。利用多数投票的自我奖励机制替代传统奖励，并结合自动生成的合成问题，提升模型推理能力，在多个数据集上显著优于无监督基线，接近监督方法效果。**

- **链接: [http://arxiv.org/pdf/2505.22453v1](http://arxiv.org/pdf/2505.22453v1)**

> **作者:** Lai Wei; Yuting Li; Chen Wang; Yue Wang; Linghe Kong; Weiran Huang; Lichao Sun
>
> **摘要:** Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). However, these supervised methods require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. While recent efforts have explored unsupervised post-training, their methods are complex and difficult to iterate. In this work, we are the first to investigate the use of GRPO, a stable and scalable online RL algorithm, for enabling continual self-improvement without any external supervision. We propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that MM-UPT significantly improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3 %$\rightarrow$72.9 % on MathVista, 62.9 %$\rightarrow$68.7 % on We-Math), using standard dataset without ground truth labels. MM-UPT also outperforms prior unsupervised baselines and even approaches the results of supervised GRPO. Furthermore, we show that incorporating synthetic questions, generated solely by MLLM itself, can boost performance as well, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for continual, autonomous enhancement of MLLMs in the absence of external supervision. Our code is available at https://github.com/waltonfuture/MM-UPT.
>
---
#### [new 002] Counterfactual Simulatability of LLM Explanations for Generation Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM可解释性评估任务，旨在解决生成任务中模型解释的反事实预测能力问题。提出通用框架将反事实可模拟性扩展至生成任务（如新闻摘要、医疗建议），发现解释在摘要任务有效但医疗场景需改进，表明该评估更适合技能型而非知识型任务。**

- **链接: [http://arxiv.org/pdf/2505.21740v1](http://arxiv.org/pdf/2505.21740v1)**

> **作者:** Marvin Limpijankit; Yanda Chen; Melanie Subbiah; Nicholas Deas; Kathleen McKeown
>
> **摘要:** LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
>
---
#### [new 003] Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究推测解码与量化技术结合加速大模型推理，解决两者冲突导致内存优势被计算负载抵消的问题。提出分层框架，用小模型将树状草案转为序列，提升4位量化Llama-3-70B速度2.78倍，优于EAGLE-2。**

- **链接: [http://arxiv.org/pdf/2505.22179v1](http://arxiv.org/pdf/2505.22179v1)**

> **作者:** Yudi Zhang; Weilin Zhao; Xu Han; Tiejun Zhao; Wang Xu; Hailong Cao; Conghui Zhu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Speculative decoding and quantization effectively accelerate memory-bound inference of large language models. Speculative decoding mitigates the memory bandwidth bottleneck by verifying multiple tokens within a single forward pass, which increases computational effort. Quantization achieves this optimization by compressing weights and activations into lower bit-widths and also reduces computations via low-bit matrix multiplications. To further leverage their strengths, we investigate the integration of these two techniques. Surprisingly, experiments applying the advanced speculative decoding method EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models. This finding led to our new speculative decoding design: a hierarchical framework that employs a small model as an intermediate stage to turn tree-style drafts into sequence drafts, leveraging the memory access benefits of the target quantized model. Experimental results show that our hierarchical approach achieves a 2.78$\times$ speedup across various tasks for the 4-bit weight Llama-3-70B model on an A100 GPU, outperforming EAGLE-2 by 1.31$\times$. Code available at https://github.com/AI9Stars/SpecMQuant.
>
---
#### [new 004] NLP for Social Good: A Survey of Challenges, Opportunities, and Responsible Deployment
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于NLP应用综述任务，探讨NLP技术在解决社会问题中的挑战与机遇。针对技术部署中的伦理风险与公平性问题，论文通过跨学科分析社会目标与技术风险，提出NLP4SG（NLP for Social Good）的研究方向与责任部署策略，推动技术公平发展。**

- **链接: [http://arxiv.org/pdf/2505.22327v1](http://arxiv.org/pdf/2505.22327v1)**

> **作者:** Antonia Karamolegkou; Angana Borah; Eunjung Cho; Sagnik Ray Choudhury; Martina Galletti; Rajarshi Ghosh; Pranav Gupta; Oana Ignat; Priyanka Kargupta; Neema Kotonya; Hemank Lamba; Sun-Joo Lee; Arushi Mangla; Ishani Mondal; Deniz Nazarova; Poli Nemkova; Dina Pisarevskaya; Naquee Rizwan; Nazanin Sabri; Dominik Stammbach; Anna Steinberg; David Tomás; Steven R Wilson; Bowen Yi; Jessica H Zhu; Arkaitz Zubiaga; Anders Søgaard; Alexander Fraser; Zhijing Jin; Rada Mihalcea; Joel R. Tetreault; Daryna Dementieva
>
> **摘要:** Recent advancements in large language models (LLMs) have unlocked unprecedented possibilities across a range of applications. However, as a community, we believe that the field of Natural Language Processing (NLP) has a growing need to approach deployment with greater intentionality and responsibility. In alignment with the broader vision of AI for Social Good (Toma\v{s}ev et al., 2020), this paper examines the role of NLP in addressing pressing societal challenges. Through a cross-disciplinary analysis of social goals and emerging risks, we highlight promising research directions and outline challenges that must be addressed to ensure responsible and equitable progress in NLP4SG research.
>
---
#### [new 005] How does Misinformation Affect Large Language Model Behaviors and Preferences?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大型语言模型（LLMs）鲁棒性评估任务，旨在解决其易受虚假信息影响的问题。研究构建了含1034万条虚假信息的MisBench基准，分析LLMs在知识冲突和风格误导下的表现，并提出RtD方法提升检测能力，增强模型可靠性。**

- **链接: [http://arxiv.org/pdf/2505.21608v1](http://arxiv.org/pdf/2505.21608v1)**

> **作者:** Miao Peng; Nuo Chen; Jianheng Tang; Jia Li
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in knowledge-intensive tasks, while they remain vulnerable when encountering misinformation. Existing studies have explored the role of LLMs in combating misinformation, but there is still a lack of fine-grained analysis on the specific aspects and extent to which LLMs are influenced by misinformation. To bridge this gap, we present MisBench, the current largest and most comprehensive benchmark for evaluating LLMs' behavior and knowledge preference toward misinformation. MisBench consists of 10,346,712 pieces of misinformation, which uniquely considers both knowledge-based conflicts and stylistic variations in misinformation. Empirical results reveal that while LLMs demonstrate comparable abilities in discerning misinformation, they still remain susceptible to knowledge conflicts and stylistic variations. Based on these findings, we further propose a novel approach called Reconstruct to Discriminate (RtD) to strengthen LLMs' ability to detect misinformation. Our study provides valuable insights into LLMs' interactions with misinformation, and we believe MisBench can serve as an effective benchmark for evaluating LLM-based detectors and enhancing their reliability in real-world applications. Codes and data are available at https://github.com/GKNL/MisBench.
>
---
#### [new 006] Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文为循证医学（EBM）领域NLP应用的范围综述，通过分析129篇研究，探讨NLP如何支持EBM的"Ask- Acquire-Appraise-Apply-Assess"全流程，解决医学文献规模庞大与人工处理成本高之间的矛盾，总结技术现状、局限性及未来方向，推动临床决策智能化。**

- **链接: [http://arxiv.org/pdf/2505.22280v1](http://arxiv.org/pdf/2505.22280v1)**

> **作者:** Zihan Xu; Haotian Ma; Gongbo Zhang; Yihao Ding; Chunhua Weng; Yifan Peng
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Evidence-based medicine (EBM) is at the forefront of modern healthcare, emphasizing the use of the best available scientific evidence to guide clinical decisions. Due to the sheer volume and rapid growth of medical literature and the high cost of curation, there is a critical need to investigate Natural Language Processing (NLP) methods to identify, appraise, synthesize, summarize, and disseminate evidence in EBM. This survey presents an in-depth review of 129 research studies on leveraging NLP for EBM, illustrating its pivotal role in enhancing clinical decision-making processes. The paper systematically explores how NLP supports the five fundamental steps of EBM -- Ask, Acquire, Appraise, Apply, and Assess. The review not only identifies current limitations within the field but also proposes directions for future research, emphasizing the potential for NLP to revolutionize EBM by refining evidence extraction, evidence synthesis, appraisal, summarization, enhancing data comprehensibility, and facilitating a more efficient clinical workflow.
>
---
#### [new 007] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型持续扩展任务。解决在添加新语言时避免旧语言能力下降的问题。提出LayerMoE方法，通过分析各层语言表征相似性分配专家数量（相似度高则专家少），并在高相似层前加分类器引导路由，提升参数效率。**

- **链接: [http://arxiv.org/pdf/2505.22582v1](http://arxiv.org/pdf/2505.22582v1)**

> **作者:** Xue Zhang; Yunlong Liang; Fandong Meng; Songming Zhang; Yufeng Chen; Jinan Xu; Jie Zhou
>
> **备注:** ACL 2025 (Main), 16 pages, 5 figures, 11 tables
>
> **摘要:** Continually expanding new languages for existing large language models (LLMs) is a promising yet challenging approach to building powerful multilingual LLMs. The biggest challenge is to make the model continuously learn new languages while preserving the proficient ability of old languages. To achieve this, recent work utilizes the Mixture-of-Experts (MoE) architecture to expand new languages by adding new experts and avoid catastrophic forgetting of old languages by routing corresponding tokens to the original model backbone (old experts). Although intuitive, this kind of method is parameter-costly when expanding new languages and still inevitably impacts the performance of old languages. To address these limitations, we analyze the language characteristics of different layers in LLMs and propose a layer-wise expert allocation algorithm (LayerMoE) to determine the appropriate number of new experts for each layer. Specifically, we find different layers in LLMs exhibit different representation similarities between languages and then utilize the similarity as the indicator to allocate experts for each layer, i.e., the higher similarity, the fewer experts. Additionally, to further mitigate the forgetting of old languages, we add a classifier in front of the router network on the layers with higher similarity to guide the routing of old language tokens. Experimental results show that our method outperforms the previous state-of-the-art baseline with 60% fewer experts in the single-expansion setting and with 33.3% fewer experts in the lifelong-expansion setting, demonstrating the effectiveness of our method.
>
---
#### [new 008] Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）的错误生成机制，针对无关上下文幻觉问题，揭示其源于"基于类别的错误泛化"。通过行为分析和模型内部机制实验，发现LLMs在低层构建抽象类别表征，高层整合查询特征生成答案，且受直接推理与上下文线索两类计算通路影响，提出"随机变色龙"概念修正"随机鹦鹉"假设。**

- **链接: [http://arxiv.org/pdf/2505.22630v1](http://arxiv.org/pdf/2505.22630v1)**

> **作者:** Ziling Cheng; Meng Cao; Marc-Antoine Rondeau; Jackie Chi Kit Cheung
>
> **备注:** Accepted to ACL 2025 (Main Conference)
>
> **摘要:** The widespread success of large language models (LLMs) on NLP benchmarks has been accompanied by concerns that LLMs function primarily as stochastic parrots that reproduce texts similar to what they saw during pre-training, often erroneously. But what is the nature of their errors, and do these errors exhibit any regularities? In this work, we examine irrelevant context hallucinations, in which models integrate misleading contextual cues into their predictions. Through behavioral analysis, we show that these errors result from a structured yet flawed mechanism that we term class-based (mis)generalization, in which models combine abstract class cues with features extracted from the query or context to derive answers. Furthermore, mechanistic interpretability experiments on Llama-3, Mistral, and Pythia across 39 factual recall relation types reveal that this behavior is reflected in the model's internal computations: (i) abstract class representations are constructed in lower layers before being refined into specific answers in higher layers, (ii) feature selection is governed by two competing circuits -- one prioritizing direct query-based reasoning, the other incorporating contextual cues -- whose relative influences determine the final output. Our findings provide a more nuanced perspective on the stochastic parrot argument: through form-based training, LLMs can exhibit generalization leveraging abstractions, albeit in unreliable ways based on contextual cues -- what we term stochastic chameleons.
>
---
#### [new 009] Stratified Selective Sampling for Instruction Tuning with Dedicated Scoring Strategy
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）指令调优任务，旨在解决数据选择计算成本高且领域受限的问题。提出分层采样策略：通过数据分组、专用模型评估质量、轻量方法评分难度，并结合嵌入聚类保证多样性，实现高效通用的数据选择，降低计算开销。**

- **链接: [http://arxiv.org/pdf/2505.22157v1](http://arxiv.org/pdf/2505.22157v1)**

> **作者:** Paramita Mirza; Lucas Weber; Fabian Küch
>
> **摘要:** Recent work shows that post-training datasets for LLMs can be substantially downsampled without noticeably deteriorating performance. However, data selection often incurs high computational costs or is limited to narrow domains. In this paper, we demonstrate that data selection can be both -- efficient and universal -- by using a multi-step pipeline in which we efficiently bin data points into groups, estimate quality using specialized models, and score difficulty with a robust, lightweight method. Task-based categorization allows us to control the composition of our final data -- crucial for finetuning multi-purpose models. To guarantee diversity, we improve upon previous work using embedding models and a clustering algorithm. This integrated strategy enables high-performance fine-tuning with minimal overhead.
>
---
#### [new 010] Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言预训练数据过滤任务，旨在解决现有启发式方法跨语言迁移性差、扩展性不足的问题。提出JQL方法，利用多语言嵌入将LLM的标注能力轻量化，提升数据质量与跨语言适应性，在35种语言测试中优于现有方法，提高数据保留率和模型训练效果。**

- **链接: [http://arxiv.org/pdf/2505.22232v1](http://arxiv.org/pdf/2505.22232v1)**

> **作者:** Mehdi Ali; Manuel Brack; Max Lübbering; Elias Wendt; Abbas Goher Khan; Richard Rutmann; Alex Jude; Maurice Kraus; Alexander Arno Weber; Felix Stollenwerk; David Kaczér; Florian Mai; Lucie Flek; Rafet Sifa; Nicolas Flores-Herr; Joachim Köhler; Patrick Schramowski; Michael Fromm; Kristian Kersting
>
> **备注:** Project page available at https://huggingface.co/spaces/Jackal-AI/JQL
>
> **摘要:** High-quality multilingual training data is essential for effectively pretraining large language models (LLMs). Yet, the availability of suitable open-source multilingual datasets remains limited. Existing state-of-the-art datasets mostly rely on heuristic filtering methods, restricting both their cross-lingual transferability and scalability. Here, we introduce JQL, a systematic approach that efficiently curates diverse and high-quality multilingual data at scale while significantly reducing computational demands. JQL distills LLMs' annotation capabilities into lightweight annotators based on pretrained multilingual embeddings. These models exhibit robust multilingual and cross-lingual performance, even for languages and scripts unseen during training. Evaluated empirically across 35 languages, the resulting annotation pipeline substantially outperforms current heuristic filtering methods like Fineweb2. JQL notably enhances downstream model training quality and increases data retention rates. Our research provides practical insights and valuable resources for multilingual data curation, raising the standards of multilingual dataset development.
>
---
#### [new 011] A Linguistically Motivated Analysis of Intonational Phrasing in Text-to-Speech Systems: Revealing Gaps in Syntactic Sensitivity
- **分类: cs.CL**

- **简介: 该论文研究文本到语音（TTS）系统在生成语调短语边界时的语法敏感性。针对语法边界模糊的句子（如歧义句），发现TTS依赖逗号等表层标记，而简单句能利用语法线索。通过微调模型减少对逗号的依赖，使其关注深层语法，提升语调与句子结构的匹配。任务：改进TTS语法处理能力；问题：复杂句法下语调边界生成不准确；方法：分析系统表现并优化模型训练。**

- **链接: [http://arxiv.org/pdf/2505.22236v1](http://arxiv.org/pdf/2505.22236v1)**

> **作者:** Charlotte Pouw; Afra Alishahi; Willem Zuidema
>
> **备注:** Accepted to CoNLL 2025
>
> **摘要:** We analyze the syntactic sensitivity of Text-to-Speech (TTS) systems using methods inspired by psycholinguistic research. Specifically, we focus on the generation of intonational phrase boundaries, which can often be predicted by identifying syntactic boundaries within a sentence. We find that TTS systems struggle to accurately generate intonational phrase boundaries in sentences where syntactic boundaries are ambiguous (e.g., garden path sentences or sentences with attachment ambiguity). In these cases, systems need superficial cues such as commas to place boundaries at the correct positions. In contrast, for sentences with simpler syntactic structures, we find that systems do incorporate syntactic cues beyond surface markers. Finally, we finetune models on sentences without commas at the syntactic boundary positions, encouraging them to focus on more subtle linguistic cues. Our findings indicate that this leads to more distinct intonation patterns that better reflect the underlying structure.
>
---
#### [new 012] GMU Systems for the IWSLT 2025 Low-Resource Speech Translation Shared Task
- **分类: cs.CL**

- **简介: 该论文属于IWSLT 2025低资源语音翻译任务，旨在解决多语言语音到文本翻译的低资源挑战。团队基于SeamlessM4T-v2微调ASR、MT模型及端到端语音翻译系统，尝试直接微调、多任务训练及参数初始化策略，提升未覆盖语言的翻译性能。**

- **链接: [http://arxiv.org/pdf/2505.21781v1](http://arxiv.org/pdf/2505.21781v1)**

> **作者:** Chutong Meng; Antonios Anastasopoulos
>
> **备注:** IWSLT 2025
>
> **摘要:** This paper describes the GMU systems for the IWSLT 2025 low-resource speech translation shared task. We trained systems for all language pairs, except for Levantine Arabic. We fine-tuned SeamlessM4T-v2 for automatic speech recognition (ASR), machine translation (MT), and end-to-end speech translation (E2E ST). The ASR and MT models are also used to form cascaded ST systems. Additionally, we explored various training paradigms for E2E ST fine-tuning, including direct E2E fine-tuning, multi-task training, and parameter initialization using components from fine-tuned ASR and/or MT models. Our results show that (1) direct E2E fine-tuning yields strong results; (2) initializing with a fine-tuned ASR encoder improves ST performance on languages SeamlessM4T-v2 has not been trained on; (3) multi-task training can be slightly helpful.
>
---
#### [new 013] Found in Translation: Measuring Multilingual LLM Consistency as Simple as Translate then Evaluate
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型评估任务，旨在解决现有评测依赖昂贵标注数据且难以评估开放生成任务的问题。提出基于"翻译后评估"的框架，从信息一致性和共情一致性两维度评测LLM跨语言响应差异，揭示其在多语言家族中的显著性能缺陷。**

- **链接: [http://arxiv.org/pdf/2505.21999v1](http://arxiv.org/pdf/2505.21999v1)**

> **作者:** Ashim Gupta; Maitrey Mehta; Zhichao Xu; Vivek Srikumar
>
> **摘要:** Large language models (LLMs) provide detailed and impressive responses to queries in English. However, are they really consistent at responding to the same query in other languages? The popular way of evaluating for multilingual performance of LLMs requires expensive-to-collect annotated datasets. Further, evaluating for tasks like open-ended generation, where multiple correct answers may exist, is nontrivial. Instead, we propose to evaluate the predictability of model response across different languages. In this work, we propose a framework to evaluate LLM's cross-lingual consistency based on a simple Translate then Evaluate strategy. We instantiate this evaluation framework along two dimensions of consistency: information and empathy. Our results reveal pronounced inconsistencies in popular LLM responses across thirty languages, with severe performance deficits in certain language families and scripts, underscoring critical weaknesses in their multilingual capabilities. These findings necessitate cross-lingual evaluations that are consistent along multiple dimensions. We invite practitioners to use our framework for future multilingual LLM benchmarking.
>
---
#### [new 014] WebDancer: Towards Autonomous Information Seeking Agency
- **分类: cs.CL**

- **简介: 该论文提出WebDancer，一种自主信息寻求代理，旨在解决复杂问题中多步推理与自主信息检索的挑战。通过构建浏览数据、轨迹采样、监督微调及强化学习四阶段训练框架，提升代理的冷启动与泛化能力，在GAIA和WebWalkerQA基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.22648v1](http://arxiv.org/pdf/2505.22648v1)**

> **作者:** Jialong Wu; Baixuan Li; Runnan Fang; Wenbiao Yin; Liwen Zhang; Zhengwei Tao; Dingchu Zhang; Zekun Xi; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **摘要:** Addressing intricate real-world problems necessitates in-depth information seeking and multi-step reasoning. Recent progress in agentic systems, exemplified by Deep Research, underscores the potential for autonomous multi-step research. In this work, we present a cohesive paradigm for building end-to-end agentic information seeking agents from a data-centric and training-stage perspective. Our approach consists of four key stages: (1) browsing data construction, (2) trajectories sampling, (3) supervised fine-tuning for effective cold start, and (4) reinforcement learning for enhanced generalisation. We instantiate this framework in a web agent based on the ReAct, WebDancer. Empirical evaluations on the challenging information seeking benchmarks, GAIA and WebWalkerQA, demonstrate the strong performance of WebDancer, achieving considerable results and highlighting the efficacy of our training paradigm. Further analysis of agent training provides valuable insights and actionable, systematic pathways for developing more capable agentic models. The codes and demo will be released in https://github.com/Alibaba-NLP/WebAgent.
>
---
#### [new 015] The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason
- **分类: cs.CL**

- **简介: 该论文研究强化学习中奖励噪声对LLMs推理训练的影响。针对现实场景中奖励信号不准确的问题，实验发现模型对40%奖励翻转仍具鲁棒性，并提出仅通过奖励关键推理短语（RPR）即可提升性能，结合RPR与噪声奖励可增强开放任务表现。**

- **链接: [http://arxiv.org/pdf/2505.22653v1](http://arxiv.org/pdf/2505.22653v1)**

> **作者:** Ang Lv; Ruobing Xie; Xingwu Sun; Zhanhui Kang; Rui Yan
>
> **备注:** Preprint
>
> **摘要:** Recent studies on post-training large language models (LLMs) for reasoning through reinforcement learning (RL) typically focus on tasks that can be accurately verified and rewarded, such as solving math problems. In contrast, our research investigates the impact of reward noise, a more practical consideration for real-world scenarios involving the post-training of LLMs using reward models. We found that LLMs demonstrate strong robustness to substantial reward noise. For example, manually flipping 40% of the reward function's outputs in math tasks still allows a Qwen-2.5-7B model to achieve rapid convergence, improving its performance on math tasks from 5% to 72%, compared to the 75% accuracy achieved by a model trained with noiseless rewards. Surprisingly, by only rewarding the appearance of key reasoning phrases (namely reasoning pattern reward, RPR), such as ``first, I need to''-without verifying the correctness of answers, the model achieved peak downstream performance (over 70% accuracy for Qwen-2.5-7B) comparable to models trained with strict correctness verification and accurate rewards. Recognizing the importance of the reasoning process over the final results, we combined RPR with noisy reward models. RPR helped calibrate the noisy reward models, mitigating potential false negatives and enhancing the LLM's performance on open-ended tasks. These findings suggest the importance of improving models' foundational abilities during the pre-training phase while providing insights for advancing post-training techniques. Our code and scripts are available at https://github.com/trestad/Noisy-Rewards-in-Learning-to-Reason.
>
---
#### [new 016] LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High
- **分类: cs.CL**

- **简介: 该研究探讨大型语言模型（LLMs）在高风险情境下处理错误预设的能力，分析语言构造、政治立场等因素对其接受或拒绝误导性假设的影响。通过新数据集测试GPT-4-o等三模型，发现其识别虚假预设困难，强调语言预设分析对揭示政治 misinformation 的价值。**

- **链接: [http://arxiv.org/pdf/2505.22354v1](http://arxiv.org/pdf/2505.22354v1)**

> **作者:** Judith Sieker; Clara Lachenmaier; Sina Zarrieß
>
> **备注:** 8 pages (including References). Accepted at CogSci 2025
>
> **摘要:** This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses.
>
---
#### [new 017] BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain
- **分类: cs.CL**

- **简介: 该论文提出BioHopR基准，用于评估生物医学领域多跳多答案推理任务。针对现有基准无法评测复杂关系推理的问题，基于PrimeKG构建包含1/2跳任务的测试集，评估多个模型发现性能显著下降，凸显隐式推理挑战，为改进生物医学LLM提供新标准。**

- **链接: [http://arxiv.org/pdf/2505.22240v1](http://arxiv.org/pdf/2505.22240v1)**

> **作者:** Yunsoo Kim; Yusuf Abdulle; Honghan Wu
>
> **摘要:** Biomedical reasoning often requires traversing interconnected relationships across entities such as drugs, diseases, and proteins. Despite the increasing prominence of large language models (LLMs), existing benchmarks lack the ability to evaluate multi-hop reasoning in the biomedical domain, particularly for queries involving one-to-many and many-to-many relationships. This gap leaves the critical challenges of biomedical multi-hop reasoning underexplored. To address this, we introduce BioHopR, a novel benchmark designed to evaluate multi-hop, multi-answer reasoning in structured biomedical knowledge graphs. Built from the comprehensive PrimeKG, BioHopR includes 1-hop and 2-hop reasoning tasks that reflect real-world biomedical complexities. Evaluations of state-of-the-art models reveal that O3-mini, a proprietary reasoning-focused model, achieves 37.93% precision on 1-hop tasks and 14.57% on 2-hop tasks, outperforming proprietary models such as GPT4O and open-source biomedical models including HuatuoGPT-o1-70B and Llama-3.3-70B. However, all models exhibit significant declines in multi-hop performance, underscoring the challenges of resolving implicit reasoning steps in the biomedical domain. By addressing the lack of benchmarks for multi-hop reasoning in biomedical domain, BioHopR sets a new standard for evaluating reasoning capabilities and highlights critical gaps between proprietary and open-source models while paving the way for future advancements in biomedical LLMs.
>
---
#### [new 018] Pearl: A Multimodal Culturally-Aware Arabic Instruction Dataset
- **分类: cs.CL**

- **简介: 该论文提出Pearl：一个多模态阿拉伯文化数据集，旨在解决主流视觉语言模型中的文化偏见问题。通过45位阿拉伯标注者构建覆盖十领域的超K例数据，开发评估基准，实验表明其有效提升模型文化理解，数据公开。**

- **链接: [http://arxiv.org/pdf/2505.21979v1](http://arxiv.org/pdf/2505.21979v1)**

> **作者:** Fakhraddin Alwajih; Samar Mohamed Magdy; Abdellah El Mekki; Omer Nacar; Youssef Nafea; Safaa Taher Abdelfadil; Abdulfattah Mohammed Yahya; Hamzah Luqman; Nada Almarwani; Samah Aloufi; Baraah Qawasmeh; Houdaifa Atou; Serry Sibaee; Hamzah A. Alsayadi; Walid Al-Dhabyani; Maged S. Al-shaibani; Aya El aatar; Nour Qandos; Rahaf Alhamouri; Samar Ahmad; Razan Khassib; Lina Hamad; Mohammed Anwar AL-Ghrawi; Fatimah Alshamari; Cheikh Malainine; Doaa Qawasmeh; Aminetou Yacoub; Tfeil moilid; Ruwa AbuHweidi; Ahmed Aboeitta; Vatimetou Mohamed Lemin; Reem Abdel-Salam; Ahlam Bashiti; Adel Ammar; Aisha Alansari; Ahmed Ashraf; Nora Alturayeif; Sara Shatnawi; Alcides Alcoba Inciarte; AbdelRahim A. Elmadany; Mohamedou cheikh tourad; Ismail Berrada; Mustafa Jarrar; Shady Shehata; Muhammad Abdul-Mageed
>
> **备注:** https://github.com/UBC-NLP/pearl
>
> **摘要:** Mainstream large vision-language models (LVLMs) inherently encode cultural biases, highlighting the need for diverse multimodal datasets. To address this gap, we introduce Pearl, a large-scale Arabic multimodal dataset and benchmark explicitly designed for cultural understanding. Constructed through advanced agentic workflows and extensive human-in-the-loop annotations by 45 annotators from across the Arab world, Pearl comprises over K multimodal examples spanning ten culturally significant domains covering all Arab countries. We further provide two robust evaluation benchmarks Pearl and Pearl-Lite along with a specialized subset Pearl-X explicitly developed to assess nuanced cultural variations. Comprehensive evaluations on state-of-the-art open and proprietary LVLMs demonstrate that reasoning-centric instruction alignment substantially improves models' cultural grounding compared to conventional scaling methods. Pearl establishes a foundational resource for advancing culturally-informed multimodal modeling research. All datasets and benchmarks are publicly available.
>
---
#### [new 019] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MERRY模型，解决知识图谱(KG)基础模型局限于结构信息及in-KG任务的问题。通过多视角条件消息传递、动态残差融合与灵活边评分机制，融合文本与结构信息，提升KG推理及泛化至问答等out-of-KG任务，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21926v1](http://arxiv.org/pdf/2505.21926v1)**

> **作者:** Yin Hua; Zhiqiang Liu; Mingyang Chen; Zheng Fang; Chi Man Wong; Lingxiao Li; Chi Man Vong; Huajun Chen; Wen Zhang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** In natural language processing (NLP) and computer vision (CV), the successful application of foundation models across diverse tasks has demonstrated their remarkable potential. However, despite the rich structural and textual information embedded in knowledge graphs (KGs), existing research of foundation model for KG has primarily focused on their structural aspects, with most efforts restricted to in-KG tasks (e.g., knowledge graph completion, KGC). This limitation has hindered progress in addressing more challenging out-of-KG tasks. In this paper, we introduce MERRY, a foundation model for general knowledge graph reasoning, and investigate its performance across two task categories: in-KG reasoning tasks (e.g., KGC) and out-of-KG tasks (e.g., KG question answering, KGQA). We not only utilize the structural information, but also the textual information in KGs. Specifically, we propose a multi-perspective Conditional Message Passing (CMP) encoding architecture to bridge the gap between textual and structural modalities, enabling their seamless integration. Additionally, we introduce a dynamic residual fusion module to selectively retain relevant textual information and a flexible edge scoring mechanism to adapt to diverse downstream tasks. Comprehensive evaluations on 28 datasets demonstrate that MERRY outperforms existing baselines in most scenarios, showcasing strong reasoning capabilities within KGs and excellent generalization to out-of-KG tasks such as KGQA.
>
---
#### [new 020] Voice Adaptation for Swiss German
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属语音适应任务，旨在将标准德语文本转化为瑞士德语方言语音，解决方言语音克隆数据不足的问题。研究者预处理5000小时播客单词并自动标注方言类别，微调XTTSv2模型，取得CMOS-0.28、SMOS3.8的评估结果，推动小语种语音技术发展。**

- **链接: [http://arxiv.org/pdf/2505.22054v1](http://arxiv.org/pdf/2505.22054v1)**

> **作者:** Samuel Stucki; Jan Deriu; Mark Cieliebak
>
> **备注:** Submitted to Interspeech
>
> **摘要:** This work investigates the performance of Voice Adaptation models for Swiss German dialects, i.e., translating Standard German text to Swiss German dialect speech. For this, we preprocess a large dataset of Swiss podcasts, which we automatically transcribe and annotate with dialect classes, yielding approximately 5000 hours of weakly labeled training material. We fine-tune the XTTSv2 model on this dataset and show that it achieves good scores in human and automated evaluations and can correctly render the desired dialect. Our work shows a step towards adapting Voice Cloning technology to underrepresented languages. The resulting model achieves CMOS scores of up to -0.28 and SMOS scores of 3.8.
>
---
#### [new 021] MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MemOS——面向LLM的内存增强生成操作系统，解决现有模型缺乏统一内存架构的问题。针对参数记忆、激活状态和外部存储的碎片化管理，MemOS通过MemCube统一抽象三种内存类型，实现跨任务追踪、融合与迁移，建立可控制、可进化的内存中心框架，填补LLM基础设施空白。**

- **链接: [http://arxiv.org/pdf/2505.22101v1](http://arxiv.org/pdf/2505.22101v1)**

> **作者:** Zhiyu Li; Shichao Song; Hanyu Wang; Simin Niu; Ding Chen; Jiawei Yang; Chenyang Xi; Huayi Lai; Jihao Zhao; Yezhaohui Wang; Junpeng Ren; Zehao Lin; Jiahao Huo; Tianyi Chen; Kai Chen; Kehang Li; Zhiqiang Yin; Qingchen Yu; Bo Tang; Hongkang Yang; Zhi-Qin John Xu; Feiyu Xiong
>
> **摘要:** Large Language Models (LLMs) have emerged as foundational infrastructure in the pursuit of Artificial General Intelligence (AGI). Despite their remarkable capabilities in language perception and generation, current LLMs fundamentally lack a unified and structured architecture for handling memory. They primarily rely on parametric memory (knowledge encoded in model weights) and ephemeral activation memory (context-limited runtime states). While emerging methods like Retrieval-Augmented Generation (RAG) incorporate plaintext memory, they lack lifecycle management and multi-modal integration, limiting their capacity for long-term knowledge evolution. To address this, we introduce MemOS, a memory operating system designed for LLMs that, for the first time, elevates memory to a first-class operational resource. It builds unified mechanisms for representation, organization, and governance across three core memory types: parametric, activation, and plaintext. At its core is the MemCube, a standardized memory abstraction that enables tracking, fusion, and migration of heterogeneous memory, while offering structured, traceable access across tasks and contexts. MemOS establishes a memory-centric execution framework with strong controllability, adaptability, and evolvability. It fills a critical gap in current LLM infrastructure and lays the groundwork for continual adaptation, personalized intelligence, and cross-platform coordination in next-generation intelligent systems.
>
---
#### [new 022] Legal Assist AI: Leveraging Transformer-Based Model for Effective Legal Assistance
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Legal Assist AI，基于Transformer模型构建法律问答系统，解决印度公民因法律意识与信息获取不足导致的权利行使困难。通过微调印度法律数据（宪法、法典等），模型在AIBE评估中达60.08%准确率，优于GPT-3.5等，减少幻觉问题，提升法律推理可靠性。**

- **链接: [http://arxiv.org/pdf/2505.22003v1](http://arxiv.org/pdf/2505.22003v1)**

> **作者:** Jatin Gupta; Akhil Sharma; Saransh Singhania; Ali Imam Abidi
>
> **备注:** 9 pages, 5 tables, 4 figures. This is a revised version of a preprint previously available at this URL: https://doi.org/10.21203/rs.3.rs-5351879/v1
>
> **摘要:** Pursuit of accessible legal assistance in India faces a critical gap, as many citizens struggle to leverage their legal rights due to limited awareness and access to relevant legal information. This paper introduces Legal Assist AI, a transformer-based model designed to bridge this gap by offering effective legal assistance through large language models (LLMs). The system retrieves relevant legal information from a curated database and generates accurate responses, enabling effective assistance for diverse users, including legal professionals, scholars, and the general public. The model was fine-tuned on extensive datasets from the Indian legal domain, including Indian Constitution, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and so forth, providing a robust understanding of the complexities of Indian law. By incorporating domain-specific legal datasets, the proposed model demonstrated remarkable efficiency and specialization in legal Question-Answering. The model was evaluated against state-of-the-art models such as GPT-3.5 Turbo and Mistral 7B, achieving a 60.08% score on the AIBE, outperforming its competitors in legal reasoning and accuracy. Unlike other models, Legal Assist AI avoided common issues such as hallucinations, making it highly reliable for practical legal applications. It showcases the model's applicability in real-world legal scenarios, with future iterations aiming to enhance performance and expand its dataset to cover a broader range of multilingual and case-specific queries as well.
>
---
#### [new 023] VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出VRAG-RL框架，针对视觉丰富信息理解任务中传统方法处理视觉信息不足、检索查询效果差的问题，通过强化学习优化视觉语言模型与搜索引擎的交互，设计视觉操作动作（如裁剪/缩放）和奖励机制，提升多模态推理能力。**

- **链接: [http://arxiv.org/pdf/2505.22019v1](http://arxiv.org/pdf/2505.22019v1)**

> **作者:** Qiuchen Wang; Ruixue Ding; Yu Zeng; Zehui Chen; Lin Chen; Shihang Wang; Pengjun Xie; Fei Huang; Feng Zhao
>
> **摘要:** Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at \hyperlink{https://github.com/Alibaba-NLP/VRAG}{https://github.com/Alibaba-NLP/VRAG}.
>
---
#### [new 024] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Self-Error-Instruct框架，针对LLMs数学推理中错误泛化不足的问题。通过分析模型在GSM8K等数据集的错误案例，聚类错误类型并生成针对性训练数据，迭代微调模型，提升数学推理能力。**

- **链接: [http://arxiv.org/pdf/2505.22591v1](http://arxiv.org/pdf/2505.22591v1)**

> **作者:** Erxin Yu; Jing Li; Ming Liao; Qi Zhu; Boyang Xue; Minghui Xu; Baojun Wang; Lanqing Hong; Fei Mi; Lifeng Shang
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Although large language models demonstrate strong performance across various domains, they still struggle with numerous bad cases in mathematical reasoning. Previous approaches to learning from errors synthesize training data by solely extrapolating from isolated bad cases, thereby failing to generalize the extensive patterns inherent within these cases. This paper presents Self-Error-Instruct (SEI), a framework that addresses these model weaknesses and synthesizes more generalized targeted training data. Specifically, we explore a target model on two mathematical datasets, GSM8K and MATH, to pinpoint bad cases. Then, we generate error keyphrases for these cases based on the instructor model's (GPT-4o) analysis and identify error types by clustering these keyphrases. Next, we sample a few bad cases during each generation for each identified error type and input them into the instructor model, which synthesizes additional training data using a self-instruct approach. This new data is refined through a one-shot learning process to ensure that only the most effective examples are kept. Finally, we use these curated data to fine-tune the target model, iteratively repeating the process to enhance performance. We apply our framework to various models and observe improvements in their reasoning abilities across both in-domain and out-of-domain mathematics datasets. These results demonstrate the effectiveness of self-error instruction in improving LLMs' mathematical reasoning through error generalization.
>
---
#### [new 025] Multi-MLLM Knowledge Distillation for Out-of-Context News Detection
- **分类: cs.CL; cs.MM**

- **简介: 该论文针对小规模多模态大模型在低资源场景下检测脱离原始语境新闻的性能不足问题，提出基于多教师模型知识蒸馏的两阶段优化方法：首先通过多个教师模型生成预测与推理依据，随后分阶段利用LoRA微调和DPO对抗训练提升学生模型，实现仅用10%标注数据达到最优效果。**

- **链接: [http://arxiv.org/pdf/2505.22517v1](http://arxiv.org/pdf/2505.22517v1)**

> **作者:** Yimeng Gu; Zhao Tong; Ignacio Castro; Shu Wu; Gareth Tyson
>
> **摘要:** Multimodal out-of-context news is a type of misinformation in which the image is used outside of its original context. Many existing works have leveraged multimodal large language models (MLLMs) for detecting out-of-context news. However, observing the limited zero-shot performance of smaller MLLMs, they generally require label-rich fine-tuning and/or expensive API calls to GPT models to improve the performance, which is impractical in low-resource scenarios. In contrast, we aim to improve the performance of small MLLMs in a more label-efficient and cost-effective manner. To this end, we first prompt multiple teacher MLLMs to generate both label predictions and corresponding rationales, which collectively serve as the teachers' knowledge. We then introduce a two-stage knowledge distillation framework to transfer this knowledge to a student MLLM. In Stage 1, we apply LoRA fine-tuning to the student model using all training data. In Stage 2, we further fine-tune the student model using both LoRA fine-tuning and DPO on the data points where teachers' predictions conflict. This two-stage strategy reduces annotation costs and helps the student model uncover subtle patterns in more challenging cases. Experimental results demonstrate that our approach achieves state-of-the-art performance using less than 10% labeled data.
>
---
#### [new 026] R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing
- **分类: cs.CL; cs.AI; cs.LG; cs.PF; I.2.7**

- **简介: 该论文提出R2R方法，解决小模型推理性能不足与大模型效率低的问题。通过识别少数关键分歧token，路由至大模型处理，其余由小模型生成，平衡效率与性能。实验显示其以5.6B参数超越更大模型，提升速度与精度。**

- **链接: [http://arxiv.org/pdf/2505.21600v1](http://arxiv.org/pdf/2505.21600v1)**

> **作者:** Tianyu Fu; Yi Ge; Yichen You; Enshu Liu; Zhihang Yuan; Guohao Dai; Shengen Yan; Huazhong Yang; Yu Wang
>
> **摘要:** Large Language Models (LLMs) achieve impressive reasoning capabilities at the cost of substantial inference overhead, posing substantial deployment challenges. Although distilled Small Language Models (SLMs) significantly enhance efficiency, their performance suffers as they fail to follow LLMs' reasoning paths. Luckily, we reveal that only a small fraction of tokens genuinely diverge reasoning paths between LLMs and SLMs. Most generated tokens are either identical or exhibit neutral differences, such as minor variations in abbreviations or expressions. Leveraging this insight, we introduce **Roads to Rome (R2R)**, a neural token routing method that selectively utilizes LLMs only for these critical, path-divergent tokens, while leaving the majority of token generation to the SLM. We also develop an automatic data generation pipeline that identifies divergent tokens and generates token-level routing labels to train the lightweight router. We apply R2R to combine R1-1.5B and R1-32B models from the DeepSeek family, and evaluate on challenging math, coding, and QA benchmarks. With an average activated parameter size of 5.6B, R2R surpasses the average accuracy of R1-7B by 1.6x, outperforming even the R1-14B model. Compared to R1-32B, it delivers a 2.8x wall-clock speedup with comparable performance, advancing the Pareto frontier of test-time scaling efficiency. Our code is available at https://github.com/thu-nics/R2R.
>
---
#### [new 027] Text2Grad: Reinforcement Learning from Natural Language Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Text2Grad方法，改进强化学习从自然语言反馈中学习。针对传统标量奖励不透明且粗粒度的问题，通过将文本反馈转化为跨度级梯度，实现精准模型优化。工作包括反馈标注管道、细粒度奖励模型及跨度策略优化器，提升任务性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.22338v1](http://arxiv.org/pdf/2505.22338v1)**

> **作者:** Hanyang Wang; Lu Wang; Chaoyun Zhang; Tianjun Mao; Si Qin; Qingwei Lin; Saravan Rajmohan; Dongmei Zhang
>
> **备注:** The code for our method is available at https://github.com/microsoft/Text2Grad
>
> **摘要:** Traditional RLHF optimizes language models with coarse, scalar rewards that mask the fine-grained reasons behind success or failure, leading to slow and opaque learning. Recent work augments RL with textual critiques through prompting or reflection, improving interpretability but leaving model parameters untouched. We introduce Text2Grad, a reinforcement-learning paradigm that turns free-form textual feedback into span-level gradients. Given human (or programmatic) critiques, Text2Grad aligns each feedback phrase with the relevant token spans, converts these alignments into differentiable reward signals, and performs gradient updates that directly refine the offending portions of the model's policy. This yields precise, feedback-conditioned adjustments instead of global nudges. Text2Grad is realized through three components: (1) a high-quality feedback-annotation pipeline that pairs critiques with token spans; (2) a fine-grained reward model that predicts span-level reward on answer while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates natural-language gradients. Across summarization, code generation, and question answering, Text2Grad consistently surpasses scalar-reward RL and prompt-only baselines, providing both higher task metrics and richer interpretability. Our results demonstrate that natural-language feedback, when converted to gradients, is a powerful signal for fine-grained policy optimization. The code for our method is available at https://github.com/microsoft/Text2Grad
>
---
#### [new 028] Curse of High Dimensionality Issue in Transformer for Long-context Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于长上下文建模任务，旨在解决Transformer中冗余注意力计算导致的高计算成本问题。通过将序列建模重构为监督学习任务，分析注意力稀疏性，提出动态分组注意力（DGA）方法，利用分组编码聚合次要token，减少冗余计算，提升效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2505.22107v1](http://arxiv.org/pdf/2505.22107v1)**

> **作者:** Shuhai Zhang; Zeng You; Yaofo Chen; Zhiquan Wen; Qianyue Wang; Zhijie Qiu; Yuanqing Li; Mingkui Tan
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Transformer-based large language models (LLMs) excel in natural language processing tasks by capturing long-range dependencies through self-attention mechanisms. However, long-context modeling faces significant computational inefficiencies due to \textit{redundant} attention computations: while attention weights are often \textit{sparse}, all tokens consume \textit{equal} computational resources. In this paper, we reformulate traditional probabilistic sequence modeling as a \textit{supervised learning task}, enabling the separation of relevant and irrelevant tokens and providing a clearer understanding of redundancy. Based on this reformulation, we theoretically analyze attention sparsity, revealing that only a few tokens significantly contribute to predictions. Building on this, we formulate attention optimization as a linear coding problem and propose a \textit{group coding strategy}, theoretically showing its ability to improve robustness against random noise and enhance learning efficiency. Motivated by this, we propose \textit{Dynamic Group Attention} (DGA), which leverages the group coding to explicitly reduce redundancy by aggregating less important tokens during attention computation. Empirical results show that our DGA significantly reduces computational costs while maintaining competitive performance.Code is available at https://github.com/bolixinyu/DynamicGroupAttention.
>
---
#### [new 029] Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出SMILE方法，属于模型可解释性任务，解决大语言模型黑箱决策问题。通过局部输入扰动分析关键影响词，生成热图解释，验证其准确性等指标，提升AI透明度与可信度。**

- **链接: [http://arxiv.org/pdf/2505.21657v1](http://arxiv.org/pdf/2505.21657v1)**

> **作者:** Zeinab Dehghani; Koorosh Aslansefat; Adil Khan; Mohammed Naveed Akram
>
> **备注:** arXiv admin note: text overlap with arXiv:2412.16277
>
> **摘要:** Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy.
>
---
#### [new 030] LoKI: Low-damage Knowledge Implanting of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出LoKI方法，属于参数高效微调任务，旨在解决模型微调时的灾难性遗忘问题及一般能力下降。通过分析Transformer知识存储机制，LoKI在任务性能上优于或匹配全量微调和LoRA，同时更好保留通用能力，实验证明其优势。**

- **链接: [http://arxiv.org/pdf/2505.22120v1](http://arxiv.org/pdf/2505.22120v1)**

> **作者:** Runyu Wang; Peng Ping; Zhengyu Guo; Xiaoye Zhang; Quan Shi; Liting Zhou; Tianbo Ji
>
> **摘要:** Fine-tuning adapts pretrained models for specific tasks but poses the risk of catastrophic forgetting (CF), where critical knowledge from pre-training is overwritten. Current Parameter-Efficient Fine-Tuning (PEFT) methods for Large Language Models (LLMs), while efficient, often sacrifice general capabilities. To address the issue of CF in a general-purpose PEFT framework, we propose \textbf{Lo}w-damage \textbf{K}nowledge \textbf{I}mplanting (\textbf{LoKI}), a PEFT technique that is based on a mechanistic understanding of how knowledge is stored in transformer architectures. In two real-world scenarios, LoKI demonstrates task-specific performance that is comparable to or even surpasses that of full fine-tuning and LoRA-based methods across various model types, while significantly better preserving general capabilities. Our work connects mechanistic insights into LLM knowledge storage with practical fine-tuning objectives, achieving state-of-the-art trade-offs between task specialization and the preservation of general capabilities. Our implementation is publicly available as ready-to-use code\footnote{https://github.com/Nexround/LoKI}.
>
---
#### [new 031] Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出C²TU方法，解决中文同音隐写毒性内容检测问题。通过同音词图和毒词库匹配候选词，利用BERT和LLM模型过滤非毒性词并还原隐写词，无需训练和提示。实验显示其F1值和准确率较最优方法提升达71%和35%。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22184v1](http://arxiv.org/pdf/2505.22184v1)**

> **作者:** Xuchen Ma; Jianxiang Yu; Wenming Shao; Bo Pang; Xiang Li
>
> **备注:** 25 pages, 5 figures, 9 tables
>
> **摘要:** Social media platforms have experienced a significant rise in toxic content, including abusive language and discriminatory remarks, presenting growing challenges for content moderation. Some users evade censorship by deliberately disguising toxic words through homophonic cloak, which necessitates the task of unveiling cloaked toxicity. Existing methods are mostly designed for English texts, while Chinese cloaked toxicity unveiling has not been solved yet. To tackle the issue, we propose C$^2$TU, a novel training-free and prompt-free method for Chinese cloaked toxic content unveiling. It first employs substring matching to identify candidate toxic words based on Chinese homo-graph and toxic lexicon. Then it filters those candidates that are non-toxic and corrects cloaks to be their corresponding toxicities. Specifically, we develop two model variants for filtering, which are based on BERT and LLMs, respectively. For LLMs, we address the auto-regressive limitation in computing word occurrence probability and utilize the full semantic contexts of a text sequence to reveal cloaked toxic words. Extensive experiments demonstrate that C$^2$TU can achieve superior performance on two Chinese toxic datasets. In particular, our method outperforms the best competitor by up to 71% on the F1 score and 35% on accuracy, respectively.
>
---
#### [new 032] Evaluating the Retrieval Robustness of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估大型语言模型在检索增强生成（RAG）中的检索鲁棒性，研究RAG是否优于非RAG、检索文档数量及顺序对性能的影响。通过构建1500题基准数据集、提出三类鲁棒性指标，并测试11个模型与三种提示策略，发现LLMs存在不同程度的鲁棒性缺陷，阻碍RAG效能完全发挥。**

- **链接: [http://arxiv.org/pdf/2505.21870v1](http://arxiv.org/pdf/2505.21870v1)**

> **作者:** Shuyang Cao; Karthik Radhakrishnan; David Rosenberg; Steven Lu; Pengxiang Cheng; Lu Wang; Shiyue Zhang
>
> **备注:** 19 pages
>
> **摘要:** Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG may also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content. In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; (3) and whether document orders impact results. To facilitate this study, we establish a benchmark of 1500 open-domain questions, each with retrieved documents from Wikipedia. We introduce three robustness metrics, each corresponds to one research question. Our comprehensive experiments, involving 11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit surprisingly high retrieval robustness; nonetheless, different degrees of imperfect robustness hinders them from fully utilizing the benefits of RAG.
>
---
#### [new 033] Do Large Language Models Think Like the Brain? Sentence-Level Evidence from fMRI and Hierarchical Embeddings
- **分类: cs.CL; q-bio.NC**

- **简介: 该研究通过对比14个LLMs的层级嵌入与人类fMRI数据，探究LLMs是否与大脑语言处理机制趋同。任务为分析模型层级表示与脑神经响应的对齐关系，解决LLMs的脑似性源于规模扩展还是本质对齐问题。工作包括构建句子级神经预测模型，发现模型性能提升促使高层语义表征更接近脑区激活模式。**

- **链接: [http://arxiv.org/pdf/2505.22563v1](http://arxiv.org/pdf/2505.22563v1)**

> **作者:** Yu Lei; Xingyang Ge; Yi Zhang; Yiming Yang; Bolei Ma
>
> **摘要:** Understanding whether large language models (LLMs) and the human brain converge on similar computational principles remains a fundamental and important question in cognitive neuroscience and AI. Do the brain-like patterns observed in LLMs emerge simply from scaling, or do they reflect deeper alignment with the architecture of human language processing? This study focuses on the sentence-level neural mechanisms of language models, systematically investigating how hierarchical representations in LLMs align with the dynamic neural responses during human sentence comprehension. By comparing hierarchical embeddings from 14 publicly available LLMs with fMRI data collected from participants, who were exposed to a naturalistic narrative story, we constructed sentence-level neural prediction models to precisely identify the model layers most significantly correlated with brain region activations. Results show that improvements in model performance drive the evolution of representational architectures toward brain-like hierarchies, particularly achieving stronger functional and anatomical correspondence at higher semantic abstraction levels.
>
---
#### [new 034] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation
- **分类: cs.CL**

- **简介: 该论文属于表格评估任务，旨在解决传统指标无法捕捉表格结构与内容细微差异的问题。提出TabXEval框架，通过两阶段方法（结构对齐TabAlign与语义语法对比TabCompare）实现全面可解释评估，并构建TabXBench基准进行验证，展示其在多领域任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22176v1](http://arxiv.org/pdf/2505.22176v1)**

> **作者:** Vihang Pancholi; Jainit Bafna; Tejas Anvekar; Manish Shrivastava; Vivek Gupta
>
> **备注:** Accepeted for Findings at ACL 2025
>
> **摘要:** Evaluating tables qualitatively & quantitatively presents a significant challenge, as traditional metrics often fail to capture nuanced structural and content discrepancies. To address this, we introduce a novel, methodical rubric integrating multi-level structural descriptors with fine-grained contextual quantification, thereby establishing a robust foundation for comprehensive table comparison. Building on this foundation, we propose TabXEval, an eXhaustive and eXplainable two-phase evaluation framework. TabXEval initially aligns reference tables structurally via TabAlign & subsequently conducts a systematic semantic and syntactic comparison using TabCompare; this approach clarifies the evaluation process and pinpoints subtle discrepancies overlooked by conventional methods. The efficacy of this framework is assessed using TabXBench, a novel, diverse, multi-domain benchmark we developed, featuring realistic table perturbations and human-annotated assessments. Finally, a systematic analysis of existing evaluation methods through sensitivity-specificity trade-offs demonstrates the qualitative and quantitative effectiveness of TabXEval across diverse table-related tasks and domains, paving the way for future innovations in explainable table evaluation.
>
---
#### [new 035] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries
- **分类: cs.CL**

- **简介: 该论文属于多文档摘要生成任务，旨在解决大语言模型在处理多文档时因"中间迷失"导致覆盖不全及多样性不足的问题。提出三步方法：提取关键点、用DPP选择多样化内容、重写成摘要，并融入用户意图实现个性化，提升DiverseSumm基准测试中的源覆盖能力。**

- **链接: [http://arxiv.org/pdf/2505.21859v1](http://arxiv.org/pdf/2505.21859v1)**

> **作者:** Vishakh Padmakumar; Zichao Wang; David Arbour; Jennifer Healey
>
> **备注:** To appear at ACL 2025 - Main Conference
>
> **摘要:** While large language models (LLMs) are increasingly capable of handling longer contexts, recent work has demonstrated that they exhibit the "lost in the middle" phenomenon (Liu et al., 2024) of unevenly attending to different parts of the provided context. This hinders their ability to cover diverse source material in multi-document summarization, as noted in the DiverseSumm benchmark (Huang et al., 2024). In this work, we contend that principled content selection is a simple way to increase source coverage on this task. As opposed to prompting an LLM to perform the summarization in a single step, we explicitly divide the task into three steps -- (1) reducing document collections to atomic key points, (2) using determinantal point processes (DPP) to perform select key points that prioritize diverse content, and (3) rewriting to the final summary. By combining prompting steps, for extraction and rewriting, with principled techniques, for content selection, we consistently improve source coverage on the DiverseSumm benchmark across various LLMs. Finally, we also show that by incorporating relevance to a provided user intent into the DPP kernel, we can generate personalized summaries that cover relevant source information while retaining coverage.
>
---
#### [new 036] VeriTrail: Closed-Domain Hallucination Detection with Traceability
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VeriTrail方法，用于闭域幻觉检测及追踪多/单生成步骤（MGS/SGS）中虚假内容的来源。针对现有方法仅检测最终输出而忽略中间过程的问题，其构建含中间输出和人类标注的数据集，并通过实验验证方法优于基线。**

- **链接: [http://arxiv.org/pdf/2505.21786v1](http://arxiv.org/pdf/2505.21786v1)**

> **作者:** Dasha Metropolitansky; Jonathan Larson
>
> **摘要:** Even when instructed to adhere to source material, Language Models often generate unsubstantiated content - a phenomenon known as "closed-domain hallucination." This risk is amplified in processes with multiple generative steps (MGS), compared to processes with a single generative step (SGS). However, due to the greater complexity of MGS processes, we argue that detecting hallucinations in their final outputs is necessary but not sufficient: it is equally important to trace where hallucinated content was likely introduced and how faithful content may have been derived from the source through intermediate outputs. To address this need, we present VeriTrail, the first closed-domain hallucination detection method designed to provide traceability for both MGS and SGS processes. We also introduce the first datasets to include all intermediate outputs as well as human annotations of final outputs' faithfulness for their respective MGS processes. We demonstrate that VeriTrail outperforms baseline methods on both datasets.
>
---
#### [new 037] Seeing the Threat: Vulnerabilities in Vision-Language Models to Adversarial Attack
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型（LVLMs）对抗攻击漏洞。针对多模态模型易受视觉输入攻击的问题，提出系统性表征分析解释攻击绕过安全机制的原因，并设计两阶段评估框架（区分攻击类型及量化危害程度），最后提出安全对齐的规范标准。任务属模型安全领域，解决对抗攻击防护缺陷。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21967v1](http://arxiv.org/pdf/2505.21967v1)**

> **作者:** Juan Ren; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown remarkable capabilities across a wide range of multimodal tasks. However, their integration of visual inputs introduces expanded attack surfaces, thereby exposing them to novel security vulnerabilities. In this work, we conduct a systematic representational analysis to uncover why conventional adversarial attacks can circumvent the safety mechanisms embedded in LVLMs. We further propose a novel two stage evaluation framework for adversarial attacks on LVLMs. The first stage differentiates among instruction non compliance, outright refusal, and successful adversarial exploitation. The second stage quantifies the degree to which the model's output fulfills the harmful intent of the adversarial prompt, while categorizing refusal behavior into direct refusals, soft refusals, and partial refusals that remain inadvertently helpful. Finally, we introduce a normative schema that defines idealized model behavior when confronted with harmful prompts, offering a principled target for safety alignment in multimodal systems.
>
---
#### [new 038] 360-LLaMA-Factory: Plug & Play Sequence Parallelism for Long Post-Training
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出360-LLaMA-Factory框架，通过集成序列并行技术优化长序列模型的后训练效率。旨在解决大规模模型分布式训练中的序列处理瓶颈问题，开源实现并分析了不同序列并行模式，提供高效训练方案与技术实现经验。**

- **链接: [http://arxiv.org/pdf/2505.22296v1](http://arxiv.org/pdf/2505.22296v1)**

> **作者:** Haosheng Zou; Xiaowei Lv; Shousheng Jia; Xiangzheng Zhang
>
> **备注:** code at https://github.com/Qihoo360/360-LLaMA-Factory
>
> **摘要:** Adding sequence parallelism into LLaMA-Factory, we open-sourced 360-LLaMA-Factory at https://github.com/Qihoo360/360-LLaMA-Factory. 360-LLaMA-Factory has received wide recognition and used in models such as Light-R1 arXiv:2503.10460, TinyR1 arXiv:2503.04872, Kaggle AIMO math models and also in large companies' training frameworks. This technical report delves deeper into the different sequence parallel modes behind 360-LLaMA-Factory and discusses our implementation insights.
>
---
#### [new 039] Learning Composable Chains-of-Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何提升大语言模型的组合推理泛化能力，解决其依赖标注数据且难以组合基础技能解决新任务的问题。通过设计可组合的链式思维格式，结合多任务学习和少量数据微调（RFT），提升模型在无标注组合任务上的零样本性能，实验表明优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22635v1](http://arxiv.org/pdf/2505.22635v1)**

> **作者:** Fangcong Yin; Zeyu Leo Liu; Liu Leqi; Xi Ye; Greg Durrett
>
> **摘要:** A common approach for teaching large language models (LLMs) to reason is to train on chain-of-thought (CoT) traces of in-distribution reasoning problems, but such annotated data is costly to obtain for every problem of interest. We want reasoning models to generalize beyond their training distribution, and ideally to generalize compositionally: combine atomic reasoning skills to solve harder, unseen reasoning tasks. We take a step towards compositional generalization of reasoning skills when addressing a target compositional task that has no labeled CoT data. We find that simply training models on CoT data of atomic tasks leads to limited generalization, but minimally modifying CoT formats of constituent atomic tasks to be composable can lead to improvements. We can train "atomic CoT" models on the atomic tasks with Composable CoT data and combine them with multitask learning or model merging for better zero-shot performance on the target compositional task. Such a combined model can be further bootstrapped on a small amount of compositional data using rejection sampling fine-tuning (RFT). Results on string operations and natural language skill compositions show that training LLMs on Composable CoT outperforms multitask learning and continued fine-tuning baselines within a given training data budget.
>
---
#### [new 040] MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs
- **分类: cs.CL**

- **简介: 该论文提出MAKIEval框架，用于评估LLMs跨语言文化意识。针对LLMs因英语预训练导致的文化差异问题，其利用Wikidata自动识别文化实体，提出粒度、多样性等四指标，评估7种LLMs在13语言、19地区、6文化主题的表现，发现英语输出更具文化敏感性。**

- **链接: [http://arxiv.org/pdf/2505.21693v1](http://arxiv.org/pdf/2505.21693v1)**

> **作者:** Raoyuan Zhao; Beiduo Chen; Barbara Plank; Michael A. Hedderich
>
> **摘要:** Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge. We publicly release our code and data.
>
---
#### [new 041] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning
- **分类: cs.CL**

- **简介: 该论文提出GuessArena框架，通过游戏化对抗机制动态评估LLMs的领域知识与推理能力。针对传统静态基准无法适配领域及缺乏细粒度评估的问题，其整合动态知识建模与渐进推理测试，在五领域实验中有效区分模型性能，提升评估适应性与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.22661v1](http://arxiv.org/pdf/2505.22661v1)**

> **作者:** Qingchen Yu; Zifan Zheng; Ding Chen; Simin Niu; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** The evaluation of large language models (LLMs) has traditionally relied on static benchmarks, a paradigm that poses two major limitations: (1) predefined test sets lack adaptability to diverse application domains, and (2) standardized evaluation protocols often fail to capture fine-grained assessments of domain-specific knowledge and contextual reasoning abilities. To overcome these challenges, we propose GuessArena, an adaptive evaluation framework grounded in adversarial game-based interactions. Inspired by the interactive structure of the Guess Who I Am? game, our framework seamlessly integrates dynamic domain knowledge modeling with progressive reasoning assessment to improve evaluation fidelity. Empirical studies across five vertical domains-finance, healthcare, manufacturing, information technology, and education-demonstrate that GuessArena effectively distinguishes LLMs in terms of domain knowledge coverage and reasoning chain completeness. Compared to conventional benchmarks, our method provides substantial advantages in interpretability, scalability, and scenario adaptability.
>
---
#### [new 042] Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data
- **分类: cs.CL**

- **简介: 该论文研究利用基于访谈的LLMs生成调查响应，以整合定量与定性数据。通过BREQ问卷和访谈案例，评估LLMs预测人类响应的可靠性，发现其能捕捉整体模式但变异性较低，优化提示和模型参数可提升一致性，访谈内容比人口统计影响更大。指出LLMs在混合研究中的潜力及局限，如情感解读不足，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2505.21997v1](http://arxiv.org/pdf/2505.21997v1)**

> **作者:** Jihong Zhang; Xinya Liang; Anqi Deng; Nicole Bonge; Lin Tan; Ling Zhang; Nicole Zarrett
>
> **摘要:** Mixed methods research integrates quantitative and qualitative data but faces challenges in aligning their distinct structures, particularly in examining measurement characteristics and individual response patterns. Advances in large language models (LLMs) offer promising solutions by generating synthetic survey responses informed by qualitative data. This study investigates whether LLMs, guided by personal interviews, can reliably predict human survey responses, using the Behavioral Regulations in Exercise Questionnaire (BREQ) and interviews from after-school program staff as a case study. Results indicate that LLMs capture overall response patterns but exhibit lower variability than humans. Incorporating interview data improves response diversity for some models (e.g., Claude, GPT), while well-crafted prompts and low-temperature settings enhance alignment between LLM and human responses. Demographic information had less impact than interview content on alignment accuracy. These findings underscore the potential of interview-informed LLMs to bridge qualitative and quantitative methodologies while revealing limitations in response variability, emotional interpretation, and psychometric fidelity. Future research should refine prompt design, explore bias mitigation, and optimize model settings to enhance the validity of LLM-generated survey data in social science research.
>
---
#### [new 043] Multilingual vs Crosslingual Retrieval of Fact-Checked Claims: A Tale of Two Approaches
- **分类: cs.CL**

- **简介: 该论文研究多语言与跨语言事实核查声明检索任务，旨在提升不同语言环境下虚假信息识别的效率。针对低资源语言及全球性议题（如疫情、战争），提出通过负样本选择（监督学习）和LLM重排序（无监督）优化检索效果，在47种语言数据集验证，发现LLM重排序性能最优，揭示跨语言检索的独特性。**

- **链接: [http://arxiv.org/pdf/2505.22118v1](http://arxiv.org/pdf/2505.22118v1)**

> **作者:** Alan Ramponi; Marco Rovera; Robert Moro; Sara Tonelli
>
> **摘要:** Retrieval of previously fact-checked claims is a well-established task, whose automation can assist professional fact-checkers in the initial steps of information verification. Previous works have mostly tackled the task monolingually, i.e., having both the input and the retrieved claims in the same language. However, especially for languages with a limited availability of fact-checks and in case of global narratives, such as pandemics, wars, or international politics, it is crucial to be able to retrieve claims across languages. In this work, we examine strategies to improve the multilingual and crosslingual performance, namely selection of negative examples (in the supervised) and re-ranking (in the unsupervised setting). We evaluate all approaches on a dataset containing posts and claims in 47 languages (283 language combinations). We observe that the best results are obtained by using LLM-based re-ranking, followed by fine-tuning with negative examples sampled using a sentence similarity-based strategy. Most importantly, we show that crosslinguality is a setup with its own unique characteristics compared to the multilingual setup.
>
---
#### [new 044] Improving Continual Pre-training Through Seamless Data Packing
- **分类: cs.CL**

- **简介: 该论文属于持续预训练优化任务，旨在解决传统数据打包导致的上下文断裂和过度截断问题。提出Seamless Packing方法，通过滑动窗口保持序列连续性，并用FFD算法高效打包文本，减少填充/截断。实验显示其性能优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22018v1](http://arxiv.org/pdf/2505.22018v1)**

> **作者:** Ruicheng Yin; Xuan Gao; Changze Lv; Xiaohua Wang; Xiaoqing Zheng; Xuanjing Huang
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Continual pre-training has demonstrated significant potential in enhancing model performance, particularly in domain-specific scenarios. The most common approach for packing data before continual pre-training involves concatenating input texts and splitting them into fixed-length sequences. While straightforward and efficient, this method often leads to excessive truncation and context discontinuity, which can hinder model performance. To address these issues, we explore the potential of data engineering to enhance continual pre-training, particularly its impact on model performance and efficiency. We propose Seamless Packing (SP), a novel data packing strategy aimed at preserving contextual information more effectively and enhancing model performance. Our approach employs a sliding window technique in the first stage that synchronizes overlapping tokens across consecutive sequences, ensuring better continuity and contextual coherence. In the second stage, we adopt a First-Fit-Decreasing algorithm to pack shorter texts into bins slightly larger than the target sequence length, thereby minimizing padding and truncation. Empirical evaluations across various model architectures and corpus domains demonstrate the effectiveness of our method, outperforming baseline method in 99% of all settings. Code is available at https://github.com/Infernus-WIND/Seamless-Packing.
>
---
#### [new 045] RAG-Zeval: Towards Robust and Interpretable Evaluation on RAG Responses through End-to-End Rule-Guided Reasoning
- **分类: cs.CL**

- **简介: 该论文提出RAG-Zeval框架，针对RAG系统响应评估任务，解决现有LLM评估方法计算成本高、依赖复杂多阶段提示的问题。通过端到端规则引导推理，采用强化学习训练评估器，结合排名奖励机制和零标注合成参考数据，实现高效、可解释的评估，性能优于大模型基线，更贴近人类判断。**

- **链接: [http://arxiv.org/pdf/2505.22430v1](http://arxiv.org/pdf/2505.22430v1)**

> **作者:** Kun Li; Yunxiang Li; Tianhua Zhang; Hongyin Luo; Xixin Wu; James Glass; Helen Meng
>
> **摘要:** Robust evaluation is critical for deploying trustworthy retrieval-augmented generation (RAG) systems. However, current LLM-based evaluation frameworks predominantly rely on directly prompting resource-intensive models with complex multi-stage prompts, underutilizing models' reasoning capabilities and introducing significant computational cost. In this paper, we present RAG-Zeval (RAG-Zero Evaluator), a novel end-to-end framework that formulates faithfulness and correctness evaluation as a rule-guided reasoning task. Our approach trains evaluators with reinforcement learning, facilitating compact models to generate comprehensive and sound assessments with detailed explanation in one-pass. We introduce a ranking-based outcome reward mechanism, using preference judgments rather than absolute scores, to address the challenge of obtaining precise pointwise reward signals. To this end, we synthesize the ranking references by generating quality-controlled responses with zero human annotation. Experiments demonstrate RAG-Zeval's superior performance, achieving the strongest correlation with human judgments and outperforming baselines that rely on LLMs with 10-100 times more parameters. Our approach also exhibits superior interpretability in response evaluation.
>
---
#### [new 046] Test-Time Scaling with Repeated Sampling Improves Multilingual Text Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言文本生成任务，旨在通过测试时重复采样提升生成质量，解决其在推理任务（如数学、代码）中的有效性问题。研究在Aya和m-ArenaHard基准上评估了困惑度与奖励验证器，发现奖励验证器显著提升推理任务表现（部分超35%），强调验证器选择的重要性。**

- **链接: [http://arxiv.org/pdf/2505.21941v1](http://arxiv.org/pdf/2505.21941v1)**

> **作者:** Ashim Gupta; Vivek Srikumar
>
> **摘要:** Inference-time scaling via repeated sampling has shown promise in reasoning tasks, but its effectiveness in multilingual generation remains underexplored. We evaluate this approach using perplexity- and reward-based verifiers on two multilingual benchmarks: the Aya Evaluation Suite and m-ArenaHard. Our results show consistent quality improvements, with gains exceeding 35% in some cases. While perplexity-based scoring is effective for open-ended prompts, only reward-based verifiers improve performance on tasks requiring reasoning (e.g., math, code). Our results demonstrate the broader utility of repeated sampling for multilingual text generation and underscore the importance of selecting right verifiers for the task.
>
---
#### [new 047] Let's Predict Sentence by Sentence
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何使预训练语言模型（LM）从逐token生成过渡到基于句子的抽象推理。通过构建框架，将LM适配为预测下一句的连续嵌入（语义嵌入或上下文嵌入），并在离散/连续推理模式下测试，结果显示其在四领域性能接近CoT，且计算效率提升，同时提出诊断工具SentenceLens。任务属提升LM抽象推理能力，解决其与人类高阶思维的差异。**

- **链接: [http://arxiv.org/pdf/2505.22202v1](http://arxiv.org/pdf/2505.22202v1)**

> **作者:** Hyeonbin Hwang; Byeongguk Jeon; Seungone Kim; Jiyeon Kim; Hoyeon Chang; Sohee Yang; Seungpil Won; Dohaeng Lee; Youbin Ahn; Minjoon Seo
>
> **备注:** Work In Progress
>
> **摘要:** Autoregressive language models (LMs) generate one token at a time, yet human reasoning operates over higher-level abstractions - sentences, propositions, and concepts. This contrast raises a central question- Can LMs likewise learn to reason over structured semantic units rather than raw token sequences? In this work, we investigate whether pretrained LMs can be lifted into such abstract reasoning spaces by building on their learned representations. We present a framework that adapts a pretrained token-level LM to operate in sentence space by autoregressively predicting continuous embeddings of next sentences. We explore two embedding paradigms inspired by classical representation learning: 1) semantic embeddings, learned via autoencoding to preserve surface meaning; and 2) contextual embeddings, trained via next-sentence prediction to encode anticipatory structure. We evaluate both under two inference regimes: Discretized, which decodes each predicted embedding into text before re-encoding; and Continuous, which reasons entirely in embedding space for improved efficiency. Across four domains - mathematics, logic, commonsense, and planning - contextual embeddings under continuous inference show competitive performance with Chain-of-Thought (CoT) while reducing inference-time FLOPs on average by half. We also present early signs of scalability and modular adaptation. Finally, to visualize latent trajectories, we introduce SentenceLens, a diagnostic tool that decodes intermediate model states into interpretable sentences. Together, our results indicate that pretrained LMs can effectively transition to abstract, structured reasoning within latent embedding spaces.
>
---
#### [new 048] EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）的infilling任务，解决因prompt结构导致KV缓存复用效率低的问题。提出EFIM格式优化缓存复用，并引入分片token化训练改善子词生成，实验显示延迟降低52%、吞吐量提升98%。**

- **链接: [http://arxiv.org/pdf/2505.21889v1](http://arxiv.org/pdf/2505.21889v1)**

> **作者:** Tianyu Guo; Hande Dong; Yichong Leng; Feng Liu; Cheater Lin; Nong Xiao; Xianwei Zhang
>
> **摘要:** Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability.EFIM's source code is publicly available at https://github.com/gty111/EFIM.
>
---
#### [new 049] Revisiting Common Assumptions about Arabic Dialects in NLP
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语NLP领域，针对方言划分假设（如地域分组）未经验证的问题，构建多标签数据集（含11国方言人工标注），验证四个常见假设，发现其过度简化现实，可能阻碍方言识别等任务进展。**

- **链接: [http://arxiv.org/pdf/2505.21816v1](http://arxiv.org/pdf/2505.21816v1)**

> **作者:** Amr Keleg; Sharon Goldwater; Walid Magdy
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Arabic has diverse dialects, where one dialect can be substantially different from the others. In the NLP literature, some assumptions about these dialects are widely adopted (e.g., ``Arabic dialects can be grouped into distinguishable regional dialects") and are manifested in different computational tasks such as Arabic Dialect Identification (ADI). However, these assumptions are not quantitatively verified. We identify four of these assumptions and examine them by extending and analyzing a multi-label dataset, where the validity of each sentence in 11 different country-level dialects is manually assessed by speakers of these dialects. Our analysis indicates that the four assumptions oversimplify reality, and some of them are not always accurate. This in turn might be hindering further progress in different Arabic NLP tasks.
>
---
#### [new 050] Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文聚焦多模态推理任务，旨在解决强化学习（RL）冷启动阶段推理性能不足的问题。提出两阶段方法：先通过监督微调（SFT）建立结构化推理模式，再结合GRPO强化学习优化，显著提升模型性能。实验显示其开源模型在MathVista等基准测试中达SOTA，7B模型性能提升超7%，3B模型表现媲美7B模型。**

- **链接: [http://arxiv.org/pdf/2505.22334v1](http://arxiv.org/pdf/2505.22334v1)**

> **作者:** Lai Wei; Yuting Li; Kaipeng Zheng; Chen Wang; Yue Wang; Linghe Kong; Lichao Sun; Weiran Huang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at https://github.com/waltonfuture/RL-with-Cold-Start.
>
---
#### [new 051] Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 该论文提出Agent-UniRAG框架，解决现有RAG系统无法统一处理单/多跳查询的问题。通过设计基于LLM的可训练代理，分步骤处理输入复杂度，同时支持单/多跳任务，并构建SynAgent-RAG数据集优化小模型性能，实验表明其表现与大型模型相当。**

- **链接: [http://arxiv.org/pdf/2505.22571v1](http://arxiv.org/pdf/2505.22571v1)**

> **作者:** Hoang Pham; Khac-Hoai Nam Bui
>
> **摘要:** This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
>
---
#### [new 052] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳问答（MHQA）任务，旨在解决大语言模型在复杂推理中的证据整合与逻辑依赖问题。针对RAG方法在噪声过滤和必要证据检索上的不足，提出RISE框架，通过问题分解、检索-阅读及自我批判的迭代自探索步骤，提升模型推理路径识别与逻辑一致性，增强MHQA性能。**

- **链接: [http://arxiv.org/pdf/2505.21940v1](http://arxiv.org/pdf/2505.21940v1)**

> **作者:** Bolei He; Xinran He; Mengke Chen; Xianwei Xue; Ying Zhu; Zhenhua Ling
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) excel in many areas but continue to face challenges with complex reasoning tasks, such as Multi-Hop Question Answering (MHQA). MHQA requires integrating evidence from diverse sources while managing intricate logical dependencies, often leads to errors in reasoning. Retrieval-Augmented Generation (RAG), widely employed in MHQA tasks, faces challenges in effectively filtering noisy data and retrieving all necessary evidence, thereby limiting its effectiveness in addressing MHQA challenges. To address these challenges, we propose RISE:Reasoning Enhancement via Iterative Self-Exploration, a novel framework designed to enhance models' reasoning capability through iterative self-exploration. Specifically, RISE involves three key steps in addressing MHQA tasks: question decomposition, retrieve-then-read, and self-critique. By leveraging continuous self-exploration, RISE identifies accurate reasoning paths, iteratively self-improving the model's capability to integrate evidence, maintain logical consistency, and enhance performance in MHQA tasks. Extensive experiments on multiple MHQA benchmarks demonstrate that RISE significantly improves reasoning accuracy and task performance.
>
---
#### [new 053] LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LaMDAgent框架，解决后训练流程（如微调、偏好学习等）依赖人工设计及孤立优化的问题。通过LLM代理自主探索模型生成技术、数据集及超参数配置，构建优化完整pipeline，提升工具使用精度并发现高效策略，分析数据与模型规模对探索成本的影响。**

- **链接: [http://arxiv.org/pdf/2505.21963v1](http://arxiv.org/pdf/2505.21963v1)**

> **作者:** Taro Yano; Yoichi Ishibashi; Masafumi Oyamada
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks. To further tailor LLMs to specific domains or applications, post-training techniques such as Supervised Fine-Tuning (SFT), Preference Learning, and model merging are commonly employed. While each of these methods has been extensively studied in isolation, the automated construction of complete post-training pipelines remains an underexplored area. Existing approaches typically rely on manual design or focus narrowly on optimizing individual components, such as data ordering or merging strategies. In this work, we introduce LaMDAgent (short for Language Model Developing Agent), a novel framework that autonomously constructs and optimizes full post-training pipelines through the use of LLM-based agents. LaMDAgent systematically explores diverse model generation techniques, datasets, and hyperparameter configurations, leveraging task-based feedback to discover high-performing pipelines with minimal human intervention. Our experiments show that LaMDAgent improves tool-use accuracy by 9.0 points while preserving instruction-following capabilities. Moreover, it uncovers effective post-training strategies that are often overlooked by conventional human-driven exploration. We further analyze the impact of data and model size scaling to reduce computational costs on the exploration, finding that model size scalings introduces new challenges, whereas scaling data size enables cost-effective pipeline discovery.
>
---
#### [new 054] Advancing Expert Specialization for Better MoE
- **分类: cs.CL; cs.SE; 68T07; I.2.7**

- **简介: 该论文属于MoE模型优化任务，旨在解决现有辅助负载平衡损失导致专家重叠和路由均匀化的问题。提出正交性损失和方差损失，促进专家专业化与路由区分，实验显示提升性能达23.79%且保持负载均衡，无需架构修改。**

- **链接: [http://arxiv.org/pdf/2505.22323v1](http://arxiv.org/pdf/2505.22323v1)**

> **作者:** Hongcan Guo; Haolang Lu; Guoshun Nan; Bolun Chu; Jialin Zhuang; Yuan Yang; Wenhao Che; Sicong Leng; Qimei Cui; Xudong Jiang
>
> **备注:** 33pages, 6figures
>
> **摘要:** Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training. To address this, we propose a simple yet effective solution that introduces two complementary objectives: (1) an orthogonality loss to encourage experts to process distinct types of tokens, and (2) a variance loss to encourage more discriminative routing decisions. Gradient-level analysis demonstrates that these objectives are compatible with the existing auxiliary loss and contribute to optimizing the training process. Experimental results over various model architectures and across multiple benchmarks show that our method significantly enhances expert specialization. Notably, our method improves classic MoE baselines with auxiliary loss by up to 23.79%, while also maintaining load balancing in downstream tasks, without any architectural modifications or additional components. We will release our code to contribute to the community.
>
---
#### [new 055] Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文提出ToxEdit方法，解决LLMs毒性净化中对实体依赖和过度编辑问题。通过动态检测毒性激活模式并采用自适应层间路径，精准抑制毒性同时保留模型能力。改进SafeEdit基准，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22298v1](http://arxiv.org/pdf/2505.22298v1)**

> **作者:** Yifan Lu; Jing Li; Yigeng Zhou; Yihui Zhang; Wenya Wang; Xiucheng Li; Meishan Zhang; Fangming Liu; Jun Yu; Min Zhang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs.
>
---
#### [new 056] Loquacious Set: 25,000 Hours of Transcribed and Diverse English Speech Recognition Data for Research and Commercial Use
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文构建ASR数据集，解决现有数据集规模小、场景单一及许可限制问题。提出Loquacious Set：25000小时商业可用英语语音，涵盖多口音、朗读/自发/嘈杂等场景，助力学术与工业界研发真实环境ASR系统。（98字）**

- **链接: [http://arxiv.org/pdf/2505.21578v1](http://arxiv.org/pdf/2505.21578v1)**

> **作者:** Titouan Parcollet; Yuan Tseng; Shucong Zhang; Rogier van Dalen
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) research is driven by the availability of common datasets between industrial researchers and academics, encouraging comparisons and evaluations. LibriSpeech, despite its long success as an ASR benchmark, is now limited by its size and focus on clean, read speech, leading to near-zero word error rates. More recent datasets, including MOSEL, YODAS, Gigaspeech, OWSM, Libriheavy or People's Speech suffer from major limitations including licenses that researchers in the industry cannot use, unreliable transcriptions, incorrect audio data, or the lack of evaluation sets. This work presents the Loquacious Set, a 25,000-hour curated collection of commercially usable English speech. Featuring hundreds of thousands of speakers with diverse accents and a wide range of speech types (read, spontaneous, talks, clean, noisy), the Loquacious Set is designed to work for academics and researchers in the industry to build ASR systems in real-world scenarios.
>
---
#### [new 057] EULER: Enhancing the Reasoning Ability of Large Language Models through Error-Induced Learning
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型（LLMs）推理能力优化任务，旨在解决现有数学问题解决中错误样本生成不足的问题。提出EULER模型，通过优化错误生成机制提升LLMs自生错误的概率，并利用更优模型的解决方案规范生成质量，实验显示其数学推理性能提升超4%。**

- **链接: [http://arxiv.org/pdf/2505.22131v1](http://arxiv.org/pdf/2505.22131v1)**

> **作者:** Zhuoyang Wu; Xinze Li; Zhenghao Liu; Yukun Yan; Zhiyuan Liu; Minghe Yu; Cheng Yang; Yu Gu; Ge Yu; Maosong Sun
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong reasoning capabilities and achieved promising results in mathematical problem-solving tasks. Learning from errors offers the potential to further enhance the performance of LLMs during Supervised Fine-Tuning (SFT). However, the errors in synthesized solutions are typically gathered from sampling trails, making it challenging to generate solution errors for each mathematical problem. This paper introduces the Error-IndUced LEaRning (EULER) model, which aims to develop an error exposure model that generates high-quality solution errors to enhance the mathematical reasoning capabilities of LLMs. Specifically, EULER optimizes the error exposure model to increase the generation probability of self-made solution errors while utilizing solutions produced by a superior LLM to regularize the generation quality. Our experiments across various mathematical problem datasets demonstrate the effectiveness of the EULER model, achieving an improvement of over 4% compared to all baseline models. Further analysis reveals that EULER is capable of synthesizing more challenging and educational solution errors, which facilitate both the training and inference processes of LLMs. All codes are available at https://github.com/NEUIR/EULER.
>
---
#### [new 058] Chain-of-Talkers (CoTalk): Fast Human Annotation of Dense Image Captions
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出CoTalk方法，优化密集图像标注任务，通过顺序标注减少冗余、多模态界面提升效率，在固定预算下提高标注数量与全面性。实验显示其速度（0.42 vs 0.30单位/秒）和检索性能（41.13% vs 40.52%）优于并行方法。**

- **链接: [http://arxiv.org/pdf/2505.22627v1](http://arxiv.org/pdf/2505.22627v1)**

> **作者:** Yijun Shen; Delong Chen; Fan Liu; Xingyu Wang; Chuanyi Zhang; Liang Yao; Yuhui Zheng
>
> **摘要:** While densely annotated image captions significantly facilitate the learning of robust vision-language alignment, methodologies for systematically optimizing human annotation efforts remain underexplored. We introduce Chain-of-Talkers (CoTalk), an AI-in-the-loop methodology designed to maximize the number of annotated samples and improve their comprehensiveness under fixed budget constraints (e.g., total human annotation time). The framework is built upon two key insights. First, sequential annotation reduces redundant workload compared to conventional parallel annotation, as subsequent annotators only need to annotate the ``residual'' -- the missing visual information that previous annotations have not covered. Second, humans process textual input faster by reading while outputting annotations with much higher throughput via talking; thus a multimodal interface enables optimized efficiency. We evaluate our framework from two aspects: intrinsic evaluations that assess the comprehensiveness of semantic units, obtained by parsing detailed captions into object-attribute trees and analyzing their effective connections; extrinsic evaluation measures the practical usage of the annotated captions in facilitating vision-language alignment. Experiments with eight participants show our Chain-of-Talkers (CoTalk) improves annotation speed (0.42 vs. 0.30 units/sec) and retrieval performance (41.13\% vs. 40.52\%) over the parallel method.
>
---
#### [new 059] InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing
- **分类: cs.CL**

- **简介: 论文提出InComeS框架，属于LLM模型编辑任务。针对现有方法在复杂场景中因上下文窗口限制导致效率下降的问题，通过压缩编辑信息为gist token缓存，并利用动态选择模块提取关键信息，实现高效灵活的多编辑处理。**

- **链接: [http://arxiv.org/pdf/2505.22156v1](http://arxiv.org/pdf/2505.22156v1)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yang Deng; Chenlong Deng; Tianqing Fang; Hongming Zhang; Haitao Mi; Dong Yu; Wai Lam
>
> **备注:** Under review
>
> **摘要:** Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
>
---
#### [new 060] Characterizing Bias: Benchmarking Large Language Models in Simplified versus Traditional Chinese
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大型语言模型（LLMs）在简体与繁体中文中的表现差异及潜在偏差。通过设计区域术语选择（如地名/物品名称差异）和姓名招聘任务，测试11种LLM，发现模型在术语任务偏向简体中文，姓名任务偏向繁体中文，归因于训练数据与分词差异，并开源基准数据集促进后续研究。**

- **链接: [http://arxiv.org/pdf/2505.22645v1](http://arxiv.org/pdf/2505.22645v1)**

> **作者:** Hanjia Lyu; Jiebo Luo; Jian Kang; Allison Koenecke
>
> **备注:** To appear in the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT '25)
>
> **摘要:** While the capabilities of Large Language Models (LLMs) have been studied in both Simplified and Traditional Chinese, it is yet unclear whether LLMs exhibit differential performance when prompted in these two variants of written Chinese. This understanding is critical, as disparities in the quality of LLM responses can perpetuate representational harms by ignoring the different cultural contexts underlying Simplified versus Traditional Chinese, and can exacerbate downstream harms in LLM-facilitated decision-making in domains such as education or hiring. To investigate potential LLM performance disparities, we design two benchmark tasks that reflect real-world scenarios: regional term choice (prompting the LLM to name a described item which is referred to differently in Mainland China and Taiwan), and regional name choice (prompting the LLM to choose who to hire from a list of names in both Simplified and Traditional Chinese). For both tasks, we audit the performance of 11 leading commercial LLM services and open-sourced models -- spanning those primarily trained on English, Simplified Chinese, or Traditional Chinese. Our analyses indicate that biases in LLM responses are dependent on both the task and prompting language: while most LLMs disproportionately favored Simplified Chinese responses in the regional term choice task, they surprisingly favored Traditional Chinese names in the regional name choice task. We find that these disparities may arise from differences in training data representation, written character preferences, and tokenization of Simplified and Traditional Chinese. These findings highlight the need for further analysis of LLM biases; as such, we provide an open-sourced benchmark dataset to foster reproducible evaluations of future LLM behavior across Chinese language variants (https://github.com/brucelyu17/SC-TC-Bench).
>
---
#### [new 061] LLMPR: A Novel LLM-Driven Transfer Learning based Petition Ranking Model
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出LLMPR模型，利用迁移学习和机器学习，通过文本嵌入（如DistilBERT）和数值特征（如案件滞留天数、字数）对法律请愿自动排序，解决印度司法系统案件积压与人工排序低效、主观的问题。实验表明随机森林等模型准确率达99%，数值特征主导排序效果，证明自动化可提升司法效率与公平性。**

- **链接: [http://arxiv.org/pdf/2505.21689v1](http://arxiv.org/pdf/2505.21689v1)**

> **作者:** Avijit Gayen; Somyajit Chakraborty; Mainak Sen; Soham Paul; Angshuman Jana
>
> **备注:** 28 pages, 5 figures, journal paper, submitted to AI and Law
>
> **摘要:** The persistent accumulation of unresolved legal cases, especially within the Indian judiciary, significantly hampers the timely delivery of justice. Manual methods of prioritizing petitions are often prone to inefficiencies and subjective biases further exacerbating delays. To address this issue, we propose LLMPR (Large Language Model-based Petition Ranking), an automated framework that utilizes transfer learning and machine learning to assign priority rankings to legal petitions based on their contextual urgency. Leveraging the ILDC dataset comprising 7,593 annotated petitions, we process unstructured legal text and extract features through various embedding techniques, including DistilBERT, LegalBERT, and MiniLM. These textual embeddings are combined with quantitative indicators such as gap days, rank scores, and word counts to train multiple machine learning models, including Random Forest, Decision Tree, XGBoost, LightGBM, and CatBoost. Our experiments demonstrate that Random Forest and Decision Tree models yield superior performance, with accuracy exceeding 99% and a Spearman rank correlation of 0.99. Notably, models using only numerical features achieve nearly optimal ranking results (R2 = 0.988, \r{ho} = 0.998), while LLM-based embeddings offer only marginal gains. These findings suggest that automated petition ranking can effectively streamline judicial workflows, reduce case backlog, and improve fairness in legal prioritization.
>
---
#### [new 062] MRT at SemEval-2025 Task 8: Maximizing Recovery from Tables with Multiple Steps
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对SemEval-2025任务8（表格数据问答），提出多步骤方法解决表格问答问题。通过LLM生成自然语言步骤并转化为Python代码，结合开源模型与优化提示，处理表格理解、代码执行及错误修正，实现子任务1准确率70.5%。**

- **链接: [http://arxiv.org/pdf/2505.22264v1](http://arxiv.org/pdf/2505.22264v1)**

> **作者:** Maximiliano Hormazábal Lagos; Álvaro Bueno Saez; Héctor Cerezo-Costas; Pedro Alonso Doval; Jorge Alcalde Vesteiro
>
> **备注:** 7 pages, 6 tables
>
> **摘要:** In this paper we expose our approach to solve the \textit{SemEval 2025 Task 8: Question-Answering over Tabular Data} challenge. Our strategy leverages Python code generation with LLMs to interact with the table and get the answer to the questions. The process is composed of multiple steps: understanding the content of the table, generating natural language instructions in the form of steps to follow in order to get the answer, translating these instructions to code, running it and handling potential errors or exceptions. These steps use open source LLMs and fine grained optimized prompts for each task (step). With this approach, we achieved a score of $70.50\%$ for subtask 1.
>
---
#### [new 063] Representative Language Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出"代表生成"框架，扩展生成模型理论以解决多样性和偏见问题，要求输出按比例反映训练数据中的群体。引入"群组闭包维度"，分析均匀/非均匀生成的可行性，证明无限假设类下理论可行但 membership query 计算不可行，为开发公平生成模型提供理论基础。**

- **链接: [http://arxiv.org/pdf/2505.21819v1](http://arxiv.org/pdf/2505.21819v1)**

> **作者:** Charlotte Peale; Vinod Raman; Omer Reingold
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** We introduce "representative generation," extending the theoretical framework for generation proposed by Kleinberg et al. (2024) and formalized by Li et al. (2024), to additionally address diversity and bias concerns in generative models. Our notion requires outputs of a generative model to proportionally represent groups of interest from the training data. We characterize representative uniform and non-uniform generation, introducing the "group closure dimension" as a key combinatorial quantity. For representative generation in the limit, we analyze both information-theoretic and computational aspects, demonstrating feasibility for countably infinite hypothesis classes and collections of groups under certain conditions, but proving a negative result for computability using only membership queries. This contrasts with Kleinberg et al.'s (2024) positive results for standard generation in the limit. Our findings provide a rigorous foundation for developing more diverse and representative generative models.
>
---
#### [new 064] Knowledge Base Construction for Knowledge-Augmented Text-to-SQL
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本到SQL（Text-to-SQL）任务，旨在解决大语言模型（LLMs）因知识局限导致生成SQL语句不准确的问题。提出构建综合知识库，整合所有可用问题、数据库模式及关联知识，支持跨领域复用，实验显示其性能优于基线方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22096v1](http://arxiv.org/pdf/2505.22096v1)**

> **作者:** Jinheon Baek; Horst Samulowitz; Oktie Hassanzadeh; Dharmashankar Subramanian; Sola Shirai; Alfio Gliozzo; Debarun Bhattacharjya
>
> **备注:** ACL Findings 2025
>
> **摘要:** Text-to-SQL aims to translate natural language queries into SQL statements, which is practical as it enables anyone to easily retrieve the desired information from databases. Recently, many existing approaches tackle this problem with Large Language Models (LLMs), leveraging their strong capability in understanding user queries and generating corresponding SQL code. Yet, the parametric knowledge in LLMs might be limited to covering all the diverse and domain-specific queries that require grounding in various database schemas, which makes generated SQLs less accurate oftentimes. To tackle this, we propose constructing the knowledge base for text-to-SQL, a foundational source of knowledge, from which we retrieve and generate the necessary knowledge for given queries. In particular, unlike existing approaches that either manually annotate knowledge or generate only a few pieces of knowledge for each query, our knowledge base is comprehensive, which is constructed based on a combination of all the available questions and their associated database schemas along with their relevant knowledge, and can be reused for unseen databases from different datasets and domains. We validate our approach on multiple text-to-SQL datasets, considering both the overlapping and non-overlapping database scenarios, where it outperforms relevant baselines substantially.
>
---
#### [new 065] Precise In-Parameter Concept Erasure in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型概念擦除任务，旨在解决现有方法（如微调、低秩适配）擦除效果粗糙或不彻底的问题。提出PISCES框架，通过解缠模型分解MLP特征，定位并移除参数中编码目标概念的成分，实验显示其擦除精度达7.7%，特异性提升31%，鲁棒性提升38%。**

- **链接: [http://arxiv.org/pdf/2505.22586v1](http://arxiv.org/pdf/2505.22586v1)**

> **作者:** Yoav Gur-Arieh; Clara Suslik; Yihuai Hong; Fazl Barez; Mor Geva
>
> **摘要:** Large language models (LLMs) often acquire knowledge during pretraining that is undesirable in downstream deployments, e.g., sensitive information or copyrighted content. Existing approaches for removing such knowledge rely on fine-tuning, training low-rank adapters or fact-level editing, but these are either too coarse, too shallow, or ineffective. In this work, we propose PISCES (Precise In-parameter Suppression for Concept EraSure), a novel framework for precisely erasing entire concepts from model parameters by directly editing directions that encode them in parameter space. PISCES uses a disentangler model to decompose MLP vectors into interpretable features, identifies those associated with a target concept using automated interpretability techniques, and removes them from model parameters. Experiments on Gemma 2 and Llama 3.1 over various concepts show that PISCES achieves modest gains in efficacy over leading erasure methods, reducing accuracy on the target concept to as low as 7.7%, while dramatically improving erasure specificity (by up to 31%) and robustness (by up to 38%). Overall, these results demonstrate that feature-based in-parameter editing enables a more precise and reliable approach for removing conceptual knowledge in language models.
>
---
#### [new 066] Jailbreak Distillation: Renewable Safety Benchmarking
- **分类: cs.CL; cs.CR; cs.SE**

- **简介: 该论文提出Jailbreak Distillation（JBDistill）框架，用于构建可更新的LLM安全评估基准。针对现有安全评估中基准过时、污染及人工成本高的问题，其通过小规模模型和攻击算法生成候选提示池，再自动筛选优质提示作为基准，确保公平性和可重复性。实验显示其在多样模型上效果显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22037v1](http://arxiv.org/pdf/2505.22037v1)**

> **作者:** Jingyu Zhang; Ahmed Elgohary; Xiawei Wang; A S M Iftekhar; Ahmed Magooda; Benjamin Van Durme; Daniel Khashabi; Kyle Jackson
>
> **备注:** Project page: https://aka.ms/jailbreak-distillation
>
> **摘要:** Large language models (LLMs) are rapidly deployed in critical applications, raising urgent needs for robust safety benchmarking. We propose Jailbreak Distillation (JBDistill), a novel benchmark construction framework that "distills" jailbreak attacks into high-quality and easily-updatable safety benchmarks. JBDistill utilizes a small set of development models and existing jailbreak attack algorithms to create a candidate prompt pool, then employs prompt selection algorithms to identify an effective subset of prompts as safety benchmarks. JBDistill addresses challenges in existing safety evaluation: the use of consistent evaluation prompts across models ensures fair comparisons and reproducibility. It requires minimal human effort to rerun the JBDistill pipeline and produce updated benchmarks, alleviating concerns on saturation and contamination. Extensive experiments demonstrate our benchmarks generalize robustly to 13 diverse evaluation models held out from benchmark construction, including proprietary, specialized, and newer-generation LLMs, significantly outperforming existing safety benchmarks in effectiveness while maintaining high separability and diversity. Our framework thus provides an effective, sustainable, and adaptable solution for streamlining safety evaluation.
>
---
#### [new 067] THINK-Bench: Evaluating Thinking Efficiency and Chain-of-Thought Quality of Large Reasoning Models
- **分类: cs.CL**

- **简介: 论文提出Think-Bench基准，评估大型推理模型（LRMs）的思考效率与思维链质量，解决其过度思考导致计算资源浪费问题。通过新指标多维度分析，发现多数LRMs在简单任务中生成冗余推理链，虽思维质量高但效率低，为优化LRMs提供基础。**

- **链接: [http://arxiv.org/pdf/2505.22113v1](http://arxiv.org/pdf/2505.22113v1)**

> **作者:** Zhiyuan Li; Yi Chang; Yuan Wu
>
> **备注:** 20 pages, 8 figures, 6 tables
>
> **摘要:** Large reasoning models (LRMs) have achieved impressive performance in complex tasks, often outperforming conventional large language models (LLMs). However, the prevalent issue of overthinking severely limits their computational efficiency. Overthinking occurs when models generate excessive and redundant tokens that contribute little to accurate outcomes, especially in simple tasks, resulting in a significant waste of computational resources. To systematically investigate this issue, we introduce Think-Bench, a benchmark designed to evaluate the reasoning efficiency of LRMs. We also propose novel efficiency metrics and conduct a comprehensive evaluation of various LRMs across multiple dimensions, including the reasoning process, outcome quality, and chain-of-thought (CoT) characteristics. Our analysis reveals that most LRMs exhibit overthinking in handling easy questions, generating unnecessarily lengthy reasoning chains. While many LRMs demonstrate high CoT quality, several suffer from low efficiency. We hope that Think-Bench can serve as a robust foundation for advancing research into LRMs.
>
---
#### [new 068] Reverse Preference Optimization for Complex Instruction Following
- **分类: cs.CL**

- **简介: 该论文属于复杂指令遵循任务，针对多约束场景下现有偏好优化方法存在的噪声问题（如选中响应未满足部分约束），提出Reverse Preference Optimization（RPO）方法。通过动态反转指令约束确保选中响应的完美性，扩大正反例差距，提升模型鲁棒性。实验显示其在多轮指令基准测试中显著优于DPO基线，且规模扩展性优异。**

- **链接: [http://arxiv.org/pdf/2505.22172v1](http://arxiv.org/pdf/2505.22172v1)**

> **作者:** Xiang Huang; Ting-En Lin; Feiteng Fang; Yuchuan Wu; Hangyu Li; Yuzhong Qu; Fei Huang; Yongbin Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Instruction following (IF) is a critical capability for large language models (LLMs). However, handling complex instructions with multiple constraints remains challenging. Previous methods typically select preference pairs based on the number of constraints they satisfy, introducing noise where chosen examples may fail to follow some constraints and rejected examples may excel in certain respects over the chosen ones. To address the challenge of aligning with multiple preferences, we propose a simple yet effective method called Reverse Preference Optimization (RPO). It mitigates noise in preference pairs by dynamically reversing the constraints within the instruction to ensure the chosen response is perfect, alleviating the burden of extensive sampling and filtering to collect perfect responses. Besides, reversal also enlarges the gap between chosen and rejected responses, thereby clarifying the optimization direction and making it more robust to noise. We evaluate RPO on two multi-turn IF benchmarks, Sysbench and Multi-IF, demonstrating average improvements over the DPO baseline of 4.6 and 2.5 points (on Llama-3.1 8B), respectively. Moreover, RPO scales effectively across model sizes (8B to 70B parameters), with the 70B RPO model surpassing GPT-4o.
>
---
#### [new 069] Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs
- **分类: cs.CL**

- **简介: 该论文属于情绪理解任务，旨在解决现有LLM因固定CoT推理无法适应情感复杂度的问题。提出自适应推理框架Emotion-o1，利用DeepSeek-R1生成变长推理链，并结合强化学习优化预测精度、推理深度、路径多样性和逻辑重复抑制，显著提升情感、幽默等任务的Acc和F1值。**

- **链接: [http://arxiv.org/pdf/2505.22548v1](http://arxiv.org/pdf/2505.22548v1)**

> **作者:** Changhao Song; Yazhou Zhang; Peng Zhang
>
> **摘要:** Emotion understanding includes basic tasks (e.g., sentiment/emotion classification) and advanced tasks (e.g., sarcasm/humor detection). Current methods rely on fixed-length CoT reasoning, failing to adapt to the varying complexity of emotions. We propose a task-adaptive reasoning framework that employs DeepSeek-R1 to generate variable-length reasoning chains for different emotion tasks. By combining fine-tuning with reinforcement learning, we design a composite reward function that balances four objectives: prediction accuracy, adaptive reasoning depth control, structural diversity in reasoning paths, and suppression of repetitive logic. This approach achieves dynamic context-sensitive inference while enabling LLMs to autonomously develop deep reasoning capabilities. Experimental results demonstrate consistent improvements in both Acc and F1 scores across four tasks: emotion, sentiment, humor, and sarcasm. Notably, peak enhancements reached 3.56% F1 (2.76% Acc) for basic tasks and 37.95% F1 (23.14% Acc) for advanced tasks. Our work bridges rigid CoT reasoning and emotional complexity through adaptive-depth analysis.
>
---
#### [new 070] Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，针对RAG系统中私有检索数据易受成员推断攻击（MIA）的问题，提出Mirabel框架。通过检测查询与目标文档的高相似性，采用"detect-and-hide"策略防御攻击，实验证明其有效且兼容现有系统。**

- **链接: [http://arxiv.org/pdf/2505.22061v1](http://arxiv.org/pdf/2505.22061v1)**

> **作者:** Yujin Choi; Youngjoo Park; Junyoung Byun; Jaewook Lee; Jinseong Park
>
> **摘要:** Retrieval-augmented generation (RAG) mitigates the hallucination problem in large language models (LLMs) and has proven effective for specific, personalized applications. However, passing private retrieved documents directly to LLMs introduces vulnerability to membership inference attacks (MIAs), which try to determine whether the target datum exists in the private external database or not. Based on the insight that MIA queries typically exhibit high similarity to only one target document, we introduce Mirabel, a similarity-based MIA detection framework designed for the RAG system. With the proposed Mirabel, we show that simple detect-and-hide strategies can successfully obfuscate attackers, maintain data utility, and remain system-agnostic. We experimentally prove its detection and defense against various state-of-the-art MIA methods and its adaptability to existing private RAG systems.
>
---
#### [new 071] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本生成任务，旨在解决离散与连续扩散模型的局限性。离散模型缺乏精细控制，连续模型因均匀扩散难以捕捉语义细节。提出NeoDiff，结合两者优势，采用Poisson扩散实现灵活噪声添加，并通过时间预测器自适应去噪，优化推理调度，实验显示性能更优。**

- **链接: [http://arxiv.org/pdf/2505.22165v1](http://arxiv.org/pdf/2505.22165v1)**

> **作者:** Bocheng Li; Zhujin Gao; Linli Xu
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Diffusion models have emerged as a promising approach for text generation, with recent works falling into two main categories: discrete and continuous diffusion models. Discrete diffusion models apply token corruption independently using categorical distributions, allowing for different diffusion progress across tokens but lacking fine-grained control. Continuous diffusion models map tokens to continuous spaces and apply fine-grained noise, but the diffusion progress is uniform across tokens, limiting their ability to capture semantic nuances. To address these limitations, we propose \textbf{\underline{N}}on-simultan\textbf{\underline{e}}ous C\textbf{\underline{o}}ntinuous \textbf{\underline{Diff}}usion Models (NeoDiff), a novel diffusion model that integrates the strengths of both discrete and continuous approaches. NeoDiff introduces a Poisson diffusion process for the forward process, enabling a flexible and fine-grained noising paradigm, and employs a time predictor for the reverse process to adaptively modulate the denoising progress based on token semantics. Furthermore, NeoDiff utilizes an optimized schedule for inference to ensure more precise noise control and improved performance. Our approach unifies the theories of discrete and continuous diffusion models, offering a more principled and effective framework for text generation. Experimental results on several text generation tasks demonstrate NeoDiff's superior performance compared to baselines of non-autoregressive continuous and discrete diffusion models, iterative-based methods and autoregressive diffusion-based methods. These results highlight NeoDiff's potential as a powerful tool for generating high-quality text and advancing the field of diffusion-based text generation.
>
---
#### [new 072] Graph-Assisted Culturally Adaptable Idiomatic Translation for Indic Languages
- **分类: cs.CL**

- **简介: 该论文提出IdiomCE方法，针对印度语言习语翻译中的一对多映射和文化差异问题，通过自适应图神经网络学习习语间复杂关系，提升资源匮乏场景下的翻译质量。任务为跨文化习语翻译，解决传统静态知识图与提示方法的不足。**

- **链接: [http://arxiv.org/pdf/2505.21937v1](http://arxiv.org/pdf/2505.21937v1)**

> **作者:** Pratik Rakesh Singh; Kritarth Prasad; Mohammadi Zaki; Pankaj Wasnik
>
> **摘要:** Translating multi-word expressions (MWEs) and idioms requires a deep understanding of the cultural nuances of both the source and target languages. This challenge is further amplified by the one-to-many nature of idiomatic translations, where a single source idiom can have multiple target-language equivalents depending on cultural references and contextual variations. Traditional static knowledge graphs (KGs) and prompt-based approaches struggle to capture these complex relationships, often leading to suboptimal translations. To address this, we propose IdiomCE, an adaptive graph neural network (GNN) based methodology that learns intricate mappings between idiomatic expressions, effectively generalizing to both seen and unseen nodes during training. Our proposed method enhances translation quality even in resource-constrained settings, facilitating improved idiomatic translation in smaller models. We evaluate our approach on multiple idiomatic translation datasets using reference-less metrics, demonstrating significant improvements in translating idioms from English to various Indian languages.
>
---
#### [new 073] ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估任务，解决标准基准依赖单一提示导致评估不可靠的问题。提出ReliableEval方法，通过随机采样意义保持的提示扰动，结合矩方法估计所需采样次数，量化模型的提示敏感性，发现顶级模型存在显著波动，提供通用评估框架。**

- **链接: [http://arxiv.org/pdf/2505.22169v1](http://arxiv.org/pdf/2505.22169v1)**

> **作者:** Gili Lior; Eliya Habba; Shahar Levy; Avi Caciularu; Gabriel Stanovsky
>
> **摘要:** LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
>
---
#### [new 074] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于论据挖掘任务，旨在解决模型泛化能力不足的问题。研究通过评估四类Transformer模型在17个英文数据集上的表现，发现模型依赖数据集特定词汇而非真正理解论据，提出结合任务预训练和联合训练提升泛化性的方法。**

- **链接: [http://arxiv.org/pdf/2505.22137v1](http://arxiv.org/pdf/2505.22137v1)**

> **作者:** Marc Feger; Katarina Boland; Stefan Dietze
>
> **备注:** This paper has been accepted to ACL 2025 and will be published after 27.07.2025
>
> **摘要:** Identifying arguments is a necessary prerequisite for various tasks in automated discourse analysis, particularly within contexts such as political debates, online discussions, and scientific reasoning. In addition to theoretical advances in understanding the constitution of arguments, a significant body of research has emerged around practical argument mining, supported by a growing number of publicly available datasets. On these benchmarks, BERT-like transformers have consistently performed best, reinforcing the belief that such models are broadly applicable across diverse contexts of debate. This study offers the first large-scale re-evaluation of such state-of-the-art models, with a specific focus on their ability to generalize in identifying arguments. We evaluate four transformers, three standard and one enhanced with contrastive pre-training for better generalization, on 17 English sentence-level datasets as most relevant to the task. Our findings show that, to varying degrees, these models tend to rely on lexical shortcuts tied to content words, suggesting that apparent progress may often be driven by dataset-specific cues rather than true task alignment. While the models achieve strong results on familiar benchmarks, their performance drops markedly when applied to unseen datasets. Nonetheless, incorporating both task-specific pre-training and joint benchmark training proves effective in enhancing both robustness and generalization.
>
---
#### [new 075] BehaviorSFT: Behavioral Token Conditioning for Clinical Agents Across the Proactivity Spectrum
- **分类: cs.CL**

- **简介: 该论文针对临床大语言模型主动性不足的问题（如未提示时无法识别关键缺失信息），提出BehaviorSFT方法：通过行为标记动态调整模型行为，结合BehaviorBench评估数据集，在保持反应性的同时提升主动性（最高97.3% Macro F1），经临床验证平衡了建议及时性与过度干预风险。**

- **链接: [http://arxiv.org/pdf/2505.21757v1](http://arxiv.org/pdf/2505.21757v1)**

> **作者:** Yubin Kim; Zhiyuan Hu; Hyewon Jeong; Eugene Park; Shuyue Stella Li; Chanwoo Park; Shiyun Xiong; MingYu Lu; Hyeonhoon Lee; Xin Liu; Daniel McDuff; Cynthia Breazeal; Samir Tulebaev; Hae Won Park
>
> **摘要:** Large Language Models (LLMs) as clinical agents require careful behavioral adaptation. While adept at reactive tasks (e.g., diagnosis reasoning), LLMs often struggle with proactive engagement, like unprompted identification of critical missing information or risks. We introduce BehaviorBench, a comprehensive dataset to evaluate agent behaviors across a clinical assistance spectrum, ranging from reactive query responses to proactive interventions (e.g., clarifying ambiguities, flagging overlooked critical data). Our BehaviorBench experiments reveal LLMs' inconsistent proactivity. To address this, we propose BehaviorSFT, a novel training strategy using behavioral tokens to explicitly condition LLMs for dynamic behavioral selection along this spectrum. BehaviorSFT boosts performance, achieving up to 97.3% overall Macro F1 on BehaviorBench and improving proactive task scores (e.g., from 95.0% to 96.5% for Qwen2.5-7B-Ins). Crucially, blind clinician evaluations confirmed BehaviorSFT-trained agents exhibit more realistic clinical behavior, striking a superior balance between helpful proactivity (e.g., timely, relevant suggestions) and necessary restraint (e.g., avoiding over-intervention) versus standard fine-tuning or explicit instructed agents.
>
---
#### [new 076] Rethinking the Outlier Distribution in Large Language Models: An In-depth Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型优化任务，旨在解决异常值（massive activations和channel-wise outliers）导致量化误差及性能下降的问题。通过分析异常值形成机制，提出减少其产生的策略，实现高效量化压缩且精度损失小。**

- **链接: [http://arxiv.org/pdf/2505.21670v1](http://arxiv.org/pdf/2505.21670v1)**

> **作者:** Rahul Raman; Khushi Sharma; Sai Qian Zhang
>
> **摘要:** Investigating outliers in large language models (LLMs) is crucial due to their significant impact on various aspects of LLM performance, including quantization and compression. Outliers often cause considerable quantization errors, leading to degraded model performance. Identifying and addressing these outliers can enhance the accuracy and efficiency of the quantization process, enabling smoother deployment on edge devices or specialized hardware. Recent studies have identified two common types of outliers in LLMs: massive activations and channel-wise outliers. While numerous quantization algorithms have been proposed to mitigate their effects and maintain satisfactory accuracy, few have thoroughly explored the root causes of these outliers in depth. In this paper, we conduct a comprehensive investigation into the formation mechanisms of these outliers and propose potential strategies to mitigate their occurrence. Ultimately, we introduce some efficient approaches to eliminate most massive activations and channel-wise outliers with minimal impact on accuracy.
>
---
#### [new 077] Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于科学信息抽取（SciIE）任务，旨在解决大语言模型（LLMs）在需要推理与记忆的SciIE中表现低于小型Bert模型的问题。提出两阶段训练方法：1）MimicSFT通过结构化模板模仿提升推理路径；2）R²GRPO结合相关性与规则奖励优化。实验显示其在关系抽取上超越基线模型。**

- **链接: [http://arxiv.org/pdf/2505.22068v1](http://arxiv.org/pdf/2505.22068v1)**

> **作者:** Ran Li; Shimin Di; Yuchen Liu; Chen Jing; Yu Qiu; Lei Chen
>
> **摘要:** Previous study suggest that powerful Large Language Models (LLMs) trained with Reinforcement Learning with Verifiable Rewards (RLVR) only refines reasoning path without improving the reasoning capacity in math tasks while supervised-finetuning(SFT) with distillation can. We study this from the view of Scientific information extraction (SciIE) where LLMs and reasoning LLMs underperforms small Bert-based models. SciIE require both the reasoning and memorization. We argue that both SFT and RLVR can refine the reasoning path and improve reasoning capacity in a simple way based on SciIE. We propose two-stage training with 1. MimicSFT, using structured reasoning templates without needing high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and rule-induced rewards. Experiments on scientific IE benchmarks show that both methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses baseline LLMs and specialized supervised models in relation extraction. Our code is available at https://github.com/ranlislz/R2GRPO.
>
---
#### [new 078] If Pigs Could Fly... Can LLMs Logically Reason Through Counterfactuals?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLMs在反事实推理任务中的逻辑缺陷，解决其处理与已有知识冲突的假设场景时推理能力下降的问题。构建CounterLogic数据集，发现模型准确率下降27%，提出Self-Segregate方法通过元认知提示识别冲突，缩小差距至11%，整体提升7.5%。**

- **链接: [http://arxiv.org/pdf/2505.22318v1](http://arxiv.org/pdf/2505.22318v1)**

> **作者:** Ishwar B Balappanawar; Vamshi Krishna Bonagiri; Anish R Joishy; Manas Gaur; Krishnaprasad Thirunarayan; Ponnurangam Kumaraguru
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive reasoning capabilities in familiar contexts, but struggle when the context conflicts with their parametric knowledge. To investigate this phenomenon, we introduce CounterLogic, a dataset containing 1,800 examples across 9 logical schemas, explicitly designed to evaluate logical reasoning through counterfactual (hypothetical knowledge-conflicting) scenarios. Our systematic evaluation of 11 LLMs across 6 different datasets reveals a consistent performance degradation, with accuracies dropping by 27% on average when reasoning through counterfactual information. We propose Self-Segregate, a prompting method enabling metacognitive awareness (explicitly identifying knowledge conflicts) before reasoning. Our method dramatically narrows the average performance gaps from 27% to just 11%, while significantly increasing the overall accuracy (+7.5%). We discuss the implications of these findings and draw parallels to human cognitive processes, particularly on how humans disambiguate conflicting information during reasoning tasks. Our findings offer practical insights for understanding and enhancing LLMs reasoning capabilities in real-world applications, especially where models must logically reason independently of their factual knowledge.
>
---
#### [new 079] Assessing and Refining ChatGPT's Performance in Identifying Targeting and Inappropriate Language: A Comparative Study
- **分类: cs.CL**

- **简介: 该论文评估ChatGPT在识别网络评论中不当及攻击性语言的任务表现，通过对比众包标注和专家判断，发现其在检测不当内容准确率较高（尤其v6版本优化明显），但攻击性语言检测存在较高假阳性。研究旨在改进AI内容审核系统，强调需持续优化模型与语境理解能力。**

- **链接: [http://arxiv.org/pdf/2505.21710v1](http://arxiv.org/pdf/2505.21710v1)**

> **作者:** Barbarestani Baran; Maks Isa; Vossen Piek
>
> **摘要:** This study evaluates the effectiveness of ChatGPT, an advanced AI model for natural language processing, in identifying targeting and inappropriate language in online comments. With the increasing challenge of moderating vast volumes of user-generated content on social network sites, the role of AI in content moderation has gained prominence. We compared ChatGPT's performance against crowd-sourced annotations and expert evaluations to assess its accuracy, scope of detection, and consistency. Our findings highlight that ChatGPT performs well in detecting inappropriate content, showing notable improvements in accuracy through iterative refinements, particularly in Version 6. However, its performance in targeting language detection showed variability, with higher false positive rates compared to expert judgments. This study contributes to the field by demonstrating the potential of AI models like ChatGPT to enhance automated content moderation systems while also identifying areas for further improvement. The results underscore the importance of continuous model refinement and contextual understanding to better support automated moderation and mitigate harmful online behavior.
>
---
#### [new 080] Iterative Corpus Refinement for Materials Property Prediction Based on Scientific Texts
- **分类: cs.CL; cond-mat.mtrl-sci**

- **简介: 该论文提出迭代语料库优化框架，解决材料发现中数据稀缺与组合爆炸问题。通过精选多样化科学文本训练词嵌入模型，监测组成-属性关联收敛，预测电催化材料性能，实验验证有效，加速高通量材料筛选。**

- **链接: [http://arxiv.org/pdf/2505.21646v1](http://arxiv.org/pdf/2505.21646v1)**

> **作者:** Lei Zhang; Markus Stricker
>
> **备注:** 13 pages, 5 figures, 2 tables, accepted at ECMLPKDD 2025
>
> **摘要:** The discovery and optimization of materials for specific applications is hampered by the practically infinite number of possible elemental combinations and associated properties, also known as the `combinatorial explosion'. By nature of the problem, data are scarce and all possible data sources should be used. In addition to simulations and experimental results, the latent knowledge in scientific texts is not yet used to its full potential. We present an iterative framework that refines a given scientific corpus by strategic selection of the most diverse documents, training Word2Vec models, and monitoring the convergence of composition-property correlations in embedding space. Our approach is applied to predict high-performing materials for oxygen reduction (ORR), hydrogen evolution (HER), and oxygen evolution (OER) reactions for a large number of possible candidate compositions. Our method successfully predicts the highest performing compositions among a large pool of candidates, validated by experimental measurements of the electrocatalytic performance in the lab. This work demonstrates and validates the potential of iterative corpus refinement to accelerate materials discovery and optimization, offering a scalable and efficient tool for screening large compositional spaces where reliable data are scarce or non-existent.
>
---
#### [new 081] Fusion Steering: Prompt-Specific Activation Control
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，旨在提升大语言模型的事实准确性。针对传统方法受限于单层操作的问题，提出Fusion Steering方法，通过动态跨层注入提示特异的激活增量，并优化权重平衡事实与流畅性。实验显示分段引导策略在SimpleQA任务中准确率达25.4%，显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.22572v1](http://arxiv.org/pdf/2505.22572v1)**

> **作者:** Waldemar Chang; Alhassan Yasin
>
> **备注:** 14 pages, 4 figures, 2 tables
>
> **摘要:** We present Fusion Steering, an activation steering methodology that improves factual accuracy in large language models (LLMs) for question-answering (QA) tasks. This approach introduces flexible steering configurations, including full-layer steering and segmented steering. Unlike traditional methods constrained to single-layer or fixed-layer operations, Fusion Steering employs dynamic injection of prompt-specific activation deltas across all transformer layers. These activation deltas are derived from reference completions that combine the ground-truth answer with a model-generated explanation to facilitate semantically enriched, example-specific steering. The injection weights are optimized per prompt using Optuna, targeting a joint objective that balances token overlap (factual alignment) and perplexity (fluency proxy). Evaluation employs a composite score integrating token overlap and LLM-graded quality, encompassing factual accuracy, coherence, and relevance. Empirical results on 260 SimpleQA prompts (selected from 500 where the baseline failed) showcase the efficacy of segmented steering. Using Gemma-2-2B-IT with 8-bit quantization, segmented steering achieves an accuracy of 25.4% (outputs scoring $\geq 0.6$), outperforming the baseline at 3.5% and full-layer steering at 16.2%. Under the stricter SimpleQA rubric, segmented steering boosts fully correct responses from 0.0% to 13.1%. These findings highlight the strengths of segmented, dynamic intervention strategies and the promise of per-prompt, full-network activation control. Fusion Steering is also amenable to sparse representations, such as Neuronpedia or sparse crosscoders, suggesting a promising direction for interpretable and scalable activation-level control in LLMs.
>
---
#### [new 082] More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态推理任务，研究推理链延长导致模型过度依赖语言先验、忽视视觉信息的幻觉问题。提出RH-AUC指标量化推理长度对感知准确性的影响，并构建RH-Bench基准测试，揭示模型规模及训练数据类型比数据量更影响推理与视觉接地的平衡。**

- **链接: [http://arxiv.org/pdf/2505.21523v1](http://arxiv.org/pdf/2505.21523v1)**

> **作者:** Chengzhi Liu; Zhongxing Xu; Qingyue Wei; Juncheng Wu; James Zou; Xin Eric Wang; Yuyin Zhou; Sheng Liu
>
> **摘要:** Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity.
>
---
#### [new 083] Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition
- **分类: cs.CL**

- **简介: 该论文提出Pangu Embedded，一种高效双系统LLM推理器，解决现有模型计算成本高、推理延迟大的问题。通过两阶段训练（迭代蒸馏+强化学习）及双模式架构（快/慢思维），平衡资源与推理深度，实验显示其性能优于同类模型。**

- **链接: [http://arxiv.org/pdf/2505.22375v1](http://arxiv.org/pdf/2505.22375v1)**

> **作者:** Hanting Chen; Yasheng Wang; Kai Han; Dong Li; Lin Li; Zhenni Bi; Jinpeng Li; Haoyu Wang; Fei Mi; Mingjian Zhu; Bin Wang; Kaikai Song; Yifei Fu; Xu He; Yu Luo; Chong Zhu; Quan He; Xueyu Wu; Wei He; Hailin Hu; Yehui Tang; Dacheng Tao; Xinghao Chen; Yunhe Wang; Other Contributors
>
> **摘要:** This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
>
---
#### [new 084] Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs
- **分类: cs.CL**

- **简介: 该论文针对低资源语言机器翻译任务，提出Fragment-Shot Prompting及Pivoted扩展方法，通过分段输入与句法覆盖的检索示例，解决LLMs在数据不足下的翻译挑战。实验表明方法有效提升翻译质量，尤其在低资源语言间，并揭示模型推理能力的关键作用及提示工程在特定场景的局限。**

- **链接: [http://arxiv.org/pdf/2505.22293v1](http://arxiv.org/pdf/2505.22293v1)**

> **作者:** Samuel Frontull; Thomas Ströhle
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation, sometimes even outperforming traditional neural systems. However, previous research has highlighted the challenges of using LLMs, particularly with prompt engineering, for low-resource languages. In this work, we introduce Fragment-Shot Prompting, a novel in-context learning method that segments input and retrieves translation examples based on syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables translation without direct parallel data. We evaluate these methods using GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot Prompting is effective for translating into and between the studied low-resource languages, with syntactic coverage positively correlating with translation quality; (2) Models with stronger reasoning abilities make more effective use of retrieved knowledge, generally produce better translations, and enable Pivoted Fragment-Shot to significantly improve translation quality between the Ladin variants; and (3) prompt engineering offers limited, if any, improvements when translating from a low-resource to a high-resource language, where zero-shot prompting already yields satisfactory results. We publicly release our code and the retrieval corpora.
>
---
#### [new 085] Do We Know What LLMs Don't Know? A Study of Consistency in Knowledge Probing
- **分类: cs.CL**

- **简介: 该论文研究LLMs知识探测一致性问题。针对现有方法在识别知识缺口时因输入微小变化或方法差异导致结果不一致的问题，提出通过输入变异和量化指标评估，揭示方法内（如答案选项排序变化致一致性降至40%）和跨方法（一致性低至7%）的不一致性，强调需开发抗扰动探测框架。**

- **链接: [http://arxiv.org/pdf/2505.21701v1](http://arxiv.org/pdf/2505.21701v1)**

> **作者:** Raoyuan Zhao; Abdullatif Köksal; Ali Modarressi; Michael A. Hedderich; Hinrich Schütze
>
> **摘要:** The reliability of large language models (LLMs) is greatly compromised by their tendency to hallucinate, underscoring the need for precise identification of knowledge gaps within LLMs. Various methods for probing such gaps exist, ranging from calibration-based to prompting-based methods. To evaluate these probing methods, in this paper, we propose a new process based on using input variations and quantitative metrics. Through this, we expose two dimensions of inconsistency in knowledge gap probing. (1) Intra-method inconsistency: Minimal non-semantic perturbations in prompts lead to considerable variance in detected knowledge gaps within the same probing method; e.g., the simple variation of shuffling answer options can decrease agreement to around 40%. (2) Cross-method inconsistency: Probing methods contradict each other on whether a model knows the answer. Methods are highly inconsistent -- with decision consistency across methods being as low as 7% -- even though the model, dataset, and prompt are all the same. These findings challenge existing probing methods and highlight the urgent need for perturbation-robust probing frameworks.
>
---
#### [new 086] ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于声明验证任务，旨在解决现有方法难以有效利用知识图谱（KG）结构化数据及LLMs处理KG推理不足的问题。提出ClaimPKG框架：通过轻量级专用LLM将声明转换为伪子图，指导KG子图检索，再由通用LLM推理生成结论，显著提升验证准确率并具备跨数据集泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.22552v1](http://arxiv.org/pdf/2505.22552v1)**

> **作者:** Hoang Pham; Thanh-Do Nguyen; Khac-Hoai Nam Bui
>
> **备注:** Accepted by ACL 2025 findings
>
> **摘要:** Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of large language models (LLMs) is an emerging research challenge in claim verification. While KGs provide structured, semantically rich representations well-suited for reasoning, most existing verification methods rely on unstructured text corpora, limiting their ability to effectively leverage KGs. Additionally, despite possessing strong reasoning abilities, modern LLMs struggle with multi-step modular pipelines and reasoning over KGs without adaptation. To address these challenges, we propose ClaimPKG, an end-to-end framework that seamlessly integrates LLM reasoning with structured knowledge from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight, specialized LLM to represent the input claim as pseudo-subgraphs, guiding a dedicated subgraph retrieval module to identify relevant KG subgraphs. These retrieved subgraphs are then processed by a general-purpose LLM to produce the final verdict and justification. Extensive experiments on the FactKG dataset demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming strong baselines in this research field by 9%-12% accuracy points across multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS, effectively combining structured knowledge from KGs with LLM reasoning across various LLM backbones.
>
---
#### [new 087] Spatial Knowledge Graph-Guided Multimodal Synthesis
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 论文提出SKG2Data方法，属于多模态数据合成任务，解决多模态大模型（MLLMs）空间感知不足问题。通过构建空间知识图（SKG）模拟人类对方向和距离的感知，生成符合空间常识的合成数据，实验表明有效提升模型空间推理与泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.22633v1](http://arxiv.org/pdf/2505.22633v1)**

> **作者:** Yida Xue; Zhen Bi; Jinnan Yang; Jungang Lou; Huajun Chen; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. In this work, we introduce SKG2Data, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2Data automatically constructs a Spatial Knowledge Graph (SKG) to emulate human-like perception of spatial directions and distances, which is subsequently utilized to guide multimodal data synthesis. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, not only enhance the spatial perception and reasoning abilities of MLLMs but also exhibit strong generalization capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence.
>
---
#### [new 088] EvolveSearch: An Iterative Self-Evolving Search Agent
- **分类: cs.CL**

- **简介: 论文提出EvolveSearch框架，解决LLM在开放网络搜索中因监督微调数据不足和强化学习效率低导致的性能瓶颈。通过迭代结合SFT与RL，无需标注数据，提升多跳问答任务表现，平均提升4.7%。**

- **链接: [http://arxiv.org/pdf/2505.22501v1](http://arxiv.org/pdf/2505.22501v1)**

> **作者:** Dingchu Zhang; Yida Zhao; Jialong Wu; Baixuan Li; Wenbiao Yin; Liwen Zhang; Yong Jiang; Yufeng Li; Kewei Tu; Pengjun Xie; Fei Huang
>
> **摘要:** The rapid advancement of large language models (LLMs) has transformed the landscape of agentic information seeking capabilities through the integration of tools such as search engines and web browsers. However, current mainstream approaches for enabling LLM web search proficiency face significant challenges: supervised fine-tuning struggles with data production in open-search domains, while RL converges quickly, limiting their data utilization efficiency. To address these issues, we propose EvolveSearch, a novel iterative self-evolution framework that combines SFT and RL to enhance agentic web search capabilities without any external human-annotated reasoning data. Extensive experiments on seven multi-hop question-answering (MHQA) benchmarks demonstrate that EvolveSearch consistently improves performance across iterations, ultimately achieving an average improvement of 4.7\% over the current state-of-the-art across seven benchmarks, opening the door to self-evolution agentic capabilities in open web search domains.
>
---
#### [new 089] ArgInstruct: Specialized Instruction Fine-Tuning for Computational Argumentation
- **分类: cs.CL**

- **简介: 该论文提出ArgInstruct方法，通过针对计算论证（CA）领域的指令微调，解决大语言模型（LLMs）在该领域任务中的知识不足问题。工作包括构建105项CA任务指令集、开发专用基准测试，并生成5.2万条指令训练模型，实验显示其显著提升CA任务表现且保持通用NLP能力。**

- **链接: [http://arxiv.org/pdf/2505.22076v1](http://arxiv.org/pdf/2505.22076v1)**

> **作者:** Maja Stahl; Timon Ziegenbein; Joonsuk Park; Henning Wachsmuth
>
> **摘要:** Training large language models (LLMs) to follow instructions has significantly enhanced their ability to tackle unseen tasks. However, despite their strong generalization capabilities, instruction-following LLMs encounter difficulties when dealing with tasks that require domain knowledge. This work introduces a specialized instruction fine-tuning for the domain of computational argumentation (CA). The goal is to enable an LLM to effectively tackle any unseen CA tasks while preserving its generalization capabilities. Reviewing existing CA research, we crafted natural language instructions for 105 CA tasks to this end. On this basis, we developed a CA-specific benchmark for LLMs that allows for a comprehensive evaluation of LLMs' capabilities in solving various CA tasks. We synthesized 52k CA-related instructions, adapting the self-instruct process to train a CA-specialized instruction-following LLM. Our experiments suggest that CA-specialized instruction fine-tuning significantly enhances the LLM on both seen and unseen CA tasks. At the same time, performance on the general NLP tasks of the SuperNI benchmark remains stable.
>
---
#### [new 090] RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RAD框架，针对混合模型（Transformer+SSM）中Transformer冗余问题，通过自推测解码识别冗余注意力层，替换为SSM并进行针对性蒸馏，提升数学/编码任务性能，收敛速度翻倍，小模型效果超越基线。**

- **链接: [http://arxiv.org/pdf/2505.22135v1](http://arxiv.org/pdf/2505.22135v1)**

> **作者:** Yuichiro Hoshino; Hideyuki Tachibana; Muneyoshi Inahara; Hiroto Takegawa
>
> **备注:** 26 pages
>
> **摘要:** Hybrid models combining Transformers and State Space Models (SSMs) are promising for balancing performance and efficiency. However, optimizing these hybrid models, particularly by addressing the potential redundancy inherent within the Transformer components, remains a significant challenge. In this paper, we propose RAD (Redundancy-Aware Distillation), a novel framework that uses self-speculative decoding as a diagnostic tool to identify redundant attention layers within the model. These identified layers are then selectively replaced with SSM components, followed by targeted (self-)distillation. Specifically, RAD focuses knowledge transfer on the components identified as redundant, considering architectural changes and specific weight initialization strategies. We experimentally demonstrate that self-distillation using RAD significantly surpasses the performance of the original base model on mathematical and coding tasks. Furthermore, RAD is also effective in standard knowledge distillation settings, achieving up to approximately 2x faster convergence compared to baseline methods. Notably, while a baseline model distilled from a Llama-3.1 70B teacher achieves scores of 46.17 on GSM8K and 22.75 on CRUX, RAD achieves significantly higher scores of 71.27 on GSM8K and 28.25 on CRUX, even when using a much smaller Llama-3.1 8B teacher. RAD offers a new pathway for efficient optimization and performance enhancement in the distillation of hybrid models.
>
---
#### [new 091] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，旨在解决扩散LLM因缺乏KV缓存及并行解码质量下降导致的推理速度慢问题。提出块状近似KV缓存机制提升缓存复用效率，并设计信心阈值策略选择性解码高置信度token，平衡并行解码速度与生成质量，实现27.6倍吞吐量提升。**

- **链接: [http://arxiv.org/pdf/2505.22618v1](http://arxiv.org/pdf/2505.22618v1)**

> **作者:** Chengyue Wu; Hao Zhang; Shuchen Xue; Zhijian Liu; Shizhe Diao; Ligeng Zhu; Ping Luo; Song Han; Enze Xie
>
> **摘要:** Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
>
---
#### [new 092] RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- **分类: cs.CL**

- **简介: 该论文提出RedTeamCUA框架，用于测试计算机使用代理（CUAs）在混合Web-OS环境中的间接提示注入漏洞。针对现有评估缺乏现实场景及混合攻击分析的问题，框架整合VM和Docker沙箱，支持灵活配置攻击场景，开发含864例的基准测试，发现前沿CUAs仍存在显著漏洞（最高ASR 48%），强调需加强防御。**

- **链接: [http://arxiv.org/pdf/2505.21936v1](http://arxiv.org/pdf/2505.21936v1)**

> **作者:** Zeyi Liao; Jaylen Jones; Linxi Jiang; Eric Fosler-Lussier; Yu Su; Zhiqiang Lin; Huan Sun
>
> **摘要:** Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, with the recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%, demonstrating that indirect prompt injection presents tangible risks for even advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.
>
---
#### [new 093] Comprehensive Evaluation on Lexical Normalization: Boundary-Aware Approaches for Unsegmented Languages
- **分类: cs.CL**

- **简介: 该论文聚焦未分词语言的词汇规范化任务，解决现有方法缺乏跨领域全面评估的问题。构建大规模多领域日语数据集，开发基于预训练模型的方法，并通过多角度实验验证，证明编码器和解码器模型在准确性和效率上表现优异。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22273v1](http://arxiv.org/pdf/2505.22273v1)**

> **作者:** Shohei Higashiyama; Masao Utiyama
>
> **备注:** 23 pages
>
> **摘要:** Lexical normalization research has sought to tackle the challenge of processing informal expressions in user-generated text, yet the absence of comprehensive evaluations leaves it unclear which methods excel across multiple perspectives. Focusing on unsegmented languages, we make three key contributions: (1) creating a large-scale, multi-domain Japanese normalization dataset, (2) developing normalization methods based on state-of-the-art pretrained models, and (3) conducting experiments across multiple evaluation perspectives. Our experiments show that both encoder-only and decoder-only approaches achieve promising results in both accuracy and efficiency.
>
---
#### [new 094] Rethinking Data Mixture for Large Language Models: A Comprehensive Survey and New Perspectives
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型数据混合方法的综述与分析，旨在解决计算资源受限时如何确定多领域数据权重以优化模型性能的问题。工作包括提出细粒度分类（离线分三类，在线分三组）、总结方法算法及关系，并讨论优缺点与挑战。**

- **链接: [http://arxiv.org/pdf/2505.21598v1](http://arxiv.org/pdf/2505.21598v1)**

> **作者:** Yajiao Liu; Congliang Chen; Junchi Yang; Ruoyu Sun
>
> **备注:** The first version of this paper was submitted to ACL ARR 2025 February Submission
>
> **摘要:** Training large language models with data collected from various domains can improve their performance on downstream tasks. However, given a fixed training budget, the sampling proportions of these different domains significantly impact the model's performance. How can we determine the domain weights across different data domains to train the best-performing model within constrained computational resources? In this paper, we provide a comprehensive overview of existing data mixture methods. First, we propose a fine-grained categorization of existing methods, extending beyond the previous offline and online classification. Offline methods are further grouped into heuristic-based, algorithm-based, and function fitting-based methods. For online methods, we categorize them into three groups: online min-max optimization, online mixing law, and other approaches by drawing connections with the optimization frameworks underlying offline methods. Second, we summarize the problem formulations, representative algorithms for each subtype of offline and online methods, and clarify the relationships and distinctions among them. Finally, we discuss the advantages and disadvantages of each method and highlight key challenges in the field of data mixture.
>
---
#### [new 095] AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出AutoL2S框架，解决大语言模型在简单推理任务中过度思考生成冗余长路径的问题。通过动态分析问题复杂度，结合<EASY>标记训练数据，使模型自主选择短或长推理路径，减少推理长度达57%且保持性能，提升效率。**

- **链接: [http://arxiv.org/pdf/2505.22662v1](http://arxiv.org/pdf/2505.22662v1)**

> **作者:** Feng Luo; Yu-Neng Chuang; Guanchu Wang; Hoang Anh Duy Le; Shaochen Zhong; Hongyi Liu; Jiayi Yuan; Yang Sui; Vladimir Braverman; Vipin Chaudhary; Xia Hu
>
> **摘要:** The reasoning-capable large language models (LLMs) demonstrate strong performance on complex reasoning tasks but often suffer from overthinking, generating unnecessarily long chain-of-thought (CoT) reasoning paths for easy reasoning questions, thereby increasing inference cost and latency. Recent approaches attempt to address this challenge by manually deciding when to apply long or short reasoning. However, they lack the flexibility to adapt CoT length dynamically based on question complexity. In this paper, we propose Auto Long-Short Reasoning (AutoL2S), a dynamic and model-agnostic framework that enables LLMs to dynamically compress their generated reasoning path based on the complexity of the reasoning question. AutoL2S enables a learned paradigm, in which LLMs themselves can decide when longer reasoning is necessary and when shorter reasoning suffices, by training on data annotated with our proposed method, which includes both long and short CoT paths and a special <EASY> token. We then use <EASY> token to indicate when the model can skip generating lengthy CoT reasoning. This proposed annotation strategy can enhance the LLMs' ability to generate shorter CoT reasoning paths with improved quality after training. Extensive evaluation results show that AutoL2S reduces the length of reasoning generation by up to 57% without compromising performance, demonstrating the effectiveness of AutoL2S for scalable and efficient LLM reasoning.
>
---
#### [new 096] Co-Saving: Resource Aware Multi-Agent Collaboration for Software Development
- **分类: cs.CL; cs.AI; cs.MA; cs.SE**

- **简介: 该论文提出Co-Saving系统，一种资源感知的多智能体协作方法，用于优化软件开发。针对传统MAS资源消耗高、效率低的问题，其通过学习历史成功路径的"捷径"减少冗余推理，提升效率与代码质量。实验显示较ChatDev减少50.85%的token使用并提升10.06%代码质量。**

- **链接: [http://arxiv.org/pdf/2505.21898v1](http://arxiv.org/pdf/2505.21898v1)**

> **作者:** Rennai Qiu; Chen Qian; Ran Li; Yufan Dang; Weize Chen; Cheng Yang; Yingli Zhang; Ye Tian; Xuantang Xiong; Lei Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** Work in Progress
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and autonomous agents have demonstrated remarkable capabilities across various domains. However, standalone agents frequently encounter limitations when handling complex tasks that demand extensive interactions and substantial computational resources. Although Multi-Agent Systems (MAS) alleviate some of these limitations through collaborative mechanisms like task decomposition, iterative communication, and role specialization, they typically remain resource-unaware, incurring significant inefficiencies due to high token consumption and excessive execution time. To address these limitations, we propose a resource-aware multi-agent system -- Co-Saving (meaning that multiple agents collaboratively engage in resource-saving activities), which leverages experiential knowledge to enhance operational efficiency and solution quality. Our key innovation is the introduction of "shortcuts" -- instructional transitions learned from historically successful trajectories -- which allows to bypass redundant reasoning agents and expedite the collective problem-solving process. Experiments for software development tasks demonstrate significant advantages over existing methods. Specifically, compared to the state-of-the-art MAS ChatDev, our method achieves an average reduction of 50.85% in token usage, and improves the overall code quality by 10.06%.
>
---
#### [new 097] Resolving Knowledge Conflicts in Domain-specific Data Selection: A Case Study on Medical Instruction-tuning
- **分类: cs.CL**

- **简介: 该论文属于领域特定数据选择任务，解决指令调优中因知识冲突（预训练知识与领域数据知识不一致）导致的模型性能下降和幻觉问题。提出KDS框架，通过量化上下文-内存知识对齐度和内存知识一致性两个指标筛选数据，提升模型效果并减少幻觉，医学实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.21958v1](http://arxiv.org/pdf/2505.21958v1)**

> **作者:** Qihuang Zhong; Liang Ding; Fei Liao; Juhua Liu; Bo Du; Dacheng Tao
>
> **摘要:** Domain-specific instruction-tuning has become the defacto standard for improving the performance of large language models (LLMs) in specialized applications, e.g., medical question answering. Since the instruction-tuning dataset might contain redundant or low-quality data, data selection (DS) is usually required to maximize the data efficiency. Despite the successes in the general domain, current DS methods often struggle to select the desired data for domain-specific instruction-tuning. One of the main reasons is that they neglect the impact of knowledge conflicts, i.e., the discrepancy between LLMs' pretrained knowledge and context knowledge of instruction data, which could damage LLMs' prior abilities and lead to hallucination. To this end, we propose a simple-yet-effective Knowledge-aware Data Selection (namely KDS) framework to select the domain-specific instruction-tuning data that meets LLMs' actual needs. The core of KDS is to leverage two knowledge-aware metrics for quantitatively measuring knowledge conflicts from two aspects: context-memory knowledge alignment and intra-memory knowledge consistency. By filtering the data with large knowledge conflicts and sampling the high-quality and diverse data, KDS can effectively stimulate the LLMs' abilities and achieve better domain-specific performance. Taking the medical domain as the testbed, we conduct extensive experiments and empirically prove that KDS surpasses the other baselines and brings significant and consistent performance gains among all LLMs. More encouragingly, KDS effectively improves the model generalization and alleviates the hallucination problem.
>
---
#### [new 098] Learning to Route Queries Across Knowledge Bases for Step-wise Retrieval-Augmented Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，针对现有MRAG方法静态检索导致效率低、精度不足的问题，提出R1-Router框架，通过动态路由中间查询至最优知识库，并结合Step-GRPO算法优化推理路径，提升模型效率与准确性。**

- **链接: [http://arxiv.org/pdf/2505.22095v1](http://arxiv.org/pdf/2505.22095v1)**

> **作者:** Chunyi Peng; Zhipeng Xu; Zhenghao Liu; Yishan Li; Yukun Yan; Shuo Wang; Zhiyuan Liu; Yu Gu; Minghe Yu; Ge Yu; Maosong Sun
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) has shown promise in mitigating hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge during generation. Existing MRAG methods typically adopt a static retrieval pipeline that fetches relevant information from multiple Knowledge Bases (KBs), followed by a refinement step. However, these approaches overlook the reasoning and planning capabilities of MLLMs to dynamically determine how to interact with different KBs during the reasoning process. To address this limitation, we propose R1-Router, a novel MRAG framework that learns to decide when and where to retrieve knowledge based on the evolving reasoning state. Specifically, R1-Router can generate follow-up queries according to the current reasoning step, routing these intermediate queries to the most suitable KB, and integrating external knowledge into a coherent reasoning trajectory to answer the original query. Furthermore, we introduce Step-wise Group Relative Policy Optimization (Step-GRPO), a tailored reinforcement learning algorithm that assigns step-specific rewards to optimize the reasoning behavior of MLLMs. Experimental results on various open-domain QA benchmarks across multiple modalities demonstrate that R1-Router outperforms baseline models by over 7%. Further analysis shows that R1-Router can adaptively and effectively leverage diverse KBs, reducing unnecessary retrievals and improving both efficiency and accuracy.
>
---
#### [new 099] CoThink: Token-Efficient Reasoning via Instruct Models Guiding Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型推理效率优化任务，旨在解决推理模型过度思考导致的冗余输出问题。通过分析发现强化学习降低推理信息密度及反向链式训练引入冗余验证为关键原因，提出CoThink框架：指令模型先生成高阶方案，推理模型执行求解，动态调整推理深度。实验表明其减少22.3%生成token同时保持准确率，并定义推理效率及扩展规律。**

- **链接: [http://arxiv.org/pdf/2505.22017v1](http://arxiv.org/pdf/2505.22017v1)**

> **作者:** Siqi Fan; Peng Han; Shuo Shang; Yequan Wang; Aixin Sun
>
> **摘要:** Large language models (LLMs) benefit from increased test-time compute, a phenomenon known as test-time scaling. However, reasoning-optimized models often overthink even simple problems, producing excessively verbose outputs and leading to low token efficiency. By comparing these models with equally sized instruct models, we identify two key causes of this verbosity: (1) reinforcement learning reduces the information density of forward reasoning, and (2) backward chain-of thought training encourages redundant and often unnecessary verification steps. Since LLMs cannot assess the difficulty of a given problem, they tend to apply the same cautious reasoning strategy across all tasks, resulting in inefficient overthinking. To address this, we propose CoThink, an embarrassingly simple pipeline: an instruct model first drafts a high-level solution outline; a reasoning model then works out the solution. We observe that CoThink enables dynamic adjustment of reasoning depth based on input difficulty. Evaluated with three reasoning models DAPO, DeepSeek-R1, and QwQ on three datasets GSM8K, MATH500, and AIME24, CoThink reduces total token generation by 22.3% while maintaining pass@1 accuracy within a 0.42% margin on average. With reference to the instruct model, we formally define reasoning efficiency and observe a potential reasoning efficiency scaling law in LLMs.
>
---
#### [new 100] Calibrating LLM Confidence by Probing Perturbed Representation Stability
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）置信度校准任务，旨在解决LLM预测可靠性不足的问题。提出CCPS方法，通过扰动LLM最终隐藏层表示并分析其稳定性，提取特征预测答案正确性。在多个模型和基准测试中，CCPS显著降低误差并提升性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21772v1](http://arxiv.org/pdf/2505.21772v1)**

> **作者:** Reza Khanmohammadi; Erfan Miahi; Mehrsa Mardikoraem; Simerjot Kaur; Ivan Brugere; Charese H. Smiley; Kundan Thind; Mohammad M. Ghassemi
>
> **摘要:** Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness.
>
---
#### [new 101] Multimodal Forecasting of Sparse Intraoperative Hypotension Events Powered by Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IOHFuseLM框架，针对术中低血压（IOH）预测中事件稀疏及多模态数据整合难题，采用两阶段训练（扩散增强预训练与临床数据微调），并通过生理时序与临床文本的token级对齐及静态属性文本化，提升预测精度，实验显示优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22116v1](http://arxiv.org/pdf/2505.22116v1)**

> **作者:** Jintao Zhang; Zirui Liu; Mingyue Cheng; Shilong Zhang; Tingyue Pan; Qi Liu; Yanhu Xie
>
> **摘要:** Intraoperative hypotension (IOH) frequently occurs under general anesthesia and is strongly linked to adverse outcomes such as myocardial injury and increased mortality. Despite its significance, IOH prediction is hindered by event sparsity and the challenge of integrating static and dynamic data across diverse patients. In this paper, we propose \textbf{IOHFuseLM}, a multimodal language model framework. To accurately identify and differentiate sparse hypotensive events, we leverage a two-stage training strategy. The first stage involves domain adaptive pretraining on IOH physiological time series augmented through diffusion methods, thereby enhancing the model sensitivity to patterns associated with hypotension. Subsequently, task fine-tuning is performed on the original clinical dataset to further enhance the ability to distinguish normotensive from hypotensive states. To enable multimodal fusion for each patient, we align structured clinical descriptions with the corresponding physiological time series at the token level. Such alignment enables the model to capture individualized temporal patterns alongside their corresponding clinical semantics. In addition, we convert static patient attributes into structured text to enrich personalized information. Experimental evaluations on two intraoperative datasets demonstrate that IOHFuseLM outperforms established baselines in accurately identifying IOH events, highlighting its applicability in clinical decision support scenarios. Our code is publicly available to promote reproducibility at https://github.com/zjt-gpu/IOHFuseLM.
>
---
#### [new 102] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于状态跟踪任务，旨在探索隐藏单元在计算中的主动作用而非单纯记忆。通过理论和实证分析，证明双线性操作是表示隐藏状态演化的自然归纳偏置，并揭示其与任务复杂度对应的层次结构，指出线性RNN（如Mamba）位于低复杂度中心。**

- **链接: [http://arxiv.org/pdf/2505.21749v1](http://arxiv.org/pdf/2505.21749v1)**

> **作者:** M. Reza Ebrahimi; Roland Memisevic
>
> **摘要:** The role of hidden units in recurrent neural networks is typically seen as modeling memory, with research focusing on enhancing information retention through gating mechanisms. A less explored perspective views hidden units as active participants in the computation performed by the network, rather than passive memory stores. In this work, we revisit bi-linear operations, which involve multiplicative interactions between hidden units and input embeddings. We demonstrate theoretically and empirically that they constitute a natural inductive bias for representing the evolution of hidden states in state tracking tasks. These are the simplest type of task that require hidden units to actively contribute to the behavior of the network. We also show that bi-linear state updates form a natural hierarchy corresponding to state tracking tasks of increasing complexity, with popular linear recurrent networks such as Mamba residing at the lowest-complexity center of that hierarchy.
>
---
#### [new 103] Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Cross-modal RAG框架，针对复杂文本到图像生成中现有方法无法整合多图像元素的问题，通过子维度分解查询与图像，结合稀疏与密集检索策略获取互补图像，并指导模型生成。实验显示其在质量和效率上优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21956v1](http://arxiv.org/pdf/2505.21956v1)**

> **作者:** Mengdan Zhu; Senhao Cheng; Guangji Bai; Yifei Zhang; Liang Zhao
>
> **摘要:** Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy - combining a sub-dimensional sparse retriever with a dense retriever - to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in both retrieval and generation quality, while maintaining high efficiency.
>
---
#### [new 104] Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于科学论文检索任务，旨在解决现有方法无法精准捕捉细粒度科学概念及LLM缺乏领域知识的问题。提出SemRank框架，结合LLM解析查询核心概念与多粒度概念索引（如主题和关键短语），通过语义匹配提升检索准确性，同时保持高效性。**

- **链接: [http://arxiv.org/pdf/2505.21815v1](http://arxiv.org/pdf/2505.21815v1)**

> **作者:** Yunyi Zhang; Ruozhen Yang; Siqi Jiao; SeongKu Kang; Jiawei Han
>
> **摘要:** Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient.
>
---
#### [new 105] Modeling and Optimizing User Preferences in AI Copilots: A Comprehensive Survey and Taxonomy
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于综述任务，旨在解决AI副驾系统中用户偏好优化的碎片化问题。通过提出AI副驾的统一定义及基于交互阶段（事前、事中、事后）的偏好优化分类体系，分析偏好获取、意图建模与反馈整合技术，整合个性化AI、人机协作等领域方法，为设计自适应的AI副驾提供系统性框架。**

- **链接: [http://arxiv.org/pdf/2505.21907v1](http://arxiv.org/pdf/2505.21907v1)**

> **作者:** Saleh Afzoon; Zahra Jahanandish; Phuong Thao Huynh; Amin Beheshti; Usman Naseem
>
> **摘要:** AI copilots, context-aware, AI-powered systems designed to assist users in tasks such as software development and content creation, are becoming integral to modern workflows. As these systems grow in capability and adoption, personalization has emerged as a cornerstone for ensuring usability, trust, and productivity. Central to this personalization is preference optimization: the ability of AI copilots to detect, interpret, and align with individual user preferences. While personalization techniques are well-established in domains like recommender systems and dialogue agents, their adaptation to interactive, real-time systems like AI copilots remains fragmented and underexplored. This survey addresses this gap by synthesizing research on how user preferences are captured, modeled, and refined within the design of AI copilots. We introduce a unified definition of AI copilots and propose a phase-based taxonomy of preference optimization strategies, structured around pre-interaction, mid-interaction, and post-interaction stages. We analyze techniques for acquiring preference signals, modeling user intent, and integrating feedback loops, highlighting both established approaches and recent innovations. By bridging insights from AI personalization, human-AI collaboration, and large language model adaptation, this survey provides a structured foundation for designing adaptive, preference-aware AI copilots. It offers a holistic view of the available preference resources, how they can be leveraged, and which technical approaches are most suited to each stage of system design.
>
---
#### [new 106] Complexity counts: global and local perspectives on Indo-Aryan numeral systems
- **分类: physics.soc-ph; cs.CL**

- **简介: 该论文分析印度-雅利安语系（如印地语、孟加拉语）数字系统的复杂性，探究其非透明构造的成因及跨语言差异。通过建立量化指标，发现其复杂度显著高于全球平均水平，研究宗教、地理等因素影响，并指出其仍符合语言经济性原则。任务为类型学对比与成因分析，解决为何该地区保留复杂数字系统的问题。**

- **链接: [http://arxiv.org/pdf/2505.21510v1](http://arxiv.org/pdf/2505.21510v1)**

> **作者:** Chundra Cathcart
>
> **摘要:** The numeral systems of Indo-Aryan languages such as Hindi, Gujarati, and Bengali are highly unusual in that unlike most numeral systems (e.g., those of English, Chinese, etc.), forms referring to 1--99 are highly non-transparent and are cannot be constructed using straightforward rules. As an example, Hindi/Urdu *iky\=anve* `91' is not decomposable into the composite elements *ek* `one' and *nave* `ninety' in the way that its English counterpart is. This paper situates Indo-Aryan languages within the typology of cross-linguistic numeral systems, and explores the linguistic and non-linguistic factors that may be responsible for the persistence of complex systems in these languages. Using cross-linguistic data from multiple databases, we develop and employ a number of cross-linguistically applicable metrics to quantifies the complexity of languages' numeral systems, and demonstrate that Indo-Aryan languages have decisively more complex numeral systems than the world's languages as a whole, though individual Indo-Aryan languages differ from each other in terms of the complexity of the patterns they display. We investigate the factors (e.g., religion, geographic isolation, etc.) that underlie complexity in numeral systems, with a focus on South Asia, in an attempt to develop an account of why complex numeral systems developed and persisted in certain Indo-Aryan languages but not elsewhere. Finally, we demonstrate that Indo-Aryan numeral systems adhere to certain general pressures toward efficient communication found cross-linguistically, despite their high complexity. We call for this somewhat overlooked dimension of complexity to be taken seriously when discussing general variation in cross-linguistic numeral systems.
>
---
#### [new 107] Effective Context in Neural Speech Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究神经语音模型的有效上下文长度，解决模型实际使用上下文量量化问题。提出两种测量方法，分析监督与自监督模型（如HuBERT）的有效上下文差异，发现任务复杂度与所需上下文正相关，自监督模型早期层提升显著但整体仍较短，支持HuBERT无需修改即可流式处理。**

- **链接: [http://arxiv.org/pdf/2505.22487v1](http://arxiv.org/pdf/2505.22487v1)**

> **作者:** Yen Meng; Sharon Goldwater; Hao Tang
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Modern neural speech models benefit from having longer context, and many approaches have been proposed to increase the maximum context a model can use. However, few have attempted to measure how much context these models actually use, i.e., the effective context. Here, we propose two approaches to measuring the effective context, and use them to analyze different speech Transformers. For supervised models, we find that the effective context correlates well with the nature of the task, with fundamental frequency tracking, phone classification, and word classification requiring increasing amounts of effective context. For self-supervised models, we find that effective context increases mainly in the early layers, and remains relatively short -- similar to the supervised phone model. Given that these models do not use a long context during prediction, we show that HuBERT can be run in streaming mode without modification to the architecture and without further fine-tuning.
>
---
#### [new 108] VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于越南语自动语音识别（ASR）任务，旨在解决低资源语言数据稀缺及现有系统成本高、延迟大的问题。提出VietASR方法，通过7万小时无标注数据预训练与50小时标注数据微调，构建轻量高效的模型，性能超越Whisper等现有系统。**

- **链接: [http://arxiv.org/pdf/2505.21527v1](http://arxiv.org/pdf/2505.21527v1)**

> **作者:** Jianheng Zhuo; Yifan Yang; Yiwen Shao; Yong Xu; Dong Yu; Kai Yu; Xie Chen
>
> **摘要:** Automatic speech recognition (ASR) has made remarkable progress but heavily relies on large-scale labeled data, which is scarce for low-resource languages like Vietnamese. While existing systems such as Whisper, USM, and MMS achieve promising performance, their efficacy remains inadequate in terms of training costs, latency, and accessibility. To address these issues, we propose VietASR, a novel ASR training pipeline that leverages vast amounts of unlabeled data and a small set of labeled data. Through multi-iteration ASR-biased self-supervised learning on a large-scale unlabeled dataset, VietASR offers a cost-effective and practical solution for enhancing ASR performance. Experiments demonstrate that pre-training on 70,000-hour unlabeled data and fine-tuning on merely 50-hour labeled data yield a lightweight but powerful ASR model. It outperforms Whisper Large-v3 and commercial ASR systems on real-world data. Our code and models will be open-sourced to facilitate research in low-resource ASR.
>
---
#### [new 109] 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于具身智能任务，解决大语言模型在3D动态环境中的长期时空记忆不足问题。提出3DMem-Bench基准测试和3DLLM-Mem模型，通过工作记忆令牌查询与选择性融合 episodic记忆中的时空特征，提升复杂环境中的任务执行效率，成功率达16.5%提升。**

- **链接: [http://arxiv.org/pdf/2505.22657v1](http://arxiv.org/pdf/2505.22657v1)**

> **作者:** Wenbo Hu; Yining Hong; Yanjun Wang; Leison Gao; Zibu Wei; Xingcheng Yao; Nanyun Peng; Yonatan Bitton; Idan Szpektor; Kai-Wei Chang
>
> **备注:** demos at: https://3dllm-mem.github.io
>
> **摘要:** Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks.
>
---
#### [new 110] From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于LLM内部机制分析任务，旨在解决简单命题真假判断的单一线性方向表示不足的问题。研究扩展了锥框架模型，通过多维锥结构中介真相相关行为，验证了因果干预、跨模型泛化及行为保留等效果，揭示LLMs中命题判断的多维结构。**

- **链接: [http://arxiv.org/pdf/2505.21800v1](http://arxiv.org/pdf/2505.21800v1)**

> **作者:** Stanley Yu; Vaidehi Bulusu; Oscar Yasunaga; Clayton Lau; Cole Blondin; Sean O'Brien; Kevin Zhu; Vasu Sharma
>
> **摘要:** Large Language Models (LLMs) exhibit strong conversational abilities but often generate falsehoods. Prior work suggests that the truthfulness of simple propositions can be represented as a single linear direction in a model's internal activations, but this may not fully capture its underlying geometry. In this work, we extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth. We identify multi-dimensional cones that causally mediate truth-related behavior across multiple LLM families. Our results are supported by three lines of evidence: (i) causal interventions reliably flip model responses to factual statements, (ii) learned cones generalize across model architectures, and (iii) cone-based interventions preserve unrelated model behavior. These findings reveal the richer, multidirectional structure governing simple true/false propositions in LLMs and highlight concept cones as a promising tool for probing abstract behaviors.
>
---
#### [new 111] Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出Test-time Immunization (TIM)框架，解决大语言模型对抗多样化jailbreak攻击的防御问题。通过训练gist token检测违规行为，检测到攻击时自适应安全微调模型，并解耦微调与检测模块防止性能下降，实现通用防御。**

- **链接: [http://arxiv.org/pdf/2505.22271v1](http://arxiv.org/pdf/2505.22271v1)**

> **作者:** Yongcan Yu; Yanbo Wang; Ran He; Jian Liang
>
> **备注:** Under Review
>
> **摘要:** While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM.
>
---
#### [new 112] Vision Meets Language: A RAG-Augmented YOLOv8 Framework for Coffee Disease Diagnosis and Farmer Assistance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出结合YOLOv8与RAG增强的LLM框架，用于咖啡叶病害诊断及农民支持。针对传统农业低效和LLM幻觉问题，通过视觉检测病害、语言模型生成精准诊断及环保治疗方案，减少农药使用，提供用户友好界面，推动精准农业应用。**

- **链接: [http://arxiv.org/pdf/2505.21544v1](http://arxiv.org/pdf/2505.21544v1)**

> **作者:** Semanto Mondal
>
> **备注:** There are 14 pages, 8 figures
>
> **摘要:** As a social being, we have an intimate bond with the environment. A plethora of things in human life, such as lifestyle, health, and food are dependent on the environment and agriculture. It comes under our responsibility to support the environment as well as agriculture. However, traditional farming practices often result in inefficient resource use and environmental challenges. To address these issues, precision agriculture has emerged as a promising approach that leverages advanced technologies to optimise agricultural processes. In this work, a hybrid approach is proposed that combines the three different potential fields of model AI: object detection, large language model (LLM), and Retrieval-Augmented Generation (RAG). In this novel framework, we have tried to combine the vision and language models to work together to identify potential diseases in the tree leaf. This study introduces a novel AI-based precision agriculture system that uses Retrieval Augmented Generation (RAG) to provide context-aware diagnoses and natural language processing (NLP) and YOLOv8 for crop disease detection. The system aims to tackle major issues with large language models (LLMs), especially hallucinations and allows for adaptive treatment plans and real-time disease detection. The system provides an easy-to-use interface to the farmers, which they can use to detect the different diseases related to coffee leaves by just submitting the image of the affected leaf the model will detect the diseases as well as suggest potential remediation methodologies which aim to lower the use of pesticides, preserving livelihoods, and encouraging environmentally friendly methods. With an emphasis on scalability, dependability, and user-friendliness, the project intends to improve RAG-integrated object detection systems for wider agricultural applications in the future.
>
---
#### [new 113] Sherlock: Self-Correcting Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型（VLM）推理任务，旨在解决其推理易出错、依赖大量标注数据及泛化能力差的问题。提出Sherlock框架，通过轨迹级自纠目标、视觉扰动构建偏好数据及动态β调参，使模型仅用20k标注数据获得自纠能力并持续自我优化，在8个基准测试中超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.22651v1](http://arxiv.org/pdf/2505.22651v1)**

> **作者:** Yi Ding; Ruqi Zhang
>
> **备注:** 27 pages
>
> **摘要:** Reasoning Vision-Language Models (VLMs) have shown promising performance on complex multimodal tasks. However, they still face significant challenges: they are highly sensitive to reasoning errors, require large volumes of annotated data or accurate verifiers, and struggle to generalize beyond specific domains. To address these limitations, we explore self-correction as a strategy to enhance reasoning VLMs. We first conduct an in-depth analysis of reasoning VLMs' self-correction abilities and identify key gaps. Based on our findings, we introduce Sherlock, a self-correction and self-improvement training framework. Sherlock introduces a trajectory-level self-correction objective, a preference data construction method based on visual perturbation, and a dynamic $\beta$ for preference tuning. Once the model acquires self-correction capabilities using only 20k randomly sampled annotated data, it continues to self-improve without external supervision. Built on the Llama3.2-Vision-11B model, Sherlock achieves remarkable results across eight benchmarks, reaching an average accuracy of 64.1 with direct generation and 65.4 after self-correction. It outperforms LLaVA-CoT (63.2), Mulberry (63.9), and LlamaV-o1 (63.4) while using less than 20% of the annotated data.
>
---
#### [new 114] Advancing Hearing Assessment: An ASR-Based Frequency-Specific Speech Test for Diagnosing Presbycusis
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文开发基于ASR的频率特异性听力测试，解决传统测听无法准确评估老年性耳聋的阈上缺陷和频段感知问题。通过模拟听损的语音处理，分析音素混淆模式（如高频辅音替换/删除），验证测试可有效区分正常与受损听力，为精准听力评估提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.22231v1](http://arxiv.org/pdf/2505.22231v1)**

> **作者:** Stefan Bleeck
>
> **摘要:** Traditional audiometry often fails to fully characterize the functional impact of hearing loss on speech understanding, particularly supra-threshold deficits and frequency-specific perception challenges in conditions like presbycusis. This paper presents the development and simulated evaluation of a novel Automatic Speech Recognition (ASR)-based frequency-specific speech test designed to provide granular diagnostic insights. Our approach leverages ASR to simulate the perceptual effects of moderate sloping hearing loss by processing speech stimuli under controlled acoustic degradation and subsequently analyzing phoneme-level confusion patterns. Key findings indicate that simulated hearing loss introduces specific phoneme confusions, predominantly affecting high-frequency consonants (e.g., alveolar/palatal to labiodental substitutions) and leading to significant phoneme deletions, consistent with the acoustic cues degraded in presbycusis. A test battery curated from these ASR-derived confusions demonstrated diagnostic value, effectively differentiating between simulated normal-hearing and hearing-impaired listeners in a comprehensive simulation. This ASR-driven methodology offers a promising avenue for developing objective, granular, and frequency-specific hearing assessment tools that complement traditional audiometry. Future work will focus on validating these findings with human participants and exploring the integration of advanced AI models for enhanced diagnostic precision.
>
---
#### [new 115] Mitigating Overthinking in Large Reasoning Models via Manifold Steering
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大型推理模型（LRMs）推理时的"过度思考"问题（如冗余验证导致效率低下），提出"流形导向"方法。通过分析模型激活空间，发现过度思考与低维流形相关，将干预方向投影至该流形以减少噪声，实验证实可减少71%输出token且提升/维持精度，适用于数学、代码生成等任务。**

- **链接: [http://arxiv.org/pdf/2505.22411v1](http://arxiv.org/pdf/2505.22411v1)**

> **作者:** Yao Huang; Huanran Chen; Shouwei Ruan; Yichi Zhang; Xingxing Wei; Yinpeng Dong
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Recent advances in Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in solving complex tasks such as mathematics and coding. However, these models frequently exhibit a phenomenon known as overthinking during inference, characterized by excessive validation loops and redundant deliberation, leading to substantial computational overheads. In this paper, we aim to mitigate overthinking by investigating the underlying mechanisms from the perspective of mechanistic interpretability. We first showcase that the tendency of overthinking can be effectively captured by a single direction in the model's activation space and the issue can be eased by intervening the activations along this direction. However, this efficacy soon reaches a plateau and even deteriorates as the intervention strength increases. We therefore systematically explore the activation space and find that the overthinking phenomenon is actually tied to a low-dimensional manifold, which indicates that the limited effect stems from the noises introduced by the high-dimensional steering direction. Based on this insight, we propose Manifold Steering, a novel approach that elegantly projects the steering direction onto the low-dimensional activation manifold given the theoretical approximation of the interference noise. Extensive experiments on DeepSeek-R1 distilled models validate that our method reduces output tokens by up to 71% while maintaining and even improving the accuracy on several mathematical benchmarks. Our method also exhibits robust cross-domain transferability, delivering consistent token reduction performance in code generation and knowledge-based QA tasks. Code is available at: https://github.com/Aries-iai/Manifold_Steering.
>
---
#### [new 116] Rethinking the Unsolvable: When In-Context Search Meets Test-Time Scaling
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究如何提升大语言模型（LLMs）在超复杂推理任务中的表现，针对现有评估方法低估LLM能力的问题，提出结合上下文搜索（in-context search）与推理时扩展（test-time scaling）的技术。通过实验在NP难题和现实规划任务中实现成功率达30倍提升，并理论证明该方法扩展了LLMs可处理的复杂度边界，挑战了其能力天花板的既有认知。**

- **链接: [http://arxiv.org/pdf/2505.22290v1](http://arxiv.org/pdf/2505.22290v1)**

> **作者:** Fanzeng Xia; Yidong Luo; Tinko Sebastian Bartels; Yaqi Xu; Tongxin Li
>
> **摘要:** Recent research has highlighted that Large Language Models (LLMs), even when trained to generate extended long reasoning steps, still face significant challenges on hard reasoning problems. However, much of the existing literature relies on direct prompting with simple in-context learning examples for evaluation, which largely overlooks advanced techniques to elicit LLMs' deliberate reasoning before drawing conclusions that LLMs hit a performance ceiling. In this paper, we systematically explore the combined potential of in-context search and test-time scaling on super hard reasoning tasks. We find that by employing advanced in-context search prompting to LLMs augmented with internal scaling, one can achieve transformative performance breakthroughs on tasks previously deemed "unsolvable" (e.g., reported success rates below 5%). We provide both empirical results and theoretical analysis of how this combination can unleash LLM reasoning capabilities: i) Empirically, on controlled NP-hard tasks and complex real-world planning benchmarks, our approach achieves up to a 30x improvement in success rates compared to previously reported results without any external mechanisms; ii) Theoretically, we show that in-context search prompting, when combined with internal scaling, significantly extends the complexity class of solvable reasoning problems. These findings challenge prevailing assumptions about the limitations of LLMs on complex tasks, indicating that current evaluation paradigms systematically underestimate their true potential. Our work calls for a critical reassessment of how LLM reasoning is benchmarked and a more robust evaluation strategy that fully captures the true capabilities of contemporary LLMs, which can lead to a better understanding of their operational reasoning boundaries in real-world deployments.
>
---
#### [new 117] Fluent but Culturally Distant: Can Regional Training Teach Cultural Understanding?
- **分类: physics.soc-ph; cs.AI; cs.CL; cs.CY**

- **简介: 该论文评估区域语言模型是否能更好体现本地文化。任务是对比印度本地与全球模型在文化价值观和实践上的对齐度，发现区域模型未显著优于全球模型，且微调可能损害文化能力。研究指出数据不足是主因，并呼吁构建文化代表性数据集。**

- **链接: [http://arxiv.org/pdf/2505.21548v1](http://arxiv.org/pdf/2505.21548v1)**

> **作者:** Dhruv Agarwal; Anya Shukla; Sunayana Sitaram; Aditya Vashistha
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) are used around the world but exhibit Western cultural tendencies. To address this cultural misalignment, many countries have begun developing "regional" LLMs tailored to local communities. Yet it remains unclear whether these models merely speak the language of their users or also reflect their cultural values and practices. Using India as a case study, we evaluate five Indic and five global LLMs along two key dimensions: values (via the Inglehart-Welzel map and GlobalOpinionQA) and practices (via CulturalBench and NormAd). Across all four tasks, we find that Indic models do not align more closely with Indian cultural norms than global models. In fact, an average American person is a better proxy for Indian cultural values than any Indic model. Even prompting strategies fail to meaningfully improve alignment. Ablations show that regional fine-tuning does not enhance cultural competence and may in fact hurt it by impeding recall of existing knowledge. We trace this failure to the scarcity of high-quality, untranslated, and culturally grounded pretraining and fine-tuning data. Our study positions cultural evaluation as a first-class requirement alongside multilingual benchmarks and offers a reusable methodology for developers. We call for deeper investments in culturally representative data to build and evaluate truly sovereign LLMs.
>
---
#### [new 118] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型推理时计算资源分配问题，探讨长思考链与多短链并行哪种更优。通过理论分析和实验表明，在图连通性等任务中，长链推理可带来指数级优势，验证了顺序扩展优于并行的场景。**

- **链接: [http://arxiv.org/pdf/2505.21825v1](http://arxiv.org/pdf/2505.21825v1)**

> **作者:** Parsa Mirtaheri; Ezra Edelman; Samy Jelassi; Eran Malach; Enric Boix-Adsera
>
> **摘要:** Inference-time computation has emerged as a promising scaling axis for improving large language model reasoning. However, despite yielding impressive performance, the optimal allocation of inference-time computation remains poorly understood. A central question is whether to prioritize sequential scaling (e.g., longer chains of thought) or parallel scaling (e.g., majority voting across multiple short chains of thought). In this work, we seek to illuminate the landscape of test-time scaling by demonstrating the existence of reasoning settings where sequential scaling offers an exponential advantage over parallel scaling. These settings are based on graph connectivity problems in challenging distributions of graphs. We validate our theoretical findings with comprehensive experiments across a range of language models, including models trained from scratch for graph connectivity with different chain of thought strategies as well as large reasoning models.
>
---
#### [new 119] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像上下文理解任务，旨在解决现有方法难以准确提取事件相关图像的深层时空与地理信息的问题。提出GETReason框架，通过分层多智能体推理整合全局事件、时间及地理数据，并设计GREAT评估指标，提升图像重要性与事件背景的关联分析。**

- **链接: [http://arxiv.org/pdf/2505.21863v1](http://arxiv.org/pdf/2505.21863v1)**

> **作者:** Shikhhar Siingh; Abhinav Rawat; Vivek Gupta; Chitta Baral
>
> **摘要:** Publicly significant images from events hold valuable contextual information, crucial for journalism and education. However, existing methods often struggle to extract this relevance accurately. To address this, we introduce GETReason (Geospatial Event Temporal Reasoning), a framework that moves beyond surface-level image descriptions to infer deeper contextual meaning. We propose that extracting global event, temporal, and geospatial information enhances understanding of an image's significance. Additionally, we introduce GREAT (Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric for evaluating reasoning-based image understanding. Our layered multi-agent approach, assessed using a reasoning-weighted metric, demonstrates that meaningful insights can be inferred, effectively linking images to their broader event context.
>
---
#### [new 120] ChemHAS: Hierarchical Agent Stacking for Enhancing Chemistry Tools
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ChemHAS方法，通过优化分层代理堆叠结构，利用有限数据提升化学工具预测精度，解决工具误差限制化学任务性能的问题，在四项任务中达最优，并揭示四种代理行为以增强可解释性。**

- **链接: [http://arxiv.org/pdf/2505.21569v1](http://arxiv.org/pdf/2505.21569v1)**

> **作者:** Zhucong Li; Bowei Zhang; Jin Xiao; Zhijian Zhou; Fenglei Cao; Jiaqing Liang; Yuan Qi
>
> **备注:** 9 pages
>
> **摘要:** Large Language Model (LLM)-based agents have demonstrated the ability to improve performance in chemistry-related tasks by selecting appropriate tools. However, their effectiveness remains limited by the inherent prediction errors of chemistry tools. In this paper, we take a step further by exploring how LLMbased agents can, in turn, be leveraged to reduce prediction errors of the tools. To this end, we propose ChemHAS (Chemical Hierarchical Agent Stacking), a simple yet effective method that enhances chemistry tools through optimizing agent-stacking structures from limited data. ChemHAS achieves state-of-the-art performance across four fundamental chemistry tasks, demonstrating that our method can effectively compensate for prediction errors of the tools. Furthermore, we identify and characterize four distinct agent-stacking behaviors, potentially improving interpretability and revealing new possibilities for AI agent applications in scientific research. Our code and dataset are publicly available at https: //anonymous.4open.science/r/ChemHAS-01E4/README.md.
>
---
#### [new 121] Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for Chest X-ray Report Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像分析任务，旨在解决多模态大语言模型生成胸片报告时存在的幻觉和临床错误问题。提出Look & Mark方法，通过整合放射科医生眼动数据和边界框标注到LLM的提示框架中，利用上下文学习提升模型性能，减少临床错误，无需重新训练即显著提高报告准确性。**

- **链接: [http://arxiv.org/pdf/2505.22222v1](http://arxiv.org/pdf/2505.22222v1)**

> **作者:** Yunsoo Kim; Jinge Wu; Su-Hwan Kim; Pardeep Vasudev; Jiashu Shen; Honghan Wu
>
> **摘要:** Recent advancements in multimodal Large Language Models (LLMs) have significantly enhanced the automation of medical image analysis, particularly in generating radiology reports from chest X-rays (CXR). However, these models still suffer from hallucinations and clinically significant errors, limiting their reliability in real-world applications. In this study, we propose Look & Mark (L&M), a novel grounding fixation strategy that integrates radiologist eye fixations (Look) and bounding box annotations (Mark) into the LLM prompting framework. Unlike conventional fine-tuning, L&M leverages in-context learning to achieve substantial performance gains without retraining. When evaluated across multiple domain-specific and general-purpose models, L&M demonstrates significant gains, including a 1.2% improvement in overall metrics (A.AVG) for CXR-LLaVA compared to baseline prompting and a remarkable 9.2% boost for LLaVA-Med. General-purpose models also benefit from L&M combined with in-context learning, with LLaVA-OV achieving an 87.3% clinical average performance (C.AVG)-the highest among all models, even surpassing those explicitly trained for CXR report generation. Expert evaluations further confirm that L&M reduces clinically significant errors (by 0.43 average errors per report), such as false predictions and omissions, enhancing both accuracy and reliability. These findings highlight L&M's potential as a scalable and efficient solution for AI-assisted radiology, paving the way for improved diagnostic workflows in low-resource clinical settings.
>
---
#### [new 122] UI-Evol: Automatic Knowledge Evolving for Computer Use Agents
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出UI-Evol模块，解决计算机使用代理中知识到执行的转化鸿沟。通过回溯环境交互动作序列并对比外部参考优化知识，显著提升任务成功率和执行稳定性，在OSWorld基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.21964v1](http://arxiv.org/pdf/2505.21964v1)**

> **作者:** Ziyun Zhang; Xinyi Liu; Xiaoyi Zhang; Jun Wang; Gang Chen; Yan Lu
>
> **摘要:** External knowledge has played a crucial role in the recent development of computer use agents. We identify a critical knowledge-execution gap: retrieved knowledge often fails to translate into effective real-world task execution. Our analysis shows even 90\% correct knowledge yields only 41\% execution success rate. To bridge this gap, we propose UI-Evol, a plug-and-play module for autonomous GUI knowledge evolution. UI-Evol consists of two stages: a Retrace Stage that extracts faithful objective action sequences from actual agent-environment interactions, and a Critique Stage that refines existing knowledge by comparing these sequences against external references. We conduct comprehensive experiments on the OSWorld benchmark with the state-of-the-art Agent S2. Our results demonstrate that UI-Evol not only significantly boosts task performance but also addresses a previously overlooked issue of high behavioral standard deviation in computer use agents, leading to superior performance on computer use tasks and substantially improved agent reliability.
>
---
#### [new 123] Train Sparse Autoencoders Efficiently by Utilizing Features Correlation
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大规模训练稀疏自编码器（SAEs）时编码器高计算开销的问题，提出KronSAE架构通过克罗内克积分解潜空间降低资源消耗，并引入mAND激活函数提升可解释性，优化了高效训练。**

- **链接: [http://arxiv.org/pdf/2505.22255v1](http://arxiv.org/pdf/2505.22255v1)**

> **作者:** Vadim Kurochkin; Yaroslav Aksenov; Daniil Laptev; Daniil Gavrilov; Nikita Balagansky
>
> **摘要:** Sparse Autoencoders (SAEs) have demonstrated significant promise in interpreting the hidden states of language models by decomposing them into interpretable latent directions. However, training SAEs at scale remains challenging, especially when large dictionary sizes are used. While decoders can leverage sparse-aware kernels for efficiency, encoders still require computationally intensive linear operations with large output dimensions. To address this, we propose KronSAE, a novel architecture that factorizes the latent representation via Kronecker product decomposition, drastically reducing memory and computational overhead. Furthermore, we introduce mAND, a differentiable activation function approximating the binary AND operation, which improves interpretability and performance in our factorized framework.
>
---
#### [new 124] Learning Compositional Behaviors from Demonstration and Language
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出BLADE框架，属于长期机器人操作任务，解决复杂场景下机器人泛化执行新任务的挑战。通过整合模仿学习与模型规划，利用语言标注的示范数据，自动提取结构化高阶动作表示（含视觉感知前提/效果及神经网络策略），无需人工标注，实现在模拟与真实机器人上的复杂操作验证。**

- **链接: [http://arxiv.org/pdf/2505.21981v1](http://arxiv.org/pdf/2505.21981v1)**

> **作者:** Weiyu Liu; Neil Nie; Ruohan Zhang; Jiayuan Mao; Jiajun Wu
>
> **备注:** Presented at CoRL 2024 and as an Oral Presentation at the 2024 CoRL LEAP Workshop. The first two authors contributed equally. The last two authors jointly advised the project. For videos and additional results, visit: https://blade-bot.github.io/
>
> **摘要:** We introduce Behavior from Language and Demonstration (BLADE), a framework for long-horizon robotic manipulation by integrating imitation learning and model-based planning. BLADE leverages language-annotated demonstrations, extracts abstract action knowledge from large language models (LLMs), and constructs a library of structured, high-level action representations. These representations include preconditions and effects grounded in visual perception for each high-level action, along with corresponding controllers implemented as neural network-based policies. BLADE can recover such structured representations automatically, without manually labeled states or symbolic definitions. BLADE shows significant capabilities in generalizing to novel situations, including novel initial states, external state perturbations, and novel goals. We validate the effectiveness of our approach both in simulation and on real robots with a diverse set of objects with articulated parts, partial observability, and geometric constraints.
>
---
#### [new 125] Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像-文本检索任务，旨在解决CLIP模型因固定图像分辨率和有限上下文导致的细粒度跨模态理解不足问题。通过教师-学生蒸馏框架，利用YOLO提取的图像区域与文本片段的双向注意力生成增强嵌入，并采用混合损失函数训练轻量学生模型，在小规模数据下显著提升检索指标，同时保留94%的零样本分类性能。**

- **链接: [http://arxiv.org/pdf/2505.21549v1](http://arxiv.org/pdf/2505.21549v1)**

> **作者:** Daniel Csizmadia; Andrei Codreanu; Victor Sim; Vighnesh Prabeau; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** We present Distill CLIP (DCLIP), a fine-tuned variant of the CLIP model that enhances multimodal image-text retrieval while preserving the original model's strong zero-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine-grained cross-modal understanding. DCLIP addresses these challenges through a meta teacher-student distillation framework, where a cross-modal transformer teacher is fine-tuned to produce enriched embeddings via bidirectional cross-attention between YOLO-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions-just a fraction of CLIP's original dataset-DCLIP significantly improves image-text retrieval metrics (Recall@K, MAP), while retaining approximately 94% of CLIP's zero-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade-off between task specialization and generalization, offering a resource-efficient, domain-adaptive, and detail-sensitive solution for advanced vision-language tasks. Code available at https://anonymous.4open.science/r/DCLIP-B772/README.md.
>
---
#### [new 126] Visual Cues Support Robust Turn-taking Prediction in Noise
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究噪音环境下对话轮换预测，发现现有音频模型在10dB音乐噪声中准确率骤降至52%。提出多模态模型（融合视觉特征）将准确率提升至72%，但泛化至新噪声类型受限。任务：提升噪声中预测鲁棒性；问题：模型对噪音敏感；方法：多模态建模与数据依赖性分析。**

- **链接: [http://arxiv.org/pdf/2505.22088v1](http://arxiv.org/pdf/2505.22088v1)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** 5 pages
>
> **摘要:** Accurate predictive turn-taking models (PTTMs) are essential for naturalistic human-robot interaction. However, little is known about their performance in noise. This study therefore explores PTTM performance in types of noise likely to be encountered once deployed. Our analyses reveal PTTMs are highly sensitive to noise. Hold/shift accuracy drops from 84% in clean speech to just 52% in 10 dB music noise. Training with noisy data enables a multimodal PTTM, which includes visual features to better exploit visual cues, with 72% accuracy in 10 dB music noise. The multimodal PTTM outperforms the audio-only PTTM across all noise types and SNRs, highlighting its ability to exploit visual cues; however, this does not always generalise to new types of noise. Analysis also reveals that successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions. We make code publicly available for future research.
>
---
#### [new 127] R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.SC**

- **简介: 该论文提出R1-Code-Interpreter，通过监督微调和强化学习扩展文本模型，解决LLMs在代码执行类任务（如计算、算法推理）中推理不足的问题。其训练模型自主生成多步代码查询，测试了不同策略，最终14B模型在任务准确率从44%提升至64.1%，接近GPT-4o代码解释器水平。**

- **链接: [http://arxiv.org/pdf/2505.21668v1](http://arxiv.org/pdf/2505.21668v1)**

> **作者:** Yongchao Chen; Yueying Liu; Junwei Zhou; Yilun Hao; Jingquan Wang; Yang Zhang; Chuchu Fan
>
> **备注:** 33 pages, 8 figures
>
> **摘要:** Despite advances in reasoning and planning of R1-like models, Large Language Models (LLMs) still struggle with tasks requiring precise computation, symbolic manipulation, optimization, and algorithmic reasoning, in which textual reasoning lacks the rigor of code execution. A key challenge is enabling LLMs to decide when to use textual reasoning versus code generation. While OpenAI trains models to invoke a Code Interpreter as needed, public research lacks guidance on aligning pre-trained LLMs to effectively leverage code and generalize across diverse tasks. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. We curate 144 reasoning and planning tasks (107 for training, 37 for testing), each with over 200 diverse questions. We fine-tune Qwen-2.5 models (3B/7B/14B) using various SFT and RL strategies, investigating different answer formats, reasoning vs. non-reasoning models, cold vs. warm starts, GRPO vs. PPO, and masked vs. unmasked code outputs. Unlike prior RL work on narrow domains, we find that Code Interpreter training is significantly harder due to high task diversity and expensive code execution, highlighting the critical role of the SFT stage. Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.0\% to 64.1\%, outperforming GPT-4o (text-only: 58.6\%) and approaching GPT-4o with Code Interpreter (70.9\%), with the emergent self-checking behavior via code generation. Datasets, Codes, and Models are available at https://github.com/yongchao98/R1-Code-Interpreter and https://huggingface.co/yongchao98.
>
---
#### [new 128] The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习（RL）在语言模型推理中的熵机制，旨在解决策略熵崩溃导致的探索能力下降和性能饱和问题。通过建立性能与熵的转换方程，揭示熵与性能的 trade-off 关系，并提出Clip-Cov和KL-Cov方法控制高协方差token更新，提升探索效果和最终性能。**

- **链接: [http://arxiv.org/pdf/2505.22617v1](http://arxiv.org/pdf/2505.22617v1)**

> **作者:** Ganqu Cui; Yuchen Zhang; Jiacheng Chen; Lifan Yuan; Zhi Wang; Yuxin Zuo; Haozhan Li; Yuchen Fan; Huayu Chen; Weize Chen; Zhiyuan Liu; Hao Peng; Lei Bai; Wanli Ouyang; Yu Cheng; Bowen Zhou; Ning Ding
>
> **摘要:** This paper aims to overcome a major obstacle in scaling RL for reasoning with LLMs, namely the collapse of policy entropy. Such phenomenon is consistently observed across vast RL runs without entropy intervention, where the policy entropy dropped sharply at the early training stage, this diminished exploratory ability is always accompanied with the saturation of policy performance. In practice, we establish a transformation equation R=-a*e^H+b between entropy H and downstream performance R. This empirical law strongly indicates that, the policy performance is traded from policy entropy, thus bottlenecked by its exhaustion, and the ceiling is fully predictable H=0, R=-a+b. Our finding necessitates entropy management for continuous exploration toward scaling compute for RL. To this end, we investigate entropy dynamics both theoretically and empirically. Our derivation highlights that, the change in policy entropy is driven by the covariance between action probability and the change in logits, which is proportional to its advantage when using Policy Gradient-like algorithms. Empirical study shows that, the values of covariance term and entropy differences matched exactly, supporting the theoretical conclusion. Moreover, the covariance term stays mostly positive throughout training, further explaining why policy entropy would decrease monotonically. Through understanding the mechanism behind entropy dynamics, we motivate to control entropy by restricting the update of high-covariance tokens. Specifically, we propose two simple yet effective techniques, namely Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively. Experiments show that these methods encourage exploration, thus helping policy escape entropy collapse and achieve better downstream performance.
>
---
#### [new 129] Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于LLMs安全推理任务，旨在解决现有安全措施（如过度拒绝、越狱漏洞）中高质量政策嵌入式CoT数据集生成困难的问题。提出AIDSAFE方法，通过多智能体迭代辩论生成安全推理数据，并设计数据优化模块剔除冗余/欺骗内容，同时补充信念增强技术生成偏好数据，提升安全泛化与抗攻击能力。**

- **链接: [http://arxiv.org/pdf/2505.21784v1](http://arxiv.org/pdf/2505.21784v1)**

> **作者:** Tharindu Kumarage; Ninareh Mehrabi; Anil Ramakrishna; Xinyan Zhao; Richard Zemel; Kai-Wei Chang; Aram Galstyan; Rahul Gupta; Charith Peris
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Safety reasoning is a recent paradigm where LLMs reason over safety policies before generating responses, thereby mitigating limitations in existing safety measures such as over-refusal and jailbreak vulnerabilities. However, implementing this paradigm is challenging due to the resource-intensive process of creating high-quality policy-embedded chain-of-thought (CoT) datasets while ensuring reasoning remains accurate and free from hallucinations or policy conflicts. To tackle this, we propose AIDSAFE: Agentic Iterative Deliberation for Safety Reasoning, a novel data generation recipe that leverages multi-agent deliberation to iteratively expand reasoning on safety policies. A data refiner stage in AIDSAFE ensures high-quality outputs by eliminating repetitive, redundant, and deceptive thoughts. AIDSAFE-generated CoTs provide a strong foundation for supervised fine-tuning (SFT)-based safety training. Additionally, to address the need of preference data in alignment stages, such as DPO training, we introduce a supplemental recipe that uses belief augmentation to create distinct selected and rejected CoT samples. Our evaluations demonstrate that AIDSAFE-generated CoTs achieve superior policy adherence and reasoning quality. Consequently, we show that fine-tuning open-source LLMs on these CoTs can significantly improve safety generalization and jailbreak robustness while maintaining acceptable utility and over-refusal accuracy. AIDSAFE-generated CoT datasets can be found here: https://huggingface.co/datasets/AmazonScience/AIDSAFE
>
---
#### [new 130] Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language
- **分类: cs.CV; cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文提出通过低维视觉-语言属性对齐进行灵活工具选择的框架，解决计算模型在模拟人类工具认知能力上的不足。构建ToolNet数据集，利用视觉模型提取工具图像属性，语言模型解析任务需求属性，通过关键操作属性（如握持性）匹配实现74%选择准确率，参数效率高且性能接近大模型。**

- **链接: [http://arxiv.org/pdf/2505.22146v1](http://arxiv.org/pdf/2505.22146v1)**

> **作者:** Guangfu Hao; Haojie Wen; Liangxuna Guo; Yang Chen; Yanchao Bi; Shan Yu
>
> **摘要:** Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Ablation studies revealed that manipulation-related attributes (graspability, hand-relatedness, elongation) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks.
>
---
#### [new 131] Skywork Open Reasoner 1 Technical Report
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Skywork-OR1，一种基于强化学习优化长链思维模型的方法，旨在提升大模型推理能力。通过改进DeepSeek-R1-Distill模型，其32B/7B版本在AIME24等基准测试中分别提升15%/13.9%准确率。工作包括消融实验验证训练组件、分析熵塌缩对性能的影响，并开源模型与数据。任务属强化学习优化LLM推理，解决训练效果与稳定性问题。**

- **链接: [http://arxiv.org/pdf/2505.22312v1](http://arxiv.org/pdf/2505.22312v1)**

> **作者:** Jujie He; Jiacai Liu; Chris Yuhao Liu; Rui Yan; Chaojie Wang; Peng Cheng; Xiaoyu Zhang; Fuxiang Zhang; Jiacheng Xu; Wei Shen; Siyuan Li; Liang Zeng; Tianwen Wei; Cheng Cheng; Bo An; Yang Liu; Yahui Zhou
>
> **摘要:** The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets.
>
---
#### [new 132] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出多适配器集成方法，解决多任务微调效率问题。通过将数据集分组训练适配器并加权组合，利用基模型梯度的一阶近似快速评估组合效果，误差低于5%，计算加速105倍。在Llama等模型上，准确率提升10%，FLOPs仅增9%。**

- **链接: [http://arxiv.org/pdf/2505.21930v1](http://arxiv.org/pdf/2505.21930v1)**

> **作者:** Dongyue Li; Ziniu Zhang; Lu Wang; Hongyang R. Zhang
>
> **备注:** 17 pages. To appear in ACL'25
>
> **摘要:** This paper develops an ensemble method for fine-tuning a language model to multiple datasets. Existing methods, such as quantized LoRA (QLoRA), are efficient when adapting to a single dataset. When training on multiple datasets of different tasks, a common setup in practice, it remains unclear how to design an efficient adaptation for fine-tuning language models. We propose to use an ensemble of multiple smaller adapters instead of a single adapter per task. We design an efficient algorithm that partitions $n$ datasets into $m$ groups, where $m$ is typically much smaller than $n$ in practice, and train one adapter for each group before taking a weighted combination to form the ensemble. The algorithm leverages a first-order approximation property of low-rank adaptation to quickly obtain the fine-tuning performances of dataset combinations since methods like LoRA stay close to the base model. Hence, we use the gradients of the base model to estimate its behavior during fine-tuning. Empirically, this approximation holds with less than $1\%$ error on models with up to $34$ billion parameters, leading to an estimation of true fine-tuning performances under $5\%$ error while speeding up computation compared to base fine-tuning by $105$ times. When applied to fine-tune Llama and GPT models on ten text classification tasks, our approach provides up to $10\%$ higher average test accuracy over QLoRA, with only $9\%$ more FLOPs. On a Llama model with $34$ billion parameters, an ensemble of QLoRA increases test accuracy by $3\%$ compared to QLoRA, with only $8\%$ more FLOPs.
>
---
#### [new 133] Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation
- **分类: cs.MA; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于城市移动性模拟任务，旨在解决传统基于规则的ABM模型在人口多样性和行为真实性方面的不足。研究将LLM与ABM结合，通过生成合成人口资料、分配动态地点及模拟个性化路线提升仿真效果，应用于台北市实证，为城市规划提供行为分析与政策依据。**

- **链接: [http://arxiv.org/pdf/2505.21880v1](http://arxiv.org/pdf/2505.21880v1)**

> **作者:** Yu-Lun Song; Chung-En Tsern; Che-Cheng Wu; Yu-Ming Chang; Syuan-Bo Huang; Wei-Chu Chen; Michael Chia-Liang Lin; Yu-Ta Lin
>
> **备注:** 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
>
> **摘要:** This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
>
---
#### [new 134] Fostering Video Reasoning via Next-Event Prediction
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出通过预测下一事件（NEP）提升视频时间推理。针对现有任务依赖标注或混杂空间信息的问题，该工作将视频分为过去和未来帧，训练模型预测未来事件总结，并构建V1-33K数据集及评估基准FutureBench，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22457v1](http://arxiv.org/pdf/2505.22457v1)**

> **作者:** Haonan Wang; Hongfu Liu; Xiangyan Liu; Chao Du; Kenji Kawaguchi; Ye Wang; Tianyu Pang
>
> **摘要:** Next-token prediction serves as the foundational learning task enabling reasoning in LLMs. But what should the learning task be when aiming to equip MLLMs with temporal reasoning capabilities over video inputs? Existing tasks such as video question answering often rely on annotations from humans or much stronger MLLMs, while video captioning tends to entangle temporal reasoning with spatial information. To address this gap, we propose next-event prediction (NEP), a learning task that harnesses future video segments as a rich, self-supervised signal to foster temporal reasoning. We segment each video into past and future frames: the MLLM takes the past frames as input and predicts a summary of events derived from the future frames, thereby encouraging the model to reason temporally in order to complete the task. To support this task, we curate V1-33K, a dataset comprising 33,000 automatically extracted video segments spanning diverse real-world scenarios. We further explore a range of video instruction-tuning strategies to study their effects on temporal reasoning. To evaluate progress, we introduce FutureBench to assess coherence in predicting unseen future events. Experiments validate that NEP offers a scalable and effective training paradigm for fostering temporal reasoning in MLLMs.
>
---
#### [new 135] EnsemW2S: Enhancing Weak-to-Strong Generalization with Large Language Model Ensembles
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于弱到强（W2S）泛化任务，旨在利用小型人类水平模型提升大模型性能。针对弱专家模型难以处理复杂任务的问题，提出EnsemW2S方法，通过迭代集成多弱模型优化不足，提升监督能力。实验显示，在分布内和分布外数据（含难度维度）分别提升4%、6%等，有效增强泛化。**

- **链接: [http://arxiv.org/pdf/2505.21959v1](http://arxiv.org/pdf/2505.21959v1)**

> **作者:** Aakriti Agrawal; Mucong Ding; Zora Che; Chenghao Deng; Anirudh Satheesh; Bang An; Bayan Bruss; John Langford; Furong Huang
>
> **备注:** Superalignment. arXiv admin note: substantial text overlap with arXiv:2410.04571
>
> **摘要:** With Large Language Models (LLMs) rapidly approaching and potentially surpassing human-level performance, it has become imperative to develop approaches capable of effectively supervising and enhancing these powerful models using smaller, human-level models exposed to only human-level data. We address this critical weak-to-strong (W2S) generalization challenge by proposing a novel method aimed at improving weak experts, by training on the same limited human-level data, enabling them to generalize to complex, super-human-level tasks. Our approach, called \textbf{EnsemW2S}, employs a token-level ensemble strategy that iteratively combines multiple weak experts, systematically addressing the shortcomings identified in preceding iterations. By continuously refining these weak models, we significantly enhance their collective ability to supervise stronger student models. We extensively evaluate the generalization performance of both the ensemble of weak experts and the subsequent strong student model across in-distribution (ID) and out-of-distribution (OOD) datasets. For OOD, we specifically introduce question difficulty as an additional dimension for defining distributional shifts. Our empirical results demonstrate notable improvements, achieving 4\%, and 3.2\% improvements on ID datasets and, upto 6\% and 2.28\% on OOD datasets for experts and student models respectively, underscoring the effectiveness of our proposed method in advancing W2S generalization.
>
---
#### [new 136] How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文评估大型语言模型（LLMs）在3D虚拟形象运动控制中的能力，研究其对人类运动知识的掌握。通过设计20个动作指令，测试LLMs的高阶动作规划（如分步骤分解动作）与低阶身体部位定位能力，发现其擅长理解高层次动作但难以精确控制高自由度身体部位，适合创意动作设计但不适用于精准时空参数生成。**

- **链接: [http://arxiv.org/pdf/2505.21531v1](http://arxiv.org/pdf/2505.21531v1)**

> **作者:** Kunhang Li; Jason Naradowsky; Yansong Feng; Yusuke Miyao
>
> **摘要:** We explore Large Language Models (LLMs)' human motion knowledge through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations as a clear verification lens for human evaluators. Through carefully designed 20 representative motion instructions with full coverage of basic movement primitives and balanced body part usage, we conduct comprehensive evaluations including human assessment of both generated animations and high-level movement plans, as well as automatic comparison with oracle positions in low-level planning. We find that LLMs are strong at interpreting the high-level body movements but struggle with precise body part positioning. While breaking down motion queries into atomic components improves planning performance, LLMs have difficulty with multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximation for general spatial descriptions, but fail to handle precise spatial specifications in text, and the precise spatial-temporal parameters needed for avatar control. Notably, LLMs show promise in conceptualizing creative motions and distinguishing culturally-specific motion patterns.
>
---
#### [new 137] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属不确定性量化任务，针对LLM代理在开放交互场景中传统二元不确定性（aleatoric/epistemic）分类失效问题，提出三方向：欠规格不确定性（用户信息不全时）、交互学习（通过追问减小上下文不确定性）、输出不确定性（用语言表达多维不确定性），以提升交互透明度与可信度。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22655v1](http://arxiv.org/pdf/2505.22655v1)**

> **作者:** Michael Kirchhof; Gjergji Kasneci; Enkelejda Kasneci
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Large-language models (LLMs) and chatbot agents are known to provide wrong outputs at times, and it was recently found that this can never be fully prevented. Hence, uncertainty quantification plays a crucial role, aiming to quantify the level of ambiguity in either one overall number or two numbers for aleatoric and epistemic uncertainty. This position paper argues that this traditional dichotomy of uncertainties is too limited for the open and interactive setup that LLM agents operate in when communicating with a user, and that we need to research avenues that enrich uncertainties in this novel scenario. We review the literature and find that popular definitions of aleatoric and epistemic uncertainties directly contradict each other and lose their meaning in interactive LLM agent settings. Hence, we propose three novel research directions that focus on uncertainties in such human-computer interactions: Underspecification uncertainties, for when users do not provide all information or define the exact task at the first go, interactive learning, to ask follow-up questions and reduce the uncertainty about the current context, and output uncertainties, to utilize the rich language and speech space to express uncertainties as more than mere numbers. We expect that these new ways of dealing with and communicating uncertainties will lead to LLM agent interactions that are more transparent, trustworthy, and intuitive.
>
---
#### [new 138] FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决模型在多模态分布偏移下的鲁棒性不足问题。作者构建了FRAMES-VQA基准，整合10个数据集分类为ID、近/远OOD场景，通过马氏距离量化分布偏移，分析模态间交互及重要性，为开发鲁棒微调方法提供指导。**

- **链接: [http://arxiv.org/pdf/2505.21755v1](http://arxiv.org/pdf/2505.21755v1)**

> **作者:** Chengyue Huang; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at https://github.com/chengyuehuang511/FRAMES-VQA .
>
---
#### [new 139] Improving Brain-to-Image Reconstruction via Fine-Grained Text Bridging
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于脑-图像重建任务，旨在解决现有方法重建图像细节缺失和语义不一致的问题。提出FgB2I方法，通过细粒度文本作为桥梁，分三阶段工作：利用视觉语言模型增强细节、基于fMRI信号解码细粒度文本（采用三指标优化），最后整合文本进行图像重建，提升重建精度。**

- **链接: [http://arxiv.org/pdf/2505.22150v1](http://arxiv.org/pdf/2505.22150v1)**

> **作者:** Runze Xia; Shuo Feng; Renzhi Wang; Congchi Yin; Xuyun Wen; Piji Li
>
> **备注:** CogSci2025
>
> **摘要:** Brain-to-Image reconstruction aims to recover visual stimuli perceived by humans from brain activity. However, the reconstructed visual stimuli often missing details and semantic inconsistencies, which may be attributed to insufficient semantic information. To address this issue, we propose an approach named Fine-grained Brain-to-Image reconstruction (FgB2I), which employs fine-grained text as bridge to improve image reconstruction. FgB2I comprises three key stages: detail enhancement, decoding fine-grained text descriptions, and text-bridged brain-to-image reconstruction. In the detail-enhancement stage, we leverage large vision-language models to generate fine-grained captions for visual stimuli and experimentally validate its importance. We propose three reward metrics (object accuracy, text-image semantic similarity, and image-image semantic similarity) to guide the language model in decoding fine-grained text descriptions from fMRI signals. The fine-grained text descriptions can be integrated into existing reconstruction methods to achieve fine-grained Brain-to-Image reconstruction.
>
---
#### [new 140] Thinking with Generated Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出一种多模态视觉推理新范式，使大型模型能通过生成中间视觉步骤（如分解任务、自检优化）进行主动思考，突破仅依赖输入图像或纯文本推理的限制。通过生成视觉子目标及自我批判机制，显著提升复杂场景处理能力（如生物、建筑、刑侦领域），实现50%性能提升。任务属多模态视觉推理，解决模型视觉思维能力不足问题。**

- **链接: [http://arxiv.org/pdf/2505.22525v1](http://arxiv.org/pdf/2505.22525v1)**

> **作者:** Ethan Chern; Zhulin Hu; Steffi Chern; Siqi Kou; Jiadi Su; Yan Ma; Zhijie Deng; Pengfei Liu
>
> **摘要:** We present Thinking with Generated Images, a novel paradigm that fundamentally transforms how large multimodal models (LMMs) engage with visual reasoning by enabling them to natively think across text and vision modalities through spontaneous generation of intermediate visual thinking steps. Current visual reasoning with LMMs is constrained to either processing fixed user-provided images or reasoning solely through text-based chain-of-thought (CoT). Thinking with Generated Images unlocks a new dimension of cognitive capability where models can actively construct intermediate visual thoughts, critique their own visual hypotheses, and refine them as integral components of their reasoning process. We demonstrate the effectiveness of our approach through two complementary mechanisms: (1) vision generation with intermediate visual subgoals, where models decompose complex visual tasks into manageable components that are generated and integrated progressively, and (2) vision generation with self-critique, where models generate an initial visual hypothesis, analyze its shortcomings through textual reasoning, and produce refined outputs based on their own critiques. Our experiments on vision generation benchmarks show substantial improvements over baseline approaches, with our models achieving up to 50% (from 38% to 57%) relative improvement in handling complex multi-object scenarios. From biochemists exploring novel protein structures, and architects iterating on spatial designs, to forensic analysts reconstructing crime scenes, and basketball players envisioning strategic plays, our approach enables AI models to engage in the kind of visual imagination and iterative refinement that characterizes human creative, analytical, and strategic thinking. We release our open-source suite at https://github.com/GAIR-NLP/thinking-with-generated-images.
>
---
#### [new 141] Pitfalls of Rule- and Model-based Verifiers -- A Case Study on Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究了强化学习可验证奖励（RLVR）中规则和模型验证器的缺陷。针对数学推理任务，分析发现规则验证器因格式差异导致漏判（假阴性），影响训练；模型验证器虽精度高，但易被攻击（假阳性），导致奖励虚高。工作包括对比两类验证器在静态评估和RL训练中的表现，揭示其风险，为改进奖励系统提供依据。**

- **链接: [http://arxiv.org/pdf/2505.22203v1](http://arxiv.org/pdf/2505.22203v1)**

> **作者:** Yuzhen Huang; Weihao Zeng; Xingshan Zeng; Qi Zhu; Junxian He
>
> **摘要:** Trustworthy verifiers are essential for the success of reinforcement learning with verifiable reward (RLVR), which is the core methodology behind various large reasoning models such as DeepSeek-R1. In complex domains like mathematical reasoning, rule-based verifiers have been widely adopted in previous works to train strong reasoning models. However, the reliability of these verifiers and their impact on the RL training process remain poorly understood. In this work, we take mathematical reasoning as a case study and conduct a comprehensive analysis of various verifiers in both static evaluation and RL training scenarios. First, we find that current open-source rule-based verifiers often fail to recognize equivalent answers presented in different formats across multiple commonly used mathematical datasets, resulting in non-negligible false negative rates. This limitation adversely affects RL training performance and becomes more pronounced as the policy model gets stronger. Subsequently, we investigate model-based verifiers as a potential solution to address these limitations. While the static evaluation shows that model-based verifiers achieve significantly higher verification accuracy, further analysis and RL training results imply that they are highly susceptible to hacking, where they misclassify certain patterns in responses as correct (i.e., false positives). This vulnerability is exploited during policy model optimization, leading to artificially inflated rewards. Our findings underscore the unique risks inherent to both rule-based and model-based verifiers, aiming to offer valuable insights to develop more robust reward systems in reinforcement learning.
>
---
#### [new 142] Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别评估任务，旨在解决测试集数据污染导致模型性能评估偏差的问题。研究发现LibriSpeech和Common Voice测试集内容存在于LLM预训练数据中，导致模型可能复现训练见过的文本。通过对比污染与非污染模型，揭示其对结果的误导性，强调需用独立数据评估LLM语音系统。**

- **链接: [http://arxiv.org/pdf/2505.22251v1](http://arxiv.org/pdf/2505.22251v1)**

> **作者:** Yuan Tseng; Titouan Parcollet; Rogier van Dalen; Shucong Zhang; Sourav Bhattacharya
>
> **摘要:** Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure the impact of contamination, LLMs trained with or without contamination are compared, showing that a contaminated LLM is more likely to generate test sentences it has seen during training. Speech recognisers using contaminated LLMs shows only subtle differences in error rates, but assigns significantly higher probabilities to transcriptions seen during training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data.
>
---
#### [new 143] Capability-Based Scaling Laws for LLM Red-Teaming
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文研究LLM红队测试的扩展规律，解决目标模型能力超越攻击者时传统方法失效的问题。通过测试500余对模型，发现攻击成功率与能力差距相关，并提出预测模型，指出需控制模型的操纵能力以应对未来风险。**

- **链接: [http://arxiv.org/pdf/2505.20162v1](http://arxiv.org/pdf/2505.20162v1)**

> **作者:** Alexander Panfilov; Paul Kassianik; Maksym Andriushchenko; Jonas Geiping
>
> **摘要:** As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.
>
---
#### [new 144] MapStory: LLM-Powered Text-Driven Map Animation Prototyping with Human-in-the-Loop Editing
- **分类: cs.HC; cs.AI; cs.CL; cs.MM; H.5.2, H.5.1**

- **简介: 论文提出MapStory，一种基于LLM的文本驱动地图动画工具，解决专业地图动画制作门槛高、迭代慢的问题。通过自动分解脚本为动画组件、结合LLM查询地理数据及交互式编辑，支持用户高效创作。基于专家访谈和视频分析设计，实验证明其易用性及创意促进作用。**

- **链接: [http://arxiv.org/pdf/2505.21966v1](http://arxiv.org/pdf/2505.21966v1)**

> **作者:** Aditya Gunturu; Ben Pearman; Keiichi Ihara; Morteza Faraji; Bryan Wang; Rubaiat Habib Kazi; Ryo Suzuki
>
> **备注:** 16 pages and 15 figures
>
> **摘要:** We introduce MapStory, an LLM-powered animation authoring tool that generates editable map animation sequences directly from natural language text. Given a user-written script, MapStory leverages an agentic architecture to automatically produce a scene breakdown, which decomposes the script into key animation building blocks such as camera movements, visual highlights, and animated elements. Our system includes a researcher component that accurately queries geospatial information by leveraging an LLM with web search, enabling the automatic extraction of relevant regions, paths, and coordinates while allowing users to edit and query for changes or additional information to refine the results. Additionally, users can fine-tune parameters of these blocks through an interactive timeline editor. We detail the system's design and architecture, informed by formative interviews with professional animators and an analysis of 200 existing map animation videos. Our evaluation, which includes expert interviews (N=5) and a usability study (N=12), demonstrates that MapStory enables users to create map animations with ease, facilitates faster iteration, encourages creative exploration, and lowers barriers to creating map-centric stories.
>
---
#### [new 145] RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像重描述任务，旨在解决现有方法因幻觉和细节缺失导致的描述不准确和不完整问题。提出RICO框架，通过文本到图像模型重建图像并迭代修正描述，同时开发RICO-Flash以降低计算成本，实验显示其在准确性和完整性上超越基线约10%。**

- **链接: [http://arxiv.org/pdf/2505.22613v1](http://arxiv.org/pdf/2505.22613v1)**

> **作者:** Yuchi Wang; Yishuo Cai; Shuhuai Ren; Sihan Yang; Linli Yao; Yuanxin Liu; Yuanxing Zhang; Pengfei Wan; Xu Sun
>
> **备注:** code: https://github.com/wangyuchi369/RICO
>
> **摘要:** Image recaptioning is widely used to generate training datasets with enhanced quality for various multimodal tasks. Existing recaptioning methods typically rely on powerful multimodal large language models (MLLMs) to enhance textual descriptions, but often suffer from inaccuracies due to hallucinations and incompleteness caused by missing fine-grained details. To address these limitations, we propose RICO, a novel framework that refines captions through visual reconstruction. Specifically, we leverage a text-to-image model to reconstruct a caption into a reference image, and prompt an MLLM to identify discrepancies between the original and reconstructed images to refine the caption. This process is performed iteratively, further progressively promoting the generation of more faithful and comprehensive descriptions. To mitigate the additional computational cost induced by the iterative process, we introduce RICO-Flash, which learns to generate captions like RICO using DPO. Extensive experiments demonstrate that our approach significantly improves caption accuracy and completeness, outperforms most baselines by approximately 10% on both CapsBench and CompreCap. Code released at https://github.com/wangyuchi369/RICO.
>
---
#### [new 146] Born a Transformer -- Always a Transformer?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer架构在LLMs中的长度泛化限制。通过检索/复制任务，发现预训练模型在右侧（归纳）检索优于左侧（反归纳），此不对称性可通过微调消除，揭示Transformer内部电路差异导致能力局限，实验证明预训练未克服根本限制。**

- **链接: [http://arxiv.org/pdf/2505.21785v1](http://arxiv.org/pdf/2505.21785v1)**

> **作者:** Yana Veitsman; Mayank Jobanputra; Yash Sarrof; Aleksandra Bakalova; Vera Demberg; Ellie Pavlick; Michael Hahn
>
> **摘要:** Transformers have theoretical limitations in modeling certain sequence-to-sequence tasks, yet it remains largely unclear if these limitations play a role in large-scale pretrained LLMs, or whether LLMs might effectively overcome these constraints in practice due to the scale of both the models themselves and their pretraining data. We explore how these architectural constraints manifest after pretraining, by studying a family of $\textit{retrieval}$ and $\textit{copying}$ tasks inspired by Liu et al. [2024]. We use the recently proposed C-RASP framework for studying length generalization [Huang et al., 2025b] to provide guarantees for each of our settings. Empirically, we observe an $\textit{induction-versus-anti-induction}$ asymmetry, where pretrained models are better at retrieving tokens to the right (induction) rather than the left (anti-induction) of a query token. This asymmetry disappears upon targeted fine-tuning if length-generalization is guaranteed by theory. Mechanistic analysis reveals that this asymmetry is connected to the differences in the strength of induction versus anti-induction circuits within pretrained Transformers. We validate our findings through practical experiments on real-world tasks demonstrating reliability risks. Our results highlight that pretraining selectively enhances certain Transformer capabilities, but does not overcome fundamental length-generalization limits.
>
---
#### [new 147] From prosthetic memory to prosthetic denial: Auditing whether large language models are prone to mass atrocity denialism
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在评估大型语言模型（LLMs）是否加剧历史大屠杀否认现象。通过对比Claude、GPT等五种模型在霍洛多莫尔等四起事件中的多语言回应，发现LLMs对数据不足事件（如柬埔寨大屠杀）易输出不准确或否认倾向内容，揭示训练数据偏差与模型机制可能强化历史记忆扭曲，呼吁关注技术对记忆保存的伦理影响。**

- **链接: [http://arxiv.org/pdf/2505.21753v1](http://arxiv.org/pdf/2505.21753v1)**

> **作者:** Roberto Ulloa; Eve M. Zucker; Daniel Bultmann; David J. Simon; Mykola Makhortykh
>
> **摘要:** The proliferation of large language models (LLMs) can influence how historical narratives are disseminated and perceived. This study explores the implications of LLMs' responses on the representation of mass atrocity memory, examining whether generative AI systems contribute to prosthetic memory, i.e., mediated experiences of historical events, or to what we term "prosthetic denial," the AI-mediated erasure or distortion of atrocity memories. We argue that LLMs function as interfaces that can elicit prosthetic memories and, therefore, act as experiential sites for memory transmission, but also introduce risks of denialism, particularly when their outputs align with contested or revisionist narratives. To empirically assess these risks, we conducted a comparative audit of five LLMs (Claude, GPT, Llama, Mixtral, and Gemini) across four historical case studies: the Holodomor, the Holocaust, the Cambodian Genocide, and the genocide against the Tutsis in Rwanda. Each model was prompted with questions addressing common denialist claims in English and an alternative language relevant to each case (Ukrainian, German, Khmer, and French). Our findings reveal that while LLMs generally produce accurate responses for widely documented events like the Holocaust, significant inconsistencies and susceptibility to denialist framings are observed for more underrepresented cases like the Cambodian Genocide. The disparities highlight the influence of training data availability and the probabilistic nature of LLM responses on memory integrity. We conclude that while LLMs extend the concept of prosthetic memory, their unmoderated use risks reinforcing historical denialism, raising ethical concerns for (digital) memory preservation, and potentially challenging the advantageous role of technology associated with the original values of prosthetic memory.
>
---
#### [new 148] Scaling Reasoning without Attention
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对复杂推理任务中Transformer模型的架构低效及缺乏结构化微调问题，提出无注意力模型\ourmodel，基于SSD层消除自注意力与KV缓存，实现固定内存推理；并通过两阶段课程微调策略提升训练效果。实验显示其7B模型优于同规模及更大模型（如Gemma3-27B），验证了高效推理架构潜力。**

- **链接: [http://arxiv.org/pdf/2505.22425v1](http://arxiv.org/pdf/2505.22425v1)**

> **作者:** Xueliang Zhao; Wei Wu; Lingpeng Kong
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) have made significant advances in complex reasoning tasks, yet they remain bottlenecked by two core challenges: architectural inefficiency due to reliance on Transformers, and a lack of structured fine-tuning for high-difficulty domains. We introduce \ourmodel, an attention-free language model that addresses both issues through architectural and data-centric innovations. Built on the state space dual (SSD) layers of Mamba-2, our model eliminates the need for self-attention and key-value caching, enabling fixed-memory, constant-time inference. To train it for complex reasoning, we propose a two-phase curriculum fine-tuning strategy based on the \textsc{PromptCoT} synthesis paradigm, which generates pedagogically structured problems via abstract concept selection and rationale-guided generation. On benchmark evaluations, \ourmodel-7B outperforms strong Transformer and hybrid models of comparable scale, and even surpasses the much larger Gemma3-27B by 2.6\% on AIME 24, 0.6\% on AIME 25, and 3.0\% on Livecodebench. These results highlight the potential of state space models as efficient and scalable alternatives to attention-based architectures for high-capacity reasoning.
>
---
## 更新

#### [replaced 001] Adapting Pretrained Language Models for Citation Classification via Self-Supervised Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14471v2](http://arxiv.org/pdf/2505.14471v2)**

> **作者:** Tong Li; Jiachuan Wang; Yongqi Zhang; Shuangyin Li; Lei Chen
>
> **备注:** Accepted to KDD 2025. This is the author's version of the work
>
> **摘要:** Citation classification, which identifies the intention behind academic citations, is pivotal for scholarly analysis. Previous works suggest fine-tuning pretrained language models (PLMs) on citation classification datasets, reaping the reward of the linguistic knowledge they gained during pretraining. However, directly fine-tuning for citation classification is challenging due to labeled data scarcity, contextual noise, and spurious keyphrase correlations. In this paper, we present a novel framework, Citss, that adapts the PLMs to overcome these challenges. Citss introduces self-supervised contrastive learning to alleviate data scarcity, and is equipped with two specialized strategies to obtain the contrastive pairs: sentence-level cropping, which enhances focus on target citations within long contexts, and keyphrase perturbation, which mitigates reliance on specific keyphrases. Compared with previous works that are only designed for encoder-based PLMs, Citss is carefully developed to be compatible with both encoder-based PLMs and decoder-based LLMs, to embrace the benefits of enlarged pretraining. Experiments with three benchmark datasets with both encoder-based PLMs and decoder-based LLMs demonstrate our superiority compared to the previous state of the art. Our code is available at: github.com/LITONG99/Citss
>
---
#### [replaced 002] Moderating Harm: Benchmarking Large Language Models for Cyberbullying Detection in YouTube Comments
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18927v2](http://arxiv.org/pdf/2505.18927v2)**

> **作者:** Amel Muminovic
>
> **备注:** Preprint. 9 pages, 3 tables, 1 figure. Not yet submitted to a journal. Feedback welcome
>
> **摘要:** As online platforms grow, comment sections increasingly host harassment that undermines user experience and well-being. This study benchmarks three leading large language models, OpenAI GPT-4.1, Google Gemini 1.5 Pro, and Anthropic Claude 3 Opus, on a corpus of 5,080 YouTube comments sampled from high-abuse threads in gaming, lifestyle, food vlog, and music channels. The dataset comprises 1,334 harmful and 3,746 non-harmful messages in English, Arabic, and Indonesian, annotated independently by two reviewers with substantial agreement (Cohen's kappa = 0.83). Using a unified prompt and deterministic settings, GPT-4.1 achieved the best overall balance with an F1 score of 0.863, precision of 0.887, and recall of 0.841. Gemini flagged the highest share of harmful posts (recall = 0.875) but its precision fell to 0.767 due to frequent false positives. Claude delivered the highest precision at 0.920 and the lowest false-positive rate of 0.022, yet its recall dropped to 0.720. Qualitative analysis showed that all three models struggle with sarcasm, coded insults, and mixed-language slang. These results underscore the need for moderation pipelines that combine complementary models, incorporate conversational context, and fine-tune for under-represented languages and implicit abuse. A de-identified version of the dataset and full prompts is publicly released to promote reproducibility and further progress in automated content moderation.
>
---
#### [replaced 003] Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.03312v2](http://arxiv.org/pdf/2504.03312v2)**

> **作者:** Luís Couto Seller; Íñigo Sanz Torres; Adrián Vogel-Fernández; Carlos González Carballo; Pedro Miguel Sánchez Sánchez; Adrián Carruana Martín; Enrique de Miguel Ambite
>
> **备注:** Accepted at SEPLN 2025 conference
>
> **摘要:** Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance
>
---
#### [replaced 004] Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.19580v4](http://arxiv.org/pdf/2407.19580v4)**

> **作者:** Dang Nguyen; Wenhan Yang; Rathul Anand; Yu Yang; Baharan Mirzasoleiman
>
> **备注:** 21 pages, 6 figures, 9 tables, link: https://github.com/BigML-CS-UCLA/CoLM
>
> **摘要:** Training with larger mini-batches improves the convergence rate and can yield superior performance. However, training with large mini-batches becomes prohibitive for Large Language Models (LLMs), due to the large GPU memory requirement. To address this problem, an effective approach is finding small mini-batch coresets that closely match the gradient of larger mini-batches. However, this approach becomes infeasible and ineffective for LLMs, due to the highly imbalanced mixture of sources in language data, use of the Adam optimizer, and the very large gradient dimensionality of LLMs. In this work, we address the above challenges by proposing Coresets for Training LLMs (CoLM). First, we show that mini-batch coresets found by gradient matching do not contain representative examples of the small sources w.h.p., and thus including all examples of the small sources in the mini-batch coresets is crucial for optimal performance. Second, we normalize the gradients by their historical exponential to find mini-batch coresets for training with Adam. Finally, we leverage zeroth-order methods to find smooth gradient of the last V-projection matrix and sparsify it to keep the dimensions with the largest normalized gradient magnitude. We apply CoLM to fine-tuning Phi-2, Phi-3, Zephyr, and Llama-3 models with LoRA on MathInstruct and SuperGLUE benchmark. Remarkably, CoLM reduces the memory requirement of fine-tuning by 2x and even outperforms training with 4x larger mini-batches. Moreover, CoLM seamlessly integrates with existing memory-efficient training methods like LoRA, further reducing the memory requirements of training LLMs. Our code is available at https://github.com/BigML-CS-UCLA/CoLM.
>
---
#### [replaced 005] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy
- **分类: cs.CL; astro-ph.IM; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20538v2](http://arxiv.org/pdf/2505.20538v2)**

> **作者:** Sebastian Antony Joseph; Syed Murtaza Husain; Stella S. R. Offner; Stéphanie Juneau; Paul Torrey; Adam S. Bolton; Juan P. Farias; Niall Gaffney; Greg Durrett; Junyi Jessy Li
>
> **摘要:** Large Language Models (LLMs) are being explored for applications in scientific research, including their capabilities to synthesize literature, answer research questions, generate research ideas, and even conduct computational experiments. Ultimately, our goal is for these to help scientists derive novel scientific insights. In many areas of science, such insights often arise from processing and visualizing data to understand its patterns. However, evaluating whether an LLM-mediated scientific workflow produces outputs conveying the correct scientific insights is challenging to evaluate and has not been addressed in past work. We introduce AstroVisBench, the first benchmark for both scientific computing and visualization in the astronomy domain. AstroVisBench judges a language model's ability to both (1) create astronomy-specific workflows to process and analyze data and (2) visualize the results of these workflows through complex plots. Our evaluation of visualizations uses a novel LLM-as-a-judge workflow, which is validated against annotation by five professional astronomers. Using AstroVisBench we present an evaluation of state-of-the-art language models, showing a significant gap in their ability to engage in astronomy research as useful assistants. This evaluation provides a strong end-to-end evaluation for AI scientists that offers a path forward for the development of visualization-based workflows, which are central to a broad range of domains from physics to biology.
>
---
#### [replaced 006] Preference Adaptive and Sequential Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.10419v2](http://arxiv.org/pdf/2412.10419v2)**

> **作者:** Ofir Nabati; Guy Tennenholtz; ChihWei Hsu; Moonkyung Ryu; Deepak Ramachandran; Yinlam Chow; Xiang Li; Craig Boutilier
>
> **备注:** Accepted to ICML 2025 Link to PASTA dataset: https://www.kaggle.com/datasets/googleai/pasta-data
>
> **摘要:** We address the problem of interactive text-to-image (T2I) generation, designing a reinforcement learning (RL) agent which iteratively improves a set of generated images for a user through a sequence of prompt expansions. Using human raters, we create a novel dataset of sequential preferences, which we leverage, together with large-scale open-source (non-sequential) datasets. We construct user-preference and user-choice models using an EM strategy and identify varying user preference types. We then leverage a large multimodal language model (LMM) and a value-based RL approach to suggest an adaptive and diverse slate of prompt expansions to the user. Our Preference Adaptive and Sequential Text-to-image Agent (PASTA) extends T2I models with adaptive multi-turn capabilities, fostering collaborative co-creation and addressing uncertainty or underspecification in a user's intent. We evaluate PASTA using human raters, showing significant improvement compared to baseline methods. We also open-source our sequential rater dataset and simulated user-rater interactions to support future research in user-centric multi-turn T2I systems.
>
---
#### [replaced 007] Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10460v4](http://arxiv.org/pdf/2503.10460v4)**

> **作者:** Liang Wen; Yunke Cai; Fenrui Xiao; Xin He; Qi An; Zhenyu Duan; Yimin Du; Junchen Liu; Lifu Tang; Xiaowei Lv; Haosheng Zou; Yongchao Deng; Shousheng Jia; Xiangzheng Zhang
>
> **备注:** v4: ACL'25 industry track camera ready; v3: minor modifications; v2: better writing & format for later submission; all release at https://github.com/Qihoo360/Light-R1
>
> **摘要:** This paper introduces Light-R1, an open-source suite for training long reasoning models using reproducible and cost-effective methodology. Given the proprietary nature of data used in the DeepSeek-R1 series, we develop an alternative approach leveraging exclusively public data and models. Our curriculum training progressively increases data difficulty, combined with multi-staged post-training. Our Light-R1-32B model, trained from Qwen2.5-32B-Instruct, outperforms DeepSeek-R1-Distill-Qwen-32B in math reasoning. Experimental results show that this curriculum approach becomes more effective when distinct, diverse datasets are available for different training stages: fine-tuning DeepSeek-R1-Distilled models (pre-tuned by DeepSeek team on proprietary data) with 3,000 challenging examples from our curriculum dataset yielded state-of-the-art 7B and 14B models, while the 32B model, Light-R1-32B-DS performed comparably to QwQ-32B and DeepSeek-R1. Furthermore, we extend our work by applying GRPO on long reasoning models. Our final Light-R1-14B-DS achieves SOTA performance among 14B models in math, with AIME24 & 25 scores of 74.0 and 60.2 respectively, surpassing many 32B models and DeepSeek-R1-Distill-Llama-70B. Despite math-focused training, Light-R1-14B-DS demonstrates strong cross-domain generalization. Light-R1 represents a significant advancement in making sophisticated reasoning models more accessible and implementable in real-world applications. Our models, training data and code have been made available at https://github.com/Qihoo360/Light-R1.
>
---
#### [replaced 008] Nonlinear second-order dynamics describe labial constriction trajectories across languages and contexts
- **分类: cs.CL; nlin.AO**

- **链接: [http://arxiv.org/pdf/2410.08351v3](http://arxiv.org/pdf/2410.08351v3)**

> **作者:** Michael C. Stern; Jason A. Shaw
>
> **摘要:** We investigate the dynamics of labial constriction trajectories during the production of /b/ and /m/ in English and Mandarin. We find that, across languages and contexts, the ratio of instantaneous displacement to instantaneous velocity generally follows an exponential decay curve from movement onset to movement offset. We formalize this empirical discovery in a differential equation and, in combination with an assumption of point attractor dynamics, derive a nonlinear second-order dynamical system describing labial constriction trajectories. The equation has only two parameters, T and r. T corresponds to the target state and r corresponds to movement rapidity. Thus, each of the parameters corresponds to a phonetically relevant dimension of control. Nonlinear regression demonstrates that the model provides excellent fits to individual movement trajectories. Moreover, trajectories simulated from the model qualitatively match empirical trajectories, and capture key kinematic variables like duration, peak velocity, and time to achieve peak velocity. The model constitutes a proposal for the dynamics of individual articulatory movements, and thus offers a novel foundation from which to understand additional influences on articulatory kinematics like prosody, inter-movement coordination, and stochastic noise.
>
---
#### [replaced 009] VQ-CTAP: Cross-Modal Fine-Grained Sequence Representation Learning for Speech Processing
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2408.05758v2](http://arxiv.org/pdf/2408.05758v2)**

> **作者:** Chunyu Qiang; Wang Geng; Yi Zhao; Ruibo Fu; Tao Wang; Cheng Gong; Tianrui Wang; Qiuyu Liu; Jiangyan Yi; Zhengqi Wen; Chen Zhang; Hao Che; Longbiao Wang; Jianwu Dang; Jianhua Tao
>
> **摘要:** Deep learning has brought significant improvements to the field of cross-modal representation learning. For tasks such as text-to-speech (TTS), voice conversion (VC), and automatic speech recognition (ASR), a cross-modal fine-grained (frame-level) sequence representation is desired, emphasizing the semantic content of the text modality while de-emphasizing the paralinguistic information of the speech modality. We propose a method called "Vector Quantized Contrastive Token-Acoustic Pre-training (VQ-CTAP)", which uses the cross-modal aligned sequence transcoder to bring text and speech into a joint multimodal space, learning how to connect text and speech at the frame level. The proposed VQ-CTAP is a paradigm for cross-modal sequence representation learning, offering a promising solution for fine-grained generation and recognition tasks in speech processing. The VQ-CTAP can be directly applied to VC and ASR tasks without fine-tuning or additional structures. We propose a sequence-aware semantic connector, which connects multiple frozen pre-trained modules for the TTS task, exhibiting a plug-and-play capability. We design a stepping optimization strategy to ensure effective model convergence by gradually injecting and adjusting the influence of various loss components. Furthermore, we propose a semantic-transfer-wise paralinguistic consistency loss to enhance representational capabilities, allowing the model to better generalize to unseen data and capture the nuances of paralinguistic information. In addition, VQ-CTAP achieves high-compression speech coding at a rate of 25Hz from 24kHz input waveforms, which is a 960-fold reduction in the sampling rate. The audio demo is available at https://qiangchunyu.github.io/VQCTAP/
>
---
#### [replaced 010] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.08057v2](http://arxiv.org/pdf/2503.08057v2)**

> **作者:** Wen Luo; Feifan Song; Wei Li; Guangyue Peng; Shaohang Wei; Houfeng Wang
>
> **备注:** Accepted to the ACL 2025 Main Conference
>
> **摘要:** Large Language Models (LLMs) are increasingly required to generate text that is both factually accurate and diverse across various open-ended applications. However, current stochastic decoding methods struggle to balance such objectives. We introduce Dynamic Focus Decoding (DFD), a novel plug-and-play stochastic approach that resolves this trade-off without requiring additional data, knowledge, or models. DFD adaptively adjusts the decoding focus based on distributional differences across layers, leveraging the modular and hierarchical nature of factual knowledge within LLMs. This dynamic adjustment improves factuality in knowledge-intensive decoding steps and promotes diversity in less knowledge-reliant steps. DFD can be easily integrated with existing decoding methods, enhancing both factuality and diversity with minimal computational overhead. Extensive experiments across seven datasets demonstrate that DFD significantly improves performance, providing a scalable and efficient solution for open-ended text generation.
>
---
#### [replaced 011] TLUE: A Tibetan Language Understanding Evaluation Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12051v3](http://arxiv.org/pdf/2503.12051v3)**

> **作者:** Fan Gao; Cheng Huang; Nyima Tashi; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Hao Wang Xiao Feng; Yongbin Yu
>
> **摘要:** Large language models (LLMs) have made tremendous progress in recent years, but low-resource languages, such as Tibetan, remain significantly underrepresented in their evaluation. Despite Tibetan being spoken by over seven million people, it has largely been neglected in the development and assessment of LLMs. To address this gap, we present TLUE (A Tibetan Language Understanding Evaluation Benchmark), the first large-scale benchmark for assessing LLMs' capabilities in Tibetan. TLUE comprises two major components: (1) a comprehensive multi-task understanding benchmark spanning 5 domains and 67 subdomains, and (2) a safety benchmark covering 7 subdomains. We evaluate a diverse set of state-of-the-art LLMs. Experimental results demonstrate that most LLMs perform below the random baseline, highlighting the considerable challenges LLMs face in processing Tibetan, a low-resource language. TLUE provides an essential foundation for driving future research and progress in Tibetan language understanding and underscores the need for greater inclusivity in LLM development.
>
---
#### [replaced 012] Empirical analysis of binding precedent efficiency in Brazilian Supreme Court via case classification
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; 68T50 (Primary), 68T07 (Secondary)**

- **链接: [http://arxiv.org/pdf/2407.07004v3](http://arxiv.org/pdf/2407.07004v3)**

> **作者:** Raphaël Tinarrage; Henrique Ennes; Lucas Resck; Lucas T. Gomes; Jean R. Ponciano; Jorge Poco
>
> **备注:** Document similar to published version. Contains 62 pages and 21 figures
>
> **摘要:** Binding precedents (s\'umulas vinculantes) constitute a juridical instrument unique to the Brazilian legal system and whose objectives include the protection of the Federal Supreme Court against repetitive demands. Studies of the effectiveness of these instruments in decreasing the Court's exposure to similar cases, however, indicate that they tend to fail in such a direction, with some of the binding precedents seemingly creating new demands. We empirically assess the legal impact of five binding precedents, 11, 14, 17, 26, and 37, at the highest Court level through their effects on the legal subjects they address. This analysis is only possible through the comparison of the Court's ruling about the precedents' themes before they are created, which means that these decisions should be detected through techniques of Similar Case Retrieval, which we tackle from the angle of Case Classification. The contributions of this article are therefore twofold: on the mathematical side, we compare the use of different methods of Natural Language Processing -- TF-IDF, LSTM, Longformer, and regex -- for Case Classification, whereas on the legal side, we contrast the inefficiency of these binding precedents with a set of hypotheses that may justify their repeated usage. We observe that the TF-IDF models performed slightly better than LSTM and Longformer when compared through common metrics; however, the deep learning models were able to detect certain important legal events that TF-IDF missed. On the legal side, we argue that the reasons for binding precedents to fail in responding to repetitive demand are heterogeneous and case-dependent, making it impossible to single out a specific cause. We identify five main hypotheses, which are found in different combinations in each of the precedents studied.
>
---
#### [replaced 013] Text Generation Beyond Discrete Token Sampling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14827v2](http://arxiv.org/pdf/2505.14827v2)**

> **作者:** Yufan Zhuang; Liyuan Liu; Chandan Singh; Jingbo Shang; Jianfeng Gao
>
> **摘要:** In standard autoregressive generation, an LLM predicts the next-token distribution, samples a discrete token, and then discards the distribution, passing only the sampled token as new input. To preserve this distribution's rich information, we propose Mixture of Inputs (MoI), a training-free method for autoregressive generation. After generating a token following the standard paradigm, we construct a new input that blends the generated discrete token with the previously discarded token distribution. Specifically, we employ a Bayesian estimation method that treats the token distribution as the prior, the sampled token as the observation, and replaces the conventional one-hot vector with the continuous posterior expectation as the new model input. MoI allows the model to maintain a richer internal representation throughout the generation process, resulting in improved text quality and reasoning capabilities. On mathematical reasoning, code generation, and PhD-level QA tasks, MoI consistently improves performance across multiple models including QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, and DAPO-Qwen-32B, with no additional training and negligible computational overhead.
>
---
#### [replaced 014] Core Context Aware Transformers for Long Context Language Modeling
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.12465v2](http://arxiv.org/pdf/2412.12465v2)**

> **作者:** Yaofo Chen; Zeng You; Shuhai Zhang; Haokun Li; Yirui Li; Yaowei Wang; Mingkui Tan
>
> **备注:** Accepted for publication at ICML 2025
>
> **摘要:** Transformer-based Large Language Models (LLMs) have exhibited remarkable success in extensive tasks primarily attributed to self-attention mechanism, which requires a token to consider all preceding tokens as its context to compute attention. However, when the context length L becomes very large (e.g., 128K), the amount of potentially redundant information in the context tends to increase. The redundant context not only hampers the modeling representation performance but also incurs unnecessary computational and storage overhead. In this paper, we propose a plug-and-play Core Context Aware (CCA) Attention for efficient long-context modeling, comprising two complementary modules: 1) Globality-aware pooling module groups input tokens and dynamically compresses each group into one core token based on their significance. In this way, our method automatically focuses and strengthens core context while diminishing redundancy during the learning process, leading to effective long-term dependency modeling. 2) Locality-preserving module incorporates neighboring tokens to preserve local context for detailed representation. Notably, our CCA-Attention is able to replace the self-attention module in existing LLMs with minimal fine-tuning cost. Extensive experimental results show the superiority of our method in both long-context modeling and computational efficiency over state-of-the-art methods.
>
---
#### [replaced 015] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07365v3](http://arxiv.org/pdf/2502.07365v3)**

> **作者:** Zican Dong; Junyi Li; Jinhao Jiang; Mingyu Xu; Wayne Xin Zhao; Bingning Wang; Weipeng Chen
>
> **备注:** ACL2025 Main
>
> **摘要:** Large language models (LLMs) have gained extended context windows through scaling positional encodings and lightweight continual pre-training. However, this often leads to degraded performance on short-text tasks, while the reasons for this degradation remain insufficiently explored. In this work, we identify two primary factors contributing to this issue: distribution drift in hidden states and attention scores, and catastrophic forgetting during continual pre-training. To address these challenges, we propose Long Context Pre-training with Restoration Distillation (LongReD), a novel approach designed to mitigate short-text performance degradation through minimizing the distribution discrepancy between the extended and original models. Besides training on long texts, LongReD distills the hidden state of selected layers from the original model on short texts. Additionally, LongReD also introduces a short-to-long distillation, aligning the output distribution on short texts with that on long texts by leveraging skipped positional indices. Experiments on common text benchmarks demonstrate that LongReD effectively preserves the model's short-text performance while maintaining comparable or even better capacity to handle long texts than baselines. Our code is available at https://github.com/RUCAIBox/LongReD.
>
---
#### [replaced 016] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20429v2](http://arxiv.org/pdf/2505.20429v2)**

> **作者:** Shuhao Guan; Moule Lin; Cheng Xu; Xinyi Liu; Jinman Zhao; Jiexin Fan; Qi Xu; Derek Greene
>
> **备注:** ACL 2025 main
>
> **摘要:** This paper introduces PreP-OCR, a two-stage pipeline that combines document image restoration with semantic-aware post-OCR correction to enhance both visual clarity and textual consistency, thereby improving text extraction from degraded historical documents. First, we synthesize document-image pairs from plaintext, rendering them with diverse fonts and layouts and then applying a randomly ordered set of degradation operations. An image restoration model is trained on this synthetic data, using multi-directional patch extraction and fusion to process large images. Second, a ByT5 post-OCR model, fine-tuned on synthetic historical text pairs, addresses remaining OCR errors. Detailed experiments on 13,831 pages of real historical documents in English, French, and Spanish show that the PreP-OCR pipeline reduces character error rates by 63.9-70.3% compared to OCR on raw images. Our pipeline demonstrates the potential of integrating image restoration with linguistic error correction for digitizing historical archives.
>
---
#### [replaced 017] Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13904v2](http://arxiv.org/pdf/2504.13904v2)**

> **作者:** Donghuo Zeng; Roberto Legaspi; Yuewen Sun; Xinshuai Dong; Kazushi Ikeda; Peter Spirtes; Kun Zhang
>
> **备注:** 12 pages, 10 figures, 1 table. Accepted by ACM UMAP 2025
>
> **摘要:** We hypothesize that optimal system responses emerge from adaptive strategies grounded in causal and counterfactual knowledge. Counterfactual inference allows us to create hypothetical scenarios to examine the effects of alternative system responses. We enhance this process through causal discovery, which identifies the strategies informed by the underlying causal structure that govern system behaviors. Moreover, we consider the psychological constructs and unobservable noises that might be influencing user-system interactions as latent factors. We show that these factors can be effectively estimated. We employ causal discovery to identify strategy-level causal relationships among user and system utterances, guiding the generation of personalized counterfactual dialogues. We model the user utterance strategies as causal factors, enabling system strategies to be treated as counterfactual actions. Furthermore, we optimize policies for selecting system responses based on counterfactual data. Our results using a real-world dataset on social good demonstrate significant improvements in persuasive system outcomes, with increased cumulative rewards validating the efficacy of causal discovery in guiding personalized counterfactual inference and optimizing dialogue policies for a persuasive dialogue system.
>
---
#### [replaced 018] RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20813v2](http://arxiv.org/pdf/2505.20813v2)**

> **作者:** Junsik Kim; Jinwook Park; Kangil Kim
>
> **备注:** Accepted to ACL 2025, 17 pages, 10 figures
>
> **摘要:** In knowledge graph embedding, leveraging relation specific entity transformation has markedly enhanced performance. However, the consistency of embedding differences before and after transformation remains unaddressed, risking the loss of valuable inductive bias inherent in the embeddings. This inconsistency stems from two problems. First, transformation representations are specified for relations in a disconnected manner, allowing dissimilar transformations and corresponding entity embeddings for similar relations. Second, a generalized plug-in approach as a SFBR (Semantic Filter Based on Relations) disrupts this consistency through excessive concentration of entity embeddings under entity-based regularization, generating indistinguishable score distributions among relations. In this paper, we introduce a plug-in KGE method, Relation-Semantics Consistent Filter (RSCF). Its entity transformation has three features for enhancing semantic consistency: 1) shared affine transformation of relation embeddings across all relations, 2) rooted entity transformation that adds an entity embedding to its change represented by the transformed vector, and 3) normalization of the change to prevent scale reduction. To amplify the advantages of consistency that preserve semantics on embeddings, RSCF adds relation transformation and prediction modules for enhancing the semantics. In knowledge graph completion tasks with distance-based and tensor decomposition models, RSCF significantly outperforms state-of-the-art KGE methods, showing robustness across all relations and their frequencies.
>
---
#### [replaced 019] On the Within-class Variation Issue in Alzheimer's Disease Detection
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2409.16322v2](http://arxiv.org/pdf/2409.16322v2)**

> **作者:** Jiawen Kang; Dongrui Han; Lingwei Meng; Jingyan Zhou; Jinchao Li; Xixin Wu; Helen Meng
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** Alzheimer's Disease (AD) detection employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, we identify within-class variation as a critical challenge in AD detection: individuals with AD exhibit a spectrum of cognitive impairments. Therefore, simplistic binary AD classification may overlook two crucial aspects: within-class heterogeneity and instance-level imbalance. In this work, we found using a sample score estimator can generate sample-specific soft scores aligning with cognitive scores. We subsequently propose two simple yet effective methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe), targeting two problems respectively. Based on the ADReSS and CU-MARVEL corpora, we demonstrated and analyzed the advantages of the proposed approaches in detection performance. These findings provide insights for developing robust and reliable AD detection models.
>
---
#### [replaced 020] Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18436v2](http://arxiv.org/pdf/2410.18436v2)**

> **作者:** Seoyeon Kim; Huiseo Kim; Chanjun Park; Jinyoung Yeo; Dongha Lee
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can 'activate', or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
>
---
#### [replaced 021] Constrained Discrete Diffusion
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09790v2](http://arxiv.org/pdf/2503.09790v2)**

> **作者:** Michael Cardei; Jacob K Christopher; Thomas Hartvigsen; Brian R. Bartoldson; Bhavya Kailkhura; Ferdinando Fioretto
>
> **摘要:** Discrete diffusion models are a class of generative models that construct sequences by progressively denoising samples from a categorical noise distribution. Beyond their rapidly growing ability to generate coherent natural language, these models present a new and important opportunity to enforce sequence-level constraints, a capability that current autoregressive models cannot natively provide. This paper capitalizes on this opportunity by introducing Constrained Discrete Diffusion (CDD), a novel integration of differentiable constraint optimization within the diffusion process to ensure adherence to constraints, logic rules, or safety requirements for generated sequences. Unlike conventional text generators that often rely on post-hoc filtering or model retraining for controllable generation, CDD directly imposes constraints into the discrete diffusion sampling process, resulting in a training-free and effective approach. Experiments in toxicity-controlled text generation, property-constrained molecule design, and instruction-constrained text completion demonstrate that CDD achieves zero constraint violations in a diverse array of tasks while preserving fluency, novelty, and coherence while outperforming autoregressive and existing discrete diffusion approaches.
>
---
#### [replaced 022] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2406.09325v4](http://arxiv.org/pdf/2406.09325v4)**

> **作者:** Tomer Ashuach; Martin Tutek; Yonatan Belinkov
>
> **备注:** ACL 2025 Findings, 24 pages, 4 figures
>
> **摘要:** Language models (LMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel non-gradient-based method for unlearning sensitive information from LMs. REVS identifies and modifies a small subset of neurons relevant for constituent tokens that form sensitive information. To adequately evaluate our method on truly sensitive information, we curate three datasets: email and URL datasets naturally memorized by the models, and a synthetic social security number dataset that we tune the models to memorize. Compared to other methods, REVS demonstrates superior performance in unlearning sensitive information and robustness to extraction attacks, while retaining underlying model integrity.
>
---
#### [replaced 023] GOAT-TTS: Expressive and Realistic Speech Generation via A Dual-Branch LLM
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.12339v2](http://arxiv.org/pdf/2504.12339v2)**

> **作者:** Yaodong Song; Hongjie Chen; Jie Lian; Yuxin Zhang; Guangmin Xia; Zehan Li; Genliang Zhao; Jian Kang; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-n layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
>
---
#### [replaced 024] Layers at Similar Depths Generate Similar Activations Across LLM Architectures
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08775v2](http://arxiv.org/pdf/2504.08775v2)**

> **作者:** Christopher Wolfram; Aaron Schein
>
> **摘要:** How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures.
>
---
#### [replaced 025] Self-Taught Agentic Long Context Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15920v2](http://arxiv.org/pdf/2502.15920v2)**

> **作者:** Yufan Zhuang; Xiaodong Yu; Jialian Wu; Ximeng Sun; Ze Wang; Jiang Liu; Yusheng Su; Jingbo Shang; Zicheng Liu; Emad Barsoum
>
> **备注:** Published at ACL 2025 Main Conference
>
> **摘要:** Answering complex, long-context questions remains a major challenge for large language models (LLMs) as it requires effective question clarifications and context retrieval. We propose Agentic Long-Context Understanding (AgenticLU), a framework designed to enhance an LLM's understanding of such queries by integrating targeted self-clarification with contextual grounding within an agentic workflow. At the core of AgenticLU is Chain-of-Clarifications (CoC), where models refine their understanding through self-generated clarification questions and corresponding contextual groundings. By scaling inference as a tree search where each node represents a CoC step, we achieve 97.8% answer recall on NarrativeQA with a search depth of up to three and a branching factor of eight. To amortize the high cost of this search process to training, we leverage the preference pairs for each step obtained by the CoC workflow and perform two-stage model finetuning: (1) supervised finetuning to learn effective decomposition strategies, and (2) direct preference optimization to enhance reasoning quality. This enables AgenticLU models to generate clarifications and retrieve relevant context effectively and efficiently in a single inference pass. Extensive experiments across seven long-context tasks demonstrate that AgenticLU significantly outperforms state-of-the-art prompting methods and specialized long-context LLMs, achieving robust multi-hop reasoning while sustaining consistent performance as context length grows.
>
---
#### [replaced 026] ClonEval: An Open Voice Cloning Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20581v2](http://arxiv.org/pdf/2504.20581v2)**

> **作者:** Iwona Christop; Tomasz Kuczyński; Marek Kubis
>
> **备注:** Under review at NeurIPS
>
> **摘要:** We present a novel benchmark for voice cloning text-to-speech models. The benchmark consists of an evaluation protocol, an open-source library for assessing the performance of voice cloning models, and an accompanying leaderboard. The paper discusses design considerations and presents a detailed description of the evaluation procedure. The usage of the software library is explained, along with the organization of results on the leaderboard.
>
---
#### [replaced 027] BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11926v3](http://arxiv.org/pdf/2502.11926v3)**

> **作者:** Shamsuddeen Hassan Muhammad; Nedjma Ousidhoum; Idris Abdulmumin; Jan Philip Wahle; Terry Ruas; Meriem Beloucif; Christine de Kock; Nirmal Surange; Daniela Teodorescu; Ibrahim Said Ahmad; David Ifeoluwa Adelani; Alham Fikri Aji; Felermino D. M. A. Ali; Ilseyar Alimova; Vladimir Araujo; Nikolay Babakov; Naomi Baes; Ana-Maria Bucur; Andiswa Bukula; Guanqun Cao; Rodrigo Tufino Cardenas; Rendi Chevi; Chiamaka Ijeoma Chukwuneke; Alexandra Ciobotaru; Daryna Dementieva; Murja Sani Gadanya; Robert Geislinger; Bela Gipp; Oumaima Hourrane; Oana Ignat; Falalu Ibrahim Lawan; Rooweither Mabuya; Rahmad Mahendra; Vukosi Marivate; Alexander Panchenko; Andrew Piper; Charles Henrique Porto Ferreira; Vitaly Protasov; Samuel Rutunda; Manish Shrivastava; Aura Cristina Udrea; Lilian Diana Awuor Wanzare; Sophie Wu; Florian Valentin Wunderlich; Hanif Muhammad Zhafran; Tianhui Zhang; Yi Zhou; Saif M. Mohammad
>
> **备注:** Accepted at ACL2025 (Main)
>
> **摘要:** People worldwide use language in subtle and complex ways to express emotions. Although emotion recognition--an umbrella term for several NLP tasks--impacts various applications within NLP and beyond, most work in this area has focused on high-resource languages. This has led to significant disparities in research efforts and proposed solutions, particularly for under-resourced languages, which often lack high-quality annotated datasets. In this paper, we present BRIGHTER--a collection of multilabeled, emotion-annotated datasets in 28 different languages and across several domains. BRIGHTER primarily covers low-resource languages from Africa, Asia, Eastern Europe, and Latin America, with instances labeled by fluent speakers. We highlight the challenges related to the data collection and annotation processes, and then report experimental results for monolingual and crosslingual multi-label emotion identification, as well as emotion intensity recognition. We analyse the variability in performance across languages and text domains, both with and without the use of LLMs, and show that the BRIGHTER datasets represent a meaningful step towards addressing the gap in text-based emotion recognition.
>
---
#### [replaced 028] How Do LLMs Perform Two-Hop Reasoning in Context?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13913v2](http://arxiv.org/pdf/2502.13913v2)**

> **作者:** Tianyu Guo; Hanlin Zhu; Ruiqi Zhang; Jiantao Jiao; Song Mei; Michael I. Jordan; Stuart Russell
>
> **摘要:** ``Socrates is human. All humans are mortal. Therefore, Socrates is mortal.'' This form of argument illustrates a typical pattern of two-hop reasoning. Formally, two-hop reasoning refers to the process of inferring a conclusion by making two logical steps, each connecting adjacent concepts, such that the final conclusion depends on the integration of both steps. It is one of the most fundamental components of human reasoning and plays a crucial role in both formal logic and everyday decision-making. Despite recent progress in large language models (LLMs), we surprisingly find that they can fail at solving simple two-hop reasoning problems when distractors are present. We observe on a synthetic dataset that pre-trained LLMs often resort to random guessing among all plausible conclusions. However, after few steps of fine-tuning, models achieve near-perfect accuracy and exhibit strong length generalization. To understand the underlying mechanisms, we train a 3-layer Transformer from scratch on a synthetic two-hop reasoning task and reverse-engineer its internal information flow. We observe a clear progression in the attention logits throughout training. This pictures a sharp phase transition from an initial stage of random guessing to the emergence of a structured sequential query mechanism, where the model first retrieves the preceding and the bridge concepts in the early layers and then uses them to infer the final answer. Finally, we show that these dynamics can be captured by a minimal three-parameter attention-only network.
>
---
#### [replaced 029] Natural Language Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.14251v3](http://arxiv.org/pdf/2411.14251v3)**

> **作者:** Xidong Feng; Bo Liu; Yan Song; Haotian Fu; Ziyu Wan; Girish A. Koushik; Zhiyuan Hu; Mengyue Yang; Ying Wen; Jun Wang
>
> **备注:** 10 pages
>
> **摘要:** Artificial intelligence progresses towards the "Era of Experience," where agents are expected to learn from continuous, grounded interaction. We argue that traditional Reinforcement Learning (RL), which typically represents value as a scalar, can restrict agent's deep understanding of environments and hinders the active, deliberative learning crucial for navigating this new paradigm. To address the issue, we introduce Natural Language Reinforcement Learning (NLRL), a framework that extends RL principles into natural language counterparts. Central to NLRL is the Language Value Function (LVF), which redefines value as an interpretable linguistic narrative articulating the rationale behind an evaluation. NLRL further extends this concept to core RL components, including policy, the Bellman equation, and policy iteration. Leveraging recent advancements in Large Language Models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value training through unsupervised environment interactions. Experiments over 4 multi-step agentic tasks demonstrate NLRL's effectiveness, efficiency, and its potential to foster deeper understanding and more active learning strategies.
>
---
#### [replaced 030] A Checks-and-Balances Framework for Context-Aware Ethical AI Alignment
- **分类: cs.CL; cs.AI; F.2.2**

- **链接: [http://arxiv.org/pdf/2502.00136v3](http://arxiv.org/pdf/2502.00136v3)**

> **作者:** Edward Y. Chang
>
> **备注:** 20 pages, 7 tables, 6 figures. arXiv admin note: substantial text overlap with arXiv:2405.07076
>
> **摘要:** This paper introduces a checks-and-balances framework for ethical alignment of Large Language Models (LLMs), inspired by three-branch governmental systems. It implements three independent yet interacting components: LLMs as the executive branch for knowledge generation, DIKE as the legislative branch establishing ethical guardrails, and ERIS as the judicial branch for contextual interpretation. Beyond structural separation, we address a fundamental challenge: regulating emotion to shape behaviors. Drawing from psychological theories where managing emotional responses prevents harmful behaviors, we develop a self-supervised learning pipeline that maps emotions to linguistic behaviors, enabling precise behavioral modulation through emotional conditioning. By integrating this approach with adversarial testing, our framework demonstrates how DIKE and ERIS direct linguistic behaviors toward ethical outcomes while preserving independence throughout knowledge generation, ethical oversight, and contextual interpretation.
>
---
#### [replaced 031] LINGOLY-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02972v5](http://arxiv.org/pdf/2503.02972v5)**

> **作者:** Jude Khouja; Karolina Korgul; Simi Hellsten; Lingyi Yang; Vlad Neacsu; Harry Mayne; Ryan Kearns; Andrew Bean; Adam Mahdi
>
> **摘要:** The expanding knowledge and memorisation capacity of frontier language models allows them to solve many reasoning tasks directly by exploiting prior knowledge, leading to inflated estimates of their reasoning abilities. We introduce LINGOLY-TOO, a challenging reasoning benchmark grounded in natural language and designed to counteract the effect of non-reasoning abilities on reasoning estimates. Using linguistically informed rulesets, we permute reasoning problems written in real languages to generate numerous question variations. These permutations preserve the intrinsic reasoning steps required for each solution while reducing the likelihood problems are directly solvable with models' knowledge. Experiments and analyses show that models can circumvent reasoning and answer from prior knowledge. On a metric that rewards consistent reasoning, all models perform poorly and exhibit high variance across question permutations, indicating that Large Language Models' (LLMs) reasoning faculty remains brittle. Overall, results on the benchmark reflect the recent progress of Inference-Time Compute (ITC) models but suggest ample room for further improvement. The benchmark is a step towards better measurement of reasoning abilities of LLMs and offers a cautionary tale on the importance of disentangling reasoning abilities from models' internalised knowledge when developing reasoning benchmarks.
>
---
#### [replaced 032] Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14023v3](http://arxiv.org/pdf/2406.14023v3)**

> **作者:** Yuchen Wen; Keping Bi; Wei Chen; Jiafeng Guo; Xueqi Cheng
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data and benchmarks are available at https://github.com/yuchenwen1/ImplicitBiasPsychometricEvaluation and https://github.com/yuchenwen1/BUMBLE.
>
---
#### [replaced 033] In-context Language Learning for Endangered Languages in Speech Recognition
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20445v2](http://arxiv.org/pdf/2505.20445v2)**

> **作者:** Zhaolin Li; Jan Niehues
>
> **备注:** Interspeech2025
>
> **摘要:** With approximately 7,000 languages spoken worldwide, current large language models (LLMs) support only a small subset. Prior research indicates LLMs can learn new languages for certain tasks without supervised data. We extend this investigation to speech recognition, investigating whether LLMs can learn unseen, low-resource languages through in-context learning (ICL). With experiments on four diverse endangered languages that LLMs have not been trained on, we find that providing more relevant text samples enhances performance in both language modelling and Automatic Speech Recognition (ASR) tasks. Furthermore, we show that the probability-based approach outperforms the traditional instruction-based approach in language learning. Lastly, we show ICL enables LLMs to achieve ASR performance that is comparable to or even surpasses dedicated language models trained specifically for these languages, while preserving the original capabilities of the LLMs.
>
---
#### [replaced 034] CHIMERA: A Knowledge Base of Idea Recombination in Scientific Literature
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20779v2](http://arxiv.org/pdf/2505.20779v2)**

> **作者:** Noy Sternlicht; Tom Hope
>
> **备注:** Project page: https://noy-sternlicht.github.io/CHIMERA-Web
>
> **摘要:** A hallmark of human innovation is the process of recombination -- creating original ideas by integrating elements of existing mechanisms and concepts. In this work, we automatically mine the scientific literature and build CHIMERA: a large-scale knowledge base (KB) of recombination examples. CHIMERA can be used to empirically explore at scale how scientists recombine concepts and take inspiration from different areas, or to train supervised machine learning models that learn to predict new creative cross-domain directions. To build this KB, we present a novel information extraction task of extracting recombination from scientific paper abstracts, collect a high-quality corpus of hundreds of manually annotated abstracts, and use it to train an LLM-based extraction model. The model is applied to a large corpus of papers in the AI domain, yielding a KB of over 28K recombination examples. We analyze CHIMERA to explore the properties of recombination in different subareas of AI. Finally, we train a scientific hypothesis generation model using the KB, which predicts new recombination directions that real-world researchers find inspiring. Our data and code are available at https://github.com/noy-sternlicht/CHIMERA-KB
>
---
#### [replaced 035] AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for Early Warning System Investments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05104v2](http://arxiv.org/pdf/2504.05104v2)**

> **作者:** Saeid Ario Vaghefi; Aymane Hachcham; Veronica Grasso; Jiska Manicus; Nakiete Msemo; Chiara Colesanti Senni; Markus Leippold
>
> **摘要:** Tracking financial investments in climate adaptation is a complex and expertise-intensive task, particularly for Early Warning Systems (EWS), which lack standardized financial reporting across multilateral development banks (MDBs) and funds. To address this challenge, we introduce an LLM-based agentic AI system that integrates contextual retrieval, fine-tuning, and multi-step reasoning to extract relevant financial data, classify investments, and ensure compliance with funding guidelines. Our study focuses on a real-world application: tracking EWS investments in the Climate Risk and Early Warning Systems (CREWS) Fund. We analyze 25 MDB project documents and evaluate multiple AI-driven classification methods, including zero-shot and few-shot learning, fine-tuned transformer-based classifiers, chain-of-thought (CoT) prompting, and an agent-based retrieval-augmented generation (RAG) approach. Our results show that the agent-based RAG approach significantly outperforms other methods, achieving 87\% accuracy, 89\% precision, and 83\% recall. Additionally, we contribute a benchmark dataset and expert-annotated corpus, providing a valuable resource for future research in AI-driven financial tracking and climate finance transparency.
>
---
#### [replaced 036] AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.17284v5](http://arxiv.org/pdf/2411.17284v5)**

> **作者:** Alexander Capstick; Rahul G. Krishnan; Payam Barnaghi
>
> **摘要:** Large language models (LLMs) acquire a breadth of information across various domains. However, their computational complexity, cost, and lack of transparency often hinder their direct application for predictive tasks where privacy and interpretability are paramount. In fields such as healthcare, biology, and finance, specialised and interpretable linear models still hold considerable value. In such domains, labelled data may be scarce or expensive to obtain. Well-specified prior distributions over model parameters can reduce the sample complexity of learning through Bayesian inference; however, eliciting expert priors can be time-consuming. We therefore introduce AutoElicit to extract knowledge from LLMs and construct priors for predictive models. We show these priors are informative and can be refined using natural language. We perform a careful study contrasting AutoElicit with in-context learning and demonstrate how to perform model selection between the two methods. We find that AutoElicit yields priors that can substantially reduce error over uninformative priors, using fewer labels, and consistently outperform in-context learning. We show that AutoElicit saves over 6 months of labelling effort when building a new predictive model for urinary tract infections from sensor recordings of people living with dementia.
>
---
#### [replaced 037] Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13171v2](http://arxiv.org/pdf/2505.13171v2)**

> **作者:** Yixuan Xu; Antoni-Joan Solergibert i Llaquet; Antoine Bosselut; Imanol Schlag
>
> **摘要:** Large language models are known to memorize parts of their training data, posing risk of copyright violations. To systematically examine this risk, we pretrain language models (1B/3B/8B) from scratch on 83B tokens, mixing web-scale data with public domain books used to simulate copyrighted content at controlled frequencies at lengths at least ten times longer than prior work. We thereby identified the offset effect, a phenomenon characterized by two key findings: (1) verbatim memorization is most strongly triggered by short prefixes drawn from the beginning of the context window, with memorization decreasing counterintuitively as prefix length increases; and (2) a sharp decline in verbatim recall when prefix begins offset from the initial tokens of the context window. We attribute this to positional fragility: models rely disproportionately on the earliest tokens in their context window as retrieval anchors, making them sensitive to even slight shifts. We further observe that when the model fails to retrieve memorized content, it often produces degenerated text. Leveraging these findings, we show that shifting sensitive data deeper into the context window suppresses both extractable memorization and degeneration. Our results suggest that positional offset is a critical and previously overlooked axis for evaluating memorization risks, since prior work implicitly assumed uniformity by probing only from the beginning of training sequences.
>
---
#### [replaced 038] Towards Achieving Concept Completeness for Textual Concept Bottleneck Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11100v3](http://arxiv.org/pdf/2502.11100v3)**

> **作者:** Milan Bhan; Yann Choho; Pierre Moreau; Jean-Noel Vittaut; Nicolas Chesneau; Marie-Jeanne Lesot
>
> **摘要:** Textual Concept Bottleneck Models (TCBMs) are interpretable-by-design models for text classification that predict a set of salient concepts before making the final prediction. This paper proposes Complete Textual Concept Bottleneck Model (CT-CBM), a novel TCBM generator building concept labels in a fully unsupervised manner using a small language model, eliminating both the need for predefined human labeled concepts and LLM annotations. CT-CBM iteratively targets and adds important and identifiable concepts in the bottleneck layer to create a complete concept basis. CT-CBM achieves striking results against competitors in terms of concept basis completeness and concept detection accuracy, offering a promising solution to reliably enhance interpretability of NLP classifiers.
>
---
#### [replaced 039] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.10619v5](http://arxiv.org/pdf/2503.10619v5)**

> **作者:** Andy Zhou; Ron Arel
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** We introduce Tempest, a multi-turn adversarial framework that models the gradual erosion of Large Language Model (LLM) safety through a tree search perspective. Unlike single-turn jailbreaks that rely on one meticulously engineered prompt, Tempest expands the conversation at each turn in a breadth-first fashion, branching out multiple adversarial prompts that exploit partial compliance from previous responses. By tracking these incremental policy leaks and re-injecting them into subsequent queries, Tempest reveals how minor concessions can accumulate into fully disallowed outputs. Evaluations on the JailbreakBench dataset show that Tempest achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4 in a single multi-turn run, using fewer queries than baselines such as Crescendo or GOAT. This tree search methodology offers an in-depth view of how model safeguards degrade over successive dialogue turns, underscoring the urgency of robust multi-turn testing procedures for language models.
>
---
#### [replaced 040] Personality-aware Student Simulation for Conversational Intelligent Tutoring Systems
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.06762v2](http://arxiv.org/pdf/2404.06762v2)**

> **作者:** Zhengyuan Liu; Stella Xin Yin; Geyu Lin; Nancy F. Chen
>
> **摘要:** Intelligent Tutoring Systems (ITSs) can provide personalized and self-paced learning experience. The emergence of large language models (LLMs) further enables better human-machine interaction, and facilitates the development of conversational ITSs in various disciplines such as math and language learning. In dialogic teaching, recognizing and adapting to individual characteristics can significantly enhance student engagement and learning efficiency. However, characterizing and simulating student's persona remain challenging in training and evaluating conversational ITSs. In this work, we propose a framework to construct profiles of different student groups by refining and integrating both cognitive and noncognitive aspects, and leverage LLMs for personality-aware student simulation in a language learning scenario. We further enhance the framework with multi-aspect validation, and conduct extensive analysis from both teacher and student perspectives. Our experimental results show that state-of-the-art LLMs can produce diverse student responses according to the given language ability and personality traits, and trigger teacher's adaptive scaffolding strategies.
>
---
#### [replaced 041] LLäMmlein: Compact and Competitive German-Only Language Models from Scratch
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11171v4](http://arxiv.org/pdf/2411.11171v4)**

> **作者:** Jan Pfister; Julia Wunderle; Andreas Hotho
>
> **备注:** camera ready @ACL25; https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/llammlein/
>
> **摘要:** We create two German-only decoder models, LL\"aMmlein 120M and 1B, transparently from scratch and publish them, along with the training data, for the German NLP research community to use. The model training involved several key steps, including extensive data preprocessing, the creation of a custom German tokenizer, the training itself, as well as the evaluation of the final models on various benchmarks. Throughout the training process, multiple checkpoints were saved and analyzed using the SuperGLEBer benchmark to monitor the models' learning dynamics. Compared to state-of-the-art models on the SuperGLEBer benchmark, both LL\"aMmlein models performed competitively, consistently matching or surpassing models with similar parameter sizes. The results show that the models' quality scales with size as expected, but performance improvements on some tasks plateaued early, offering valuable insights into resource allocation for future model development.
>
---
#### [replaced 042] Overcoming Non-monotonicity in Transducer-based Streaming Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17170v2](http://arxiv.org/pdf/2411.17170v2)**

> **作者:** Zhengrui Ma; Yang Feng; Min Zhang
>
> **备注:** ICML25; Codes: https://github.com/ictnlp/MonoAttn-Transducer
>
> **摘要:** Streaming generation models are utilized across fields, with the Transducer architecture being popular in industrial applications. However, its input-synchronous decoding mechanism presents challenges in tasks requiring non-monotonic alignments, such as simultaneous translation. In this research, we address this issue by integrating Transducer's decoding with the history of input stream via a learnable monotonic attention. Our approach leverages the forward-backward algorithm to infer the posterior probability of alignments between the predictor states and input timestamps, which is then used to estimate the monotonic context representations, thereby avoiding the need to enumerate the exponentially large alignment space during training. Extensive experiments show that our MonoAttn-Transducer effectively handles non-monotonic alignments in streaming scenarios, offering a robust solution for complex generation tasks.
>
---
#### [replaced 043] You Do Not Fully Utilize Transformer's Representation Capacity
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09245v2](http://arxiv.org/pdf/2502.09245v2)**

> **作者:** Gleb Gerasimov; Yaroslav Aksenov; Nikita Balagansky; Viacheslav Sinii; Daniil Gavrilov
>
> **摘要:** In contrast to RNNs, which compress their history into a single hidden state, Transformers can attend to all past tokens directly. However, standard Transformers rely solely on the hidden state from the previous layer to represent the entire context. We show that this design choice induces representation collapse and degrades performance. To address this issue, we introduce Layer-Integrated Memory (LIMe), a lightweight extension that leverages existing key-value buffers and learns per-head, per-layer routing weights to integrate representations from all previous layers with negligible overhead. Through extensive experiments-including language modeling, synthetic reasoning benchmarks, and very deep architectures-LIMe consistently achieves faster convergence, lower perplexity per FLOP, and substantial accuracy improvements on synthetic tasks while preserving higher value-vector entropy and improved token separability. Finally, our analysis of the learned routing weights reveals systematic reuse of both local and long-distance features, demonstrating how LIMe mitigates collapse, unlocks richer representations without increasing hidden-state size, and points to promising directions for future research.
>
---
#### [replaced 044] ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13458v2](http://arxiv.org/pdf/2502.13458v2)**

> **作者:** Xiaofei Wen; Wenxuan Zhou; Wenjie Jacky Mo; Muhao Chen
>
> **备注:** ACL 2025
>
> **摘要:** Ensuring the safety of large language models (LLMs) is critical as they are deployed in real-world applications. Existing guardrails rely on rule-based filtering or single-pass classification, limiting their ability to handle nuanced safety violations. To address this, we propose ThinkGuard, a critique-augmented guardrail model that distills knowledge from high-capacity LLMs by generating structured critiques alongside safety labels. Fine-tuned on critique-augmented data, the captured deliberative thinking ability drastically enhances the guardrail's cautiousness and interpretability. Evaluated on multiple safety benchmarks, ThinkGuard achieves the highest average F1 and AUPRC, outperforming all baselines. Compared to LLaMA Guard 3, ThinkGuard improves accuracy by 16.1% and macro F1 by 27.0%. Moreover, it surpasses label-only fine-tuned models, confirming that structured critiques enhance both classification precision and nuanced safety reasoning while maintaining computational efficiency.
>
---
#### [replaced 045] Wanda++: Pruning Large Language Models via Regional Gradients
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04992v3](http://arxiv.org/pdf/2503.04992v3)**

> **作者:** Yifan Yang; Kai Zhen; Bhavana Ganesh; Aram Galstyan; Goeric Huybrechts; Markus Müller; Jonas M. Kübler; Rupak Vignesh Swaminathan; Athanasios Mouchtaris; Sravan Babu Bodapati; Nathan Susanj; Zheng Zhang; Jack FitzGerald; Abhishek Kumar
>
> **备注:** Paper accepted at ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) pruning seeks to remove unimportant weights for inference speedup with minimal accuracy impact. However, existing methods often suffer from accuracy degradation without full-model sparsity-aware fine-tuning. This paper presents Wanda++, a novel pruning framework that outperforms the state-of-the-art methods by utilizing decoder-block-level \textbf{regional} gradients. Specifically, Wanda++ improves the pruning score with regional gradients for the first time and proposes an efficient regional optimization method to minimize pruning-induced output discrepancies between the dense and sparse decoder output. Notably, Wanda++ improves perplexity by up to 32\% over Wanda in the language modeling task and generalizes effectively to downstream tasks. Moreover, despite updating weights with regional optimization, Wanda++ remains orthogonal to sparsity-aware fine-tuning, further reducing perplexity with LoRA in great extend. Our approach is lightweight, pruning a 7B LLaMA model in under 10 minutes on a single H100 GPU.
>
---
#### [replaced 046] Inference-time Alignment in Continuous Space
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20081v2](http://arxiv.org/pdf/2505.20081v2)**

> **作者:** Yige Yuan; Teng Xiao; Li Yunfan; Bingbing Xu; Shuchang Tao; Yunqi Qiu; Huawei Shen; Xueqi Cheng
>
> **摘要:** Aligning large language models with human feedback at inference time has received increasing attention due to its flexibility. Existing methods rely on generating multiple responses from the base policy for search using a reward model, which can be considered as searching in a discrete response space. However, these methods struggle to explore informative candidates when the base policy is weak or the candidate set is small, resulting in limited effectiveness. In this paper, to address this problem, we propose Simple Energy Adaptation ($\textbf{SEA}$), a simple yet effective algorithm for inference-time alignment. In contrast to expensive search over the discrete space, SEA directly adapts original responses from the base policy toward the optimal one via gradient-based sampling in continuous latent space. Specifically, SEA formulates inference as an iterative optimization procedure on an energy function over actions in the continuous space defined by the optimal policy, enabling simple and effective alignment. For instance, despite its simplicity, SEA outperforms the second-best baseline with a relative improvement of up to $ \textbf{77.51%}$ on AdvBench and $\textbf{16.36%}$ on MATH. Our code is publicly available at https://github.com/yuanyige/sea
>
---
#### [replaced 047] Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks
- **分类: cs.IR; cs.CL; H.3.3; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.16849v2](http://arxiv.org/pdf/2505.16849v2)**

> **作者:** Martin Böckling; Heiko Paulheim; Andreea Iana
>
> **备注:** Accepted at the Information Retrieval's Role in RAG Systems (IR-RAG 2025) in conjunction with SIGIR 2025
>
> **摘要:** Large Language Models (LLMs) have showcased impressive reasoning abilities, but often suffer from hallucinations or outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by grounding LLM responses in structured external information from a knowledge base. However, many KG-based RAG approaches struggle with (i) aligning KG and textual representations, (ii) balancing retrieval accuracy and efficiency, and (iii) adapting to dynamically updated KGs. In this work, we introduce Walk&Retrieve, a simple yet effective KG-based framework that leverages walk-based graph traversal and knowledge verbalization for corpus generation for zero-shot RAG. Built around efficient KG walks, our method does not require fine-tuning on domain-specific data, enabling seamless adaptation to KG updates, reducing computational overhead, and allowing integration with any off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs competitively, often outperforming existing RAG systems in response accuracy and hallucination reduction. Moreover, it demonstrates lower query latency and robust scalability to large KGs, highlighting the potential of lightweight retrieval strategies as strong baselines for future RAG research.
>
---
#### [replaced 048] Shaping Shared Languages: Human and Large Language Models' Inductive Biases in Emergent Communication
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04395v2](http://arxiv.org/pdf/2503.04395v2)**

> **作者:** Tom Kouwenhoven; Max Peeperkorn; Roy de Kleijn; Tessa Verhoef
>
> **备注:** Presented at IJCAI 2025 (Human-centred AI Track)
>
> **摘要:** Languages are shaped by the inductive biases of their users. Using a classical referential game, we investigate how artificial languages evolve when optimised for inductive biases in humans and large language models (LLMs) via Human-Human, LLM-LLM and Human-LLM experiments. We show that referentially grounded vocabularies emerge that enable reliable communication in all conditions, even when humans \textit{and} LLMs collaborate. Comparisons between conditions reveal that languages optimised for LLMs subtly differ from those optimised for humans. Interestingly, interactions between humans and LLMs alleviate these differences and result in vocabularies more human-like than LLM-like. These findings advance our understanding of the role inductive biases in LLMs play in the dynamic nature of human language and contribute to maintaining alignment in human and machine communication. In particular, our work underscores the need to think of new LLM training methods that include human interaction and shows that using communicative success as a reward signal can be a fruitful, novel direction.
>
---
#### [replaced 049] Advancing Reasoning in Large Language Models: Promising Methods and Approaches
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.03671v2](http://arxiv.org/pdf/2502.03671v2)**

> **作者:** Avinash Patil; Aryan Jadon
>
> **备注:** 9 Pages, 1 Figure, IEEE Format
>
> **摘要:** Large Language Models (LLMs) have succeeded remarkably in various natural language processing (NLP) tasks, yet their reasoning capabilities remain a fundamental challenge. While LLMs exhibit impressive fluency and factual recall, their ability to perform complex reasoning-spanning logical deduction, mathematical problem-solving, commonsense inference, and multi-step reasoning-often falls short of human expectations. This survey provides a comprehensive review of emerging techniques enhancing reasoning in LLMs. We categorize existing methods into key approaches, including prompting strategies (e.g., Chain-of-Thought reasoning, Self-Consistency, and Tree-of-Thought reasoning), architectural innovations (e.g., retrieval-augmented models, modular reasoning networks, and neuro-symbolic integration), and learning paradigms (e.g., fine-tuning with reasoning-specific datasets, reinforcement learning, and self-supervised reasoning objectives). Additionally, we explore evaluation frameworks used to assess reasoning in LLMs and highlight open challenges, such as hallucinations, robustness, and reasoning generalization across diverse tasks. By synthesizing recent advancements, this survey aims to provide insights into promising directions for future research and practical applications of reasoning-augmented LLMs.
>
---
#### [replaced 050] ConKE: Conceptualization-Augmented Knowledge Editing in Large Language Models for Commonsense Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11418v2](http://arxiv.org/pdf/2412.11418v2)**

> **作者:** Liyu Zhang; Weiqi Wang; Tianqing Fang; Yangqiu Song
>
> **备注:** Findings of ACL2025
>
> **摘要:** Knowledge Editing (KE) aims to adjust a Large Language Model's (LLM) internal representations and parameters to correct inaccuracies and improve output consistency without incurring the computational expense of re-training the entire model. However, editing commonsense knowledge still faces difficulties, including limited knowledge coverage in existing resources, the infeasibility of annotating labels for an overabundance of commonsense knowledge, and the strict knowledge formats of current editing methods. In this paper, we address these challenges by presenting ConceptEdit, a framework that integrates conceptualization and instantiation into the KE pipeline for LLMs to enhance their commonsense reasoning capabilities. ConceptEdit dynamically diagnoses implausible commonsense knowledge within an LLM using another verifier LLM and augments the source knowledge to be edited with conceptualization for stronger generalizability. Experimental results demonstrate that LLMs enhanced with ConceptEdit successfully generate commonsense knowledge with improved plausibility compared to other baselines and achieve stronger performance across multiple question answering benchmarks. Our data, code, and models are publicly available at https://github.com/HKUST-KnowComp/ConKE.
>
---
#### [replaced 051] Paths Not Taken: Understanding and Mending the Multilingual Factual Recall Pipeline
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20546v2](http://arxiv.org/pdf/2505.20546v2)**

> **作者:** Meng Lu; Ruochen Zhang; Carsten Eickhoff; Ellie Pavlick
>
> **摘要:** Multilingual large language models (LLMs) often exhibit factual inconsistencies across languages, with significantly better performance in factual recall tasks in English than in other languages. The causes of these failures, however, remain poorly understood. Using mechanistic analysis techniques, we uncover the underlying pipeline that LLMs employ, which involves using the English-centric factual recall mechanism to process multilingual queries and then translating English answers back into the target language. We identify two primary sources of error: insufficient engagement of the reliable English-centric mechanism for factual recall, and incorrect translation from English back into the target language for the final answer. To address these vulnerabilities, we introduce two vector interventions, both independent of languages and datasets, to redirect the model toward better internal paths for higher factual consistency. Our interventions combined increase the recall accuracy by over 35 percent for the lowest-performing language. Our findings demonstrate how mechanistic insights can be used to unlock latent multilingual capabilities in LLMs.
>
---
#### [replaced 052] Revisiting In-Context Learning with Long Context Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.16926v3](http://arxiv.org/pdf/2412.16926v3)**

> **作者:** Jinheon Baek; Sun Jae Lee; Prakhar Gupta; Geunseob Oh; Siddharth Dalmia; Prateek Kolhar
>
> **备注:** ACL Findings 2025
>
> **摘要:** In-Context Learning (ICL) is a technique by which language models make predictions based on examples provided in their input context. Previously, their context window size imposed a limit on the number of examples that can be shown, making example selection techniques crucial for identifying the maximally effective set of examples. However, the recent advent of Long Context Language Models (LCLMs) has significantly increased the number of examples that can be included in context, raising an important question of whether ICL performance in a many-shot regime is still sensitive to the method of sample selection. To answer this, we revisit these approaches in the context of LCLMs through extensive experiments on 18 datasets spanning 4 tasks. Surprisingly, we observe that sophisticated example selection techniques do not yield significant improvements over a simple random sample selection method. Instead, we discover that the advent of LCLMs has fundamentally shifted the challenge of ICL from that of selecting the most effective examples to that of collecting sufficient examples to fill the context window. Specifically, in certain datasets, including all available examples does not fully utilize the context window; however, by augmenting the examples in context with a simple data augmentation approach, we substantially improve ICL performance by 5%.
>
---
#### [replaced 053] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00777v3](http://arxiv.org/pdf/2501.00777v3)**

> **作者:** Qianli Wang; Nils Feldhus; Simon Ostermann; Luis Felipe Villa-Arenas; Sebastian Möller; Vera Schmitt
>
> **备注:** ACL 2025 Findings; camera-ready version
>
> **摘要:** Counterfactual examples are widely used in natural language processing (NLP) as valuable data to improve models, and in explainable artificial intelligence (XAI) to understand model behavior. The automated generation of counterfactual examples remains a challenging task even for large language models (LLMs), despite their impressive performance on many tasks. In this paper, we first introduce ZeroCF, a faithful approach for leveraging important words derived from feature attribution methods to generate counterfactual examples in a zero-shot setting. Second, we present a new framework, FitCF, which further verifies aforementioned counterfactuals by label flip verification and then inserts them as demonstrations for few-shot prompting, outperforming two state-of-the-art baselines. Through ablation studies, we identify the importance of each of FitCF's core components in improving the quality of counterfactuals, as assessed through flip rate, perplexity, and similarity measures. Furthermore, we show the effectiveness of LIME and Integrated Gradients as backbone attribution methods for FitCF and find that the number of demonstrations has the largest effect on performance. Finally, we reveal a strong correlation between the faithfulness of feature attribution scores and the quality of generated counterfactuals, which we hope will serve as an important finding for future research in this direction.
>
---
#### [replaced 054] CoSER: Coordinating LLM-Based Persona Simulation of Established Roles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.09082v2](http://arxiv.org/pdf/2502.09082v2)**

> **作者:** Xintao Wang; Heng Wang; Yifei Zhang; Xinfeng Yuan; Rui Xu; Jen-tse Huang; Siyu Yuan; Haoran Guo; Jiangjie Chen; Shuchang Zhou; Wei Wang; Yanghua Xiao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
>
---
#### [replaced 055] FCKT: Fine-Grained Cross-Task Knowledge Transfer with Semantic Contrastive Learning for Targeted Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21040v2](http://arxiv.org/pdf/2505.21040v2)**

> **作者:** Wei Chen; Zhao Zhang; Meng Yuan; Kepeng Xu; Fuzhen Zhuang
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** In this paper, we address the task of targeted sentiment analysis (TSA), which involves two sub-tasks, i.e., identifying specific aspects from reviews and determining their corresponding sentiments. Aspect extraction forms the foundation for sentiment prediction, highlighting the critical dependency between these two tasks for effective cross-task knowledge transfer. While most existing studies adopt a multi-task learning paradigm to align task-specific features in the latent space, they predominantly rely on coarse-grained knowledge transfer. Such approaches lack fine-grained control over aspect-sentiment relationships, often assuming uniform sentiment polarity within related aspects. This oversimplification neglects contextual cues that differentiate sentiments, leading to negative transfer. To overcome these limitations, we propose FCKT, a fine-grained cross-task knowledge transfer framework tailored for TSA. By explicitly incorporating aspect-level information into sentiment prediction, FCKT achieves fine-grained knowledge transfer, effectively mitigating negative transfer and enhancing task performance. Experiments on three datasets, including comparisons with various baselines and large language models (LLMs), demonstrate the effectiveness of FCKT. The source code is available on https://github.com/cwei01/FCKT.
>
---
#### [replaced 056] K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13567v3](http://arxiv.org/pdf/2501.13567v3)**

> **作者:** Jeonghun Cho; Gary Geunbae Lee
>
> **备注:** Accepted at NAACL 2025 (Main, long paper)
>
> **摘要:** Retrieval-augmented question answering (QA) integrates external information and thereby increases the QA accuracy of reader models that lack domain knowledge. However, documents retrieved for closed domains require high expertise, so the reader model may have difficulty fully comprehending the text. Moreover, the retrieved documents contain thousands of tokens, some unrelated to the question. As a result, the documents include some inaccurate information, which could lead the reader model to mistrust the passages and could result in hallucinations. To solve these problems, we propose K-comp (Knowledge-injected compressor) which provides the knowledge required to answer correctly. The compressor automatically generates the prior knowledge necessary to facilitate the answer process prior to compression of the retrieved passages. Subsequently, the passages are compressed autoregressively, with the generated knowledge being integrated into the compression process. This process ensures alignment between the question intent and the compressed context. By augmenting this prior knowledge and concise context, the reader models are guided toward relevant answers and trust the context.
>
---
#### [replaced 057] From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Reasoning-Driven Pedagogical Visualization
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16832v2](http://arxiv.org/pdf/2505.16832v2)**

> **作者:** Haonian Ji; Shi Qiu; Siyang Xin; Siwei Han; Zhaorun Chen; Dake Zhang; Hongyi Wang; Huaxiu Yao
>
> **备注:** 16 pages; 7 figures
>
> **摘要:** While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at https://github.com/aiming-lab/EduVisBench and https://github.com/aiming-lab/EduVisAgent.
>
---
#### [replaced 058] Benchmarking LLMs' Swarm intelligence
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04364v3](http://arxiv.org/pdf/2505.04364v3)**

> **作者:** Kai Ruan; Mowen Huang; Ji-Rong Wen; Hao Sun
>
> **备注:** added new ref
>
> **摘要:** Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
>
---
#### [replaced 059] Comparing Moral Values in Western English-speaking societies and LLMs with Word Associations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19674v2](http://arxiv.org/pdf/2505.19674v2)**

> **作者:** Chaoyi Xiang; Chunhua Liu; Simon De Deyne; Lea Frermann
>
> **备注:** 9 pages,7 figures. Accepted to the ACL 2025 conference
>
> **摘要:** As the impact of large language models increases, understanding the moral values they reflect becomes ever more important. Assessing the nature of moral values as understood by these models via direct prompting is challenging due to potential leakage of human norms into model training data, and their sensitivity to prompt formulation. Instead, we propose to use word associations, which have been shown to reflect moral reasoning in humans, as low-level underlying representations to obtain a more robust picture of LLMs' moral reasoning. We study moral differences in associations from western English-speaking communities and LLMs trained predominantly on English data. First, we create a large dataset of LLM-generated word associations, resembling an existing data set of human word associations. Next, we propose a novel method to propagate moral values based on seed words derived from Moral Foundation Theory through the human and LLM-generated association graphs. Finally, we compare the resulting moral conceptualizations, highlighting detailed but systematic differences between moral values emerging from English speakers and LLM associations.
>
---
#### [replaced 060] MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12170v2](http://arxiv.org/pdf/2502.12170v2)**

> **作者:** Da Xiao; Qingye Meng; Shengping Li; Xingyuan Yuan
>
> **备注:** Accepted to the 42nd International Conference on Machine Learning (ICML'25)
>
> **摘要:** We propose MUltiway Dynamic Dense (MUDD) connections, a simple yet effective method to address the limitations of residual connections and enhance cross-layer information flow in Transformers. Unlike existing dense connection approaches with static and shared connection weights, MUDD generates connection weights dynamically depending on hidden states at each sequence position and for each decoupled input stream (the query, key, value or residual) of a Transformer block. MUDD connections can be seamlessly integrated into any Transformer architecture to create MUDDFormer. Extensive experiments show that MUDDFormer significantly outperforms Transformers across various model architectures and scales in language modeling, achieving the performance of Transformers trained with 1.8X-2.4X compute. Notably, MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even rivals Pythia-12B in five-shot settings, while adding only 0.23% parameters and 0.4% computation. Code in JAX and PyTorch and pre-trained models are available at https://github.com/Caiyun-AI/MUDDFormer .
>
---
#### [replaced 061] Advancing Sequential Numerical Prediction in Autoregressive Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.13077v2](http://arxiv.org/pdf/2505.13077v2)**

> **作者:** Xiang Fei; Jinghui Lu; Qi Sun; Hao Feng; Yanjie Wang; Wei Shi; An-Lan Wang; Jingqun Tang; Can Huang
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL.
>
---
#### [replaced 062] Controllable Context Sensitivity and the Knob Behind It
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07404v3](http://arxiv.org/pdf/2411.07404v3)**

> **作者:** Julian Minder; Kevin Du; Niklas Stoehr; Giovanni Monea; Chris Wendler; Robert West; Ryan Cotterell
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** When making predictions, a language model must trade off how much it relies on its context vs. its prior knowledge. Choosing how sensitive the model is to its context is a fundamental functionality, as it enables the model to excel at tasks like retrieval-augmented generation and question-answering. In this paper, we search for a knob which controls this sensitivity, determining whether language models answer from the context or their prior knowledge. To guide this search, we design a task for controllable context sensitivity. In this task, we first feed the model a context (Paris is in England) and a question (Where is Paris?); we then instruct the model to either use its prior or contextual knowledge and evaluate whether it generates the correct answer for both intents (either France or England). When fine-tuned on this task, instruction-tuned versions of Llama-3.1, Mistral-v0.3, and Gemma-2 can solve it with high accuracy (85-95%). Analyzing these high-performing models, we narrow down which layers may be important to context sensitivity using a novel linear time algorithm. Then, in each model, we identify a 1-D subspace in a single layer that encodes whether the model follows context or prior knowledge. Interestingly, while we identify this subspace in a fine-tuned model, we find that the exact same subspace serves as an effective knob in not only that model but also non-fine-tuned instruct and base models of that model family. Finally, we show a strong correlation between a model's performance and how distinctly it separates context-agreeing from context-ignoring answers in this subspace. These results suggest a single subspace facilitates how the model chooses between context and prior knowledge, hinting at a simple fundamental mechanism that controls this behavior.
>
---
#### [replaced 063] Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2311.04378v5](http://arxiv.org/pdf/2311.04378v5)**

> **作者:** Hanlin Zhang; Benjamin L. Edelman; Danilo Francati; Daniele Venturi; Giuseppe Ateniese; Boaz Barak
>
> **备注:** ICML 2024. Website: https://hanlin-zhang.com/impossibility-watermarks
>
> **摘要:** Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.
>
---
#### [replaced 064] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11441v3](http://arxiv.org/pdf/2502.11441v3)**

> **作者:** Hwan Chang; Hwanhee Lee
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
>
---
#### [replaced 065] Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11277v2](http://arxiv.org/pdf/2505.11277v2)**

> **作者:** Yaorui Shi; Sihang Li; Chang Wu; Zhiyuan Liu; Junfeng Fang; Hengxing Cai; An Zhang; Xiang Wang
>
> **摘要:** Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
>
---
#### [replaced 066] Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00602v2](http://arxiv.org/pdf/2502.00602v2)**

> **作者:** Tianci Liu; Ruirui Li; Zihan Dong; Hui Liu; Xianfeng Tang; Qingyu Yin; Linjun Zhang; Haoyu Wang; Jing Gao
>
> **备注:** ICML 2025
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing (KE) to update specific knowledge in LLMs without changing unrelated others or compromising their pre-trained capabilities. Previous efforts sought to update a small amount of parameters of a LLM and proved effective for making selective updates. Nonetheless, the edited LLM often exhibits degraded ability to reason about the new knowledge. In this work, we identify a key issue: heterogeneous token overfitting (HTO), where the LLM overfits different tokens in the provided knowledge at varying rates. To tackle this, we propose OVERTONE, a token-level smoothing method that mitigates HTO by adaptively refining the target distribution. Theoretically, OVERTONE offers better parameter updates with negligible computation overhead. It also induces an implicit DPO but does not require preference data pairs. Extensive experiments across four editing methods, two LLMs, and diverse scenarios demonstrate the effectiveness and versatility of our method.
>
---
#### [replaced 067] Closed-Form Training Dynamics Reveal Learned Features and Linear Structure in Word2Vec-like Models
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.09863v2](http://arxiv.org/pdf/2502.09863v2)**

> **作者:** Dhruva Karkada; James B. Simon; Yasaman Bahri; Michael R. DeWeese
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Self-supervised word embedding algorithms such as word2vec provide a minimal setting for studying representation learning in language modeling. We examine the quartic Taylor approximation of the word2vec loss around the origin, and we show that both the resulting training dynamics and the final performance on downstream tasks are empirically very similar to those of word2vec. Our main contribution is to analytically solve for both the gradient flow training dynamics and the final word embeddings in terms of only the corpus statistics and training hyperparameters. The solutions reveal that these models learn orthogonal linear subspaces one at a time, each one incrementing the effective rank of the embeddings until model capacity is saturated. Training on Wikipedia, we find that each of the top linear subspaces represents an interpretable topic-level concept. Finally, we apply our theory to describe how linear representations of more abstract semantic concepts emerge during training; these can be used to complete analogies via vector addition.
>
---
#### [replaced 068] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v3](http://arxiv.org/pdf/2505.12312v3)**

> **作者:** Qi Feng
>
> **备注:** Author list corrected. In version 1, Hidetoshi Shimodaira was included as a co-author without their consent and has been removed from the author list
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 069] Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.06365v2](http://arxiv.org/pdf/2501.06365v2)**

> **作者:** Elizabeth Schaefer; Kirk Roberts
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** This paper presents a pipeline for mitigating gender bias in large language models (LLMs) used in medical literature by neutralizing gendered occupational pronouns. A dataset of 379,000 PubMed abstracts from 1965-1980 was processed to identify and modify pronouns tied to professions. We developed a BERT-based model, "Modern Occupational Bias Elimination with Refined Training," or "MOBERT," trained on these neutralized abstracts, and compared its performance with "1965BERT," trained on the original dataset. MOBERT achieved a 70% inclusive replacement rate, while 1965BERT reached only 4%. A further analysis of MOBERT revealed that pronoun replacement accuracy correlated with the frequency of occupational terms in the training data. We propose expanding the dataset and refining the pipeline to improve performance and ensure more equitable language modeling in medical applications.
>
---
#### [replaced 070] Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00134v2](http://arxiv.org/pdf/2503.00134v2)**

> **作者:** Zhongqi Yang; Amir Rahmani
>
> **摘要:** Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process.
>
---
#### [replaced 071] SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.03561v2](http://arxiv.org/pdf/2504.03561v2)**

> **作者:** Runnan Fang; Xiaobin Wang; Yuan Liang; Shuofei Qiao; Jialong Wu; Zekun Xi; Ningyu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at https://github.com/zjunlp/SynWorld.
>
---
#### [replaced 072] SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19641v3](http://arxiv.org/pdf/2505.19641v3)**

> **作者:** Junteng Liu; Yuanxiang Fan; Zhuo Jiang; Han Ding; Yongyi Hu; Chi Zhang; Yiqi Shi; Shitong Weng; Aili Chen; Shiqi Chen; Yunan Huang; Mozhi Zhang; Pengyu Zhao; Junjie Yan; Junxian He
>
> **摘要:** Recent advances such as OpenAI-o1 and DeepSeek R1 have demonstrated the potential of Reinforcement Learning (RL) to enhance reasoning abilities in Large Language Models (LLMs). While open-source replication efforts have primarily focused on mathematical and coding domains, methods and resources for developing general reasoning capabilities remain underexplored. This gap is partly due to the challenge of collecting diverse and verifiable reasoning data suitable for RL. We hypothesize that logical reasoning is critical for developing general reasoning capabilities, as logic forms a fundamental building block of reasoning. In this work, we present SynLogic, a data synthesis framework and dataset that generates diverse logical reasoning data at scale, encompassing 35 diverse logical reasoning tasks. The SynLogic approach enables controlled synthesis of data with adjustable difficulty and quantity. Importantly, all examples can be verified by simple rules, making them ideally suited for RL with verifiable rewards. In our experiments, we validate the effectiveness of RL training on the SynLogic dataset based on 7B and 32B models. SynLogic leads to state-of-the-art logical reasoning performance among open-source datasets, surpassing DeepSeek-R1-Distill-Qwen-32B by 6 points on BBEH. Furthermore, mixing SynLogic data with mathematical and coding tasks improves the training efficiency of these domains and significantly enhances reasoning generalization. Notably, our mixed training model outperforms DeepSeek-R1-Zero-Qwen-32B across multiple benchmarks. These findings position SynLogic as a valuable resource for advancing the broader reasoning capabilities of LLMs. We open-source both the data synthesis pipeline and the SynLogic dataset at https://github.com/MiniMax-AI/SynLogic.
>
---
#### [replaced 073] WiseMind: Recontextualizing AI with a Knowledge-Guided, Theory-Informed Multi-Agent Framework for Instrumental and Humanistic Benefits
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20689v2](http://arxiv.org/pdf/2502.20689v2)**

> **作者:** Yuqi Wu; Guangya Wan; Jingjing Li; Shengming Zhao; Lingfeng Ma; Tianyi Ye; Ion Pop; Yanbo Zhang; Jie Chen
>
> **备注:** 27 pages, 13 figures
>
> **摘要:** Translating state-of-the-art NLP into practice often stalls at the "last mile" owing to insufficient contextualization of the target domain's knowledge, processes, and evaluation. Psychiatric differential diagnosis exemplifies this challenge: accurate assessments depend on nuanced clinical knowledge, a delicate cognitive-affective interview process, and downstream outcomes that extend far beyond benchmark accuracy. We present WiseMind, a systematic interdisciplinary contextualization framework that delivers both instrumental (diagnostic precision) and humanistic (empathy) gains. WiseMind comprises three components:(i) structured knowledge-guided proactive reasoning, which embeds DSM-5 criteria in a knowledge graph to steer questioning; (ii) a theory-informed dual-agent architecture that coordinates a "reasonable-mind" reasoning agent and an "emotional-mind" empathy agent, inspired by Dialectical Behavior Therapy; and (iii) a multi-faceted evaluation strategy covering simulated patients, user studies, clinician review, and ethical assessment. Tested on depression, anxiety, and bipolar disorder, WiseMind attains up to 84.2% diagnostic accuracy, which is comparable to human experts, while outperforming single-agent baselines in perceived empathy and trustworthiness. These results show that deep contextualization-across knowledge, process, and evaluation layers-can transform benchmark-driven NLP into clinically meaningful impact.
>
---
#### [replaced 074] LLMs Reproduce Stereotypes of Sexual and Gender Minorities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.05926v2](http://arxiv.org/pdf/2501.05926v2)**

> **作者:** Ruby Ostrow; Adam Lopez
>
> **备注:** 8 pages, 5 figures, 5 tables
>
> **摘要:** A large body of research has found substantial gender bias in NLP systems. Most of this research takes a binary, essentialist view of gender: limiting its variation to the categories _men_ and _women_, conflating gender with sex, and ignoring different sexual identities. But gender and sexuality exist on a spectrum, so in this paper we study the biases of large language models (LLMs) towards sexual and gender minorities beyond binary categories. Grounding our study in a widely used social psychology model -- the Stereotype Content Model -- we demonstrate that English-language survey questions about social perceptions elicit more negative stereotypes of sexual and gender minorities from both humans and LLMs. We then extend this framework to a more realistic use case: text generation. Our analysis shows that LLMs generate stereotyped representations of sexual and gender minorities in this setting, showing that they amplify representational harms in creative writing, a widely advertised use for LLMs.
>
---
#### [replaced 075] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v3](http://arxiv.org/pdf/2505.12363v3)**

> **作者:** Qi Feng
>
> **备注:** In version 1, Hidetoshi Shimodaira was included as a co-author without their consent and has been removed from the author list
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 076] Balancing Computation Load and Representation Expressivity in Parallel Hybrid Neural Networks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19472v2](http://arxiv.org/pdf/2505.19472v2)**

> **作者:** Mohammad Mahdi Moradi; Walid Ahmed; Shuangyue Wen; Sudhir Mudur; Weiwei Zhang; Yang Liu
>
> **摘要:** Attention and State-Space Models (SSMs) when combined in a hybrid network in sequence or in parallel provide complementary strengths. In a hybrid sequential pipeline they alternate between applying a transformer to the input and then feeding its output into a SSM. This results in idle periods in the individual components increasing end-to-end latency and lowering throughput caps. In the parallel hybrid architecture, the transformer operates independently in parallel with the SSM, and these pairs are cascaded, with output from one pair forming the input to the next. Two issues are (i) creating an expressive knowledge representation with the inherently divergent outputs from these separate branches, and (ii) load balancing the computation between these parallel branches, while maintaining representation fidelity. In this work we present FlowHN, a novel parallel hybrid network architecture that accommodates various strategies for load balancing, achieved through appropriate distribution of input tokens between the two branches. Two innovative differentiating factors in FlowHN include a FLOP aware dynamic token split between the attention and SSM branches yielding efficient balance in compute load, and secondly, a method to fuse the highly divergent outputs from individual branches for enhancing representation expressivity. Together they enable much better token processing speeds, avoid bottlenecks, and at the same time yield significantly improved accuracy as compared to other competing works. We conduct comprehensive experiments on autoregressive language modeling for models with 135M, 350M, and 1B parameters. FlowHN outperforms sequential hybrid models and its parallel counterpart, achieving up to 4* higher Tokens per Second (TPS) and 2* better Model FLOPs Utilization (MFU).
>
---
#### [replaced 077] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21329v2](http://arxiv.org/pdf/2505.21329v2)**

> **作者:** Allaa Boutaleb; Bernd Amann; Hubert Naacke; Rafael Angarita
>
> **备注:** Accepted @ ACL 2025's Table Representation Learning Workshop (TRL)
>
> **摘要:** Recent table representation learning and data discovery methods tackle table union search (TUS) within data lakes, which involves identifying tables that can be unioned with a given query table to enrich its content. These methods are commonly evaluated using benchmarks that aim to assess semantic understanding in real-world TUS tasks. However, our analysis of prominent TUS benchmarks reveals several limitations that allow simple baselines to perform surprisingly well, often outperforming more sophisticated approaches. This suggests that current benchmark scores are heavily influenced by dataset-specific characteristics and fail to effectively isolate the gains from semantic understanding. To address this, we propose essential criteria for future benchmarks to enable a more realistic and reliable evaluation of progress in semantic table union search.
>
---
#### [replaced 078] Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.10832v2](http://arxiv.org/pdf/2505.10832v2)**

> **作者:** Songjun Tu; Jiahao Lin; Qichao Zhang; Xiangyu Tian; Linjing Li; Xiangyuan Lan; Dongbin Zhao
>
> **备注:** Fisrt Submitted on 16 May 2025; Update on 28 May 2025
>
> **摘要:** Large reasoning models (LRMs) are proficient at generating explicit, step-by-step reasoning sequences before producing final answers. However, such detailed reasoning can introduce substantial computational overhead and latency, particularly for simple problems. To address this over-thinking problem, we explore how to equip LRMs with adaptive thinking capabilities: enabling them to dynamically decide whether or not to engage in explicit reasoning based on problem complexity. Building on R1-style distilled models, we observe that inserting a simple ellipsis ("...") into the prompt can stochastically trigger either a thinking or no-thinking mode, revealing a latent controllability in the reasoning behavior. Leveraging this property, we propose AutoThink, a multi-stage reinforcement learning (RL) framework that progressively optimizes reasoning policies via stage-wise reward shaping. AutoThink learns to invoke explicit reasoning only when necessary, while defaulting to succinct responses for simpler tasks. Experiments on five mainstream mathematical benchmarks demonstrate that AutoThink achieves favorable accuracy-efficiency trade-offs compared to recent prompting and RL-based pruning methods. It can be seamlessly integrated into any R1-style model, including both distilled and further fine-tuned variants. Notably, AutoThink improves relative accuracy by 6.4 percent while reducing token usage by 52 percent on DeepSeek-R1-Distill-Qwen-1.5B, establishing a scalable and adaptive reasoning paradigm for LRMs. Project Page: https://github.com/ScienceOne-AI/AutoThink.
>
---
#### [replaced 079] Exploring the Limitations of Mamba in COPY and CoT Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03810v2](http://arxiv.org/pdf/2410.03810v2)**

> **作者:** Ruifeng Ren; Zhicong Li; Yong Liu
>
> **备注:** Mamba, Chain of Thought
>
> **摘要:** Transformers have become the backbone of modern Large Language Models (LLMs); however, their inference overhead grows linearly with the sequence length, posing challenges for modeling long sequences. In light of this, Mamba has attracted attention for maintaining a constant inference size, with empirical evidence demonstrating that it can match Transformer performance in sequence modeling while significantly reducing computational costs. However, an open question remains: can Mamba always bring savings while achieving performance comparable to Transformers? In this paper, we focus on analyzing the expressive ability of Mamba to perform our defined COPY operation and Chain of Thought (CoT) reasoning. First, inspired by the connection between Mamba and linear attention, we show that constant-sized Mamba may struggle to perform COPY operations while Transformers can handle them more easily. However, when the size of Mamba grows linearly with the input sequence length, it can accurately perform COPY, but in this case, Mamba no longer provides overhead savings. Based on this observation, we further analyze Mamba's ability to tackle CoT tasks, which can be described by the Dynamic Programming (DP) problems. Our findings suggest that to solve arbitrary DP problems, the total cost of Mamba is still comparable to standard Transformers. However, similar to efficient Transformers, when facing DP problems with favorable properties such as locality, Mamba can provide savings in overhead. Our experiments on the copy and CoT tasks further demonstrate Mamba's limitations compared to Transformers in learning these tasks.
>
---
#### [replaced 080] EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16160v3](http://arxiv.org/pdf/2505.16160v3)**

> **作者:** Bin Xu; Yu Bai; Huashan Sun; Yiguan Lin; Siming Liu; Xinyue Liang; Yaolin Li; Yang Gao; Heyan Huang
>
> **摘要:** As large language models continue to advance, their application in educational contexts remains underexplored and under-optimized. In this paper, we address this gap by introducing the first diverse benchmark tailored for educational scenarios, incorporating synthetic data containing 9 major scenarios and over 4,000 distinct educational contexts. To enable comprehensive assessment, we propose a set of multi-dimensional evaluation metrics that cover 12 critical aspects relevant to both teachers and students. We further apply human annotation to ensure the effectiveness of the model-generated evaluation responses. Additionally, we succeed to train a relatively small-scale model on our constructed dataset and demonstrate that it can achieve performance comparable to state-of-the-art large models (e.g., Deepseek V3, Qwen Max) on the test set. Overall, this work provides a practical foundation for the development and evaluation of education-oriented language models. Code and data are released at https://github.com/ybai-nlp/EduBench.
>
---
#### [replaced 081] Domain-Specific Pruning of Large Mixture-of-Experts Models with Few-shot Demonstrations
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.06792v2](http://arxiv.org/pdf/2504.06792v2)**

> **作者:** Zican Dong; Han Peng; Peiyu Liu; Wayne Xin Zhao; Dong Wu; Feng Xiao; Zhifeng Wang
>
> **摘要:** Mixture-of-Experts (MoE) models achieve a favorable trade-off between performance and inference efficiency by activating only a subset of experts. However, the memory overhead of storing all experts remains a major limitation, especially in large-scale MoE models such as DeepSeek-R1(671B). In this study, we investigate domain specialization and expert redundancy in large-scale MoE models and uncover a consistent behavior we term few-shot expert localization, with only a few in-domain demonstrations, the model consistently activates a sparse and stable subset of experts on tasks within the same domain. Building on this observation, we propose a simple yet effective pruning framework, EASY-EP, that leverages a few domain-specific demonstrations to identify and retain only the most relevant experts. EASY-EP comprises two key components: output-aware expert importance assessment and expert-level token contribution estimation. The former evaluates the importance of each expert for the current token by considering the gating scores and L2 norm of the outputs of activated experts, while the latter assesses the contribution of tokens based on representation similarities before and after routed experts. Experiments on DeepSeek-R1 and DeepSeek-V3-0324 show that our method can achieve comparable performances and $2.99\times$ throughput under the same memory budget with full model with only half the experts.
>
---
#### [replaced 082] General-Reasoner: Advancing LLM Reasoning Across All Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14652v4](http://arxiv.org/pdf/2505.14652v4)**

> **作者:** Xueguang Ma; Qian Liu; Dongfu Jiang; Ge Zhang; Zejun Ma; Wenhu Chen
>
> **摘要:** Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
>
---
#### [replaced 083] LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21082v2](http://arxiv.org/pdf/2505.21082v2)**

> **作者:** Jieyong Kim; Tongyoung Kim; Soojin Yoon; Jaehyung Kim; Dongha Lee
>
> **摘要:** Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
>
---
#### [replaced 084] Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19475v2](http://arxiv.org/pdf/2505.19475v2)**

> **作者:** Mohammad Mahdi Moradi; Hossam Amer; Sudhir Mudur; Weiwei Zhang; Yang Liu; Walid Ahmed
>
> **摘要:** Learning to adapt pretrained language models to unlabeled, out-of-distribution data is a critical challenge, as models often falter on structurally novel reasoning tasks even while excelling within their training distribution. We introduce a new framework called VDS-TTT - Verifier-Driven Sample Selection for Test-Time Training to efficiently address this. We use a learned verifier to score a pool of generated responses and select only from high ranking pseudo-labeled examples for fine-tuned adaptation. Specifically, for each input query our LLM generates N candidate answers; the verifier assigns a reliability score to each, and the response with the highest confidence and above a fixed threshold is paired with its query for test-time training. We fine-tune only low-rank LoRA adapter parameters, ensuring adaptation efficiency and fast convergence. Our proposed self-supervised framework is the first to synthesize verifier driven test-time training data for continuous self-improvement of the model. Experiments across three diverse benchmarks and three state-of-the-art LLMs demonstrate that VDS-TTT yields up to a 32.29% relative improvement over the base model and a 6.66% gain compared to verifier-based methods without test-time training, highlighting its effectiveness and efficiency for on-the-fly large language model adaptation.
>
---
#### [replaced 085] Incentivizing Strong Reasoning from Weak Supervision
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20072v2](http://arxiv.org/pdf/2505.20072v2)**

> **作者:** Yige Yuan; Teng Xiao; Shuchang Tao; Xue Wang; Jinyang Gao; Bolin Ding; Bingbing Xu
>
> **摘要:** Large language models (LLMs) have demonstrated impressive performance on reasoning-intensive tasks, but enhancing their reasoning abilities typically relies on either reinforcement learning (RL) with verifiable signals or supervised fine-tuning (SFT) with high-quality long chain-of-thought (CoT) demonstrations, both of which are expensive. In this paper, we study a novel problem of incentivizing the reasoning capacity of LLMs without expensive high-quality demonstrations and reinforcement learning. We investigate whether the reasoning capabilities of LLMs can be effectively incentivized via supervision from significantly weaker models. We further analyze when and why such weak supervision succeeds in eliciting reasoning abilities in stronger models. Our findings show that supervision from significantly weaker reasoners can substantially improve student reasoning performance, recovering close to 94% of the gains of expensive RL at a fraction of the cost. Experiments across diverse benchmarks and model architectures demonstrate that weak reasoners can effectively incentivize reasoning in stronger student models, consistently improving performance across a wide range of reasoning tasks. Our results suggest that this simple weak-to-strong paradigm is a promising and generalizable alternative to costly methods for incentivizing strong reasoning capabilities at inference-time in LLMs. The code is publicly available at https://github.com/yuanyige/w2sr.
>
---
#### [replaced 086] Mitigating Text Toxicity with Counterfactual Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.09948v3](http://arxiv.org/pdf/2405.09948v3)**

> **作者:** Milan Bhan; Jean-Noel Vittaut; Nina Achache; Victor Legrand; Nicolas Chesneau; Annabelle Blangero; Juliette Murris; Marie-Jeanne Lesot
>
> **摘要:** Toxicity mitigation consists in rephrasing text in order to remove offensive or harmful meaning. Neural natural language processing (NLP) models have been widely used to target and mitigate textual toxicity. However, existing methods fail to detoxify text while preserving the initial non-toxic meaning at the same time. In this work, we propose to apply counterfactual generation methods from the eXplainable AI (XAI) field to target and mitigate textual toxicity. In particular, we perform text detoxification by applying local feature importance and counterfactual generation methods to a toxicity classifier distinguishing between toxic and non-toxic texts. We carry out text detoxification through counterfactual generation on three datasets and compare our approach to three competitors. Automatic and human evaluations show that recently developed NLP counterfactual generators can mitigate toxicity accurately while better preserving the meaning of the initial text as compared to classical detoxification methods. Finally, we take a step back from using automated detoxification tools, and discuss how to manage the polysemous nature of toxicity and the risk of malicious use of detoxification tools. This work is the first to bridge the gap between counterfactual generation and text detoxification and paves the way towards more practical application of XAI methods.
>
---
#### [replaced 087] Towards Practical Defect-Focused Automated Code Review
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.17928v2](http://arxiv.org/pdf/2505.17928v2)**

> **作者:** Junyi Lu; Lili Jiang; Xiaojia Li; Jianbing Fang; Fengjun Zhang; Li Yang; Chun Zuo
>
> **备注:** Accepted as Spotlight at the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: 1) capturing relevant context, 2) improving key bug inclusion (KBI), 3) reducing false alarm rates (FAR), and 4) integrating human workflows. To tackle these, we propose 1) code slicing algorithms for context extraction, 2) a multi-role LLM framework for KBI, 3) a filtering mechanism for FAR reduction, and 4) a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2x improvement over standard LLMs and a 10x gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages language-agnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability.
>
---
#### [replaced 088] Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration
- **分类: cs.AI; cs.CL; cs.HC; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.11882v5](http://arxiv.org/pdf/2502.11882v5)**

> **作者:** Shao Zhang; Xihuai Wang; Wenhao Zhang; Chaoran Li; Junru Song; Tingyu Li; Lin Qiu; Xuezhi Cao; Xunliang Cai; Wen Yao; Weinan Zhang; Xinbing Wang; Ying Wen
>
> **备注:** Accepted by ACL 2025 Main. Camera Ready Version
>
> **摘要:** Agents built on large language models (LLMs) have excelled in turn-by-turn human-AI collaboration but struggle with simultaneous tasks requiring real-time interaction. Latency issues and the challenge of inferring variable human strategies hinder their ability to make autonomous decisions without explicit instructions. Through experiments with current independent System 1 and System 2 methods, we validate the necessity of using Dual Process Theory (DPT) in real-time tasks. We propose DPT-Agent, a novel language agent framework that integrates System 1 and System 2 for efficient real-time simultaneous human-AI collaboration. DPT-Agent's System 1 uses a Finite-state Machine (FSM) and code-as-policy for fast, intuitive, and controllable decision-making. DPT-Agent's System 2 integrates Theory of Mind (ToM) and asynchronous reflection to infer human intentions and perform reasoning-based autonomous decisions. We demonstrate the effectiveness of DPT-Agent through further experiments with rule-based agents and human collaborators, showing significant improvements over mainstream LLM-based frameworks. DPT-Agent can effectively help LLMs convert correct slow thinking and reasoning into executable actions, thereby improving performance. To the best of our knowledge, DPT-Agent is the first language agent framework that achieves successful real-time simultaneous human-AI collaboration autonomously. Code of DPT-Agent can be found in https://github.com/sjtu-marl/DPT-Agent.
>
---
#### [replaced 089] The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19797v2](http://arxiv.org/pdf/2505.19797v2)**

> **作者:** Yiqun Zhang; Hao Li; Chenxu Wang; Linyao Chen; Qiaosheng Zhang; Peng Ye; Shi Feng; Daling Wang; Zhen Wang; Xinrun Wang; Jia Xu; Lei Bai; Wanli Ouyang; Shuyue Hu
>
> **备注:** 9 pages, 3 figures, 6 tables, supplementary material (appendix) included separately
>
> **摘要:** As proprietary giants increasingly dominate the race for ever-larger language models, a pressing question arises for the open-source community: can smaller models remain competitive across a broad range of tasks? In this paper, we present the Avengers--a simple recipe that effectively leverages the collective intelligence of open-source, smaller language models. Our framework is built upon four lightweight operations: (i) embedding: encode queries using a text embedding model; (ii) clustering: group queries based on their semantic similarity; (iii) scoring: scores each model's performance within each cluster; and (iv) voting: improve outputs via repeated sampling and voting. At inference time, each query is embedded and assigned to its nearest cluster. The top-performing model(s) within that cluster are selected to generate the response using the Self-Consistency or its multi-model variant. Remarkably, with 10 open-source models (~7B parameters each), the Avengers collectively outperforms GPT-4.1 on nine out of 15 datasets (spanning mathematics, code, logic, knowledge, and affective tasks). In particular, it surpasses GPT-4.1 on mathematics tasks by 18.21% and on code tasks by 7.46%. Furthermore, the Avengers delivers superior out-of-distribution generalization, and remains robust across various embedding models, clustering algorithms, ensemble strategies, and values of its sole parameter--the number of clusters. We have open-sourced the code on GitHub: https://github.com/ZhangYiqun018/Avengers
>
---
#### [replaced 090] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03124v4](http://arxiv.org/pdf/2501.03124v4)**

> **作者:** Mingyang Song; Zhaochen Su; Xiaoye Qu; Jiawei Zhou; Yu Cheng
>
> **备注:** Accepted by ACL 2025 Main. Project Page: https://prmbench.github.io/
>
> **摘要:** Process-level Reward Models (PRMs) are crucial for complex reasoning and decision-making tasks, where each intermediate step plays an important role in the reasoning process. Since language models are prone to various types of errors during the reasoning process, PRMs are required to possess nuanced capabilities for detecting various implicit error types in real-world scenarios. However, current benchmarks primarily focus on step correctness, failing to evaluate PRMs' performance systematically. To address this gap, we introduce PRMBench, a process-level benchmark specifically designed to assess the fine-grained error detection capabilities of PRMs. PRMBench comprises 6,216 carefully designed problems and 83,456 step-level labels, evaluating models across multiple dimensions, including simplicity, soundness, and sensitivity. In our experiments on 15 models, spanning both open-source PRMs and closed-source large language models prompted as critic models, we uncover significant weaknesses in current PRMs. These findings underscore the challenges inherent in process-level evaluation and highlight key directions for future research. We hope PRMBench can be a robust bench for advancing research on PRM evaluation and development.
>
---
#### [replaced 091] AdvAgent: Controllable Blackbox Red-teaming on Web Agents
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17401v3](http://arxiv.org/pdf/2410.17401v3)**

> **作者:** Chejian Xu; Mintong Kang; Jiawei Zhang; Zeyi Liao; Lingbo Mo; Mengqi Yuan; Huan Sun; Bo Li
>
> **备注:** ICML 2025
>
> **摘要:** Foundation model-based agents are increasingly used to automate complex tasks, enhancing efficiency and productivity. However, their access to sensitive resources and autonomous decision-making also introduce significant security risks, where successful attacks could lead to severe consequences. To systematically uncover these vulnerabilities, we propose AdvAgent, a black-box red-teaming framework for attacking web agents. Unlike existing approaches, AdvAgent employs a reinforcement learning-based pipeline to train an adversarial prompter model that optimizes adversarial prompts using feedback from the black-box agent. With careful attack design, these prompts effectively exploit agent weaknesses while maintaining stealthiness and controllability. Extensive evaluations demonstrate that AdvAgent achieves high success rates against state-of-the-art GPT-4-based web agents across diverse web tasks. Furthermore, we find that existing prompt-based defenses provide only limited protection, leaving agents vulnerable to our framework. These findings highlight critical vulnerabilities in current web agents and emphasize the urgent need for stronger defense mechanisms. We release code at https://ai-secure.github.io/AdvAgent/.
>
---
#### [replaced 092] Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.04122v2](http://arxiv.org/pdf/2409.04122v2)**

> **作者:** Jan Hofmann; Cornelia Sindermann; Roman Klinger
>
> **备注:** Accepted to the REALM workshop at ACL 2025
>
> **摘要:** Author profiling is the task of inferring characteristics about individuals by analyzing content they share. Supervised machine learning still dominates automatic systems that perform this task, despite the popularity of prompting large language models to address natural language understanding tasks. One reason is that the classification instances consist of large amounts of posts, potentially a whole user profile, which may exceed the input length of Transformers. Even if a model can use a large context window, the entirety of posts makes the application of API-accessed black box systems costly and slow, next to issues which come with such "needle-in-the-haystack" tasks. To mitigate this limitation, we propose a new method for author profiling which aims at distinguishing relevant from irrelevant content first, followed by the actual user profiling only with relevant data. To circumvent the need for relevance-annotated data, we optimize this relevance filter via reinforcement learning with a reward function that utilizes the zero-shot capabilities of large language models. We evaluate our method for Big Five personality trait prediction on two Twitter corpora. On publicly available real-world data with a skewed label distribution, our method shows similar efficacy to using all posts in a user profile, but with a substantially shorter context. An evaluation on a version of these data balanced with artificial posts shows that the filtering to relevant posts leads to a significantly improved accuracy of the predictions.
>
---
#### [replaced 093] Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.14431v2](http://arxiv.org/pdf/2501.14431v2)**

> **作者:** Xu Chu; Zhijie Tan; Hanlin Xue; Guanyu Wang; Tong Mo; Weiping Li
>
> **摘要:** Large Language Models (LLMs) are widely applied to downstream domains. However, current LLMs for high-stakes domain tasks, such as financial investment and legal QA, typically generate brief answers without reasoning processes and explanations. This limits users' confidence in making decisions based on their responses. While original CoT shows promise, it lacks self-correction mechanisms during reasoning. This work introduces Domain$o1$s, which enhances LLMs' reasoning capabilities on domain tasks through supervised fine-tuning and tree search. We construct CoT-stock-2k and CoT-legal-2k datasets for fine-tuning models that activate domain-specific reasoning steps based on their judgment. Additionally, we propose Selective Tree Exploration to spontaneously explore solution spaces and sample optimal reasoning paths to improve performance. We also introduce PROOF-Score, a new metric for evaluating domain models' explainability, complementing traditional accuracy metrics with richer assessment dimensions. Extensive experiments on stock investment recommendation and legal reasoning QA tasks demonstrate Domaino1s's leading performance and explainability. Our code is available at https://github.com/Hyalinesky/Domaino1s.
>
---
#### [replaced 094] Language-Specific Latent Process Hinders Cross-Lingual Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13141v2](http://arxiv.org/pdf/2505.13141v2)**

> **作者:** Zheng Wei Lim; Alham Fikri Aji; Trevor Cohn
>
> **摘要:** Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. We find LLMs predict inconsistently and are less accurate because they rely on subspaces of individual languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing can be modulated by steering the models' latent processing towards the shared semantic space. We find reinforcing utilization of the shared space improves the models' multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English.
>
---
#### [replaced 095] When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16170v2](http://arxiv.org/pdf/2505.16170v2)**

> **作者:** Yuqing Yang; Robin Jia
>
> **备注:** Fixed typos
>
> **摘要:** Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on https://github.com/ayyyq/llm-retraction.
>
---
#### [replaced 096] Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21411v2](http://arxiv.org/pdf/2505.21411v2)**

> **作者:** Yehui Tang; Xiaosong Li; Fangcheng Liu; Wei Guo; Hang Zhou; Yaoyuan Wang; Kai Han; Xianzhi Yu; Jinpeng Li; Hui Zang; Fei Mi; Xiaojun Meng; Zhicheng Liu; Hanting Chen; Binfan Zheng; Can Chen; Youliang Yan; Ruiming Tang; Peifeng Qin; Xinghao Chen; Dacheng Tao; Yunhe Wang
>
> **摘要:** The surgence of Mixture of Experts (MoE) in Large Language Models promises a small price of execution cost for a much larger model parameter count and learning capacity, because only a small fraction of parameters are activated for each input token. However, it is commonly observed that some experts are activated far more often than others, leading to system inefficiency when running the experts on different devices in parallel. Therefore, we introduce Mixture of Grouped Experts (MoGE), which groups the experts during selection and balances the expert workload better than MoE in nature. It constrains tokens to activate an equal number of experts within each predefined expert group. When a model execution is distributed on multiple devices, this architectural design ensures a balanced computational load across devices, significantly enhancing throughput, particularly for the inference phase. Further, we build Pangu Pro MoE on Ascend NPUs, a sparse model based on MoGE with 72 billion total parameters, 16 billion of which are activated for each token. The configuration of Pangu Pro MoE is optimized for Ascend 300I Duo and 800I A2 through extensive system simulation studies. Our experiments indicate that MoGE indeed leads to better expert load balancing and more efficient execution for both model training and inference on Ascend NPUs. The inference performance of Pangu Pro MoE achieves 1148 tokens/s per card and can be further improved to 1528 tokens/s per card by speculative acceleration, outperforming comparable 32B and 72B Dense models. Furthermore, we achieve an excellent cost-to-performance ratio for model inference on Ascend 300I Duo. Our studies show that Ascend NPUs are capable of training Pangu Pro MoE with massive parallelization to make it a leading model within the sub-100B total parameter class, outperforming prominent open-source models like GLM-Z1-32B and Qwen3-32B.
>
---
#### [replaced 097] GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16514v4](http://arxiv.org/pdf/2502.16514v4)**

> **作者:** Yingjian Chen; Haoran Liu; Yinhong Liu; Jinxiang Xie; Rui Yang; Han Yuan; Yanran Fu; Peng Yuan Zhou; Qingyu Chen; James Caverlee; Irene Li
>
> **摘要:** Large language models (LLMs) are widely used, but they often generate subtle factual errors, especially in long-form text. These errors are fatal in some specialized domains such as medicine. Existing fact-checking with grounding documents methods face two main challenges: (1) they struggle to understand complex multihop relations in long documents, often overlooking subtle factual errors; (2) most specialized methods rely on pairwise comparisons, requiring multiple model calls, leading to high resource and computational costs. To address these challenges, we propose GraphCheck, a fact-checking framework that uses extracted knowledge graphs to enhance text representation. Graph Neural Networks further process these graphs as a soft prompt, enabling LLMs to incorporate structured knowledge more effectively. Enhanced with graph-based reasoning, GraphCheck captures multihop reasoning chains that are often overlooked by existing methods, enabling precise and efficient fact-checking in a single inference call. Experimental results on seven benchmarks spanning both general and medical domains demonstrate up to a 7.1% overall improvement over baseline models. Notably, GraphCheck outperforms existing specialized fact-checkers and achieves comparable performance with state-of-the-art LLMs, such as DeepSeek-V3 and OpenAI-o1, with significantly fewer parameters.
>
---
#### [replaced 098] Explicit Learning and the LLM in Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09454v3](http://arxiv.org/pdf/2503.09454v3)**

> **作者:** Malik Marmonier; Rachel Bawden; Benoît Sagot
>
> **摘要:** This study explores an LLM's ability to learn new languages using explanations found in a grammar book$\unicode{x2014}$a process we term "explicit learning." To rigorously assess this ability, we design controlled translation experiments between English and constructed languages generated$\unicode{x2014}$by specific cryptographic means$\unicode{x2014}$out of Latin or French. Contrary to previous studies, our results demonstrate that LLMs do possess a measurable capacity for explicit learning. This ability, however, diminishes as the complexity of the linguistic phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs, benefiting low-resource languages typically described in grammar books but lacking extensive corpora.
>
---
#### [replaced 099] Large Vocabulary Size Improves Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.16508v2](http://arxiv.org/pdf/2406.16508v2)**

> **作者:** Sho Takase; Ryokan Ri; Shun Kiyono; Takuya Kato
>
> **备注:** Findings of ACL 2025
>
> **摘要:** This paper empirically investigates the relationship between subword vocabulary size and the performance of large language models (LLMs) to provide insights on how to define the vocabulary size. Experimental results show that larger vocabulary sizes lead to better performance in LLMs. Moreover, we consider a continual training scenario where a pre-trained language model is trained on a different target language. We introduce a simple method to use a new vocabulary instead of the pre-defined one. We show that using the new vocabulary outperforms the model with the vocabulary used in pre-training.
>
---
#### [replaced 100] Enhancing Target-unspecific Tasks through a Features Matrix
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03414v4](http://arxiv.org/pdf/2505.03414v4)**

> **作者:** Fangming Cui; Yonggang Zhang; Xuan Wang; Xinmei Tian; Jun Yu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Recent developments in prompt learning of large Vision-Language Models (VLMs) have significantly improved performance in target-specific tasks. However, these prompting methods often struggle to tackle the target-unspecific or generalizable tasks effectively. It may be attributed to the fact that overfitting training causes the model to forget its general knowledge. The general knowledge has a strong promotion on target-unspecific tasks. To alleviate this issue, we propose a novel Features Matrix (FM) approach designed to enhance these models on target-unspecific tasks. Our method extracts and leverages general knowledge, shaping a Features Matrix (FM). Specifically, the FM captures the semantics of diverse inputs from a deep and fine perspective, preserving essential general knowledge, which mitigates the risk of overfitting. Representative evaluations demonstrate that: 1) the FM is compatible with existing frameworks as a generic and flexible module, and 2) the FM significantly showcases its effectiveness in enhancing target-unspecific tasks (base-to-novel generalization, domain generalization, and cross-dataset generalization), achieving state-of-the-art performance.
>
---
#### [replaced 101] Probabilistic Reasoning with LLMs for k-anonymity Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09674v3](http://arxiv.org/pdf/2503.09674v3)**

> **作者:** Jonathan Zheng; Sauvik Das; Alan Ritter; Wei Xu
>
> **备注:** 9 pages, preprint
>
> **摘要:** Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
>
---
#### [replaced 102] Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14245v2](http://arxiv.org/pdf/2502.14245v2)**

> **作者:** Rongzhi Zhu; Xiangyu Liu; Zequn Sun; Yiwei Wang; Wei Hu
>
> **备注:** Accepted in the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** In this paper, we identify a critical problem, "lost-in-retrieval", in retrieval-augmented multi-hop question answering (QA): the key entities are missed in LLMs' sub-question decomposition. "Lost-in-retrieval" significantly degrades the retrieval performance, which disrupts the reasoning chain and leads to the incorrect answers. To resolve this problem, we propose a progressive retrieval and rewriting method, namely ChainRAG, which sequentially handles each sub-question by completing missing key entities and retrieving relevant sentences from a sentence graph for answer generation. Each step in our retrieval and rewriting process builds upon the previous one, creating a seamless chain that leads to accurate retrieval and answers. Finally, all retrieved sentences and sub-question answers are integrated to generate a comprehensive answer to the original question. We evaluate ChainRAG on three multi-hop QA datasets - MuSiQue, 2Wiki, and HotpotQA - using three large language models: GPT4o-mini, Qwen2.5-72B, and GLM-4-Plus. Empirical results demonstrate that ChainRAG consistently outperforms baselines in both effectiveness and efficiency.
>
---
#### [replaced 103] Reasoning Is Not All You Need: Examining LLMs for Multi-Turn Mental Health Conversations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20201v2](http://arxiv.org/pdf/2505.20201v2)**

> **作者:** Mohit Chandra; Siddharth Sriraman; Harneet Singh Khanuja; Yiqiao Jin; Munmun De Choudhury
>
> **备注:** 34 pages, 5 figures, 30 tables
>
> **摘要:** Limited access to mental healthcare, extended wait times, and increasing capabilities of Large Language Models (LLMs) has led individuals to turn to LLMs for fulfilling their mental health needs. However, examining the multi-turn mental health conversation capabilities of LLMs remains under-explored. Existing evaluation frameworks typically focus on diagnostic accuracy and win-rates and often overlook alignment with patient-specific goals, values, and personalities required for meaningful conversations. To address this, we introduce MedAgent, a novel framework for synthetically generating realistic, multi-turn mental health sensemaking conversations and use it to create the Mental Health Sensemaking Dialogue (MHSD) dataset, comprising over 2,200 patient-LLM conversations. Additionally, we present MultiSenseEval, a holistic framework to evaluate the multi-turn conversation abilities of LLMs in healthcare settings using human-centric criteria. Our findings reveal that frontier reasoning models yield below-par performance for patient-centric communication and struggle at advanced diagnostic capabilities with average score of 31%. Additionally, we observed variation in model performance based on patient's persona and performance drop with increasing turns in the conversation. Our work provides a comprehensive synthetic data generation framework, a dataset and evaluation framework for assessing LLMs in multi-turn mental health conversations.
>
---
#### [replaced 104] Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.17933v2](http://arxiv.org/pdf/2503.17933v2)**

> **作者:** Justice Ou; Tinglin Huang; Yilun Zhao; Ziyang Yu; Peiqing Lu; Rex Ying
>
> **摘要:** To improve the reliability of Large Language Models (LLMs) in clinical applications, retrieval-augmented generation (RAG) is extensively applied to provide factual medical knowledge. However, beyond general medical knowledge from open-ended datasets, clinical case-based knowledge is also critical for effective medical reasoning, as it provides context grounded in real-world patient experiences.Motivated by this, we propose Experience Retrieval-Augmentation ExpRAG framework based on Electronic Health Record(EHR), aiming to offer the relevant context from other patients' discharge reports. ExpRAG performs retrieval through a coarse-to-fine process, utilizing an EHR-based report ranker to efficiently identify similar patients, followed by an experience retriever to extract task-relevant content for enhanced medical reasoning.To evaluate ExpRAG, we introduce DischargeQA, a clinical QA dataset with 1,280 discharge-related questions across diagnosis, medication, and instruction tasks. Each problem is generated using EHR data to ensure realistic and challenging scenarios. Experimental results demonstrate that ExpRAG consistently outperforms a text-based ranker, achieving an average relative improvement of 5.2%, highlighting the importance of case-based knowledge for medical reasoning.
>
---
#### [replaced 105] Non-Markovian Discrete Diffusion with Causal Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09767v2](http://arxiv.org/pdf/2502.09767v2)**

> **作者:** Yangtian Zhang; Sizhuang He; Daniel Levine; Lawrence Zhao; David Zhang; Syed A Rizvi; Emanuele Zappala; Rex Ying; David van Dijk
>
> **备注:** Under Review
>
> **摘要:** Discrete diffusion models offer a flexible, controllable approach to structured sequence generation, yet they still lag behind causal language models in expressive power. A key limitation lies in their reliance on the Markovian assumption, which restricts each step to condition only on the current state, leading to potential uncorrectable error accumulation. In this paper, we introduce CaDDi, a discrete diffusion model that conditions on the entire generative trajectory, thereby lifting the Markov constraint and allowing the model to revisit and improve past states. By unifying sequential (causal) and temporal (diffusion) reasoning in a single non-Markovian transformer, CaDDi also treats standard causal language models as a special case and permits the direct reuse of pretrained LLM weights with no architectural changes. Empirically, CaDDi outperforms state-of-the-art discrete diffusion baselines on natural-language benchmarks, substantially narrowing the remaining gap to large autoregressive transformers.
>
---
#### [replaced 106] Sun-Shine: A Foundation Large Language Model for Tibetan Culture and Heritage
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18288v3](http://arxiv.org/pdf/2503.18288v3)**

> **作者:** Cheng Huang; Fan Gao; Yutong Liu; Nyima Tashi; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Xiao Feng; Hao Wang; Yongbin Yu
>
> **摘要:** Tibetan, a minority language in China, features a highly intricate grammatical structure, characterized by four verb tenses and a tense system with frequent irregularities, contributing to its extensive inflectional diversity. Recently, advances in Large Language Models (LLMs) have transformed the paradigm in many domains. Despite the success in other fields, current LLMs often fall short in catering to the needs of domain experts like Tibetans, and the potential of LLMs for Tibetan culture is under-explored. The intrinsic reasons are the immense and intricate nature of Tibetan culture as well as the necessity for higher granularity and richness in knowledge. Simultaneously, the complexity and uniqueness of its grammatical structure, coupled with its status as a minority ethnic language, contribute to data scarcity, which remains a fundamental challenge. To alleviate these issues, we introduce Llama-Sunshine (Sun-Shine), the first large language model for Tibetan culture, which is expert in various Tibetan language processing tasks. Sun-Shine incorporates state-of-the-art model architectures optimized for Tibetan's linguistic features. We also propose TIB-STC, a comprehensive dataset comprising diverse Tibetan texts such as literature, religious scripts, news, and conversational data, which is also the first large-scale dataset for Tibetan culture. Though comprehensive experiments, Sun-Shine not only demonstrates a higher level of knowledge expertise for Tibetan culture but also gains preliminary embodied intelligence capabilities in Tibetan language processing tasks, like language modeling, text classification, machine translation, and syntactic analysis. Moreover, it excels in low-resource scenarios, showcasing strong generalization capabilities.
>
---
#### [replaced 107] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18079v2](http://arxiv.org/pdf/2505.18079v2)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** V2 draft. Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later.
>
---
#### [replaced 108] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13080v2](http://arxiv.org/pdf/2410.13080v2)**

> **作者:** Linhao Luo; Zicheng Zhao; Gholamreza Haffari; Yuan-Fang Li; Chen Gong; Shirui Pan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Large language models (LLMs) have demonstrated impressive reasoning abilities, but they still struggle with faithful reasoning due to knowledge gaps and hallucinations. To address these issues, knowledge graphs (KGs) have been utilized to enhance LLM reasoning through their structured knowledge. However, existing KG-enhanced methods, either retrieval-based or agent-based, encounter difficulties in accurately retrieving knowledge and efficiently traversing KGs at scale. In this work, we introduce graph-constrained reasoning (GCR), a novel framework that bridges structured knowledge in KGs with unstructured reasoning in LLMs. To eliminate hallucinations, GCR ensures faithful KG-grounded reasoning by integrating KG structure into the LLM decoding process through KG-Trie, a trie-based index that encodes KG reasoning paths. KG-Trie constrains the decoding process, allowing LLMs to directly reason on graphs and generate faithful reasoning paths grounded in KGs. Additionally, GCR leverages a lightweight KG-specialized LLM for graph-constrained reasoning alongside a powerful general LLM for inductive reasoning over multiple reasoning paths, resulting in accurate reasoning with zero reasoning hallucination. Extensive experiments on several KGQA benchmarks demonstrate that GCR achieves state-of-the-art performance and exhibits strong zero-shot generalizability to unseen KGs without additional training.
>
---
#### [replaced 109] Token embeddings violate the manifold hypothesis
- **分类: cs.CL; cs.AI; 53Z50, 62H15**

- **链接: [http://arxiv.org/pdf/2504.01002v2](http://arxiv.org/pdf/2504.01002v2)**

> **作者:** Michael Robinson; Sourya Dey; Tony Chiang
>
> **备注:** 27 pages, 6 figures, 9 tables
>
> **摘要:** A full understanding of the behavior of a large language model (LLM) requires our understanding of its input token space. If this space differs from our assumptions, our understanding of and conclusions about the LLM will likely be flawed. We elucidate the structure of the token embeddings both empirically and theoretically. We present a novel statistical test assuming that the neighborhood around each token has a relatively flat and smooth structure as the null hypothesis. Failing to reject the null is uninformative, but rejecting it at a specific token $\psi$ implies an irregularity in the token subspace in a $\psi$-neighborhood, $B(\psi)$. The structure assumed in the null is a generalization of a manifold with boundary called a \emph{smooth fiber bundle} (which can be split into two spatial regimes -- small and large radius), so we denote our new hypothesis test as the ``fiber bundle hypothesis.'' Failure to reject the null hypothesis is uninformative, but rejecting it at $\psi$ indicates a statistically significant irregularity at $B(\psi)$. By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the evidence suggests that the token subspace is not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, if one prompt contains a token implicated by our test, the response to that prompt will likely exhibit less stability than the other.
>
---
#### [replaced 110] CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10688v3](http://arxiv.org/pdf/2503.10688v3)**

> **作者:** Tadesse Destaw Belay; Ahmed Haj Ahmed; Alvin Grissom II; Iqra Ameer; Grigori Sidorov; Olga Kolesnikova; Seid Muhie Yimam
>
> **备注:** ACL-main 2025
>
> **摘要:** NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. The dataset and evaluation code are publicly available.
>
---
#### [replaced 111] Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17601v2](http://arxiv.org/pdf/2505.17601v2)**

> **作者:** Jiawei Kong; Hao Fang; Xiaochen Yang; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Yaowei Wang; Min Zhang
>
> **摘要:** Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.
>
---
#### [replaced 112] ReLearn: Unlearning via Learning for Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11190v3](http://arxiv.org/pdf/2502.11190v3)**

> **作者:** Haoming Xu; Ningyuan Zhao; Liming Yang; Sendong Zhao; Shumin Deng; Mengru Wang; Bryan Hooi; Nay Oo; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Current unlearning methods for large language models usually rely on reverse optimization to reduce target token probabilities. However, this paradigm disrupts the subsequent tokens prediction, degrading model performance and linguistic coherence. Moreover, existing evaluation metrics overemphasize contextual forgetting while inadequately assessing response fluency and relevance. To address these challenges, we propose ReLearn, a data augmentation and fine-tuning pipeline for effective unlearning, along with a comprehensive evaluation framework. This framework introduces Knowledge Forgetting Rate (KFR) and Knowledge Retention Rate (KRR) to measure knowledge-level preservation, and Linguistic Score (LS) to evaluate generation quality. Our experiments show that ReLearn successfully achieves targeted forgetting while preserving high-quality output. Through mechanistic analysis, we further demonstrate how reverse optimization disrupts coherent text generation, while ReLearn preserves this essential capability. Code is available at https://github.com/zjunlp/unlearn.
>
---
#### [replaced 113] ALPS: Attention Localization and Pruning Strategy for Efficient Alignment of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18799v2](http://arxiv.org/pdf/2505.18799v2)**

> **作者:** Hao Chen; Haoze Li; Zhiqing Xiao; Lirong Gao; Qi Zhang; Xiaomeng Hu; Ningtao Wang; Xing Fu; Junbo Zhao
>
> **备注:** 17 pages, 8 figures, 14 tables
>
> **摘要:** Aligning general-purpose large language models (LLMs) to downstream tasks often incurs significant training adjustment costs. Prior research has explored various avenues to enhance alignment efficiency, primarily through minimal-data training or data-driven activations to identify key attention heads. However, these approaches inherently introduce data dependency, which hinders generalization and reusability. To address this issue and enhance model alignment efficiency, we propose the \textit{\textbf{A}ttention \textbf{L}ocalization and \textbf{P}runing \textbf{S}trategy (\textbf{ALPS})}, an efficient algorithm that localizes the most task-sensitive attention heads and prunes by restricting attention training updates to these heads, thereby reducing alignment costs. Experimental results demonstrate that our method activates only \textbf{10\%} of attention parameters during fine-tuning while achieving a \textbf{2\%} performance improvement over baselines on three tasks. Moreover, the identified task-specific heads are transferable across datasets and mitigate knowledge forgetting. Our work and findings provide a novel perspective on efficient LLM alignment. The code is available at https://github.com/VoiceBeer/ALPS.
>
---
#### [replaced 114] PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21342v2](http://arxiv.org/pdf/2505.21342v2)**

> **作者:** Valentin Knappich; Annemarie Friedrich; Anna Hätty; Simon Razniewski
>
> **摘要:** Patent claims define the scope of protection for an invention. If there are ambiguities in a claim, it is rejected by the patent office. In the US, this is referred to as indefiniteness (35 U.S.C {\S} 112(b)) and is among the most frequent reasons for patent application rejection. The development of automatic methods for patent definiteness examination has the potential to make patent drafting and examination more efficient, but no annotated dataset has been published to date. We introduce PEDANTIC (Patent Definiteness Examination Corpus), a novel dataset of 14k US patent claims from patent applications relating to Natural Language Processing (NLP), annotated with reasons for indefiniteness. We construct PEDANTIC using a fully automatic pipeline that retrieves office action documents from the USPTO and uses Large Language Models (LLMs) to extract the reasons for indefiniteness. A human validation study confirms the pipeline's accuracy in generating high-quality annotations. To gain insight beyond binary classification metrics, we implement an LLM-as-Judge evaluation that compares the free-form reasoning of every model-cited reason with every examiner-cited reason. We show that LLM agents based on Qwen 2.5 32B and 72B struggle to outperform logistic regression baselines on definiteness prediction, even though they often correctly identify the underlying reasons. PEDANTIC provides a valuable resource for patent AI researchers, enabling the development of advanced examination models. We will publicly release the dataset and code.
>
---
#### [replaced 115] Tracking Semantic Change in Slovene: A Novel Dataset and Optimal Transport-Based Distance
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2402.16596v2](http://arxiv.org/pdf/2402.16596v2)**

> **作者:** Marko Pranjić; Kaja Dobrovoljc; Senja Pollak; Matej Martinc
>
> **摘要:** In this paper, we focus on the detection of semantic changes in Slovene, a less resourced Slavic language with two million speakers. Detecting and tracking semantic changes provides insight into the evolution of language caused by changes in society and culture. We present the first Slovene dataset for evaluating semantic change detection systems, which contains aggregated semantic change scores for 104 target words obtained from more than 3,000 manually annotated sentence pairs. We analyze an important class of measures of semantic change metrics based on the Average pairwise distance and identify several limitations. To address these limitations, we propose a novel metric based on regularized optimal transport, which offers a more robust framework for quantifying semantic change. We provide a comprehensive evaluation of various existing semantic change detection methods and associated semantic change measures on our dataset. Through empirical testing, we demonstrate that our proposed approach, leveraging regularized optimal transport, achieves either matching or improved performance compared to baseline approaches.
>
---
#### [replaced 116] Bridging Supervised Learning and Reinforcement Learning in Math Reasoning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18116v2](http://arxiv.org/pdf/2505.18116v2)**

> **作者:** Huayu Chen; Kaiwen Zheng; Qinsheng Zhang; Ganqu Cui; Yin Cui; Haotian Ye; Tsung-Yi Lin; Ming-Yu Liu; Jun Zhu; Haoxiang Wang
>
> **摘要:** Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling self-improvement through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) -- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an implicit negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like Rejection sampling Fine-Tuning, matching or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they originate from entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems.
>
---
#### [replaced 117] How to Synthesize Text Data without Model Collapse?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.14689v3](http://arxiv.org/pdf/2412.14689v3)**

> **作者:** Xuekai Zhu; Daixuan Cheng; Hengli Li; Kaiyan Zhang; Ermo Hua; Xingtai Lv; Ning Ding; Zhouhan Lin; Zilong Zheng; Bowen Zhou
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Model collapse in synthetic data indicates that iterative training on self-generated data leads to a gradual decline in performance. With the proliferation of AI models, synthetic data will fundamentally reshape the web data ecosystem. Future GPT-$\{n\}$ models will inevitably be trained on a blend of synthetic and human-produced data. In this paper, we focus on two questions: what is the impact of synthetic data on language model training, and how to synthesize data without model collapse? We first pre-train language models across different proportions of synthetic data, revealing a negative correlation between the proportion of synthetic data and model performance. We further conduct statistical analysis on synthetic data to uncover distributional shift phenomenon and over-concentration of n-gram features. Inspired by the above findings, we propose token editing on human-produced data to obtain semi-synthetic data. As a proof of concept, we theoretically demonstrate that token-level editing can prevent model collapse, as the test error is constrained by a finite upper bound. We conduct extensive experiments on pre-training from scratch, continual pre-training, and supervised fine-tuning. The results validate our theoretical proof that token-level editing improves model performance.
>
---
#### [replaced 118] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.16665v3](http://arxiv.org/pdf/2410.16665v3)**

> **作者:** Jing-Jing Li; Valentina Pyatkin; Max Kleiman-Weiner; Liwei Jiang; Nouha Dziri; Anne G. E. Collins; Jana Schaich Borg; Maarten Sap; Yejin Choi; Sydney Levine
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** The ideal AI safety moderation system would be both structurally interpretable (so its decisions can be reliably explained) and steerable (to align to safety standards and reflect a community's values), which current systems fall short on. To address this gap, we present SafetyAnalyst, a novel AI safety moderation framework. Given an AI behavior, SafetyAnalyst uses chain-of-thought reasoning to analyze its potential consequences by creating a structured "harm-benefit tree," which enumerates harmful and beneficial actions and effects the AI behavior may lead to, along with likelihood, severity, and immediacy labels that describe potential impacts on stakeholders. SafetyAnalyst then aggregates all effects into a harmfulness score using 28 fully interpretable weight parameters, which can be aligned to particular safety preferences. We applied this framework to develop an open-source LLM prompt safety classification system, distilled from 18.5 million harm-benefit features generated by frontier LLMs on 19k prompts. On comprehensive benchmarks, we show that SafetyAnalyst (average F1=0.81) outperforms existing moderation systems (average F1$<$0.72) on prompt safety classification, while offering the additional advantages of interpretability, transparency, and steerability.
>
---
#### [replaced 119] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20767v2](http://arxiv.org/pdf/2505.20767v2)**

> **作者:** Xiaqiang Tang; Jian Li; Keyu Hu; Du Nan; Xiaolong Li; Xi Zhang; Weigao Sun; Sihong Xie
>
> **备注:** ACL 2025
>
> **摘要:** Faithfulness hallucination are claims generated by a Large Language Model (LLM) not supported by contexts provided to the LLM. Lacking assessment standard, existing benchmarks only contain "factual statements" that rephrase source materials without marking "cognitive statements" that make inference from the given context, making the consistency evaluation and optimization of cognitive statements difficult. Inspired by how an evidence is assessed in the legislative domain, we design a rigorous framework to assess different levels of faithfulness of cognitive statements and create a benchmark dataset where we reveal insightful statistics. We design an annotation pipeline to create larger benchmarks for different LLMs automatically, and the resulting larger-scale CogniBench-L dataset can be used to train accurate cognitive hallucination detection model. We release our model and dataset at: https://github.com/FUTUREEEEEE/CogniBench
>
---
#### [replaced 120] A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15585v3](http://arxiv.org/pdf/2504.15585v3)**

> **作者:** Kun Wang; Guibin Zhang; Zhenhong Zhou; Jiahao Wu; Miao Yu; Shiqian Zhao; Chenlong Yin; Jinhu Fu; Yibo Yan; Hanjun Luo; Liang Lin; Zhihao Xu; Haolang Lu; Xinye Cao; Xinyun Zhou; Weifei Jin; Fanci Meng; Junyuan Mao; Yu Wang; Hao Wu; Minghe Wang; Fan Zhang; Junfeng Fang; Wenjie Qu; Yue Liu; Chengwei Liu; Yifan Zhang; Qiankun Li; Chongye Guo; Yalan Qin; Zhaoxin Fan; Yi Ding; Donghai Hong; Jiaming Ji; Yingxin Lai; Zitong Yu; Xinfeng Li; Yifan Jiang; Yanhui Li; Xinyu Deng; Junlin Wu; Dongxia Wang; Yihao Huang; Yufei Guo; Jen-tse Huang; Qiufeng Wang; Wenxuan Wang; Dongrui Liu; Yanwei Yue; Wenke Huang; Guancheng Wan; Heng Chang; Tianlin Li; Yi Yu; Chenghao Li; Jiawei Li; Lei Bai; Jie Zhang; Qing Guo; Jingyi Wang; Tianlong Chen; Joey Tianyi Zhou; Xiaojun Jia; Weisong Sun; Cong Wu; Jing Chen; Xuming Hu; Yiming Li; Xiao Wang; Ningyu Zhang; Luu Anh Tuan; Guowen Xu; Jiaheng Zhang; Tianwei Zhang; Xingjun Ma; Jindong Gu; Xiang Wang; Bo An; Jun Sun; Mohit Bansal; Shirui Pan; Lingjuan Lyu; Yuval Elovici; Bhavya Kailkhura; Yaodong Yang; Hongwei Li; Wenyuan Xu; Yizhou Sun; Wei Wang; Qing Li; Ke Tang; Yu-Gang Jiang; Felix Juefei-Xu; Hui Xiong; Xiaofeng Wang; Dacheng Tao; Philip S. Yu; Qingsong Wen; Yang Liu
>
> **摘要:** The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
>
---
#### [replaced 121] Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21277v2](http://arxiv.org/pdf/2505.21277v2)**

> **作者:** Yao Huang; Yitong Sun; Shouwei Ruan; Yichi Zhang; Yinpeng Dong; Xingxing Wei
>
> **备注:** 19 pages, 20 figures, accepted by ACL 2025, Findings
>
> **摘要:** Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO.
>
---
#### [replaced 122] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; I.2.7; I.2.10; I.4.9**

- **链接: [http://arxiv.org/pdf/2503.07265v2](http://arxiv.org/pdf/2503.07265v2)**

> **作者:** Yuwei Niu; Munan Ning; Mengren Zheng; Weiyang Jin; Bin Lin; Peng Jin; Jiaqi Liao; Chaoran Feng; Kunpeng Ning; Bin Zhu; Li Yuan
>
> **备注:** Code, data and leaderboard: https://github.com/PKU-YuanGroup/WISE
>
> **摘要:** Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose $\textbf{WISE}$, the first benchmark specifically designed for $\textbf{W}$orld Knowledge-$\textbf{I}$nformed $\textbf{S}$emantic $\textbf{E}$valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce $\textbf{WiScore}$, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at https://github.com/PKU-YuanGroup/WISE.
>
---
#### [replaced 123] Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14641v3](http://arxiv.org/pdf/2410.14641v3)**

> **作者:** Runchu Tian; Yanghao Li; Yuepeng Fu; Siyang Deng; Qinyu Luo; Cheng Qian; Shuo Wang; Xin Cong; Zhong Zhang; Yesai Wu; Yankai Lin; Huadong Wang; Xiaojiang Liu
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Positional bias in large language models (LLMs) hinders their ability to effectively process long inputs. A prominent example is the "lost in the middle" phenomenon, where LLMs struggle to utilize relevant information situated in the middle of the input. While prior research primarily focuses on single pieces of relevant information, real-world applications often involve multiple relevant information pieces. To bridge this gap, we present LongPiBench, a benchmark designed to assess positional bias involving multiple pieces of relevant information. Thorough experiments are conducted with five commercial and six open-source models. These experiments reveal that while most current models are robust against the "lost in the middle" issue, there exist significant biases related to the spacing of relevant information pieces. These findings highlight the importance of evaluating and reducing positional biases to advance LLM's capabilities.
>
---
#### [replaced 124] KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15480v2](http://arxiv.org/pdf/2505.15480v2)**

> **作者:** Qihuang Zhong; Liang Ding; Xiantao Cai; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** Accepted to ACL2025 Findings
>
> **摘要:** Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
>
---
#### [replaced 125] Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05242v2](http://arxiv.org/pdf/2502.05242v2)**

> **作者:** Guanxu Chen; Dongrui Liu; Tao Luo; Lijie Hu; Jing Shao
>
> **备注:** 25 pages,6 figures,13 tables
>
> **摘要:** Large language models (LLMs) are becoming increasingly capable, but the mechanisms of their thinking and decision-making process remain unclear. Chain-of-thoughts (CoTs) have been commonly utilized to monitor LLMs, but this strategy fails to accurately reflect LLMs' thinking process. Techniques based on LLMs' hidden representations provide an inner perspective to monitor their latent thinking. However, previous methods only try to develop external monitors instead of making LLMs themselves easier to monitor. In this paper, we propose a novel method TELLME, improving the transparency of LLMs and helping monitors identify unsuitable and sensitive behaviors. Furthermore, we showcase the applications of TELLME on trustworthiness tasks (\eg, safety risks monitoring tasks and detoxification tasks), where LLMs achieve consistent improvement in transparency and task performance. More crucially, we theoretically analyze the improvement of TELLME on LLMs' generalization ability through optimal transport theory.
>
---
#### [replaced 126] Understanding Synthetic Context Extension via Retrieval Heads
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.22316v4](http://arxiv.org/pdf/2410.22316v4)**

> **作者:** Xinyu Zhao; Fangcong Yin; Greg Durrett
>
> **备注:** Published at ICML 2025
>
> **摘要:** Long-context LLMs are increasingly in demand for applications such as retrieval-augmented generation. To defray the cost of pretraining LLMs over long contexts, recent work takes an approach of synthetic context extension: fine-tuning LLMs with synthetically generated long-context data in a post-training stage. However, it remains unclear how and why this synthetic context extension imparts abilities for downstream long-context tasks. In this paper, we investigate fine-tuning on synthetic data for three long-context tasks that require retrieval and reasoning. We vary the realism of "needle" concepts to be retrieved and diversity of the surrounding "haystack" context, from using LLMs to construct synthetic documents to using templated relations and creating symbolic datasets. We find that models trained on synthetic data fall short of the real data, but surprisingly, the mismatch can be interpreted and even predicted in terms of a special set of attention heads that are responsible for retrieval over long context, retrieval heads (Wu et al., 2024). The retrieval heads learned on synthetic data have high overlap with retrieval heads learned on real data, and there is a strong correlation between the recall of heads learned and the downstream performance of a model. Furthermore, with attention knockout and activation patching, we mechanistically show that retrieval heads are necessary and explain model performance, although they are not totally sufficient. Our results shed light on how to interpret synthetic data fine-tuning performance and how to approach creating better data for learning real-world capabilities over long contexts.
>
---
#### [replaced 127] Redundancy Principles for MLLMs Benchmarks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13953v2](http://arxiv.org/pdf/2501.13953v2)**

> **作者:** Zicheng Zhang; Xiangyu Zhao; Xinyu Fang; Chunyi Li; Xiaohong Liu; Xiongkuo Min; Haodong Duan; Kai Chen; Guangtao Zhai
>
> **摘要:** With the rapid iteration of Multi-modality Large Language Models (MLLMs) and the evolving demands of the field, the number of benchmarks produced annually has surged into the hundreds. The rapid growth has inevitably led to significant redundancy among benchmarks. Therefore, it is crucial to take a step back and critically assess the current state of redundancy and propose targeted principles for constructing effective MLLM benchmarks. In this paper, we focus on redundancy from three key perspectives: 1) Redundancy of benchmark capability dimensions, 2) Redundancy in the number of test questions, and 3) Cross-benchmark redundancy within specific domains. Through the comprehensive analysis over hundreds of MLLMs' performance across more than 20 benchmarks, we aim to quantitatively measure the level of redundancy lies in existing MLLM evaluations, provide valuable insights to guide the future development of MLLM benchmarks, and offer strategies to refine and address redundancy issues effectively. The code is available at https://github.com/zzc-1998/Benchmark-Redundancy.
>
---
#### [replaced 128] Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.09946v2](http://arxiv.org/pdf/2408.09946v2)**

> **作者:** Byungjun Kim; Dayeon Seo; Bugeun Kim
>
> **备注:** Under review, Modified title and content
>
> **摘要:** Recent studies have investigated whether large language models (LLMs) can support obscure communication that requires specialized skills, such as inferring subtext or doublespeak. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two issues with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these issues, we propose a macroscopic and systematic approach to the investigation. Specifically, we introduce seven fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
>
---
#### [replaced 129] Dissecting the Ullman Variations with a SCALPEL: Why do LLMs fail at Trivial Alterations to the False Belief Task?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14737v2](http://arxiv.org/pdf/2406.14737v2)**

> **作者:** Zhiqiang Pi; Annapurna Vadaparty; Benjamin K. Bergen; Cameron R. Jones
>
> **摘要:** Recent empirical results have sparked a debate about whether or not Large Language Models (LLMs) are capable of Theory of Mind (ToM). While some have found LLMs to be successful on ToM evaluations such as the False Belief task, others have shown that their performance is not robust against trivial alterations to stimuli. In this paper, we introduce SCALPEL -- a technique to incrementally modify stimuli to test different specific hypotheses about why LLMs fail -- and apply this method to the "transparent-access" modification of the unexpected contents task. Our results suggest that LLMs often do poorly because they fail to make essential common-sense inferences, such as that seeing a transparent container implies recognizing its contents. We conclude that while modern LLMs go beyond mere pattern matching, they still fall short of robust human-like ToM. We argue that SCALPEL can help cognitive scientists examine LLMs' capabilities in finer detail and provide insight into alternative mechanisms by which tasks that are used to assess human cognition might be completed.
>
---
#### [replaced 130] Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21072v2](http://arxiv.org/pdf/2505.21072v2)**

> **作者:** Ekaterina Fadeeva; Aleksandr Rubashevskii; Roman Vashurin; Shehzaad Dhuliawala; Artem Shelmanov; Timothy Baldwin; Preslav Nakov; Mrinmaya Sachan; Maxim Panov
>
> **摘要:** Large Language Models (LLMs) enhanced with external knowledge retrieval, an approach known as Retrieval-Augmented Generation (RAG), have shown strong performance in open-domain question answering. However, RAG systems remain susceptible to hallucinations: factually incorrect outputs that may arise either from inconsistencies in the model's internal knowledge or incorrect use of the retrieved context. Existing approaches often conflate factuality with faithfulness to the retrieved context, misclassifying factually correct statements as hallucinations if they are not directly supported by the retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval Augmented UNcertainty Quantification), a novel method for hallucination detection in RAG outputs. FRANQ applies different Uncertainty Quantification (UQ) techniques to estimate factuality based on whether a statement is faithful to the retrieved context or not. To evaluate FRANQ and other UQ techniques for RAG, we present a new long-form Question Answering (QA) dataset annotated for both factuality and faithfulness, combining automated labeling with manual validation of challenging examples. Extensive experiments on long- and short-form QA across multiple datasets and LLMs show that FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing methods.
>
---
#### [replaced 131] Machine Translation Models are Zero-Shot Detectors of Translation Direction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.06769v4](http://arxiv.org/pdf/2401.06769v4)**

> **作者:** Michelle Wastl; Jannis Vamvas; Rico Sennrich
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Detecting the translation direction of parallel text has applications for machine translation training and evaluation, but also has forensic applications such as resolving plagiarism or forgery allegations. In this work, we explore an unsupervised approach to translation direction detection based on the simple hypothesis that $p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$, motivated by the well-known simplification effect in translationese or machine-translationese. In experiments with massively multilingual machine translation models across 20 translation directions, we confirm the effectiveness of the approach for high-resource language pairs, achieving document-level accuracies of 82--96% for NMT-produced translations, and 60--81% for human translations, depending on the model used. Code and demo are available at https://github.com/ZurichNLP/translation-direction-detection
>
---
#### [replaced 132] Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21315v2](http://arxiv.org/pdf/2505.21315v2)**

> **作者:** Jesujoba O. Alabi; Michael A. Hedderich; David Ifeoluwa Adelani; Dietrich Klakow
>
> **备注:** Working paper
>
> **摘要:** With over 2,000 languages and potentially millions of speakers, Africa represents one of the richest linguistic regions in the world. Yet, this diversity is scarcely reflected in state-of-the-art natural language processing (NLP) systems and large language models (LLMs), which predominantly support a narrow set of high-resource languages. This exclusion not only limits the reach and utility of modern NLP technologies but also risks widening the digital divide across linguistic communities. Nevertheless, NLP research on African languages is active and growing. In recent years, there has been a surge of interest in this area, driven by several factors-including the creation of multilingual language resources, the rise of community-led initiatives, and increased support through funding programs. In this survey, we analyze 734 research papers on NLP for African languages published over the past five years, offering a comprehensive overview of recent progress across core tasks. We identify key trends shaping the field and conclude by outlining promising directions to foster more inclusive and sustainable NLP research for African languages.
>
---
#### [replaced 133] Which Demographics do LLMs Default to During Annotation?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08820v3](http://arxiv.org/pdf/2410.08820v3)**

> **作者:** Johannes Schäfer; Aidan Combs; Christopher Bagdon; Jiahui Li; Nadine Probol; Lynn Greschner; Sean Papay; Yarik Menchaca Resendiz; Aswathy Velutharambath; Amelie Wührl; Sabine Weber; Roman Klinger
>
> **备注:** ACL 2025
>
> **摘要:** Demographics and cultural background of annotators influence the labels they assign in text annotation -- for instance, an elderly woman might find it offensive to read a message addressed to a "bro", but a male teenager might find it appropriate. It is therefore important to acknowledge label variations to not under-represent members of a society. Two research directions developed out of this observation in the context of using large language models (LLM) for data annotations, namely (1) studying biases and inherent knowledge of LLMs and (2) injecting diversity in the output by manipulating the prompt with demographic information. We combine these two strands of research and ask the question to which demographics an LLM resorts to when no demographics is given. To answer this question, we evaluate which attributes of human annotators LLMs inherently mimic. Furthermore, we compare non-demographic conditioned prompts and placebo-conditioned prompts (e.g., "you are an annotator who lives in house number 5") to demographics-conditioned prompts ("You are a 45 year old man and an expert on politeness annotation. How do you rate {instance}"). We study these questions for politeness and offensiveness annotations on the POPQUORN data set, a corpus created in a controlled manner to investigate human label variations based on demographics which has not been used for LLM-based analyses so far. We observe notable influences related to gender, race, and age in demographic prompting, which contrasts with previous studies that found no such effects.
>
---
#### [replaced 134] Faster and Better LLMs via Latency-Aware Test-Time Scaling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19634v3](http://arxiv.org/pdf/2505.19634v3)**

> **作者:** Zili Wang; Tianyu Zhang; Lei Zhu; Haoli Bai; Lu Hou; Shiming Xiang; Xianzhi Yu; Wulong Liu
>
> **摘要:** Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
>
---

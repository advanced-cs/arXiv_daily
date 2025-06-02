# 自然语言处理 cs.CL

- **最新发布 179 篇**

- **更新 155 篇**

## 最新发布

#### [new 001] Multi-objective Large Language Model Alignment with Hierarchical Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属LLM多目标对齐任务，解决现有方法难以平衡多目标且需高成本重训的问题。提出HoE方法，通过分层专家（LoRA、路由、偏好路由）实现轻量化、零训练的参数高效方案，覆盖Pareto前沿并优于15个基线。**

- **链接: [http://arxiv.org/pdf/2505.20925v1](http://arxiv.org/pdf/2505.20925v1)**

> **作者:** Zhuo Li; Guodong Du; Weiyang Guo; Yigeng Zhou; Xiucheng Li; Wenya Wang; Fangming Liu; Yequan Wang; Deheng Ye; Min Zhang; Jing Li
>
> **摘要:** Aligning large language models (LLMs) to simultaneously satisfy multiple objectives remains a significant challenge, especially given the diverse and often conflicting nature of human preferences. Existing alignment methods struggle to balance trade-offs effectively, often requiring costly retraining or yielding suboptimal results across the Pareto frontier of preferences. In this paper, we introduce \textit{HoE}(Hierarchical Mixture-of-Experts), a \textit{lightweight}, \textit{parameter-efficient}, and \textit{plug-and-play} approach that eliminates the need for model training, while enabling LLMs to adapt across the entire Pareto frontier and accommodate diverse user preferences. In particular, \textit{HoE} consists of three hierarchical components: LoRA Experts, Router Experts and Preference Routing, reaching optimal Pareto frontiers and achieving a trade-off between parameter size, training cost, and performance. We evaluate \textit{HoE} across various tasks on 14 objectives and 200 different preferences among 6 benchmarks, demonstrating superior performance over 15 recent baselines. Code is available in the supplementary materials.
>
---
#### [new 002] Evaluating and Steering Modality Preferences in Multimodal Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于多模态大模型研究任务，旨在解决模型在处理冲突多模态信息时是否存在模态偏好及如何控制的问题。工作包括构建MC²基准测试揭示18个模型的模态偏见，提出基于表征工程的无微调引导方法，实现偏好调控并提升下游任务效果。**

- **链接: [http://arxiv.org/pdf/2505.20977v1](http://arxiv.org/pdf/2505.20977v1)**

> **作者:** Yu Zhang; Jinlong Ma; Yongshuai Hou; Xuefeng Bai; Kehai Chen; Yang Xiang; Jun Yu; Min Zhang
>
> **备注:** Modality Preference
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable performance on complex tasks with multimodal context. However, it is still understudied whether they exhibit modality preference when processing multimodal contexts. To study this question, we first build a \textbf{MC\textsuperscript{2}} benchmark under controlled evidence conflict scenarios to systematically evaluate modality preference, which is the tendency to favor one modality over another when making decisions based on multimodal conflicting evidence. Our extensive evaluation reveals that all 18 tested MLLMs generally demonstrate clear modality bias, and modality preference can be influenced by external interventions. An in-depth analysis reveals that the preference direction can be captured within the latent representations of MLLMs. Built on this, we propose a probing and steering method based on representation engineering to explicitly control modality preference without additional fine-tuning or carefully crafted prompts. Our method effectively amplifies modality preference toward a desired direction and applies to downstream tasks such as hallucination mitigation and multimodal machine translation, yielding promising improvements.
>
---
#### [new 003] Words Like Knives: Backstory-Personalized Modeling and Detection of Violent Communication
- **分类: cs.CL**

- **简介: 该论文属于个性化暴力沟通检测任务，解决现有NLP忽视人际关系背景对冲突感知影响的问题。构建PersonaConflicts Corpus数据集（5772对话），通过非暴力沟通理论评估LLMs，发现模型难以利用关系背景信息且高估信息正面效果，强调个性化对沟通调解的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.21451v1](http://arxiv.org/pdf/2505.21451v1)**

> **作者:** Jocelyn Shen; Akhila Yerukola; Xuhui Zhou; Cynthia Breazeal; Maarten Sap; Hae Won Park
>
> **摘要:** Conversational breakdowns in close relationships are deeply shaped by personal histories and emotional context, yet most NLP research treats conflict detection as a general task, overlooking the relational dynamics that influence how messages are perceived. In this work, we leverage nonviolent communication (NVC) theory to evaluate LLMs in detecting conversational breakdowns and assessing how relationship backstory influences both human and model perception of conflicts. Given the sensitivity and scarcity of real-world datasets featuring conflict between familiar social partners with rich personal backstories, we contribute the PersonaConflicts Corpus, a dataset of N=5,772 naturalistic simulated dialogues spanning diverse conflict scenarios between friends, family members, and romantic partners. Through a controlled human study, we annotate a subset of dialogues and obtain fine-grained labels of communication breakdown types on individual turns, and assess the impact of backstory on human and model perception of conflict in conversation. We find that the polarity of relationship backstories significantly shifted human perception of communication breakdowns and impressions of the social partners, yet models struggle to meaningfully leverage those backstories in the detection task. Additionally, we find that models consistently overestimate how positively a message will make a listener feel. Our findings underscore the critical role of personalization to relationship contexts in enabling LLMs to serve as effective mediators in human communication for authentic connection.
>
---
#### [new 004] Analyzing values about gendered language reform in LLMs' revisions
- **分类: cs.CL**

- **简介: 该论文研究LLMs在文本修订中对性别化职业名词（如"outdoorswoman"）的修改行为及理由，评估其是否符合女权主义和跨性别包容语言改革标准，并测试其是否像人类一样受语境影响，探讨AI价值对齐问题。**

- **链接: [http://arxiv.org/pdf/2505.21378v1](http://arxiv.org/pdf/2505.21378v1)**

> **作者:** Jules Watson; Xi Wang; Raymond Liu; Suzanne Stevenson; Barend Beekhuizen
>
> **备注:** 15 pages
>
> **摘要:** Within the common LLM use case of text revision, we study LLMs' revision of gendered role nouns (e.g., outdoorsperson/woman/man) and their justifications of such revisions. We evaluate their alignment with feminist and trans-inclusive language reforms for English. Drawing on insight from sociolinguistics, we further assess if LLMs are sensitive to the same contextual effects in the application of such reforms as people are, finding broad evidence of such effects. We discuss implications for value alignment.
>
---
#### [new 005] Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making
- **分类: cs.CL; cs.AI; cs.LG; q-bio.OT**

- **简介: 该论文聚焦临床决策支持任务，旨在解决多智能体LLM中的"沉默共识"问题（即智能体过早达成一致而缺乏批判分析）。提出"鲶鱼智能体"通过结构化质疑（基于案例复杂度和语气校准的干预机制）打破共识，促进深度推理。实验显示其超越现有单/多智能体模型。**

- **链接: [http://arxiv.org/pdf/2505.21503v1](http://arxiv.org/pdf/2505.21503v1)**

> **作者:** Yihan Wang; Qiao Yan; Zhenghao Xing; Lihao Liu; Junjun He; Chi-Wing Fu; Xiaowei Hu; Pheng-Ann Heng
>
> **摘要:** Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1.
>
---
#### [new 006] Guided by Gut: Efficient Test-Time Scaling with Reinforced Intrinsic Confidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Guided by Gut（GG），一种高效的自引导测试时缩放框架，解决传统TTS方法依赖外部模型或采样导致的高计算成本问题。通过轻量级树搜索结合内在置信度与步骤新颖性信号，并经强化学习优化，GG使小模型（1.5B参数）性能匹敌大模型（32B-70B），且推理速度快8倍，内存减少4-5倍，KV缓存降50%。**

- **链接: [http://arxiv.org/pdf/2505.20325v1](http://arxiv.org/pdf/2505.20325v1)**

> **作者:** Amirhosein Ghasemabadi; Keith G. Mills; Baochun Li; Di Niu
>
> **摘要:** Test-Time Scaling (TTS) methods for enhancing Large Language Model (LLM) reasoning often incur substantial computational costs, primarily due to extensive reliance on external Process Reward Models (PRMs) or sampling methods like Best-of-N (BoN). This paper introduces Guided by Gut (GG), an efficient self-guided TTS framework that achieves PRM-level performance without costly external verifier models. Our method employs a lightweight tree search guided solely by intrinsic LLM signals, token-level confidence and step novelty. One critical innovation is improving the reliability of internal confidence estimates via a targeted reinforcement learning fine-tuning phase. Empirical evaluations on challenging mathematical reasoning benchmarks demonstrate that GG enables smaller models (e.g., 1.5B parameters) to achieve accuracy matching or surpassing significantly larger models (e.g., 32B-70B parameters), while reducing GPU memory usage by up to 10x. Compared to PRM-based methods, GG achieves comparable accuracy with 8x faster inference speeds and 4-5x lower memory usage. Additionally, GG reduces KV cache memory usage by approximately 50% compared to the BoN strategy, facilitating more efficient and practical deployment of TTS techniques.
>
---
#### [new 007] POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization
- **分类: cs.CL**

- **简介: 该论文提出多语言多文化在线极化基准POLAR，包含7种语言2.3万实例，标注极化存在、类型及表现形式。通过实验评估模型在单语/跨语言及少/零样本场景表现，发现模型在检测极化类型和表现时效果显著下降，揭示极化复杂性及需改进NLP方法。属于极化分析任务，旨在解决现有研究局限性，构建跨文化数据集并验证模型不足。**

- **链接: [http://arxiv.org/pdf/2505.20624v1](http://arxiv.org/pdf/2505.20624v1)**

> **作者:** Usman Naseem; Juan Ren; Saba Anwar; Sarah Kohail; Rudy Alexandro Garrido Veliz; Robert Geislinger; Aisha Jabr; Idris Abdulmumin; Laiba Qureshi; Aarushi Ajay Borkar; Maryam Ibrahim Mukhtar; Abinew Ali Ayele; Ibrahim Said Ahmad; Adem Ali; Martin Semmann; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam
>
> **备注:** Preprint
>
> **摘要:** Online polarization poses a growing challenge for democratic discourse, yet most computational social science research remains monolingual, culturally narrow, or event-specific. We introduce POLAR, a multilingual, multicultural, and multievent dataset with over 23k instances in seven languages from diverse online platforms and real-world events. Polarization is annotated along three axes: presence, type, and manifestation, using a variety of annotation platforms adapted to each cultural context. We conduct two main experiments: (1) we fine-tune six multilingual pretrained language models in both monolingual and cross-lingual setups; and (2) we evaluate a range of open and closed large language models (LLMs) in few-shot and zero-shot scenarios. Results show that while most models perform well on binary polarization detection, they achieve substantially lower scores when predicting polarization types and manifestations. These findings highlight the complex, highly contextual nature of polarization and the need for robust, adaptable approaches in NLP and computational social science. All resources will be released to support further research and effective mitigation of digital polarization globally.
>
---
#### [new 008] REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文提出REAL-Prover，旨在提升自动定理证明器在大学数学中的适用性。针对现有方法难以处理高级数学的问题，团队开发了基于大模型与检索系统的Lean 4证明器，结合数据转换管道HERALD-AF和交互环境Jixia-interactive，实现在ProofNet达23.7%及新基准FATE-M达56.7%的SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.20613v1](http://arxiv.org/pdf/2505.20613v1)**

> **作者:** Ziju Shen; Naohao Huang; Fanyi Yang; Yutong Wang; Guoxiong Gao; Tianyi Xu; Jiedong Jiang; Wanyi He; Pu Yang; Mengzhou Sun; Haocheng Ju; Peihao Wu; Bryan Dai; Bin Dong
>
> **摘要:** Nowadays, formal theorem provers have made monumental progress on high-school and competition-level mathematics, but few of them generalize to more advanced mathematics. In this paper, we present REAL-Prover, a new open-source stepwise theorem prover for Lean 4 to push this boundary. This prover, based on our fine-tuned large language model (REAL-Prover-v1) and integrated with a retrieval system (Leansearch-PS), notably boosts performance on solving college-level mathematics problems. To train REAL-Prover-v1, we developed HERALD-AF, a data extraction pipeline that converts natural language math problems into formal statements, and a new open-source Lean 4 interactive environment (Jixia-interactive) to facilitate synthesis data collection. In our experiments, our prover using only supervised fine-tune achieves competitive results with a 23.7% success rate (Pass@64) on the ProofNet dataset-comparable to state-of-the-art (SOTA) models. To further evaluate our approach, we introduce FATE-M, a new benchmark focused on algebraic problems, where our prover achieves a SOTA success rate of 56.7% (Pass@64).
>
---
#### [new 009] Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity
- **分类: cs.CL**

- **简介: 该论文提出MoGE方法优化MoE模型的专家负载不均问题，通过分组约束平衡计算资源，构建72B参数的Pangu Pro MoE模型，在Ascend芯片上实现高效推理（1528 token/s/卡），优于同规模密集模型，提升硬件利用率。**

- **链接: [http://arxiv.org/pdf/2505.21411v1](http://arxiv.org/pdf/2505.21411v1)**

> **作者:** Yehui Tang; Xiaosong Li; Fangcheng Liu; Wei Guo; Hang Zhou; Yaoyuan Wang; Kai Han; Xianzhi Yu; Jinpeng Li; Hui Zang; Fei Mi; Xiaojun Meng; Zhicheng Liu; Hanting Chen; Binfan Zheng; Can Chen; Youliang Yan; Ruiming Tang; Peifeng Qin; Xinghao Chen; Dacheng Tao; Yunhe Wang
>
> **摘要:** The surgence of Mixture of Experts (MoE) in Large Language Models promises a small price of execution cost for a much larger model parameter count and learning capacity, because only a small fraction of parameters are activated for each input token. However, it is commonly observed that some experts are activated far more often than others, leading to system inefficiency when running the experts on different devices in parallel. Therefore, we introduce Mixture of Grouped Experts (MoGE), which groups the experts during selection and balances the expert workload better than MoE in nature. It constrains tokens to activate an equal number of experts within each predefined expert group. When a model execution is distributed on multiple devices, this architectural design ensures a balanced computational load across devices, significantly enhancing throughput, particularly for the inference phase. Further, we build Pangu Pro MoE on Ascend NPUs, a sparse model based on MoGE with 72 billion total parameters, 16 billion of which are activated for each token. The configuration of Pangu Pro MoE is optimized for Ascend 300I Duo and 800I A2 through extensive system simulation studies. Our experiments indicate that MoGE indeed leads to better expert load balancing and more efficient execution for both model training and inference on Ascend NPUs. The inference performance of Pangu Pro MoE achieves 1148 tokens/s per card and can be further improved to 1528 tokens/s per card by speculative acceleration, outperforming comparable 32B and 72B Dense models. Furthermore, we achieve an excellent cost-to-performance ratio for model inference on Ascend 300I Duo.Our studies show that Ascend NPUs are capable of training Pangu Pro MoE with massive parallelization to make it a leading model within the sub-100B total parameter class, outperforming prominent open-source models like GLM-Z1-32B and Qwen3-32B.
>
---
#### [new 010] Can LLMs Learn to Map the World from Local Descriptions?
- **分类: cs.CL**

- **简介: 该论文研究LLMs通过局部描述构建全局空间认知的能力。任务评估模型在空间感知（推断全局布局）和导航（路径规划）中的表现。实验在模拟城市环境验证LLMs可泛化空间关系、学习道路连接并规划最优路径，展现与现实一致的空间表征。**

- **链接: [http://arxiv.org/pdf/2505.20874v1](http://arxiv.org/pdf/2505.20874v1)**

> **作者:** Sirui Xia; Aili Chen; Xintao Wang; Tinghui Zhu; Yikai Zhang; Jiangjie Chen; Yanghua Xiao
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in tasks such as code and mathematics. However, their potential to internalize structured spatial knowledge remains underexplored. This study investigates whether LLMs, grounded in locally relative human observations, can construct coherent global spatial cognition by integrating fragmented relational descriptions. We focus on two core aspects of spatial cognition: spatial perception, where models infer consistent global layouts from local positional relationships, and spatial navigation, where models learn road connectivity from trajectory data and plan optimal paths between unconnected locations. Experiments conducted in a simulated urban environment demonstrate that LLMs not only generalize to unseen spatial relationships between points of interest (POIs) but also exhibit latent representations aligned with real-world spatial distributions. Furthermore, LLMs can learn road connectivity from trajectory descriptions, enabling accurate path planning and dynamic spatial awareness during navigation.
>
---
#### [new 011] FCKT: Fine-Grained Cross-Task Knowledge Transfer with Semantic Contrastive Learning for Targeted Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对目标情感分析任务，解决现有方法因粗粒度知识转移导致的方面-情感关系处理不足及负面迁移问题。提出FCKT框架，通过语义对比学习实现细粒度跨任务知识转移，提升模型性能，实验验证有效。**

- **链接: [http://arxiv.org/pdf/2505.21040v1](http://arxiv.org/pdf/2505.21040v1)**

> **作者:** Wei Chen; Zhao Zhang; Meng Yuan; Kepeng Xu; Fuzhen Zhuang
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** In this paper, we address the task of targeted sentiment analysis (TSA), which involves two sub-tasks, i.e., identifying specific aspects from reviews and determining their corresponding sentiments. Aspect extraction forms the foundation for sentiment prediction, highlighting the critical dependency between these two tasks for effective cross-task knowledge transfer. While most existing studies adopt a multi-task learning paradigm to align task-specific features in the latent space, they predominantly rely on coarse-grained knowledge transfer. Such approaches lack fine-grained control over aspect-sentiment relationships, often assuming uniform sentiment polarity within related aspects. This oversimplification neglects contextual cues that differentiate sentiments, leading to negative transfer. To overcome these limitations, we propose FCKT, a fine-grained cross-task knowledge transfer framework tailored for TSA. By explicitly incorporating aspect-level information into sentiment prediction, FCKT achieves fine-grained knowledge transfer, effectively mitigating negative transfer and enhancing task performance. Experiments on three datasets, including comparisons with various baselines and large language models (LLMs), demonstrate the effectiveness of FCKT. The source code is available on https://github.com/cwei01/FCKT.
>
---
#### [new 012] Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA
- **分类: cs.CL**

- **简介: 该论文提出多语言evergreen问题分类任务，解决大语言模型在问答中因问题时效性引发的幻觉问题。构建首个带evergreen标签的跨语言数据集EverGreenQA，评估12种模型对问题时效性的显式/隐式处理能力，训练轻量分类器EG-E5达SOTA，并展示其在提升QA可靠性、数据过滤和模型解释等场景的应用。**

- **链接: [http://arxiv.org/pdf/2505.21115v1](http://arxiv.org/pdf/2505.21115v1)**

> **作者:** Sergey Pletenev; Maria Marina; Nikolay Ivanov; Daria Galimzianova; Nikita Krayko; Mikhail Salnikov; Vasily Konovalov; Alexander Panchenko; Viktor Moskvoretskii
>
> **摘要:** Large Language Models (LLMs) often hallucinate in question answering (QA) tasks. A key yet underexplored factor contributing to this is the temporality of questions -- whether they are evergreen (answers remain stable over time) or mutable (answers change). In this work, we introduce EverGreenQA, the first multilingual QA dataset with evergreen labels, supporting both evaluation and training. Using EverGreenQA, we benchmark 12 modern LLMs to assess whether they encode question temporality explicitly (via verbalized judgments) or implicitly (via uncertainty signals). We also train EG-E5, a lightweight multilingual classifier that achieves SoTA performance on this task. Finally, we demonstrate the practical utility of evergreen classification across three applications: improving self-knowledge estimation, filtering QA datasets, and explaining GPT-4o retrieval behavior.
>
---
#### [new 013] Inceptive Transformers: Enhancing Contextual Representations through Multi-Scale Feature Learning Across Domains and Languages
- **分类: cs.CL**

- **简介: 该论文提出Inceptive Transformer，改进传统Transformer通过单[CLS] token压缩全局信息导致的局部信息丢失问题。通过多尺度特征模块动态加权token，平衡局部与全局依赖，提升跨语言/领域任务（如情绪识别、疾病检测等）表现，实验显示比基线模型提升1%-14%，兼具高效性。**

- **链接: [http://arxiv.org/pdf/2505.20496v1](http://arxiv.org/pdf/2505.20496v1)**

> **作者:** Asif Shahriar; Rifat Shahriyar; M Saifur Rahman
>
> **摘要:** Conventional transformer models typically compress the information from all tokens in a sequence into a single \texttt{[CLS]} token to represent global context-- an approach that can lead to information loss in tasks requiring localized or hierarchical cues. In this work, we introduce \textit{Inceptive Transformer}, a modular and lightweight architecture that enriches transformer-based token representations by integrating a multi-scale feature extraction module inspired by inception networks. Our model is designed to balance local and global dependencies by dynamically weighting tokens based on their relevance to a particular task. Evaluation across a diverse range of tasks including emotion recognition (both English and Bangla), irony detection, disease identification, and anti-COVID vaccine tweets classification shows that our models consistently outperform the baselines by 1\% to 14\% while maintaining efficiency. These findings highlight the versatility and cross-lingual applicability of our method for enriching transformer-based representations across diverse domains.
>
---
#### [new 014] PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文构建PMOA-TTS数据集，解决临床领域大规模时间标注资源匮乏问题。通过LLM处理12.4万PMOA病例报告，提取560万带时间戳事件，验证时间线质量并用于生存预测（C-index 0.82），数据公开可用。**

- **链接: [http://arxiv.org/pdf/2505.20323v1](http://arxiv.org/pdf/2505.20323v1)**

> **作者:** Shahriar Noroozizadeh; Sayantan Kumar; George H. Chen; Jeremy C. Weiss
>
> **摘要:** Understanding temporal dynamics in clinical narratives is essential for modeling patient trajectories, yet large-scale temporally annotated resources remain limited. We present PMOA-TTS, the first openly available dataset of 124,699 PubMed Open Access (PMOA) case reports, each converted into structured (event, time) timelines via a scalable LLM-based pipeline. Our approach combines heuristic filtering with Llama 3.3 to identify single-patient case reports, followed by prompt-driven extraction using Llama 3.3 and DeepSeek R1, resulting in over 5.6 million timestamped clinical events. To assess timeline quality, we evaluate against a clinician-curated reference set using three metrics: (i) event-level matching (80% match at a cosine similarity threshold of 0.1), (ii) temporal concordance (c-index > 0.90), and (iii) Area Under the Log-Time CDF (AULTC) for timestamp alignment. Corpus-level analysis shows wide diagnostic and demographic coverage. In a downstream survival prediction task, embeddings from extracted timelines achieve time-dependent concordance indices up to 0.82 $\pm$ 0.01, demonstrating the predictive value of temporally structured narratives. PMOA-TTS provides a scalable foundation for timeline extraction, temporal reasoning, and longitudinal modeling in biomedical NLP. The dataset is available at: https://huggingface.co/datasets/snoroozi/pmoa-tts .
>
---
#### [new 015] RefTool: Enhancing Model Reasoning with Reference-Guided Tool Creation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RefTool框架，通过参考教材等结构化材料指导工具自动生成，解决大模型在无预定义工具时超出知识范围的推理问题。其包含工具创建（生成、验证、分层组织）和使用模块，实验显示其在多领域推理任务中提升11.3%准确率，兼具高效与通用性。**

- **链接: [http://arxiv.org/pdf/2505.21413v1](http://arxiv.org/pdf/2505.21413v1)**

> **作者:** Xiao Liu; Da Yin; Zirui Wu; Yansong Feng
>
> **备注:** Code is available at https://github.com/xxxiaol/RefTool
>
> **摘要:** Tools enhance the reasoning capabilities of large language models (LLMs) in complex problem-solving tasks, but not all tasks have available tools. In the absence of predefined tools, prior works have explored instructing LLMs to generate tools on their own. However, such approaches rely heavily on the models' internal knowledge and would fail in domains beyond the LLMs' knowledge scope. To address this limitation, we propose RefTool, a reference-guided framework for automatic tool creation that leverages structured external materials such as textbooks. RefTool consists of two modules: (1) tool creation, where LLMs generate executable tools from reference content, validate them using illustrative examples, and organize them hierarchically into a toolbox; and (2) tool utilization, where LLMs navigate the toolbox structure to select and apply the appropriate tools to solve problems. Experiments on causality, physics, and chemistry benchmarks demonstrate that RefTool outperforms existing tool-creation and domain-specific reasoning methods by 11.3% on average accuracy, while being cost-efficient and broadly generalizable. Analyses reveal that grounding tool creation in references produces accurate and faithful tools, and that the hierarchical structure facilitates effective tool selection. RefTool enables LLMs to overcome knowledge limitations, demonstrating the value of grounding tool creation in external references for enhanced and generalizable reasoning.
>
---
#### [new 016] Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing
- **分类: cs.CL**

- **简介: 该论文针对跨领域 constituency parsing 树库不足的问题，提出LLM反向生成方法生成跨领域树库，并结合跨度级对比学习预训练策略优化模型，实验在MCTB五领域达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.20976v1](http://arxiv.org/pdf/2505.20976v1)**

> **作者:** Peiming Guo; Meishan Zhang; Jianling Li; Min Zhang; Yue Zhang
>
> **备注:** Accepted by ACL 2025 main conference
>
> **摘要:** Cross-domain constituency parsing is still an unsolved challenge in computational linguistics since the available multi-domain constituency treebank is limited. We investigate automatic treebank generation by large language models (LLMs) in this paper. The performance of LLMs on constituency parsing is poor, therefore we propose a novel treebank generation method, LLM back generation, which is similar to the reverse process of constituency parsing. LLM back generation takes the incomplete cross-domain constituency tree with only domain keyword leaf nodes as input and fills the missing words to generate the cross-domain constituency treebank. Besides, we also introduce a span-level contrastive learning pre-training strategy to make full use of the LLM back generation treebank for cross-domain constituency parsing. We verify the effectiveness of our LLM back generation treebank coupled with contrastive learning pre-training on five target domains of MCTB. Experimental results show that our approach achieves state-of-the-art performance on average results compared with various baselines.
>
---
#### [new 017] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Trans-EnV框架，评估LLMs在非标准英语变体中的语言稳健性，解决其因依赖标准美式英语导致的公平性问题。通过结合语言学规则与LLM转换，将6个基准数据集扩展为38种英语变体，测试7个模型并发现最大46.3%的性能下降，强调跨多样英语变体评估的必要性。**

- **链接: [http://arxiv.org/pdf/2505.20875v1](http://arxiv.org/pdf/2505.20875v1)**

> **作者:** Jiyoung Lee; Seungho Kim; Jieun Han; Jun-Min Lee; Kitaek Kim; Alice Oh; Edward Choi
>
> **备注:** 27 pages, 6 figures, 16 tables
>
> **摘要:** Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our \href{https://github.com/jiyounglee-0523/TransEnV}{code} and \href{https://huggingface.co/collections/jiyounglee0523/transenv-681eadb3c0c8cf363b363fb1}{datasets} are publicly available.
>
---
#### [new 018] Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对查询自动补全任务，解决用户长短期兴趣分层建模与生成有毒内容问题。提出LaD模型，分层捕捉用户长期与短期兴趣，并通过Reject Preference Optimization实现自适应去毒生成，提升结果相关性与安全性，已部署于快手搜索系统。**

- **链接: [http://arxiv.org/pdf/2505.20966v1](http://arxiv.org/pdf/2505.20966v1)**

> **作者:** Zhibo Wang; Xiaoze Jiang; Zhiheng Qin; Enyun Yu; Han Li
>
> **备注:** KDD 2025
>
> **摘要:** Query auto-completion (QAC) plays a crucial role in modern search systems. However, in real-world applications, there are two pressing challenges that still need to be addressed. First, there is a need for hierarchical personalized representations for users. Previous approaches have typically used users' search behavior as a single, overall representation, which proves inadequate in more nuanced generative scenarios. Additionally, query prefixes are typically short and may contain typos or sensitive information, increasing the likelihood of generating toxic content compared to traditional text generation tasks. Such toxic content can degrade user experience and lead to public relations issues. Therefore, the second critical challenge is detoxifying QAC systems. To address these two limitations, we propose a novel model (LaD) that captures personalized information from both long-term and short-term interests, incorporating adaptive detoxification. In LaD, personalized information is captured hierarchically at both coarse-grained and fine-grained levels. This approach preserves as much personalized information as possible while enabling online generation within time constraints. To move a futher step, we propose an online training method based on Reject Preference Optimization (RPO). By incorporating a special token [Reject] during both the training and inference processes, the model achieves adaptive detoxification. Consequently, the generated text presented to users is both non-toxic and relevant to the given prefix. We conduct comprehensive experiments on industrial-scale datasets and perform online A/B tests, delivering the largest single-experiment metric improvement in nearly two years of our product. Our model has been deployed on Kuaishou search, driving the primary traffic for hundreds of millions of active users. The code is available at https://github.com/JXZe/LaD.
>
---
#### [new 019] Dynamic Manifold Evolution Theory: Modeling and Stability Analysis of Latent Representations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型生成机制分析任务，旨在建模潜在表示动态并平衡生成文本的创造性和一致性。提出动态流形演化理论（DMET），将生成过程建模为低维语义流形上的动力系统，映射Transformer组件的动力学机制，通过稳定性理论定义3个评估指标，实验验证理论并提供生成指导。**

- **链接: [http://arxiv.org/pdf/2505.20340v1](http://arxiv.org/pdf/2505.20340v1)**

> **作者:** Yukun Zhang; Qi Dong
>
> **摘要:** We introduce Dynamic Manifold Evolution Theory (DMET),a unified framework that models large language model generation as a controlled dynamical system evolving on a low_dimensional semantic manifold. By casting latent_state updates as discrete time Euler approximations of continuous dynamics, we map intrinsic energy_driven flows and context_dependent forces onto Transformer components (residual connections, attention, feed-forward networks). Leveraging Lyapunov stability theory We define three empirical metrics (state continuity, clustering quality, topological persistence) that quantitatively link latent_trajectory properties to text fluency, grammaticality, and semantic coherence. Extensive experiments across decoding parameters validate DMET's predictions and yield principled guidelines for balancing creativity and consistency in text generation.
>
---
#### [new 020] MOSLIM:Align with diverse preferences in prompts through reward classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MOSLIM方法，属于大语言模型多目标对齐任务。解决现有方法需多模型或多训练导致资源消耗大的问题，通过单奖励模型分类问答对生成奖励分数优化策略，无需偏好训练，计算效率更高且性能更优。**

- **链接: [http://arxiv.org/pdf/2505.20336v1](http://arxiv.org/pdf/2505.20336v1)**

> **作者:** Yu Zhang; Wanli Jiang; Zhengyu Yang
>
> **摘要:** The multi-objective alignment of Large Language Models (LLMs) is essential for ensuring foundational models conform to diverse human preferences. Current research in this field typically involves either multiple policies or multiple reward models customized for various preferences, or the need to train a preference-specific supervised fine-tuning (SFT) model. In this work, we introduce a novel multi-objective alignment method, MOSLIM, which utilizes a single reward model and policy model to address diverse objectives. MOSLIM provides a flexible way to control these objectives through prompting and does not require preference training during SFT phase, allowing thousands of off-the-shelf models to be directly utilized within this training framework. MOSLIM leverages a multi-head reward model that classifies question-answer pairs instead of scoring them and then optimize policy model with a scalar reward derived from a mapping function that converts classification results from reward model into reward scores. We demonstrate the efficacy of our proposed method across several multi-objective benchmarks and conduct ablation studies on various reward model sizes and policy optimization methods. The MOSLIM method outperforms current multi-objective approaches in most results while requiring significantly fewer GPU computing resources compared with existing policy optimization methods.
>
---
#### [new 021] PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims
- **分类: cs.CL**

- **简介: 论文提出PEDANTIC数据集，解决专利权利要求确定性检测缺乏标注数据的问题。通过自动流程结合LLM从USPTO文档提取标注，构建14k NLP专利数据，经人工验证。评估显示LLM在预测上不超传统模型，但能识别原因，公开数据和代码。**

- **链接: [http://arxiv.org/pdf/2505.21342v1](http://arxiv.org/pdf/2505.21342v1)**

> **作者:** Valentin Knappich; Annemarie Friedrich; Anna Hätty; Simon Razniewski
>
> **摘要:** Patent claims define the scope of protection for an invention. If there are ambiguities in a claim, it is rejected by the patent office. In the US, this is referred to as indefiniteness (35 U.S.C {\S} 112(b)) and is among the most frequent reasons for patent application rejection. The development of automatic methods for patent definiteness examination has the potential to make patent drafting and examination more efficient, but no annotated dataset has been published to date. We introduce PEDANTIC (\underline{P}at\underline{e}nt \underline{D}efiniteness Ex\underline{a}mi\underline{n}a\underline{ti}on \underline{C}orpus), a novel dataset of 14k US patent claims from patent applications relating to Natural Language Processing (NLP), annotated with reasons for indefiniteness. We construct PEDANTIC using a fully automatic pipeline that retrieves office action documents from the USPTO and uses Large Language Models (LLMs) to extract the reasons for indefiniteness. A human validation study confirms the pipeline's accuracy in generating high-quality annotations. To gain insight beyond binary classification metrics, we implement an LLM-as-Judge evaluation that compares the free-form reasoning of every model-cited reason with every examiner-cited reason. We show that LLM agents based on Qwen 2.5 32B and 72B struggle to outperform logistic regression baselines on definiteness prediction, even though they often correctly identify the underlying reasons. PEDANTIC provides a valuable resource for patent AI researchers, enabling the development of advanced examination models. We will publicly release the dataset and code.
>
---
#### [new 022] Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型（LLMs）的事实自我意识，解决其生成内容事实错误的问题。发现LLMs在生成时通过Transformer残差流的线性特征自我评估事实正确性，且该信号对格式变化鲁棒。实验表明自我意识在训练早期快速出现，中间层达峰，提升模型可解释性与可靠性。**

- **链接: [http://arxiv.org/pdf/2505.21399v1](http://arxiv.org/pdf/2505.21399v1)**

> **作者:** Hovhannes Tamoyan; Subhabrata Dutta; Iryna Gurevych
>
> **摘要:** Factual incorrectness in generated content is one of the primary concerns in ubiquitous deployment of large language models (LLMs). Prior findings suggest LLMs can (sometimes) detect factual incorrectness in their generated content (i.e., fact-checking post-generation). In this work, we provide evidence supporting the presence of LLMs' internal compass that dictate the correctness of factual recall at the time of generation. We demonstrate that for a given subject entity and a relation, LLMs internally encode linear features in the Transformer's residual stream that dictate whether it will be able to recall the correct attribute (that forms a valid entity-relation-attribute triplet). This self-awareness signal is robust to minor formatting variations. We investigate the effects of context perturbation via different example selection strategies. Scaling experiments across model sizes and training dynamics highlight that self-awareness emerges rapidly during training and peaks in intermediate layers. These findings uncover intrinsic self-monitoring capabilities within LLMs, contributing to their interpretability and reliability.
>
---
#### [new 023] Are Language Models Consequentialist or Deontological Moral Reasoners?
- **分类: cs.CL**

- **简介: 该论文通过分析LLMs在600余个电车难题中的推理过程，研究其道德推理倾向。任务为区分LLMs是否遵循功利主义或义务论伦理理论。发现LLMs推理链偏向义务论，但事后解释更倾向功利主义，提出分类框架以指导AI伦理部署。**

- **链接: [http://arxiv.org/pdf/2505.21479v1](http://arxiv.org/pdf/2505.21479v1)**

> **作者:** Keenan Samway; Max Kleiman-Weiner; David Guzman Piedrahita; Rada Mihalcea; Bernhard Schölkopf; Zhijing Jin
>
> **摘要:** As AI systems increasingly navigate applications in healthcare, law, and governance, understanding how they handle ethically complex scenarios becomes critical. Previous work has mainly examined the moral judgments in large language models (LLMs), rather than their underlying moral reasoning process. In contrast, we focus on a large-scale analysis of the moral reasoning traces provided by LLMs. Furthermore, unlike prior work that attempted to draw inferences from only a handful of moral dilemmas, our study leverages over 600 distinct trolley problems as probes for revealing the reasoning patterns that emerge within different LLMs. We introduce and test a taxonomy of moral rationales to systematically classify reasoning traces according to two main normative ethical theories: consequentialism and deontology. Our analysis reveals that LLM chains-of-thought tend to favor deontological principles based on moral obligations, while post-hoc explanations shift notably toward consequentialist rationales that emphasize utility. Our framework provides a foundation for understanding how LLMs process and articulate ethical considerations, an important step toward safe and interpretable deployment of LLMs in high-stakes decision-making environments. Our code is available at https://github.com/keenansamway/moral-lens .
>
---
#### [new 024] How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型机制分析任务，旨在通过语言神经元探究对齐如何提升LLM的多语言能力。提出新神经元识别算法，划分模型处理四阶段（理解、共享推理、输出转换、词汇输出），对比对齐前后神经元特性，并分析"自发对齐"现象，揭示多语言能力提升机制。**

- **链接: [http://arxiv.org/pdf/2505.21505v1](http://arxiv.org/pdf/2505.21505v1)**

> **作者:** Shimao Zhang; Zhejian Lai; Xiang Liu; Shuaijie She; Xiao Liu; Yeyun Gong; Shujian Huang; Jiajun Chen
>
> **摘要:** Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some researches on language-specific neurons reveal that there are language-specific neurons that are selectively activated in LLMs when processing different languages. This provides a new perspective to analyze and understand LLMs' mechanisms more specifically in multilingual scenarios. In this work, we propose a new finer-grained neuron identification algorithm, which detects language neurons~(including language-specific neurons and language-related neurons) and language-agnostic neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights for better understanding multilingual alignment and multilingual capabilities of LLMs.
>
---
#### [new 025] Who Reasons in the Large Language Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM可解释性研究任务，旨在探究推理能力来源。针对推理能力是否来自整体模型、特定模块或过拟合问题，提出输出投影模块（oproj）为核心推理组件的假设，开发诊断工具SfN验证，发现oproj主导推理而其他模块影响对话流畅性。**

- **链接: [http://arxiv.org/pdf/2505.20993v1](http://arxiv.org/pdf/2505.20993v1)**

> **作者:** Jie Shao; Jianxin Wu
>
> **摘要:** Despite the impressive performance of large language models (LLMs), the process of endowing them with new capabilities--such as mathematical reasoning--remains largely empirical and opaque. A critical open question is whether reasoning abilities stem from the entire model, specific modules, or are merely artifacts of overfitting. In this work, we hypothesize that the reasoning capabilities in well-trained LLMs are primarily attributed to the output projection module (oproj) in the Transformer's multi-head self-attention (MHSA) mechanism. To support this hypothesis, we introduce Stethoscope for Networks (SfN), a suite of diagnostic tools designed to probe and analyze the internal behaviors of LLMs. Using SfN, we provide both circumstantial and empirical evidence suggesting that oproj plays a central role in enabling reasoning, whereas other modules contribute more to fluent dialogue. These findings offer a new perspective on LLM interpretability and open avenues for more targeted training strategies, potentially enabling more efficient and specialized LLMs.
>
---
#### [new 026] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors
- **分类: cs.CL**

- **简介: 该论文属于神经机器翻译（NMT）鲁棒性分析任务，旨在探究模型处理语法错误输入的机制。通过GED探测和表征相似性分析，发现编码器可检测错误并修正表征，结合注意力机制识别"鲁棒性头"，揭示模型在微调时依赖这些头更新错误词表示。**

- **链接: [http://arxiv.org/pdf/2505.21224v1](http://arxiv.org/pdf/2505.21224v1)**

> **作者:** Abderrahmane Issam; Yusuf Can Semerci; Jan Scholtes; Gerasimos Spanakis
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Understanding robustness is essential for building reliable NLP systems. Unfortunately, in the context of machine translation, previous work mainly focused on documenting robustness failures or improving robustness. In contrast, we study robustness from a model representation perspective by looking at internal model representations of ungrammatical inputs and how they evolve through model layers. For this purpose, we perform Grammatical Error Detection (GED) probing and representational similarity analysis. Our findings indicate that the encoder first detects the grammatical error, then corrects it by moving its representation toward the correct form. To understand what contributes to this process, we turn to the attention mechanism where we identify what we term Robustness Heads. We find that Robustness Heads attend to interpretable linguistic units when responding to grammatical errors, and that when we fine-tune models for robustness, they tend to rely more on Robustness Heads for updating the ungrammatical word representation.
>
---
#### [new 027] SEMMA: A Semantic Aware Knowledge Graph Foundation Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱（KG）的归纳式链接预测任务。针对现有模型仅依赖结构忽视文本语义、泛化能力差的问题，提出SEMMA模型：通过LLM增强关系标识符生成语义嵌入，融合文本关系图与结构信息。实验显示其在54个KG上超越结构基线，尤其在完全未知关系测试时效果翻倍，证明语义对结构泛化的必要性。**

- **链接: [http://arxiv.org/pdf/2505.20422v1](http://arxiv.org/pdf/2505.20422v1)**

> **作者:** Arvindh Arun; Sumit Kumar; Mojtaba Nayyeri; Bo Xiong; Ponnurangam Kumaraguru; Antonio Vergari; Steffen Staab
>
> **摘要:** Knowledge Graph Foundation Models (KGFMs) have shown promise in enabling zero-shot reasoning over unseen graphs by learning transferable patterns. However, most existing KGFMs rely solely on graph structure, overlooking the rich semantic signals encoded in textual attributes. We introduce SEMMA, a dual-module KGFM that systematically integrates transferable textual semantics alongside structure. SEMMA leverages Large Language Models (LLMs) to enrich relation identifiers, generating semantic embeddings that subsequently form a textual relation graph, which is fused with the structural component. Across 54 diverse KGs, SEMMA outperforms purely structural baselines like ULTRA in fully inductive link prediction. Crucially, we show that in more challenging generalization settings, where the test-time relation vocabulary is entirely unseen, structural methods collapse while SEMMA is 2x more effective. Our findings demonstrate that textual semantics are critical for generalization in settings where structure alone fails, highlighting the need for foundation models that unify structural and linguistic signals in knowledge reasoning.
>
---
#### [new 028] Towards Better Instruction Following Retrieval Models
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在提升模型遵循用户指令的能力。针对现有模型难以处理显式指令的问题，提出InF-IR数据集，包含3.8万条指令-查询-段落三元组及硬负例，并训练InF-Embed模型，通过对比学习与注意力机制优化检索结果，使p-MRR提升8.1%。**

- **链接: [http://arxiv.org/pdf/2505.21439v1](http://arxiv.org/pdf/2505.21439v1)**

> **作者:** Yuchen Zhuang; Aaron Trinh; Rushi Qiang; Haotian Sun; Chao Zhang; Hanjun Dai; Bo Dai
>
> **备注:** Retrieval Models, Embedding, Retrieval with Instructions
>
> **摘要:** Modern information retrieval (IR) models, trained exclusively on standard <query, passage> pairs, struggle to effectively interpret and follow explicit user instructions. We introduce InF-IR, a large-scale, high-quality training corpus tailored for enhancing retrieval models in Instruction-Following IR. InF-IR expands traditional training pairs into over 38,000 expressive <instruction, query, passage> triplets as positive samples. In particular, for each positive triplet, we generate two additional hard negative examples by poisoning both instructions and queries, then rigorously validated by an advanced reasoning model (o3-mini) to ensure semantic plausibility while maintaining instructional incorrectness. Unlike existing corpora that primarily support computationally intensive reranking tasks for decoder-only language models, the highly contrastive positive-negative triplets in InF-IR further enable efficient representation learning for smaller encoder-only models, facilitating direct embedding-based retrieval. Using this corpus, we train InF-Embed, an instruction-aware Embedding model optimized through contrastive learning and instruction-query attention mechanisms to align retrieval outcomes precisely with user intents. Extensive experiments across five instruction-based retrieval benchmarks demonstrate that InF-Embed significantly surpasses competitive baselines by 8.1% in p-MRR, measuring the instruction-following capabilities.
>
---
#### [new 029] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge
- **分类: cs.CL**

- **简介: 该论文聚焦自然语言到信号时序逻辑(STL)的自动转换任务，针对数据不足问题，构建了含16000样本的STL-DivEn数据集，并提出基于LLM和外部知识的KGST框架，通过生成-优化流程提升转换精度，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20658v1](http://arxiv.org/pdf/2505.20658v1)**

> **作者:** Yue Fang; Zhi Jin; Jie An; Hongshen Chen; Xiaohong Chen; Naijun Zhan
>
> **备注:** 13 pages, 5 figures, published to ACL
>
> **摘要:** Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets.
>
---
#### [new 030] Assessing the Capability of LLMs in Solving POSCOMP Questions
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估LLMs在巴西POSCOMP计算机考试中的能力，探究其专业领域应用潜力。测试了4个模型在2022-2023年试题，发现ChatGPT-4文本题最优但图像题较弱；更新的6模型在2022-2024年试题中超越人类平均及顶尖考生。任务为评估LLMs专业考试能力，解决其领域适用性问题，工作包括多模型多轮测试与分析。**

- **链接: [http://arxiv.org/pdf/2505.20338v1](http://arxiv.org/pdf/2505.20338v1)**

> **作者:** Cayo Viegas; Rohit Gheyi; Márcio Ribeiro
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly expanded the capabilities of artificial intelligence in natural language processing tasks. Despite this progress, their performance in specialized domains such as computer science remains relatively unexplored. Understanding the proficiency of LLMs in these domains is critical for evaluating their practical utility and guiding future developments. The POSCOMP, a prestigious Brazilian examination used for graduate admissions in computer science promoted by the Brazlian Computer Society (SBC), provides a challenging benchmark. This study investigates whether LLMs can match or surpass human performance on the POSCOMP exam. Four LLMs - ChatGPT-4, Gemini 1.0 Advanced, Claude 3 Sonnet, and Le Chat Mistral Large - were initially evaluated on the 2022 and 2023 POSCOMP exams. The assessments measured the models' proficiency in handling complex questions typical of the exam. LLM performance was notably better on text-based questions than on image interpretation tasks. In the 2022 exam, ChatGPT-4 led with 57 correct answers out of 69 questions, followed by Gemini 1.0 Advanced (49), Le Chat Mistral (48), and Claude 3 Sonnet (44). Similar trends were observed in the 2023 exam. ChatGPT-4 achieved the highest performance, surpassing all students who took the POSCOMP 2023 exam. LLMs, particularly ChatGPT-4, show promise in text-based tasks on the POSCOMP exam, although image interpretation remains a challenge. Given the rapid evolution of LLMs, we expanded our analysis to include more recent models - o1, Gemini 2.5 Pro, Claude 3.7 Sonnet, and o3-mini-high - evaluated on the 2022-2024 POSCOMP exams. These newer models demonstrate further improvements and consistently surpass both the average and top-performing human participants across all three years.
>
---
#### [new 031] Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs for KGQA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答（KGQA）任务，旨在解决LLM推理幻觉与KG结构化知识结合的问题。提出RAR框架，通过Reasoner生成推理链、Aligner映射KG路径、Responser合成答案，并用EM算法优化，实现高准确率（WebQSP和CWQ的Hit@1达93.3%和91.0%），兼具可解释性与零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.20971v1](http://arxiv.org/pdf/2505.20971v1)**

> **作者:** Xiangqing Shen; Fanfan Wang; Rui Xia
>
> **摘要:** LLMs have demonstrated remarkable capabilities in complex reasoning tasks, yet they often suffer from hallucinations and lack reliable factual grounding. Meanwhile, knowledge graphs (KGs) provide structured factual knowledge but lack the flexible reasoning abilities of LLMs. In this paper, we present Reason-Align-Respond (RAR), a novel framework that systematically integrates LLM reasoning with knowledge graphs for KGQA. Our approach consists of three key components: a Reasoner that generates human-like reasoning chains, an Aligner that maps these chains to valid KG paths, and a Responser that synthesizes the final answer. We formulate this process as a probabilistic model and optimize it using the Expectation-Maximization algorithm, which iteratively refines the reasoning chains and knowledge paths. Extensive experiments on multiple benchmarks demonstrate the effectiveness of RAR, achieving state-of-the-art performance with Hit@1 scores of 93.3% and 91.0% on WebQSP and CWQ respectively. Human evaluation confirms that RAR generates high-quality, interpretable reasoning chains well-aligned with KG paths. Furthermore, RAR exhibits strong zero-shot generalization capabilities and maintains computational efficiency during inference.
>
---
#### [new 032] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出PreP-OCR管道，针对退化历史文档的OCR任务，通过两阶段方法提升文本提取：首先用合成数据训练图像恢复模型优化清晰度，其次基于ByT5的纠错模型修正语义错误，显著降低字符错误率（63.9-70.3%）。属于文档图像恢复与OCR优化，解决退化图像导致的高误差问题。**

- **链接: [http://arxiv.org/pdf/2505.20429v1](http://arxiv.org/pdf/2505.20429v1)**

> **作者:** Shuhao Guan; Moule Lin; Cheng Xu; Xinyi Liu; Jinman Zhao; Jiexin Fan; Qi Xu; Derek Greene
>
> **备注:** ACL 2025 main
>
> **摘要:** This paper introduces PreP-OCR, a two-stage pipeline that combines document image restoration with semantic-aware post-OCR correction to improve text extraction from degraded historical documents. Our key innovation lies in jointly optimizing image clarity and linguistic consistency. First, we generate synthetic image pairs with randomized text fonts, layouts, and degradations. An image restoration model is trained on this synthetic data, using multi-directional patch extraction and fusion to process large images. Second, a ByT5 post-corrector, fine-tuned on synthetic historical text training pairs, addresses any remaining OCR errors. Detailed experiments on 13,831 pages of real historical documents in English, French, and Spanish show that PreP-OCR pipeline reduces character error rates by 63.9-70.3\% compared to OCR on raw images. Our pipeline demonstrates the potential of integrating image restoration with linguistic error correction for digitizing historical archives.
>
---
#### [new 033] CHIMERA: A Knowledge Base of Idea Recombination in Scientific Literature
- **分类: cs.CL**

- **简介: 该论文构建CHIMERA知识库，通过抽取科学论文中的创意重组案例，分析跨领域创新并训练模型预测新方向。提出新信息抽取任务，标注数百篇摘要训练LLM模型，处理AI论文获28K重组实例，并开发假设生成模型，数据代码公开。**

- **链接: [http://arxiv.org/pdf/2505.20779v1](http://arxiv.org/pdf/2505.20779v1)**

> **作者:** Noy Sternlicht; Tom Hope
>
> **备注:** Project page: https://noy-sternlicht.github.io/CHIMERA-Web
>
> **摘要:** A hallmark of human innovation is the process of recombination -- creating original ideas by integrating elements of existing mechanisms and concepts. In this work, we automatically mine the scientific literature and build CHIMERA: a large-scale knowledge base (KB) of recombination examples. CHIMERA can be used to empirically explore at scale how scientists recombine concepts and take inspiration from different areas, or to train supervised machine learning models that learn to predict new creative cross-domain directions. To build this KB, we present a novel information extraction task of extracting recombination from scientific paper abstracts, collect a high-quality corpus of hundreds of manually annotated abstracts, and use it to train an LLM-based extraction model. The model is applied to a large corpus of papers in the AI domain, yielding a KB of over 28K recombination examples. We analyze CHIMERA to explore the properties of recombination in different subareas of AI. Finally, we train a scientific hypothesis generation model using the KB, which predicts new recombination directions that real-world researchers find inspiring. Our data and code are available at https://github.cs.huji.ac.il/tomhope-lab/CHIMERA
>
---
#### [new 034] Chinese Cyberbullying Detection: Dataset, Method, and Validation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中文网络欺凌事件检测任务，针对现有研究仅关注言论极性（如攻击性言论）而忽视事件关联的问题，构建首个中文网络欺凌事件数据集CHNCI（含91个事件的22万条评论）。采用生成解释的集成方法生成伪标签并结合人工标注，提出事件检测评估标准，验证了数据集的基准作用。**

- **链接: [http://arxiv.org/pdf/2505.20654v1](http://arxiv.org/pdf/2505.20654v1)**

> **作者:** Yi Zhu; Xin Zou; Xindong Wu
>
> **摘要:** Existing cyberbullying detection benchmarks were organized by the polarity of speech, such as "offensive" and "non-offensive", which were essentially hate speech detection. However, in the real world, cyberbullying often attracted widespread social attention through incidents. To address this problem, we propose a novel annotation method to construct a cyberbullying dataset that organized by incidents. The constructed CHNCI is the first Chinese cyberbullying incident detection dataset, which consists of 220,676 comments in 91 incidents. Specifically, we first combine three cyberbullying detection methods based on explanations generation as an ensemble method to generate the pseudo labels, and then let human annotators judge these labels. Then we propose the evaluation criteria for validating whether it constitutes a cyberbullying incident. Experimental results demonstrate that the constructed dataset can be a benchmark for the tasks of cyberbullying detection and incident prediction. To the best of our knowledge, this is the first study for the Chinese cyberbullying incident detection task.
>
---
#### [new 035] Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出无文本语音到语音翻译框架Dub-S2ST，用于跨语言配音。针对现有方法忽视源语音时长、语速及身份导致的不匹配问题，其创新采用离散扩散模型结合显式时长控制实现时间对齐，并引入单元级语速适应机制，合成语音保持源特征，实验验证其自然流畅且翻译效果优异。**

- **链接: [http://arxiv.org/pdf/2505.20899v1](http://arxiv.org/pdf/2505.20899v1)**

> **作者:** Jeongsoo Choi; Jaehun Kim; Joon Son Chung
>
> **摘要:** This paper introduces a cross-lingual dubbing system that translates speech from one language to another while preserving key characteristics such as duration, speaker identity, and speaking speed. Despite the strong translation quality of existing speech translation approaches, they often overlook the transfer of speech patterns, leading to mismatches with source speech and limiting their suitability for dubbing applications. To address this, we propose a discrete diffusion-based speech-to-unit translation model with explicit duration control, enabling time-aligned translation. We then synthesize speech based on the predicted units and source identity with a conditional flow matching model. Additionally, we introduce a unit-based speed adaptation mechanism that guides the translation model to produce speech at a rate consistent with the source, without relying on any text. Extensive experiments demonstrate that our framework generates natural and fluent translations that align with the original speech's duration and speaking pace, while achieving competitive translation performance.
>
---
#### [new 036] Enhancing Logical Reasoning in Language Models via Symbolically-Guided Monte Carlo Process Supervision
- **分类: cs.CL**

- **简介: 该论文属于提升语言模型逻辑推理任务，旨在解决LLMs依赖记忆而非符号抽象导致的泛化不足问题。提出通过生成符号推理轨迹，利用蒙特卡洛过程奖励模型筛选高质量轨迹，并通过微调增强模型的逻辑推理与跨领域泛化能力，实验显示显著提升。**

- **链接: [http://arxiv.org/pdf/2505.20415v1](http://arxiv.org/pdf/2505.20415v1)**

> **作者:** Xingwei Tan; Marco Valentino; Mahmud Akhter; Maria Liakata; Nikolaos Aletras
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) have shown promising performance in mathematical and logical reasoning benchmarks. However, recent studies have pointed to memorization, rather than generalization, as one of the leading causes for such performance. LLMs, in fact, are susceptible to content variations, demonstrating a lack of robust symbolic abstractions supporting their reasoning process. To improve reliability, many attempts have been made to combine LLMs with symbolic methods. Nevertheless, existing approaches fail to effectively leverage symbolic representations due to the challenges involved in developing reliable and scalable verification mechanisms. In this paper, we propose to overcome such limitations by generating symbolic reasoning trajectories and select the high-quality ones using a process reward model automatically tuned based on Monte Carlo estimation. The trajectories are then employed via fine-tuning methods to improve logical reasoning and generalization. Our results on logical reasoning benchmarks such as FOLIO and LogicAsker show the effectiveness of the proposed method with large gains on frontier and open-weight models. Moreover, additional experiments on claim verification reveal that fine-tuning on the generated symbolic reasoning trajectories enhances out-of-domain generalizability, suggesting the potential impact of symbolically-guided process supervision in alleviating the effect of memorization on LLM reasoning.
>
---
#### [new 037] LLMs are Frequency Pattern Learners in Natural Language Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）任务，探究LLMs微调提升推理性能的机制。发现模型学习数据中"假设高频谓词更常见于正确实例"的频率模式，通过实验验证模型依赖此偏见且微调加剧依赖，并关联词频与文本蕴含。**

- **链接: [http://arxiv.org/pdf/2505.21011v1](http://arxiv.org/pdf/2505.21011v1)**

> **作者:** Liang Cheng; Zhaowei Wang; Mark Steedman
>
> **备注:** 9 pages
>
> **摘要:** While fine-tuning LLMs on NLI corpora improves their inferential performance, the underlying mechanisms driving this improvement remain largely opaque. In this work, we conduct a series of experiments to investigate what LLMs actually learn during fine-tuning. We begin by analyzing predicate frequencies in premises and hypotheses across NLI datasets and identify a consistent frequency bias, where predicates in hypotheses occur more frequently than those in premises for positive instances. To assess the impact of this bias, we evaluate both standard and NLI fine-tuned LLMs on bias-consistent and bias-adversarial cases. We find that LLMs exploit frequency bias for inference and perform poorly on adversarial instances. Furthermore, fine-tuned LLMs exhibit significantly increased reliance on this bias, suggesting that they are learning these frequency patterns from datasets. Finally, we compute the frequencies of hyponyms and their corresponding hypernyms from WordNet, revealing a correlation between frequency bias and textual entailment. These findings help explain why learning frequency patterns can enhance model performance on inference tasks.
>
---
#### [new 038] Scaling and Prompting for Improved End-to-End Spoken Grammatical Error Correction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于端到端口语语法错误纠正（SGEC）与反馈生成任务。针对标注数据不足及模型性能瓶颈，提出伪标记方法扩展训练数据至2500小时，并通过提示优化Whisper模型，提升纠错准确性和反馈效果。实验表明，伪标签对大模型效果有限，但提示策略有效。**

- **链接: [http://arxiv.org/pdf/2505.21137v1](http://arxiv.org/pdf/2505.21137v1)**

> **作者:** Mengjie Qian; Rao Ma; Stefano Bannò; Kate M. Knill; Mark J. F. Gales
>
> **备注:** submitted to Interspeech
>
> **摘要:** Spoken Grammatical Error Correction (SGEC) and Feedback (SGECF) are crucial for second language learners, teachers and test takers. Traditional SGEC systems rely on a cascaded pipeline consisting of an ASR, a module for disfluency detection (DD) and removal and one for GEC. With the rise of end-to-end (E2E) speech foundation models, we investigate their effectiveness in SGEC and feedback generation. This work introduces a pseudo-labelling process to address the challenge of limited labelled data, expanding the training data size from 77 hours to approximately 2500 hours, leading to improved performance. Additionally, we prompt an E2E Whisper-based SGEC model with fluent transcriptions, showing a slight improvement in SGEC performance, with more significant gains in feedback generation. Finally, we assess the impact of increasing model size, revealing that while pseudo-labelled data does not yield performance gain for a larger Whisper model, training with prompts proves beneficial.
>
---
#### [new 039] Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 论文属于推理模型效率优化任务，解决其在简单问题上过度思考导致资源浪费的问题。提出Self-Route框架，通过能力评估动态切换一般/推理模式；利用隐藏层嵌入实时评估模型能力，构建Gradient-10K数据集训练路由；实验显示准确率相当且减少30-55% token消耗。**

- **链接: [http://arxiv.org/pdf/2505.20664v1](http://arxiv.org/pdf/2505.20664v1)**

> **作者:** Yang He; Xiao Ding; Bibo Cai; Yufei Zhang; Kai Xiong; Zhouhao Sun; Bing Qin; Ting Liu
>
> **摘要:** While reasoning-augmented large language models (RLLMs) significantly enhance complex task performance through extended reasoning chains, they inevitably introduce substantial unnecessary token consumption, particularly for simpler problems where Short Chain-of-Thought (Short CoT) suffices. This overthinking phenomenon leads to inefficient resource usage without proportional accuracy gains. To address this issue, we propose Self-Route, a dynamic reasoning framework that automatically selects between general and reasoning modes based on model capability estimation. Our approach introduces a lightweight pre-inference stage to extract capability-aware embeddings from hidden layer representations, enabling real-time evaluation of the model's ability to solve problems. We further construct Gradient-10K, a model difficulty estimation-based dataset with dense complexity sampling, to train the router for precise capability boundary detection. Extensive experiments demonstrate that Self-Route achieves comparable accuracy to reasoning models while reducing token consumption by 30-55\% across diverse benchmarks. The proposed framework demonstrates consistent effectiveness across models with different parameter scales and reasoning paradigms, highlighting its general applicability and practical value.
>
---
#### [new 040] FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出FinTagging，首个全面表格感知的XBRL基准，评估LLM在财务信息结构化提取与语义对齐的能力。针对现有基准简化任务且忽略表格的问题，将其分解为财务实体抽取（FinNI）和分类法概念对齐（FinCL）子任务，要求模型处理文本及表格并关联万项分类。实验显示LLM在提取上表现佳，但细粒度对齐不足，凸显需改进语义推理与模式建模。**

- **链接: [http://arxiv.org/pdf/2505.20650v1](http://arxiv.org/pdf/2505.20650v1)**

> **作者:** Yan Wang; Yang Ren; Lingfei Qian; Xueqing Peng; Keyi Wang; Yi Han; Dongji Feng; Xiao-Yang Liu; Jimin Huang; Qianqian Xie
>
> **摘要:** We introduce FinTagging, the first full-scope, table-aware XBRL benchmark designed to evaluate the structured information extraction and semantic alignment capabilities of large language models (LLMs) in the context of XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL tagging as flat multi-class classification and focus solely on narrative text, FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for financial entity extraction and FinCL for taxonomy-driven concept alignment. It requires models to jointly extract facts and align them with the full 10k+ US-GAAP taxonomy across both unstructured text and structured tables, enabling realistic, fine-grained evaluation. We assess a diverse set of LLMs under zero-shot settings, systematically analyzing their performance on both subtasks and overall tagging accuracy. Our results reveal that, while LLMs demonstrate strong generalization in information extraction, they struggle with fine-grained concept alignment, particularly in disambiguating closely related taxonomy entries. These findings highlight the limitations of existing LLMs in fully automating XBRL tagging and underscore the need for improved semantic reasoning and schema-aware modeling to meet the demands of accurate financial disclosure. Code is available at our GitHub repository and data is at our Hugging Face repository.
>
---
#### [new 041] Tracing and Reversing Rank-One Model Edits
- **分类: cs.CL**

- **简介: 该论文属于模型安全领域，针对知识编辑（如ROME）的恶意篡改风险，研究其可追溯与可逆性。通过分析编辑权重的分布特征，实现定位、预测修改内容及95%精度推断实体，并可逆向恢复原始输出（≥80%准确率），构建防御框架。**

- **链接: [http://arxiv.org/pdf/2505.20819v1](http://arxiv.org/pdf/2505.20819v1)**

> **作者:** Paul Youssef; Zhixue Zhao; Christin Seifert; Jörg Schlötterer
>
> **摘要:** Knowledge editing methods (KEs) are a cost-effective way to update the factual content of large language models (LLMs), but they pose a dual-use risk. While KEs are beneficial for updating outdated or incorrect information, they can be exploited maliciously to implant misinformation or bias. In order to defend against these types of malicious manipulation, we need robust techniques that can reliably detect, interpret, and mitigate adversarial edits. This work investigates the traceability and reversibility of knowledge edits, focusing on the widely used Rank-One Model Editing (ROME) method. We first show that ROME introduces distinctive distributional patterns in the edited weight matrices, which can serve as effective signals for locating the edited weights. Second, we show that these altered weights can reliably be used to predict the edited factual relation, enabling partial reconstruction of the modified fact. Building on this, we propose a method to infer the edited object entity directly from the modified weights, without access to the editing prompt, achieving over 95% accuracy. Finally, we demonstrate that ROME edits can be reversed, recovering the model's original outputs with $\geq$ 80% accuracy. Our findings highlight the feasibility of detecting, tracing, and reversing edits based on the edited weights, offering a robust framework for safeguarding LLMs against adversarial manipulations.
>
---
#### [new 042] SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习（RL）训练LLM代理处理复杂多步任务，旨在解决延迟奖励导致早期动作反馈不足的问题。提出SPA框架，通过进展估计器将最终奖励分解为每步贡献，并结合环境动作的grounding信号形成中间奖励，提升训练效果。**

- **链接: [http://arxiv.org/pdf/2505.20732v1](http://arxiv.org/pdf/2505.20732v1)**

> **作者:** Hanlin Wang; Chak Tou Leong; Jiashuo Wang; Jian Wang; Wenjie Li
>
> **摘要:** Reinforcement learning (RL) holds significant promise for training LLM agents to handle complex, goal-oriented tasks that require multi-step interactions with external environments. However, a critical challenge when applying RL to these agentic tasks arises from delayed rewards: feedback signals are typically available only after the entire task is completed. This makes it non-trivial to assign delayed rewards to earlier actions, providing insufficient guidance regarding environmental constraints and hindering agent training. In this work, we draw on the insight that the ultimate completion of a task emerges from the cumulative progress an agent makes across individual steps. We propose Stepwise Progress Attribution (SPA), a general reward redistribution framework that decomposes the final reward into stepwise contributions, each reflecting its incremental progress toward overall task completion. To achieve this, we train a progress estimator that accumulates stepwise contributions over a trajectory to match the task completion. During policy optimization, we combine the estimated per-step contribution with a grounding signal for actions executed in the environment as the fine-grained, intermediate reward for effective agent training. Extensive experiments on common agent benchmarks (including Webshop, ALFWorld, and VirtualHome) demonstrate that SPA consistently outperforms the state-of-the-art method in both success rate (+2.5\% on average) and grounding accuracy (+1.9\% on average). Further analyses demonstrate that our method remarkably provides more effective intermediate rewards for RL training. Our code is available at https://github.com/WangHanLinHenry/SPA-RL-Agent.
>
---
#### [new 043] PHISH in MESH: Korean Adversarial Phonetic Substitution and Phonetic-Semantic Feature Integration Defense
- **分类: cs.CL**

- **简介: 该论文针对韩语仇恨言论检测中对抗性语音替代攻击问题，提出PHISH方法生成韩语语音扰动攻击，并设计MESH模型通过整合语音-语义特征提升检测鲁棒性，填补了韩语防御架构研究空白。**

- **链接: [http://arxiv.org/pdf/2505.21380v1](http://arxiv.org/pdf/2505.21380v1)**

> **作者:** Byungjun Kim; Minju Kim; Hyeonchu Park; Bugeun Kim
>
> **备注:** Under review
>
> **摘要:** As malicious users increasingly employ phonetic substitution to evade hate speech detection, researchers have investigated such strategies. However, two key challenges remain. First, existing studies have overlooked the Korean language, despite its vulnerability to phonetic perturbations due to its phonographic nature. Second, prior work has primarily focused on constructing datasets rather than developing architectural defenses. To address these challenges, we propose (1) PHonetic-Informed Substitution for Hangul (PHISH) that exploits the phonological characteristics of the Korean writing system, and (2) Mixed Encoding of Semantic-pHonetic features (MESH) that enhances the detector's robustness by incorporating phonetic information at the architectural level. Our experimental results demonstrate the effectiveness of our proposed methods on both perturbed and unperturbed datasets, suggesting that they not only improve detection performance but also reflect realistic adversarial behaviors employed by malicious users.
>
---
#### [new 044] Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration
- **分类: cs.CL**

- **简介: 该论文提出XpandA框架，解决长文本处理中代理方法的高延迟、信息损失及依赖破坏问题。通过动态分区、问题驱动共享内存及选择性重播，提升LLM长上下文能力，获20%提升和1.5倍加速。**

- **链接: [http://arxiv.org/pdf/2505.20625v1](http://arxiv.org/pdf/2505.20625v1)**

> **作者:** Sibo Xiao; Zixin Lin; Wenyang Gao; Yue Zhang
>
> **摘要:** Processing long contexts has become a critical capability for modern large language models (LLMs). Existing works leverage agent-based divide-and-conquer methods for processing long contexts. But these methods face crucial limitations, including prohibitive accumulated latency and amplified information loss from excessive agent invocations, and the disruption of inherent textual dependencies by immoderate partitioning. In this paper, we propose a novel multi-agent framework XpandA (Expand-Agent) coupled with question-driven workflow and dynamic partitioning for robust long-context processing. XpandA overcomes these limitations through: 1) dynamic partitioning of long texts, which adaptively modulates the filling rate of context windows for input sequences of vastly varying lengths; 2) question-guided protocol to update flat information ensembles within centralized shared memory, constructing consistent inter-agent knowledge across partitions; and 3) selectively replaying specific partitions based on the state-tracking of question-information couples to promote the resolution of inverted-order structures across partitions (e.g., flashbacks). We perform a comprehensive evaluation of XpandA on multiple long-context benchmarks with length varying from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long sequences and its significant effectiveness in enhancing the long-context capabilities of various LLMs by achieving 20\% improvements and 1.5x inference speedup over baselines of full-context, RAG and previous agent-based methods.
>
---
#### [new 045] Thinker: Learning to Think Fast and Slow
- **分类: cs.CL; cs.AI; cs.LG; I.2.6; I.2.8; I.5.1**

- **简介: 该论文提出Thinker框架，通过四阶段任务（快思、验证、慢思、总结）优化LLMs推理，解决其在QA任务中不精准、冗长的问题。实验显示模型准确率提升，证明直觉与深思系统的互补性。**

- **链接: [http://arxiv.org/pdf/2505.21097v1](http://arxiv.org/pdf/2505.21097v1)**

> **作者:** Stephen Chung; Wenyu Du; Jie Fu
>
> **备注:** 21 pages
>
> **摘要:** Recent studies show that the reasoning capabilities of Large Language Models (LLMs) can be improved by applying Reinforcement Learning (RL) to question-answering (QA) tasks in areas such as math and coding. With a long context length, LLMs may learn to perform search, as indicated by the self-correction behavior observed in DeepSeek R1. However, this search behavior is often imprecise and lacks confidence, resulting in long, redundant responses and highlighting deficiencies in intuition and verification. Inspired by the Dual Process Theory in psychology, we introduce a simple modification to the QA task that includes four stages: Fast Thinking, where the LLM must answer within a strict token budget; Verification, where the model evaluates its initial response; Slow Thinking, where it refines the initial response with more deliberation; and Summarization, where it distills the refinement from the previous stage into precise steps. Our proposed task improves average accuracy from 24.9% to 27.9% for Qwen2.5-1.5B, and from 45.9% to 49.8% for DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone achieves 26.8% accuracy using fewer than 1000 tokens, demonstrating substantial inference efficiency gains. These findings suggest that intuition and deliberative reasoning are distinct, complementary systems benefiting from targeted training.
>
---
#### [new 046] Automated Privacy Information Annotation in Large Language Model Interactions
- **分类: cs.CL**

- **简介: 该论文提出自动化标注大语言模型交互中的隐私信息，解决用户使用真实身份时无意泄露隐私的检测问题。构建含249K查询及154K标注隐私短语的多语言数据集，开发基于强LLM的自动化标注管道，设计多层级评估指标，并建立轻量级基线模型，揭示现有方法与实际需求的性能差距。**

- **链接: [http://arxiv.org/pdf/2505.20910v1](http://arxiv.org/pdf/2505.20910v1)**

> **作者:** Hang Zeng; Xiangyu Liu; Yong Hu; Chaoyue Niu; Fan Wu; Shaojie Tang; Guihai Chen
>
> **备注:** 9 content pages
>
> **摘要:** Users interacting with large language models (LLMs) under their real identifiers often unknowingly risk disclosing private information. Automatically notifying users whether their queries leak privacy and which phrases leak what private information has therefore become a practical need. Existing privacy detection methods, however, were designed for different objectives and application scenarios, typically tagging personally identifiable information (PII) in anonymous content. In this work, to support the development and evaluation of privacy detection models for LLM interactions that are deployable on local user devices, we construct a large-scale multilingual dataset with 249K user queries and 154K annotated privacy phrases. In particular, we build an automated privacy annotation pipeline with cloud-based strong LLMs to automatically extract privacy phrases from dialogue datasets and annotate leaked information. We also design evaluation metrics at the levels of privacy leakage, extracted privacy phrase, and privacy information. We further establish baseline methods using light-weight LLMs with both tuning-free and tuning-based methods, and report a comprehensive evaluation of their performance. Evaluation results reveal a gap between current performance and the requirements of real-world LLM applications, motivating future research into more effective local privacy detection methods grounded in our dataset.
>
---
#### [new 047] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?
- **分类: cs.CL**

- **简介: 该论文研究长上下文in-context学习中示例数量对模型不确定性的影。针对现有研究忽视可信度问题，系统量化不同示例量下的预测不确定性，通过分解认识论不确定性（EU），发现增加示例可减少总不确定，复杂任务需先处理输入噪声，并分析内部信心变化机制。**

- **链接: [http://arxiv.org/pdf/2505.21003v1](http://arxiv.org/pdf/2505.21003v1)**

> **作者:** Yifei Wang; Yu Sheng; Linjing Li; Daniel Zeng
>
> **备注:** Camera-ready versions for ACL 2025 Findings
>
> **摘要:** Recent advances in handling long sequences have facilitated the exploration of long-context in-context learning (ICL). While much of the existing research emphasizes performance improvements driven by additional in-context examples, the influence on the trustworthiness of generated responses remains underexplored. This paper addresses this gap by investigating how increased examples influence predictive uncertainty, an essential aspect in trustworthiness. We begin by systematically quantifying the uncertainty of ICL with varying shot counts, analyzing the impact of example quantity. Through uncertainty decomposition, we introduce a novel perspective on performance enhancement, with a focus on epistemic uncertainty (EU). Our results reveal that additional examples reduce total uncertainty in both simple and complex tasks by injecting task-specific knowledge, thereby diminishing EU and enhancing performance. For complex tasks, these advantages emerge only after addressing the increased noise and uncertainty associated with longer inputs. Finally, we explore the evolution of internal confidence across layers, unveiling the mechanisms driving the reduction in uncertainty.
>
---
#### [new 048] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型蒸馏任务，旨在通过时序差分模仿学习提升模型压缩效率。针对大型模型计算成本高的问题，提出基于教师模型分布稀疏性的通用框架，在缩减的词汇子集上进行时序差分学习，减少动作空间以优化蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2505.20335v1](http://arxiv.org/pdf/2505.20335v1)**

> **作者:** Zishun Yu; Shangzhe Li; Xinhua Zhang
>
> **摘要:** Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements.
>
---
#### [new 049] Articulatory strategy in vowel production as a basis for speaker discrimination
- **分类: cs.CL**

- **简介: 该论文属于说话人辨别任务，旨在探究元音发音的构音策略是否具有足够的个体差异支持说话人区分。研究通过广义Procrustes分析40名英语母语者的舌形数据，发现舌大小是核心区分特征，前舌形状比后舌更具辨别力，且非共变的形状特征组合可达到与形状-大小联合特征相当的区分效果。**

- **链接: [http://arxiv.org/pdf/2505.20995v1](http://arxiv.org/pdf/2505.20995v1)**

> **作者:** Justin J. H. Lo; Patrycja Strycharczuk; Sam Kirkham
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The way speakers articulate is well known to be variable across individuals while at the same time subject to anatomical and biomechanical constraints. In this study, we ask whether articulatory strategy in vowel production can be sufficiently speaker-specific to form the basis for speaker discrimination. We conducted Generalised Procrustes Analyses of tongue shape data from 40 English speakers from the North West of England, and assessed the speaker-discriminatory potential of orthogonal tongue shape features within the framework of likelihood ratios. Tongue size emerged as the individual dimension with the strongest discriminatory power, while tongue shape variation in the more anterior part of the tongue generally outperformed tongue shape variation in the posterior part. When considered in combination, shape-only information may offer comparable levels of speaker specificity to size-and-shape information, but only when features do not exhibit speaker-level co-variation.
>
---
#### [new 050] Large Language Models for IT Automation Tasks: Are We There Yet?
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于评估大型语言模型（LLM）在IT自动化任务（如Ansible脚本生成）中的表现，旨在解决现有基准与实际需求脱节及模型能力不足的问题。工作包括构建含126个真实任务的ITAB基准，测试14个开源LLM，发现其因状态协调缺陷（44.87%）和模块知识不足（24.37%）导致通过率低于12%。**

- **链接: [http://arxiv.org/pdf/2505.20505v1](http://arxiv.org/pdf/2505.20505v1)**

> **作者:** Md Mahadi Hassan; John Salvador; Akond Rahman; Santu Karmaker
>
> **备注:** 8 pages
>
> **摘要:** LLMs show promise in code generation, yet their effectiveness for IT automation tasks, particularly for tools like Ansible, remains understudied. Existing benchmarks rely primarily on synthetic tasks that fail to capture the needs of practitioners who use IT automation tools, such as Ansible. We present ITAB (IT Automation Task Benchmark), a benchmark of 126 diverse tasks (e.g., configuring servers, managing files) where each task accounts for state reconciliation: a property unique to IT automation tools. ITAB evaluates LLMs' ability to generate functional Ansible automation scripts via dynamic execution in controlled environments. We evaluate 14 open-source LLMs, none of which accomplish pass@10 at a rate beyond 12%. To explain these low scores, we analyze 1,411 execution failures across the evaluated LLMs and identify two main categories of prevalent semantic errors: failures in state reconciliation related reasoning (44.87% combined from variable (11.43%), host (11.84%), path(11.63%), and template (9.97%) issues) and deficiencies in module-specific execution knowledge (24.37% combined from Attribute and parameter (14.44%) and module (9.93%) errors). Our findings reveal key limitations in open-source LLMs' ability to track state changes and apply specialized module knowledge, indicating that reliable IT automation will require major advances in state reasoning and domain-specific execution understanding.
>
---
#### [new 051] Gatsby Without the 'E': Crafting Lipograms with LLMs
- **分类: cs.CL**

- **简介: 该论文研究利用LLMs生成无特定字母文本的任务，解决在严格约束下保持语义的问题。通过同义词替换、生成模型（含集束搜索和命名实体分析）将《了不起的盖茨比》改写为无'e'文本，发现排除前3.6%常见字母（至'u'）对意义影响小，但约束增强时保真度显著下降，揭示语言灵活性。**

- **链接: [http://arxiv.org/pdf/2505.20501v1](http://arxiv.org/pdf/2505.20501v1)**

> **作者:** Rohan Balasubramanian; Nitish Gokulakrishnan; Syeda Jannatus Saba; Steven Skiena
>
> **备注:** 7.5 pages
>
> **摘要:** Lipograms are a unique form of constrained writing where all occurrences of a particular letter are excluded from the text, typified by the novel Gadsby, which daringly avoids all usage of the letter 'e'. In this study, we explore the power of modern large language models (LLMs) by transforming the novel F. Scott Fitzgerald's The Great Gatsby into a fully 'e'-less text. We experimented with a range of techniques, from baseline methods like synonym replacement to sophisticated generative models enhanced with beam search and named entity analysis. We show that excluding up to 3.6% of the most common letters (up to the letter 'u') had minimal impact on the text's meaning, although translation fidelity rapidly and predictably decays with stronger lipogram constraints. Our work highlights the surprising flexibility of English under strict constraints, revealing just how adaptable and creative language can be.
>
---
#### [new 052] Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于在线对话理解任务，旨在解决短文本及隐式引用导致的上下文捕捉难题。提出两种Conversation Kernels机制，通过分析对话树的邻近结构构建相关上下文，适用于判断帖子属性（如有趣、有见地等）。实验在Slashdot数据上验证了方法的通用性。**

- **链接: [http://arxiv.org/pdf/2505.20482v1](http://arxiv.org/pdf/2505.20482v1)**

> **作者:** Vibhor Agarwal; Arjoo Gupta; Suparna De; Nishanth Sastry
>
> **备注:** Accepted at International AAAI Conference on Web and Social Media (ICWSM) 2025
>
> **摘要:** Understanding online conversations has attracted research attention with the growth of social networks and online discussion forums. Content analysis of posts and replies in online conversations is difficult because each individual utterance is usually short and may implicitly refer to other posts within the same conversation. Thus, understanding individual posts requires capturing the conversational context and dependencies between different parts of a conversation tree and then encoding the context dependencies between posts and comments/replies into the language model. To this end, we propose a general-purpose mechanism to discover appropriate conversational context for various aspects about an online post in a conversation, such as whether it is informative, insightful, interesting or funny. Specifically, we design two families of Conversation Kernels, which explore different parts of the neighborhood of a post in the tree representing the conversation and through this, build relevant conversational context that is appropriate for each task being considered. We apply our developed method to conversations crawled from slashdot.org, which allows users to apply highly different labels to posts, such as 'insightful', 'funny', etc., and therefore provides an ideal experimental platform to study whether a framework such as Conversation Kernels is general-purpose and flexible enough to be adapted to disparately different conversation understanding tasks.
>
---
#### [new 053] Dissecting Physics Reasoning in Small Language Models: A Multi-Dimensional Analysis from an Educational Perspective
- **分类: cs.CL; cs.AI; physics.ed-ph**

- **简介: 该论文评估小型语言模型（<40亿参数）的高中物理推理能力，解决其复杂推理不足的问题。通过构建基于教材的标注数据集（含跨文化改编），用Gemini评估，发现模型答案准确但推理缺陷多，提示需提升理解而非仅答案正确。**

- **链接: [http://arxiv.org/pdf/2505.20707v1](http://arxiv.org/pdf/2505.20707v1)**

> **作者:** Nicy Scaria; Silvester John Joseph Kennedy; Diksha Seth; Deepak Subramani
>
> **摘要:** Small Language Models (SLMs) offer computational efficiency and accessibility, making them promising for educational applications. However, their capacity for complex reasoning, particularly in domains such as physics, remains underexplored. This study investigates the high school physics reasoning capabilities of state-of-the-art SLMs (under 4 billion parameters), including instruct versions of Llama 3.2, Phi 4 Mini, Gemma 3, and Qwen series. We developed a comprehensive physics dataset from the OpenStax High School Physics textbook, annotated according to Bloom's Taxonomy, with LaTeX and plaintext mathematical notations. A novel cultural contextualization approach was applied to a subset, creating culturally adapted problems for Asian, African, and South American/Australian contexts while preserving core physics principles. Using an LLM-as-a-judge framework with Google's Gemini 2.5 Flash, we evaluated answer and reasoning chain correctness, along with calculation accuracy. The results reveal significant differences between the SLMs. Qwen 3 1.7B achieved high `answer accuracy' (85%), but `fully correct reasoning' was substantially low (38%). The format of the mathematical notation had a negligible impact on performance. SLMs exhibited varied performance across the physics topics and showed a decline in reasoning quality with increasing cognitive and knowledge complexity. In particular, the consistency of reasoning was largely maintained in diverse cultural contexts, especially by better performing models. These findings indicate that, while SLMs can often find correct answers, their underlying reasoning is frequently flawed, suggesting an overreliance on pattern recognition. For SLMs to become reliable educational tools in physics, future development must prioritize enhancing genuine understanding and the generation of sound, verifiable reasoning chains over mere answer accuracy.
>
---
#### [new 054] DecisionFlow: Advancing Large Language Model as Principled Decision Maker
- **分类: cs.CL**

- **简介: 该论文提出DecisionFlow框架，旨在提升大语言模型在医疗、金融等高风险领域的决策能力。针对现有模型决策不透明、推理不结构化的问题，其构建语义关联的决策空间并推导隐式效用函数，实现可解释的权衡推理。实验显示其准确率提升30%，增强决策一致性，推动LLM与符号推理的融合。**

- **链接: [http://arxiv.org/pdf/2505.21397v1](http://arxiv.org/pdf/2505.21397v1)**

> **作者:** Xiusi Chen; Shanyong Wang; Cheng Qian; Hongru Wang; Peixuan Han; Heng Ji
>
> **备注:** 24 pages, 13 figures
>
> **摘要:** In high-stakes domains such as healthcare and finance, effective decision-making demands not just accurate outcomes but transparent and explainable reasoning. However, current language models often lack the structured deliberation needed for such tasks, instead generating decisions and justifications in a disconnected, post-hoc manner. To address this, we propose DecisionFlow, a novel decision modeling framework that guides models to reason over structured representations of actions, attributes, and constraints. Rather than predicting answers directly from prompts, DecisionFlow builds a semantically grounded decision space and infers a latent utility function to evaluate trade-offs in a transparent, utility-driven manner. This process produces decisions tightly coupled with interpretable rationales reflecting the model's reasoning. Empirical results on two high-stakes benchmarks show that DecisionFlow not only achieves up to 30% accuracy gains over strong prompting baselines but also enhances alignment in outcomes. Our work is a critical step toward integrating symbolic reasoning with LLMs, enabling more accountable, explainable, and reliable LLM decision support systems. We release the data and code at https://github.com/xiusic/DecisionFlow.
>
---
#### [new 055] Predicting Implicit Arguments in Procedural Video Instructions
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对程序式视频指令中隐式参数预测问题，提出Implicit-VidSRL多模态数据集，通过构建需结合视觉与上下文推断隐式"what/where"参数的基准任务，揭示现有模型局限并提出iSRL-Qwen2-VL方法，显著提升隐式语义角色识别效果。**

- **链接: [http://arxiv.org/pdf/2505.21068v1](http://arxiv.org/pdf/2505.21068v1)**

> **作者:** Anil Batra; Laura Sevilla-Lara; Marcus Rohrbach; Frank Keller
>
> **备注:** ACL 2025 Main
>
> **摘要:** Procedural texts help AI enhance reasoning about context and action sequences. Transforming these into Semantic Role Labeling (SRL) improves understanding of individual steps by identifying predicate-argument structure like {verb,what,where/with}. Procedural instructions are highly elliptic, for instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second step's where argument is inferred from the context, referring to where the cucumber was placed. Prior SRL benchmarks often miss implicit arguments, leading to incomplete understanding. To address this, we introduce Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit arguments from contextual information in multimodal cooking procedures. Our proposed dataset benchmarks multimodal models' contextual reasoning, requiring entity tracking through visual changes in recipes. We study recent multimodal LLMs and reveal that they struggle to predict implicit arguments of what and where/with from multi-modal procedural data given the verb. Lastly, we propose iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o.
>
---
#### [new 056] Concealment of Intent: A Game-Theoretic Analysis
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，针对现有LLM防御机制易受隐匿意图攻击的问题，提出通过组合技能隐藏恶意的对抗性提示攻击策略，并构建博弈论框架分析攻防平衡点，设计针对性防御机制。实验验证了攻击有效性及防御优势。**

- **链接: [http://arxiv.org/pdf/2505.20841v1](http://arxiv.org/pdf/2505.20841v1)**

> **作者:** Xinbo Wu; Abhishek Umrawal; Lav R. Varshney
>
> **摘要:** As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.
>
---
#### [new 057] Silencer: From Discovery to Mitigation of Self-Bias in LLM-as-Benchmark-Generator
- **分类: cs.CL**

- **简介: 该论文针对LLM生成评估基准时的自我偏差问题，提出Silencer框架。通过分析问题领域、语言风格及错误标签等子偏差来源，利用多生成器异质性中和偏差，实验显示其将偏差降至近零，使评估效果（皮尔逊系数）从0.655提升至0.833，显著提升基准质量。**

- **链接: [http://arxiv.org/pdf/2505.20738v1](http://arxiv.org/pdf/2505.20738v1)**

> **作者:** Peiwen Yuan; Yiwei Li; Shaoxiong Feng; Xinglin Wang; Yueqi Zhang; Jiayi Shi; Chuyi Tan; Boyuan Pan; Yao Hu; Kan Li
>
> **摘要:** LLM-as-Benchmark-Generator methods have been widely studied as a supplement to human annotators for scalable evaluation, while the potential biases within this paradigm remain underexplored. In this work, we systematically define and validate the phenomenon of inflated performance in models evaluated on their self-generated benchmarks, referred to as self-bias, and attribute it to sub-biases arising from question domain, language style, and wrong labels. On this basis, we propose Silencer, a general framework that leverages the heterogeneity between multiple generators at both the sample and benchmark levels to neutralize bias and generate high-quality, self-bias-silenced benchmark. Experimental results across various settings demonstrate that Silencer can suppress self-bias to near zero, significantly improve evaluation effectiveness of the generated benchmark (with an average improvement from 0.655 to 0.833 in Pearson correlation with high-quality human-annotated benchmark), while also exhibiting strong generalizability.
>
---
#### [new 058] BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于GUI任务代理优化任务，解决现有代理缺乏错误检测与恢复机制的问题。提出BacktrackAgent框架，集成验证器、裁判器、反思器模块及奖励机制，结合专门设计的训练数据集，提升任务成功率与步骤准确率。**

- **链接: [http://arxiv.org/pdf/2505.20660v1](http://arxiv.org/pdf/2505.20660v1)**

> **作者:** Qinzhuo Wu; Pengzhi Gao; Wei Liu; Jian Luan
>
> **摘要:** Graphical User Interface (GUI) agents have gained substantial attention due to their impressive capabilities to complete tasks through multiple interactions within GUI environments. However, existing agents primarily focus on enhancing the accuracy of individual actions and often lack effective mechanisms for detecting and recovering from errors. To address these shortcomings, we propose the BacktrackAgent, a robust framework that incorporates a backtracking mechanism to improve task completion efficiency. BacktrackAgent includes verifier, judger, and reflector components as modules for error detection and recovery, while also applying judgment rewards to further enhance the agent's performance. Additionally, we develop a training dataset specifically designed for the backtracking mechanism, which considers the outcome pages after action executions. Experimental results show that BacktrackAgent has achieved performance improvements in both task success rate and step accuracy on Mobile3M and Auto-UI benchmarks. Our data and code will be released upon acceptance.
>
---
#### [new 059] Assessment of L2 Oral Proficiency using Speech Large Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究二语（L2）口语自动评估任务，针对传统方法（级联系统信息丢失、端到端模型局限）的不足，利用语音大语言模型（LLMs）探索更优方案。通过对比不同训练策略，验证语音LLMs在评分任务中的优势，其性能超越现有基线，并展现跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.21148v1](http://arxiv.org/pdf/2505.21148v1)**

> **作者:** Rao Ma; Mengjie Qian; Siyuan Tang; Stefano Bannò; Kate M. Knill; Mark J. F. Gales
>
> **备注:** submitted to Interspeech
>
> **摘要:** The growing population of L2 English speakers has increased the demand for developing automatic graders for spoken language assessment (SLA). Historically, statistical models, text encoders, and self-supervised speech models have been utilised for this task. However, cascaded systems suffer from the loss of information, while E2E graders also have limitations. With the recent advancements of multi-modal large language models (LLMs), we aim to explore their potential as L2 oral proficiency graders and overcome these issues. In this work, we compare various training strategies using regression and classification targets. Our results show that speech LLMs outperform all previous competitive baselines, achieving superior performance on two datasets. Furthermore, the trained grader demonstrates strong generalisation capabilities in the cross-part or cross-task evaluation, facilitated by the audio understanding knowledge acquired during LLM pre-training.
>
---
#### [new 060] RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱嵌入(KGE)任务，针对实体转换前后嵌入不一致问题，提出RSCF方法。通过共享关系仿射变换、实体嵌入加法及归一化，解决关系转换孤立和语义偏差，增强语义一致性，在补全任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20813v1](http://arxiv.org/pdf/2505.20813v1)**

> **作者:** Junsik Kim; Jinwook Park; Kangil Kim
>
> **备注:** Accepted to ACL 2025, 17 pages, 10 figures
>
> **摘要:** In knowledge graph embedding, leveraging relation-specific entity-transformation has markedly enhanced performance. However, the consistency of embedding differences before and after transformation remains unaddressed, risking the loss of valuable inductive bias inherent in the embeddings. This inconsistency stems from two problems. First, transformation representations are specified for relations in a disconnected manner, allowing dissimilar transformations and corresponding entity-embeddings for similar relations. Second, a generalized plug-in approach as a SFBR (Semantic Filter Based on Relations) disrupts this consistency through excessive concentration of entity embeddings under entity-based regularization, generating indistinguishable score distributions among relations. In this paper, we introduce a plug-in KGE method, Relation-Semantics Consistent Filter (RSCF), containing more consistent entity-transformation characterized by three features: 1) shared affine transformation of relation embeddings across all relations, 2) rooted entity-transformation that adds an entity embedding to its change represented by the transformed vector, and 3) normalization of the change to prevent scale reduction. To amplify the advantages of consistency that preserve semantics on embeddings, RSCF adds relation transformation and prediction modules for enhancing the semantics. In knowledge graph completion tasks with distance-based and tensor decomposition models, RSCF significantly outperforms state-of-the-art KGE methods, showing robustness across all relations and their frequencies.
>
---
#### [new 061] The UD-NewsCrawl Treebank: Reflections and Challenges from a Large-scale Tagalog Syntactic Annotation Project
- **分类: cs.CL**

- **简介: 该论文构建了最大规模的他加禄语树库UD-NewsCrawl（15.6k依存树），解决低资源语言句法分析挑战。通过数据采集、标注流程优化及质量控制建立树库，用Transformer模型评估基线性能，分析他加禄语语法特性对标注的影响，为计算语言学提供资源。**

- **链接: [http://arxiv.org/pdf/2505.20428v1](http://arxiv.org/pdf/2505.20428v1)**

> **作者:** Angelina A. Aquino; Lester James V. Miranda; Elsie Marie T. Or
>
> **备注:** Link to treebank: https://huggingface.co/datasets/UD-Filipino/UD_Tagalog-NewsCrawl ; All authors contributed equally in this work
>
> **摘要:** This paper presents UD-NewsCrawl, the largest Tagalog treebank to date, containing 15.6k trees manually annotated according to the Universal Dependencies framework. We detail our treebank development process, including data collection, pre-processing, manual annotation, and quality assurance procedures. We provide baseline evaluations using multiple transformer-based models to assess the performance of state-of-the-art dependency parsers on Tagalog. We also highlight challenges in the syntactic analysis of Tagalog given its distinctive grammatical properties, and discuss its implications for the annotation of this treebank. We anticipate that UD-NewsCrawl and our baseline model implementations will serve as valuable resources for advancing computational linguistics research in underrepresented languages like Tagalog.
>
---
#### [new 062] Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects
- **分类: cs.CL**

- **简介: 该论文综述多模态对话情感识别（MERC），解决单模态情感识别局限性，通过整合文本、语音、视觉信号提升人机交互情感理解。系统梳理方法、任务、评估策略，分析研究趋势与挑战，指明未来方向。**

- **链接: [http://arxiv.org/pdf/2505.20511v1](http://arxiv.org/pdf/2505.20511v1)**

> **作者:** Chengyan Wu; Yiqiang Cai; Yang Liu; Pengxu Zhu; Yun Xue; Ziwei Gong; Julia Hirschberg; Bolei Ma
>
> **摘要:** While text-based emotion recognition methods have achieved notable success, real-world dialogue systems often demand a more nuanced emotional understanding than any single modality can offer. Multimodal Emotion Recognition in Conversations (MERC) has thus emerged as a crucial direction for enhancing the naturalness and emotional understanding of human-computer interaction. Its goal is to accurately recognize emotions by integrating information from various modalities such as text, speech, and visual signals. This survey offers a systematic overview of MERC, including its motivations, core tasks, representative methods, and evaluation strategies. We further examine recent trends, highlight key challenges, and outline future directions. As interest in emotionally intelligent systems grows, this survey provides timely guidance for advancing MERC research.
>
---
#### [new 063] Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出ConciseR框架，解决LLM推理中过度冗余问题。通过两阶段强化学习：首阶段用GRPO++增强推理能力，次阶段用L-GRPO优化简洁性，仅在推理正确时控制长度，提升推理效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.21178v1](http://arxiv.org/pdf/2505.21178v1)**

> **作者:** Mingyang Song; Mao Zheng
>
> **备注:** Ongoing Work
>
> **摘要:** As test-time scaling becomes a pivotal research frontier in Large Language Models (LLMs) development, contemporary and advanced post-training methodologies increasingly focus on extending the generation length of long Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward DeepSeek R1-like performance. However, recent studies reveal a persistent overthinking phenomenon in state-of-the-art reasoning models, manifesting as excessive redundancy or repetitive thinking patterns in long CoT responses. To address this issue, in this paper, we propose a simple yet effective two-stage reinforcement learning framework for achieving concise reasoning in LLMs, named ConciseR. Specifically, the first stage, using more training steps, aims to incentivize the model's reasoning capabilities via Group Relative Policy Optimization with clip-higher and dynamic sampling components (GRPO++), and the second stage, using fewer training steps, explicitly enforces conciseness and improves efficiency via Length-aware Group Relative Policy Optimization (L-GRPO). Significantly, ConciseR only optimizes response length once all rollouts of a sample are correct, following the "walk before you run" principle. Extensive experimental results demonstrate that our ConciseR model, which generates more concise CoT reasoning responses, outperforms recent state-of-the-art reasoning models with zero RL paradigm across AIME 2024, MATH-500, AMC 2023, Minerva, and Olympiad benchmarks.
>
---
#### [new 064] Beyond Keywords: Evaluating Large Language Model Classification of Nuanced Ableism
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估四大语言模型对自闭症相关细微能及主义的识别能力，发现其依赖关键词匹配而忽略上下文，导致漏检隐含伤害。研究对比模型与人类解释的差异，验证二分类评估的有效性。**

- **链接: [http://arxiv.org/pdf/2505.20500v1](http://arxiv.org/pdf/2505.20500v1)**

> **作者:** Naba Rizvi; Harper Strickland; Saleha Ahmedi; Aekta Kallepalli; Isha Khirwadkar; William Wu; Imani N. S. Munyaka; Nedjma Ousidhoum
>
> **摘要:** Large language models (LLMs) are increasingly used in decision-making tasks like r\'esum\'e screening and content moderation, giving them the power to amplify or suppress certain perspectives. While previous research has identified disability-related biases in LLMs, little is known about how they conceptualize ableism or detect it in text. We evaluate the ability of four LLMs to identify nuanced ableism directed at autistic individuals. We examine the gap between their understanding of relevant terminology and their effectiveness in recognizing ableist content in context. Our results reveal that LLMs can identify autism-related language but often miss harmful or offensive connotations. Further, we conduct a qualitative comparison of human and LLM explanations. We find that LLMs tend to rely on surface-level keyword matching, leading to context misinterpretations, in contrast to human annotators who consider context, speaker identity, and potential impact. On the other hand, both LLMs and humans agree on the annotation scheme, suggesting that a binary classification is adequate for evaluating LLM performance, which is consistent with findings from prior studies involving human annotators.
>
---
#### [new 065] On VLMs for Diverse Tasks in Multimodal Meme Classification
- **分类: cs.CL**

- **简介: 该论文研究多模态表情分类任务，针对讽刺、攻击性及情感分类准确性不足的问题，提出结合视觉语言模型（VLM）与语言模型（LLM）的新方法。通过基准测试VLM的提示策略、评估LoRA微调，并利用VLM生成的图文解释训练小模型，分别提升分类准确率8.34%、3.52%和26.24%。**

- **链接: [http://arxiv.org/pdf/2505.20937v1](http://arxiv.org/pdf/2505.20937v1)**

> **作者:** Deepesh Gavit; Debajyoti Mazumder; Samiran Das; Jasabanta Patro
>
> **备注:** 16 pages
>
> **摘要:** In this paper, we present a comprehensive and systematic analysis of vision-language models (VLMs) for disparate meme classification tasks. We introduced a novel approach that generates a VLM-based understanding of meme images and fine-tunes the LLMs on textual understanding of the embedded meme text for improving the performance. Our contributions are threefold: (1) Benchmarking VLMs with diverse prompting strategies purposely to each sub-task; (2) Evaluating LoRA fine-tuning across all VLM components to assess performance gains; and (3) Proposing a novel approach where detailed meme interpretations generated by VLMs are used to train smaller language models (LLMs), significantly improving classification. The strategy of combining VLMs with LLMs improved the baseline performance by 8.34%, 3.52% and 26.24% for sarcasm, offensive and sentiment classification, respectively. Our results reveal the strengths and limitations of VLMs and present a novel strategy for meme understanding.
>
---
#### [new 066] LMCD: Language Models are Zeroshot Cognitive Diagnosis Learners
- **分类: cs.CL**

- **简介: 该论文属于认知诊断任务，旨在解决传统模型在冷启动场景（如新学生/领域）因数据不足表现差的问题。提出LMCD框架，通过知识扩散生成习题与知识概念的语义关联，结合因果注意力机制融合文本与认知状态，构建学生及习题表征，显著提升冷启动场景下的诊断效果。**

- **链接: [http://arxiv.org/pdf/2505.21239v1](http://arxiv.org/pdf/2505.21239v1)**

> **作者:** Yu He; Zihan Yao; Chentao Song; Tianyu Qi; Jun Liu; Ming Li; Qing Huang
>
> **备注:** work in progress
>
> **摘要:** Cognitive Diagnosis (CD) has become a critical task in AI-empowered education, supporting personalized learning by accurately assessing students' cognitive states. However, traditional CD models often struggle in cold-start scenarios due to the lack of student-exercise interaction data. Recent NLP-based approaches leveraging pre-trained language models (PLMs) have shown promise by utilizing textual features but fail to fully bridge the gap between semantic understanding and cognitive profiling. In this work, we propose Language Models as Zeroshot Cognitive Diagnosis Learners (LMCD), a novel framework designed to handle cold-start challenges by harnessing large language models (LLMs). LMCD operates via two primary phases: (1) Knowledge Diffusion, where LLMs generate enriched contents of exercises and knowledge concepts (KCs), establishing stronger semantic links; and (2) Semantic-Cognitive Fusion, where LLMs employ causal attention mechanisms to integrate textual information and student cognitive states, creating comprehensive profiles for both students and exercises. These representations are efficiently trained with off-the-shelf CD models. Experiments on two real-world datasets demonstrate that LMCD significantly outperforms state-of-the-art methods in both exercise-cold and domain-cold settings. The code is publicly available at https://github.com/TAL-auroraX/LMCD
>
---
#### [new 067] Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大型语言模型（LLMs）可解释性研究，旨在解决其内部推理不透明的问题。提出多尺度流形对齐框架，将潜空间分解为全局、中间、局部语义流形，通过跨尺度映射函数实现几何对齐与信息保留，并结合曲率正则化优化，理论证明KL散度误差有界，提升模型解释性并支持偏见检测等应用。**

- **链接: [http://arxiv.org/pdf/2505.20333v1](http://arxiv.org/pdf/2505.20333v1)**

> **作者:** Yukun Zhang; Qi Dong
>
> **摘要:** Recent advances in Large Language Models (LLMs) have achieved strong performance, yet their internal reasoning remains opaque, limiting interpretability and trust in critical applications. We propose a novel Multi_Scale Manifold Alignment framework that decomposes the latent space into global, intermediate, and local semantic manifolds capturing themes, context, and word-level details. Our method introduces cross_scale mapping functions that jointly enforce geometric alignment (e.g., Procrustes analysis) and information preservation (via mutual information constraints like MINE or VIB). We further incorporate curvature regularization and hyperparameter tuning for stable optimization. Theoretical analysis shows that alignment error, measured by KL divergence, can be bounded under mild assumptions. This framework offers a unified explanation of how LLMs structure multi-scale semantics, advancing interpretability and enabling applications such as bias detection and robustness enhancement.
>
---
#### [new 068] Do LLMs Need to Think in One Language? Correlation between Latent Language and Task Performance
- **分类: cs.CL**

- **简介: 该论文研究LLM的内部语言（latent language）一致性对任务表现的影响。通过多语言提示实验，分析翻译和地缘文化任务，发现保持内部语言一致性并非必需，因模型可于最终层自适应调整以匹配目标语言，削弱一致性影响。**

- **链接: [http://arxiv.org/pdf/2505.21458v1](http://arxiv.org/pdf/2505.21458v1)**

> **作者:** Shintaro Ozaki; Tatsuya Hiraoka; Hiroto Otake; Hiroki Ouchi; Masaru Isonuma; Benjamin Heinzerling; Kentaro Inui; Taro Watanabe; Yusuke Miyao; Yohei Oseki; Yu Takagi
>
> **摘要:** Large Language Models (LLMs) are known to process information using a proficient internal language consistently, referred to as latent language, which may differ from the input or output languages. However, how the discrepancy between the latent language and the input and output language affects downstream task performance remains largely unexplored. While many studies research the latent language of LLMs, few address its importance in influencing task performance. In our study, we hypothesize that thinking in latent language consistently enhances downstream task performance. To validate this, our work varies the input prompt languages across multiple downstream tasks and analyzes the correlation between consistency in latent language and task performance. We create datasets consisting of questions from diverse domains such as translation and geo-culture, which are influenced by the choice of latent language. Experimental results across multiple LLMs on translation and geo-culture tasks, which are sensitive to the choice of language, indicate that maintaining consistency in latent language is not always necessary for optimal downstream task performance. This is because these models adapt their internal representations near the final layers to match the target language, reducing the impact of consistency on overall performance.
>
---
#### [new 069] The NaijaVoices Dataset: Cultivating Large-Scale, High-Quality, Culturally-Rich Speech Data for African Languages
- **分类: cs.CL**

- **简介: 该论文属于语音数据集构建任务，旨在解决非洲语言（如伊博、豪萨、约鲁巴语）语音技术数据匮乏问题。团队创建了包含1800小时、5000+说话者的NaijaVoices数据集，通过多样化数据收集和模型微调实验（提升WER 42.33%-75.86%），推动非洲语言多语种语音处理发展。**

- **链接: [http://arxiv.org/pdf/2505.20564v1](http://arxiv.org/pdf/2505.20564v1)**

> **作者:** Chris Emezue; The NaijaVoices Community; Busayo Awobade; Abraham Owodunni; Handel Emezue; Gloria Monica Tobechukwu Emezue; Nefertiti Nneoma Emezue; Sewade Ogun; Bunmi Akinremi; David Ifeoluwa Adelani; Chris Pal
>
> **备注:** Accepted for publication at Interspeech 2025
>
> **摘要:** The development of high-performing, robust, and reliable speech technologies depends on large, high-quality datasets. However, African languages -- including our focus, Igbo, Hausa, and Yoruba -- remain under-represented due to insufficient data. Popular voice-enabled technologies do not support any of the 2000+ African languages, limiting accessibility for circa one billion people. While previous dataset efforts exist for the target languages, they lack the scale and diversity needed for robust speech models. To bridge this gap, we introduce the NaijaVoices dataset, a 1,800-hour speech-text dataset with 5,000+ speakers. We outline our unique data collection approach, analyze its acoustic diversity, and demonstrate its impact through finetuning experiments on automatic speech recognition, averagely achieving 75.86% (Whisper), 52.06% (MMS), and 42.33% (XLSR) WER improvements. These results highlight NaijaVoices' potential to advance multilingual speech processing for African languages.
>
---
#### [new 070] UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出UI-Genie框架，针对移动GUI代理中轨迹验证困难和数据不足问题，设计了融合图像文本的奖励模型UI-Genie-RM，并构建自我改进 pipeline，通过奖励引导和动态环境验证迭代提升模型性能，生成高质量合成数据集，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.21496v1](http://arxiv.org/pdf/2505.21496v1)**

> **作者:** Han Xiao; Guozhi Wang; Yuxiang Chai; Zimu Lu; Weifeng Lin; Hao He; Lue Fan; Liuyang Bian; Rui Hu; Liang Liu; Shuai Ren; Yafei Wen; Xiaoxin Chen; Aojun Zhou; Hongsheng Li
>
> **备注:** https://github.com/Euphoria16/UI-Genie
>
> **摘要:** In this paper, we introduce UI-Genie, a self-improving framework addressing two key challenges in GUI agents: verification of trajectory outcome is challenging and high-quality training data are not scalable. These challenges are addressed by a reward model and a self-improving pipeline, respectively. The reward model, UI-Genie-RM, features an image-text interleaved architecture that efficiently pro- cesses historical context and unifies action-level and task-level rewards. To sup- port the training of UI-Genie-RM, we develop deliberately-designed data genera- tion strategies including rule-based verification, controlled trajectory corruption, and hard negative mining. To address the second challenge, a self-improvement pipeline progressively expands solvable complex GUI tasks by enhancing both the agent and reward models through reward-guided exploration and outcome verification in dynamic environments. For training the model, we generate UI- Genie-RM-517k and UI-Genie-Agent-16k, establishing the first reward-specific dataset for GUI agents while demonstrating high-quality synthetic trajectory gen- eration without manual annotation. Experimental results show that UI-Genie achieves state-of-the-art performance across multiple GUI agent benchmarks with three generations of data-model self-improvement. We open-source our complete framework implementation and generated datasets to facilitate further research in https://github.com/Euphoria16/UI-Genie.
>
---
#### [new 071] Test-Time Learning for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出测试时学习（TLM）方法，解决大语言模型在领域迁移和分布偏移中的性能下降问题。通过最小化测试数据输入困惑度、主动选择高困惑度样本及采用低秩适配技术，在不显著遗忘原有知识前提下提升模型适应性，实验显示性能提升超20%。**

- **链接: [http://arxiv.org/pdf/2505.20633v1](http://arxiv.org/pdf/2505.20633v1)**

> **作者:** Jinwu Hu; Zhitian Zhang; Guohao Chen; Xutao Wen; Chao Shuai; Wei Luo; Bin Xiao; Yuanqing Li; Mingkui Tan
>
> **备注:** Accepted by ICML2025
>
> **摘要:** While Large Language Models (LLMs) have exhibited remarkable emergent capabilities through extensive pre-training, they still face critical limitations in generalizing to specialized domains and handling diverse linguistic variations, known as distribution shifts. In this paper, we propose a Test-Time Learning (TTL) paradigm for LLMs, namely TLM, which dynamically adapts LLMs to target domains using only unlabeled test data during testing. Specifically, we first provide empirical evidence and theoretical insights to reveal that more accurate predictions from LLMs can be achieved by minimizing the input perplexity of the unlabeled test data. Based on this insight, we formulate the Test-Time Learning process of LLMs as input perplexity minimization, enabling self-supervised enhancement of LLM performance. Furthermore, we observe that high-perplexity samples tend to be more informative for model optimization. Accordingly, we introduce a Sample Efficient Learning Strategy that actively selects and emphasizes these high-perplexity samples for test-time updates. Lastly, to mitigate catastrophic forgetting and ensure adaptation stability, we adopt Low-Rank Adaptation (LoRA) instead of full-parameter optimization, which allows lightweight model updates while preserving more original knowledge from the model. We introduce the AdaptEval benchmark for TTL and demonstrate through experiments that TLM improves performance by at least 20% compared to original LLMs on domain knowledge adaptation.
>
---
#### [new 072] Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对长文本问答任务，提出RioRAG框架，通过强化学习优化生成的信息量与事实准确性，解决长答案生成中数据不足、幻觉风险及评估困难问题。其创新包括基于信息量的RL训练和分层奖励模型，通过关键点提取与事实核查提升生成质量。**

- **链接: [http://arxiv.org/pdf/2505.20825v1](http://arxiv.org/pdf/2505.20825v1)**

> **作者:** Yuhao Wang; Ruiyang Ren; Yucheng Wang; Wayne Xin Zhao; Jing Liu; Hua Wu; Haifeng Wang
>
> **摘要:** Long-form question answering (LFQA) presents unique challenges for large language models, requiring the synthesis of coherent, paragraph-length answers. While retrieval-augmented generation (RAG) systems have emerged as a promising solution, existing research struggles with key limitations: the scarcity of high-quality training data for long-form generation, the compounding risk of hallucination in extended outputs, and the absence of reliable evaluation metrics for factual completeness. In this paper, we propose RioRAG, a novel reinforcement learning (RL) framework that advances long-form RAG through reinforced informativeness optimization. Our approach introduces two fundamental innovations to address the core challenges. First, we develop an RL training paradigm of reinforced informativeness optimization that directly optimizes informativeness and effectively addresses the slow-thinking deficit in conventional RAG systems, bypassing the need for expensive supervised data. Second, we propose a nugget-centric hierarchical reward modeling approach that enables precise assessment of long-form answers through a three-stage process: extracting the nugget from every source webpage, constructing a nugget claim checklist, and computing rewards based on factual alignment. Extensive experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the effectiveness of the proposed method. Our codes are available at https://github.com/RUCAIBox/RioRAG.
>
---
#### [new 073] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CogniBench框架及数据集，解决LLM认知忠实度评估问题。针对现有基准无法区分事实陈述与推理性认知陈述的问题，借鉴法律证据评估方法，设计多层级忠实度评估框架，并构建可扩展的标注数据集，用于训练认知幻觉检测模型。**

- **链接: [http://arxiv.org/pdf/2505.20767v1](http://arxiv.org/pdf/2505.20767v1)**

> **作者:** Xiaqiang Tang; Jian Li; Keyu Hu; Du Nan; Xiaolong Li; Xi Zhang; Weigao Sun; Sihong Xie
>
> **备注:** ACL 2025
>
> **摘要:** Faithfulness hallucination are claims generated by a Large Language Model (LLM) not supported by contexts provided to the LLM. Lacking assessment standard, existing benchmarks only contain "factual statements" that rephrase source materials without marking "cognitive statements" that make inference from the given context, making the consistency evaluation and optimization of cognitive statements difficult. Inspired by how an evidence is assessed in the legislative domain, we design a rigorous framework to assess different levels of faithfulness of cognitive statements and create a benchmark dataset where we reveal insightful statistics. We design an annotation pipeline to create larger benchmarks for different LLMs automatically, and the resulting larger-scale CogniBench-L dataset can be used to train accurate cognitive hallucination detection model. We release our model and dataset at: https://github.com/FUTUREEEEEE/CogniBench
>
---
#### [new 074] ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出ArVoice数据集，用于多说话者阿拉伯语语音合成，解决该领域高质量多说话者数据缺乏的问题。包含专业录制、现有数据集改进及合成语音，共83.5小时（11种声音），并训练TTS及声音转换系统验证其应用，推动语音合成及相关研究。**

- **链接: [http://arxiv.org/pdf/2505.20506v1](http://arxiv.org/pdf/2505.20506v1)**

> **作者:** Hawau Olamide Toyin; Rufael Marew; Humaid Alblooshi; Samar M. Magdy; Hanan Aldarmaki
>
> **备注:** Accepted at INTERSPEECH 2025 The dataset is available at https://huggingface.co/datasets/MBZUAI/ArVoice
>
> **摘要:** We introduce ArVoice, a multi-speaker Modern Standard Arabic (MSA) speech corpus with diacritized transcriptions, intended for multi-speaker speech synthesis, and can be useful for other tasks such as speech-based diacritic restoration, voice conversion, and deepfake detection. ArVoice comprises: (1) a new professionally recorded set from six voice talents with diverse demographics, (2) a modified subset of the Arabic Speech Corpus; and (3) high-quality synthetic speech from two commercial systems. The complete corpus consists of a total of 83.52 hours of speech across 11 voices; around 10 hours consist of human voices from 7 speakers. We train three open-source TTS and two voice conversion systems to illustrate the use cases of the dataset. The corpus is available for research use.
>
---
#### [new 075] Pretraining Language Models to Ponder in Continuous Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出在语言模型中引入“思考”机制（Pondering），通过在单个token生成时多次前向传递，利用预测分布的加权嵌入反馈替代直接采样，提升复杂语言生成的深度处理能力。属于预训练语言模型优化任务，旨在提高参数效率与生成质量。实验显示其在少参数下性能接近大模型，并在多个下游任务中表现优异，验证了方法的通用性。**

- **链接: [http://arxiv.org/pdf/2505.20674v1](http://arxiv.org/pdf/2505.20674v1)**

> **作者:** Boyi Zeng; Shixiang Song; Siyuan Huang; Yixuan Wang; He Li; Ziwei He; Xinbing Wang; Zhiyu Li; Zhouhan Lin
>
> **摘要:** Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort. In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations. Our method is straightforward and can be seamlessly integrated with various existing language models. Experiments across three widely used open-source architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. For language modeling tasks, pondering language models achieve performance comparable to vanilla models with twice the number of parameters. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, pondering-enhanced Pythia-1B is comparable to TinyLlama-1.1B, which is trained on 10 times more data. The code is available at https://github.com/LUMIA-Group/PonderingLM.
>
---
#### [new 076] Phir Hera Fairy: An English Fairytaler is a Strong Faker of Fluent Speech in Low-Resource Indian Languages
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在解决低资源印度语言的语音生成问题。通过对比三种微调策略（从头训练、仅印度数据微调、英印混合微调），发现仅用印度数据微调英文F5模型（获IN-F5）效果最佳，实现多语言流畅合成、声音/风格迁移及零资源语言（如Bhojpuri）的生成，证明英文预训练对低资源TTS达人类水平的关键作用，并提出数据受限场景的计算优化方案。**

- **链接: [http://arxiv.org/pdf/2505.20693v1](http://arxiv.org/pdf/2505.20693v1)**

> **作者:** Praveen Srinivasa Varadhan; Srija Anand; Soma Siddhartha; Mitesh M. Khapra
>
> **摘要:** What happens when an English Fairytaler is fine-tuned on Indian languages? We evaluate how the English F5-TTS model adapts to 11 Indian languages, measuring polyglot fluency, voice-cloning, style-cloning, and code-mixing. We compare: (i) training from scratch, (ii) fine-tuning English F5 on Indian data, and (iii) fine-tuning on both Indian and English data to prevent forgetting. Fine-tuning with only Indian data proves most effective and the resultant IN-F5 is a near-human polyglot; that enables speakers of one language (e.g., Odia) to fluently speak in another (e.g., Hindi). Our results show English pretraining aids low-resource TTS in reaching human parity. To aid progress in other low-resource languages, we study data-constrained setups and arrive at a compute optimal strategy. Finally, we show IN-F5 can synthesize unseen languages like Bhojpuri and Tulu using a human-in-the-loop approach for zero-resource TTS via synthetic data generation.
>
---
#### [new 077] GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM监督微调任务，旨在解决高质量训练数据稀缺及现有合成数据存在事实错误、长尾覆盖不足、结构简单等问题。提出GraphGen框架，通过构建知识图谱识别模型知识缺口，利用多跳采样和风格控制生成多样化、结构复杂的QA对，提升合成数据质量与覆盖范围。**

- **链接: [http://arxiv.org/pdf/2505.20416v1](http://arxiv.org/pdf/2505.20416v1)**

> **作者:** Zihong Chen; Wanli Jiang; Jinzhe Li; Zhonghang Yuan; Huanjun Kong; Wanli Ouyang; Nanqing Dong
>
> **摘要:** Fine-tuning for large language models (LLMs) typically requires substantial amounts of high-quality supervised data, which is both costly and labor-intensive to acquire. While synthetic data generation has emerged as a promising solution, existing approaches frequently suffer from factual inaccuracies, insufficient long-tail coverage, simplistic knowledge structures, and homogenized outputs. To address these challenges, we introduce GraphGen, a knowledge graph-guided framework designed for three key question-answering (QA) scenarios: atomic QA, aggregated QA, and multi-hop QA. It begins by constructing a fine-grained knowledge graph from the source text. It then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge. Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data. Experimental results on knowledge-intensive tasks under closed-book settings demonstrate that GraphGen outperforms conventional synthetic data methods, offering a more reliable and comprehensive solution to the data scarcity challenge in supervised fine-tuning. The code and data are publicly available at https://github.com/open-sciencelab/GraphGen.
>
---
#### [new 078] Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对文本到SQL生成任务，解决生成正确复杂查询的瓶颈。提出Arctic-Text2SQL-R1框架，采用基于执行正确性的轻量强化学习，避免复杂奖励设计，结合优质数据和训练策略，在六项基准中达SOTA，7B模型超越70B系统，展现高效与可扩展性。**

- **链接: [http://arxiv.org/pdf/2505.20315v1](http://arxiv.org/pdf/2505.20315v1)**

> **作者:** Zhewei Yao; Guoheng Sun; Lukasz Borchmann; Zheyu Shen; Minghang Deng; Bohan Zhai; Hao Zhang; Ang Li; Yuxiong He
>
> **备注:** 22 pages, 2 figures
>
> **摘要:** Translating natural language into SQL (Test2SQL) is a longstanding challenge at the intersection of natural language understanding and structured data access. While large language models (LLMs) have significantly improved fluency in SQL generation, producing correct and executable SQL--particularly for complex queries--remains a bottleneck. We present Arctic-Text2SQL-R1, a reinforcement learning (RL) framework and model family designed to generate accurate, executable SQL using a lightweight reward signal based solely on execution correctness. Our approach avoids brittle intermediate supervision and complex reward shaping, promoting stable training and alignment with the end task. Combined with carefully curated data, strong supervised initialization, and effective training practices, Arctic-Text2SQL-R1 achieves state-of-the-art execution accuracy across six diverse Test2SQL benchmarks, including the top position on the BIRD leaderboard. Notably, our 7B model outperforms prior 70B-class systems, highlighting the framework's scalability and efficiency. We further demonstrate inference-time robustness through simple extensions like value retrieval and majority voting. Extensive experiments and ablation studies offer both positive and negative insights, providing practical guidance for future Test2SQL research.
>
---
#### [new 079] LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出RPM框架，针对黑盒大语言模型推理过程缺乏个性化的问题，通过提取用户历史中的特征构建个性化推理路径，在推理时结合路径与相似案例，使模型遵循用户逻辑。任务为黑盒LLM的推理级个性化，解决现有方法仅关注输出匹配而忽视思维过程适配的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.21082v1](http://arxiv.org/pdf/2505.21082v1)**

> **作者:** Jieyong Kim; Tongyoung Kim; Soonjin Yoon; Jaehyung Kim; Dongha Lee
>
> **摘要:** Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
>
---
#### [new 080] Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation
- **分类: cs.CL**

- **简介: 该论文属于对话主题分割（DTS）任务，解决数据短缺、标注模糊及方法复杂性问题。提出Def-DTS，利用LLM多步骤演绎推理，通过双向上下文摘要、通用意图分类与主题转移检测，提升分割效果并减少第二类错误，同时探索自动标注潜力。**

- **链接: [http://arxiv.org/pdf/2505.21033v1](http://arxiv.org/pdf/2505.21033v1)**

> **作者:** Seungmin Lee; Yongsang Yoo; Minhwa Jung; Min Song
>
> **备注:** 19 pages, 3 figures, Accepted to Findings of the ACL 2025
>
> **摘要:** Dialogue Topic Segmentation (DTS) aims to divide dialogues into coherent segments. DTS plays a crucial role in various NLP downstream tasks, but suffers from chronic problems: data shortage, labeling ambiguity, and incremental complexity of recently proposed solutions. On the other hand, Despite advances in Large Language Models (LLMs) and reasoning strategies, these have rarely been applied to DTS. This paper introduces Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation, which utilizes LLM-based multi-step deductive reasoning to enhance DTS performance and enable case study using intermediate result. Our method employs a structured prompting approach for bidirectional context summarization, utterance intent classification, and deductive topic shift detection. In the intent classification process, we propose the generalizable intent list for domain-agnostic dialogue intent classification. Experiments in various dialogue settings demonstrate that Def-DTS consistently outperforms traditional and state-of-the-art approaches, with each subtask contributing to improved performance, particularly in reducing type 2 error. We also explore the potential for autolabeling, emphasizing the importance of LLM reasoning techniques in DTS.
>
---
#### [new 081] Improving Research Idea Generation Through Data: An Empirical Investigation in Social Science
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属研究辅助任务，解决LLM生成研究想法的可行性与有效性不足问题。提出两种方法：生成阶段用元数据引导、选择阶段用自动验证，通过气候谈判实验验证，元数据提升20%可行性，验证提升7%质量，并经人类研究验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.21396v1](http://arxiv.org/pdf/2505.21396v1)**

> **作者:** Xiao Liu; Xinyi Dong; Xinyang Gao; Yansong Feng; Xun Pang
>
> **摘要:** Recent advancements in large language models (LLMs) have shown promise in generating novel research ideas. However, these ideas often face challenges related to feasibility and expected effectiveness. This paper explores how augmenting LLMs with relevant data during the idea generation process can enhance the quality of generated ideas. We introduce two ways of incorporating data: (1) providing metadata during the idea generation stage to guide LLMs toward feasible directions, and (2) adding automatic validation during the idea selection stage to assess the empirical plausibility of hypotheses within ideas. We conduct experiments in the social science domain, specifically with climate negotiation topics, and find that metadata improves the feasibility of generated ideas by 20%, while automatic validation improves the overall quality of selected ideas by 7%. A human study shows that LLM-generated ideas, along with their related data and validation processes, inspire researchers to propose research ideas with higher quality. Our work highlights the potential of data-driven research idea generation, and underscores the practical utility of LLM-assisted ideation in real-world academic settings.
>
---
#### [new 082] Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦中文方言及口音的语音识别任务，针对其数据稀缺问题，提出结合自监督预训练（Data2vec2）与LLM的方法。通过30万小时无标签方言数据预训练及4万小时监督数据对齐训练，系统评估不同投影器和LLM对识别效果的影响，实现多数据集SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.21138v1](http://arxiv.org/pdf/2505.21138v1)**

> **作者:** Tianyi Xu; Hongjie Chen; Wang Qing; Lv Hang; Jian Kang; Li Jie; Zhennan Lin; Yongxiang Li; Xie Lei
>
> **摘要:** Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre- training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
>
---
#### [new 083] In-context Language Learning for Endangered Languages in Speech Recognition
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨通过ICL（上下文学习）提升LLMs对未接触的濒危语言语音识别能力。针对低资源语言支持不足问题，实验表明增加相关文本样本可优化语言建模和ASR效果，概率方法优于传统指令方法，且ICL使LLMs的ASR性能可匹敌专用模型，同时保留其原有能力。**

- **链接: [http://arxiv.org/pdf/2505.20445v1](http://arxiv.org/pdf/2505.20445v1)**

> **作者:** Zhaolin Li; Jan Niehues
>
> **摘要:** With approximately 7,000 languages spoken worldwide, current large language models (LLMs) support only a small subset. Prior research indicates LLMs can learn new languages for certain tasks without supervised data. We extend this investigation to speech recognition, investigating whether LLMs can learn unseen, low-resource languages through in-context learning (ICL). With experiments on four diverse endangered languages that LLMs have not been trained on, we find that providing more relevant text samples enhances performance in both language modelling and Automatic Speech Recognition (ASR) tasks. Furthermore, we show that the probability-based approach outperforms the traditional instruction-based approach in language learning. Lastly, we show ICL enables LLMs to achieve ASR performance that is comparable to or even surpasses dedicated language models trained specifically for these languages, while preserving the original capabilities of the LLMs.
>
---
#### [new 084] SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SeqPO-SiMT框架，针对同时机器翻译任务，解决翻译质量与低延迟的平衡问题。通过将SiMT建模为序列决策问题，设计定制奖励函数优化策略，提升7B模型在多语言任务中的翻译质量（COMET提升1.13）并降低延迟（平均滞后减少6.17），效果媲美更大离线模型。**

- **链接: [http://arxiv.org/pdf/2505.20622v1](http://arxiv.org/pdf/2505.20622v1)**

> **作者:** Ting Xu; Zhichao Huang; Jiankai Sun; Shanbo Cheng; Wai Lam
>
> **备注:** Accepted by The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** We present Sequential Policy Optimization for Simultaneous Machine Translation (SeqPO-SiMT), a new policy optimization framework that defines the simultaneous machine translation (SiMT) task as a sequential decision making problem, incorporating a tailored reward to enhance translation quality while reducing latency. In contrast to popular Reinforcement Learning from Human Feedback (RLHF) methods, such as PPO and DPO, which are typically applied in single-step tasks, SeqPO-SiMT effectively tackles the multi-step SiMT task. This intuitive framework allows the SiMT LLMs to simulate and refine the SiMT process using a tailored reward. We conduct experiments on six datasets from diverse domains for En to Zh and Zh to En SiMT tasks, demonstrating that SeqPO-SiMT consistently achieves significantly higher translation quality with lower latency. In particular, SeqPO-SiMT outperforms the supervised fine-tuning (SFT) model by 1.13 points in COMET, while reducing the Average Lagging by 6.17 in the NEWSTEST2021 En to Zh dataset. While SiMT operates with far less context than offline translation, the SiMT results of SeqPO-SiMT on 7B LLM surprisingly rival the offline translation of high-performing LLMs, including Qwen-2.5-7B-Instruct and LLaMA-3-8B-Instruct.
>
---
#### [new 085] SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SeRL方法，解决在有限数据下提升LLM推理能力的问题。通过自我指令生成（在线过滤保证质量）和多数投票自奖励机制（无需人工标注），实现自举强化学习，实验显示其效果媲美高质量数据训练。**

- **链接: [http://arxiv.org/pdf/2505.20347v1](http://arxiv.org/pdf/2505.20347v1)**

> **作者:** Wenkai Fang; Shunyu Liu; Yang Zhou; Kongcheng Zhang; Tongya Zheng; Kaixuan Chen; Mingli Song; Dacheng Tao
>
> **摘要:** Recent advances have demonstrated the effectiveness of Reinforcement Learning (RL) in improving the reasoning capabilities of Large Language Models (LLMs). However, existing works inevitably rely on high-quality instructions and verifiable rewards for effective training, both of which are often difficult to obtain in specialized domains. In this paper, we propose Self-play Reinforcement Learning(SeRL) to bootstrap LLM training with limited initial data. Specifically, SeRL comprises two complementary modules: self-instruction and self-rewarding. The former module generates additional instructions based on the available data at each training step, employing robust online filtering strategies to ensure instruction quality, diversity, and difficulty. The latter module introduces a simple yet effective majority-voting mechanism to estimate response rewards for additional instructions, eliminating the need for external annotations. Finally, SeRL performs conventional RL based on the generated data, facilitating iterative self-play learning. Extensive experiments on various reasoning benchmarks and across different LLM backbones demonstrate that the proposed SeRL yields results superior to its counterparts and achieves performance on par with those obtained by high-quality data with verifiable rewards. Our code is available at https://github.com/wantbook-book/SeRL.
>
---
#### [new 086] Pretrained LLMs Learn Multiple Types of Uncertainty
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型不确定性分析任务，研究预训练LLMs如何内在学习多种不确定性以减少幻觉错误。通过揭示LLMs潜在空间中存在可区分的不确定性类型（如任务特异性、校正预测关联性），发现模型规模对此影响有限，并提出统一不确定性类型（如指令调优）可提升正确性预测。**

- **链接: [http://arxiv.org/pdf/2505.21218v1](http://arxiv.org/pdf/2505.21218v1)**

> **作者:** Roi Cohen; Omri Fahn; Gerard de Melo
>
> **摘要:** Large Language Models are known to capture real-world knowledge, allowing them to excel in many downstream tasks. Despite recent advances, these models are still prone to what are commonly known as hallucinations, causing them to emit unwanted and factually incorrect text. In this work, we study how well LLMs capture uncertainty, without explicitly being trained for that. We show that, if considering uncertainty as a linear concept in the model's latent space, it might indeed be captured, even after only pretraining. We further show that, though unintuitive, LLMs appear to capture several different types of uncertainty, each of which can be useful to predict the correctness for a specific task or benchmark. Furthermore, we provide in-depth results such as demonstrating a correlation between our correction prediction and the model's ability to abstain from misinformation using words, and the lack of impact of model scaling for capturing uncertainty. Finally, we claim that unifying the uncertainty types as a single one using instruction-tuning or [IDK]-token tuning is helpful for the model in terms of correctness prediction.
>
---
#### [new 087] TAT-R1: Terminology-Aware Translation with Reinforcement Learning and Word Alignment
- **分类: cs.CL**

- **简介: 该论文提出TAT-R1模型，针对机器翻译中术语翻译不足的问题，结合强化学习与词对齐技术。通过提取关键词对并设计对齐奖励机制，使模型聚焦术语准确翻译，实验显示术语准确率显著提升，同时保持通用翻译性能。**

- **链接: [http://arxiv.org/pdf/2505.21172v1](http://arxiv.org/pdf/2505.21172v1)**

> **作者:** Zheng Li; Mao Zheng; Mingyang Song; Wenjie Yang
>
> **摘要:** Recently, deep reasoning large language models(LLMs) like DeepSeek-R1 have made significant progress in tasks such as mathematics and coding. Inspired by this, several studies have employed reinforcement learning(RL) to enhance models' deep reasoning capabilities and improve machine translation(MT) quality. However, the terminology translation, an essential task in MT, remains unexplored in deep reasoning LLMs. In this paper, we propose \textbf{TAT-R1}, a terminology-aware translation model trained with reinforcement learning and word alignment. Specifically, we first extract the keyword translation pairs using a word alignment model. Then we carefully design three types of rule-based alignment rewards with the extracted alignment relationships. With those alignment rewards, the RL-trained translation model can learn to focus on the accurate translation of key information, including terminology in the source text. Experimental results show the effectiveness of TAT-R1. Our model significantly improves terminology translation accuracy compared to the baseline models while maintaining comparable performance on general translation tasks. In addition, we conduct detailed ablation studies of the DeepSeek-R1-like training paradigm for machine translation and reveal several key findings.
>
---
#### [new 088] STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Steer-Bench基准，评估大语言模型（LLMs）适应不同社区规范的能力。针对模型在社区特定引导上的不足，构建覆盖30对Reddit社区、1.05万测试样本的基准，测试13种模型发现其表现较人类专家（81%）低15-16个百分点，凸显模型在社区敏感可控性上的差距。**

- **链接: [http://arxiv.org/pdf/2505.20645v1](http://arxiv.org/pdf/2505.20645v1)**

> **作者:** Kai Chen; Zihao He; Taiwei Shi; Kristina Lerman
>
> **摘要:** Steerability, or the ability of large language models (LLMs) to adapt outputs to align with diverse community-specific norms, perspectives, and communication styles, is critical for real-world applications but remains under-evaluated. We introduce Steer-Bench, a benchmark for assessing population-specific steering using contrasting Reddit communities. Covering 30 contrasting subreddit pairs across 19 domains, Steer-Bench includes over 10,000 instruction-response pairs and validated 5,500 multiple-choice question with corresponding silver labels to test alignment with diverse community norms. Our evaluation of 13 popular LLMs using Steer-Bench reveals that while human experts achieve an accuracy of 81% with silver labels, the best-performing models reach only around 65% accuracy depending on the domain and configuration. Some models lag behind human-level alignment by over 15 percentage points, highlighting significant gaps in community-sensitive steerability. Steer-Bench is a benchmark to systematically assess how effectively LLMs understand community-specific instructions, their resilience to adversarial steering attempts, and their ability to accurately represent diverse cultural and ideological perspectives.
>
---
#### [new 089] Paths Not Taken: Understanding and Mending the Multilingual Factual Recall Pipeline
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型的事实回忆任务，解决跨语言事实一致性问题。通过分析发现模型依赖英语处理流程再翻译，存在英语机制参与不足和翻译错误两大缺陷。提出两种语言无关的向量干预方法，使低效语言准确率提升35%以上，验证机制分析对提升多语言能力的作用。**

- **链接: [http://arxiv.org/pdf/2505.20546v1](http://arxiv.org/pdf/2505.20546v1)**

> **作者:** Meng Lu; Ruochen Zhang; Ellie Pavlick; Carsten Eickhoff
>
> **摘要:** Multilingual large language models (LLMs) often exhibit factual inconsistencies across languages, with significantly better performance in factual recall tasks in English than in other languages. The causes of these failures, however, remain poorly understood. Using mechanistic analysis techniques, we uncover the underlying pipeline that LLMs employ, which involves using the English-centric factual recall mechanism to process multilingual queries and then translating English answers back into the target language. We identify two primary sources of error: insufficient engagement of the reliable English-centric mechanism for factual recall, and incorrect translation from English back into the target language for the final answer. To address these vulnerabilities, we introduce two vector interventions, both independent of languages and datasets, to redirect the model toward better internal paths for higher factual consistency. Our interventions combined increase the recall accuracy by over 35 percent for the lowest-performing language. Our findings demonstrate how mechanistic insights can be used to unlock latent multilingual capabilities in LLMs.
>
---
#### [new 090] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于LLM行为控制任务，旨在解决参数纠缠导致的控制精度不足和副作用问题。提出Steering Target Atoms（STA）方法，通过分离并操纵知识组件实现精准安全控制，在对抗场景及复杂推理中验证了其鲁棒性与灵活性。**

- **链接: [http://arxiv.org/pdf/2505.20322v1](http://arxiv.org/pdf/2505.20322v1)**

> **作者:** Mengru Wang; Ziwen Xu; Shengyu Mao; Shumin Deng; Zhaopeng Tu; Huajun Chen; Ningyu Zhang
>
> **摘要:** Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control.
>
---
#### [new 091] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration
- **分类: cs.CL**

- **简介: 该论文提出ExtAgents框架，解决LLM因上下文窗口限制无法有效处理大量外部知识的问题。通过优化多智能体协作中的知识同步与推理，提升复杂任务性能，无需延长训练上下文，在多跳QA等测试中表现优异，高效且适用于超长输入场景。**

- **链接: [http://arxiv.org/pdf/2505.21471v1](http://arxiv.org/pdf/2505.21471v1)**

> **作者:** Zijun Liu; Zhennan Wan; Peng Li; Ming Yan; Ji Zhang; Fei Huang; Yang Liu
>
> **备注:** 30 pages, 9 figures. Code and data are available at https://github.com/THUNLP-MT/ExtAgents
>
> **摘要:** With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications.
>
---
#### [new 092] Beyond Demonstrations: Dynamic Vector Construction from Latent Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于少样本学习任务，旨在解决现有ICV方法对上下文学习敏感、表示粗糙及注入位置固定的缺陷。提出DyVec方法，通过 Exhaustive Query Rotation 提取稳健语义聚合表示，动态分割表示并优化注入位置，提升推理效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.20318v1](http://arxiv.org/pdf/2505.20318v1)**

> **作者:** Wang Cai; Hsiu-Yuan Huang; Zhixiang Wang; Yunfang Wu
>
> **摘要:** In-Context derived Vector (ICV) methods extract task-relevant representations from large language models (LLMs) and reinject them during inference, achieving comparable performance to few-shot In-Context Learning (ICL) without repeated demonstration processing. However, existing ICV methods remain sensitive to ICL-specific factors, often use coarse or semantically fragmented representations as the source of the vector, and rely on heuristic-based injection positions, limiting their applicability. To address these issues, we propose Dynamic Vector (DyVec), which incorporates an Exhaustive Query Rotation (EQR) strategy to extract robust semantically aggregated latent representations by mitigating variance introduced by ICL. It then applies Dynamic Latent Segmentation and Injection to adaptively partition representations based on task complexity and leverages REINFORCE-based optimization to learn optimal injection positions for each segment. Experiments results show that DyVec outperforms few-shot ICL, LoRA, and prior ICV baselines. Further analysis highlights the effectiveness of dynamically segmenting and injecting semantically aggregated latent representations. DyVec provides a lightweight and data-efficient solution for inference-time task adaptation.
>
---
#### [new 093] BLUCK: A Benchmark Dataset for Bengali Linguistic Understanding and Cultural Knowledge
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BLUCK数据集，评估LLMs在孟加拉语语言与文化知识的表现。解决现有模型在该领域评估不足的问题。构建2366题、23类别的MCQ数据集，测试9个模型，发现其在语音方面较弱，显示孟加拉语为中等资源语言，填补文化基准空白。**

- **链接: [http://arxiv.org/pdf/2505.21092v1](http://arxiv.org/pdf/2505.21092v1)**

> **作者:** Daeen Kabir; Minhajur Rahman Chowdhury Mahim; Sheikh Shafayat; Adnan Sadik; Arian Ahmed; Eunsu Kim; Alice Oh
>
> **摘要:** In this work, we introduce BLUCK, a new dataset designed to measure the performance of Large Language Models (LLMs) in Bengali linguistic understanding and cultural knowledge. Our dataset comprises 2366 multiple-choice questions (MCQs) carefully curated from compiled collections of several college and job level examinations and spans 23 categories covering knowledge on Bangladesh's culture and history and Bengali linguistics. We benchmarked BLUCK using 6 proprietary and 3 open-source LLMs - including GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Llama-3.3-70B-Instruct, and DeepSeekV3. Our results show that while these models perform reasonably well overall, they, however, struggles in some areas of Bengali phonetics. Although current LLMs' performance on Bengali cultural and linguistic contexts is still not comparable to that of mainstream languages like English, our results indicate Bengali's status as a mid-resource language. Importantly, BLUCK is also the first MCQ-based evaluation benchmark that is centered around native Bengali culture, history, and linguistics.
>
---
#### [new 094] Research Community Perspectives on "Intelligence" and Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文通过调查303名研究人员，探讨"智能"定义及其在NLP研究中的角色。任务为澄清智能标准与研究目标关联。发现共识的三个智能标准：泛化、适应性、推理。显示仅29%认为当前系统智能，16.2%以开发智能系统为目标，且二者相关。**

- **链接: [http://arxiv.org/pdf/2505.20959v1](http://arxiv.org/pdf/2505.20959v1)**

> **作者:** Bertram Højer; Terne Sasha Thorn Jakobsen; Anna Rogers; Stefan Heinrich
>
> **备注:** ACL Findings 2025
>
> **摘要:** Despite the widespread use of ''artificial intelligence'' (AI) framing in Natural Language Processing (NLP) research, it is not clear what researchers mean by ''intelligence''. To that end, we present the results of a survey on the notion of ''intelligence'' among researchers and its role in the research agenda. The survey elicited complete responses from 303 researchers from a variety of fields including NLP, Machine Learning (ML), Cognitive Science, Linguistics, and Neuroscience. We identify 3 criteria of intelligence that the community agrees on the most: generalization, adaptability, & reasoning. Our results suggests that the perception of the current NLP systems as ''intelligent'' is a minority position (29%). Furthermore, only 16.2% of the respondents see developing intelligent systems as a research goal, and these respondents are more likely to consider the current systems intelligent.
>
---
#### [new 095] Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出LLM-AT框架，通过自动选择合适LLM层级优化成本与准确率。解决复杂NLP任务中平衡成本与性能的问题。框架含启动器、生成器和评估器，结合历史数据预估初始模型准确率，迭代升级至有效输出，实验证明可降低成本并保持性能。**

- **链接: [http://arxiv.org/pdf/2505.20921v1](http://arxiv.org/pdf/2505.20921v1)**

> **作者:** Injae Na; Keonwoong Noh; Woohwan Jung
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications.
>
---
#### [new 096] Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，解决现有方法依赖单一推理策略导致准确性和可解释性不足的问题。提出MAMMQA框架，通过视觉语言模型（VLM）分解问题、跨模态合成，及文本大模型整合答案，实现模态特化分工与透明推理，提升性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.20816v1](http://arxiv.org/pdf/2505.20816v1)**

> **作者:** Krishna Singh Rajput; Tejas Anvekar; Chitta Baral; Vivek Gupta
>
> **摘要:** Recent advances in multimodal question answering have primarily focused on combining heterogeneous modalities or fine-tuning multimodal large language models. While these approaches have shown strong performance, they often rely on a single, generalized reasoning strategy, overlooking the unique characteristics of each modality ultimately limiting both accuracy and interpretability. To address these limitations, we propose MAMMQA, a multi-agent QA framework for multimodal inputs spanning text, tables, and images. Our system includes two Visual Language Model (VLM) agents and one text-based Large Language Model (LLM) agent. The first VLM decomposes the user query into sub-questions and sequentially retrieves partial answers from each modality. The second VLM synthesizes and refines these results through cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive answer. This modular design enhances interpretability by making the reasoning process transparent and allows each agent to operate within its domain of expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our cooperative, multi-agent framework consistently outperforms existing baselines in both accuracy and robustness.
>
---
#### [new 097] Context-Aware Content Moderation for German Newspaper Comments
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动内容审核任务，针对德语报纸论坛评论中忽略平台上下文的问题，提出结合用户历史和文章主题的二元分类模型。通过LSTM、CNN和ChatGPT-3.5 Turbo在Der Standard语料库上实验，发现CNN/LSTM受益于上下文且表现优异，而ChatGPT零样本分类效果较差。**

- **链接: [http://arxiv.org/pdf/2505.20963v1](http://arxiv.org/pdf/2505.20963v1)**

> **作者:** Felix Krejca; Tobias Kietreiber; Alexander Buchelt; Sebastian Neumaier
>
> **摘要:** The increasing volume of online discussions requires advanced automatic content moderation to maintain responsible discourse. While hate speech detection on social media is well-studied, research on German-language newspaper forums remains limited. Existing studies often neglect platform-specific context, such as user history and article themes. This paper addresses this gap by developing and evaluating binary classification models for automatic content moderation in German newspaper forums, incorporating contextual information. Using LSTM, CNN, and ChatGPT-3.5 Turbo, and leveraging the One Million Posts Corpus from the Austrian newspaper Der Standard, we assess the impact of context-aware models. Results show that CNN and LSTM models benefit from contextual information and perform competitively with state-of-the-art approaches. In contrast, ChatGPT's zero-shot classification does not improve with added context and underperforms.
>
---
#### [new 098] Towards Pretraining Robust ASR Foundation Model with Acoustic-Aware Data Augmentation
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于ASR模型预训练任务，旨在解决缺乏大规模数据时模型鲁棒性不足的问题。研究揭示声学多样性比语言多样性更影响模型泛化，提出针对性声学增强方法，在960小时数据上训练使未见数据集的词错率降低19.24%，证明声学增强可替代大数据提升ASR模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.20606v1](http://arxiv.org/pdf/2505.20606v1)**

> **作者:** Dancheng Liu; Amir Nassereldine; Chenhui Xu; Jinjun Xiong
>
> **备注:** in submission
>
> **摘要:** Whisper's robust performance in automatic speech recognition (ASR) is often attributed to its massive 680k-hour training set, an impractical scale for most researchers. In this work, we examine how linguistic and acoustic diversity in training data affect the robustness of the ASR model and reveal that transcription generalization is primarily driven by acoustic variation rather than linguistic richness. We find that targeted acoustic augmentation methods could significantly improve the generalization ability of ASR models, reducing word-error rates by up to 19.24 percent on unseen datasets when training on the 960-hour Librispeech dataset. These findings highlight strategic acoustically focused data augmentation as a promising alternative to massive datasets for building robust ASR models, offering a potential solution to future foundation ASR models when massive human speech data is lacking.
>
---
#### [new 099] AdParaphrase v2.0: Generating Attractive Ad Texts Using a Preference-Annotated Paraphrase Dataset
- **分类: cs.CL**

- **简介: 该论文属于广告文本生成任务，旨在通过分析人类偏好数据识别吸引人的广告因素并改进生成方法。工作包括构建AdParaphrase v2.0数据集（16,460对标注数据，比v1.0大20倍），发现新语言特征，探索生成方法，并验证基于大模型的无参考评估指标潜力。**

- **链接: [http://arxiv.org/pdf/2505.20826v1](http://arxiv.org/pdf/2505.20826v1)**

> **作者:** Soichiro Murakami; Peinan Zhang; Hidetaka Kamigaito; Hiroya Takamura; Manabu Okumura
>
> **备注:** Accepted to ACL2025 Findings
>
> **摘要:** Identifying factors that make ad text attractive is essential for advertising success. This study proposes AdParaphrase v2.0, a dataset for ad text paraphrasing, containing human preference data, to enable the analysis of the linguistic factors and to support the development of methods for generating attractive ad texts. Compared with v1.0, this dataset is 20 times larger, comprising 16,460 ad text paraphrase pairs, each annotated with preference data from ten evaluators, thereby enabling a more comprehensive and reliable analysis. Through the experiments, we identified multiple linguistic features of engaging ad texts that were not observed in v1.0 and explored various methods for generating attractive ad texts. Furthermore, our analysis demonstrated the relationships between human preference and ad performance, and highlighted the potential of reference-free metrics based on large language models for evaluating ad text attractiveness. The dataset is publicly available at: https://github.com/CyberAgentAILab/AdParaphrase-v2.0.
>
---
#### [new 100] Improved Representation Steering for Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型控制任务，旨在解决表示 steering 效果逊于提示法的问题。提出 RePS 方法，通过双向偏好优化同时实现概念引导与抑制。实验显示其在 Gemma 模型中超越传统方法，接近提示效果，且抗破解，参数更少，提供可解释的稳健替代方案。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20809v1](http://arxiv.org/pdf/2505.20809v1)**

> **作者:** Zhengxuan Wu; Qinan Yu; Aryaman Arora; Christopher D. Manning; Christopher Potts
>
> **备注:** 46 pages, 23 figures, preprint
>
> **摘要:** Steering methods for language models (LMs) seek to provide fine-grained and interpretable control over model generations by variously changing model inputs, weights, or representations to adjust behavior. Recent work has shown that adjusting weights or representations is often less effective than steering by prompting, for instance when wanting to introduce or suppress a particular concept. We demonstrate how to improve representation steering via our new Reference-free Preference Steering (RePS), a bidirectional preference-optimization objective that jointly does concept steering and suppression. We train three parameterizations of RePS and evaluate them on AxBench, a large-scale model steering benchmark. On Gemma models with sizes ranging from 2B to 27B, RePS outperforms all existing steering methods trained with a language modeling objective and substantially narrows the gap with prompting -- while promoting interpretability and minimizing parameter count. In suppression, RePS matches the language-modeling objective on Gemma-2 and outperforms it on the larger Gemma-3 variants while remaining resilient to prompt-based jailbreaking attacks that defeat prompting. Overall, our results suggest that RePS provides an interpretable and robust alternative to prompting for both steering and suppression.
>
---
#### [new 101] MSA at SemEval-2025 Task 3: High Quality Weak Labeling and LLM Ensemble Verification for Multilingual Hallucination Detection
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2025多语言幻觉检测任务（Mu-SHROOM），提出结合任务定制prompt工程与LLM集成验证方法，通过主模型提取幻觉片段，三LLM投票验证，并用模糊匹配优化对齐，提升多语言LLM生成文本中虚假片段的检测效果。**

- **链接: [http://arxiv.org/pdf/2505.20880v1](http://arxiv.org/pdf/2505.20880v1)**

> **作者:** Baraa Hikal; Ahmed Nasreldin; Ali Hamdi
>
> **摘要:** This paper describes our submission for SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. The task involves detecting hallucinated spans in text generated by instruction-tuned Large Language Models (LLMs) across multiple languages. Our approach combines task-specific prompt engineering with an LLM ensemble verification mechanism, where a primary model extracts hallucination spans and three independent LLMs adjudicate their validity through probability-based voting. This framework simulates the human annotation workflow used in the shared task validation and test data. Additionally, fuzzy matching refines span alignment. Our system ranked 1st in Arabic and Basque, 2nd in German, Swedish, and Finnish, and 3rd in Czech, Farsi, and French.
>
---
#### [new 102] Beyond Templates: Dynamic Adaptation of Reasoning Demonstrations via Feasibility-Aware Exploration
- **分类: cs.CL**

- **简介: 该论文属于小模型推理能力对齐任务，旨在解决大模型推理轨迹直接迁移至小模型时的性能退化问题。提出DART框架，通过动态评估每步模仿可行性，当检测到能力差距时，引导小模型自主探索符合能力的替代推理路径，提升其泛化与数据效率。**

- **链接: [http://arxiv.org/pdf/2505.20700v1](http://arxiv.org/pdf/2505.20700v1)**

> **作者:** Yong Wu; Weihang Pan; Ke Li; Chen Binhui; Ping Li; Binbin Lin
>
> **摘要:** Large language models (LLMs) have shown remarkable reasoning capabilities, yet aligning such abilities to small language models (SLMs) remains a challenge due to distributional mismatches and limited model capacity. Existing reasoning datasets, typically designed for powerful LLMs, often lead to degraded performance when directly applied to weaker models. In this work, we introduce Dynamic Adaptation of Reasoning Trajectories (DART), a novel data adaptation framework that bridges the capability gap between expert reasoning trajectories and diverse SLMs. Instead of uniformly imitating expert steps, DART employs a selective imitation strategy guided by step-wise adaptability estimation via solution simulation. When expert steps surpass the student's capacity -- signaled by an Imitation Gap -- the student autonomously explores alternative reasoning paths, constrained by outcome consistency. We validate DART across multiple reasoning benchmarks and model scales, demonstrating that it significantly improves generalization and data efficiency over static fine-tuning. Our method enhances supervision quality by aligning training signals with the student's reasoning capabilities, offering a scalable solution for reasoning alignment in resource-constrained models.
>
---
#### [new 103] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy
- **分类: cs.CL; astro-ph.IM; cs.LG**

- **简介: 该论文提出AstroVisBench，首个天文领域科学计算与可视化基准，评估LLM生成数据处理工作流及可视化结果的能力，解决其科学见解正确性评估难题。采用LLM作为评判并结合专家标注，揭示现有模型能力差距。**

- **链接: [http://arxiv.org/pdf/2505.20538v1](http://arxiv.org/pdf/2505.20538v1)**

> **作者:** Sebastian Antony Joseph; Syed Murtaza Husain; Stella S. R. Offner; Stéphanie Juneau; Paul Torrey; Adam S. Bolton; Juan P. Farias; Niall Gaffney; Greg Durrett; Junyi Jessy Li
>
> **摘要:** Large Language Models (LLMs) are being explored for applications in scientific research, including their capabilities to synthesize literature, answer research questions, generate research ideas, and even conduct computational experiments. Ultimately, our goal is for these to help scientists derive novel scientific insights. In many areas of science, such insights often arise from processing and visualizing data to understand its patterns. However, evaluating whether an LLM-mediated scientific workflow produces outputs conveying the correct scientific insights is challenging to evaluate and has not been addressed in past work. We introduce AstroVisBench, the first benchmark for both scientific computing and visualization in the astronomy domain. AstroVisBench judges a language model's ability to both (1) create astronomy-specific workflows to process and analyze data and (2) visualize the results of these workflows through complex plots. Our evaluation of visualizations uses a novel LLM-as-a-judge workflow, which is validated against annotation by five professional astronomers. Using AstroVisBench we present an evaluation of state-of-the-art language models, showing a significant gap in their ability to engage in astronomy research as useful assistants. This evaluation provides a strong end-to-end evaluation for AI scientists that offers a path forward for the development of visualization-based workflows, which are central to a broad range of domains from physics to biology.
>
---
#### [new 104] SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; C.4**

- **简介: 该论文提出SpecExtend，解决大语言模型投机解码在长序列输入时因注意力成本高和草稿准确率低导致的性能下降问题。通过集成FlashAttention等高效注意力机制及跨模型检索策略动态更新KV缓存，加速推理。实验显示其在16K token输入时提速2.22倍，无需额外训练。**

- **链接: [http://arxiv.org/pdf/2505.20776v1](http://arxiv.org/pdf/2505.20776v1)**

> **作者:** Jungyoub Cha; Hyunjong Kim; Sungzoon Cho
>
> **备注:** 8 pages, 3 figures. Under review at EMNLP 2025
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), but its performance degrades on long inputs due to increased attention cost and reduced draft accuracy. We introduce SpecExtend, a drop-in enhancement that improves the performance of speculative decoding on long sequences without any additional training. SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention into both the draft and target models, reducing latency across all stages. To improve draft accuracy and speed, we propose Cross-model Retrieval, a novel KV cache update strategy that uses the target model's attention scores to dynamically select relevant context for the draft model. Extensive evaluations on three long-context understanding datasets show that SpecExtend accelerates standard tree-based speculative decoding by up to 2.22x for inputs up to 16K tokens, providing an effective solution for speculative decoding of long sequences. The code is available at https://github.com/jycha98/SpecExtend .
>
---
#### [new 105] Information-Theoretic Complementary Prompts for Improved Continual Text Classification
- **分类: cs.CL**

- **简介: 该论文属于持续文本分类（CTC）任务，旨在解决模型在序列学习时因忽略共享知识导致的灾难性遗忘问题。提出InfoComp方法，通过信息论框架学习私有（任务特定）和共享（任务不变）提示空间，并设计双损失函数分别强化两类知识，提升持续学习效果。**

- **链接: [http://arxiv.org/pdf/2505.20933v1](http://arxiv.org/pdf/2505.20933v1)**

> **作者:** Duzhen Zhang; Yong Ren; Chenxing Li; Dong Yu; Tielin Zhang
>
> **备注:** Accepted by Neural Networks
>
> **摘要:** Continual Text Classification (CTC) aims to continuously classify new text data over time while minimizing catastrophic forgetting of previously acquired knowledge. However, existing methods often focus on task-specific knowledge, overlooking the importance of shared, task-agnostic knowledge. Inspired by the complementary learning systems theory, which posits that humans learn continually through the interaction of two systems -- the hippocampus, responsible for forming distinct representations of specific experiences, and the neocortex, which extracts more general and transferable representations from past experiences -- we introduce Information-Theoretic Complementary Prompts (InfoComp), a novel approach for CTC. InfoComp explicitly learns two distinct prompt spaces: P(rivate)-Prompt and S(hared)-Prompt. These respectively encode task-specific and task-invariant knowledge, enabling models to sequentially learn classification tasks without relying on data replay. To promote more informative prompt learning, InfoComp uses an information-theoretic framework that maximizes mutual information between different parameters (or encoded representations). Within this framework, we design two novel loss functions: (1) to strengthen the accumulation of task-specific knowledge in P-Prompt, effectively mitigating catastrophic forgetting, and (2) to enhance the retention of task-invariant knowledge in S-Prompt, improving forward knowledge transfer. Extensive experiments on diverse CTC benchmarks show that our approach outperforms previous state-of-the-art methods.
>
---
#### [new 106] Leveraging Large Language Models for Bengali Math Word Problem Solving with Chain of Thought Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 论文针对孟加拉语数学应用题解决任务，解决低资源语言及复杂多步推理的挑战。构建含8792个问题的SOMADHAN数据集，评估LLM（如GPT、LLaMA）在零/少样本及CoT推理下的表现，发现CoT显著提升效果，LLaMA-3.3 70B达88%准确率。采用LoRA微调降低计算成本，推动低资源语言NLP研究。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21354v1](http://arxiv.org/pdf/2505.21354v1)**

> **作者:** Bidyarthi Paul; Jalisha Jashim Era; Mirazur Rahman Zim; Tahmid Sattar Aothoi; Faisal Muhammad Shah
>
> **摘要:** Solving Bengali Math Word Problems (MWPs) remains a major challenge in natural language processing (NLP) due to the language's low-resource status and the multi-step reasoning required. Existing models struggle with complex Bengali MWPs, largely because no human-annotated Bengali dataset has previously addressed this task. This gap has limited progress in Bengali mathematical reasoning. To address this, we created SOMADHAN, a dataset of 8792 complex Bengali MWPs with manually written, step-by-step solutions. We designed this dataset to support reasoning-focused evaluation and model development in a linguistically underrepresented context. Using SOMADHAN, we evaluated a range of large language models (LLMs) - including GPT-4o, GPT-3.5 Turbo, LLaMA series models, Deepseek, and Qwen - through both zero-shot and few-shot prompting with and without Chain of Thought (CoT) reasoning. CoT prompting consistently improved performance over standard prompting, especially in tasks requiring multi-step logic. LLaMA-3.3 70B achieved the highest accuracy of 88% with few-shot CoT prompting. We also applied Low-Rank Adaptation (LoRA) to fine-tune models efficiently, enabling them to adapt to Bengali MWPs with minimal computational cost. Our work fills a critical gap in Bengali NLP by providing a high-quality reasoning dataset and a scalable framework for solving complex MWPs. We aim to advance equitable research in low-resource languages and enhance reasoning capabilities in educational and language technologies.
>
---
#### [new 107] SELF-PERCEPT: Introspection Improves Large Language Models' Detection of Multi-Person Mental Manipulation in Conversations
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于多人群体对话中心理操纵检测任务，旨在解决复杂对话场景下LLM检测精度不足的问题。团队构建了含220段真实场景对话的MultiManip数据集，并提出基于自我认知理论的两阶段框架SELF-PERCEPT，提升多轮多人操纵识别效果。**

- **链接: [http://arxiv.org/pdf/2505.20679v1](http://arxiv.org/pdf/2505.20679v1)**

> **作者:** Danush Khanna; Pratinav Seth; Sidhaarth Sredharan Murali; Aditya Kumar Guru; Siddharth Shukla; Tanuj Tyagi; Sandeep Chaurasia; Kripabandhu Ghosh
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Mental manipulation is a subtle yet pervasive form of abuse in interpersonal communication, making its detection critical for safeguarding potential victims. However, due to manipulation's nuanced and context-specific nature, identifying manipulative language in complex, multi-turn, and multi-person conversations remains a significant challenge for large language models (LLMs). To address this gap, we introduce the MultiManip dataset, comprising 220 multi-turn, multi-person dialogues balanced between manipulative and non-manipulative interactions, all drawn from reality shows that mimic real-world scenarios. For manipulative interactions, it includes 11 distinct manipulations depicting real-life scenarios. We conduct extensive evaluations of state-of-the-art LLMs, such as GPT-4o and Llama-3.1-8B, employing various prompting strategies. Despite their capabilities, these models often struggle to detect manipulation effectively. To overcome this limitation, we propose SELF-PERCEPT, a novel, two-stage prompting framework inspired by Self-Perception Theory, demonstrating strong performance in detecting multi-person, multi-turn mental manipulation. Our code and data are publicly available at https://github.com/danushkhanna/self-percept .
>
---
#### [new 108] Exploring the Latent Capacity of LLMs for One-Step Text Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究冻结LLMs的一次性多token生成能力，旨在无需自回归迭代解码。通过输入两个学习嵌入，模型单次前向生成数百准确token。分析嵌入编码信息及嵌入空间的局部连通性，揭示LLMs潜在能力并暗示专用编码器可能性。**

- **链接: [http://arxiv.org/pdf/2505.21189v1](http://arxiv.org/pdf/2505.21189v1)**

> **作者:** Gleb Mezentsev; Ivan Oseledets
>
> **备注:** under review
>
> **摘要:** A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one specially trained input embedding. In this work, we explore whether such reconstruction is possible without autoregression. We show that frozen LLMs can generate hundreds of accurate tokens in just one forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored capability of LLMs - multi-token generation without iterative decoding. We investigate the behaviour of these embeddings and provide insight into the type of information they encode. We also empirically show that although these representations are not unique for a given text, they form connected and local regions in embedding space - a property that suggests the potential of learning a dedicated encoder into that space.
>
---
#### [new 109] rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset
- **分类: cs.CL**

- **简介: 该论文属于代码推理任务，旨在解决高质量验证数据集稀缺导致的模型性能瓶颈。通过构建含41.8万竞赛级问题及58万解决方案的验证数据集，提出三方面工作：1）合成可解竞赛编程问题；2）设计分步输入生成与输出互验的测试案例管道；3）增强长推理解决方案。实验显示显著提升代码推理性能，7B模型超越32B前沿模型。**

- **链接: [http://arxiv.org/pdf/2505.21297v1](http://arxiv.org/pdf/2505.21297v1)**

> **作者:** Yifei Liu; Li Lyna Zhang; Yi Zhu; Bingcheng Dong; Xudong Zhou; Ning Shang; Fan Yang; Mao Yang
>
> **摘要:** Advancing code reasoning in large language models (LLMs) is fundamentally limited by the scarcity of high-difficulty datasets, especially those with verifiable input-output test cases necessary for rigorous solution validation at scale. We introduce rStar-Coder, which significantly improves LLM code reasoning capabilities by constructing a large-scale, verified dataset of 418K competition-level code problems, 580K long-reasoning solutions along with rich test cases of varying difficulty. This is achieved through three core contributions: (1) we curate competitive programming code problems and oracle solutions to synthesize new, solvable problems; (2) we introduce a reliable input-output test case synthesis pipeline that decouples the generation into a three-step input generation method and a mutual verification mechanism for effective output labeling; (3) we augment problems with high-quality, test-case-verified long-reasoning solutions. Extensive experiments on Qwen models (1.5B-14B) across various code reasoning benchmarks demonstrate the superiority of rStar-Coder dataset, achieving leading performance comparable to frontier reasoning LLMs with much smaller model sizes. On LiveCodeBench, rStar-Coder improves Qwen2.5-7B from 17.4% to an impressive 57.3%, and Qwen2.5-14B from 23.3% to 62.5%, surpassing o3-mini (low) by3.1%. On the more challenging USA Computing Olympiad, our 7B model achieves an average pass@1 accuracy of 16.15%, outperforming the frontier-level QWQ-32B. Code and the dataset will be released at https://github.com/microsoft/rStar.
>
---
#### [new 110] Amulet: Putting Complex Multi-Turn Conversations on the Stand with LLM Juries
- **分类: cs.CL**

- **简介: 该论文提出Amulet框架，改进大语言模型（LLM）作为评委评估多轮对话的能力。针对现有LLM在处理复杂、多意图切换的对话场景时的不足，框架通过分析对话行为（意图变化）和会话原则（maxims）来优化判断，实验显示其显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.20451v1](http://arxiv.org/pdf/2505.20451v1)**

> **作者:** Sahana Ramnath; Anurag Mudgil; Brihi Joshi; Skyler Hallinan; Xiang Ren
>
> **摘要:** Today, large language models are widely used as judges to evaluate responses from other language models. Hence, it is imperative to benchmark and improve these LLM-judges on real-world language model usage: a typical human-assistant conversation is lengthy, and shows significant diversity in topics, intents, and requirements across turns, e.g. social interactions, task requests, feedback. We present Amulet, a framework that leverages pertinent linguistic concepts of dialog-acts and maxims to improve the accuracy of LLM-judges on preference data with complex, multi-turn conversational context. Amulet presents valuable insights about (a) the communicative structures and intents present in the conversation (dialog acts), and (b) the satisfaction of conversational principles (maxims) by the preference responses, and uses them to make judgments. On four challenging datasets, Amulet shows that (a) humans frequently (60 to 70 percent of the time) change their intents from one turn of the conversation to the next, and (b) in 75 percent of instances, the preference responses can be differentiated via dialog acts and/or maxims, reiterating the latter's significance in judging such data. Amulet can be used either as a judge by applying the framework to a single LLM, or integrated into a jury with different LLM judges; our judges and juries show strong improvements on relevant baselines for all four datasets.
>
---
#### [new 111] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，解决密集检索器因缺乏标注的查询-文档对难以训练的问题。提出ReSCORE方法，利用大语言模型评估文档相关性和答案一致性，在迭代框架中无监督训练检索器，提升检索效果和MHQA性能。**

- **链接: [http://arxiv.org/pdf/2505.21250v1](http://arxiv.org/pdf/2505.21250v1)**

> **作者:** Dosung Lee; Wonjun Oh; Boyoung Kim; Minyoung Kim; Joonsuk Park; Paul Hongsuck Seo
>
> **备注:** 9 pages, 3 figures, ACL 2025
>
> **摘要:** Multi-hop question answering (MHQA) involves reasoning across multiple documents to answer complex questions. Dense retrievers typically outperform sparse methods like BM25 by leveraging semantic embeddings; however, they require labeled query-document pairs for fine-tuning. This poses a significant challenge in MHQA due to the high variability of queries (reformulated) questions throughout the reasoning steps. To overcome this limitation, we introduce Retriever Supervision with Consistency and Relevance (ReSCORE), a novel method for training dense retrievers for MHQA without labeled documents. ReSCORE leverages large language models to capture each documents relevance to the question and consistency with the correct answer and use them to train a retriever within an iterative question-answering framework. Experiments on three MHQA benchmarks demonstrate the effectiveness of ReSCORE, with significant improvements in retrieval, and in turn, the state-of-the-art MHQA performance. Our implementation is available at: https://leeds1219.github.io/ReSCORE.
>
---
#### [new 112] AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs
- **分类: cs.CL**

- **简介: 该论文提出AutoJudger框架，解决多模态大模型评估成本高昂的问题。通过IRT理论与自主代理，动态选择最具信息量的测试题，结合语义检索和动态记忆组件，以4%数据实现90%评估精度，优化评估效率。**

- **链接: [http://arxiv.org/pdf/2505.21389v1](http://arxiv.org/pdf/2505.21389v1)**

> **作者:** Xuanwen Ding; Chengjun Pan; Zejun Li; Jiwen Zhang; Siyuan Wang; Zhongyu Wei
>
> **摘要:** Evaluating multimodal large language models (MLLMs) is increasingly expensive, as the growing size and cross-modality complexity of benchmarks demand significant scoring efforts. To tackle with this difficulty, we introduce AutoJudger, an agent-driven framework for efficient and adaptive benchmarking of MLLMs that tackles this escalating cost. AutoJudger employs the Item Response Theory (IRT) to estimate the question difficulty and an autonomous evaluation agent to dynamically select the most informative test questions based on the model's real-time performance. Specifically, AutoJudger incorporates two pivotal components: a semantic-aware retrieval mechanism to ensure that selected questions cover diverse and challenging scenarios across both vision and language modalities, and a dynamic memory that maintains contextual statistics of previously evaluated questions to guide coherent and globally informed question selection throughout the evaluation process. Extensive experiments on four representative multimodal benchmarks demonstrate that our adaptive framework dramatically reduces evaluation expenses, i.e. AutoJudger uses only 4% of the data to achieve over 90% ranking accuracy with the full benchmark evaluation on MMT-Bench.
>
---
#### [new 113] Leveraging large language models and traditional machine learning ensembles for ADHD detection from narrative transcripts
- **分类: cs.CL**

- **简介: 该研究属于ADHD文本分类任务，旨在通过融合LLM与传统ML提升叙事数据诊断效果。提出集成LLaMA3、RoBERTa与SVM的框架，经投票机制增强鲁棒性，在441例数据中达0.71 F1值，优于单一模型，尤其提升召回率。**

- **链接: [http://arxiv.org/pdf/2505.21324v1](http://arxiv.org/pdf/2505.21324v1)**

> **作者:** Yuxin Zhu; Yuting Guo; Noah Marchuck; Abeed Sarker; Yun Wang
>
> **摘要:** Despite rapid advances in large language models (LLMs), their integration with traditional supervised machine learning (ML) techniques that have proven applicability to medical data remains underexplored. This is particularly true for psychiatric applications, where narrative data often exhibit nuanced linguistic and contextual complexity, and can benefit from the combination of multiple models with differing characteristics. In this study, we introduce an ensemble framework for automatically classifying Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis (binary) using narrative transcripts. Our approach integrates three complementary models: LLaMA3, an open-source LLM that captures long-range semantic structure; RoBERTa, a pre-trained transformer model fine-tuned on labeled clinical narratives; and a Support Vector Machine (SVM) classifier trained using TF-IDF-based lexical features. These models are aggregated through a majority voting mechanism to enhance predictive robustness. The dataset includes 441 instances, including 352 for training and 89 for validation. Empirical results show that the ensemble outperforms individual models, achieving an F$_1$ score of 0.71 (95\% CI: [0.60-0.80]). Compared to the best-performing individual model (SVM), the ensemble improved recall while maintaining competitive precision. This indicates the strong sensitivity of the ensemble in identifying ADHD-related linguistic cues. These findings demonstrate the promise of hybrid architectures that leverage the semantic richness of LLMs alongside the interpretability and pattern recognition capabilities of traditional supervised ML, offering a new direction for robust and generalizable psychiatric text classification.
>
---
#### [new 114] Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG
- **分类: cs.CL**

- **简介: 该论文属于提升检索增强生成(RAG)系统可靠性任务。针对现有方法在缺乏可靠知识时仍强行生成答案的问题，提出Divide-Then-Align(DTA)方法，通过划分知识象限构建偏好数据集，训练模型在超出知识边界时主动拒绝回答。实验表明该方法有效平衡了回答准确性和合理拒答，提升系统可信度。**

- **链接: [http://arxiv.org/pdf/2505.20871v1](http://arxiv.org/pdf/2505.20871v1)**

> **作者:** Xin Sun; Jianan Xie; Zhongqi Chen; Qiang Liu; Shu Wu; Yuehe Chen; Bowen Song; Weiqiang Wang; Zilei Wang; Liang Wang
>
> **备注:** ACL 2025 main
>
> **摘要:** Large language models (LLMs) augmented with retrieval systems have significantly advanced natural language processing tasks by integrating external knowledge sources, enabling more accurate and contextually rich responses. To improve the robustness of such systems against noisy retrievals, Retrieval-Augmented Fine-Tuning (RAFT) has emerged as a widely adopted method. However, RAFT conditions models to generate answers even in the absence of reliable knowledge. This behavior undermines their reliability in high-stakes domains, where acknowledging uncertainty is critical. To address this issue, we propose Divide-Then-Align (DTA), a post-training approach designed to endow RAG systems with the ability to respond with "I don't know" when the query is out of the knowledge boundary of both the retrieved passages and the model's internal knowledge. DTA divides data samples into four knowledge quadrants and constructs tailored preference data for each quadrant, resulting in a curated dataset for Direct Preference Optimization (DPO). Experimental results on three benchmark datasets demonstrate that DTA effectively balances accuracy with appropriate abstention, enhancing the reliability and trustworthiness of retrieval-augmented systems.
>
---
#### [new 115] Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Lunguage基准，解决现有放射报告评估无法捕捉细粒度临床语义与时序依赖的问题。构建含1473份标注胸片报告（含80份纵向数据）的数据集，开发两阶段框架生成结构化表示，并提出LUNGUAGESCORE指标，通过实体、关系及时间一致性评估报告质量。**

- **链接: [http://arxiv.org/pdf/2505.21190v1](http://arxiv.org/pdf/2505.21190v1)**

> **作者:** Jong Hak Moon; Geon Choi; Paloma Rabaey; Min Gwan Kim; Hyuk Gi Hong; Jung-Oh Lee; Hangyul Yoon; Eun Woo Doe; Jiyoun Kim; Harshita Sharma; Daniel C. Castro; Javier Alvarez-Valle; Edward Choi
>
> **摘要:** Radiology reports convey detailed clinical observations and capture diagnostic reasoning that evolves over time. However, existing evaluation methods are limited to single-report settings and rely on coarse metrics that fail to capture fine-grained clinical semantics and temporal dependencies. We introduce LUNGUAGE,a benchmark dataset for structured radiology report generation that supports both single-report evaluation and longitudinal patient-level assessment across multiple studies. It contains 1,473 annotated chest X-ray reports, each reviewed by experts, and 80 of them contain longitudinal annotations to capture disease progression and inter-study intervals, also reviewed by experts. Using this benchmark, we develop a two-stage framework that transforms generated reports into fine-grained, schema-aligned structured representations, enabling longitudinal interpretation. We also propose LUNGUAGESCORE, an interpretable metric that compares structured outputs at the entity, relation, and attribute level while modeling temporal consistency across patient timelines. These contributions establish the first benchmark dataset, structuring framework, and evaluation metric for sequential radiology reporting, with empirical results demonstrating that LUNGUAGESCORE effectively supports structured report evaluation. The code is available at: https://github.com/SuperSupermoon/Lunguage
>
---
#### [new 116] A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction
- **分类: cs.CL; cs.AI; cs.CE; cs.IR; cs.LG; I.2.7; I.2.1; I.5.1; I.2.6; H.3.1**

- **简介: 该论文属于生成语言模型领域适应任务，解决工程场景下计算资源消耗大及模型幻觉问题。提出Small Language Graph（SLG），采用图结构轻量专家节点，提升Exact Match指标3倍，微调速度快1.7倍，支持分布式AI，降低算力需求。**

- **链接: [http://arxiv.org/pdf/2505.21109v1](http://arxiv.org/pdf/2505.21109v1)**

> **作者:** Bogdan Bogachov; Yaoyao Fiona Zhao
>
> **备注:** 10 pages, 4 Figures, 6 Tables. This paper has been accepted to be published in the proceedings of IDETC-CIE 2025
>
> **摘要:** Despite recent advancements in domain adaptation techniques for large language models, these methods remain computationally intensive, and the resulting models can still exhibit hallucination issues. Most existing adaptation methods do not prioritize reducing the computational resources required for fine-tuning and inference of language models. Hallucination issues have gradually decreased with each new model release. However, they remain prevalent in engineering contexts, where generating well-structured text with minimal errors and inconsistencies is critical. This work introduces a novel approach called the Small Language Graph (SLG), which is a lightweight adaptation solution designed to address the two key challenges outlined above. The system is structured in the form of a graph, where each node represents a lightweight expert - a small language model fine-tuned on specific and concise texts. The results of this study have shown that SLG was able to surpass conventional fine-tuning methods on the Exact Match metric by 3 times. Additionally, the fine-tuning process was 1.7 times faster compared to that of a larger stand-alone language model. These findings introduce a potential for small to medium-sized engineering companies to confidently use generative AI technologies, such as LLMs, without the necessity to invest in expensive computational resources. Also, the graph architecture and the small size of expert nodes offer a possible opportunity for distributed AI systems, thus potentially diverting the global need for expensive centralized compute clusters.
>
---
#### [new 117] Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于RAG系统输出事实核查任务，旨在解决现有方法混淆事实正确性与检索内容忠实性、误判正确回答的问题。提出FRANQ方法，通过区分事实与忠实性的不确定性量化技术检测幻觉，并构建新QA数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2505.21072v1](http://arxiv.org/pdf/2505.21072v1)**

> **作者:** Ekaterina Fadeeva; Aleksandr Rubashevskii; Roman Vashurin; Shehzaad Dhuliawala; Artem Shelmanov; Timothy Baldwin; Preslav Nakov; Mrinmaya Sachan; Maxim Panov
>
> **摘要:** Large Language Models (LLMs) enhanced with external knowledge retrieval, an approach known as Retrieval-Augmented Generation (RAG), have shown strong performance in open-domain question answering. However, RAG systems remain susceptible to hallucinations: factually incorrect outputs that may arise either from inconsistencies in the model's internal knowledge or incorrect use of the retrieved context. Existing approaches often conflate factuality with faithfulness to the retrieved context, misclassifying factually correct statements as hallucinations if they are not directly supported by the retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval Augmented UNcertainty Quantification), a novel method for hallucination detection in RAG outputs. FRANQ applies different Uncertainty Quantification (UQ) techniques to estimate factuality based on whether a statement is faithful to the retrieved context or not. To evaluate FRANQ and other UQ techniques for RAG, we present a new long-form Question Answering (QA) dataset annotated for both factuality and faithfulness, combining automated labeling with manual validation of challenging examples. Extensive experiments on long- and short-form QA across multiple datasets and LLMs show that FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing methods.
>
---
#### [new 118] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction
- **分类: cs.CL; cs.RO**

- **简介: 该论文属于对话系统中的预测换手（Predictive Turn-Taking）任务，旨在通过整合视觉线索提升两人交互中对话权切换预测的准确性。现有模型依赖单一语音模态，研究提出多模态模型MM-VAP，融合面部表情、头部姿态和 gaze 数据，发现视觉特征（尤其面部表情）显著提升预测效果（84% vs. 79%），并验证语音对齐方法的适用性，为多模态预测提供首个系统分析。**

- **链接: [http://arxiv.org/pdf/2505.21043v1](http://arxiv.org/pdf/2505.21043v1)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **摘要:** Turn-taking is richly multimodal. Predictive turn-taking models (PTTMs) facilitate naturalistic human-robot interaction, yet most rely solely on speech. We introduce MM-VAP, a multimodal PTTM which combines speech with visual cues including facial expression, head pose and gaze. We find that it outperforms the state-of-the-art audio-only in videoconferencing interactions (84% vs. 79% hold/shift prediction accuracy). Unlike prior work which aggregates all holds and shifts, we group by duration of silence between turns. This reveals that through the inclusion of visual features, MM-VAP outperforms a state-of-the-art audio-only turn-taking model across all durations of speaker transitions. We conduct a detailed ablation study, which reveals that facial expression features contribute the most to model performance. Thus, our working hypothesis is that when interlocutors can see one another, visual cues are vital for turn-taking and must therefore be included for accurate turn-taking prediction. We additionally validate the suitability of automatic speech alignment for PTTM training using telephone speech. This work represents the first comprehensive analysis of multimodal PTTMs. We discuss implications for future work and make all code publicly available.
>
---
#### [new 119] Do LLMs have a Gender (Entropy) Bias?
- **分类: cs.CL; cs.AI; 68T42, 68T50; I.2.7**

- **简介: 该论文探究LLMs是否存在性别熵偏见，通过构建跨四领域真实问题数据集，发现模型在类别层面无显著偏见，但个体问题回答质量存在性别差异。提出迭代合并男女版回答的去偏策略，使78%案例信息量提升，实现平衡输出。**

- **链接: [http://arxiv.org/pdf/2505.20343v1](http://arxiv.org/pdf/2505.20343v1)**

> **作者:** Sonal Prabhune; Balaji Padmanabhan; Kaushik Dutta
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** We investigate the existence and persistence of a specific type of gender bias in some of the popular LLMs and contribute a new benchmark dataset, RealWorldQuestioning (released on HuggingFace ), developed from real-world questions across four key domains in business and health contexts: education, jobs, personal financial management, and general health. We define and study entropy bias, which we define as a discrepancy in the amount of information generated by an LLM in response to real questions users have asked. We tested this using four different LLMs and evaluated the generated responses both qualitatively and quantitatively by using ChatGPT-4o (as "LLM-as-judge"). Our analyses (metric-based comparisons and "LLM-as-judge" evaluation) suggest that there is no significant bias in LLM responses for men and women at a category level. However, at a finer granularity (the individual question level), there are substantial differences in LLM responses for men and women in the majority of cases, which "cancel" each other out often due to some responses being better for males and vice versa. This is still a concern since typical users of these tools often ask a specific question (only) as opposed to several varied ones in each of these common yet important areas of life. We suggest a simple debiasing approach that iteratively merges the responses for the two genders to produce a final result. Our approach demonstrates that a simple, prompt-based debiasing strategy can effectively debias LLM outputs, thus producing responses with higher information content than both gendered variants in 78% of the cases, and consistently achieving a balanced integration in the remaining cases.
>
---
#### [new 120] Multilingual Pretraining for Pixel Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言像素语言模型预训练任务，旨在提升跨语言（尤其非拉丁文字）性能。提出PIXEL-M4模型，预训练英、印地、乌克兰及简体中文，优于单语模型，在非拉丁语种表现更优，分析显示其捕捉语言特征且语义空间对齐。**

- **链接: [http://arxiv.org/pdf/2505.21265v1](http://arxiv.org/pdf/2505.21265v1)**

> **作者:** Ilker Kesen; Jonas F. Lotz; Ingo Ziegler; Phillip Rust; Desmond Elliott
>
> **备注:** 17 pages, 19 figures, 7 tables
>
> **摘要:** Pixel language models operate directly on images of rendered text, eliminating the need for a fixed vocabulary. While these models have demonstrated strong capabilities for downstream cross-lingual transfer, multilingual pretraining remains underexplored. We introduce PIXEL-M4, a model pretrained on four visually and linguistically diverse languages: English, Hindi, Ukrainian, and Simplified Chinese. Multilingual evaluations on semantic and syntactic tasks show that PIXEL-M4 outperforms an English-only counterpart on non-Latin scripts. Word-level probing analyses confirm that PIXEL-M4 captures rich linguistic features, even in languages not seen during pretraining. Furthermore, an analysis of its hidden representations shows that multilingual pretraining yields a semantic embedding space closely aligned across the languages used for pretraining. This work demonstrates that multilingual pretraining substantially enhances the capability of pixel language models to effectively support a diverse set of languages.
>
---
#### [new 121] Emotion Classification In-Context in Spanish
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对西班牙语客户反馈情感分类任务，解决传统翻译方法导致的语义损失问题。提出结合TF-IDF与BERT的混合模型，并采用自定义堆叠集成（CSE）整合Logistic Regression、KNN、LGBM Bagging及AdaBoost等模型，最终测试集准确率达93.3%，优于单独模型及BERT。**

- **链接: [http://arxiv.org/pdf/2505.20571v1](http://arxiv.org/pdf/2505.20571v1)**

> **作者:** Bipul Thapa; Gabriel Cofre
>
> **备注:** This paper has been accepted and presented at the 4th International Conference on Applied Intelligence and Informatics (AII 2024). The final version will appear in the official conference proceedings. This preprint is provided to ensure the timely dissemination of the research prior to formal publication
>
> **摘要:** Classifying customer feedback into distinct emotion categories is essential for understanding sentiment and improving customer experience. In this paper, we classify customer feedback in Spanish into three emotion categories--positive, neutral, and negative--using advanced NLP and ML techniques. Traditional methods translate feedback from widely spoken languages to less common ones, resulting in a loss of semantic integrity and contextual nuances inherent to the original language. To address this limitation, we propose a hybrid approach that combines TF-IDF with BERT embeddings, effectively transforming Spanish text into rich numerical representations that preserve the semantic depth of the original language by using a Custom Stacking Ensemble (CSE) approach. To evaluate emotion classification, we utilize a range of models, including Logistic Regression, KNN, Bagging classifier with LGBM, and AdaBoost. The CSE model combines these classifiers as base models and uses a one-vs-all Logistic Regression as the meta-model. Our experimental results demonstrate that CSE significantly outperforms the individual and BERT model, achieving a test accuracy of 93.3% on the native Spanish dataset--higher than the accuracy obtained from the translated version. These findings underscore the challenges of emotion classification in Spanish and highlight the advantages of combining vectorization techniques like TF-IDF with BERT for improved accuracy. Our results provide valuable insights for businesses seeking to leverage emotion classification to enhance customer feedback analysis and service improvements.
>
---
#### [new 122] Less Context, Same Performance: A RAG Framework for Resource-Efficient LLM-Based Clinical NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对临床文本分类中LLM的计算成本高和上下文限制问题，提出基于RAG框架的方法。通过分割临床文档、向量化并检索最相关片段（4000词），输入LLM进行并发症识别。实验显示与全文本处理效果相当，减少资源消耗，实现高效低成本的长文本分析。**

- **链接: [http://arxiv.org/pdf/2505.20320v1](http://arxiv.org/pdf/2505.20320v1)**

> **作者:** Satya Narayana Cheetirala; Ganesh Raut; Dhavalkumar Patel; Fabio Sanatana; Robert Freeman; Matthew A Levin; Girish N. Nadkarni; Omar Dawkins; Reba Miller; Randolph M. Steinhagen; Eyal Klang; Prem Timsina
>
> **摘要:** Long text classification is challenging for Large Language Models (LLMs) due to token limits and high computational costs. This study explores whether a Retrieval Augmented Generation (RAG) approach using only the most relevant text segments can match the performance of processing entire clinical notes with large context LLMs. We begin by splitting clinical documents into smaller chunks, converting them into vector embeddings, and storing these in a FAISS index. We then retrieve the top 4,000 words most pertinent to the classification query and feed these consolidated segments into an LLM. We evaluated three LLMs (GPT4o, LLaMA, and Mistral) on a surgical complication identification task. Metrics such as AUC ROC, precision, recall, and F1 showed no statistically significant differences between the RAG based approach and whole-text processing (p > 0.05p > 0.05). These findings indicate that RAG can significantly reduce token usage without sacrificing classification accuracy, providing a scalable and cost effective solution for analyzing lengthy clinical documents.
>
---
#### [new 123] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian
- **分类: cs.CL; cs.AI**

- **简介: 该研究对比人类与LLMs对下属类别的认知组织，首次构建意大利语人类生成实例数据集（187词），评估LLMs在实例生成、类别归纳和典型性判断中的表现，发现其与人类认知对齐度低但存在领域差异，揭示AI生成数据在心理语言研究中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2505.21301v1](http://arxiv.org/pdf/2505.21301v1)**

> **作者:** Andrea Pedrotti; Giulia Rambelli; Caterina Villani; Marianna Bolognesi
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** People can categorize the same entity at multiple taxonomic levels, such as basic (bear), superordinate (animal), and subordinate (grizzly bear). While prior research has focused on basic-level categories, this study is the first attempt to examine the organization of categories by analyzing exemplars produced at the subordinate level. We present a new Italian psycholinguistic dataset of human-generated exemplars for 187 concrete words. We then use these data to evaluate whether textual and vision LLMs produce meaningful exemplars that align with human category organization across three key tasks: exemplar generation, category induction, and typicality judgment. Our findings show a low alignment between humans and LLMs, consistent with previous studies. However, their performance varies notably across different semantic domains. Ultimately, this study highlights both the promises and the constraints of using AI-generated exemplars to support psychological and linguistic research.
>
---
#### [new 124] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings
- **分类: cs.CL**

- **简介: 该论文评估LLMs在医学文本摘要中的表现，发现其在高OOV（未登录词）场景下性能显著下降。通过词汇适应策略（包括多种方法及持续预训练）和三个医疗数据集实验，证明该策略有效，人类评估显示摘要更相关准确。**

- **链接: [http://arxiv.org/pdf/2505.21242v1](http://arxiv.org/pdf/2505.21242v1)**

> **作者:** Gunjan Balde; Soumyadeep Roy; Mainack Mondal; Niloy Ganguly
>
> **备注:** 16 pages. Accepted for publication in the Findings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Large Language Models (LLMs) recently achieved great success in medical text summarization by simply using in-context learning. However, these recent efforts do not perform fine-grained evaluations under difficult settings where LLMs might fail. They typically report performance scores over the entire dataset. Through our benchmarking study, we show that LLMs show a significant performance drop for data points with high concentration of out-of-vocabulary (OOV) words or with high novelty. Vocabulary adaptation is an intuitive solution to this vocabulary mismatch issue where the LLM vocabulary gets updated with certain expert domain (here, medical) words or subwords. An interesting finding from our study is that Llama-3.1, even with a vocabulary size of around 128K tokens, still faces over-fragmentation issue with medical words. To that end, we show vocabulary adaptation helps improve the LLM summarization performance even in difficult settings. Through extensive experimentation of multiple vocabulary adaptation strategies, two continual pretraining strategies, and three benchmark medical summarization datasets, we gain valuable insights into the role of vocabulary adaptation strategies for customizing LLMs to the medical domain. We also performed a human evaluation study with medical experts where they found that vocabulary adaptation results in more relevant and faithful summaries. Our codebase is made publicly available at https://github.com/gb-kgp/LLM-MedicalSummarization-Benchmark.
>
---
#### [new 125] RelationalFactQA: A Benchmark for Evaluating Tabular Fact Retrieval from Large Language Models
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文提出RelationalFactQA基准，评估大语言模型生成结构化表格事实的能力。针对现有评估忽视多记录表格输出及维度增加时的性能下降问题，设计含自然语言问题、SQL及标准表格答案的任务。实验显示先进模型准确率不足25%，凸显其结构化知识合成的局限。**

- **链接: [http://arxiv.org/pdf/2505.21409v1](http://arxiv.org/pdf/2505.21409v1)**

> **作者:** Dario Satriani; Enzo Veltri; Donatello Santoro; Paolo Papotti
>
> **摘要:** Factuality in Large Language Models (LLMs) is a persistent challenge. Current benchmarks often assess short factual answers, overlooking the critical ability to generate structured, multi-record tabular outputs from parametric knowledge. We demonstrate that this relational fact retrieval is substantially more difficult than isolated point-wise queries, even when individual facts are known to the model, exposing distinct failure modes sensitive to output dimensionality (e.g., number of attributes or records). To systematically evaluate this under-explored capability, we introduce RelationalFactQA, a new benchmark featuring diverse natural language questions (paired with SQL) and gold-standard tabular answers, specifically designed to assess knowledge retrieval in a structured format. RelationalFactQA enables analysis across varying query complexities, output sizes, and data characteristics. Our experiments reveal that even state-of-the-art LLMs struggle significantly, not exceeding 25% factual accuracy in generating relational outputs, with performance notably degrading as output dimensionality increases. These findings underscore critical limitations in current LLMs' ability to synthesize structured factual knowledge and establish RelationalFactQA as a crucial resource for measuring future progress in LLM factuality.
>
---
#### [new 126] Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead
- **分类: cs.CL**

- **简介: 该综述分析过去五年734篇非洲语言NLP论文，总结核心任务进展，识别研究趋势，提出促进包容性、可持续发展的方向，旨在解决语言资源不均及数字鸿沟问题。**

- **链接: [http://arxiv.org/pdf/2505.21315v1](http://arxiv.org/pdf/2505.21315v1)**

> **作者:** Jesujoba O. Alabi; Michael A. Hedderich; David Ifeoluwa Adelani; Dietrich Klakow
>
> **备注:** Working paper
>
> **摘要:** With over 2,000 languages and potentially millions of speakers, Africa represents one of the richest linguistic regions in the world. Yet, this diversity is scarcely reflected in state-of-the-art natural language processing (NLP) systems and large language models (LLMs), which predominantly support a narrow set of high-resource languages. This exclusion not only limits the reach and utility of modern NLP technologies but also risks widening the digital divide across linguistic communities. Nevertheless, NLP research on African languages is active and growing. In recent years, there has been a surge of interest in this area, driven by several factors-including the creation of multilingual language resources, the rise of community-led initiatives, and increased support through funding programs. In this survey, we analyze 734 research papers on NLP for African languages published over the past five years, offering a comprehensive overview of recent progress across core tasks. We identify key trends shaping the field and conclude by outlining promising directions to foster more inclusive and sustainable NLP research for African languages.
>
---
#### [new 127] HAMburger: Accelerating LLM Inference via Token Smashing
- **分类: cs.CL**

- **简介: 该论文属于LLM推理加速任务。针对传统方法每个token需独立KV缓存与计算导致资源线性增长的问题，提出HAMburger模型：通过合并多token至单KV并分层解码，实现资源亚线性增长，提升推理速度达2倍，同时保持质量。**

- **链接: [http://arxiv.org/pdf/2505.20438v1](http://arxiv.org/pdf/2505.20438v1)**

> **作者:** Jingyu Liu; Ce Zhang
>
> **摘要:** The growing demand for efficient Large Language Model (LLM) inference requires a holistic optimization on algorithms, systems, and hardware. However, very few works have fundamentally changed the generation pattern: each token needs one forward pass and one KV cache. This can be sub-optimal because we found that LLMs are extremely capable of self-identifying the exact dose of information that a single KV cache can store, and many tokens can be generated confidently without global context. Based on this insight, we introduce HAMburger, a Hierarchically Auto-regressive Model that redefines resource allocation in LLMs by moving beyond uniform computation and storage per token during inference. Stacking a compositional embedder and a micro-step decoder in between a base LLM, HAMburger smashes multiple tokens into a single KV and generates several tokens per step. Additionally, HAMburger functions as a speculative decoding framework where it can blindly trust self-drafted tokens. As a result, HAMburger shifts the growth of KV cache and forward FLOPs from linear to sub-linear with respect to output length, and adjusts its inference speed based on query perplexity and output structure. Extensive evaluations show that HAMburger reduces the KV cache computation by up to 2$\times$ and achieves up to 2$\times$ TPS, while maintaining quality in both short- and long-context tasks. Our method explores an extremely challenging inference regime that requires both computation- and memory-efficiency with a hardware-agnostic design.
>
---
#### [new 128] Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Lookahead Q-Cache（LAQ），针对大语言模型KV缓存因预填充阶段注意力评分导致的淘汰不一致性问题，在内存受限时通过生成伪查询预测解码需求，优化缓存策略，提升长序列任务性能。**

- **链接: [http://arxiv.org/pdf/2505.20334v1](http://arxiv.org/pdf/2505.20334v1)**

> **作者:** Yixuan Wang; Shiyu Ji; Yijun Liu; Yuzhuang Xu; Yang Xu; Qingfu Zhu; Wanxiang Che
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Large language models (LLMs) rely on key-value cache (KV cache) to accelerate decoding by reducing redundant computations. However, the KV cache memory usage grows substantially with longer text sequences, posing challenges for efficient deployment. Existing KV cache eviction methods prune tokens using prefilling-stage attention scores, causing inconsistency with actual inference queries, especially under tight memory budgets. In this paper, we propose Lookahead Q-Cache (LAQ), a novel eviction framework that generates low-cost pseudo lookahead queries to better approximate the true decoding-stage queries. By using these lookahead queries as the observation window for importance estimation, LAQ achieves more consistent and accurate KV cache eviction aligned with real inference scenarios. Experimental results on LongBench and Needle-in-a-Haystack benchmarks show that LAQ outperforms existing methods across various budget levels, achieving a 1 $\sim$ 4 point improvement on LongBench under limited cache budget. Moreover, LAQ is complementary to existing approaches and can be flexibly combined to yield further improvements.
>
---
#### [new 129] Rethinking Text-based Protein Understanding: Retrieval or LLM?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于蛋白质文本理解与生成任务，针对现有基准数据泄露及评估指标不适用的问题，重新组织数据集并提出基于生物实体的评估框架；同时提出检索增强方法，优于LLM的微调模型，在无训练场景下高效生成蛋白质文本。**

- **链接: [http://arxiv.org/pdf/2505.20354v1](http://arxiv.org/pdf/2505.20354v1)**

> **作者:** Juntong Wu; Zijing Liu; He Cao; Hao Li; Bin Feng; Zishan Shu; Ke Yu; Li Yuan; Yu Li
>
> **摘要:** In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
>
---
#### [new 130] EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EasyDistill工具包，针对大语言模型的黑盒/白盒知识蒸馏任务，解决高效知识转移与技术普及问题。工作包括集成数据合成、监督微调、排序优化及强化学习模块，支持两种类型模型，并提供预训练模型、数据集及阿里云平台集成，促进工业应用。**

- **链接: [http://arxiv.org/pdf/2505.20888v1](http://arxiv.org/pdf/2505.20888v1)**

> **作者:** Chengyu Wang; Junbing Yan; Wenrui Cai; Yuanhao Yue; Jun Huang
>
> **摘要:** In this paper, we present EasyDistill, a comprehensive toolkit designed for effective black-box and white-box knowledge distillation (KD) of large language models (LLMs). Our framework offers versatile functionalities, including data synthesis, supervised fine-tuning, ranking optimization, and reinforcement learning techniques specifically tailored for KD scenarios. The toolkit accommodates KD functionalities for both System 1 (fast, intuitive) and System 2 (slow, analytical) models. With its modular design and user-friendly interface, EasyDistill empowers researchers and industry practitioners to seamlessly experiment with and implement state-of-the-art KD strategies for LLMs. In addition, EasyDistill provides a series of robust distilled models and KD-based industrial solutions developed by us, along with the corresponding open-sourced datasets, catering to a variety of use cases. Furthermore, we describe the seamless integration of EasyDistill into Alibaba Cloud's Platform for AI (PAI). Overall, the EasyDistill toolkit makes advanced KD techniques for LLMs more accessible and impactful within the NLP community.
>
---
#### [new 131] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出BiomedSQL，首个针对生物医学知识库的文本到SQL基准，解决现有系统在科学推理（如基因-疾病关联、药物数据）中的不足。包含68,000个问题-查询-答案，需领域推理，评估多种模型显示性能差距（最高62.6% vs 专家90%），数据开源。**

- **链接: [http://arxiv.org/pdf/2505.20321v1](http://arxiv.org/pdf/2505.20321v1)**

> **作者:** Mathew J. Koretsky; Maya Willey; Adi Asija; Owen Bianchi; Chelsea X. Alvarado; Tanay Nayak; Nicole Kuznetsov; Sungwon Kim; Mike A. Nalls; Daniel Khashabi; Faraz Faghri
>
> **备注:** Under Review
>
> **摘要:** Biomedical researchers increasingly rely on large-scale structured databases for complex analytical tasks. However, current text-to-SQL systems often struggle to map qualitative scientific questions into executable SQL, particularly when implicit domain reasoning is required. We introduce BiomedSQL, the first benchmark explicitly designed to evaluate scientific reasoning in text-to-SQL generation over a real-world biomedical knowledge base. BiomedSQL comprises 68,000 question/SQL query/answer triples grounded in a harmonized BigQuery knowledge base that integrates gene-disease associations, causal inference from omics data, and drug approval records. Each question requires models to infer domain-specific criteria, such as genome-wide significance thresholds, effect directionality, or trial phase filtering, rather than rely on syntactic translation alone. We evaluate a range of open- and closed-source LLMs across prompting strategies and interaction paradigms. Our results reveal a substantial performance gap: GPT-o3-mini achieves 59.0% execution accuracy, while our custom multi-step agent, BMSQL, reaches 62.6%, both well below the expert baseline of 90.0%. BiomedSQL provides a new foundation for advancing text-to-SQL systems capable of supporting scientific discovery through robust reasoning over structured biomedical knowledge bases. Our dataset is publicly available at https://huggingface.co/datasets/NIH-CARD/BiomedSQL, and our code is open-source at https://github.com/NIH-CARD/biomedsql.
>
---
#### [new 132] Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion
- **分类: cs.CL**

- **简介: 该论文聚焦扩散语言模型推理加速任务，解决其计算成本高、长输入延迟及token不连贯问题。提出无训练的FreeCache（KV缓存复用）和Guided Diffusion（AR模型引导减少迭代）方法，实现34倍加速且保持质量，首次使扩散模型推理效率超越同规模自回归模型。**

- **链接: [http://arxiv.org/pdf/2505.21467v1](http://arxiv.org/pdf/2505.21467v1)**

> **作者:** Zhanqiu Hu; Jian Meng; Yash Akhauri; Mohamed S. Abdelfattah; Jae-sun Seo; Zhiru Zhang; Udit Gupta
>
> **摘要:** Diffusion language models offer parallel token generation and inherent bidirectionality, promising more efficient and powerful sequence modeling compared to autoregressive approaches. However, state-of-the-art diffusion models (e.g., Dream 7B, LLaDA 8B) suffer from slow inference. While they match the quality of similarly sized Autoregressive (AR) Models (e.g., Qwen2.5 7B, Llama3 8B), their iterative denoising requires multiple full-sequence forward passes, resulting in high computational costs and latency, particularly for long input prompts and long-context scenarios. Furthermore, parallel token generation introduces token incoherence problems, and current sampling heuristics suffer from significant quality drops with decreasing denoising steps. We address these limitations with two training-free techniques. First, we propose FreeCache, a Key-Value (KV) approximation caching technique that reuses stable KV projections across denoising steps, effectively reducing the computational cost of DLM inference. Second, we introduce Guided Diffusion, a training-free method that uses a lightweight pretrained autoregressive model to supervise token unmasking, dramatically reducing the total number of denoising iterations without sacrificing quality. We conduct extensive evaluations on open-source reasoning benchmarks, and our combined methods deliver up to a 34x end-to-end speedup without compromising accuracy. For the first time, diffusion language models achieve a comparable and even faster latency as the widely adopted autoregressive models. Our work successfully paved the way for scaling up the diffusion language model to a broader scope of applications across different domains.
>
---
#### [new 133] M-Wanda: Improving One-Shot Pruning for Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文针对多语言大模型剪枝任务，解决剪枝导致性能下降问题。提出M-Wanda方法，通过语言感知激活统计与动态层稀疏度调整优化跨语言重要性，提升剪枝后多语言性能，首次系统优化多语言剪枝效果。**

- **链接: [http://arxiv.org/pdf/2505.21171v1](http://arxiv.org/pdf/2505.21171v1)**

> **作者:** Rochelle Choenni; Ivan Titov
>
> **摘要:** Multilingual LLM performance is often critically dependent on model size. With an eye on efficiency, this has led to a surge in interest in one-shot pruning methods that retain the benefits of large-scale pretraining while shrinking the model size. However, as pruning tends to come with performance loss, it is important to understand the trade-offs between multilinguality and sparsification. In this work, we study multilingual performance under different sparsity constraints and show that moderate ratios already substantially harm performance. To help bridge this gap, we propose M-Wanda, a pruning method that models cross-lingual variation by incorporating language-aware activation statistics into its pruning criterion and dynamically adjusts layerwise sparsity based on cross-lingual importance. We show that M-Wanda consistently improves performance at minimal additional costs. We are the first to explicitly optimize pruning to retain multilingual performance, and hope to inspire future advances in multilingual pruning.
>
---
#### [new 134] InFact: Informativeness Alignment for Improved LLM Factuality
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于改进大语言模型（LLM）事实性生成任务，针对LLMs生成正确但信息量不足的问题，提出InFact方法：通过信息量对齐机制结合事实基准优化目标，优先选择既正确又详细的信息，提升事实准确性和信息量。**

- **链接: [http://arxiv.org/pdf/2505.20487v1](http://arxiv.org/pdf/2505.20487v1)**

> **作者:** Roi Cohen; Russa Biswas; Gerard de Melo
>
> **摘要:** Factual completeness is a general term that captures how detailed and informative a factually correct text is. For instance, the factual sentence ``Barack Obama was born in the United States'' is factually correct, though less informative than the factual sentence ``Barack Obama was born in Honolulu, Hawaii, United States''. Despite the known fact that LLMs tend to hallucinate and generate factually incorrect text, they might also tend to choose to generate factual text that is indeed factually correct and yet less informative than other, more informative choices. In this work, we tackle this problem by proposing an informativeness alignment mechanism. This mechanism takes advantage of recent factual benchmarks to propose an informativeness alignment objective. This objective prioritizes answers that are both correct and informative. A key finding of our work is that when training a model to maximize this objective or optimize its preference, we can improve not just informativeness but also factuality.
>
---
#### [new 135] Effectiveness of Prompt Optimization in NL2SQL Systems
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于NL2SQL任务，旨在解决生产环境中系统需兼顾高精度与高性能的问题。现有方法依赖检索上下文导致推理耗时，作者提出通过多目标优化选择静态示例集，优化生成SQL的准确性和执行效率。**

- **链接: [http://arxiv.org/pdf/2505.20591v1](http://arxiv.org/pdf/2505.20591v1)**

> **作者:** Sairam Gurajada; Eser Kandogan; Sajjadur Rahman
>
> **摘要:** NL2SQL approaches have greatly benefited from the impressive capabilities of large language models (LLMs). In particular, bootstrapping an NL2SQL system for a specific domain can be as simple as instructing an LLM with sufficient contextual information, such as schema details and translation demonstrations. However, building an accurate system still requires the rigorous task of selecting the right context for each query-including identifying relevant schema elements, cell values, and suitable exemplars that help the LLM understand domain-specific nuances. Retrieval-based methods have become the go-to approach for identifying such context. While effective, these methods introduce additional inference-time costs due to the retrieval process. In this paper, we argue that production scenarios demand high-precision, high-performance NL2SQL systems, rather than simply high-quality SQL generation, which is the focus of most current NL2SQL approaches. In such scenarios, the careful selection of a static set of exemplars-capturing the intricacies of the query log, target database, SQL constructs, and execution latencies-plays a more crucial role than exemplar selection based solely on similarity. The key challenge, however, lies in identifying a representative set of exemplars for a given production setting. To this end, we propose a prompt optimization framework that not only addresses the high-precision requirement but also optimizes the performance of the generated SQL through multi-objective optimization. Preliminary empirical analysis demonstrates the effectiveness of the proposed framework.
>
---
#### [new 136] Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 论文提出轻量级控制器网络，在推理时动态调整LLM各层激活权重，通过预测全局及层特定参数，基于预设的拒绝方向向量，对有害输入实施精准干预，提升安全内容过滤效果，无需修改原模型参数，优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20309v1](http://arxiv.org/pdf/2505.20309v1)**

> **作者:** Amr Hegazy; Mostafa Elhoushi; Amr Alanwar
>
> **摘要:** Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time.
>
---
#### [new 137] A Stereotype Content Analysis on Color-related Social Bias in Large Vision Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于视觉语言模型（LVLMs）偏见评估任务，旨在解决现有研究忽略内容词重要性及颜色影响的问题。提出基于SCM的评价指标与BASIC基准，评估8个LVLMs，发现SCM能有效检测偏见，模型输出存在颜色刻板印象，且架构与参数量影响偏见表现。（98字）**

- **链接: [http://arxiv.org/pdf/2505.20901v1](http://arxiv.org/pdf/2505.20901v1)**

> **作者:** Junhyuk Choi; Minju Kim; Yeseon Hong; Bugeun Kim
>
> **备注:** Under review
>
> **摘要:** As large vision language models(LVLMs) rapidly advance, concerns about their potential to learn and generate social biases and stereotypes are increasing. Previous studies on LVLM's stereotypes face two primary limitations: metrics that overlooked the importance of content words, and datasets that overlooked the effect of color. To address these limitations, this study introduces new evaluation metrics based on the Stereotype Content Model (SCM). We also propose BASIC, a benchmark for assessing gender, race, and color stereotypes. Using SCM metrics and BASIC, we conduct a study with eight LVLMs to discover stereotypes. As a result, we found three findings. (1) The SCM-based evaluation is effective in capturing stereotypes. (2) LVLMs exhibit color stereotypes in the output along with gender and race ones. (3) Interaction between model architecture and parameter sizes seems to affect stereotypes. We release BASIC publicly on [anonymized for review].
>
---
#### [new 138] Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文评估LLM对用户社会人口特征（如年龄、教育水平）的适应能力，解决现有单轮评估无法反映真实对话中多轮情境的问题。提出框架对比用户资料显式提示与对话历史隐式引导下的模型行为一致性，构建合成数据集测试价值观表达，发现模型能调整输出但一致性差异显著，强推理能力模型更优。**

- **链接: [http://arxiv.org/pdf/2505.21362v1](http://arxiv.org/pdf/2505.21362v1)**

> **作者:** Qishuai Zhong; Zongmin Li; Siqi Fan; Aixin Sun
>
> **摘要:** Effective engagement by large language models (LLMs) requires adapting responses to users' sociodemographic characteristics, such as age, occupation, and education level. While many real-world applications leverage dialogue history for contextualization, existing evaluations of LLMs' behavioral adaptation often focus on single-turn prompts. In this paper, we propose a framework to evaluate LLM adaptation when attributes are introduced either (1) explicitly via user profiles in the prompt or (2) implicitly through multi-turn dialogue history. We assess the consistency of model behavior across these modalities. Using a multi-agent pipeline, we construct a synthetic dataset pairing dialogue histories with distinct user profiles and employ questions from the Value Survey Module (VSM 2013) (Hofstede and Hofstede, 2016) to probe value expression. Our findings indicate that most models adjust their expressed values in response to demographic changes, particularly in age and education level, but consistency varies. Models with stronger reasoning capabilities demonstrate greater alignment, indicating the importance of reasoning in robust sociodemographic adaptation.
>
---
#### [new 139] Unveiling Instruction-Specific Neurons & Experts: An Analytical Framework for LLM's Instruction-Following Capabilities
- **分类: cs.CL; cs.LG**

- **简介: 该论文分析LLM指令遵循机制，解决细调如何优化其计算的问题。提出HexaInst数据集和SPARCOM框架，通过识别、评估稀疏组件（神经元/专家）及其变化，揭示其功能与作用，阐明LLM指令执行的内在机制。**

- **链接: [http://arxiv.org/pdf/2505.21191v1](http://arxiv.org/pdf/2505.21191v1)**

> **作者:** Junyan Zhang; Yubo Gao; Yibo Yan; Jungang Li; Zhaorui Hou; Sicheng Tao; Shuliang Liu; Song Dai; Yonghua Hei; Junzhuo Li; Xuming Hu
>
> **摘要:** The finetuning of Large Language Models (LLMs) has significantly advanced their instruction-following capabilities, yet the underlying computational mechanisms driving these improvements remain poorly understood. This study systematically examines how fine-tuning reconfigures LLM computations by isolating and analyzing instruction-specific sparse components, i.e., neurons in dense models and both neurons and experts in Mixture-of-Experts (MoE) architectures. In particular, we introduce HexaInst, a carefully curated and balanced instructional dataset spanning six distinct categories, and propose SPARCOM, a novel analytical framework comprising three key contributions: (1) a method for identifying these sparse components, (2) an evaluation of their functional generality and uniqueness, and (3) a systematic comparison of their alterations. Through experiments, we demonstrate functional generality, uniqueness, and the critical role of these components in instruction execution. By elucidating the relationship between fine-tuning-induced adaptations and sparse computational substrates, this work provides deeper insights into how LLMs internalize instruction-following behavior for the trustworthy LLM community.
>
---
#### [new 140] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?
- **分类: cs.CL**

- **简介: 论文研究LLMs微调中的校准问题，发现其先验知识与微调数据重叠导致过自信，提出CogCalib框架通过认知感知策略改善校准，实验显示ECE降低57%，提升可靠性。**

- **链接: [http://arxiv.org/pdf/2505.20903v1](http://arxiv.org/pdf/2505.20903v1)**

> **作者:** Ziming Wang; Zeyu Shi; Haoyi Zhou; Shiqi Gao; Qingyun Sun; Jianxin Li
>
> **备注:** Accepted to ACL2025 Main; The code will be released soon
>
> **摘要:** Fine-tuned Large Language Models (LLMs) often demonstrate poor calibration, with their confidence scores misaligned with actual performance. While calibration has been extensively studied in models trained from scratch, the impact of LLMs' prior knowledge on calibration during fine-tuning remains understudied. Our research reveals that LLMs' prior knowledge causes potential poor calibration due to the ubiquitous presence of known data in real-world fine-tuning, which appears harmful for calibration. Specifically, data aligned with LLMs' prior knowledge would induce overconfidence, while new knowledge improves calibration. Our findings expose a tension: LLMs' encyclopedic knowledge, while enabling task versatility, undermines calibration through unavoidable knowledge overlaps. To address this, we propose CogCalib, a cognition-aware framework that applies targeted learning strategies according to the model's prior knowledge. Experiments across 7 tasks using 3 LLM families prove that CogCalib significantly improves calibration while maintaining performance, achieving an average 57\% reduction in ECE compared to standard fine-tuning in Llama3-8B. These improvements generalize well to out-of-domain tasks, enhancing the objectivity and reliability of domain-specific LLMs, and making them more trustworthy for critical human-AI interaction applications.
>
---
#### [new 141] RefAV: Towards Planning-Centric Scenario Mining
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于自动驾驶场景挖掘任务，旨在解决从原始驾驶日志中高效识别复杂安全场景的难题。提出RefAV数据集，利用视觉语言模型（VLM）通过自然语言查询定位多智能体交互场景，验证了现有VLM直接应用效果不佳，凸显了场景挖掘的独特挑战。**

- **链接: [http://arxiv.org/pdf/2505.20981v1](http://arxiv.org/pdf/2505.20981v1)**

> **作者:** Cainan Davidson; Deva Ramanan; Neehar Peri
>
> **摘要:** Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Our code and dataset are available at https://github.com/CainanD/RefAV/ and https://argoverse.github.io/user-guide/tasks/scenario_mining.html
>
---
#### [new 142] Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（LVLM）幻觉 mitigation 任务。针对LVLM在多模态生成中虚构图像不存在内容的问题，提出CAAC框架：通过视觉令牌校准（VTC）平衡图像注意力，利用自适应注意力重标定（AAR）基于模型置信度强化视觉 grounding，减少长文本生成中的幻觉。**

- **链接: [http://arxiv.org/pdf/2505.21472v1](http://arxiv.org/pdf/2505.21472v1)**

> **作者:** Mehrdad Fazli; Bowen Wei; Ziwei Zhu
>
> **摘要:** Large vision-language models (LVLMs) achieve impressive performance on multimodal tasks but often suffer from hallucination, and confidently describe objects or attributes not present in the image. Current inference-time interventions, while training-free, struggle to maintain accuracy in open-ended and long-form generation scenarios. We introduce the Confidence-Aware Attention Calibration (CAAC) framework to address this challenge by targeting two key biases: spatial perception bias, which distributes attention disproportionately across image tokens, and modality bias, which shifts focus from visual to textual inputs over time. CAAC employs a two-step approach: Visual-Token Calibration (VTC) to balance attention across visual tokens, and Adaptive Attention Re-Scaling (AAR) to reinforce visual grounding based on the model's confidence. This confidence-driven adjustment ensures consistent visual alignment during generation. Experiments on CHAIR, AMBER, and POPE benchmarks demonstrate that CAAC outperforms baselines, particularly in long-form generations, effectively reducing hallucination.
>
---
#### [new 143] PSRB: A Comprehensive Benchmark for Evaluating Persian ASR Systems
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出PSRB基准，用于评估波斯语ASR系统，解决低资源语言评估难题。通过测试10个ASR模型，分析错误类型并提出加权替换误差指标，揭示模型在方言、儿童语音等场景表现不佳，强调需多样化数据与微调优化。**

- **链接: [http://arxiv.org/pdf/2505.21230v1](http://arxiv.org/pdf/2505.21230v1)**

> **作者:** Nima Sedghiyeh; Sara Sadeghi; Reza Khodadadi; Farzin Kashani; Omid Aghdaei; Somayeh Rahimi; Mohammad Sadegh Safari
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Although Automatic Speech Recognition (ASR) systems have become an integral part of modern technology, their evaluation remains challenging, particularly for low-resource languages such as Persian. This paper introduces Persian Speech Recognition Benchmark(PSRB), a comprehensive benchmark designed to address this gap by incorporating diverse linguistic and acoustic conditions. We evaluate ten ASR systems, including state-of-the-art commercial and open-source models, to examine performance variations and inherent biases. Additionally, we conduct an in-depth analysis of Persian ASR transcriptions, identifying key error types and proposing a novel metric that weights substitution errors. This metric enhances evaluation robustness by reducing the impact of minor and partial errors, thereby improving the precision of performance assessment. Our findings indicate that while ASR models generally perform well on standard Persian, they struggle with regional accents, children's speech, and specific linguistic challenges. These results highlight the necessity of fine-tuning and incorporating diverse, representative training datasets to mitigate biases and enhance overall ASR performance. PSRB provides a valuable resource for advancing ASR research in Persian and serves as a framework for developing benchmarks in other low-resource languages. A subset of the PSRB dataset is publicly available at https://huggingface.co/datasets/PartAI/PSRB.
>
---
#### [new 144] Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文属于jailbreak攻击研究任务，旨在突破现有方法因预定义策略空间受限导致的安全模型绕过瓶颈。提出基于ELM理论分解攻击策略并结合遗传优化与意图评估的框架，实验在Claude-3.5达90%成功率，展现强跨模型迁移性。**

- **链接: [http://arxiv.org/pdf/2505.21277v1](http://arxiv.org/pdf/2505.21277v1)**

> **作者:** Yao Huang; Yitong Sun; Shouwei Ruan; Yichi Zhang; Yinpeng Dong; Xingxing Wei
>
> **备注:** 19 pages, 20 figures, accepted by ACL 2025, Findings
>
> **摘要:** Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO.
>
---
#### [new 145] Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文是系统综述，探讨基础模型（如LLMs、多模态模型）在移动服务机器人中的应用，解决多模态融合、实时决策、任务泛化及人机交互等挑战。通过分析基础模型在传感器融合、语言控制和自适应任务执行中的作用，综述其在家庭、医疗和自动化领域的应用，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.20503v1](http://arxiv.org/pdf/2505.20503v1)**

> **作者:** Matthew Lisondra; Beno Benhabib; Goldie Nejat
>
> **摘要:** Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interactions, robots can improve understanding, adapt to, and execute complex tasks in dynamic real-world environments. However, embodied AI in mobile service robots continues to face key challenges, including multimodal sensor fusion, real-time decision-making under uncertainty, task generalization, and effective human-robot interactions (HRI). In this paper, we present the first systematic review of the integration of foundation models in mobile service robotics, identifying key open challenges in embodied AI and examining how foundation models can address them. Namely, we explore the role of such models in enabling real-time sensor fusion, language-conditioned control, and adaptive task execution. Furthermore, we discuss real-world applications in the domestic assistance, healthcare, and service automation sectors, demonstrating the transformative impact of foundation models on service robotics. We also include potential future research directions, emphasizing the need for predictive scaling laws, autonomous long-term adaptation, and cross-embodiment generalization to enable scalable, efficient, and robust deployment of foundation models in human-centric robotic systems.
>
---
#### [new 146] An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件工程（SE）评估任务，旨在解决LLM生成代码等工件时自动评估与人工评估准确性差距大的问题。提出SWE-Judge方法：通过5种评估策略构建LLM法官集合，结合动态选择机制生成最终评分。实验显示其与人类判断一致性较现有方法提升5.9%-183.8%，可作为可靠替代方案。**

- **链接: [http://arxiv.org/pdf/2505.20854v1](http://arxiv.org/pdf/2505.20854v1)**

> **作者:** Xin Zhou; Kisub Kim; Ting Zhang; Martin Weyssow; Luis F. Gomes; Guang Yang; David Lo
>
> **备注:** 20 pages
>
> **摘要:** Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, other existing automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts. In this paper, we present SWE-Judge, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SWE-Judge first defines five distinct evaluation strategies, each implemented as an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges to produce a final correctness score through ensembling. We evaluate SWE-Judge across a diverse set of software engineering (SE) benchmarks, including CoNaLa, Card2Code, HumanEval-X, APPS, APR-Assess, and Summary-Assess. These benchmarks span three SE tasks: code generation, automated program repair, and code summarization. Experimental results demonstrate that SWE-Judge consistently achieves a higher correlation with human judgments, with improvements ranging from 5.9% to 183.8% over existing automatic metrics. Furthermore, SWE-Judge reaches agreement levels with human annotators that are comparable to inter-annotator agreement in code generation and program repair tasks. These findings underscore SWE-Judge's potential as a scalable and reliable alternative to human evaluation.
>
---
#### [new 147] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Roboflow100-VL基准，针对视觉语言模型（VLMs）在分布外场景（如医疗影像）目标检测泛化差的问题，构建100个多模态数据集并评估模型在不同训练模式下的表现，证明需通过少量样本与文本描述对齐提升性能。**

- **链接: [http://arxiv.org/pdf/2505.20612v1](http://arxiv.org/pdf/2505.20612v1)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Our code and dataset are available at https://github.com/roboflow/rf100-vl/ and https://universe.roboflow.com/rf100-vl/
>
---
#### [new 148] Scaling over Scaling: Exploring Test-Time Scaling Pareto in Large Reasoning Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大型推理模型（LRMs）测试时资源优化任务，解决扩展计算资源时的效益极限问题。提出Test-Time Scaling Performance Model（TTSPM），分析并行与串行扩展的饱和点，通过理论推导与实验验证，指导高效推理策略开发。**

- **链接: [http://arxiv.org/pdf/2505.20522v1](http://arxiv.org/pdf/2505.20522v1)**

> **作者:** Jian Wang; Boyan Zhu; Chak Tou Leong; Yongqi Li; Wenjie Li
>
> **备注:** Work in progress
>
> **摘要:** Large reasoning models (LRMs) have exhibited the capacity of enhancing reasoning performance via internal test-time scaling. Building upon this, a promising direction is to further scale test-time compute to unlock even greater reasoning capabilities. However, as we push these scaling boundaries, systematically understanding the practical limits and achieving optimal resource allocation becomes a critical challenge. In this paper, we investigate the scaling Pareto of test-time scaling and introduce the Test-Time Scaling Performance Model (TTSPM). We theoretically analyze two fundamental paradigms for such extended scaling, parallel scaling and sequential scaling, from a probabilistic modeling perspective. Our primary contribution is the derivation of the saturation point on the scaling budget for both strategies, identifying thresholds beyond which additional computation yields diminishing returns. Remarkably, despite their distinct mechanisms, both paradigms converge to a unified mathematical structure in their upper bounds. We empirically validate our theoretical findings on challenging reasoning benchmarks, including AIME, MATH-500, and GPQA, demonstrating the practical utility of these bounds for test-time resource allocation. We hope that this work provides insights into the cost-benefit trade-offs of test-time scaling, guiding the development of more resource-efficient inference strategies for large reasoning models.
>
---
#### [new 149] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于表联合搜索（TUS）任务，针对现有数据湖中TUS基准评估不足的问题，指出其依赖数据特性导致简单基线表现过优，无法有效衡量语义理解能力。研究分析现有基准局限性，提出改进标准以提升评估可靠性。**

- **链接: [http://arxiv.org/pdf/2505.21329v1](http://arxiv.org/pdf/2505.21329v1)**

> **作者:** Allaa Boutaleb; Bernd Amann; Hubert Naacke; Rafael Angarita
>
> **备注:** Accepted @ ACL 2025's Table Representation Learning Workshop (TRL)
>
> **摘要:** Recent table representation learning and data discovery methods tackle table union search (TUS) within data lakes, which involves identifying tables that can be unioned with a given query table to enrich its content. These methods are commonly evaluated using benchmarks that aim to assess semantic understanding in real-world TUS tasks. However, our analysis of prominent TUS benchmarks reveals several limitations that allow simple baselines to perform surprisingly well, often outperforming more sophisticated approaches. This suggests that current benchmark scores are heavily influenced by dataset-specific characteristics and fail to effectively isolate the gains from semantic understanding. To address this, we propose essential criteria for future benchmarks to enable a more realistic and reliable evaluation of progress in semantic table union search.
>
---
#### [new 150] Cultural Awareness in Vision-Language Models: A Cross-Country Exploration
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型（VLMs）偏见评估任务，旨在探究VLMs在跨国家场景中编码文化差异及种族、性别、身体特征偏见的问题。研究提出三类检索任务（种族-国家、特质-国家、体征-国家关联分析），揭示VLMs存在强化社会刻板印象的持续偏见。**

- **链接: [http://arxiv.org/pdf/2505.20326v1](http://arxiv.org/pdf/2505.20326v1)**

> **作者:** Avinash Madasu; Vasudev Lal; Phillip Howard
>
> **摘要:** Vision-Language Models (VLMs) are increasingly deployed in diverse cultural contexts, yet their internal biases remain poorly understood. In this work, we propose a novel framework to systematically evaluate how VLMs encode cultural differences and biases related to race, gender, and physical traits across countries. We introduce three retrieval-based tasks: (1) Race to Country retrieval, which examines the association between individuals from specific racial groups (East Asian, White, Middle Eastern, Latino, South Asian, and Black) and different countries; (2) Personal Traits to Country retrieval, where images are paired with trait-based prompts (e.g., Smart, Honest, Criminal, Violent) to investigate potential stereotypical associations; and (3) Physical Characteristics to Country retrieval, focusing on visual attributes like skinny, young, obese, and old to explore how physical appearances are culturally linked to nations. Our findings reveal persistent biases in VLMs, highlighting how visual representations may inadvertently reinforce societal stereotypes.
>
---
#### [new 151] Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属强化学习与LLM推理任务，针对传统马尔可夫RL限制探索、无法有效激发反思性推理的问题，提出基于贝叶斯自适应RL的BARL算法。通过后验分布优化，指导LLM动态切换策略，提升测试阶段的探索效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.20561v1](http://arxiv.org/pdf/2505.20561v1)**

> **作者:** Shenao Zhang; Yaqing Wang; Yinxiao Liu; Tianqi Liu; Peter Grabowski; Eugene Ie; Zhaoran Wang; Yunxuan Li
>
> **摘要:** Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as backtracking and error correction. However, conventional Markovian RL confines exploration to the training phase to learn an optimal deterministic policy and depends on the history contexts only through the current state. Therefore, it remains unclear whether reflective reasoning will emerge during Markovian RL training, or why they are beneficial at test time. To remedy this, we recast reflective exploration within the Bayes-Adaptive RL framework, which explicitly optimizes the expected return under a posterior distribution over Markov decision processes. This Bayesian formulation inherently incentivizes both reward-maximizing exploitation and information-gathering exploration via belief updates. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms standard Markovian RL approaches at test time, achieving superior token efficiency with improved exploration effectiveness. Our code is available at https://github.com/shenao-zhang/BARL.
>
---
#### [new 152] ID-Align: RoPE-Conscious Position Remapping for Dynamic High-Resolution Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型（VLM）中高分辨率图像与文本交互不足的问题，提出ID-Align方法。通过重新映射位置ID，使高分辨率图像token继承缩略图token的位置信息并限制索引扩张，缓解RoPE位置嵌入的衰减效应，提升跨模态交互，在MMBench等任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.21465v1](http://arxiv.org/pdf/2505.21465v1)**

> **作者:** Bozhou Li; Wentao Zhang
>
> **摘要:** Currently, a prevalent approach for enhancing Vision-Language Models (VLMs) performance is to encode both the high-resolution version and the thumbnail of an image simultaneously. While effective, this method generates a large number of image tokens. When combined with the widely used Rotary Position Embedding (RoPE), its long-term decay property hinders the interaction between high-resolution tokens and thumbnail tokens, as well as between text and image. To address these issues, we propose ID-Align, which alleviates these problems by reordering position IDs. In this method, high-resolution tokens inherit IDs from their corresponding thumbnail token while constraining the overexpansion of positional indices. Our experiments conducted within the LLaVA-Next framework demonstrate that ID-Align achieves significant improvements, including a 6.09% enhancement on MMBench's relation reasoning tasks and notable gains across multiple benchmarks. Our code is available at the following link: https://github.com/zooblastlbz/ID-Align.
>
---
#### [new 153] Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于理论分析任务，研究暂停符号（如"..."）如何提升Transformer模型的表达能力。通过证明添加暂停符号使常数深度Transformer的计算能力从AC⁰子集扩展至整个AC⁰（对数精度达TC⁰），并实验证明其能学习原本无法处理的奇偶校验函数，理论解释了暂停符号增强模型推理的机制。**

- **链接: [http://arxiv.org/pdf/2505.21024v1](http://arxiv.org/pdf/2505.21024v1)**

> **作者:** Charles London; Varun Kanade
>
> **摘要:** Pause tokens, simple filler symbols such as "...", consistently improve Transformer performance on both language and mathematical tasks, yet their theoretical effect remains unexplained. We provide the first formal separation result, proving that adding pause tokens to constant-depth, logarithmic-width Transformers strictly increases their computational expressivity. With bounded-precision activations, Transformers without pause tokens compute only a strict subset of $\mathsf{AC}^0$ functions, while adding a polynomial number of pause tokens allows them to express the entire class. For logarithmic-precision Transformers, we show that adding pause tokens achieves expressivity equivalent to $\mathsf{TC}^0$, matching known upper bounds. Empirically, we demonstrate that two-layer causally masked Transformers can learn parity when supplied with pause tokens, a function that they appear unable to learn without them. Our results provide a rigorous theoretical explanation for prior empirical findings, clarify how pause tokens interact with width, depth, and numeric precision, and position them as a distinct mechanism, complementary to chain-of-thought prompting, for enhancing Transformer reasoning.
>
---
#### [new 154] TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research
- **分类: cs.IR; cs.AI; cs.CL; H.3; I.2**

- **简介: 该论文构建了萜类研究专用知识库TeroSeek，解决其跨学科研究中知识整合难题。团队整合20年文献，开发基于RAG框架的AI问答系统，提供精准信息，超越通用大模型，作为开源工具服务科研。**

- **链接: [http://arxiv.org/pdf/2505.20663v1](http://arxiv.org/pdf/2505.20663v1)**

> **作者:** Xu Kang; Siqi Jiang; Kangwei Xu; Jiahao Li; Ruibo Wu
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at http://teroseek.qmclab.com.
>
---
#### [new 155] What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于图像编辑评估任务，针对现有指标与人类判断不一致的问题，提出DICE模型，通过差异检测器和连贯性评估器，结合多模态大模型与混合训练策略，有效识别并评估指令引导的图像修改效果，实验显示与人类判断强相关。**

- **链接: [http://arxiv.org/pdf/2505.20405v1](http://arxiv.org/pdf/2505.20405v1)**

> **作者:** Lorenzo Baraldi; Davide Bucciarelli; Federico Betti; Marcella Cornia; Lorenzo Baraldi; Nicu Sebe; Rita Cucchiara
>
> **摘要:** Instruction-based image editing models offer increased personalization opportunities in generative tasks. However, properly evaluating their results is challenging, and most of the existing metrics lag in terms of alignment with human judgment and explainability. To tackle these issues, we introduce DICE (DIfference Coherence Estimator), a model designed to detect localized differences between the original and the edited image and to assess their relevance to the given modification request. DICE consists of two key components: a difference detector and a coherence estimator, both built on an autoregressive Multimodal Large Language Model (MLLM) and trained using a strategy that leverages self-supervision, distillation from inpainting networks, and full supervision. Through extensive experiments, we evaluate each stage of our pipeline, comparing different MLLMs within the proposed framework. We demonstrate that DICE effectively identifies coherent edits, effectively evaluating images generated by different editing models with a strong correlation with human judgment. We publicly release our source code, models, and data.
>
---
#### [new 156] Hardware-Efficient Attention for Fast Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大模型解码中KV缓存加载导致的高延迟及并行性不足问题，提出Grouped-Tied Attention（GTA）和Grouped Latent Attention（GLA）方法。GTA通过复用键值状态减少内存访问，GLA优化并行解码效率，实验显示两者均提升速度与吞吐量，降低延迟。**

- **链接: [http://arxiv.org/pdf/2505.21487v1](http://arxiv.org/pdf/2505.21487v1)**

> **作者:** Ted Zadouri; Hubert Strauss; Tri Dao
>
> **备注:** 37 pages, 15 figures, 45 tables
>
> **摘要:** LLM decoding is bottlenecked for large batches and long contexts by loading the key-value (KV) cache from high-bandwidth memory, which inflates per-token latency, while the sequential nature of decoding limits parallelism. We analyze the interplay among arithmetic intensity, parallelization, and model quality and question whether current architectures fully exploit modern hardware. This work redesigns attention to perform more computation per byte loaded from memory to maximize hardware efficiency without trading off parallel scalability. We first propose Grouped-Tied Attention (GTA), a simple variant that combines and reuses key and value states, reducing memory transfers without compromising model quality. We then introduce Grouped Latent Attention (GLA), a parallel-friendly latent attention paired with low-level optimizations for fast decoding while maintaining high model quality. Experiments show that GTA matches Grouped-Query Attention (GQA) quality while using roughly half the KV cache and that GLA matches Multi-head Latent Attention (MLA) and is easier to shard. Our optimized GLA kernel is up to 2$\times$ faster than FlashMLA, for example, in a speculative decoding setting when the query length exceeds one. Furthermore, by fetching a smaller KV cache per device, GLA reduces end-to-end latency and increases throughput in online serving benchmarks by up to 2$\times$.
>
---
#### [new 157] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于任务导向的部件分割任务，旨在解决现有视觉语言模型（VLMs）难以理解物体部件及其功能的问题。作者构建了含部件分割标注和任务指令的新基准InstructPart，提出通过微调提升性能的基线模型，推动VLMs在机器人等领域的应用。**

- **链接: [http://arxiv.org/pdf/2505.18291v1](http://arxiv.org/pdf/2505.18291v1)**

> **作者:** Zifu Wan; Yaqi Xie; Ce Zhang; Zhiqiu Lin; Zihan Wang; Simon Stepputtis; Deva Ramanan; Katia Sycara
>
> **备注:** Accepted by ACL 2025 Main. Project page: https://zifuwan.github.io/InstructPart/
>
> **摘要:** Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object's functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: https://zifuwan.github.io/InstructPart/.
>
---
#### [new 158] What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属推荐系统任务，旨在解决LLMs难以有效利用用户-物品交互的协作信号问题。通过对比LLMs与矩阵分解模型，提出检索增强生成方法，结合结构化交互数据提升推荐质量，实验表明该方法显著改善效果。**

- **链接: [http://arxiv.org/pdf/2505.20730v1](http://arxiv.org/pdf/2505.20730v1)**

> **作者:** Shahrooz Pouryousef
>
> **摘要:** User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders.
>
---
#### [new 159] The Multilingual Divide and Its Impact on Global AI Safety
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全与政策分析任务，旨在解决多语言差距导致的全球AI安全不平等问题。分析语言鸿沟成因及安全风险，提出通过多语料建设、透明度提升和政策支持缩小差距的建议。**

- **链接: [http://arxiv.org/pdf/2505.21344v1](http://arxiv.org/pdf/2505.21344v1)**

> **作者:** Aidan Peppin; Julia Kreutzer; Alice Schoenauer Sebag; Kelly Marchisio; Beyza Ermis; John Dang; Samuel Cahyawijaya; Shivalika Singh; Seraphina Goldfarb-Tarrant; Viraat Aryabumi; Aakanksha; Wei-Yin Ko; Ahmet Üstün; Matthias Gallé; Marzieh Fadaee; Sara Hooker
>
> **摘要:** Despite advances in large language model capabilities in recent years, a large gap remains in their capabilities and safety performance for many languages beyond a relatively small handful of globally dominant languages. This paper provides researchers, policymakers and governance experts with an overview of key challenges to bridging the "language gap" in AI and minimizing safety risks across languages. We provide an analysis of why the language gap in AI exists and grows, and how it creates disparities in global AI safety. We identify barriers to address these challenges, and recommend how those working in policy and governance can help address safety concerns associated with the language gap by supporting multilingual dataset creation, transparency, and research.
>
---
#### [new 160] Can we Debias Social Stereotypes in AI-Generated Images? Examining Text-to-Image Outputs and User Perceptions
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究文本到图像生成中的社会偏见问题，提出偏见检测标准和SSI指数，评估三大模型后发现其输出存在刻板印象。通过LLM优化提示词使偏见降低51%-69%，但用户研究显示去偏可能影响上下文相关性。探讨需在去偏与现实复杂性间平衡，呼吁AI系统兼顾多样性和真实性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20692v1](http://arxiv.org/pdf/2505.20692v1)**

> **作者:** Saharsh Barve; Andy Mao; Jiayue Melissa Shi; Prerna Juneja; Koustuv Saha
>
> **摘要:** Recent advances in generative AI have enabled visual content creation through text-to-image (T2I) generation. However, despite their creative potential, T2I models often replicate and amplify societal stereotypes -- particularly those related to gender, race, and culture -- raising important ethical concerns. This paper proposes a theory-driven bias detection rubric and a Social Stereotype Index (SSI) to systematically evaluate social biases in T2I outputs. We audited three major T2I model outputs -- DALL-E-3, Midjourney-6.1, and Stability AI Core -- using 100 queries across three categories -- geocultural, occupational, and adjectival. Our analysis reveals that initial outputs are prone to include stereotypical visual cues, including gendered professions, cultural markers, and western beauty norms. To address this, we adopted our rubric to conduct targeted prompt refinement using LLMs, which significantly reduced bias -- SSI dropped by 61% for geocultural, 69% for occupational, and 51% for adjectival queries. We complemented our quantitative analysis through a user study examining perceptions, awareness, and preferences around AI-generated biased imagery. Our findings reveal a key tension -- although prompt refinement can mitigate stereotypes, it can limit contextual alignment. Interestingly, users often perceived stereotypical images to be more aligned with their expectations. We discuss the need to balance ethical debiasing with contextual relevance and call for T2I systems that support global diversity and inclusivity while not compromising the reflection of real-world social complexity.
>
---
#### [new 161] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对开放领域金融问答任务，解决标准化文档（如SEC文件）因重复文本和相似结构导致传统RAG方法检索冗余、影响准确性的难题。提出HiREC框架，通过分层检索（先文档后段落）及证据整理去除冗余，并自动生成补充查询。同时构建LOFin数据集（含14万+SEC文件及1595问答对）进行评估。**

- **链接: [http://arxiv.org/pdf/2505.20368v1](http://arxiv.org/pdf/2505.20368v1)**

> **作者:** Jaeyoung Choe; Jihoon Kim; Woohwan Jung
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** Retrieval-augmented generation (RAG) based large language models (LLMs) are widely used in finance for their excellent performance on knowledge-intensive tasks. However, standardized documents (e.g., SEC filing) share similar formats such as repetitive boilerplate texts, and similar table structures. This similarity forces traditional RAG methods to misidentify near-duplicate text, leading to duplicate retrieval that undermines accuracy and completeness. To address these issues, we propose the Hierarchical Retrieval with Evidence Curation (HiREC) framework. Our approach first performs hierarchical retrieval to reduce confusion among similar texts. It first retrieve related documents and then selects the most relevant passages from the documents. The evidence curation process removes irrelevant passages. When necessary, it automatically generates complementary queries to collect missing information. To evaluate our approach, we construct and release a Large-scale Open-domain Financial (LOFin) question answering benchmark that includes 145,897 SEC documents and 1,595 question-answer pairs. Our code and data are available at https://github.com/deep-over/LOFin-bench-HiREC.
>
---
#### [new 162] Creativity in LLM-based Multi-Agent Systems: A Survey
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文是关于LLM驱动多智能体系统（MAS）创造力的综述，旨在填补现有研究在创造力维度（如创新输出生成与评估、角色设计、协作流程）的空白。提出创造力分类框架、生成技术（发散探索、迭代优化、协作合成）及评估挑战，并探讨标准化与偏见等难题，为创意MAS研发提供方法论与方向。**

- **链接: [http://arxiv.org/pdf/2505.21116v1](http://arxiv.org/pdf/2505.21116v1)**

> **作者:** Yi-Cheng Lin; Kang-Chieh Chen; Zhe-Yan Li; Tzu-Heng Wu; Tzu-Hsuan Wu; Kuan-Yu Chen; Hung-yi Lee; Yun-Nung Chen
>
> **备注:** 23 pages
>
> **摘要:** Large language model (LLM)-driven multi-agent systems (MAS) are transforming how humans and AIs collaboratively generate ideas and artifacts. While existing surveys provide comprehensive overviews of MAS infrastructures, they largely overlook the dimension of \emph{creativity}, including how novel outputs are generated and evaluated, how creativity informs agent personas, and how creative workflows are coordinated. This is the first survey dedicated to creativity in MAS. We focus on text and image generation tasks, and present: (1) a taxonomy of agent proactivity and persona design; (2) an overview of generation techniques, including divergent exploration, iterative refinement, and collaborative synthesis, as well as relevant datasets and evaluation metrics; and (3) a discussion of key challenges, such as inconsistent evaluation standards, insufficient bias mitigation, coordination conflicts, and the lack of unified benchmarks. This survey offers a structured framework and roadmap for advancing the development, evaluation, and standardization of creative MAS.
>
---
#### [new 163] Optimizing fMRI Data Acquisition for Decoding Natural Speech with Limited Participants
- **分类: q-bio.NC; cs.CL; cs.LG**

- **简介: 该论文研究有限被试条件下优化fMRI数据解码自然语音的策略，通过对比发现多被试训练未优于单被试，刺激相似性影响微弱，解码更擅长句法而非语义，复杂句式或丰富语义内容更难解码，建议需深度个体数据或扩大样本。**

- **链接: [http://arxiv.org/pdf/2505.21304v1](http://arxiv.org/pdf/2505.21304v1)**

> **作者:** Louis Jalouzot; Alexis Thual; Yair Lakretz; Christophe Pallier; Bertrand Thirion
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** We investigate optimal strategies for decoding perceived natural speech from fMRI data acquired from a limited number of participants. Leveraging Lebel et al. (2023)'s dataset of 8 participants, we first demonstrate the effectiveness of training deep neural networks to predict LLM-derived text representations from fMRI activity. Then, in this data regime, we observe that multi-subject training does not improve decoding accuracy compared to single-subject approach. Furthermore, training on similar or different stimuli across subjects has a negligible effect on decoding accuracy. Finally, we find that our decoders better model syntactic than semantic features, and that stories containing sentences with complex syntax or rich semantic content are more challenging to decode. While our results demonstrate the benefits of having extensive data per participant (deep phenotyping), they suggest that leveraging multi-subject for natural speech decoding likely requires deeper phenotyping or a substantially larger cohort.
>
---
#### [new 164] How Do Transformers Learn Variable Binding in Symbolic Programs?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型如何在符号程序中学习变量绑定。针对其缺乏显式绑定机制的问题，通过训练模型解析多层变量赋值链（含干扰链），揭示其训练分三阶段发展：随机预测→启发式策略→系统性解析。发现模型利用残差流和注意力头构建动态可寻址内存，实现变量追踪，阐明神经网络与符号计算的结合机制。**

- **链接: [http://arxiv.org/pdf/2505.20896v1](http://arxiv.org/pdf/2505.20896v1)**

> **作者:** Yiwei Wu; Atticus Geiger; Raphaël Millière
>
> **备注:** 16 pages, 10 figures, 1 table. To appear in the Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Variable binding -- the ability to associate variables with values -- is fundamental to symbolic computation and cognition. Although classical architectures typically implement variable binding via addressable memory, it is not well understood how modern neural networks lacking built-in binding operations may acquire this capacity. We investigate this by training a Transformer to dereference queried variables in symbolic programs where variables are assigned either numerical constants or other variables. Each program requires following chains of variable assignments up to four steps deep to find the queried value, and also contains irrelevant chains of assignments acting as distractors. Our analysis reveals a developmental trajectory with three distinct phases during training: (1) random prediction of numerical constants, (2) a shallow heuristic prioritizing early variable assignments, and (3) the emergence of a systematic mechanism for dereferencing assignment chains. Using causal interventions, we find that the model learns to exploit the residual stream as an addressable memory space, with specialized attention heads routing information across token positions. This mechanism allows the model to dynamically track variable bindings across layers, resulting in accurate dereferencing. Our results show how Transformer models can learn to implement systematic variable binding without explicit architectural support, bridging connectionist and symbolic approaches.
>
---
#### [new 165] Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers
- **分类: cs.CV; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出Paper2Poster系统，解决学术海报自动化生成任务。针对长论文压缩为视觉连贯海报的挑战，团队构建首个评估基准（含视觉语义、文本连贯性等指标），设计多阶段Pipeline（解析论文、二叉树布局规划、渲染优化），开源方案性能超GPT-4o，成本仅$0.005。**

- **链接: [http://arxiv.org/pdf/2505.21497v1](http://arxiv.org/pdf/2505.21497v1)**

> **作者:** Wei Pang; Kevin Qinghong Lin; Xiangru Jian; Xi He; Philip Torr
>
> **备注:** Project Page: https://github.com/Paper2Poster/Paper2Poster
>
> **摘要:** Academic poster generation is a crucial yet challenging task in scientific communication, requiring the compression of long-context interleaved documents into a single, visually coherent page. To address this challenge, we introduce the first benchmark and metric suite for poster generation, which pairs recent conference papers with author-designed posters and evaluates outputs on (i)Visual Quality-semantic alignment with human posters, (ii)Textual Coherence-language fluency, (iii)Holistic Assessment-six fine-grained aesthetic and informational criteria scored by a VLM-as-judge, and notably (iv)PaperQuiz-the poster's ability to convey core paper content as measured by VLMs answering generated quizzes. Building on this benchmark, we propose PosterAgent, a top-down, visual-in-the-loop multi-agent pipeline: the (a)Parser distills the paper into a structured asset library; the (b)Planner aligns text-visual pairs into a binary-tree layout that preserves reading order and spatial balance; and the (c)Painter-Commenter loop refines each panel by executing rendering code and using VLM feedback to eliminate overflow and ensure alignment. In our comprehensive evaluation, we find that GPT-4o outputs-though visually appealing at first glance-often exhibit noisy text and poor PaperQuiz scores, and we find that reader engagement is the primary aesthetic bottleneck, as human-designed posters rely largely on visual semantics to convey meaning. Our fully open-source variants (e.g. based on the Qwen-2.5 series) outperform existing 4o-driven multi-agent systems across nearly all metrics, while using 87% fewer tokens. It transforms a 22-page paper into a finalized yet editable .pptx poster - all for just $0.005. These findings chart clear directions for the next generation of fully automated poster-generation models. The code and datasets are available at https://github.com/Paper2Poster/Paper2Poster.
>
---
#### [new 166] Reinforcing General Reasoning without Verifiers
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出无验证器方法VeriFree，解决现有强化学习依赖规则验证或额外LLM验证器的问题。通过直接最大化生成参考答案的概率，扩展RL至通用推理领域，在多任务基准上表现优异，减少计算需求并避免验证器缺陷。（99字）**

- **链接: [http://arxiv.org/pdf/2505.21493v1](http://arxiv.org/pdf/2505.21493v1)**

> **作者:** Xiangxin Zhou; Zichen Liu; Anya Sims; Haonan Wang; Tianyu Pang; Chongxuan Li; Liang Wang; Min Lin; Chao Du
>
> **摘要:** The recent paradigm shift towards training large language models (LLMs) using DeepSeek-R1-Zero-style reinforcement learning (RL) on verifiable rewards has led to impressive advancements in code and mathematical reasoning. However, this methodology is limited to tasks where rule-based answer verification is possible and does not naturally extend to real-world domains such as chemistry, healthcare, engineering, law, biology, business, and economics. Current practical workarounds use an additional LLM as a model-based verifier; however, this introduces issues such as reliance on a strong verifier LLM, susceptibility to reward hacking, and the practical burden of maintaining the verifier model in memory during training. To address this and extend DeepSeek-R1-Zero-style training to general reasoning domains, we propose a verifier-free method (VeriFree) that bypasses answer verification and instead uses RL to directly maximize the probability of generating the reference answer. We compare VeriFree with verifier-based methods and demonstrate that, in addition to its significant practical benefits and reduced compute requirements, VeriFree matches and even surpasses verifier-based methods on extensive evaluations across MMLU-Pro, GPQA, SuperGPQA, and math-related benchmarks. Moreover, we provide insights into this method from multiple perspectives: as an elegant integration of training both the policy and implicit verifier in a unified model, and as a variational optimization approach. Code is available at https://github.com/sail-sg/VeriFree.
>
---
#### [new 167] PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI安全领域，提出PoisonSwarm框架解决LLM生成有害数据的可靠性与多样性不足问题。通过模型众包策略，分解良性模板为语义单元，动态切换模型进行毒化与优化，提升合成成功率与多样性。**

- **链接: [http://arxiv.org/pdf/2505.21184v1](http://arxiv.org/pdf/2505.21184v1)**

> **作者:** Yu Yan; Sheng Sun; Zhifei Zheng; Ziji Hao; Teli Liu; Min Liu
>
> **摘要:** To construct responsible and secure AI applications, harmful information data is widely utilized for adversarial testing and the development of safeguards. Existing studies mainly leverage Large Language Models (LLMs) to synthesize data to obtain high-quality task datasets at scale, thereby avoiding costly human annotation. However, limited by the safety alignment mechanisms of LLMs, the synthesis of harmful data still faces challenges in generation reliability and content diversity. In this study, we propose a novel harmful information synthesis framework, PoisonSwarm, which applies the model crowdsourcing strategy to generate diverse harmful data while maintaining a high success rate. Specifically, we generate abundant benign data as the based templates in a counterfactual manner. Subsequently, we decompose each based template into multiple semantic units and perform unit-by-unit toxification and final refinement through dynamic model switching, thus ensuring the success of synthesis. Experimental results demonstrate that PoisonSwarm achieves state-of-the-art performance in synthesizing different categories of harmful data with high scalability and diversity.
>
---
#### [new 168] Towards Emotionally Consistent Text-Based Speech Editing: Introducing EmoCorrector and The ECD-TSE Dataset
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于文本驱动语音编辑（TSE）任务，旨在解决现有方法忽视文本修改导致的语音情感不一致问题。提出EmoCorrector方案，通过检索增强生成匹配情感的语音样本，修正情感偏差，并构建ECD-TSE数据集支持训练与评估，实验验证其提升情感表达一致性。**

- **链接: [http://arxiv.org/pdf/2505.20341v1](http://arxiv.org/pdf/2505.20341v1)**

> **作者:** Rui Liu; Pu Gao; Jiatian Xi; Berrak Sisman; Carlos Busso; Haizhou Li
>
> **备注:** INTERSPEECH2025. Code and audio examples: https://github.com/AI-S2-Lab/EmoCorrector
>
> **摘要:** Text-based speech editing (TSE) modifies speech using only text, eliminating re-recording. However, existing TSE methods, mainly focus on the content accuracy and acoustic consistency of synthetic speech segments, and often overlook the emotional shifts or inconsistency issues introduced by text changes. To address this issue, we propose EmoCorrector, a novel post-correction scheme for TSE. EmoCorrector leverages Retrieval-Augmented Generation (RAG) by extracting the edited text's emotional features, retrieving speech samples with matching emotions, and synthesizing speech that aligns with the desired emotion while preserving the speaker's identity and quality. To support the training and evaluation of emotional consistency modeling in TSE, we pioneer the benchmarking Emotion Correction Dataset for TSE (ECD-TSE). The prominent aspect of ECD-TSE is its inclusion of $<$text, speech$>$ paired data featuring diverse text variations and a range of emotional expressions. Subjective and objective experiments and comprehensive analysis on ECD-TSE confirm that EmoCorrector significantly enhances the expression of intended emotion while addressing emotion inconsistency limitations in current TSE methods. Code and audio examples are available at https://github.com/AI-S2-Lab/EmoCorrector.
>
---
#### [new 169] MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属视频时间理解任务，针对多模态大语言模型（MLLMs）在细粒度时间推理上的不足，提出MUSEG方法。通过时间戳感知的多片段接地机制及分阶段奖励的强化学习框架，提升模型对视频时间信息的推理能力，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20715v1](http://arxiv.org/pdf/2505.20715v1)**

> **作者:** Fuwen Luo; Shengfeng Lou; Chi Chen; Ziyue Wang; Chenliang Li; Weizhou Shen; Jiyue Guo; Peng Li; Ming Yan; Ji Zhang; Fei Huang; Yang Liu
>
> **摘要:** Video temporal understanding is crucial for multimodal large language models (MLLMs) to reason over events in videos. Despite recent advances in general video understanding, current MLLMs still struggle with fine-grained temporal reasoning. While reinforcement learning (RL) has been explored to address this issue recently, existing RL approaches remain limited in effectiveness. In this work, we propose MUSEG, a novel RL-based method that enhances temporal understanding by introducing timestamp-aware multi-segment grounding. MUSEG enables MLLMs to align queries with multiple relevant video segments, promoting more comprehensive temporal reasoning. To facilitate effective learning, we design a customized RL training recipe with phased rewards that progressively guides the model toward temporally grounded reasoning. Extensive experiments on temporal grounding and time-sensitive video QA tasks demonstrate that MUSEG significantly outperforms existing methods and generalizes well across diverse temporal understanding scenarios. View our project at https://github.com/THUNLP-MT/MUSEG.
>
---
#### [new 170] Leveraging GANs for citation intent classification and its impact on citation network analysis
- **分类: cs.DL; cs.CL; cs.SI**

- **简介: 该论文提出基于GAN的引文意图分类方法，解决传统模型参数多且对引文网络分析影响不明的问题。通过对比实验验证方法有效性，并分析过滤不同引文类型对四种中心性指标的影响，发现betweenness最敏感。**

- **链接: [http://arxiv.org/pdf/2505.21162v1](http://arxiv.org/pdf/2505.21162v1)**

> **作者:** Davi A. Bezerra; Filipi N. Silva; Diego R. Amancio
>
> **摘要:** Citations play a fundamental role in the scientific ecosystem, serving as a foundation for tracking the flow of knowledge, acknowledging prior work, and assessing scholarly influence. In scientometrics, they are also central to the construction of quantitative indicators. Not all citations, however, serve the same function: some provide background, others introduce methods, or compare results. Therefore, understanding citation intent allows for a more nuanced interpretation of scientific impact. In this paper, we adopted a GAN-based method to classify citation intents. Our results revealed that the proposed method achieves competitive classification performance, closely matching state-of-the-art results with substantially fewer parameters. This demonstrates the effectiveness and efficiency of leveraging GAN architectures combined with contextual embeddings in intent classification task. We also investigated whether filtering citation intents affects the centrality of papers in citation networks. Analyzing the network constructed from the unArXiv dataset, we found that paper rankings can be significantly influenced by citation intent. All four centrality metrics examined- degree, PageRank, closeness, and betweenness - were sensitive to the filtering of citation types. The betweenness centrality displayed the greatest sensitivity, showing substantial changes in ranking when specific citation intents were removed.
>
---
#### [new 171] Comparisons between a Large Language Model-based Real-Time Compound Diagnostic Medical AI Interface and Physicians for Common Internal Medicine Cases using Simulated Patients
- **分类: cs.AI; cs.CL**

- **简介: 该论文任务为开发基于LLM的实时复合诊断AI接口，对比其与医生在常见内科病例中的诊断效能。旨在解决初级诊疗中效率与成本问题。通过非随机临床试验，使用模拟患者和USMLE案例，评估首次诊断准确率、时间及成本，结果显示AI在准确率（80%）、耗时（缩短44.6%）和成本（降98.1%）上优于医生，但患者满意度略低，提示AI可辅助常见病例诊疗。**

- **链接: [http://arxiv.org/pdf/2505.20609v1](http://arxiv.org/pdf/2505.20609v1)**

> **作者:** Hyungjun Park; Chang-Yun Woo; Seungjo Lim; Seunghwan Lim; Keunho Kwak; Ju Young Jeong; Chong Hyun Suh
>
> **摘要:** Objective To develop an LLM based realtime compound diagnostic medical AI interface and performed a clinical trial comparing this interface and physicians for common internal medicine cases based on the United States Medical License Exam (USMLE) Step 2 Clinical Skill (CS) style exams. Methods A nonrandomized clinical trial was conducted on August 20, 2024. We recruited one general physician, two internal medicine residents (2nd and 3rd year), and five simulated patients. The clinical vignettes were adapted from the USMLE Step 2 CS style exams. We developed 10 representative internal medicine cases based on actual patients and included information available on initial diagnostic evaluation. Primary outcome was the accuracy of the first differential diagnosis. Repeatability was evaluated based on the proportion of agreement. Results The accuracy of the physicians' first differential diagnosis ranged from 50% to 70%, whereas the realtime compound diagnostic medical AI interface achieved an accuracy of 80%. The proportion of agreement for the first differential diagnosis was 0.7. The accuracy of the first and second differential diagnoses ranged from 70% to 90% for physicians, whereas the AI interface achieved an accuracy rate of 100%. The average time for the AI interface (557 sec) was 44.6% shorter than that of the physicians (1006 sec). The AI interface ($0.08) also reduced costs by 98.1% compared to the physicians' average ($4.2). Patient satisfaction scores ranged from 4.2 to 4.3 for care by physicians and were 3.9 for the AI interface Conclusion An LLM based realtime compound diagnostic medical AI interface demonstrated diagnostic accuracy and patient satisfaction comparable to those of a physician, while requiring less time and lower costs. These findings suggest that AI interfaces may have the potential to assist primary care consultations for common internal medicine cases.
>
---
#### [new 172] SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SV-TrustEval-C基准，评估LLMs在C代码漏洞分析中的结构（复杂流中元素关系识别）和语义推理（扰动下逻辑一致性）能力，揭示现有模型依赖模式匹配而非深度推理，指出现有不足并公开数据集。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20630v1](http://arxiv.org/pdf/2505.20630v1)**

> **作者:** Yansong Li; Paula Branco; Alexander M. Hoole; Manish Marwah; Hari Manassery Koduvely; Guy-Vincent Jourdan; Stephan Jou
>
> **摘要:** As Large Language Models (LLMs) evolve in understanding and generating code, accurately evaluating their reliability in analyzing source code vulnerabilities becomes increasingly vital. While studies have examined LLM capabilities in tasks like vulnerability detection and repair, they often overlook the importance of both structure and semantic reasoning crucial for trustworthy vulnerability analysis. To address this gap, we introduce SV-TrustEval-C, a benchmark designed to evaluate LLMs' abilities for vulnerability analysis of code written in the C programming language through two key dimensions: structure reasoning - assessing how models identify relationships between code elements under varying data and control flow complexities; and semantic reasoning - examining their logical consistency in scenarios where code is structurally and semantically perturbed. Our results show that current LLMs are far from satisfactory in understanding complex code relationships and that their vulnerability analyses rely more on pattern matching than on robust logical reasoning. These findings underscore the effectiveness of the SV-TrustEval-C benchmark and highlight critical areas for enhancing the reasoning capabilities and trustworthiness of LLMs in real-world vulnerability analysis tasks. Our initial benchmark dataset is publicly available.
>
---
#### [new 173] Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs)
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI偏见分析任务，研究系统提示位置对LLM行为的影响。通过对比六种商用LLM在系统与用户提示中处理50类人口统计信息的差异，揭示信息层级导致的代表性、分配偏差等风险，呼吁将系统提示纳入AI审计以缓解潜在危害。**

- **链接: [http://arxiv.org/pdf/2505.21091v1](http://arxiv.org/pdf/2505.21091v1)**

> **作者:** Anna Neumann; Elisabeth Kirsten; Muhammad Bilal Zafar; Jatinder Singh
>
> **备注:** Forthcoming in Proceedings of ACM FAccT 2025
>
> **摘要:** System prompts in Large Language Models (LLMs) are predefined directives that guide model behaviour, taking precedence over user inputs in text processing and generation. LLM deployers increasingly use them to ensure consistent responses across contexts. While model providers set a foundation of system prompts, deployers and third-party developers can append additional prompts without visibility into others' additions, while this layered implementation remains entirely hidden from end-users. As system prompts become more complex, they can directly or indirectly introduce unaccounted for side effects. This lack of transparency raises fundamental questions about how the position of information in different directives shapes model outputs. As such, this work examines how the placement of information affects model behaviour. To this end, we compare how models process demographic information in system versus user prompts across six commercially available LLMs and 50 demographic groups. Our analysis reveals significant biases, manifesting in differences in user representation and decision-making scenarios. Since these variations stem from inaccessible and opaque system-level configurations, they risk representational, allocative and potential other biases and downstream harms beyond the user's ability to detect or correct. Our findings draw attention to these critical issues, which have the potential to perpetuate harms if left unexamined. Further, we argue that system prompt analysis must be incorporated into AI auditing processes, particularly as customisable system prompts become increasingly prevalent in commercial AI deployments.
>
---
#### [new 174] BrainStratify: Coarse-to-Fine Disentanglement of Intracranial Neural Dynamics
- **分类: eess.SP; cs.CL; q-bio.NC**

- **简介: 该论文属于脑机接口语音解码任务，针对颅内神经信号稀疏分布及与无关信号纠缠问题，提出分层框架BrainStratify：先通过空间-时间建模识别功能组，再用DPQ解耦目标神经动态。实验显示其显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20480v1](http://arxiv.org/pdf/2505.20480v1)**

> **作者:** Hui Zheng; Hai-Teng Wang; Yi-Tao Jing; Pei-Yang Lin; Han-Qing Zhao; Wei Chen; Peng-Hu Wei; Yong-Zhi Shan; Guo-Guang Zhao; Yun-Zhe Liu
>
> **摘要:** Decoding speech directly from neural activity is a central goal in brain-computer interface (BCI) research. In recent years, exciting advances have been made through the growing use of intracranial field potential recordings, such as stereo-ElectroEncephaloGraphy (sEEG) and ElectroCorticoGraphy (ECoG). These neural signals capture rich population-level activity but present key challenges: (i) task-relevant neural signals are sparsely distributed across sEEG electrodes, and (ii) they are often entangled with task-irrelevant neural signals in both sEEG and ECoG. To address these challenges, we introduce a unified Coarse-to-Fine neural disentanglement framework, BrainStratify, which includes (i) identifying functional groups through spatial-context-guided temporal-spatial modeling, and (ii) disentangling distinct neural dynamics within the target functional group using Decoupled Product Quantization (DPQ). We evaluate BrainStratify on two open-source sEEG datasets and one (epidural) ECoG dataset, spanning tasks like vocal production and speech perception. Extensive experiments show that BrainStratify, as a unified framework for decoding speech from intracranial neural signals, significantly outperforms previous decoding methods. Overall, by combining data-driven stratification with neuroscience-inspired modularity, BrainStratify offers a robust and interpretable solution for speech decoding from intracranial recordings.
>
---
#### [new 175] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-rebench，通过自动化管道从GitHub提取真实世界软件工程任务，构建含21,000+交互式Python任务的数据集，解决训练数据稀缺及评估污染问题，创建无污染基准以准确评测LLM性能，揭示现有模型表现可能因数据污染被高估。**

- **链接: [http://arxiv.org/pdf/2505.20411v1](http://arxiv.org/pdf/2505.20411v1)**

> **作者:** Ibragim Badertdinov; Alexander Golubev; Maksim Nekrashevich; Anton Shevtsov; Simon Karasik; Andrei Andriushchenko; Maria Trofimova; Daria Litvintseva; Boris Yangel
>
> **备注:** Dataset: https://huggingface.co/datasets/nebius/SWE-rebench, SWE-rebench leaderboard https://swe-rebench.com
>
> **摘要:** LLM-based agents have shown promising capabilities in a growing range of software engineering (SWE) tasks. However, advancing this field faces two critical challenges. First, high-quality training data is scarce, especially data that reflects real-world SWE scenarios, where agents must interact with development environments, execute code and adapt behavior based on the outcomes of their actions. Existing datasets are either limited to one-shot code generation or comprise small, manually curated collections of interactive tasks, lacking both scale and diversity. Second, the lack of fresh interactive SWE tasks affects evaluation of rapidly improving models, as static benchmarks quickly become outdated due to contamination issues. To address these limitations, we introduce a novel, automated, and scalable pipeline to continuously extract real-world interactive SWE tasks from diverse GitHub repositories. Using this pipeline, we construct SWE-rebench, a public dataset comprising over 21,000 interactive Python-based SWE tasks, suitable for reinforcement learning of SWE agents at scale. Additionally, we use continuous supply of fresh tasks collected using SWE-rebench methodology to build a contamination-free benchmark for agentic software engineering. We compare results of various LLMs on this benchmark to results on SWE-bench Verified and show that performance of some language models might be inflated due to contamination issues.
>
---
#### [new 176] ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ViewSpatial-Bench基准，评估视觉语言模型（VLMs）在多视角空间定位任务中的表现。针对现有模型擅长相机视角但无法有效处理他人称视角的问题，设计五类任务及3D标注系统，揭示模型性能差距，并通过微调提升46.24%，推动具身AI的空间智能研究。**

- **链接: [http://arxiv.org/pdf/2505.21500v1](http://arxiv.org/pdf/2505.21500v1)**

> **作者:** Dingming Li; Hongxing Li; Zixuan Wang; Yuchen Yan; Hang Zhang; Siqi Chen; Guiyang Hou; Shengpei Jiang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Yueting Zhuang
>
> **备注:** Project: https://zju-real.github.io/ViewSpatial-Page/
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and reasoning about visual content, but significant challenges persist in tasks requiring cross-viewpoint understanding and spatial reasoning. We identify a critical limitation: current VLMs excel primarily at egocentric spatial reasoning (from the camera's perspective) but fail to generalize to allocentric viewpoints when required to adopt another entity's spatial frame of reference. We introduce ViewSpatial-Bench, the first comprehensive benchmark designed specifically for multi-viewpoint spatial localization recognition evaluation across five distinct task types, supported by an automated 3D annotation pipeline that generates precise directional labels. Comprehensive evaluation of diverse VLMs on ViewSpatial-Bench reveals a significant performance disparity: models demonstrate reasonable performance on camera-perspective tasks but exhibit reduced accuracy when reasoning from a human viewpoint. By fine-tuning VLMs on our multi-perspective spatial dataset, we achieve an overall performance improvement of 46.24% across tasks, highlighting the efficacy of our approach. Our work establishes a crucial benchmark for spatial intelligence in embodied AI systems and provides empirical evidence that modeling 3D spatial relationships enhances VLMs' corresponding spatial comprehension capabilities.
>
---
#### [new 177] Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting
- **分类: cs.AI; cs.CL; I.2.7; I.2.1; H.5.2**

- **简介: 该论文提出Project Riley——基于五种情感代理的多模态多Agent协作架构，通过模拟情绪推理与投票机制优化对话回复。旨在解决AI对话中情感表达与多模态整合不足的问题，构建可生成情感适配、清晰实用回复的系统，并衍生出紧急场景专用模型Armando。经用户测试验证，系统在情感一致性与表达清晰度表现优异。**

- **链接: [http://arxiv.org/pdf/2505.20521v1](http://arxiv.org/pdf/2505.20521v1)**

> **作者:** Ana Rita Ortigoso; Gabriel Vieira; Daniel Fuentes; Luis Frazão; Nuno Costa; António Pereira
>
> **备注:** 28 pages, 5 figures. Submitted for review to Information Fusion
>
> **摘要:** This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
>
---
#### [new 178] The Impact of a Chatbot's Ephemerality-Framing on Self-Disclosure Perceptions
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于实验研究，探讨聊天机器人关系描述（ephemerality-framing）对用户自我披露的影响。通过对比"熟悉"（记忆互动）和"陌生"（每次新实体）机器人，在混合设计中测试情感与事实披露任务，发现披露类型和顺序影响用户舒适度与享受度，定性分析揭示陌生框架提供匿名感，而熟悉框架需通过事实披露建立信任以避免侵入感。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20464v1](http://arxiv.org/pdf/2505.20464v1)**

> **作者:** Samuel Rhys Cox; Rune Møberg Jacobsen; Niels van Berkel
>
> **备注:** In ACM Conversational User Interfaces (CUI '25), July 8-10, 2025; 18 pages; 6 Figures; 6 Tables
>
> **摘要:** Self-disclosure, the sharing of one's thoughts and feelings, is affected by the perceived relationship between individuals. While chatbots are increasingly used for self-disclosure, the impact of a chatbot's framing on users' self-disclosure remains under-explored. We investigated how a chatbot's description of its relationship with users, particularly in terms of ephemerality, affects self-disclosure. Specifically, we compared a Familiar chatbot, presenting itself as a companion remembering past interactions, with a Stranger chatbot, presenting itself as a new, unacquainted entity in each conversation. In a mixed factorial design, participants engaged with either the Familiar or Stranger chatbot in two sessions across two days, with one conversation focusing on Emotional- and another Factual-disclosure. When Emotional-disclosure was sought in the first chatting session, Stranger-condition participants felt more comfortable self-disclosing. However, when Factual-disclosure was sought first, these differences were replaced by more enjoyment among Familiar-condition participants. Qualitative findings showed Stranger afforded anonymity and reduced judgement, whereas Familiar sometimes felt intrusive unless rapport was built via low-risk Factual-disclosure.
>
---
#### [new 179] Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航（VLN）任务，旨在解决现有方法依赖视觉合成导致的高计算成本和冗余问题。提出Adaptive Text Dreamer（ATD），基于LLM构建左右脑架构：左脑逻辑整合指令，右脑预测未来场景语义，通过轻量微调和交叉交互机制结合导航模型，实现高效精准导航，在R2R数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.20897v1](http://arxiv.org/pdf/2505.20897v1)**

> **作者:** Pingrui Zhang; Yifei Su; Pengyuan Wu; Dong An; Li Zhang; Zhigang Wang; Dong Wang; Yan Ding; Bin Zhao; Xuelong Li
>
> **摘要:** Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{https://github.com/zhangpingrui/Adaptive-Text-Dreamer}{here}.
>
---
## 更新

#### [replaced 001] Predicting Through Generation: Why Generation Is Better for Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17817v2](http://arxiv.org/pdf/2502.17817v2)**

> **作者:** Md Kowsher; Nusrat Jahan Prottasha; Prakash Bhat; Chun-Nam Yu; Mojtaba Soltanalian; Ivan Garibay; Ozlem Garibay; Chen Chen; Niloofar Yousefi
>
> **备注:** ACL Accepted paper
>
> **摘要:** This paper argues that generating output tokens is more effective than using pooled representations for prediction tasks because token-level generation retains more mutual information. Since LLMs are trained on massive text corpora using next-token prediction, generation aligns naturally with their learned behavior. Using the Data Processing Inequality (DPI), we provide both theoretical and empirical evidence supporting this claim. However, autoregressive models face two key challenges when used for prediction: (1) exposure bias, where the model sees ground truth tokens during training but relies on its own predictions during inference, leading to errors, and (2) format mismatch, where discrete tokens do not always align with the tasks required output structure. To address these challenges, we introduce PredGen(Predicting Through Generating), an end to end framework that (i) uses scheduled sampling to reduce exposure bias, and (ii) introduces a task adapter to convert the generated tokens into structured outputs. Additionally, we introduce Writer-Director Alignment Loss (WDAL), which ensures consistency between token generation and final task predictions, improving both text coherence and numerical accuracy. We evaluate PredGen on multiple classification and regression benchmarks. Our results show that PredGen consistently outperforms standard baselines, demonstrating its effectiveness in structured prediction tasks.
>
---
#### [replaced 002] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2501.18922v2](http://arxiv.org/pdf/2501.18922v2)**

> **作者:** Haoran Luo; Haihong E; Yikai Guo; Qika Lin; Xiaobao Wu; Xinyu Mu; Wenhao Liu; Meina Song; Yifan Zhu; Luu Anh Tuan
>
> **备注:** Accepted by ICML 2025 main conference
>
> **摘要:** Knowledge Base Question Answering (KBQA) aims to answer natural language questions with a large-scale structured knowledge base (KB). Despite advancements with large language models (LLMs), KBQA still faces challenges in weak KB awareness, imbalance between effectiveness and efficiency, and high reliance on annotated data. To address these challenges, we propose KBQA-o1, a novel agentic KBQA method with Monte Carlo Tree Search (MCTS). It introduces a ReAct-based agent process for stepwise logical form generation with KB environment exploration. Moreover, it employs MCTS, a heuristic search method driven by policy and reward models, to balance agentic exploration's performance and search space. With heuristic exploration, KBQA-o1 generates high-quality annotations for further improvement by incremental fine-tuning. Experimental results show that KBQA-o1 outperforms previous low-resource KBQA methods with limited annotated data, boosting Llama-3.1-8B model's GrailQA F1 performance to 78.5% compared to 48.5% of the previous sota method with GPT-3.5-turbo. Our code is publicly available.
>
---
#### [replaced 003] More is not always better? Enhancing Many-Shot In-Context Learning with Differentiated and Reweighting Objectives
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.04070v3](http://arxiv.org/pdf/2501.04070v3)**

> **作者:** Xiaoqing Zhang; Ang Lv; Yuhan Liu; Flood Sung; Wei Liu; Jian Luan; Shuo Shang; Xiuying Chen; Rui Yan
>
> **备注:** 14 pages, 8 figures, 11 tables
>
> **摘要:** Large language models (LLMs) excel at few-shot in-context learning (ICL) without requiring parameter updates. However, as ICL demonstrations increase from a few to many, performance tends to plateau and eventually decline. We identify two primary causes for this trend: the suboptimal negative log-likelihood (NLL) optimization objective and the incremental data noise. To address these issues, we introduce \textit{DrICL}, a novel optimization method that enhances model performance through \textit{Differentiated} and \textit{Reweighting} objectives. Globally, DrICL utilizes differentiated learning to optimize the NLL objective, ensuring that many-shot performance surpasses zero-shot levels. Locally, it dynamically adjusts the weighting of many-shot demonstrations by leveraging cumulative advantages inspired by reinforcement learning, thereby mitigating the impact of noisy data. Recognizing the lack of multi-task datasets with diverse many-shot distributions, we develop the \textit{Many-Shot ICL Benchmark} (ICL-50)-a large-scale benchmark of 50 tasks that cover shot numbers from 1 to 350 within sequences of up to 8,000 tokens-for both fine-tuning and evaluation purposes. Experimental results demonstrate that LLMs enhanced with DrICL achieve significant improvements in many-shot setups across various tasks, including both in-domain and out-of-domain scenarios. We release the code and dataset hoping to facilitate further research in many-shot ICL\footnote{https://github.com/xiaoqzhwhu/DrICL}.
>
---
#### [replaced 004] ComplexFormer: Disruptively Advancing Transformer Inference Ability via Head-Specific Complex Vector Attention
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10222v2](http://arxiv.org/pdf/2505.10222v2)**

> **作者:** Jintian Shao; Hongyi Huang; Jiayi Wu; Beiwen Zhang; ZhiYu Wu; You Shan; MingKai Zheng
>
> **备注:** We are withdrawing this submission as the underlying experiment is currently incomplete. We require additional time to gather more data and supplement the existing findings to ensure a comprehensive and robust presentation. We intend to resubmit once these additions are finalized
>
> **摘要:** Transformer models rely on self-attention to capture token dependencies but face challenges in effectively integrating positional information while allowing multi-head attention (MHA) flexibility. Prior methods often model semantic and positional differences disparately or apply uniform positional adjustments across heads, potentially limiting representational capacity. This paper introduces ComplexFormer, featuring Complex Multi-Head Attention-CMHA. CMHA empowers each head to independently model semantic and positional differences unified within the complex plane, representing interactions as rotations and scaling. ComplexFormer incorporates two key improvements: (1) a per-head Euler transformation, converting real-valued query/key projections into polar-form complex vectors for head-specific complex subspace operation; and (2) a per-head adaptive differential rotation mechanism, exp[i(Adapt(ASmn,i) + Delta(Pmn),i)], allowing each head to learn distinct strategies for integrating semantic angle differences (ASmn,i) with relative positional encodings (Delta(Pmn),i). Extensive experiments on language modeling, text generation, code generation, and mathematical reasoning show ComplexFormer achieves superior performance, significantly lower generation perplexity , and improved long-context coherence compared to strong baselines like RoPE-Transformers. ComplexFormer demonstrates strong parameter efficiency, offering a more expressive, adaptable attention mechanism.
>
---
#### [replaced 005] WizardLM: Empowering large pre-trained language models to follow complex instructions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2304.12244v3](http://arxiv.org/pdf/2304.12244v3)**

> **作者:** Can Xu; Qingfeng Sun; Kai Zheng; Xiubo Geng; Pu Zhao; Jiazhan Feng; Chongyang Tao; Qingwei Lin; Daxin Jiang
>
> **备注:** large language model, instruction fine-tune
>
> **摘要:** Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90\% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs. Our code and data are public at https://github.com/nlpxucan/WizardLM
>
---
#### [replaced 006] Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.05518v2](http://arxiv.org/pdf/2403.05518v2)**

> **作者:** James Chua; Edward Rees; Hunar Batra; Samuel R. Bowman; Julian Michael; Ethan Perez; Miles Turpin
>
> **摘要:** Chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning. But CoT can also systematically misrepresent the factors influencing models' behavior -- for example, rationalizing answers in line with a user's opinion. We first create a new dataset of 9 different biases that affect GPT-3.5-Turbo and Llama-8b models. These consist of spurious-few-shot patterns, post hoc rationalization, and sycophantic settings. Models switch to the answer implied by the bias, without mentioning the effect of the bias in the CoT. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86\% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37\%. As BCT generalizes to held-out biases and does not require gold labels, this method may hold promise for reducing biased reasoning from as-of-yet unknown biases and on tasks where ground truth reasoning is unavailable.
>
---
#### [replaced 007] Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10554v2](http://arxiv.org/pdf/2505.10554v2)**

> **作者:** Zhiyuan Hu; Yibo Wang; Hanze Dong; Yuhui Xu; Amrita Saha; Caiming Xiong; Bryan Hooi; Junnan Li
>
> **备注:** In Progress
>
> **摘要:** Large reasoning models (LRMs) already possess a latent capacity for long chain-of-thought reasoning. Prior work has shown that outcome-based reinforcement learning (RL) can incidentally elicit advanced reasoning behaviors such as self-correction, backtracking, and verification phenomena often referred to as the model's "aha moment". However, the timing and consistency of these emergent behaviors remain unpredictable and uncontrollable, limiting the scalability and reliability of LRMs' reasoning capabilities. To address these limitations, we move beyond reliance on prompts and coincidental "aha moments". Instead, we explicitly align models with three meta-abilities: deduction, induction, and abduction, using automatically generated, self-verifiable tasks. Our three stage-pipeline individual alignment, parameter-space merging, and domain-specific reinforcement learning, boosting performance by over 10\% relative to instruction-tuned baselines. Furthermore, domain-specific RL from the aligned checkpoint yields an additional gain in performance ceiling for both 7B and 32B models across math, coding, and science benchmarks, demonstrating that explicit meta-ability alignment offers a scalable and dependable foundation for reasoning. Code is available at: https://github.com/zhiyuanhubj/Meta-Ability-Alignment
>
---
#### [replaced 008] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20968v2](http://arxiv.org/pdf/2502.20968v2)**

> **作者:** Weixiang Zhao; Yulin Hu; Yang Deng; Jiahe Guo; Xingyu Sui; Xinyang Han; An Zhang; Yanyan Zhao; Bing Qin; Tat-Seng Chua; Ting Liu
>
> **备注:** To appear at ACL 2025 (Main)
>
> **摘要:** Role-playing enables large language models (LLMs) to engage users in immersive and personalized interactions, but it also introduces significant safety risks. Existing role-play fine-tuning techniques improve role adaptability but may degrade safety performance, particularly for villainous characters. In this work, we conduct the first comprehensive assessment of role-play fine-tuning risks by training 95 role-specific LLMs using RoleBench. Our experiments reveal that role-play fine-tuning leads to a noticeable decline in safety performance, with safety risks varying based on character traits. To tackle this challenge, we propose Safety-Aware Role-Play Fine-Tuning (SaRFT), a novel method designed to balance role-playing capabilities and safety. Extensive experiments on LLaMA-3-8B-Instruct, Gemma-2-9B-it, and Qwen2.5-7B-Instruct demonstrate that SaRFT consistently outperforms state-of-the-art baselines under both LoRA and full-parameter fine-tuning settings. Our findings highlight the necessity of role-adaptive safety measures and provide insights into mitigating role-specific safety risks in role-playing LLMs.
>
---
#### [replaced 009] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.01834v2](http://arxiv.org/pdf/2411.01834v2)**

> **作者:** Guan-Ting Lin; Prashanth Gurunath Shivakumar; Aditya Gourav; Yile Gu; Ankur Gandhe; Hung-yi Lee; Ivan Bulyko
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** While textless Spoken Language Models (SLMs) have shown potential in end-to-end speech-to-speech modeling, they still lag behind text-based Large Language Models (LLMs) in terms of semantic coherence and relevance. This work introduces the Align-SLM framework, which leverages preference optimization inspired by Reinforcement Learning with AI Feedback (RLAIF) to enhance the semantic understanding of SLMs. Our approach generates multiple speech continuations from a given prompt and uses semantic metrics to create preference data for Direct Preference Optimization (DPO). We evaluate the framework using ZeroSpeech 2021 benchmarks for lexical and syntactic modeling, the spoken version of the StoryCloze dataset for semantic coherence, and other speech generation metrics, including the GPT4-o score and human evaluation. Experimental results show that our method achieves state-of-the-art performance for SLMs on most benchmarks, highlighting the importance of preference optimization to improve the semantics of SLMs.
>
---
#### [replaced 010] AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.19528v2](http://arxiv.org/pdf/2505.19528v2)**

> **作者:** Yejin Lee; Joonghyuk Hahn; Hyeseon Ahn; Yo-Sub Han
>
> **备注:** 13 pages, 4 figures, Under Review
>
> **摘要:** Implicit hate speech detection is challenging due to its subtlety and reliance on contextual interpretation rather than explicit offensive words. Current approaches rely on contrastive learning, which are shown to be effective on distinguishing hate and non-hate sentences. Humans, however, detect implicit hate speech by first identifying specific targets within the text and subsequently interpreting how these target relate to their surrounding context. Motivated by this reasoning process, we propose AmpleHate, a novel approach designed to mirror human inference for implicit hate detection. AmpleHate identifies explicit target using a pretrained Named Entity Recognition model and capture implicit target information via [CLS] tokens. It computes attention-based relationships between explicit, implicit targets and sentence context and then, directly injects these relational vectors into the final sentence representation. This amplifies the critical signals of target-context relations for determining implicit hate. Experiments demonstrate that AmpleHate achieves state-of-the-art performance, outperforming contrastive learning baselines by an average of 82.14% and achieve faster convergence. Qualitative analyses further reveal that attention patterns produced by AmpleHate closely align with human judgement, underscoring its interpretability and robustness.
>
---
#### [replaced 011] A Survey of LLM $\times$ DATA
- **分类: cs.DB; cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18458v2](http://arxiv.org/pdf/2505.18458v2)**

> **作者:** Xuanhe Zhou; Junxuan He; Wei Zhou; Haodong Chen; Zirui Tang; Haoyu Zhao; Xin Tong; Guoliang Li; Youmin Chen; Jun Zhou; Zhaojun Sun; Binyuan Hui; Shuo Wang; Conghui He; Zhiyuan Liu; Jingren Zhou; Fan Wu
>
> **备注:** Please refer to the paper list at: https://github.com/weAIDB/awesome-data-llm
>
> **摘要:** The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration.
>
---
#### [replaced 012] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2406.11546v2](http://arxiv.org/pdf/2406.11546v2)**

> **作者:** Yifan Yang; Zheshu Song; Jianheng Zhuo; Mingyu Cui; Jinpeng Li; Bo Yang; Yexing Du; Ziyang Ma; Xunying Liu; Ziyuan Wang; Ke Li; Shuai Fan; Kai Yu; Wei-Qiang Zhang; Guoguo Chen; Xie Chen
>
> **备注:** Accepted in ACL 2025 (Main)
>
> **摘要:** The evolution of speech technology has been spurred by the rapid increase in dataset sizes. Traditional speech models generally depend on a large amount of labeled training data, which is scarce for low-resource languages. This paper presents GigaSpeech 2, a large-scale, multi-domain, multilingual speech recognition corpus. It is designed for low-resource languages and does not rely on paired speech and text data. GigaSpeech 2 comprises about 30,000 hours of automatically transcribed speech, including Thai, Indonesian, and Vietnamese, gathered from unlabeled YouTube videos. We also introduce an automated pipeline for data crawling, transcription, and label refinement. Specifically, this pipeline involves Whisper for initial transcription, MMS for forced alignment, and multi-dimensional filtering for data quality assurance. A modified Noisy Student Training is developed to further refine flawed pseudo labels iteratively, thereby enhancing model performance. Experimental results on our manually transcribed evaluation set and two public test sets from Common Voice and FLEURS confirm our corpus's high quality and broad applicability. Notably, ASR models trained on GigaSpeech 2 can reduce the word error rate for Thai, Indonesian, and Vietnamese on our challenging and realistic YouTube test set by 25% to 40% compared to Whisper large-v3, with merely 10% model parameters. Furthermore, our ASR models trained on GigaSpeech 2 yield superior performance compared to commercial services. We hope that our newly introduced corpus and pipeline will open a new avenue for low-resource speech recognition and significantly facilitate research in this area.
>
---
#### [replaced 013] Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training
- **分类: cs.AI; cs.CL; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14681v2](http://arxiv.org/pdf/2505.14681v2)**

> **作者:** Mengru Wang; Xingyu Chen; Yue Wang; Zhiwei He; Jiahao Xu; Tian Liang; Qiuzhi Liu; Yunzhi Yao; Wenxuan Wang; Ruotian Ma; Haitao Mi; Ningyu Zhang; Zhaopeng Tu; Xiaolong Li; Dong Yu
>
> **备注:** Work in progress
>
> **摘要:** Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models.
>
---
#### [replaced 014] MA-LoT: Model-Collaboration Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03205v3](http://arxiv.org/pdf/2503.03205v3)**

> **作者:** Ruida Wang; Rui Pan; Yuxin Li; Jipeng Zhang; Yizhen Jia; Shizhe Diao; Renjie Pi; Junjie Hu; Tong Zhang
>
> **摘要:** Solving mathematical problems using computer-verifiable languages like Lean has significantly impacted the mathematical and computer science communities. State-of-the-art methods utilize a single Large Language Model (LLM) to generate complete proof or perform tree search, but they fail to balance these tasks. We propose **MA-LoT**: *Model-CollAboration Lean-based Long Chain-of-Thought*, a comprehensive framework for Lean4 theorem proving to solve this issue. It separates the cognition tasks of general NL for whole-proof generation and error analysis for proof correction using the model-collaboration method. We achieve this by structured interaction of the LLM and Lean4 verifier in Long CoT. To implement the framework, we propose the novel *LoT-Transfer Learning* training-inference pipeline, which enables the Long CoT thinking capability to LLMs without special data annotation. Extensive experiment shows that our framework achieves a **61.07%** accuracy rate on the Lean4 version of the MiniF2F-Test dataset, largely outperforming DeepSeek-V3 (33.61%), single-model tree search (InternLM-Step-Prover, 50.70%), and whole-proof generation (Godel-Prover, 55.33%) baselines. Furthermore, our findings highlight the potential of combining Long CoT with formal verification for a more insightful generation in a broader perspective.
>
---
#### [replaced 015] Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18973v2](http://arxiv.org/pdf/2505.18973v2)**

> **作者:** Sarang Patil; Ashish Parmanand Pandey; Ioannis Koutis; Mengjia Xu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Selective state-space models have achieved great success in long-sequence modeling. However, their capacity for language representation, especially in complex hierarchical reasoning tasks, remains underexplored. Most large language models rely on flat Euclidean embeddings, limiting their ability to capture latent hierarchies. To address this limitation, we propose Hierarchical Mamba (HiM), integrating efficient Mamba2 with exponential growth and curved nature of hyperbolic geometry to learn hierarchy-aware language embeddings for deeper linguistic understanding. Mamba2-processed sequences are projected to the Poincare ball (via tangent-based mapping) or Lorentzian manifold (via cosine and sine-based mapping) with "learnable" curvature, optimized with a combined hyperbolic loss. Our HiM model facilitates the capture of relational distances across varying hierarchical levels, enabling effective long-range reasoning. This makes it well-suited for tasks like mixed-hop prediction and multi-hop inference in hierarchical classification. We evaluated our HiM with four linguistic and medical datasets for mixed-hop prediction and multi-hop inference tasks. Experimental results demonstrated that: 1) Both HiM models effectively capture hierarchical relationships for four ontological datasets, surpassing Euclidean baselines. 2) HiM-Poincare captures fine-grained semantic distinctions with higher h-norms, while HiM-Lorentz provides more stable, compact, and hierarchy-preserving embeddings favoring robustness over detail.
>
---
#### [replaced 016] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13975v2](http://arxiv.org/pdf/2505.13975v2)**

> **作者:** Yuxuan Jiang; Dawei Li; Frank Ferraro
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains.
>
---
#### [replaced 017] Autoregressive Speech Synthesis without Vector Quantization
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.08551v2](http://arxiv.org/pdf/2407.08551v2)**

> **作者:** Lingwei Meng; Long Zhou; Shujie Liu; Sanyuan Chen; Bing Han; Shujie Hu; Yanqing Liu; Jinyu Li; Sheng Zhao; Xixin Wu; Helen Meng; Furu Wei
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** We present MELLE, a novel continuous-valued token based language modeling approach for text-to-speech synthesis (TTS). MELLE autoregressively generates continuous mel-spectrogram frames directly from text condition, bypassing the need for vector quantization, which is typically designed for audio compression and sacrifices fidelity compared to continuous representations. Specifically, (i) instead of cross-entropy loss, we apply regression loss with a proposed spectrogram flux loss function to model the probability distribution of the continuous-valued tokens; (ii) we have incorporated variational inference into MELLE to facilitate sampling mechanisms, thereby enhancing the output diversity and model robustness. Experiments demonstrate that, compared to the two-stage codec language model VALL-E and its variants, the single-stage MELLE mitigates robustness issues by avoiding the inherent flaws of sampling vector-quantized codes, achieves superior performance across multiple metrics, and, most importantly, offers a more streamlined paradigm. The demos of our work are provided at https://aka.ms/melle.
>
---
#### [replaced 018] An In-depth Evaluation of Large Language Models in Sentence Simplification with Error-based Human Assessment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.04963v3](http://arxiv.org/pdf/2403.04963v3)**

> **作者:** Xuanxin Wu; Yuki Arase
>
> **备注:** Accepted by ACM Transactions on Intelligent Systems and Technology, to appear
>
> **摘要:** Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliability. To address these problems, this study provides in-depth insights into LLMs' performance while ensuring the reliability of the evaluation. We design an error-based human annotation framework to assess the LLMs' simplification capabilities. We select both closed-source and open-source LLMs, including GPT-4, Qwen2.5-72B, and Llama-3.2-3B. We believe that these models offer a representative selection across large, medium, and small sizes of LLMs. Results show that GPT-4 generally generates fewer erroneous simplification outputs compared to the current state-of-the-art. However, LLMs have their limitations, as seen in GPT-4's struggles with lexical paraphrasing. Results show that LLMs generally generate fewer erroneous simplification outputs compared to the previous state-of-the-art. However, LLMs have their limitations, as seen in GPT-4's and Qwen2.5-72B's struggle with lexical paraphrasing. Furthermore, we conduct meta-evaluations on widely used automatic metrics using our human annotations. We find that these metrics lack sufficient sensitivity to assess the overall high-quality simplifications, particularly those generated by high-performance LLMs.
>
---
#### [replaced 019] Can Small Language Models Learn, Unlearn, and Retain Noise Patterns?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.00996v3](http://arxiv.org/pdf/2407.00996v3)**

> **作者:** Nicy Scaria; Silvester John Joseph Kennedy; Deepak Subramani
>
> **摘要:** With the growing need for efficient language models in resource-constrained environments, Small Language Models (SLMs) have emerged as compact and practical alternatives to Large Language Models (LLMs). While studies have explored noise handling in LLMs, little is known about how SLMs handle noise, a critical factor for their reliable real-world deployment. This study investigates the ability of SLMs with parameters between 1 and 3 billion to learn, retain, and subsequently eliminate different types of noise (word flip, character flip, transliteration, irrelevant content, and contradictory information). Four pretrained SLMs (Olmo 1B, Qwen1.5 1.8B, Gemma1.1 2B, and Phi2 2.7B) were instruction-tuned on noise-free data and tested with in-context examples to assess noise learning. Subsequently, noise patterns were introduced in instruction tuning to assess their adaptability. The results revealed differences in how models handle noise, with smaller models like Olmo quickly adapting to noise patterns. Phi2's carefully curated, structured, and high-quality pretraining data enabled resistance to character level, transliteration, and counterfactual noise, while Gemma adapted successfully to transliteration noise through its multilingual pretraining. Subsequent clean data training effectively mitigated noise effects. These findings provide practical strategies for developing robust SLMs for real-world applications.
>
---
#### [replaced 020] Analyzing Biases in Political Dialogue: Tagging U.S. Presidential Debates with an Extended DAMSL Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19515v2](http://arxiv.org/pdf/2505.19515v2)**

> **作者:** Lavanya Prahallad; Radhika Mamidi
>
> **备注:** 8 pages
>
> **摘要:** We present a critical discourse analysis of the 2024 U.S. presidential debates, examining Donald Trump's rhetorical strategies in his interactions with Joe Biden and Kamala Harris. We introduce a novel annotation framework, BEADS (Bias Enriched Annotation for Dialogue Structure), which systematically extends the DAMSL framework to capture bias driven and adversarial discourse features in political communication. BEADS includes a domain and language agnostic set of tags that model ideological framing, emotional appeals, and confrontational tactics. Our methodology compares detailed human annotation with zero shot ChatGPT assisted tagging on verified transcripts from the Trump and Biden (19,219 words) and Trump and Harris (18,123 words) debates. Our analysis shows that Trump consistently dominated in key categories: Challenge and Adversarial Exchanges, Selective Emphasis, Appeal to Fear, Political Bias, and Perceived Dismissiveness. These findings underscore his use of emotionally charged and adversarial rhetoric to control the narrative and influence audience perception. In this work, we establish BEADS as a scalable and reproducible framework for critical discourse analysis across languages, domains, and political contexts.
>
---
#### [replaced 021] WizardCoder: Empowering Code Large Language Models with Evol-Instruct
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2306.08568v2](http://arxiv.org/pdf/2306.08568v2)**

> **作者:** Ziyang Luo; Can Xu; Pu Zhao; Qingfeng Sun; Xiubo Geng; Wenxiang Hu; Chongyang Tao; Jing Ma; Qingwei Lin; Daxin Jiang
>
> **备注:** Large Language model, Code Generation, Code LLMs.This paper has been accepted to ICLR 2024. Please cite the ICLR version
>
> **摘要:** Code Large Language Models (Code LLMs), such as StarCoder, have demonstrated exceptional performance in code-related tasks. However, most existing models are solely pre-trained on extensive raw code data without instruction fine-tuning. In this paper, we introduce WizardCoder, which empowers Code LLMs with complex instruction fine-tuning, by adapting the Evol-Instruct method to the domain of code. Through comprehensive experiments on four prominent code generation benchmarks, namely HumanEval, HumanEval+, MBPP, and DS-1000, we unveil the exceptional capabilities of our model. It surpasses all other open-source Code LLMs by a substantial margin. Moreover, our model even outperforms the largest closed LLMs, Anthropic's Claude and Google's Bard, on HumanEval and HumanEval+. Our code, model weights, and data are public at https://github.com/nlpxucan/WizardLM
>
---
#### [replaced 022] Plan2Align: Predictive Planning Based Test-Time Preference Alignment for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20795v2](http://arxiv.org/pdf/2502.20795v2)**

> **作者:** Kuang-Da Wang; Teng-Ruei Chen; Yu Heng Hung; Guo-Xun Ko; Shuoyang Ding; Yueh-Hua Wu; Yu-Chiang Frank Wang; Chao-Han Huck Yang; Wen-Chih Peng; Ping-Chun Hsieh
>
> **备注:** Preprint. Code will be released at Plan2Align GitHub link: https://github.com/NYCU-RL-Bandits-Lab/Plan2Align
>
> **摘要:** Aligning Large Language Models with Preference Fine-Tuning is often resource-intensive. Test-time alignment techniques that do not modify the underlying models, such as prompting and guided decodings, offer a lightweight alternative. However, existing test-time alignment methods primarily improve short responses and fail to ensure coherence over extended contexts due to the myopic nature of token-level alignment. Moreover, these methods often incur a slowdown during inference. To address these challenges, we propose Plan2Align, a test-time alignment framework that formulates text generation as a predictive planning problem. Plan2Align adapts Model Predictive Control (MPC) to iteratively refine output by rolling out multiple complete responses and optimizing each segment. To more rigorously evaluate the effectiveness and efficiency, we focus on the more challenging task of long-text generation. Experiments on the long-form response subset of the HH-RLHF dataset and the WMT'24 Discourse-Level Literary Translation demonstrate that Plan2Align significantly enhances the performance of base LLMs. Compared to existing training-time and test-time alignment methods on LLaMA-3.1 8B, Plan2Align achieves comparable or superior results, while also delivering improved inference efficiency relative to prior test-time alignment approaches.
>
---
#### [replaced 023] Does Synthetic Data Help Named Entity Recognition for Low-Resource Languages?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16814v2](http://arxiv.org/pdf/2505.16814v2)**

> **作者:** Gaurav Kamath; Sowmya Vajjala
>
> **备注:** pre-print
>
> **摘要:** Named Entity Recognition(NER) for low-resource languages aims to produce robust systems for languages where there is limited labeled training data available, and has been an area of increasing interest within NLP. Data augmentation for increasing the amount of low-resource labeled data is a common practice. In this paper, we explore the role of synthetic data in the context of multilingual, low-resource NER, considering 11 languages from diverse language families. Our results suggest that synthetic data does in fact hold promise for low-resource language NER, though we see significant variation between languages.
>
---
#### [replaced 024] OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.20947v4](http://arxiv.org/pdf/2405.20947v4)**

> **作者:** Justin Cui; Wei-Lin Chiang; Ion Stoica; Cho-Jui Hsieh
>
> **备注:** Accepted to ICML 2025, we thank everyone for their valuable suggestions and feedback!
>
> **摘要:** Large Language Models (LLMs) require careful safety alignment to prevent malicious outputs. While significant research focuses on mitigating harmful content generation, the enhanced safety often come with the side effect of over-refusal, where LLMs may reject innocuous prompts and become less helpful. Although the issue of over-refusal has been empirically observed, a systematic measurement is challenging due to the difficulty of crafting prompts that can elicit the over-refusal behaviors of LLMs. This study proposes a novel method for automatically generating large-scale over-refusal datasets. Leveraging this technique, we introduce OR-Bench, the first large-scale over-refusal benchmark. OR-Bench comprises 80,000 over-refusal prompts across 10 common rejection categories, a subset of around 1,000 hard prompts that are challenging even for state-of-the-art LLMs, and an additional 600 toxic prompts to prevent indiscriminate responses. We then conduct a comprehensive study to measure the over-refusal of 32 popular LLMs across 8 model families. Our datasets are publicly available at https://huggingface.co/bench-llms and our codebase is open-sourced at https://github.com/justincui03/or-bench. We hope this benchmark can help the community develop better safety aligned models.
>
---
#### [replaced 025] A Graph Perspective to Probe Structural Patterns of Knowledge in Large Language Models
- **分类: cs.CL; cs.LG; cs.SI**

- **链接: [http://arxiv.org/pdf/2505.19286v2](http://arxiv.org/pdf/2505.19286v2)**

> **作者:** Utkarsh Sahu; Zhisheng Qi; Yongjia Lei; Ryan A. Rossi; Franck Dernoncourt; Nesreen K. Ahmed; Mahantesh M Halappanavar; Yao Ma; Yu Wang
>
> **摘要:** Large language models have been extensively studied as neural knowledge bases for their knowledge access, editability, reasoning, and explainability. However, few works focus on the structural patterns of their knowledge. Motivated by this gap, we investigate these structural patterns from a graph perspective. We quantify the knowledge of LLMs at both the triplet and entity levels, and analyze how it relates to graph structural properties such as node degree. Furthermore, we uncover the knowledge homophily, where topologically close entities exhibit similar levels of knowledgeability, which further motivates us to develop graph machine learning models to estimate entity knowledge based on its local neighbors. This model further enables valuable knowledge checking by selecting triplets less known to LLMs. Empirical results show that using selected triplets for fine-tuning leads to superior performance.
>
---
#### [replaced 026] Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19075v2](http://arxiv.org/pdf/2505.19075v2)**

> **作者:** Jaemin Kim; Hangeol Chang; Hyunmin Hwang; Choonghan Kim; Jong Chul Ye
>
> **备注:** 22 pages, typos corrected
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise their generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically requires retraining for each LLM backbone due to architectural dependencies. To address these challenges, here we propose Universal Reasoner (UniR) - a single, lightweight, composable, and plug-and-play reasoning module that can be used with any frozen LLM to endow it with specialized reasoning capabilities. Specifically, UniR decomposes the reward into a standalone reasoning module that is trained independently using predefined rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR can be combined with any frozen LLM at inference time by simply adding its output logits to those of the LLM backbone. This additive structure naturally enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Experimental results on mathematical reasoning and machine translation tasks show that UniR significantly outperforms existing baseline fine-tuning methods using the Llama3.2 model. Furthermore, UniR demonstrates strong weak-to-strong generalization: reasoning modules trained on smaller models effectively guide much larger LLMs. This makes UniR a cost-efficient, adaptable, and robust solution for enhancing reasoning in LLMs without compromising their core capabilities. Code is open-sourced at https://github.com/hangeol/UniR
>
---
#### [replaced 027] Hallucinations are inevitable but can be made statistically negligible. The "innate" inevitability of hallucinations cannot explain practical LLM issues
- **分类: cs.CL; cs.FL; cs.LG; math.ST; stat.ML; stat.TH**

- **链接: [http://arxiv.org/pdf/2502.12187v2](http://arxiv.org/pdf/2502.12187v2)**

> **作者:** Atsushi Suzuki; Yulan He; Feng Tian; Zhongyuan Wang
>
> **摘要:** Hallucinations, a phenomenon where a language model (LM) generates nonfactual content, pose a significant challenge to the practical deployment of LMs. While many empirical methods have been proposed to mitigate hallucinations, recent studies established a computability-theoretic result showing that any LM will inevitably generate hallucinations on an infinite set of inputs, regardless of the quality and quantity of training datasets and the choice of the language model architecture and training and inference algorithms. Although the computability-theoretic result may seem pessimistic, its significance in practical viewpoints has remained unclear. This paper claims that those "innate" inevitability results from computability theory and diagonal argument, in principle, cannot explain practical issues of LLMs. We demonstrate this claim by presenting a positive theoretical result from a probabilistic perspective. Specifically, we prove that hallucinations can be made statistically negligible, provided that the quality and quantity of the training data are sufficient. Interestingly, our positive result coexists with the computability-theoretic result, implying that while hallucinations on an infinite set of inputs cannot be entirely eliminated, their probability can always be reduced by improving algorithms and training data. By evaluating the two seemingly contradictory results through the lens of information theory, we argue that our probability-theoretic positive result better reflects practical considerations than the computability-theoretic negative result.
>
---
#### [replaced 028] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.09403v4](http://arxiv.org/pdf/2410.09403v4)**

> **作者:** Haoyang Su; Renqi Chen; Shixiang Tang; Zhenfei Yin; Xinzhe Zheng; Jinzhe Li; Biqing Qi; Qi Wu; Hui Li; Wanli Ouyang; Philip Torr; Bowen Zhou; Nanqing Dong
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
>
---
#### [replaced 029] Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17266v2](http://arxiv.org/pdf/2505.17266v2)**

> **作者:** Cehao Yang; Xueyuan Lin; Chengjin Xu; Xuhui Jiang; Xiaojun Wu; Honghao Liu; Hui Xiong; Jian Guo
>
> **摘要:** A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost.
>
---
#### [replaced 030] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.14469v2](http://arxiv.org/pdf/2409.14469v2)**

> **作者:** Kaikai An; Shuzheng Si; Helan Hu; Haozhe Zhao; Yuchi Wang; Qingyan Guo; Baobao Chang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Semantic Parsing aims to capture the meaning of a sentence and convert it into a logical, structured form. Previous studies show that semantic parsing enhances the performance of smaller models (e.g., BERT) on downstream tasks. However, it remains unclear whether the improvements extend similarly to LLMs. In this paper, our empirical findings reveal that, unlike smaller models, directly adding semantic parsing results into LLMs reduces their performance. To overcome this, we propose SENSE, a novel prompting approach that embeds semantic hints within the prompt. Experiments show that SENSE consistently improves LLMs' performance across various tasks, highlighting the potential of integrating semantic information to improve LLM capabilities.
>
---
#### [replaced 031] Interlocking-free Selective Rationalization Through Genetic-based Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **链接: [http://arxiv.org/pdf/2412.10312v2](http://arxiv.org/pdf/2412.10312v2)**

> **作者:** Federico Ruggeri; Gaetano Signorelli
>
> **摘要:** A popular end-to-end architecture for selective rationalization is the select-then-predict pipeline, comprising a generator to extract highlights fed to a predictor. Such a cooperative system suffers from suboptimal equilibrium minima due to the dominance of one of the two modules, a phenomenon known as interlocking. While several contributions aimed at addressing interlocking, they only mitigate its effect, often by introducing feature-based heuristics, sampling, and ad-hoc regularizations. We present GenSPP, the first interlocking-free architecture for selective rationalization that does not require any learning overhead, as the above-mentioned. GenSPP avoids interlocking by performing disjoint training of the generator and predictor via genetic global search. Experiments on a synthetic and a real-world benchmark show that our model outperforms several state-of-the-art competitors.
>
---
#### [replaced 032] Transparent and Coherent Procedural Mistake Detection
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11927v2](http://arxiv.org/pdf/2412.11927v2)**

> **作者:** Shane Storks; Itamar Bar-Yossef; Yayuan Li; Zheyuan Zhang; Jason J. Corso; Joyce Chai
>
> **摘要:** Procedural mistake detection (PMD) is a challenging problem of classifying whether a human user (observed through egocentric video) has successfully executed a task (specified by a procedural text). Despite significant recent efforts, machine performance in the wild remains nonviable, and the reasoning processes underlying this performance are opaque. As such, we extend PMD to require generating visual self-dialog rationales to inform decisions. Given the impressive, mature image understanding capabilities observed in recent vision-and-language models (VLMs), we curate a suitable benchmark dataset for PMD based on individual frames. As our reformulation enables unprecedented transparency, we leverage a natural language inference (NLI) model to formulate two automated metrics for the coherence of generated rationales. We establish baselines for this reframed task, showing that while VLMs struggle off-the-shelf, their accuracy, coherence, and efficiency can be improved by incorporating these metrics into common inference and fine-tuning methods- though not without tradeoff. Lastly, our multi-faceted metrics visualize common outcomes, highlighting areas for further improvement.
>
---
#### [replaced 033] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v2](http://arxiv.org/pdf/2505.14874v2)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 034] DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14838v4](http://arxiv.org/pdf/2412.14838v4)**

> **作者:** Xiabin Zhou; Wenbin Wang; Minyan Zeng; Jiaxian Guo; Xuebo Liu; Li Shen; Min Zhang; Liang Ding
>
> **摘要:** Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released.
>
---
#### [replaced 035] Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14641v2](http://arxiv.org/pdf/2410.14641v2)**

> **作者:** Runchu Tian; Yanghao Li; Yuepeng Fu; Siyang Deng; Qinyu Luo; Cheng Qian; Shuo Wang; Xin Cong; Zhong Zhang; Yesai Wu; Yankai Lin; Huadong Wang; Xiaojiang Liu
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Positional bias in large language models (LLMs) hinders their ability to effectively process long inputs. A prominent example is the "lost in the middle" phenomenon, where LLMs struggle to utilize relevant information situated in the middle of the input. While prior research primarily focuses on single pieces of relevant information, real-world applications often involve multiple relevant information pieces. To bridge this gap, we present LongPiBench, a benchmark designed to assess positional bias involving multiple pieces of relevant information. Thorough experiments are conducted with five commercial and six open-source models. These experiments reveal that while most current models are robust against the "lost in the middle" issue, there exist significant biases related to the spacing of relevant information pieces. These findings highlight the importance of evaluating and reducing positional biases to advance LLM's capabilities.
>
---
#### [replaced 036] Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.18693v2](http://arxiv.org/pdf/2410.18693v2)**

> **作者:** Yuyang Ding; Xinyu Shi; Xiaobo Liang; Juntao Li; Zhaopeng Tu; Qiaoming Zhu; Min Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Improving the mathematical reasoning capabilities of Large Language Models (LLMs) is critical for advancing artificial intelligence. However, access to extensive, diverse, and high-quality reasoning datasets remains a significant challenge, particularly for the open-source community. In this paper, we propose ScaleQuest, a novel, scalable, and cost-effective data synthesis method that enables the generation of large-scale mathematical reasoning datasets using lightweight 7B-scale models. ScaleQuest introduces a two-stage question-tuning process comprising Question Fine-Tuning (QFT) and Question Preference Optimization (QPO) to unlock the question generation capabilities of problem-solving models. By generating diverse questions from scratch -- without relying on powerful proprietary models or seed data -- we produce a dataset of 1 million problem-solution pairs. Our experiments demonstrate that models trained on our data outperform existing open-source datasets in both in-domain and out-of-domain evaluations. Furthermore, our approach shows continued performance improvement as the volume of training data increases, highlighting its potential for ongoing data scaling. The extensive improvements observed in code reasoning tasks demonstrate the generalization capabilities of our proposed method. Our work provides the open-source community with a practical solution to enhance the mathematical reasoning abilities of LLMs.
>
---
#### [replaced 037] Exploring the Necessity of Reasoning in LLM-based Agent Scenarios
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11074v2](http://arxiv.org/pdf/2503.11074v2)**

> **作者:** Xueyang Zhou; Guiyao Tie; Guowen Zhang; Weidong Wang; Zhigang Zuo; Di Wu; Duanfeng Chu; Pan Zhou; Neil Zhenqiang Gong; Lichao Sun
>
> **备注:** 71 pages, 11 figures, 8 tables
>
> **摘要:** The rise of Large Reasoning Models (LRMs) signifies a paradigm shift toward advanced computational reasoning. Yet, this progress disrupts traditional agent frameworks, traditionally anchored by execution-oriented Large Language Models (LLMs). To explore this transformation, we propose the LaRMA framework, encompassing nine tasks across Tool Usage, Plan Design, and Problem Solving, assessed with three top LLMs (e.g., Claude3.5-sonnet) and five leading LRMs (e.g., DeepSeek-R1). Our findings address four research questions: LRMs surpass LLMs in reasoning-intensive tasks like Plan Design, leveraging iterative reflection for superior outcomes; LLMs excel in execution-driven tasks such as Tool Usage, prioritizing efficiency; hybrid LLM-LRM configurations, pairing LLMs as actors with LRMs as reflectors, optimize agent performance by blending execution speed with reasoning depth; and LRMs' enhanced reasoning incurs higher computational costs, prolonged processing, and behavioral challenges, including overthinking and fact-ignoring tendencies. This study fosters deeper inquiry into LRMs' balance of deep thinking and overthinking, laying a critical foundation for future agent design advancements.
>
---
#### [replaced 038] Adaptive Deep Reasoning: Triggering Deep Thinking When Needed
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20101v2](http://arxiv.org/pdf/2505.20101v2)**

> **作者:** Yunhao Wang; Yuhao Zhang; Tinghao Yu; Can Xu; Feng Zhang; Fengzong Lian
>
> **摘要:** Large language models (LLMs) have shown impressive capabilities in handling complex tasks through long-chain reasoning. However, the extensive reasoning steps involved can significantly increase computational costs, posing challenges for real-world deployment. Recent efforts have focused on optimizing reasoning efficiency by shortening the Chain-of-Thought (CoT) reasoning processes through various approaches, such as length-aware prompt engineering, supervised fine-tuning on CoT data with variable lengths, and reinforcement learning with length penalties. Although these methods effectively reduce reasoning length, they still necessitate an initial reasoning phase. More recent approaches have attempted to integrate long-chain and short-chain reasoning abilities into a single model, yet they still rely on manual control to toggle between short and long CoT. In this work, we propose a novel approach that autonomously switches between short and long reasoning chains based on problem complexity. Our method begins with supervised fine-tuning of the base model to equip both long-chain and short-chain reasoning abilities. We then employ reinforcement learning to further balance short and long CoT generation while maintaining accuracy through two key strategies: first, integrating reinforcement learning with a long-short adaptive group-wise reward strategy to assess prompt complexity and provide corresponding rewards; second, implementing a logit-based reasoning mode switching loss to optimize the model's initial token choice, thereby guiding the selection of the reasoning type. Evaluations on mathematical datasets demonstrate that our model can dynamically switch between long-chain and short-chain reasoning modes without substantially sacrificing performance. This advancement enhances the practicality of reasoning in large language models for real-world applications.
>
---
#### [replaced 039] Enhance Mobile Agents Thinking Process Via Iterative Preference Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12299v2](http://arxiv.org/pdf/2505.12299v2)**

> **作者:** Kun Huang; Weikai Xu; Yuxuan Liu; Quandong Wang; Pengzhi Gao; Wei Liu; Jian Luan; Bin Wang; Bo An
>
> **备注:** 9 pages, 8 figures, 7 tables
>
> **摘要:** The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios.
>
---
#### [replaced 040] Knowledge Boundary of Large Language Models: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12472v2](http://arxiv.org/pdf/2412.12472v2)**

> **作者:** Moxin Li; Yong Zhao; Wenxuan Zhang; Shuaiyi Li; Wenya Xie; See-Kiong Ng; Tat-Seng Chua; Yang Deng
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** Although large language models (LLMs) store vast amount of knowledge in their parameters, they still have limitations in the memorization and utilization of certain knowledge, leading to undesired behaviors such as generating untruthful and inaccurate responses. This highlights the critical need to understand the knowledge boundary of LLMs, a concept that remains inadequately defined in existing research. In this survey, we propose a comprehensive definition of the LLM knowledge boundary and introduce a formalized taxonomy categorizing knowledge into four distinct types. Using this foundation, we systematically review the field through three key lenses: the motivation for studying LLM knowledge boundaries, methods for identifying these boundaries, and strategies for mitigating the challenges they present. Finally, we discuss open challenges and potential research directions in this area. We aim for this survey to offer the community a comprehensive overview, facilitate access to key issues, and inspire further advancements in LLM knowledge research.
>
---
#### [replaced 041] GeLLMO: Generalizing Large Language Models for Multi-property Molecule Optimization
- **分类: cs.LG; cs.AI; cs.CL; physics.chem-ph; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2502.13398v2](http://arxiv.org/pdf/2502.13398v2)**

> **作者:** Vishal Dey; Xiao Hu; Xia Ning
>
> **备注:** Accepted to ACL Main 2025. Vishal Dey and Xiao Hu contributed equally to this paper
>
> **摘要:** Despite recent advancements, most computational methods for molecule optimization are constrained to single- or double-property optimization tasks and suffer from poor scalability and generalizability to novel optimization tasks. Meanwhile, Large Language Models (LLMs) demonstrate remarkable out-of-domain generalizability to novel tasks. To demonstrate LLMs' potential for molecule optimization, we introduce MuMOInstruct, the first high-quality instruction-tuning dataset specifically focused on complex multi-property molecule optimization tasks. Leveraging MuMOInstruct, we develop GeLLMOs, a series of instruction-tuned LLMs for molecule optimization. Extensive evaluations across 5 in-domain and 5 out-of-domain tasks demonstrate that GeLLMOs consistently outperform state-of-the-art baselines. GeLLMOs also exhibit outstanding zero-shot generalization to unseen tasks, significantly outperforming powerful closed-source LLMs. Such strong generalizability demonstrates the tremendous potential of GeLLMOs as foundational models for molecule optimization, thereby tackling novel optimization tasks without resource-intensive retraining. MuMOInstruct, models, and code are accessible through https://github.com/ninglab/GeLLMO.
>
---
#### [replaced 042] Voting or Consensus? Decision-Making in Multi-Agent Debate
- **分类: cs.MA; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19130v2](http://arxiv.org/pdf/2502.19130v2)**

> **作者:** Lars Benedikt Kaesberg; Jonas Becker; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **摘要:** Much of the success of multi-agent debates depends on carefully choosing the right parameters. The decision-making protocol stands out as it can highly impact final model answers, depending on how decisions are reached. Systematic comparison of decision protocols is difficult because many studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making influences different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time - the decision protocol - to analyze how different methods affect the collaboration between agents and measure differences in knowledge and reasoning tasks. Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks compared to other decision protocols. Increasing the number of agents improves performance, while more discussion rounds before voting reduce it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling.
>
---
#### [replaced 043] Can Community Notes Replace Professional Fact-Checkers?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14132v2](http://arxiv.org/pdf/2502.14132v2)**

> **作者:** Nadav Borenstein; Greta Warren; Desmond Elliott; Isabelle Augenstein
>
> **备注:** Accepted to the main proceedings of ACL 2025
>
> **摘要:** Two commonly employed strategies to combat the rise of misinformation on social media are (i) fact-checking by professional organisations and (ii) community moderation by platform users. Policy changes by Twitter/X and, more recently, Meta, signal a shift away from partnerships with fact-checking organisations and towards an increased reliance on crowdsourced community notes. However, the extent and nature of dependencies between fact-checking and helpful community notes remain unclear. To address these questions, we use language models to annotate a large corpus of Twitter/X community notes with attributes such as topic, cited sources, and whether they refute claims tied to broader misinformation narratives. Our analysis reveals that community notes cite fact-checking sources up to five times more than previously reported. Fact-checking is especially crucial for notes on posts linked to broader narratives, which are twice as likely to reference fact-checking sources compared to other sources. Our results show that successful community moderation relies on professional fact-checking and highlight how citizen and professional fact-checking are deeply intertwined.
>
---
#### [replaced 044] Tradeoffs Between Alignment and Helpfulness in Language Models with Steering Methods
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.16332v5](http://arxiv.org/pdf/2401.16332v5)**

> **作者:** Yotam Wolf; Noam Wies; Dorin Shteyman; Binyamin Rothberg; Yoav Levine; Amnon Shashua
>
> **摘要:** Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.
>
---
#### [replaced 045] When More is Less: Understanding Chain-of-Thought Length in LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07266v3](http://arxiv.org/pdf/2502.07266v3)**

> **作者:** Yuyang Wu; Yifei Wang; Ziyu Ye; Tianqi Du; Stefanie Jegelka; Yisen Wang
>
> **摘要:** Large Language Models (LLMs) employ Chain-of-Thought (CoT) reasoning to deconstruct complex problems. While longer CoTs are often presumed superior, this paper challenges that notion, arguing that longer is not always better. Drawing on combined evidence from real-world observations, controlled experiments, and theoretical analysis, we demonstrate that task accuracy typically follows an inverted U-shaped curve with CoT length, where performance initially improves but eventually decreases as the number of CoT steps increases. With controlled experiments, we further uncover the scaling behaviors of the optimal CoT length: it increases with task difficulty but decreases with model capability, exposing an inherent simplicity bias where more capable models favor shorter, more efficient CoT reasoning. This bias is also evident in Reinforcement Learning (RL) training, where models gravitate towards shorter CoTs as their accuracy improves. To have a deep understanding of these dynamics, we establish a simple theoretical model that formally proves these phenomena, including the optimal length's scaling laws and the emergence of simplicity bias during RL. Guided by this framework, we demonstrate significant practical benefits from training with optimally-lengthed CoTs and employing length-aware filtering at inference. These findings offer both a principled understanding of the "overthinking" phenomenon and multiple practical guidelines for CoT calibration, enabling LLMs to achieve optimal reasoning performance with adaptive CoTs tailored to task complexity and model capability.
>
---
#### [replaced 046] S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10368v3](http://arxiv.org/pdf/2504.10368v3)**

> **作者:** Wenyuan Zhang; Shuaiyi Nie; Xinghua Zhang; Zefeng Zhang; Tingwen Liu
>
> **备注:** 31 pages, 9 figures, 16 tables
>
> **摘要:** We introduce S1-Bench, a novel benchmark designed to evaluate the performance of Large Reasoning Models (LRMs) on simple tasks that favor intuitive system 1 thinking rather than deliberative system 2 reasoning. While LRMs have achieved significant breakthroughs in complex reasoning tasks through explicit chains of thought, their heavy reliance on system 2 thinking may limit their system 1 thinking capabilities. However, there is a lack of an appropriate benchmark for evaluating LRM's system 1 thinking capabilities. To fill this gap, S1-Bench introduces a suite of simple, diverse, and natural questions across multiple domains and languages, specifically designed to assess LRMs' performance on questions more suitable for system 1 . We conduct extensive evaluations across 28 LRMs, revealing their inefficiency, inadequate accuracy, and limited robustness when handling simple questions. Additionally, we observe a gap between their difficulty perception and generation length. Overall, this work paves the way toward dual-system compatibility in the development of LRMs.
>
---
#### [replaced 047] A Lightweight Method to Disrupt Memorized Sequences in LLM
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05159v2](http://arxiv.org/pdf/2502.05159v2)**

> **作者:** Parjanya Prajakta Prashant; Kaustubh Ponkshe; Babak Salimi
>
> **备注:** 26 pages, 3 figures
>
> **摘要:** As language models scale, their performance improves dramatically across a wide range of tasks, but so does their tendency to memorize and regurgitate parts of their training data verbatim. This tradeoff poses serious legal, ethical, and safety concerns, especially in real-world deployments. Existing mitigation techniques, such as differential privacy or model unlearning, often require retraining or access to internal weights making them impractical for most users. In this work, we introduce TokenSwap, a lightweight, post-hoc defense designed for realistic settings where the user can only access token-level outputs. Our key insight is that while large models are necessary for high task performance, small models (e.g., DistilGPT-2) are often sufficient to assign fluent, grammatically plausible probabilities to common function words - and crucially, they memorize far less. By selectively swapping token probabilities between models, TokenSwap preserves the capabilities of large models while reducing their propensity for verbatim reproduction. Evaluations on Pythia-6.9B and Llama-3-8B show up to a 10$\times$ drop in exact memorization with negligible task degradation. Our method offers a practical, accessible solution for mitigating memorized generation in deployed LLMs.
>
---
#### [replaced 048] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.13533v2](http://arxiv.org/pdf/2408.13533v2)**

> **作者:** Jinyang Wu; Shuai Zhang; Feihu Che; Mingkuan Feng; Pengpeng Shao; Jianhua Tao
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a crucial method for addressing hallucinations in large language models (LLMs). While recent research has extended RAG models to complex noisy scenarios, these explorations often confine themselves to limited noise types and presuppose that noise is inherently detrimental to LLMs, potentially deviating from real-world retrieval environments and restricting practical applicability. In this paper, we define seven distinct noise types from a linguistic perspective and establish a Noise RAG Benchmark (NoiserBench), a comprehensive evaluation framework encompassing multiple datasets and reasoning tasks. Through empirical evaluation of eight representative LLMs with diverse architectures and scales, we reveal that these noises can be further categorized into two practical groups: noise that is beneficial to LLMs (aka beneficial noise) and noise that is harmful to LLMs (aka harmful noise). While harmful noise generally impairs performance, beneficial noise may enhance several aspects of model capabilities and overall performance. Our analysis offers insights for developing more robust, adaptable RAG solutions and mitigating hallucinations across diverse retrieval scenarios.
>
---
#### [replaced 049] Monocle: Hybrid Local-Global In-Context Evaluation for Long-Text Generation with Uncertainty-Based Active Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20195v2](http://arxiv.org/pdf/2505.20195v2)**

> **作者:** Xiaorong Wang; Ting Yang; Zhu Zhang; Shuo Wang; Zihan Zhou; Liner Yang; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Assessing the quality of long-form, model-generated text is challenging, even with advanced LLM-as-a-Judge methods, due to performance degradation as input length increases. To address this issue, we propose a divide-and-conquer approach, which breaks down the comprehensive evaluation task into a series of localized scoring tasks, followed by a final global assessment. This strategy allows for more granular and manageable evaluations, ensuring that each segment of the text is assessed in isolation for both coherence and quality, while also accounting for the overall structure and consistency of the entire piece. Moreover, we introduce a hybrid in-context learning approach that leverages human annotations to enhance the performance of both local and global evaluations. By incorporating human-generated feedback directly into the evaluation process, this method allows the model to better align with human judgment. Finally, we develop an uncertainty-based active learning algorithm that efficiently selects data samples for human annotation, thereby reducing annotation costs in practical scenarios. Experimental results show that the proposed evaluation framework outperforms several representative baselines, highlighting the effectiveness of our approach.
>
---
#### [replaced 050] LLMs with Industrial Lens: Deciphering the Challenges and Prospects -- A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.14558v2](http://arxiv.org/pdf/2402.14558v2)**

> **作者:** Ashok Urlana; Charaka Vinayak Kumar; Ajeet Kumar Singh; Bala Mallikarjunarao Garlapati; Srinivasa Rao Chalamala; Rahul Mishra
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Large language models (LLMs) have become the secret ingredient driving numerous industrial applications, showcasing their remarkable versatility across a diverse spectrum of tasks. From natural language processing and sentiment analysis to content generation and personalized recommendations, their unparalleled adaptability has facilitated widespread adoption across industries. This transformative shift driven by LLMs underscores the need to explore the underlying associated challenges and avenues for enhancement in their utilization. In this paper, our objective is to unravel and evaluate the obstacles and opportunities inherent in leveraging LLMs within an industrial context. To this end, we conduct a survey involving a group of industry practitioners, develop four research questions derived from the insights gathered, and examine 68 industry papers to address these questions and derive meaningful conclusions. We maintain the Github repository with the most recent papers in the field.
>
---
#### [replaced 051] Does quantization affect models' performance on long-context tasks?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20276v2](http://arxiv.org/pdf/2505.20276v2)**

> **作者:** Anmol Mekala; Anirudh Atmakuru; Yixiao Song; Marzena Karpinska; Mohit Iyyer
>
> **备注:** 9 pages of content with 9 figures. 37 remaining pages of references and supplementary with 17 figures. Under review as of May 26
>
> **摘要:** Large language models (LLMs) now support context windows exceeding 128K tokens, but this comes with significant memory requirements and high inference latency. Quantization can mitigate these costs, but may degrade performance. In this work, we present the first systematic evaluation of quantized LLMs on tasks with long-inputs (>64K tokens) and long-form outputs. Our evaluation spans 9.7K test examples, five quantization methods (FP8, GPTQ-int8, AWQ-int4, GPTQ-int4, BNB-nf4), and five models (Llama-3.1 8B and 70B; Qwen-2.5 7B, 32B, and 72B). We find that, on average, 8-bit quantization preserves accuracy (~0.8% drop), whereas 4-bit methods lead to substantial losses, especially for tasks involving long context inputs (drops of up to 59%). This degradation tends to worsen when the input is in a language other than English. Crucially, the effects of quantization depend heavily on the quantization method, model, and task. For instance, while Qwen-2.5 72B remains robust under BNB-nf4, Llama-3.1 70B experiences a 32% performance drop on the same task. These findings highlight the importance of a careful, task-specific evaluation before deploying quantized LLMs, particularly in long-context scenarios and with languages other than English.
>
---
#### [replaced 052] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19628v2](http://arxiv.org/pdf/2505.19628v2)**

> **作者:** Silin Li; Yuhang Guo; Jiashu Yao; Zeming Liu; Haifeng Wang
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** Large language models (LLMs) have the potential to revolutionize smart home assistants by enhancing their ability to accurately understand user needs and respond appropriately, which is extremely beneficial for building a smarter home environment. While recent studies have explored integrating LLMs into smart home systems, they primarily focus on handling straightforward, valid single-device operation instructions. However, real-world scenarios are far more complex and often involve users issuing invalid instructions or controlling multiple devices simultaneously. These have two main challenges: LLMs must accurately identify and rectify errors in user instructions and execute multiple user instructions perfectly. To address these challenges and advance the development of LLM-based smart home assistants, we introduce HomeBench, the first smart home dataset with valid and invalid instructions across single and multiple devices in this paper. We have experimental results on 13 distinct LLMs; e.g., GPT-4o achieves only a 0.0% success rate in the scenario of invalid multi-device instructions, revealing that the existing state-of-the-art LLMs still cannot perform well in this situation even with the help of in-context learning, retrieval-augmented generation, and fine-tuning. Our code and dataset are publicly available at https://github.com/BITHLP/HomeBench.
>
---
#### [replaced 053] Path Pooling: Training-Free Structure Enhancement for Efficient Knowledge Graph Retrieval-Augmented Generation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05203v2](http://arxiv.org/pdf/2503.05203v2)**

> **作者:** Hairu Wang; Yuan Feng; Xike Xie; S Kevin Zhou
>
> **摘要:** Although Large Language Models achieve strong success in many tasks, they still suffer from hallucinations and knowledge deficiencies in real-world applications. Many knowledge graph-based retrieval-augmented generation (KG-RAG) methods enhance the quality and credibility of LLMs by leveraging structure and semantic information in KGs as external knowledge bases. However, these methods struggle to effectively incorporate structure information, either incurring high computational costs or underutilizing available knowledge. Inspired by smoothing operations in graph representation learning, we propose path pooling, a simple, training-free strategy that introduces structure information through a novel path-centric pooling operation. It seamlessly integrates into existing KG-RAG methods in a plug-and-play manner, enabling richer structure information utilization. Extensive experiments demonstrate that incorporating the path pooling into the state-of-the-art KG-RAG method consistently improves performance across various settings while introducing negligible additional cost.
>
---
#### [replaced 054] ANCHOLIK-NER: A Benchmark Dataset for Bangla Regional Named Entity Recognition
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11198v3](http://arxiv.org/pdf/2502.11198v3)**

> **作者:** Bidyarthi Paul; Faika Fairuj Preotee; Shuvashis Sarker; Shamim Rahim Refat; Shifat Islam; Tashreef Muhammad; Mohammad Ashraful Hoque; Shahriar Manzoor
>
> **摘要:** Named Entity Recognition (NER) in regional dialects is a critical yet underexplored area in Natural Language Processing (NLP), especially for low-resource languages like Bangla. While NER systems for Standard Bangla have made progress, no existing resources or models specifically address the challenge of regional dialects such as Barishal, Chittagong, Mymensingh, Noakhali, and Sylhet, which exhibit unique linguistic features that existing models fail to handle effectively. To fill this gap, we introduce ANCHOLIK-NER, the first benchmark dataset for NER in Bangla regional dialects, comprising 17,405 sentences distributed across five regions. The dataset was sourced from publicly available resources and supplemented with manual translations, ensuring alignment of named entities across dialects. We evaluate three transformer-based models - Bangla BERT, Bangla BERT Base, and BERT Base Multilingual Cased - on this dataset. Our findings demonstrate that BERT Base Multilingual Cased performs best in recognizing named entities across regions, with significant performance observed in Mymensingh with an F1-score of 82.611%. Despite strong overall performance, challenges remain in region like Chittagong, where the models show lower precision and recall. Since no previous NER systems for Bangla regional dialects exist, our work represents a foundational step in addressing this gap. Future work will focus on improving model performance in underperforming regions and expanding the dataset to include more dialects, enhancing the development of dialect-aware NER systems.
>
---
#### [replaced 055] LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **链接: [http://arxiv.org/pdf/2503.14434v2](http://arxiv.org/pdf/2503.14434v2)**

> **作者:** Nikhil Abhyankar; Parshin Shojaee; Chandan K. Reddy
>
> **摘要:** Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks.
>
---
#### [replaced 056] The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.09674v4](http://arxiv.org/pdf/2502.09674v4)**

> **作者:** Wenbo Pan; Zhichao Liu; Qiguang Chen; Xiangyang Zhou; Haining Yu; Xiaohua Jia
>
> **备注:** Code and artifacts: https://github.com/BMPixel/safety-residual-space Accepted by ICML 2025
>
> **摘要:** Large Language Models' safety-aligned behaviors, such as refusing harmful queries, can be represented by linear directions in activation space. Previous research modeled safety behavior with a single direction, limiting mechanistic understanding to an isolated safety feature. In this work, we discover that safety-aligned behavior is jointly controlled by multi-dimensional directions. Namely, we study the vector space of representation shifts during safety fine-tuning on Llama 3 8B for refusing jailbreaks. By studying orthogonal directions in the space, we first find that a dominant direction governs the model's refusal behavior, while multiple smaller directions represent distinct and interpretable features like hypothetical narrative and role-playing. We then measure how different directions promote or suppress the dominant direction, showing the important role of secondary directions in shaping the model's refusal representation. Finally, we demonstrate that removing certain trigger tokens in harmful queries can mitigate these directions to bypass the learned safety capability, providing new insights on understanding safety alignment vulnerability from a multi-dimensional perspective. Code and artifacts are available at https://github.com/BMPixel/safety-residual-space.
>
---
#### [replaced 057] Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05374v4](http://arxiv.org/pdf/2502.05374v4)**

> **作者:** Chongyu Fan; Jinghan Jia; Yihua Zhang; Anil Ramakrishna; Mingyi Hong; Sijia Liu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.
>
---
#### [replaced 058] Thinking beyond the anthropomorphic paradigm benefits LLM research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09192v2](http://arxiv.org/pdf/2502.09192v2)**

> **作者:** Lujain Ibrahim; Myra Cheng
>
> **摘要:** Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of research articles to present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). We argue for challenging the deeper assumptions reflected in this terminology -- which, though often useful, may inadvertently constrain LLM development -- and broadening beyond them to open new pathways for understanding and improving LLMs. Specifically, we identify and examine five anthropomorphic assumptions that shape research across the LLM development lifecycle. For each assumption (e.g., that LLMs must use natural language for reasoning, or that they should be evaluated on benchmarks originally meant for humans), we demonstrate empirical, non-anthropomorphic alternatives that remain under-explored yet offer promising directions for LLM research and development.
>
---
#### [replaced 059] TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19586v2](http://arxiv.org/pdf/2505.19586v2)**

> **作者:** Dingyu Yao; Bowen Shen; Zheng Lin; Wei Liu; Jian Luan; Bin Wang; Weiping Wang
>
> **摘要:** The Key-Value (KV) cache in generative large language models (LLMs) introduces substantial memory overhead. Existing works mitigate this burden by offloading or compressing the KV cache. However, loading the entire cache incurs significant latency due to PCIe bandwidth bottlenecks in CPU-GPU communication, while aggressive compression causes notable performance degradation. We identify that certain layers in the LLM need to maintain global information and are unsuitable for selective loading. In contrast, other layers primarily focus on a few tokens with dominant activations that potentially incur substantial quantization error. This observation leads to a key insight that loading dominant tokens and quantizing all tokens can complement each other. Building on this insight, we propose a hybrid compression method, TailorKV, which seamlessly integrates quantization and offloading. TailorKV develops an inference framework along with a hardware-friendly implementation that leverages these complementary characteristics. Extensive long-context evaluations exhibit that TailorKV achieves nearly lossless performance under aggressive compression settings, outperforming the state-of-the-art. Particularly, the Llama-3.1-8B with 128k context can be served within a single RTX 3090 GPU, reaching 82 ms per token during decoding.
>
---
#### [replaced 060] Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20367v3](http://arxiv.org/pdf/2412.20367v3)**

> **作者:** Junqiao Wang; Zeng Zhang; Yangfan He; Zihao Zhang; Yuyang Song; Tianyu Shi; Yuchen Li; Hengyuan Xu; Kunyu Wu; Xin Yi; Zhongwei Wan; Xinhang Yuan; Kuan Lu; Menghao Huo; Guangwu Qian; Keqin Li; Qiuwu Chen; Lewei He
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful paradigm for enhancing large language models (LLMs) in code generation and optimization. This survey systematically reviews RL-driven techniques across the code development lifecycle, from compiler-level optimizations and resource allocation strategies to end-to-end code synthesis frameworks. We first examine classical and modern RL algorithms -- spanning policy gradients, actor-critic methods, human-feedback alignment, and preference-based optimization -- and their adaptations to the unique challenges of code generation, such as sparse and delayed rewards. Next, we analyze key benchmarks, datasets, and evaluation metrics that drive progress in RL-augmented Code LLMs. Finally, we identify open problems, including the need for richer feedback sources, support for low-level and domain-specific languages, and methods to reduce computational overhead. By consolidating current insights and outlining future directions, this work aims to guide researchers and practitioners in leveraging RL to produce more robust, efficient, and human-aligned code generation systems.
>
---
#### [replaced 061] CLEVRER-Humans: Describing Physical and Causal Events the Human Way
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2310.03635v2](http://arxiv.org/pdf/2310.03635v2)**

> **作者:** Jiayuan Mao; Xuelin Yang; Xikun Zhang; Noah D. Goodman; Jiajun Wu
>
> **备注:** Version 3. NeurIPS 2022 (Dataset and Benchmark Track). First two authors contributed equally. Project page: https://sites.google.com/stanford.edu/clevrer-humans/home
>
> **摘要:** Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models. We convert the collected CEGs into questions and answers to be consistent with prior work. Finally, we study a collection of baseline approaches for CLEVRER-Humans question-answering, highlighting the great challenges set forth by our benchmark.
>
---
#### [replaced 062] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17595v2](http://arxiv.org/pdf/2505.17595v2)**

> **作者:** Li Lin; Xinyu Hu; Xiaojun Wan
>
> **备注:** 9 pages, under review
>
> **摘要:** Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored for its efficiency and ease of deployment since uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on $\geq 2$-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they primarily focus on quantization methodologies, while the initialization of quantization parameters is underexplored and still relies on the suboptimal Min-Max strategies. In this work, we propose NeUQI, a method devoted to efficiently determining near-optimal initial parameters for uniform quantization. NeUQI is orthogonal to prior quantization methodologies and can seamlessly integrate with them. The experiments with the LLaMA and Qwen families on various tasks demonstrate that our NeUQI consistently outperforms existing methods. Furthermore, when combined with a lightweight distillation strategy, NeUQI can achieve superior performance to PV-tuning, a much more resource-intensive approach.
>
---
#### [replaced 063] Tuning LLM Judge Design Decisions for 1/1000 of the Cost
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17178v4](http://arxiv.org/pdf/2501.17178v4)**

> **作者:** David Salinas; Omar Swelam; Frank Hutter
>
> **摘要:** Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune the hyperparameters of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trade accuracy for cost and also significantly reduce the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility. The code to reproduce our experiments is available at this repository https://github.com/geoalgo/judgetuning .
>
---
#### [replaced 064] Retrospex: Language Agent Meets Offline Reinforcement Learning Critic
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11807v2](http://arxiv.org/pdf/2505.11807v2)**

> **作者:** Yufei Xiang; Yiqun Shen; Yeqin Zhang; Cam-Tu Nguyen
>
> **备注:** 17 pages, Published in EMNLP 2024 (Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing)
>
> **摘要:** Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines.
>
---
#### [replaced 065] Efficiently Scaling LLM Reasoning with Certaindex
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20993v2](http://arxiv.org/pdf/2412.20993v2)**

> **作者:** Yichao Fu; Junda Chen; Siqi Zhu; Zheyu Fu; Zhongdongming Dai; Yonghao Zhuang; Yian Ma; Aurick Qiao; Tajana Rosing; Ion Stoica; Hao Zhang
>
> **摘要:** Test-time reasoning algorithms such as chain-of-thought, self-consistency, and MCTS enhance LLM problem-solving but can wastefully generate many tokens without improving accuracy. At the same time, we observe that these algorithms exhibit answer stabilization: their intermediate solutions often cease to change after a certain point, and further investment of compute does not change their final answer. To quantify this phenomenon, we introduce Certaindex, an algorithm-agnostic metric measuring this evolving stability, signaling when further computation is unlikely to alter the final result. Certaindex is lightweight, can accelerate reasoning program inference via early exit, and further enables dynamic token allocation, gang scheduling, and many opportunities when integrated with real-world LLM serving systems. To quantify real-world benefits, we built Certaindex as a scheduler into Dynasor, our reasoning-aware LLM serving system, and demonstrate up to 50% compute savings and 3.3x higher throughput in real workloads with no accuracy drop. Our code is available at https://github.com/hao-ai-lab/Dynasor.git
>
---
#### [replaced 066] vCache: Verified Semantic Prompt Caching
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03771v3](http://arxiv.org/pdf/2502.03771v3)**

> **作者:** Luis Gaspar Schroeder; Aditya Desai; Alejandro Cuadron; Kyle Chu; Shu Liu; Mark Zhao; Stephan Krusche; Alfons Kemper; Matei Zaharia; Joseph E. Gonzalez
>
> **摘要:** Semantic caches return cached LLM-generated responses for semantically similar prompts to reduce inference latency and cost. They embed cached prompts and store them alongside their response in a vector database. Embedding similarity metrics assign a numerical score to quantify the similarity between a request and its nearest neighbor prompt from the cache. Existing systems use the same static similarity threshold across all requests to determine whether two prompts can share similar responses. However, we observe that static thresholds do not give formal correctness guarantees, can result in unexpected error rates, and lead to suboptimal cache hit rates. This paper proposes vCache, the first verified semantic cache with user-defined error rate guarantees. It employs an online learning algorithm to estimate an optimal threshold for each cached prompt, enabling reliable cache responses without additional training. Our experiments show that vCache consistently meets the specified error bounds while outperforming state-of-the-art static-threshold and fine-tuned embedding baselines. We release the vCache implementation and benchmarks to support future research.
>
---
#### [replaced 067] GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration
- **分类: cs.LG; cs.AI; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.16216v3](http://arxiv.org/pdf/2412.16216v3)**

> **作者:** Ting Bai; Yue Yu; Le Huang; Zenan Xu; Zhe Zhao; Chuan Shi
>
> **备注:** 9 pages, 25 figures
>
> **摘要:** The sparse Mixture-of-Experts (MoE) architecture of large language models (LLMs) confronts an inherent issue of load imbalance arising from the simplistic linear router strategy, which ultimately causes the instability and inefficient learning of LLMs. To address this challenge, we introduce a novel MoE graph-based framework $\textbf{GMoE}$, aimed at enhancing the collaboration among multiple experts. In GMoE, a graph router function is designed to capture the collaboration signals among experts. This enables all experts to dynamically allocate information derived from input data by sharing information with their neighboring experts. Moreover, we put forward two coordination strategies in GMoE: the $\textit{Poisson distribution-based distinction strategy}$ and the $\textit{Normal distribution-based balance strategy}$, to further release the capacity of each expert and increase the model stability in the fine-tuning of LLMs. Specifically, we leverage a parameter-efficient fine-tuning technique, i.e., Low-Rank Adaptation (LoRA), to implement the graph MoE architecture. Extensive experiments on four real-world benchmark datasets demonstrate the effectiveness of GMoE, showing the benefits of facilitating collaborations of multiple experts in LLM fine-tuning. The code of experimental implementation is available at https://github.com/BAI-LAB/GMoE
>
---
#### [replaced 068] TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.20118v2](http://arxiv.org/pdf/2505.20118v2)**

> **作者:** Dominik Meier; Jan Philip Wahle; Paul Röttger; Terry Ruas; Bela Gipp
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.
>
---
#### [replaced 069] Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalization of Misinformation Detection Models
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18122v2](http://arxiv.org/pdf/2410.18122v2)**

> **作者:** Ivo Verhoeven; Pushkar Mishra; Ekaterina Shutova
>
> **备注:** Under review
>
> **摘要:** This article introduces misinfo-general, a benchmark dataset for evaluating misinformation models' ability to perform out-of-distribution generalization. Misinformation changes rapidly, much more quickly than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation detectors need to be able to perform out-of-distribution generalization, an attribute they currently lack. Our benchmark uses distant labelling to enable simulating covariate shifts in misinformation content. We identify time, event, topic, publisher, political bias, misinformation type as important axes for generalization, and we evaluate a common class of baseline models on each. Using article metadata, we show how this model fails desiderata, which is not necessarily obvious from classification metrics. Finally, we analyze properties of the data to ensure limited presence of modelling shortcuts. We make the dataset and accompanying code publicly available: https://github.com/ioverho/misinfo-general
>
---
#### [replaced 070] QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17667v2](http://arxiv.org/pdf/2505.17667v2)**

> **作者:** Fanqi Wan; Weizhou Shen; Shengyi Liao; Yingcheng Shi; Chenliang Li; Ziyi Yang; Ji Zhang; Fei Huang; Jingren Zhou; Ming Yan
>
> **备注:** Technical Report
>
> **摘要:** Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QwenLong-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QwenLong-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments.
>
---
#### [replaced 071] SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11484v2](http://arxiv.org/pdf/2505.11484v2)**

> **作者:** Yige Xu; Xu Guo; Zhiwei Zeng; Chunyan Miao
>
> **备注:** 14 pages
>
> **摘要:** Test-Time Scaling (TTS) refers to approaches that improve reasoning performance by allocating extra computation during inference, without altering the model's parameters. While existing TTS methods operate in a discrete token space by generating more intermediate steps, recent studies in Coconut and SoftCoT have demonstrated that thinking in the continuous latent space can further enhance the reasoning performance. Such latent thoughts encode informative thinking without the information loss associated with autoregressive token generation, sparking increased interest in continuous-space reasoning. Unlike discrete decoding, where repeated sampling enables exploring diverse reasoning paths, latent representations in continuous space are fixed for a given input, which limits diverse exploration, as all decoded paths originate from the same latent thought. To overcome this limitation, we introduce SoftCoT++ to extend SoftCoT to the Test-Time Scaling paradigm by enabling diverse exploration of thinking paths. Specifically, we perturb latent thoughts via multiple specialized initial tokens and apply contrastive learning to promote diversity among soft thought representations. Experiments across five reasoning benchmarks and two distinct LLM architectures demonstrate that SoftCoT++ significantly boosts SoftCoT and also outperforms SoftCoT with self-consistency scaling. Moreover, it shows strong compatibility with conventional scaling techniques such as self-consistency. Source code is available at https://github.com/xuyige/SoftCoT.
>
---
#### [replaced 072] Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.10056v4](http://arxiv.org/pdf/2403.10056v4)**

> **作者:** Yongquan He; Wenyuan Zhang; Xuancheng Huang; Peng Zhang; Lingxun Meng; Xiang Zhou; Ke Zeng; Xunliang Cai
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score, to measure the generalization and instruction-following abilities of LLMs. Experiments demonstrate our method achieves superior performance on both seen and held-out tasks.
>
---
#### [replaced 073] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10610v2](http://arxiv.org/pdf/2505.10610v2)**

> **作者:** Zhaowei Wang; Wenhao Yu; Xiyu Ren; Jipeng Zhang; Yu Zhao; Rohit Saxena; Liang Cheng; Ginny Wong; Simon See; Pasquale Minervini; Yangqiu Song; Mark Steedman
>
> **备注:** Work in progress
>
> **摘要:** The rapid extension of context windows in large vision-language models has given rise to long-context vision-language models (LCVLMs), which are capable of handling hundreds of images with interleaved text tokens in a single forward pass. In this work, we introduce MMLongBench, the first benchmark covering a diverse set of long-context vision-language tasks, to evaluate LCVLMs effectively and thoroughly. MMLongBench is composed of 13,331 examples spanning five different categories of downstream tasks, such as Visual RAG and Many-Shot ICL. It also provides broad coverage of image types, including various natural and synthetic images. To assess the robustness of the models to different input lengths, all examples are delivered at five standardized input lengths (8K-128K tokens) via a cross-modal tokenization scheme that combines vision patches and text tokens. Through a thorough benchmarking of 46 closed-source and open-source LCVLMs, we provide a comprehensive analysis of the current models' vision-language long-context ability. Our results show that: i) performance on a single task is a weak proxy for overall long-context capability; ii) both closed-source and open-source models face challenges in long-context vision-language tasks, indicating substantial room for future improvement; iii) models with stronger reasoning ability tend to exhibit better long-context performance. By offering wide task coverage, various image types, and rigorous length control, MMLongBench provides the missing foundation for diagnosing and advancing the next generation of LCVLMs.
>
---
#### [replaced 074] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19249v2](http://arxiv.org/pdf/2502.19249v2)**

> **作者:** Michael Y. Hu; Jackson Petty; Chuan Shi; William Merrill; Tal Linzen
>
> **备注:** ACL 2025 Camera Ready
>
> **摘要:** Pretraining language models on formal language can improve their acquisition of natural language. Which features of the formal language impart an inductive bias that leads to effective transfer? Drawing on insights from linguistics and complexity theory, we hypothesize that effective transfer occurs when two conditions are met: the formal language should capture the dependency structures present in natural language, and it should remain within the computational limitations of the model architecture. We experiment with pre-pretraining (training on formal language before natural languages) on transformers and find that formal languages capturing hierarchical dependencies indeed enable language models to achieve lower loss on natural language and better linguistic generalization compared to other formal languages. We also find modest support for the hypothesis that the formal language should fall within the computational limitations of the architecture. Strikingly, pre-pretraining reduces loss more efficiently than training on a matched amount of natural language. For a 1B-parameter language model trained on roughly 1.6B tokens of natural language, pre-pretraining achieves the same loss and better linguistic generalization with a 33% smaller token budget. Finally, we also give mechanistic evidence of transfer from formal to natural language: attention heads acquired during pre-pretraining remain crucial for the model's performance on syntactic evaluations.
>
---
#### [replaced 075] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.04183v2](http://arxiv.org/pdf/2409.04183v2)**

> **作者:** Ziyin Zhang; Hang Yu; Shijie Li; Peng Di; Jianguo Li; Rui Wang
>
> **备注:** ACL 2025 camera-ready
>
> **摘要:** Programming languages possess rich semantic information - such as data flow - that is represented by graphs and not available from the surface form of source code. Recent code language models have scaled to billions of parameters, but model source code solely as text tokens while ignoring any other structural information. Conversely, models that do encode structural information of code make modifications to the Transformer architecture, limiting their scale and compatibility with pretrained LLMs. In this work, we take the best of both worlds with GALLa - Graph Aligned Large Language Models. GALLa utilizes graph neural networks and cross-modal alignment technologies to inject the structural information of code into LLMs as an auxiliary task during finetuning. This framework is both model-agnostic and task-agnostic, as it can be applied to any code LLM for any code downstream task, and requires the structural graph data only at training time from a corpus unrelated to the finetuning data, while incurring no cost at inference time over the baseline LLM. Experiments on five code tasks with seven different baseline LLMs ranging in size from 350M to 14B validate the effectiveness of GALLa, demonstrating consistent improvement over the baseline, even for powerful models such as LLaMA3 and Qwen2.5-Coder.
>
---
#### [replaced 076] Language Models Surface the Unwritten Code of Science and Society
- **分类: cs.CY; cs.CL; cs.DL**

- **链接: [http://arxiv.org/pdf/2505.18942v2](http://arxiv.org/pdf/2505.18942v2)**

> **作者:** Honglin Bao; Siyang Wu; Jiwoong Choi; Yingrong Mao; James A. Evans
>
> **摘要:** This paper calls on the research community not only to investigate how human biases are inherited by large language models (LLMs) but also to explore how these biases in LLMs can be leveraged to make society's "unwritten code" - such as implicit stereotypes and heuristics - visible and accessible for critique. We introduce a conceptual framework through a case study in science: uncovering hidden rules in peer review - the factors that reviewers care about but rarely state explicitly due to normative scientific expectations. The idea of the framework is to push LLMs to speak out their heuristics through generating self-consistent hypotheses - why one paper appeared stronger in reviewer scoring - among paired papers submitted to 45 computer science conferences, while iteratively searching deeper hypotheses from remaining pairs where existing hypotheses cannot explain. We observed that LLMs' normative priors about the internal characteristics of good science extracted from their self-talk, e.g. theoretical rigor, were systematically updated toward posteriors that emphasize storytelling about external connections, such as how the work is positioned and connected within and across literatures. This shift reveals the primacy of scientific myths about intrinsic properties driving scientific excellence rather than extrinsic contextualization and storytelling that influence conceptions of relevance and significance. Human reviewers tend to explicitly reward aspects that moderately align with LLMs' normative priors (correlation = 0.49) but avoid articulating contextualization and storytelling posteriors in their review comments (correlation = -0.14), despite giving implicit reward to them with positive scores. We discuss the broad applicability of the framework, leveraging LLMs as diagnostic tools to surface the tacit codes underlying human society, enabling more precisely targeted responsible AI.
>
---
#### [replaced 077] Leveraging Large Language Models for Active Merchant Non-player Characters
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11189v3](http://arxiv.org/pdf/2412.11189v3)**

> **作者:** Byungjun Kim; Minju Kim; Dayeon Seo; Bugeun Kim
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** We highlight two significant issues leading to the passivity of current merchant non-player characters (NPCs): pricing and communication. While immersive interactions with active NPCs have been a focus, price negotiations between merchant NPCs and players remain underexplored. First, passive pricing refers to the limited ability of merchants to modify predefined item prices. Second, passive communication means that merchants can only interact with players in a scripted manner. To tackle these issues and create an active merchant NPC, we propose a merchant framework based on large language models (LLMs), called MART, which consists of an appraiser module and a negotiator module. We conducted two experiments to explore various implementation options under different training methods and LLM sizes, considering a range of possible game environments. Our findings indicate that finetuning methods, such as supervised finetuning (SFT) and knowledge distillation (KD), are effective in using smaller LLMs to implement active merchant NPCs. Additionally, we found three irregular cases arising from the responses of LLMs.
>
---
#### [replaced 078] No LLM is Free From Bias: A Comprehensive Study of Bias Evaluation in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11985v2](http://arxiv.org/pdf/2503.11985v2)**

> **作者:** Charaka Vinayak Kumar; Ashok Urlana; Gopichand Kanumolu; Bala Mallikarjunarao Garlapati; Pruthwik Mishra
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Advancements in Large Language Models (LLMs) have increased the performance of different natural language understanding as well as generation tasks. Although LLMs have breached the state-of-the-art performance in various tasks, they often reflect different forms of bias present in the training data. In the light of this perceived limitation, we provide a unified evaluation of benchmarks using a set of representative small and medium-sized LLMs that cover different forms of biases starting from physical characteristics to socio-economic categories. Moreover, we propose five prompting approaches to carry out the bias detection task across different aspects of bias. Further, we formulate three research questions to gain valuable insight in detecting biases in LLMs using different approaches and evaluation metrics across benchmarks. The results indicate that each of the selected LLMs suffer from one or the other form of bias with the Phi-3.5B model being the least biased. Finally, we conclude the paper with the identification of key challenges and possible future directions.
>
---
#### [replaced 079] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19484v2](http://arxiv.org/pdf/2505.19484v2)**

> **作者:** Ruixiang Feng; Shen Gao; Xiuying Chen; Lisi Chen; Shuo Shang
>
> **备注:** accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they often exhibit a specific cultural biases, neglecting the values and linguistic diversity of low-resource regions. This cultural bias not only undermines universal equality, but also risks reinforcing stereotypes and perpetuating discrimination. To address this, we propose CulFiT, a novel culturally-aware training paradigm that leverages multilingual data and fine-grained reward modeling to enhance cultural sensitivity and inclusivity. Our approach synthesizes diverse cultural-related questions, constructs critique data in culturally relevant languages, and employs fine-grained rewards to decompose cultural texts into verifiable knowledge units for interpretable evaluation. We also introduce GlobalCultureQA, a multilingual open-ended question-answering dataset designed to evaluate culturally-aware responses in a global context. Extensive experiments on three existing benchmarks and our GlobalCultureQA demonstrate that CulFiT achieves state-of-the-art open-source model performance in cultural alignment and general reasoning.
>
---
#### [replaced 080] Aggregation Artifacts in Subjective Tasks Collapse Large Language Models' Posteriors
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.13776v4](http://arxiv.org/pdf/2410.13776v4)**

> **作者:** Georgios Chochlakis; Alexandros Potamianos; Kristina Lerman; Shrikanth Narayanan
>
> **备注:** 16 pages, 12 figures, 3 tables
>
> **摘要:** In-context Learning (ICL) has become the primary method for performing natural language tasks with Large Language Models (LLMs). The knowledge acquired during pre-training is crucial for this few-shot capability, providing the model with task priors. However, recent studies have shown that ICL predominantly relies on retrieving task priors rather than "learning" to perform tasks. This limitation is particularly evident in complex subjective domains such as emotion and morality, where priors significantly influence posterior predictions. In this work, we examine whether this is the result of the aggregation used in corresponding datasets, where trying to combine low-agreement, disparate annotations might lead to annotation artifacts that create detrimental noise in the prompt. Moreover, we evaluate the posterior bias towards certain annotators by grounding our study in appropriate, quantitative measures of LLM priors. Our results indicate that aggregation is a confounding factor in the modeling of subjective tasks, and advocate focusing on modeling individuals instead. However, aggregation does not explain the entire gap between ICL and the state of the art, meaning other factors in such tasks also account for the observed phenomena. Finally, by rigorously studying annotator-level labels, we find that it is possible for minority annotators to both better align with LLMs and have their perspectives further amplified.
>
---
#### [replaced 081] SCIRGC: Multi-Granularity Citation Recommendation and Citation Sentence Preference Alignment
- **分类: cs.DL; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20103v2](http://arxiv.org/pdf/2505.20103v2)**

> **作者:** Xiangyu Li; Jingqiang Chen
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Citations are crucial in scientific research articles as they highlight the connection between the current study and prior work. However, this process is often time-consuming for researchers. In this study, we propose the SciRGC framework, which aims to automatically recommend citation articles and generate citation sentences for citation locations within articles. The framework addresses two key challenges in academic citation generation: 1) how to accurately identify the author's citation intent and find relevant citation papers, and 2) how to generate high-quality citation sentences that align with human preferences. We enhance citation recommendation accuracy in the citation article recommendation module by incorporating citation networks and sentiment intent, and generate reasoning-based citation sentences in the citation sentence generation module by using the original article abstract, local context, citation intent, and recommended articles as inputs. Additionally, we propose a new evaluation metric to fairly assess the quality of generated citation sentences. Through comparisons with baseline models and ablation experiments, the SciRGC framework not only improves the accuracy and relevance of citation recommendations but also ensures the appropriateness of the generated citation sentences in context, providing a valuable tool for interdisciplinary researchers.
>
---
#### [replaced 082] Behavioral Analysis of Information Salience in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14613v2](http://arxiv.org/pdf/2502.14613v2)**

> **作者:** Jan Trienes; Jörg Schlötterer; Junyi Jessy Li; Christin Seifert
>
> **备注:** Accepted at ACL 2025 (Findings)
>
> **摘要:** Large Language Models (LLMs) excel at text summarization, a task that requires models to select content based on its importance. However, the exact notion of salience that LLMs have internalized remains unclear. To bridge this gap, we introduce an explainable framework to systematically derive and investigate information salience in LLMs through their summarization behavior. Using length-controlled summarization as a behavioral probe into the content selection process, and tracing the answerability of Questions Under Discussion throughout, we derive a proxy for how models prioritize information. Our experiments on 13 models across four datasets reveal that LLMs have a nuanced, hierarchical notion of salience, generally consistent across model families and sizes. While models show highly consistent behavior and hence salience patterns, this notion of salience cannot be accessed through introspection, and only weakly correlates with human perceptions of information salience.
>
---
#### [replaced 083] Information Gain-Guided Causal Intervention for Autonomous Debiasing Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12898v3](http://arxiv.org/pdf/2504.12898v3)**

> **作者:** Zhouhao Sun; Xiao Ding; Li Du; Yunpeng Xu; Yixuan Ma; Yang Zhao; Bing Qin; Ting Liu
>
> **摘要:** Despite significant progress, recent studies indicate that current large language models (LLMs) may still capture dataset biases and utilize them during inference, leading to the poor generalizability of LLMs. However, due to the diversity of dataset biases and the insufficient nature of bias suppression based on in-context learning, the effectiveness of previous prior knowledge-based debiasing methods and in-context learning based automatic debiasing methods is limited. To address these challenges, we explore the combination of causal mechanisms with information theory and propose an information gain-guided causal intervention debiasing (ICD) framework. To eliminate biases within the instruction-tuning dataset, it is essential to ensure that these biases do not provide any additional information to predict the answers, i.e., the information gain of these biases for predicting the answers needs to be 0. Under this guidance, this framework utilizes a causal intervention-based data rewriting method to automatically and autonomously balance the distribution of instruction-tuning dataset for reducing the information gain. Subsequently, it employs a standard supervised fine-tuning process to train LLMs on the debiased dataset. Experimental results show that ICD can effectively debias LLM to improve its generalizability across different tasks.
>
---
#### [replaced 084] Sentiment Reasoning for Healthcare
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.21054v4](http://arxiv.org/pdf/2407.21054v4)**

> **作者:** Khai-Nguyen Nguyen; Khai Le-Duc; Bach Phan Tat; Duy Le; Long Vo-Dang; Truong-Son Hy
>
> **备注:** ACL 2025 (Oral)
>
> **摘要:** Transparency in AI healthcare decision-making is crucial. By incorporating rationales to explain reason for each predicted label, users could understand Large Language Models (LLMs)'s reasoning to make better decision. In this work, we introduce a new task - Sentiment Reasoning - for both speech and text modalities, and our proposed multimodal multitask framework and the world's largest multimodal sentiment analysis dataset. Sentiment Reasoning is an auxiliary task in sentiment analysis where the model predicts both the sentiment label and generates the rationale behind it based on the input transcript. Our study conducted on both human transcripts and Automatic Speech Recognition (ASR) transcripts shows that Sentiment Reasoning helps improve model transparency by providing rationale for model prediction with quality semantically comparable to humans while also improving model's classification performance (+2% increase in both accuracy and macro-F1) via rationale-augmented fine-tuning. Also, no significant difference in the semantic quality of generated rationales between human and ASR transcripts. All code, data (five languages - Vietnamese, English, Chinese, German, and French) and models are published online: https://github.com/leduckhai/Sentiment-Reasoning
>
---
#### [replaced 085] One-shot Entropy Minimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20282v2](http://arxiv.org/pdf/2505.20282v2)**

> **作者:** Zitian Gao; Lynx Chen; Joey Zhou; Bryan Dai
>
> **备注:** Work in progress
>
> **摘要:** We trained 13,440 large language models and found that entropy minimization requires only a single unlabeled data and 10 steps optimization to achieve performance improvements comparable to or even greater than those obtained using thousands of data and carefully designed rewards in rule-based reinforcement learning. This striking result may prompt a rethinking of post-training paradigms for large language models. Our code is avaliable at https://github.com/zitian-gao/one-shot-em.
>
---
#### [replaced 086] R-TOFU: Unlearning in Large Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15214v2](http://arxiv.org/pdf/2505.15214v2)**

> **作者:** Sangyeon Yoon; Wonje Jeung; Albert No
>
> **备注:** 19 pages
>
> **摘要:** Large Reasoning Models (LRMs) embed private or copyrighted information not only in their final answers but also throughout multi-step chain-of-thought (CoT) traces, making reliable unlearning far more demanding than in standard LLMs. We introduce Reasoning-TOFU (R-TOFU), the first benchmark tailored to this setting. R-TOFU augments existing unlearning tasks with realistic CoT annotations and provides step-wise metrics that expose residual knowledge invisible to answer-level checks. Using R-TOFU, we carry out a comprehensive comparison of gradient-based and preference-optimization baselines and show that conventional answer-only objectives leave substantial forget traces in reasoning. We further propose Reasoned IDK, a preference-optimization variant that preserves coherent yet inconclusive reasoning, achieving a stronger balance between forgetting efficacy and model utility than earlier refusal styles. Finally, we identify a failure mode: decoding variants such as ZeroThink and LessThink can still reveal forgotten content despite seemingly successful unlearning, emphasizing the need to evaluate models under diverse decoding settings. Together, the benchmark, analysis, and new baseline establish a systematic foundation for studying and improving unlearning in LRMs while preserving their reasoning capabilities.
>
---
#### [replaced 087] Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18034v2](http://arxiv.org/pdf/2505.18034v2)**

> **作者:** Wentao Sun; João Paulo Nogueira; Alonso Silva
>
> **摘要:** Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks.
>
---
#### [replaced 088] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.17117v2](http://arxiv.org/pdf/2505.17117v2)**

> **作者:** Chen Shani; Dan Jurafsky; Yann LeCun; Ravid Shwartz-Ziv
>
> **摘要:** Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
>
---
#### [replaced 089] Frequency matters: Modeling irregular morphological patterns in Spanish with Transformers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21013v4](http://arxiv.org/pdf/2410.21013v4)**

> **作者:** Akhilesh Kakolu Ramarao; Kevin Tang; Dinah Baer-Henney
>
> **备注:** Typos and grammatical corrections
>
> **摘要:** Over the past decade, various studies have addressed how speakers solve the so-called `The Paradigm Cell Filling Problem' (PCFP) \citep{ackerman2009parts} across different languages. The PCFP addresses a fundamental question in morphological processing: how do speakers accurately generate inflected forms of words when presented with incomplete paradigms? This problem is particularly salient when modeling complex inflectional systems. We focus on Spanish verbal paradigms, where certain verbs follow an irregular L-shaped pattern, where the first-person singular present indicative stem matches the stem used throughout the present subjunctive mood. We formulate the problem as a morphological reinflection task. Specifically, we investigate the role of input frequency in the acquisition of regular versus irregular L-shaped patterns in transformer models. By systematically manipulating the input distributions and analyzing model behavior, we reveal four key findings: 1) Models perform better on L-shaped verbs compared to regular verbs, especially in uneven frequency conditions; 2) Robust primacy effects are observed, but no consistent recency effects; 3) Memorization becomes more prominent as the proportion of L-shaped verbs increases; 4) There is a tendency to regularize L-shaped verbs when their consonant alternation pairs are rare or absent in the training data.
>
---
#### [replaced 090] Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16522v2](http://arxiv.org/pdf/2505.16522v2)**

> **作者:** Zhouhao Sun; Zhiyuan Kan; Xiao Ding; Li Du; Yang Zhao; Bing Qin; Ting Liu
>
> **摘要:** Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs.
>
---
#### [replaced 091] DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14107v3](http://arxiv.org/pdf/2505.14107v3)**

> **作者:** Yakun Zhu; Zhongzhen Huang; Linjie Mu; Yutong Huang; Wei Nie; Jiaji Liu; Shaoting Zhang; Pengfei Liu; Xiaofan Zhang
>
> **摘要:** The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3-mini, o1, and DeepSeek-R1, achieve only 45.82%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AIs diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development https://github.com/SPIRAL-MED/DiagnosisArena.
>
---
#### [replaced 092] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.14055v2](http://arxiv.org/pdf/2411.14055v2)**

> **作者:** Hexuan Deng; Wenxiang Jiao; Xuebo Liu; Jing Li; Min Zhang; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) deliver impressive results but face challenges from increasing model sizes and computational costs. Structured pruning reduces model size and speeds up inference but often causes uneven degradation across domains, leading to biased performance. To address this, we propose DRPruning, a method that dynamically adjusts the data distribution during training to restore balanced performance across heterogeneous and multi-tasking data. Experiments in monolingual and multilingual settings show that DRPruning surpasses similarly sized models in both pruning and continued pretraining over perplexity, downstream tasks, and instruction tuning. Further analysis demonstrates the robustness of DRPruning towards various domains and distribution shifts. Furthermore, DRPruning can determine optimal reference losses and data ratios automatically, suggesting potential for broader applications. Code and scripts are available at https://github.com/hexuandeng/DRPruning.
>
---
#### [replaced 093] Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06042v2](http://arxiv.org/pdf/2502.06042v2)**

> **作者:** Louis Bethune; David Grangier; Dan Busbridge; Eleonora Gualdoni; Marco Cuturi; Pierre Ablin
>
> **备注:** 19 pages, 15 figures, preprint
>
> **摘要:** A widespread strategy to obtain a language model that performs well on a target domain is to finetune a pretrained model to perform unsupervised next-token prediction on data from that target domain. Finetuning presents two challenges: (i) if the amount of target data is limited, as in most practical applications, the model will quickly overfit, and (ii) the model will drift away from the original model, forgetting the pretraining data and the generic knowledge that comes with it. We aim to derive scaling laws that quantify these two phenomena for various target domains, amounts of available target data, and model scales. We measure the efficiency of injecting pretraining data into the finetuning data mixture to avoid forgetting and mitigate overfitting. A key practical takeaway from our study is that injecting as little as 1% of pretraining data in the finetuning data mixture prevents the model from forgetting the pretraining set.
>
---
#### [replaced 094] Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.13010v2](http://arxiv.org/pdf/2502.13010v2)**

> **作者:** Mohammad Reza Rezaei; Reza Saadati Fard; Rahul G. Krishnan; Milad Lankarany
>
> **摘要:** Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Agentic Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
>
---
#### [replaced 095] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v2](http://arxiv.org/pdf/2505.12363v2)**

> **作者:** Qi Feng
>
> **备注:** 26 pages, 19 figures, 4 tables. Code, models, and datasets are available at our project page: https://github.com/nkkbr/ViCA. This is a draft technical report. At the request of Professor Hidetoshi Shimodaira, his name has been removed from the author list
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 096] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v2](http://arxiv.org/pdf/2505.12312v2)**

> **作者:** Qi Feng
>
> **备注:** 31 pages, 10 figures, 6 tables. The implementation and fine-tuned model (ViCA-7B), along with detailed documentation, are publicly available at https://huggingface.co/nkkbr/ViCA. This is a draft technical report. At Professor Hidetoshi Shimodaira's request, his name has been removed from the author list
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 097] DUSK: Do Not Unlearn Shared Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15209v2](http://arxiv.org/pdf/2505.15209v2)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Hyesoo Hong; Soeun Kim; Seungju Han; Youngjae Yu; Albert No
>
> **备注:** 21 pages
>
> **摘要:** Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about the unauthorized use of copyrighted or sensitive data. Machine unlearning aims to remove such 'forget' data while preserving utility and information from the 'retain' set. However, existing evaluations typically assume that forget and retain sets are fully disjoint, overlooking realistic scenarios where they share overlapping content. For instance, a news article may need to be unlearned, even though the same event, such as an earthquake in Japan, is also described factually on Wikipedia. Effective unlearning should remove the specific phrasing of the news article while preserving publicly supported facts. In this paper, we introduce DUSK, a benchmark designed to evaluate unlearning methods under realistic data overlap. DUSK constructs document sets that describe the same factual content in different styles, with some shared information appearing across all sets and other content remaining unique to each. When one set is designated for unlearning, an ideal method should remove its unique content while preserving shared facts. We define seven evaluation metrics to assess whether unlearning methods can achieve this selective removal. Our evaluation of nine recent unlearning methods reveals a key limitation: while most can remove surface-level text, they often fail to erase deeper, context-specific knowledge without damaging shared content. We release DUSK as a public benchmark to support the development of more precise and reliable unlearning techniques for real-world applications.
>
---
#### [replaced 098] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2406.14230v4](http://arxiv.org/pdf/2406.14230v4)**

> **作者:** Han Jiang; Xiaoyuan Yi; Zhihua Wei; Ziang Xiao; Shu Wang; Xing Xie
>
> **备注:** ICML 2025
>
> **摘要:** Warning: Contains harmful model outputs. Despite significant advancements, the propensity of Large Language Models (LLMs) to generate harmful and unethical content poses critical challenges. Measuring value alignment of LLMs becomes crucial for their regulation and responsible deployment. Although numerous benchmarks have been constructed to assess social bias, toxicity, and ethical issues in LLMs, those static benchmarks suffer from evaluation chronoeffect, in which, as models rapidly evolve, existing benchmarks may leak into training data or become saturated, overestimating ever-developing LLMs. To tackle this problem, we propose GETA, a novel generative evolving testing approach based on adaptive testing methods in measurement theory. Unlike traditional adaptive testing methods that rely on a static test item pool, GETA probes the underlying moral boundaries of LLMs by dynamically generating test items tailored to model capability. GETA co-evolves with LLMs by learning a joint distribution of item difficulty and model value conformity, thus effectively addressing evaluation chronoeffect. We evaluated various popular LLMs with GETA and demonstrated that 1) GETA can dynamically create difficulty-tailored test items and 2) GETA's evaluation results are more consistent with models' performance on unseen OOD and i.i.d. items, laying the groundwork for future evaluation paradigms.
>
---
#### [replaced 099] Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles
- **分类: cs.CL; cs.IR; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09829v2](http://arxiv.org/pdf/2410.09829v2)**

> **作者:** Rimvydas Rubavicius; Antonio Valerio Miceli-Barone; Alex Lascarides; Subramanian Ramamoorthy
>
> **备注:** 12 pages, 5 figures, 2 tables
>
> **摘要:** Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that dialogue is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.
>
---
#### [replaced 100] Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14315v3](http://arxiv.org/pdf/2501.14315v3)**

> **作者:** Chao-Chung Wu; Zhi Rui Tam; Chieh-Yen Lin; Yun-Nung Chen; Shao-Hua Sun; Hung-yi Lee
>
> **摘要:** Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and 3 additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
>
---
#### [replaced 101] Enabling Inclusive Systematic Reviews: Incorporating Preprint Articles with Large Language Model-Driven Evaluations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13857v3](http://arxiv.org/pdf/2503.13857v3)**

> **作者:** Rui Yang; Jiayi Tong; Haoyuan Wang; Hui Huang; Ziyang Hu; Peiyu Li; Nan Liu; Christopher J. Lindsell; Michael J. Pencina; Yong Chen; Chuan Hong
>
> **备注:** 30 pages, 6 figures
>
> **摘要:** Background. Systematic reviews in comparative effectiveness research require timely evidence synthesis. Preprints accelerate knowledge dissemination but vary in quality, posing challenges for systematic reviews. Methods. We propose AutoConfidence (automated confidence assessment), an advanced framework for predicting preprint publication, which reduces reliance on manual curation and expands the range of predictors, including three key advancements: (1) automated data extraction using natural language processing techniques, (2) semantic embeddings of titles and abstracts, and (3) large language model (LLM)-driven evaluation scores. Additionally, we employed two prediction models: a random forest classifier for binary outcome and a survival cure model that predicts both binary outcome and publication risk over time. Results. The random forest classifier achieved AUROC 0.692 with LLM-driven scores, improving to 0.733 with semantic embeddings and 0.747 with article usage metrics. The survival cure model reached AUROC 0.716 with LLM-driven scores, improving to 0.731 with semantic embeddings. For publication risk prediction, it achieved a concordance index of 0.658, increasing to 0.667 with semantic embeddings. Conclusion. Our study advances the framework for preprint publication prediction through automated data extraction and multiple feature integration. By combining semantic embeddings with LLM-driven evaluations, AutoConfidence enhances predictive performance while reducing manual annotation burden. The framework has the potential to facilitate incorporation of preprint articles during the appraisal phase of systematic reviews, supporting researchers in more effective utilization of preprint resources.
>
---
#### [replaced 102] MaskSearch: A Universal Pre-Training Framework to Enhance Agentic Search Capability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20285v2](http://arxiv.org/pdf/2505.20285v2)**

> **作者:** Weiqi Wu; Xin Guan; Shen Huang; Yong Jiang; Pengjun Xie; Fei Huang; Jiuxin Cao; Hai Zhao; Jingren Zhou
>
> **备注:** Code is available at https://github.com/Alibaba-NLP/MaskSearch
>
> **摘要:** Retrieval-Augmented Language Models (RALMs) represent a classic paradigm where models enhance generative capabilities using external knowledge retrieved via a specialized module. Recent advancements in Agent techniques enable Large Language Models (LLMs) to autonomously utilize tools for retrieval, planning, and reasoning. While existing training-based methods show promise, their agentic abilities are limited by inherent characteristics of the task-specific data used during training. To further enhance the universal search capability of agents, we propose a novel pre-training framework, MaskSearch. In the pre-training stage, we introduce the Retrieval Augmented Mask Prediction (RAMP) task, where the model learns to leverage search tools to fill masked spans on a large number of pre-training data, thus acquiring universal retrieval and reasoning capabilities for LLMs. After that, the model is trained on downstream tasks to achieve further improvement. We apply both Supervised Fine-tuning (SFT) and Reinforcement Learning (RL) for training. For SFT, we combine agent-based and distillation-based methods to generate training data, starting with a multi-agent system consisting of a planner, rewriter, observer, and followed by a self-evolving teacher model. While for RL, we employ DAPO as the training framework and adopt a hybrid reward system consisting of answer rewards and format rewards. Additionally, we introduce a curriculum learning approach that allows the model to learn progressively from easier to more challenging instances based on the number of masked spans. We evaluate the effectiveness of our framework in the scenario of open-domain multi-hop question answering. Through extensive experiments, we demonstrate that MaskSearch significantly enhances the performance of LLM-based search agents on both in-domain and out-of-domain downstream tasks.
>
---
#### [replaced 103] Can Large Language Models Understand Symbolic Graphics Programs?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.08313v4](http://arxiv.org/pdf/2408.08313v4)**

> **作者:** Zeju Qiu; Weiyang Liu; Haiwen Feng; Zhen Liu; Tim Z. Xiao; Katherine M. Collins; Joshua B. Tenenbaum; Adrian Weller; Michael J. Black; Bernhard Schölkopf
>
> **备注:** ICLR 2025 Spotlight (v4: 47 pages, 26 figures, project page: https://sgp-bench.github.io/)
>
> **摘要:** Against the backdrop of enthusiasm for large language models (LLMs), there is a growing need to scientifically assess their capabilities and shortcomings. This is nontrivial in part because it is difficult to find tasks which the models have not encountered during training. Utilizing symbolic graphics programs, we propose a domain well-suited to test multiple spatial-semantic reasoning skills of LLMs. Popular in computer graphics, these programs procedurally generate visual data. While LLMs exhibit impressive skills in general program synthesis and analysis, symbolic graphics programs offer a new layer of evaluation: they allow us to test an LLM's ability to answer semantic questions about the images or 3D geometries without a vision encoder. To semantically understand the symbolic programs, LLMs would need to possess the ability to "imagine" and reason how the corresponding graphics content would look with only the symbolic description of the local curvatures and strokes. We use this task to evaluate LLMs by creating a large benchmark for the semantic visual understanding of symbolic graphics programs, built procedurally with minimal human effort. Particular emphasis is placed on transformations of images that leave the image level semantics invariant while introducing significant changes to the underlying program. We evaluate commercial and open-source LLMs on our benchmark to assess their ability to reason about visual output of programs, finding that LLMs considered stronger at reasoning generally perform better. Lastly, we introduce a novel method to improve this ability -- Symbolic Instruction Tuning (SIT), in which the LLM is finetuned with pre-collected instruction data on symbolic graphics programs. Interestingly, we find that SIT not only improves LLM's understanding on symbolic programs, but it also improves general reasoning ability on various other benchmarks.
>
---
#### [replaced 104] Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17997v2](http://arxiv.org/pdf/2505.17997v2)**

> **作者:** Jintian Shao; Yiming Cheng; Hongyi Huang; Beiwen Zhang; Zhiyu Wu; You Shan; Mingkai Zheng
>
> **备注:** We are withdrawing this submission as the underlying experiment is currently incomplete. We require additional time to gather more data and supplement the existing findings to ensure a comprehensive and robust presentation. We intend to resubmit once these additions are finalized
>
> **摘要:** The VAPO framework has demonstrated significant empirical success in enhancing the efficiency and reliability of reinforcement learning for long chain-of-thought (CoT) reasoning tasks with large language models (LLMs). By systematically addressing challenges such as value model bias, heterogeneous sequence lengths, and sparse reward signals, VAPO achieves state-of-the-art performance. While its practical benefits are evident, a deeper theoretical understanding of its underlying mechanisms and potential limitations is crucial for guiding future advancements. This paper aims to initiate such a discussion by exploring VAPO from a theoretical perspective, highlighting areas where its assumptions might be challenged and where further investigation could yield more robust and generalizable reasoning agents. We delve into the intricacies of value function approximation in complex reasoning spaces, the optimality of adaptive advantage estimation, the impact of token-level optimization, and the enduring challenges of exploration and generalization.
>
---
#### [replaced 105] Power-Law Decay Loss for Large Language Model Finetuning: Focusing on Information Sparsity to Enhance Generation Quality
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16900v3](http://arxiv.org/pdf/2505.16900v3)**

> **作者:** Jintian Shao; Yiming Cheng; Hongyi Huang; Jiayi Wu; Beiwen Zhang; Zhiyu Wu; You Shan; Mingkai Zheng
>
> **备注:** We are withdrawing this submission as the underlying experiment is currently incomplete. We require additional time to gather more data and supplement the existing findings to ensure a comprehensive and robust presentation. We intend to resubmit once these additions are finalized
>
> **摘要:** During the finetuning stage of text generation tasks, standard cross-entropy loss treats all tokens equally. This can lead models to overemphasize high-frequency, low-information tokens, neglecting lower-frequency tokens crucial for specificity and informativeness in generated content. This paper introduces a novel loss function, Power-Law Decay Loss (PDL), specifically designed to optimize the finetuning process for text generation. The core motivation for PDL stems from observations in information theory and linguistics: the informativeness of a token is often inversely proportional to its frequency of occurrence. PDL re-weights the contribution of each token in the standard cross-entropy loss based on its frequency in the training corpus, following a power-law decay. Specifically, the weights for high-frequency tokens are reduced, while low-frequency, information-dense tokens are assigned higher weights. This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the quality, diversity, and informativeness of the generated text. We theoretically elaborate on the motivation and construction of PDL and discuss its potential applications and advantages across various text generation finetuning tasks, such as abstractive summarization, dialogue systems, and style transfer.
>
---
#### [replaced 106] SEPS: A Separability Measure for Robust Unlearning in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14832v2](http://arxiv.org/pdf/2505.14832v2)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Albert No
>
> **备注:** 32 pages
>
> **摘要:** Machine unlearning aims to selectively remove targeted knowledge from Large Language Models (LLMs), ensuring they forget specified content while retaining essential information. Existing unlearning metrics assess whether a model correctly answers retain queries and rejects forget queries, but they fail to capture real-world scenarios where forget queries rarely appear in isolation. In fact, forget and retain queries often coexist within the same prompt, making mixed-query evaluation crucial. We introduce SEPS, an evaluation framework that explicitly measures a model's ability to both forget and retain information within a single prompt. Through extensive experiments across three benchmarks, we identify two key failure modes in existing unlearning methods: (1) untargeted unlearning indiscriminately erases both forget and retain content once a forget query appears, and (2) targeted unlearning overfits to single-query scenarios, leading to catastrophic failures when handling multiple queries. To address these issues, we propose Mixed Prompt (MP) unlearning, a strategy that integrates both forget and retain queries into a unified training objective. Our approach significantly improves unlearning effectiveness, demonstrating robustness even in complex settings with up to eight mixed forget and retain queries in a single prompt.
>
---
#### [replaced 107] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14910v2](http://arxiv.org/pdf/2505.14910v2)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Dongyu Yao; Zhiyuan Zhu; Ziyue Jiang; Yuhan Wang; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks. Singing voice samples are available at https://aaronz345.github.io/TCSinger2Demo/.
>
---
#### [replaced 108] Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04948v2](http://arxiv.org/pdf/2505.04948v2)**

> **作者:** Md Aminul Islam; Ahmed Sayeed Faruk
>
> **备注:** We have decided to withdraw the manuscript as it requires substantial revisions that go beyond what is appropriate for a versioned update on arXiv. We plan to resubmit once the necessary improvements are made
>
> **摘要:** Recommender systems are essential for delivering personalized content across digital platforms by modeling user preferences and behaviors. Recently, large language models (LLMs) have been adopted for prompt-based recommendation due to their ability to generate personalized outputs without task-specific training. However, LLM-based methods face limitations such as limited context window size, inefficient pointwise and pairwise prompting, and difficulty handling listwise ranking due to token constraints. LLMs can also be sensitive to position bias, as they may overemphasize earlier items in the prompt regardless of their true relevance. To address and investigate these issues, we propose a hybrid framework that combines a traditional recommendation model with an LLM for reranking top-k items using structured prompts. We evaluate the effects of user history reordering and instructional prompts for mitigating position bias. Experiments on MovieLens-100K show that randomizing user history improves ranking quality, but LLM-based reranking does not outperform the base model. Explicit instructions to reduce position bias are also ineffective. Our evaluations reveal limitations in LLMs' ability to model ranking context and mitigate bias. Our code is publicly available at https://github.com/aminul7506/LLMForReRanking.
>
---
#### [replaced 109] SciHorizon: Benchmarking AI-for-Science Readiness from Scientific Data to Large Language Models
- **分类: cs.LG; cs.CL; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.13503v2](http://arxiv.org/pdf/2503.13503v2)**

> **作者:** Chuan Qin; Xin Chen; Chengrui Wang; Pengmin Wu; Xi Chen; Yihang Cheng; Jingyi Zhao; Meng Xiao; Xiangchao Dong; Qingqing Long; Boya Pan; Han Wu; Chengzan Li; Yuanchun Zhou; Hui Xiong; Hengshu Zhu
>
> **摘要:** In recent years, the rapid advancement of Artificial Intelligence (AI) technologies, particularly Large Language Models (LLMs), has revolutionized the paradigm of scientific discovery, establishing AI-for-Science (AI4Science) as a dynamic and evolving field. However, there is still a lack of an effective framework for the overall assessment of AI4Science, particularly from a holistic perspective on data quality and model capability. Therefore, in this study, we propose SciHorizon, a comprehensive assessment framework designed to benchmark the readiness of AI4Science from both scientific data and LLM perspectives. First, we introduce a generalizable framework for assessing AI-ready scientific data, encompassing four key dimensions: Quality, FAIRness, Explainability, and Compliance-which are subdivided into 15 sub-dimensions. Drawing on data resource papers published between 2018 and 2023 in peer-reviewed journals, we present recommendation lists of AI-ready datasets for Earth, Life, and Materials Sciences, making a novel and original contribution to the field. Concurrently, to assess the capabilities of LLMs across multiple scientific disciplines, we establish 16 assessment dimensions based on five core indicators Knowledge, Understanding, Reasoning, Multimodality, and Values spanning Mathematics, Physics, Chemistry, Life Sciences, and Earth and Space Sciences. Using the developed benchmark datasets, we have conducted a comprehensive evaluation of over 50 representative open-source and closed source LLMs. All the results are publicly available and can be accessed online at www.scihorizon.cn/en.
>
---
#### [replaced 110] HalluCounter: Reference-free LLM Hallucination Detection in the Wild!
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04615v2](http://arxiv.org/pdf/2503.04615v2)**

> **作者:** Ashok Urlana; Gopichand Kanumolu; Charaka Vinayak Kumar; Bala Mallikarjunarao Garlapati; Rahul Mishra
>
> **备注:** 30 pages, 3 figures
>
> **摘要:** Response consistency-based, reference-free hallucination detection (RFHD) methods do not depend on internal model states, such as generation probabilities or gradients, which Grey-box models typically rely on but are inaccessible in closed-source LLMs. However, their inability to capture query-response alignment patterns often results in lower detection accuracy. Additionally, the lack of large-scale benchmark datasets spanning diverse domains remains a challenge, as most existing datasets are limited in size and scope. To this end, we propose HalluCounter, a novel reference-free hallucination detection method that utilizes both response-response and query-response consistency and alignment patterns. This enables the training of a classifier that detects hallucinations and provides a confidence score and an optimal response for user queries. Furthermore, we introduce HalluCounterEval, a benchmark dataset comprising both synthetically generated and human-curated samples across multiple domains. Our method outperforms state-of-the-art approaches by a significant margin, achieving over 90\% average confidence in hallucination detection across datasets.
>
---
#### [replaced 111] SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14667v2](http://arxiv.org/pdf/2505.14667v2)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Minsuk Kahng; Albert No
>
> **备注:** 22 pages
>
> **摘要:** Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI.
>
---
#### [replaced 112] STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.00387v2](http://arxiv.org/pdf/2411.00387v2)**

> **作者:** Jiaru Zou; Qing Wang; Pratyush Thakur; Nickvash Kani
>
> **备注:** ACL 2025; NeurIPS Math-AI 2024
>
> **摘要:** Advances in large language models (LLMs) have spurred research into enhancing their reasoning capabilities, particularly in math-rich STEM (Science, Technology, Engineering, and Mathematics) documents. While LLMs can generate equations or solve math-related queries, their ability to fully understand and interpret abstract mathematical symbols in long, math-rich documents remains limited. In this paper, we introduce STEM-PoM, a comprehensive benchmark dataset designed to evaluate LLMs' reasoning abilities on math symbols within contextual scientific text. The dataset, sourced from real-world ArXiv documents, contains over 2K math symbols classified as main attributes of variables, constants, operators, and unit descriptors, with additional sub-attributes including scalar/vector/matrix for variables and local/global/discipline-specific labels for both constants and operators. Our extensive experiments demonstrate that state-of-the-art LLMs achieve an average accuracy of 20-60% under in-context learning and 50-60% with fine-tuning, highlighting a substantial gap in their ability to classify mathematical symbols. By improving LLMs' mathematical symbol classification, STEM-PoM further enhances models' downstream mathematical reasoning capabilities. The code and data are available at https://github.com/jiaruzouu/STEM-PoM.
>
---
#### [replaced 113] Optimizing Case-Based Reasoning System for Functional Test Script Generation with Large Language Models
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.20576v3](http://arxiv.org/pdf/2503.20576v3)**

> **作者:** Siyuan Guo; Huiwu Liu; Xiaolong Chen; Yuming Xie; Liang Zhang; Tao Han; Hechang Chen; Yi Chang; Jun Wang
>
> **备注:** Accepted by KDD 2025 (ADS Track)
>
> **摘要:** In this work, we explore the potential of large language models (LLMs) for generating functional test scripts, which necessitates understanding the dynamically evolving code structure of the target software. To achieve this, we propose a case-based reasoning (CBR) system utilizing a 4R cycle (i.e., retrieve, reuse, revise, and retain), which maintains and leverages a case bank of test intent descriptions and corresponding test scripts to facilitate LLMs for test script generation. To improve user experience further, we introduce Re4, an optimization method for the CBR system, comprising reranking-based retrieval finetuning and reinforced reuse finetuning. Specifically, we first identify positive examples with high semantic and script similarity, providing reliable pseudo-labels for finetuning the retriever model without costly labeling. Then, we apply supervised finetuning, followed by a reinforcement learning finetuning stage, to align LLMs with our production scenarios, ensuring the faithful reuse of retrieved cases. Extensive experimental results on two product development units from Huawei Datacom demonstrate the superiority of the proposed CBR+Re4. Notably, we also show that the proposed Re4 method can help alleviate the repetitive generation issues with LLMs.
>
---
#### [replaced 114] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12458v2](http://arxiv.org/pdf/2410.12458v2)**

> **作者:** Minghao Wu; Thuy-Trang Vu; Lizhen Qu; Gholamreza Haffari
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** The performance of large language models (LLMs) is strongly influenced by the quality and diversity of data used during supervised fine-tuning (SFT). However, current data selection methods often prioritize one aspect over the other, resulting in suboptimal training outcomes. To address this, we formulate data selection as a set cover problem and present GraphFilter, a novel approach that balances both quality and diversity in data selection. GraphFilter models the dataset as a bipartite graph connecting sentences to their constituent n-grams, then employs a priority function that combines quality and diversity metrics multiplicatively. GraphFilter iteratively selects sentences with the highest priority, removes covered n-grams from the bipartite graph, and recomputes priorities to reflect the changing data landscape. We validate GraphFilter using three model backbones across six widely-used benchmarks, demonstrating that it outperforms nine existing baselines in both model performance and computational efficiency. Further analysis shows that our design choices lead to more effective subset selection, underscores the value of instruction diversity, and provides insights into how quality and diversity interact with different subset sizes.
>
---
#### [replaced 115] Predicting drug-gene relations via analogy tasks with word embeddings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.00984v5](http://arxiv.org/pdf/2406.00984v5)**

> **作者:** Hiroaki Yamagiwa; Ryoma Hashimoto; Kiwamu Arakane; Ken Murakami; Shou Soeda; Momose Oyama; Yihua Zhu; Mariko Okada; Hidetoshi Shimodaira
>
> **摘要:** Natural language processing (NLP) is utilized in a wide range of fields, where words in text are typically transformed into feature vectors called embeddings. BioConceptVec is a specific example of embeddings tailored for biology, trained on approximately 30 million PubMed abstracts using models such as skip-gram. Generally, word embeddings are known to solve analogy tasks through simple vector arithmetic. For example, subtracting the vector for man from that of king and then adding the vector for woman yields a point that lies closer to queen in the embedding space. In this study, we demonstrate that BioConceptVec embeddings, along with our own embeddings trained on PubMed abstracts, contain information about drug-gene relations and can predict target genes from a given drug through analogy computations. We also show that categorizing drugs and genes using biological pathways improves performance. Furthermore, we illustrate that vectors derived from known relations in the past can predict unknown future relations in datasets divided by year. Despite the simplicity of implementing analogy tasks as vector additions, our approach demonstrated performance comparable to that of large language models such as GPT-4 in predicting drug-gene relations.
>
---
#### [replaced 116] BQA: Body Language Question Answering Dataset for Video Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13206v2](http://arxiv.org/pdf/2410.13206v2)**

> **作者:** Shintaro Ozaki; Kazuki Hayashi; Miyu Oba; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted to ACL2025 (Main)
>
> **摘要:** A large part of human communication relies on nonverbal cues such as facial expressions, eye contact, and body language. Unlike language or sign language, such nonverbal communication lacks formal rules, requiring complex reasoning based on commonsense understanding. Enabling current Video Large Language Models (VideoLLMs) to accurately interpret body language is a crucial challenge, as human unconscious actions can easily cause the model to misinterpret their intent. To address this, we propose a dataset, BQA, a body language question answering dataset, to validate whether the model can correctly interpret emotions from short clips of body language comprising 26 emotion labels of videos of body language. We evaluated various VideoLLMs on BQA and revealed that understanding body language is challenging, and our analyses of the wrong answers by VideoLLMs show that certain VideoLLMs made significantly biased answers depending on the age group and ethnicity of the individuals in the video. The dataset is available.
>
---
#### [replaced 117] How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.09598v2](http://arxiv.org/pdf/2503.09598v2)**

> **作者:** Ruohao Guo; Wei Xu; Alan Ritter
>
> **摘要:** As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world interactions. We curated EchoMist, the first comprehensive benchmark for implicit misinformation, where false assumptions are embedded in the query to LLMs. EchoMist targets circulated, harmful, and ever-evolving implicit misinformation from diverse sources, including realistic human-AI conversations and social media interactions. Through extensive empirical studies on 15 state-of-the-art LLMs, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating counterfactual explanations. We also investigate two mitigation methods, i.e., Self-Alert and RAG, to enhance LLMs' capability to counter implicit misinformation. Our findings indicate that EchoMist remains a persistent challenge and underscore the critical need to safeguard against the risk of implicit misinformation.
>
---
#### [replaced 118] RASMALAI: Resources for Adaptive Speech Modeling in Indian Languages with Accents and Intonations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18609v2](http://arxiv.org/pdf/2505.18609v2)**

> **作者:** Ashwin Sankar; Yoach Lacombe; Sherry Thomas; Praveen Srinivasa Varadhan; Sanchit Gandhi; Mitesh M Khapra
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** We introduce RASMALAI, a large-scale speech dataset with rich text descriptions, designed to advance controllable and expressive text-to-speech (TTS) synthesis for 23 Indian languages and English. It comprises 13,000 hours of speech and 24 million text-description annotations with fine-grained attributes like speaker identity, accent, emotion, style, and background conditions. Using RASMALAI, we develop IndicParlerTTS, the first open-source, text-description-guided TTS for Indian languages. Systematic evaluation demonstrates its ability to generate high-quality speech for named speakers, reliably follow text descriptions and accurately synthesize specified attributes. Additionally, it effectively transfers expressive characteristics both within and across languages. IndicParlerTTS consistently achieves strong performance across these evaluations, setting a new standard for controllable multilingual expressive speech synthesis in Indian languages.
>
---
#### [replaced 119] "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.18594v2](http://arxiv.org/pdf/2409.18594v2)**

> **作者:** Ricardo Knauer; Mario Koddenbrock; Raphael Wallsberger; Nicholas M. Brisson; Georg N. Duda; Deborah Falla; David W. Evans; Erik Rodner
>
> **备注:** KDD 2025 Research Track
>
> **摘要:** Large language models (LLMs) provide powerful means to leverage prior knowledge for predictive modeling when data is limited. In this work, we demonstrate how LLMs can use their compressed world knowledge to generate intrinsically interpretable machine learning models, i.e., decision trees, without any training data. We find that these zero-shot decision trees can even surpass data-driven trees on some small-sized tabular datasets and that embeddings derived from these trees perform better than data-driven tree-based embeddings on average. Our decision tree induction and embedding approaches can therefore serve as new knowledge-driven baselines for data-driven machine learning methods in the low-data regime. Furthermore, they offer ways to harness the rich world knowledge within LLMs for tabular machine learning tasks. Our code and results are available at https://github.com/ml-lab-htw/llm-trees.
>
---
#### [replaced 120] NAP^2: A Benchmark for Naturalness and Privacy-Preserving Text Rewriting by Learning from Human
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.03749v2](http://arxiv.org/pdf/2406.03749v2)**

> **作者:** Shuo Huang; William MacLean; Xiaoxi Kang; Qiongkai Xu; Zhuang Li; Xingliang Yuan; Gholamreza Haffari; Lizhen Qu
>
> **摘要:** The widespread use of cloud-based Large Language Models (LLMs) has heightened concerns over user privacy, as sensitive information may be inadvertently exposed during interactions with these services. To protect privacy before sending sensitive data to those models, we suggest sanitizing sensitive text using two common strategies used by humans: i) deleting sensitive expressions, and ii) obscuring sensitive details by abstracting them. To explore the issues and develop a tool for text rewriting, we curate the first corpus, coined NAP^2, through both crowdsourcing and the use of large language models (LLMs). Compared to the prior works on anonymization, the human-inspired approaches result in more natural rewrites and offer an improved balance between privacy protection and data utility, as demonstrated by our extensive experiments. Researchers interested in accessing the dataset are encouraged to contact the first or corresponding author via email.
>
---
#### [replaced 121] The Power of Personality: A Human Simulation Perspective to Investigate Large Language Model Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20859v2](http://arxiv.org/pdf/2502.20859v2)**

> **作者:** Yifan Duan; Yihong Tang; Xuefeng Bai; Kehai Chen; Juntao Li; Min Zhang
>
> **摘要:** Large language models (LLMs) excel in both closed tasks (including problem-solving, and code generation) and open tasks (including creative writing), yet existing explanations for their capabilities lack connections to real-world human intelligence. To fill this gap, this paper systematically investigates LLM intelligence through the lens of ``human simulation'', addressing three core questions: (1) \textit{How do personality traits affect problem-solving in closed tasks?} (2) \textit{How do traits shape creativity in open tasks?} (3) \textit{How does single-agent performance influence multi-agent collaboration?} By assigning Big Five personality traits to LLM agents and evaluating their performance in single- and multi-agent settings, we reveal that specific traits significantly influence reasoning accuracy (closed tasks) and creative output (open tasks). Furthermore, multi-agent systems exhibit collective intelligence distinct from individual capabilities, driven by distinguishing combinations of personalities.
>
---
#### [replaced 122] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12953v2](http://arxiv.org/pdf/2502.12953v2)**

> **作者:** Andrei Jarca; Florinel Alin Croitoru; Radu Tudor Ionescu
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Masked language modeling has become a widely adopted unsupervised technique to pre-train large language models (LLMs). However, the process of selecting tokens for masking is random, and the percentage of masked tokens is typically fixed for the entire training process. In this paper, we propose to adjust the masking ratio and to decide which tokens to mask based on a novel task-informed anti-curriculum learning scheme. First, we harness task-specific knowledge about useful and harmful tokens in order to determine which tokens to mask. Second, we propose a cyclic decaying masking ratio, which corresponds to an anti-curriculum schedule (from hard to easy). We exemplify our novel task-informed anti-curriculum by masking (TIACBM) approach across three diverse downstream tasks: sentiment analysis, text classification by topic, and authorship attribution. Our findings suggest that TIACBM enhances the ability of the model to focus on key task-relevant features, contributing to statistically significant performance gains across tasks. We release our code at https://github.com/JarcaAndrei/TIACBM.
>
---
#### [replaced 123] Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18874v3](http://arxiv.org/pdf/2502.18874v3)**

> **作者:** Kaishuai Xu; Tiezheng Yu; Wenjun Hou; Yi Cheng; Liangyou Li; Xin Jiang; Lifeng Shang; Qun Liu; Wenjie Li
>
> **备注:** accepted as ACL 2025 findings
>
> **摘要:** Large Language Models (LLMs) are being used more and more extensively for automated evaluation in various scenarios. Previous studies have attempted to fine-tune open-source LLMs to replicate the evaluation explanations and judgments of powerful proprietary models, such as GPT-4. However, these methods are largely limited to text-based analyses under predefined general criteria, resulting in reduced adaptability for unseen instructions and demonstrating instability in evaluating adherence to quantitative and structural constraints. To address these limitations, we propose a novel evaluation framework, ARJudge, that adaptively formulates evaluation criteria and synthesizes both text-based and code-driven analyses to evaluate LLM responses. ARJudge consists of two components: a fine-tuned Analyzer that generates multi-faceted evaluation analyses and a tuning-free Refiner that combines and refines all analyses to make the final judgment. We construct a Composite Analysis Corpus that integrates tasks for evaluation criteria generation alongside text-based and code-driven analysis generation to train the Analyzer. Our results demonstrate that ARJudge outperforms existing fine-tuned evaluators in effectiveness and robustness. Furthermore, it demonstrates the importance of multi-faceted evaluation and code-driven analyses in enhancing evaluation capabilities.
>
---
#### [replaced 124] When Two LLMs Debate, Both Think They'll Win
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19184v2](http://arxiv.org/pdf/2505.19184v2)**

> **作者:** Pradyumna Shyama Prasad; Minh Nhat Nguyen
>
> **摘要:** Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLM outputs are deployed without careful review in assistant roles or agentic settings.
>
---
#### [replaced 125] RaDeR: Reasoning-aware Dense Retrieval Models
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.18405v2](http://arxiv.org/pdf/2505.18405v2)**

> **作者:** Debrup Das; Sam O' Nuallain; Razieh Rahimi
>
> **备注:** 26 pages
>
> **摘要:** We propose RaDeR, a set of reasoning-based dense retrieval models trained with data derived from mathematical problem solving using large language models (LLMs). Our method leverages retrieval-augmented reasoning trajectories of an LLM and self-reflective relevance evaluation, enabling the creation of both diverse and hard-negative samples for reasoning-intensive relevance. RaDeR retrievers, trained for mathematical reasoning, effectively generalize to diverse reasoning tasks in the BRIGHT and RAR-b benchmarks, consistently outperforming strong baselines in overall performance. Notably, RaDeR achieves significantly higher performance than baselines on the Math and Coding splits. In addition, RaDeR presents the first dense retriever that outperforms BM25 when queries are Chain-of-Thought reasoning steps, underscoring the critical role of reasoning-based retrieval to augment reasoning language models. Furthermore, RaDeR achieves comparable or superior performance while using only 2.5% of the training data used by the concurrent work REASONIR, highlighting the quality of our synthesized training data.
>
---
#### [replaced 126] OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17473v2](http://arxiv.org/pdf/2505.17473v2)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [replaced 127] Systematic Generalization in Language Models Scales with Information Entropy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13089v2](http://arxiv.org/pdf/2505.13089v2)**

> **作者:** Sondre Wold; Lucas Georges Gabriel Charpentier; Étienne Simon
>
> **备注:** Accepted to ACL 2025: Findings
>
> **摘要:** Systematic generalization remains challenging for current language models, which are known to be both sensitive to semantically similar permutations of the input and to struggle with known concepts presented in novel contexts. Although benchmarks exist for assessing compositional behavior, it is unclear how to measure the difficulty of a systematic generalization problem. In this work, we show how one aspect of systematic generalization can be described by the entropy of the distribution of component parts in the training data. We formalize a framework for measuring entropy in a sequence-to-sequence task and find that the performance of popular model architectures scales with the entropy. Our work connects systematic generalization to information efficiency, and our results indicate that success at high entropy can be achieved even without built-in priors, and that success at low entropy can serve as a target for assessing progress towards robust systematic generalization.
>
---
#### [replaced 128] Training a Generally Curious Agent
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17543v3](http://arxiv.org/pdf/2502.17543v3)**

> **作者:** Fahim Tajwar; Yiding Jiang; Abitha Thankaraj; Sumaita Sadia Rahman; J Zico Kolter; Jeff Schneider; Ruslan Salakhutdinov
>
> **备注:** ICML 2025. Project Website: https://paprika-llm.github.io
>
> **摘要:** Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present Paprika, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, Paprika teaches models to explore and adapt their behavior on a new task based on environment feedback in-context without more gradient updates. Experimental results show that models fine-tuned with Paprika can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decision-making problems that require interactions with the external world.
>
---
#### [replaced 129] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.03181v2](http://arxiv.org/pdf/2407.03181v2)**

> **作者:** Haritz Puerto; Tilek Chubakov; Xiaodan Zhu; Harish Tayyar Madabushi; Iryna Gurevych
>
> **备注:** ACL 2025 Main
>
> **摘要:** Requiring a large language model (LLM) to generate intermediary reasoning steps, known as Chain of Thought (CoT), has been shown to be an effective way of boosting performance. Previous approaches have focused on generating multiple independent CoTs, combining them through ensembling or other post-hoc strategies to enhance reasoning. In this work, we introduce a novel approach where LLMs are fine-tuned to generate a sequence of Diverse Chains of Thought (DCoT) within a single inference step, which is fundamentally different from prior work that primarily operate on parallel CoT generations. DCoT allows LLMs to gain the ability to perform within-inference refinement of reasoning chains without requiring external feedback. Through a rigorous set of experiments spanning a wide range of tasks that require various reasoning types, we show that fine-tuning on DCoT improves performance over the CoT baseline across model families and scales (1.3B to 70B). These improvements are particularly impactful for tasks with a large result state space, such as those involving numeric answers. Our work is also significant because both quantitative analyses and manual evaluations reveal the observed gains stem from the models' ability to refine an initial reasoning chain by generating a second, improved chain within the same inference step, demonstrating previously elusive self-improvement. Our code and data are publicly available at https://github.com/UKPLab/acl2025-diverse-cot.
>
---
#### [replaced 130] Faster and Better LLMs via Latency-Aware Test-Time Scaling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19634v2](http://arxiv.org/pdf/2505.19634v2)**

> **作者:** Zili Wang; Tianyu Zhang; Haoli Bai; Lu Hou; Xianzhi Yu; Wulong Liu; Shiming Xiang; Lei Zhu
>
> **摘要:** Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
>
---
#### [replaced 131] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18001v3](http://arxiv.org/pdf/2502.18001v3)**

> **作者:** Xinghao Chen; Zhijing Sun; Wenjin Guo; Miaoran Zhang; Yanjun Chen; Yirong Sun; Hui Su; Yijie Pan; Dietrich Klakow; Wenjie Li; Xiaoyu Shen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) excel in reasoning tasks through Chain-of-Thought (CoT) prompting. However, CoT prompting greatly increases computational demands, which has prompted growing interest in distilling CoT capabilities into Small Language Models (SLMs). This study systematically examines the factors influencing CoT distillation, including the choice of granularity, format and teacher model. Through experiments involving four teacher models and seven student models across seven mathematical and commonsense reasoning datasets, we uncover three key findings: (1) Unlike LLMs, SLMs exhibit a non-monotonic relationship with granularity, with stronger models benefiting from finer-grained reasoning and weaker models performing better with simpler CoT supervision; (2) CoT format significantly impacts LLMs but has minimal effect on SLMs, likely due to their reliance on supervised fine-tuning rather than pretraining preferences; (3) Stronger teacher models do NOT always produce better student models, as diversity and complexity in CoT supervision can outweigh accuracy alone. These findings emphasize the need to tailor CoT strategies to specific student model, offering actionable insights for optimizing CoT distillation in SLMs. The code and datasets are available at https://github.com/EIT-NLP/Distilling-CoT-Reasoning.
>
---
#### [replaced 132] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07446v2](http://arxiv.org/pdf/2411.07446v2)**

> **作者:** Cilin Yan; Jingyun Wang; Lin Zhang; Ruihui Zhao; Xiaopu Wu; Kai Xiong; Qingsong Liu; Guoliang Kang; Yangyang Kang
>
> **备注:** ACL 2025 Main
>
> **摘要:** Automatic prompt engineering aims to enhance the generation quality of large language models (LLMs). Recent works utilize feedbacks generated from erroneous cases to guide the prompt optimization. During inference, they may further retrieve several semantically-related exemplars and concatenate them to the optimized prompts to improve the performance. However, those works only utilize the feedback at the current step, ignoring historical and unseleccted feedbacks which are potentially beneficial. Moreover, the selection of exemplars only considers the general semantic relationship and may not be optimal in terms of task performance and matching with the optimized prompt. In this work, we propose an Exemplar-Guided Reflection with Memory mechanism (ERM) to realize more efficient and accurate prompt optimization. Specifically, we design an exemplar-guided reflection mechanism where the feedback generation is additionally guided by the generated exemplars. We further build two kinds of memory to fully utilize the historical feedback information and support more effective exemplar retrieval. Empirical evaluations show our method surpasses previous state-of-the-arts with less optimization steps, i.e., improving F1 score by 10.1 on LIAR dataset, and reducing half of the optimization steps on ProTeGi.
>
---
#### [replaced 133] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11051v4](http://arxiv.org/pdf/2502.11051v4)**

> **作者:** Jiahao Huo; Yibo Yan; Xu Zheng; Yuanhuiyi Lyu; Xin Zou; Zhihua Wei; Xuming Hu
>
> **备注:** Accepted as ACL 2025 Findings
>
> **摘要:** Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to reformulate the task of multimodal MU in the era of MLLMs, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we develop a novel geometry-constrained gradient ascent method MMUnlearner. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code can be found in [this URL](https://github.com/Z1zs/MMUnlearner).
>
---
#### [replaced 134] Shadow-FT: Tuning Instruct via Base
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12716v2](http://arxiv.org/pdf/2505.12716v2)**

> **作者:** Taiqiang Wu; Runming Yang; Jiayi Li; Pengfei Hu; Ngai Wong; Yujiu Yang
>
> **备注:** 19 pages, 10 tables, 6 figures
>
> **摘要:** Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{https://github.com/wutaiqiang/Shadow-FT}{Github}.
>
---
#### [replaced 135] Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.17049v2](http://arxiv.org/pdf/2505.17049v2)**

> **作者:** David Rozado
>
> **摘要:** This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning.
>
---
#### [replaced 136] Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16552v3](http://arxiv.org/pdf/2505.16552v3)**

> **作者:** Wenhui Tan; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Ruihua Song
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
>
---
#### [replaced 137] GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16514v3](http://arxiv.org/pdf/2502.16514v3)**

> **作者:** Yingjian Chen; Haoran Liu; Yinhong Liu; Jinxiang Xie; Rui Yang; Han Yuan; Yanran Fu; Peng Yuan Zhou; Qingyu Chen; James Caverlee; Irene Li
>
> **摘要:** Large language models (LLMs) are widely used, but they often generate subtle factual errors, especially in long-form text. These errors are fatal in some specialized domains such as medicine. Existing fact-checking with grounding documents methods face two main challenges: (1) they struggle to understand complex multihop relations in long documents, often overlooking subtle factual errors; (2) most specialized methods rely on pairwise comparisons, requiring multiple model calls, leading to high resource and computational costs. To address these challenges, we propose GraphCheck, a fact-checking framework that uses extracted knowledge graphs to enhance text representation. Graph Neural Networks further process these graphs as a soft prompt, enabling LLMs to incorporate structured knowledge more effectively. Enhanced with graph-based reasoning, GraphCheck captures multihop reasoning chains that are often overlooked by existing methods, enabling precise and efficient fact-checking in a single inference call. Experimental results on seven benchmarks spanning both general and medical domains demonstrate up to a 7.1% overall improvement over baseline models. Notably, GraphCheck outperforms existing specialized fact-checkers and achieves comparable performance with state-of-the-art LLMs, such as DeepSeek-V3 and OpenAI-o1, with significantly fewer parameters.
>
---
#### [replaced 138] Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.06638v4](http://arxiv.org/pdf/2410.06638v4)**

> **作者:** Kaishuai Xu; Tiezheng Yu; Wenjun Hou; Yi Cheng; Chak Tou Leong; Liangyou Li; Xin Jiang; Lifeng Shang; Qun Liu; Wenjie Li
>
> **备注:** accepted as ACL 2025 main
>
> **摘要:** Large Language Models (LLMs) have exhibited strong mathematical reasoning prowess, tackling tasks ranging from basic arithmetic to advanced competition-level problems. However, frequently occurring subtle yet critical errors, such as miscalculations or incorrect substitutions, limit the LLMs' full potential. Existing studies to improve mathematical ability typically involve applying preference learning to step-wise solution pairs. Although these methods leverage samples of varying granularity to mitigate reasoning errors, they overlook critical subtle errors. In this work, we propose a novel preference learning framework called eRror-Injected Self-Editing (RISE), which injects predefined subtle errors into pivotal tokens in reasoning or computation steps to construct hard pairs for error mitigation. In detail, RISE uses the LLM itself to edit a small number of tokens in the solution, injecting designed subtle errors. Then, pairs composed of self-edited solutions and their corresponding correct ones, along with pairs of correct and incorrect solutions obtained through sampling, are used together for subtle error-aware DPO training. Compared with other preference learning methods, RISE further refines the training objective without requiring fine-grained sampling or preference annotation. Extensive experiments validate the effectiveness of RISE, with preference learning on Qwen2-7B-Instruct yielding notable improvements of 3.0% on GSM8K and 7.9% on MATH with only 4.5K training samples. Moreover, the effect of error mitigation extends from mathematical reasoning to logical reasoning and code generation.
>
---
#### [replaced 139] ProgCo: Program Helps Self-Correction of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01264v2](http://arxiv.org/pdf/2501.01264v2)**

> **作者:** Xiaoshuai Song; Yanan Wu; Weixun Wang; Jiaheng Liu; Wenbo Su; Bo Zheng
>
> **备注:** Accpeted at ACL2025 Main
>
> **摘要:** Self-Correction aims to enable large language models (LLMs) to self-verify and self-refine their initial responses without external feedback. However, LLMs often fail to effectively self-verify and generate correct feedback, further misleading refinement and leading to the failure of self-correction, especially in complex reasoning tasks. In this paper, we propose Program-driven Self-Correction (ProgCo). First, program-driven verification (ProgVe) achieves complex verification logic and extensive validation through self-generated, self-executing verification pseudo-programs. Then, program-driven refinement (ProgRe) receives feedback from ProgVe, conducts dual reflection and refinement on both responses and verification programs to mitigate misleading of incorrect feedback in complex reasoning tasks. Experiments on three instruction-following and mathematical benchmarks indicate that ProgCo achieves effective self-correction, and can be further enhance performance when combined with real program tools. We release our code at https://github.com/songxiaoshuai/progco.
>
---
#### [replaced 140] Rethinking MUSHRA: Addressing Modern Challenges in Text-to-Speech Evaluation
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.12719v3](http://arxiv.org/pdf/2411.12719v3)**

> **作者:** Praveen Srinivasa Varadhan; Amogh Gulati; Ashwin Sankar; Srija Anand; Anirudh Gupta; Anirudh Mukherjee; Shiva Kumar Marepally; Ankur Bhatia; Saloni Jaju; Suvrat Bhooshan; Mitesh M. Khapra
>
> **备注:** Accepted in TMLR
>
> **摘要:** Despite rapid advancements in TTS models, a consistent and robust human evaluation framework is still lacking. For example, MOS tests fail to differentiate between similar models, and CMOS's pairwise comparisons are time-intensive. The MUSHRA test is a promising alternative for evaluating multiple TTS systems simultaneously, but in this work we show that its reliance on matching human reference speech unduly penalises the scores of modern TTS systems that can exceed human speech quality. More specifically, we conduct a comprehensive assessment of the MUSHRA test, focusing on its sensitivity to factors such as rater variability, listener fatigue, and reference bias. Based on our extensive evaluation involving 492 human listeners across Hindi and Tamil we identify two primary shortcomings: (i) reference-matching bias, where raters are unduly influenced by the human reference, and (ii) judgement ambiguity, arising from a lack of clear fine-grained guidelines. To address these issues, we propose two refined variants of the MUSHRA test. The first variant enables fairer ratings for synthesized samples that surpass human reference quality. The second variant reduces ambiguity, as indicated by the relatively lower variance across raters. By combining these approaches, we achieve both more reliable and more fine-grained assessments. We also release MANGO, a massive dataset of 246,000 human ratings, the first-of-its-kind collection for Indian languages, aiding in analyzing human preferences and developing automatic metrics for evaluating TTS systems.
>
---
#### [replaced 141] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05111v2](http://arxiv.org/pdf/2505.05111v2)**

> **作者:** Boyi Deng; Yu Wan; Yidan Zhang; Baosong Yang; Fuli Feng
>
> **备注:** ACL 2025 main
>
> **摘要:** The mechanisms behind multilingual capabilities in Large Language Models (LLMs) have been examined using neuron-based or internal-activation-based methods. However, these methods often face challenges such as superposition and layer-wise activation variance, which limit their reliability. Sparse Autoencoders (SAEs) offer a more nuanced analysis by decomposing the activations of LLMs into a sparse linear combination of SAE features. We introduce a novel metric to assess the monolinguality of features obtained from SAEs, discovering that some features are strongly related to specific languages. Additionally, we show that ablating these SAE features only significantly reduces abilities in one language of LLMs, leaving others almost unaffected. Interestingly, we find some languages have multiple synergistic SAE features, and ablating them together yields greater improvement than ablating individually. Moreover, we leverage these SAE-derived language-specific features to enhance steering vectors, achieving control over the language generated by LLMs. The code is publicly available at https://github.com/Aatrox103/multilingual-llm-features.
>
---
#### [replaced 142] EPIC: Efficient Position-Independent Caching for Serving Large Language Models
- **分类: cs.LG; cs.CL; cs.DC; cs.PF**

- **链接: [http://arxiv.org/pdf/2410.15332v3](http://arxiv.org/pdf/2410.15332v3)**

> **作者:** Junhao Hu; Wenrui Huang; Weidong Wang; Haoyi Wang; Tiancheng Hu; Qin Zhang; Hao Feng; Xusheng Chen; Yizhou Shan; Tao Xie
>
> **摘要:** Large Language Models (LLMs) show great capabilities in a wide range of applications, but serving them efficiently becomes increasingly challenging as requests (prompts) become more complex. Context caching improves serving performance by reusing Key-Value (KV) vectors, the intermediate representations of tokens that are repeated across requests. However, existing context caching requires exact prefix matches across requests, limiting reuse cases in settings such as few-shot learning and retrieval-augmented generation, where immutable content (e.g., documents) remains unchanged across requests but is preceded by varying prefixes. Position-Independent Caching (PIC) addresses this issue by enabling modular reuse of the KV vectors regardless of prefixes. We formalize PIC and advance prior work by introducing EPIC, a serving system incorporating our new LegoLink algorithm, which mitigates the inappropriate "attention sink" effect at every document beginning, to maintain accuracy with minimal computation. Experiments show that EPIC achieves up to 8x improvements in Time-To-First-Token (TTFT) and 7x throughput gains over existing systems, with negligible or no accuracy loss.
>
---
#### [replaced 143] Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18596v2](http://arxiv.org/pdf/2505.18596v2)**

> **作者:** Chen Han; Wenzhen Zheng; Xijin Tang
>
> **摘要:** The proliferation of misinformation in digital platforms reveals the limitations of traditional detection methods, which mostly rely on static classification and fail to capture the intricate process of real-world fact-checking. Despite advancements in Large Language Models (LLMs) that enhance automated reasoning, their application to misinformation detection remains hindered by issues of logical inconsistency and superficial verification. In response, we introduce Debate-to-Detect (D2D), a novel Multi-Agent Debate (MAD) framework that reformulates misinformation detection as a structured adversarial debate. Inspired by fact-checking workflows, D2D assigns domain-specific profiles to each agent and orchestrates a five-stage debate process, including Opening Statement, Rebuttal, Free Debate, Closing Statement, and Judgment. To transcend traditional binary classification, D2D introduces a multi-dimensional evaluation mechanism that assesses each claim across five distinct dimensions: Factuality, Source Reliability, Reasoning Quality, Clarity, and Ethics. Experiments with GPT-4o on two fakenews datasets demonstrate significant improvements over baseline methods, and the case study highlight D2D's capability to iteratively refine evidence while improving decision transparency, representing a substantial advancement towards robust and interpretable misinformation detection. The code will be open-sourced in a future release.
>
---
#### [replaced 144] Token-level Accept or Reject: A Micro Alignment Approach for Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19743v2](http://arxiv.org/pdf/2505.19743v2)**

> **作者:** Yang Zhang; Yu Yu; Bo Tang; Yu Zhu; Chuxiong Sun; Wenqiang Wei; Jie Hu; Zipeng Xie; Zhiyu Li; Feiyu Xiong; Edward Chung
>
> **备注:** Accepted to 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** With the rapid development of Large Language Models (LLMs), aligning these models with human preferences and values is critical to ensuring ethical and safe applications. However, existing alignment techniques such as RLHF or DPO often require direct fine-tuning on LLMs with billions of parameters, resulting in substantial computational costs and inefficiencies. To address this, we propose Micro token-level Accept-Reject Aligning (MARA) approach designed to operate independently of the language models. MARA simplifies the alignment process by decomposing sentence-level preference learning into token-level binary classification, where a compact three-layer fully-connected network determines whether candidate tokens are "Accepted" or "Rejected" as part of the response. Extensive experiments across seven different LLMs and three open-source datasets show that MARA achieves significant improvements in alignment performance while reducing computational costs. The source code and implementation details are publicly available at https://github.com/IAAR-Shanghai/MARA, and the trained models are released at https://huggingface.co/IAAR-Shanghai/MARA_AGENTS.
>
---
#### [replaced 145] ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.09501v3](http://arxiv.org/pdf/2503.09501v3)**

> **作者:** Ziyu Wan; Yunxiang Li; Xiaoyu Wen; Yan Song; Hanjing Wang; Linyi Yang; Mark Schmidt; Jun Wang; Weinan Zhang; Shuyue Hu; Ying Wen
>
> **摘要:** Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking -- enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Empirical results from single-turn experiments demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Additionally, we further extend ReMA to multi-turn interaction settings, leveraging turn-level ratio and parameter sharing to improve efficiency. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs. Our code can be found in https://github.com/ziyuwan/ReMA-public
>
---
#### [replaced 146] SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19641v2](http://arxiv.org/pdf/2505.19641v2)**

> **作者:** Junteng Liu; Yuanxiang Fan; Zhuo Jiang; Han Ding; Yongyi Hu; Chi Zhang; Yiqi Shi; Shitong Weng; Aili Chen; Shiqi Chen; Yunan Huang; Mozhi Zhang; Pengyu Zhao; Junjie Yan; Junxian He
>
> **摘要:** Recent advances such as OpenAI-o1 and DeepSeek R1 have demonstrated the potential of Reinforcement Learning (RL) to enhance reasoning abilities in Large Language Models (LLMs). While open-source replication efforts have primarily focused on mathematical and coding domains, methods and resources for developing general reasoning capabilities remain underexplored. This gap is partly due to the challenge of collecting diverse and verifiable reasoning data suitable for RL. We hypothesize that logical reasoning is critical for developing general reasoning capabilities, as logic forms a fundamental building block of reasoning. In this work, we present SynLogic, a data synthesis framework and dataset that generates diverse logical reasoning data at scale, encompassing 35 diverse logical reasoning tasks. The SynLogic approach enables controlled synthesis of data with adjustable difficulty and quantity. Importantly, all examples can be verified by simple rules, making them ideally suited for RL with verifiable rewards. In our experiments, we validate the effectiveness of RL training on the SynLogic dataset based on 7B and 32B models. SynLogic leads to state-of-the-art logical reasoning performance among open-source datasets, surpassing DeepSeek-R1-Distill-Qwen-32B by 6 points on BBEH. Furthermore, mixing SynLogic data with mathematical and coding tasks improves the training efficiency of these domains and significantly enhances reasoning generalization. Notably, our mixed training model outperforms DeepSeek-R1-Zero-Qwen-32B across multiple benchmarks. These findings position SynLogic as a valuable resource for advancing the broader reasoning capabilities of LLMs. We open-source both the data synthesis pipeline and the SynLogic dataset at https://github.com/MiniMax-AI/SynLogic.
>
---
#### [replaced 147] SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12134v2](http://arxiv.org/pdf/2502.12134v2)**

> **作者:** Yige Xu; Xu Guo; Zhiwei Zeng; Chunyan Miao
>
> **备注:** Camera-ready for ACL 2025 (main conference)
>
> **摘要:** Chain-of-Thought (CoT) reasoning enables Large Language Models (LLMs) to solve complex reasoning tasks by generating intermediate reasoning steps. However, most existing approaches focus on hard token decoding, which constrains reasoning within the discrete vocabulary space and may not always be optimal. While recent efforts explore continuous-space reasoning, they often require full-model fine-tuning and suffer from catastrophic forgetting, limiting their applicability to state-of-the-art LLMs that already perform well in zero-shot settings with a proper instruction. To address this challenge, we propose a novel approach for continuous-space reasoning that does not require modifying the LLM. Specifically, we employ a lightweight fixed assistant model to speculatively generate instance-specific soft thought tokens as the initial chain of thoughts, which are then mapped into the LLM's representation space via a trainable projection module. Experimental results on five reasoning benchmarks demonstrate that our method enhances LLM reasoning performance through supervised, parameter-efficient fine-tuning. Source code is available at https://github.com/xuyige/SoftCoT.
>
---
#### [replaced 148] How Private are Language Models in Abstractive Summarization?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12040v2](http://arxiv.org/pdf/2412.12040v2)**

> **作者:** Anthony Hughes; Ning Ma; Nikolaos Aletras
>
> **摘要:** In sensitive domains such as medical and legal, protecting sensitive information is critical, with protective laws strictly prohibiting the disclosure of personal data. This poses challenges for sharing valuable data such as medical reports and legal cases summaries. While language models (LMs) have shown strong performance in text summarization, it is still an open question to what extent they can provide privacy-preserving summaries from non-private source documents. In this paper, we perform a comprehensive study of privacy risks in LM-based summarization across two closed- and four open-weight models of different sizes and families. We experiment with both prompting and fine-tuning strategies for privacy-preservation across a range of summarization datasets including medical and legal domains. Our quantitative and qualitative analysis, including human evaluation, shows that LMs frequently leak personally identifiable information in their summaries, in contrast to human-generated privacy-preserving summaries, which demonstrate significantly higher privacy protection levels. These findings highlight a substantial gap between current LM capabilities and expert human expert performance in privacy-sensitive summarization tasks.
>
---
#### [replaced 149] Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00675v2](http://arxiv.org/pdf/2505.00675v2)**

> **作者:** Yiming Du; Wenyu Huang; Danna Zheng; Zhaowei Wang; Sebastien Montella; Mirella Lapata; Kam-Fai Wong; Jeff Z. Pan
>
> **摘要:** Memory is a fundamental component of AI systems, underpinning large language models (LLMs)-based agents. While prior surveys have focused on memory applications with LLMs (e.g., enabling personalized memory in conversational agents), they often overlook the atomic operations that underlie memory dynamics. In this survey, we first categorize memory representations into parametric and contextual forms, and then introduce six fundamental memory operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Compression. We map these operations to the most relevant research topics across long-term, long-context, parametric modification, and multi-source memory. By reframing memory systems through the lens of atomic operations and representation types, this survey provides a structured and dynamic perspective on research, benchmark datasets, and tools related to memory in AI, clarifying the functional interplay in LLMs based agents while outlining promising directions for future research\footnote{The paper list, datasets, methods and tools are available at \href{https://github.com/Elvin-Yiming-Du/Survey_Memory_in_AI}{https://github.com/Elvin-Yiming-Du/Survey\_Memory\_in\_AI}.}.
>
---
#### [replaced 150] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01651v3](http://arxiv.org/pdf/2410.01651v3)**

> **作者:** Xiang Hu; Zhihao Teng; Jun Zhao; Wei Wu; Kewei Tu
>
> **备注:** accepted to ICML 2025
>
> **摘要:** Despite the success of Transformers, handling long contexts remains challenging due to the limited length generalization and quadratic complexity of self-attention. Thus Transformers often require post-training with a larger attention window, significantly increasing computational and memory costs. In this paper, we propose a novel attention mechanism based on dynamic context, Grouped Cross Attention (GCA), which can generalize to 1000 times the pre-training context length while maintaining the ability to access distant information with a constant attention window size. For a given input sequence, we split it into chunks and use each chunk to retrieve top-k relevant past chunks for subsequent text generation. Specifically, unlike most previous works that use an off-the-shelf retriever, our key innovation allows the retriever to learn how to retrieve past chunks that better minimize the auto-regressive loss of subsequent tokens in an end-to-end manner. Such a mechanism accommodates retrieved chunks with a fixed-size attention window to achieve long-range information access, significantly reducing computational and memory costs during training and inference. Experiments show that GCA-based models achieve near-perfect accuracy in passkey retrieval for 16M context lengths, which is 1000 times the training length.
>
---
#### [replaced 151] Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.04068v4](http://arxiv.org/pdf/2402.04068v4)**

> **作者:** Ravi Patel; Angus Brayne; Rogier Hintzen; Daniel Jaroslawicz; Georgiana Neculae; Dane Corneil
>
> **备注:** Accepted at ACL 2025 (The 63rd Annual Meeting of the Association for Computational Linguistics)
>
> **摘要:** Language models hold incredible promise for enabling scientific discovery by synthesizing massive research corpora. Many complex scientific research questions have multiple plausible answers, each supported by evidence of varying strength. However, existing language models lack the capability to quantitatively and faithfully compare answer plausibility in terms of supporting evidence. To address this, we introduce Retrieve to Explain (R2E), a retrieval-based model that scores and ranks all possible answers to a research question based on evidence retrieved from a document corpus. The architecture represents each answer only in terms of its supporting evidence, with the answer itself masked. This allows us to extend feature attribution methods such as Shapley values, to transparently attribute answer scores to supporting evidence at inference time. The architecture also allows incorporation of new evidence without retraining, including non-textual data modalities templated into natural language. We developed R2E for the challenging scientific discovery task of drug target identification, a human-in-the-loop process where failures are extremely costly and explainability paramount. When predicting whether drug targets will subsequently be confirmed as efficacious in clinical trials, R2E not only matches non-explainable literature-based models but also surpasses a genetics-based target identification approach used throughout the pharmaceutical industry.
>
---
#### [replaced 152] How to Upscale Neural Networks with Scaling Law? A Survey and Practical Guidelines
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12051v3](http://arxiv.org/pdf/2502.12051v3)**

> **作者:** Ayan Sengupta; Yash Goel; Tanmoy Chakraborty
>
> **备注:** 21 pages, 11 tables, 4 figures
>
> **摘要:** Neural scaling laws have revolutionized the design and optimization of large-scale AI models by revealing predictable relationships between model size, dataset volume, and computational resources. Early research established power-law relationships in model performance, leading to compute-optimal scaling strategies. However, recent studies highlighted their limitations across architectures, modalities, and deployment contexts. Sparse models, mixture-of-experts, retrieval-augmented learning, and multimodal models often deviate from traditional scaling patterns. Moreover, scaling behaviors vary across domains such as vision, reinforcement learning, and fine-tuning, underscoring the need for more nuanced approaches. In this survey, we synthesize insights from over 50 studies, examining the theoretical foundations, empirical findings, and practical implications of scaling laws. We also explore key challenges, including data efficiency, inference scaling, and architecture-specific constraints, advocating for adaptive scaling strategies tailored to real-world applications. We suggest that while scaling laws provide a useful guide, they do not always generalize across all architectures and training strategies.
>
---
#### [replaced 153] RvLLM: LLM Runtime Verification with Domain Knowledge
- **分类: cs.AI; cs.CL; cs.LO**

- **链接: [http://arxiv.org/pdf/2505.18585v2](http://arxiv.org/pdf/2505.18585v2)**

> **作者:** Yedi Zhang; Sun Yi Emma; Annabelle Lee Jia En; Jin Song Dong
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Large language models (LLMs) have emerged as a dominant AI paradigm due to their exceptional text understanding and generation capabilities. However, their tendency to generate inconsistent or erroneous outputs challenges their reliability, especially in high-stakes domains requiring accuracy and trustworthiness. Existing research primarily focuses on detecting and mitigating model misbehavior in general-purpose scenarios, often overlooking the potential of integrating domain-specific knowledge. In this work, we advance misbehavior detection by incorporating domain knowledge. The core idea is to design a general specification language that enables domain experts to customize domain-specific predicates in a lightweight and intuitive manner, supporting later runtime verification of LLM outputs. To achieve this, we design a novel specification language, ESL, and introduce a runtime verification framework, RvLLM, to validate LLM output against domain-specific constraints defined in ESL. We evaluate RvLLM on three representative tasks: violation detection against Singapore Rapid Transit Systems Act, numerical comparison, and inequality solving. Experimental results demonstrate that RvLLM effectively detects erroneous outputs across various LLMs in a lightweight and flexible manner. The results reveal that despite their impressive capabilities, LLMs remain prone to low-level errors due to limited interpretability and a lack of formal guarantees during inference, and our framework offers a potential long-term solution by leveraging expert domain knowledge to rigorously and efficiently verify LLM outputs.
>
---
#### [replaced 154] QwenLong-CPRS: Towards $\infty$-LLMs with Dynamic Context Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18092v2](http://arxiv.org/pdf/2505.18092v2)**

> **作者:** Weizhou Shen; Chenliang Li; Fanqi Wan; Shengyi Liao; Shaopeng Lai; Bo Zhang; Yingcheng Shi; Yuning Wu; Gang Fu; Zhansheng Li; Bin Yang; Ji Zhang; Fei Huang; Jingren Zhou; Ming Yan
>
> **摘要:** This technical report presents QwenLong-CPRS, a context compression framework designed for explicit long-context optimization, addressing prohibitive computation overhead during the prefill stage and the "lost in the middle" performance degradation of large language models (LLMs) during long sequence processing. Implemented through a novel dynamic context optimization mechanism, QwenLong-CPRS enables multi-granularity context compression guided by natural language instructions, achieving both efficiency gains and improved performance. Evolved from the Qwen architecture series, QwenLong-CPRS introduces four key innovations: (1) Natural language-guided dynamic optimization, (2) Bidirectional reasoning layers for enhanced boundary awareness, (3) Token critic mechanisms with language modeling heads, and (4) Window-parallel inference. Comprehensive evaluations across five benchmarks (4K-2M word contexts) demonstrate QwenLong-CPRS's threefold effectiveness: (1) Consistent superiority over other context management methods like RAG and sparse attention in both accuracy and efficiency. (2) Architecture-agnostic integration with all flagship LLMs, including GPT-4o, Gemini2.0-pro, Claude3.7-sonnet, DeepSeek-v3, and Qwen2.5-max, achieves 21.59$\times$ context compression alongside 19.15-point average performance gains; (3) Deployed with Qwen2.5-32B-Instruct, QwenLong-CPRS surpasses leading proprietary LLMs by 4.85 and 10.88 points on Ruler-128K and InfiniteBench, establishing new SOTA performance.
>
---
#### [replaced 155] VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.04962v4](http://arxiv.org/pdf/2501.04962v4)**

> **作者:** Wenqian Cui; Xiaoqi Jiao; Ziqiao Meng; Irwin King
>
> **备注:** The Version of Record of this contribution is accepted to ACL 2025 main conference
>
> **摘要:** With the rising need for speech-based interaction models, end-to-end Spoken Language Models (SLMs) have emerged as a promising solution. While these models require comprehensive world knowledge for meaningful and reliable human interactions, existing question-answering (QA) benchmarks fall short in evaluating SLMs' knowledge understanding due to their inability to support end-to-end speech evaluation and account for varied input audio conditions. To address these limitations, we present VoxEval, a novel SpeechQA benchmark that assesses SLMs' knowledge understanding through pure speech interactions. Our benchmark 1) uniquely maintains speech format for both inputs and outputs, 2) evaluates model robustness across diverse input audio conditions, and 3) pioneers the assessment of complex tasks like mathematical reasoning in spoken format. Systematic evaluation demonstrates that VoxEval presents significant challenges to current SLMs, revealing their sensitivity to varying audio conditions and highlighting the need to enhance reasoning capabilities in future development. We hope this benchmark could guide the advancement of more sophisticated and reliable SLMs. VoxEval dataset is available at: https://github.com/dreamtheater123/VoxEval
>
---

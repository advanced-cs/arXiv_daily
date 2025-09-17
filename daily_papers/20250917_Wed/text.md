# 自然语言处理 cs.CL

- **最新发布 69 篇**

- **更新 63 篇**

## 最新发布

#### [new 001] SitLLM: Large Language Models for Sitting Posture Health Understanding via Pressure Sensor Data
- **分类: cs.CL**

- **简介: 该论文提出SitLLM，一种结合压力传感器与大语言模型的轻量级多模态框架，用于细粒度坐姿健康分析与个性化反馈。旨在解决现有系统识别粗糙、语义表达不足的问题，通过多模块设计提升坐姿理解与健康响应能力。**

- **链接: [http://arxiv.org/pdf/2509.12994v1](http://arxiv.org/pdf/2509.12994v1)**

> **作者:** Jian Gao; Fufangchen Zhao; Yiyang Zhang; Danfeng Yan
>
> **摘要:** Poor sitting posture is a critical yet often overlooked factor contributing to long-term musculoskeletal disorders and physiological dysfunctions. Existing sitting posture monitoring systems, although leveraging visual, IMU, or pressure-based modalities, often suffer from coarse-grained recognition and lack the semantic expressiveness necessary for personalized feedback. In this paper, we propose \textbf{SitLLM}, a lightweight multimodal framework that integrates flexible pressure sensing with large language models (LLMs) to enable fine-grained posture understanding and personalized health-oriented response generation. SitLLM comprises three key components: (1) a \textit{Gaussian-Robust Sensor Embedding Module} that partitions pressure maps into spatial patches and injects local noise perturbations for robust feature extraction; (2) a \textit{Prompt-Driven Cross-Modal Alignment Module} that reprograms sensor embeddings into the LLM's semantic space via multi-head cross-attention using the pre-trained vocabulary embeddings; and (3) a \textit{Multi-Context Prompt Module} that fuses feature-level, structure-level, statistical-level, and semantic-level contextual information to guide instruction comprehension.
>
---
#### [new 002] PAC: Pronunciation-Aware Contextualized Large Language Model-based Automatic Speech Recognition
- **分类: cs.CL; eess.AS**

- **简介: 论文提出PAC框架，用于改进基于大语言模型的语音识别系统。旨在解决发音建模与同音词区分问题，采用两阶段学习方法，显著降低词错误率，提升长尾词识别效果。属于自动语音识别任务。**

- **链接: [http://arxiv.org/pdf/2509.12647v1](http://arxiv.org/pdf/2509.12647v1)**

> **作者:** Li Fu; Yu Xin; Sunlu Zeng; Lu Fan; Youzheng Wu; Xiaodong He
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** This paper presents a Pronunciation-Aware Contextualized (PAC) framework to address two key challenges in Large Language Model (LLM)-based Automatic Speech Recognition (ASR) systems: effective pronunciation modeling and robust homophone discrimination. Both are essential for raw or long-tail word recognition. The proposed approach adopts a two-stage learning paradigm. First, we introduce a pronunciation-guided context learning method. It employs an interleaved grapheme-phoneme context modeling strategy that incorporates grapheme-only distractors, encouraging the model to leverage phonemic cues for accurate recognition. Then, we propose a pronunciation-discriminative reinforcement learning method with perturbed label sampling to further enhance the model\'s ability to distinguish contextualized homophones. Experimental results on the public English Librispeech and Mandarin AISHELL-1 datasets indicate that PAC: (1) reduces relative Word Error Rate (WER) by 30.2% and 53.8% compared to pre-trained LLM-based ASR models, and (2) achieves 31.8% and 60.5% relative reductions in biased WER for long-tail words compared to strong baselines, respectively.
>
---
#### [new 003] WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research
- **分类: cs.CL**

- **简介: 该论文提出WebWeaver框架，解决开放性深度研究中信息整合与长文本生成问题。通过动态规划与分步写作机制，有效提升报告质量与结构，实现新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.13312v1](http://arxiv.org/pdf/2509.13312v1)**

> **作者:** Zijian Li; Xin Guan; Bo Zhang; Shen Huang; Houquan Zhou; Shaopeng Lai; Ming Yan; Yong Jiang; Pengjun Xie; Fei Huang; Jun Zhang; Jingren Zhou
>
> **备注:** An agent system for open-ended deep research
>
> **摘要:** This paper tackles open-ended deep research (OEDR), a complex challenge where AI agents must synthesize vast web-scale information into insightful reports. Current approaches are plagued by dual-fold limitations: static research pipelines that decouple planning from evidence acquisition and one-shot generation paradigms that easily suffer from long-context failure issues like "loss in the middle" and hallucinations. To address these challenges, we introduce WebWeaver, a novel dual-agent framework that emulates the human research process. The planner operates in a dynamic cycle, iteratively interleaving evidence acquisition with outline optimization to produce a comprehensive, source-grounded outline linking to a memory bank of evidence. The writer then executes a hierarchical retrieval and writing process, composing the report section by section. By performing targeted retrieval of only the necessary evidence from the memory bank for each part, it effectively mitigates long-context issues. Our framework establishes a new state-of-the-art across major OEDR benchmarks, including DeepResearch Bench, DeepConsult, and DeepResearchGym. These results validate our human-centric, iterative methodology, demonstrating that adaptive planning and focused synthesis are crucial for producing high-quality, reliable, and well-structured reports.
>
---
#### [new 004] The Few-shot Dilemma: Over-prompting Large Language Models
- **分类: cs.CL**

- **简介: 论文研究大语言模型在少量样本学习中的“过度提示”问题，探讨过多示例对模型性能的负面影响。通过对比多种提示方法，发现适量示例更优，并在软件需求分类任务中取得优于现有方法1%的性能提升。属于自然语言处理中的少样本学习任务。**

- **链接: [http://arxiv.org/pdf/2509.13196v1](http://arxiv.org/pdf/2509.13196v1)**

> **作者:** Yongjian Tang; Doruk Tuncel; Christian Koerner; Thomas Runkler
>
> **备注:** accepted for the main track of FLLM
>
> **摘要:** Over-prompting, a phenomenon where excessive examples in prompts lead to diminished performance in Large Language Models (LLMs), challenges the conventional wisdom about in-context few-shot learning. To investigate this few-shot dilemma, we outline a prompting framework that leverages three standard few-shot selection methods - random sampling, semantic embedding, and TF-IDF vectors - and evaluate these methods across multiple LLMs, including GPT-4o, GPT-3.5-turbo, DeepSeek-V3, Gemma-3, LLaMA-3.1, LLaMA-3.2, and Mistral. Our experimental results reveal that incorporating excessive domain-specific examples into prompts can paradoxically degrade performance in certain LLMs, which contradicts the prior empirical conclusion that more relevant few-shot examples universally benefit LLMs. Given the trend of LLM-assisted software engineering and requirement analysis, we experiment with two real-world software requirement classification datasets. By gradually increasing the number of TF-IDF-selected and stratified few-shot examples, we identify their optimal quantity for each LLM. This combined approach achieves superior performance with fewer examples, avoiding the over-prompting problem, thus surpassing the state-of-the-art by 1% in classifying functional and non-functional requirements.
>
---
#### [new 005] MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 论文提出MORABLES基准，用于评估LLMs的抽象道德推理能力。通过寓言和短篇故事中的多选题，测试模型是否能超越浅层回答进行深层道德推断，并发现大模型虽表现更好，但仍易受对抗样本影响，依赖表面模式而非真实推理。**

- **链接: [http://arxiv.org/pdf/2509.12371v1](http://arxiv.org/pdf/2509.12371v1)**

> **作者:** Matteo Marcuzzo; Alessandro Zangari; Andrea Albarelli; Jose Camacho-Collados; Mohammad Taher Pilehvar
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** As LLMs excel on standard reading comprehension benchmarks, attention is shifting toward evaluating their capacity for complex abstract reasoning and inference. Literature-based benchmarks, with their rich narrative and moral depth, provide a compelling framework for evaluating such deeper comprehension skills. Here, we present MORABLES, a human-verified benchmark built from fables and short stories drawn from historical literature. The main task is structured as multiple-choice questions targeting moral inference, with carefully crafted distractors that challenge models to go beyond shallow, extractive question answering. To further stress-test model robustness, we introduce adversarial variants designed to surface LLM vulnerabilities and shortcuts due to issues such as data contamination. Our findings show that, while larger models outperform smaller ones, they remain susceptible to adversarial manipulation and often rely on superficial patterns rather than true moral reasoning. This brittleness results in significant self-contradiction, with the best models refuting their own answers in roughly 20% of cases depending on the framing of the moral choice. Interestingly, reasoning-enhanced models fail to bridge this gap, suggesting that scale - not reasoning ability - is the primary driver of performance.
>
---
#### [new 006] Case-Based Decision-Theoretic Decoding with Quality Memories
- **分类: cs.CL**

- **简介: 论文提出基于案例的决策理论解码（CBDT），用于文本生成任务，解决MBR解码在领域外数据表现不佳的问题。CBDT利用领域数据样例估计期望效用，结合MBR与CBDT可提升翻译和图像字幕生成质量。**

- **链接: [http://arxiv.org/pdf/2509.12677v1](http://arxiv.org/pdf/2509.12677v1)**

> **作者:** Hiroyuki Deguchi; Masaaki Nagata
>
> **备注:** Accepted at EMNLP2025 main
>
> **摘要:** Minimum Bayes risk (MBR) decoding is a decision rule of text generation, which selects the hypothesis that maximizes the expected utility and robustly generates higher-quality texts than maximum a posteriori (MAP) decoding. However, it depends on sample texts drawn from the text generation model; thus, it is difficult to find a hypothesis that correctly captures the knowledge or information of out-of-domain. To tackle this issue, we propose case-based decision-theoretic (CBDT) decoding, another method to estimate the expected utility using examples of domain data. CBDT decoding not only generates higher-quality texts than MAP decoding, but also the combination of MBR and CBDT decoding outperformed MBR decoding in seven domain De--En and Ja$\leftrightarrow$En translation tasks and image captioning tasks on MSCOCO and nocaps datasets.
>
---
#### [new 007] Do Natural Language Descriptions of Model Activations Convey Privileged Information?
- **分类: cs.CL; cs.LG**

- **简介: 该论文评估自然语言描述模型激活是否揭示模型内部信息。研究发现，现有方法依赖于生成描述的LLM参数知识，而非目标模型激活。论文指出需设计更严格的基准与实验以验证此类方法的有效性。属于模型可解释性任务。**

- **链接: [http://arxiv.org/pdf/2509.13316v1](http://arxiv.org/pdf/2509.13316v1)**

> **作者:** Millicent Li; Alberto Mario Ceballos Arroyo; Giordano Rogers; Naomi Saphra; Byron C. Wallace
>
> **备注:** 34 pages, 6 figures
>
> **摘要:** Recent interpretability methods have proposed to translate LLM internal representations into natural language descriptions using a second verbalizer LLM. This is intended to illuminate how the target model represents and operates on inputs. But do such activation verbalization approaches actually provide privileged knowledge about the internal workings of the target model, or do they merely convey information about its inputs? We critically evaluate popular verbalization methods across datasets used in prior work and find that they succeed at benchmarks without any access to target model internals, suggesting that these datasets are not ideal for evaluating verbalization methods. We then run controlled experiments which reveal that verbalizations often reflect the parametric knowledge of the verbalizer LLM which generated them, rather than the activations of the target LLM being decoded. Taken together, our results indicate a need for targeted benchmarks and experimental controls to rigorously assess whether verbalization methods provide meaningful insights into the operations of LLMs.
>
---
#### [new 008] LLM Hallucination Detection: A Fast Fourier Transform Method Based on Hidden Layer Temporal Signals
- **分类: cs.CL**

- **简介: 该论文提出HSAD方法，通过FFT分析隐藏层时序信号检测大语言模型幻觉。属于幻觉检测任务，解决现有方法依赖外部知识或静态分析的问题，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2509.13154v1](http://arxiv.org/pdf/2509.13154v1)**

> **作者:** Jinxin Li; Gang Tu; ShengYu Cheng; Junjie Hu; Jinting Wang; Rui Chen; Zhilong Zhou; Dongbo Shan
>
> **摘要:** Hallucination remains a critical barrier for deploying large language models (LLMs) in reliability-sensitive applications. Existing detection methods largely fall into two categories: factuality checking, which is fundamentally constrained by external knowledge coverage, and static hidden-state analysis, that fails to capture deviations in reasoning dynamics. As a result, their effectiveness and robustness remain limited. We propose HSAD (Hidden Signal Analysis-based Detection), a novel hallucination detection framework that models the temporal dynamics of hidden representations during autoregressive generation. HSAD constructs hidden-layer signals by sampling activations across layers, applies Fast Fourier Transform (FFT) to obtain frequency-domain representations, and extracts the strongest non-DC frequency component as spectral features. Furthermore, by leveraging the autoregressive nature of LLMs, HSAD identifies optimal observation points for effective and reliable detection. Across multiple benchmarks, including TruthfulQA, HSAD achieves over 10 percentage points improvement compared to prior state-of-the-art methods. By integrating reasoning-process modeling with frequency-domain analysis, HSAD establishes a new paradigm for robust hallucination detection in LLMs.
>
---
#### [new 009] ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization
- **分类: cs.CL**

- **简介: 该论文提出ReSum框架，解决LLM基于网络代理在长周期任务中因上下文窗口限制导致的性能瓶颈。通过周期性上下文摘要，实现无限探索。实验表明，ReSum显著提升任务完成率，尤其在少量样本下表现优异，属于网络搜索智能任务。**

- **链接: [http://arxiv.org/pdf/2509.13313v1](http://arxiv.org/pdf/2509.13313v1)**

> **作者:** Xixi Wu; Kuan Li; Yida Zhao; Liwen Zhang; Litu Ou; Huifeng Yin; Zhongwang Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Minhao Cheng; Shuai Wang; Hong Cheng; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Large Language Model (LLM)-based web agents demonstrate strong performance on knowledge-intensive tasks but are hindered by context window limitations in paradigms like ReAct. Complex queries involving multiple entities, intertwined relationships, and high uncertainty demand extensive search cycles that rapidly exhaust context budgets before reaching complete solutions. To overcome this challenge, we introduce ReSum, a novel paradigm that enables indefinite exploration through periodic context summarization. ReSum converts growing interaction histories into compact reasoning states, maintaining awareness of prior discoveries while bypassing context constraints. For paradigm adaptation, we propose ReSum-GRPO, integrating GRPO with segmented trajectory training and advantage broadcasting to familiarize agents with summary-conditioned reasoning. Extensive experiments on web agents of varying scales across three benchmarks demonstrate that ReSum delivers an average absolute improvement of 4.5\% over ReAct, with further gains of up to 8.2\% following ReSum-GRPO training. Notably, with only 1K training samples, our WebResummer-30B (a ReSum-GRPO-trained version of WebSailor-30B) achieves 33.3\% Pass@1 on BrowseComp-zh and 18.3\% on BrowseComp-en, surpassing existing open-source web agents.
>
---
#### [new 010] MORQA: Benchmarking Evaluation Metrics for Medical Open-Ended Question Answering
- **分类: cs.CL; 68T50 (Primary) 68T45 (Secondary); I.2.7; I.2.10**

- **简介: 该论文提出MORQA，用于评估医疗开放问答的NLG指标。针对传统指标不足，构建多语言数据集，对比传统与LLM评估方法，发现LLM更符合专家判断，推动医疗领域生成式模型的评价研究。**

- **链接: [http://arxiv.org/pdf/2509.12405v1](http://arxiv.org/pdf/2509.12405v1)**

> **作者:** Wen-wai Yim; Asma Ben Abacha; Zixuan Yu; Robert Doerning; Fei Xia; Meliha Yetisgen
>
> **备注:** 9 pages, 8 tables
>
> **摘要:** Evaluating natural language generation (NLG) systems in the medical domain presents unique challenges due to the critical demands for accuracy, relevance, and domain-specific expertise. Traditional automatic evaluation metrics, such as BLEU, ROUGE, and BERTScore, often fall short in distinguishing between high-quality outputs, especially given the open-ended nature of medical question answering (QA) tasks where multiple valid responses may exist. In this work, we introduce MORQA (Medical Open-Response QA), a new multilingual benchmark designed to assess the effectiveness of NLG evaluation metrics across three medical visual and text-based QA datasets in English and Chinese. Unlike prior resources, our datasets feature 2-4+ gold-standard answers authored by medical professionals, along with expert human ratings for three English and Chinese subsets. We benchmark both traditional metrics and large language model (LLM)-based evaluators, such as GPT-4 and Gemini, finding that LLM-based approaches significantly outperform traditional metrics in correlating with expert judgments. We further analyze factors driving this improvement, including LLMs' sensitivity to semantic nuances and robustness to variability among reference answers. Our results provide the first comprehensive, multilingual qualitative study of NLG evaluation in the medical domain, highlighting the need for human-aligned evaluation methods. All datasets and annotations will be publicly released to support future research.
>
---
#### [new 011] WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents
- **分类: cs.CL**

- **简介: 论文提出WebResearcher框架，解决长时程智能体在自主知识发现与合成中的上下文限制与噪声干扰问题。通过迭代研究范式和数据合成引擎，提升工具使用能力和多智能体协作效率，实现前沿性能。**

- **链接: [http://arxiv.org/pdf/2509.13309v1](http://arxiv.org/pdf/2509.13309v1)**

> **作者:** Zile Qiao; Guoxin Chen; Xuanzhong Chen; Donglei Yu; Wenbiao Yin; Xinyu Wang; Zhen Zhang; Baixuan Li; Huifeng Yin; Kuan Li; Rui Min; Minpeng Liao; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Recent advances in deep-research systems have demonstrated the potential for AI agents to autonomously discover and synthesize knowledge from external sources. In this paper, we introduce WebResearcher, a novel framework for building such agents through two key components: (1) WebResearcher, an iterative deep-research paradigm that reformulates deep research as a Markov Decision Process, where agents periodically consolidate findings into evolving reports while maintaining focused workspaces, overcoming the context suffocation and noise contamination that plague existing mono-contextual approaches; and (2) WebFrontier, a scalable data synthesis engine that generates high-quality training data through tool-augmented complexity escalation, enabling systematic creation of research tasks that bridge the gap between passive knowledge recall and active knowledge construction. Notably, we find that the training data from our paradigm significantly enhances tool-use capabilities even for traditional mono-contextual methods. Furthermore, our paradigm naturally scales through parallel thinking, enabling concurrent multi-agent exploration for more comprehensive conclusions. Extensive experiments across 6 challenging benchmarks demonstrate that WebResearcher achieves state-of-the-art performance, even surpassing frontier proprietary systems.
>
---
#### [new 012] Mitigating Strategy Preference Bias in Emotional Support Conversation via Uncertainty Estimations
- **分类: cs.CL**

- **简介: 论文研究情感支持对话中的策略偏好偏差问题，提出基于不确定性的强化学习方法，通过双奖励函数优化策略规划，提升LLM在情感支持任务中的效果。属于对话系统优化任务。**

- **链接: [http://arxiv.org/pdf/2509.12661v1](http://arxiv.org/pdf/2509.12661v1)**

> **作者:** Yougen Zhou; Qin Chen; Ningning Zhou; Jie Zhou; Xingjiao Wu; Liang He
>
> **摘要:** Emotional support conversation (ESC) aims to alleviate distress through empathetic dialogue, yet large language models (LLMs) face persistent challenges in delivering effective ESC due to low accuracy in strategy planning. Moreover, there is a considerable preference bias towards specific strategies. Prior methods using fine-tuned strategy planners have shown potential in reducing such bias, while the underlying causes of the preference bias in LLMs have not well been studied. To address these issues, we first reveal the fundamental causes of the bias by identifying the knowledge boundaries of LLMs in strategy planning. Then, we propose an approach to mitigate the bias by reinforcement learning with a dual reward function, which optimizes strategy planning via both accuracy and entropy-based confidence for each region according to the knowledge boundaries. Experiments on the ESCov and ExTES datasets with multiple LLM backbones show that our approach outperforms the baselines, confirming the effectiveness of our approach.
>
---
#### [new 013] Towards General Agentic Intelligence via Environment Scaling
- **分类: cs.CL**

- **简介: 该论文旨在提升通用智能体能力，通过扩展训练环境增强模型调用函数的多样性与鲁棒性。提出自动构建异构环境框架及两阶段微调策略，显著提升模型在多个基准测试中的表现。属于人工智能领域智能体训练任务。**

- **链接: [http://arxiv.org/pdf/2509.13311v1](http://arxiv.org/pdf/2509.13311v1)**

> **作者:** Runnan Fang; Shihao Cai; Baixuan Li; Jialong Wu; Guangyu Li; Wenbiao Yin; Xinyu Wang; Xiaobin Wang; Liangcai Su; Zhen Zhang; Shibin Wu; Zhengwei Tao; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Advanced agentic intelligence is a prerequisite for deploying Large Language Models in practical, real-world applications. Diverse real-world APIs demand precise, robust function-calling intelligence, which needs agents to develop these capabilities through interaction in varied environments. The breadth of function-calling competence is closely tied to the diversity of environments in which agents are trained. In this work, we scale up environments as a step towards advancing general agentic intelligence. This gives rise to two central challenges: (i) how to scale environments in a principled manner, and (ii) how to effectively train agentic capabilities from experiences derived through interactions with these environments. To address these, we design a scalable framework that automatically constructs heterogeneous environments that are fully simulated, systematically broadening the space of function-calling scenarios. We further adapt a two-phase agent fine-tuning strategy: first endowing agents with fundamental agentic capabilities, then specializing them for domain-specific contexts. Extensive experiments on agentic benchmarks, tau-bench, tau2-Bench, and ACEBench, demonstrate that our trained model, AgentScaler, significantly enhances the function-calling capability of models.
>
---
#### [new 014] ChartGaze: Enhancing Chart Understanding in LVLMs with Eye-Tracking Guided Attention Refinement
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于图表问答任务，旨在解决LVLMs关注无关区域导致准确率低的问题。作者构建了ChartGaze眼动数据集，并提出基于注视的注意力优化方法，提升模型回答准确性和注意力对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.13282v1](http://arxiv.org/pdf/2509.13282v1)**

> **作者:** Ali Salamatian; Amirhossein Abaskohi; Wan-Cyuan Fan; Mir Rayat Imtiaz Hossain; Leonid Sigal; Giuseppe Carenini
>
> **备注:** EMNLP 2025
>
> **摘要:** Charts are a crucial visual medium for communicating and representing information. While Large Vision-Language Models (LVLMs) have made progress on chart question answering (CQA), the task remains challenging, particularly when models attend to irrelevant regions of the chart. In this work, we present ChartGaze, a new eye-tracking dataset that captures human gaze patterns during chart reasoning tasks. Through a systematic comparison of human and model attention, we find that LVLMs often diverge from human gaze, leading to reduced interpretability and accuracy. To address this, we propose a gaze-guided attention refinement that aligns image-text attention with human fixations. Our approach improves both answer accuracy and attention alignment, yielding gains of up to 2.56 percentage points across multiple models. These results demonstrate the promise of incorporating human gaze to enhance both the reasoning quality and interpretability of chart-focused LVLMs.
>
---
#### [new 015] Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Conan-Embedding-v2，从零训练1.4B参数LLM用于文本嵌入。针对数据和训练方式的差距，引入多语言数据、软掩码机制及动态负采样，提升跨语言嵌入效果，在MTEB基准上取得SOTA性能。属于文本嵌入任务。**

- **链接: [http://arxiv.org/pdf/2509.12892v1](http://arxiv.org/pdf/2509.12892v1)**

> **作者:** Shiyu Li; Yang Tang; Ruijie Liu; Shi-Zhe Chen; Xi Chen
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Large language models (LLMs) have recently demonstrated excellent performance in text embedding tasks. Previous work usually use LoRA to fine-tune existing LLMs, which are limited by the data and training gap between LLMs and embedding models. In this work, we introduce Conan-embedding-v2, a new 1.4B-parameter LLM trained from scratch and fine-tuned as a text embedder. First, we add news data and multilingual pairs for LLM pretraining to bridge the data gap. Based on this, we propose a cross-lingual retrieval dataset that enables the LLM to better integrate embeddings across different languages. Second, whereas LLMs use a causal mask with token-level loss, embedding models use a bidirectional mask with sentence-level loss. This training gap makes full fine-tuning less effective than LoRA. We introduce a soft-masking mechanism to gradually transition between these two types of masks, enabling the model to learn more comprehensive representations. Based on this, we propose a dynamic hard negative mining method that exposes the model to more difficult negative examples throughout the training process. Being intuitive and effective, with only approximately 1.4B parameters, Conan-embedding-v2 achieves SOTA performance on both the Massive Text Embedding Benchmark (MTEB) and Chinese MTEB (May 19, 2025).
>
---
#### [new 016] ConvergeWriter: Data-Driven Bottom-Up Article Construction
- **分类: cs.CL**

- **简介: 论文提出ConvergeWriter，一种基于数据的自底向上文章生成框架，解决大语言模型生成长文时事实错误和结构不连贯的问题。通过先检索知识再聚类构建结构，确保内容准确且可追溯，提升生成文档的可靠性与结构一致性。**

- **链接: [http://arxiv.org/pdf/2509.12811v1](http://arxiv.org/pdf/2509.12811v1)**

> **作者:** Binquan Ji; Jiaqi Wang; Ruiting Li; Xingchen Han; Yiyang Qi; Shichao Wang; Yifei Lu; Yuantao Han; Feiliang Ren
>
> **摘要:** Large Language Models (LLMs) have shown remarkable prowess in text generation, yet producing long-form, factual documents grounded in extensive external knowledge bases remains a significant challenge. Existing "top-down" methods, which first generate a hypothesis or outline and then retrieve evidence, often suffer from a disconnect between the model's plan and the available knowledge, leading to content fragmentation and factual inaccuracies. To address these limitations, we propose a novel "bottom-up," data-driven framework that inverts the conventional generation pipeline. Our approach is predicated on a "Retrieval-First for Knowledge, Clustering for Structure" strategy, which first establishes the "knowledge boundaries" of the source corpus before any generative planning occurs. Specifically, we perform exhaustive iterative retrieval from the knowledge base and then employ an unsupervised clustering algorithm to organize the retrieved documents into distinct "knowledge clusters." These clusters form an objective, data-driven foundation that directly guides the subsequent generation of a hierarchical outline and the final document content. This bottom-up process ensures that the generated text is strictly constrained by and fully traceable to the source material, proactively adapting to the finite scope of the knowledge base and fundamentally mitigating the risk of hallucination. Experimental results on both 14B and 32B parameter models demonstrate that our method achieves performance comparable to or exceeding state-of-the-art baselines, and is expected to demonstrate unique advantages in knowledge-constrained scenarios that demand high fidelity and structural coherence. Our work presents an effective paradigm for generating reliable, structured, long-form documents, paving the way for more robust LLM applications in high-stakes, knowledge-intensive domains.
>
---
#### [new 017] Does Language Model Understand Language?
- **分类: cs.CL**

- **简介: 该论文评估主流语言模型在语言理解任务中的表现，尤其关注时态、否定等细微语言现象。提出LUCID数据集和HCE指标，发现Compound-Beta模型在多语言环境下表现最佳。属于自然语言理解任务，旨在提升模型与人类语言解释的一致性。**

- **链接: [http://arxiv.org/pdf/2509.12459v1](http://arxiv.org/pdf/2509.12459v1)**

> **作者:** Suvojit Acharjee; Utathya Aich; Asfak Ali
>
> **摘要:** Despite advances in natural language generation and understanding, LM still struggle with fine grained linguistic phenomena such as tense, negation, voice, and modality which are the elements central to effective human communication. In the context of the United Nations SDG 4, where linguistic clarity is critical, the deployment of LMs in educational technologies demands careful scrutiny. As LMs are increasingly powering applications like tutoring systems, automated grading, and translation, their alignment with human linguistic interpretation becomes essential for effective learning. In this study, we conduct a evaluation of SOTA language models across these challenging contexts in both English and Bengali. To ensure a structured assessment, we introduce a new Route for Evaluation of Cognitive Inference in Systematic Environments guidelines. Our proposed LUCID dataset, composed of carefully crafted sentence pairs in English and Bengali, specifically challenges these models on critical aspects of language comprehension, including negation, tense, voice variations. We assess the performance of SOTA models including MISTRAL-SABA-24B, LLaMA-4-Scout-17B, LLaMA-3.3-70B, Gemma2-9B, and Compound-Beta using standard metrics like Pearson correlation, Spearman correlation, and Mean Absolute Error, as well as novel, linguistically inspired metric the HCE accuracy. The HCE accuracy measures how often model predictions fall within one standard deviation of the mean human rating, thus capturing human like tolerance for variability in language interpretation. Our findings highlight Compound-Beta as the most balanced model, consistently achieving high correlations and low MAEs across diverse language conditions. It records the highest Pearson correlation in English and demonstrates robust performance on mixed-language data, indicating a strong alignment with human judgments in cross lingual scenarios.
>
---
#### [new 018] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MedFact，用于评估大语言模型在中文医学文本事实核查中的能力。任务是检验LLMs的医学事实准确性。研究构建了一个包含2116个专家标注样本的基准数据集，并测试了20个主流模型，发现模型在错误定位上表现不佳，且存在过度批评现象。**

- **链接: [http://arxiv.org/pdf/2509.12440v1](http://arxiv.org/pdf/2509.12440v1)**

> **作者:** Jiayi He; Yangmin Huang; Qianyun Du; Xiangying Zhou; Zhiyang He; Jiaxue Hu; Xiaodong Tao; Lixian Lai
>
> **摘要:** The increasing deployment of Large Language Models (LLMs) in healthcare necessitates a rigorous evaluation of their factual reliability. However, existing benchmarks are often limited by narrow domains of data, failing to capture the complexity of real-world medical information. To address this critical gap, we introduce MedFact, a new and challenging benchmark for Chinese medical fact-checking. MedFact comprises 2,116 expert-annotated instances curated from diverse real-world texts, spanning 13 medical specialties, 8 fine-grained error types, 4 writing styles, and multiple difficulty levels. Its construction employs a hybrid AI-human framework where iterative expert feedback refines an AI-driven, multi-criteria filtering process, ensuring both high data quality and difficulty. We conduct a comprehensive evaluation of 20 leading LLMs, benchmarking their performance on veracity classification and error localization against a human expert baseline. Our results reveal that while models can often determine if a text contains an error, precisely localizing it remains a substantial challenge, with even top-performing models falling short of human performance. Furthermore, our analysis uncovers a frequent ``over-criticism'' phenomenon, a tendency for models to misidentify correct information as erroneous, which is exacerbated by advanced reasoning techniques such as multi-agent collaboration and inference-time scaling. By highlighting these critical challenges for deploying LLMs in medical applications, MedFact provides a robust resource to drive the development of more factually reliable and medically aware models.
>
---
#### [new 019] Contrastive Learning with Enhanced Abstract Representations using Grouped Loss of Abstract Semantic Supervision
- **分类: cs.CL**

- **简介: 论文提出一种基于分组对比损失的视觉-语言模型训练方法，旨在提升模型对图像中抽象概念的理解能力。通过构建包含多组图像和概念标签的数据集MAGIC，模型在无显式概念标注情况下，学习到更高层次的语义表示，从而增强抽象概念识别能力。**

- **链接: [http://arxiv.org/pdf/2509.12771v1](http://arxiv.org/pdf/2509.12771v1)**

> **作者:** Omri Suissa; Muhiim Ali; Shengmai Chen; Yinuo Cai; Shekhar Pradhan
>
> **摘要:** Humans can recognize an image as an instance of a general concept, beyond simply identifying its objects and their relationships. In this paper, we investigate 1. The extent to which VLMs have this concept abstraction capacity, and 2. Strategies for encoding the sort of higher-concept information in images that would enable the resulting VLM model (CLEAR GLASS model) to have this capability to a greater degree. To this end, we introduce a grouped image-caption dataset (MAGIC), which consists of several groups of image captions and for each group a set of associated images and higher-level conceptual labels. We use a novel contrastive loss technique to induce the model to encode in the representation of each image (caption) in a group the information that is common to all members of the image-caption group. Our main contribution is a grouped contrastive loss function based on text-image contrastive groups (outer contrastive loss) as well as an inner loss which measures the distances between image-caption instances in the group. Our training methodology results in the CLEAR GLASS model having the concept abstraction capacity as an emergent capacity because the model is not exposed to the higher-level concepts associated with each group. Instead, the training forces the model to create for each image-caption group a semantic representation that brings it closer to the semantic representation of the higher-level concepts in the latent semantic space. Our experiments show that this training methodology results in a model which shows improvement in abstract concept recognition compared to SOTA models.
>
---
#### [new 020] Scaling Agents via Continual Pre-training
- **分类: cs.CL**

- **简介: 该论文提出Agentic CPT方法，解决LLMs在代理任务中性能不足的问题。通过持续预训练构建强代理基础模型AgentFounder-30B，在多个基准上取得SOTA结果，提升工具使用能力。属于代理系统优化任务。**

- **链接: [http://arxiv.org/pdf/2509.13310v1](http://arxiv.org/pdf/2509.13310v1)**

> **作者:** Liangcai Su; Zhen Zhang; Guangyu Li; Zhuo Chen; Chenxi Wang; Maojia Song; Xinyu Wang; Kuan Li; Jialong Wu; Xuanzhong Chen; Zile Qiao; Zhongwang Zhang; Huifeng Yin; Shihao Cai; Runnan Fang; Zhengwei Tao; Wenbiao Yin; Chenxiong Qian; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Large language models (LLMs) have evolved into agentic systems capable of autonomous tool use and multi-step reasoning for complex problem-solving. However, post-training approaches building upon general-purpose foundation models consistently underperform in agentic tasks, particularly in open-source implementations. We identify the root cause: the absence of robust agentic foundation models forces models during post-training to simultaneously learn diverse agentic behaviors while aligning them to expert demonstrations, thereby creating fundamental optimization tensions. To this end, we are the first to propose incorporating Agentic Continual Pre-training (Agentic CPT) into the deep research agents training pipeline to build powerful agentic foundational models. Based on this approach, we develop a deep research agent model named AgentFounder. We evaluate our AgentFounder-30B on 10 benchmarks and achieve state-of-the-art performance while retains strong tool-use ability, notably 39.9% on BrowseComp-en, 43.3% on BrowseComp-zh, and 31.5% Pass@1 on HLE.
>
---
#### [new 021] Positional Encoding via Token-Aware Phase Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Token-Aware Phase Attention（TAPA）方法，用于改进位置编码。针对RoPE在长上下文建模中的距离依赖偏差问题，TAPA通过引入可学习的相位函数优化注意力机制，实现更有效的长距离交互与上下文扩展，提升长文本处理效果。**

- **链接: [http://arxiv.org/pdf/2509.12635v1](http://arxiv.org/pdf/2509.12635v1)**

> **作者:** Yu; Wang; Sheng Shen; Rémi Munos; Hongyuan Zhan; Yuandong Tian
>
> **备注:** 21 pages
>
> **摘要:** We prove under practical assumptions that Rotary Positional Embedding (RoPE) introduces an intrinsic distance-dependent bias in attention scores that limits RoPE's ability to model long-context. RoPE extension methods may alleviate this issue, but they typically require post-hoc adjustments after pretraining, such as rescaling or hyperparameters retuning. This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism. TAPA preserves token interactions over long range, extends to longer contexts with direct and light fine-tuning, extrapolates to unseen lengths, and attains significantly lower perplexity on long-context than RoPE families.
>
---
#### [new 022] All Roads Lead to Rome: Graph-Based Confidence Estimation for Large Language Model Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLM）的推理任务，旨在解决现有置信度估计方法在推理任务中泛化能力差的问题。提出了一种无需训练的图基方法，通过建模推理路径并利用图特性进行置信度估计，提升了推理性能。**

- **链接: [http://arxiv.org/pdf/2509.12908v1](http://arxiv.org/pdf/2509.12908v1)**

> **作者:** Caiqi Zhang; Chang Shu; Ehsan Shareghi; Nigel Collier
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Confidence estimation is essential for the reliable deployment of large language models (LLMs). Existing methods are primarily designed for factual QA tasks and often fail to generalize to reasoning tasks. To address this gap, we propose a set of training-free, graph-based confidence estimation methods tailored to reasoning tasks. Our approach models reasoning paths as directed graphs and estimates confidence by exploiting graph properties such as centrality, path convergence, and path weighting. Experiments with two LLMs on three reasoning datasets demonstrate improved confidence estimation and enhanced performance on two downstream tasks.
>
---
#### [new 023] HistoryBankQA: Multilingual Temporal Question Answering on Historical Events
- **分类: cs.CL**

- **简介: 该论文提出HistoryBankQA，解决多语言历史事件时间推理问题。构建包含10M+历史事件的多语言数据库，并创建涵盖6类时间问答任务的基准测试，评估多个大模型性能，推动多语言时序自然语言理解研究。**

- **链接: [http://arxiv.org/pdf/2509.12720v1](http://arxiv.org/pdf/2509.12720v1)**

> **作者:** Biswadip Mandal; Anant Khandelwal; Manish Gupta
>
> **摘要:** Temporal reasoning about historical events is a critical skill for NLP tasks like event extraction, historical entity linking, temporal question answering, timeline summarization, temporal event clustering and temporal natural language inference. Yet efforts on benchmarking temporal reasoning capabilities of large language models (LLMs) are rather limited. Existing temporal reasoning datasets are limited in scale, lack multilingual coverage and focus more on contemporary events. To address these limitations, we present HistoryBank, a multilingual database of 10M+ historical events extracted from Wikipedia timeline pages and article infoboxes. Our database provides unprecedented coverage in both historical depth and linguistic breadth with 10 languages. Additionally, we construct a comprehensive question answering benchmark for temporal reasoning across all languages. This benchmark covers a diverse set of 6 temporal QA reasoning tasks, and we evaluate a suite of popular language models (LLaMA-3-8B, Mistral-7B, Gemma-2-9b, Qwen3-8B, GPT4o) to assess their performance on these tasks. As expected GPT4o performs best across all answer types and languages; Gemma-2 outperforms the other small language models. Our work aims to provide a comprehensive resource for advancing multilingual and temporally-aware natural language understanding of historical events. To facilitate further research, we will make our code and datasets publicly available upon acceptance of this paper.
>
---
#### [new 024] Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction
- **分类: cs.CL**

- **简介: 该论文提出R2tA方法，通过LLM引导的逐步评估与修正，解决小模型在数据稀缺场景下训练任务特定推理模型的问题。方法生成并优化推理轨迹，进行两阶段对齐，提升模型推理准确性，应用于数据库设计等复杂任务。**

- **链接: [http://arxiv.org/pdf/2509.12476v1](http://arxiv.org/pdf/2509.12476v1)**

> **作者:** Sumanta Bhattacharyya; Sara Riaz; Pedram Rooshenas
>
> **摘要:** Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond.
>
---
#### [new 025] SENTRA: Selected-Next-Token Transformer for LLM Text Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SENTRA，一种基于Transformer的LLM生成文本检测模型。任务是识别未明确声明的LLM生成文本。通过利用选中的下一个token概率序列和对比预训练，SENTRA在多个领域数据上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12385v1](http://arxiv.org/pdf/2509.12385v1)**

> **作者:** Mitchell Plyler; Yilun Zhang; Alexander Tuzhilin; Saoud Khalifah; Sen Tian
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** LLMs are becoming increasingly capable and widespread. Consequently, the potential and reality of their misuse is also growing. In this work, we address the problem of detecting LLM-generated text that is not explicitly declared as such. We present a novel, general-purpose, and supervised LLM text detector, SElected-Next-Token tRAnsformer (SENTRA). SENTRA is a Transformer-based encoder leveraging selected-next-token-probability sequences and utilizing contrastive pre-training on large amounts of unlabeled data. Our experiments on three popular public datasets across 24 domains of text demonstrate SENTRA is a general-purpose classifier that significantly outperforms popular baselines in the out-of-domain setting.
>
---
#### [new 026] Don't Change My View: Ideological Bias Auditing in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于意识形态偏见检测任务，旨在解决大语言模型是否被引导至特定意识形态立场的问题。研究提出一种无需模型内部信息的统计方法，通过分析输出分布变化检测潜在引导行为，适用于黑盒系统审计。**

- **链接: [http://arxiv.org/pdf/2509.12652v1](http://arxiv.org/pdf/2509.12652v1)**

> **作者:** Paul Kröger; Emilio Barkett
>
> **摘要:** As large language models (LLMs) become increasingly embedded in products used by millions, their outputs may influence individual beliefs and, cumulatively, shape public opinion. If the behavior of LLMs can be intentionally steered toward specific ideological positions, such as political or religious views, then those who control these systems could gain disproportionate influence over public discourse. Although it remains an open question whether LLMs can reliably be guided toward coherent ideological stances and whether such steering can be effectively prevented, a crucial first step is to develop methods for detecting when such steering attempts occur. In this work, we adapt a previously proposed statistical method to the new context of ideological bias auditing. Our approach carries over the model-agnostic design of the original framework, which does not require access to the internals of the language model. Instead, it identifies potential ideological steering by analyzing distributional shifts in model outputs across prompts that are thematically related to a chosen topic. This design makes the method particularly suitable for auditing proprietary black-box systems. We validate our approach through a series of experiments, demonstrating its practical applicability and its potential to support independent post hoc audits of LLM behavior.
>
---
#### [new 027] Topic Coverage-based Demonstration Retrieval for In-Context Learning
- **分类: cs.CL**

- **简介: 该论文提出TopicK框架，用于改进上下文学习中的演示检索任务。针对现有方法依赖嵌入相似度或生成概率导致的无关冗余问题，TopicK通过主题覆盖机制选择能补充模型知识缺口的演示，提升学习效果。**

- **链接: [http://arxiv.org/pdf/2509.12451v1](http://arxiv.org/pdf/2509.12451v1)**

> **作者:** Wonbin Kweon; SeongKu Kang; Runchu Tian; Pengcheng Jiang; Jiawei Han; Hwanjo Yu
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** The effectiveness of in-context learning relies heavily on selecting demonstrations that provide all the necessary information for a given test input. To achieve this, it is crucial to identify and cover fine-grained knowledge requirements. However, prior methods often retrieve demonstrations based solely on embedding similarity or generation probability, resulting in irrelevant or redundant examples. In this paper, we propose TopicK, a topic coverage-based retrieval framework that selects demonstrations to comprehensively cover topic-level knowledge relevant to both the test input and the model. Specifically, TopicK estimates the topics required by the input and assesses the model's knowledge on those topics. TopicK then iteratively selects demonstrations that introduce previously uncovered required topics, in which the model exhibits low topical knowledge. We validate the effectiveness of TopicK through extensive experiments across various datasets and both open- and closed-source LLMs. Our source code is available at https://github.com/WonbinKweon/TopicK_EMNLP2025.
>
---
#### [new 028] Automated Generation of Research Workflows from Academic Papers: A Full-text Mining Framework
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 该论文提出一个端到端框架，从学术论文中自动提取完整研究流程，解决现有方法仅提取碎片化步骤的问题。采用PU学习和Flan-T5生成流程短语，并分类为数据准备、处理与分析阶段，最终生成可视化流程图，提升科研可复现性。**

- **链接: [http://arxiv.org/pdf/2509.12955v1](http://arxiv.org/pdf/2509.12955v1)**

> **作者:** Heng Zhang; Chengzhi Zhang
>
> **摘要:** The automated generation of research workflows is essential for improving the reproducibility of research and accelerating the paradigm of "AI for Science". However, existing methods typically extract merely fragmented procedural components and thus fail to capture complete research workflows. To address this gap, we propose an end-to-end framework that generates comprehensive, structured research workflows by mining full-text academic papers. As a case study in the Natural Language Processing (NLP) domain, our paragraph-centric approach first employs Positive-Unlabeled (PU) Learning with SciBERT to identify workflow-descriptive paragraphs, achieving an F1-score of 0.9772. Subsequently, we utilize Flan-T5 with prompt learning to generate workflow phrases from these paragraphs, yielding ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.4543, 0.2877, and 0.4427, respectively. These phrases are then systematically categorized into data preparation, data processing, and data analysis stages using ChatGPT with few-shot learning, achieving a classification precision of 0.958. By mapping categorized phrases to their document locations in the documents, we finally generate readable visual flowcharts of the entire research workflows. This approach facilitates the analysis of workflows derived from an NLP corpus and reveals key methodological shifts over the past two decades, including the increasing emphasis on data analysis and the transition from feature engineering to ablation studies. Our work offers a validated technical framework for automated workflow generation, along with a novel, process-oriented perspective for the empirical investigation of evolving scientific paradigms. Source code and data are available at: https://github.com/ZH-heng/research_workflow.
>
---
#### [new 029] MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch
- **分类: cs.CL**

- **简介: 该论文提出MTEB-NL和E5-NL，旨在填补荷兰语嵌入资源的不足。通过构建基准数据集和训练数据，并发布高效模型，推动荷兰语嵌入的发展。属于自然语言处理中的嵌入模型研究任务。**

- **链接: [http://arxiv.org/pdf/2509.12340v1](http://arxiv.org/pdf/2509.12340v1)**

> **作者:** Nikolay Banar; Ehsan Lotfi; Jens Van Nooten; Cristina Arhiliuc; Marija Kliocaite; Walter Daelemans
>
> **摘要:** Recently, embedding resources, including models, benchmarks, and datasets, have been widely released to support a variety of languages. However, the Dutch language remains underrepresented, typically comprising only a small fraction of the published multilingual resources. To address this gap and encourage the further development of Dutch embeddings, we introduce new resources for their evaluation and generation. First, we introduce the Massive Text Embedding Benchmark for Dutch (MTEB-NL), which includes both existing Dutch datasets and newly created ones, covering a wide range of tasks. Second, we provide a training dataset compiled from available Dutch retrieval datasets, complemented with synthetic data generated by large language models to expand task coverage beyond retrieval. Finally, we release a series of E5-NL models compact yet efficient embedding models that demonstrate strong performance across multiple tasks. We make our resources publicly available through the Hugging Face Hub and the MTEB package.
>
---
#### [new 030] A comparison of pipelines for the translation of a low resource language based on transformers
- **分类: cs.CL; cs.CE; cs.CY; cs.LG**

- **简介: 该论文比较三种基于Transformer的翻译管道，用于训练低资源语言Bambara的机器翻译系统。任务是提升BLEU和chrF指标，解决低资源语言翻译难题，通过不同模型结构与训练策略进行实验评估。**

- **链接: [http://arxiv.org/pdf/2509.12514v1](http://arxiv.org/pdf/2509.12514v1)**

> **作者:** Chiara Bonfanti; Michele Colombino; Giulia Coucourde; Faeze Memari; Stefano Pinardi; Rosa Meo
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** This work compares three pipelines for training transformer-based neural networks to produce machine translators for Bambara, a Mand\`e language spoken in Africa by about 14,188,850 people. The first pipeline trains a simple transformer to translate sentences from French into Bambara. The second fine-tunes LLaMA3 (3B-8B) instructor models using decoder-only architectures for French-to-Bambara translation. Models from the first two pipelines were trained with different hyperparameter combinations to improve BLEU and chrF scores, evaluated on both test sentences and official Bambara benchmarks. The third pipeline uses language distillation with a student-teacher dual neural network to integrate Bambara into a pre-trained LaBSE model, which provides language-agnostic embeddings. A BERT extension is then applied to LaBSE to generate translations. All pipelines were tested on Dokotoro (medical) and Bayelemagaba (mixed domains). Results show that the first pipeline, although simpler, achieves the best translation accuracy (10% BLEU, 21% chrF on Bayelemagaba), consistent with low-resource translation results. On the Yiri dataset, created for this work, it achieves 33.81% BLEU and 41% chrF. Instructor-based models perform better on single datasets than on aggregated collections, suggesting they capture dataset-specific patterns more effectively.
>
---
#### [new 031] Multi-Model Synthetic Training for Mission-Critical Small Language Models
- **分类: cs.CL; cs.AI; cs.LG; 68T50 68T50; I.2.7; I.2.6**

- **简介: 论文提出多模型合成训练方法，解决专业领域小语言模型数据稀缺问题。利用LLM生成合成数据，训练出高精度、低成本的Qwen2.5-7B模型，应用于海事任务，提升安全与管理效率。**

- **链接: [http://arxiv.org/pdf/2509.13047v1](http://arxiv.org/pdf/2509.13047v1)**

> **作者:** Nolan Platt; Pragyansmita Nayak
>
> **备注:** 8 pages. Accepted as a full paper to the 3rd International Conference on Foundation and Large Language Models (IEEE FLLM) 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across many domains, yet their appli- cation to specialized fields remains constrained by the scarcity and complexity of domain-specific training data. We present a novel approach that achieves a 261x cost reduction for maritime intelligence by using LLMs as one-time teachers rather than using them directly for inference. Our method transforms 3.2 billion Automatic Identification System (AIS) vessel tracking records into 21,543 synthetic question and answer pairs through multi-model generation (GPT-4o and o3-mini), preventing over- fitting and ensuring accurate reasoning. The resulting fine-tuned Qwen2.5-7B model achieves 75% accuracy on maritime tasks, while being substantially cheaper than using a larger model for inference. We show that smaller, cheaper models - when fine tuned properly - can provide similar accuracy compared to larger models that are prohibitively expensive. Our work contributes to the growing field of synthetic dataset generation for specialized AI applications and presents a highly reproducible framework for domains where manual annotation is infeasible. Beyond expand- ing research in the growing field of specialized small language models, our approach has immediate applications in maritime safety, security operations, and vessel traffic management systems in various industries.
>
---
#### [new 032] MAGIC-Enhanced Keyword Prompting for Zero-Shot Audio Captioning with CLIP Models
- **分类: cs.CL**

- **简介: 论文提出一种零样本音频字幕生成方法，利用预训练CLIP模型提取音频特征并生成结构化提示，引导大语言模型生成字幕。通过MAGIC搜索优化关键词选择，提升生成质量，解决了音频字幕数据不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.12591v1](http://arxiv.org/pdf/2509.12591v1)**

> **作者:** Vijay Govindarajan; Pratik Patel; Sahil Tripathi; Md Azizul Hoque; Gautam Siddharth Kashyap
>
> **备注:** Accepted in The 26th International Conference on Web Information Systems Engineering (WISE), scheduled for 15-17 December 2025 in Marrakech, Morocco
>
> **摘要:** Automated Audio Captioning (AAC) generates captions for audio clips but faces challenges due to limited datasets compared to image captioning. To overcome this, we propose the zero-shot AAC system that leverages pre-trained models, eliminating the need for extensive training. Our approach uses a pre-trained audio CLIP model to extract auditory features and generate a structured prompt, which guides a Large Language Model (LLM) in caption generation. Unlike traditional greedy decoding, our method refines token selection through the audio CLIP model, ensuring alignment with the audio content. Experimental results demonstrate a 35% improvement in NLG mean score (from 4.7 to 7.3) using MAGIC search with the WavCaps model. The performance is heavily influenced by the audio-text matching model and keyword selection, with optimal results achieved using a single keyword prompt, and a 50% performance drop when no keyword list is used.
>
---
#### [new 033] Investigating ReLoRA: Effects on the Learning Dynamics of Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究ReLoRA在小语言模型预训练中的效果，分析其性能与学习动态。发现ReLoRA在损失、困惑度等指标上表现较差，且加剧了小模型的秩缺陷，指出低秩更新策略可能不适用于小模型预训练。**

- **链接: [http://arxiv.org/pdf/2509.12960v1](http://arxiv.org/pdf/2509.12960v1)**

> **作者:** Yuval Weiss; David Demitri Africa; Paula Buttery; Richard Diehl Martinez
>
> **备注:** 12 Pages, 6 Tables, 8 Figures
>
> **摘要:** Parameter-efficient methods such as LoRA have revolutionised the fine-tuning of LLMs. Still, their extension to pretraining via ReLoRA is less well understood, especially for small language models (SLMs), which offer lower computational and environmental costs. This work is the first systematic study of ReLoRA in SLMs (11M-66M parameters), evaluating both performance and learning dynamics. Through ablation experiments, we find that ReLoRA generally performs worse than standard training on loss, Paloma perplexity and BLiMP, with the gap widening for the larger models. Further analysis of the learning dynamics of the models indicates that ReLoRA reinforces the rank deficiencies found in smaller models. These results indicate that low-rank update strategies may not transfer easily to SLM pretraining, highlighting the need for more research in the low-compute regime.
>
---
#### [new 034] Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews
- **分类: cs.CL**

- **简介: 该论文研究跨文化葡萄酒评论的适应问题，属于文化感知语言任务。通过构建中英文专业评论语料库，提出文化导向评估标准，测试翻译模型在文化细节处理上的不足，揭示其在跨文化内容生成中的挑战。**

- **链接: [http://arxiv.org/pdf/2509.12961v1](http://arxiv.org/pdf/2509.12961v1)**

> **作者:** Chenye Zou; Xingyue Wen; Tianyi Hu; Qian Janice Wang; Daniel Hershcovich
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Recent advances in large language models (LLMs) have opened the door to culture-aware language tasks. We introduce the novel problem of adapting wine reviews across Chinese and English, which goes beyond literal translation by incorporating regional taste preferences and culture-specific flavor descriptors. In a case study on cross-cultural wine review adaptation, we compile the first parallel corpus of professional reviews, containing 8k Chinese and 16k Anglophone reviews. We benchmark both neural-machine-translation baselines and state-of-the-art LLMs with automatic metrics and human evaluation. For the latter, we propose three culture-oriented criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness -- to assess how naturally a translated review resonates with target-culture readers. Our analysis shows that current models struggle to capture cultural nuances, especially in translating wine descriptions across different cultures. This highlights the challenges and limitations of translation models in handling cultural content.
>
---
#### [new 035] Chat-Driven Text Generation and Interaction for Person Retrieval
- **分类: cs.CL; I.2.7; I.4.9**

- **简介: 该论文属于文本驱动的人像检索任务，旨在解决高质量标注依赖问题。提出MTG和MTI模块，通过多轮对话生成伪标签并优化查询，实现无需人工标注的高效检索。**

- **链接: [http://arxiv.org/pdf/2509.12662v1](http://arxiv.org/pdf/2509.12662v1)**

> **作者:** Zequn Xie; Chuxin Wang; Sihang Cai; Yeqiang Wang; Shulei Wang; Tao Jin
>
> **备注:** Accepted by EMNLP 2025. 13 pages, 3 figures
>
> **摘要:** Text-based person search (TBPS) enables the retrieval of person images from large-scale databases using natural language descriptions, offering critical value in surveillance applications. However, a major challenge lies in the labor-intensive process of obtaining high-quality textual annotations, which limits scalability and practical deployment. To address this, we introduce two complementary modules: Multi-Turn Text Generation (MTG) and Multi-Turn Text Interaction (MTI). MTG generates rich pseudo-labels through simulated dialogues with MLLMs, producing fine-grained and diverse visual descriptions without manual supervision. MTI refines user queries at inference time through dynamic, dialogue-based reasoning, enabling the system to interpret and resolve vague, incomplete, or ambiguous descriptions - characteristics often seen in real-world search scenarios. Together, MTG and MTI form a unified and annotation-free framework that significantly improves retrieval accuracy, robustness, and usability. Extensive evaluations demonstrate that our method achieves competitive or superior results while eliminating the need for manual captions, paving the way for scalable and practical deployment of TBPS systems.
>
---
#### [new 036] FunAudio-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出FunAudio-ASR，一种结合大规模数据、大模型与LLM的ASR系统，通过强化学习优化，解决LLM幻觉问题，提升实际应用中的识别性能与鲁棒性。属于语音识别任务。**

- **链接: [http://arxiv.org/pdf/2509.12508v1](http://arxiv.org/pdf/2509.12508v1)**

> **作者:** Keyu An; Yanni Chen; Chong Deng; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Wen Wang; Wupeng Wang; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings.
>
---
#### [new 037] The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations
- **分类: cs.CL; cs.AI**

- **简介: 论文提出一种基于LLM隐藏表示估计问题难度的新方法，无需生成输出即可高效评估难度，并用于指导自适应推理策略。属于自然语言处理中的难度估计任务，解决传统方法计算成本高、泛化性差的问题。**

- **链接: [http://arxiv.org/pdf/2509.12886v1](http://arxiv.org/pdf/2509.12886v1)**

> **作者:** Yubo Zhu; Dongrui Liu; Zecheng Lin; Wei Tong; Sheng Zhong; Jing Shao
>
> **摘要:** Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens.
>
---
#### [new 038] Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO
- **分类: cs.CL; cs.AI**

- **简介: 论文提出一种基于语义奖励模型的GRPO方法，用于提升生成解释的质量。任务为复杂生成，解决LLM输出与教育目标对齐问题，使用轻量编码器模型替代传统指标，提升解释的准确性和清晰度。**

- **链接: [http://arxiv.org/pdf/2509.13081v1](http://arxiv.org/pdf/2509.13081v1)**

> **作者:** Francesco Pappone; Ruggero Marino Lazzaroni; Federico Califano; Niccolò Gentile; Roberto Marras
>
> **摘要:** While Large Language Models (LLMs) excel at generating human-like text, aligning their outputs with complex, qualitative goals like pedagogical soundness remains a significant challenge. Standard reinforcement learning techniques often rely on slow and expensive LLM-as-a-judge evaluations or on brittle, keyword-based metrics like ROUGE, which fail to capture the semantic essence of a high-quality explanation. In this work, we introduce a novel approach to reward shaping within the Group Relative Policy Optimisation (GRPO) framework. Our central contribution is the use of a small, efficient encoder-only transformer as a semantic reward model. This model provides a dense, semantically rich reward signal based on the cosine similarity between a generated explanation and a ground-truth reference, guiding the policy towards explanations that are not just factually correct but also structurally and conceptually aligned with expert reasoning. We apply this method to the task of training a model for the Italian medical-school entrance examinations, following standard domain-adaptive continued pre-training (CPT) and supervised fine-tuning (SFT). Our results demonstrate that GRPO with our proposed semantic reward significantly improves explanation faithfulness and clarity over a strong SFT baseline, showcasing the power of using lightweight encoder models for nuanced reward shaping in complex generation tasks
>
---
#### [new 039] Benchmarking and Improving LVLMs on Event Extraction from Multimedia Documents
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多媒体事件抽取（M2E2）任务，旨在提升LVLM在跨模态场景下的性能。论文系统评估了多个LVLM模型，并通过微调和少样本提示方法改进其表现，揭示了跨模态协同与现存挑战。**

- **链接: [http://arxiv.org/pdf/2509.12876v1](http://arxiv.org/pdf/2509.12876v1)**

> **作者:** Fuyu Xing; Zimu Wang; Wei Wang; Haiyang Zhang
>
> **备注:** Accepted at INLG 2025. Camera-ready version
>
> **摘要:** The proliferation of multimedia content necessitates the development of effective Multimedia Event Extraction (M2E2) systems. Though Large Vision-Language Models (LVLMs) have shown strong cross-modal capabilities, their utility in the M2E2 task remains underexplored. In this paper, we present the first systematic evaluation of representative LVLMs, including DeepSeek-VL2 and the Qwen-VL series, on the M2E2 dataset. Our evaluations cover text-only, image-only, and cross-media subtasks, assessed under both few-shot prompting and fine-tuning settings. Our key findings highlight the following valuable insights: (1) Few-shot LVLMs perform notably better on visual tasks but struggle significantly with textual tasks; (2) Fine-tuning LVLMs with LoRA substantially enhances model performance; and (3) LVLMs exhibit strong synergy when combining modalities, achieving superior performance in cross-modal settings. We further provide a detailed error analysis to reveal persistent challenges in areas such as semantic precision, localization, and cross-modal grounding, which remain critical obstacles for advancing M2E2 capabilities.
>
---
#### [new 040] Data Augmentation for Maltese NLP using Transliterated and Machine Translated Arabic Data
- **分类: cs.CL**

- **简介: 该论文研究如何利用阿拉伯语资源通过跨语言增强技术提升马耳他语NLP任务。论文提出多种对齐策略，包括新的转写系统和机器翻译方法，并验证其对单语和多语模型的有效性。**

- **链接: [http://arxiv.org/pdf/2509.12853v1](http://arxiv.org/pdf/2509.12853v1)**

> **作者:** Kurt Micallef; Nizar Habash; Claudia Borg
>
> **备注:** EMNLP Camera-Ready
>
> **摘要:** Maltese is a unique Semitic language that has evolved under extensive influence from Romance and Germanic languages, particularly Italian and English. Despite its Semitic roots, its orthography is based on the Latin script, creating a gap between it and its closest linguistic relatives in Arabic. In this paper, we explore whether Arabic-language resources can support Maltese natural language processing (NLP) through cross-lingual augmentation techniques. We investigate multiple strategies for aligning Arabic textual data with Maltese, including various transliteration schemes and machine translation (MT) approaches. As part of this, we also introduce novel transliteration systems that better represent Maltese orthography. We evaluate the impact of these augmentations on monolingual and mutlilingual models and demonstrate that Arabic-based augmentation can significantly benefit Maltese NLP tasks.
>
---
#### [new 041] EconProver: Towards More Economical Test-Time Scaling for Automated Theorem Proving
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动定理证明（ATP）任务，旨在解决现有模型测试时扩展策略计算成本高的问题。提出动态CoT切换和强化学习方法，在保持性能的同时大幅降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.12603v1](http://arxiv.org/pdf/2509.12603v1)**

> **作者:** Mukai Li; Linfeng Song; Zhenwen Liang; Jiahao Xu; Shansan Gong; Qi Liu; Haitao Mi; Dong Yu
>
> **摘要:** Large Language Models (LLMs) have recently advanced the field of Automated Theorem Proving (ATP), attaining substantial performance gains through widely adopted test-time scaling strategies, notably reflective Chain-of-Thought (CoT) reasoning and increased sampling passes. However, they both introduce significant computational overhead for inference. Moreover, existing cost analyses typically regulate only the number of sampling passes, while neglecting the substantial disparities in sampling costs introduced by different scaling strategies. In this paper, we systematically compare the efficiency of different test-time scaling strategies for ATP models and demonstrate the inefficiency of the current state-of-the-art (SOTA) open-source approaches. We then investigate approaches to significantly reduce token usage and sample passes while maintaining the original performance. Specifically, we propose two complementary methods that can be integrated into a unified EconRL pipeline for amplified benefits: (1) a dynamic Chain-of-Thought (CoT) switching mechanism designed to mitigate unnecessary token consumption, and (2) Diverse parallel-scaled reinforcement learning (RL) with trainable prefixes to enhance pass rates under constrained sampling passes. Experiments on miniF2F and ProofNet demonstrate that our EconProver achieves comparable performance to baseline methods with only 12% of the computational cost. This work provides actionable insights for deploying lightweight ATP models without sacrificing performance.
>
---
#### [new 042] Evaluating LLM Alignment on Personality Inference from Real-World Interview Data
- **分类: cs.CL**

- **简介: 该论文评估大语言模型在从真实访谈数据中推断人格特质的能力。任务是检验LLMs与连续人格评估的对齐程度。研究构建新基准，采用多种方法测试模型表现，发现其与真实人格评分相关性较低，揭示当前LLMs在心理属性对齐上的局限性。**

- **链接: [http://arxiv.org/pdf/2509.13244v1](http://arxiv.org/pdf/2509.13244v1)**

> **作者:** Jianfeng Zhu; Julina Maharjan; Xinyu Li; Karin G. Coifman; Ruoming Jin
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in roles requiring nuanced psychological understanding, such as emotional support agents, counselors, and decision-making assistants. However, their ability to interpret human personality traits, a critical aspect of such applications, remains unexplored, particularly in ecologically valid conversational settings. While prior work has simulated LLM "personas" using discrete Big Five labels on social media data, the alignment of LLMs with continuous, ground-truth personality assessments derived from natural interactions is largely unexamined. To address this gap, we introduce a novel benchmark comprising semi-structured interview transcripts paired with validated continuous Big Five trait scores. Using this dataset, we systematically evaluate LLM performance across three paradigms: (1) zero-shot and chain-of-thought prompting with GPT-4.1 Mini, (2) LoRA-based fine-tuning applied to both RoBERTa and Meta-LLaMA architectures, and (3) regression using static embeddings from pretrained BERT and OpenAI's text-embedding-3-small. Our results reveal that all Pearson correlations between model predictions and ground-truth personality traits remain below 0.26, highlighting the limited alignment of current LLMs with validated psychological constructs. Chain-of-thought prompting offers minimal gains over zero-shot, suggesting that personality inference relies more on latent semantic representation than explicit reasoning. These findings underscore the challenges of aligning LLMs with complex human attributes and motivate future work on trait-specific prompting, context-aware modeling, and alignment-oriented fine-tuning.
>
---
#### [new 043] Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning
- **分类: cs.CL**

- **简介: 论文提出PLAP框架，解决LLM在复杂对抗长期环境中行动规划问题。通过参数化技能库、技能规划器和执行器，提升LLM代理的长期规划能力，并在MicroRTS中验证有效性，发布LLM技能规划能力排行榜。**

- **链接: [http://arxiv.org/pdf/2509.13127v1](http://arxiv.org/pdf/2509.13127v1)**

> **作者:** Sijia Cui; Shuai Xu; Aiyao He; Yanna Wang; Bo Xu
>
> **备注:** Accepted to IJCNN 2025
>
> **摘要:** Recent advancements in Large Language Models(LLMs) have led to the development of LLM-based AI agents. A key challenge is the creation of agents that can effectively ground themselves in complex, adversarial long-horizon environments. Existing methods mainly focus on (1) using LLMs as policies to interact with the environment through generating low-level feasible actions, and (2) utilizing LLMs to generate high-level tasks or language guides to stimulate action generation. However, the former struggles to generate reliable actions, while the latter relies heavily on expert experience to translate high-level tasks into specific action sequences. To address these challenges, we introduce the Plan with Language, Act with Parameter (PLAP) planning framework that facilitates the grounding of LLM-based agents in long-horizon environments. The PLAP method comprises three key components: (1) a skill library containing environment-specific parameterized skills, (2) a skill planner powered by LLMs, and (3) a skill executor converting the parameterized skills into executable action sequences. We implement PLAP in MicroRTS, a long-horizon real-time strategy game that provides an unfamiliar and challenging environment for LLMs. The experimental results demonstrate the effectiveness of PLAP. In particular, GPT-4o-driven PLAP in a zero-shot setting outperforms 80% of baseline agents, and Qwen2-72B-driven PLAP, with carefully crafted few-shot examples, surpasses the top-tier scripted agent, CoacAI. Additionally, we design comprehensive evaluation metrics and test 6 closed-source and 2 open-source LLMs within the PLAP framework, ultimately releasing an LLM leaderboard ranking long-horizon skill planning ability. Our code is available at https://github.com/AI-Research-TeamX/PLAP.
>
---
#### [new 044] Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content
- **分类: cs.CL**

- **简介: 该论文属于毒性内容检测任务，旨在解决LLM生成内容导致的误分类和对抗攻击问题。研究通过机制可解释性技术识别并抑制脆弱组件，提升模型鲁棒性与公平性。**

- **链接: [http://arxiv.org/pdf/2509.12672v1](http://arxiv.org/pdf/2509.12672v1)**

> **作者:** Shaz Furniturewala; Arkaitz Zubiaga
>
> **摘要:** The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.
>
---
#### [new 045] LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation
- **分类: cs.CL; H.3.3; I.2.7; I.2.6**

- **简介: 论文提出用大语言模型（LLM）作为评判者，评估检索增强生成系统在法律文档推荐中的表现。旨在解决传统指标无法准确衡量法律领域推荐质量的问题，通过实验确定更可靠的评估方法，实现高效、自动化的系统比较。**

- **链接: [http://arxiv.org/pdf/2509.12382v1](http://arxiv.org/pdf/2509.12382v1)**

> **作者:** Anu Pradhan; Alexandra Ortan; Apurv Verma; Madhavan Seshadri
>
> **备注:** Accepted in EARL 25: The 2nd Workshop on Evaluating and Applying Recommender Systems with Large Language Models at RecSys 2025
>
> **摘要:** The evaluation bottleneck in recommendation systems has become particularly acute with the rise of Generative AI, where traditional metrics fall short of capturing nuanced quality dimensions that matter in specialized domains like legal research. Can we trust Large Language Models to serve as reliable judges of their own kind? This paper investigates LLM-as-a-Judge as a principled approach to evaluating Retrieval-Augmented Generation systems in legal contexts, where the stakes of recommendation quality are exceptionally high. We tackle two fundamental questions that determine practical viability: which inter-rater reliability metrics best capture the alignment between LLM and human assessments, and how do we conduct statistically sound comparisons between competing systems? Through systematic experimentation, we discover that traditional agreement metrics like Krippendorff's alpha can be misleading in the skewed distributions typical of AI system evaluations. Instead, Gwet's AC2 and rank correlation coefficients emerge as more robust indicators for judge selection, while the Wilcoxon Signed-Rank Test with Benjamini-Hochberg corrections provides the statistical rigor needed for reliable system comparisons. Our findings suggest a path toward scalable, cost-effective evaluation that maintains the precision demanded by legal applications, transforming what was once a human-intensive bottleneck into an automated, yet statistically principled, evaluation framework.
>
---
#### [new 046] RepIt: Representing Isolated Targets to Steer Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RepIt框架，用于隔离大语言模型中的特定概念表示，实现精准干预。旨在解决激活引导方法影响范围过广的问题，通过少量数据高效提取目标概念向量，提升模型行为控制的粒度。**

- **链接: [http://arxiv.org/pdf/2509.13281v1](http://arxiv.org/pdf/2509.13281v1)**

> **作者:** Vincent Siu; Nathan W. Henry; Nicholas Crispino; Yang Liu; Dawn Song; Chenguang Wang
>
> **摘要:** While activation steering in large language models (LLMs) is a growing area of research, methods can often incur broader effects than desired. This motivates isolation of purer concept vectors to enable targeted interventions and understand LLM behavior at a more granular level. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations. Across five frontier LLMs, RepIt enables precise interventions: it selectively suppresses refusal on targeted concepts while preserving refusal elsewhere, producing models that answer WMD-related questions while still scoring as safe on standard benchmarks. We further show that the corrective signal localizes to just 100-200 neurons and that robust target representations can be extracted from as few as a dozen examples on a single A6000. This efficiency raises a dual concern: manipulations can be performed with modest compute and data to extend to underrepresented data-scarce topics while evading existing benchmarks. By disentangling refusal vectors with RepIt, this work demonstrates that targeted interventions can counteract overgeneralization, laying the foundation for more granular control of model behavior.
>
---
#### [new 047] LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 论文提出LEAF框架，用于文本嵌入模型的知识蒸馏。该方法使学生模型与教师模型对齐，在信息检索中实现灵活的异构架构，并提升检索性能。同时适用于多任务场景，无需硬负样本或大量数据，训练要求低。**

- **链接: [http://arxiv.org/pdf/2509.12539v1](http://arxiv.org/pdf/2509.12539v1)**

> **作者:** Robin Vujanic; Thomas Rueckstiess
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** We present LEAF ("Lightweight Embedding Alignment Framework"), a knowledge distillation framework for text embedding models. A key distinguishing feature is that our distilled leaf models are aligned to their teacher. In the context of information retrieval, this allows for flexible asymmetric architectures where documents are encoded with the larger teacher model, while queries can be served with the smaller leaf models. We also show that leaf models automatically inherit MRL and robustness to output quantization whenever these properties are present in the teacher model, without explicitly training for them. To demonstrate the capability of our framework we publish leaf-ir, a 23M parameters information retrieval oriented text embedding model trained using LEAF, which sets a new state-of-the-art (SOTA) on BEIR, ranking #1 on the public leaderboard for this benchmark and for models of its size. When run in asymmetric mode, its retrieval performance is further increased. Our scheme is however not restricted to the information retrieval setting, and we demonstrate its wider applicability by synthesizing the multi-task leaf-mt model. This also sets a new SOTA, ranking #1 on the public MTEB v2 (English) leaderboard for its size. LEAF is applicable to black-box models and in contrast to other embedding model training frameworks, it does not require judgments nor hard negatives, and training can be conducted using small batch sizes. Thus, dataset and training infrastructure requirements for our framework are modest. We make our models publicly available under a permissive Apache 2.0 license.
>
---
#### [new 048] InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文提出InfoGain-RAG框架，通过文档信息增益（DIG）度量检索文档对生成答案的贡献，解决RAG中难以过滤无关或误导性内容的问题。实验表明该方法在多个基准上显著优于现有方法。属于检索增强生成任务。**

- **链接: [http://arxiv.org/pdf/2509.12765v1](http://arxiv.org/pdf/2509.12765v1)**

> **作者:** Zihan Wang; Zihan Liang; Zhou Shao; Yufei Ma; Huangyu Dai; Ben Chen; Lingtao Mao; Chenyi Lei; Yuqing Ding; Han Li
>
> **备注:** EMNLP'25 Oral Presentation. Contact: benchen4395@gmail.com
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising approach to address key limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and lacking reference. However, current RAG frameworks often struggle with identifying whether retrieved documents meaningfully contribute to answer generation. This shortcoming makes it difficult to filter out irrelevant or even misleading content, which notably impacts the final performance. In this paper, we propose Document Information Gain (DIG), a novel metric designed to quantify the contribution of retrieved documents to correct answer generation. DIG measures a document's value by computing the difference of LLM's generation confidence with and without the document augmented. Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to train a specialized reranker, which prioritizes each retrieved document from exact distinguishing and accurate sorting perspectives. This approach can effectively filter out irrelevant documents and select the most valuable ones for better answer generation. Extensive experiments across various models and benchmarks demonstrate that InfoGain-RAG can significantly outperform existing approaches, on both single and multiple retrievers paradigm. Specifically on NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG respectively, and even an average of 15.3% increment on advanced proprietary model GPT-4o across all datasets. These results demonstrate the feasibility of InfoGain-RAG as it can offer a reliable solution for RAG in multiple applications.
>
---
#### [new 049] Context-Aware Language Models for Forecasting Market Impact from Sequences of Financial News
- **分类: cs.CE; cs.CL; q-fin.CP; I.2.7; J.4**

- **简介: 该论文属于金融新闻市场影响预测任务，旨在利用历史上下文提升大语言模型对新闻市场影响的理解。提出一种高效方法，结合大模型与小模型处理主文与历史摘要，实验证明其能显著提升预测性能并应用于投资模拟。**

- **链接: [http://arxiv.org/pdf/2509.12519v1](http://arxiv.org/pdf/2509.12519v1)**

> **作者:** Ross Koval; Nicholas Andrews; Xifeng Yan
>
> **备注:** Preprint
>
> **摘要:** Financial news plays a critical role in the information diffusion process in financial markets and is a known driver of stock prices. However, the information in each news article is not necessarily self-contained, often requiring a broader understanding of the historical news coverage for accurate interpretation. Further, identifying and incorporating the most relevant contextual information presents significant challenges. In this work, we explore the value of historical context in the ability of large language models to understand the market impact of financial news. We find that historical context provides a consistent and significant improvement in performance across methods and time horizons. To this end, we propose an efficient and effective contextualization method that uses a large LM to process the main article, while a small LM encodes the historical context into concise summary embeddings that are then aligned with the large model's representation space. We explore the behavior of the model through multiple qualitative and quantitative interpretability tests and reveal insights into the value of contextualization. Finally, we demonstrate that the value of historical context in model predictions has real-world applications, translating to substantial improvements in simulated investment performance.
>
---
#### [new 050] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出LightVLA，一种通过可微分视觉token剪枝提升视觉-语言-动作模型效率的方法。针对VLA模型在资源受限平台部署时计算量大的问题，通过动态评估token重要性实现高效剪枝，提升性能并减少计算开销。**

- **链接: [http://arxiv.org/pdf/2509.12594v1](http://arxiv.org/pdf/2509.12594v1)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.9% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [new 051] When Inverse Data Outperforms: Exploring the Pitfalls of Mixed Data in Multi-Stage Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究多阶段微调中混合数据的负面影响，构建反向推理数据集r1k，分析SFT与DPO在双向推理目标下的对齐效果，揭示混合数据引入冲突信号，提出需方向感知的对齐策略。属于自然语言处理中的模型微调任务。**

- **链接: [http://arxiv.org/pdf/2509.13079v1](http://arxiv.org/pdf/2509.13079v1)**

> **作者:** Mengyi Deng; Xin Li; Tingyu Zhu; Zhicheng Yang; Zhijiang Guo; Wei Wang
>
> **摘要:** Existing work has shown that o1-level performance can be achieved with limited data distillation, but most existing methods focus on unidirectional supervised fine-tuning (SFT), overlooking the intricate interplay between diverse reasoning patterns. In this paper, we construct r1k, a high-quality reverse reasoning dataset derived by inverting 1,000 forward examples from s1k, and examine how SFT and Direct Preference Optimization (DPO) affect alignment under bidirectional reasoning objectives. SFT on r1k yields a 1.6%--6.8% accuracy improvement over s1k across evaluated benchmarks. However, naively mixing forward and reverse data during SFT weakens the directional distinction. Although DPO can partially recover this distinction, it also suppresses less preferred reasoning paths by shifting the probability mass toward irrelevant outputs. These findings suggest that mixed reasoning data introduce conflicting supervision signals, underscoring the need for robust and direction-aware alignment strategies.
>
---
#### [new 052] MEUV: Achieving Fine-Grained Capability Activation in Large Language Models via Mutually Exclusive Unlock Vectors
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出MEUV框架，解决大语言模型中安全对齐导致的合法功能受限问题。通过分解单一拒绝方向为多个互斥主题向量，实现细粒度能力激活，在保持安全性的前提下提升模型在高风险场景中的可控性与实用性。**

- **链接: [http://arxiv.org/pdf/2509.12221v1](http://arxiv.org/pdf/2509.12221v1)**

> **作者:** Xin Tong; Zhi Lin; Jingya Wang; Meng Han; Bo Jin
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) enforce safety alignment to reliably refuse malicious requests, yet the same blanket safeguards also block legitimate uses in policing, defense, and other high-stakes settings. Earlier "refusal-direction" edits can bypass those layers, but they rely on a single vector that indiscriminately unlocks all hazardous topics, offering no semantic control. We introduce Mutually Exclusive Unlock Vectors (MEUV), a lightweight framework that factorizes the monolithic refusal direction into topic-aligned, nearly orthogonal vectors, each dedicated to one sensitive capability. MEUV is learned in a single epoch with a multi-task objective that blends a differential-ablation margin, cross-topic and orthogonality penalties, and several auxiliary terms. On bilingual malicious-prompt benchmarks, MEUV achieves an attack success rate of no less than 87% on Gemma-2-2B, LLaMA-3-8B, and Qwen-7B, yet cuts cross-topic leakage by up to 90% compared with the best single-direction baseline. Vectors trained in Chinese transfer almost unchanged to English (and vice versa), suggesting a language-agnostic refusal subspace. The results show that fine-grained, topic-level capability activation is achievable with minimal utility loss, paving the way for controlled LLMs deployment in security-sensitive domains.
>
---
#### [new 053] Jailbreaking Large Language Models Through Content Concretization
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文提出一种名为“内容具体化”（CC）的新越狱技术，通过迭代优化生成恶意代码，提升大语言模型的越狱成功率。该研究旨在揭示当前LLM安全机制的漏洞，属于AI安全领域，解决了传统越狱方法成功率低的问题。**

- **链接: [http://arxiv.org/pdf/2509.12937v1](http://arxiv.org/pdf/2509.12937v1)**

> **作者:** Johan Wahréus; Ahmed Hussain; Panos Papadimitratos
>
> **备注:** Accepted for presentation in the Conference on Game Theory and AI for Security (GameSec) 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed for task automation and content generation, yet their safety mechanisms remain vulnerable to circumvention through different jailbreaking techniques. In this paper, we introduce \textit{Content Concretization} (CC), a novel jailbreaking technique that iteratively transforms abstract malicious requests into concrete, executable implementations. CC is a two-stage process: first, generating initial LLM responses using lower-tier, less constrained safety filters models, then refining them through higher-tier models that process both the preliminary output and original prompt. We evaluate our technique using 350 cybersecurity-specific prompts, demonstrating substantial improvements in jailbreak Success Rates (SRs), increasing from 7\% (no refinements) to 62\% after three refinement iterations, while maintaining a cost of 7.5\textcent~per prompt. Comparative A/B testing across nine different LLM evaluators confirms that outputs from additional refinement steps are consistently rated as more malicious and technically superior. Moreover, manual code analysis reveals that generated outputs execute with minimal modification, although optimal deployment typically requires target-specific fine-tuning. With eventual improved harmful code generation, these results highlight critical vulnerabilities in current LLM safety frameworks.
>
---
#### [new 054] LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LLMAP系统，解决多目标路线规划问题，结合LLM解析用户自然语言偏好，并采用MSGS算法优化路线。工作包括任务识别、偏好提取与多约束下的路径搜索，实验证明其性能优越。**

- **链接: [http://arxiv.org/pdf/2509.12273v1](http://arxiv.org/pdf/2509.12273v1)**

> **作者:** Liangqi Yuan; Dong-Jun Han; Christopher G. Brinton; Sabine Brunswicker
>
> **摘要:** The rise of large language models (LLMs) has made natural language-driven route planning an emerging research area that encompasses rich user objectives. Current research exhibits two distinct approaches: direct route planning using LLM-as-Agent and graph-based searching strategies. However, LLMs in the former approach struggle to handle extensive map data, while the latter shows limited capability in understanding natural language preferences. Additionally, a more critical challenge arises from the highly heterogeneous and unpredictable spatio-temporal distribution of users across the globe. In this paper, we introduce a novel LLM-Assisted route Planning (LLMAP) system that employs an LLM-as-Parser to comprehend natural language, identify tasks, and extract user preferences and recognize task dependencies, coupled with a Multi-Step Graph construction with iterative Search (MSGS) algorithm as the underlying solver for optimal route finding. Our multi-objective optimization approach adaptively tunes objective weights to maximize points of interest (POI) quality and task completion rate while minimizing route distance, subject to three key constraints: user time limits, POI opening hours, and task dependencies. We conduct extensive experiments using 1,000 routing prompts sampled with varying complexity across 14 countries and 27 cities worldwide. The results demonstrate that our approach achieves superior performance with guarantees across multiple constraints.
>
---
#### [new 055] DaSAThco: Data-Aware SAT Heuristics Combinations Optimization via Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DaSAThco框架，利用大语言模型结合问题原型，学习从实例特征到启发式组合的映射，解决SAT求解器配置优化问题，实现一次训练广泛适应，提升求解性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.12602v1](http://arxiv.org/pdf/2509.12602v1)**

> **作者:** Minyu Chen; Guoqiang Li
>
> **备注:** 11 pages
>
> **摘要:** The performance of Conflict-Driven Clause Learning solvers hinges on internal heuristics, yet the heterogeneity of SAT problems makes a single, universally optimal configuration unattainable. While prior automated methods can find specialized configurations for specific problem families, this dataset-specific approach lacks generalizability and requires costly re-optimization for new problem types. We introduce DaSAThco, a framework that addresses this challenge by learning a generalizable mapping from instance features to tailored heuristic ensembles, enabling a train-once, adapt-broadly model. Our framework uses a Large Language Model, guided by systematically defined Problem Archetypes, to generate a diverse portfolio of specialized heuristic ensembles and subsequently learns an adaptive selection mechanism to form the final mapping. Experiments show that DaSAThco achieves superior performance and, most notably, demonstrates robust out-of-domain generalization where non-adaptive methods show limitations. Our work establishes a more scalable and practical path toward automated algorithm design for complex, configurable systems.
>
---
#### [new 056] Textarium: Entangling Annotation, Abstraction and Argument
- **分类: cs.HC; cs.CL; H.5.2; H.5.4; I.7.1; J.5**

- **简介: 论文提出Textarium，一种结合标注、抽象与论证的在线阅读写作工具，旨在促进学术文本的深度解读与共享。通过可视化界面实现分析过程的透明化，解决传统阅读与写作分离的问题。**

- **链接: [http://arxiv.org/pdf/2509.13191v1](http://arxiv.org/pdf/2509.13191v1)**

> **作者:** Philipp Proff; Marian Dörk
>
> **备注:** This is the authors' version of the article presented at VIS4DH and published in the proceedings of IEEE VIS 2025
>
> **摘要:** We present a web-based environment that connects annotation, abstraction, and argumentation during the interpretation of text. As a visual interface for scholarly reading and writing, Textarium combines human analysis with lightweight computational processing to bridge close and distant reading practices. Readers can highlight text, group keywords into concepts, and embed these observations as anchors in essays. The interface renders these interpretive actions as parameterized visualization states. Through a speculative design process of co-creative and iterative prototyping, we developed a reading-writing approach that makes interpretive processes transparent and shareable within digital narratives.
>
---
#### [new 057] Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs
- **分类: cs.AI; cs.CL**

- **简介: 论文提出GRRAF方法，利用LLM生成代码查询图数据库，解决无需训练的图推理任务。该方法克服了传统方法依赖预定义算法或微调的限制，实现高准确率与高效性。**

- **链接: [http://arxiv.org/pdf/2509.12743v1](http://arxiv.org/pdf/2509.12743v1)**

> **作者:** Hanqing Li; Kiran Sheena Jyothi; Henry Liang; Sharika Mahadevan; Diego Klabjan
>
> **摘要:** We propose a new, training-free method, Graph Reasoning via Retrieval Augmented Framework (GRRAF), that harnesses retrieval-augmented generation (RAG) alongside the code-generation capabilities of large language models (LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target graph is stored in a graph database, and the LLM is prompted to generate executable code queries that retrieve the necessary information. This approach circumvents the limitations of existing methods that require extensive finetuning or depend on predefined algorithms, and it incorporates an error feedback loop with a time-out mechanism to ensure both correctness and efficiency. Experimental evaluations on the GraphInstruct dataset reveal that GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle detection, bipartite graph checks, shortest path computation, and maximum flow, while maintaining consistent token costs regardless of graph sizes. Imperfect but still very high performance is observed on subgraph matching. Notably, GRRAF scales effectively to large graphs with up to 10,000 nodes.
>
---
#### [new 058] Humor in Pixels: Benchmarking Large Multimodal Models Understanding of Online Comics
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出PixelHumor基准数据集，用于评估大语言模型对网络漫画中多模态幽默和叙事的理解能力。任务是测试模型在视觉与文本整合、叙事推理方面的表现，揭示当前模型在社会智能交互中的不足。**

- **链接: [http://arxiv.org/pdf/2509.12248v1](http://arxiv.org/pdf/2509.12248v1)**

> **作者:** Yuriel Ryan; Rui Yang Tan; Kenny Tsu Wei Choo; Roy Ka-Wei Lee
>
> **备注:** 27 pages, 8 figures, EMNLP 2025
>
> **摘要:** Understanding humor is a core aspect of social intelligence, yet it remains a significant challenge for Large Multimodal Models (LMMs). We introduce PixelHumor, a benchmark dataset of 2,800 annotated multi-panel comics designed to evaluate LMMs' ability to interpret multimodal humor and recognize narrative sequences. Experiments with state-of-the-art LMMs reveal substantial gaps: for instance, top models achieve only 61% accuracy in panel sequencing, far below human performance. This underscores critical limitations in current models' integration of visual and textual cues for coherent narrative and humor understanding. By providing a rigorous framework for evaluating multimodal contextual and narrative reasoning, PixelHumor aims to drive the development of LMMs that better engage in natural, socially aware interactions.
>
---
#### [new 059] Podcasts as a Medium for Participation in Collective Action: A Case Study of Black Lives Matter
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文研究播客作为集体行动参与的媒介，以BLM运动为例，分析其话语中的参与表达及情感维度。任务是探索音频格式中集体行动的表达方式，解决传统文本研究的不足，通过SPoRC语料库识别参与阶段与情绪关联，揭示不同阶段的情绪特征。**

- **链接: [http://arxiv.org/pdf/2509.13197v1](http://arxiv.org/pdf/2509.13197v1)**

> **作者:** Theodora Moldovan; Arianna Pera; Davide Vega; Luca Maria Aiello
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** We study how participation in collective action is articulated in podcast discussions, using the Black Lives Matter (BLM) movement as a case study. While research on collective action discourse has primarily focused on text-based content, this study takes a first step toward analyzing audio formats by using podcast transcripts. Using the Structured Podcast Research Corpus (SPoRC), we investigated spoken language expressions of participation in collective action, categorized as problem-solution, call-to-action, intention, and execution. We identified podcast episodes discussing racial justice after important BLM-related events in May and June of 2020, and extracted participatory statements using a layered framework adapted from prior work on social media. We examined the emotional dimensions of these statements, detecting eight key emotions and their association with varying stages of activism. We found that emotional profiles vary by stage, with different positive emotions standing out during calls-to-action, intention, and execution. We detected negative associations between collective action and negative emotions, contrary to theoretical expectations. Our work contributes to a better understanding of how activism is expressed in spoken digital discourse and how emotional framing may depend on the format of the discussion.
>
---
#### [new 060] Match Chat: Real Time Generative AI and Generative Computing for Tennis
- **分类: cs.AI; cs.CL**

- **简介: 论文提出Match Chat系统，结合生成式AI与生成计算技术，实时回答网球赛事相关问题。该系统旨在提升观赛体验，解决实时数据查询的准确性与响应速度问题，采用智能代理架构实现高并发下的稳定服务。**

- **链接: [http://arxiv.org/pdf/2509.12592v1](http://arxiv.org/pdf/2509.12592v1)**

> **作者:** Aaron Baughman; Gozde Akay; Eduardo Morales; Rahul Agarwal; Preetika Srivastava
>
> **备注:** 12 pages, 5 Figures, 4 Tables
>
> **摘要:** We present Match Chat, a real-time, agent-driven assistant designed to enhance the tennis fan experience by delivering instant, accurate responses to match-related queries. Match Chat integrates Generative Artificial Intelligence (GenAI) with Generative Computing (GenComp) techniques to synthesize key insights during live tennis singles matches. The system debuted at the 2025 Wimbledon Championships and the 2025 US Open, where it provided about 1 million users with seamless access to streaming and static data through natural language queries. The architecture is grounded in an Agent-Oriented Architecture (AOA) combining rule engines, predictive models, and agents to pre-process and optimize user queries before passing them to GenAI components. The Match Chat system had an answer accuracy of 92.83% with an average response time of 6.25 seconds under loads of up to 120 requests per second (RPS). Over 96.08% of all queries were guided using interactive prompt design, contributing to a user experience that prioritized clarity, responsiveness, and minimal effort. The system was designed to mask architectural complexity, offering a frictionless and intuitive interface that required no onboarding or technical familiarity. Across both Grand Slam deployments, Match Chat maintained 100% uptime and supported nearly 1 million unique users, underscoring the scalability and reliability of the platform. This work introduces key design patterns for real-time, consumer-facing AI systems that emphasize speed, precision, and usability that highlights a practical path for deploying performant agentic systems in dynamic environments.
>
---
#### [new 061] HARMONIC: A Content-Centric Cognitive Robotic Architecture
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 论文提出HARMONIC认知机器人架构，用于人机团队中的机器人。该架构支持语义感知、类人决策与语言交流，解决数据稀缺、可解释性与安全性问题。通过两个系统验证其在仿真与实体平台的有效性。属于机器人认知架构设计任务。**

- **链接: [http://arxiv.org/pdf/2509.13279v1](http://arxiv.org/pdf/2509.13279v1)**

> **作者:** Sanjay Oruganti; Sergei Nirenburg; Marjorie McShane; Jesse English; Michael K. Roberts; Christian Arndt; Carlos Gonzalez; Mingyo Seo; Luis Sentis
>
> **摘要:** This paper introduces HARMONIC, a cognitive-robotic architecture designed for robots in human-robotic teams. HARMONIC supports semantic perception interpretation, human-like decision-making, and intentional language communication. It addresses the issues of safety and quality of results; aims to solve problems of data scarcity, explainability, and safety; and promotes transparency and trust. Two proof-of-concept HARMONIC-based robotic systems are demonstrated, each implemented in both a high-fidelity simulation environment and on physical robotic platforms.
>
---
#### [new 062] Similarity-Distance-Magnitude Activations
- **分类: cs.LG; cs.CL**

- **简介: 论文提出一种改进的softmax激活函数——SDM，用于提升神经网络在面对分布偏移和高概率区域输入时的鲁棒性与可解释性。该方法结合相似性、距离和幅度感知，适用于语言模型的选择性分类任务。**

- **链接: [http://arxiv.org/pdf/2509.12760v1](http://arxiv.org/pdf/2509.12760v1)**

> **作者:** Allen Schmaltz
>
> **备注:** 17 pages, 5 tables, 1 algorithm. arXiv admin note: substantial text overlap with arXiv:2502.20167
>
> **摘要:** We introduce a more robust and interpretable formulation of the standard softmax activation function commonly used with neural networks by adding Similarity (i.e., correctly predicted depth-matches into training) awareness and Distance-to-training-distribution awareness to the existing output Magnitude (i.e., decision-boundary) awareness. When used as the final-layer activation with language models, the resulting Similarity-Distance-Magnitude (SDM) activation function is more robust than the softmax function to co-variate shifts and out-of-distribution inputs in high-probability regions, and provides interpretability-by-exemplar via dense matching. Complementing the prediction-conditional estimates, the SDM activation enables a partitioning of the class-wise empirical CDFs to guard against low class-wise recall among selective classifications. These properties make it preferable for selective classification, even when considering post-hoc calibration methods over the softmax.
>
---
#### [new 063] Exact Coset Sampling for Quantum Lattice Algorithms
- **分类: quant-ph; cs.CL; cs.CR**

- **简介: 该论文提出一种精确的陪集采样方法，用于修正量子晶格算法中争议的“域扩展”步骤，解决周期性与支持不匹配问题，构造可逆量子电路以实现模线性关系，提升算法正确性与效率。属于量子计算领域中的晶格算法优化任务。**

- **链接: [http://arxiv.org/pdf/2509.12341v1](http://arxiv.org/pdf/2509.12341v1)**

> **作者:** Yifan Zhang
>
> **备注:** Project Page: https://github.com/yifanzhang-pro/quantum-lattice
>
> **摘要:** We give a simple, fully correct, and assumption-light replacement for the contested "domain-extension" in Step 9 of a recent windowed-QFT lattice algorithm with complex-Gaussian windows~\citep{chen2024quantum}. The published Step~9 suffers from a periodicity/support mismatch. We present a pair-shift difference construction that coherently cancels all unknown offsets, produces an exact uniform CRT-coset state over $\mathbb{Z}_{P}$, and then uses the QFT to enforce the intended modular linear relation. The unitary is reversible, uses $\mathrm{poly}(\log M_2)$ gates, and preserves the algorithm's asymptotics. Project Page: https://github.com/yifanzhang-pro/quantum-lattice.
>
---
#### [new 064] Rethinking the Evaluation of Alignment Methods: Insights into Diversity, Generalisation, and Safety
- **分类: cs.LG; cs.CL**

- **简介: 该论文评估大语言模型对齐方法（如DPO、PPO等）在事实性、安全性和多样性等方面的性能，提出统一框架揭示其权衡关系，旨在指导更平衡可靠的LLM开发。**

- **链接: [http://arxiv.org/pdf/2509.12936v1](http://arxiv.org/pdf/2509.12936v1)**

> **作者:** Denis Janiak; Julia Moska; Dawid Motyka; Karolina Seweryn; Paweł Walkowiak; Bartosz Żuk; Arkadiusz Janz
>
> **摘要:** Large language models (LLMs) require careful alignment to balance competing objectives - factuality, safety, conciseness, proactivity, and diversity. Existing studies focus on individual techniques or specific dimensions, lacking a holistic assessment of the inherent trade-offs. We propose a unified evaluation framework that compares LLM alignment methods (PPO, DPO, ORPO, KTO) across these five axes, using both in-distribution and out-of-distribution datasets. Leveraging a specialized LLM-as-Judge prompt, validated through human studies, we reveal that DPO and KTO excel in factual accuracy, PPO and DPO lead in safety, and PPO best balances conciseness with proactivity. Our findings provide insights into trade-offs of common alignment methods, guiding the development of more balanced and reliable LLMs.
>
---
#### [new 065] Yet Another Watermark for Large Language Models
- **分类: cs.CR; cs.CL**

- **简介: 论文提出一种新型大语言模型水印方法，通过调整模型内部参数嵌入水印，实现黑盒场景下高效提取。该方法兼顾水印鲁棒性与隐蔽性，为LLM水印研究提供新视角。**

- **链接: [http://arxiv.org/pdf/2509.12574v1](http://arxiv.org/pdf/2509.12574v1)**

> **作者:** Siyuan Bao; Ying Shi; Zhiguang Yang; Hanzhou Wu; Xinpeng Zhang
>
> **备注:** https://scholar.google.com/citations?hl=en&user=IdiF7M0AAAAJ
>
> **摘要:** Existing watermarking methods for large language models (LLMs) mainly embed watermark by adjusting the token sampling prediction or post-processing, lacking intrinsic coupling with LLMs, which may significantly reduce the semantic quality of the generated marked texts. Traditional watermarking methods based on training or fine-tuning may be extendable to LLMs. However, most of them are limited to the white-box scenario, or very time-consuming due to the massive parameters of LLMs. In this paper, we present a new watermarking framework for LLMs, where the watermark is embedded into the LLM by manipulating the internal parameters of the LLM, and can be extracted from the generated text without accessing the LLM. Comparing with related methods, the proposed method entangles the watermark with the intrinsic parameters of the LLM, which better balances the robustness and imperceptibility of the watermark. Moreover, the proposed method enables us to extract the watermark under the black-box scenario, which is computationally efficient for use. Experimental results have also verified the feasibility, superiority and practicality. This work provides a new perspective different from mainstream works, which may shed light on future research.
>
---
#### [new 066] Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition
- **分类: cs.AI; cs.CL**

- **简介: 论文提出一种分解方法，通过结构化交互摘要和意图提取，提升小模型在资源受限设备上对用户意图的理解能力，解决其在准确意图推理上的不足，任务为意图识别。**

- **链接: [http://arxiv.org/pdf/2509.12423v1](http://arxiv.org/pdf/2509.12423v1)**

> **作者:** Danielle Cohen; Yoni Halpern; Noam Kahlon; Joel Oren; Omri Berkovitch; Sapir Caduri; Ido Dagan; Anatoly Efros
>
> **摘要:** Understanding user intents from UI interaction trajectories remains a challenging, yet crucial, frontier in intelligent agent development. While massive, datacenter-based, multi-modal large language models (MLLMs) possess greater capacity to handle the complexities of such sequences, smaller models which can run on-device to provide a privacy-preserving, low-cost, and low-latency user experience, struggle with accurate intent inference. We address these limitations by introducing a novel decomposed approach: first, we perform structured interaction summarization, capturing key information from each user action. Second, we perform intent extraction using a fine-tuned model operating on the aggregated summaries. This method improves intent understanding in resource-constrained models, even surpassing the base performance of large MLLMs.
>
---
#### [new 067] A Novel Recurrent Neural Network Framework for Prediction and Treatment of Oncogenic Mutation Progression
- **分类: cs.LG; cs.CL; q-bio.QM**

- **简介: 该论文提出一种基于RNN的AI框架，用于预测癌症突变进展并推荐治疗方案。通过分析TCGA数据，结合预处理与药物数据库，实现高效、低成本的癌症诊断与治疗建议，无需依赖传统湿实验。**

- **链接: [http://arxiv.org/pdf/2509.12732v1](http://arxiv.org/pdf/2509.12732v1)**

> **作者:** Rishab Parthasarathy; Achintya Bhowmik
>
> **备注:** 12 pages, 11 figures, work originally done in 2022/2023 and was awarded as one of the Regeneron Science Talent Search Finalists in 2022
>
> **摘要:** Despite significant medical advancements, cancer remains the second leading cause of death, with over 600,000 deaths per year in the US. One emerging field, pathway analysis, is promising but still relies on manually derived wet lab data, which is time-consuming to acquire. This work proposes an efficient, effective end-to-end framework for Artificial Intelligence (AI) based pathway analysis that predicts both cancer severity and mutation progression, thus recommending possible treatments. The proposed technique involves a novel combination of time-series machine learning models and pathway analysis. First, mutation sequences were isolated from The Cancer Genome Atlas (TCGA) Database. Then, a novel preprocessing algorithm was used to filter key mutations by mutation frequency. This data was fed into a Recurrent Neural Network (RNN) that predicted cancer severity. Then, the model probabilistically used the RNN predictions, information from the preprocessing algorithm, and multiple drug-target databases to predict future mutations and recommend possible treatments. This framework achieved robust results and Receiver Operating Characteristic (ROC) curves (a key statistical metric) with accuracies greater than 60%, similar to existing cancer diagnostics. In addition, preprocessing played an instrumental role in isolating important mutations, demonstrating that each cancer stage studied may contain on the order of a few-hundred key driver mutations, consistent with current research. Heatmaps based on predicted gene frequency were also generated, highlighting key mutations in each cancer. Overall, this work is the first to propose an efficient, cost-effective end-to-end framework for projecting cancer progression and providing possible treatments without relying on expensive, time-consuming wet lab work.
>
---
#### [new 068] The Adaptation Paradox: Agency vs. Mimicry in Companion Chatbots
- **分类: cs.HC; cs.CL; H.5.2; I.2.7**

- **简介: 论文探讨陪伴聊天机器人如何通过用户生成的虚拟形象和语言风格匹配建立真实连接。研究发现，用户生成形象提升亲密度，但自适应语言风格反而降低满意度，提出“适应悖论”并建议优先可见的用户驱动个性化设计。**

- **链接: [http://arxiv.org/pdf/2509.12525v1](http://arxiv.org/pdf/2509.12525v1)**

> **作者:** T. James Brandt; Cecilia Xi Wang
>
> **备注:** 31 pages, 17 figures, 2 tables. Submitted to CHI 2026 (under review). Preregistered: https://osf.io/f4h5b ; Code/Materials: https://doi.org/10.5281/zenodo.15801081
>
> **摘要:** Generative AI powers a growing wave of companion chatbots, yet principles for fostering genuine connection remain unsettled. We test two routes: visible user authorship versus covert language-style mimicry. In a preregistered 3x2 experiment (N = 162), we manipulated user-controlled avatar generation (none, premade, user-generated) and Language Style Matching (LSM) (static vs. adaptive). Generating an avatar boosted rapport ($\omega^2$ = .040, p = .013), whereas adaptive LSM underperformed static style on personalization and satisfaction (d = 0.35, p = .009) and was paradoxically judged less adaptive (t = 3.07, p = .003, d = 0.48). We term this an Adaptation Paradox: synchrony erodes connection when perceived as incoherent, destabilizing persona. To explain, we propose a stability-and-legibility account: visible authorship fosters natural interaction, while covert mimicry risks incoherence. Our findings suggest designers should prioritize legible, user-driven personalization and limit stylistic shifts rather than rely on opaque mimicry.
>
---
#### [new 069] WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出WebSailor-V2方法，旨在提升开放源代码模型在复杂信息检索任务中的能力，缩小与专有代理的差距。通过合成数据和可扩展强化学习，增强模型处理高不确定性的系统化推理能力。**

- **链接: [http://arxiv.org/pdf/2509.13305v1](http://arxiv.org/pdf/2509.13305v1)**

> **作者:** Kuan Li; Zhongwang Zhang; Huifeng Yin; Rui Ye; Yida Zhao; Liwen Zhang; Litu Ou; Dingchu Zhang; Xixi Wu; Jialong Wu; Xinyu Wang; Zile Qiao; Zhen Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all open-source agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.
>
---
## 更新

#### [replaced 001] Do Large Language Models Truly Grasp Addition? A Rule-Focused Diagnostic Using Two-Integer Arithmetic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05262v2](http://arxiv.org/pdf/2504.05262v2)**

> **作者:** Yang Yan; Yu Lu; Renjun Xu; Zhenzhong Lan
>
> **备注:** Accepted by EMNLP'25 Main
>
> **摘要:** Large language models (LLMs) achieve impressive results on advanced mathematics benchmarks but sometimes fail on basic arithmetic tasks, raising the question of whether they have truly grasped fundamental arithmetic rules or are merely relying on pattern matching. To unravel this issue, we systematically probe LLMs' understanding of two-integer addition (0 to $2^64$) by testing three crucial properties: commutativity (A+B=B+A), representation invariance via symbolic remapping (e.g., $7 -> Y$), and consistent accuracy scaling with operand length. Our evaluation of 12 leading LLMs reveals a stark disconnect: while models achieve high numeric accuracy (73.8-99.8%), they systematically fail these diagnostics. Specifically, accuracy plummets to <= 7.5% with symbolic inputs, commutativity is violated in up to 20% of cases, and accuracy scaling is non-monotonic. These findings demonstrate that current LLMs address elementary addition via pattern matching, not robust rule induction, motivating new diagnostic benchmarks and innovations in model architecture and training to cultivate genuine mathematical reasoning. Our dataset and generating code are available at https://github.com/kuri-leo/llm-arithmetic-diagnostic.
>
---
#### [replaced 002] AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection: Improving Model Performance by Span-Level Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07459v2](http://arxiv.org/pdf/2509.07459v2)**

> **作者:** Christian Rene Thelen; Patrick Gustav Blaneck; Tobias Bornheim; Niklas Grieger; Stephan Bialonski
>
> **备注:** 6 pages, 1 figure, 2 tables
>
> **摘要:** Positive, supportive online communication in social media (candy speech) has the potential to foster civility, yet automated detection of such language remains underexplored, limiting systematic analysis of its impact. We investigate how candy speech can be reliably detected in a 46k-comment German YouTube corpus by monolingual and multilingual language models, including GBERT, Qwen3 Embedding, and XLM-RoBERTa. We find that a multilingual XLM-RoBERTa-Large model trained to detect candy speech at the span level outperforms other approaches, ranking first in both binary positive F1: 0.8906) and categorized span-based detection (strict F1: 0.6307) subtasks at the GermEval 2025 Shared Task on Candy Speech Detection. We speculate that span-based training, multilingual capabilities, and emoji-aware tokenizers improved detection performance. Our results demonstrate the effectiveness of multilingual models in identifying positive, supportive language.
>
---
#### [replaced 003] Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11206v2](http://arxiv.org/pdf/2509.11206v2)**

> **作者:** Tae Soo Kim; Heechan Lee; Yoonjoo Lee; Joseph Seering; Juho Kim
>
> **备注:** The first two authors hold equal contribution
>
> **摘要:** Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior.
>
---
#### [replaced 004] Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19331v3](http://arxiv.org/pdf/2411.19331v3)**

> **作者:** Luca Barsellotti; Lorenzo Bianchi; Nicola Messina; Fabio Carrara; Marcella Cornia; Lorenzo Baraldi; Fabrizio Falchi; Rita Cucchiara
>
> **备注:** ICCV 2025
>
> **摘要:** Open-Vocabulary Segmentation (OVS) aims at segmenting images from free-form textual concepts without predefined training classes. While existing vision-language models such as CLIP can generate segmentation masks by leveraging coarse spatial information from Vision Transformers, they face challenges in spatial localization due to their global alignment of image and text features. Conversely, self-supervised visual models like DINO excel in fine-grained visual encoding but lack integration with language. To bridge this gap, we present Talk2DINO, a novel hybrid approach that combines the spatial accuracy of DINOv2 with the language understanding of CLIP. Our approach aligns the textual embeddings of CLIP to the patch-level features of DINOv2 through a learned mapping function without the need to fine-tune the underlying backbones. At training time, we exploit the attention maps of DINOv2 to selectively align local visual patches with textual embeddings. We show that the powerful semantic and localization abilities of Talk2DINO can enhance the segmentation process, resulting in more natural and less noisy segmentations, and that our approach can also effectively distinguish foreground objects from the background. Experimental results demonstrate that Talk2DINO achieves state-of-the-art performance across several unsupervised OVS benchmarks. Source code and models are publicly available at: https://lorebianchi98.github.io/Talk2DINO/.
>
---
#### [replaced 005] Executable Ontologies: Synthesizing Event Semantics with Dataflow Architecture
- **分类: cs.AI; cs.CL; cs.FL; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.09775v2](http://arxiv.org/pdf/2509.09775v2)**

> **作者:** Aleksandr Boldachev
>
> **备注:** 22 pages, 6 figures. Corrected captions on Figure 4
>
> **摘要:** This paper presents boldsea, Boldachev's semantic-event approach -- an architecture for modeling complex dynamic systems using executable ontologies -- semantic models that act as dynamic structures, directly controlling process execution. We demonstrate that integrating event semantics with a dataflow architecture addresses the limitations of traditional Business Process Management (BPM) systems and object-oriented semantic technologies. The paper presents the formal BSL (boldsea Semantic Language), including its BNF grammar, and outlines the boldsea-engine's architecture, which directly interprets semantic models as executable algorithms without compilation. It enables the modification of event models at runtime, ensures temporal transparency, and seamlessly merges data and business logic within a unified semantic framework.
>
---
#### [replaced 006] Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.01129v5](http://arxiv.org/pdf/2404.01129v5)**

> **作者:** Bohao Yang; Kun Zhao; Dong Liu; Chen Tang; Liang Zhan; Chenghua Lin
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Automatic open-domain dialogue evaluation has attracted increasing attention, yet remains challenging due to the complexity of assessing response appropriateness. Traditional evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with contexts. However, adversarial negative responses, despite possessing high lexical overlap with contexts, can be semantically incongruous. Consequently, existing metrics struggle to effectively evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in handling adversarial negative examples. We propose a novel evaluation framework that integrates Abstract Meaning Representation (AMR) enhanced domain-specific language models (SLMs) with LLMs. Our SLMs explicitly incorporate AMR graph information through a gating mechanism for enhanced semantic representation learning, while both SLM predictions and AMR knowledge are integrated into LLM prompts for robust evaluation. Extensive experiments on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to state-of-the-art baselines. Our comprehensive ablation studies reveal that AMR graph information contributes substantially more to performance improvements. Our framework achieves strong correlations with human judgments across multiple datasets, establishing a new benchmark for dialogue evaluation. Our code and data are publicly available.
>
---
#### [replaced 007] Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.08045v3](http://arxiv.org/pdf/2502.08045v3)**

> **作者:** Mohsinul Kabir; Ajwad Abrar; Sophia Ananiadou
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
>
---
#### [replaced 008] SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning
- **分类: eess.SP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19668v3](http://arxiv.org/pdf/2502.19668v3)**

> **作者:** Mingsheng Cai; Jiuming Jiang; Wenhao Huang; Che Liu; Rossella Arcucci
>
> **备注:** Findings of The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) is critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME is pre-trained using structured diagnostic labels derived from ECG report entities through a one-time offline extraction with Large Language Models (LLMs), which help denoise, standardize cardiac concepts, and improve clinical representation learning. By fusing ECG signals with textual cardiac queries instead of fixed labels, SuPreME enables zero-shot classification of unseen conditions without further fine-tuning. We evaluate SuPreME on six downstream datasets covering 106 cardiac conditions, achieving superior zero-shot AUC performance of $77.20\%$, surpassing state-of-the-art eSSLs by $4.98\%$. Results demonstrate SuPreME's effectiveness in leveraging structured, clinically relevant knowledge for high-quality ECG representations.
>
---
#### [replaced 009] Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04655v2](http://arxiv.org/pdf/2509.04655v2)**

> **作者:** Ayush Gupta; Ramneet Kaur; Anirban Roy; Adam D. Cobb; Rama Chellappa; Susmit Jha
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives.
>
---
#### [replaced 010] Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17088v3](http://arxiv.org/pdf/2506.17088v3)**

> **作者:** Jiahao Cheng; Tiancheng Su; Jia Yuan; Guoxiu He; Jiawei Liu; Xinqi Tao; Jingwen Xie; Huaxia Li
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) often exhibit \textit{hallucinations}, generating factually incorrect or semantically irrelevant content in response to prompts. Chain-of-Thought (CoT) prompting can mitigate hallucinations by encouraging step-by-step reasoning, but its impact on hallucination detection remains underexplored. To bridge this gap, we conduct a systematic empirical evaluation. We begin with a pilot experiment, revealing that CoT reasoning significantly affects the LLM's internal states and token probability distributions. Building on this, we evaluate the impact of various CoT prompting methods on mainstream hallucination detection methods across both instruction-tuned and reasoning-oriented LLMs. Specifically, we examine three key dimensions: changes in hallucination score distributions, variations in detection accuracy, and shifts in detection confidence. Our findings show that while CoT prompting helps reduce hallucination frequency, it also tends to obscure critical signals used for detection, impairing the effectiveness of various detection methods. Our study highlights an overlooked trade-off in the use of reasoning. Code is publicly available at: https://github.com/ECNU-Text-Computing/cot-hallu-detect .
>
---
#### [replaced 011] VARCO-VISION-2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10105v2](http://arxiv.org/pdf/2509.10105v2)**

> **作者:** Young-rok Cha; Jeongho Ju; SunYoung Park; Jong-Hyeon Lee; Younghyun Yu; Youngjune Kim
>
> **备注:** 19 pages, 1 figure, 14 tables. Technical report for VARCO-VISION-2.0, a Korean-English bilingual VLM in 14B and 1.7B variants. Key features: multi-image understanding, OCR with text localization, improved Korean capabilities
>
> **摘要:** We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model.
>
---
#### [replaced 012] UniversalCEFR: Enabling Open Multilingual Research on Language Proficiency Assessment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01419v2](http://arxiv.org/pdf/2506.01419v2)**

> **作者:** Joseph Marvin Imperial; Abdullah Barayan; Regina Stodden; Rodrigo Wilkens; Ricardo Munoz Sanchez; Lingyun Gao; Melissa Torgbi; Dawn Knight; Gail Forey; Reka R. Jablonkai; Ekaterina Kochmar; Robert Reynolds; Eugénio Ribeiro; Horacio Saggion; Elena Volodina; Sowmya Vajjala; Thomas François; Fernando Alva-Manchego; Harish Tayyar Madabushi
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** We introduce UniversalCEFR, a large-scale multilingual and multidimensional dataset of texts annotated with CEFR (Common European Framework of Reference) levels in 13 languages. To enable open research in automated readability and language proficiency assessment, UniversalCEFR comprises 505,807 CEFR-labeled texts curated from educational and learner-oriented resources, standardized into a unified data format to support consistent processing, analysis, and modelling across tasks and languages. To demonstrate its utility, we conduct benchmarking experiments using three modelling paradigms: a) linguistic feature-based classification, b) fine-tuning pre-trained LLMs, and c) descriptor-based prompting of instruction-tuned LLMs. Our results support using linguistic features and fine-tuning pretrained models in multilingual CEFR level assessment. Overall, UniversalCEFR aims to establish best practices in data distribution for language proficiency research by standardising dataset formats, and promoting their accessibility to the global research community.
>
---
#### [replaced 013] Teaching Your Models to Understand Code via Focal Preference Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02783v3](http://arxiv.org/pdf/2503.02783v3)**

> **作者:** Jie Wu; Haoling Li; Xin Zhang; Jianwen Luo; Yangyu Huang; Ruihang Chu; Yujiu Yang; Scarlett Li
>
> **备注:** Accepted by EMNLP'25
>
> **摘要:** Preference learning extends the performance of Code LLMs beyond traditional supervised fine-tuning by leveraging relative quality comparisons. In existing approaches, a set of n candidate solutions is evaluated based on test case success rates, with the candidate demonstrating a higher pass rate being labeled as positive and its counterpart with a lower pass rate as negative. However, because this approach aligns entire failing code blocks rather than pinpointing specific errors, it lacks the granularity necessary to capture meaningful error-correction relationships. As a result, the model is unable to learn more informative error-correction patterns. To address these issues, we propose Target-DPO, a new preference alignment framework that mimics human iterative debugging to refine Code LLMs. Target-DPO explicitly locates error regions and aligns the corresponding tokens via a tailored DPO algorithm. To facilitate it, we introduce the CodeFlow dataset, where samples are iteratively refined until passing tests, with modifications capturing error corrections. Extensive experiments show that a diverse suite of Code LLMs equipped with Target-DPO achieves significant performance gains in code generation and improves on challenging tasks like BigCodeBench. In-depth analysis reveals that Target-DPO yields fewer errors. Code, model and datasets are in: https://github.com/JieWu02/Target-DPO.
>
---
#### [replaced 014] JoPA:Explaining Large Language Model's Generation via Joint Prompt Attribution
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.20404v3](http://arxiv.org/pdf/2405.20404v3)**

> **作者:** Yurui Chang; Bochuan Cao; Yujia Wang; Jinghui Chen; Lu Lin
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performances in complex text generation tasks. However, the contribution of the input prompt to the generated content still remains obscure to humans, underscoring the necessity of understanding the causality between input and output pairs. Existing works for providing prompt-specific explanation often confine model output to be classification or next-word prediction. Few initial attempts aiming to explain the entire language generation often treat input prompt texts independently, ignoring their combinatorial effects on the follow-up generation. In this study, we introduce a counterfactual explanation framework based on Joint Prompt Attribution, JoPA, which aims to explain how a few prompt texts collaboratively influences the LLM's complete generation. Particularly, we formulate the task of prompt attribution for generation interpretation as a combinatorial optimization problem, and introduce a probabilistic algorithm to search for the casual input combination in the discrete space. We define and utilize multiple metrics to evaluate the produced explanations, demonstrating both the faithfulness and efficiency of our framework.
>
---
#### [replaced 015] How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12769v3](http://arxiv.org/pdf/2502.12769v3)**

> **作者:** Saad Obaid ul Islam; Anne Lauscher; Goran Glavaš
>
> **备注:** EMNLP 2025
>
> **摘要:** In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
>
---
#### [replaced 016] Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15805v2](http://arxiv.org/pdf/2505.15805v2)**

> **作者:** Hwan Chang; Yumin Kim; Yonghyun Jun; Hwanhee Lee
>
> **备注:** EMNLP 2025 (Main Conference)
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.
>
---
#### [replaced 017] Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05179v3](http://arxiv.org/pdf/2503.05179v3)**

> **作者:** Simon A. Aytes; Jinheon Baek; Sung Ju Hwang
>
> **备注:** EMNLP 2025
>
> **摘要:** Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
>
---
#### [replaced 018] Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22388v3](http://arxiv.org/pdf/2503.22388v3)**

> **作者:** Zhiyu Yang; Shuo Wang; Yukun Yan; Yang Deng
>
> **备注:** Accepted at EMNLP 2025 Main, Oral
>
> **摘要:** LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future. DSDBench is publicly available at github.com/KevinCL16/DSDBench.
>
---
#### [replaced 019] Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19081v3](http://arxiv.org/pdf/2507.19081v3)**

> **作者:** Hao Li; Yizheng Sun; Viktor Schlegel; Kailai Yang; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Preprint
>
> **摘要:** Argument summarization aims to generate concise, structured representations of complex, multi-perspective debates. While recent work has advanced the identification and clustering of argumentative components, the generation stage remains underexplored. Existing approaches typically rely on single-pass generation, offering limited support for factual correction or structural refinement. To address this gap, we introduce Arg-LLaDA, a novel large language diffusion framework that iteratively improves summaries via sufficiency-guided remasking and regeneration. Our method combines a flexible masking controller with a sufficiency-checking module to identify and revise unsupported, redundant, or incomplete spans, yielding more faithful, concise, and coherent outputs. Empirical results on two benchmark datasets demonstrate that Arg-LLaDA surpasses state-of-the-art baselines in 7 out of 10 automatic evaluation metrics. In addition, human evaluations reveal substantial improvements across core dimensions, coverage, faithfulness, and conciseness, validating the effectiveness of our iterative, sufficiency-aware generation strategy.
>
---
#### [replaced 020] Benchmarking Gender and Political Bias in Large Language Models
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06164v2](http://arxiv.org/pdf/2509.06164v2)**

> **作者:** Jinrui Yang; Xudong Han; Timothy Baldwin
>
> **摘要:** We introduce EuroParlVote, a novel benchmark for evaluating large language models (LLMs) in politically sensitive contexts. It links European Parliament debate speeches to roll-call vote outcomes and includes rich demographic metadata for each Member of the European Parliament (MEP), such as gender, age, country, and political group. Using EuroParlVote, we evaluate state-of-the-art LLMs on two tasks -- gender classification and vote prediction -- revealing consistent patterns of bias. We find that LLMs frequently misclassify female MEPs as male and demonstrate reduced accuracy when simulating votes for female speakers. Politically, LLMs tend to favor centrist groups while underperforming on both far-left and far-right ones. Proprietary models like GPT-4o outperform open-weight alternatives in terms of both robustness and fairness. We release the EuroParlVote dataset, code, and demo to support future research on fairness and accountability in NLP within political contexts.
>
---
#### [replaced 021] Game-RL: Synthesizing Verifiable Game Tasks at Scale to Boost VLMs General Reasoning
- **分类: cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.13886v4](http://arxiv.org/pdf/2505.13886v4)**

> **作者:** Jingqi Tong; Jixin Tang; Hangcheng Li; Yurong Mou; Ming Zhang; Jun Zhao; Yanbo Wen; Fan Song; Jiahao Zhan; Yuyang Lu; Chaoran Tao; Zhiyuan Guo; Jizhou Yu; Tianhao Cheng; Changhao Jiang; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Weifeng Ge; Guanhua Chen; Tao Gui; Xipeng Qiu; Qi Zhang; Xuanjing Huang
>
> **备注:** 63 pages, 23 figures, submitted to NeurIPS 2025
>
> **摘要:** Real-world vision language reasoning scenarios often include diverse and complex tasks. However, vision language reinforcement learning has primarily focused on a narrow set of tasks (e.g. geometry or chart reasoning), limiting the improvement of Vision Language Models' (VLMs) general reasoning. Therefore, we propose a novel Code2Logic approach, using Large Language Models (LLMs) to synthesize verifiable game reasoning tasks at scale via adapting game code. Using the Code2Logic, we developed the GameQA dataset to train and evaluate VLMs. GameQA is verifiable and scalable, offers controllable difficulty gradation and is diverse with 30 games and 158 tasks. Then we apply Game-RL, which is simple reinforcement learning on GameQA. Surprisingly, despite training solely on game tasks, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33% across 7 diverse vision-language benchmarks. Our code, dataset and models are available at the GitHub repository.
>
---
#### [replaced 022] Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.09689v4](http://arxiv.org/pdf/2411.09689v4)**

> **作者:** Seongmin Lee; Hsiang Hsu; Chun-Fu Chen; Duen Horng Chau
>
> **备注:** 22 pages, 15 figures
>
> **摘要:** LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
>
---
#### [replaced 023] LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20999v3](http://arxiv.org/pdf/2507.20999v3)**

> **作者:** Yining Huang; Bin Li; Keke Tang; Meilian Chen
>
> **备注:** 12 pages
>
> **摘要:** Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
>
---
#### [replaced 024] GTA: Supervised-Guided Reinforcement Learning for Text Classification with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12108v2](http://arxiv.org/pdf/2509.12108v2)**

> **作者:** Min Zeng; Jingfei Sun; Xueyou Luo; Caiquan Liu; Shiqi Zhang; Li Xie; Xiaoxin Chen
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** In natural language processing tasks, pure reinforcement learning (RL) fine-tuning methods often suffer from inefficient exploration and slow convergence; while supervised fine-tuning (SFT) methods, although efficient in training, have limited performance ceiling and less solid theoretical foundation compared to RL. To address efficiency-capability trade-off, we propose the Guess-Think-Answer (GTA) framework that combines the efficiency of SFT with the capability gains of RL in a unified training paradigm. GTA works by having the model first produce a provisional guess (optimized via cross-entropy loss), then reflect on this guess before generating the final answer, with RL rewards shaping both the final output and the format of the entire GTA structure. This hybrid approach achieves both faster convergence than pure RL and higher performance ceiling than pure SFT. To mitigate gradient conflicts between the two training signals, we employ loss masking and gradient constraints. Empirical results on four text classification benchmarks demonstrate that GTA substantially accelerates convergence while outperforming both standalone SFT and RL baselines.
>
---
#### [replaced 025] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05983v2](http://arxiv.org/pdf/2509.05983v2)**

> **作者:** Minh N. H. Nguyen; Anh Nguyen Tran; Dung Truong Dinh; Nam Van Vo
>
> **备注:** I need to withdraw the paper as there something wrong
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios.
>
---
#### [replaced 026] Evaluating the Robustness of Open-Source Vision-Language Models to Domain Shift in Object Captioning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.19579v2](http://arxiv.org/pdf/2506.19579v2)**

> **作者:** Federico Tavella; Amber Drinkwater; Angelo Cangelosi
>
> **摘要:** Vision-Language Models (VLMs) have emerged as powerful tools for generating textual descriptions from visual data. While these models excel on web-scale datasets, their robustness to the domain shifts inherent in many real-world applications remains under-explored. This paper presents a systematic evaluation of VLM performance on a single-view object captioning task when faced with a controlled, physical domain shift. We compare captioning accuracy across two distinct object sets: a collection of multi-material, real-world tools and a set of single-material, 3D-printed items. The 3D-printed set introduces a significant domain shift in texture and material properties, challenging the models' generalization capabilities. Our quantitative results demonstrate that all tested VLMs show a marked performance degradation when describing the 3D-printed objects compared to the real-world tools. This underscores a critical limitation in the ability of current models to generalize beyond surface-level features and highlights the need for more robust architectures for real-world signal processing applications.
>
---
#### [replaced 027] Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11177v2](http://arxiv.org/pdf/2509.11177v2)**

> **作者:** Hang Guo; Yawei Li; Luca Benini
>
> **备注:** Preprint
>
> **摘要:** Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
>
---
#### [replaced 028] Dynamic Relation Inference via Verb Embeddings
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13021v2](http://arxiv.org/pdf/2503.13021v2)**

> **作者:** Omri Suissa; Muhiim Ali; Ariana Azarbal; Hui Shen; Shekhar Pradhan
>
> **摘要:** CLIP has demonstrated exceptional image-text matching capabilities due to its training on contrastive learning tasks. Past research has suggested that whereas CLIP effectively matches text to images when the matching can be achieved just by matching the text with the objects in the image, CLIP struggles when the matching depends on representing the relationship among the objects in the images (i.e., inferring relations). Previous attempts to address this limitation by training CLIP on relation detection datasets with only linguistic supervision have met with limited success. In this paper, we offer insights and practical methods to advance the field of relation inference from images. This paper approaches the task of creating a model that effectively detects relations among the objects in images by producing text and image embeddings that capture relationships through linguistic supervision. To this end, we propose Dynamic Relation Inference via Verb Embeddings (DRIVE), which augments the COCO dataset, fine-tunes CLIP with hard negatives subject-relation-object triples and corresponding images, and introduces a novel loss function to improve relation detection. Evaluated on multiple CLIP-based models, our method significantly improves zero-shot relation inference accuracy in both frozen and fine-tuned settings, significantly outperforming CLIP and state-of-the-art models while generalizing well on unseen data.
>
---
#### [replaced 029] The Belief State Transformer
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.23506v3](http://arxiv.org/pdf/2410.23506v3)**

> **作者:** Edward S. Hu; Kwangjun Ahn; Qinghua Liu; Haoran Xu; Manan Tomar; Ada Langford; Jayden Teoh; Bryon Xu; David Yan; Dinesh Jayaraman; Alex Lamb; John Langford
>
> **备注:** Updated report with new improvements and authors
>
> **摘要:** We introduce the "Belief State Transformer", a next-token predictor that takes both a prefix and suffix as inputs, with a novel objective of predicting both the next token for the prefix and the previous token for the suffix. The Belief State Transformer effectively learns to solve challenging problems that conventional forward-only transformers struggle with, in a domain-independent fashion. Key to this success is learning a compact belief state that captures all relevant information necessary for accurate predictions. Empirical ablations show that each component of the model is essential in difficult scenarios where standard Transformers fall short. For the task of story writing with known prefixes and suffixes, our approach outperforms the Fill-in-the-Middle method for reaching known goals and demonstrates improved performance even when the goals are unknown. Altogether, the Belief State Transformer enables more efficient goal-conditioned decoding, better test-time inference, and high-quality text representations on small scale problems. Website: https://edwhu.github.io/bst-website
>
---
#### [replaced 030] From Understanding to Generation: An Efficient Shortcut for Evaluating Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03592v2](http://arxiv.org/pdf/2506.03592v2)**

> **作者:** Viktor Hangya; Fabian Küch; Darina Gold
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Iterative evaluation of LLMs during training is essential to ensure expected capability development, but can be time- and compute-intensive. While NLU tasks, where the model selects from fixed answer choices, are cheap to evaluate, essential capabilities like reasoning and code generation rely on the more time-consuming NLG (token-by-token generation) format. In this work, our aim is to decrease the computational burden of NLG benchmarks in order to enable monitoring crucial LLM capabilities during model training. We reformulate generative tasks into computationally cheaper NLU alternatives. We test the performance correlation between the original and reformulated tasks using 8 LMs of various sizes and 4 capabilities: mathematical reasoning, code generation, factual knowledge and reading comprehension. Our results show a strong correlation between task formats, supporting capability assessment via cheaper alternatives and achieving over 35x average reduction in evaluation time. Our project is available at: https://github.com/Fraunhofer-IIS/EvalShortcut
>
---
#### [replaced 031] GPT-4.1 Sets the Standard in Automated Experiment Design Using Novel Python Libraries
- **分类: cs.SE; cs.AI; cs.CL; 68T50; I.2.2; I.2.7; D.2.3**

- **链接: [http://arxiv.org/pdf/2508.00033v2](http://arxiv.org/pdf/2508.00033v2)**

> **作者:** Nuno Fachada; Daniel Fernandes; Carlos M. Fernandes; Bruno D. Ferreira-Saraiva; João P. Matos-Carvalho
>
> **备注:** The peer-reviewed version of this paper is published in Future Internet at https://doi.org/10.3390/fi17090412. This version is typeset by the author and differs only in pagination and typographical detail
>
> **摘要:** Large Language Models (LLMs) have advanced rapidly as tools for automating code generation in scientific research, yet their ability to interpret and use unfamiliar Python APIs for complex computational experiments remains poorly characterized. This study systematically benchmarks a selection of state-of-the-art LLMs in generating functional Python code for two increasingly challenging scenarios: conversational data analysis with the \textit{ParShift} library, and synthetic data generation and clustering using \textit{pyclugen} and \textit{scikit-learn}. Both experiments use structured, zero-shot prompts specifying detailed requirements but omitting in-context examples. Model outputs are evaluated quantitatively for functional correctness and prompt compliance over multiple runs, and qualitatively by analyzing the errors produced when code execution fails. Results show that only a small subset of models consistently generate correct, executable code. GPT-4.1 achieved a 100\% success rate across all runs in both experimental tasks, whereas most other models succeeded in fewer than half of the runs, with only Grok-3 and Mistral-Large approaching comparable performance. In addition to benchmarking LLM performance, this approach helps identify shortcomings in third-party libraries, such as unclear documentation or obscure implementation bugs. Overall, these findings highlight current limitations of LLMs for end-to-end scientific automation and emphasize the need for careful prompt design, comprehensive library documentation, and continued advances in language model capabilities.
>
---
#### [replaced 032] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13773v2](http://arxiv.org/pdf/2507.13773v2)**

> **作者:** Pu Jian; Donglei Yu; Wen Yang; Shuo Ren; Jiajun Zhang
>
> **备注:** ACL2025 Main (SAC Highlight Award)
>
> **摘要:** In visual question answering (VQA) context, users often pose ambiguous questions to visual language models (VLMs) due to varying expression habits. Existing research addresses such ambiguities primarily by rephrasing questions. These approaches neglect the inherently interactive nature of user interactions with VLMs, where ambiguities can be clarified through user feedback. However, research on interactive clarification faces two major challenges: (1) Benchmarks are absent to assess VLMs' capacity for resolving ambiguities through interaction; (2) VLMs are trained to prefer answering rather than asking, preventing them from seeking clarification. To overcome these challenges, we introduce \textbf{ClearVQA} benchmark, which targets three common categories of ambiguity in VQA context, and encompasses various VQA scenarios.
>
---
#### [replaced 033] ToM-SSI: Evaluating Theory of Mind in Situated Social Interactions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.05066v2](http://arxiv.org/pdf/2509.05066v2)**

> **作者:** Matteo Bortoletto; Constantin Ruhdorfer; Andreas Bulling
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** Most existing Theory of Mind (ToM) benchmarks for foundation models rely on variations of the Sally-Anne test, offering only a very limited perspective on ToM and neglecting the complexity of human social interactions. To address this gap, we propose ToM-SSI: a new benchmark specifically designed to test ToM capabilities in environments rich with social interactions and spatial dynamics. While current ToM benchmarks are limited to text-only or dyadic interactions, ToM-SSI is multimodal and includes group interactions of up to four agents that communicate and move in situated environments. This unique design allows us to study, for the first time, mixed cooperative-obstructive settings and reasoning about multiple agents' mental state in parallel, thus capturing a wider range of social cognition than existing benchmarks. Our evaluations reveal that the current models' performance is still severely limited, especially in these new tasks, highlighting critical gaps for future research.
>
---
#### [replaced 034] Cutting Through the Noise: Boosting LLM Performance on Math Word Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.15444v5](http://arxiv.org/pdf/2406.15444v5)**

> **作者:** Ujjwala Anantheswaran; Himanshu Gupta; Kevin Scaria; Shreyas Verma; Chitta Baral; Swaroop Mishra
>
> **备注:** Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
>
> **摘要:** Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
>
---
#### [replaced 035] HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16281v3](http://arxiv.org/pdf/2505.16281v3)**

> **作者:** Shijie Zhang; Renhao Li; Songsheng Wang; Philipp Koehn; Min Yang; Derek F. Wong
>
> **摘要:** The advancement of Large Language Models (LLMs) enables flexible and interpretable automatic evaluations. In the field of machine translation evaluation, utilizing LLMs with translation error annotations based on Multidimensional Quality Metrics (MQM) yields more human-aligned judgments. However, current LLM-based evaluation methods still face challenges in accurately identifying error spans and assessing their severity. In this paper, we propose HiMATE, a Hierarchical Multi-Agent Framework for Machine Translation Evaluation. We argue that existing approaches inadequately exploit the fine-grained structural and semantic information within the MQM hierarchy. To address this, we develop a hierarchical multi-agent system grounded in the MQM error typology, enabling granular evaluation of subtype errors. Two key strategies are incorporated to further mitigate systemic hallucinations within the framework: the utilization of the model's self-reflection capability and the facilitation of agent discussion involving asymmetric information. Empirically, HiMATE outperforms competitive baselines across different datasets in conducting human-aligned evaluations. Further analyses underscore its significant advantage in error span detection and severity assessment, achieving an average F1-score improvement of 89% over the best-performing baseline. We make our code and data publicly available at https://github.com/nlp2ct-shijie/HiMATE.
>
---
#### [replaced 036] Responsible AI in NLP: GUS-Net Span-Level Bias Detection Dataset and Benchmark for Generalizations, Unfairness, and Stereotypes
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.08388v5](http://arxiv.org/pdf/2410.08388v5)**

> **作者:** Maximus Powers; Shaina Raza; Alex Chang; Rehana Riaz; Umang Mavani; Harshitha Reddy Jonala; Ansh Tiwari; Hua Wei
>
> **摘要:** Representational harms in language technologies often occur in short spans within otherwise neutral text, where phrases may simultaneously convey generalizations, unfairness, or stereotypes. Framing bias detection as sentence-level classification obscures which words carry bias and what type is present, limiting both auditability and targeted mitigation. We introduce the GUS-Net Framework, comprising the GUS dataset and a multi-label token-level detector for span-level analysis of social bias. The GUS dataset contains 3,739 unique snippets across multiple domains, with over 69,000 token-level annotations. Each token is labeled using BIO tags (Begin, Inside, Outside) for three pathways of representational harm: Generalizations, Unfairness, and Stereotypes. To ensure reliable data annotation, we employ an automated multi-agent pipeline that proposes candidate spans which are subsequently verified and corrected by human experts. We formulate bias detection as multi-label token-level classification and benchmark both encoder-based models (e.g., BERT family variants) and decoder-based large language models (LLMs). Our evaluations cover token-level identification and span-level entity recognition on our test set, and out-of-distribution generalization. Empirical results show that encoder-based models consistently outperform decoder-based baselines on nuanced and overlapping spans while being more computationally efficient. The framework delivers interpretable, fine-grained diagnostics that enable systematic auditing and mitigation of representational harms in real-world NLP systems.
>
---
#### [replaced 037] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13061v4](http://arxiv.org/pdf/2502.13061v4)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Weizhe Lin; Bill Byrne
>
> **备注:** EMNLP 2025 Main (Oral)
>
> **摘要:** Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL
>
---
#### [replaced 038] Counterfactual Simulatability of LLM Explanations for Generation Tasks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21740v2](http://arxiv.org/pdf/2505.21740v2)**

> **作者:** Marvin Limpijankit; Yanda Chen; Melanie Subbiah; Nicholas Deas; Kathleen McKeown
>
> **摘要:** LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
>
---
#### [replaced 039] PatentScore: Multi-dimensional Evaluation of LLM-Generated Patent Claims
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19345v2](http://arxiv.org/pdf/2505.19345v2)**

> **作者:** Yongmin Yoo; Qiongkai Xu; Longbing Cao
>
> **摘要:** High-stakes texts such as patent claims, medical records, and technical reports are structurally complex and demand a high degree of reliability and precision. While large language models (LLMs) have recently been applied to automate their generation in high-stakes domains, reliably evaluating such outputs remains a major challenge. Conventional natural language generation (NLG) metrics are effective for generic documents but fail to capture the structural and legal characteristics essential to evaluating complex high-stakes documents. To address this gap, we propose PatentScore, a multi-dimensional evaluation framework specifically designed for one of the most intricate and rigorous domains, patent claims. PatentScore integrates hierarchical decomposition of claim elements, validation patterns grounded in legal and technical standards, and scoring across structural, semantic, and legal dimensions. In experiments on our dataset which consists of 400 Claim1, PatentScore achieved the highest correlation with expert annotations ($r = 0.819$), significantly outperforming widely used NLG metrics. This work establishes a new standard for evaluating LLM-generated patent claims, providing a solid foundation for research on patent generation and validation.
>
---
#### [replaced 040] OpenWHO: A Document-Level Parallel Corpus for Health Translation in Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16048v2](http://arxiv.org/pdf/2508.16048v2)**

> **作者:** Raphaël Merx; Hanna Suominen; Trevor Cohn; Ekaterina Vylomova
>
> **备注:** Accepted at WMT 2025
>
> **摘要:** In machine translation (MT), health is a high-stakes domain characterised by widespread deployment and domain-specific vocabulary. However, there is a lack of MT evaluation datasets for low-resource languages in this domain. To address this gap, we introduce OpenWHO, a document-level parallel corpus of 2,978 documents and 26,824 sentences from the World Health Organization's e-learning platform. Sourced from expert-authored, professionally translated materials shielded from web-crawling, OpenWHO spans a diverse range of over 20 languages, of which nine are low-resource. Leveraging this new resource, we evaluate modern large language models (LLMs) against traditional MT models. Our findings reveal that LLMs consistently outperform traditional MT models, with Gemini 2.5 Flash achieving a +4.79 ChrF point improvement over NLLB-54B on our low-resource test set. Further, we investigate how LLM context utilisation affects accuracy, finding that the benefits of document-level translation are most pronounced in specialised domains like health. We release the OpenWHO corpus to encourage further research into low-resource MT in the health domain.
>
---
#### [replaced 041] A funny companion: Distinct neural responses to perceived AI- versus human-generated humor
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.10847v2](http://arxiv.org/pdf/2509.10847v2)**

> **作者:** Xiaohui Rao; Hanlin Wu; Zhenguang G. Cai
>
> **摘要:** As AI companions become capable of human-like communication, including telling jokes, understanding how people cognitively and emotionally respond to AI humor becomes increasingly important. This study used electroencephalography (EEG) to compare how people process humor from AI versus human sources. Behavioral analysis revealed that participants rated AI and human humor as comparably funny. However, neurophysiological data showed that AI humor elicited a smaller N400 effect, suggesting reduced cognitive effort during the processing of incongruity. This was accompanied by a larger Late Positive Potential (LPP), indicating a greater degree of surprise and emotional response. This enhanced LPP likely stems from the violation of low initial expectations regarding AI's comedic capabilities. Furthermore, a key temporal dynamic emerged: human humor showed habituation effects, marked by an increasing N400 and a decreasing LPP over time. In contrast, AI humor demonstrated increasing processing efficiency and emotional reward, with a decreasing N400 and an increasing LPP. This trajectory reveals how the brain can dynamically update its predictive model of AI capabilities. This process of cumulative reinforcement challenges "algorithm aversion" in humor, as it demonstrates how cognitive adaptation to AI's language patterns can lead to an intensified emotional reward. Additionally, participants' social attitudes toward AI modulated these neural responses, with higher perceived AI trustworthiness correlating with enhanced emotional engagement. These findings indicate that the brain responds to AI humor with surprisingly positive and intense reactions, highlighting humor's potential for fostering genuine engagement in human-AI social interaction.
>
---
#### [replaced 042] Do predictability factors towards signing avatars hold across cultures?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2307.02103v2](http://arxiv.org/pdf/2307.02103v2)**

> **作者:** Abdelhadi Soudi; Manal El Hakkaoui; Kristof Van Laerhoven
>
> **备注:** updated version
>
> **摘要:** Avatar technology can offer accessibility possibilities and improve the Deaf-and-Hard of Hearing sign language users access to communication, education and services, such as the healthcare system. However, sign language users acceptance of signing avatars as well as their attitudes towards them vary and depend on many factors. Furthermore, research on avatar technology is mostly done by researchers who are not Deaf. The study examines the extent to which intrinsic or extrinsic factors contribute to predict the attitude towards avatars across cultures. Intrinsic factors include the characteristics of the avatar, such as appearance, movements and facial expressions. Extrinsic factors include users technology experience, their hearing status, age and their sign language fluency. This work attempts to answer questions such as, if lower attitude ratings are related to poor technology experience with ASL users, for example, is that also true for Moroccan Sign Language (MSL) users? For the purposes of the study, we designed a questionnaire to understand MSL users attitude towards avatars. Three groups of participants were surveyed: Deaf (57), Hearing (20) and Hard-of-Hearing (3). The results of our study were then compared with those reported in other relevant studies.
>
---
#### [replaced 043] MVPBench: A Benchmark and Fine-Tuning Framework for Aligning Large Language Models with Diverse Human Values
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.08022v2](http://arxiv.org/pdf/2509.08022v2)**

> **作者:** Yao Liang; Dongcheng Zhao; Feifei Zhao; Guobin Shen; Yuwei Wang; Dongqi Liang; Yi Zeng
>
> **备注:** Some parts of the paper need to be revised. We would therefore like to withdraw the paper and resubmit it after making the necessary changes
>
> **摘要:** The alignment of large language models (LLMs) with human values is critical for their safe and effective deployment across diverse user populations. However, existing benchmarks often neglect cultural and demographic diversity, leading to limited understanding of how value alignment generalizes globally. In this work, we introduce MVPBench, a novel benchmark that systematically evaluates LLMs' alignment with multi-dimensional human value preferences across 75 countries. MVPBench contains 24,020 high-quality instances annotated with fine-grained value labels, personalized questions, and rich demographic metadata, making it the most comprehensive resource of its kind to date. Using MVPBench, we conduct an in-depth analysis of several state-of-the-art LLMs, revealing substantial disparities in alignment performance across geographic and demographic lines. We further demonstrate that lightweight fine-tuning methods, such as Low-Rank Adaptation (LoRA) and Direct Preference Optimization (DPO), can significantly enhance value alignment in both in-domain and out-of-domain settings. Our findings underscore the necessity for population-aware alignment evaluation and provide actionable insights for building culturally adaptive and value-sensitive LLMs. MVPBench serves as a practical foundation for future research on global alignment, personalized value modeling, and equitable AI development.
>
---
#### [replaced 044] UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.09767v2](http://arxiv.org/pdf/2508.09767v2)**

> **作者:** Shuhei Kato
>
> **备注:** 5 pages
>
> **摘要:** We propose UtterTune, a lightweight adaptation method that fine-tunes a multilingual text-to-speech (TTS) system based on a large language model (LLM) architecture, designed to enhance the controllability of pronunciation in a target language while preserving performance in others. While LLM architectures have enabled TTS models to achieve remarkable naturalness, accurately modeling grapheme-to-phoneme (G2P) mapping and prosody remains challenging, especially when the model omits an explicit G2P module and directly processes minimally encoded text (e.g., byte-pair encoding). UtterTune leverages low-rank adaptation to enable the control of segmental pronunciation and pitch accent at the phoneme level for Japanese speech, the target language in this paper, while maintaining naturalness and speaker similarity in a zero-shot setting. Objective and subjective evaluations confirm its effectiveness.
>
---
#### [replaced 045] Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-$k$
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.08479v2](http://arxiv.org/pdf/2506.08479v2)**

> **作者:** Chihiro Taguchi; Seiji Maekawa; Nikita Bhutani
>
> **备注:** 26 pages, 16 tables, 5 figures. Accepted at EMNLP 2025 (Main)
>
> **摘要:** Retrieval-augmented generation (RAG) and long-context language models (LCLMs) both address context limitations of LLMs in open-domain question answering (QA). However, optimal external context to retrieve remains an open problem: fixing the retrieval size risks either wasting tokens or omitting key evidence. Existing adaptive methods like Self-RAG and Self-Route rely on iterative LLM prompting and perform well on factoid QA, but struggle with aggregation QA, where the optimal context size is both unknown and variable. We present Adaptive-$k$ retrieval, a simple and effective single-pass method that adaptively selects the number of passages based on the distribution of the similarity scores between the query and the candidate passages. It does not require model fine-tuning, extra LLM inferences or changes to existing retriever-reader pipelines. On both factoid and aggregation QA benchmarks, Adaptive-$k$ matches or outperforms fixed-$k$ baselines while using up to 10x fewer tokens than full-context input, yet still retrieves 70% of relevant passages. It improves accuracy across five LCLMs and two embedding models, highlighting that dynamically adjusting context size leads to more efficient and accurate QA.
>
---
#### [replaced 046] ICR: Iterative Clarification and Rewriting for Conversational Search
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.05100v2](http://arxiv.org/pdf/2509.05100v2)**

> **作者:** Zhiyu Cao; Peifeng Li; Qiaoming Zhu
>
> **摘要:** Most previous work on Conversational Query Rewriting employs an end-to-end rewriting paradigm. However, this approach is hindered by the issue of multiple fuzzy expressions within the query, which complicates the simultaneous identification and rewriting of multiple positions. To address this issue, we propose a novel framework ICR (Iterative Clarification and Rewriting), an iterative rewriting scheme that pivots on clarification questions. Within this framework, the model alternates between generating clarification questions and rewritten queries. The experimental results show that our ICR can continuously improve retrieval performance in the clarification-rewriting iterative process, thereby achieving state-of-the-art performance on two popular datasets.
>
---
#### [replaced 047] MillStone: How Open-Minded Are LLMs?
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11967v2](http://arxiv.org/pdf/2509.11967v2)**

> **作者:** Harold Triedman; Vitaly Shmatikov
>
> **备注:** 19 pages, 7 tables, 7 figures
>
> **摘要:** Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
>
---
#### [replaced 048] Understanding and Leveraging the Expert Specialization of Context Faithfulness in Mixture-of-Experts LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19594v2](http://arxiv.org/pdf/2508.19594v2)**

> **作者:** Jun Bai; Minghao Tong; Yang Liu; Zixia Jia; Zilong Zheng
>
> **备注:** Accepted by EMNLP 2025 Main
>
> **摘要:** Context faithfulness is essential for reliable reasoning in context-dependent scenarios. However, large language models often struggle to ground their outputs in the provided context, resulting in irrelevant responses. Inspired by the emergent expert specialization observed in mixture-of-experts architectures, this work investigates whether certain experts exhibit specialization in context utilization, offering a potential pathway toward targeted optimization for improved context faithfulness. To explore this, we propose Router Lens, a method that accurately identifies context-faithful experts. Our analysis reveals that these experts progressively amplify attention to relevant contextual information, thereby enhancing context grounding. Building on this insight, we introduce Context-faithful Expert Fine-Tuning (CEFT), a lightweight optimization approach that selectively fine-tunes context-faithful experts. Experiments across a wide range of benchmarks and models demonstrate that CEFT matches or surpasses the performance of full fine-tuning while being significantly more efficient.
>
---
#### [replaced 049] TAPS: Tool-Augmented Personalisation via Structured Tagging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20409v3](http://arxiv.org/pdf/2506.20409v3)**

> **作者:** Ekaterina Taktasheva; Jeff Dalton
>
> **备注:** Accepted to EMNLP 2026 Main
>
> **摘要:** Recent advancements in tool-augmented large language models have enabled them to interact with external tools, enhancing their ability to perform complex user tasks. However, existing approaches overlook the role of personalisation in guiding tool use. This work investigates how user preferences can be effectively integrated into goal-oriented dialogue agents. Through extensive analysis, we identify key weaknesses in the ability of LLMs to personalise tool use. To this end, we introduce TAPS, a novel solution that enhances personalised tool use by leveraging a structured tagging tool and an uncertainty-based tool detector. TAPS significantly improves the ability of LLMs to incorporate user preferences, achieving the new state-of-the-art for open source models on the NLSI task.
>
---
#### [replaced 050] Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2507.02844v2](http://arxiv.org/pdf/2507.02844v2)**

> **作者:** Ziqi Miao; Yi Ding; Lijun Li; Jing Shao
>
> **备注:** Accepted to EMNLP 2025 (Main). 17 pages, 7 figures
>
> **摘要:** With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.
>
---
#### [replaced 051] From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16408v2](http://arxiv.org/pdf/2505.16408v2)**

> **作者:** Muhammad Farid Adilazuarda; Chen Cecilia Liu; Iryna Gurevych; Alham Fikri Aji
>
> **摘要:** Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior. We release our code at https://github.com/faridlazuarda/from-surveys-to-narratives.
>
---
#### [replaced 052] Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16467v2](http://arxiv.org/pdf/2505.16467v2)**

> **作者:** Vera Neplenbroek; Arianna Bisazza; Raquel Fernández
>
> **备注:** Accepted at EMNLP Main 2025
>
> **摘要:** Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
>
---
#### [replaced 053] The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14172v3](http://arxiv.org/pdf/2505.14172v3)**

> **作者:** Adrian Cosma; Stefan Ruseti; Emilian Radoi; Mihai Dascalu
>
> **备注:** Accepted at EMNLP 2025 Main as Oral Presentation (Top 15% of accepted papers)
>
> **摘要:** Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge suddenly and only late in training. We find that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available.
>
---
#### [replaced 054] HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.11552v2](http://arxiv.org/pdf/2509.11552v2)**

> **作者:** Wensheng Lu; Keyu Chen; Ruizhi Qiao; Xing Sun
>
> **备注:** 17 pages, 5 figures, 6 tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems.
>
---
#### [replaced 055] TokenSkip: Controllable Chain-of-Thought Compression in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12067v3](http://arxiv.org/pdf/2502.12067v3)**

> **作者:** Heming Xia; Chak Tou Leong; Wenjie Wang; Yongqi Li; Wenjie Li
>
> **备注:** EMNLP 2025 (Long Paper), camera-ready version
>
> **摘要:** Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https://github.com/hemingkx/TokenSkip.
>
---
#### [replaced 056] References Matter: Investigating the Impact of Reference Set Variation on Summarization Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14335v3](http://arxiv.org/pdf/2506.14335v3)**

> **作者:** Silvia Casola; Yang Janet Liu; Siyao Peng; Oliver Kraus; Albert Gatt; Barbara Plank
>
> **摘要:** Human language production exhibits remarkable richness and variation, reflecting diverse communication styles and intents. However, this variation is often overlooked in summarization evaluation. While having multiple reference summaries is known to improve correlation with human judgments, the impact of the reference set on reference-based metrics has not been systematically investigated. This work examines the sensitivity of widely used reference-based metrics in relation to the choice of reference sets, analyzing three diverse multi-reference summarization datasets: SummEval, GUMSum, and DUC2004. We demonstrate that many popular metrics exhibit significant instability. This instability is particularly concerning for n-gram-based metrics like ROUGE, where model rankings vary depending on the reference sets, undermining the reliability of model comparisons. We also collect human judgments on LLM outputs for genre-diverse data and examine their correlation with metrics to supplement existing findings beyond newswire summaries, finding weak-to-no correlation. Taken together, we recommend incorporating reference set variation into summarization evaluation to enhance consistency alongside correlation with human judgments, especially when evaluating LLMs.
>
---
#### [replaced 057] Concurrent Linguistic Error Detection (CLED): a New Methodology for Error Detection in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.16393v2](http://arxiv.org/pdf/2403.16393v2)**

> **作者:** Jinhua Zhu; Javier Conde; Zhen Gao; Pedro Reviriego; Shanshan Liu; Fabrizio Lombardi
>
> **备注:** 11 pages, 6 figures, 30 references
>
> **摘要:** The wide adoption of Large language models (LLMs) makes their dependability a pressing concern. Detection of errors is the first step to mitigating their impact on a system and thus, efficient error detection for LLMs is an important issue. In many settings, the LLM is considered as a black box with no access to the internal nodes; this prevents the use of many error detection schemes that need access to the model's internal nodes. An interesting observation is that the output of LLMs in error-free operation should be valid and normal text. Therefore, when the text is not valid or differs significantly from normal text, it is likely that there is an error. Based on this observation we propose to perform Concurrent Linguistic Error Detection (CLED); this scheme extracts some linguistic features of the text generated by the LLM and feeds them to a concurrent classifier that detects errors. Since the proposed error detection mechanism only relies on the outputs of the model, then it can be used on LLMs in which there is no access to the internal nodes. The proposed CLED scheme has been evaluated on the T5 model when used for news summarization and on the OPUS-MT model when used for translation. In both cases, the same set of linguistic features has been used for error detection to illustrate the applicability of the proposed scheme beyond a specific case. The results show that CLED can detect most of the errors at a low overhead penalty. The use of the concurrent classifier also enables a trade-off between error detection effectiveness and its associated overhead, so providing flexibility to a designer.
>
---
#### [replaced 058] IAG: Input-aware Backdoor Attack on VLMs for Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2508.09456v2](http://arxiv.org/pdf/2508.09456v2)**

> **作者:** Junxian Li; Beining Xu; Di Zhang
>
> **备注:** 13 pages, 13 Figures
>
> **摘要:** Vision-language models (VLMs) have shown significant advancements in tasks such as visual grounding, where they localize specific objects in images based on natural language queries and images. However, security issues in visual grounding tasks for VLMs remain underexplored, especially in the context of backdoor attacks. In this paper, we introduce a novel input-aware backdoor attack method, IAG, designed to manipulate the grounding behavior of VLMs. This attack forces the model to ground a specific target object in the input image, regardless of the user's query. We propose an adaptive trigger generator that embeds the semantic information of the attack target's description into the original image using a text-conditional U-Net, thereby overcoming the open-vocabulary attack challenge. To ensure the attack's stealthiness, we utilize a reconstruction loss to minimize visual discrepancies between poisoned and clean images. Additionally, we introduce a unified method for generating attack data. IAG is evaluated theoretically and empirically, demonstrating its feasibility and effectiveness. Notably, our ASR@0.5 on InternVL-2.5-8B reaches over 65\% on various testing sets. IAG also shows promising potential on manipulating Ferret-7B and LlaVA-1.5-7B with very little accuracy decrease on clean samples. Extensive specific experiments, such as ablation study and potential defense, also indicate the robustness and transferability of our attack.
>
---
#### [replaced 059] EIFBENCH: Extremely Complex Instruction Following Benchmark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08375v2](http://arxiv.org/pdf/2506.08375v2)**

> **作者:** Tao Zou; Xinghua Zhang; Haiyang Yu; Minzheng Wang; Fei Huang; Yongbin Li
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** With the development and widespread application of large language models (LLMs), the new paradigm of "Model as Product" is rapidly evolving, and demands higher capabilities to address complex user needs, often requiring precise workflow execution which involves the accurate understanding of multiple tasks. However, existing benchmarks focusing on single-task environments with limited constraints lack the complexity required to fully reflect real-world scenarios. To bridge this gap, we present the Extremely Complex Instruction Following Benchmark (EIFBENCH), meticulously crafted to facilitate a more realistic and robust evaluation of LLMs. EIFBENCH not only includes multi-task scenarios that enable comprehensive assessment across diverse task types concurrently, but also integrates a variety of constraints, replicating complex operational environments. Furthermore, we propose the Segment Policy Optimization (SegPO) algorithm to enhance the LLM's ability to accurately fulfill multi-task workflow. Evaluations on EIFBENCH have unveiled considerable performance discrepancies in existing LLMs when challenged with these extremely complex instructions. This finding underscores the necessity for ongoing optimization to navigate the intricate challenges posed by LLM applications.
>
---
#### [replaced 060] Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18436v3](http://arxiv.org/pdf/2410.18436v3)**

> **作者:** Seoyeon Kim; Huiseo Kim; Chanjun Park; Jinyoung Yeo; Dongha Lee
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
>
---
#### [replaced 061] Is the Top Still Spinning? Evaluating Subjectivity in Narrative Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.01132v2](http://arxiv.org/pdf/2504.01132v2)**

> **作者:** Melanie Subbiah; Akankshya Mishra; Grace Kim; Liyan Tang; Greg Durrett; Kathleen McKeown
>
> **备注:** EMNLP 2025
>
> **摘要:** Determining faithfulness of a claim to a source document is an important problem across many domains. This task is generally treated as a binary judgment of whether the claim is supported or unsupported in relation to the source. In many cases, though, whether a claim is supported can be ambiguous. For instance, it may depend on making inferences from given evidence, and different people can reasonably interpret the claim as either supported or unsupported based on their agreement with those inferences. Forcing binary labels upon such claims lowers the reliability of evaluation. In this work, we reframe the task to manage the subjectivity involved with factuality judgments of ambiguous claims. We introduce LLM-generated edits of summaries as a method of providing a nuanced evaluation of claims: how much does a summary need to be edited to be unambiguous? Whether a claim gets rewritten and how much it changes can be used as an automatic evaluation metric, the Ambiguity Rewrite Metric (ARM), with a much richer feedback signal than a binary judgment of faithfulness. We focus on the area of narrative summarization as it is particularly rife with ambiguity and subjective interpretation. We show that ARM produces a 21% absolute improvement in annotator agreement on claim faithfulness, indicating that subjectivity is reduced.
>
---
#### [replaced 062] Context-Aware Membership Inference Attacks against Pre-trained Large Language Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2409.13745v2](http://arxiv.org/pdf/2409.13745v2)**

> **作者:** Hongyan Chang; Ali Shahin Shamsabadi; Kleomenis Katevas; Hamed Haddadi; Reza Shokri
>
> **摘要:** Membership Inference Attacks (MIAs) on pre-trained Large Language Models (LLMs) aim at determining if a data point was part of the model's training set. Prior MIAs that are built for classification models fail at LLMs, due to ignoring the generative nature of LLMs across token sequences. In this paper, we present a novel attack on pre-trained LLMs that adapts MIA statistical tests to the perplexity dynamics of subsequences within a data point. Our method significantly outperforms prior approaches, revealing context-dependent memorization patterns in pre-trained LLMs.
>
---
#### [replaced 063] MachineLearningLM: Scaling Many-shot In-context Learning via Continued Pretraining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06806v5](http://arxiv.org/pdf/2509.06806v5)**

> **作者:** Haoyu Dong; Pengkun Zhang; Mingzhe Lu; Yanzhen Shen; Guolin Ke
>
> **摘要:** Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows. Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference. Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU.
>
---

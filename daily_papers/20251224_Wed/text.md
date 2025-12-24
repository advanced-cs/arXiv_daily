# 自然语言处理 cs.CL

- **最新发布 36 篇**

- **更新 29 篇**

## 最新发布

#### [new 001] Distilling to Hybrid Attention Models via KL-Guided Layer Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属模型压缩任务，旨在提升大语言模型推理效率。它提出一种基于KL散度引导的层选择方法，自动识别Transformer中适合替换为线性注意力的层，再结合RADLADS蒸馏流程，实现softmax与线性注意力混合架构的高效知识迁移。**

- **链接: [https://arxiv.org/pdf/2512.20569v1](https://arxiv.org/pdf/2512.20569v1)**

> **作者:** Yanhong Li; Songlin Yang; Shawn Tan; Mayank Mishra; Rameswar Panda; Jiawei Zhou; Yoon Kim
>
> **摘要:** Distilling pretrained softmax attention Transformers into more efficient hybrid architectures that interleave softmax and linear attention layers is a promising approach for improving the inference efficiency of LLMs without requiring expensive pretraining from scratch. A critical factor in the conversion process is layer selection, i.e., deciding on which layers to convert to linear attention variants. This paper describes a simple and efficient recipe for layer selection that uses layer importance scores derived from a small amount of training on generic text data. Once the layers have been selected we use a recent pipeline for the distillation process itself \citep[RADLADS;][]{goldstein2025radlads}, which consists of attention weight transfer, hidden state alignment, KL-based distribution matching, followed by a small amount of finetuning. We find that this approach is more effective than existing approaches for layer selection, including heuristics that uniformly interleave linear attentions based on a fixed ratio, as well as more involved approaches that rely on specialized diagnostic datasets.
>
---
#### [new 002] Retrieval-augmented Prompt Learning for Pre-trained Foundation Models
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文提出RetroPrompt，属提示学习任务，旨在解决预训练基础模型在少样本下过拟合、依赖死记硬背的问题。其通过引入基于训练数据构建的公开知识库与全程检索机制，解耦知识与记忆，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.20145v1](https://arxiv.org/pdf/2512.20145v1)**

> **作者:** Xiang Chen; Yixin Ou; Quan Feng; Lei Li; Piji Li; Haibo Ye; Sheng-Jun Huang; Shuofei Qiao; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** The pre-trained foundation models (PFMs) have become essential for facilitating large-scale multimodal learning. Researchers have effectively employed the ``pre-train, prompt, and predict'' paradigm through prompt learning to induce improved few-shot performance. However, prompt learning approaches for PFMs still follow a parametric learning paradigm. As such, the stability of generalization in memorization and rote learning can be compromised. More specifically, conventional prompt learning might face difficulties in fully utilizing atypical instances and avoiding overfitting to shallow patterns with limited data during the process of fully-supervised training. To overcome these constraints, we present our approach, named RetroPrompt, which aims to achieve a balance between memorization and generalization by decoupling knowledge from mere memorization. Unlike traditional prompting methods, RetroPrompt leverages a publicly accessible knowledge base generated from the training data and incorporates a retrieval mechanism throughout the input, training, and inference stages. This enables the model to actively retrieve relevant contextual information from the corpus, thereby enhancing the available cues. We conduct comprehensive experiments on a variety of datasets across natural language processing and computer vision tasks to demonstrate the superior performance of our proposed approach, RetroPrompt, in both zero-shot and few-shot scenarios. Through detailed analysis of memorization patterns, we observe that RetroPrompt effectively reduces the reliance on rote memorization, leading to enhanced generalization.
>
---
#### [new 003] Corpus of Cross-lingual Dialogues with Minutes and Detection of Misunderstandings
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向跨语言会议场景，构建了含5小时多语语音、ASR转录、英译文本及会议纪要的评测语料库，并提出并标注了跨语言误解；还探索了大模型（如Gemini）自动检测误解的能力，验证其77%召回与47%精度。**

- **链接: [https://arxiv.org/pdf/2512.20204v1](https://arxiv.org/pdf/2512.20204v1)**

> **作者:** Marko Čechovič; Natália Komorníková; Dominik Macháček; Ondřej Bojar
>
> **备注:** 12 pages, 2 figures, 6 tables, published as a conference paper in Text, Speech, and Dialogue 28th International Conference, TSD 2025, Erlangen, Germany, August 25-28, 2025, Proceedings, Part II. This version published here on arXiv.org is before review comments and seedings of the TSD conference staff
>
> **摘要:** Speech processing and translation technology have the potential to facilitate meetings of individuals who do not share any common language. To evaluate automatic systems for such a task, a versatile and realistic evaluation corpus is needed. Therefore, we create and present a corpus of cross-lingual dialogues between individuals without a common language who were facilitated by automatic simultaneous speech translation. The corpus consists of 5 hours of speech recordings with ASR and gold transcripts in 12 original languages and automatic and corrected translations into English. For the purposes of research into cross-lingual summarization, our corpus also includes written summaries (minutes) of the meetings. Moreover, we propose automatic detection of misunderstandings. For an overview of this task and its complexity, we attempt to quantify misunderstandings in cross-lingual meetings. We annotate misunderstandings manually and also test the ability of current large language models to detect them automatically. The results show that the Gemini model is able to identify text spans with misunderstandings with recall of 77% and precision of 47%.
>
---
#### [new 004] Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出多LLM主题分析框架，解决定性研究中AI编码信度不足问题。融合Cohen’s Kappa与语义相似度双指标，支持可配置参数与共识主题提取。在 psychedelic艺术治疗访谈数据上验证三模型可靠性，开源实现以推动AI辅助质性研究可信化。**

- **链接: [https://arxiv.org/pdf/2512.20352v1](https://arxiv.org/pdf/2512.20352v1)**

> **作者:** Nilesh Jain; Seyi Adeyinka; Leor Roseman; Aza Allsop
>
> **备注:** 11 pages, 1 figure, 3 tables
>
> **摘要:** Qualitative research faces a critical reliability challenge: traditional inter-rater agreement methods require multiple human coders, are time-intensive, and often yield moderate consistency. We present a multi-perspective validation framework for LLM-based thematic analysis that combines ensemble validation with dual reliability metrics: Cohen's Kappa ($κ$) for inter-rater agreement and cosine similarity for semantic consistency. Our framework enables configurable analysis parameters (1-6 seeds, temperature 0.0-2.0), supports custom prompt structures with variable substitution, and provides consensus theme extraction across any JSON format. As proof-of-concept, we evaluate three leading LLMs (Gemini 2.5 Pro, GPT-4o, Claude 3.5 Sonnet) on a psychedelic art therapy interview transcript, conducting six independent runs per model. Results demonstrate Gemini achieves highest reliability ($κ= 0.907$, cosine=95.3%), followed by GPT-4o ($κ= 0.853$, cosine=92.6%) and Claude ($κ= 0.842$, cosine=92.1%). All three models achieve a high agreement ($κ> 0.80$), validating the multi-run ensemble approach. The framework successfully extracts consensus themes across runs, with Gemini identifying 6 consensus themes (50-83% consistency), GPT-4o identifying 5 themes, and Claude 4 themes. Our open-source implementation provides researchers with transparent reliability metrics, flexible configuration, and structure-agnostic consensus extraction, establishing methodological foundations for reliable AI-assisted qualitative research.
>
---
#### [new 005] Sentiment-Aware Extractive and Abstractive Summarization for Unstructured Text Mining
- **分类: cs.CL**

- **简介: 该论文属文本摘要任务，旨在解决现有方法在噪声大、情感丰富的短用户文本（如社交帖、评论）上摘要效果差的问题。作者提出情感感知框架，将情感信号融入TextRank（抽取式）和UniLM（生成式）模型，提升摘要的情感准确性与主题相关性。**

- **链接: [https://arxiv.org/pdf/2512.20404v1](https://arxiv.org/pdf/2512.20404v1)**

> **作者:** Junyi Liu; Stanley Kok
>
> **备注:** WITS 2025 (Workshop on Information Technologies and Systems 2025)
>
> **摘要:** With the rapid growth of unstructured data from social media, reviews, and forums, text mining has become essential in Information Systems (IS) for extracting actionable insights. Summarization can condense fragmented, emotion-rich posts, but existing methods-optimized for structured news-struggle with noisy, informal content. Emotional cues are critical for IS tasks such as brand monitoring and market analysis, yet few studies integrate sentiment modeling into summarization of short user-generated texts. We propose a sentiment-aware framework extending extractive (TextRank) and abstractive (UniLM) approaches by embedding sentiment signals into ranking and generation processes. This dual design improves the capture of emotional nuances and thematic relevance, producing concise, sentiment-enriched summaries that enhance timely interventions and strategic decision-making in dynamic online environments.
>
---
#### [new 006] Step-DeepResearch Technical Report
- **分类: cs.CL**

- **简介: 该论文面向自主研究代理任务，解决现有基准无法评估开放性深度研究能力的问题。提出Step-DeepResearch代理模型，采用原子能力数据合成、渐进式训练及清单式评判器，并构建中文ADR-Bench基准，验证中等规模模型在成本效益下达到专家级研究能力。**

- **链接: [https://arxiv.org/pdf/2512.20491v1](https://arxiv.org/pdf/2512.20491v1)**

> **作者:** Chen Hu; Haikuo Du; Heng Wang; Lin Lin; Mingrui Chen; Peng Liu; Ruihang Miao; Tianchi Yue; Wang You; Wei Ji; Wei Yuan; Wenjin Deng; Xiaojian Yuan; Xiaoyun Zhang; Xiangyu Liu; Xikai Liu; Yanming Xu; Yicheng Cao; Yifei Zhang; Yongyao Wang; Yubo Shu; Yurong Zhang; Yuxiang Zhang; Zheng Gong; Zhichao Chang; Binyan Li; Dan Ma; Furong Jia; Hongyuan Wang; Jiayu Liu; Jing Bai; Junlan Liu; Manjiao Liu; Na Wang; Qiuping Wu; Qinxin Du; Shiwei Li; Wen Sun; Yifeng Gong; Yonglin Chen; Yuling Zhao; Yuxuan Lin; Ziqi Ren; Zixuan Wang; Aihu Zhang; Brian Li; Buyun Ma; Kang An; Li Xie; Mingliang Li; Pan Li; Shidong Yang; Xi Chen; Xiaojia Liu; Yuchu Luo; Yuan Song; YuanHao Ding; Yuanwei Liang; Zexi Li; Zhaoning Zhang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** As LLMs shift toward autonomous agents, Deep Research has emerged as a pivotal metric. However, existing academic benchmarks like BrowseComp often fail to meet real-world demands for open-ended research, which requires robust skills in intent recognition, long-horizon decision-making, and cross-source verification. To address this, we introduce Step-DeepResearch, a cost-effective, end-to-end agent. We propose a Data Synthesis Strategy Based on Atomic Capabilities to reinforce planning and report writing, combined with a progressive training path from agentic mid-training to SFT and RL. Enhanced by a Checklist-style Judger, this approach significantly improves robustness. Furthermore, to bridge the evaluation gap in the Chinese domain, we establish ADR-Bench for realistic deep research scenarios. Experimental results show that Step-DeepResearch (32B) scores 61.4% on Scale AI Research Rubrics. On ADR-Bench, it significantly outperforms comparable models and rivals SOTA closed-source models like OpenAI and Gemini DeepResearch. These findings prove that refined training enables medium-sized models to achieve expert-level capabilities at industry-leading cost-efficiency.
>
---
#### [new 007] Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents
- **分类: cs.CL**

- **简介: 该论文面向多轮对话中的时序推理任务，解决长历史对话中模型难以精准定位时间相关证据的问题。提出Memory-T1框架：用强化学习学习时间感知的记忆选择策略，通过粗粒度过滤与细粒度RL选择关键会话，并设计含时序一致性的多级奖励函数。**

- **链接: [https://arxiv.org/pdf/2512.20092v1](https://arxiv.org/pdf/2512.20092v1)**

> **作者:** Yiming Du; Baojun Wang; Yifan Xiang; Zhaowei Wang; Wenyu Huang; Boyang Xue; Bin Liang; Xingshan Zeng; Fei Mi; Haoli Bai; Lifeng Shang; Jeff Z. Pan; Yuxin Jiang; Kam-Fai Wong
>
> **摘要:** Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. However, existing works and our pilot study have shown that as dialogue histories grow in length and accumulate noise, current long-context models struggle to accurately identify temporally pertinent information, significantly impairing reasoning performance. To address this, we introduce Memory-T1, a framework that learns a time-aware memory selection policy using reinforcement learning (RL). It employs a coarse-to-fine strategy, first pruning the dialogue history into a candidate set using temporal and relevance filters, followed by an RL agent that selects the precise evidence sessions. The RL training is guided by a multi-level reward function optimizing (i) answer accuracy, (ii) evidence grounding, and (iii) temporal consistency. In particular, the temporal consistency reward provides a dense signal by evaluating alignment with the query time scope at both the session-level (chronological proximity) and the utterance-level (chronological fidelity), enabling the agent to resolve subtle chronological ambiguities. On the Time-Dialog benchmark, Memory-T1 boosts a 7B model to an overall score of 67.0\%, establishing a new state-of-the-art performance for open-source models and outperforming a 14B baseline by 10.2\%. Ablation studies show temporal consistency and evidence grounding rewards jointly contribute to a 15.0\% performance gain. Moreover, Memory-T1 maintains robustness up to 128k tokens, where baseline models collapse, proving effectiveness against noise in extensive dialogue histories. The code and datasets are publicly available at https://github.com/Elvin-Yiming-Du/Memory-T1/
>
---
#### [new 008] Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits
- **分类: cs.CL**

- **简介: 该论文研究LLM自我失败预测任务，旨在解决模型无法识别自身错误与幻觉的问题。作者提出Gnosis机制，通过轻量解码隐藏状态和注意力模式，实现冻结LLM的内在自验证，零样本泛化，且开销极小。**

- **链接: [https://arxiv.org/pdf/2512.20578v1](https://arxiv.org/pdf/2512.20578v1)**

> **作者:** Amirhosein Ghasemabadi; Di Niu
>
> **摘要:** Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision.
>
---
#### [new 009] MoE-DiffuSeq: Enhancing Long-Document Diffusion Models with Sparse Attention and Mixture of Experts
- **分类: cs.CL**

- **简介: 该论文面向长文档文本生成任务，解决扩散模型（如DiffuSeq）在长序列上计算开销大、内存占用高的问题。提出MoE-DiffuSeq框架，融合稀疏注意力与混合专家（MoE）架构，并引入软吸收态加速重建，显著提升训练效率、采样速度与生成质量。**

- **链接: [https://arxiv.org/pdf/2512.20604v1](https://arxiv.org/pdf/2512.20604v1)**

> **作者:** Alexandros Christoforos; Chadbourne Davis
>
> **备注:** Under submission
>
> **摘要:** We present MoE-DiffuSeq, a mixture of experts based framework for enhancing diffusion models in long document generation. Existing diffusion based text generation models, such as DiffuSeq, suffer from high computational cost and memory overhead when applied to extended sequences. To address these challenges, MoE-DiffuSeq integrates sparse attention with a mixture of experts architecture, enabling efficient and scalable long sequence modeling. Our approach introduces a customized sparse attention mechanism designed to reduce computational complexity while preserving text quality and coherence. In addition, we incorporate a soft absorbing state within the diffusion process to accelerate sequence reconstruction and improve generation precision. Extensive experiments demonstrate that MoE-DiffuSeq significantly improves training efficiency and sampling speed compared to existing diffusion models. These advantages are particularly effective for long document scenarios, including scientific article generation, code repository modeling, and long form dialogue generation. Benchmark results further show that MoE-DiffuSeq improves efficiency, speed, accuracy, and expressiveness, advancing the practical applicability of diffusion models for high quality long form text generation.
>
---
#### [new 010] FaithLens: Detecting and Explaining Faithfulness Hallucination
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FaithLens模型，解决大语言模型输出中“忠实性幻觉”（即事实性错误）的检测与解释问题。通过合成带解释的训练数据、数据过滤、监督微调及规则强化学习，实现高效、可信的二分类检测与高质量解释生成。**

- **链接: [https://arxiv.org/pdf/2512.20182v1](https://arxiv.org/pdf/2512.20182v1)**

> **作者:** Shuzheng Si; Qingyi Wang; Haozhe Zhao; Yuzhuo Bai; Guanqiao Chen; Kangyang Luo; Gang Chen; Fanchao Qi; Minjia Zhang; Baobao Chang; Maosong Sun
>
> **摘要:** Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness.
>
---
#### [new 011] Bias Beneath the Tone: Empirical Characterisation of Tone Bias in LLM-Driven UX Systems
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究大语言模型在UX对话系统中的隐性语气偏见问题，属AI公平性与可解释性任务。它构建中性/导向性合成对话数据集，利用DistilBERT弱监督标注并训练分类器，发现模型存在固有语气偏差（如过度礼貌），证实其系统性、可测性，为设计公正可信的对话AI提供依据。**

- **链接: [https://arxiv.org/pdf/2512.19950v1](https://arxiv.org/pdf/2512.19950v1)**

> **作者:** Heet Bodara; Md Masum Mushfiq; Isma Farah Siddiqui
>
> **摘要:** Large Language Models are increasingly used in conversational systems such as digital personal assistants, shaping how people interact with technology through language. While their responses often sound fluent and natural, they can also carry subtle tone biases such as sounding overly polite, cheerful, or cautious even when neutrality is expected. These tendencies can influence how users perceive trust, empathy, and fairness in dialogue. In this study, we explore tone bias as a hidden behavioral trait of large language models. The novelty of this research lies in the integration of controllable large language model based dialogue synthesis with tone classification models, enabling robust and ethical emotion recognition in personal assistant interactions. We created two synthetic dialogue datasets, one generated from neutral prompts and another explicitly guided to produce positive or negative tones. Surprisingly, even the neutral set showed consistent tonal skew, suggesting that bias may stem from the model's underlying conversational style. Using weak supervision through a pretrained DistilBERT model, we labeled tones and trained several classifiers to detect these patterns. Ensemble models achieved macro F1 scores up to 0.92, showing that tone bias is systematic, measurable, and relevant to designing fair and trustworthy conversational AI.
>
---
#### [new 012] AprielGuard
- **分类: cs.CL**

- **简介: 该论文提出AprielGuard，一种8B参数的统一安全防护模型，旨在同时应对LLM的有害内容（如毒性、偏见）与对抗威胁（如提示注入、越狱）。它基于多源数据与结构化推理训练，在多步、推理密集场景中显著优于Llama-Guard等开源方案。**

- **链接: [https://arxiv.org/pdf/2512.20293v1](https://arxiv.org/pdf/2512.20293v1)**

> **作者:** Jaykumar Kasundra; Anjaneya Praharaj; Sourabh Surana; Lakshmi Sirisha Chodisetty; Sourav Sharma; Abhigya Verma; Abhishek Bhardwaj; Debasish Kanhar; Aakash Bhagat; Khalil Slimi; Seganrasan Subramanian; Sathwik Tejaswi Madhusudhan; Ranga Prasad Chenna; Srinivas Sunkara
>
> **摘要:** Safeguarding large language models (LLMs) against unsafe or adversarial behavior is critical as they are increasingly deployed in conversational and agentic settings. Existing moderation tools often treat safety risks (e.g. toxicity, bias) and adversarial threats (e.g. prompt injections, jailbreaks) as separate problems, limiting their robustness and generalizability. We introduce AprielGuard, an 8B parameter safeguard model that unify these dimensions within a single taxonomy and learning framework. AprielGuard is trained on a diverse mix of open and synthetic data covering standalone prompts, multi-turn conversations, and agentic workflows, augmented with structured reasoning traces to improve interpretability. Across multiple public and proprietary benchmarks, AprielGuard achieves strong performance in detecting harmful content and adversarial manipulations, outperforming existing opensource guardrails such as Llama-Guard and Granite Guardian, particularly in multi-step and reasoning intensive scenarios. By releasing the model, we aim to advance transparent and reproducible research on reliable safeguards for LLMs.
>
---
#### [new 013] PRISM: A Personality-Driven Multi-Agent Framework for Social Media Simulation
- **分类: cs.CL**

- **简介: 该论文提出PRISM框架，解决传统ABM因忽略心理异质性而难以建模网络极化的问题。它融合SDE与MBTI驱动的PC-POMDP，用多模态大模型代理模拟人格一致的社交行为，成功复现理性抑制、情感共振等现象。**

- **链接: [https://arxiv.org/pdf/2512.19933v1](https://arxiv.org/pdf/2512.19933v1)**

> **作者:** Zhixiang Lu; Xueyuan Deng; Yiran Liu; Yulong Li; Qiang Yan; Imran Razzak; Jionglong Su
>
> **摘要:** Traditional agent-based models (ABMs) of opinion dynamics often fail to capture the psychological heterogeneity driving online polarization due to simplistic homogeneity assumptions. This limitation obscures the critical interplay between individual cognitive biases and information propagation, thereby hindering a mechanistic understanding of how ideological divides are amplified. To address this challenge, we introduce the Personality-Refracted Intelligent Simulation Model (PRISM), a hybrid framework coupling stochastic differential equations (SDE) for continuous emotional evolution with a personality-conditional partially observable Markov decision process (PC-POMDP) for discrete decision-making. In contrast to continuous trait approaches, PRISM assigns distinct Myers-Briggs Type Indicator (MBTI) based cognitive policies to multimodal large language model (MLLM) agents, initialized via data-driven priors from large-scale social media datasets. PRISM achieves superior personality consistency aligned with human ground truth, significantly outperforming standard homogeneous and Big Five benchmarks. This framework effectively replicates emergent phenomena such as rational suppression and affective resonance, offering a robust tool for analyzing complex social media ecosystems.
>
---
#### [new 014] Counterfactual LLM-based Framework for Measuring Rhetorical Style
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出基于LLM的反事实框架，量化机器学习论文中的修辞风格，剥离内容影响。任务是修辞风格测量；解决“ hype”难量化问题；工作包括构建修辞人格、生成反事实文本、LLM成对评估与Bradley-Terry建模，并在ICLR论文上实证分析。**

- **链接: [https://arxiv.org/pdf/2512.19908v1](https://arxiv.org/pdf/2512.19908v1)**

> **作者:** Jingyi Qiu; Hong Chen; Zongyi Li
>
> **摘要:** The rise of AI has fueled growing concerns about ``hype'' in machine learning papers, yet a reliable way to quantify rhetorical style independently of substantive content has remained elusive. Because bold language can stem from either strong empirical results or mere rhetorical style, it is often difficult to distinguish between the two. To disentangle rhetorical style from substantive content, we introduce a counterfactual, LLM-based framework: multiple LLM rhetorical personas generate counterfactual writings from the same substantive content, an LLM judge compares them through pairwise evaluations, and the outcomes are aggregated using a Bradley--Terry model. Applying this method to 8,485 ICLR submissions sampled from 2017 to 2025, we generate more than 250,000 counterfactual writings and provide a large-scale quantification of rhetorical style in ML papers. We find that visionary framing significantly predicts downstream attention, including citations and media attention, even after controlling for peer-review evaluations. We also observe a sharp rise in rhetorical strength after 2023, and provide empirical evidence showing that this increase is largely driven by the adoption of LLM-based writing assistance. The reliability of our framework is validated by its robustness to the choice of personas and the high correlation between LLM judgments and human annotations. Our work demonstrates that LLMs can serve as instruments to measure and improve scientific evaluation.
>
---
#### [new 015] Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属临床诊断评估任务，旨在检验LLM在人格障碍诊断中是否可替代人类专家。研究首次对比Gemini Pro等LLM与精神科医生对波兰语自述文本的BPD/NPD诊断能力，发现模型整体准确率更高但严重低估NPD，暴露其模式偏好与价值偏见问题。**

- **链接: [https://arxiv.org/pdf/2512.20298v1](https://arxiv.org/pdf/2512.20298v1)**

> **作者:** Karolina Drożdż; Kacper Dudzic; Anna Sterna; Marcin Moskalewicz
>
> **摘要:** Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. We present the first direct comparison between state-of-the-art LLMs and mental health professionals in diagnosing Borderline (BPD) and Narcissistic (NPD) Personality Disorders utilizing Polish-language first-person autobiographical accounts. We show that the top-performing Gemini Pro models surpassed human professionals in overall diagnostic accuracy by 21.91 percentage points (65.48% vs. 43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patient's sense of self and temporal experience. Our findings demonstrate that while LLMs are highly competent at interpreting complex first-person clinical data, they remain subject to critical reliability and bias issues.
>
---
#### [new 016] AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在简历筛选等专用场景中的对抗脆弱性，提出“对抗指令”攻击新问题。构建简历筛选基准，发现攻击成功率超80%；对比prompt防御与新方法FIDS（基于LoRA），验证训练时防御更优。**

- **链接: [https://arxiv.org/pdf/2512.20164v1](https://arxiv.org/pdf/2512.20164v1)**

> **作者:** Honglin Mu; Jinghao Liu; Kaiyang Wan; Rui Xing; Xiuying Chen; Timothy Baldwin; Wanxiang Che
>
> **摘要:** Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.
>
---
#### [new 017] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文提出SlideTailor，解决科学论文自动生成个性化幻灯片任务。针对用户偏好隐式、难标注问题，用示例对+模板隐式建模偏好，并引入链式语音机制对齐口述内容，提升幻灯片质量与可用性。**

- **链接: [https://arxiv.org/pdf/2512.20292v1](https://arxiv.org/pdf/2512.20292v1)**

> **作者:** Wenzheng Zeng; Mingyu Ouyang; Langyuan Cui; Hwee Tou Ng
>
> **备注:** AAAI 2026 (with appendix)
>
> **摘要:** Automatic presentation slide generation can greatly streamline content creation. However, since preferences of each user may vary, existing under-specified formulations often lead to suboptimal results that fail to align with individual user needs. We introduce a novel task that conditions paper-to-slides generation on user-specified preferences. We propose a human behavior-inspired agentic framework, SlideTailor, that progressively generates editable slides in a user-aligned manner. Instead of requiring users to write their preferences in detailed textual form, our system only asks for a paper-slides example pair and a visual template - natural and easy-to-provide artifacts that implicitly encode rich user preferences across content and visual style. Despite the implicit and unlabeled nature of these inputs, our framework effectively distills and generalizes the preferences to guide customized slide generation. We also introduce a novel chain-of-speech mechanism to align slide content with planned oral narration. Such a design significantly enhances the quality of generated slides and enables downstream applications like video presentations. To support this new task, we construct a benchmark dataset that captures diverse user preferences, with carefully designed interpretable metrics for robust evaluation. Extensive experiments demonstrate the effectiveness of our framework.
>
---
#### [new 018] SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出SpidR模型，面向无文本监督的口语语言建模任务，旨在从原始语音中直接学习稳定、语义丰富的离散语言单元。它通过掩码预测、自蒸馏与在线聚类联合训练，提升单元质量与预训练效率，显著优于wav2vec 2.0等基线，并开源代码与模型。**

- **链接: [https://arxiv.org/pdf/2512.20308v1](https://arxiv.org/pdf/2512.20308v1)**

> **作者:** Maxime Poli; Mahi Luthra; Youssef Benchekroun; Yosuke Higuchi; Martin Gleize; Jiayi Shen; Robin Algayres; Yu-An Chung; Mido Assran; Juan Pino; Emmanuel Dupoux
>
> **备注:** 30 pages, 16 figures
>
> **摘要:** The parallel advances in language modeling and speech representation learning have raised the prospect of learning language directly from speech without textual intermediates. This requires extracting semantic representations directly from speech. Our contributions are threefold. First, we introduce SpidR, a self-supervised speech representation model that efficiently learns representations with highly accessible phonetic information, which makes it particularly suited for textless spoken language modeling. It is trained on raw waveforms using a masked prediction objective combined with self-distillation and online clustering. The intermediate layers of the student model learn to predict assignments derived from the teacher's intermediate layers. This learning objective stabilizes the online clustering procedure compared to previous approaches, resulting in higher quality codebooks. SpidR outperforms wav2vec 2.0, HuBERT, WavLM, and DinoSR on downstream language modeling benchmarks (sWUGGY, sBLIMP, tSC). Second, we systematically evaluate across models and layers the correlation between speech unit quality (ABX, PNMI) and language modeling performance, validating these metrics as reliable proxies. Finally, SpidR significantly reduces pretraining time compared to HuBERT, requiring only one day of pretraining on 16 GPUs, instead of a week. This speedup is enabled by the pretraining method and an efficient codebase, which allows faster iteration and easier experimentation. We open-source the training code and model checkpoints at https://github.com/facebookresearch/spidr.
>
---
#### [new 019] Schoenfeld's Anatomy of Mathematical Reasoning by Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属AI可解释性任务，旨在解析大模型数学推理的内在认知结构。针对推理步骤难识别问题，提出ThinkARM框架，基于Schoenfeld理论将推理迹抽象为分析、探索、验证等功能步骤，揭示模型间结构性差异与推理动态机制。**

- **链接: [https://arxiv.org/pdf/2512.19995v1](https://arxiv.org/pdf/2512.19995v1)**

> **作者:** Ming Li; Chenrui Fan; Yize Cheng; Soheil Feizi; Tianyi Zhou
>
> **摘要:** Large language models increasingly expose reasoning traces, yet their underlying cognitive structure and steps remain difficult to identify and analyze beyond surface-level statistics. We adopt Schoenfeld's Episode Theory as an inductive, intermediate-scale lens and introduce ThinkARM (Anatomy of Reasoning in Models), a scalable framework that explicitly abstracts reasoning traces into functional reasoning steps such as Analysis, Explore, Implement, Verify, etc. When applied to mathematical problem solving by diverse models, this abstraction reveals reproducible thinking dynamics and structural differences between reasoning and non-reasoning models, which are not apparent from token-level views. We further present two diagnostic case studies showing that exploration functions as a critical branching step associated with correctness, and that efficiency-oriented methods selectively suppress evaluative feedback steps rather than uniformly shortening responses. Together, our results demonstrate that episode-level representations make reasoning steps explicit, enabling systematic analysis of how reasoning is structured, stabilized, and altered in modern language models.
>
---
#### [new 020] M$^3$KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属多模态检索增强生成（RAG）任务，旨在解决音频-视觉领域中现有多模态知识图谱（MMKG）覆盖窄、多跳连通性弱及检索不精准问题。提出M³KG-RAG框架：构建多跳MMKG，并引入GRASP模块实现查询对齐、相关性评估与冗余剪枝。**

- **链接: [https://arxiv.org/pdf/2512.20136v1](https://arxiv.org/pdf/2512.20136v1)**

> **作者:** Hyeongcheol Park; Jiyoung Seo; Jaewon Mun; Hogun Park; Wonmin Byeon; Sung June Kim; Hyeonsoo Im; JeungSub Lee; Sangpil Kim
>
> **摘要:** Retrieval-Augmented Generation (RAG) has recently been extended to multimodal settings, connecting multimodal large language models (MLLMs) with vast corpora of external knowledge such as multimodal knowledge graphs (MMKGs). Despite their recent success, multimodal RAG in the audio-visual domain remains challenging due to 1) limited modality coverage and multi-hop connectivity of existing MMKGs, and 2) retrieval based solely on similarity in a shared multimodal embedding space, which fails to filter out off-topic or redundant knowledge. To address these limitations, we propose M$^3$KG-RAG, a Multi-hop Multimodal Knowledge Graph-enhanced RAG that retrieves query-aligned audio-visual knowledge from MMKGs, improving reasoning depth and answer faithfulness in MLLMs. Specifically, we devise a lightweight multi-agent pipeline to construct multi-hop MMKG (M$^3$KG), which contains context-enriched triplets of multimodal entities, enabling modality-wise retrieval based on input queries. Furthermore, we introduce GRASP (Grounded Retrieval And Selective Pruning), which ensures precise entity grounding to the query, evaluates answer-supporting relevance, and prunes redundant context to retain only knowledge essential for response generation. Extensive experiments across diverse multimodal benchmarks demonstrate that M$^3$KG-RAG significantly enhances MLLMs' multimodal reasoning and grounding over existing approaches.
>
---
#### [new 021] ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ABBEL框架，解决长程决策中LLM记忆爆炸问题：用自然语言信念状态压缩交互历史，通过贝叶斯式更新实现低内存、可解释的多步推理；并引入RL后训练优化信念质量与压缩率，提升性能。**

- **链接: [https://arxiv.org/pdf/2512.20111v1](https://arxiv.org/pdf/2512.20111v1)**

> **作者:** Aly Lidayan; Jakob Bjorner; Satvik Golechha; Kartik Goyal; Alane Suhr
>
> **摘要:** As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches.
>
---
#### [new 022] Multi-hop Reasoning via Early Knowledge Alignment
- **分类: cs.CL**

- **简介: 该论文面向多跳问答任务，解决迭代RAG中因规划脱离检索语料导致的低效检索与级联错误问题；提出无需训练的Early Knowledge Alignment（EKA）模块，在推理前将LLM与检索结果对齐，提升检索精度、减少错误传播，并增强推理效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.20144v1](https://arxiv.org/pdf/2512.20144v1)**

> **作者:** Yuxin Wang; Shicheng Fang; Bo Wang; Qi Luo; Xuanjing Huang; Yining Zheng; Xipeng Qiu
>
> **备注:** 16 pages
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for Large Language Models (LLMs) to address knowledge-intensive queries requiring domain-specific or up-to-date information. To handle complex multi-hop questions that are challenging for single-step retrieval, iterative RAG approaches incorporating reinforcement learning have been proposed. However, existing iterative RAG systems typically plan to decompose questions without leveraging information about the available retrieval corpus, leading to inefficient retrieval and reasoning chains that cascade into suboptimal performance. In this paper, we introduce Early Knowledge Alignment (EKA), a simple but effective module that aligns LLMs with retrieval set before planning in iterative RAG systems with contextually relevant retrieved knowledge. Extensive experiments on six standard RAG datasets demonstrate that by establishing a stronger reasoning foundation, EKA significantly improves retrieval precision, reduces cascading errors, and enhances both performance and efficiency. Our analysis from an entropy perspective demonstrate that incorporating early knowledge reduces unnecessary exploration during the reasoning process, enabling the model to focus more effectively on relevant information subsets. Moreover, EKA proves effective as a versatile, training-free inference strategy that scales seamlessly to large models. Generalization tests across diverse datasets and retrieval corpora confirm the robustness of our approach. Overall, EKA advances the state-of-the-art in iterative RAG systems while illuminating the critical interplay between structured reasoning and efficient exploration in reinforcement learning-augmented frameworks. The code is released at \href{https://github.com/yxzwang/EarlyKnowledgeAlignment}{Github}.
>
---
#### [new 023] Cube Bench: A Benchmark for Spatial Visual Reasoning in MLLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Cube Bench基准，用于评估多模态大语言模型（MLLMs）的空间与序列推理能力。针对Rubik's Cube任务，分解为五项技能，统一评测框架下对比七种模型，揭示其在复杂度提升时性能骤降及开源/闭源差距，并验证自校正的有限增益。**

- **链接: [https://arxiv.org/pdf/2512.20595v1](https://arxiv.org/pdf/2512.20595v1)**

> **作者:** Dhruv Anand; Ehsan Shareghi
>
> **备注:** 27 pages, 5 figures, 9 tables. Cube available at https://github.com/dana-23/cube-bench
>
> **摘要:** We introduce Cube Bench, a Rubik's-cube benchmark for evaluating spatial and sequential reasoning in multimodal large language models (MLLMs). The benchmark decomposes performance into five skills: (i) reconstructing cube faces from images and text, (ii) choosing the optimal next move, (iii) predicting the outcome of a candidate move without applying it, (iv) executing multi-step plans while recovering from mistakes, and (v) detecting and revising one's own errors. Using a shared set of scrambled cube states, identical prompts and parsers, and a single distance-to-solved metric, we compare recent MLLMs side by side as a function of scramble depth. Across seven MLLMs, accuracy drops sharply with depth; once a trajectory stalls or diverges, models rarely recover, and high face-reconstruction accuracy does not guarantee competent action selection or multi-step execution. A pronounced closed- vs open-source gap emerges: the strongest closed model leads on both single-step perception tasks and multi-step control tasks, while open-weight models cluster near chance on the hardest settings; yet even the best MLLM degrades at higher cube complexity. A simple self-correction via reflective thinking yields modest gains but can also introduce overthinking. Cube Bench offers a compact, reproducible probe of sequential spatial reasoning in MLLMs.
>
---
#### [new 024] Fun-Audio-Chat Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-Audio-Chat，属大型音频语言模型（LALM）任务，旨在解决语音-文本模态间时序分辨率失配、计算开销大及文本LLM知识灾难性遗忘问题。通过双分辨率语音表征、Core-Cocktail训练和多任务DPO训练，实现高效高质语音理解与生成，并开源8B模型及代码。**

- **链接: [https://arxiv.org/pdf/2512.20156v1](https://arxiv.org/pdf/2512.20156v1)**

> **作者:** Qian Chen; Luyao Cheng; Chong Deng; Xiangang Li; Jiaqing Liu; Chao-Hong Tan; Wen Wang; Junhao Xu; Jieping Ye; Qinglin Zhang; Qiquan Zhang; Jingren Zhou
>
> **备注:** 21 pages, https://github.com/FunAudioLLM/Fun-Audio-Chat
>
> **摘要:** Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo.
>
---
#### [new 025] HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属医疗信息抽取任务，旨在解决真实世界肿瘤学笔记中结构化数据提取难的问题。提出HARMON-E框架，利用LLM作为推理代理，通过分层、自适应的多步推理，实现跨多文档的患者级临床变量（如生物标志物、用药）高精度结构化提取。**

- **链接: [https://arxiv.org/pdf/2512.19864v1](https://arxiv.org/pdf/2512.19864v1)**

> **作者:** Shashi Kant Gupta; Arijeet Pramanik; Jerrin John Thomas; Regina Schwind; Lauren Wiener; Avi Raju; Jeremy Kornbluth; Yanshan Wang; Zhaohui Su; Hrituraj Singh
>
> **备注:** 39 Pages, Supplementary Included
>
> **摘要:** Unstructured notes within the electronic health record (EHR) contain rich clinical information vital for cancer treatment decision making and research, yet reliably extracting structured oncology data remains challenging due to extensive variability, specialized terminology, and inconsistent document formats. Manual abstraction, although accurate, is prohibitively costly and unscalable. Existing automated approaches typically address narrow scenarios - either using synthetic datasets, restricting focus to document-level extraction, or isolating specific clinical variables (e.g., staging, biomarkers, histology) - and do not adequately handle patient-level synthesis across the large number of clinical documents containing contradictory information. In this study, we propose an agentic framework that systematically decomposes complex oncology data extraction into modular, adaptive tasks. Specifically, we use large language models (LLMs) as reasoning agents, equipped with context-sensitive retrieval and iterative synthesis capabilities, to exhaustively and comprehensively extract structured clinical variables from real-world oncology notes. Evaluated on a large-scale dataset of over 400,000 unstructured clinical notes and scanned PDF reports spanning 2,250 cancer patients, our method achieves an average F1-score of 0.93, with 100 out of 103 oncology-specific clinical variables exceeding 0.85, and critical variables (e.g., biomarkers and medications) surpassing 0.95. Moreover, integration of the agentic system into a data curation workflow resulted in 0.94 direct manual approval rate, significantly reducing annotation costs. To our knowledge, this constitutes the first exhaustive, end-to-end application of LLM-based agents for structured oncology data extraction at scale
>
---
#### [new 026] Can LLMs Solve My Grandma's Riddle? Evaluating Multilingual Large Language Models on Reasoning Traditional Bangla Tricky Riddles
- **分类: cs.CL**

- **简介: 该论文聚焦低资源、文化特异性推理任务，提出BanglaRiddleEval基准（1244个传统孟加拉语谜语），评估多语言大模型在生成问答、多项选择、歧义解析等任务上的表现。结果显示模型远低于人类水平，揭示其在隐喻与文化推理上的严重不足。**

- **链接: [https://arxiv.org/pdf/2512.20324v1](https://arxiv.org/pdf/2512.20324v1)**

> **作者:** Nurul Labib Sayeedi; Md. Faiyaz Abdullah Sayeedi; Khushnur Binte Jahangir; Swakkhar Shatabda; Sarah Masud Preum
>
> **摘要:** Large Language Models (LLMs) show impressive performance on many NLP benchmarks, yet their ability to reason in figurative, culturally grounded, and low-resource settings remains underexplored. We address this gap for Bangla by introducing BanglaRiddleEval, a benchmark of 1,244 traditional Bangla riddles instantiated across four tasks (4,976 riddle-task artifacts in total). Using an LLM-based pipeline, we generate Chain-of-Thought explanations, semantically coherent distractors, and fine-grained ambiguity annotations, and evaluate a diverse suite of open-source and closed-source models under different prompting strategies. Models achieve moderate semantic overlap on generative QA but low correctness, MCQ accuracy peaks at only about 56% versus an 83% human baseline, and ambiguity resolution ranges from roughly 26% to 68%, with high-quality explanations confined to the strongest models. These results show that current LLMs capture some cues needed for Bangla riddle reasoning but remain far from human-level performance, establishing BanglaRiddleEval as a challenging new benchmark for low-resource figurative reasoning. All data, code, and evaluation scripts are available on GitHub: https://github.com/Labib1610/BanglaRiddleEval.
>
---
#### [new 027] A Novel Graph-Sequence Learning Model for Inductive Text Classification
- **分类: cs.CL**

- **简介: 该论文面向归纳式文本分类任务，旨在解决现有GNN方法忽视多类型词对关系（如共现、句法、语义）及序列信息的问题。提出TextGSL模型：构建单文本图、设计多类型边与自适应消息传递，并融合Transformer捕获序列信息，提升文本表征能力。**

- **链接: [https://arxiv.org/pdf/2512.20097v1](https://arxiv.org/pdf/2512.20097v1)**

> **作者:** Zuo Wang; Ye Yuan
>
> **摘要:** Text classification plays an important role in various downstream text-related tasks, such as sentiment analysis, fake news detection, and public opinion analysis. Recently, text classification based on Graph Neural Networks (GNNs) has made significant progress due to their strong capabilities of structural relationship learning. However, these approaches still face two major limitations. First, these approaches fail to fully consider the diverse structural information across word pairs, e.g., co-occurrence, syntax, and semantics. Furthermore, they neglect sequence information in the text graph structure information learning module and can not classify texts with new words and relations. In this paper, we propose a Novel Graph-Sequence Learning Model for Inductive Text Classification (TextGSL) to address the previously mentioned issues. More specifically, we construct a single text-level graph for all words in each text and establish different edge types based on the diverse relationships between word pairs. Building upon this, we design an adaptive multi-edge message-passing paradigm to aggregate diverse structural information between word pairs. Additionally, sequential information among text data can be captured by the proposed TextGSL through the incorporation of Transformer layers. Therefore, TextGSL can learn more discriminative text representations. TextGSL has been comprehensively compared with several strong baselines. The experimental results on diverse benchmarking datasets demonstrate that TextGSL outperforms these baselines in terms of accuracy.
>
---
#### [new 028] How well do Large Language Models Recognize Instructional Moves? Establishing Baselines for Foundation Models in Educational Discourse
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在教育话语中识别“教学行为”（instructional moves）的基线能力。任务是分类真实课堂转录中的教学行为类型。作者评估6个LLM在零样本、单样本和少样本提示下的表现，发现少样本提示显著提升性能（最高Kappa=0.58），但存在类别差异与精度-召回权衡。**

- **链接: [https://arxiv.org/pdf/2512.19903v1](https://arxiv.org/pdf/2512.19903v1)**

> **作者:** Kirk Vanacore; Rene F. Kizilcec
>
> **摘要:** Large language models (LLMs) are increasingly adopted in educational technologies for a variety of tasks, from generating instructional materials and assisting with assessment design to tutoring. While prior work has investigated how models can be adapted or optimized for specific tasks, far less is known about how well LLMs perform at interpreting authentic educational scenarios without significant customization. As LLM-based systems become widely adopted by learners and educators in everyday academic contexts, understanding their out-of-the-box capabilities is increasingly important for setting expectations and benchmarking. We compared six LLMs to estimate their baseline performance on a simple but important task: classifying instructional moves in authentic classroom transcripts. We evaluated typical prompting methods: zero-shot, one-shot, and few-shot prompting. We found that while zero-shot performance was moderate, providing comprehensive examples (few-shot prompting) significantly improved performance for state-of-the-art models, with the strongest configuration reaching Cohen's Kappa = 0.58 against expert-coded annotations. At the same time, improvements were neither uniform nor complete: performance varied considerably by instructional move, and higher recall frequently came at the cost of increased false positives. Overall, these findings indicate that foundation models demonstrate meaningful yet limited capacity to interpret instructional discourse, with prompt design helping to surface capability but not eliminating fundamental reliability constraints.
>
---
#### [new 029] Making Large Language Models Efficient Dense Retrievers
- **分类: cs.IR; cs.CL**

- **简介: 该论文属信息检索任务，旨在解决LLM作为稠密检索器时参数量大、推理低效的问题。作者分析发现其MLP层冗余高、注意力层关键，据此提出EffiR框架，通过粗粒度深度剪枝与细粒度宽度压缩，显著减小模型并降低计算开销，同时保持检索性能。**

- **链接: [https://arxiv.org/pdf/2512.20612v1](https://arxiv.org/pdf/2512.20612v1)**

> **作者:** Yibin Lei; Shwai He; Ang Li; Andrew Yates
>
> **摘要:** Recent work has shown that directly fine-tuning large language models (LLMs) for dense retrieval yields strong performance, but their substantial parameter counts make them computationally inefficient. While prior studies have revealed significant layer redundancy in LLMs for generative tasks, it remains unclear whether similar redundancy exists when these models are adapted for retrieval tasks, which require encoding entire sequences into fixed representations rather than generating tokens iteratively. To this end, we conduct a comprehensive analysis of layer redundancy in LLM-based dense retrievers. We find that, in contrast to generative settings, MLP layers are substantially more prunable, while attention layers remain critical for semantic aggregation. Building on this insight, we propose EffiR, a framework for developing efficient retrievers that performs large-scale MLP compression through a coarse-to-fine strategy (coarse-grained depth reduction followed by fine-grained width reduction), combined with retrieval-specific fine-tuning. Across diverse BEIR datasets and LLM backbones, EffiR achieves substantial reductions in model size and inference cost while preserving the performance of full-size models.
>
---
#### [new 030] Reason2Decide: Rationale-Driven Multi-Task Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Reason2Decide，面向临床决策支持中的自解释预测任务，解决预测与解释不一致（暴露偏差）及任务割裂问题。通过两阶段训练：先用LLM生成理由预训练，再联合预测与理由生成并引入计划采样，提升准确性与解释一致性，且适配小模型。**

- **链接: [https://arxiv.org/pdf/2512.20074v1](https://arxiv.org/pdf/2512.20074v1)**

> **作者:** H M Quamran Hasan; Housam Khalifa Bashier; Jiayi Dai; Mi-Young Kim; Randy Goebel
>
> **摘要:** Despite the wide adoption of Large Language Models (LLM)s, clinical decision support systems face a critical challenge: achieving high predictive accuracy while generating explanations aligned with the predictions. Current approaches suffer from exposure bias leading to misaligned explanations. We propose Reason2Decide, a two-stage training framework that addresses key challenges in self-rationalization, including exposure bias and task separation. In Stage-1, our model is trained on rationale generation, while in Stage-2, we jointly train on label prediction and rationale generation, applying scheduled sampling to gradually transition from conditioning on gold labels to model predictions. We evaluate Reason2Decide on three medical datasets, including a proprietary triage dataset and public biomedical QA datasets. Across model sizes, Reason2Decide outperforms other fine-tuning baselines and some zero-shot LLMs in prediction (F1) and rationale fidelity (BERTScore, BLEU, LLM-as-a-Judge). In triage, Reason2Decide is rationale source-robust across LLM-generated, nurse-authored, and nurse-post-processed rationales. In our experiments, while using only LLM-generated rationales in Stage-1, Reason2Decide outperforms other fine-tuning variants. This indicates that LLM-generated rationales are suitable for pretraining models, reducing reliance on human annotations. Remarkably, Reason2Decide achieves these gains with models 40x smaller than contemporary foundation models, making clinical reasoning more accessible for resource-constrained deployments while still providing explainable decision support.
>
---
#### [new 031] Brain-Grounded Axes for Reading and Steering LLM States
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属LLM可解释性与可控性任务，旨在解决文本监督方向缺乏外部 grounding 的问题。作者利用MEG脑活动构建词级脑图谱，提取ICA轴作为坐标系，训练轻量适配器将LLM隐状态映射至脑轴，实现无需微调的神经生理学驱动的状态读取与 steering。**

- **链接: [https://arxiv.org/pdf/2512.19399v1](https://arxiv.org/pdf/2512.19399v1)**

> **作者:** Sandro Andric
>
> **备注:** 10 pages, 4 figures. Code: https://github.com/sandroandric/Brain-Grounded-Axes-for-Reading-and-Steering-LLM-States
>
> **摘要:** Interpretability methods for large language models (LLMs) typically derive directions from textual supervision, which can lack external grounding. We propose using human brain activity not as a training signal but as a coordinate system for reading and steering LLM states. Using the SMN4Lang MEG dataset, we construct a word-level brain atlas of phase-locking value (PLV) patterns and extract latent axes via ICA. We validate axes with independent lexica and NER-based labels (POS/log-frequency used as sanity checks), then train lightweight adapters that map LLM hidden states to these brain axes without fine-tuning the LLM. Steering along the resulting brain-derived directions yields a robust lexical (frequency-linked) axis in a mid TinyLlama layer, surviving perplexity-matched controls, and a brain-vs-text probe comparison shows larger log-frequency shifts (relative to the text probe) with lower perplexity for the brain axis. A function/content axis (axis 13) shows consistent steering in TinyLlama, Qwen2-0.5B, and GPT-2, with PPL-matched text-level corroboration. Layer-4 effects in TinyLlama are large but inconsistent, so we treat them as secondary (Appendix). Axis structure is stable when the atlas is rebuilt without GPT embedding-change features or with word2vec embeddings (|r|=0.64-0.95 across matched axes), reducing circularity concerns. Exploratory fMRI anchoring suggests potential alignment for embedding change and log frequency, but effects are sensitive to hemodynamic modeling assumptions and are treated as population-level evidence only. These results support a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior.
>
---
#### [new 032] Automated stereotactic radiosurgery planning using a human-in-the-loop reasoning large language model agent
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文面向SRS自动计划任务，解决黑盒AI临床难采纳问题。提出人机协同的LLM代理SAGE，对比推理型与非推理型模型：前者在剂量指标上媲美人工，显著降低耳蜗剂量，并展现可审计的约束验证与权衡推理行为。**

- **链接: [https://arxiv.org/pdf/2512.20586v1](https://arxiv.org/pdf/2512.20586v1)**

> **作者:** Humza Nusrat; Luke Francisco; Bing Luo; Hassan Bagher-Ebadian; Joshua Kim; Karen Chin-Snyder; Salim Siddiqui; Mira Shah; Eric Mellon; Mohammad Ghassemi; Anthony Doemer; Benjamin Movsas; Kundan Thind
>
> **摘要:** Stereotactic radiosurgery (SRS) demands precise dose shaping around critical structures, yet black-box AI systems have limited clinical adoption due to opacity concerns. We tested whether chain-of-thought reasoning improves agentic planning in a retrospective cohort of 41 patients with brain metastases treated with 18 Gy single-fraction SRS. We developed SAGE (Secure Agent for Generative Dose Expertise), an LLM-based planning agent for automated SRS treatment planning. Two variants generated plans for each case: one using a non-reasoning model, one using a reasoning model. The reasoning variant showed comparable plan dosimetry relative to human planners on primary endpoints (PTV coverage, maximum dose, conformity index, gradient index; all p > 0.21) while reducing cochlear dose below human baselines (p = 0.022). When prompted to improve conformity, the reasoning model demonstrated systematic planning behaviors including prospective constraint verification (457 instances) and trade-off deliberation (609 instances), while the standard model exhibited none of these deliberative processes (0 and 7 instances, respectively). Content analysis revealed that constraint verification and causal explanation concentrated in the reasoning agent. The optimization traces serve as auditable logs, offering a path toward transparent automated planning.
>
---
#### [new 033] Learning to Reason in LLMs by Expectation Maximization
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属大模型推理优化任务，旨在提升LLM通过生成理由（rationale）进行推理的准确性。提出将推理建模为隐变量问题，用EM框架统一学习目标，并比较多种采样策略；实验表明简化的提示后验采样（PPS）效果最优。**

- **链接: [https://arxiv.org/pdf/2512.20169v1](https://arxiv.org/pdf/2512.20169v1)**

> **作者:** Junghyun Lee; Branislav Kveton; Sunav Choudhary; Subhojyoti Mukherjee; Anup Rao; Ryan A. Rossi; Alexa Siu
>
> **备注:** 12 pages, 3 figures, 1 table
>
> **摘要:** Large language models (LLMs) solve reasoning problems by first generating a rationale and then answering. We formalize reasoning as a latent variable model and derive an expectation-maximization (EM) objective for learning to reason. This view connects EM and modern reward-based optimization, and shows that the main challenge lies in designing a sampling distribution that generates rationales that justify correct answers. We instantiate and compare several sampling schemes: rejection sampling with a budget, self-taught reasoner (STaR), and prompt posterior sampling (PPS), which only keeps the rationalization stage of STaR. Our experiments on the ARC, MMLU, and OpenBookQA datasets with the Llama and Qwen models show that the sampling scheme can significantly affect the accuracy of learned reasoning models. Despite its simplicity, we observe that PPS outperforms the other sampling schemes.
>
---
#### [new 034] Coherence in the brain unfolds across separable temporal regimes
- **分类: q-bio.NC; cs.CL**

- **简介: 该论文属神经语言学任务，探究语言理解中“语境渐进整合”与“事件快速重配置”两种时间尺度的神经机制。作者用LLM提取无标注的drift/shift信号，结合7T fMRI数据构建编码模型，发现默认网络主司慢速drift，听觉/语言皮层主司快速shift。**

- **链接: [https://arxiv.org/pdf/2512.20481v1](https://arxiv.org/pdf/2512.20481v1)**

> **作者:** Davide Stauba; Finn Rabe; Akhil Misra; Yves Pauli; Roya Hüppi; Nils Lang; Lars Michels; Victoria Edkins; Sascha Frühholz; Iris Sommer; Wolfram Hinzen; Philipp Homan
>
> **摘要:** Coherence in language requires the brain to satisfy two competing temporal demands: gradual accumulation of meaning across extended context and rapid reconfiguration of representations at event boundaries. Despite their centrality to language and thought, how these processes are implemented in the human brain during naturalistic listening remains unclear. Here, we tested whether these two processes can be captured by annotation-free drift and shift signals and whether their neural expression dissociates across large-scale cortical systems. These signals were derived from a large language model (LLM) and formalized contextual drift and event shifts directly from the narrative input. To enable high-precision voxelwise encoding models with stable parameter estimates, we densely sampled one healthy adult across more than 7 hours of listening to thirteen crime stories while collecting ultra high-field (7T) BOLD data. We then modeled the feature-informed hemodynamic response using a regularized encoding framework validated on independent stories. Drift predictions were prevalent in default-mode network hubs, whereas shift predictions were evident bilaterally in the primary auditory cortex and language association cortex. Furthermore, activity in default-mode and parietal networks was best explained by a signal capturing how meaning accumulates and gradually fades over the course of the narrative. Together, these findings show that coherence during language comprehension is implemented through dissociable neural regimes of slow contextual integration and rapid event-driven reconfiguration, offering a mechanistic entry point for understanding disturbances of language coherence in psychiatric disorders.
>
---
#### [new 035] Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出视觉-语言仿真模型（VLSM），解决工业数字孪生中从草图和自然语言生成可执行FlexScript代码的问题。构建首个12万+三元组数据集，设计SVR、PMR、ESR三项新指标，并通过模型消融验证其高结构准确率与执行鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.20387v1](https://arxiv.org/pdf/2512.20387v1)**

> **作者:** YuChe Hsu; AnJui Wang; TsaiChing Ni; YuanFu Yang
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** We propose a Vision-Language Simulation Model (VLSM) that unifies visual and textual understanding to synthesize executable FlexScript from layout sketches and natural-language prompts, enabling cross-modal reasoning for industrial simulation systems. To support this new paradigm, the study constructs the first large-scale dataset for generative digital twins, comprising over 120,000 prompt-sketch-code triplets that enable multimodal learning between textual descriptions, spatial structures, and simulation logic. In parallel, three novel evaluation metrics, Structural Validity Rate (SVR), Parameter Match Rate (PMR), and Execution Success Rate (ESR), are proposed specifically for this task to comprehensively evaluate structural integrity, parameter fidelity, and simulator executability. Through systematic ablation across vision encoders, connectors, and code-pretrained language backbones, the proposed models achieve near-perfect structural accuracy and high execution robustness. This work establishes a foundation for generative digital twins that integrate visual reasoning and language understanding into executable industrial simulation systems.
>
---
#### [new 036] Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文提出自然语言驱动的文档图像检索（NL-DIR）任务，解决现有方法仅支持图像查询、难以处理细粒度文本查询的问题。构建含41K文档图像及205K人工校验文本查询的新基准数据集，评估主流视觉语言模型，并设计高效两阶段检索方法。**

- **链接: [https://arxiv.org/pdf/2512.20174v1](https://arxiv.org/pdf/2512.20174v1)**

> **作者:** Hao Guo; Xugong Qin; Jun Jie Ou Yang; Peng Zhang; Gangyan Zeng; Yubo Li; Hailun Lin
>
> **备注:** CVPR 2025
>
> **摘要:** Document image retrieval (DIR) aims to retrieve document images from a gallery according to a given query. Existing DIR methods are primarily based on image queries that retrieve documents within the same coarse semantic category, e.g., newspapers or receipts. However, these methods struggle to effectively retrieve document images in real-world scenarios where textual queries with fine-grained semantics are usually provided. To bridge this gap, we introduce a new Natural Language-based Document Image Retrieval (NL-DIR) benchmark with corresponding evaluation metrics. In this work, natural language descriptions serve as semantically rich queries for the DIR task. The NL-DIR dataset contains 41K authentic document images, each paired with five high-quality, fine-grained semantic queries generated and evaluated through large language models in conjunction with manual verification. We perform zero-shot and fine-tuning evaluations of existing mainstream contrastive vision-language models and OCR-free visual document understanding (VDU) models. A two-stage retrieval method is further investigated for performance improvement while achieving both time and space efficiency. We hope the proposed NL-DIR benchmark can bring new opportunities and facilitate research for the VDU community. Datasets and codes will be publicly available at huggingface.co/datasets/nianbing/NL-DIR.
>
---
## 更新

#### [replaced 001] Vision Language Models are Confused Tourists
- **分类: cs.CV; cs.CL**

- **简介: 该论文属多模态鲁棒性评估任务，旨在解决VLMs在多元文化混合输入下性能骤降的问题。作者提出ConfusedTourist基准，通过图像堆叠等扰动测试模型对地理文化线索的稳定性，并发现模型因注意力被干扰而失效，揭示其文化鲁棒性缺陷。**

- **链接: [https://arxiv.org/pdf/2511.17004v3](https://arxiv.org/pdf/2511.17004v3)**

> **作者:** Patrick Amadeus Irawan; Ikhlasul Akmal Hanif; Muhammad Dehan Al Kautsar; Genta Indra Winata; Fajri Koto; Alham Fikri Aji
>
> **摘要:** Although the cultural dimension has been one of the key aspects in evaluating Vision-Language Models (VLMs), their ability to remain stable across diverse cultural inputs remains largely untested, despite being crucial to support diversity and multicultural societies. Existing evaluations often rely on benchmarks featuring only a singular cultural concept per image, overlooking scenarios where multiple, potentially unrelated cultural cues coexist. To address this gap, we introduce ConfusedTourist, a novel cultural adversarial robustness suite designed to assess VLMs' stability against perturbed geographical cues. Our experiments reveal a critical vulnerability, where accuracy drops heavily under simple image-stacking perturbations and even worsens with its image-generation-based variant. Interpretability analyses further show that these failures stem from systematic attention shifts toward distracting cues, diverting the model from its intended focus. These findings highlight a critical challenge: visual cultural concept mixing can substantially impair even state-of-the-art VLMs, underscoring the urgent need for more culturally robust multimodal understanding.
>
---
#### [replaced 002] Zero-Overhead Introspection for Adaptive Test-Time Compute
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ZIP-RC方法，解决大语言模型缺乏实时奖励与计算成本自省能力的问题。它在单次前向传播中复用logits，零开销预测最终奖励与剩余长度，实现自适应测试时推理，在数学任务上以更低或相等成本提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.01457v4](https://arxiv.org/pdf/2512.01457v4)**

> **作者:** Rohin Manvi; Joey Hong; Tim Seyde; Maxime Labonne; Mathias Lechner; Sergey Levine
>
> **摘要:** Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, which equips models with zero-overhead introspective predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.
>
---
#### [replaced 003] C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文面向大语言模型的推理任务，旨在解决RL方法（如GRPO）在推理中过度自信、缺乏自我认知的问题。提出C²GSPG方法：构建组序列策略梯度框架消除token级偏差，并引入基于序列概率与奖励的置信度校准正则项，提升推理准确率与置信度一致性。**

- **链接: [https://arxiv.org/pdf/2509.23129v2](https://arxiv.org/pdf/2509.23129v2)**

> **作者:** Haotian Liu; Shuo Wang; Hongteng Xu
>
> **摘要:** Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models. However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants. In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration. The code of C$^2$GSPG is available at https://github.com/HaotianLiu123/CCGSPG.
>
---
#### [replaced 004] SiamGPT: Quality-First Fine-Tuning for Stable Thai Text Generation
- **分类: cs.CL**

- **简介: 该论文属大语言模型本地化任务，旨在解决开源大模型在泰语生成中指令遵循差、多轮不稳定等问题。作者提出Quality-First微调策略，基于Qwen3-32B构建SiamGPT-32B，融合翻译高复杂度指令数据与泰语适配的AutoIF框架，仅用监督微调提升泰语生成质量。**

- **链接: [https://arxiv.org/pdf/2512.19455v2](https://arxiv.org/pdf/2512.19455v2)**

> **作者:** Thittipat Pairatsuppawat; Abhibhu Tachaapornchai; Paweekorn Kusolsomboon; Chutikan Chaiwong; Thodsaporn Chay-intr; Kobkrit Viriyayudhakorn; Nongnuch Ketui; Aslan B. Wong
>
> **摘要:** Open-weights large language models remain difficult to deploy for Thai due to unstable generation under complex instructions, despite strong English performance. To mitigate these limitations, We present SiamGPT-32B, an open-weights model based on Qwen3-32B, fine-tuned with a Quality-First strategy emphasizing curated supervision over data scale. The fine-tuning pipeline combines translated high-complexity English instruction data with a Thai-adapted AutoIF framework for instruction and linguistic constraints. Using supervised fine-tuning only, without continual pretraining or corpus expansion, SiamGPT-32B improves instruction adherence, multi-turn robustness, and linguistic stability. Evaluations on the SEA-HELM benchmark show that SiamGPT-32B achieves the strongest overall performance among similar-scale open-weights Thai models, with consistent gains in instruction following, multi-turn dialogue, and natural language understanding.
>
---
#### [replaced 005] Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning
- **分类: eess.AS; cs.CL**

- **简介: 该论文面向低资源语音识别（ASR）域适应任务，解决Speech LLM在缺乏配对语音-文本数据时难以适配新领域的问题。提出仅用目标域无标注文本进行微调的方法，并引入实时评估机制保持语音-文本对齐，避免遗忘源域性能。**

- **链接: [https://arxiv.org/pdf/2506.05671v2](https://arxiv.org/pdf/2506.05671v2)**

> **作者:** Yangui Fang; Jing Peng; Xu Li; Yu Xi; Chengwei Zhang; Guohui Zhong; Kai Yu
>
> **备注:** This paper has been ACCEPTED for publication in ASRU
>
> **摘要:** Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
>
---
#### [replaced 006] GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators
- **分类: cs.CL**

- **简介: 该论文提出GenEnv框架，属LLM智能体训练任务，旨在解决真实交互数据成本高、静态导致训练低效的问题。通过构建代理与生成式环境模拟器的难度对齐协同进化机制，动态生成适配代理当前能力的任务，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.19682v2](https://arxiv.org/pdf/2512.19682v2)**

> **作者:** Jiacheng Guo; Ling Yang; Peter Chen; Qixin Xiao; Yinjie Wang; Xinzhe Juan; Jiahao Qiu; Ke Shen; Mengdi Wang
>
> **备注:** Our codes are available at https://github.com/Gen-Verse/GenEnv
>
> **摘要:** Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
>
---
#### [replaced 007] External Hippocampus: Topological Cognitive Maps for Guiding Large Language Model Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“外部海马体”框架，属小模型多步推理任务，旨在解决认知僵局问题。通过构建语义空间的拓扑认知图，实现测试时低开销、可干预的能量流导航，无需训练，显著提升准确率与推理速度。**

- **链接: [https://arxiv.org/pdf/2512.18190v2](https://arxiv.org/pdf/2512.18190v2)**

> **作者:** Jian Yan
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** This paper proposes the External Hippocampus framework, which models language model reasoning from a cognitive dynamics perspective as the flow of information energy in semantic space. Unlike traditional weight-space optimization methods, this framework constructs topological cognitive maps through dimensionality reduction projection, enabling precise navigation and intervention of energy flow at test time while avoiding substantial computational requirements and demonstrating predictable intervention patterns. The method effectively addresses the cognitive deadlock problem in multi-step reasoning for small models. Experiments on models <=7B parameters show: map-guided methods achieve 81.20% accuracy on 500 challenging problems (relative baseline +16.80%), reduce reasoning time by >= 15x, with key findings revealing that reasoning stagnation manifests as "Cognitive Vortex" and low-entropy potential wells, while temperature perturbations effectively restart energy flow. The framework requires no additional training, possesses autonomous growth capability, and provides an efficient and controllable topological-aware solution for small model reasoning.
>
---
#### [replaced 008] On Structured State-Space Duality
- **分类: cs.LG; cs.CL; cs.CV; stat.ML**

- **简介: 该论文研究结构化状态空间模型（SSM）与注意力机制的等价性问题，属模型理论分析任务。它将SSD从标量恒等矩阵推广至一般对角SSM，给出等价于1-半可分掩码注意力的充要条件，并证明其不适用于标准Softmax注意力。**

- **链接: [https://arxiv.org/pdf/2510.04944v2](https://arxiv.org/pdf/2510.04944v2)**

> **作者:** Jerry Yao-Chieh Hu; Xiwen Zhang; Ali ElSheikh; Weimin Wu; Han Liu
>
> **备注:** v2 fixed typos and added numerical results (Appendix B)
>
> **摘要:** Structured State-Space Duality (SSD) [Dao & Gu, ICML 2024] is an equivalence between a simple Structured State-Space Model (SSM) and a masked attention mechanism. In particular, a state-space model with a scalar-times-identity state matrix is equivalent to a masked self-attention with a $1$-semiseparable causal mask. Consequently, the same sequence transformation (model) has two algorithmic realizations: as a linear-time $O(T)$ recurrence or as a quadratic-time $O(T^2)$ attention. In this note, we formalize and generalize this duality: (i) we extend SSD from the scalar-identity case to general diagonal SSMs (diagonal state matrices); (ii) we show that these diagonal SSMs match the scalar case's training complexity lower bounds while supporting richer dynamics; (iii) we establish a necessary and sufficient condition under which an SSM is equivalent to $1$-semiseparable masked attention; and (iv) we show that such duality fails to extend to standard softmax attention due to rank explosion. Together, these results tighten bridge between recurrent SSMs and Transformers, and widen the design space for expressive yet efficient sequence models.
>
---
#### [replaced 009] Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属序列建模任务，旨在解决Transformer在测试序列远超训练长度时性能骤降的问题。作者提出PRISM方法，用概率化、连续相对位置编码替代传统确定性位置嵌入，实现高达10倍长度外推，在算法推理等基准上达SOTA。**

- **链接: [https://arxiv.org/pdf/2506.00920v2](https://arxiv.org/pdf/2506.00920v2)**

> **作者:** Philip Heejun Lee
>
> **备注:** Note: v2: working paper; code, additional baselines, ablations, will follow in v3
>
> **摘要:** Deep sequence models typically degrade in accuracy when test sequences significantly exceed their training lengths, yet many critical tasks--such as algorithmic reasoning, multi-step arithmetic, and compositional generalization--require robust length extrapolation. We introduce PRISM, a Probabilistic Relative-position Implicit Superposition Model, a novel positional encoding mechanism that enables Transformers to extrapolate accurately up to 10x beyond their training length. PRISM learns continuous relative positions through a differentiable histogram-filter update, preserving position uncertainty via a probabilistic superposition rather than conventional deterministic embeddings. Empirically, PRISM achieves state-of-the-art length extrapolation, successfully generalizing to previously intractable sequence lengths across algorithmic benchmarks--including arithmetic (addition, multiplication), SCAN compositionality tasks, and complex copy variants derived from DeepMind's recent datasets. Our analysis demonstrates that PRISM's stochastic positional encoding maintains sharp and interpretable internal states, providing a theoretical basis for reliable length generalization. These results advance the goal of neural sequence models that remain algorithmically robust at lengths far exceeding their training horizon.
>
---
#### [replaced 010] Thematic Dispersion in Arabic Applied Linguistics: A Bibliometric Analysis using Brookes' Measure
- **分类: cs.CL**

- **简介: 该论文属 bibliometric 分析任务，旨在量化阿拉伯应用语言学领域的主题分散程度。它采用 Brookes' Δ 指标，基于2019–2025年1564篇文献，计算得 Δ=0.194，证实领域高度异质、无主导范式，并验证了方法的适用性与可复现性。**

- **链接: [https://arxiv.org/pdf/2512.15328v2](https://arxiv.org/pdf/2512.15328v2)**

> **作者:** Ayman Eddakrouri; Amani Ramadan
>
> **摘要:** This study applies Brookes' Measure of Categorical Dispersion (Δ) to analyze the thematic structure of contemporary Arabic Applied Linguistics research. Using a comprehensive, real-world dataset of 1,564 publications from 2019 to 2025, classified into eight core sub-disciplines, we calculate a dispersion index of Δ = 0.194. This remarkably low value indicates extreme thematic dispersion, revealing that the field is characterized by pronounced heterogeneity rather than concentration. The analysis identifies Computational Linguistics as a dominant but non-hegemonic force, coexisting with robust research in Sociolinguistics, Language Teaching, and other subfields. This study clarifies the correct application of Brookes' original formula, demonstrates its utility for field characterization, and provides a replicable bibliometric methodology for assessing disciplinary structure across domains.
>
---
#### [replaced 011] DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations
- **分类: cs.CL**

- **简介: 该论文提出DrVoice模型，解决端到端语音-文本联合生成中模态感知弱、计算开销大问题。通过双分辨率语音表征（将输入频率从12.5Hz降至5Hz），实现语音与文本的并行自回归建模，提升LLM对语音的理解与生成能力，在多项语音基准上达到7B级SOTA。**

- **链接: [https://arxiv.org/pdf/2506.09349v4](https://arxiv.org/pdf/2506.09349v4)**

> **作者:** Chao-Hong Tan; Qian Chen; Wen Wang; Chong Deng; Qinglin Zhang; Luyao Cheng; Hai Yu; Xin Zhang; Xiang Lv; Tianyu Zhao; Chong Zhang; Yukun Ma; Yafeng Chen; Hui Wang; Jiaqing Liu; Xiangang Li; Jieping Ye
>
> **备注:** Work in progress
>
> **摘要:** Recent studies on end-to-end (E2E) speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing E2E approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM's autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech-text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents DrVoice, a parallel speech-text voice conversation model based on joint autoregressive modeling, featuring dual-resolution speech representations. Notably, while current methods utilize mainly 12.5Hz input audio representation, our proposed dual-resolution mechanism reduces the input frequency for the LLM to 5Hz, significantly reducing computational cost and alleviating the frequency discrepancy between speech and text tokens and in turn better exploiting LLMs' capabilities. Experimental results demonstrate that DrVoice-7B establishes new state-of-the-art (SOTA) on prominent speech benchmarks including OpenAudioBench, VoiceBench, UltraEval-Audio and Big Bench Audio, making it a leading open-source speech foundation model in ~7B models.
>
---
#### [replaced 012] Learning without training: The implicit dynamics of in-context learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）的上下文学习机制，旨在解释其“无需训练即可学习”的现象。作者理论结合实验，揭示Transformer块中自注意力与MLP层协同作用，能将输入上下文隐式转化为MLP权重的低秩更新，从而实现推理时的快速适应。**

- **链接: [https://arxiv.org/pdf/2507.16003v3](https://arxiv.org/pdf/2507.16003v3)**

> **作者:** Benoit Dherin; Michael Munn; Hanna Mazzawi; Michael Wunder; Javier Gonzalvo
>
> **摘要:** One of the most striking features of Large Language Models (LLMs) is their ability to learn in-context. Namely at inference time an LLM is able to learn new patterns without any additional weight update when these patterns are presented in the form of examples in the prompt, even if these patterns were not seen during training. The mechanisms through which this can happen are still largely unknown. In this work, we show that the stacking of a self-attention layer with an MLP, allows the transformer block to implicitly modify the weights of the MLP layer according to the context. We argue through theory and experimentation that this simple mechanism may be the reason why LLMs can learn in-context and not only during training. Specifically, we show how a transformer block implicitly transforms a context into a low-rank weight-update of its MLP layer.
>
---
#### [replaced 013] AWPO: Enhancing Tool-Use of Large Language Models through Explicit Integration of Reasoning Rewards
- **分类: cs.CL**

- **简介: 该论文属大语言模型工具使用增强任务，旨在解决现有RL方法忽视显式推理奖励、导致推理与工具调用能力不足的问题。提出AWPO框架，通过方差感知门控、难度感知加权及裁剪机制，自适应融合推理奖励，显著提升多轮工具使用性能。**

- **链接: [https://arxiv.org/pdf/2512.19126v2](https://arxiv.org/pdf/2512.19126v2)**

> **作者:** Zihan Lin; Xiaohan Wang; Hexiong Yang; Jiajun Chai; Jie Cao; Guojun Yin; Wei Lin; Ran He
>
> **摘要:** While reinforcement learning (RL) shows promise in training tool-use large language models (LLMs) using verifiable outcome rewards, existing methods largely overlook the potential of explicit reasoning rewards to bolster reasoning and tool utilization. Furthermore, natively combining reasoning and outcome rewards may yield suboptimal performance or conflict with the primary optimization objective. To address this, we propose advantage-weighted policy optimization (AWPO) -- a principled RL framework that effectively integrates explicit reasoning rewards to enhance tool-use capability. AWPO incorporates variance-aware gating and difficulty-aware weighting to adaptively modulate advantages from reasoning signals based on group-relative statistics, alongside a tailored clipping mechanism for stable optimization. Extensive experiments demonstrate that AWPO achieves state-of-the-art performance across standard tool-use benchmarks, significantly outperforming strong baselines and leading closed-source models in challenging multi-turn scenarios. Notably, with exceptional parameter efficiency, our 4B model surpasses Grok-4 by 16.0 percent in multi-turn accuracy while preserving generalization capability on the out-of-distribution MMLU-Pro benchmark.
>
---
#### [replaced 014] Why mask diffusion does not work
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属生成式语言模型研究任务，旨在揭示掩码扩散模型难以实现并行生成与双向注意力的根本原因。作者分析其内在缺陷，并提出最优的训练与推理策略。**

- **链接: [https://arxiv.org/pdf/2510.03289v2](https://arxiv.org/pdf/2510.03289v2)**

> **作者:** Haocheng Sun; Cynthia Xin Wen; Edward Hong Wang
>
> **摘要:** The main advantages of diffusion language models over autoregressive (AR) models lie in their ability to support parallel generation and bidirectional attention, enabling a more controllable generation process. In recent years, open-source mask diffusion language models have emerged, most of which are based on a variant known as absorbing diffusion. However, this paper demonstrates why mask diffusion faces inherent difficulties in achieving parallel generation and bidirectional attention. We also propose the most effective training and inference strategies for mask diffusion.
>
---
#### [replaced 015] Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction
- **分类: cs.CL; eess.AS**

- **简介: 该论文属ASR错误纠正任务，旨在解决LLM直接纠错易产生幻觉、误改正确文本的问题。提出三阶段无训练框架RLLM-CF：错误预检测、思维链迭代修正、推理过程验证，显著降低CER/WER。**

- **链接: [https://arxiv.org/pdf/2505.24347v3](https://arxiv.org/pdf/2505.24347v3)**

> **作者:** Yangui Fang; Baixu Chen; Jing Peng; Xu Li; Yu Xi; Chengwei Zhang; Guohui Zhong
>
> **备注:** This paper has been ACCEPTED for publication in ASRU
>
> **摘要:** Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
>
---
#### [replaced 016] DramaBench: A Six-Dimensional Evaluation Framework for Drama Script Continuation
- **分类: cs.CL**

- **简介: 该论文面向戏剧剧本续写任务，解决现有基准无法全面评估角色一致性、情节连贯性等多维质量的问题。作者提出首个六维评测框架DramaBench（含格式、叙事效率、角色、情感、逻辑、冲突），融合规则分析与LLM标注，对8个模型在千余剧本上开展系统评测与验证。**

- **链接: [https://arxiv.org/pdf/2512.19012v2](https://arxiv.org/pdf/2512.19012v2)**

> **作者:** Shijian Ma; Yunqi Huang; Yan Lin
>
> **备注:** Project page: https://dramabench.pages.dev/
>
> **摘要:** Drama script continuation requires models to maintain character consistency, advance plot coherently, and preserve dramatic structurecapabilities that existing benchmarks fail to evaluate comprehensively. We present DramaBench, the first large-scale benchmark for evaluating drama script continuation across six independent dimensions: Format Standards, Narrative Efficiency, Character Consistency, Emotional Depth, Logic Consistency, and Conflict Handling. Our framework combines rulebased analysis with LLM-based labeling and statistical metrics, ensuring objective and reproducible evaluation. We conduct comprehensive evaluation of 8 state-of-the-art language models on 1,103 scripts (8,824 evaluations total), with rigorous statistical significance testing (252 pairwise comparisons, 65.9% significant) and human validation (188 scripts, substantial agreement on 3/5 dimensions). Our ablation studies confirm all six dimensions capture independent quality aspects (mean | r | = 0.020). DramaBench provides actionable, dimensionspecific feedback for model improvement and establishes a rigorous standard for creative writing evaluation.
>
---
#### [replaced 017] Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief
- **分类: cs.CL**

- **简介: 该论文属模型校准任务，旨在解决LLM过度自信、不确定性估计不准的问题。提出EAGLE方法，利用多层内部隐状态提取并聚合“内部信念”，通过期望计算生成更可靠的置信度分数，提升模型校准性能。**

- **链接: [https://arxiv.org/pdf/2509.01564v2](https://arxiv.org/pdf/2509.01564v2)**

> **作者:** Zeguan Xiao; Diyang Dou; Boya Xiong; Yun Chen; Guanhua Chen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range.
>
---
#### [replaced 018] SoK: Are Watermarks in LLMs Ready for Deployment?
- **分类: cs.CR; cs.CL**

- **简介: 该论文属系统性综述（SoK）任务，旨在评估LLM水印技术的实用部署成熟度。它构建水印分类法、提出IP分类器评估水印效果与影响、分析现有方法局限，并指出水印损害模型实用性，尚难满足真实场景需求。**

- **链接: [https://arxiv.org/pdf/2506.05594v3](https://arxiv.org/pdf/2506.05594v3)**

> **作者:** Kieu Dang; Phung Lai; NhatHai Phan; Yelong Shen; Ruoming Jin; Abdallah Khreishah; My T. Thai
>
> **摘要:** Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
>
---
#### [replaced 019] Latent learning: episodic memory complements parametric learning by enabling flexible reuse of experiences
- **分类: cs.LG; cs.CL**

- **简介: 该论文属AI泛化能力研究任务，旨在解决机器学习系统因缺乏“潜伏学习”导致的泛化失败问题。作者受认知科学启发，提出用外显记忆（如检索机制）补充参数化学习，并验证其在语言建模、导航等任务中提升泛化与数据效率的效果。**

- **链接: [https://arxiv.org/pdf/2509.16189v3](https://arxiv.org/pdf/2509.16189v3)**

> **作者:** Andrew Kyle Lampinen; Martin Engelcke; Yuxuan Li; Arslan Chaudhry; James L. McClelland
>
> **摘要:** When do machine learning systems fail to generalize, and what mechanisms could improve their generalization? Here, we draw inspiration from cognitive science to argue that one weakness of parametric machine learning systems is their failure to exhibit latent learning -- learning information that is not relevant to the task at hand, but that might be useful in a future task. We show how this perspective links failures ranging from the reversal curse in language modeling to new findings on agent-based navigation. We then highlight how cognitive science points to episodic memory as a potential part of the solution to these issues. Correspondingly, we show that a system with an oracle retrieval mechanism can use learning experiences more flexibly to generalize better across many of these challenges. We also identify some of the essential components for effectively using retrieval, including the importance of within-example in-context learning for acquiring the ability to use information across retrieved examples. In summary, our results illustrate one possible contributor to the relative data inefficiency of current machine learning systems compared to natural intelligence, and help to understand how retrieval methods can complement parametric learning to improve generalization. We close by discussing some of the links between these findings and prior results in cognitive science and neuroscience, and the broader implications.
>
---
#### [replaced 020] GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大语言模型推理增强任务，旨在解决传统MoE中专家独立、缺乏协作的问题。提出GRAPHMOE，通过伪图结构与自反思机制实现专家间信息流动，并引入循环路由策略模拟迭代思考，结合LoRA高效训练，在多个基准上达SOTA。**

- **链接: [https://arxiv.org/pdf/2501.07890v3](https://arxiv.org/pdf/2501.07890v3)**

> **作者:** Bo Lv; Chen Tang; Zifan Zheng; Bohao Yang; Kun Zhao; Ning Liao; Xiaoxing Wang; Feiyu Xiong; Zhiyu Li; Nayu Liu; Jingchi Jiang
>
> **备注:** 10 pages
>
> **摘要:** Traditional Mixture-of-Experts (MoE) networks benefit from utilizing multiple smaller expert models as opposed to a single large network. However, these experts typically operate independently, leaving a question open about whether interconnecting these models could enhance the performance of MoE networks. In response, we introduce GRAPHMOE, a novel method aimed at augmenting the cognitive depth of language models via a self-rethinking mechanism constructed on Pseudo GraphMoE networks. GRAPHMOE employs a recurrent routing strategy to simulate iterative thinking steps, thereby facilitating the flow of information among expert nodes. We implement the GRAPHMOE architecture using Low-Rank Adaptation techniques (LoRA) and conduct extensive experiments on various benchmark datasets. The experimental results reveal that GRAPHMOE outperforms other LoRA based models, achieving state-of-the-art (SOTA) performance. Additionally, this study explores a novel recurrent routing strategy that may inspire further advancements in enhancing the reasoning capabilities of language models.
>
---
#### [replaced 021] Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History
- **分类: cs.CL; cs.AI**

- **简介: 该论文属AI安全与可信赖性评估任务，旨在解决大语言模型（LLM）人格测量不稳定的难题。作者提出PERSIST框架，系统评测25个开源模型在不同规模、提示、推理模式及对话历史下的人格一致性，发现当前LLM缺乏行为稳定性基础，现有对齐策略难以保障安全应用所需的一致性。**

- **链接: [https://arxiv.org/pdf/2508.04826v3](https://arxiv.org/pdf/2508.04826v3)**

> **作者:** Tommaso Tosato; Saskia Helbling; Yorguin-Jose Mantilla-Ramos; Mahmood Hegazy; Alberto Tosato; David John Lemay; Irina Rish; Guillaume Dumas
>
> **备注:** Accepted at AAAI 2026, Track on AI Alignment
>
> **摘要:** Large language models require consistent behavioral patterns for safe deployment, yet there are indications of large variability that may lead to an instable expression of personality traits in these models. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25 open-source models (1B-685B parameters) across 2 million+ responses. Using traditional (BFI, SD3) and novel LLM-adapted personality questionnaires, we systematically vary model size, personas, reasoning modes, question order or paraphrasing, and conversation history. Our findings challenge fundamental assumptions: (1) Question reordering alone can introduce large shifts in personality measurements; (2) Scaling provides limited stability gains: even 400B+ models exhibit standard deviations >0.3 on 5-point scales; (3) Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability; (4) Detailed persona instructions produce mixed effects, with misaligned personas showing significantly higher variability than the helpful assistant baseline; (5) The LLM-adapted questionnaires, despite their improved ecological validity, exhibit instability comparable to human-centric versions. This persistent instability across scales and mitigation strategies suggests that current LLMs lack the architectural foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that current alignment strategies may be inadequate.
>
---
#### [replaced 022] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文聚焦MLLM作为验证器的任务，旨在解决其在开放域（如网页导航、机器人操作）中普遍存在的“同意偏差”——即过度认可错误行为的问题。作者提出两步式自扎根验证（SGV）方法，通过先生成先验再条件评估，显著提升失败检测与准确性，并推动下游任务性能突破。**

- **链接: [https://arxiv.org/pdf/2507.11662v2](https://arxiv.org/pdf/2507.11662v2)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** Our code, models, and data are publicly available at https://mshalimay.github.io/agreement-bias-sgv/
>
> **摘要:** Verifiers--functions assigning rewards to agent behavior--have been key for AI progress in domains like math and code. However, extending gains to domains without clear-cut success criteria (e.g., computer use) remains a challenge: while humans can recognize desired outcomes, translating this intuition into scalable rules is nontrivial. Multimodal Large Language Models (MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers across web navigation, computer use, and robotic manipulation, and identify a critical limitation: a strong tendency to over-validate agent behavior, a phenomenon we term agreement bias. This bias is pervasive across models, resilient to test-time scaling, and poses risks to existing methods relying on MLLM evaluations. We discuss methods to evaluate and improve MLLM verifiers and introduce Self-Grounded Verification (SGV), a lightweight method that harnesses MLLMs' own sampling mechanisms by modulating (un)conditional generation to better leverage their knowledge, alignment, and reasoning. SGV operates in two steps: first, the MLLM is elicited to generate broad priors about desired behavior, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. SGV yields more human-aligned evaluations with gains of up to 25pp in failure detection, 14pp in accuracy, and benefits extending to downstream applications. In self-refinement and online supervision, SGV boosts task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena--setting a new state of the art, surpassing the previous best by 20pp. We release an updated version of VisualWebArena featuring more human-aligned evaluators, high-fidelity environment parallelism, and speedups of over 10x.
>
---
#### [replaced 023] Large Language Models Develop Novel Social Biases Through Adaptive Exploration
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属AI伦理与偏见研究任务，旨在揭示LLMs在决策中自发生成新社会偏见的问题。作者通过心理学范式发现，模型因探索不足而对人工群体形成刻板印象，导致不公平任务分配；并验证了显式激励探索等干预措施的有效性。**

- **链接: [https://arxiv.org/pdf/2511.06148v2](https://arxiv.org/pdf/2511.06148v2)**

> **作者:** Addison J. Wu; Ryan Liu; Xuechunzi Bai; Thomas L. Griffiths
>
> **摘要:** As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time.
>
---
#### [replaced 024] VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属多模态长上下文理解任务，旨在探究视觉-文本压缩（VTC）对VLM长上下文能力的影响。作者构建首个VTC专用基准VTCBench，含检索、推理、记忆三类子任务，并评估主流模型，发现VTC虽提升效率，却严重损害VLM的长依赖建模能力。**

- **链接: [https://arxiv.org/pdf/2512.15649v2](https://arxiv.org/pdf/2512.15649v2)**

> **作者:** Hongbo Zhao; Meng Wang; Fei Zhu; Wenzhuo Liu; Bolin Ni; Fanhu Zeng; Gaofeng Meng; Zhaoxiang Zhang
>
> **摘要:** The computational and memory overheads associated with expanding the context window of LLMs severely limit their scalability. A noteworthy solution is vision-text compression (VTC), exemplified by frameworks like DeepSeek-OCR and Glyph, which convert long texts into dense 2D visual representations, thereby achieving token compression ratios of 3x-20x. However, the impact of this high information density on the core long-context capabilities of vision-language models (VLMs) remains under-investigated. To address this gap, we introduce the first benchmark for VTC and systematically assess the performance of VLMs across three long-context understanding settings: VTC-Retrieval, which evaluates the model's ability to retrieve and aggregate information; VTC-Reasoning, which requires models to infer latent associations to locate facts with minimal lexical overlap; and VTC-Memory, which measures comprehensive question answering within long-term dialogue memory. Furthermore, we establish the VTCBench-Wild to simulate diverse input scenarios.We comprehensively evaluate leading open-source and proprietary models on our benchmarks. The results indicate that, despite being able to decode textual information (e.g., OCR) well, most VLMs exhibit a surprisingly poor long-context understanding ability with VTC-processed information, failing to capture long associations or dependencies in the context.This study provides a deep understanding of VTC and serves as a foundation for designing more efficient and scalable VLMs.
>
---
#### [replaced 025] Generative Retrieval with Few-shot Indexing
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属生成式检索任务，旨在解决训练式索引成本高、难适配动态语料等问题。提出无需训练的少样本索引框架：用LLM直接生成文档ID构建ID库，检索时约束LLM在库中生成ID并映射回文档，并引入一对多映射提升效果。**

- **链接: [https://arxiv.org/pdf/2408.02152v3](https://arxiv.org/pdf/2408.02152v3)**

> **作者:** Arian Askari; Chuan Meng; Mohammad Aliannejadi; Zhaochun Ren; Evangelos Kanoulas; Suzan Verberne
>
> **备注:** Accepted for publication at the 48th European Conference on Information Retrieval (ECIR 2026)
>
> **摘要:** Existing generative retrieval (GR) methods rely on training-based indexing, which fine-tunes a model to memorise associations between queries and the document identifiers (docids) of relevant documents. Training-based indexing suffers from high training costs, under-utilisation of pre-trained knowledge in large language models (LLMs), and limited adaptability to dynamic document corpora. To address the issues, we propose a few-shot indexing-based GR framework (Few-Shot GR). It has a few-shot indexing process without any training, where we prompt an LLM to generate docids for all documents in a corpus, ultimately creating a docid bank for the entire corpus. During retrieval, we feed a query to the same LLM and constrain it to generate a docid within the docid bank created during indexing, and then map the generated docid back to its corresponding document. Moreover, we devise few-shot indexing with one-to-many mapping to further enhance Few-Shot GR. Experiments show that Few-Shot GR achieves superior performance to state-of-the-art GR methods requiring heavy training.
>
---
#### [replaced 026] Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属Transformer位置编码研究任务，旨在解决RoPE中“内容”与“位置”信息纠缠导致的建模偏差问题。作者提出极坐标位置编码（PoPE），解耦二者，提升位置/内容独立匹配能力，在语言、音乐、基因等序列建模中显著降低损失并增强零-shot长度外推性能。**

- **链接: [https://arxiv.org/pdf/2509.10534v2](https://arxiv.org/pdf/2509.10534v2)**

> **作者:** Anand Gopalakrishnan; Robert Csordás; Jürgen Schmidhuber; Michael C. Mozer
>
> **备注:** Comparison to YaRN added + additional bias visualization + model ablation
>
> **摘要:** The attention mechanism in a Transformer architecture matches key to query based on both content -- the what -- and position in a sequence -- the where. We present an analysis indicating that what and where are entangled in the popular RoPE rotary position embedding. This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call Polar Coordinate Position Embeddings or PoPE, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities compared not only to RoPE but even a method designed for extrapolation, YaRN, which requires additional fine tuning and frequency interpolation.
>
---
#### [replaced 027] Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Object-Oriented Programming
- **分类: cs.CL; cs.SE**

- **简介: 该论文属AI工程实践任务，旨在解决机器学习/深度学习系统代码复杂、难维护、不可复用等问题。工作是系统阐述OOP四大原则在AI项目中的应用，结合Python示例，展示如何用封装、继承、多态等构建模块化、可扩展的机器学习系统。**

- **链接: [https://arxiv.org/pdf/2409.19916v5](https://arxiv.org/pdf/2409.19916v5)**

> **作者:** Tianyang Wang; Ziqian Bi; Keyu Chen; Jiawei Xu; Qian Niu; Junyu Liu; Benji Peng; Ming Li; Sen Zhang; Xuanhe Pan; Jinlang Wang; Pohsun Feng; Yizhu Wen; Xinyuan Song; Ming Liu
>
> **备注:** 49pages
>
> **摘要:** Object-Oriented Programming (OOP) has become a crucial paradigm for managing the growing complexity of modern software systems, particularly in fields like machine learning, deep learning, large language models (LLM), and data analytics. This work provides a comprehensive introduction to the integration of OOP techniques within these domains, with a focus on improving code modularity, maintainability, and scalability. We begin by outlining the evolution of computing and the rise of OOP, followed by an in-depth discussion of key OOP principles such as encapsulation, inheritance, polymorphism, and abstraction. The practical application of these principles is demonstrated using Python, a widely adopted language in AI and data science. Furthermore, we examine how design patterns and modular programming can be employed to enhance the structure and efficiency of machine learning systems. In subsequent sections, we apply these OOP concepts to real-world AI tasks, including the encapsulation of preprocessing workflows, machine learning model training, and evaluation. Detailed examples illustrate how OOP can be used to build reusable, scalable machine learning systems while maintaining code clarity and reducing redundancy.This work is intended to serve as a bridge for both beginners and experienced developers, equipping them with the necessary knowledge to apply OOP methodologies in AI-driven projects, ultimately fostering the development of more robust and maintainable systems.
>
---
#### [replaced 028] Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank
- **分类: cs.CL; cs.LG**

- **简介: 该论文面向极端多标签文本分类任务，解决注意力机制难训练、尤其在少样本和罕见标签下效果差的问题。提出PLANT方法：利用学习排序模型，基于互信息增益预训练并“种植”标签特异性注意力，实现即插即用的注意力初始化，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2410.23066v3](https://arxiv.org/pdf/2410.23066v3)**

> **作者:** Debjyoti Saha Roy; Byron C. Wallace; Javed A. Aslam
>
> **摘要:** State-of-the-art Extreme Multi-Label Text Classification models rely on multi-label attention to focus on key tokens in input text, but learning good attention weights is challenging. We introduce PLANT - Pretrained and Leveraged Attention - a plug-and-play strategy for initializing attention. PLANT works by planting label-specific attention using a pretrained Learning-to-Rank model guided by mutual information gain. This architecture-agnostic approach integrates seamlessly with large language model backbones such as Mistral-7B, LLaMA3-8B, DeepSeek-V3, and Phi-3. PLANT outperforms state-of-the-art methods across tasks including ICD coding, legal topic classification, and content recommendation. Gains are especially pronounced in few-shot settings, with substantial improvements on rare labels. Ablation studies confirm that attention initialization is a key driver of these gains. For code and trained models, see https://github.com/debjyotiSRoy/xcube/tree/plant
>
---
#### [replaced 029] Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大模型指令微调任务，旨在解决长链思维（long-CoT）指令数据规模大、训练开销高、缺乏高效自动筛选方法的问题。提出Select2Reason框架，基于问题难度与推理轨迹长度联合加权排序，高效选取高价值样本，显著提升微调效率与性能。**

- **链接: [https://arxiv.org/pdf/2505.17266v3](https://arxiv.org/pdf/2505.17266v3)**

> **作者:** Cehao Yang; Xueyuan Lin; Xiaojun Wu; Chengjin Xu; Xuhui Jiang; Honghao Liu; Hui Xiong; Jian Guo
>
> **摘要:** A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost.
>
---

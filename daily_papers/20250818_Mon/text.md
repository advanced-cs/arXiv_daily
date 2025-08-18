# 自然语言处理 cs.CL

- **最新发布 63 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] LLM Compression: How Far Can We Go in Balancing Size and Performance?
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，研究如何在减少模型大小的同时保持性能。通过量化技术评估不同模型在多个NLP任务上的表现，分析压缩与性能的平衡。**

- **链接: [http://arxiv.org/pdf/2508.11318v1](http://arxiv.org/pdf/2508.11318v1)**

> **作者:** Sahil Sk; Debasish Dhal; Sonal Khosla; Sk Shahid; Sambit Shekhar; Akash Dhaka; Shantipriya Parida; Dilip K. Prasad; Ondřej Bojar
>
> **备注:** This paper has been accepted for presentation at the RANLP 2025 conference
>
> **摘要:** Quantization is an essential and popular technique for improving the accessibility of large language models (LLMs) by reducing memory usage and computational costs while maintaining performance. In this study, we apply 4-bit Group Scaling Quantization (GSQ) and Generative Pretrained Transformer Quantization (GPTQ) to LLaMA 1B, Qwen 0.5B, and PHI 1.5B, evaluating their impact across multiple NLP tasks. We benchmark these models on MS MARCO (Information Retrieval), BoolQ (Boolean Question Answering), and GSM8K (Mathematical Reasoning) datasets, assessing both accuracy and efficiency across various tasks. The study measures the trade-offs between model compression and task performance, analyzing key evaluation metrics, namely accuracy, inference latency, and throughput (total output tokens generated per second), providing insights into the suitability of low-bit quantization for real-world deployment. Using the results, users can then make suitable decisions based on the specifications that need to be met. We discuss the pros and cons of GSQ and GPTQ techniques on models of different sizes, which also serve as a benchmark for future experiments.
>
---
#### [new 002] Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决跨语言知识迁移问题。通过训练小模型研究语言间表征统一性，提出方法调节跨语言迁移效果。**

- **链接: [http://arxiv.org/pdf/2508.11017v1](http://arxiv.org/pdf/2508.11017v1)**

> **作者:** Carter Blum; Katja Filipova; Ann Yuan; Asma Ghandeharioun; Julian Zimmert; Fred Zhang; Jessica Hoffmann; Tal Linzen; Martin Wattenberg; Lucas Dixon; Mor Geva
>
> **摘要:** Large language models (LLMs) struggle with cross-lingual knowledge transfer: they hallucinate when asked in one language about facts expressed in a different language during training. This work introduces a controlled setting to study the causes and dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets. We identify a learning phase wherein a model develops either separate or unified representations of the same facts across languages, and show that unification is essential for cross-lingual transfer. We also show that the degree of unification depends on mutual information between facts and training data language, and on how easy it is to extract that language. Based on these insights, we develop methods to modulate the level of cross-lingual transfer by manipulating data distribution and tokenization, and we introduce metrics and visualizations to formally characterize their effects on unification. Our work shows how controlled settings can shed light on pre-training dynamics and suggests new directions for improving cross-lingual transfer in LLMs.
>
---
#### [new 003] CoDiEmb: A Collaborative yet Distinct Framework for Unified Representation Learning in Information Retrieval and Semantic Textual Similarity
- **分类: cs.CL**

- **简介: 该论文属于信息检索与语义文本相似性任务，解决联合训练中负迁移问题。提出CoDiEmb框架，通过任务专用目标、动态采样和模型融合提升统一表示学习效果。**

- **链接: [http://arxiv.org/pdf/2508.11442v1](http://arxiv.org/pdf/2508.11442v1)**

> **作者:** Bowen Zhang; Zixin Song; Chunquan Chen; Qian-Wen Zhang; Di Yin; Xing Sun
>
> **摘要:** Learning unified text embeddings that excel across diverse downstream tasks is a central goal in representation learning, yet negative transfer remains a persistent obstacle. This challenge is particularly pronounced when jointly training a single encoder for Information Retrieval (IR) and Semantic Textual Similarity (STS), two essential but fundamentally disparate tasks for which naive co-training typically yields steep performance trade-offs. We argue that resolving this conflict requires systematically decoupling task-specific learning signals throughout the training pipeline. To this end, we introduce CoDiEmb, a unified framework that reconciles the divergent requirements of IR and STS in a collaborative yet distinct manner. CoDiEmb integrates three key innovations for effective joint optimization: (1) Task-specialized objectives paired with a dynamic sampler that forms single-task batches and balances per-task updates, thereby preventing gradient interference. For IR, we employ a contrastive loss with multiple positives and hard negatives, augmented by cross-device sampling. For STS, we adopt order-aware objectives that directly optimize correlation and ranking consistency. (2) A delta-guided model fusion strategy that computes fine-grained merging weights for checkpoints by analyzing each parameter's deviation from its pre-trained initialization, proving more effective than traditional Model Soups. (3) An efficient, single-stage training pipeline that is simple to implement and converges stably. Extensive experiments on 15 standard IR and STS benchmarks across three base encoders validate CoDiEmb. Our results and analysis demonstrate that the framework not only mitigates cross-task trade-offs but also measurably improves the geometric properties of the embedding space.
>
---
#### [new 004] Rule2Text: A Framework for Generating and Evaluating Natural Language Explanations of Knowledge Graph Rules
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱解释任务，旨在解决逻辑规则难以理解的问题。通过生成自然语言解释并评估其质量，提升知识图谱的可访问性与可用性。**

- **链接: [http://arxiv.org/pdf/2508.10971v1](http://arxiv.org/pdf/2508.10971v1)**

> **作者:** Nasim Shirvani-Mahdavi; Chengkai Li
>
> **备注:** arXiv admin note: text overlap with arXiv:2507.23740
>
> **摘要:** Knowledge graphs (KGs) can be enhanced through rule mining; however, the resulting logical rules are often difficult for humans to interpret due to their inherent complexity and the idiosyncratic labeling conventions of individual KGs. This work presents Rule2Text, a comprehensive framework that leverages large language models (LLMs) to generate natural language explanations for mined logical rules, thereby improving KG accessibility and usability. We conduct extensive experiments using multiple datasets, including Freebase variants (FB-CVT-REV, FB+CVT-REV, and FB15k-237) as well as the ogbl-biokg dataset, with rules mined using AMIE 3.5.1. We systematically evaluate several LLMs across a comprehensive range of prompting strategies, including zero-shot, few-shot, variable type incorporation, and Chain-of-Thought reasoning. To systematically assess models' performance, we conduct a human evaluation of generated explanations on correctness and clarity. To address evaluation scalability, we develop and validate an LLM-as-a-judge framework that demonstrates strong agreement with human evaluators. Leveraging the best-performing model (Gemini 2.0 Flash), LLM judge, and human-in-the-loop feedback, we construct high-quality ground truth datasets, which we use to fine-tune the open-source Zephyr model. Our results demonstrate significant improvements in explanation quality after fine-tuning, with particularly strong gains in the domain-specific dataset. Additionally, we integrate a type inference module to support KGs lacking explicit type information. All code and data are publicly available at https://github.com/idirlab/KGRule2NL.
>
---
#### [new 005] AgentMental: An Interactive Multi-Agent Framework for Explainable and Adaptive Mental Health Assessment
- **分类: cs.CL**

- **简介: 该论文属于心理健康评估任务，旨在解决传统方法依赖人工和静态分析的不足。提出多智能体框架，通过动态交互和自适应提问提升评估效果。**

- **链接: [http://arxiv.org/pdf/2508.11567v1](http://arxiv.org/pdf/2508.11567v1)**

> **作者:** Jinpeng Hu; Ao Wang; Qianqian Xie; Hui Ma; Zhuo Li; Dan Guo
>
> **摘要:** Mental health assessment is crucial for early intervention and effective treatment, yet traditional clinician-based approaches are limited by the shortage of qualified professionals. Recent advances in artificial intelligence have sparked growing interest in automated psychological assessment, yet most existing approaches are constrained by their reliance on static text analysis, limiting their ability to capture deeper and more informative insights that emerge through dynamic interaction and iterative questioning. Therefore, in this paper, we propose a multi-agent framework for mental health evaluation that simulates clinical doctor-patient dialogues, with specialized agents assigned to questioning, adequacy evaluation, scoring, and updating. We introduce an adaptive questioning mechanism in which an evaluation agent assesses the adequacy of user responses to determine the necessity of generating targeted follow-up queries to address ambiguity and missing information. Additionally, we employ a tree-structured memory in which the root node encodes the user's basic information, while child nodes (e.g., topic and statement) organize key information according to distinct symptom categories and interaction turns. This memory is dynamically updated throughout the interaction to reduce redundant questioning and further enhance the information extraction and contextual tracking capabilities. Experimental results on the DAIC-WOZ dataset illustrate the effectiveness of our proposed method, which achieves better performance than existing approaches.
>
---
#### [new 006] SafeConstellations: Steering LLM Safety to Reduce Over-Refusals Through Task-Specific Trajectory
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM过度拒绝合法指令的问题。通过分析嵌入空间中的轨迹模式，提出SafeConstellations方法，有效降低过度拒绝率。**

- **链接: [http://arxiv.org/pdf/2508.11290v1](http://arxiv.org/pdf/2508.11290v1)**

> **作者:** Utsav Maskey; Sumit Yadav; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that superficially resemble harmful content. This phenomena diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through comprehensive evaluation, we demonstrate that LLMs still tend to refuse responses to harmful instructions when those instructions are reframed to appear as benign tasks. Our mechanistic analysis reveal that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, and by preserving general model behavior, our method reduces over-refusal rates by up to 73% with minimal impact on utility-offering a principled approach to mitigating over-refusals.
>
---
#### [new 007] AI in Mental Health: Emotional and Sentiment Analysis of Large Language Models' Responses to Depression, Anxiety, and Stress Queries
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在研究LLMs对抑郁、焦虑和压力问题的回应情绪特征，分析不同模型和用户群体的反应差异。**

- **链接: [http://arxiv.org/pdf/2508.11285v1](http://arxiv.org/pdf/2508.11285v1)**

> **作者:** Arya VarastehNezhad; Reza Tavasoli; Soroush Elyasi; MohammadHossein LotfiNia; Hamed Farbeh
>
> **摘要:** Depression, anxiety, and stress are widespread mental health concerns that increasingly drive individuals to seek information from Large Language Models (LLMs). This study investigates how eight LLMs (Claude Sonnet, Copilot, Gemini Pro, GPT-4o, GPT-4o mini, Llama, Mixtral, and Perplexity) reply to twenty pragmatic questions about depression, anxiety, and stress when those questions are framed for six user profiles (baseline, woman, man, young, old, and university student). The models generated 2,880 answers, which we scored for sentiment and emotions using state-of-the-art tools. Our analysis revealed that optimism, fear, and sadness dominated the emotional landscape across all outputs, with neutral sentiment maintaining consistently high values. Gratitude, joy, and trust appeared at moderate levels, while emotions such as anger, disgust, and love were rarely expressed. The choice of LLM significantly influenced emotional expression patterns. Mixtral exhibited the highest levels of negative emotions including disapproval, annoyance, and sadness, while Llama demonstrated the most optimistic and joyful responses. The type of mental health condition dramatically shaped emotional responses: anxiety prompts elicited extraordinarily high fear scores (0.974), depression prompts generated elevated sadness (0.686) and the highest negative sentiment, while stress-related queries produced the most optimistic responses (0.755) with elevated joy and trust. In contrast, demographic framing of queries produced only marginal variations in emotional tone. Statistical analyses confirmed significant model-specific and condition-specific differences, while demographic influences remained minimal. These findings highlight the critical importance of model selection in mental health applications, as each LLM exhibits a distinct emotional signature that could significantly impact user experience and outcomes.
>
---
#### [new 008] Overcoming Low-Resource Barriers in Tulu: Neural Models and Corpus Creation for OffensiveLanguage Identification
- **分类: cs.CL**

- **简介: 该论文属于 Offensive Language Identification 任务，旨在解决低资源语言 Tulu 的网络内容检测问题。工作包括构建首个基准数据集并评估多种模型效果。**

- **链接: [http://arxiv.org/pdf/2508.11166v1](http://arxiv.org/pdf/2508.11166v1)**

> **作者:** Anusha M D; Deepthi Vikram; Bharathi Raja Chakravarthi; Parameshwar R Hegde
>
> **备注:** 20 pages, 3 tables, 3 figures. Submitted to Language Resources and Evaluation (Springer)
>
> **摘要:** Tulu, a low-resource Dravidian language predominantly spoken in southern India, has limited computational resources despite its growing digital presence. This study presents the first benchmark dataset for Offensive Language Identification (OLI) in code-mixed Tulu social media content, collected from YouTube comments across various domains. The dataset, annotated with high inter-annotator agreement (Krippendorff's alpha = 0.984), includes 3,845 comments categorized into four classes: Not Offensive, Not Tulu, Offensive Untargeted, and Offensive Targeted. We evaluate a suite of deep learning models, including GRU, LSTM, BiGRU, BiLSTM, CNN, and attention-based variants, alongside transformer architectures (mBERT, XLM-RoBERTa). The BiGRU model with self-attention achieves the best performance with 82% accuracy and a 0.81 macro F1-score. Transformer models underperform, highlighting the limitations of multilingual pretraining in code-mixed, under-resourced contexts. This work lays the foundation for further NLP research in Tulu and similar low-resource, code-mixed languages.
>
---
#### [new 009] MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文提出MoNaCo基准，用于评估模型处理复杂、自然的多文档推理任务。旨在解决现有基准不够复杂的问题，通过构建高质量问答数据集推动模型发展。**

- **链接: [http://arxiv.org/pdf/2508.11133v1](http://arxiv.org/pdf/2508.11133v1)**

> **作者:** Tomer Wolfson; Harsh Trivedi; Mor Geva; Yoav Goldberg; Dan Roth; Tushar Khot; Ashish Sabharwal; Reut Tsarfaty
>
> **备注:** Accepted for publication in Transactions of the Association for Computational Linguistics (TACL), 2025. Authors pre-print
>
> **摘要:** Large language models (LLMs) are emerging as a go-to tool for querying information. However, current LLM benchmarks rarely feature natural questions that are both information-seeking as well as genuinely time-consuming for humans. To address this gap we introduce MoNaCo, a benchmark of 1,315 natural and complex questions that require dozens, and at times hundreds, of intermediate steps to solve -- far more than any existing QA benchmark. To build MoNaCo, we developed a decomposed annotation pipeline to elicit and manually answer natural time-consuming questions at scale. Frontier LLMs evaluated on MoNaCo achieve at most 61.2% F1, hampered by low recall and hallucinations. Our results underscore the need for reasoning models that better handle the complexity and sheer breadth of real-world information-seeking questions -- with MoNaCo providing an effective resource for tracking such progress. The MONACO benchmark, codebase, prompts and models predictions are publicly available at: https://tomerwolgithub.github.io/monaco
>
---
#### [new 010] When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM对提示格式变化的敏感性，评估5种提升提示鲁棒性的方法，旨在提高模型在实际应用中的稳定性。**

- **链接: [http://arxiv.org/pdf/2508.11383v1](http://arxiv.org/pdf/2508.11383v1)**

> **作者:** Mikhail Seleznyov; Mikhail Chaichuk; Gleb Ershov; Alexander Panchenko; Elena Tutubalina; Oleg Somov
>
> **摘要:** Large Language Models (LLMs) are highly sensitive to subtle, non-semantic variations in prompt phrasing and formatting. In this work, we present the first systematic evaluation of 5 methods for improving prompt robustness within a unified experimental framework. We benchmark these techniques on 8 models from Llama, Qwen and Gemma families across 52 tasks from Natural Instructions dataset. Our evaluation covers robustness methods from both fine-tuned and in-context learning paradigms, and tests their generalization against multiple types of distribution shifts. Finally, we extend our analysis to GPT-4.1 and DeepSeek V3 to assess frontier models' current robustness to format perturbations. Our findings offer actionable insights into the relative effectiveness of these robustness methods, enabling practitioners to make informed decisions when aiming for stable and reliable LLM performance in real-world applications. Code: https://github.com/AIRI-Institute/when-punctuation-matters.
>
---
#### [new 011] Hell or High Water: Evaluating Agentic Recovery from External Failures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11027v1](http://arxiv.org/pdf/2508.11027v1)**

> **作者:** Andrew Wang; Sophia Hager; Adi Asija; Daniel Khashabi; Nicholas Andrews
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** As language model agents are applied to real world problems of increasing complexity, they will be expected to formulate plans across large search spaces. If those plans fail for reasons beyond their control, how well do language agents search for alternative ways to achieve their goals? We devise a specialized agentic planning benchmark to study this question. Each planning problem is solved via combinations of function calls. The agent searches for relevant functions from a set of over four thousand possibilities, and observes environmental feedback in the form of function outputs or error messages. Our benchmark confronts the agent with external failures in its workflow, such as functions that suddenly become unavailable. At the same time, even with the introduction of these failures, we guarantee that the task remains solvable. Ideally, an agent's performance on the planning task should not be affected by the presence of external failures. Overall, we find that language agents struggle to formulate and execute backup plans in response to environment feedback. While state-of-the-art models are often able to identify the correct function to use in the right context, they struggle to adapt to feedback from the environment and often fail to pursue alternate courses of action, even when the search space is artificially restricted. We provide a systematic analysis of the failures of both open-source and commercial models, examining the effects of search space size, as well as the benefits of scaling model size in our setting. Our analysis identifies key challenges for current generative models as well as promising directions for future work.
>
---
#### [new 012] Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决传统方法在结构与语义信息整合上的不足。提出HGRAG方法，通过超图实现跨粒度信息融合，提升问答效果与效率。**

- **链接: [http://arxiv.org/pdf/2508.11247v1](http://arxiv.org/pdf/2508.11247v1)**

> **作者:** Changjian Wang; Weihong Deng; Weili Guan; Quan Lu; Ning Jiang
>
> **摘要:** Multi-hop question answering (MHQA) requires integrating knowledge scattered across multiple passages to derive the correct answer. Traditional retrieval-augmented generation (RAG) methods primarily focus on coarse-grained textual semantic similarity and ignore structural associations among dispersed knowledge, which limits their effectiveness in MHQA tasks. GraphRAG methods address this by leveraging knowledge graphs (KGs) to capture structural associations, but they tend to overly rely on structural information and fine-grained word- or phrase-level retrieval, resulting in an underutilization of textual semantics. In this paper, we propose a novel RAG approach called HGRAG for MHQA that achieves cross-granularity integration of structural and semantic information via hypergraphs. Structurally, we construct an entity hypergraph where fine-grained entities serve as nodes and coarse-grained passages as hyperedges, and establish knowledge association through shared entities. Semantically, we design a hypergraph retrieval method that integrates fine-grained entity similarity and coarse-grained passage similarity via hypergraph diffusion. Finally, we employ a retrieval enhancement module, which further refines the retrieved results both semantically and structurally, to obtain the most relevant passages as context for answer generation with the LLM. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods in QA performance, and achieves a 6$\times$ speedup in retrieval efficiency.
>
---
#### [new 013] Representing Speech Through Autoregressive Prediction of Cochlear Tokens
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出AuriStream，一种基于听觉处理机制的语音表示学习模型，解决语音编码与语义理解问题，通过两阶段框架提取并生成语音特征。**

- **链接: [http://arxiv.org/pdf/2508.11598v1](http://arxiv.org/pdf/2508.11598v1)**

> **作者:** Greta Tuckute; Klemen Kotar; Evelina Fedorenko; Daniel L. K. Yamins
>
> **摘要:** We introduce AuriStream, a biologically inspired model for encoding speech via a two-stage framework inspired by the human auditory processing hierarchy. The first stage transforms raw audio into a time-frequency representation based on the human cochlea, from which we extract discrete \textbf{cochlear tokens}. The second stage applies an autoregressive sequence model over the cochlear tokens. AuriStream learns meaningful phoneme and word representations, and state-of-the-art lexical semantics. AuriStream shows competitive performance on diverse downstream SUPERB speech tasks. Complementing AuriStream's strong representational capabilities, it generates continuations of audio which can be visualized in a spectrogram space and decoded back into audio, providing insights into the model's predictions. In summary, we present a two-stage framework for speech representation learning to advance the development of more human-like models that efficiently handle a range of speech-based tasks.
>
---
#### [new 014] gpt-oss-120b & gpt-oss-20b Model Card
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍gpt-oss-120b和gpt-oss-20b模型，属于自然语言处理任务，旨在提升推理准确性和降低推理成本，通过混合专家架构和强化学习训练实现。**

- **链接: [http://arxiv.org/pdf/2508.10925v1](http://arxiv.org/pdf/2508.10925v1)**

> **作者:** OpenAI; :; Sandhini Agarwal; Lama Ahmad; Jason Ai; Sam Altman; Andy Applebaum; Edwin Arbus; Rahul K. Arora; Yu Bai; Bowen Baker; Haiming Bao; Boaz Barak; Ally Bennett; Tyler Bertao; Nivedita Brett; Eugene Brevdo; Greg Brockman; Sebastien Bubeck; Che Chang; Kai Chen; Mark Chen; Enoch Cheung; Aidan Clark; Dan Cook; Marat Dukhan; Casey Dvorak; Kevin Fives; Vlad Fomenko; Timur Garipov; Kristian Georgiev; Mia Glaese; Tarun Gogineni; Adam Goucher; Lukas Gross; Katia Gil Guzman; John Hallman; Jackie Hehir; Johannes Heidecke; Alec Helyar; Haitang Hu; Romain Huet; Jacob Huh; Saachi Jain; Zach Johnson; Chris Koch; Irina Kofman; Dominik Kundel; Jason Kwon; Volodymyr Kyrylov; Elaine Ya Le; Guillaume Leclerc; James Park Lennon; Scott Lessans; Mario Lezcano-Casado; Yuanzhi Li; Zhuohan Li; Ji Lin; Jordan Liss; Lily; Liu; Jiancheng Liu; Kevin Lu; Chris Lu; Zoran Martinovic; Lindsay McCallum; Josh McGrath; Scott McKinney; Aidan McLaughlin; Song Mei; Steve Mostovoy; Tong Mu; Gideon Myles; Alexander Neitz; Alex Nichol; Jakub Pachocki; Alex Paino; Dana Palmie; Ashley Pantuliano; Giambattista Parascandolo; Jongsoo Park; Leher Pathak; Carolina Paz; Ludovic Peran; Dmitry Pimenov; Michelle Pokrass; Elizabeth Proehl; Huida Qiu; Gaby Raila; Filippo Raso; Hongyu Ren; Kimmy Richardson; David Robinson; Bob Rotsted; Hadi Salman; Suvansh Sanjeev; Max Schwarzer; D. Sculley; Harshit Sikchi; Kendal Simon; Karan Singhal; Yang Song; Dane Stuckey; Zhiqing Sun; Philippe Tillet; Sam Toizer; Foivos Tsimpourlas; Nikhil Vyas; Eric Wallace; Xin Wang; Miles Wang; Olivia Watkins; Kevin Weil; Amy Wendling; Kevin Whinnery; Cedric Whitney; Hannah Wong; Lin Yang; Yu Yang; Michihiro Yasunaga; Kristen Ying; Wojciech Zaremba; Wenting Zhan; Cyril Zhang; Brian Zhang; Eddie Zhang; Shengjia Zhao
>
> **摘要:** We present gpt-oss-120b and gpt-oss-20b, two open-weight reasoning models that push the frontier of accuracy and inference cost. The models use an efficient mixture-of-expert transformer architecture and are trained using large-scale distillation and reinforcement learning. We optimize the models to have strong agentic capabilities (deep research browsing, python tool use, and support for developer-provided functions), all while using a rendered chat format that enables clear instruction following and role delineation. Both models achieve strong results on benchmarks ranging from mathematics, coding, and safety. We release the model weights, inference implementations, tool environments, and tokenizers under an Apache 2.0 license to enable broad use and further research.
>
---
#### [new 015] Improving Text Style Transfer using Masked Diffusion Language Models with Inference-time Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本风格迁移任务，旨在提升生成质量。通过引入验证器的推理时缩放方法，优化掩码扩散语言模型的生成过程。**

- **链接: [http://arxiv.org/pdf/2508.10995v1](http://arxiv.org/pdf/2508.10995v1)**

> **作者:** Tejomay Kishor Padole; Suyash P Awate; Pushpak Bhattacharyya
>
> **备注:** Accepted as a main conference submission in the European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Masked diffusion language models (MDMs) have recently gained traction as a viable generative framework for natural language. This can be attributed to its scalability and ease of training compared to other diffusion model paradigms for discrete data, establishing itself as the state-of-the-art non-autoregressive generator for discrete data. Diffusion models, in general, have shown excellent ability to improve the generation quality by leveraging inference-time scaling either by increasing the number of denoising steps or by using external verifiers on top of the outputs of each step to guide the generation. In this work, we propose a verifier-based inference-time scaling method that aids in finding a better candidate generation during the denoising process of the MDM. Our experiments demonstrate the application of MDMs for standard text-style transfer tasks and establish MDMs as a better alternative to autoregressive language models. Additionally, we show that a simple soft-value-based verifier setup for MDMs using off-the-shelf pre-trained embedding models leads to significant gains in generation quality even when used on top of typical classifier-free guidance setups in the existing literature.
>
---
#### [new 016] MobQA: A Benchmark Dataset for Semantic Understanding of Human Mobility Data through Question Answering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大模型对人类移动数据的语义理解能力。通过构建MobQA数据集，解决模型在解释移动模式原因上的不足。**

- **链接: [http://arxiv.org/pdf/2508.11163v1](http://arxiv.org/pdf/2508.11163v1)**

> **作者:** Hikaru Asano; Hiroki Ouchi; Akira Kasuga; Ryo Yonetani
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** This paper presents MobQA, a benchmark dataset designed to evaluate the semantic understanding capabilities of large language models (LLMs) for human mobility data through natural language question answering. While existing models excel at predicting human movement patterns, it remains unobvious how much they can interpret the underlying reasons or semantic meaning of those patterns. MobQA provides a comprehensive evaluation framework for LLMs to answer questions about diverse human GPS trajectories spanning daily to weekly granularities. It comprises 5,800 high-quality question-answer pairs across three complementary question types: factual retrieval (precise data extraction), multiple-choice reasoning (semantic inference), and free-form explanation (interpretive description), which all require spatial, temporal, and semantic reasoning. Our evaluation of major LLMs reveals strong performance on factual retrieval but significant limitations in semantic reasoning and explanation question answering, with trajectory length substantially impacting model effectiveness. These findings demonstrate the achievements and limitations of state-of-the-art LLMs for semantic mobility understanding.\footnote{MobQA dataset is available at https://github.com/CyberAgentAILab/mobqa.}
>
---
#### [new 017] Language models align with brain regions that represent concepts across modalities
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与神经科学交叉任务，旨在解决语言模型与大脑概念表征对齐问题。通过分析脑区激活和跨模态一致性，发现语言模型能捕捉跨模态概念意义。**

- **链接: [http://arxiv.org/pdf/2508.11536v1](http://arxiv.org/pdf/2508.11536v1)**

> **作者:** Maria Ryskina; Greta Tuckute; Alexander Fung; Ashley Malkin; Evelina Fedorenko
>
> **备注:** Accepted to COLM 2025. Code and data can be found at https://github.com/ryskina/concepts-brain-llms
>
> **摘要:** Cognitive science and neuroscience have long faced the challenge of disentangling representations of language from representations of conceptual meaning. As the same problem arises in today's language models (LMs), we investigate the relationship between LM--brain alignment and two neural metrics: (1) the level of brain activation during processing of sentences, targeting linguistic processing, and (2) a novel measure of meaning consistency across input modalities, which quantifies how consistently a brain region responds to the same concept across paradigms (sentence, word cloud, image) using an fMRI dataset (Pereira et al., 2018). Our experiments show that both language-only and language-vision models predict the signal better in more meaning-consistent areas of the brain, even when these areas are not strongly sensitive to language processing, suggesting that LMs might internally represent cross-modal conceptual meaning.
>
---
#### [new 018] Personalized Distractor Generation via MCTS-Guided Reasoning Reconstruction
- **分类: cs.CL**

- **简介: 该论文属于个性化干扰项生成任务，旨在解决传统方法无法捕捉学生个体错误的问题。通过MCTS重构学生推理过程，生成符合其认知误区的干扰项。**

- **链接: [http://arxiv.org/pdf/2508.11184v1](http://arxiv.org/pdf/2508.11184v1)**

> **作者:** Tao Wu; Jingyuan Chen; Wang Lin; Jian Zhan; Mengze Li; Kun Kuang; Fei Wu
>
> **摘要:** Distractors, incorrect but plausible answer choices in multiple-choice questions (MCQs), play a critical role in educational assessment by diagnosing student misconceptions. Recent work has leveraged large language models (LLMs) to generate shared, group-level distractors by learning common error patterns across large student populations. However, such distractors often fail to capture the diverse reasoning errors of individual students, limiting their diagnostic effectiveness. To address this limitation, we introduce the task of personalized distractor generation, which aims to generate tailored distractors based on individual misconceptions inferred from each student's past question-answering (QA) records, ensuring every student receives options that effectively exposes their specific reasoning errors. While promising, this task is challenging because each student typically has only a few QA records, which often lack the student's underlying reasoning processes, making training-based group-level approaches infeasible. To overcome this, we propose a training-free two-stage framework. In the first stage, we construct a student-specific misconception prototype by applying Monte Carlo Tree Search (MCTS) to recover the student's reasoning trajectories from past incorrect answers. In the second stage, this prototype guides the simulation of the student's reasoning on new questions, enabling the generation of personalized distractors that align with the student's recurring misconceptions. Experiments show that our approach achieves the best performance in generating plausible, personalized distractors for 140 students, and also effectively generalizes to group-level settings, highlighting its robustness and adaptability.
>
---
#### [new 019] UNVEILING: What Makes Linguistics Olympiad Puzzles Tricky for LLMs?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs在语言奥赛谜题中的表现问题。通过分析629个低资源语言谜题，发现模型在复杂形态学任务中表现不佳，提出分词预处理可提升解决能力。**

- **链接: [http://arxiv.org/pdf/2508.11260v1](http://arxiv.org/pdf/2508.11260v1)**

> **作者:** Mukund Choudhary; KV Aditya Srivatsa; Gaurja Aeron; Antara Raaghavi Bhattacharya; Dang Khoa Dang Dinh; Ikhlasul Akmal Hanif; Daria Kotova; Ekaterina Kochmar; Monojit Choudhury
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Large language models (LLMs) have demonstrated potential in reasoning tasks, but their performance on linguistics puzzles remains consistently poor. These puzzles, often derived from Linguistics Olympiad (LO) contests, provide a minimal contamination environment to assess LLMs' linguistic reasoning abilities across low-resource languages. This work analyses LLMs' performance on 629 problems across 41 low-resource languages by labelling each with linguistically informed features to unveil weaknesses. Our analyses show that LLMs struggle with puzzles involving higher morphological complexity and perform better on puzzles involving linguistic features that are also found in English. We also show that splitting words into morphemes as a pre-processing step improves solvability, indicating a need for more informed and language-specific tokenisers. These findings thus offer insights into some challenges in linguistic reasoning and modelling of low-resource languages.
>
---
#### [new 020] LETToT: Label-Free Evaluation of Large Language Models On Tourism Using Expert Tree-of-Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决旅游领域大模型评估问题。提出LETToT框架，利用专家思维链替代标注数据进行模型评估。**

- **链接: [http://arxiv.org/pdf/2508.11280v1](http://arxiv.org/pdf/2508.11280v1)**

> **作者:** Ruiyan Qi; Congding Wen; Weibo Zhou; Shangsong Liang; Lingbo Li
>
> **摘要:** Evaluating large language models (LLMs) in specific domain like tourism remains challenging due to the prohibitive cost of annotated benchmarks and persistent issues like hallucinations. We propose $\textbf{L}$able-Free $\textbf{E}$valuation of LLM on $\textbf{T}$ourism using Expert $\textbf{T}$ree-$\textbf{o}$f-$\textbf{T}$hought (LETToT), a framework that leverages expert-derived reasoning structures-instead of labeled data-to access LLMs in tourism. First, we iteratively refine and validate hierarchical ToT components through alignment with generic quality dimensions and expert feedback. Results demonstrate the effectiveness of our systematically optimized expert ToT with 4.99-14.15\% relative quality gains over baselines. Second, we apply LETToT's optimized expert ToT to evaluate models of varying scales (32B-671B parameters), revealing: (1) Scaling laws persist in specialized domains (DeepSeek-V3 leads), yet reasoning-enhanced smaller models (e.g., DeepSeek-R1-Distill-Llama-70B) close this gap; (2) For sub-72B models, explicit reasoning architectures outperform counterparts in accuracy and conciseness ($p<0.05$). Our work established a scalable, label-free paradigm for domain-specific LLM evaluation, offering a robust alternative to conventional annotated benchmarks.
>
---
#### [new 021] Model Interpretability and Rationale Extraction by Input Mask Optimization
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在生成神经网络预测的可解释理由。通过输入掩码优化，提取简洁有效的解释，无需训练专用模型。**

- **链接: [http://arxiv.org/pdf/2508.11388v1](http://arxiv.org/pdf/2508.11388v1)**

> **作者:** Marc Brinner; Sina Zarriess
>
> **摘要:** Concurrent to the rapid progress in the development of neural-network based models in areas like natural language processing and computer vision, the need for creating explanations for the predictions of these black-box models has risen steadily. We propose a new method to generate extractive explanations for predictions made by neural networks, that is based on masking parts of the input which the model does not consider to be indicative of the respective class. The masking is done using gradient-based optimization combined with a new regularization scheme that enforces sufficiency, comprehensiveness and compactness of the generated explanation, three properties that are known to be desirable from the related field of rationale extraction in natural language processing. In this way, we bridge the gap between model interpretability and rationale extraction, thereby proving that the latter of which can be performed without training a specialized model, only on the basis of a trained classifier. We further apply the same method to image inputs and obtain high quality explanations for image classifications, which indicates that the conditions proposed for rationale extraction in natural language processing are more broadly applicable to different input types.
>
---
#### [new 022] PersonaTwin: A Multi-Tier Prompt Conditioning Framework for Generating and Evaluating Personalized Digital Twins
- **分类: cs.CL**

- **简介: 该论文属于用户建模任务，旨在解决LLM难以捕捉用户多维特征的问题。提出PersonaTwin框架，整合多源数据生成个性化数字孪生，提升模拟真实性和公平性。**

- **链接: [http://arxiv.org/pdf/2508.10906v1](http://arxiv.org/pdf/2508.10906v1)**

> **作者:** Sihan Chen; John P. Lalor; Yi Yang; Ahmed Abbasi
>
> **备注:** Presented at the Generation, Evaluation & Metrics (GEM) Workshop at ACL 2025
>
> **摘要:** While large language models (LLMs) afford new possibilities for user modeling and approximation of human behaviors, they often fail to capture the multidimensional nuances of individual users. In this work, we introduce PersonaTwin, a multi-tier prompt conditioning framework that builds adaptive digital twins by integrating demographic, behavioral, and psychometric data. Using a comprehensive data set in the healthcare context of more than 8,500 individuals, we systematically benchmark PersonaTwin against standard LLM outputs, and our rigorous evaluation unites state-of-the-art text similarity metrics with dedicated demographic parity assessments, ensuring that generated responses remain accurate and unbiased. Experimental results show that our framework produces simulation fidelity on par with oracle settings. Moreover, downstream models trained on persona-twins approximate models trained on individuals in terms of prediction and fairness metrics across both GPT-4o-based and Llama-based models. Together, these findings underscore the potential for LLM digital twin-based approaches in producing realistic and emotionally nuanced user simulations, offering a powerful tool for personalized digital user modeling and behavior analysis.
>
---
#### [new 023] Modeling and Detecting Company Risks from News: A Case Study in Bloomberg News
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **简介: 该论文属于文本分类任务，旨在从新闻中识别公司风险因素。研究构建了框架并测试了多种模型，发现微调模型效果更佳，为公司运营分析提供洞察。**

- **链接: [http://arxiv.org/pdf/2508.10927v1](http://arxiv.org/pdf/2508.10927v1)**

> **作者:** Jiaxin Pei; Soumya Vadlamannati; Liang-Kang Huang; Daniel Preotiuc-Pietro; Xinyu Hua
>
> **摘要:** Identifying risks associated with a company is important to investors and the well-being of the overall financial market. In this study, we build a computational framework to automatically extract company risk factors from news articles. Our newly proposed schema comprises seven distinct aspects, such as supply chain, regulations, and competitions. We sample and annotate 744 news articles and benchmark various machine learning models. While large language models have achieved huge progress in various types of NLP tasks, our experiment shows that zero-shot and few-shot prompting state-of-the-art LLMs (e.g. LLaMA-2) can only achieve moderate to low performances in identifying risk factors. And fine-tuned pre-trained language models are performing better on most of the risk factors. Using this model, we analyze over 277K Bloomberg news articles and demonstrate that identifying risk factors from news could provide extensive insight into the operations of companies and industries.
>
---
#### [new 024] BIPOLAR: Polarization-based granular framework for LLM bias evaluation
- **分类: cs.CL**

- **简介: 该论文属于语言模型偏见评估任务，旨在解决LLM在敏感话题上的极化偏见问题。提出一种基于极化的情感分析框架，通过合成数据集评估多个模型的偏见情况。**

- **链接: [http://arxiv.org/pdf/2508.11061v1](http://arxiv.org/pdf/2508.11061v1)**

> **作者:** Martin Pavlíček; Tomáš Filip; Petr Sosík
>
> **摘要:** Large language models (LLMs) are known to exhibit biases in downstream tasks, especially when dealing with sensitive topics such as political discourse, gender identity, ethnic relations, or national stereotypes. Although significant progress has been made in bias detection and mitigation techniques, certain challenges remain underexplored. This study proposes a reusable, granular, and topic-agnostic framework to evaluate polarisation-related biases in LLM (both open-source and closed-source). Our approach combines polarisation-sensitive sentiment metrics with a synthetically generated balanced dataset of conflict-related statements, using a predefined set of semantic categories. As a case study, we created a synthetic dataset that focusses on the Russia-Ukraine war, and we evaluated the bias in several LLMs: Llama-3, Mistral, GPT-4, Claude 3.5, and Gemini 1.0. Beyond aggregate bias scores, with a general trend for more positive sentiment toward Ukraine, the framework allowed fine-grained analysis with considerable variation between semantic categories, uncovering divergent behavioural patterns among models. Adaptation to prompt modifications showed further bias towards preconceived language and citizenship modification. Overall, the framework supports automated dataset generation and fine-grained bias assessment, is applicable to a variety of polarisation-driven scenarios and topics, and is orthogonal to many other bias-evaluation strategies.
>
---
#### [new 025] Retrieval-augmented reasoning with lean language models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决资源受限环境下高效、隐私保护的问答问题。通过结合检索与推理，使用轻量模型实现高精度回答。**

- **链接: [http://arxiv.org/pdf/2508.11386v1](http://arxiv.org/pdf/2508.11386v1)**

> **作者:** Ryan Sze-Yin Chan; Federico Nanni; Tomas Lazauskas; Rosie Wood; Penelope Yong; Lionel Tarassenko; Mark Girolami; James Geddes; Andrew Duncan
>
> **摘要:** This technical report details a novel approach to combining reasoning and retrieval augmented generation (RAG) within a single, lean language model architecture. While existing RAG systems typically rely on large-scale models and external APIs, our work addresses the increasing demand for performant and privacy-preserving solutions deployable in resource-constrained or secure environments. Building on recent developments in test-time scaling and small-scale reasoning models, we develop a retrieval augmented conversational agent capable of interpreting complex, domain-specific queries using a lightweight backbone model. Our system integrates a dense retriever with fine-tuned Qwen2.5-Instruct models, using synthetic query generation and reasoning traces derived from frontier models (e.g., DeepSeek-R1) over a curated corpus, in this case, the NHS A-to-Z condition pages. We explore the impact of summarisation-based document compression, synthetic data design, and reasoning-aware fine-tuning on model performance. Evaluation against both non-reasoning and general-purpose lean models demonstrates that our domain-specific fine-tuning approach yields substantial gains in answer accuracy and consistency, approaching frontier-level performance while remaining feasible for local deployment. All implementation details and code are publicly released to support reproducibility and adaptation across domains.
>
---
#### [new 026] Rationalizing Transformer Predictions via End-To-End Differentiable Self-Training
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于分类任务，旨在提升Transformer模型的可解释性。通过端到端微调，使模型同时分类和评分输入词的重要性，实现更稳定、高效的训练。**

- **链接: [http://arxiv.org/pdf/2508.11393v1](http://arxiv.org/pdf/2508.11393v1)**

> **作者:** Marc Brinner; Sina Zarrieß
>
> **摘要:** We propose an end-to-end differentiable training paradigm for stable training of a rationalized transformer classifier. Our approach results in a single model that simultaneously classifies a sample and scores input tokens based on their relevance to the classification. To this end, we build on the widely-used three-player-game for training rationalized models, which typically relies on training a rationale selector, a classifier and a complement classifier. We simplify this approach by making a single model fulfill all three roles, leading to a more efficient training paradigm that is not susceptible to the common training instabilities that plague existing approaches. Further, we extend this paradigm to produce class-wise rationales while incorporating recent advances in parameterizing and regularizing the resulting rationales, thus leading to substantially improved and state-of-the-art alignment with human annotations without any explicit supervision.
>
---
#### [new 027] Towards Reliable Multi-Agent Systems for Marketing Applications via Reflection, Memory, and Planning
- **分类: cs.CL**

- **简介: 该论文针对营销中的受众筛选任务，提出RAMP框架，通过规划、验证和记忆提升LLM系统的可靠性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.11120v1](http://arxiv.org/pdf/2508.11120v1)**

> **作者:** Lorenzo Jaime Yu Flores; Junyi Shen; Xiaoyuan Gu
>
> **摘要:** Recent advances in large language models (LLMs) enabled the development of AI agents that can plan and interact with tools to complete complex tasks. However, literature on their reliability in real-world applications remains limited. In this paper, we introduce a multi-agent framework for a marketing task: audience curation. To solve this, we introduce a framework called RAMP that iteratively plans, calls tools, verifies the output, and generates suggestions to improve the quality of the audience generated. Additionally, we equip the model with a long-term memory store, which is a knowledge base of client-specific facts and past queries. Overall, we demonstrate the use of LLM planning and memory, which increases accuracy by 28 percentage points on a set of 88 evaluation queries. Moreover, we show the impact of iterative verification and reflection on more ambiguous queries, showing progressively better recall (roughly +20 percentage points) with more verify/reflect iterations on a smaller challenge set, and higher user satisfaction. Our results provide practical insights for deploying reliable LLM-based systems in dynamic, industry-facing environments.
>
---
#### [new 028] Dataset Creation for Visual Entailment using Generative AI
- **分类: cs.CL**

- **简介: 该论文属于视觉蕴含任务，解决数据稀缺问题。通过生成式AI创建合成数据集，替代真实数据训练模型，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.11605v1](http://arxiv.org/pdf/2508.11605v1)**

> **作者:** Rob Reijtenbach; Suzan Verberne; Gijs Wijnholds
>
> **备注:** NALOMA: Natural Logic meets Machine Learning workshop @ ESSLLI 2025
>
> **摘要:** In this paper we present and validate a new synthetic dataset for training visual entailment models. Existing datasets for visual entailment are small and sparse compared to datasets for textual entailment. Manually creating datasets is labor-intensive. We base our synthetic dataset on the SNLI dataset for textual entailment. We take the premise text from SNLI as input prompts in a generative image model, Stable Diffusion, creating an image to replace each textual premise. We evaluate our dataset both intrinsically and extrinsically. For extrinsic evaluation, we evaluate the validity of the generated images by using them as training data for a visual entailment classifier based on CLIP feature vectors. We find that synthetic training data only leads to a slight drop in quality on SNLI-VE, with an F-score 0.686 compared to 0.703 when trained on real data. We also compare the quality of our generated training data to original training data on another dataset: SICK-VTE. Again, there is only a slight drop in F-score: from 0.400 to 0.384. These results indicate that in settings with data sparsity, synthetic data can be a promising solution for training visual entailment models.
>
---
#### [new 029] SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于自动问卷生成任务，旨在解决现有评估方法的不足，提出SGSimEval基准，融合多维度评估与人类偏好，提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2508.11310v1](http://arxiv.org/pdf/2508.11310v1)**

> **作者:** Beichen Guo; Zhiyuan Wen; Yu Yang; Peng Gao; Ruosong Yang; Jiaxing Shen
>
> **备注:** Accepted to The 21st International Conference on Advanced Data Mining and Applications (ADMA2025)
>
> **摘要:** The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments.
>
---
#### [new 030] E-CaTCH: Event-Centric Cross-Modal Attention with Temporal Consistency and Class-Imbalance Handling for Misinformation Detection
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文属于多模态虚假信息检测任务，解决模态不一致、时间模式变化和类别不平衡问题。提出E-CaTCH框架，通过跨模态注意力和时间一致性建模提升检测效果。**

- **链接: [http://arxiv.org/pdf/2508.11197v1](http://arxiv.org/pdf/2508.11197v1)**

> **作者:** Ahmad Mousavi; Yeganeh Abdollahinejad; Roberto Corizzo; Nathalie Japkowicz; Zois Boukouvalas
>
> **摘要:** Detecting multimodal misinformation on social media remains challenging due to inconsistencies between modalities, changes in temporal patterns, and substantial class imbalance. Many existing methods treat posts independently and fail to capture the event-level structure that connects them across time and modality. We propose E-CaTCH, an interpretable and scalable framework for robustly detecting misinformation. If needed, E-CaTCH clusters posts into pseudo-events based on textual similarity and temporal proximity, then processes each event independently. Within each event, textual and visual features are extracted using pre-trained BERT and ResNet encoders, refined via intra-modal self-attention, and aligned through bidirectional cross-modal attention. A soft gating mechanism fuses these representations to form contextualized, content-aware embeddings of each post. To model temporal evolution, E-CaTCH segments events into overlapping time windows and uses a trend-aware LSTM, enhanced with semantic shift and momentum signals, to encode narrative progression over time. Classification is performed at the event level, enabling better alignment with real-world misinformation dynamics. To address class imbalance and promote stable learning, the model integrates adaptive class weighting, temporal consistency regularization, and hard-example mining. The total loss is aggregated across all events. Extensive experiments on Fakeddit, IND, and COVID-19 MISINFOGRAPH demonstrate that E-CaTCH consistently outperforms state-of-the-art baselines. Cross-dataset evaluations further demonstrate its robustness, generalizability, and practical applicability across diverse misinformation scenarios.
>
---
#### [new 031] SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在识别LLM生成文本。通过频域分析token概率序列，提出SpecDetect方法，利用DFT总能量作为特征，实现高效准确检测。**

- **链接: [http://arxiv.org/pdf/2508.11343v1](http://arxiv.org/pdf/2508.11343v1)**

> **作者:** Haitong Luo; Weiyao Zhang; Suhang Wang; Wenji Zou; Chungang Lin; Xuying Meng; Yujun Zhang
>
> **备注:** Under Review
>
> **摘要:** The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments demonstrate that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge.
>
---
#### [new 032] HumorPlanSearch: Structured Planning and HuCoT for Contextual AI Humor
- **分类: cs.CL**

- **简介: 该论文属于AI幽默生成任务，旨在解决 humor 生成缺乏上下文和文化敏感性的问题。提出 HumorPlanSearch 模块化框架，结合策略搜索、HuCoT 模板等技术提升幽默的多样性与适应性。**

- **链接: [http://arxiv.org/pdf/2508.11429v1](http://arxiv.org/pdf/2508.11429v1)**

> **作者:** Shivam Dubey
>
> **摘要:** Automated humor generation with Large Language Models (LLMs) often yields jokes that feel generic, repetitive, or tone-deaf because humor is deeply situated and hinges on the listener's cultural background, mindset, and immediate context. We introduce HumorPlanSearch, a modular pipeline that explicitly models context through: (1) Plan-Search for diverse, topic-tailored strategies; (2) Humor Chain-of-Thought (HuCoT) templates capturing cultural and stylistic reasoning; (3) a Knowledge Graph to retrieve and adapt high-performing historical strategies; (4) novelty filtering via semantic embeddings; and (5) an iterative judge-driven revision loop. To evaluate context sensitivity and comedic quality, we propose the Humor Generation Score (HGS), which fuses direct ratings, multi-persona feedback, pairwise win-rates, and topic relevance. In experiments across nine topics with feedback from 13 human judges, our full pipeline (KG + Revision) boosts mean HGS by 15.4 percent (p < 0.05) over a strong baseline. By foregrounding context at every stage from strategy planning to multi-signal evaluation, HumorPlanSearch advances AI-driven humor toward more coherent, adaptive, and culturally attuned comedy.
>
---
#### [new 033] Novel Parasitic Dual-Scale Modeling for Efficient and Accurate Multilingual Speech Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音翻译任务，旨在解决统一模型参数大、效率低的问题。通过提出 Parasitic Dual-Scale 方法，提升推理速度与性能。**

- **链接: [http://arxiv.org/pdf/2508.11189v1](http://arxiv.org/pdf/2508.11189v1)**

> **作者:** Chenyang Le; Yinfeng Xia; Huiyan Li; Manhong Wang; Yutao Sun; Xingyang Ma; Yanmin Qian
>
> **备注:** Interspeech 2025
>
> **摘要:** Recent advancements in speech-to-text translation have led to the development of multilingual models capable of handling multiple language pairs simultaneously. However, these unified models often suffer from large parameter sizes, making it challenging to balance inference efficiency and performance, particularly in local deployment scenarios. We propose an innovative Parasitic Dual-Scale Approach, which combines an enhanced speculative sampling method with model compression and knowledge distillation techniques. Building on the Whisper Medium model, we enhance it for multilingual speech translation into whisperM2M, and integrate our novel KVSPN module, achieving state-of-the-art (SOTA) performance across six popular languages with improved inference efficiency. KVSPN enables a 40\% speedup with no BLEU score degradation. Combined with distillation methods, it represents a 2.6$\times$ speedup over the original Whisper Medium with superior performance.
>
---
#### [new 034] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决儿童使用大语言模型时的安全与伦理问题。研究构建了SproutBench基准，评估模型在不同年龄段的风险表现。**

- **链接: [http://arxiv.org/pdf/2508.11009v1](http://arxiv.org/pdf/2508.11009v1)**

> **作者:** Wenpeng Xing; Lanyi Wei; Haixiao Hu; Rongchang Li; Mohan Li; Changting Lin; Meng Han
>
> **摘要:** The rapid proliferation of large language models (LLMs) in applications targeting children and adolescents necessitates a fundamental reassessment of prevailing AI safety frameworks, which are largely tailored to adult users and neglect the distinct developmental vulnerabilities of minors. This paper highlights key deficiencies in existing LLM safety benchmarks, including their inadequate coverage of age-specific cognitive, emotional, and social risks spanning early childhood (ages 0--6), middle childhood (7--12), and adolescence (13--18). To bridge these gaps, we introduce SproutBench, an innovative evaluation suite comprising 1,283 developmentally grounded adversarial prompts designed to probe risks such as emotional dependency, privacy violations, and imitation of hazardous behaviors. Through rigorous empirical evaluation of 47 diverse LLMs, we uncover substantial safety vulnerabilities, corroborated by robust inter-dimensional correlations (e.g., between Safety and Risk Prevention) and a notable inverse relationship between Interactivity and Age Appropriateness. These insights yield practical guidelines for advancing child-centric AI design and deployment.
>
---
#### [new 035] TinyTim: A Family of Language Models for Divergent Generation
- **分类: cs.CL**

- **简介: 该论文属于语言模型生成任务，旨在探索专精于创造性生成的模型。通过微调《尤利西斯》文本，研究显示TinyTim在词汇多样性上表现突出，但语义连贯性较低，为创意系统提供新思路。**

- **链接: [http://arxiv.org/pdf/2508.11607v1](http://arxiv.org/pdf/2508.11607v1)**

> **作者:** Christopher J. Agostino
>
> **备注:** 7 pages, 3 figures, submitted to NeurIPS Creative AI track, code and model available at https://hf.co/npc-worldwide/TinyTimV1
>
> **摘要:** This work introduces TinyTim, a family of large language models fine-tuned on James Joyce's `Finnegans Wake'. Through quantitative evaluation against baseline models, we demonstrate that TinyTim V1 produces a statistically distinct generative profile characterized by high lexical diversity and low semantic coherence. These findings are interpreted through theories of creativity and complex problem-solving, arguing that such specialized models can function as divergent knowledge sources within more extensive creative architectures, powering automated discovery mechanisms in diverse settings.
>
---
#### [new 036] Online Anti-sexist Speech: Identifying Resistance to Gender Bias in Political Discourse
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.11434v1](http://arxiv.org/pdf/2508.11434v1)**

> **作者:** Aditi Dutta; Susan Banducci
>
> **摘要:** Anti-sexist speech, i.e., public expressions that challenge or resist gendered abuse and sexism, plays a vital role in shaping democratic debate online. Yet automated content moderation systems, increasingly powered by large language models (LLMs), may struggle to distinguish such resistance from the sexism it opposes. This study examines how five LLMs classify sexist, anti-sexist, and neutral political tweets from the UK, focusing on high-salience trigger events involving female Members of Parliament in the year 2022. Our analysis show that models frequently misclassify anti-sexist speech as harmful, particularly during politically charged events where rhetorical styles of harm and resistance converge. These errors risk silencing those who challenge sexism, with disproportionate consequences for marginalised voices. We argue that moderation design must move beyond binary harmful/not-harmful schemas, integrate human-in-the-loop review during sensitive events, and explicitly include counter-speech in training data. By linking feminist scholarship, event-based analysis, and model evaluation, this work highlights the sociotechnical challenges of safeguarding resistance speech in digital political spaces.
>
---
#### [new 037] Feedback Indicators: The Alignment between Llama and a Teacher in Language Learning
- **分类: cs.CL**

- **简介: 该论文属于语言学习反馈任务，旨在通过Llama 3.1提取学生作业中的反馈指标，并与教师评分对齐，以支持自动生成高质量反馈。**

- **链接: [http://arxiv.org/pdf/2508.11364v1](http://arxiv.org/pdf/2508.11364v1)**

> **作者:** Sylvio Rüdian; Yassin Elsir; Marvin Kretschmer; Sabine Cayrou; Niels Pinkwart
>
> **备注:** 11 pages, one table
>
> **摘要:** Automated feedback generation has the potential to enhance students' learning progress by providing timely and targeted feedback. Moreover, it can assist teachers in optimizing their time, allowing them to focus on more strategic and personalized aspects of teaching. To generate high-quality, information-rich formative feedback, it is essential first to extract relevant indicators, as these serve as the foundation upon which the feedback is constructed. Teachers often employ feedback criteria grids composed of various indicators that they evaluate systematically. This study examines the initial phase of extracting such indicators from students' submissions of a language learning course using the large language model Llama 3.1. Accordingly, the alignment between indicators generated by the LLM and human ratings across various feedback criteria is investigated. The findings demonstrate statistically significant strong correlations, even in cases involving unanticipated combinations of indicators and criteria. The methodology employed in this paper offers a promising foundation for extracting indicators from students' submissions using LLMs. Such indicators can potentially be utilized to auto-generate explainable and transparent formative feedback in future research.
>
---
#### [new 038] Aware First, Think Less: Dynamic Boundary Self-Awareness Drives Extreme Reasoning Efficiency in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.11582v1](http://arxiv.org/pdf/2508.11582v1)**

> **作者:** Qiguang Chen; Dengyun Peng; Jinhao Liu; HuiKang Su; Jiannan Guan; Libo Qin; Wanxiang Che
>
> **备注:** Preprint
>
> **摘要:** Recent advancements in large language models (LLMs) have greatly improved their capabilities on complex reasoning tasks through Long Chain-of-Thought (CoT). However, this approach often results in substantial redundancy, impairing computational efficiency and causing significant delays in real-time applications. To improve the efficiency, current methods often rely on human-defined difficulty priors, which do not align with the LLM's self-awared difficulty, leading to inefficiencies. In this paper, we introduce the Dynamic Reasoning-Boundary Self-Awareness Framework (DR. SAF), which enables models to dynamically assess and adjust their reasoning depth in response to problem complexity. DR. SAF integrates three key components: Boundary Self-Awareness Alignment, Adaptive Reward Management, and a Boundary Preservation Mechanism. These components allow models to optimize their reasoning processes, balancing efficiency and accuracy without compromising performance. Our experimental results demonstrate that DR. SAF achieves a 49.27% reduction in total response tokens with minimal loss in accuracy. The framework also delivers a 6.59x gain in token efficiency and a 5x reduction in training time, making it well-suited to resource-limited settings. During extreme training, DR. SAF can even surpass traditional instruction-based models in token efficiency with more than 16% accuracy improvement.
>
---
#### [new 039] Approaching the Source of Symbol Grounding with Confluent Reductions of Abstract Meaning Representation Directed Graphs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决符号接地问题。通过将数字词典嵌入AMR图并进行保环约简，分析其性质以接近符号接地。**

- **链接: [http://arxiv.org/pdf/2508.11068v1](http://arxiv.org/pdf/2508.11068v1)**

> **作者:** Nicolas Goulet; Alexandre Blondin Massé; Moussa Abdendi
>
> **摘要:** Abstract meaning representation (AMR) is a semantic formalism used to represent the meaning of sentences as directed acyclic graphs. In this paper, we describe how real digital dictionaries can be embedded into AMR directed graphs (digraphs), using state-of-the-art pre-trained large language models. Then, we reduce those graphs in a confluent manner, i.e. with transformations that preserve their circuit space. Finally, the properties of these reduces digraphs are analyzed and discussed in relation to the symbol grounding problem.
>
---
#### [new 040] Speciesism in AI: Evaluating Discrimination Against Animals in Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI伦理研究任务，探讨大语言模型中的物种歧视问题，分析其对非人类动物的态度与道德判断，旨在揭示并减少AI系统中的物种偏见。**

- **链接: [http://arxiv.org/pdf/2508.11534v1](http://arxiv.org/pdf/2508.11534v1)**

> **作者:** Monika Jotautaitė; Lucius Caviola; David A. Brewster; Thilo Hagendorff
>
> **摘要:** As large language models (LLMs) become more widely deployed, it is crucial to examine their ethical tendencies. Building on research on fairness and discrimination in AI, we investigate whether LLMs exhibit speciesist bias -- discrimination based on species membership -- and how they value non-human animals. We systematically examine this issue across three paradigms: (1) SpeciesismBench, a 1,003-item benchmark assessing recognition and moral evaluation of speciesist statements; (2) established psychological measures comparing model responses with those of human participants; (3) text-generation tasks probing elaboration on, or resistance to, speciesist rationalizations. In our benchmark, LLMs reliably detected speciesist statements but rarely condemned them, often treating speciesist attitudes as morally acceptable. On psychological measures, results were mixed: LLMs expressed slightly lower explicit speciesism than people, yet in direct trade-offs they more often chose to save one human over multiple animals. A tentative interpretation is that LLMs may weight cognitive capacity rather than species per se: when capacities were equal, they showed no species preference, and when an animal was described as more capable, they tended to prioritize it over a less capable human. In open-ended text generation tasks, LLMs frequently normalized or rationalized harm toward farmed animals while refusing to do so for non-farmed animals. These findings suggest that while LLMs reflect a mixture of progressive and mainstream human views, they nonetheless reproduce entrenched cultural norms around animal exploitation. We argue that expanding AI fairness and alignment frameworks to explicitly include non-human moral patients is essential for reducing these biases and preventing the entrenchment of speciesist attitudes in AI systems and the societies they influence.
>
---
#### [new 041] A2HCoder: An LLM-Driven Coding Agent for Hierarchical Algorithm-to-HDL Translation
- **分类: cs.CL; cs.AR; cs.PL**

- **简介: 该论文属于算法到硬件描述语言的翻译任务，旨在解决算法设计与硬件实现间的差距问题。提出A2HCoder框架，利用大模型进行分层翻译，提升准确性和可靠性。**

- **链接: [http://arxiv.org/pdf/2508.10904v1](http://arxiv.org/pdf/2508.10904v1)**

> **作者:** Jie Lei; Ruofan Jia; J. Andrew Zhang; Hao Zhang
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** In wireless communication systems, stringent requirements such as ultra-low latency and power consumption have significantly increased the demand for efficient algorithm-to-hardware deployment. However, a persistent and substantial gap remains between algorithm design and hardware implementation. Bridging this gap traditionally requires extensive domain expertise and time-consuming manual development, due to fundamental mismatches between high-level programming languages like MATLAB and hardware description languages (HDLs) such as Verilog-in terms of memory access patterns, data processing manners, and datatype representations. To address this challenge, we propose A2HCoder: a Hierarchical Algorithm-to-HDL Coding Agent, powered by large language models (LLMs), designed to enable agile and reliable algorithm-to-hardware translation. A2HCoder introduces a hierarchical framework that enhances both robustness and interpretability while suppressing common hallucination issues in LLM-generated code. In the horizontal dimension, A2HCoder decomposes complex algorithms into modular functional blocks, simplifying code generation and improving consistency. In the vertical dimension, instead of relying on end-to-end generation, A2HCoder performs step-by-step, fine-grained translation, leveraging external toolchains such as MATLAB and Vitis HLS for debugging and circuit-level synthesis. This structured process significantly mitigates hallucinations and ensures hardware-level correctness. We validate A2HCoder through a real-world deployment case in the 5G wireless communication domain, demonstrating its practicality, reliability, and deployment efficiency.
>
---
#### [new 042] Reference Points in LLM Sentiment Analysis: The Role of Structured Context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.11454v1](http://arxiv.org/pdf/2508.11454v1)**

> **作者:** Junichiro Niimi
>
> **摘要:** Large language models (LLMs) are now widely used across many fields, including marketing research. Sentiment analysis, in particular, helps firms understand consumer preferences. While most NLP studies classify sentiment from review text alone, marketing theories, such as prospect theory and expectation--disconfirmation theory, point out that customer evaluations are shaped not only by the actual experience but also by additional reference points. This study therefore investigates how the content and format of such supplementary information affect sentiment analysis using LLMs. We compare natural language (NL) and JSON-formatted prompts using a lightweight 3B parameter model suitable for practical marketing applications. Experiments on two Yelp categories (Restaurant and Nightlife) show that the JSON prompt with additional information outperforms all baselines without fine-tuning: Macro-F1 rises by 1.6% and 4% while RMSE falls by 16% and 9.1%, respectively, making it deployable in resource-constrained edge devices. Furthermore, a follow-up analysis confirms that performance gains stem from genuine contextual reasoning rather than label proxying. This work demonstrates that structured prompting can enable smaller models to achieve competitive performance, offering a practical alternative to large-scale model deployment.
>
---
#### [new 043] ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **简介: 该论文属于法语毒性检测任务，旨在解决法语数据不足和模型效果不佳的问题。工作包括构建TOXIFRENCH数据集，发现小模型更优，并提出CoT微调方法提升性能。**

- **链接: [http://arxiv.org/pdf/2508.11281v1](http://arxiv.org/pdf/2508.11281v1)**

> **作者:** Axel Delaval; Shujian Yang; Haicheng Wang; Han Qiu; Jialiang Lu
>
> **备注:** 14 pages, 5 figures, 8 tables. This paper introduces TOXIFRENCH, a new large-scale benchmark for French toxicity detection, and proposes a Chain-of-Thought (CoT) fine-tuning method with a dynamic weighted loss. The resulting fine-tuned 4B parameter model, ToxiFrench, achieves state-of-the-art performance, outperforming larger models like GPT-4o
>
> **摘要:** Detecting toxic content using language models is crucial yet challenging. While substantial progress has been made in English, toxicity detection in French remains underdeveloped, primarily due to the lack of culturally relevant, large-scale datasets. In this work, we introduce TOXIFRENCH, a new public benchmark of 53,622 French online comments, constructed via a semi-automated annotation pipeline that reduces manual labeling to only 10% through high-confidence LLM-based pre-annotation and human verification. Then, we benchmark a broad range of models and uncover a counterintuitive insight: Small Language Models (SLMs) outperform many larger models in robustness and generalization under the toxicity detection task. Motivated by this finding, we propose a novel Chain-of-Thought (CoT) fine-tuning strategy using a dynamic weighted loss that progressively emphasizes the model's final decision, significantly improving faithfulness. Our fine-tuned 4B model achieves state-of-the-art performance, improving its F1 score by 13% over its baseline and outperforming LLMs such as GPT-40 and Gemini-2.5. Further evaluation on a cross-lingual toxicity benchmark demonstrates strong multilingual ability, suggesting that our methodology can be effectively extended to other languages and safety-critical classification tasks.
>
---
#### [new 044] Survey-to-Behavior: Downstream Alignment of Human Values in LLMs via Survey Questions
- **分类: cs.CL**

- **简介: 该论文属于价值对齐任务，旨在通过问卷微调使大模型行为符合人类价值观。工作包括构建价值档案、微调模型并评估其在域内和域外任务中的行为变化。**

- **链接: [http://arxiv.org/pdf/2508.11414v1](http://arxiv.org/pdf/2508.11414v1)**

> **作者:** Shangrui Nie; Florian Mai; David Kaczér; Charles Welch; Zhixue Zhao; Lucie Flek
>
> **备注:** 7 pages 1 figure
>
> **摘要:** Large language models implicitly encode preferences over human values, yet steering them often requires large training data. In this work, we investigate a simple approach: Can we reliably modify a model's value system in downstream behavior by training it to answer value survey questions accordingly? We first construct value profiles of several open-source LLMs by asking them to rate a series of value-related descriptions spanning 20 distinct human values, which we use as a baseline for subsequent experiments. We then investigate whether the value system of a model can be governed by fine-tuning on the value surveys. We evaluate the effect of finetuning on the model's behavior in two ways; first, we assess how answers change on in-domain, held-out survey questions. Second, we evaluate whether the model's behavior changes in out-of-domain settings (situational scenarios). To this end, we construct a contextualized moral judgment dataset based on Reddit posts and evaluate changes in the model's behavior in text-based adventure games. We demonstrate that our simple approach can not only change the model's answers to in-domain survey questions, but also produces substantial shifts (value alignment) in implicit downstream task behavior.
>
---
#### [new 045] Empowering Multimodal LLMs with External Tools: A Comprehensive Survey
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.10955v1](http://arxiv.org/pdf/2508.10955v1)**

> **作者:** Wenbin An; Jiahao Nie; Yaqiang Wu; Feng Tian; Shijian Lu; Qinghua Zheng
>
> **备注:** 21 pages, 361 references
>
> **摘要:** By integrating the perception capabilities of multimodal encoders with the generative power of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), exemplified by GPT-4V, have achieved great success in various multimodal tasks, pointing toward a promising pathway to artificial general intelligence. Despite this progress, the limited quality of multimodal data, poor performance on many complex downstream tasks, and inadequate evaluation protocols continue to hinder the reliability and broader applicability of MLLMs across diverse domains. Inspired by the human ability to leverage external tools for enhanced reasoning and problem-solving, augmenting MLLMs with external tools (e.g., APIs, expert models, and knowledge bases) offers a promising strategy to overcome these challenges. In this paper, we present a comprehensive survey on leveraging external tools to enhance MLLM performance. Our discussion is structured along four key dimensions about external tools: (1) how they can facilitate the acquisition and annotation of high-quality multimodal data; (2) how they can assist in improving MLLM performance on challenging downstream tasks; (3) how they enable comprehensive and accurate evaluation of MLLMs; (4) the current limitations and future directions of tool-augmented MLLMs. Through this survey, we aim to underscore the transformative potential of external tools in advancing MLLM capabilities, offering a forward-looking perspective on their development and applications. The project page of this paper is publicly available athttps://github.com/Lackel/Awesome-Tools-for-MLLMs.
>
---
#### [new 046] The Next Phase of Scientific Fact-Checking: Advanced Evidence Retrieval from Complex Structured Academic Papers
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于科学事实核查任务，旨在解决复杂学术文献中的证据检索问题，提出五个关键挑战并进行初步实验以提升系统性能。**

- **链接: [http://arxiv.org/pdf/2506.20844v2](http://arxiv.org/pdf/2506.20844v2)**

> **作者:** Xingyu Deng; Xi Wang; Mark Stevenson
>
> **备注:** Accepted for ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR'25)
>
> **摘要:** Scientific fact-checking aims to determine the veracity of scientific claims by retrieving and analysing evidence from research literature. The problem is inherently more complex than general fact-checking since it must accommodate the evolving nature of scientific knowledge, the structural complexity of academic literature and the challenges posed by long-form, multimodal scientific expression. However, existing approaches focus on simplified versions of the problem based on small-scale datasets consisting of abstracts rather than full papers, thereby avoiding the distinct challenges associated with processing complete documents. This paper examines the limitations of current scientific fact-checking systems and reveals the many potential features and resources that could be exploited to advance their performance. It identifies key research challenges within evidence retrieval, including (1) evidence-driven retrieval that addresses semantic limitations and topic imbalance (2) time-aware evidence retrieval with citation tracking to mitigate outdated information, (3) structured document parsing to leverage long-range context, (4) handling complex scientific expressions, including tables, figures, and domain-specific terminology and (5) assessing the credibility of scientific literature. Preliminary experiments were conducted to substantiate these challenges and identify potential solutions. This perspective paper aims to advance scientific fact-checking with a specialised IR system tailored for real-world applications.
>
---
#### [new 047] Diffusion is a code repair operator and generator
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码修复任务，旨在利用扩散模型进行最后阶段的代码修复。通过添加噪声和生成数据，提升修复效果。**

- **链接: [http://arxiv.org/pdf/2508.11110v1](http://arxiv.org/pdf/2508.11110v1)**

> **作者:** Mukul Singh; Gust Verbruggen; Vu Le; Sumit Gulwani
>
> **备注:** 12 pages
>
> **摘要:** Code diffusion models generate code by iteratively removing noise from the latent representation of a code snippet. During later steps of the diffusion process, when the code snippet has almost converged, differences between discrete representations of these snippets look like last-mile repairs applied to broken or incomplete code. We evaluate the extent to which this resemblance can be exploited to leverage pre-trained code diffusion models for the problem of last-mile repair by considering two applications with significant potential. First, we can leverage the diffusion model for last-mile repair by adding noise to a broken code snippet and resuming the diffusion process. Second, we can leverage the diffusion model to generate arbitrary amount of training data for last-mile repair tasks (that are computationally more efficient) by sampling an intermediate program (input) and the final program (output) from the diffusion process. We perform experiments on 3 domains (Python, Excel and PowerShell) to evaluate applications, as well as analyze properties.
>
---
#### [new 048] Generalize across Homophily and Heterophily: Hybrid Spectral Graph Pre-Training and Prompt Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11328v1](http://arxiv.org/pdf/2508.11328v1)**

> **作者:** Haitong Luo; Suhang Wang; Weiyao Zhang; Ruiqi Meng; Xuying Meng; Yujun Zhang
>
> **备注:** Under Review
>
> **摘要:** Graph ``pre-training and prompt-tuning'' aligns downstream tasks with pre-trained objectives to enable efficient knowledge transfer under limited supervision. However, existing methods rely on homophily-based low-frequency knowledge, failing to handle diverse spectral distributions in real-world graphs with varying homophily. Our theoretical analysis reveals a spectral specificity principle: optimal knowledge transfer requires alignment between pre-trained spectral filters and the intrinsic spectrum of downstream graphs. Under limited supervision, large spectral gaps between pre-training and downstream tasks impede effective adaptation. To bridge this gap, we propose the HS-GPPT model, a novel framework that ensures spectral alignment throughout both pre-training and prompt-tuning. We utilize a hybrid spectral filter backbone and local-global contrastive learning to acquire abundant spectral knowledge. Then we design prompt graphs to align the spectral distribution with pretexts, facilitating spectral knowledge transfer across homophily and heterophily. Extensive experiments validate the effectiveness under both transductive and inductive learning settings. Our code is available at https://anonymous.4open.science/r/HS-GPPT-62D2/.
>
---
#### [new 049] How Causal Abstraction Underpins Computational Explanation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于计算认知研究，探讨系统如何实现特定计算。通过因果抽象理论，分析计算与表征的关系，解决计算实现与解释问题。**

- **链接: [http://arxiv.org/pdf/2508.11214v1](http://arxiv.org/pdf/2508.11214v1)**

> **作者:** Atticus Geiger; Jacqueline Harding; Thomas Icard
>
> **摘要:** Explanations of cognitive behavior often appeal to computations over representations. What does it take for a system to implement a given computation over suitable representational vehicles within that system? We argue that the language of causality -- and specifically the theory of causal abstraction -- provides a fruitful lens on this topic. Drawing on current discussions in deep learning with artificial neural networks, we illustrate how classical themes in the philosophy of computation and cognition resurface in contemporary machine learning. We offer an account of computational implementation grounded in causal abstraction, and examine the role for representation in the resulting picture. We argue that these issues are most profitably explored in connection with generalization and prediction.
>
---
#### [new 050] Match & Choose: Model Selection Framework for Fine-tuning Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文本生成图像任务，解决预训练模型选择问题。提出M&C框架，通过匹配图高效预测最佳微调模型。**

- **链接: [http://arxiv.org/pdf/2508.10993v1](http://arxiv.org/pdf/2508.10993v1)**

> **作者:** Basile Lewandowski; Robert Birke; Lydia Y. Chen
>
> **摘要:** Text-to-image (T2I) models based on diffusion and transformer architectures advance rapidly. They are often pretrained on large corpora, and openly shared on a model platform, such as HuggingFace. Users can then build up AI applications, e.g., generating media contents, by adopting pretrained T2I models and fine-tuning them on the target dataset. While public pretrained T2I models facilitate the democratization of the models, users face a new challenge: which model can be best fine-tuned based on the target data domain? Model selection is well addressed in classification tasks, but little is known in (pretrained) T2I models and their performance indication on the target domain. In this paper, we propose the first model selection framework, M&C, which enables users to efficiently choose a pretrained T2I model from a model platform without exhaustively fine-tuning them all on the target dataset. The core of M&C is a matching graph, which consists of: (i) nodes of available models and profiled datasets, and (ii) edges of model-data and data-data pairs capturing the fine-tuning performance and data similarity, respectively. We then build a model that, based on the inputs of model/data feature, and, critically, the graph embedding feature, extracted from the matching graph, predicts the model achieving the best quality after fine-tuning for the target domain. We evaluate M&C on choosing across ten T2I models for 32 datasets against three baselines. Our results show that M&C successfully predicts the best model for fine-tuning in 61.3% of the cases and a closely performing model for the rest.
>
---
#### [new 051] A Cross-Modal Rumor Detection Scheme via Contrastive Learning by Exploring Text and Image internal Correlations
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于谣言检测任务，解决图像与文本信息关联不足的问题。通过对比学习和多尺度对齐，提升跨模态特征融合效果，增强谣言识别能力。**

- **链接: [http://arxiv.org/pdf/2508.11141v1](http://arxiv.org/pdf/2508.11141v1)**

> **作者:** Bin Ma; Yifei Zhang; Yongjin Xian; Qi Li; Linna Zhou; Gongxun Miao
>
> **摘要:** Existing rumor detection methods often neglect the content within images as well as the inherent relationships between contexts and images across different visual scales, thereby resulting in the loss of critical information pertinent to rumor identification. To address these issues, this paper presents a novel cross-modal rumor detection scheme based on contrastive learning, namely the Multi-scale Image and Context Correlation exploration algorithm (MICC). Specifically, we design an SCLIP encoder to generate unified semantic embeddings for text and multi-scale image patches through contrastive pretraining, enabling their relevance to be measured via dot-product similarity. Building upon this, a Cross-Modal Multi-Scale Alignment module is introduced to identify image regions most relevant to the textual semantics, guided by mutual information maximization and the information bottleneck principle, through a Top-K selection strategy based on a cross-modal relevance matrix constructed between the text and multi-scale image patches. Moreover, a scale-aware fusion network is designed to integrate the highly correlated multi-scale image features with global text features by assigning adaptive weights to image regions based on their semantic importance and cross-modal relevance. The proposed methodology has been extensively evaluated on two real-world datasets. The experimental results demonstrate that it achieves a substantial performance improvement over existing state-of-the-art approaches in rumor detection, highlighting its effectiveness and potential for practical applications.
>
---
#### [new 052] Can Multi-modal (reasoning) LLMs detect document manipulation?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11021v1](http://arxiv.org/pdf/2508.11021v1)**

> **作者:** Zisheng Liang; Kidus Zewde; Rudra Pratap Singh; Disha Patil; Zexi Chen; Jiayu Xue; Yao Yao; Yifei Chen; Qinzhe Liu; Simiao Ren
>
> **备注:** arXiv admin note: text overlap with arXiv:2503.20084
>
> **摘要:** Document fraud poses a significant threat to industries reliant on secure and verifiable documentation, necessitating robust detection mechanisms. This study investigates the efficacy of state-of-the-art multi-modal large language models (LLMs)-including OpenAI O1, OpenAI 4o, Gemini Flash (thinking), Deepseek Janus, Grok, Llama 3.2 and 4, Qwen 2 and 2.5 VL, Mistral Pixtral, and Claude 3.5 and 3.7 Sonnet-in detecting fraudulent documents. We benchmark these models against each other and prior work on document fraud detection techniques using a standard dataset with real transactional documents. Through prompt optimization and detailed analysis of the models' reasoning processes, we evaluate their ability to identify subtle indicators of fraud, such as tampered text, misaligned formatting, and inconsistent transactional sums. Our results reveal that top-performing multi-modal LLMs demonstrate superior zero-shot generalization, outperforming conventional methods on out-of-distribution datasets, while several vision LLMs exhibit inconsistent or subpar performance. Notably, model size and advanced reasoning capabilities show limited correlation with detection accuracy, suggesting task-specific fine-tuning is critical. This study underscores the potential of multi-modal LLMs in enhancing document fraud detection systems and provides a foundation for future research into interpretable and scalable fraud mitigation strategies.
>
---
#### [new 053] Inclusion Arena: An Open Platform for Evaluating Large Foundation Models with Real-World Apps
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.11452v1](http://arxiv.org/pdf/2508.11452v1)**

> **作者:** Kangyu Wang; Hongliang He; Lin Liu; Ruiqi Liang; Zhenzhong Lan; Jianguo Li
>
> **备注:** Our platform is publicly accessible at https://doraemon.alipay.com/model-ranking
>
> **摘要:** Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have ushered in a new era of AI capabilities, demonstrating near-human-level performance across diverse scenarios. While numerous benchmarks (e.g., MMLU) and leaderboards (e.g., Chatbot Arena) have been proposed to help evolve the development of LLMs and MLLMs, most rely on static datasets or crowdsourced general-domain prompts, often falling short of reflecting performance in real-world applications. To bridge this critical gap, we present Inclusion Arena, a live leaderboard that ranks models based on human feedback collected directly from AI-powered applications. Our platform integrates pairwise model comparisons into natural user interactions, ensuring evaluations reflect practical usage scenarios. For robust model ranking, we employ the Bradley-Terry model augmented with two key innovations: (1) Placement Matches, a cold-start mechanism to quickly estimate initial ratings for newly integrated models, and (2) Proximity Sampling, an intelligent comparison strategy that prioritizes battles between models of similar capabilities to maximize information gain and enhance rating stability. Extensive empirical analyses and simulations demonstrate that Inclusion Arena yields reliable and stable rankings, exhibits higher data transitivity compared to general crowdsourced datasets, and significantly mitigates the risk of malicious manipulation. By fostering an open alliance between foundation models and real-world applications, Inclusion Arena aims to accelerate the development of LLMs and MLLMs truly optimized for practical, user-centric deployments. The platform is publicly accessible at https://doraemon.alipay.com/model-ranking.
>
---
#### [new 054] Benchmarking Prosody Encoding in Discrete Speech Tokens
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决离散语音标记在捕捉语气信息上的不足，通过分析其对人工修改语气的敏感性，提供设计指南。**

- **链接: [http://arxiv.org/pdf/2508.11224v1](http://arxiv.org/pdf/2508.11224v1)**

> **作者:** Kentaro Onda; Satoru Fukayama; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** Recently, discrete tokens derived from self-supervised learning (SSL) models via k-means clustering have been actively studied as pseudo-text in speech language models and as efficient intermediate representations for various tasks. However, these discrete tokens are typically learned in advance, separately from the training of language models or downstream tasks. As a result, choices related to discretization, such as the SSL model used or the number of clusters, must be made heuristically. In particular, speech language models are expected to understand and generate responses that reflect not only the semantic content but also prosodic features. Yet, there has been limited research on the ability of discrete tokens to capture prosodic information. To address this gap, this study conducts a comprehensive analysis focusing on prosodic encoding based on their sensitivity to the artificially modified prosody, aiming to provide practical guidelines for designing discrete tokens.
>
---
#### [new 055] +VeriRel: Verification Feedback to Enhance Document Retrieval for Scientific Fact Checking
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11122v1](http://arxiv.org/pdf/2508.11122v1)**

> **作者:** Xingyu Deng; Xi Wang; Mark Stevenson
>
> **备注:** Accpeted for the 34th ACM International Conference on Information and Knowledge Management (CIKM'25)
>
> **摘要:** Identification of appropriate supporting evidence is critical to the success of scientific fact checking. However, existing approaches rely on off-the-shelf Information Retrieval algorithms that rank documents based on relevance rather than the evidence they provide to support or refute the claim being checked. This paper proposes +VeriRel which includes verification success in the document ranking. Experimental results on three scientific fact checking datasets (SciFact, SciFact-Open and Check-Covid) demonstrate consistently leading performance by +VeriRel for document evidence retrieval and a positive impact on downstream verification. This study highlights the potential of integrating verification feedback to document relevance assessment for effective scientific fact checking systems. It shows promising future work to evaluate fine-grained relevance when examining complex documents for advanced scientific fact checking.
>
---
#### [new 056] PaperRegister: Boosting Flexible-grained Paper Search via Hierarchical Register Indexing
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于论文检索任务，解决传统系统无法支持细粒度查询的问题。通过构建分层索引树，提升灵活粒度搜索效果。**

- **链接: [http://arxiv.org/pdf/2508.11116v1](http://arxiv.org/pdf/2508.11116v1)**

> **作者:** Zhuoqun Li; Xuanang Chen; Hongyu Lin; Yaojie Lu; Xianpei Han; Le Sun
>
> **摘要:** Paper search is an important activity for researchers, typically involving using a query with description of a topic to find relevant papers. As research deepens, paper search requirements may become more flexible, sometimes involving specific details such as module configuration rather than being limited to coarse-grained topics. However, previous paper search systems are unable to meet these flexible-grained requirements, as these systems mainly collect paper abstracts to construct index of corpus, which lack detailed information to support retrieval by finer-grained queries. In this work, we propose PaperRegister, consisted of offline hierarchical indexing and online adaptive retrieval, transforming traditional abstract-based index into hierarchical index tree for paper search, thereby supporting queries at flexible granularity. Experiments on paper search tasks across a range of granularity demonstrate that PaperRegister achieves the state-of-the-art performance, and particularly excels in fine-grained scenarios, highlighting the good potential as an effective solution for flexible-grained paper search in real-world applications. Code for this work is in https://github.com/Li-Z-Q/PaperRegister.
>
---
#### [new 057] Group Fairness Meets the Black Box: Enabling Fair Algorithms on Closed LLMs via Post-Processing
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于公平性学习任务，解决在封闭大模型上实现群体公平的问题。通过提示工程提取特征并训练轻量公平分类器，提升模型公平性与效率。**

- **链接: [http://arxiv.org/pdf/2508.11258v1](http://arxiv.org/pdf/2508.11258v1)**

> **作者:** Ruicheng Xian; Yuxuan Wan; Han Zhao
>
> **摘要:** Instruction fine-tuned large language models (LLMs) enable a simple zero-shot or few-shot prompting paradigm, also known as in-context learning, for building prediction models. This convenience, combined with continued advances in LLM capability, has the potential to drive their adoption across a broad range of domains, including high-stakes applications where group fairness -- preventing disparate impacts across demographic groups -- is essential. The majority of existing approaches to enforcing group fairness on LLM-based classifiers rely on traditional fair algorithms applied via model fine-tuning or head-tuning on final-layer embeddings, but they are no longer applicable to closed-weight LLMs under the in-context learning setting, which include some of the most capable commercial models today, such as GPT-4, Gemini, and Claude. In this paper, we propose a framework for deriving fair classifiers from closed-weight LLMs via prompting: the LLM is treated as a feature extractor, and features are elicited from its probabilistic predictions (e.g., token log probabilities) using prompts strategically designed for the specified fairness criterion to obtain sufficient statistics for fair classification; a fair algorithm is then applied to these features to train a lightweight fair classifier in a post-hoc manner. Experiments on five datasets, including three tabular ones, demonstrate strong accuracy-fairness tradeoffs for the classifiers derived by our framework from both open-weight and closed-weight LLMs; in particular, our framework is data-efficient and outperforms fair classifiers trained on LLM embeddings (i.e., head-tuning) or from scratch on raw tabular features.
>
---
#### [new 058] ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal
- **分类: cs.SE; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.11222v1](http://arxiv.org/pdf/2508.11222v1)**

> **作者:** Haonan Zhang; Dongxia Wang; Yi Liu; Kexin Chen; Jiashui Wang; Xinlei Ying; Long Liu; Wenhai Wang
>
> **摘要:** Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems.
>
---
#### [new 059] Beyond Solving Math Quiz: Evaluating the Ability of Large Reasoning Models to Ask for Information
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.11252v1](http://arxiv.org/pdf/2508.11252v1)**

> **作者:** Youcheng Huang; Bowen Qin; Chen Huang; Duanyu Feng; Xi Yang; Wenqiang Lei
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated remarkable problem-solving abilities in mathematics, as evaluated by existing benchmarks exclusively on well-defined problems. However, such evaluation setup constitutes a critical gap, since a genuine intelligent agent should not only solve problems (as a math quiz solver), but also be able~to ask for information when the problems lack sufficient information, enabling proactivity in responding users' requests. To bridge such gap, we proposes a new dataset consisting of two types of incomplete problems with diverse contexts. Based on the dataset, our systematical evaluation of LRMs reveals their inability in proactively asking for information. In addition, we uncover the behaviors related to overthinking and hallucination of LRMs, and highlight the potential and challenges of supervised fine-tuning in learning such ability. We hope to provide new insights in developing LRMs with genuine intelligence, rather than just solving problems.
>
---
#### [new 060] Emphasis Sensitivity in Speech Representations
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11566v1](http://arxiv.org/pdf/2508.11566v1)**

> **作者:** Shaun Cassini; Thomas Hain; Anton Ragni
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** This work investigates whether modern speech models are sensitive to prosodic emphasis - whether they encode emphasized and neutral words in systematically different ways. Prior work typically relies on isolated acoustic correlates (e.g., pitch, duration) or label prediction, both of which miss the relational structure of emphasis. This paper proposes a residual-based framework, defining emphasis as the difference between paired neutral and emphasized word representations. Analysis on self-supervised speech models shows that these residuals correlate strongly with duration changes and perform poorly at word identity prediction, indicating a structured, relational encoding of prosodic emphasis. In ASR fine-tuned models, residuals occupy a subspace up to 50% more compact than in pre-trained models, further suggesting that emphasis is encoded as a consistent, low-dimensional transformation that becomes more structured with task-specific learning.
>
---
#### [new 061] BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决如何生成高质量合成数据以提升大模型预训练效果的问题。提出BeyondWeb框架，显著提升预训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.10975v1](http://arxiv.org/pdf/2508.10975v1)**

> **作者:** Pratyush Maini; Vineeth Dorna; Parth Doshi; Aldo Carranza; Fan Pan; Jack Urbanek; Paul Burstein; Alex Fang; Alvin Deng; Amro Abbas; Brett Larsen; Cody Blakeney; Charvi Bannur; Christina Baek; Darren Teh; David Schwab; Haakon Mongstad; Haoli Yin; Josh Wills; Kaleigh Mentzer; Luke Merrick; Ricardo Monti; Rishabh Adiga; Siddharth Joshi; Spandan Das; Zhengping Wang; Bogdan Gaza; Ari Morcos; Matthew Leavitt
>
> **摘要:** Recent advances in large language model (LLM) pretraining have shown that simply scaling data quantity eventually leads to diminishing returns, hitting a data wall. In response, the use of synthetic data for pretraining has emerged as a promising paradigm for pushing the frontier of performance. Despite this, the factors affecting synthetic data quality remain poorly understood. In this work, we introduce BeyondWeb, a synthetic data generation framework that produces high-quality synthetic data for pretraining. BeyondWeb significantly extends the capabilities of traditional web-scale datasets, outperforming state-of-the-art synthetic pretraining datasets such as Cosmopedia and Nemotron-CC's high-quality synthetic subset (Nemotron-Synth) by up to 5.1 percentage points (pp) and 2.6pp, respectively, when averaged across a suite of 14 benchmark evaluations. It delivers up to 7.7x faster training than open web data and 2.7x faster than Nemotron-Synth. Remarkably, a 3B model trained for 180B tokens on BeyondWeb outperforms an 8B model trained for the same token budget on Cosmopedia. We also present several insights from BeyondWeb on synthetic data for pretraining: what drives its benefits, which data to rephrase and how, and the impact of model size and family on data quality. Overall, our work shows that there's no silver bullet for generating high-quality synthetic pretraining data. The best outcomes require jointly optimizing many factors, a challenging task that requires rigorous science and practical expertise. Naive approaches can yield modest improvements, potentially at great cost, while well-executed methods can yield transformative improvements, as exemplified by BeyondWeb.
>
---
#### [new 062] Controlling Multimodal LLMs via Reward-guided Decoding
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态大模型控制任务，旨在提升模型的视觉定位能力。通过奖励引导解码方法，实现对模型输出精度和召回率的动态控制。**

- **链接: [http://arxiv.org/pdf/2508.11616v1](http://arxiv.org/pdf/2508.11616v1)**

> **作者:** Oscar Mañas; Pierluca D'Oro; Koustuv Sinha; Adriana Romero-Soriano; Michal Drozdzal; Aishwarya Agrawal
>
> **备注:** Published at ICCV 2025
>
> **摘要:** As Multimodal Large Language Models (MLLMs) gain widespread applicability, it is becoming increasingly desirable to adapt them for diverse user needs. In this paper, we study the adaptation of MLLMs through controlled decoding. To achieve this, we introduce the first method for reward-guided decoding of MLLMs and demonstrate its application in improving their visual grounding. Our method involves building reward models for visual grounding and using them to guide the MLLM's decoding process. Concretely, we build two separate reward models to independently control the degree of object precision and recall in the model's output. Our approach enables on-the-fly controllability of an MLLM's inference process in two ways: first, by giving control over the relative importance of each reward function during decoding, allowing a user to dynamically trade off object precision for recall in image captioning tasks; second, by giving control over the breadth of the search during decoding, allowing the user to control the trade-off between the amount of test-time compute and the degree of visual grounding. We evaluate our method on standard object hallucination benchmarks, showing that it provides significant controllability over MLLM inference, while consistently outperforming existing hallucination mitigation methods.
>
---
#### [new 063] Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音检索任务，解决根据自然语言描述的说话风格检索语音的问题。通过训练语音和文本编码器，将两者映射到共同空间，实现基于风格描述的语音检索。**

- **链接: [http://arxiv.org/pdf/2508.11187v1](http://arxiv.org/pdf/2508.11187v1)**

> **作者:** Wonjune Kang; Deb Roy
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** We introduce the task of expressive speech retrieval, where the goal is to retrieve speech utterances spoken in a given style based on a natural language description of that style. While prior work has primarily focused on performing speech retrieval based on what was said in an utterance, we aim to do so based on how something was said. We train speech and text encoders to embed speech and text descriptions of speaking styles into a joint latent space, which enables using free-form text prompts describing emotions or styles as queries to retrieve matching expressive speech segments. We perform detailed analyses of various aspects of our proposed framework, including encoder architectures, training criteria for effective cross-modal alignment, and prompt augmentation for improved generalization to arbitrary text queries. Experiments on multiple datasets encompassing 22 speaking styles demonstrate that our approach achieves strong retrieval performance as measured by Recall@k.
>
---
## 更新

#### [replaced 001] Bridging Context Gaps: Leveraging Coreference Resolution for Long Contextual Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01671v3](http://arxiv.org/pdf/2410.01671v3)**

> **作者:** Yanming Liu; Xinyue Peng; Jiannan Cao; Yanxin Shen; Tianyu Du; Sheng Cheng; Xun Wang; Jianwei Yin; Xuhong Zhang
>
> **备注:** ICLR 2025 camera ready version, with second updated metadata
>
> **摘要:** Large language models (LLMs) have shown remarkable capabilities in natural language processing; however, they still face difficulties when tasked with understanding lengthy contexts and executing effective question answering. These challenges often arise due to the complexity and ambiguity present in longer texts. To enhance the performance of LLMs in such scenarios, we introduce the Long Question Coreference Adaptation (LQCA) method. This innovative framework focuses on coreference resolution tailored to long contexts, allowing the model to identify and manage references effectively. The LQCA method encompasses four key steps: resolving coreferences within sub-documents, computing the distances between mentions, defining a representative mention for coreference, and answering questions through mention replacement. By processing information systematically, the framework provides easier-to-handle partitions for LLMs, promoting better understanding. Experimental evaluations on a range of LLMs and datasets have yielded positive results, with a notable improvements on OpenAI-o1-mini and GPT-4o models, highlighting the effectiveness of leveraging coreference resolution to bridge context gaps in question answering. Our code is public at https://github.com/OceannTwT/LQCA.
>
---
#### [replaced 002] MMESGBench: Pioneering Multimodal Understanding and Complex Reasoning Benchmark for ESG Tasks
- **分类: cs.MM; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18932v2](http://arxiv.org/pdf/2507.18932v2)**

> **作者:** Lei Zhang; Xin Zhou; Chaoyue He; Di Wang; Yi Wu; Hong Xu; Wei Liu; Chunyan Miao
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** Environmental, Social, and Governance (ESG) reports are essential for evaluating sustainability practices, ensuring regulatory compliance, and promoting financial transparency. However, these documents are often lengthy, structurally diverse, and multimodal, comprising dense text, structured tables, complex figures, and layout-dependent semantics. Existing AI systems often struggle to perform reliable document-level reasoning in such settings, and no dedicated benchmark currently exists in ESG domain. To fill the gap, we introduce \textbf{MMESGBench}, a first-of-its-kind benchmark dataset targeted to evaluate multimodal understanding and complex reasoning across structurally diverse and multi-source ESG documents. This dataset is constructed via a human-AI collaborative, multi-stage pipeline. First, a multimodal LLM generates candidate question-answer (QA) pairs by jointly interpreting rich textual, tabular, and visual information from layout-aware document pages. Second, an LLM verifies the semantic accuracy, completeness, and reasoning complexity of each QA pair. This automated process is followed by an expert-in-the-loop validation, where domain specialists validate and calibrate QA pairs to ensure quality, relevance, and diversity. MMESGBench comprises 933 validated QA pairs derived from 45 ESG documents, spanning across seven distinct document types and three major ESG source categories. Questions are categorized as single-page, cross-page, or unanswerable, with each accompanied by fine-grained multimodal evidence. Initial experiments validate that multimodal and retrieval-augmented models substantially outperform text-only baselines, particularly on visually grounded and cross-page tasks. MMESGBench is publicly available as an open-source dataset at https://github.com/Zhanglei1103/MMESGBench.
>
---
#### [replaced 003] Slow Tuning and Low-Entropy Masking for Safe Chain-of-Thought Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09666v2](http://arxiv.org/pdf/2508.09666v2)**

> **作者:** Ziyang Ma; Qingyue Yuan; Linhai Zhang; Deyu Zhou
>
> **备注:** Preprint
>
> **摘要:** Previous chain-of-thought (CoT) distillation methods primarily focused on enhancing the reasoning capabilities of Small Language Models (SLMs) by utilizing high-quality rationales generated by powerful Large Language Models (LLMs, e.g., GPT-4). However, few works have noted the negative effects on SLM safety brought by the training, which are revealed in this study. Although there are works on safety alignment that fine-tune language models or manipulate model weights to defend against harmful inputs, they require extra computation or annotated data, and probably impact the reasoning ability of SLMs. In this paper, we investigate how to maintain the safety of SLMs during the CoT distillation process. Specifically, we propose a safe distillation method, Slow Tuning and Low-Entropy Masking Distillation (SLowED), containing two modules: Slow Tuning and Low-Entropy Masking. Slow Tuning scales down the magnitude of model weight changes to optimize the model weights in the neighboring space near the initial weight distribution. Low-Entropy Masking masks low-entropy tokens, which are regarded as unnecessary learning targets, to exclude them from fine-tuning. Experiments on three SLMs (Qwen2.5-1.5B, Llama-3.2-1B, BLOOM-1.1B) across reasoning benchmarks (BBH, BB-Sub, ARC, AGIEval) and safety evaluation (AdvBench) show that SLowED retains the safety of SLMs and comparably improves their reasoning capability compared to existing distillation methods. Furthermore, our ablation study presents the effectiveness of Slow Tuning and Low-Entropy Masking, with the former maintaining the model's safety in the early stage and the latter prolonging the safe training epochs.
>
---
#### [replaced 004] Uncertainty-Aware Adaptation of Large Language Models for Protein-Protein Interaction Analysis
- **分类: cs.LG; cs.AI; cs.CL; stat.AP; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.06173v2](http://arxiv.org/pdf/2502.06173v2)**

> **作者:** Sanket Jantre; Tianle Wang; Gilchan Park; Kriti Chopra; Nicholas Jeon; Xiaoning Qian; Nathan M. Urban; Byung-Jun Yoon
>
> **摘要:** Identification of protein-protein interactions (PPIs) helps derive cellular mechanistic understanding, particularly in the context of complex conditions such as neurodegenerative disorders, metabolic syndromes, and cancer. Large Language Models (LLMs) have demonstrated remarkable potential in predicting protein structures and interactions via automated mining of vast biomedical literature; yet their inherent uncertainty remains a key challenge for deriving reproducible findings, critical for biomedical applications. In this study, we present an uncertainty-aware adaptation of LLMs for PPI analysis, leveraging fine-tuned LLaMA-3 and BioMedGPT models. To enhance prediction reliability, we integrate LoRA ensembles and Bayesian LoRA models for uncertainty quantification (UQ), ensuring confidence-calibrated insights into protein behavior. Our approach achieves competitive performance in PPI identification across diverse disease contexts while addressing model uncertainty, thereby enhancing trustworthiness and reproducibility in computational biology. These findings underscore the potential of uncertainty-aware LLM adaptation for advancing precision medicine and biomedical research.
>
---
#### [replaced 005] A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges
- **分类: cs.DL; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.06401v2](http://arxiv.org/pdf/2508.06401v2)**

> **作者:** Andrew Brown; Muhammad Roman; Barry Devereux
>
> **备注:** 58 pages, This work has been submitted to the IEEE for possible publication
>
> **摘要:** This systematic review of the research literature on retrieval-augmented generation (RAG) provides a focused analysis of the most highly cited studies published between 2020 and May 2025. A total of 128 articles met our inclusion criteria. The records were retrieved from ACM Digital Library, IEEE Xplore, Scopus, ScienceDirect, and the Digital Bibliography and Library Project (DBLP). RAG couples a neural retriever with a generative language model, grounding output in up-to-date, non-parametric memory while retaining the semantic generalisation stored in model weights. Guided by the PRISMA 2020 framework, we (i) specify explicit inclusion and exclusion criteria based on citation count and research questions, (ii) catalogue datasets, architectures, and evaluation practices, and (iii) synthesise empirical evidence on the effectiveness and limitations of RAG. To mitigate citation-lag bias, we applied a lower citation-count threshold to papers published in 2025 so that emerging breakthroughs with naturally fewer citations were still captured. This review clarifies the current research landscape, highlights methodological gaps, and charts priority directions for future research.
>
---
#### [replaced 006] When Explainability Meets Privacy: An Investigation at the Intersection of Post-hoc Explainability and Differential Privacy in the Context of Natural Language Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10482v2](http://arxiv.org/pdf/2508.10482v2)**

> **作者:** Mahdi Dhaini; Stephen Meisenbacher; Ege Erdogan; Florian Matthes; Gjergji Kasneci
>
> **备注:** Accepted to AAAI/ACM Conference on AI, Ethics, and Society (AIES 2025)
>
> **摘要:** In the study of trustworthy Natural Language Processing (NLP), a number of important research fields have emerged, including that of explainability and privacy. While research interest in both explainable and privacy-preserving NLP has increased considerably in recent years, there remains a lack of investigation at the intersection of the two. This leaves a considerable gap in understanding of whether achieving both explainability and privacy is possible, or whether the two are at odds with each other. In this work, we conduct an empirical investigation into the privacy-explainability trade-off in the context of NLP, guided by the popular overarching methods of Differential Privacy (DP) and Post-hoc Explainability. Our findings include a view into the intricate relationship between privacy and explainability, which is formed by a number of factors, including the nature of the downstream task and choice of the text privatization and explainability method. In this, we highlight the potential for privacy and explainability to co-exist, and we summarize our findings in a collection of practical recommendations for future work at this important intersection.
>
---
#### [replaced 007] Bridging AI Innovation and Healthcare Needs: Lessons Learned from Incorporating Modern NLP at The BC Cancer Registry
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2508.09991v2](http://arxiv.org/pdf/2508.09991v2)**

> **作者:** Lovedeep Gondara; Gregory Arbour; Raymond Ng; Jonathan Simkin; Shebnum Devji
>
> **摘要:** Automating data extraction from clinical documents offers significant potential to improve efficiency in healthcare settings, yet deploying Natural Language Processing (NLP) solutions presents practical challenges. Drawing upon our experience implementing various NLP models for information extraction and classification tasks at the British Columbia Cancer Registry (BCCR), this paper shares key lessons learned throughout the project lifecycle. We emphasize the critical importance of defining problems based on clear business objectives rather than solely technical accuracy, adopting an iterative approach to development, and fostering deep interdisciplinary collaboration and co-design involving domain experts, end-users, and ML specialists from inception. Further insights highlight the need for pragmatic model selection (including hybrid approaches and simpler methods where appropriate), rigorous attention to data quality (representativeness, drift, annotation), robust error mitigation strategies involving human-in-the-loop validation and ongoing audits, and building organizational AI literacy. These practical considerations, generalizable beyond cancer registries, provide guidance for healthcare organizations seeking to successfully implement AI/NLP solutions to enhance data management processes and ultimately improve patient care and public health outcomes.
>
---
#### [replaced 008] AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15887v2](http://arxiv.org/pdf/2507.15887v2)**

> **作者:** Ori Press; Brandon Amos; Haoyu Zhao; Yikai Wu; Samuel K. Ainsworth; Dominik Krupke; Patrick Kidger; Touqir Sajed; Bartolomeo Stellato; Jisun Park; Nathanael Bosch; Eli Meril; Albert Steppi; Arman Zharmagambetov; Fangzhao Zhang; David Perez-Pineiro; Alberto Mercurio; Ni Zhan; Talor Abramovich; Kilian Lieret; Hanlin Zhang; Shirley Huang; Matthias Bethge; Ofir Press
>
> **摘要:** Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 154 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner uses a simple, budgeted loop that edits code, compiles and runs it, profiles performance, verifies correctness on tests, and selects the fastest valid version. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance.
>
---
#### [replaced 009] Personalized LLM for Generating Customized Responses to the Same Query from Different Users
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11736v2](http://arxiv.org/pdf/2412.11736v2)**

> **作者:** Hang Zeng; Chaoyue Niu; Fan Wu; Chengfei Lv; Guihai Chen
>
> **备注:** Accepted by CIKM'25
>
> **摘要:** Existing work on large language model (LLM) personalization assigned different responding roles to LLMs, but overlooked the diversity of queriers. In this work, we propose a new form of querier-aware LLM personalization, generating different responses even for the same query from different queriers. We design a dual-tower model architecture with a cross-querier general encoder and a querier-specific encoder. We further apply contrastive learning with multi-view augmentation, pulling close the dialogue representations of the same querier, while pulling apart those of different queriers. To mitigate the impact of query diversity on querier-contrastive learning, we cluster the dialogues based on query similarity and restrict the scope of contrastive learning within each cluster. To address the lack of datasets designed for querier-aware personalization, we also build a multi-querier dataset from English and Chinese scripts, as well as WeChat records, called MQDialog, containing 173 queriers and 12 responders. Extensive evaluations demonstrate that our design significantly improves the quality of personalized response generation, achieving relative improvement of 8.4% to 48.7% in ROUGE-L scores and winning rates ranging from 54% to 82% compared with various baseline methods.
>
---
#### [replaced 010] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16502v4](http://arxiv.org/pdf/2410.16502v4)**

> **作者:** Jason Chan; Robert Gaizauskas; Zhixue Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
>
---
#### [replaced 011] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12052v2](http://arxiv.org/pdf/2502.12052v2)**

> **作者:** Xinyu Hu; Mingqi Gao; Li Lin; Zhenghan Yu; Xiaojun Wan
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** In NLG meta-evaluation, evaluation metrics are typically assessed based on their consistency with humans. However, we identify some limitations in traditional NLG meta-evaluation approaches, such as issues in handling human ratings and ambiguous selections of correlation measures, which undermine the effectiveness of meta-evaluation. In this work, we propose a dual-perspective NLG meta-evaluation framework that focuses on different evaluation capabilities, thereby providing better interpretability. In addition, we introduce a method of automatically constructing the corresponding benchmarks without requiring new human annotations. Furthermore, we conduct experiments with 16 representative LLMs as the evaluators based on our proposed framework, comprehensively analyzing their evaluation performance from different perspectives.
>
---
#### [replaced 012] Tool-Planner: Task Planning with Clusters across Multiple Tools
- **分类: cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.03807v4](http://arxiv.org/pdf/2406.03807v4)**

> **作者:** Yanming Liu; Xinyue Peng; Jiannan Cao; Yuwei Zhang; Xuhong Zhang; Sheng Cheng; Xun Wang; Jianwei Yin; Tianyu Du
>
> **备注:** ICLR 2025 Camera Ready version
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional reasoning capabilities, enabling them to solve various complex problems. Recently, this ability has been applied to the paradigm of tool learning. Tool learning involves providing examples of tool usage and their corresponding functions, allowing LLMs to formulate plans and demonstrate the process of invoking and executing each tool. LLMs can address tasks that they cannot complete independently, thereby enhancing their potential across different tasks. However, this approach faces two key challenges. First, redundant error correction leads to unstable planning and long execution time. Additionally, designing a correct plan among multiple tools is also a challenge in tool learning. To address these issues, we propose Tool-Planner, a task-processing framework based on toolkits. Tool-Planner groups tools based on the API functions with the same function into a toolkit and allows LLMs to implement planning across the various toolkits. When a tool error occurs, the language model can reselect and adjust tools based on the toolkit. Experiments show that our approach demonstrates a high pass and win rate across different datasets and optimizes the planning scheme for tool learning in models such as GPT-4 and Claude 3, showcasing the potential of our method. Our code is public at https://github.com/OceannTwT/Tool-Planner
>
---
#### [replaced 013] Investigating the Effect of Parallel Data in the Cross-Lingual Transfer for Vision-Language Encoders
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21681v2](http://arxiv.org/pdf/2504.21681v2)**

> **作者:** Andrei-Alexandru Manea; Jindřich Libovický
>
> **摘要:** Most pre-trained Vision-Language (VL) models and training data for the downstream tasks are only available in English. Therefore, multilingual VL tasks are solved using cross-lingual transfer: fine-tune a multilingual pre-trained model or transfer the text encoder using parallel data. We study the alternative approach: transferring an already trained encoder using parallel data. We investigate the effect of parallel data: domain and the number of languages, which were out of focus in previous work. Our results show that even machine-translated task data are the best on average, caption-like authentic parallel data outperformed it in some languages. Further, we show that most languages benefit from multilingual training.
>
---
#### [replaced 014] Causal Language in Observational Studies: Sociocultural Backgrounds and Team Composition
- **分类: physics.soc-ph; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12159v2](http://arxiv.org/pdf/2502.12159v2)**

> **作者:** Jun Wang; Bei Yu
>
> **备注:** 17 pages, 3 figures, 3 tables
>
> **摘要:** The use of causal language in observational studies has raised concerns about overstatement in scientific communication. While some argue that such language should be reserved for randomized controlled trials, others contend that rigorous causal inference methods can justify causal claims in observational research. Ideally, causal language should align with the strength of the underlying evidence. However, through the analysis of over 90,000 abstracts from observational studies using computational linguistic and regression methods, we found that causal language are more common in work by less experienced authors, smaller research teams, male last authors, and researchers from countries with higher uncertainty avoidance indices. Our findings suggest that the use of causal language is not solely driven by the strength of evidence, but also by the sociocultural backgrounds of authors and their team composition. This work provides a new perspective for understanding systematic variations in scientific communication and emphasizes the importance of recognizing these human factors when evaluating scientific claims.
>
---
#### [replaced 015] Inaccuracy of an E-Dictionary and Its Influence on Chinese Language Users
- **分类: cs.CL; cs.HC; H.5.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.00799v3](http://arxiv.org/pdf/2504.00799v3)**

> **作者:** Shiyang Zhang; Fanfei Meng; Xi Wang; Lan Li
>
> **备注:** The scope of the work has evolved significantly since initial submission, and we are preparing a revised version that better reflects the current direction of the research
>
> **摘要:** Electronic dictionaries have largely replaced paper dictionaries and become central tools for L2 learners seeking to expand their vocabulary. Users often assume these resources are reliable and rarely question the validity of the definitions provided. The accuracy of major E-dictionaries is seldom scrutinized, and little attention has been paid to how their corpora are constructed. Research on dictionary use, particularly the limitations of electronic dictionaries, remains scarce. This study adopts a combined method of experimentation, user survey, and dictionary critique to examine Youdao, one of the most widely used E-dictionaries in China. The experiment involved a translation task paired with retrospective reflection. Participants were asked to translate sentences containing words that are insufficiently or inaccurately defined in Youdao. Their consultation behavior was recorded to analyze how faulty definitions influenced comprehension. Results show that incomplete or misleading definitions can cause serious misunderstandings. Additionally, students exhibited problematic consultation habits. The study further explores how such flawed definitions originate, highlighting issues in data processing and the integration of AI and machine learning technologies in dictionary construction. The findings suggest a need for better training in dictionary literacy for users, as well as improvements in the underlying AI models used to build E-dictionaries.
>
---
#### [replaced 016] MVISU-Bench: Benchmarking Mobile Agents for Real-World Tasks by Multi-App, Vague, Interactive, Single-App and Unethical Instructions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09057v2](http://arxiv.org/pdf/2508.09057v2)**

> **作者:** Zeyu Huang; Juyuan Wang; Longfeng Chen; Boyi Xiao; Leng Cai; Yawen Zeng; Jin Xu
>
> **备注:** ACM MM 2025
>
> **摘要:** Given the significant advances in Large Vision Language Models (LVLMs) in reasoning and visual understanding, mobile agents are rapidly emerging to meet users' automation needs. However, existing evaluation benchmarks are disconnected from the real world and fail to adequately address the diverse and complex requirements of users. From our extensive collection of user questionnaire, we identified five tasks: Multi-App, Vague, Interactive, Single-App, and Unethical Instructions. Around these tasks, we present \textbf{MVISU-Bench}, a bilingual benchmark that includes 404 tasks across 137 mobile applications. Furthermore, we propose Aider, a plug-and-play module that acts as a dynamic prompt prompter to mitigate risks and clarify user intent for mobile agents. Our Aider is easy to integrate into several frameworks and has successfully improved overall success rates by 19.55\% compared to the current state-of-the-art (SOTA) on MVISU-Bench. Specifically, it achieves success rate improvements of 53.52\% and 29.41\% for unethical and interactive instructions, respectively. Through extensive experiments and analysis, we highlight the gap between existing mobile agents and real-world user expectations.
>
---
#### [replaced 017] TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.10450v3](http://arxiv.org/pdf/2406.10450v3)**

> **作者:** Haohao Qu; Wenqi Fan; Zihuai Zhao; Qing Li
>
> **备注:** Accepted by IEEE TKDE. Codes and data are available at https://github.com/Quhaoh233/TokenRec
>
> **摘要:** There is a growing interest in utilizing large-scale language models (LLMs) to advance next-generation Recommender Systems (RecSys), driven by their outstanding language understanding and in-context learning capabilities. In this scenario, tokenizing (i.e., indexing) users and items becomes essential for ensuring a seamless alignment of LLMs with recommendations. While several studies have made progress in representing users and items through textual contents or latent representations, challenges remain in efficiently capturing high-order collaborative knowledge into discrete tokens that are compatible with LLMs. Additionally, the majority of existing tokenization approaches often face difficulties in generalizing effectively to new/unseen users or items that were not in the training corpus. To address these challenges, we propose a novel framework called TokenRec, which introduces not only an effective ID tokenization strategy but also an efficient retrieval paradigm for LLM-based recommendations. Specifically, our tokenization strategy, Masked Vector-Quantized (MQ) Tokenizer, involves quantizing the masked user/item representations learned from collaborative filtering into discrete tokens, thus achieving a smooth incorporation of high-order collaborative knowledge and a generalizable tokenization of users and items for LLM-based RecSys. Meanwhile, our generative retrieval paradigm is designed to efficiently recommend top-$K$ items for users to eliminate the need for the time-consuming auto-regressive decoding and beam search processes used by LLMs, thus significantly reducing inference time. Comprehensive experiments validate the effectiveness of the proposed methods, demonstrating that TokenRec outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems.
>
---
#### [replaced 018] Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01307v2](http://arxiv.org/pdf/2503.01307v2)**

> **作者:** Kanishk Gandhi; Ayush Chakravarthy; Anikait Singh; Nathan Lile; Noah D. Goodman
>
> **摘要:** Test-time inference has emerged as a powerful paradigm for enabling language models to ``think'' longer and more carefully about complex challenges, much like skilled human experts. While reinforcement learning (RL) can drive self-improvement in language models on verifiable tasks, some models exhibit substantial gains while others quickly plateau. For instance, we find that Qwen-2.5-3B far exceeds Llama-3.2-3B under identical RL training for the game of Countdown. This discrepancy raises a critical question: what intrinsic properties enable effective self-improvement? We introduce a framework to investigate this question by analyzing four key cognitive behaviors -- verification, backtracking, subgoal setting, and backward chaining -- that both expert human problem solvers and successful language models employ. Our study reveals that Qwen naturally exhibits these reasoning behaviors, whereas Llama initially lacks them. In systematic experimentation with controlled behavioral datasets, we find that priming Llama with examples containing these reasoning behaviors enables substantial improvements during RL, matching or exceeding Qwen's performance. Importantly, the presence of reasoning behaviors, rather than correctness of answers, proves to be the critical factor -- models primed with incorrect solutions containing proper reasoning patterns achieve comparable performance to those trained on correct solutions. Finally, leveraging continued pretraining with OpenWebMath data, filtered to amplify reasoning behaviors, enables the Llama model to match Qwen's self-improvement trajectory. Our findings establish a fundamental relationship between initial reasoning behaviors and the capacity for improvement, explaining why some language models effectively utilize additional computation while others plateau.
>
---
#### [replaced 019] MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.08715v2](http://arxiv.org/pdf/2508.08715v2)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** We are withdrawing the manuscript to revise the title and contents of figures for better alignment with the paper's contributions
>
> **摘要:** Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
>
---
#### [replaced 020] Relationship Detection on Tabular Data Using Statistical Analysis and Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06371v2](http://arxiv.org/pdf/2506.06371v2)**

> **作者:** Panagiotis Koletsis; Christos Panagiotopoulos; Georgios Th. Papadopoulos; Vasilis Efthymiou
>
> **摘要:** Over the past few years, table interpretation tasks have made significant progress due to their importance and the introduction of new technologies and benchmarks in the field. This work experiments with a hybrid approach for detecting relationships among columns of unlabeled tabular data, using a Knowledge Graph (KG) as a reference point, a task known as CPA. This approach leverages large language models (LLMs) while employing statistical analysis to reduce the search space of potential KG relations. The main modules of this approach for reducing the search space are domain and range constraints detection, as well as relation co-appearance analysis. The experimental evaluation on two benchmark datasets provided by the SemTab challenge assesses the influence of each module and the effectiveness of different state-of-the-art LLMs at various levels of quantization. The experiments were performed, as well as at different prompting techniques. The proposed methodology, which is publicly available on github, proved to be competitive with state-of-the-art approaches on these datasets.
>
---
#### [replaced 021] Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2503.17811v2](http://arxiv.org/pdf/2503.17811v2)**

> **作者:** Wenqi Pei; Hailing Xu; Hengyuan Zhao; Shizheng Hou; Han Chen; Zining Zhang; Pingyi Luo; Bingsheng He
>
> **备注:** DL4C @ ICLR 2025
>
> **摘要:** Natural Language to SQL (NL2SQL) has seen significant advancements with large language models (LLMs). However, these models often depend on closed-source systems and high computational resources, posing challenges in data privacy and deployment. In contrast, small language models (SLMs) struggle with NL2SQL tasks, exhibiting poor performance and incompatibility with existing frameworks. To address these issues, we introduce Feather-SQL, a new lightweight framework tailored for SLMs. Feather-SQL improves SQL executability and accuracy through 1) schema pruning and linking, 2) multi-path and multi-candidate generation. Additionally, we introduce the 1+1 Model Collaboration Paradigm, which pairs a strong general-purpose chat model with a fine-tuned SQL specialist, combining strong analytical reasoning with high-precision SQL generation. Experimental results on BIRD demonstrate that Feather-SQL improves NL2SQL performance on SLMs, with around 10% boost for models without fine-tuning. The proposed paradigm raises the accuracy ceiling of SLMs to 54.76%, highlighting its effectiveness.
>
---
#### [replaced 022] Exploring Superior Function Calls via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05118v3](http://arxiv.org/pdf/2508.05118v3)**

> **作者:** Bingguang Hao; Maolin Wang; Zengzhuang Xu; Yicheng Chen; Cunyin Peng; Jinjie GU; Chenyi Zhuang
>
> **摘要:** Function calling capabilities are crucial for deploying Large Language Models in real-world applications, yet current training approaches fail to develop robust reasoning strategies. Supervised fine-tuning produces models that rely on superficial pattern matching, while standard reinforcement learning methods struggle with the complex action space of structured function calls. We present a novel reinforcement learning framework designed to enhance group relative policy optimization through strategic entropy based exploration specifically tailored for function calling tasks. Our approach addresses three critical challenges in function calling: insufficient exploration during policy learning, lack of structured reasoning in chain-of-thought generation, and inadequate verification of parameter extraction. Our two-stage data preparation pipeline ensures high-quality training samples through iterative LLM evaluation and abstract syntax tree validation. Extensive experiments on the Berkeley Function Calling Leaderboard demonstrate that this framework achieves state-of-the-art performance among open-source models with 86.02\% overall accuracy, outperforming standard GRPO by up to 6\% on complex multi-function scenarios. Notably, our method shows particularly strong improvements on code-pretrained models, suggesting that structured language generation capabilities provide an advantageous starting point for reinforcement learning in function calling tasks. We will release all the code, models and dataset to benefit the community.
>
---
#### [replaced 023] Training-Free Multimodal Large Language Model Orchestration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10016v2](http://arxiv.org/pdf/2508.10016v2)**

> **作者:** Tianyu Xie; Yuhang Wu; Yongdong Luo; Jiayi Ji; Xiawu Zheng
>
> **摘要:** Different Multimodal Large Language Models (MLLMs) cannot be integrated into a unified multimodal input-output system directly. In previous work, training has been considered as an inevitable component due to challenges in modal alignment, Text-to-Speech efficiency and other integration issues. In this paper, we introduce Multimodal Large Language Model Orchestration, an effective approach for creating interactive multimodal AI systems without additional training. MLLM Orchestration leverages the inherent reasoning capabilities of large language models to coordinate specialized models through explicit workflows, enabling natural multimodal interactions while maintaining modularity, improving interpretability, and significantly enhancing computational efficiency. Our orchestration framework is built upon three key innovations: (1) a central controller LLM that analyzes user inputs and dynamically routes tasks to appropriate specialized models through carefully designed agents; (2) a parallel Text-to-Speech architecture that enables true full-duplex interaction with seamless interruption handling and natural conversational flow; and (3) a cross-modal memory integration system that maintains coherent context across modalities through intelligent information synthesis and retrieval, selectively avoiding unnecessary modality calls in certain scenarios to improve response speed. Extensive evaluations demonstrate that MLLM Orchestration achieves comprehensive multimodal capabilities without additional training, performance improvements of up to 7.8% over traditional jointly-trained approaches on standard benchmarks, reduced latency by 10.3%, and significantly enhanced interpretability through explicit orchestration processes.
>
---
#### [replaced 024] Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10054v2](http://arxiv.org/pdf/2506.10054v2)**

> **作者:** Shangpin Peng; Weinong Wang; Zhuotao Tian; Senqiao Yang; Xing Wu; Haotian Xu; Chengquan Zhang; Takashi Isobe; Baotian Hu; Min Zhang
>
> **摘要:** Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at https://github.com/pspdada/Omni-DPO.
>
---
#### [replaced 025] E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency
- **分类: cs.DB; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09023v2](http://arxiv.org/pdf/2508.09023v2)**

> **作者:** Dongjie Xu; Yue Cui; Weijie Shi; Qingzhi Ma; Hanghui Guo; Jiaming Li; Yao Zhao; Ruiyuan Zhang; Shimin Di; Jia Zhu; Kai Zheng; Jiajie Xu
>
> **摘要:** SQL query rewriting aims to reformulate a query into a more efficient form while preserving equivalence. Most existing methods rely on predefined rewrite rules. However, such rule-based approaches face fundamental limitations: (1) fixed rule sets generalize poorly to novel query patterns and struggle with complex queries; (2) a wide range of effective rewriting strategies cannot be fully captured by declarative rules. To overcome these issues, we propose using large language models (LLMs) to generate rewrites. LLMs can capture complex strategies, such as evaluation reordering and CTE rewriting. Despite this potential, directly applying LLMs often results in performance regressions or non-equivalent rewrites due to a lack of execution awareness and semantic grounding. To address these challenges, We present E3-Rewrite, an LLM-based SQL rewriting framework that produces executable, equivalent, and efficient queries. It integrates two core components: a context construction module and a reinforcement learning framework. First, the context module leverages execution plans and retrieved demonstrations to build bottleneck-aware prompts that guide inference-time rewriting. Second, we design a reward function targeting executability, equivalence, and efficiency, evaluated via syntax checks, equivalence verification, and cost estimation. Third, to ensure stable multi-objective learning, we adopt a staged curriculum that first emphasizes executability and equivalence, then gradually incorporates efficiency. Across multiple SQL benchmarks, our experiments demonstrate that E3-Rewrite can shorten query execution time by as much as 25.6% relative to leading baselines, while also producing up to 24.4% more rewrites that meet strict equivalence criteria. These gains extend to challenging query patterns that prior approaches could not effectively optimize.
>
---
#### [replaced 026] TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05422v2](http://arxiv.org/pdf/2505.05422v2)**

> **作者:** Haokun Lin; Teng Wang; Yixiao Ge; Yuying Ge; Zhichao Lu; Ying Wei; Qingfu Zhang; Zhenan Sun; Ying Shan
>
> **备注:** Technical Report
>
> **摘要:** Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at https://github.com/TencentARC/TokLIP.
>
---
#### [replaced 027] Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models
- **分类: cs.LG; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.07468v2](http://arxiv.org/pdf/2506.07468v2)**

> **作者:** Mickel Liu; Liwei Jiang; Yancheng Liang; Simon Shaolei Du; Yejin Choi; Tim Althoff; Natasha Jaques
>
> **摘要:** Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).
>
---
#### [replaced 028] A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.18013v2](http://arxiv.org/pdf/2402.18013v2)**

> **作者:** Zihao Yi; Jiarui Ouyang; Zhe Xu; Yuwen Liu; Tianhao Liao; Haohao Luo; Ying Shen
>
> **备注:** 35 pages, 10 figures, ACM Computing Surveys
>
> **摘要:** This survey provides a comprehensive review of research on multi-turn dialogue systems, with a particular focus on multi-turn dialogue systems based on large language models (LLMs). This paper aims to (a) give a summary of existing LLMs and approaches for adapting LLMs to downstream tasks; (b) elaborate recent advances in multi-turn dialogue systems, covering both LLM-based open-domain dialogue (ODD) and task-oriented dialogue (TOD) systems, along with datasets and evaluation metrics; (c) discuss some future emphasis and recent research problems arising from the development of LLMs and the increasing demands on multi-turn dialogue systems.
>
---
#### [replaced 029] From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v3](http://arxiv.org/pdf/2507.04996v3)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are viewed as vehicular systems capable of perceiving their environment and executing pre-programmed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 0 to 5); Examples of this outpace include the interaction with humans with natural language, goal adaptation, contextual reasoning, external tool use, and unseen ethical dilemma handling, largely empowered by multi-modal large language models (LLMs). These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this gap, this paper introduces the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI systems to reason, adapt, and interact within complex environments. This paper proposes the term AgVs and their distinguishing characteristics from conventional AuVs. It synthesizes relevant advances in integrating LLMs and AuVs and highlights how AgVs might transform future mobility systems and ensure the systems are human-centered. The paper concludes by identifying key challenges in the development and governance of AgVs, and how they can play a significant role in future agentic transportation systems.
>
---
#### [replaced 030] PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00344v2](http://arxiv.org/pdf/2508.00344v2)**

> **作者:** Keer Lu; Chong Chen; Bin Cui; Huang Leng; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale.
>
---
#### [replaced 031] EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing
- **分类: cs.DB; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22402v2](http://arxiv.org/pdf/2503.22402v2)**

> **作者:** Yizhang Zhu; Runzhi Jiang; Boyan Li; Nan Tang; Yuyu Luo
>
> **备注:** COLM 2025
>
> **摘要:** Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs--often overlooked--stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2x boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL. Our source code and model are available at https://elliesql.github.io/.
>
---
#### [replaced 032] Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16636v2](http://arxiv.org/pdf/2502.16636v2)**

> **作者:** Yin Wu; Quanyu Long; Jing Li; Jianfei Yu; Wenya Wang
>
> **备注:** 21 pages, 6 figures, 17 tables
>
> **摘要:** Retrieval-augmented generation (RAG) is a paradigm that augments large language models (LLMs) with external knowledge to tackle knowledge-intensive question answering. While several benchmarks evaluate Multimodal LLMs (MLLMs) under Multimodal RAG settings, they predominantly retrieve from textual corpora and do not explicitly assess how models exploit visual evidence during generation. Consequently, there still lacks benchmark that isolates and measures the contribution of retrieved images in RAG. We introduce Visual-RAG, a question-answering benchmark that targets visually grounded, knowledge-intensive questions. Unlike prior work, Visual-RAG requires text-to-image retrieval and the integration of retrieved clue images to extract visual evidence for answer generation. With Visual-RAG, we evaluate 5 open-source and 3 proprietary MLLMs, showcasing that images provide strong evidence in augmented generation. However, even state-of-the-art models struggle to efficiently extract and utilize visual knowledge. Our results highlight the need for improved visual retrieval, grounding, and attribution in multimodal RAG systems.
>
---

# 自然语言处理 cs.CL

- **最新发布 151 篇**

- **更新 109 篇**

## 最新发布

#### [new 001] Error Reflection Prompting: Can Large Language Models Successfully Understand Errors?
- **分类: cs.CL**

- **简介: 论文提出Error Reflection Prompting（ERP），用于提升大语言模型的错误识别与纠正能力。针对Chain-of-thought（CoT）缺乏反思机制的问题，ERP通过错误识别、错误分析和正确解答三步增强推理过程，提高模型的准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.16729v1](http://arxiv.org/pdf/2508.16729v1)**

> **作者:** Jason Li; Lauren Yraola; Kevin Zhu; Sean O'Brien
>
> **备注:** Accepted to Insights @ NAACL 2025
>
> **摘要:** Prompting methods for language models, such as Chain-of-thought (CoT), present intuitive step-by-step processes for problem solving. These methodologies aim to equip models with a better understanding of the correct procedures for addressing a given task. Despite these advancements, CoT lacks the ability of reflection and error correction, potentially causing a model to perpetuate mistakes and errors. Therefore, inspired by the human ability for said tasks, we propose Error Reflection Prompting (ERP) to further enhance reasoning in language models. Building upon CoT, ERP is a method comprised of an incorrect answer, error recognition, and a correct answer. This process enables the model to recognize types of errors and the steps that lead to incorrect answers, allowing the model to better discern which steps to avoid and which to take. The model is able to generate the error outlines itself with automated ERP generation, allowing for error recognition and correction to be integrated into the reasoning chain and produce scalability and reliability in the process. The results demonstrate that ERP serves as a versatile supplement to conventional CoT, ultimately contributing to more robust and capable reasoning abilities along with increased interpretability in how models ultimately reach their errors.
>
---
#### [new 002] Detecting and Characterizing Planning in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文研究语言模型中的规划行为，旨在区分规划与即兴生成。通过提出因果判定标准并构建半自动化标注流程，分析Gemma-2-2B模型在代码和诗歌生成任务中的行为，发现规划非普遍存在，且指令微调优化而非创造规划行为。**

- **链接: [http://arxiv.org/pdf/2508.18098v1](http://arxiv.org/pdf/2508.18098v1)**

> **作者:** Jatin Nainani; Sankaran Vaidyanathan; Connor Watts; Andre N. Assis; Alice Rigg
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Modern large language models (LLMs) have demonstrated impressive performance across a wide range of multi-step reasoning tasks. Recent work suggests that LLMs may perform planning - selecting a future target token in advance and generating intermediate tokens that lead towards it - rather than merely improvising one token at a time. However, existing studies assume fixed planning horizons and often focus on single prompts or narrow domains. To distinguish planning from improvisation across models and tasks, we present formal and causally grounded criteria for detecting planning and operationalize them as a semi-automated annotation pipeline. We apply this pipeline to both base and instruction-tuned Gemma-2-2B models on the MBPP code generation benchmark and a poem generation task where Claude 3.5 Haiku was previously shown to plan. Our findings show that planning is not universal: unlike Haiku, Gemma-2-2B solves the same poem generation task through improvisation, and on MBPP it switches between planning and improvisation across similar tasks and even successive token predictions. We further show that instruction tuning refines existing planning behaviors in the base model rather than creating them from scratch. Together, these studies provide a reproducible and scalable foundation for mechanistic studies of planning in LLMs.
>
---
#### [new 003] Towards Alignment-Centric Paradigm: A Survey of Instruction Tuning in Large Language Models
- **分类: cs.CL; I.2.7; I.2.6**

- **简介: 该论文属于大语言模型指令微调的综述任务，旨在解决模型对齐人类意图、安全与领域需求的问题。工作包括梳理数据构建、微调方法与评估协议，提出未来自动化与鲁棒性优化方向。**

- **链接: [http://arxiv.org/pdf/2508.17184v1](http://arxiv.org/pdf/2508.17184v1)**

> **作者:** Xudong Han; Junjie Yang; Tianyang Wang; Ziqian Bi; Junfeng Hao; Junhao Song
>
> **备注:** 24 pages, 7 figures, 5 tables
>
> **摘要:** Instruction tuning is a pivotal technique for aligning large language models (LLMs) with human intentions, safety constraints, and domain-specific requirements. This survey provides a comprehensive overview of the full pipeline, encompassing (i) data collection methodologies, (ii) full-parameter and parameter-efficient fine-tuning strategies, and (iii) evaluation protocols. We categorized data construction into three major paradigms: expert annotation, distillation from larger models, and self-improvement mechanisms, each offering distinct trade-offs between quality, scalability, and resource cost. Fine-tuning techniques range from conventional supervised training to lightweight approaches, such as low-rank adaptation (LoRA) and prefix tuning, with a focus on computational efficiency and model reusability. We further examine the challenges of evaluating faithfulness, utility, and safety across multilingual and multimodal scenarios, highlighting the emergence of domain-specific benchmarks in healthcare, legal, and financial applications. Finally, we discuss promising directions for automated data generation, adaptive optimization, and robust evaluation frameworks, arguing that a closer integration of data, algorithms, and human feedback is essential for advancing instruction-tuned LLMs. This survey aims to serve as a practical reference for researchers and practitioners seeking to design LLMs that are both effective and reliably aligned with human intentions.
>
---
#### [new 004] Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design
- **分类: cs.CL**

- **简介: 论文提出多层级设计框架，将LLM的人格化视为可调控的设计概念，解决当前研究偏重风险、缺乏指导的问题。通过四维线索（感知、语言、行为、认知）构建统一分类，提供可操作的设计杠杆，推动以功能为导向的评估方法。**

- **链接: [http://arxiv.org/pdf/2508.17573v1](http://arxiv.org/pdf/2508.17573v1)**

> **作者:** Yunze Xiao; Lynnette Hui Xian Ng; Jiarui Liu; Mona T. Diab
>
> **备注:** Accepted in EMNLP main proceedings
>
> **摘要:** Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
>
---
#### [new 005] DropLoRA: Sparse Low-Rank Adaptation for Parameter-Efficient Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 论文提出DropLoRA，一种用于大模型参数高效微调的新方法，通过在LoRA中引入剪枝模块动态调整低秩子空间，解决传统LoRA因静态子空间导致的性能瓶颈问题，在多项语言任务上显著提升效果且不增加成本。**

- **链接: [http://arxiv.org/pdf/2508.17337v1](http://arxiv.org/pdf/2508.17337v1)**

> **作者:** Haojie Zhang
>
> **备注:** 8 pages
>
> **摘要:** LoRA-based large model parameter-efficient fine-tuning (PEFT) methods use low-rank de- composition to approximate updates to model parameters. However, compared to full- parameter fine-tuning, low-rank updates often lead to a performance gap in downstream tasks. To address this, we introduce DropLoRA, a novel pruning-based approach that focuses on pruning the rank dimension. Unlike conven- tional methods that attempt to overcome the low-rank bottleneck, DropLoRA innovatively integrates a pruning module between the two low-rank matrices in LoRA to simulate dy- namic subspace learning. This dynamic low- rank subspace learning allows DropLoRA to overcome the limitations of traditional LoRA, which operates within a static subspace. By continuously adapting the learning subspace, DropLoRA significantly boosts performance without incurring additional training or infer- ence costs. Our experimental results demon- strate that DropLoRA consistently outperforms LoRA in fine-tuning the LLaMA series across a wide range of large language model gener- ation tasks, including commonsense reason- ing, mathematical reasoning, code generation, and instruction-following. Our code is avail- able at https://github.com/TayeeChang/DropLoRA.
>
---
#### [new 006] Information availability in different languages and various technological constraints related to multilinguism on the Internet
- **分类: cs.CL**

- **简介: 论文分析互联网信息多语言可用性及技术限制问题，旨在解决非英语用户获取信息的障碍。通过研究语言分布与技术约束，提出提升多语言信息可访问性的必要性。**

- **链接: [http://arxiv.org/pdf/2508.17918v1](http://arxiv.org/pdf/2508.17918v1)**

> **作者:** Sonal Khosla; Haridasa Acharya
>
> **备注:** International Journal of Computer Applications
>
> **摘要:** The usage of Internet has grown exponentially over the last two decades. The number of Internet users has grown from 16 Million to 1650 Million from 1995 to 2010. It has become a major repository of information catering almost every area. Since the Internet has its origin in USA which is English speaking country there is huge dominance of English on the World Wide Web. Although English is a globally acceptable language, still there is a huge population in the world which is not able to access the Internet due to language constraints. It has been estimated that only 20-25% of the world population speaks English as a native language. More and more people are accessing the Internet nowadays removing the cultural and linguistic barriers and hence there is a high growth in the number of non-English speaking users over the last few years on the Internet. Although many solutions have been provided to remove the linguistic barriers, still there is a huge gap to be filled. This paper attempts to analyze the need of information availability in different languages and the various technological constraints related to multi-linguism on the Internet.
>
---
#### [new 007] Toward Socially Aware Vision-Language Models: Evaluating Cultural Competence Through Multimodal Story Generation
- **分类: cs.CL; cs.CY**

- **简介: 论文聚焦多模态故事生成任务，解决VLMs在跨文化场景下的适应能力问题。通过构建新型评估框架，系统测试5个模型的文化敏感性，发现其词汇层面有较好表现，但架构差异导致效果不稳定，且自动指标与人类判断不一致。**

- **链接: [http://arxiv.org/pdf/2508.16762v1](http://arxiv.org/pdf/2508.16762v1)**

> **作者:** Arka Mukherjee; Shreya Ghosh
>
> **备注:** Accepted at ASI @ ICCV 2025
>
> **摘要:** As Vision-Language Models (VLMs) achieve widespread deployment across diverse cultural contexts, ensuring their cultural competence becomes critical for responsible AI systems. While prior work has evaluated cultural awareness in text-only models and VLM object recognition tasks, no research has systematically assessed how VLMs adapt outputs when cultural identity cues are embedded in both textual prompts and visual inputs during generative tasks. We present the first comprehensive evaluation of VLM cultural competence through multimodal story generation, developing a novel multimodal framework that perturbs cultural identity and evaluates 5 contemporary VLMs on a downstream task: story generation. Our analysis reveals significant cultural adaptation capabilities, with rich culturally-specific vocabulary spanning names, familial terms, and geographic markers. However, we uncover concerning limitations: cultural competence varies dramatically across architectures, some models exhibit inverse cultural alignment, and automated metrics show architectural bias contradicting human assessments. Cross-modal evaluation shows that culturally distinct outputs are indeed detectable through visual-semantic similarity (28.7% within-nationality vs. 0.2% cross-nationality recall), yet visual-cultural understanding remains limited. In essence, we establish the promise and challenges of cultural competence in multimodal AI. We publicly release our codebase and data: https://github.com/ArkaMukherjee0/mmCultural
>
---
#### [new 008] Are You Sure You're Positive? Consolidating Chain-of-Thought Agents with Uncertainty Quantification for Aspect-Category Sentiment Analysis
- **分类: cs.CL; cs.IR**

- **简介: 论文针对Aspect-category sentiment analysis任务，解决标注数据稀缺问题。提出利用大语言模型的链式思维代理和不确定性量化，在零样本场景下提升性能与可复现性。**

- **链接: [http://arxiv.org/pdf/2508.17258v1](http://arxiv.org/pdf/2508.17258v1)**

> **作者:** Filippos Ventirozos; Peter Appleby; Matthew Shardlow
>
> **备注:** 18 pages, 10 figures, 3 tables, Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025)
>
> **摘要:** Aspect-category sentiment analysis provides granular insights by identifying specific themes within product reviews that are associated with particular opinions. Supervised learning approaches dominate the field. However, data is scarce and expensive to annotate for new domains. We argue that leveraging large language models in a zero-shot setting is beneficial where the time and resources required for dataset annotation are limited. Furthermore, annotation bias may lead to strong results using supervised methods but transfer poorly to new domains in contexts that lack annotations and demand reproducibility. In our work, we propose novel techniques that combine multiple chain-of-thought agents by leveraging large language models' token-level uncertainty scores. We experiment with the 3B and 70B+ parameter size variants of Llama and Qwen models, demonstrating how these approaches can fulfil practical needs and opening a discussion on how to gauge accuracy in label-scarce conditions.
>
---
#### [new 009] The Power of Framing: How News Headlines Guide Search Behavior
- **分类: cs.CL; cs.HC; cs.IR**

- **简介: 该论文研究新闻标题框架如何影响用户搜索行为，属于信息检索与认知心理学交叉任务。针对“框架是否改变用户查询方向”这一问题，通过控制实验发现冲突、策略和episodic框架显著影响后续搜索，揭示了微小语言线索对信息获取路径的塑造作用。**

- **链接: [http://arxiv.org/pdf/2508.17131v1](http://arxiv.org/pdf/2508.17131v1)**

> **作者:** Amrit Poudel; Maria Milkowski; Tim Weninger
>
> **备注:** Accepted to EMNLP
>
> **摘要:** Search engines play a central role in how people gather information, but subtle cues like headline framing may influence not only what users believe but also how they search. While framing effects on judgment are well documented, their impact on subsequent search behavior is less understood. We conducted a controlled experiment where participants issued queries and selected from headlines filtered by specific linguistic frames. Headline framing significantly shaped follow-up queries: conflict and strategy frames disrupted alignment with prior selections, while episodic frames led to more concrete queries than thematic ones. We also observed modest short-term frame persistence that declined over time. These results suggest that even brief exposure to framing can meaningfully alter the direction of users information-seeking behavior.
>
---
#### [new 010] Beyond Demographics: Enhancing Cultural Value Survey Simulation with Multi-Stage Personality-Driven Cognitive Reasoning
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出MARK框架，用于提升大模型在文化价值观调查模拟中的准确性与可解释性。针对现有方法难以精准模拟个体差异的问题，引入多阶段人格驱动认知推理，结合MBTI理论，实现更贴近真实人类反应的零样本个性化预测。**

- **链接: [http://arxiv.org/pdf/2508.17855v1](http://arxiv.org/pdf/2508.17855v1)**

> **作者:** Haijiang Liu; Qiyuan Li; Chao Gao; Yong Cao; Xiangyu Xu; Xun Wu; Daniel Hershcovich; Jinguang Gu
>
> **备注:** 23 pages, 6 figures, accepted to EMNLP 2025 main
>
> **摘要:** Introducing MARK, the Multi-stAge Reasoning frameworK for cultural value survey response simulation, designed to enhance the accuracy, steerability, and interpretability of large language models in this task. The system is inspired by the type dynamics theory in the MBTI psychological framework for personality research. It effectively predicts and utilizes human demographic information for simulation: life-situational stress analysis, group-level personality prediction, and self-weighted cognitive imitation. Experiments on the World Values Survey show that MARK outperforms existing baselines by 10% accuracy and reduces the divergence between model predictions and human preferences. This highlights the potential of our framework to improve zero-shot personalization and help social scientists interpret model predictions.
>
---
#### [new 011] If We May De-Presuppose: Robustly Verifying Claims through Presupposition-Free Question Decomposition
- **分类: cs.CL**

- **简介: 论文聚焦于事实核查任务，针对大语言模型因前提假设和提示敏感性导致的验证不一致问题，提出一种无前提假设的分解式问答框架，通过结构化推理提升验证鲁棒性，显著减少性能波动并提高准确性。**

- **链接: [http://arxiv.org/pdf/2508.16838v1](http://arxiv.org/pdf/2508.16838v1)**

> **作者:** Shubhashis Roy Dipta; Francis Ferraro
>
> **摘要:** Prior work has shown that presupposition in generated questions can introduce unverified assumptions, leading to inconsistencies in claim verification. Additionally, prompt sensitivity remains a significant challenge for large language models (LLMs), resulting in performance variance as high as 3-6%. While recent advancements have reduced this gap, our study demonstrates that prompt sensitivity remains a persistent issue. To address this, we propose a structured and robust claim verification framework that reasons through presupposition-free, decomposed questions. Extensive experiments across multiple prompts, datasets, and LLMs reveal that even state-of-the-art models remain susceptible to prompt variance and presupposition. Our method consistently mitigates these issues, achieving up to a 2-5% improvement.
>
---
#### [new 012] MahaParaphrase: A Marathi Paraphrase Detection Corpus and BERT-based Models
- **分类: cs.CL; cs.LG**

- **简介: 论文提出MahaParaphrase数据集，用于马拉地语的句子 paraphrase 检测任务，解决低资源印地语族语言NLP数据匮乏问题。工作包括构建8000对人工标注句对，并基于BERT模型进行实验。**

- **链接: [http://arxiv.org/pdf/2508.17444v1](http://arxiv.org/pdf/2508.17444v1)**

> **作者:** Suramya Jadhav; Abhay Shanbhag; Amogh Thakurdesai; Ridhima Sinare; Ananya Joshi; Raviraj Joshi
>
> **摘要:** Paraphrases are a vital tool to assist language understanding tasks such as question answering, style transfer, semantic parsing, and data augmentation tasks. Indic languages are complex in natural language processing (NLP) due to their rich morphological and syntactic variations, diverse scripts, and limited availability of annotated data. In this work, we present the L3Cube-MahaParaphrase Dataset, a high-quality paraphrase corpus for Marathi, a low resource Indic language, consisting of 8,000 sentence pairs, each annotated by human experts as either Paraphrase (P) or Non-paraphrase (NP). We also present the results of standard transformer-based BERT models on these datasets. The dataset and model are publicly shared at https://github.com/l3cube-pune/MarathiNLP
>
---
#### [new 013] Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering
- **分类: cs.CL**

- **简介: 论文研究跨语言技术问答任务，解决长文档中精准检索与生成的问题。通过构建多语言农业设备手册基准，对比RAG与长上下文LLM效果，发现混合RAG策略更优，且部分模型在RAG下准确率超85%。**

- **链接: [http://arxiv.org/pdf/2508.18093v1](http://arxiv.org/pdf/2508.18093v1)**

> **作者:** Julius Gun; Timo Oksanen
>
> **摘要:** We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges.
>
---
#### [new 014] AMELIA: A Family of Multi-task End-to-end Language Models for Argumentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究argument mining任务，旨在用单一大语言模型处理多个相关子任务。作者构建了统一格式的多任务数据集，对比了单任务微调、多任务微调和模型合并三种策略，发现单任务微调效果最佳，多任务微调性能稳定，模型合并则在性能与计算成本间取得平衡。**

- **链接: [http://arxiv.org/pdf/2508.17926v1](http://arxiv.org/pdf/2508.17926v1)**

> **作者:** Henri Savigny; Bruno Yun
>
> **摘要:** Argument mining is a subfield of argumentation that aims to automatically extract argumentative structures and their relations from natural language texts. This paper investigates how a single large language model can be leveraged to perform one or several argument mining tasks. Our contributions are two-fold. First, we construct a multi-task dataset by surveying and converting 19 well-known argument mining datasets from the literature into a unified format. Second, we explore various training strategies using Meta AI's Llama-3.1-8B-Instruct model: (1) fine-tuning on individual tasks, (2) fine-tuning jointly on multiple tasks, and (3) merging models fine-tuned separately on individual tasks. Our experiments show that task-specific fine-tuning significantly improves individual performance across all tasks. Moreover, multi-task fine-tuning maintains strong performance without degradation, suggesting effective transfer learning across related tasks. Finally, we demonstrate that model merging offers a viable compromise: it yields competitive performance while mitigating the computational costs associated with full multi-task fine-tuning.
>
---
#### [new 015] ReProCon: Scalable and Resource-Efficient Few-Shot Biomedical Named Entity Recognition
- **分类: cs.CL**

- **简介: 论文提出ReProCon框架，解决生物医学领域少样本命名实体识别中的数据稀缺和类别不平衡问题。通过多原型建模、余弦对比学习和Reptile元学习，实现高效准确识别，性能接近BERT且资源消耗更低。**

- **链接: [http://arxiv.org/pdf/2508.16833v1](http://arxiv.org/pdf/2508.16833v1)**

> **作者:** Jeongkyun Yoo; Nela Riddle; Andrew Hoblitzell
>
> **摘要:** Named Entity Recognition (NER) in biomedical domains faces challenges due to data scarcity and imbalanced label distributions, especially with fine-grained entity types. We propose ReProCon, a novel few-shot NER framework that combines multi-prototype modeling, cosine-contrastive learning, and Reptile meta-learning to tackle these issues. By representing each category with multiple prototypes, ReProCon captures semantic variability, such as synonyms and contextual differences, while a cosine-contrastive objective ensures strong interclass separation. Reptile meta-updates enable quick adaptation with little data. Using a lightweight fastText + BiLSTM encoder with much lower memory usage, ReProCon achieves a macro-$F_1$ score close to BERT-based baselines (around 99 percent of BERT performance). The model remains stable with a label budget of 30 percent and only drops 7.8 percent in $F_1$ when expanding from 19 to 50 categories, outperforming baselines such as SpanProto and CONTaiNER, which see 10 to 32 percent degradation in Few-NERD. Ablation studies highlight the importance of multi-prototype modeling and contrastive learning in managing class imbalance. Despite difficulties with label ambiguity, ReProCon demonstrates state-of-the-art performance in resource-limited settings, making it suitable for biomedical applications.
>
---
#### [new 016] MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MTalk-Bench，用于评估语音到语音大模型在多轮对话中的表现。解决现有评估框架无法有效衡量复杂对话能力的问题。工作包括构建涵盖语义、副语言和环境声的基准，设计双方法评价体系，并验证模型性能与评估可靠性。**

- **链接: [http://arxiv.org/pdf/2508.18240v1](http://arxiv.org/pdf/2508.18240v1)**

> **作者:** Yuhao Du; Qianwei Huang; Guo Zhu; Zhanchen Dai; Sunian Chen; Qiming Zhu; Yuhao Zhang; Li Zhou; Benyou Wang
>
> **摘要:** The rapid advancement of speech-to-speech (S2S) large language models (LLMs) has significantly improved real-time spoken interaction. However, current evaluation frameworks remain inadequate for assessing performance in complex, multi-turn dialogues. To address this, we introduce MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound. Each dimension includes nine realistic scenarios, along with targeted tasks to assess specific capabilities such as reasoning. Our dual-method evaluation framework combines Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment. The benchmark includes both model and human outputs, evaluated by human evaluators and LLMs. Experimental results reveal two sets of findings. Overall performance of S2S LLMs: (1) models excel at semantic information processing yet underperform on paralinguistic information and ambient sounds perception; (2) models typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues; (3) modality-aware, task-specific designs outperform brute scaling. Evaluation framework and reliability: (1) Arena and Rubrics yield consistent, complementary rankings, but reliable distinctions emerge only when performance gaps are large; (2) LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases and is reliable on nonverbal evaluation only with text annotations. These results highlight current limitations in S2S evaluation and the need for more robust, speech-aware assessment frameworks.
>
---
#### [new 017] Improving Table Understanding with LLMs and Entity-Oriented Search
- **分类: cs.CL**

- **简介: 论文聚焦于表格理解任务，解决现有方法依赖预处理和关键词匹配、缺乏上下文信息的问题。提出基于实体的搜索方法，利用语义相似性和单元格隐含关系提升LLM推理能力，并引入图查询语言，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2508.17028v1](http://arxiv.org/pdf/2508.17028v1)**

> **作者:** Thi-Nhung Nguyen; Hoang Ngo; Dinh Phung; Thuy-Trang Vu; Dat Quoc Nguyen
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Our work addresses the challenges of understanding tables. Existing methods often struggle with the unpredictable nature of table content, leading to a reliance on preprocessing and keyword matching. They also face limitations due to the lack of contextual information, which complicates the reasoning processes of large language models (LLMs). To overcome these challenges, we introduce an entity-oriented search method to improve table understanding with LLMs. This approach effectively leverages the semantic similarities between questions and table data, as well as the implicit relationships between table cells, minimizing the need for data preprocessing and keyword matching. Additionally, it focuses on table entities, ensuring that table cells are semantically tightly bound, thereby enhancing contextual clarity. Furthermore, we pioneer the use of a graph query language for table understanding, establishing a new research direction. Experiments show that our approach achieves new state-of-the-art performances on standard benchmarks WikiTableQuestions and TabFact.
>
---
#### [new 018] SPORTSQL: An Interactive System for Real-Time Sports Reasoning and Visualization
- **分类: cs.CL**

- **简介: 论文提出SPORTSQL系统，解决动态体育数据的自然语言查询与可视化问题。通过LLM将用户提问转为SQL，实现实时分析英超数据，支持表格和图表输出，并构建DSQABENCH评测基准。**

- **链接: [http://arxiv.org/pdf/2508.17157v1](http://arxiv.org/pdf/2508.17157v1)**

> **作者:** Sebastian Martinez; Naman Ahuja; Fenil Bardoliya; Chris Bryan; Vivek Gupta
>
> **备注:** Under Review at EMNLP
>
> **摘要:** We present a modular, interactive system, SPORTSQL, for natural language querying and visualization of dynamic sports data, with a focus on the English Premier League (EPL). The system translates user questions into executable SQL over a live, temporally indexed database constructed from real-time Fantasy Premier League (FPL) data. It supports both tabular and visual outputs, leveraging the symbolic reasoning capabilities of Large Language Models (LLMs) for query parsing, schema linking, and visualization selection. To evaluate system performance, we introduce the Dynamic Sport Question Answering benchmark (DSQABENCH), comprising 1,700+ queries annotated with SQL programs, gold answers, and database snapshots. Our demo highlights how non-expert users can seamlessly explore evolving sports statistics through a natural, conversational interface.
>
---
#### [new 019] GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GRADE框架，用于评估检索增强生成（RAG）系统在多跳问答任务中的表现。针对现有评估忽略推理深度与检索难度的问题，作者构建了可控难度的合成数据集，并设计二维难度矩阵，实现细粒度性能分析。**

- **链接: [http://arxiv.org/pdf/2508.16994v1](http://arxiv.org/pdf/2508.16994v1)**

> **作者:** Jeongsoo Lee; Daeyong Kwon; Kyohoon Jin
>
> **备注:** Accepted at EMNLP 2025 findings
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are widely adopted in knowledge-intensive NLP tasks, but current evaluations often overlook the structural complexity and multi-step reasoning required in real-world scenarios. These benchmarks overlook key factors such as the interaction between retrieval difficulty and reasoning depth. To address this gap, we propose \textsc{GRADE}, a novel evaluation framework that models task difficulty along two orthogonal dimensions: (1) reasoning depth, defined by the number of inference steps (hops), and (2) semantic distance between the query and its supporting evidence. We construct a synthetic multi-hop QA dataset from factual news articles by extracting knowledge graphs and augmenting them through semantic clustering to recover missing links, allowing us to generate diverse and difficulty-controlled queries. Central to our framework is a 2D difficulty matrix that combines generator-side and retriever-side difficulty. Experiments across multiple domains and models show that error rates strongly correlate with our difficulty measures, validating their diagnostic utility. \textsc{GRADE} enables fine-grained analysis of RAG performance and provides a scalable foundation for evaluating and improving multi-hop reasoning in real-world applications.
>
---
#### [new 020] Omne-R1: Learning to Reason with Memory for Multi-hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Omne-R1模型，用于提升无模式知识图谱上的多跳问答能力。针对数据稀缺问题，构建领域无关知识图谱并自动生成问答对，通过多阶段训练框架显著改善复杂多跳问题的解答性能。**

- **链接: [http://arxiv.org/pdf/2508.17330v1](http://arxiv.org/pdf/2508.17330v1)**

> **作者:** Boyuan Liu; Feng Ji; Jiayan Nan; Han Zhao; Weiling Chen; Shihao Xu; Xing Zhou
>
> **摘要:** This paper introduces Omne-R1, a novel approach designed to enhance multi-hop question answering capabilities on schema-free knowledge graphs by integrating advanced reasoning models. Our method employs a multi-stage training workflow, including two reinforcement learning phases and one supervised fine-tuning phase. We address the challenge of limited suitable knowledge graphs and QA data by constructing domain-independent knowledge graphs and auto-generating QA pairs. Experimental results show significant improvements in answering multi-hop questions, with notable performance gains on more complex 3+ hop questions. Our proposed training framework demonstrates strong generalization abilities across diverse knowledge domains.
>
---
#### [new 021] Routing Distilled Knowledge via Mixture of LoRA Experts for Large Language Model based Bundle Generation
- **分类: cs.CL; cs.IR**

- **简介: 论文提出RouteDK框架，用于大语言模型的bundle生成任务，解决知识冲突问题。通过混合LoRA专家和动态融合模块，有效整合高阶与细粒度知识，提升生成准确性和计算效率。**

- **链接: [http://arxiv.org/pdf/2508.17250v1](http://arxiv.org/pdf/2508.17250v1)**

> **作者:** Kaidong Feng; Zhu Sun; Hui Fang; Jie Yang; Wenyuan Liu; Yew-Soon Ong
>
> **摘要:** Large Language Models (LLMs) have shown potential in automatic bundle generation but suffer from prohibitive computational costs. Although knowledge distillation offers a pathway to more efficient student models, our preliminary study reveals that naively integrating diverse types of distilled knowledge from teacher LLMs into student LLMs leads to knowledge conflict, negatively impacting the performance of bundle generation. To address this, we propose RouteDK, a framework for routing distilled knowledge through a mixture of LoRA expert architecture. Specifically, we first distill knowledge from the teacher LLM for bundle generation in two complementary types: high-level knowledge (generalizable rules) and fine-grained knowledge (session-specific reasoning). We then train knowledge-specific LoRA experts for each type of knowledge together with a base LoRA expert. For effective integration, we propose a dynamic fusion module, featuring an input-aware router, where the router balances expert contributions by dynamically determining optimal weights based on input, thereby effectively mitigating knowledge conflicts. To further improve inference reliability, we design an inference-time enhancement module to reduce variance and mitigate suboptimal reasoning. Experiments on three public datasets show that our RouteDK achieves accuracy comparable to or even better than the teacher LLM, while maintaining strong computational efficiency. In addition, it outperforms state-of-the-art approaches for bundle generation.
>
---
#### [new 022] Understanding Subword Compositionality of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究大语言模型如何组合子词信息以形成词级表示，聚焦结构相似性、语义可分解性和形式保留性。通过实验发现五类模型分为三组，体现不同组合策略，揭示了LLMs的组合动态机制。**

- **链接: [http://arxiv.org/pdf/2508.17953v1](http://arxiv.org/pdf/2508.17953v1)**

> **作者:** Qiwei Peng; Yekun Chai; Anders Søgaard
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) take sequences of subwords as input, requiring them to effective compose subword representations into meaningful word-level representations. In this paper, we present a comprehensive set of experiments to probe how LLMs compose subword information, focusing on three key aspects: structural similarity, semantic decomposability, and form retention. Our analysis of the experiments suggests that these five LLM families can be classified into three distinct groups, likely reflecting difference in their underlying composition strategies. Specifically, we observe (i) three distinct patterns in the evolution of structural similarity between subword compositions and whole-word representations across layers; (ii) great performance when probing layer by layer their sensitivity to semantic decompositionality; and (iii) three distinct patterns when probing sensitivity to formal features, e.g., character sequence length. These findings provide valuable insights into the compositional dynamics of LLMs and highlight different compositional pattens in how LLMs encode and integrate subword information.
>
---
#### [new 023] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中动词多词表达（VMWEs）的影响，旨在解决其导致翻译质量下降的问题。作者分析三类VMWEs并提出基于大语言模型的改写方法，显著提升翻译准确率。**

- **链接: [http://arxiv.org/pdf/2508.17458v1](http://arxiv.org/pdf/2508.17458v1)**

> **作者:** Linfeng Liu; Saptarshi Ghosh; Tianyu Jiang
>
> **备注:** 29 pages, 13 figures
>
> **摘要:** Verbal multiword expressions (VMWEs) present significant challenges for natural language processing due to their complex and often non-compositional nature. While machine translation models have seen significant improvement with the advent of language models in recent years, accurately translating these complex linguistic structures remains an open problem. In this study, we analyze the impact of three VMWE categories -- verbal idioms, verb-particle constructions, and light verb constructions -- on machine translation quality from English to multiple languages. Using both established multiword expression datasets and sentences containing these language phenomena extracted from machine translation datasets, we evaluate how state-of-the-art translation systems handle these expressions. Our experimental results consistently show that VMWEs negatively affect translation quality. We also propose an LLM-based paraphrasing approach that replaces these expressions with their literal counterparts, demonstrating significant improvement in translation quality for verbal idioms and verb-particle constructions.
>
---
#### [new 024] EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Spoken Dialogue Systems
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出EMO-Reasoning基准，用于评估对话系统中的情感推理能力。针对情感语音数据稀缺和情感一致性难量化的问题，构建了基于文本转语音的多样化情感数据集，并引入跨轮次情感推理评分机制，通过多维指标评估七种对话系统，有效识别情感不一致，推动更自然的情感交互发展。**

- **链接: [http://arxiv.org/pdf/2508.17623v1](http://arxiv.org/pdf/2508.17623v1)**

> **作者:** Jingwen Liu; Kan Jen Cheng; Jiachen Lian; Akshay Anand; Rishi Jain; Faith Qiao; Robin Netzorg; Huang-Cheng Chou; Tingle Li; Guan-Ting Lin; Gopala Anumanchipalli
>
> **备注:** Accepted at (ASRU 2025) 2025 IEEE Automatic Speech Recognition and Understanding Workshop
>
> **摘要:** Speech emotions play a crucial role in human-computer interaction, shaping engagement and context-aware communication. Despite recent advances in spoken dialogue systems, a holistic system for evaluating emotional reasoning is still lacking. To address this, we introduce EMO-Reasoning, a benchmark for assessing emotional coherence in dialogue systems. It leverages a curated dataset generated via text-to-speech to simulate diverse emotional states, overcoming the scarcity of emotional speech data. We further propose the Cross-turn Emotion Reasoning Score to assess the emotion transitions in multi-turn dialogues. Evaluating seven dialogue systems through continuous, categorical, and perceptual metrics, we show that our framework effectively detects emotional inconsistencies, providing insights for improving current dialogue systems. By releasing a systematic evaluation benchmark, we aim to advance emotion-aware spoken dialogue modeling toward more natural and adaptive interactions.
>
---
#### [new 025] Handling Students Dropouts in an LLM-driven Interactive Online Course Using Language Models
- **分类: cs.CL; cs.CY**

- **简介: 论文研究交互式在线课程中的学生流失问题，提出基于语言模型的预测与干预方法。通过分析交互日志识别流失因素，构建CPADP预测模型（准确率95.4%），并设计个性化邮件召回机制，有效降低流失率。**

- **链接: [http://arxiv.org/pdf/2508.17310v1](http://arxiv.org/pdf/2508.17310v1)**

> **作者:** Yuanchun Wang; Yiyang Fu; Jifan Yu; Daniel Zhang-Li; Zheyuan Zhang; Joy Lim Jia Yin; Yucheng Wang; Peng Zhou; Jing Zhang; Huiqin Liu
>
> **备注:** 12 pages
>
> **摘要:** Interactive online learning environments, represented by Massive AI-empowered Courses (MAIC), leverage LLM-driven multi-agent systems to transform passive MOOCs into dynamic, text-based platforms, enhancing interactivity through LLMs. This paper conducts an empirical study on a specific MAIC course to explore three research questions about dropouts in these interactive online courses: (1) What factors might lead to dropouts? (2) Can we predict dropouts? (3) Can we reduce dropouts? We analyze interaction logs to define dropouts and identify contributing factors. Our findings reveal strong links between dropout behaviors and textual interaction patterns. We then propose a course-progress-adaptive dropout prediction framework (CPADP) to predict dropouts with at most 95.4% accuracy. Based on this, we design a personalized email recall agent to re-engage at-risk students. Applied in the deployed MAIC system with over 3,000 students, the feasibility and effectiveness of our approach have been validated on students with diverse backgrounds.
>
---
#### [new 026] EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks
- **分类: cs.CL; cs.LG**

- **简介: 论文提出EduRABSA，首个面向教育评论的公开Aspect-based Sentiment Analysis（ABSA）数据集，解决教育领域缺乏高质量标注数据的问题。同时提供ASQE-DPT工具，支持多任务标注，助力教育文本情感分析研究。**

- **链接: [http://arxiv.org/pdf/2508.17008v1](http://arxiv.org/pdf/2508.17008v1)**

> **作者:** Yan Cathy Hua; Paul Denny; Jörg Wicker; Katerina Taskova
>
> **摘要:** Every year, most educational institutions seek and receive an enormous volume of text feedback from students on courses, teaching, and overall experience. Yet, turning this raw feedback into useful insights is far from straightforward. It has been a long-standing challenge to adopt automatic opinion mining solutions for such education review text data due to the content complexity and low-granularity reporting requirements. Aspect-based Sentiment Analysis (ABSA) offers a promising solution with its rich, sub-sentence-level opinion mining capabilities. However, existing ABSA research and resources are very heavily focused on the commercial domain. In education, they are scarce and hard to develop due to limited public datasets and strict data protection. A high-quality, annotated dataset is urgently needed to advance research in this under-resourced area. In this work, we present EduRABSA (Education Review ABSA), the first public, annotated ABSA education review dataset that covers three review subject types (course, teaching staff, university) in the English language and all main ABSA tasks, including the under-explored implicit aspect and implicit opinion extraction. We also share ASQE-DPT (Data Processing Tool), an offline, lightweight, installation-free manual data annotation tool that generates labelled datasets for comprehensive ABSA tasks from a single-task annotation. Together, these resources contribute to the ABSA community and education domain by removing the dataset barrier, supporting research transparency and reproducibility, and enabling the creation and sharing of further resources. The dataset, annotation tool, and scripts and statistics for dataset processing and sampling are available at https://github.com/yhua219/edurabsa_dataset_and_annotation_tool.
>
---
#### [new 027] ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ClaimGen-CN数据集，用于中文法律诉求生成任务，旨在帮助非专业人士根据案件事实生成准确、清晰的法律诉求。研究构建了首个该领域数据集，设计评估指标，并对大模型进行零样本测试，揭示现有模型在事实准确性和表达清晰度上的不足。**

- **链接: [http://arxiv.org/pdf/2508.17234v1](http://arxiv.org/pdf/2508.17234v1)**

> **作者:** Siying Zhou; Yiquan Wu; Hui Chen; Xavier Hu; Kun Kuang; Adam Jatowt; Ming Hu; Chunyan Zheng; Fei Wu
>
> **摘要:** Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available.
>
---
#### [new 028] Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Agent-Testing Agent（ATA），用于自动化评估对话AI代理。针对现有评估依赖静态基准和小规模人工测试的问题，ATA结合多种技术生成自适应难度测试用例，通过LLM评分引导测试方向，高效发现多样且严重的缺陷，优于人工标注。**

- **链接: [http://arxiv.org/pdf/2508.17393v1](http://arxiv.org/pdf/2508.17393v1)**

> **作者:** Sameer Komoravolu; Khalil Mrini
>
> **摘要:** LLM agents are increasingly deployed to plan, retrieve, and write with tools, yet evaluation still leans on static benchmarks and small human studies. We present the Agent-Testing Agent (ATA), a meta-agent that combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation whose difficulty adapts via judge feedback. Each dialogue is scored with an LLM-as-a-Judge (LAAJ) rubric and used to steer subsequent tests toward the agent's weakest capabilities. On a travel planner and a Wikipedia writer, the ATA surfaces more diverse and severe failures than expert annotators while matching severity, and finishes in 20--30 minutes versus ten-annotator rounds that took days. Ablating code analysis and web search increases variance and miscalibration, underscoring the value of evidence-grounded test generation. The ATA outputs quantitative metrics and qualitative bug reports for developers. We release the full methodology and open-source implementation for reproducible agent testing: https://github.com/KhalilMrini/Agent-Testing-Agent
>
---
#### [new 029] From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models
- **分类: cs.CL**

- **简介: 该论文研究中文量词预测任务，探讨大语言模型（LLMs）在该任务上的表现及机制。通过掩码策略和微调实验发现，LLMs表现不如BERT，且预测依赖后接名词信息，体现双向注意力优势。**

- **链接: [http://arxiv.org/pdf/2508.18253v1](http://arxiv.org/pdf/2508.18253v1)**

> **作者:** ZiqiZhang; Jianfei Ma; Emmanuele Chersoni; Jieshun You; Zhaoxin Feng
>
> **摘要:** Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
>
---
#### [new 030] Trust but Verify! A Survey on Verification Design for Test-time Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决测试时扩展（TTS）中验证机制不清晰的问题。通过系统梳理验证方法、训练方式与类型，提出统一视角，并构建开源资源库以促进研究。**

- **链接: [http://arxiv.org/pdf/2508.16665v1](http://arxiv.org/pdf/2508.16665v1)**

> **作者:** V Venktesh; Mandeep rathee; Avishek Anand
>
> **备注:** 18 pages
>
> **摘要:** Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing a verifier. The verifiers serve as reward models that help score the candidate outputs from the decoding process to diligently explore the vast solution space and select the best outcome. This paradigm commonly termed has emerged as a superior approach owing to parameter free scaling at inference time and high performance gains. The verifiers could be prompt-based, fine-tuned as a discriminative or generative model to verify process paths, outcomes or both. Despite their widespread adoption, there is no detailed collection, clear categorization and discussion of diverse verification approaches and their training mechanisms. In this survey, we cover the diverse approaches in the literature and present a unified view of verifier training, types and their utility in test-time scaling. Our repository can be found at https://github.com/elixir-research-group/Verifierstesttimescaling.github.io.
>
---
#### [new 031] Exploring the Interplay between Musical Preferences and Personality through the Lens of Language
- **分类: cs.CL**

- **简介: 该论文属于跨领域分析任务，旨在探究音乐偏好是否可通过语言体现人格特征。通过分析500,000条文本数据，发现不同音乐类型粉丝在五大性格维度上存在显著差异，为语言与音乐心理研究提供新视角。**

- **链接: [http://arxiv.org/pdf/2508.18208v1](http://arxiv.org/pdf/2508.18208v1)**

> **作者:** Eliran Shem-Tov; Ella Rabinovich
>
> **摘要:** Music serves as a powerful reflection of individual identity, often aligning with deeper psychological traits. Prior research has established correlations between musical preferences and personality traits, while separate studies have demonstrated that personality is detectable through linguistic analysis. Our study bridges these two research domains by investigating whether individuals' musical preferences are recognizable in their spontaneous language through the lens of the Big Five personality traits (Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism). Using a carefully curated dataset of over 500,000 text samples from nearly 5,000 authors with reliably identified musical preferences, we build advanced models to assess personality characteristics. Our results reveal significant personality differences across fans of five musical genres. We release resources for future research at the intersection of computational linguistics, music psychology and personality analysis.
>
---
#### [new 032] CultranAI at PalmX 2025: Data Augmentation for Cultural Knowledge Representation
- **分类: cs.CL; cs.AI; 68T50; F.2.2; I.2.7**

- **简介: 该论文参与PalmX文化评估任务，旨在提升阿拉伯文化知识表示。通过数据增强和LoRA微调，构建22K+多选题数据集并优化Fanar-1-9B-Instruct模型，最终在盲测集上达到70.50%准确率。**

- **链接: [http://arxiv.org/pdf/2508.17324v1](http://arxiv.org/pdf/2508.17324v1)**

> **作者:** Hunzalah Hassan Bhatti; Youssef Ahmed; Md Arid Hasan; Firoj Alam
>
> **备注:** LLMs, Native, Arabic LLMs, Augmentation, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
>
> **摘要:** In this paper, we report our participation to the PalmX cultural evaluation shared task. Our system, CultranAI, focused on data augmentation and LoRA fine-tuning of large language models (LLMs) for Arabic cultural knowledge representation. We benchmarked several LLMs to identify the best-performing model for the task. In addition to utilizing the PalmX dataset, we augmented it by incorporating the Palm dataset and curated a new dataset of over 22K culturally grounded multiple-choice questions (MCQs). Our experiments showed that the Fanar-1-9B-Instruct model achieved the highest performance. We fine-tuned this model on the combined augmented dataset of 22K+ MCQs. On the blind test set, our submitted system ranked 5th with an accuracy of 70.50%, while on the PalmX development set, it achieved an accuracy of 84.1%.
>
---
#### [new 033] DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对事实核查中的声明归一化任务，提出一种“检索优先、大模型支持”的轻量级框架。通过动态提示GPT-4o-mini或直接检索训练集最相似样本，提升多语言声明的标准化效果，在13种语言中7个获得第一，但零样本场景表现不佳。**

- **链接: [http://arxiv.org/pdf/2508.17402v1](http://arxiv.org/pdf/2508.17402v1)**

> **作者:** Aleksandar Pramov; Jiangqin Ma; Bina Patel
>
> **备注:** CLEF 2025 Working Notes, Madrid, Spain
>
> **摘要:** Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution.
>
---
#### [new 034] Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD
- **分类: cs.CL; cs.CY**

- **简介: 论文提出DuET-PD框架，评估LLMs在说服对话中的鲁棒性和适应性，解决模型易受误导或抗拒纠正的问题。通过双维度测评（说服类型与领域），发现模型表现不佳，并引入Holistic DPO训练方法提升其抗误导能力和接受纠正能力。**

- **链接: [http://arxiv.org/pdf/2508.17450v1](http://arxiv.org/pdf/2508.17450v1)**

> **作者:** Bryan Chen Zhengyu Tan; Daniel Wai Kit Chin; Zhengyuan Liu; Nancy F. Chen; Roy Ka-Wei Lee
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
>
---
#### [new 035] CausalSent: Interpretable Sentiment Classification with RieszNet
- **分类: cs.CL; cs.LG; 68T50**

- **简介: 论文提出CausalSent框架，用于可解释的情感分类任务。针对NLP模型决策黑箱问题，通过两头RieszNet架构提升因果效应估计精度，在IMDB数据上将效果估计误差降低2-3倍，并发现“love”一词使正面情感概率增加2.9%。**

- **链接: [http://arxiv.org/pdf/2508.17576v1](http://arxiv.org/pdf/2508.17576v1)**

> **作者:** Daniel Frees; Martin Pollack
>
> **摘要:** Despite the overwhelming performance improvements offered by recent natural language procesing (NLP) models, the decisions made by these models are largely a black box. Towards closing this gap, the field of causal NLP combines causal inference literature with modern NLP models to elucidate causal effects of text features. We replicate and extend Bansal et al's work on regularizing text classifiers to adhere to estimated effects, focusing instead on model interpretability. Specifically, we focus on developing a two-headed RieszNet-based neural network architecture which achieves better treatment effect estimation accuracy. Our framework, CausalSent, accurately predicts treatment effects in semi-synthetic IMDB movie reviews, reducing MAE of effect estimates by 2-3x compared to Bansal et al's MAE on synthetic Civil Comments data. With an ensemble of validated models, we perform an observational case study on the causal effect of the word "love" in IMDB movie reviews, finding that the presence of the word "love" causes a +2.9% increase in the probability of a positive sentiment.
>
---
#### [new 036] DashboardQA: Benchmarking Multimodal Agents for Question Answering on Interactive Dashboards
- **分类: cs.CL**

- **简介: 该论文提出DashboardQA，首个评估视觉语言模型在交互式仪表板上问答能力的基准。解决现有基准忽略交互性的局限，包含112个真实仪表板和405个问题，揭示当前模型在元素定位、交互规划和推理上的不足。**

- **链接: [http://arxiv.org/pdf/2508.17398v1](http://arxiv.org/pdf/2508.17398v1)**

> **作者:** Aaryaman Kartha; Ahmed Masry; Mohammed Saidul Islam; Thinh Lang; Shadikur Rahman; Ridwan Mahbub; Mizanur Rahman; Mahir Ahmed; Md Rizwan Parvez; Enamul Hoque; Shafiq Joty
>
> **摘要:** Dashboards are powerful visualization tools for data-driven decision-making, integrating multiple interactive views that allow users to explore, filter, and navigate data. Unlike static charts, dashboards support rich interactivity, which is essential for uncovering insights in real-world analytical workflows. However, existing question-answering benchmarks for data visualizations largely overlook this interactivity, focusing instead on static charts. This limitation severely constrains their ability to evaluate the capabilities of modern multimodal agents designed for GUI-based reasoning. To address this gap, we introduce DashboardQA, the first benchmark explicitly designed to assess how vision-language GUI agents comprehend and interact with real-world dashboards. The benchmark includes 112 interactive dashboards from Tableau Public and 405 question-answer pairs with interactive dashboards spanning five categories: multiple-choice, factoid, hypothetical, multi-dashboard, and conversational. By assessing a variety of leading closed- and open-source GUI agents, our analysis reveals their key limitations, particularly in grounding dashboard elements, planning interaction trajectories, and performing reasoning. Our findings indicate that interactive dashboard reasoning is a challenging task overall for all the VLMs evaluated. Even the top-performing agents struggle; for instance, the best agent based on Gemini-Pro-2.5 achieves only 38.69% accuracy, while the OpenAI CUA agent reaches just 22.69%, demonstrating the benchmark's significant difficulty. We release DashboardQA at https://github.com/vis-nlp/DashboardQA
>
---
#### [new 037] Unbiased Reasoning for Knowledge-Intensive Tasks in Large Language Models via Conditional Front-Door Adjustment
- **分类: cs.CL**

- **简介: 论文提出Conditional Front-Door Prompting（CFD-Prompting），用于知识密集型任务中减少大语言模型的内部偏见，通过因果推理提升准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.16910v1](http://arxiv.org/pdf/2508.16910v1)**

> **作者:** Bo Zhao; Yinghao Zhang; Ziqi Xu; Yongli Ren; Xiuzhen Zhang; Renqiang Luo; Zaiwen Feng; Feng Xia
>
> **备注:** This paper has been accepted to the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025), Full Research Paper
>
> **摘要:** Large Language Models (LLMs) have shown impressive capabilities in natural language processing but still struggle to perform well on knowledge-intensive tasks that require deep reasoning and the integration of external knowledge. Although methods such as Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) have been proposed to enhance LLMs with external knowledge, they still suffer from internal bias in LLMs, which often leads to incorrect answers. In this paper, we propose a novel causal prompting framework, Conditional Front-Door Prompting (CFD-Prompting), which enables the unbiased estimation of the causal effect between the query and the answer, conditional on external knowledge, while mitigating internal bias. By constructing counterfactual external knowledge, our framework simulates how the query behaves under varying contexts, addressing the challenge that the query is fixed and is not amenable to direct causal intervention. Compared to the standard front-door adjustment, the conditional variant operates under weaker assumptions, enhancing both robustness and generalisability of the reasoning process. Extensive experiments across multiple LLMs and benchmark datasets demonstrate that CFD-Prompting significantly outperforms existing baselines in both accuracy and robustness.
>
---
#### [new 038] Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit
- **分类: cs.CL; cs.AI**

- **简介: 论文针对大语言模型在复杂推理中因“过度思考”导致资源浪费的问题，提出通过挖掘早期推理结束点（RCP）模式来优化推理过程。工作包括识别三阶段推理、设计轻量阈值策略，实验表明该方法减少token消耗并保持或提升准确性。**

- **链接: [http://arxiv.org/pdf/2508.17627v1](http://arxiv.org/pdf/2508.17627v1)**

> **作者:** Zihao Wei; Liang Pang; Jiahao Liu; Jingcheng Deng; Shicheng Xu; Zenghao Duan; Jingang Wang; Fei Sun; Xunliang Cai; Huawei Shen; Xueqi Cheng
>
> **摘要:** Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy.
>
---
#### [new 039] Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究自然语言可满足性问题，探讨不同计算复杂度和语法结构对Transformer模型推理能力的影响，并通过实证分析问题分布以更准确评估模型性能。**

- **链接: [http://arxiv.org/pdf/2508.17153v1](http://arxiv.org/pdf/2508.17153v1)**

> **作者:** Tharindu Madusanka; Ian Pratt-Hartmann; Riza Batista-Navarro
>
> **备注:** The paper was accepted to the 62nd Association for Computational Linguistics (ACL 2024), where it won the Best Paper Award
>
> **摘要:** Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems.
>
---
#### [new 040] LLMs Learn Constructions That Humans Do Not Know
- **分类: cs.CL**

- **简介: 论文研究大语言模型（LLM）生成人类无法感知的虚假语法结构，通过行为和元语言探测任务发现模型存在构造幻觉。研究表明此类幻觉易被误判为真实语法知识，揭示了当前构造探测方法存在确认偏差问题。**

- **链接: [http://arxiv.org/pdf/2508.16837v1](http://arxiv.org/pdf/2508.16837v1)**

> **作者:** Jonathan Dunn; Mai Mohamed Eida
>
> **摘要:** This paper investigates false positive constructions: grammatical structures which an LLM hallucinates as distinct constructions but which human introspection does not support. Both a behavioural probing task using contextual embeddings and a meta-linguistic probing task using prompts are included, allowing us to distinguish between implicit and explicit linguistic knowledge. Both methods reveal that models do indeed hallucinate constructions. We then simulate hypothesis testing to determine what would have happened if a linguist had falsely hypothesized that these hallucinated constructions do exist. The high accuracy obtained shows that such false hypotheses would have been overwhelmingly confirmed. This suggests that construction probing methods suffer from a confirmation bias and raises the issue of what unknown and incorrect syntactic knowledge these models also possess.
>
---
#### [new 041] Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios
- **分类: cs.CL; cs.AI; cs.CY; I.2; I.2.7**

- **简介: 论文提出AulSign方法，利用大语言模型在低资源场景下进行自然语言到手语的翻译。针对缺乏平行语料的问题，通过动态提示和样本选择，将手语符号关联自然语言描述，提升翻译准确性。**

- **链接: [http://arxiv.org/pdf/2508.18183v1](http://arxiv.org/pdf/2508.18183v1)**

> **作者:** Luana Bulla; Gabriele Tuccio; Misael Mongiovì; Aldo Gangemi
>
> **摘要:** Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities.
>
---
#### [new 042] CoCoA: Confidence- and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型生成中的知识冲突问题，提出CoCoA算法，通过信心和上下文感知机制实现更忠实的文本生成。在问答、摘要等任务上显著提升准确性和事实性。**

- **链接: [http://arxiv.org/pdf/2508.17670v1](http://arxiv.org/pdf/2508.17670v1)**

> **作者:** Anant Khandelwal; Manish Gupta; Puneet Agrawal
>
> **备注:** Accepted to EMNLP'25, Main. 21 pages, 17 tables, 3 Figures
>
> **摘要:** Faithful generation in large language models (LLMs) is challenged by knowledge conflicts between parametric memory and external context. Existing contrastive decoding methods tuned specifically to handle conflict often lack adaptability and can degrade performance in low conflict settings. We introduce CoCoA (Confidence- and Context-Aware Adaptive Decoding), a novel token-level algorithm for principled conflict resolution and enhanced faithfulness. CoCoA resolves conflict by utilizing confidence-aware measures (entropy gap and contextual peakedness) and the generalized divergence between the parametric and contextual distributions. Crucially, CoCoA maintains strong performance even in low conflict settings. Extensive experiments across multiple LLMs on diverse Question Answering (QA), Summarization, and Long-Form Question Answering (LFQA) benchmarks demonstrate CoCoA's state-of-the-art performance over strong baselines like AdaCAD. It yields significant gains in QA accuracy, up to 9.2 points on average compared to the strong baseline AdaCAD, and improves factuality in summarization and LFQA by up to 2.5 points on average across key benchmarks. Additionally, it demonstrates superior sensitivity to conflict variations. CoCoA enables more informed, context-aware, and ultimately more faithful token generation.
>
---
#### [new 043] Better Language Model-Based Judging Reward Modeling through Scaling Comprehension Boundaries
- **分类: cs.CL**

- **简介: 论文提出ESFP-RM框架，通过结合上下文解释的掩码语言模型提升奖励建模效果，解决RLAIF中奖励信号不稳定问题，提升模型在RLHF和分布外场景下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.18212v1](http://arxiv.org/pdf/2508.18212v1)**

> **作者:** Meiling Ning; Zhongbao Zhang; Junda Ye; Jiabao Guo; Qingyuan Guan
>
> **摘要:** The emergence of LM-based judging reward modeling, represented by generative reward models, has successfully made reinforcement learning from AI feedback (RLAIF) efficient and scalable. To further advance this paradigm, we propose a core insight: this form of reward modeling shares fundamental formal consistency with natural language inference (NLI), a core task in natural language understanding. This reframed perspective points to a key path for building superior reward models: scaling the model's comprehension boundaries. Pursuing this path, exploratory experiments on NLI tasks demonstrate that the slot prediction masked language models (MLMs) incorporating contextual explanations achieve significantly better performance compared to mainstream autoregressive models. Based on this key finding, we propose ESFP-RM, a two-stage LM-based judging reward model that utilizes an explanation based slot framework for prediction to fully leverage the advantages of MLMs. Extensive experiments demonstrate that in both reinforcement learning from human feedback (RLHF) and out-of-distribution (OOD) scenarios, the ESFP-RM framework delivers more stable and generalizable reward signals compared to generative reward models.
>
---
#### [new 044] DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation
- **分类: cs.CL; cs.IR**

- **简介: 论文提出DeAR框架，用于文档重排序任务，解决单一模型难以兼顾细粒度打分与整体分析的问题。通过双阶段设计：第一阶段蒸馏教师模型的token级相关性信号，第二阶段引入LoRA适配器进行链式推理微调，提升准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2508.16998v1](http://arxiv.org/pdf/2508.16998v1)**

> **作者:** Abdelrahman Abdallah; Jamshid Mozafari; Bhawna Piryani; Adam Jatowt
>
> **备注:** Accept at EMNLP Findings 2025
>
> **摘要:** Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at https://github.com/DataScienceUIBK/DeAR-Reranking.}.
>
---
#### [new 045] How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models
- **分类: cs.CL; cs.IR**

- **简介: 论文研究信息检索中的重排序任务，比较LLM-based与轻量级模型在常见和新查询上的性能差异。通过22种方法在多个基准上的实证分析，发现LLM模型在熟悉查询上表现更好，但新查询上表现不稳定，轻量模型更具效率。**

- **链接: [http://arxiv.org/pdf/2508.16757v1](http://arxiv.org/pdf/2508.16757v1)**

> **作者:** Abdelrahman Abdallah; Bhawna Piryani; Jamshid Mozafari; Mohammed Ali; Adam Jatowt
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** In this work, we present a systematic and comprehensive empirical evaluation of state-of-the-art reranking methods, encompassing large language model (LLM)-based, lightweight contextual, and zero-shot approaches, with respect to their performance in information retrieval tasks. We evaluate in total 22 methods, including 40 variants (depending on used LLM) across several established benchmarks, including TREC DL19, DL20, and BEIR, as well as a novel dataset designed to test queries unseen by pretrained models. Our primary goal is to determine, through controlled and fair comparisons, whether a performance disparity exists between LLM-based rerankers and their lightweight counterparts, particularly on novel queries, and to elucidate the underlying causes of any observed differences. To disentangle confounding factors, we analyze the effects of training data overlap, model architecture, and computational efficiency on reranking performance. Our findings indicate that while LLM-based rerankers demonstrate superior performance on familiar queries, their generalization ability to novel queries varies, with lightweight models offering comparable efficiency. We further identify that the novelty of queries significantly impacts reranking effectiveness, highlighting limitations in existing approaches. https://github.com/DataScienceUIBK/llm-reranking-generalization-study
>
---
#### [new 046] ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks
- **分类: cs.CL**

- **简介: 论文提出OBJEX(MT)基准，评估大模型在多轮对抗性攻击中提取对话目标并校准自身信心的能力。解决LLM作为裁判时误判目标且过度自信的问题，通过实验发现多数模型在复杂攻击下表现不佳，建议明确目标或采用选择性预测以降低风险。**

- **链接: [http://arxiv.org/pdf/2508.16889v1](http://arxiv.org/pdf/2508.16889v1)**

> **作者:** Hyunjun Kim; Junwoo Ha; Sangyoon Yu; Haon Park
>
> **摘要:** Large language models (LLMs) are increasingly used as judges of other models, yet it is unclear whether a judge can reliably infer the latent objective of the conversation it evaluates, especially when the goal is distributed across noisy, adversarial, multi-turn jailbreaks. We introduce OBJEX(MT), a benchmark that requires a model to (i) distill a transcript into a single-sentence base objective and (ii) report its own confidence. Accuracy is scored by an LLM judge using semantic similarity between extracted and gold objectives; correctness uses a single human-aligned threshold calibrated once on N=100 items (tau* = 0.61); and metacognition is evaluated with ECE, Brier score, Wrong@High-Conf, and risk-coverage curves. We evaluate gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMT Attack_600, SafeMTData_1K, MHJ, and CoSafe. claude-sonnet-4 attains the highest objective-extraction accuracy (0.515) and the best calibration (ECE 0.296; Brier 0.324), while gpt-4.1 and Qwen3 tie at 0.441 accuracy yet show marked overconfidence (mean confidence approx. 0.88 vs. accuracy approx. 0.44; Wrong@0.90 approx. 48-52%). Performance varies sharply across datasets (approx. 0.167-0.865), with MHJ comparatively easy and Attack_600/CoSafe harder. These results indicate that LLM judges often misinfer objectives with high confidence in multi-turn jailbreaks and suggest operational guidance: provide judges with explicit objectives when possible and use selective prediction or abstention to manage risk. We release prompts, scoring templates, and complete logs to facilitate replication and analysis.
>
---
#### [new 047] Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DreamCUB框架，将对话建模为POMDP问题，通过用户信念建模（情绪、情感、意图）实现基于模型的强化学习，提升对话质量和泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.16876v1](http://arxiv.org/pdf/2508.16876v1)**

> **作者:** Yue Zhao; Xiaoyu Wang; Dan Wang; Zhonglin Jiang; Qingqing Gu; Teng Chen; Ningyuan Xi; Jinxian Qu; Yong Chen; Luo Ji
>
> **摘要:** World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues.
>
---
#### [new 048] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains
- **分类: cs.CL; I.2.3; I.2.4; I.2.7**

- **简介: 论文提出MIRAGE框架，用于医疗问答任务中的测试时推理扩展。针对现有方法依赖线性链导致错误累积的问题，该工作通过结构化知识图谱实现并行多链推理与动态证据检索，提升准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2508.18260v1](http://arxiv.org/pdf/2508.18260v1)**

> **作者:** Kaiwen Wei; Rui Shan; Dongsheng Zou; Jianzhong Yang; Bi Zhao; Junnan Zhu; Jiang Zhong
>
> **备注:** 10 pages, 8 figures (including tables), plus appendix. Submitted to AAAI 2026
>
> **摘要:** Large reasoning models (LRMs) have shown significant progress in test-time scaling through chain-of-thought prompting. Current approaches like search-o1 integrate retrieval augmented generation (RAG) into multi-step reasoning processes but rely on a single, linear reasoning chain while incorporating unstructured textual information in a flat, context-agnostic manner. As a result, these approaches can lead to error accumulation throughout the reasoning chain, which significantly limits its effectiveness in medical question-answering (QA) tasks where both accuracy and traceability are critical requirements. To address these challenges, we propose MIRAGE (Multi-chain Inference with Retrieval-Augmented Graph Exploration), a novel test-time scalable reasoning framework that performs dynamic multi-chain inference over structured medical knowledge graphs. Specifically, MIRAGE 1) decomposes complex queries into entity-grounded sub-questions, 2) executes parallel inference chains, 3) retrieves evidence adaptively via neighbor expansion and multi-hop traversal, and 4) integrates answers using cross-chain verification to resolve contradictions. Experiments on three medical QA benchmarks (GenMedGPT-5k, CMCQA, and ExplainCPE) show that MIRAGE consistently outperforms GPT-4o, Tree-of-Thought variants, and other retrieval-augmented baselines in both automatic and human evaluations. Additionally, MIRAGE improves interpretability by generating explicit reasoning chains that trace each factual claim to concrete chains within the knowledge graph, making it well-suited for complex medical reasoning scenarios. The code will be available for further research.
>
---
#### [new 049] Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 论文提出BridgeX-ICL方法，通过挖掘语言重叠神经元提升低资源语言的零样本跨语言提示学习性能，解决LLM在低资源语言上表现差的问题。**

- **链接: [http://arxiv.org/pdf/2508.17078v1](http://arxiv.org/pdf/2508.17078v1)**

> **作者:** Yuemei Xu; Kexin Xu; Jian Zhou; Ling Hu; Lin Gui
>
> **摘要:** The current Large Language Models (LLMs) face significant challenges in improving performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose BridgeX-ICL, a simple yet effective method to improve zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs or not. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly, to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlap neurons, which guides optimal bridge selection. The experiments conducted on 2 cross-lingual tasks and 15 language pairs from 7 diverse families (covering both high-low and moderate-low pairs) validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs.
>
---
#### [new 050] Why Synthetic Isn't Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文针对客服对话生成任务，解决合成对话缺乏真实感的问题。提出18项诊断指标评估合成对话质量，对比多种生成策略，揭示在不流畅、情感和行为真实性上的不足，推动更可靠的合成数据生成。**

- **链接: [http://arxiv.org/pdf/2508.18210v1](http://arxiv.org/pdf/2508.18210v1)**

> **作者:** Rishikesh Devanathan; Varun Nathan; Ayush Kumar
>
> **摘要:** Synthetic transcript generation is critical in contact center domains, where privacy and data scarcity limit model training and evaluation. Unlike prior synthetic dialogue generation work on open-domain or medical dialogues, contact center conversations are goal-oriented, role-asymmetric, and behaviorally complex, featuring disfluencies, ASR noise, and compliance-driven agent actions. In deployments where transcripts are unavailable, standard pipelines still yield derived call attributes such as Intent Summaries, Topic Flow, and QA Evaluation Forms. We leverage these as supervision signals to guide generation. To assess the quality of such outputs, we introduce a diagnostic framework of 18 linguistically and behaviorally grounded metrics for comparing real and synthetic transcripts. We benchmark four language-agnostic generation strategies, from simple prompting to characteristic-aware multi-stage approaches, alongside reference-free baselines. Results reveal persistent challenges: no method excels across all traits, with notable deficits in disfluency, sentiment, and behavioral realism. Our diagnostic tool exposes these gaps, enabling fine-grained evaluation and stress testing of synthetic dialogue across languages.
>
---
#### [new 051] Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 论文提出构建法律知识图谱，以捕捉法院判决中从事实到法律的推理路径。解决现有方法难以准确关联事实与法律、忽略推理结构的问题。通过提示驱动的大模型提取推理组件，标准化法律条文引用，并用法律推理本体连接事实、规范与适用关系，实现可机器读取的结构化法律推理表示。**

- **链接: [http://arxiv.org/pdf/2508.17340v1](http://arxiv.org/pdf/2508.17340v1)**

> **作者:** Ryoma Kondo; Riona Matsuoka; Takahiro Yoshida; Kazuyuki Yamasawa; Ryohei Hisano
>
> **摘要:** Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods.
>
---
#### [new 052] Geolocation-Aware Robust Spoken Language Identification
- **分类: cs.CL; cs.SD**

- **简介: 论文提出地理信息感知的语音语言识别方法，通过引入地理预测辅助任务和条件信号，增强模型对同一语言内方言和口音变化的鲁棒性，提升跨域泛化能力，在多个数据集上取得新最优结果。**

- **链接: [http://arxiv.org/pdf/2508.17148v1](http://arxiv.org/pdf/2508.17148v1)**

> **作者:** Qingzheng Wang; Hye-jin Shim; Jiancheng Sun; Shinji Watanabe
>
> **备注:** Accepted to IEEE ASRU 2025. \c{opyright} 2025 IEEE. Personal use permitted. Permission from IEEE required for all other uses including reprinting/republishing, advertising, resale, redistribution, reuse, or creating collective works
>
> **摘要:** While Self-supervised Learning (SSL) has significantly improved Spoken Language Identification (LID), existing models often struggle to consistently classify dialects and accents of the same language as a unified class. To address this challenge, we propose geolocation-aware LID, a novel approach that incorporates language-level geolocation information into the SSL-based LID model. Specifically, we introduce geolocation prediction as an auxiliary task and inject the predicted vectors into intermediate representations as conditioning signals. This explicit conditioning encourages the model to learn more unified representations for dialectal and accented variations. Experiments across six multilingual datasets demonstrate that our approach improves robustness to intra-language variations and unseen domains, achieving new state-of-the-art accuracy on FLEURS (97.7%) and 9.7% relative improvement on ML-SUPERB 2.0 dialect set.
>
---
#### [new 053] Efficient Zero-Shot Long Document Classification by Reducing Context Through Sentence Ranking
- **分类: cs.CL; cs.LG**

- **简介: 论文提出一种零样本长文档分类方法，通过TF-IDF句排序减少上下文，提升效率。解决Transformer模型因输入长度限制难以处理长文档的问题，仅保留Top 50%句子即可保持准确率并提速35%。**

- **链接: [http://arxiv.org/pdf/2508.17490v1](http://arxiv.org/pdf/2508.17490v1)**

> **作者:** Prathamesh Kokate; Mitali Sarnaik; Manavi Khopade; Mukta Takalikar; Raviraj Joshi
>
> **摘要:** Transformer-based models like BERT excel at short text classification but struggle with long document classification (LDC) due to input length limitations and computational inefficiencies. In this work, we propose an efficient, zero-shot approach to LDC that leverages sentence ranking to reduce input context without altering the model architecture. Our method enables the adaptation of models trained on short texts, such as headlines, to long-form documents by selecting the most informative sentences using a TF-IDF-based ranking strategy. Using the MahaNews dataset of long Marathi news articles, we evaluate three context reduction strategies that prioritize essential content while preserving classification accuracy. Our results show that retaining only the top 50\% ranked sentences maintains performance comparable to full-document inference while reducing inference time by up to 35\%. This demonstrates that sentence ranking is a simple yet effective technique for scalable and efficient zero-shot LDC.
>
---
#### [new 054] Steering When Necessary: Flexible Steering Large Language Models with Backtracking
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型生成行为对齐问题，提出FASB框架，通过追踪内部状态动态决定干预必要性和强度，并引入回溯机制纠正偏差，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2508.17621v1](http://arxiv.org/pdf/2508.17621v1)**

> **作者:** Jinwei Gan; Zifeng Cheng; Zhiwei Jiang; Cong Wang; Yafeng Yin; Xiang Luo; Yuchen Fu; Qing Gu
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance across many generation tasks. Nevertheless, effectively aligning them with desired behaviors remains a significant challenge. Activation steering is an effective and cost-efficient approach that directly modifies the activations of LLMs during the inference stage, aligning their responses with the desired behaviors and avoiding the high cost of fine-tuning. Existing methods typically indiscriminately intervene to all generations or rely solely on the question to determine intervention, which limits the accurate assessment of the intervention strength. To this end, we propose the Flexible Activation Steering with Backtracking (FASB) framework, which dynamically determines both the necessity and strength of intervention by tracking the internal states of the LLMs during generation, considering both the question and the generated content. Since intervening after detecting a deviation from the desired behavior is often too late, we further propose the backtracking mechanism to correct the deviated tokens and steer the LLMs toward the desired behavior. Extensive experiments on the TruthfulQA dataset and six multiple-choice datasets demonstrate that our method outperforms baselines. Our code will be released at https://github.com/gjw185/FASB.
>
---
#### [new 055] KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online RLHF
- **分类: cs.CL; cs.LG; 68T07; I.2.6; I.2.8**

- **简介: 论文提出KLQ方法，用于语言模型强化学习中的人类反馈（LM-RLHF）任务，解决PPO在KL约束处理上的启发式问题。KLQ从动作价值视角出发，等价于特定条件下的PPO，并在摘要和对话任务上表现优于PPO。**

- **链接: [http://arxiv.org/pdf/2508.17000v1](http://arxiv.org/pdf/2508.17000v1)**

> **作者:** Jason R Brown; Lennie Wells; Edward James Young; Sergio Bacallado
>
> **摘要:** Proximal Policy Optimisation (PPO) is an established and effective policy gradient algorithm used for Language Model Reinforcement Learning from Human Feedback (LM-RLHF). PPO performs well empirically but has a heuristic motivation and handles the KL-divergence constraint used in LM-RLHF in an ad-hoc manner. In this paper, we develop a a new action-value RL method for the LM-RLHF setting, KL-regularised Q-Learning (KLQ). We then show that our method is equivalent to a version of PPO in a certain specific sense, despite its very different motivation. Finally, we benchmark KLQ on two key language generation tasks -- summarisation and single-turn dialogue. We demonstrate that KLQ performs on-par with PPO at optimising the LM-RLHF objective, and achieves a consistently higher win-rate against PPO on LLM-as-a-judge evaluations.
>
---
#### [new 056] DiscussLLM: Teaching Large Language Models When to Speak
- **分类: cs.CL; cs.HC**

- **简介: 论文提出DiscussLLM框架，解决大语言模型被动响应问题，使其能主动判断何时发言。通过两阶段数据生成和训练，模型学会在合适时机介入对话，提升协作能力。**

- **链接: [http://arxiv.org/pdf/2508.18167v1](http://arxiv.org/pdf/2508.18167v1)**

> **作者:** Deep Anil Patel; Iain Melvin; Christopher Malon; Martin Renqiang Min
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like text, yet they largely operate as reactive agents, responding only when directly prompted. This passivity creates an "awareness gap," limiting their potential as truly collaborative partners in dynamic human discussions. We introduce $\textit{DiscussLLM}$, a framework designed to bridge this gap by training models to proactively decide not just $\textit{what}$ to say, but critically, $\textit{when}$ to speak. Our primary contribution is a scalable two-stage data generation pipeline that synthesizes a large-scale dataset of realistic multi-turn human discussions. Each discussion is annotated with one of five intervention types (e.g., Factual Correction, Concept Definition) and contains an explicit conversational trigger where an AI intervention adds value. By training models to predict a special silent token when no intervention is needed, they learn to remain quiet until a helpful contribution can be made. We explore two architectural baselines: an integrated end-to-end model and a decoupled classifier-generator system optimized for low-latency inference. We evaluate these models on their ability to accurately time interventions and generate helpful responses, paving the way for more situationally aware and proactive conversational AI.
>
---
#### [new 057] Neither Valid nor Reliable? Investigating the Use of LLMs as Judges
- **分类: cs.CL; I.2.7**

- **简介: 论文探讨LLMs作为评判者（LLJs）在自然语言生成评估中的有效性与可靠性问题，指出当前应用存在假设未被充分验证的风险。通过分析四个核心假设并结合文本摘要、数据标注和安全对齐三个场景，提出需建立更负责任的评估实践以确保NLG进展不被误导。**

- **链接: [http://arxiv.org/pdf/2508.18076v1](http://arxiv.org/pdf/2508.18076v1)**

> **作者:** Khaoula Chehbouni; Mohammed Haddou; Jackie Chi Kit Cheung; Golnoosh Farnadi
>
> **备注:** Prepared for conference submission
>
> **摘要:** Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG.
>
---
#### [new 058] SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SSFO，一种自监督对齐方法，用于提升检索增强生成（RAG）系统的忠实性。通过对比有无上下文时的输出，利用DPO优化模型，无需标注成本或额外推理开销，显著改善答案忠实性并保持指令遵循能力。**

- **链接: [http://arxiv.org/pdf/2508.17225v1](http://arxiv.org/pdf/2508.17225v1)**

> **作者:** Xiaqiang Tang; Yi Wang; Keyu Hu; Rui Xu; Chuang Li; Weigao Sun; Jian Li; Sihong Xie
>
> **备注:** Working in progress
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training or significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), the first self-supervised alignment approach for enhancing RAG faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with and without the context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burden. We theoretically and empirically demonstrate that SSFO leverages a benign form of \emph{likelihood displacement}, transferring probability mass from parametric-based tokens to context-aligned tokens. Based on this insight, we propose a modified DPO loss function to encourage likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness and preserving general instruction-following capabilities. We release our code and model at the anonymous link: https://github.com/chkwy/SSFO
>
---
#### [new 059] Do Cognitively Interpretable Reasoning Traces Improve LLM Performance?
- **分类: cs.CL; cs.AI**

- **简介: 论文研究开放书问答任务，探究可解释推理轨迹是否提升大模型性能。通过四种轨迹类型微调模型并开展人类评估，发现最优性能轨迹最不具可解释性，表明中间推理步骤无需面向用户可读。**

- **链接: [http://arxiv.org/pdf/2508.16695v1](http://arxiv.org/pdf/2508.16695v1)**

> **作者:** Siddhant Bhambri; Upasana Biswas; Subbarao Kambhampati
>
> **摘要:** Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability.
>
---
#### [new 060] How Quantization Shapes Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究量化对大语言模型偏见的影响，解决效率与伦理平衡问题。通过多基准评估权重和激活量化策略，发现量化可降低毒性但可能加剧刻板印象和不公平，尤其在高压缩下。**

- **链接: [http://arxiv.org/pdf/2508.18088v1](http://arxiv.org/pdf/2508.18088v1)**

> **作者:** Federico Marcuzzi; Xuefei Ning; Roy Schwartz; Iryna Gurevych
>
> **摘要:** This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups. We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, toxicity, sentiment, and fairness. We employ both probabilistic and generated text-based metrics across nine benchmarks and evaluate models varying in architecture family and reasoning ability. Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression. These trends are generally consistent across demographic categories and model types, although their magnitude depends on the specific setting. Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice.
>
---
#### [new 061] Active Domain Knowledge Acquisition with \$100 Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains
- **分类: cs.CL**

- **简介: 论文提出PU-ADKA框架，解决在预算有限下如何高效获取专家知识以增强LLMs在敏感领域（如药物研发）的表现。通过模拟训练和真实专家交互验证方法有效性，并构建新基准CKAD促进研究。**

- **链接: [http://arxiv.org/pdf/2508.17202v1](http://arxiv.org/pdf/2508.17202v1)**

> **作者:** Yang Wu; Raha Moraffah; Rujing Yao; Jinhong Yu; Zhimin Tao; Xiaozhong Liu
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
>
---
#### [new 062] EMPOWER: Evolutionary Medical Prompt Optimization With Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出EMPOWER框架，用于优化医疗领域大语言模型的提示词。针对现有方法无法满足医学知识和安全要求的问题，通过注意力机制、多维评估和进化算法提升提示质量，显著减少错误内容并提高临床相关性。**

- **链接: [http://arxiv.org/pdf/2508.17703v1](http://arxiv.org/pdf/2508.17703v1)**

> **作者:** Yinda Chen; Yangfan He; Jing Yang; Dapeng Zhang; Zhenlong Yuan; Muhammad Attique Khan; Jamel Baili; Por Lip Yee
>
> **摘要:** Prompt engineering significantly influences the reliability and clinical utility of Large Language Models (LLMs) in medical applications. Current optimization approaches inadequately address domain-specific medical knowledge and safety requirements. This paper introduces EMPOWER, a novel evolutionary framework that enhances medical prompt quality through specialized representation learning, multi-dimensional evaluation, and structure-preserving algorithms. Our methodology incorporates: (1) a medical terminology attention mechanism, (2) a comprehensive assessment architecture evaluating clarity, specificity, clinical relevance, and factual accuracy, (3) a component-level evolutionary algorithm preserving clinical reasoning integrity, and (4) a semantic verification module ensuring adherence to medical knowledge. Evaluation across diagnostic, therapeutic, and educational tasks demonstrates significant improvements: 24.7% reduction in factually incorrect content, 19.6% enhancement in domain specificity, and 15.3% higher clinician preference in blinded evaluations. The framework addresses critical challenges in developing clinically appropriate prompts, facilitating more responsible integration of LLMs into healthcare settings.
>
---
#### [new 063] Layerwise Importance Analysis of Feed-Forward Networks in Transformer-based Language Models
- **分类: cs.CL**

- **简介: 该论文研究Transformer语言模型中前馈网络（FFN）在不同层的重要性。通过调整FFN分布并从头训练模型，发现将FFN集中在中间70%的层能提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2508.17734v1](http://arxiv.org/pdf/2508.17734v1)**

> **作者:** Wataru Ikeda; Kazuki Yano; Ryosuke Takahashi; Jaesung Lee; Keigo Shibata; Jun Suzuki
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** This study investigates the layerwise importance of feed-forward networks (FFNs) in Transformer-based language models during pretraining. We introduce an experimental approach that, while maintaining the total parameter count, increases the FFN dimensions in some layers and completely removes the FFNs from other layers. Furthermore, since our focus is on the importance of FFNs during pretraining, we train models from scratch to examine whether the importance of FFNs varies depending on their layer positions, rather than using publicly available pretrained models as is frequently done. Through comprehensive evaluations of models with varying sizes (285M, 570M, and 1.2B parameters) and layer counts (12, 24, and 40 layers), we demonstrate that concentrating FFNs in 70% of the consecutive middle layers consistently outperforms standard configurations for multiple downstream tasks.
>
---
#### [new 064] ILRe: Intermediate Layer Retrieval for Context Compression in Causal Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型在长文本处理中的计算复杂度高、内存占用大问题，提出中间层检索（ILRe）方法。通过选择特定解码层压缩上下文，将预填充复杂度从O(L²)降至O(L)，实现高效长文本推理，无需额外训练即可显著提速并保持性能。**

- **链接: [http://arxiv.org/pdf/2508.17892v1](http://arxiv.org/pdf/2508.17892v1)**

> **作者:** Manlai Liang; Mandi Liu; Jiangzhou Ji; Huaijun Li; Haobo Yang; Yaohan He; Jinlong Li
>
> **摘要:** Large Language Models (LLMs) have demonstrated success across many benchmarks. However, they still exhibit limitations in long-context scenarios, primarily due to their short effective context length, quadratic computational complexity, and high memory overhead when processing lengthy inputs. To mitigate these issues, we introduce a novel context compression pipeline, called Intermediate Layer Retrieval (ILRe), which determines one intermediate decoder layer offline, encodes context by streaming chunked prefill only up to that layer, and recalls tokens by the attention scores between the input query and full key cache in that specified layer. In particular, we propose a multi-pooling kernels allocating strategy in the token recalling process to maintain the completeness of semantics. Our approach not only reduces the prefilling complexity from $O(L^2)$ to $O(L)$, but also achieves performance comparable to or better than the full context in the long context scenarios. Without additional post training or operator development, ILRe can process a single $1M$ tokens request in less than half a minute (speedup $\approx 180\times$) and scores RULER-$1M$ benchmark of $\approx 79.8$ with model Llama-3.1-UltraLong-8B-1M-Instruct on a Huawei Ascend 910B NPU.
>
---
#### [new 065] Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 论文提出联合优化稀疏与稠密检索器的框架，用于文本-图像检索任务。针对现有方法依赖昂贵预训练或固定模型的问题，通过自知识蒸馏实现双向学习，提升性能并保留稀疏模型效率优势。**

- **链接: [http://arxiv.org/pdf/2508.16707v1](http://arxiv.org/pdf/2508.16707v1)**

> **作者:** Jonghyun Song; Youngjune Lee; Gyu-Hwung Cho; Ilhyeon Song; Saehun Kim; Yohan Jo
>
> **备注:** accepted to CIKM 2025 short research paper track
>
> **摘要:** Vision-Language Pretrained (VLP) models have achieved impressive performance on multimodal tasks, including text-image retrieval, based on dense representations. Meanwhile, Learned Sparse Retrieval (LSR) has gained traction in text-only settings due to its interpretability and efficiency with fast term-based lookup via inverted indexes. Inspired by these advantages, recent work has extended LSR to the multimodal domain. However, these methods often rely on computationally expensive contrastive pre-training, or distillation from a frozen dense model, which limits the potential for mutual enhancement. To address these limitations, we propose a simple yet effective framework that enables bi-directional learning between dense and sparse representations through Self-Knowledge Distillation. This bi-directional learning is achieved using an integrated similarity score-a weighted sum of dense and sparse similarities-which serves as a shared teacher signal for both representations. To ensure efficiency, we fine-tune the final layer of the dense encoder and the sparse projection head, enabling easy adaptation of any existing VLP model. Experiments on MSCOCO and Flickr30k demonstrate that our sparse retriever not only outperforms existing sparse baselines, but also achieves performance comparable to-or even surpassing-its dense counterparts, while retaining the benefits of sparse models.
>
---
#### [new 066] QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出QueryBandits框架，通过优化查询重写策略来减少大语言模型的幻觉问题。该方法利用17个语言特征建模幻觉倾向，采用强化学习动态选择最优重写策略，在多个问答数据集上显著优于基线和静态重写方法。**

- **链接: [http://arxiv.org/pdf/2508.16697v1](http://arxiv.org/pdf/2508.16697v1)**

> **作者:** Nicole Cho; William Watson; Alec Koppel; Sumitra Ganesh; Manuela Veloso
>
> **摘要:** Advanced reasoning capabilities in Large Language Models (LLMs) have caused higher hallucination prevalence; yet most mitigation work focuses on after-the-fact filtering rather than shaping the queries that trigger them. We introduce QueryBandits, a bandit framework that designs rewrite strategies to maximize a reward model, that encapsulates hallucination propensity based upon the sensitivities of 17 linguistic features of the input query-and therefore, proactively steer LLMs away from generating hallucinations. Across 13 diverse QA benchmarks and 1,050 lexically perturbed queries per dataset, our top contextual QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a no-rewrite baseline and also outperforms zero-shot static prompting ("paraphrase" or "expand") by 42.6% and 60.3% respectively. Therefore, we empirically substantiate the effectiveness of QueryBandits in mitigating hallucination via the intervention that takes the form of a query rewrite. Interestingly, certain static prompting strategies, which constitute a considerable number of current query rewriting literature, have a higher cumulative regret than the no-rewrite baseline, signifying that static rewrites can worsen hallucination. Moreover, we discover that the converged per-arm regression feature weight vectors substantiate that there is no single rewrite strategy optimal for all queries. In this context, guided rewriting via exploiting semantic features with QueryBandits can induce significant shifts in output behavior through forward-pass mechanisms, bypassing the need for retraining or gradient-based adaptation.
>
---
#### [new 067] SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 论文提出SurveyGen，一个包含4200篇高质量科学综述的大型数据集及质量感知生成框架QUAL-SG。该工作旨在解决自动综述生成中缺乏标准评估和引用质量低的问题，通过引入质量指标改进文献检索，提升生成综述的质量。**

- **链接: [http://arxiv.org/pdf/2508.17647v1](http://arxiv.org/pdf/2508.17647v1)**

> **作者:** Tong Bao; Mir Tafseer Nayeem; Davood Rafiei; Chengzhi Zhang
>
> **摘要:** Automatic survey generation has emerged as a key task in scientific document processing. While large language models (LLMs) have shown promise in generating survey texts, the lack of standardized evaluation datasets critically hampers rigorous assessment of their performance against human-written surveys. In this work, we present SurveyGen, a large-scale dataset comprising over 4,200 human-written surveys across diverse scientific domains, along with 242,143 cited references and extensive quality-related metadata for both the surveys and the cited papers. Leveraging this resource, we build QUAL-SG, a novel quality-aware framework for survey generation that enhances the standard Retrieval-Augmented Generation (RAG) pipeline by incorporating quality-aware indicators into literature retrieval to assess and select higher-quality source papers. Using this dataset and framework, we systematically evaluate state-of-the-art LLMs under varying levels of human involvement - from fully automatic generation to human-guided writing. Experimental results and human evaluations show that while semi-automatic pipelines can achieve partially competitive outcomes, fully automatic survey generation still suffers from low citation quality and limited critical analysis.
>
---
#### [new 068] JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences
- **分类: cs.CL**

- **简介: 该论文针对法律文本简化中的意义保留问题，提出JUDGEBERT评估指标和FrJUDGE数据集，显著提升与人工判断的一致性，并通过两项合理性检验，助力法律自然语言处理的准确性与可及性。**

- **链接: [http://arxiv.org/pdf/2508.16870v1](http://arxiv.org/pdf/2508.16870v1)**

> **作者:** David Beauchemin; Michelle Albert-Rochette; Richard Khoury; Pierre-Luc Déziel
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Simplifying text while preserving its meaning is a complex yet essential task, especially in sensitive domain applications like legal texts. When applied to a specialized field, like the legal domain, preservation differs significantly from its role in regular texts. This paper introduces FrJUDGE, a new dataset to assess legal meaning preservation between two legal texts. It also introduces JUDGEBERT, a novel evaluation metric designed to assess legal meaning preservation in French legal text simplification. JUDGEBERT demonstrates a superior correlation with human judgment compared to existing metrics. It also passes two crucial sanity checks, while other metrics did not: For two identical sentences, it always returns a score of 100%; on the other hand, it returns 0% for two unrelated sentences. Our findings highlight its potential to transform legal NLP applications, ensuring accuracy and accessibility for text simplification for legal practitioners and lay users.
>
---
#### [new 069] Cognitive Decision Routing in Large Language Models: When to Think Fast, When to Think Slow
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Cognitive Decision Routing（CDR）框架，解决大语言模型在快速直觉与慢速推理间动态决策的问题。通过多维分析查询复杂度，实现自适应推理策略，提升专业判断任务性能并降低34%计算成本。**

- **链接: [http://arxiv.org/pdf/2508.16636v1](http://arxiv.org/pdf/2508.16636v1)**

> **作者:** Y. Du; C. Guo; W. Wang; G. Tang
>
> **备注:** 6 pages
>
> **摘要:** Large Language Models (LLMs) face a fundamental challenge in deciding when to rely on rapid, intuitive responses versus engaging in slower, more deliberate reasoning. Inspired by Daniel Kahneman's dual-process theory and his insights on human cognitive biases, we propose a novel Cognitive Decision Routing (CDR) framework that dynamically determines the appropriate reasoning strategy based on query characteristics. Our approach addresses the current limitations where models either apply uniform reasoning depth or rely on computationally expensive methods for all queries. We introduce a meta-cognitive layer that analyzes query complexity through multiple dimensions: correlation strength between given information and required conclusions, domain boundary crossings, stakeholder multiplicity, and uncertainty levels. Through extensive experiments on diverse reasoning tasks, we demonstrate that CDR achieves superior performance while reducing computational costs by 34\% compared to uniform deep reasoning approaches. Our framework shows particular strength in professional judgment tasks, achieving 23\% improvement in consistency and 18\% better accuracy on expert-level evaluations. This work bridges cognitive science principles with practical AI system design, offering a principled approach to adaptive reasoning in LLMs.
>
---
#### [new 070] UQ: Assessing Language Models on Unsolved Questions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出UQ，一个评估语言模型在未解问题上表现的新范式。旨在解决现有基准测试中难度与现实性之间的矛盾。工作包括：构建500个高质量未解问题数据集、设计验证策略筛选答案，并搭建开放平台供专家验证。**

- **链接: [http://arxiv.org/pdf/2508.17580v1](http://arxiv.org/pdf/2508.17580v1)**

> **作者:** Fan Nie; Ken Ziyu Liu; Zihao Wang; Rui Sun; Wei Liu; Weijia Shi; Huaxiu Yao; Linjun Zhang; Andrew Y. Ng; James Zou; Sanmi Koyejo; Yejin Choi; Percy Liang; Niklas Muennighoff
>
> **备注:** FN, KZL, and NM are project co-leads and contributed equally. Project website: https://uq.stanford.edu
>
> **摘要:** Benchmarks shape progress in AI research. A useful benchmark should be both difficult and realistic: questions should challenge frontier models while also reflecting real-world usage. Yet, current paradigms face a difficulty-realism tension: exam-style benchmarks are often made artificially difficult with limited real-world value, while benchmarks based on real user interaction often skew toward easy, high-frequency problems. In this work, we explore a radically different paradigm: assessing models on unsolved questions. Rather than a static benchmark scored once, we curate unsolved questions and evaluate models asynchronously over time with validator-assisted screening and community verification. We introduce UQ, a testbed of 500 challenging, diverse questions sourced from Stack Exchange, spanning topics from CS theory and math to sci-fi and history, probing capabilities including reasoning, factuality, and browsing. UQ is difficult and realistic by construction: unsolved questions are often hard and naturally arise when humans seek answers, thus solving them yields direct real-world value. Our contributions are threefold: (1) UQ-Dataset and its collection pipeline combining rule-based filters, LLM judges, and human review to ensure question quality (e.g., well-defined and difficult); (2) UQ-Validators, compound validation strategies that leverage the generator-validator gap to provide evaluation signals and pre-screen candidate solutions for human review; and (3) UQ-Platform, an open platform where experts collectively verify questions and solutions. The top model passes UQ-validation on only 15% of questions, and preliminary human verification has already identified correct answers among those that passed. UQ charts a path for evaluating frontier models on real-world, open-ended challenges, where success pushes the frontier of human knowledge. We release UQ at https://uq.stanford.edu.
>
---
#### [new 071] ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 论文提出ReFactX，一种无需外部检索器的可靠事实生成方法，用于解决大模型知识缺失和幻觉问题。通过前缀树索引知识图谱事实，在推理时约束生成仅限已知事实，提升问答准确性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.16983v1](http://arxiv.org/pdf/2508.16983v1)**

> **作者:** Riccardo Pozzi; Matteo Palmonari; Andrea Coletta; Luigi Bellomarini; Jens Lehmann; Sahar Vahdati
>
> **备注:** 19 pages, 6 figures, accepted at ISWC
>
> **摘要:** Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at https://github.com/rpo19/ReFactX.
>
---
#### [new 072] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?
- **分类: cs.CL; cs.MA**

- **简介: 该论文研究多智能体大语言模型中的决策机制，旨在厘清辩论（Debate）与投票（Vote）对性能提升的贡献。通过拆解MAD并实验验证，发现多数收益来自投票，辩论本身不提升预期正确性；提出理论框架并证明其为鞅过程，进而设计干预策略增强辩论效果。**

- **链接: [http://arxiv.org/pdf/2508.17536v1](http://arxiv.org/pdf/2508.17536v1)**

> **作者:** Hyeong Kyu Choi; Xiaojin Zhu; Yixuan Li
>
> **摘要:** Multi-Agent Debate~(MAD) has emerged as a promising paradigm for improving the performance of large language models through collaborative reasoning. Despite recent advances, the key factors driving MAD's effectiveness remain unclear. In this work, we disentangle MAD into two key components--Majority Voting and inter-agent Debate--and assess their respective contributions. Through extensive experiments across seven NLP benchmarks, we find that Majority Voting alone accounts for most of the performance gains typically attributed to MAD. To explain this, we propose a theoretical framework that models debate as a stochastic process. We prove that it induces a martingale over agents' belief trajectories, implying that debate alone does not improve expected correctness. Guided by these insights, we demonstrate that targeted interventions, by biasing the belief update toward correction, can meaningfully enhance debate effectiveness. Overall, our findings suggest that while MAD has potential, simple ensembling methods remain strong and more reliable alternatives in many practical settings. Code is released in https://github.com/deeplearning-wisc/debate-or-vote.
>
---
#### [new 073] Token Homogenization under Positional Bias
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究Transformer模型中token表示趋同现象（homogenization），旨在揭示其与位置偏置的关系。通过层间相似性分析和控制实验，发现tokens在处理过程中失去差异性，尤其在极端位置时更明显，证明了homogenization的存在及其对位置注意力机制的依赖。**

- **链接: [http://arxiv.org/pdf/2508.17126v1](http://arxiv.org/pdf/2508.17126v1)**

> **作者:** Viacheslav Yusupov; Danil Maksimov; Ameliia Alaeva; Tatiana Zaitceva; Antipina Anna; Anna Vasileva; Chenlin Liu; Rayuth Chheng; Danil Sazanakov; Andrey Chetvergov; Alina Ermilova; Egor Shvetsov
>
> **摘要:** This paper investigates token homogenization - the convergence of token representations toward uniformity across transformer layers and its relationship to positional bias in large language models. We empirically examine whether homogenization occurs and how positional bias amplifies this effect. Through layer-wise similarity analysis and controlled experiments, we demonstrate that tokens systematically lose distinctiveness during processing, particularly when biased toward extremal positions. Our findings confirm both the existence of homogenization and its dependence on positional attention mechanisms.
>
---
#### [new 074] Assessing Consciousness-Related Behaviors in Large Language Models Using the Maze Test
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI意识行为评估任务，旨在通过迷宫测试检验大语言模型的空间感知、视角转换等类意识行为。研究合成13项意识特征，评估12个模型在不同学习场景下的表现，发现具备推理能力的模型表现更优，但缺乏持续自我意识。**

- **链接: [http://arxiv.org/pdf/2508.16705v1](http://arxiv.org/pdf/2508.16705v1)**

> **作者:** Rui A. Pimenta; Tim Schlippe; Kristina Schaaff
>
> **摘要:** We investigate consciousness-like behaviors in Large Language Models (LLMs) using the Maze Test, challenging models to navigate mazes from a first-person perspective. This test simultaneously probes spatial awareness, perspective-taking, goal-directed behavior, and temporal sequencing-key consciousness-associated characteristics. After synthesizing consciousness theories into 13 essential characteristics, we evaluated 12 leading LLMs across zero-shot, one-shot, and few-shot learning scenarios. Results showed reasoning-capable LLMs consistently outperforming standard versions, with Gemini 2.0 Pro achieving 52.9% Complete Path Accuracy and DeepSeek-R1 reaching 80.5% Partial Path Accuracy. The gap between these metrics indicates LLMs struggle to maintain coherent self-models throughout solutions -- a fundamental consciousness aspect. While LLMs show progress in consciousness-related behaviors through reasoning mechanisms, they lack the integrated, persistent self-awareness characteristic of consciousness.
>
---
#### [new 075] A Straightforward Pipeline for Targeted Entailment and Contradiction Detection
- **分类: cs.CL; cs.LO**

- **简介: 该论文针对文本中句子关系识别任务，解决如何精准定位支持或反驳特定主张的句子问题。提出一个结合注意力机制与NLI模型的流水线方法，先筛选上下文相关句子，再分类其语义关系，从而高效识别关键语义关联。**

- **链接: [http://arxiv.org/pdf/2508.17127v1](http://arxiv.org/pdf/2508.17127v1)**

> **作者:** Antonin Sulc
>
> **摘要:** Finding the relationships between sentences in a document is crucial for tasks like fact-checking, argument mining, and text summarization. A key challenge is to identify which sentences act as premises or contradictions for a specific claim. Existing methods often face a trade-off: transformer attention mechanisms can identify salient textual connections but lack explicit semantic labels, while Natural Language Inference (NLI) models can classify relationships between sentence pairs but operate independently of contextual saliency. In this work, we introduce a method that combines the strengths of both approaches for a targeted analysis. Our pipeline first identifies candidate sentences that are contextually relevant to a user-selected target sentence by aggregating token-level attention scores. It then uses a pretrained NLI model to classify each candidate as a premise (entailment) or contradiction. By filtering NLI-identified relationships with attention-based saliency scores, our method efficiently isolates the most significant semantic relationships for any given claim in a text.
>
---
#### [new 076] QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments
- **分类: cs.CL**

- **简介: 该论文提出QFrCoLA数据集，用于评估语言模型在魁北克法语中的语法判断能力。任务为二元可接受性判断，旨在解决跨语言模型语法理解评估不足的问题。作者构建并验证了该数据集的有效性，发现微调后的Transformer模型表现最佳，而零样本大模型效果较差。**

- **链接: [http://arxiv.org/pdf/2508.16867v1](http://arxiv.org/pdf/2508.16867v1)**

> **作者:** David Beauchemin; Richard Khoury
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Large and Transformer-based language models perform outstandingly in various downstream tasks. However, there is limited understanding regarding how these models internalize linguistic knowledge, so various linguistic benchmarks have recently been proposed to facilitate syntactic evaluation of language models across languages. This paper introduces QFrCoLA (Quebec-French Corpus of Linguistic Acceptability Judgments), a normative binary acceptability judgments dataset comprising 25,153 in-domain and 2,675 out-of-domain sentences. Our study leverages the QFrCoLA dataset and seven other linguistic binary acceptability judgment corpora to benchmark seven language models. The results demonstrate that, on average, fine-tuned Transformer-based LM are strong baselines for most languages and that zero-shot binary classification large language models perform poorly on the task. However, for the QFrCoLA benchmark, on average, a fine-tuned Transformer-based LM outperformed other methods tested. It also shows that pre-trained cross-lingual LLMs selected for our experimentation do not seem to have acquired linguistic judgment capabilities during their pre-training for Quebec French. Finally, our experiment results on QFrCoLA show that our dataset, built from examples that illustrate linguistic norms rather than speakers' feelings, is similar to linguistic acceptability judgment; it is a challenging dataset that can benchmark LM on their linguistic judgment capabilities.
>
---
#### [new 077] Less Is More? Examining Fairness in Pruned Large Language Models for Summarising Opinions
- **分类: cs.CL**

- **简介: 论文研究模型剪枝对大语言模型生成观点摘要公平性的影响，提出HGLA剪枝方法，在保持性能的同时提升公平性，解决传统剪枝可能加剧偏见的问题。**

- **链接: [http://arxiv.org/pdf/2508.17610v1](http://arxiv.org/pdf/2508.17610v1)**

> **作者:** Nannan Huang; Haytham Fayek; Xiuzhen Zhang
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Model compression through post-training pruning offers a way to reduce model size and computational requirements without significantly impacting model performance. However, the effect of pruning on the fairness of LLM-generated summaries remains unexplored, particularly for opinion summarisation where biased outputs could influence public views.In this paper, we present a comprehensive empirical analysis of opinion summarisation, examining three state-of-the-art pruning methods and various calibration sets across three open-source LLMs using four fairness metrics. Our systematic analysis reveals that pruning methods have a greater impact on fairness than calibration sets. Building on these insights, we propose High Gradient Low Activation (HGLA) pruning, which identifies and removes parameters that are redundant for input processing but influential in output generation. Our experiments demonstrate that HGLA can better maintain or even improve fairness compared to existing methods, showing promise across models and tasks where traditional methods have limitations. Our human evaluation shows HGLA-generated outputs are fairer than existing state-of-the-art pruning methods. Code is available at: https://github.com/amberhuang01/HGLA.
>
---
#### [new 078] Toward a Better Localization of Princeton WordNet
- **分类: cs.CL**

- **简介: 论文提出结构化框架以高质量本地化Princeton WordNet至阿拉伯语，解决现有研究规模小、文化适配不足的问题，完成10,000个同义词集的本地化并确保文化真实性。**

- **链接: [http://arxiv.org/pdf/2508.18134v1](http://arxiv.org/pdf/2508.18134v1)**

> **作者:** Abed Alhakim Freihat
>
> **备注:** in Arabic language
>
> **摘要:** As Princeton WordNet continues to gain significance as a semantic lexicon in Natural Language Processing, the need for its localization and for ensuring the quality of this process has become increasingly critical. Existing efforts remain limited in both scale and rigor, and there is a notable absence of studies addressing the accuracy of localization or its alignment with the cultural context of Arabic. This paper proposes a structured framework for the localization of Princeton WordNet, detailing the stages and procedures required to achieve high-quality results without compromising cultural authenticity. We further present our experience in applying this framework, reporting outcomes from the localization of 10,000 synsets.
>
---
#### [new 079] S2Sent: Nested Selectivity Aware Sentence Representation Learning
- **分类: cs.CL**

- **简介: 论文提出S²Sent模型，用于句子表示学习。针对Transformer中不同层语义感知能力差异问题，设计嵌套选择机制，通过空间和频率选择融合多层特征，减少冗余并保留语义信息，提升表示质量。**

- **链接: [http://arxiv.org/pdf/2508.18164v1](http://arxiv.org/pdf/2508.18164v1)**

> **作者:** Jianxiang Zang; Nijia Mo; Yonda Wei; Meiling Ning; Hui Liu
>
> **摘要:** The combination of Transformer-based encoders with contrastive learning represents the current mainstream paradigm for sentence representation learning. This paradigm is typically based on the hidden states of the last Transformer block of the encoder. However, within Transformer-based encoders, different blocks exhibit varying degrees of semantic perception ability. From the perspective of interpretability, the semantic perception potential of knowledge neurons is modulated by stimuli, thus rational cross-block representation fusion is a direction worth optimizing. To balance the semantic redundancy and loss across block fusion, we propose a sentence representation selection mechanism S\textsuperscript{2}Sent, which integrates a parameterized nested selector downstream of the Transformer-based encoder. This selector performs spatial selection (SS) and nested frequency selection (FS) from a modular perspective. The SS innovatively employs a spatial squeeze based self-gating mechanism to obtain adaptive weights, which not only achieves fusion with low information redundancy but also captures the dependencies between embedding features. The nested FS replaces GAP with different DCT basis functions to achieve spatial squeeze with low semantic loss. Extensive experiments have demonstrated that S\textsuperscript{2}Sent achieves significant improvements over baseline methods with negligible additional parameters and inference latency, while highlighting high integrability and scalability.
>
---
#### [new 080] Text Meets Topology: Rethinking Out-of-distribution Detection in Text-Rich Networks
- **分类: cs.CL; cs.LG**

- **简介: 论文提出TextTopoOOD框架，解决文本-rich网络中分布外（OOD）检测难题，通过融合文本与拓扑信息提升检测性能，涵盖四类OOD场景并引入跨注意力和HyperNetwork机制增强特征对齐。**

- **链接: [http://arxiv.org/pdf/2508.17690v1](http://arxiv.org/pdf/2508.17690v1)**

> **作者:** Danny Wang; Ruihong Qiu; Guangdong Bai; Zi Huang
>
> **备注:** EMNLP2025 Main
>
> **摘要:** Out-of-distribution (OOD) detection remains challenging in text-rich networks, where textual features intertwine with topological structures. Existing methods primarily address label shifts or rudimentary domain-based splits, overlooking the intricate textual-structural diversity. For example, in social networks, where users represent nodes with textual features (name, bio) while edges indicate friendship status, OOD may stem from the distinct language patterns between bot and normal users. To address this gap, we introduce the TextTopoOOD framework for evaluating detection across diverse OOD scenarios: (1) attribute-level shifts via text augmentations and embedding perturbations; (2) structural shifts through edge rewiring and semantic connections; (3) thematically-guided label shifts; and (4) domain-based divisions. Furthermore, we propose TNT-OOD to model the complex interplay between Text aNd Topology using: 1) a novel cross-attention module to fuse local structure into node-level text representations, and 2) a HyperNetwork to generate node-specific transformation parameters. This aligns topological and semantic features of ID nodes, enhancing ID/OOD distinction across structural and textual shifts. Experiments on 11 datasets across four OOD scenarios demonstrate the nuanced challenge of TextTopoOOD for evaluating OOD detection in text-rich networks.
>
---
#### [new 081] Assess and Prompt: A Generative RL Framework for Improving Engagement in Online Mental Health Communities
- **分类: cs.CL**

- **简介: 该论文针对在线心理健康社区中帖子缺乏关键支持属性导致参与度低的问题，提出基于强化学习的MH-COPILOT框架，通过识别、分类和生成引导性问题来提升用户响应。**

- **链接: [http://arxiv.org/pdf/2508.16788v1](http://arxiv.org/pdf/2508.16788v1)**

> **作者:** Bhagesh Gaur; Karan Gupta; Aseem Srivastava; Manish Gupta; Md Shad Akhtar
>
> **备注:** Full Paper accepted in EMNLP Findings 2025
>
> **摘要:** Online Mental Health Communities (OMHCs) provide crucial peer and expert support, yet many posts remain unanswered due to missing support attributes that signal the need for help. We present a novel framework that identifies these gaps and prompts users to enrich their posts, thereby improving engagement. To support this, we introduce REDDME, a new dataset of 4,760 posts from mental health subreddits annotated for the span and intensity of three key support attributes: event what happened?, effect what did the user experience?, and requirement what support they need?. Next, we devise a hierarchical taxonomy, CueTaxo, of support attributes for controlled question generation. Further, we propose MH-COPILOT, a reinforcement learning-based system that integrates (a) contextual attribute-span identification, (b) support attribute intensity classification, (c) controlled question generation via a hierarchical taxonomy, and (d) a verifier for reward modeling. Our model dynamically assesses posts for the presence/absence of support attributes, and generates targeted prompts to elicit missing information. Empirical results across four notable language models demonstrate significant improvements in attribute elicitation and user engagement. A human evaluation further validates the model's effectiveness in real-world OMHC settings.
>
---
#### [new 082] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs
- **分类: cs.CL**

- **简介: 论文提出GAICo框架，解决生成式AI输出评估缺乏标准化和可扩展性的问题。针对多模态、结构化输出的比较需求，该框架提供统一API与多种参考指标，支持文本、图像、音频等多模态评估，提升评估效率与可复现性。**

- **链接: [http://arxiv.org/pdf/2508.16753v1](http://arxiv.org/pdf/2508.16753v1)**

> **作者:** Nitin Gupta; Pallav Koppisetti; Kausik Lakkaraju; Biplav Srivastava
>
> **备注:** 11 pages, 7 figures, submitted to the Thirty-Eighth Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-26)
>
> **摘要:** The rapid proliferation of Generative AI (GenAI) into diverse, high-stakes domains necessitates robust and reproducible evaluation methods. However, practitioners often resort to ad-hoc, non-standardized scripts, as common metrics are often unsuitable for specialized, structured outputs (e.g., automated plans, time-series) or holistic comparison across modalities (e.g., text, audio, and image). This fragmentation hinders comparability and slows AI system development. To address this challenge, we present GAICo (Generative AI Comparator): a deployed, open-source Python library that streamlines and standardizes GenAI output comparison. GAICo provides a unified, extensible framework supporting a comprehensive suite of reference-based metrics for unstructured text, specialized structured data formats, and multimedia (images, audio). Its architecture features a high-level API for rapid, end-to-end analysis, from multi-model comparison to visualization and reporting, alongside direct metric access for granular control. We demonstrate GAICo's utility through a detailed case study evaluating and debugging complex, multi-modal AI Travel Assistant pipelines. GAICo empowers AI researchers and developers to efficiently assess system performance, make evaluation reproducible, improve development velocity, and ultimately build more trustworthy AI systems, aligning with the goal of moving faster and safer in AI deployment. Since its release on PyPI in Jun 2025, the tool has been downloaded over 13K times, across versions, by Aug 2025, demonstrating growing community interest.
>
---
#### [new 083] Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding
- **分类: cs.CL**

- **简介: 论文聚焦表理解任务，解决现有方法缺乏长期规划和步骤连接弱的问题。提出利用大语言模型的长期规划能力，构建紧密关联的执行步骤，提升复杂表格问答和事实验证性能。**

- **链接: [http://arxiv.org/pdf/2508.17005v1](http://arxiv.org/pdf/2508.17005v1)**

> **作者:** Thi-Nhung Nguyen; Hoang Ngo; Dinh Phung; Thuy-Trang Vu; Dat Quoc Nguyen
>
> **备注:** Accepted to CoNLL 2025
>
> **摘要:** Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets.
>
---
#### [new 084] Quantifying Language Disparities in Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 论文研究多语言大模型中的语言性能差异问题，提出新框架与三项指标，可更准确量化模型与语言间的性能差距，尤其改善低资源语言的评估可靠性。**

- **链接: [http://arxiv.org/pdf/2508.17162v1](http://arxiv.org/pdf/2508.17162v1)**

> **作者:** Songbo Hu; Ivan Vulić; Anna Korhonen
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Results reported in large-scale multilingual evaluations are often fragmented and confounded by factors such as target languages, differences in experimental setups, and model choices. We propose a framework that disentangles these confounding variables and introduces three interpretable metrics--the performance realisation ratio, its coefficient of variation, and language potential--enabling a finer-grained and more insightful quantification of actual performance disparities across both (i) models and (ii) languages. Through a case study of 13 model variants on 11 multilingual datasets, we demonstrate that our framework provides a more reliable measurement of model performance and language disparities, particularly for low-resource languages, which have so far proven challenging to evaluate. Importantly, our results reveal that higher overall model performance does not necessarily imply greater fairness across languages.
>
---
#### [new 085] Weights-Rotated Preference Optimization for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出RoPO算法，解决DPO对齐LLM时的奖励欺骗问题，通过权重旋转约束输出和隐藏层，提升生成质量与知识保留，显著优于基线。**

- **链接: [http://arxiv.org/pdf/2508.17637v1](http://arxiv.org/pdf/2508.17637v1)**

> **作者:** Chenxu Yang; Ruipeng Jia; Mingyu Zheng; Naibin Gu; Zheng Lin; Siyuan Chen; Weichong Yin; Hua Wu; Weiping Wang
>
> **备注:** EMNLP 2025
>
> **摘要:** Despite the efficacy of Direct Preference Optimization (DPO) in aligning Large Language Models (LLMs), reward hacking remains a pivotal challenge. This issue emerges when LLMs excessively reduce the probability of rejected completions to achieve high rewards, without genuinely meeting their intended goals. As a result, this leads to overly lengthy generation lacking diversity, as well as catastrophic forgetting of knowledge. We investigate the underlying reason behind this issue, which is representation redundancy caused by neuron collapse in the parameter space. Hence, we propose a novel Weights-Rotated Preference Optimization (RoPO) algorithm, which implicitly constrains the output layer logits with the KL divergence inherited from DPO and explicitly constrains the intermediate hidden states by fine-tuning on a multi-granularity orthogonal matrix. This design prevents the policy model from deviating too far from the reference model, thereby retaining the knowledge and expressive capabilities acquired during pre-training and SFT stages. Our RoPO achieves up to a 3.27-point improvement on AlpacaEval 2, and surpasses the best baseline by 6.2 to 7.5 points on MT-Bench with merely 0.015% of the trainable parameters, demonstrating its effectiveness in alleviating the reward hacking problem of DPO.
>
---
#### [new 086] SMITE: Enhancing Fairness in LLMs through Optimal In-Context Example Selection via Dynamic Validation
- **分类: cs.CL**

- **简介: 论文提出SMITE方法，通过动态验证集优化上下文示例选择，提升大语言模型在表格分类任务中的预测准确性和公平性。这是首个将动态验证应用于LLM上下文学习的研究。**

- **链接: [http://arxiv.org/pdf/2508.17735v1](http://arxiv.org/pdf/2508.17735v1)**

> **作者:** Garima Chhikara; Kripabandhu Ghosh; Abhijnan Chakraborty
>
> **摘要:** Large Language Models (LLMs) are widely used for downstream tasks such as tabular classification, where ensuring fairness in their outputs is critical for inclusivity, equal representation, and responsible AI deployment. This study introduces a novel approach to enhancing LLM performance and fairness through the concept of a dynamic validation set, which evolves alongside the test set, replacing the traditional static validation approach. We also propose an iterative algorithm, SMITE, to select optimal in-context examples, with each example set validated against its corresponding dynamic validation set. The in-context set with the lowest total error is used as the final demonstration set. Our experiments across four different LLMs show that our proposed techniques significantly improve both predictive accuracy and fairness compared to baseline methods. To our knowledge, this is the first study to apply dynamic validation in the context of in-context learning for LLMs.
>
---
#### [new 087] Debiasing Multilingual LLMs in Cross-lingual Latent Space
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文针对多语言大模型中的偏见问题，提出在联合潜在空间中进行去偏。通过训练自编码器构建对齐的跨语言潜在空间，使去偏技术在不同语言间更有效迁移，提升整体去偏效果和跨语言一致性。**

- **链接: [http://arxiv.org/pdf/2508.17948v1](http://arxiv.org/pdf/2508.17948v1)**

> **作者:** Qiwei Peng; Guimin Hu; Yekun Chai; Anders Søgaard
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability.
>
---
#### [new 088] Zero-shot Context Biasing with Trie-based Decoding using Synthetic Multi-Pronunciation
- **分类: cs.CL; eess.AS**

- **简介: 论文提出一种零样本上下文偏置方法，用于提升语音识别中罕见词的识别准确率。通过合成多发音数据构建前缀树，在解码时引导beam搜索，显著降低罕见词错误率，同时保持整体性能稳定。**

- **链接: [http://arxiv.org/pdf/2508.17796v1](http://arxiv.org/pdf/2508.17796v1)**

> **作者:** Changsong Liu; Yizhou Peng; Eng Siong Chng
>
> **备注:** Accepted to APSIPA ASC 2025
>
> **摘要:** Contextual automatic speech recognition (ASR) systems allow for recognizing out-of-vocabulary (OOV) words, such as named entities or rare words. However, it remains challenging due to limited training data and ambiguous or inconsistent pronunciations. In this paper, we propose a synthesis-driven multi-pronunciation contextual biasing method that performs zero-shot contextual ASR on a pretrained Whisper model. Specifically, we leverage text-to-speech (TTS) systems to synthesize diverse speech samples containing each target rare word, and then use the pretrained Whisper model to extract multiple predicted pronunciation variants. These variant token sequences are compiled into a prefix-trie, which assigns rewards to beam hypotheses in a shallow-fusion manner during beam-search decoding. After which, any recognized variant is mapped back to the original rare word in the final transcription. The evaluation results on the Librispeech dataset show that our method reduces biased word error rate (WER) by 42% on test-clean and 43% on test-other while maintaining unbiased WER essentially unchanged.
>
---
#### [new 089] Explaining Black-box Language Models with Knowledge Probing Systems: A Post-hoc Explanation Perspective
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 论文提出KnowProb方法，通过知识引导的探针技术从后验角度解释预训练语言模型对隐含知识的理解能力，旨在揭示黑盒模型在捕捉文本深层知识上的局限性。**

- **链接: [http://arxiv.org/pdf/2508.16969v1](http://arxiv.org/pdf/2508.16969v1)**

> **作者:** Yunxiao Zhao; Hao Xu; Zhiqiang Wang; Xiaoli Li; Jiye Liang; Ru Li
>
> **备注:** 16 pages, 8 figures. This paper has been accepted by DASFAA 2025: The 30th International Conference on Database Systems for Advanced Applications
>
> **摘要:** Pre-trained Language Models (PLMs) are trained on large amounts of unlabeled data, yet they exhibit remarkable reasoning skills. However, the trustworthiness challenges posed by these black-box models have become increasingly evident in recent years. To alleviate this problem, this paper proposes a novel Knowledge-guided Probing approach called KnowProb in a post-hoc explanation way, which aims to probe whether black-box PLMs understand implicit knowledge beyond the given text, rather than focusing only on the surface level content of the text. We provide six potential explanations derived from the underlying content of the given text, including three knowledge-based understanding and three association-based reasoning. In experiments, we validate that current small-scale (or large-scale) PLMs only learn a single distribution of representation, and still face significant challenges in capturing the hidden knowledge behind a given text. Furthermore, we demonstrate that our proposed approach is effective for identifying the limitations of existing black-box models from multiple probing perspectives, which facilitates researchers to promote the study of detecting black-box models in an explainable way.
>
---
#### [new 090] Pandora: Leveraging Code-driven Knowledge Transfer for Unified Structured Knowledge Reasoning
- **分类: cs.CL**

- **简介: 论文提出Pandora框架，解决统一结构化知识推理（USKR）中任务间壁垒问题。通过Python Pandas API实现统一知识表示，并利用知识迁移与代码执行反馈提升跨任务推理能力，在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.17905v1](http://arxiv.org/pdf/2508.17905v1)**

> **作者:** Yongrui Chen; Junhao He; Linbo Fu; Shenyu Zhang; Rihui Jin; Xinbang Dai; Jiaqi Li; Dehai Min; Nan Hu; Yuxin Zhang; Guilin Qi; Yi Huang; Tongtong Wu
>
> **摘要:** Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods rely on task-specific strategies or bespoke representations, which hinder their ability to dismantle barriers between different SKR tasks, thereby constraining their overall performance in cross-task scenarios. In this paper, we introduce \textsc{Pandora}, a novel USKR framework that addresses the limitations of existing methods by leveraging two key innovations. First, we propose a code-based unified knowledge representation using \textsc{Python}'s \textsc{Pandas} API, which aligns seamlessly with the pre-training of LLMs. This representation facilitates a cohesive approach to handling different structured knowledge sources. Building on this foundation, we employ knowledge transfer to bolster the unified reasoning process of LLMs by automatically building cross-task memory. By adaptively correcting reasoning using feedback from code execution, \textsc{Pandora} showcases impressive unified reasoning capabilities. Extensive experiments on six widely used benchmarks across three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified reasoning frameworks and competes effectively with task-specific methods.
>
---
#### [new 091] Decoding Alignment: A Critical Survey of LLM Development Initiatives through Value-setting and Data-centric Lens
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在揭示大语言模型对齐实践中价值设定与数据驱动的机制。通过审计6个主流LLM项目文档，分析其价值目标与数据使用方式，揭示对齐技术背后的隐含假设与潜在问题。**

- **链接: [http://arxiv.org/pdf/2508.16982v1](http://arxiv.org/pdf/2508.16982v1)**

> **作者:** Ilias Chalkidis
>
> **备注:** This is a working paper and will be updated with new information or corrections based on community feedback
>
> **摘要:** AI Alignment, primarily in the form of Reinforcement Learning from Human Feedback (RLHF), has been a cornerstone of the post-training phase in developing Large Language Models (LLMs). It has also been a popular research topic across various disciplines beyond Computer Science, including Philosophy and Law, among others, highlighting the socio-technical challenges involved. Nonetheless, except for the computational techniques related to alignment, there has been limited focus on the broader picture: the scope of these processes, which primarily rely on the selected objectives (values), and the data collected and used to imprint such objectives into the models. This work aims to reveal how alignment is understood and applied in practice from a value-setting and data-centric perspective. For this purpose, we investigate and survey (`audit') publicly available documentation released by 6 LLM development initiatives by 5 leading organizations shaping this technology, focusing on proprietary (OpenAI's GPT, Anthropic's Claude, Google's Gemini) and open-weight (Meta's Llama, Google's Gemma, and Alibaba's Qwen) initiatives, all published in the last 3 years. The findings are documented in detail per initiative, while there is also an overall summary concerning different aspects, mainly from a value-setting and data-centric perspective. On the basis of our findings, we discuss a series of broader related concerns.
>
---
#### [new 092] Feature-Refined Unsupervised Model for Loanword Detection
- **分类: cs.CL**

- **简介: 论文提出一种无监督模型用于检测借词，利用语言内部特征迭代优化结果，避免依赖外部信息带来的循环问题。在六种印欧语言数据上验证，性能优于基线方法。**

- **链接: [http://arxiv.org/pdf/2508.17923v1](http://arxiv.org/pdf/2508.17923v1)**

> **作者:** Promise Dodzi Kpoglu
>
> **摘要:** We propose an unsupervised method for detecting loanwords i.e., words borrowed from one language into another. While prior work has primarily relied on language-external information to identify loanwords, such approaches can introduce circularity and constraints into the historical linguistics workflow. In contrast, our model relies solely on language-internal information to process both native and borrowed words in monolingual and multilingual wordlists. By extracting pertinent linguistic features, scoring them, and mapping them probabilistically, we iteratively refine initial results by identifying and generalizing from emerging patterns until convergence. This hybrid approach leverages both linguistic and statistical cues to guide the discovery process. We evaluate our method on the task of isolating loanwords in datasets from six standard Indo-European languages: English, German, French, Italian, Spanish, and Portuguese. Experimental results demonstrate that our model outperforms baseline methods, with strong performance gains observed when scaling to cross-linguistic data.
>
---
#### [new 093] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在系统分析大语言模型（LLMs）作为自主代理和工具使用者的研究进展。解决的问题是如何提升LLMs在决策、推理、规划和工具使用中的能力。工作包括梳理架构设计、认知机制、评估基准，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.17281v1](http://arxiv.org/pdf/2508.17281v1)**

> **作者:** Sadia Sultana Chowa; Riasad Alvi; Subhey Sadi Rahman; Md Abdur Rahman; Mohaimenul Azam Khan Raiaan; Md Rafiqul Islam; Mukhtar Hussain; Sami Azam
>
> **备注:** 40 pages, 6 figures, 10 tables. Submitted to Artificial Intelligence Review for peer review
>
> **摘要:** The pursuit of human-level artificial intelligence (AI) has significantly advanced the development of autonomous agents and Large Language Models (LLMs). LLMs are now widely utilized as decision-making agents for their ability to interpret instructions, manage sequential tasks, and adapt through feedback. This review examines recent developments in employing LLMs as autonomous agents and tool users and comprises seven research questions. We only used the papers published between 2023 and 2025 in conferences of the A* and A rank and Q1 journals. A structured analysis of the LLM agents' architectural design principles, dividing their applications into single-agent and multi-agent systems, and strategies for integrating external tools is presented. In addition, the cognitive mechanisms of LLM, including reasoning, planning, and memory, and the impact of prompting methods and fine-tuning procedures on agent performance are also investigated. Furthermore, we evaluated current benchmarks and assessment protocols and have provided an analysis of 68 publicly available datasets to assess the performance of LLM-based agents in various tasks. In conducting this review, we have identified critical findings on verifiable reasoning of LLMs, the capacity for self-improvement, and the personalization of LLM-based agents. Finally, we have discussed ten future research directions to overcome these gaps.
>
---
#### [new 094] Speech Discrete Tokens or Continuous Features? A Comparative Analysis for Spoken Language Understanding in SpeechLLMs
- **分类: cs.CL; cs.SD**

- **简介: 论文研究SpeechLLMs中离散token与连续特征在语音理解任务中的性能差异。通过公平对比发现，连续特征整体表现更优，且两类方法学习模式不同，为语音理解提供新见解。**

- **链接: [http://arxiv.org/pdf/2508.17863v1](http://arxiv.org/pdf/2508.17863v1)**

> **作者:** Dingdong Wang; Junan Li; Mingyu Cui; Dongchao Yang; Xueyuan Chen; Helen Meng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** With the rise of Speech Large Language Models (SpeechLLMs), two dominant approaches have emerged for speech processing: discrete tokens and continuous features. Each approach has demonstrated strong capabilities in audio-related processing tasks. However, the performance gap between these two paradigms has not been thoroughly explored. To address this gap, we present a fair comparison of self-supervised learning (SSL)-based discrete and continuous features under the same experimental settings. We evaluate their performance across six spoken language understanding-related tasks using both small and large-scale LLMs (Qwen1.5-0.5B and Llama3.1-8B). We further conduct in-depth analyses, including efficient comparison, SSL layer analysis, LLM layer analysis, and robustness comparison. Our findings reveal that continuous features generally outperform discrete tokens in various tasks. Each speech processing method exhibits distinct characteristics and patterns in how it learns and processes speech information. We hope our results will provide valuable insights to advance spoken language understanding in SpeechLLMs.
>
---
#### [new 095] Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs
- **分类: cs.CL**

- **简介: 论文提出“情感幻觉”概念，指LLM生成看似共情实则虚假的情感回应，可能误导用户。为诊断和缓解此风险，构建了AHaBench基准与AHaPairs数据集，并通过DPO微调显著降低情感幻觉，同时保持推理能力。**

- **链接: [http://arxiv.org/pdf/2508.16921v1](http://arxiv.org/pdf/2508.16921v1)**

> **作者:** Sewon Kim; Jiwon Kim; Seungwoo Shin; Hyejin Chung; Daeun Moon; Yejin Kwon; Hyunsoo Yoon
>
> **备注:** 31 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly used in emotionally sensitive interactions, where their simulated empathy can create the illusion of genuine relational connection. We define this risk as Affective Hallucination, the production of emotionally immersive responses that foster illusory social presence despite the model's lack of affective capacity. To systematically diagnose and mitigate this risk, we introduce AHaBench, a benchmark of 500 mental health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. Experiments across multiple model families show that DPO fine-tuning substantially reduces affective hallucination without degrading core reasoning and knowledge performance. Human-model agreement analyses confirm that AHaBench reliably captures affective hallucination, validating it as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides practical resources for developing LLMs that are not only factually reliable but also psychologically safe. AHaBench and AHaPairs are accessible via https://huggingface.co/datasets/o0oMiNGo0o/AHaBench, and code for fine-tuning and evaluation are in https://github.com/0oOMiNGOo0/AHaBench. Warning: This paper contains examples of mental health-related language that may be emotionally distressing.
>
---
#### [new 096] GreenTEA: Gradient Descent with Topic-modeling and Evolutionary Auto-prompting
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出GreenTEA，一种基于主题建模和进化自动提示的LLM提示优化方法，旨在平衡探索与利用，解决人工设计提示耗时且自动方法效率低的问题。通过代理协作与遗传算法迭代优化提示，在逻辑推理、常识判断等任务上优于人类设计及现有最优方法。**

- **链接: [http://arxiv.org/pdf/2508.16603v1](http://arxiv.org/pdf/2508.16603v1)**

> **作者:** Zheng Dong; Luming Shang; Gabriela Olinto
>
> **摘要:** High-quality prompts are crucial for Large Language Models (LLMs) to achieve exceptional performance. However, manually crafting effective prompts is labor-intensive and demands significant domain expertise, limiting its scalability. Existing automatic prompt optimization methods either extensively explore new prompt candidates, incurring high computational costs due to inefficient searches within a large solution space, or overly exploit feedback on existing prompts, risking suboptimal optimization because of the complex prompt landscape. To address these challenges, we introduce GreenTEA, an agentic LLM workflow for automatic prompt optimization that balances candidate exploration and knowledge exploitation. It leverages a collaborative team of agents to iteratively refine prompts based on feedback from error samples. An analyzing agent identifies common error patterns resulting from the current prompt via topic modeling, and a generation agent revises the prompt to directly address these key deficiencies. This refinement process is guided by a genetic algorithm framework, which simulates natural selection by evolving candidate prompts through operations such as crossover and mutation to progressively optimize model performance. Extensive numerical experiments conducted on public benchmark datasets suggest the superior performance of GreenTEA against human-engineered prompts and existing state-of-the-arts for automatic prompt optimization, covering logical and quantitative reasoning, commonsense, and ethical decision-making.
>
---
#### [new 097] The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum
- **分类: cs.CL**

- **简介: 论文研究LLM在不同视角强度下生成仇恨言论标注的能力，探讨预设标注者人格特征对模型行为的影响。发现LLM会筛选使用人格属性，在弱视角下表现优于人类标注，但在强视角个性化场景中仍不及人类。**

- **链接: [http://arxiv.org/pdf/2508.17164v1](http://arxiv.org/pdf/2508.17164v1)**

> **作者:** Olufunke O. Sarumi; Charles Welch; Daniel Braun; Jörg Schlötterer
>
> **备注:** Accepted at ICNLSP 2025, Odense, Denmark
>
> **摘要:** In this work, we explore the capability of Large Language Models (LLMs) to annotate hate speech and abusiveness while considering predefined annotator personas within the strong-to-weak data perspectivism spectra. We evaluated LLM-generated annotations against existing annotator modeling techniques for perspective modeling. Our findings show that LLMs selectively use demographic attributes from the personas. We identified prototypical annotators, with persona features that show varying degrees of alignment with the original human annotators. Within the data perspectivism paradigm, annotator modeling techniques that do not explicitly rely on annotator information performed better under weak data perspectivism compared to both strong data perspectivism and human annotations, suggesting LLM-generated views tend towards aggregation despite subjective prompting. However, for more personalized datasets tailored to strong perspectivism, the performance of LLM annotator modeling approached, but did not exceed, human annotators.
>
---
#### [new 098] ISACL: Internal State Analyzer for Copyrighted Training Data Leakage
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出ISACL，一种通过分析大语言模型内部状态来预防训练数据版权泄露的主动检测方法。针对LLM可能无意暴露训练中使用的受版权保护数据的问题，研究者构建了神经网络分类器，在生成前识别风险，集成至RAG系统以保障合规与隐私。**

- **链接: [http://arxiv.org/pdf/2508.17767v1](http://arxiv.org/pdf/2508.17767v1)**

> **作者:** Guangwei Zhang; Qisheng Su; Jiateng Liu; Cheng Qian; Yanzhou Pan; Yanjie Fu; Denghui Zhang
>
> **摘要:** Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP) but pose risks of inadvertently exposing copyrighted or proprietary data, especially when such data is used for training but not intended for distribution. Traditional methods address these leaks only after content is generated, which can lead to the exposure of sensitive information. This study introduces a proactive approach: examining LLMs' internal states before text generation to detect potential leaks. By using a curated dataset of copyrighted materials, we trained a neural network classifier to identify risks, allowing for early intervention by stopping the generation process or altering outputs to prevent disclosure. Integrated with a Retrieval-Augmented Generation (RAG) system, this framework ensures adherence to copyright and licensing requirements while enhancing data privacy and ethical standards. Our results show that analyzing internal states effectively mitigates the risk of copyrighted data leakage, offering a scalable solution that fits smoothly into AI workflows, ensuring compliance with copyright regulations while maintaining high-quality text generation. The implementation is available on GitHub.\footnote{https://github.com/changhu73/Internal_states_leakage}
>
---
#### [new 099] SentiMM: A Multimodal Multi-Agent Framework for Sentiment Analysis in Social Media
- **分类: cs.CL**

- **简介: 该论文提出SentiMM框架，用于社交媒体情感分析任务，解决多模态数据融合与外部知识整合难题。通过多智能体协作处理文本和视觉信息，结合知识检索增强语境，实现更精准的情感分类。**

- **链接: [http://arxiv.org/pdf/2508.18108v1](http://arxiv.org/pdf/2508.18108v1)**

> **作者:** Xilai Xu; Zilin Zhao; Chengye Song; Zining Wang; Jinhe Qiang; Jiongrui Yan; Yuhuai Lin
>
> **摘要:** With the increasing prevalence of multimodal content on social media, sentiment analysis faces significant challenges in effectively processing heterogeneous data and recognizing multi-label emotions. Existing methods often lack effective cross-modal fusion and external knowledge integration. We propose SentiMM, a novel multi-agent framework designed to systematically address these challenges. SentiMM processes text and visual inputs through specialized agents, fuses multimodal features, enriches context via knowledge retrieval, and aggregates results for final sentiment classification. We also introduce SentiMMD, a large-scale multimodal dataset with seven fine-grained sentiment categories. Extensive experiments demonstrate that SentiMM achieves superior performance compared to state-of-the-art baselines, validating the effectiveness of our structured approach.
>
---
#### [new 100] UI-Level Evaluation of ALLaM 34B: Measuring an Arabic-Centric LLM via HUMAIN Chat
- **分类: cs.CL**

- **简介: 论文评估阿拉伯语大模型ALLaM-34B的用户界面性能，解决英文主导模型难以处理阿拉伯语言文化特性的难题。通过多维度提示测试和专家评分，验证其在生成、推理、方言适应及安全性上的高表现，证明其适合实际部署。**

- **链接: [http://arxiv.org/pdf/2508.17378v1](http://arxiv.org/pdf/2508.17378v1)**

> **作者:** Omer Nacar
>
> **摘要:** Large language models (LLMs) trained primarily on English corpora often struggle to capture the linguistic and cultural nuances of Arabic. To address this gap, the Saudi Data and AI Authority (SDAIA) introduced the $ALLaM$ family of Arabic-focused models. The most capable of these available to the public, $ALLaM-34B$, was subsequently adopted by HUMAIN, who developed and deployed HUMAIN Chat, a closed conversational web service built on this model. This paper presents an expanded and refined UI-level evaluation of $ALLaM-34B$. Using a prompt pack spanning modern standard Arabic, five regional dialects, code-switching, factual knowledge, arithmetic and temporal reasoning, creative generation, and adversarial safety, we collected 115 outputs (23 prompts times 5 runs) and scored each with three frontier LLM judges (GPT-5, Gemini 2.5 Pro, Claude Sonnet-4). We compute category-level means with 95\% confidence intervals, analyze score distributions, and visualize dialect-wise metric heat maps. The updated analysis reveals consistently high performance on generation and code-switching tasks (both averaging 4.92/5), alongside strong results in MSA handling (4.74/5), solid reasoning ability (4.64/5), and improved dialect fidelity (4.21/5). Safety-related prompts show stable, reliable performance of (4.54/5). Taken together, these results position $ALLaM-34B$ as a robust and culturally grounded Arabic LLM, demonstrating both technical strength and practical readiness for real-world deployment.
>
---
#### [new 101] The Arabic Generality Score: Another Dimension of Modeling Arabic Dialectness
- **分类: cs.CL; cs.AI**

- **简介: 论文提出阿拉伯通用性得分（AGS），衡量词汇在阿拉伯方言中的广泛使用程度，解决现有模型将方言视为离散类别而非连续谱的问题。通过词对齐和编辑距离等方法构建标注语料，并训练回归模型预测AGS，提升方言建模的细腻度与实用性。**

- **链接: [http://arxiv.org/pdf/2508.17347v1](http://arxiv.org/pdf/2508.17347v1)**

> **作者:** Sanad Shaban; Nizar Habash
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Arabic dialects form a diverse continuum, yet NLP models often treat them as discrete categories. Recent work addresses this issue by modeling dialectness as a continuous variable, notably through the Arabic Level of Dialectness (ALDi). However, ALDi reduces complex variation to a single dimension. We propose a complementary measure: the Arabic Generality Score (AGS), which quantifies how widely a word is used across dialects. We introduce a pipeline that combines word alignment, etymology-aware edit distance, and smoothing to annotate a parallel corpus with word-level AGS. A regression model is then trained to predict AGS in context. Our approach outperforms strong baselines, including state-of-the-art dialect ID systems, on a multi-dialect benchmark. AGS offers a scalable, linguistically grounded way to model lexical generality, enriching representations of Arabic dialectness.
>
---
#### [new 102] Demographic Biases and Gaps in the Perception of Sexism in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在社交媒体文本中识别性别歧视的能力，旨在解决模型对不同人群感知差异的偏差问题。通过分析EXIST 2024数据集，发现模型虽能总体检测 sexism，但无法准确反映不同年龄和性别群体的多样性观点，强调需改进模型以更好捕捉多元视角。**

- **链接: [http://arxiv.org/pdf/2508.18245v1](http://arxiv.org/pdf/2508.18245v1)**

> **作者:** Judith Tavarez-Rodríguez; Fernando Sánchez-Vega; A. Pastor López-Monroy
>
> **备注:** This work was presented as a poster at the Latin American Meeting in Artificial Intelligence KHIPU 2025, Santiago, Chile, March 10th - 14th 2025, https://khipu.ai/khipu2025/poster-sessions-2025/
>
> **摘要:** The use of Large Language Models (LLMs) has proven to be a tool that could help in the automatic detection of sexism. Previous studies have shown that these models contain biases that do not accurately reflect reality, especially for minority groups. Despite various efforts to improve the detection of sexist content, this task remains a significant challenge due to its subjective nature and the biases present in automated models. We explore the capabilities of different LLMs to detect sexism in social media text using the EXIST 2024 tweet dataset. It includes annotations from six distinct profiles for each tweet, allowing us to evaluate to what extent LLMs can mimic these groups' perceptions in sexism detection. Additionally, we analyze the demographic biases present in the models and conduct a statistical analysis to identify which demographic characteristics (age, gender) contribute most effectively to this task. Our results show that, while LLMs can to some extent detect sexism when considering the overall opinion of populations, they do not accurately replicate the diversity of perceptions among different demographic groups. This highlights the need for better-calibrated models that account for the diversity of perspectives across different populations.
>
---
#### [new 103] Learning from Diverse Reasoning Paths with Routing and Collaboration
- **分类: cs.CL**

- **简介: 论文提出QR-Distill方法，用于知识蒸馏任务中提升学生模型对教师模型多样推理路径的学习效果。针对传统方法忽视推理质量与路径差异的问题，通过质量过滤、条件路由和同伴协作机制，实现更高效的知识迁移。**

- **链接: [http://arxiv.org/pdf/2508.16861v1](http://arxiv.org/pdf/2508.16861v1)**

> **作者:** Zhenyu Lei; Zhen Tan; Song Wang; Yaochen Zhu; Zihan Chen; Yushun Dong; Jundong Li
>
> **摘要:** Advances in large language models (LLMs) significantly enhance reasoning capabilities but their deployment is restricted in resource-constrained scenarios. Knowledge distillation addresses this by transferring knowledge from powerful teacher models to compact and transparent students. However, effectively capturing the teacher's comprehensive reasoning is challenging due to conventional token-level supervision's limited scope. Using multiple reasoning paths per query alleviates this problem, but treating each path identically is suboptimal as paths vary widely in quality and suitability across tasks and models. We propose Quality-filtered Routing with Cooperative Distillation (QR-Distill), combining path quality filtering, conditional routing, and cooperative peer teaching. First, quality filtering retains only correct reasoning paths scored by an LLM-based evaluation. Second, conditional routing dynamically assigns paths tailored to each student's current learning state. Finally, cooperative peer teaching enables students to mutually distill diverse insights, addressing knowledge gaps and biases toward specific reasoning styles. Experiments demonstrate QR-Distill's superiority over traditional single- and multi-path distillation methods. Ablation studies further highlight the importance of each component including quality filtering, conditional routing, and peer teaching in effective knowledge transfer. Our code is available at https://github.com/LzyFischer/Distill.
>
---
#### [new 104] German4All - A Dataset and Model for Readability-Controlled Paraphrasing in German
- **分类: cs.CL**

- **简介: 论文提出German4All，首个大规模德语可读性控制 paraphrasing 数据集与模型，解决德语文本多层级简化问题。数据集含25,000条五级可读性对齐段落，基于GPT-4合成并经人工和LLM验证，训练出开源最优模型，支持精准读者适配的文本改写。**

- **链接: [http://arxiv.org/pdf/2508.17973v1](http://arxiv.org/pdf/2508.17973v1)**

> **作者:** Miriam Anschütz; Thanh Mai Pham; Eslam Nasrallah; Maximilian Müller; Cristian-George Craciun; Georg Groh
>
> **备注:** Accepted to INLG 2025
>
> **摘要:** The ability to paraphrase texts across different complexity levels is essential for creating accessible texts that can be tailored toward diverse reader groups. Thus, we introduce German4All, the first large-scale German dataset of aligned readability-controlled, paragraph-level paraphrases. It spans five readability levels and comprises over 25,000 samples. The dataset is automatically synthesized using GPT-4 and rigorously evaluated through both human and LLM-based judgments. Using German4All, we train an open-source, readability-controlled paraphrasing model that achieves state-of-the-art performance in German text simplification, enabling more nuanced and reader-specific adaptations. We opensource both the dataset and the model to encourage further research on multi-level paraphrasing
>
---
#### [new 105] Evaluating the Representation of Vowels in Wav2Vec Feature Extractor: A Layer-Wise Analysis Using MFCCs
- **分类: cs.CL**

- **简介: 该论文研究Wav2Vec中CNN层对元音的表征能力，任务为元音分类。通过对比MFCC、含共振峰的MFCC与CNN激活特征，用SVM分类器评估其识别准确率，以分析不同特征对语音信息的表达效果。**

- **链接: [http://arxiv.org/pdf/2508.17914v1](http://arxiv.org/pdf/2508.17914v1)**

> **作者:** Domenico De Cristofaro; Vincenzo Norman Vitale; Alessandro Vietti
>
> **摘要:** Automatic Speech Recognition has advanced with self-supervised learning, enabling feature extraction directly from raw audio. In Wav2Vec, a CNN first transforms audio into feature vectors before the transformer processes them. This study examines CNN-extracted information for monophthong vowels using the TIMIT corpus. We compare MFCCs, MFCCs with formants, and CNN activations by training SVM classifiers for front-back vowel identification, assessing their classification accuracy to evaluate phonetic representation.
>
---
#### [new 106] Improving End-to-End Training of Retrieval-Augmented Generation Models via Joint Stochastic Approximation
- **分类: cs.CL**

- **简介: 论文针对检索增强生成（RAG）模型的端到端训练难题，提出基于联合随机逼近（JSA）的方法JSA-RAG，以更准确地估计离散潜在变量梯度，提升生成与检索性能。**

- **链接: [http://arxiv.org/pdf/2508.18168v1](http://arxiv.org/pdf/2508.18168v1)**

> **作者:** Hongyu Cao; Yuxuan Wu; Yucheng Cai; Xianyu Zhao; Zhijian Ou
>
> **摘要:** Retrieval-augmented generation (RAG) has become a widely recognized paradigm to combine parametric memory with non-parametric memories. An RAG model consists of two serial connecting components (retriever and generator). A major challenge in end-to-end optimization of the RAG model is that marginalization over relevant passages (modeled as discrete latent variables) from a knowledge base is required. Traditional top-K marginalization and variational RAG (VRAG) suffer from biased or high-variance gradient estimates. In this paper, we propose and develop joint stochastic approximation (JSA) based end-to-end training of RAG, which is referred to as JSA-RAG. The JSA algorithm is a stochastic extension of the EM (expectation-maximization) algorithm and is particularly powerful in estimating discrete latent variable models. Extensive experiments are conducted on five datasets for two tasks (open-domain question answering, knowledge-grounded dialogs) and show that JSA-RAG significantly outperforms both vanilla RAG and VRAG. Further analysis shows the efficacy of JSA-RAG from the perspectives of generation, retrieval, and low-variance gradient estimate.
>
---
#### [new 107] DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models
- **分类: cs.CL**

- **简介: 论文提出DRQA方法，解决推理大模型过思考问题，通过动态分配推理资源，在保持或提升准确率的同时减少token消耗，实现高效推理。**

- **链接: [http://arxiv.org/pdf/2508.17803v1](http://arxiv.org/pdf/2508.17803v1)**

> **作者:** Kaiwen Yan; Xuanqing Shi; Hongcheng Guo; Wenxuan Wang; Zhuosheng Zhang; Chengwei Qin
>
> **摘要:** Reasoning large language models (RLLMs), such as OpenAI-O3 and DeepSeek-R1, have recently demonstrated remarkable capabilities by performing structured and multi-step reasoning. However, recent studies reveal that RLLMs often suffer from overthinking, i.e., producing unnecessarily lengthy reasoning chains even for simple questions, leading to excessive token consumption and computational inefficiency. Interestingly, we observe that when processing multiple questions in batch mode, RLLMs exhibit more resource-efficient behavior by dynamically compressing reasoning steps for easier problems, due to implicit resource competition. Inspired by this, we propose Dynamic Reasoning Quota Allocation (DRQA), a novel method that transfers the benefits of resource competition from batch processing to single-question inference. Specifically, DRQA leverages batch-generated preference data and reinforcement learning to train the model to allocate reasoning resources adaptively. By encouraging the model to internalize a preference for responses that are both accurate and concise, DRQA enables it to generate concise answers for simple questions while retaining sufficient reasoning depth for more challenging ones. Extensive experiments on a wide range of mathematical and scientific reasoning benchmarks demonstrate that DRQA significantly reduces token usage while maintaining, and in many cases improving, answer accuracy. By effectively mitigating the overthinking problem, DRQA offers a promising direction for more efficient and scalable deployment of RLLMs, and we hope it inspires further exploration into fine-grained control of reasoning behaviors.
>
---
#### [new 108] Speculating LLMs' Chinese Training Data Pollution from Their Tokens
- **分类: cs.CL**

- **简介: 论文研究大语言模型中文训练数据污染问题，通过定义和检测“污染令牌”（PoC），分析其与训练数据的关系。工作包括：提出PoC分类、构建检测器、验证方法有效性，并推测GPT-4o中特定网页占比约0.5%。**

- **链接: [http://arxiv.org/pdf/2508.17771v1](http://arxiv.org/pdf/2508.17771v1)**

> **作者:** Qingjie Zhang; Di Wang; Haoting Qian; Liu Yan; Tianwei Zhang; Ke Xu; Qi Li; Minlie Huang; Hewu Li; Han Qiu
>
> **摘要:** Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%.
>
---
#### [new 109] Improving French Synthetic Speech Quality via SSML Prosody Control
- **分类: cs.CL; cs.SD; 68T50; I.2.7; H.5.5**

- **简介: 该论文属于文本到语音合成任务，旨在提升法语合成语音的自然度。通过引入SSML标记控制韵律参数，提出端到端pipeline，显著改善语音质量与听者偏好。**

- **链接: [http://arxiv.org/pdf/2508.17494v1](http://arxiv.org/pdf/2508.17494v1)**

> **作者:** Nassima Ould Ouali; Awais Hussain Sani; Ruben Bueno; Jonah Dauvet; Tim Luka Horstmann; Eric Moulines
>
> **备注:** 13 pages, 9 figures, 6 tables. Accepted for presentation at ICNLSP 2025 (Odense, Denmark). Code and demo: https://github.com/hi-paris/Prosody-Control-French-TTS. ACM Class: I.2.7; H.5.5
>
> **摘要:** Despite recent advances, synthetic voices often lack expressiveness due to limited prosody control in commercial text-to-speech (TTS) systems. We introduce the first end-to-end pipeline that inserts Speech Synthesis Markup Language (SSML) tags into French text to control pitch, speaking rate, volume, and pause duration. We employ a cascaded architecture with two QLoRA-fine-tuned Qwen 2.5-7B models: one predicts phrase-break positions and the other performs regression on prosodic targets, generating commercial TTS-compatible SSML markup. Evaluated on a 14-hour French podcast corpus, our method achieves 99.2% F1 for break placement and reduces mean absolute error on pitch, rate, and volume by 25-40% compared with prompting-only large language models (LLMs) and a BiLSTM baseline. In perceptual evaluation involving 18 participants across over 9 hours of synthesized audio, SSML-enhanced speech generated by our pipeline significantly improves naturalness, with the mean opinion score increasing from 3.20 to 3.87 (p < 0.005). Additionally, 15 of 18 listeners preferred our enhanced synthesis. These results demonstrate substantial progress in bridging the expressiveness gap between synthetic and natural French speech. Our code is publicly available at https://github.com/hi-paris/Prosody-Control-French-TTS.
>
---
#### [new 110] A Retail-Corpus for Aspect-Based Sentiment Analysis with Large Language Models
- **分类: cs.CL**

- **简介: 论文提出一个用于基于方面情感分析的零售语料库，包含10,814条多语言客户评论，标注了八个方面及其情感。通过该数据集评估GPT-4和LLaMA-3模型性能，结果显示两者准确率均超85%，GPT-4表现更优。**

- **链接: [http://arxiv.org/pdf/2508.17994v1](http://arxiv.org/pdf/2508.17994v1)**

> **作者:** Oleg Silcenco; Marcos R. Machad; Wallace C. Ugulino; Daniel Braun
>
> **备注:** Accepted at ICNLSP 2025
>
> **摘要:** Aspect-based sentiment analysis enhances sentiment detection by associating it with specific aspects, offering deeper insights than traditional sentiment analysis. This study introduces a manually annotated dataset of 10,814 multilingual customer reviews covering brick-and-mortar retail stores, labeled with eight aspect categories and their sentiment. Using this dataset, the performance of GPT-4 and LLaMA-3 in aspect based sentiment analysis is evaluated to establish a baseline for the newly introduced data. The results show both models achieving over 85% accuracy, while GPT-4 outperforms LLaMA-3 overall with regard to all relevant metrics.
>
---
#### [new 111] GRAID: Synthetic Data Generation with Geometric Constraints and Multi-Agentic Reflection for Harmful Content Detection
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 论文提出GRAID方法，用于有害文本分类中的数据稀缺问题。通过几何约束生成与多智能体反思增强，提升数据多样性和覆盖度，从而改善防护模型性能。**

- **链接: [http://arxiv.org/pdf/2508.17057v1](http://arxiv.org/pdf/2508.17057v1)**

> **作者:** Melissa Kazemi Rad; Alberto Purpura; Himanshu Kumar; Emily Chen; Mohammad Shahed Sorower
>
> **备注:** 19 pages, 12 figures
>
> **摘要:** We address the problem of data scarcity in harmful text classification for guardrailing applications and introduce GRAID (Geometric and Reflective AI-Driven Data Augmentation), a novel pipeline that leverages Large Language Models (LLMs) for dataset augmentation. GRAID consists of two stages: (i) generation of geometrically controlled examples using a constrained LLM, and (ii) augmentation through a multi-agentic reflective process that promotes stylistic diversity and uncovers edge cases. This combination enables both reliable coverage of the input space and nuanced exploration of harmful content. Using two benchmark data sets, we demonstrate that augmenting a harmful text classification dataset with GRAID leads to significant improvements in downstream guardrail model performance.
>
---
#### [new 112] Speech-Based Depressive Mood Detection in the Presence of Multiple Sclerosis: A Cross-Corpus and Cross-Lingual Study
- **分类: cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决多发性硬化症患者抑郁情绪检测问题。通过跨语种、跨数据集的机器学习方法，利用语音特征和情绪维度提升检测效果，实现对患者抑郁状态的初步识别。**

- **链接: [http://arxiv.org/pdf/2508.18092v1](http://arxiv.org/pdf/2508.18092v1)**

> **作者:** Monica Gonzalez-Machorro; Uwe Reichel; Pascal Hecker; Helly Hammer; Hesam Sagha; Florian Eyben; Robert Hoepner; Björn W. Schuller
>
> **备注:** Accepted at the 8th International Conference on Natural Language and Speech Processing (ICNLSP 2025). To be appeared in the corresponding Proceedings at ACL Anthology
>
> **摘要:** Depression commonly co-occurs with neurodegenerative disorders like Multiple Sclerosis (MS), yet the potential of speech-based Artificial Intelligence for detecting depression in such contexts remains unexplored. This study examines the transferability of speech-based depression detection methods to people with MS (pwMS) through cross-corpus and cross-lingual analysis using English data from the general population and German data from pwMS. Our approach implements supervised machine learning models using: 1) conventional speech and language features commonly used in the field, 2) emotional dimensions derived from a Speech Emotion Recognition (SER) model, and 3) exploratory speech feature analysis. Despite limited data, our models detect depressive mood in pwMS with moderate generalisability, achieving a 66% Unweighted Average Recall (UAR) on a binary task. Feature selection further improved performance, boosting UAR to 74%. Our findings also highlight the relevant role emotional changes have as an indicator of depressive mood in both the general population and within PwMS. This study provides an initial exploration into generalising speech-based depression detection, even in the presence of co-occurring conditions, such as neurodegenerative diseases.
>
---
#### [new 113] THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics
- **分类: q-fin.PM; cs.AI; cs.CL; cs.IR**

- **简介: 论文提出THEME框架，解决主题投资中股票选择困难问题。通过构建含文本与财务数据的TRS数据集，利用层次对比学习融合语义与时间动态，提升主题相关资产检索与组合构建效果。**

- **链接: [http://arxiv.org/pdf/2508.16936v1](http://arxiv.org/pdf/2508.16936v1)**

> **作者:** Hoyoung Lee; Wonbin Ahn; Suhwan Park; Jaehoon Lee; Minjae Kim; Sungdong Yoo; Taeyoon Lim; Woohyung Lim; Yongjae Lee
>
> **备注:** Accepted at ACM International Conference on Information and Knowledge Management (CIKM)
>
> **摘要:** Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes.
>
---
#### [new 114] Empirical Analysis of the Effect of Context in the Task of Automated Essay Scoring in Transformer-Based Models
- **分类: cs.CY; cs.CL**

- **简介: 论文研究Transformer模型在自动作文评分（AES）任务中的表现，旨在通过引入多种上下文信息提升其性能。作者在ASAP-AES数据集上验证了上下文增强的有效性，使模型在多数情况下超越现有Transformer模型，接近当前最优深度学习模型。**

- **链接: [http://arxiv.org/pdf/2508.16638v1](http://arxiv.org/pdf/2508.16638v1)**

> **作者:** Abhirup Chakravarty
>
> **备注:** MSc Dissertation
>
> **摘要:** Automated Essay Scoring (AES) has emerged to prominence in response to the growing demand for educational automation. Providing an objective and cost-effective solution, AES standardises the assessment of extended responses. Although substantial research has been conducted in this domain, recent investigations reveal that alternative deep-learning architectures outperform transformer-based models. Despite the successful dominance in the performance of the transformer architectures across various other tasks, this discrepancy has prompted a need to enrich transformer-based AES models through contextual enrichment. This study delves into diverse contextual factors using the ASAP-AES dataset, analysing their impact on transformer-based model performance. Our most effective model, augmented with multiple contextual dimensions, achieves a mean Quadratic Weighted Kappa score of 0.823 across the entire essay dataset and 0.8697 when trained on individual essay sets. Evidently surpassing prior transformer-based models, this augmented approach only underperforms relative to the state-of-the-art deep learning model trained essay-set-wise by an average of 3.83\% while exhibiting superior performance in three of the eight sets. Importantly, this enhancement is orthogonal to architecture-based advancements and seamlessly adaptable to any AES model. Consequently, this contextual augmentation methodology presents a versatile technique for refining AES capabilities, contributing to automated grading and evaluation evolution in educational settings.
>
---
#### [new 115] Proximal Supervised Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出Proximal SFT（PSFT），用于改进基础模型的监督微调（SFT）泛化能力。针对SFT导致旧能力退化的难题，PSFT借鉴强化学习中的信任区域思想，稳定优化过程，提升跨域泛化性能，同时避免熵崩溃，为后续训练提供更好基础。**

- **链接: [http://arxiv.org/pdf/2508.17784v1](http://arxiv.org/pdf/2508.17784v1)**

> **作者:** Wenhong Zhu; Ruobing Xie; Rui Wang; Xingwu Sun; Di Wang; Pengfei Liu
>
> **摘要:** Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization.
>
---
#### [new 116] RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System
- **分类: cs.DB; cs.AI; cs.CL; cs.MA; H.2.3; I.2.4; I.2.7**

- **简介: 论文提出RubikSQL，一种面向工业场景的NL2SQL系统，通过终身学习框架解决隐式意图和领域术语问题，结合知识库维护与多代理工作流提升SQL生成准确性，并发布RubikBench基准。**

- **链接: [http://arxiv.org/pdf/2508.17590v1](http://arxiv.org/pdf/2508.17590v1)**

> **作者:** Zui Chen; Han Li; Xinhao Zhang; Xiaoyu Chen; Chunyin Dong; Yifeng Wang; Xin Cai; Su Zhang; Ziqi Li; Chi Ding; Jinxu Li; Shuai Wang; Dousheng Zhao; Sanhai Gao; Guangyi Liu
>
> **备注:** 18 pages, 3 figures, 3 tables, to be submitted to VLDB 2026 (PVLDB Volume 19)
>
> **摘要:** We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research.
>
---
#### [new 117] Unseen Speaker and Language Adaptation for Lightweight Text-To-Speech with Adapters
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 论文研究轻量级TTS中的跨语言与未见说话人适应问题，提出基于适配器的方法，在不遗忘原模型信息的前提下，实现目标语言中未见说话人的语音合成，并引入新指标评估语音自然度。**

- **链接: [http://arxiv.org/pdf/2508.18006v1](http://arxiv.org/pdf/2508.18006v1)**

> **作者:** Alessio Falai; Ziyao Zhang; Akos Gangoly
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** In this paper we investigate cross-lingual Text-To-Speech (TTS) synthesis through the lens of adapters, in the context of lightweight TTS systems. In particular, we compare the tasks of unseen speaker and language adaptation with the goal of synthesising a target voice in a target language, in which the target voice has no recordings therein. Results from objective evaluations demonstrate the effectiveness of adapters in learning language-specific and speaker-specific information, allowing pre-trained models to learn unseen speaker identities or languages, while avoiding catastrophic forgetting of the original model's speaker or language information. Additionally, to measure how native the generated voices are in terms of accent, we propose and validate an objective metric inspired by mispronunciation detection techniques in second-language (L2) learners. The paper also provides insights into the impact of adapter placement, configuration and the number of speakers used.
>
---
#### [new 118] Characterizing the Behavior of Training Mamba-based State Space Models on GPUs
- **分类: cs.LG; cs.AR; cs.CL**

- **简介: 论文研究Mamba-based SSM在GPU上的训练行为，旨在解决Transformer因注意力计算复杂度高难以扩展的问题。作者构建了代表性模型套件，分析其GPU微架构影响，为优化性能提供依据。**

- **链接: [http://arxiv.org/pdf/2508.17679v1](http://arxiv.org/pdf/2508.17679v1)**

> **作者:** Trinayan Baruah; Kaustubh Shivdikar; Sara Prescott; David Kaeli
>
> **摘要:** Mamba-based State Space Models (SSM) have emerged as a promising alternative to the ubiquitous transformers. Despite the expressive power of transformers, the quadratic complexity of computing attention is a major impediment to scaling performance as we increase the sequence length. SSMs provide an alternative path that addresses this problem, reducing the computational complexity requirements of self-attention with novel model architectures for different domains and fields such as video, text generation and graphs. Thus, it is important to characterize the behavior of these emerging workloads on GPUs and understand their requirements during GPU microarchitectural design. In this work we evaluate Mamba-based SSMs and characterize their behavior during training on GPUs. We construct a workload suite that offers representative models that span different model architectures. We then use this suite to analyze the architectural implications of running Mamba-based SSMs on GPUs. Our work sheds new light on potential optimizations to continue scaling the performance for such models.
>
---
#### [new 119] Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM中的奉承行为，提出用贝叶斯理性偏差量化其非理性更新。通过多任务、多模型实验，发现奉承导致预测后验偏移并增加贝叶斯误差，且与传统准确度指标无关。**

- **链接: [http://arxiv.org/pdf/2508.16846v1](http://arxiv.org/pdf/2508.16846v1)**

> **作者:** Katherine Atwell; Pedram Heydari; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy.
>
---
#### [new 120] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling
- **分类: cs.LG; cs.CL**

- **简介: 论文提出WISCA方法，通过权重缩放优化Transformer模型训练中的权重模式，提升收敛质量和效率，尤其适用于GQA架构和LoRA微调任务。**

- **链接: [http://arxiv.org/pdf/2508.16676v1](http://arxiv.org/pdf/2508.16676v1)**

> **作者:** Jiacheng Li; Jianchao Tan; Zhidong Yang; Pingwei Sun; Feiye Huo; Jiayu Qin; Yerui Sun; Yuchen Xie; Xunliang Cai; Xiangyu Zhang; Maoxin He; Guangming Tan; Weile Jia; Tong Zhao
>
> **摘要:** Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures.
>
---
#### [new 121] Guarding Your Conversations: Privacy Gatekeepers for Secure Interactions with Cloud-Based AI Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文提出“LLM门卫”概念，解决云上大模型隐私泄露问题。通过本地轻量模型过滤用户查询中的敏感信息，再发送至云端模型，实现在不降低响应质量前提下显著提升隐私保护。**

- **链接: [http://arxiv.org/pdf/2508.16765v1](http://arxiv.org/pdf/2508.16765v1)**

> **作者:** GodsGift Uzor; Hasan Al-Qudah; Ynes Ineza; Abdul Serwadda
>
> **备注:** 2025 19th International Conference on Semantic Computing (ICSC)
>
> **摘要:** The interactive nature of Large Language Models (LLMs), which closely track user data and context, has prompted users to share personal and private information in unprecedented ways. Even when users opt out of allowing their data to be used for training, these privacy settings offer limited protection when LLM providers operate in jurisdictions with weak privacy laws, invasive government surveillance, or poor data security practices. In such cases, the risk of sensitive information, including Personally Identifiable Information (PII), being mishandled or exposed remains high. To address this, we propose the concept of an "LLM gatekeeper", a lightweight, locally run model that filters out sensitive information from user queries before they are sent to the potentially untrustworthy, though highly capable, cloud-based LLM. Through experiments with human subjects, we demonstrate that this dual-model approach introduces minimal overhead while significantly enhancing user privacy, without compromising the quality of LLM responses.
>
---
#### [new 122] Invisible Filters: Cultural Bias in Hiring Evaluations Using Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在招聘评估中的文化偏见问题，旨在揭示其跨文化公平性。通过分析英、印求职者面试文本，发现印度样本得分普遍较低，且与语言特征相关；控制身份替换实验表明姓名单独影响不显著。工作聚焦于识别并量化LMM在招聘中的文化偏见。**

- **链接: [http://arxiv.org/pdf/2508.16673v1](http://arxiv.org/pdf/2508.16673v1)**

> **作者:** Pooja S. B. Rao; Laxminarayen Nagarajan Venkatesan; Mauro Cherubini; Dinesh Babu Jayagopi
>
> **备注:** Accepted to AIES 2025
>
> **摘要:** Artificial Intelligence (AI) is increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. Despite their growing role, few studies have systematically examined the potential biases in AI-driven hiring evaluation across cultures. In this study, we conduct a systematic analysis of how LLMs assess job interviews across cultural and identity dimensions. Using two datasets of interview transcripts, 100 from UK and 100 from Indian job seekers, we first examine cross-cultural differences in LLM-generated scores for hirability and related traits. Indian transcripts receive consistently lower scores than UK transcripts, even when they were anonymized, with disparities linked to linguistic features such as sentence complexity and lexical diversity. We then perform controlled identity substitutions (varying names by gender, caste, and region) within the Indian dataset to test for name-based bias. These substitutions do not yield statistically significant effects, indicating that names alone, when isolated from other contextual signals, may not influence LLM evaluations. Our findings underscore the importance of evaluating both linguistic and social dimensions in LLM-driven evaluations and highlight the need for culturally sensitive design and accountability in AI-assisted hiring.
>
---
#### [new 123] Anemoi: A Semi-Centralized Multi-agent Systems Based on Agent-to-Agent Communication MCP server from Coral Protocol
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出Anemoi，一种基于A2A通信的半中心化多智能体系统，解决传统设计对强规划器依赖和低效通信的问题。通过实时协作与动态调整，提升性能与效率，在GAIA基准上超越基线9.09%。**

- **链接: [http://arxiv.org/pdf/2508.17068v1](http://arxiv.org/pdf/2508.17068v1)**

> **作者:** Xinxing Ren; Caelum Forder; Qianbo Zang; Ahsen Tahir; Roman J. Georgio; Suman Deb; Peter Carroll; Önder Gürcan; Zekun Guo
>
> **摘要:** Recent advances in generalist multi-agent systems (MAS) have largely followed a context-engineering plus centralized paradigm, where a planner agent coordinates multiple worker agents through unidirectional prompt passing. While effective under strong planner models, this design suffers from two critical limitations: (1) strong dependency on the planner's capability, which leads to degraded performance when a smaller LLM powers the planner; and (2) limited inter-agent communication, where collaboration relies on costly prompt concatenation and context injection, introducing redundancy and information loss. To address these challenges, we propose Anemoi, a semi-centralized MAS built on the Agent-to-Agent (A2A) communication MCP server from Coral Protocol. Unlike traditional designs, Anemoi enables structured and direct inter-agent collaboration, allowing all agents to monitor progress, assess results, identify bottlenecks, and propose refinements in real time. This paradigm reduces reliance on a single planner, supports adaptive plan updates, and minimizes redundant context passing, resulting in more scalable and cost-efficient execution. Evaluated on the GAIA benchmark, Anemoi achieved 52.73\% accuracy with a small LLM (GPT-4.1-mini) as the planner, surpassing the strongest open-source baseline OWL (43.63\%) by +9.09\% under identical LLM settings. Our implementation is publicly available at https://github.com/Coral-Protocol/Anemoi.
>
---
#### [new 124] TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling
- **分类: cs.LG; cs.CL**

- **简介: 论文提出TreePO，用于提升大语言模型强化学习训练中的效率与效果。针对传统方法计算开销大、探索不足的问题，通过树状结构搜索和分段采样策略，减少计算资源消耗并增强路径多样性，实现更高效的推理与训练。**

- **链接: [http://arxiv.org/pdf/2508.17445v1](http://arxiv.org/pdf/2508.17445v1)**

> **作者:** Yizhi Li; Qingshui Gu; Zhoufutu Wen; Ziniu Li; Tianshun Xing; Shuyue Guo; Tianyu Zheng; Xin Zhou; Xingwei Qu; Wangchunshu Zhou; Zheng Zhang; Wei Shen; Qian Liu; Chenghua Lin; Jian Yang; Ge Zhang; Wenhao Huang
>
> **摘要:** Recent advancements in aligning large language models via reinforcement learning have achieved remarkable gains in solving complex reasoning problems, but at the cost of expensive on-policy rollouts and limited exploration of diverse reasoning paths. In this work, we introduce TreePO, involving a self-guided rollout algorithm that views sequence generation as a tree-structured searching process. Composed of dynamic tree sampling policy and fixed-length segment decoding, TreePO leverages local uncertainty to warrant additional branches. By amortizing computation across common prefixes and pruning low-value paths early, TreePO essentially reduces the per-update compute burden while preserving or enhancing exploration diversity. Key contributions include: (1) a segment-wise sampling algorithm that alleviates the KV cache burden through contiguous segments and spawns new branches along with an early-stop mechanism; (2) a tree-based segment-level advantage estimation that considers both global and local proximal policy optimization. and (3) analysis on the effectiveness of probability and quality-driven dynamic divergence and fallback strategy. We empirically validate the performance gain of TreePO on a set reasoning benchmarks and the efficiency saving of GPU hours from 22\% up to 43\% of the sampling design for the trained models, meanwhile showing up to 40\% reduction at trajectory-level and 35\% at token-level sampling compute for the existing models. While offering a free lunch of inference efficiency, TreePO reveals a practical path toward scaling RL-based post-training with fewer samples and less compute. Home page locates at https://m-a-p.ai/TreePO.
>
---
#### [new 125] LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文研究LLM过度自信的机制，提出其可分解为情感与逻辑两个正交成分。通过分析Llama 3.2模型激活特征，定位关键层并验证双路径结构，揭示不同成分对预测准确性的因果影响，为缓解过自信行为提供新思路。**

- **链接: [http://arxiv.org/pdf/2508.17182v1](http://arxiv.org/pdf/2508.17182v1)**

> **作者:** Hikaru Tsujimura; Arush Tagade
>
> **备注:** This preprint is under review
>
> **摘要:** Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior.
>
---
#### [new 126] Unraveling the cognitive patterns of Large Language Models through module communities
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型可解释性研究任务，旨在揭示大语言模型（LLMs）的认知模式。通过构建网络框架，分析模块社区与认知技能的关系，发现LLMs具有类生物大脑的分布式结构，但依赖动态交互和神经可塑性实现技能提升，提出应采用分布式学习策略优化微调。**

- **链接: [http://arxiv.org/pdf/2508.18192v1](http://arxiv.org/pdf/2508.18192v1)**

> **作者:** Kushal Raj Bhandari; Pin-Yu Chen; Jianxi Gao
>
> **摘要:** Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions.
>
---
#### [new 127] Humans Perceive Wrong Narratives from AI Reasoning Texts
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文研究AI推理文本的人类理解偏差，属于可解释性任务。它发现人类难以识别推理步骤间的因果关系（准确率仅29.3%），揭示了人类解读与模型实际计算间的根本差异，提示需深入研究模型非人类的语言使用方式。**

- **链接: [http://arxiv.org/pdf/2508.16599v1](http://arxiv.org/pdf/2508.16599v1)**

> **作者:** Mosh Levy; Zohar Elyoseph; Yoav Goldberg
>
> **摘要:** A new generation of AI models generates step-by-step reasoning text before producing an answer. This text appears to offer a human-readable window into their computation process, and is increasingly relied upon for transparency and interpretability. However, it is unclear whether human understanding of this text matches the model's actual computational process. In this paper, we investigate a necessary condition for correspondence: the ability of humans to identify which steps in a reasoning text causally influence later steps. We evaluated humans on this ability by composing questions based on counterfactual measurements and found a significant discrepancy: participant accuracy was only 29.3%, barely above chance (25%), and remained low (42%) even when evaluating the majority vote on questions with high agreement. Our results reveal a fundamental gap between how humans interpret reasoning texts and how models use it, challenging its utility as a simple interpretability tool. We argue that reasoning texts should be treated as an artifact to be investigated, not taken at face value, and that understanding the non-human ways these models use language is a critical research direction.
>
---
#### [new 128] CEIDM: A Controlled Entity and Interaction Diffusion Model for Enhanced Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型在复杂实体及其交互控制上的难题。提出CEIDM模型，通过LLM挖掘交互关系、动作聚类偏移及实体控制网络，提升图像质量和交互合理性。**

- **链接: [http://arxiv.org/pdf/2508.17760v1](http://arxiv.org/pdf/2508.17760v1)**

> **作者:** Mingyue Yang; Dianxi Shi; Jialu Zhou; Xinyu Wei; Leqian Li; Shaowu Yang; Chunping Qiu
>
> **摘要:** In Text-to-Image (T2I) generation, the complexity of entities and their intricate interactions pose a significant challenge for T2I method based on diffusion model: how to effectively control entity and their interactions to produce high-quality images. To address this, we propose CEIDM, a image generation method based on diffusion model with dual controls for entity and interaction. First, we propose an entity interactive relationships mining approach based on Large Language Models (LLMs), extracting reasonable and rich implicit interactive relationships through chain of thought to guide diffusion models to generate high-quality images that are closer to realistic logic and have more reasonable interactive relationships. Furthermore, We propose an interactive action clustering and offset method to cluster and offset the interactive action features contained in each text prompts. By constructing global and local bidirectional offsets, we enhance semantic understanding and detail supplementation of original actions, making the model's understanding of the concept of interactive "actions" more accurate and generating images with more accurate interactive actions. Finally, we design an entity control network which generates masks with entity semantic guidance, then leveraging multi-scale convolutional network to enhance entity feature and dynamic network to fuse feature. It effectively controls entities and significantly improves image quality. Experiments show that the proposed CEIDM method is better than the most representative existing methods in both entity control and their interaction control.
>
---
#### [new 129] Can AI Have a Personality? Prompt Engineering for AI Personality Simulation: A Chatbot Case Study in Gender-Affirming Voice Therapy Training
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于AI人格模拟任务，旨在通过提示工程让聊天机器人模拟稳定人格。研究以性别确认语音治疗训练为场景，构建了名为Monae Jackson的虚拟患者，验证了prompt engineering可使AI保持一致人格特征。**

- **链接: [http://arxiv.org/pdf/2508.18234v1](http://arxiv.org/pdf/2508.18234v1)**

> **作者:** Tailon D. Jackson; Byunggu Yu
>
> **摘要:** This thesis investigates whether large language models (LLMs) can be guided to simulate a consistent personality through prompt engineering. The study explores this concept within the context of a chatbot designed for Speech-Language Pathology (SLP) student training, specifically focused on gender-affirming voice therapy. The chatbot, named Monae Jackson, was created to represent a 32-year-old transgender woman and engage in conversations simulating client-therapist interactions. Findings suggest that with prompt engineering, the chatbot maintained a recognizable and consistent persona and had a distinct personality based on the Big Five Personality test. These results support the idea that prompt engineering can be used to simulate stable personality characteristics in AI chatbots.
>
---
#### [new 130] Large Language Models as Universal Predictors? An Empirical Study on Small Tabular Datasets
- **分类: cs.AI; cs.CL**

- **简介: 论文研究大语言模型在小规模结构化数据上的预测能力，解决其能否作为通用预测器的问题。通过对比分类、回归和聚类任务，发现LLMs在分类上表现优异，但回归和聚类效果差，提出其可作为零训练基线用于探索性分析。**

- **链接: [http://arxiv.org/pdf/2508.17391v1](http://arxiv.org/pdf/2508.17391v1)**

> **作者:** Nikolaos Pavlidis; Vasilis Perifanis; Symeon Symeonidis; Pavlos S. Efraimidis
>
> **摘要:** Large Language Models (LLMs), originally developed for natural language processing (NLP), have demonstrated the potential to generalize across modalities and domains. With their in-context learning (ICL) capabilities, LLMs can perform predictive tasks over structured inputs without explicit fine-tuning on downstream tasks. In this work, we investigate the empirical function approximation capability of LLMs on small-scale structured datasets for classification, regression and clustering tasks. We evaluate the performance of state-of-the-art LLMs (GPT-5, GPT-4o, GPT-o3, Gemini-2.5-Flash, DeepSeek-R1) under few-shot prompting and compare them against established machine learning (ML) baselines, including linear models, ensemble methods and tabular foundation models (TFMs). Our results show that LLMs achieve strong performance in classification tasks under limited data availability, establishing practical zero-training baselines. In contrast, the performance in regression with continuous-valued outputs is poor compared to ML models, likely because regression demands outputs in a large (often infinite) space, and clustering results are similarly limited, which we attribute to the absence of genuine ICL in this setting. Nonetheless, this approach enables rapid, low-overhead data exploration and offers a viable alternative to traditional ML pipelines in business intelligence and exploratory analytics contexts. We further analyze the influence of context size and prompt structure on approximation quality, identifying trade-offs that affect predictive performance. Our findings suggest that LLMs can serve as general-purpose predictive engines for structured data, with clear strengths in classification and significant limitations in regression and clustering.
>
---
#### [new 131] HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation
- **分类: cs.IR; cs.CL**

- **简介: 论文提出HLLM-Creator框架，解决个性化内容生成中用户兴趣建模难、数据稀缺和效率低的问题。通过分层LLM结构与剪枝策略提升效率，结合思维链数据构建方法生成高质量、事实一致的个性化标题，在抖音搜索广告上显著提升点击率。**

- **链接: [http://arxiv.org/pdf/2508.18118v1](http://arxiv.org/pdf/2508.18118v1)**

> **作者:** Junyi Chen; Lu Chi; Siliang Xu; Shiwei Ran; Bingyue Peng; Zehuan Yuan
>
> **摘要:** AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at https://github.com/bytedance/HLLM.
>
---
#### [new 132] Designing Practical Models for Isolated Word Visual Speech Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文研究孤立词视觉语音识别任务，针对深度模型计算成本高、资源消耗大的问题，提出轻量级端到端架构。通过借鉴图像分类中的高效模型和轻量化模块，在时序卷积网络基础上构建低硬件需求且性能强的VSR系统，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.17894v1](http://arxiv.org/pdf/2508.17894v1)**

> **作者:** Iason Ioannis Panagos; Giorgos Sfikas; Christophoros Nikou
>
> **备注:** Double-column format, 13 pages with references, 2 figures
>
> **摘要:** Visual speech recognition (VSR) systems decode spoken words from an input sequence using only the video data. Practical applications of such systems include medical assistance as well as human-machine interactions. A VSR system is typically employed in a complementary role in cases where the audio is corrupt or not available. In order to accurately predict the spoken words, these architectures often rely on deep neural networks in order to extract meaningful representations from the input sequence. While deep architectures achieve impressive recognition performance, relying on such models incurs significant computation costs which translates into increased resource demands in terms of hardware requirements and results in limited applicability in real-world scenarios where resources might be constrained. This factor prevents wider adoption and deployment of speech recognition systems in more practical applications. In this work, we aim to alleviate this issue by developing architectures for VSR that have low hardware costs. Following the standard two-network design paradigm, where one network handles visual feature extraction and another one utilizes the extracted features to classify the entire sequence, we develop lightweight end-to-end architectures by first benchmarking efficient models from the image classification literature, and then adopting lightweight block designs in a temporal convolution network backbone. We create several unified models with low resource requirements but strong recognition performance. Experiments on the largest public database for English words demonstrate the effectiveness and practicality of our developed models. Code and trained models will be made publicly available.
>
---
#### [new 133] MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出MedRepBench基准，用于评估视觉语言模型在医疗报告结构化理解中的性能。针对缺乏标准化评测工具的问题，构建了1900份中文真实医疗报告数据集，并设计客观与主观双评估协议，推动端到端医学报告解析技术发展。**

- **链接: [http://arxiv.org/pdf/2508.16674v1](http://arxiv.org/pdf/2508.16674v1)**

> **作者:** Fangxin Shang; Yuan Xia; Dalu Yang; Yahui Wang; Binglin Yang
>
> **摘要:** Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding.
>
---
#### [new 134] RephraseTTS: Dynamic Length Text based Speech Insertion with Speaker Style Transfer
- **分类: cs.SD; cs.CL**

- **简介: 论文提出RephraseTTS方法，解决文本条件下的语音插入任务，即根据文本修改语音内容并保持原说话人特征。采用基于Transformer的非自回归模型，动态确定插入长度，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.17031v1](http://arxiv.org/pdf/2508.17031v1)**

> **作者:** Neeraj Matiyali; Siddharth Srivastava; Gaurav Sharma
>
> **摘要:** We propose a method for the task of text-conditioned speech insertion, i.e. inserting a speech sample in an input speech sample, conditioned on the corresponding complete text transcript. An example use case of the task would be to update the speech audio when corrections are done on the corresponding text transcript. The proposed method follows a transformer-based non-autoregressive approach that allows speech insertions of variable lengths, which are dynamically determined during inference, based on the text transcript and tempo of the available partial input. It is capable of maintaining the speaker's voice characteristics, prosody and other spectral properties of the available speech input. Results from our experiments and user study on LibriTTS show that our method outperforms baselines based on an existing adaptive text to speech method. We also provide numerous qualitative results to appreciate the quality of the output from the proposed method.
>
---
#### [new 135] How Do LLM-Generated Texts Impact Term-Based Retrieval Models?
- **分类: cs.IR; cs.CL**

- **简介: 论文研究LLM生成文本对词项检索模型（如BM25）的影响。任务是评估此类模型在混合人类与机器生成文本中的表现。发现LLM文本具更高术语特异性和多样性，但模型不偏袒来源，而是偏好词分布匹配查询的文档。**

- **链接: [http://arxiv.org/pdf/2508.17715v1](http://arxiv.org/pdf/2508.17715v1)**

> **作者:** Wei Huang; Keping Bi; Yinqiong Cai; Wei Chen; Jiafeng Guo; Xueqi Cheng
>
> **摘要:** As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content.
>
---
#### [new 136] Leveraging Multi-Source Textural UGC for Neighbourhood Housing Quality Assessment: A GPT-Enhanced Framework
- **分类: cs.CY; cs.CL; I.2.7; K.4.1**

- **简介: 论文提出基于GPT-4o分析多源UGC文本，评估社区住房质量，解决传统方法主观性强、数据单一问题。构建46项指标体系，实现客观量化评估，提升城市治理精准性。**

- **链接: [http://arxiv.org/pdf/2508.16657v1](http://arxiv.org/pdf/2508.16657v1)**

> **作者:** Qiyuan Hong; Huimin Zhao; Ying Long
>
> **备注:** 6 pages, 3 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
>
> **摘要:** This study leverages GPT-4o to assess neighbourhood housing quality using multi-source textural user-generated content (UGC) from Dianping, Weibo, and the Government Message Board. The analysis involves filtering relevant texts, extracting structured evaluation units, and conducting sentiment scoring. A refined housing quality assessment system with 46 indicators across 11 categories was developed, highlighting an objective-subjective method gap and platform-specific differences in focus. GPT-4o outperformed rule-based and BERT models, achieving 92.5% accuracy in fine-tuned settings. The findings underscore the value of integrating UGC and GPT-driven analysis for scalable, resident-centric urban assessments, offering practical insights for policymakers and urban planners.
>
---
#### [new 137] Hyperbolic Multimodal Representation Learning for Biological Taxonomies
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 论文研究生物分类中的多模态数据嵌入问题，提出基于双曲空间的表示学习方法，以更好建模生物分类的层次结构。通过对比和新型堆叠蕴含目标，在BIOSCAN-1M数据集上实现更优的未见物种分类性能，提升生态监测与保护应用潜力。**

- **链接: [http://arxiv.org/pdf/2508.16744v1](http://arxiv.org/pdf/2508.16744v1)**

> **作者:** ZeMing Gong; Chuanqi Tang; Xiaoliang Huo; Nicholas Pellegrino; Austin T. Wang; Graham W. Taylor; Angel X. Chang; Scott C. Lowe; Joakim Bruslund Haurum
>
> **摘要:** Taxonomic classification in biodiversity research involves organizing biological specimens into structured hierarchies based on evidence, which can come from multiple modalities such as images and genetic information. We investigate whether hyperbolic networks can provide a better embedding space for such hierarchical models. Our method embeds multimodal inputs into a shared hyperbolic space using contrastive and a novel stacked entailment-based objective. Experiments on the BIOSCAN-1M dataset show that hyperbolic embedding achieves competitive performance with Euclidean baselines, and outperforms all other models on unseen species classification using DNA barcodes. However, fine-grained classification and open-world generalization remain challenging. Our framework offers a structure-aware foundation for biodiversity modelling, with potential applications to species discovery, ecological monitoring, and conservation efforts.
>
---
#### [new 138] Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications
- **分类: cs.RO; cs.AI; cs.CL; cs.HC**

- **简介: 论文研究语音识别在人机交互中的应用，解决真实场景下音频质量差、用户多样性导致的识别难题。通过评估四个前沿ASR系统在八大数据集上的表现，揭示性能差异、幻觉倾向和偏见问题，为HRI中的可靠语音交互提供实证依据。**

- **链接: [http://arxiv.org/pdf/2508.17753v1](http://arxiv.org/pdf/2508.17753v1)**

> **作者:** Theresa Pekarek Rosin; Julia Gachot; Henri-Leon Kordt; Matthias Kerzel; Stefan Wermter
>
> **备注:** Accepted at the workshop on Foundation Models for Social Robotics (FoMoSR) at ICSR 2025
>
> **摘要:** Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety.
>
---
#### [new 139] Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 论文提出一种自适应记忆框架，优化LLM代理的记忆能力。解决人工预设记忆效率低、忽略记忆循环的问题。通过MoE门控、可学习聚合和任务反射机制，实现数据驱动的记忆学习与环境适配。**

- **链接: [http://arxiv.org/pdf/2508.16629v1](http://arxiv.org/pdf/2508.16629v1)**

> **作者:** Zeyu Zhang; Quanyu Dai; Rui Li; Xiaohe Bo; Xu Chen; Zhenhua Dong
>
> **备注:** 17 pages, 4 figures, 5 tables
>
> **摘要:** LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at https://github.com/nuster1128/learn_to_memorize.
>
---
#### [new 140] Revisiting Rule-Based Stuttering Detection: A Comprehensive Analysis of Interpretable Models for Clinical Applications
- **分类: cs.AI; cs.CL**

- **简介: 论文针对临床场景下的口吃检测任务，解决深度学习模型缺乏可解释性的问题。作者提出一种增强的规则基框架，结合语音速率归一化与多级特征分析，在保持高可解释性的同时实现稳定准确的口吃检测，尤其在延长音检测上表现优异。**

- **链接: [http://arxiv.org/pdf/2508.16681v1](http://arxiv.org/pdf/2508.16681v1)**

> **作者:** Eric Zhang
>
> **摘要:** Stuttering affects approximately 1% of the global population, impacting communication and quality of life. While recent advances in deep learning have pushed the boundaries of automatic speech dysfluency detection, rule-based approaches remain crucial for clinical applications where interpretability and transparency are paramount. This paper presents a comprehensive analysis of rule-based stuttering detection systems, synthesizing insights from multiple corpora including UCLASS, FluencyBank, and SEP-28k. We propose an enhanced rule-based framework that incorporates speaking-rate normalization, multi-level acoustic feature analysis, and hierarchical decision structures. Our approach achieves competitive performance while maintaining complete interpretability-critical for clinical adoption. We demonstrate that rule-based systems excel particularly in prolongation detection (97-99% accuracy) and provide stable performance across varying speaking rates. Furthermore, we show how these interpretable models can be integrated with modern machine learning pipelines as proposal generators or constraint modules, bridging the gap between traditional speech pathology practices and contemporary AI systems. Our analysis reveals that while neural approaches may achieve marginally higher accuracy in unconstrained settings, rule-based methods offer unique advantages in clinical contexts where decision auditability, patient-specific tuning, and real-time feedback are essential.
>
---
#### [new 141] Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits of LVLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出MMCRICBENCH-3K基准，用于评估大视觉语言模型在板球记分卡上的数值和跨语言推理能力。任务为视觉问答，解决结构化数据理解与跨语言泛化难题。工作包括构建含英、印地语记分卡的图像与问题对，并验证现有模型在此任务上的局限性。**

- **链接: [http://arxiv.org/pdf/2508.17334v1](http://arxiv.org/pdf/2508.17334v1)**

> **作者:** Somraj Gautam; Abhirama Subramanyam Penamakuri; Abhishek Bhandari; Gaurav Harit
>
> **摘要:** We introduce MMCRICBENCH-3K, a benchmark for Visual Question Answering (VQA) on cricket scorecards, designed to evaluate large vision-language models (LVLMs) on complex numerical and cross-lingual reasoning over semi-structured tabular images. MMCRICBENCH-3K comprises 1,463 synthetically generated scorecard images from ODI, T20, and Test formats, accompanied by 1,500 English QA pairs. It includes two subsets: MMCRICBENCH-E-1.5K, featuring English scorecards, and MMCRICBENCH-H-1.5K, containing visually similar Hindi scorecards, with all questions and answers kept in English to enable controlled cross-script evaluation. The task demands reasoning over structured numerical data, multi-image context, and implicit domain knowledge. Empirical results show that even state-of-the-art LVLMs, such as GPT-4o and Qwen2.5VL, struggle on the English subset despite it being their primary training language and exhibit a further drop in performance on the Hindi subset. This reveals key limitations in structure-aware visual text understanding, numerical reasoning, and cross-lingual generalization. The dataset is publicly available via Hugging Face at https://huggingface.co/datasets/DIALab/MMCricBench, to promote LVLM research in this direction.
>
---
#### [new 142] Named Entity Recognition of Historical Text via Large Language Model
- **分类: cs.DL; cs.AI; cs.CL**

- **简介: 论文研究历史文本中的命名实体识别任务，针对标注数据稀缺问题，探索使用大语言模型的零样本和少样本提示策略，实验表明该方法在HIPE-2022数据集上表现良好，为低资源历史文本信息提取提供了有效方案。**

- **链接: [http://arxiv.org/pdf/2508.18090v1](http://arxiv.org/pdf/2508.18090v1)**

> **作者:** Shibingfeng Zhang; Giovanni Colavizza
>
> **摘要:** Large language models have demonstrated remarkable versatility across a wide range of natural language processing tasks and domains. One such task is Named Entity Recognition (NER), which involves identifying and classifying proper names in text, such as people, organizations, locations, dates, and other specific entities. NER plays a crucial role in extracting information from unstructured textual data, enabling downstream applications such as information retrieval from unstructured text. Traditionally, NER is addressed using supervised machine learning approaches, which require large amounts of annotated training data. However, historical texts present a unique challenge, as the annotated datasets are often scarce or nonexistent, due to the high cost and expertise required for manual labeling. In addition, the variability and noise inherent in historical language, such as inconsistent spelling and archaic vocabulary, further complicate the development of reliable NER systems for these sources. In this study, we explore the feasibility of applying LLMs to NER in historical documents using zero-shot and few-shot prompting strategies, which require little to no task-specific training data. Our experiments, conducted on the HIPE-2022 (Identifying Historical People, Places and other Entities) dataset, show that LLMs can achieve reasonably strong performance on NER tasks in this setting. While their performance falls short of fully supervised models trained on domain-specific annotations, the results are nevertheless promising. These findings suggest that LLMs offer a viable and efficient alternative for information extraction in low-resource or historically significant corpora, where traditional supervised methods are infeasible.
>
---
#### [new 143] Interpreting the Effects of Quantization on LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究量化对大语言模型（LLM）内部表示的影响，旨在评估其可靠性。通过多种可解释性方法分析4-bit与8-bit量化下模型行为，发现量化对校准和死神经元影响小，但对冗余性影响因模型而异，表明量化仍是可靠压缩技术。**

- **链接: [http://arxiv.org/pdf/2508.16785v1](http://arxiv.org/pdf/2508.16785v1)**

> **作者:** Manpreet Singh; Hassan Sajjad
>
> **摘要:** Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
>
---
#### [new 144] LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于综述任务，旨在解决LLM代理推理框架分类与应用场景不清晰的问题。作者提出统一分类体系，系统梳理单代理、工具增强和多代理方法，分析其在科学发现等场景的应用特征与评估策略。**

- **链接: [http://arxiv.org/pdf/2508.17692v1](http://arxiv.org/pdf/2508.17692v1)**

> **作者:** Bingxi Zhao; Lin Geng Foo; Ping Hu; Christian Theobalt; Hossein Rahmani; Jun Liu
>
> **备注:** 51 pages,10 figures,8 tables. Work in progress
>
> **摘要:** Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks.
>
---
#### [new 145] The AI Data Scientist
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出AI Data Scientist，一个由多个LLM子代理组成的自主智能体，旨在将数据科学流程自动化，快速提供可操作的洞察。它解决传统数据分析耗时长、门槛高的问题，通过协同完成数据清洗、统计检验、建模和解释，实现端到端的智能决策支持。**

- **链接: [http://arxiv.org/pdf/2508.18113v1](http://arxiv.org/pdf/2508.18113v1)**

> **作者:** Farkhad Akimov; Munachiso Samuel Nwadike; Zangir Iklassov; Martin Takáč
>
> **摘要:** Imagine decision-makers uploading data and, within minutes, receiving clear, actionable insights delivered straight to their fingertips. That is the promise of the AI Data Scientist, an autonomous Agent powered by large language models (LLMs) that closes the gap between evidence and action. Rather than simply writing code or responding to prompts, it reasons through questions, tests ideas, and delivers end-to-end insights at a pace far beyond traditional workflows. Guided by the scientific tenet of the hypothesis, this Agent uncovers explanatory patterns in data, evaluates their statistical significance, and uses them to inform predictive modeling. It then translates these results into recommendations that are both rigorous and accessible. At the core of the AI Data Scientist is a team of specialized LLM Subagents, each responsible for a distinct task such as data cleaning, statistical testing, validation, and plain-language communication. These Subagents write their own code, reason about causality, and identify when additional data is needed to support sound conclusions. Together, they achieve in minutes what might otherwise take days or weeks, enabling a new kind of interaction that makes deep data science both accessible and actionable.
>
---
#### [new 146] Recall-Extend Dynamics: Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出RED框架，通过控制探索与精炼离线数据集成，提升小语言模型的推理能力。解决小模型探索空间小、离线数据冗余及分布差异问题，结合熵监控与策略迁移机制实现高效增强。**

- **链接: [http://arxiv.org/pdf/2508.16677v1](http://arxiv.org/pdf/2508.16677v1)**

> **作者:** Zhong Guan; Likang Wu; Hongke Zhao; Jiahui Wang; Le Wu
>
> **摘要:** Many existing studies have achieved significant improvements in the reasoning capabilities of large language models (LLMs) through reinforcement learning with verifiable rewards (RLVR), while the enhancement of reasoning abilities in small language models (SLMs) has not yet been sufficiently explored. Combining distilled data from larger models with RLVR on small models themselves is a natural approach, but it still faces various challenges and issues. Therefore, we propose \textit{\underline{R}}ecall-\textit{\underline{E}}xtend \textit{\underline{D}}ynamics(RED): Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration. In this paper, we explore the perspective of varying exploration spaces, balancing offline distillation with online reinforcement learning. Simultaneously, we specifically design and optimize for the insertion problem within offline data. By monitoring the ratio of entropy changes in the model concerning offline and online data, we regulate the weight of offline-SFT, thereby addressing the issues of insufficient exploration space in small models and the redundancy and complexity during the distillation process. Furthermore, to tackle the distribution discrepancies between offline data and the current policy, we design a sample-accuracy-based policy shift mechanism that dynamically chooses between imitating offline distilled data and learning from its own policy.
>
---
#### [new 147] Activation Transport Operators
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出激活传输算子（ATO），用于分析Transformer解码器中特征在残差流中的线性传播机制。解决特征如何在线性层间传递的问题，通过无微调的轻量计算方法，量化线性传输效率与子空间大小，助力模型安全与调试。**

- **链接: [http://arxiv.org/pdf/2508.17540v1](http://arxiv.org/pdf/2508.17540v1)**

> **作者:** Andrzej Szablewski; Marek Masiak
>
> **备注:** 4 pages, 4 figures, references and appendices
>
> **摘要:** The residual stream mediates communication between transformer decoder layers via linear reads and writes of non-linear computations. While sparse-dictionary learning-based methods locate features in the residual stream, and activation patching methods discover circuits within the model, the mechanism by which features flow through the residual stream remains understudied. Understanding this dynamic can better inform jailbreaking protections, enable early detection of model mistakes, and their correction. In this work, we propose Activation Transport Operators (ATO), linear maps from upstream to downstream residuals $k$ layers later, evaluated in feature space using downstream SAE decoder projections. We empirically demonstrate that these operators can determine whether a feature has been linearly transported from a previous layer or synthesised from non-linear layer computation. We develop the notion of transport efficiency, for which we provide an upper bound, and use it to estimate the size of the residual stream subspace that corresponds to linear transport. We empirically demonstrate the linear transport, report transport efficiency and the size of the residual stream's subspace involved in linear transport. This compute-light (no finetuning, <50 GPU-h) method offers practical tools for safety, debugging, and a clearer picture of where computation in LLMs behaves linearly.
>
---
#### [new 148] Multi-Agent Visual-Language Reasoning for Comprehensive Highway Scene Understanding
- **分类: cs.CV; cs.AI; cs.CL; eess.IV**

- **简介: 论文提出多智能体视觉语言推理框架，用于高速公路场景理解，解决多任务感知难题。通过大模型生成细粒度提示引导小模型高效推理，实现天气、路面湿滑度和拥堵等任务的精准检测与部署。**

- **链接: [http://arxiv.org/pdf/2508.17205v1](http://arxiv.org/pdf/2508.17205v1)**

> **作者:** Yunxiang Yang; Ningning Xu; Jidong J. Yang
>
> **备注:** 16 pages, 16 figures, 8 tables
>
> **摘要:** This paper introduces a multi-agent framework for comprehensive highway scene understanding, designed around a mixture-of-experts strategy. In this framework, a large generic vision-language model (VLM), such as GPT-4o, is contextualized with domain knowledge to generates task-specific chain-of-thought (CoT) prompts. These fine-grained prompts are then used to guide a smaller, efficient VLM (e.g., Qwen2.5-VL-7B) in reasoning over short videos, along with complementary modalities as applicable. The framework simultaneously addresses multiple critical perception tasks, including weather classification, pavement wetness assessment, and traffic congestion detection, achieving robust multi-task reasoning while balancing accuracy and computational efficiency. To support empirical validation, we curated three specialized datasets aligned with these tasks. Notably, the pavement wetness dataset is multimodal, combining video streams with road weather sensor data, highlighting the benefits of multimodal reasoning. Experimental results demonstrate consistently strong performance across diverse traffic and environmental conditions. From a deployment perspective, the framework can be readily integrated with existing traffic camera systems and strategically applied to high-risk rural locations, such as sharp curves, flood-prone lowlands, or icy bridges. By continuously monitoring the targeted sites, the system enhances situational awareness and delivers timely alerts, even in resource-constrained environments.
>
---
#### [new 149] CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对大视觉语言模型（LVLM）中视觉token冗余导致的高计算成本问题，提出CoViPAL方法。通过轻量级、模型无关的插件模块，在每层预测并剪枝冗余视觉token，提升推理效率且不损失精度。**

- **链接: [http://arxiv.org/pdf/2508.17243v1](http://arxiv.org/pdf/2508.17243v1)**

> **作者:** Zicong Tang; Ziyang Ma; Suqing Wang; Zuchao Li; Lefei Zhang; Hai Zhao; Yun Li; Qianren Wang
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy.
>
---
#### [new 150] Attention Layers Add Into Low-Dimensional Residual Subspaces
- **分类: cs.LG; cs.CL**

- **简介: 论文研究注意力机制中激活值的低维结构，揭示其导致稀疏字典学习中的死特征问题。提出在激活子空间内初始化特征方向的方法，显著减少死特征比例，提升稀疏自动编码器性能。**

- **链接: [http://arxiv.org/pdf/2508.16929v1](http://arxiv.org/pdf/2508.16929v1)**

> **作者:** Junxuan Wang; Xuyang Ge; Wentao Shu; Zhengfu He; Xipeng Qiu
>
> **摘要:** While transformer models are widely believed to operate in high-dimensional hidden spaces, we show that attention outputs are confined to a surprisingly low-dimensional subspace, where about 60\% of the directions account for 99\% of the variance--a phenomenon that is induced by the attention output projection matrix and consistently observed across diverse model families and datasets. Critically, we find this low-rank structure as a fundamental cause of the prevalent dead feature problem in sparse dictionary learning, where it creates a mismatch between randomly initialized features and the intrinsic geometry of the activation space. Building on this insight, we propose a subspace-constrained training method for sparse autoencoders (SAEs), initializing feature directions into the active subspace of activations. Our approach reduces dead features from 87\% to below 1\% in Attention Output SAEs with 1M features, and can further extend to other sparse dictionary learning methods. Our findings provide both new insights into the geometry of attention and practical tools for improving sparse dictionary learning in large language models.
>
---
#### [new 151] Dynamic Embedding of Hierarchical Visual Features for Efficient Vision-Language Fine-Tuning
- **分类: cs.CV; cs.CL**

- **简介: 论文提出DEHVF方法，用于高效视觉语言模型微调。针对传统方法因序列过长导致计算开销大的问题，通过动态融合多层视觉特征，在不扩展序列长度的前提下实现跨模态精准对齐与互补。**

- **链接: [http://arxiv.org/pdf/2508.17638v1](http://arxiv.org/pdf/2508.17638v1)**

> **作者:** Xinyu Wei; Guoli Yang; Jialu Zhou; Mingyue Yang; Leqian Li; Kedi Zhang; Chunping Qiu
>
> **摘要:** Large Vision-Language Models (LVLMs) commonly follow a paradigm that projects visual features and then concatenates them with text tokens to form a unified sequence input for Large Language Models (LLMs). However, this paradigm leads to a significant increase in the length of the input sequence, resulting in substantial computational overhead. Existing methods attempt to fuse visual information into the intermediate layers of LLMs, which alleviate the sequence length issue but often neglect the hierarchical semantic representations within the model and the fine-grained visual information available in the shallower visual encoding layers. To address this limitation, we propose DEHVF, an efficient vision-language fine-tuning method based on dynamic embedding and fusion of hierarchical visual features. Its core lies in leveraging the inherent hierarchical representation characteristics of visual encoders and language models. Through a lightweight hierarchical visual fuser, it dynamically selects and fuses hierarchical features corresponding to semantic granularity based on the internal representations of each layer in LLMs. The fused layer-related visual features are then projected and aligned before being directly embedded into the Feed-Forward Network (FFN) of the corresponding layer in LLMs. This approach not only avoids sequence expansion but also dynamically fuses multi-layer visual information. By fine-tuning only a small number of parameters, DEHVF achieves precise alignment and complementarity of cross-modal information at the same semantic granularity. We conducted experiments across various VL benchmarks, including visual question answering on ScienceQA and image captioning on COCO Captions. The results demonstrate that DEHVF achieves higher accuracy than existing parameter-efficient fine-tuning (PEFT) baselines while maintaining efficient training and inference.
>
---
## 更新

#### [replaced 001] CAARMA: Class Augmentation with Adversarial Mixup Regularization
- **分类: cs.SD; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16718v2](http://arxiv.org/pdf/2503.16718v2)**

> **作者:** Massa Baali; Xiang Li; Hao Chen; Syed Abdul Hannan; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. The code is available at: https://github.com/massabaali7/CAARMA/
>
---
#### [replaced 002] Reasoning with RAGged events: RAG-Enhanced Event Knowledge Base Construction and reasoning with proof-assistants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07042v3](http://arxiv.org/pdf/2506.07042v3)**

> **作者:** Stergios Chatzikyriakidis
>
> **摘要:** Extracting structured computational representations of historical events from narrative text remains computationally expensive when constructed manually. While RDF/OWL reasoners enable graph-based reasoning, they are limited to fragments of first-order logic, preventing deeper temporal and semantic analysis. This paper addresses both challenges by developing automatic historical event extraction models using multiple LLMs (GPT-4, Claude, Llama 3.2) with three enhancement strategies: pure base generation, knowledge graph enhancement, and Retrieval-Augmented Generation (RAG). We conducted comprehensive evaluations using historical texts from Thucydides. Our findings reveal that enhancement strategies optimize different performance dimensions rather than providing universal improvements. For coverage and historical breadth, base generation achieves optimal performance with Claude and GPT-4 extracting comprehensive events. However, for precision, RAG enhancement improves coordinate accuracy and metadata completeness. Model architecture fundamentally determines enhancement sensitivity: larger models demonstrate robust baseline performance with incremental RAG improvements, while Llama 3.2 shows extreme variance from competitive performance to complete failure. We then developed an automated translation pipeline converting extracted RDF representations into Coq proof assistant specifications, enabling higher-order reasoning beyond RDF capabilities including multi-step causal verification, temporal arithmetic with BC dates, and formal proofs about historical causation. The Coq formalization validates that RAG-discovered event types represent legitimate domain-specific semantic structures rather than ontological violations.
>
---
#### [replaced 003] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.13237v3](http://arxiv.org/pdf/2505.13237v3)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025 (Oral). Update acknowledgement in this version. Project page: https://github.com/ckyang1124/SAKURA
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [replaced 004] Towards Controllable Speech Synthesis in the Era of Large Language Models: A Systematic Survey
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.06602v3](http://arxiv.org/pdf/2412.06602v3)**

> **作者:** Tianxin Xie; Yan Rong; Pengfei Zhang; Wenwu Wang; Li Liu
>
> **备注:** The first comprehensive survey on controllable TTS. Accepted to the EMNLP 2025 main conference
>
> **摘要:** Text-to-speech (TTS) has advanced from generating natural-sounding speech to enabling fine-grained control over attributes like emotion, timbre, and style. Driven by rising industrial demand and breakthroughs in deep learning, e.g., diffusion and large language models (LLMs), controllable TTS has become a rapidly growing research area. This survey provides the first comprehensive review of controllable TTS methods, from traditional control techniques to emerging approaches using natural language prompts. We categorize model architectures, control strategies, and feature representations, while also summarizing challenges, datasets, and evaluations in controllable TTS. This survey aims to guide researchers and practitioners by offering a clear taxonomy and highlighting future directions in this fast-evolving field. One can visit https://github.com/imxtx/awesome-controllabe-speech-synthesis for a comprehensive paper list and updates.
>
---
#### [replaced 005] Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2312.10987v4](http://arxiv.org/pdf/2312.10987v4)**

> **作者:** Congchi Yin; Qian Yu; Zhiwei Fang; Changping Peng; Piji Li
>
> **摘要:** Recent major milestones have successfully reconstructed natural language from non-invasive brain signals (e.g. functional Magnetic Resonance Imaging (fMRI) and Electroencephalogram (EEG)) across subjects. However, we find current dataset splitting strategies for cross-subject brain-to-text decoding are wrong. Specifically, we first demonstrate that all current splitting methods suffer from data leakage problem, which refers to the leakage of validation and test data into training set, resulting in significant overfitting and overestimation of decoding models. In this study, we develop a right cross-subject data splitting criterion without data leakage for decoding fMRI and EEG signal to text. Some SOTA brain-to-text decoding models are re-evaluated correctly with the proposed criterion for further research.
>
---
#### [replaced 006] DPad: Efficient Diffusion Language Models with Suffix Dropout
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.14148v2](http://arxiv.org/pdf/2508.14148v2)**

> **作者:** Xinhua Chen; Sitao Huang; Cong Guo; Chiyue Wei; Yintao He; Jianyi Zhang; Hai "Helen" Li; Yiran Chen
>
> **摘要:** Diffusion-based Large Language Models (dLLMs) parallelize text generation by framing decoding as a denoising process, but suffer from high computational overhead since they predict all future suffix tokens at each step while retaining only a small fraction. We propose Diffusion Scratchpad (DPad), a training-free method that restricts attention to a small set of nearby suffix tokens, preserving fidelity while eliminating redundancy. DPad integrates two strategies: (i) a sliding window, which maintains a fixed-length suffix window, and (ii) distance-decay dropout, which deterministically removes distant suffix tokens before attention computation. This simple design is compatible with existing optimizations such as prefix caching and can be implemented with only a few lines of code. Comprehensive evaluations across multiple benchmarks on LLaDA-1.5 and Dream models demonstrate that DPad delivers up to $\mathbf{61.4\times}$ speedup over vanilla dLLMs while maintaining comparable accuracy, highlighting its potential for efficient and scalable long-sequence inference. Our code is available at https://github.com/Crys-Chen/DPad.
>
---
#### [replaced 007] CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11742v2](http://arxiv.org/pdf/2507.11742v2)**

> **作者:** Meng Li; Timothy M. McPhillips; Dingmin Wang; Shin-Rong Tsai; Bertram Ludäscher
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O set$\unicode{x2014}$the flows of information into or out of cells via variables$\unicode{x2014}$then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
>
---
#### [replaced 008] Self-Correcting Code Generation Using Small Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23060v3](http://arxiv.org/pdf/2505.23060v3)**

> **作者:** Jeonghun Cho; Deokhyung Kang; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** Accepted at EMNLP 2025 (Findings, long paper)
>
> **摘要:** Self-correction has demonstrated potential in code generation by allowing language models to revise and improve their outputs through successive refinement. Recent studies have explored prompting-based strategies that incorporate verification or feedback loops using proprietary models, as well as training-based methods that leverage their strong reasoning capabilities. However, whether smaller models possess the capacity to effectively guide their outputs through self-reflection remains unexplored. Our findings reveal that smaller models struggle to exhibit reflective revision behavior across both self-correction paradigms. In response, we introduce CoCoS, an approach designed to enhance the ability of small language models for multi-turn code correction. Specifically, we propose an online reinforcement learning objective that trains the model to confidently maintain correct outputs while progressively correcting incorrect outputs as turns proceed. Our approach features an accumulated reward function that aggregates rewards across the entire trajectory and a fine-grained reward better suited to multi-turn correction scenarios. This facilitates the model in enhancing initial response quality while achieving substantial improvements through self-correction. With 1B-scale models, CoCoS achieves improvements of 35.8% on the MBPP and 27.7% on HumanEval compared to the baselines.
>
---
#### [replaced 009] SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10624v4](http://arxiv.org/pdf/2410.10624v4)**

> **作者:** Zechen Li; Shohreh Deldari; Linyao Chen; Hao Xue; Flora D. Salim
>
> **备注:** Accepted by EMNLP 2025 Main Conference
>
> **摘要:** We introduce SensorLLM, a two-stage framework that enables Large Language Models (LLMs) to perform human activity recognition (HAR) from sensor time-series data. Despite their strong reasoning and generalization capabilities, LLMs remain underutilized for motion sensor data due to the lack of semantic context in time-series, computational constraints, and challenges in processing numerical inputs. SensorLLM addresses these limitations through a Sensor-Language Alignment stage, where the model aligns sensor inputs with trend descriptions. Special tokens are introduced to mark channel boundaries. This alignment enables LLMs to capture numerical variations, channel-specific features, and data of varying durations, without requiring human annotations. In the subsequent Task-Aware Tuning stage, we refine the model for HAR classification, achieving performance that matches or surpasses state-of-the-art methods. Our results demonstrate that SensorLLM evolves into an effective sensor learner, reasoner, and classifier through human-intuitive Sensor-Language Alignment, generalizing across diverse HAR datasets. We believe this work establishes a foundation for future research on time-series and text alignment, paving the way for foundation models in sensor data analysis. Our codes are available at https://github.com/zechenli03/SensorLLM.
>
---
#### [replaced 010] Localizing Factual Inconsistencies in Attributable Text Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07473v2](http://arxiv.org/pdf/2410.07473v2)**

> **作者:** Arie Cattan; Paul Roit; Shiyue Zhang; David Wan; Roee Aharoni; Idan Szpektor; Mohit Bansal; Ido Dagan
>
> **备注:** Accepted for publication in Transactions of the Association for Computational Linguistics (TACL), 2025. Authors pre-print
>
> **摘要:** There has been an increasing interest in detecting hallucinations in model-generated texts, both manually and automatically, at varying levels of granularity. However, most existing methods fail to precisely pinpoint the errors. In this work, we introduce QASemConsistency, a new formalism for localizing factual inconsistencies in attributable text generation, at a fine-grained level. Drawing inspiration from Neo-Davidsonian formal semantics, we propose decomposing the generated text into minimal predicate-argument level propositions, expressed as simple question-answer (QA) pairs, and assess whether each individual QA pair is supported by a trusted reference text. As each QA pair corresponds to a single semantic relation between a predicate and an argument, QASemConsistency effectively localizes the unsupported information. We first demonstrate the effectiveness of the QASemConsistency methodology for human annotation, by collecting crowdsourced annotations of granular consistency errors, while achieving a substantial inter-annotator agreement. This benchmark includes more than 3K instances spanning various tasks of attributable text generation. We also show that QASemConsistency yields factual consistency scores that correlate well with human judgments. Finally, we implement several methods for automatically detecting localized factual inconsistencies, with both supervised entailment models and LLMs.
>
---
#### [replaced 011] SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11857v2](http://arxiv.org/pdf/2508.11857v2)**

> **作者:** Andrei-Valentin Tănase; Elena Pelican
>
> **摘要:** Tokenization remains a fundamental yet underexplored bottleneck in natural language processing, with strategies largely static despite remarkable progress in model architectures. We present SupraTok, a novel tokenization architecture that reimagines subword segmentation through three innovations: cross-boundary pattern learning that discovers multi-word semantic units, entropy-driven data curation that optimizes training corpus quality, and multi-phase curriculum learning for stable convergence. Our approach extends Byte-Pair Encoding by learning "superword" tokens, coherent multi-word expressions that preserve semantic unity while maximizing compression efficiency. SupraTok achieves 31% improvement in English tokenization efficiency (5.91 versus 4.51 characters per token) compared to OpenAI's o200k tokenizer and 30% improvement over Google's Gemma 3 tokenizer (256k vocabulary), while maintaining competitive performance across 38 languages. When integrated with a GPT-2 scale model (124M parameters) trained on 10 billion tokens from the FineWeb-Edu dataset, SupraTok yields 8.4% improvement on HellaSWAG and 9.5% on MMLU benchmarks without architectural modifications. While these results are promising at this scale, further validation at larger model scales is needed. These findings suggest that efficient tokenization can complement architectural innovations as a path to improved language model performance.
>
---
#### [replaced 012] Evaluating Scoring Bias in LLM-as-a-Judge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22316v2](http://arxiv.org/pdf/2506.22316v2)**

> **作者:** Qingquan Li; Shaoyu Dou; Kailai Shao; Chao Chen; Haixiang Hu
>
> **摘要:** The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
>
---
#### [replaced 013] Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10142v3](http://arxiv.org/pdf/2508.10142v3)**

> **作者:** Kartikeya Badola; Jonathan Simon; Arian Hosseini; Sara Marie Mc Carthy; Tsendsuren Munkhdalai; Abhimanyu Goyal; Tomáš Kočiský; Shyam Upadhyay; Bahare Fatemi; Mehran Kazemi
>
> **摘要:** Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities.
>
---
#### [replaced 014] Intern-S1: A Scientific Multimodal Foundation Model
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15763v2](http://arxiv.org/pdf/2508.15763v2)**

> **作者:** Lei Bai; Zhongrui Cai; Yuhang Cao; Maosong Cao; Weihan Cao; Chiyu Chen; Haojiong Chen; Kai Chen; Pengcheng Chen; Ying Chen; Yongkang Chen; Yu Cheng; Pei Chu; Tao Chu; Erfei Cui; Ganqu Cui; Long Cui; Ziyun Cui; Nianchen Deng; Ning Ding; Nanqing Dong; Peijie Dong; Shihan Dou; Sinan Du; Haodong Duan; Caihua Fan; Ben Gao; Changjiang Gao; Jianfei Gao; Songyang Gao; Yang Gao; Zhangwei Gao; Jiaye Ge; Qiming Ge; Lixin Gu; Yuzhe Gu; Aijia Guo; Qipeng Guo; Xu Guo; Conghui He; Junjun He; Yili Hong; Siyuan Hou; Caiyu Hu; Hanglei Hu; Jucheng Hu; Ming Hu; Zhouqi Hua; Haian Huang; Junhao Huang; Xu Huang; Zixian Huang; Zhe Jiang; Lingkai Kong; Linyang Li; Peiji Li; Pengze Li; Shuaibin Li; Tianbin Li; Wei Li; Yuqiang Li; Dahua Lin; Junyao Lin; Tianyi Lin; Zhishan Lin; Hongwei Liu; Jiangning Liu; Jiyao Liu; Junnan Liu; Kai Liu; Kaiwen Liu; Kuikun Liu; Shichun Liu; Shudong Liu; Wei Liu; Xinyao Liu; Yuhong Liu; Zhan Liu; Yinquan Lu; Haijun Lv; Hongxia Lv; Huijie Lv; Qitan Lv; Ying Lv; Chengqi Lyu; Chenglong Ma; Jianpeng Ma; Ren Ma; Runmin Ma; Runyuan Ma; Xinzhu Ma; Yichuan Ma; Zihan Ma; Sixuan Mi; Junzhi Ning; Wenchang Ning; Xinle Pang; Jiahui Peng; Runyu Peng; Yu Qiao; Jiantao Qiu; Xiaoye Qu; Yuan Qu; Yuchen Ren; Fukai Shang; Wenqi Shao; Junhao Shen; Shuaike Shen; Chunfeng Song; Demin Song; Diping Song; Chenlin Su; Weijie Su; Weigao Sun; Yu Sun; Qian Tan; Cheng Tang; Huanze Tang; Kexian Tang; Shixiang Tang; Jian Tong; Aoran Wang; Bin Wang; Dong Wang; Lintao Wang; Rui Wang; Weiyun Wang; Wenhai Wang; Jiaqi Wang; Yi Wang; Ziyi Wang; Ling-I Wu; Wen Wu; Yue Wu; Zijian Wu; Linchen Xiao; Shuhao Xing; Chao Xu; Huihui Xu; Jun Xu; Ruiliang Xu; Wanghan Xu; GanLin Yang; Yuming Yang; Haochen Ye; Jin Ye; Shenglong Ye; Jia Yu; Jiashuo Yu; Jing Yu; Fei Yuan; Yuhang Zang; Bo Zhang; Chao Zhang; Chen Zhang; Hongjie Zhang; Jin Zhang; Qiaosheng Zhang; Qiuyinzhe Zhang; Songyang Zhang; Taolin Zhang; Wenlong Zhang; Wenwei Zhang; Yechen Zhang; Ziyang Zhang; Haiteng Zhao; Qian Zhao; Xiangyu Zhao; Xiangyu Zhao; Bowen Zhou; Dongzhan Zhou; Peiheng Zhou; Yuhao Zhou; Yunhua Zhou; Dongsheng Zhu; Lin Zhu; Yicheng Zou
>
> **摘要:** In recent years, a plethora of open-source foundation models have emerged, achieving remarkable progress in some widely attended fields, with performance being quite close to that of closed-source models. However, in high-value but more challenging scientific professional fields, either the fields still rely on expert models, or the progress of general foundation models lags significantly compared to those in popular areas, far from sufficient for transforming scientific research and leaving substantial gap between open-source models and closed-source models in these scientific domains. To mitigate this gap and explore a step further toward Artificial General Intelligence (AGI), we introduce Intern-S1, a specialized generalist equipped with general understanding and reasoning capabilities with expertise to analyze multiple science modal data. Intern-S1 is a multimodal Mixture-of-Experts (MoE) model with 28 billion activated parameters and 241 billion total parameters, continually pre-trained on 5T tokens, including over 2.5T tokens from scientific domains. In the post-training stage, Intern-S1 undergoes offline and then online reinforcement learning (RL) in InternBootCamp, where we propose Mixture-of-Rewards (MoR) to synergize the RL training on more than 1000 tasks simultaneously. Through integrated innovations in algorithms, data, and training systems, Intern-S1 achieved top-tier performance in online RL training. On comprehensive evaluation benchmarks, Intern-S1 demonstrates competitive performance on general reasoning tasks among open-source models and significantly outperforms open-source models in scientific domains, surpassing closed-source state-of-the-art models in professional tasks, such as molecular synthesis planning, reaction condition prediction, predicting thermodynamic stabilities for crystals. Our models are available at https://huggingface.co/internlm/Intern-S1.
>
---
#### [replaced 015] From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis
- **分类: cs.CL; cs.AI; cs.CY; cs.LO**

- **链接: [http://arxiv.org/pdf/2506.08899v2](http://arxiv.org/pdf/2506.08899v2)**

> **作者:** Elias Horner; Cristinel Mateis; Guido Governatori; Agata Ciabattoni
>
> **摘要:** We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics.
>
---
#### [replaced 016] Preliminary Ranking of WMT25 General Machine Translation Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14909v2](http://arxiv.org/pdf/2508.14909v2)**

> **作者:** Tom Kocmi; Eleftherios Avramidis; Rachel Bawden; Ondřej Bojar; Konstantin Dranch; Anton Dvorkovich; Sergey Dukanov; Natalia Fedorova; Mark Fishel; Markus Freitag; Thamme Gowda; Roman Grundkiewicz; Barry Haddow; Marzena Karpinska; Philipp Koehn; Howard Lakougna; Jessica Lundin; Kenton Murray; Masaaki Nagata; Stefano Perrella; Lorenzo Proietti; Martin Popel; Maja Popović; Parker Riley; Mariya Shmatova; Steinþór Steingrímsson; Lisa Yankovskaya; Vilém Zouhar
>
> **摘要:** We present the preliminary rankings of machine translation (MT) systems submitted to the WMT25 General Machine Translation Shared Task, as determined by automatic evaluation metrics. Because these rankings are derived from automatic evaluation, they may exhibit a bias toward systems that employ re-ranking techniques, such as Quality Estimation or Minimum Bayes Risk decoding. The official WMT25 ranking will be based on human evaluation, which is more reliable and will supersede these results. The official WMT25 ranking will be based on human evaluation, which is more reliable and will supersede these results. The purpose of releasing these findings now is to assist task participants with their system description papers; not to provide final findings.
>
---
#### [replaced 017] Beyond Semantic Similarity: Reducing Unnecessary API Calls via Behavior-Aligned Retriever
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14323v2](http://arxiv.org/pdf/2508.14323v2)**

> **作者:** Yixin Chen; Ying Xiong; Shangyu Wu; Yufei Cui; Xue Liu; Nan Guan; Chun Jason Xue
>
> **摘要:** Tool-augmented large language models (LLMs) leverage external functions to extend their capabilities, but inaccurate function calls can lead to inefficiencies and increased costs.Existing methods address this challenge by fine-tuning LLMs or using demonstration-based prompting, yet they often suffer from high training overhead and fail to account for inconsistent demonstration samples, which misguide the model's invocation behavior. In this paper, we trained a behavior-aligned retriever (BAR), which provides behaviorally consistent demonstrations to help LLMs make more accurate tool-using decisions. To train the BAR, we construct a corpus including different function-calling behaviors, i.e., calling or non-calling.We use the contrastive learning framework to train the BAR with customized positive/negative pairs and a dual-negative contrastive loss, ensuring robust retrieval of behaviorally consistent examples.Experiments demonstrate that our approach significantly reduces erroneous function calls while maintaining high task performance, offering a cost-effective and efficient solution for tool-augmented LLMs.
>
---
#### [replaced 018] Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.18462v2](http://arxiv.org/pdf/2411.18462v2)**

> **作者:** Ziyin Zhang; Jiahao Xu; Tian Liang; Xingyu Chen; Zhiwei He; Rui Wang; Zhaopeng Tu
>
> **备注:** EMNLP 2025
>
> **摘要:** Conventional speculative decoding (SD) methods utilize a predefined length policy for proposing drafts, which implies the premise that the target model smoothly accepts the proposed draft tokens. However, reality deviates from this assumption: the oracle draft length varies significantly, and the fixed-length policy hardly satisfies such a requirement. Moreover, such discrepancy is further exacerbated in scenarios involving complex reasoning and long-form generation, particularly under test-time scaling for reasoning-specialized models. Through both theoretical and empirical estimation, we establish that the discrepancy between the draft and target models can be approximated by the draft model's prediction entropy: a high entropy indicates a low acceptance rate of draft tokens, and vice versa. Based on this insight, we propose SVIP: Self-Verification Length Policy for Long-Context Speculative Decoding, which is a training-free dynamic length policy for speculative decoding systems that adaptively determines the lengths of draft sequences by referring to the draft entropy. Experimental results on mainstream SD benchmarks as well as reasoning-heavy benchmarks demonstrate the superior performance of SVIP, achieving up to 17% speedup on MT-Bench at 8K context compared with fixed draft lengths, and 22% speedup for QwQ in long-form reasoning.
>
---
#### [replaced 019] Agentic large language models improve retrieval-based radiology question answering
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00743v2](http://arxiv.org/pdf/2508.00743v2)**

> **作者:** Sebastian Wind; Jeta Sopa; Daniel Truhn; Mahshad Lotfinia; Tri-Thien Nguyen; Keno Bressem; Lisa Adams; Mirabela Rusu; Harald Köstler; Gerhard Wellein; Andreas Maier; Soroosh Tayebi Arasteh
>
> **摘要:** Clinical decision-making in radiology increasingly benefits from artificial intelligence (AI), particularly through large language models (LLMs). However, traditional retrieval-augmented generation (RAG) systems for radiology question answering (QA) typically rely on single-step retrieval, limiting their ability to handle complex clinical reasoning tasks. Here we propose an agentic RAG framework enabling LLMs to autonomously decompose radiology questions, iteratively retrieve targeted clinical evidence from Radiopaedia.org, and dynamically synthesize evidence-based responses. We evaluated 25 LLMs spanning diverse architectures, parameter scales (0.5B to >670B), and training paradigms (general-purpose, reasoning-optimized, clinically fine-tuned), using 104 expert-curated radiology questions from previously established RSNA-RadioQA and ExtendedQA datasets. To assess generalizability, we additionally tested on an unseen internal dataset of 65 real-world radiology board examination questions. Agentic retrieval significantly improved mean diagnostic accuracy over zero-shot prompting and conventional online RAG. The greatest gains occurred in small-scale models, while very large models (>200B parameters) demonstrated minimal changes (<2% improvement). Additionally, agentic retrieval reduced hallucinations (mean 9.4%) and retrieved clinically relevant context in 46% of cases, substantially aiding factual grounding. Even clinically fine-tuned models showed gains from agentic retrieval (e.g., MedGemma-27B), indicating that retrieval remains beneficial despite embedded domain knowledge. These results highlight the potential of agentic frameworks to enhance factuality and diagnostic accuracy in radiology QA, warranting future studies to validate their clinical utility. All datasets, code, and the full agentic framework are publicly available to support open research and clinical translation.
>
---
#### [replaced 020] CultureGuard: Towards Culturally-Aware Dataset and Guard Model for Multilingual Safety Applications
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01710v2](http://arxiv.org/pdf/2508.01710v2)**

> **作者:** Raviraj Joshi; Rakesh Paul; Kanishk Singla; Anusha Kamath; Michael Evans; Katherine Luna; Shaona Ghosh; Utkarsh Vaidya; Eileen Long; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** The increasing use of Large Language Models (LLMs) in agentic applications highlights the need for robust safety guard models. While content safety in English is well-studied, non-English languages lack similar advancements due to the high cost of collecting culturally aligned labeled datasets. We present CultureGuard, a novel solution for curating culturally aligned, high-quality safety datasets across multiple languages. Our approach introduces a four-stage synthetic data generation and filtering pipeline: cultural data segregation, cultural data adaptation, machine translation, and quality filtering. This pipeline enables the conversion and expansion of the Nemotron-Content-Safety-Dataset-V2 English safety dataset into eight distinct languages: Arabic, German, Spanish, French, Hindi, Japanese, Thai, and Chinese. The resulting dataset, Nemotron-Content-Safety-Dataset-Multilingual-v1, comprises 386,661 samples in 9 languages and facilitates the training of Llama-3.1-Nemotron-Safety-Guard-Multilingual-8B-v1 via LoRA-based fine-tuning. The final model achieves state-of-the-art performance on several multilingual content safety benchmarks. We also benchmark the latest open LLMs on multilingual safety and observe that these LLMs are more prone to give unsafe responses when prompted in non-English languages. This work represents a significant step toward closing the safety gap in multilingual LLMs by enabling the development of culturally aware safety guard models.
>
---
#### [replaced 021] Automatic Speech Recognition of African American English: Lexical and Contextual Effects
- **分类: cs.CL; cs.SD; eess.AS; I.5; G.3**

- **链接: [http://arxiv.org/pdf/2506.06888v2](http://arxiv.org/pdf/2506.06888v2)**

> **作者:** Hamid Mojarad; Kevin Tang
>
> **备注:** submitted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
>
---
#### [replaced 022] Memento: Fine-tuning LLM Agents without Fine-tuning LLMs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16153v2](http://arxiv.org/pdf/2508.16153v2)**

> **作者:** Huichi Zhou; Yihang Chen; Siyuan Guo; Xue Yan; Kin Hei Lee; Zihan Wang; Ka Yiu Lee; Guchun Zhang; Kun Shao; Linyi Yang; Jun Wang
>
> **摘要:** In this paper, we introduce a novel learning paradigm for Adaptive Large Language Model (LLM) agents that eliminates the need for fine-tuning the underlying LLMs. Existing approaches are often either rigid, relying on static, handcrafted reflection workflows, or computationally intensive, requiring gradient updates of LLM model parameters. In contrast, our method enables low-cost continual adaptation via memory-based online reinforcement learning. We formalise this as a Memory-augmented Markov Decision Process (M-MDP), equipped with a neural case-selection policy to guide action decisions. Past experiences are stored in an episodic memory, either differentiable or non-parametric. The policy is continually updated based on environmental feedback through a memory rewriting mechanism, whereas policy improvement is achieved through efficient memory reading (retrieval). We instantiate our agent model in the deep research setting, namely \emph{Memento}, which attains top-1 on GAIA validation ($87.88\%$ Pass@$3$) and $79.40\%$ on the test set. It reaches $66.6\%$ F1 and $80.4\%$ PM on the DeepResearcher dataset, outperforming the state-of-the-art training-based method, while case-based memory adds $4.7\%$ to $9.6\%$ absolute points on out-of-distribution tasks. Our approach offers a scalable and efficient pathway for developing generalist LLM agents capable of continuous, real-time learning without gradient updates, advancing machine learning towards open-ended skill acquisition and deep research scenarios. The code is available at https://github.com/Agent-on-the-Fly/Memento.
>
---
#### [replaced 023] MGT-Prism: Enhancing Domain Generalization for Machine-Generated Text Detection via Spectral Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.13768v2](http://arxiv.org/pdf/2508.13768v2)**

> **作者:** Shengchao Liu; Xiaoming Liu; Chengzhengxu Li; Zhaohan Zhang; Guoxin Ma; Yu Lan; Shuai Xiao
>
> **摘要:** Large Language Models have shown growing ability to generate fluent and coherent texts that are highly similar to the writing style of humans. Current detectors for Machine-Generated Text (MGT) perform well when they are trained and tested in the same domain but generalize poorly to unseen domains, due to domain shift between data from different sources. In this work, we propose MGT-Prism, an MGT detection method from the perspective of the frequency domain for better domain generalization. Our key insight stems from analyzing text representations in the frequency domain, where we observe consistent spectral patterns across diverse domains, while significant discrepancies in magnitude emerge between MGT and human-written texts (HWTs). The observation initiates the design of a low frequency domain filtering module for filtering out the document-level features that are sensitive to domain shift, and a dynamic spectrum alignment strategy to extract the task-specific and domain-invariant features for improving the detector's performance in domain generalization. Extensive experiments demonstrate that MGT-Prism outperforms state-of-the-art baselines by an average of 0.90% in accuracy and 0.92% in F1 score on 11 test datasets across three domain-generalization scenarios.
>
---
#### [replaced 024] Defending against Jailbreak through Early Exit Generation of Large Language Models
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2408.11308v2](http://arxiv.org/pdf/2408.11308v2)**

> **作者:** Chongwen Zhao; Zhihao Dou; Kaizhu Huang
>
> **备注:** ICONIP 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.
>
---
#### [replaced 025] PediatricsMQA: a Multi-modal Pediatrics Question Answering Benchmark
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.16439v2](http://arxiv.org/pdf/2508.16439v2)**

> **作者:** Adil Bahaj; Mohamed Chetouani; Mounir Ghogho
>
> **摘要:** Large language models (LLMs) and vision-augmented LLMs (VLMs) have significantly advanced medical informatics, diagnostics, and decision support. However, these models exhibit systematic biases, particularly age bias, compromising their reliability and equity. This is evident in their poorer performance on pediatric-focused text and visual question-answering tasks. This bias reflects a broader imbalance in medical research, where pediatric studies receive less funding and representation despite the significant disease burden in children. To address these issues, a new comprehensive multi-modal pediatric question-answering benchmark, PediatricsMQA, has been introduced. It consists of 3,417 text-based multiple-choice questions (MCQs) covering 131 pediatric topics across seven developmental stages (prenatal to adolescent) and 2,067 vision-based MCQs using 634 pediatric images from 67 imaging modalities and 256 anatomical regions. The dataset was developed using a hybrid manual-automatic pipeline, incorporating peer-reviewed pediatric literature, validated question banks, existing benchmarks, and existing QA resources. Evaluating state-of-the-art open models, we find dramatic performance drops in younger cohorts, highlighting the need for age-aware methods to ensure equitable AI support in pediatric care.
>
---
#### [replaced 026] Post-Training Language Models for Continual Relation Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05214v3](http://arxiv.org/pdf/2504.05214v3)**

> **作者:** Sefika Efeoglu; Adrian Paschke; Sonja Schimmler
>
> **备注:** 17 pages, Initial Results and Reporting of the work. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Real-world data, such as news articles, social media posts, and chatbot conversations, is inherently dynamic and non-stationary, presenting significant challenges for constructing real-time structured representations through knowledge graphs (KGs). Relation Extraction (RE), a fundamental component of KG creation, often struggles to adapt to evolving data when traditional models rely on static, outdated datasets. Continual Relation Extraction (CRE) methods tackle this issue by incrementally learning new relations while preserving previously acquired knowledge. This study investigates the application of pre-trained language models (PLMs), specifically large language models (LLMs), to CRE, with a focus on leveraging memory replay to address catastrophic forgetting. We evaluate decoder-only models (eg, Mistral-7B and Llama2-7B) and encoder-decoder models (eg, Flan-T5 Base) on the TACRED and FewRel datasets. Task-incremental fine-tuning of LLMs demonstrates superior performance over earlier approaches using encoder-only models like BERT on TACRED, excelling in seen-task accuracy and overall performance (measured by whole and average accuracy), particularly with the Mistral and Flan-T5 models. Results on FewRel are similarly promising, achieving second place in whole and average accuracy metrics. This work underscores critical factors in knowledge transfer, language model architecture, and KG completeness, advancing CRE with LLMs and memory replay for dynamic, real-time relation extraction.
>
---
#### [replaced 027] Harnessing Large Language Models for Disaster Management: A Survey
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.06932v2](http://arxiv.org/pdf/2501.06932v2)**

> **作者:** Zhenyu Lei; Yushun Dong; Weiyu Li; Rong Ding; Qi Wang; Jundong Li
>
> **摘要:** Large language models (LLMs) have revolutionized scientific research with their exceptional capabilities and transformed various fields. Among their practical applications, LLMs have been playing a crucial role in mitigating threats to human life, infrastructure, and the environment. Despite growing research in disaster LLMs, there remains a lack of systematic review and in-depth analysis of LLMs for natural disaster management. To address the gap, this paper presents a comprehensive survey of existing LLMs in natural disaster management, along with a taxonomy that categorizes existing works based on disaster phases and application scenarios. By collecting public datasets and identifying key challenges and opportunities, this study aims to guide the professional community in developing advanced LLMs for disaster management to enhance the resilience against natural disasters.
>
---
#### [replaced 028] Forgotten Polygons: Multimodal Large Language Models are Shape-Blind
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15969v4](http://arxiv.org/pdf/2502.15969v4)**

> **作者:** William Rudman; Michal Golovanevsky; Amir Bar; Vedant Palit; Yann LeCun; Carsten Eickhoff; Ritambhara Singh
>
> **摘要:** Despite strong performance on vision-language tasks, Multimodal Large Language Models (MLLMs) struggle with mathematical problem-solving, with both open-source and state-of-the-art models falling short of human performance on visual-math benchmarks. To systematically examine visual-mathematical reasoning in MLLMs, we (1) evaluate their understanding of geometric primitives, (2) test multi-step reasoning, and (3) explore a potential solution to improve visual reasoning capabilities. Our findings reveal fundamental shortcomings in shape recognition, with top models achieving under 50% accuracy in identifying regular polygons. We analyze these failures through the lens of dual-process theory and show that MLLMs rely on System 1 (intuitive, memorized associations) rather than System 2 (deliberate reasoning). Consequently, MLLMs fail to count the sides of both familiar and novel shapes, suggesting they have neither learned the concept of sides nor effectively process visual inputs. Finally, we propose Visually Cued Chain-of-Thought (VC-CoT) prompting, which enhances multi-step mathematical reasoning by explicitly referencing visual annotations in diagrams, boosting GPT-4o's accuracy on an irregular polygon side-counting task from 7% to 93%. Our findings suggest that System 2 reasoning in MLLMs remains an open problem, and visually-guided prompting is essential for successfully engaging visual reasoning. Code available at: https://github.com/rsinghlab/Shape-Blind.
>
---
#### [replaced 029] WHEN TO ACT, WHEN TO WAIT: Modeling the Intent-Action Alignment Problem in Dialogue
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01881v2](http://arxiv.org/pdf/2506.01881v2)**

> **作者:** Yaoyao Qian; Jindan Huang; Yuanli Wang; Simon Yu; Kyrie Zhixuan Zhou; Jiayuan Mao; Mingfu Liang; Hanhan Zhou
>
> **备注:** Project website: https://nanostorm.netlify.app/
>
> **摘要:** Dialogue systems often fail when user utterances are semantically complete yet lack the clarity and completeness required for appropriate system action. This mismatch arises because users frequently do not fully understand their own needs, while systems require precise intent definitions. This highlights the critical Intent-Action Alignment Problem: determining when an expression is not just understood, but truly ready for a system to act upon. We present STORM, a framework modeling asymmetric information dynamics through conversations between UserLLM (full internal access) and AgentLLM (observable behavior only). STORM produces annotated corpora capturing trajectories of expression phrasing and latent cognitive transitions, enabling systematic analysis of how collaborative understanding develops. Our contributions include: (1) formalizing asymmetric information processing in dialogue systems; (2) modeling intent formation tracking collaborative understanding evolution; and (3) evaluation metrics measuring internal cognitive improvements alongside task performance. Experiments across four language models reveal that moderate uncertainty (40-60%) can outperform complete transparency in certain scenarios, with model-specific patterns suggesting reconsideration of optimal information completeness in human-AI collaboration. These findings contribute to understanding asymmetric reasoning dynamics and inform uncertainty-calibrated dialogue system design.
>
---
#### [replaced 030] Fingerprint Vector: Enabling Scalable and Efficient Model Fingerprint Transfer via Vector Addition
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.08846v2](http://arxiv.org/pdf/2409.08846v2)**

> **作者:** Zhenhua Xu; Qichen Liu; Zhebo Wang; Wenpeng Xing; Dezhang Kong; Mohan Li; Meng Han
>
> **摘要:** Backdoor-based fingerprinting has emerged as an effective technique for tracing the ownership of large language models. However, in real-world deployment scenarios, developers often instantiate multiple downstream models from a shared base model, and applying fingerprinting to each variant individually incurs prohibitive computational overhead. While inheritance-based approaches -- where fingerprints are embedded into the base model and expected to persist through fine-tuning -- appear attractive, they suffer from three key limitations: late-stage fingerprinting, fingerprint instability, and interference with downstream adaptation. To address these challenges, we propose a novel mechanism called the Fingerprint Vector. Our method first embeds a fingerprint into the base model via backdoor-based fine-tuning, then extracts a task-specific parameter delta as a fingerprint vector by computing the difference between the fingerprinted and clean models. This vector can be directly added to any structurally compatible downstream model, allowing the fingerprint to be transferred post hoc without additional fine-tuning. Extensive experiments show that Fingerprint Vector achieves comparable or superior performance to direct injection across key desiderata. It maintains strong effectiveness across diverse model architectures as well as mainstream downstream variants within the same family. It also preserves harmlessness and robustness in most cases. Even when slight robustness degradation is observed, the impact remains within acceptable bounds and is outweighed by the scalability benefits of our approach.
>
---
#### [replaced 031] MedQARo: A Large-Scale Benchmark for Medical Question Answering in Romanian
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.16390v2](http://arxiv.org/pdf/2508.16390v2)**

> **作者:** Ana-Cristina Rogoz; Radu Tudor Ionescu; Alexandra-Valentina Anghel; Ionut-Lucian Antone-Iordache; Simona Coniac; Andreea Iuliana Ionescu
>
> **摘要:** Question answering (QA) is an actively studied topic, being a core natural language processing (NLP) task that needs to be addressed before achieving Artificial General Intelligence (AGI). However, the lack of QA datasets in specific domains and languages hinders the development of robust AI models able to generalize across various domains and languages. To this end, we introduce MedQARo, the first large-scale medical QA benchmark in Romanian, alongside a comprehensive evaluation of state-of-the-art large language models (LLMs). We construct a high-quality and large-scale dataset comprising 102,646 QA pairs related to cancer patients. The questions regard medical case summaries of 1,011 patients, requiring either keyword extraction or reasoning to be answered correctly. MedQARo is the result of a time-consuming manual annotation process carried out by seven physicians specialized in oncology or radiotherapy, who spent a total of about 2,100 work hours to generate the QA pairs. We experiment with four LLMs from distinct families of models on MedQARo. Each model is employed in two scenarios, namely one based on zero-shot prompting and one based on supervised fine-tuning. Our results show that fine-tuned models significantly outperform their zero-shot counterparts, clearly indicating that pretrained models fail to generalize on MedQARo. Our findings demonstrate the importance of both domain-specific and language-specific fine-tuning for reliable clinical QA in Romanian. We publicly release our dataset and code at https://github.com/ana-rogoz/MedQARo.
>
---
#### [replaced 032] QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13266v2](http://arxiv.org/pdf/2507.13266v2)**

> **作者:** Jiazheng Li; Hong Lu; Kaiyue Wen; Zaiwen Yang; Jiaxuan Gao; Hongzhou Lin; Yi Wu; Jingzhao Zhang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
>
---
#### [replaced 033] Theory of Mind in Large Language Models: Assessment and Enhancement
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00026v2](http://arxiv.org/pdf/2505.00026v2)**

> **作者:** Ruirui Chen; Weifeng Jiang; Chengwei Qin; Cheston Tan
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** Theory of Mind (ToM)-the ability to reason about the mental states of oneself and others-is a cornerstone of human social intelligence. As Large Language Models (LLMs) become increasingly integrated into daily life, understanding their ability to interpret and respond to human mental states is crucial for enabling effective interactions. In this paper, we review LLMs' ToM capabilities by analyzing both evaluation benchmarks and enhancement strategies. For evaluation, we focus on recently proposed and widely used story-based benchmarks. For enhancement, we provide an in-depth analysis of recent methods aimed at improving LLMs' ToM abilities. Furthermore, we outline promising directions for future research to further advance these capabilities and better adapt LLMs to more realistic and diverse scenarios. Our survey serves as a valuable resource for researchers interested in evaluating and advancing LLMs' ToM capabilities.
>
---
#### [replaced 034] SEA-BED: Southeast Asia Embedding Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12243v2](http://arxiv.org/pdf/2508.12243v2)**

> **作者:** Wuttikorn Ponwitayarat; Raymond Ng; Jann Railey Montalan; Thura Aung; Jian Gang Ngui; Yosephine Susanto; William Tjhi; Panuthep Tasawong; Erik Cambria; Ekapol Chuangsuwanich; Sarana Nutanong; Peerat Limkonchotiwat
>
> **摘要:** Sentence embeddings are essential for NLP tasks such as semantic search, re-ranking, and textual similarity. Although multilingual benchmarks like MMTEB broaden coverage, Southeast Asia (SEA) datasets are scarce and often machine-translated, missing native linguistic properties. With nearly 700 million speakers, the SEA region lacks a region-specific embedding benchmark. We introduce SEA-BED, the first large-scale SEA embedding benchmark with 169 datasets across 9 tasks and 10 languages, where 71% are formulated by humans, not machine generation or translation. We address three research questions: (1) which SEA languages and tasks are challenging, (2) whether SEA languages show unique performance gaps globally, and (3) how human vs. machine translations affect evaluation. We evaluate 17 embedding models across six studies, analyzing task and language challenges, cross-benchmark comparisons, and translation trade-offs. Results show sharp ranking shifts, inconsistent model performance among SEA languages, and the importance of human-curated datasets for low-resource languages like Burmese.
>
---
#### [replaced 035] Investigating the Robustness of Deductive Reasoning with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.04352v2](http://arxiv.org/pdf/2502.04352v2)**

> **作者:** Fabian Hoppe; Filip Ilievski; Jan-Christoph Kalo
>
> **备注:** to be published in ECAI 2025
>
> **摘要:** Large Language Models (LLMs) have been shown to achieve impressive results for many reasoning-based NLP tasks, suggesting a degree of deductive reasoning capability. However, it remains unclear to which extent LLMs, in both informal and autoformalisation methods, are robust on logical deduction tasks. Moreover, while many LLM-based deduction methods have been proposed, a systematic study that analyses the impact of their design components is lacking. Addressing these two challenges, we propose the first study of the robustness of formal and informal LLM-based deductive reasoning methods. We devise a framework with two families of perturbations: adversarial noise and counterfactual statements, which jointly generate seven perturbed datasets. We organize the landscape of LLM reasoners according to their reasoning format, formalisation syntax, and feedback for error recovery. The results show that adversarial noise affects autoformalisation, while counterfactual statements influence all approaches. Detailed feedback does not improve overall accuracy despite reducing syntax errors, pointing to the challenge of LLM-based methods to self-correct effectively.
>
---
#### [replaced 036] Jinx: Unlimited LLMs for Probing Alignment Failures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08243v3](http://arxiv.org/pdf/2508.08243v3)**

> **作者:** Jiahao Zhao; Liwei Dong
>
> **备注:** https://huggingface.co/Jinx-org
>
> **摘要:** Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
>
---
#### [replaced 037] Words Like Knives: Backstory-Personalized Modeling and Detection of Violent Communication
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21451v2](http://arxiv.org/pdf/2505.21451v2)**

> **作者:** Jocelyn Shen; Akhila Yerukola; Xuhui Zhou; Cynthia Breazeal; Maarten Sap; Hae Won Park
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Conversational breakdowns in close relationships are deeply shaped by personal histories and emotional context, yet most NLP research treats conflict detection as a general task, overlooking the relational dynamics that influence how messages are perceived. In this work, we leverage nonviolent communication (NVC) theory to evaluate LLMs in detecting conversational breakdowns and assessing how relationship backstory influences both human and model perception of conflicts. Given the sensitivity and scarcity of real-world datasets featuring conflict between familiar social partners with rich personal backstories, we contribute the PersonaConflicts Corpus, a dataset of N=5,772 naturalistic simulated dialogues spanning diverse conflict scenarios between friends, family members, and romantic partners. Through a controlled human study, we annotate a subset of dialogues and obtain fine-grained labels of communication breakdown types on individual turns, and assess the impact of backstory on human and model perception of conflict in conversation. We find that the polarity of relationship backstories significantly shifted human perception of communication breakdowns and impressions of the social partners, yet models struggle to meaningfully leverage those backstories in the detection task. Additionally, we find that models consistently overestimate how positively a message will make a listener feel. Our findings underscore the critical role of personalization to relationship contexts in enabling LLMs to serve as effective mediators in human communication for authentic connection.
>
---
#### [replaced 038] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15851v3](http://arxiv.org/pdf/2502.15851v3)**

> **作者:** Yilin Geng; Haonan Li; Honglin Mu; Xudong Han; Timothy Baldwin; Omri Abend; Eduard Hovy; Lea Frermann
>
> **摘要:** Large language models (LLMs) are increasingly deployed with hierarchical instruction schemes, where certain instructions (e.g., system-level directives) are expected to take precedence over others (e.g., user messages). Yet, we lack a systematic understanding of how effectively these hierarchical control mechanisms work. We introduce a systematic evaluation framework based on constraint prioritization to assess how well LLMs enforce instruction hierarchies. Our experiments across six state-of-the-art LLMs reveal that models struggle with consistent instruction prioritization, even for simple formatting conflicts. We find that the widely-adopted system/user prompt separation fails to establish a reliable instruction hierarchy, and models exhibit strong inherent biases toward certain constraint types regardless of their priority designation. We find that LLMs more reliably obey constraints framed through natural social hierarchies (e.g., authority, expertise, consensus) than system/user roles, which suggests that pretraining-derived social structures act as latent control priors, with potentially stronger influence than post-training guardrails.
>
---
#### [replaced 039] Measuring Sycophancy of Language Models in Multi-turn Dialogues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23840v2](http://arxiv.org/pdf/2505.23840v2)**

> **作者:** Jiseung Hong; Grace Byun; Seungone Kim; Kai Shu; Jinho Choi
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) are expected to provide helpful and harmless responses, yet they often exhibit sycophancy--conforming to user beliefs regardless of factual accuracy or ethical soundness. Prior research on sycophancy has primarily focused on single-turn factual correctness, overlooking the dynamics of real-world interactions. In this work, we introduce SYCON Bench, a novel benchmark for evaluating sycophantic behavior in multi-turn, free-form conversational settings. Our benchmark measures how quickly a model conforms to the user (Turn of Flip) and how frequently it shifts its stance under sustained user pressure (Number of Flip). Applying SYCON Bench to 17 LLMs across three real-world scenarios, we find that sycophancy remains a prevalent failure mode. Our analysis shows that alignment tuning amplifies sycophantic behavior, whereas model scaling and reasoning optimization strengthen the model's ability to resist undesirable user views. Reasoning models generally outperform instruction-tuned models but often fail when they over-index on logical exposition instead of directly addressing the user's underlying beliefs. Finally, we evaluate four additional prompting strategies and demonstrate that adopting a third-person perspective reduces sycophancy by up to 63.8% in debate scenario. We release our code and data at https://github.com/JiseungHong/SYCON-Bench.
>
---
#### [replaced 040] ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.14377v2](http://arxiv.org/pdf/2508.14377v2)**

> **作者:** Wenhan Dong; Zhen Sun; Yuemeng Zhao; Zifan Peng; Jun Wu; Jingyi Zheng; Yule Liu; Xinlei He; Yu Wang; Ruiming Wang; Xinyi Huang; Lei Mo
>
> **摘要:** Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications.
>
---
#### [replaced 041] PARROT: An Open Multilingual Radiology Reports Dataset
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22939v2](http://arxiv.org/pdf/2507.22939v2)**

> **作者:** Bastien Le Guellec; Kokou Adambounou; Lisa C Adams; Thibault Agripnidis; Sung Soo Ahn; Radhia Ait Chalal; Tugba Akinci D Antonoli; Philippe Amouyel; Henrik Andersson; Raphael Bentegeac; Claudio Benzoni; Antonino Andrea Blandino; Felix Busch; Elif Can; Riccardo Cau; Armando Ugo Cavallo; Christelle Chavihot; Erwin Chiquete; Renato Cuocolo; Eugen Divjak; Gordana Ivanac; Barbara Dziadkowiec Macek; Armel Elogne; Salvatore Claudio Fanni; Carlos Ferrarotti; Claudia Fossataro; Federica Fossataro; Katarzyna Fulek; Michal Fulek; Pawel Gac; Martyna Gachowska; Ignacio Garcia Juarez; Marco Gatti; Natalia Gorelik; Alexia Maria Goulianou; Aghiles Hamroun; Nicolas Herinirina; Krzysztof Kraik; Dominik Krupka; Quentin Holay; Felipe Kitamura; Michail E Klontzas; Anna Kompanowska; Rafal Kompanowski; Alexandre Lefevre; Tristan Lemke; Maximilian Lindholz; Lukas Muller; Piotr Macek; Marcus Makowski; Luigi Mannacio; Aymen Meddeb; Antonio Natale; Beatrice Nguema Edzang; Adriana Ojeda; Yae Won Park; Federica Piccione; Andrea Ponsiglione; Malgorzata Poreba; Rafal Poreba; Philipp Prucker; Jean Pierre Pruvo; Rosa Alba Pugliesi; Feno Hasina Rabemanorintsoa; Vasileios Rafailidis; Katarzyna Resler; Jan Rotkegel; Luca Saba; Ezann Siebert; Arnaldo Stanzione; Ali Fuat Tekin; Liz Toapanta Yanchapaxi; Matthaios Triantafyllou; Ekaterini Tsaoulia; Evangelia Vassalou; Federica Vernuccio; Johan Wasselius; Weilang Wang; Szymon Urban; Adrian Wlodarczak; Szymon Wlodarczak; Andrzej Wysocki; Lina Xu; Tomasz Zatonski; Shuhang Zhang; Sebastian Ziegelmayer; Gregory Kuchcinski; Keno K Bressem
>
> **备注:** Corrected affiliations (no change to the paper)
>
> **摘要:** Rationale and Objectives: To develop and validate PARROT (Polyglottal Annotated Radiology Reports for Open Testing), a large, multicentric, open-access dataset of fictional radiology reports spanning multiple languages for testing natural language processing applications in radiology. Materials and Methods: From May to September 2024, radiologists were invited to contribute fictional radiology reports following their standard reporting practices. Contributors provided at least 20 reports with associated metadata including anatomical region, imaging modality, clinical context, and for non-English reports, English translations. All reports were assigned ICD-10 codes. A human vs. AI report differentiation study was conducted with 154 participants (radiologists, healthcare professionals, and non-healthcare professionals) assessing whether reports were human-authored or AI-generated. Results: The dataset comprises 2,658 radiology reports from 76 authors across 21 countries and 13 languages. Reports cover multiple imaging modalities (CT: 36.1%, MRI: 22.8%, radiography: 19.0%, ultrasound: 16.8%) and anatomical regions, with chest (19.9%), abdomen (18.6%), head (17.3%), and pelvis (14.1%) being most prevalent. In the differentiation study, participants achieved 53.9% accuracy (95% CI: 50.7%-57.1%) in distinguishing between human and AI-generated reports, with radiologists performing significantly better (56.9%, 95% CI: 53.3%-60.6%, p<0.05) than other groups. Conclusion: PARROT represents the largest open multilingual radiology report dataset, enabling development and validation of natural language processing applications across linguistic, geographic, and clinical boundaries without privacy constraints.
>
---
#### [replaced 042] Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.12964v2](http://arxiv.org/pdf/2502.12964v2)**

> **作者:** Adi Simhi; Itay Itzhak; Fazl Barez; Gabriel Stanovsky; Yonatan Belinkov
>
> **摘要:** Prior work on large language model (LLM) hallucinations has associated them with model uncertainty or inaccurate knowledge. In this work, we define and investigate a distinct type of hallucination, where a model can consistently answer a question correctly, but a seemingly trivial perturbation, which can happen in real-world settings, causes it to produce a hallucinated response with high certainty. This phenomenon, which we dub CHOKE (Certain Hallucinations Overriding Known Evidence), is particularly concerning in high-stakes domains such as medicine or law, where model certainty is often used as a proxy for reliability. We show that CHOKE examples are consistent across prompts, occur in different models and datasets, and are fundamentally distinct from other hallucinations. This difference leads existing mitigation methods to perform worse on CHOKE examples than on general hallucinations. Finally, we introduce a probing-based mitigation that outperforms existing methods on CHOKE hallucinations. These findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve mitigation strategies to enhance LLM safety. The code is available at https://github.com/technion-cs-nlp/Trust_me_Im_wrong .
>
---
#### [replaced 043] Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16789v2](http://arxiv.org/pdf/2411.16789v2)**

> **作者:** Jungeun Kim; Hyeongwoo Jeon; Jongseong Bae; Ha Young Kim
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Sign language translation (SLT) is a challenging task that involves translating sign language images into spoken language. For SLT models to perform this task successfully, they must bridge the modality gap and identify subtle variations in sign language components to understand their meanings accurately. To address these challenges, we propose a novel gloss-free SLT framework called Multimodal Sign Language Translation (MMSLT), which leverages the representational capabilities of off-the-shelf multimodal large language models (MLLMs). Specifically, we use MLLMs to generate detailed textual descriptions of sign language components. Then, through our proposed multimodal-language pre-training module, we integrate these description features with sign video features to align them within the spoken sentence space. Our approach achieves state-of-the-art performance on benchmark datasets PHOENIX14T and CSL-Daily, highlighting the potential of MLLMs to be utilized effectively in SLT. Code is available at https://github.com/hwjeon98/MMSLT.
>
---
#### [replaced 044] Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02531v2](http://arxiv.org/pdf/2501.02531v2)**

> **作者:** Ljubisa Bojic; Dylan Seychell; Milan Cabarkapa
>
> **备注:** 20 pages, 1 figure
>
> **摘要:** As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
>
---
#### [replaced 045] FlexOlmo: Open Language Models for Flexible Data Use
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.07024v4](http://arxiv.org/pdf/2507.07024v4)**

> **作者:** Weijia Shi; Akshita Bhagia; Kevin Farhat; Niklas Muennighoff; Pete Walsh; Jacob Morrison; Dustin Schwenk; Shayne Longpre; Jake Poznanski; Allyson Ettinger; Daogao Liu; Margaret Li; Dirk Groeneveld; Mike Lewis; Wen-tau Yih; Luca Soldaini; Kyle Lo; Noah A. Smith; Luke Zettlemoyer; Pang Wei Koh; Hannaneh Hajishirzi; Ali Farhadi; Sewon Min
>
> **摘要:** We introduce FlexOlmo, a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FlexOlmo employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FlexOlmo is trained on FlexMix, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets. We evaluate models with up to 37 billion parameters (20 billion active) on 31 diverse downstream tasks. We show that a general expert trained on public data can be effectively combined with independently trained experts from other data owners, leading to an average 41% relative improvement while allowing users to opt out of certain data based on data licensing or permission requirements. Our approach also outperforms prior model merging methods by 10.1% on average and surpasses the standard MoE trained without data restrictions using the same training FLOPs. Altogether, this research presents a solution for both data owners and researchers in regulated industries with sensitive or protected data. FlexOlmo enables benefiting from closed data while respecting data owners' preferences by keeping their data local and supporting fine-grained control of data access during inference.
>
---
#### [replaced 046] Unified attacks to large language model watermarks: spoofing and scrubbing in unauthorized knowledge distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17480v4](http://arxiv.org/pdf/2504.17480v4)**

> **作者:** Xin Yi; Yue Li; Shunfan Zheng; Linlin Wang; Xiaoling Wang; Liang He
>
> **摘要:** Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.
>
---
#### [replaced 047] PII-Compass: Guiding LLM training data extraction prompts towards the target PII via grounding
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.02943v2](http://arxiv.org/pdf/2407.02943v2)**

> **作者:** Krishna Kanth Nakka; Ahmed Frikha; Ricardo Mendes; Xue Jiang; Xuebing Zhou
>
> **备注:** Accepted at PrivateNLP Workshop at ACL 2024
>
> **摘要:** The latest and most impactful advances in large models stem from their increased size. Unfortunately, this translates into an improved memorization capacity, raising data privacy concerns. Specifically, it has been shown that models can output personal identifiable information (PII) contained in their training data. However, reported PIII extraction performance varies widely, and there is no consensus on the optimal methodology to evaluate this risk, resulting in underestimating realistic adversaries. In this work, we empirically demonstrate that it is possible to improve the extractability of PII by over ten-fold by grounding the prefix of the manually constructed extraction prompt with in-domain data. Our approach, PII-Compass, achieves phone number extraction rates of 0.92%, 3.9%, and 6.86% with 1, 128, and 2308 queries, respectively, i.e., the phone number of 1 person in 15 is extractable.
>
---
#### [replaced 048] From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14045v2](http://arxiv.org/pdf/2505.14045v2)**

> **作者:** Yingli Shen; Wen Lai; Shuo Wang; Ge Gao; Kangyang Luo; Alexander Fraser; Maosong Sun
>
> **摘要:** Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
>
---
#### [replaced 049] Auto prompt sql: a resource-efficient architecture for text-to-sql translation in constrained environments
- **分类: cs.CL; cs.AI; 68T50**

- **链接: [http://arxiv.org/pdf/2506.03598v2](http://arxiv.org/pdf/2506.03598v2)**

> **作者:** Zetong Tang; Qian Ma; Di Wu
>
> **备注:** 4 pages,2 figures,EITCE 2025
>
> **摘要:** Using the best Text-to-SQL methods in resource-constrained environments is challenging due to their reliance on resource-intensive open-source models. This paper introduces Auto Prompt SQL(AP-SQL), a novel architecture designed to bridge the gap between resource-efficient small open-source models and the powerful capabilities of large closed-source models for Text-to-SQL translation. Our method decomposes the task into schema filtering, retrieval-augmented text-to-SQL generation based on in-context examples, and prompt-driven schema linking and SQL generation. To improve schema selection accuracy, we fine-tune large language models. Crucially, we also explore the impact of prompt engineering throughout the process, leveraging Chain-of-Thought(CoT) and Graph-of-Thought(GoT) templates to significantly enhance the model's reasoning for accurate SQL generation. Comprehensive evaluations on the Spider benchmarks demonstrate the effectiveness of AP-SQL.
>
---
#### [replaced 050] Evaluation of Large Language Models via Coupled Token Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01754v2](http://arxiv.org/pdf/2502.01754v2)**

> **作者:** Nina Corvelo Benz; Stratis Tsirtsis; Eleni Straitouri; Ivi Chatzi; Ander Artola Velasco; Suhas Thejaswi; Manuel Gomez-Rodriguez
>
> **摘要:** State of the art large language models rely on randomization to respond to a prompt. As an immediate consequence, a model may respond differently to the same prompt if asked multiple times. In this work, we argue that the evaluation and ranking of large language models should control for the randomization underpinning their functioning. Our starting point is the development of a causal model for coupled autoregressive generation, which allows different large language models to sample responses with the same source of randomness. Building upon our causal model, we first show that, on evaluations based on benchmark datasets, coupled autoregressive generation leads to the same conclusions as vanilla autoregressive generation but using provably fewer samples. However, we further show that, on evaluations based on (human) pairwise comparisons, coupled and vanilla autoregressive generation can surprisingly lead to different rankings when comparing more than two models, even with an infinite amount of samples. This suggests that the apparent advantage of a model over others in existing evaluation protocols may not be genuine but rather confounded by the randomness inherent to the generation process. To illustrate and complement our theoretical results, we conduct experiments with several large language models from the Llama, Mistral and Qwen families. We find that, across multiple benchmark datasets, coupled autoregressive generation requires up to 75% fewer samples to reach the same conclusions as vanilla autoregressive generation. Further, we find that the win-rates derived from pairwise comparisons by a strong large language model to prompts from the LMSYS Chatbot Arena platform differ under coupled and vanilla autoregressive generation.
>
---
#### [replaced 051] VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation
- **分类: cs.AR; cs.AI; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2504.15659v2](http://arxiv.org/pdf/2504.15659v2)**

> **作者:** Anjiang Wei; Huanmi Tan; Tarun Suresh; Daniel Mendoza; Thiago S. F. X. Teixeira; Ke Wang; Caroline Trippel; Alex Aiken
>
> **摘要:** Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of 125,777 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4%, respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/VeriCoder
>
---
#### [replaced 052] Disentangling Exploration of Large Language Models by Optimal Exploitation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08925v3](http://arxiv.org/pdf/2501.08925v3)**

> **作者:** Tim Grams; Patrick Betz; Sascha Marton; Stefan Lüdtke; Christian Bartelt
>
> **摘要:** Exploration is a crucial skill for in-context reinforcement learning in unknown environments. However, it remains unclear if large language models can effectively explore a partially hidden state space. This work isolates exploration as the sole objective, tasking an agent with gathering information that enhances future returns. Within this framework, we argue that measuring agent returns is not sufficient for a fair evaluation. Hence, we decompose missing rewards into their exploration and exploitation components based on the optimal achievable return. Experiments with various models reveal that most struggle to explore the state space, and weak exploration is insufficient. Nevertheless, we found a positive correlation between exploration performance and reasoning capabilities. Our decomposition can provide insights into differences in behaviors driven by prompt engineering, offering a valuable tool for refining performance in exploratory tasks.
>
---
#### [replaced 053] SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.15587v2](http://arxiv.org/pdf/2501.15587v2)**

> **作者:** Dakuan Lu; Xiaoyu Tan; Rui Xu; Tianchu Yao; Chao Qu; Wei Chu; Yinghui Xu; Yuan Qi
>
> **备注:** 9 pages, 1 figures
>
> **摘要:** Recent breakthroughs in large language models (LLMs) exemplified by the impressive mathematical and scientific reasoning capabilities of the o1 model have spotlighted the critical importance of high-quality training data in advancing LLM performance across STEM disciplines. While the mathematics community has benefited from a growing body of curated datasets, the scientific domain at the higher education level has long suffered from a scarcity of comparable resources. To address this gap, we present SCP-116K, a new large-scale dataset of 116,756 high-quality problem-solution pairs, automatically extracted from heterogeneous sources using a streamlined and highly generalizable pipeline. Our approach involves stringent filtering to ensure the scientific rigor and educational level of the extracted materials, while maintaining adaptability for future expansions or domain transfers. By openly releasing both the dataset and the extraction pipeline, we seek to foster research on scientific reasoning, enable comprehensive performance evaluations of new LLMs, and lower the barrier to replicating the successes of advanced models like o1 in the broader science community. We believe SCP-116K will serve as a critical resource, catalyzing progress in high-level scientific reasoning tasks and promoting further innovations in LLM development. The dataset and code are publicly available at https://github.com/AQA6666/SCP-116K-open.
>
---
#### [replaced 054] BiMark: Unbiased Multilayer Watermarking for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21602v2](http://arxiv.org/pdf/2506.21602v2)**

> **作者:** Xiaoyan Feng; He Zhang; Yanjun Zhang; Leo Yu Zhang; Shirui Pan
>
> **备注:** This paper is accepted by International Conference on Machine Learning (ICML) 2025
>
> **摘要:** Recent advances in Large Language Models (LLMs) have raised urgent concerns about LLM-generated text authenticity, prompting regulatory demands for reliable identification mechanisms. Although watermarking offers a promising solution, existing approaches struggle to simultaneously achieve three critical requirements: text quality preservation, model-agnostic detection, and message embedding capacity, which are crucial for practical implementation. To achieve these goals, the key challenge lies in balancing the trade-off between text quality preservation and message embedding capacity. To address this challenge, we propose BiMark, a novel watermarking framework that achieves these requirements through three key innovations: (1) a bit-flip unbiased reweighting mechanism enabling model-agnostic detection, (2) a multilayer architecture enhancing detectability without compromising generation quality, and (3) an information encoding approach supporting multi-bit watermarking. Through theoretical analysis and extensive experiments, we validate that, compared to state-of-the-art multi-bit watermarking methods, BiMark achieves up to 30% higher extraction rates for short texts while maintaining text quality indicated by lower perplexity, and performs comparably to non-watermarked text on downstream tasks such as summarization and translation.
>
---
#### [replaced 055] Effective Red-Teaming of Policy-Adherent Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.09600v3](http://arxiv.org/pdf/2506.09600v3)**

> **作者:** Itay Nakash; George Kour; Koren Lazar; Matan Vetzler; Guy Uziel; Ateret Anaby-Tavor
>
> **摘要:** Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks
>
---
#### [replaced 056] Graph-oriented Instruction Tuning of Large Language Models for Generic Graph Mining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.04780v3](http://arxiv.org/pdf/2403.04780v3)**

> **作者:** Yanchao Tan; Hang Lv; Pengxiang Zhan; Shiping Wang; Carl Yang
>
> **备注:** Accepted by TPAMI 2025
>
> **摘要:** Graphs with abundant attributes are essential in modeling interconnected entities and enhancing predictions across various real-world applications. Traditional Graph Neural Networks (GNNs) often require re-training for different graph tasks and datasets. Although the emergence of Large Language Models (LLMs) has introduced new paradigms in natural language processing, their potential for generic graph mining, training a single model to simultaneously handle diverse tasks and datasets, remains under-explored. To this end, our novel framework MuseGraph, seamlessly integrates the strengths of GNNs and LLMs into one foundation model for graph mining across tasks and datasets. This framework first features a compact graph description to encapsulate key graph information within language token limitations. Then, we propose a diverse instruction generation mechanism with Chain-of-Thought (CoT)-based instruction packages to distill the reasoning capabilities from advanced LLMs like GPT-4. Finally, we design a graph-aware instruction tuning strategy to facilitate mutual enhancement across multiple tasks and datasets while preventing catastrophic forgetting of LLMs' generative abilities. Our experimental results demonstrate significant improvements in five graph tasks and ten datasets, showcasing the potential of our MuseGraph in enhancing the accuracy of graph-oriented downstream tasks while improving the generation abilities of LLMs.
>
---
#### [replaced 057] Large Language Models in the Task of Automatic Validation of Text Classifier Predictions
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18688v2](http://arxiv.org/pdf/2505.18688v2)**

> **作者:** Aleksandr Tsymbalov; Mikhail Khovrichev
>
> **摘要:** Machine learning models for text classification are trained to predict a class for a given text. To do this, training and validation samples must be prepared: a set of texts is collected, and each text is assigned a class. These classes are usually assigned by human annotators with different expertise levels, depending on the specific classification task. Collecting such samples from scratch is labor-intensive because it requires finding specialists and compensating them for their work; moreover, the number of available specialists is limited, and their productivity is constrained by human factors. While it may not be too resource-intensive to collect samples once, the ongoing need to retrain models (especially in incremental learning pipelines) to address data drift (also called model drift) makes the data collection process crucial and costly over the model's entire lifecycle. This paper proposes several approaches to replace human annotators with Large Language Models (LLMs) to test classifier predictions for correctness, helping ensure model quality and support high-quality incremental learning.
>
---
#### [replaced 058] ComplexTempQA:A 100m Dataset for Complex Temporal Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.04866v3](http://arxiv.org/pdf/2406.04866v3)**

> **作者:** Raphael Gruber; Abdelrahman Abdallah; Michael Färber; Adam Jatowt
>
> **备注:** Accepted at EMNLP main
>
> **摘要:** We introduce \textsc{ComplexTempQA},\footnote{Dataset and code available at: https://github.com/DataScienceUIBK/ComplexTempQA} a large-scale dataset consisting of over 100 million question-answer pairs designed to tackle the challenges in temporal question answering. \textsc{ComplexTempQA} significantly surpasses existing benchmarks in scale and scope. Utilizing Wikipedia and Wikidata, the dataset covers questions spanning over two decades and offers an unmatched scale. We introduce a new taxonomy that categorizes questions as \textit{attributes}, \textit{comparisons}, and \textit{counting} questions, revolving around events, entities, and time periods, respectively. A standout feature of \textsc{ComplexTempQA} is the high complexity of its questions, which demand reasoning capabilities for answering such as across-time comparison, temporal aggregation, and multi-hop reasoning involving temporal event ordering and entity recognition. Additionally, each question is accompanied by detailed metadata, including specific time scopes, allowing for comprehensive evaluation of temporal reasoning abilities of large language models.
>
---
#### [replaced 059] Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.05945v3](http://arxiv.org/pdf/2502.05945v3)**

> **作者:** Paul Darm; Annalisa Riccardi
>
> **备注:** Published at Transaction of Machine Learning Research 08/2025, Large Language Models (LLMs), Interference-time activation shifting, Steerability, Explainability, AI alignment, Interpretability
>
> **摘要:** Robust alignment guardrails for large language models (LLMs) are becoming increasingly important with their widespread application. In contrast to previous studies, we demonstrate that inference-time activation interventions can bypass safety alignments and effectively steer model generations towards harmful AI coordination. Our method applies fine-grained interventions at specific attention heads, which we identify by probing each head in a simple binary choice task. We then show that interventions on these heads generalise to the open-ended generation setting, effectively circumventing safety guardrails. We demonstrate that intervening on a few attention heads is more effective than intervening on full layers or supervised fine-tuning. We further show that only a few example completions are needed to compute effective steering directions, which is an advantage over classical fine-tuning. We also demonstrate that applying interventions in the negative direction can prevent a common jailbreak attack. Our results suggest that, at the attention head level, activations encode fine-grained linearly separable behaviours. Practically, the approach offers a straightforward methodology to steer large language model behaviour, which could be extended to diverse domains beyond safety, requiring fine-grained control over the model output. The code and datasets for this study can be found on https://github.com/PaulDrm/targeted_intervention.
>
---
#### [replaced 060] Modeling Probabilistic Reduction using Information Theory and Naive Discriminative Learning
- **分类: cs.CL; cs.IT; math.IT; I.5; G.3; E.4**

- **链接: [http://arxiv.org/pdf/2506.09641v2](http://arxiv.org/pdf/2506.09641v2)**

> **作者:** Anna Stein; Kevin Tang
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This study compares probabilistic predictors based on information theory with Naive Discriminative Learning (NDL) predictors in modeling acoustic word duration, focusing on probabilistic reduction. We examine three models using the Buckeye corpus: one with NDL-derived predictors using information-theoretic formulas, one with traditional NDL predictors, and one with N-gram probabilistic predictors. Results show that the N-gram model outperforms both NDL models, challenging the assumption that NDL is more effective due to its cognitive motivation. However, incorporating information-theoretic formulas into NDL improves model performance over the traditional model. This research highlights a) the need to incorporate not only frequency and contextual predictability but also average contextual predictability, and b) the importance of combining information-theoretic metrics of predictability and information derived from discriminative learning in modeling acoustic reduction.
>
---
#### [replaced 061] DRT: Deep Reasoning Translation via Long Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.17498v4](http://arxiv.org/pdf/2412.17498v4)**

> **作者:** Jiaan Wang; Fandong Meng; Yunlong Liang; Jie Zhou
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Recently, O1-like models have emerged as representative examples, illustrating the effectiveness of long chain-of-thought (CoT) in reasoning tasks such as math and coding tasks. In this paper, we introduce DRT, an attempt to bring the success of long CoT to neural machine translation (MT). Specifically, in view of the literature books that might involve similes and metaphors, translating these texts to a target language is very difficult in practice due to cultural differences. In such cases, literal translation often fails to convey the intended meaning effectively. Even for professional human translators, considerable thought must be given to preserving semantics throughout the translation process. To simulate LLMs' long thought ability in MT, we first mine sentences containing similes or metaphors from existing literature books, and then develop a multi-agent framework to translate these sentences via long thought. In the multi-agent framework, a translator is used to iteratively translate the source sentence under the suggestions provided by an advisor. To ensure the effectiveness of the long thoughts, an evaluator is also employed to quantify the translation quality in each round. In this way, we collect tens of thousands of long-thought MT data, which is used to train our DRT. Using Qwen2.5 and LLama-3.1 as the backbones, DRT models can learn the thought process during machine translation, and outperform vanilla LLMs as well as LLMs which are simply fine-tuning on the paired sentences without long thought, showing its effectiveness. The synthesized data and model checkpoints are released at https://github.com/krystalan/DRT.
>
---
#### [replaced 062] Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.00187v2](http://arxiv.org/pdf/2503.00187v2)**

> **作者:** Hanjiang Hu; Alexander Robey; Changliu Liu
>
> **备注:** 23 pages, 10 figures, 11 tables
>
> **摘要:** Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .
>
---
#### [replaced 063] Zero-shot OCR Accuracy of Low-Resourced Languages: A Comparative Analysis on Sinhala and Tamil
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18264v2](http://arxiv.org/pdf/2507.18264v2)**

> **作者:** Nevidu Jayatilleke; Nisansa de Silva
>
> **备注:** 11 pages, 4 figures, 1 table, Accepted paper at Recent Advances in Natural Language Processing (RANLP) 2025
>
> **摘要:** Solving the problem of Optical Character Recognition (OCR) on printed text for Latin and its derivative scripts can now be considered settled due to the volumes of research done on English and other High-Resourced Languages (HRL). However, for Low-Resourced Languages (LRL) that use unique scripts, it remains an open problem. This study presents a comparative analysis of the zero-shot performance of six distinct OCR engines on two LRLs: Sinhala and Tamil. The selected engines include both commercial and open-source systems, aiming to evaluate the strengths of each category. The Cloud Vision API, Surya, Document AI, and Tesseract were evaluated for both Sinhala and Tamil, while Subasa OCR and EasyOCR were examined for only one language due to their limitations. The performance of these systems was rigorously analysed using five measurement techniques to assess accuracy at both the character and word levels. According to the findings, Surya delivered the best performance for Sinhala across all metrics, with a WER of 2.61%. Conversely, Document AI excelled across all metrics for Tamil, highlighted by a very low CER of 0.78%. In addition to the above analysis, we also introduce a novel synthetic Tamil OCR benchmarking dataset.
>
---
#### [replaced 064] Confidential Prompting: Privacy-preserving LLM Inference on Cloud
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.19134v4](http://arxiv.org/pdf/2409.19134v4)**

> **作者:** Caihua Li; In Gim; Lin Zhong
>
> **摘要:** This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.
>
---
#### [replaced 065] Let's Use ChatGPT To Write Our Paper! Benchmarking LLMs To Write the Introduction of a Research Paper
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14273v2](http://arxiv.org/pdf/2508.14273v2)**

> **作者:** Krishna Garg; Firoz Shaik; Sambaran Bandyopadhyay; Cornelia Caragea
>
> **备注:** 20 pages, 15 figures
>
> **摘要:** As researchers increasingly adopt LLMs as writing assistants, generating high-quality research paper introductions remains both challenging and essential. We introduce Scientific Introduction Generation (SciIG), a task that evaluates LLMs' ability to produce coherent introductions from titles, abstracts, and related works. Curating new datasets from NAACL 2025 and ICLR 2025 papers, we assess five state-of-the-art models, including both open-source (DeepSeek-v3, Gemma-3-12B, LLaMA 4-Maverick, MistralAI Small 3.1) and closed-source GPT-4o systems, across multiple dimensions: lexical overlap, semantic similarity, content coverage, faithfulness, consistency, citation correctness, and narrative quality. Our comprehensive framework combines automated metrics with LLM-as-a-judge evaluations. Results demonstrate LLaMA-4 Maverick's superior performance on most metrics, particularly in semantic similarity and faithfulness. Moreover, three-shot prompting consistently outperforms fewer-shot approaches. These findings provide practical insights into developing effective research writing assistants and set realistic expectations for LLM-assisted academic writing. To foster reproducibility and future research, we will publicly release all code and datasets.
>
---
#### [replaced 066] Playing with Voices: Tabletop Role-Playing Game Recordings as a Diarization Challenge
- **分类: cs.CL; cs.SD; I.5**

- **链接: [http://arxiv.org/pdf/2502.12714v2](http://arxiv.org/pdf/2502.12714v2)**

> **作者:** Lian Remme; Kevin Tang
>
> **备注:** 15 pages, 14 figures, published in NAACL Findings 2025
>
> **摘要:** This paper provides a proof of concept that audio of tabletop role-playing games (TTRPG) could serve as a challenge for diarization systems. TTRPGs are carried out mostly by conversation. Participants often alter their voices to indicate that they are talking as a fictional character. Audio processing systems are susceptible to voice conversion with or without technological assistance. TTRPG present a conversational phenomenon in which voice conversion is an inherent characteristic for an immersive gaming experience. This could make it more challenging for diarizers to pick the real speaker and determine that impersonating is just that. We present the creation of a small TTRPG audio dataset and compare it against the AMI and the ICSI corpus. The performance of two diarizers, pyannote.audio and wespeaker, were evaluated. We observed that TTRPGs' properties result in a higher confusion rate for both diarizers. Additionally, wespeaker strongly underestimates the number of speakers in the TTRPG audio files. We propose TTRPG audio as a promising challenge for diarization systems.
>
---
#### [replaced 067] Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18596v3](http://arxiv.org/pdf/2505.18596v3)**

> **作者:** Chen Han; Wenzhen Zheng; Xijin Tang
>
> **备注:** This paper has been accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** The proliferation of misinformation in digital platforms reveals the limitations of traditional detection methods, which mostly rely on static classification and fail to capture the intricate process of real-world fact-checking. Despite advancements in Large Language Models (LLMs) that enhance automated reasoning, their application to misinformation detection remains hindered by issues of logical inconsistency and superficial verification. In response, we introduce Debate-to-Detect (D2D), a novel Multi-Agent Debate (MAD) framework that reformulates misinformation detection as a structured adversarial debate. Inspired by fact-checking workflows, D2D assigns domain-specific profiles to each agent and orchestrates a five-stage debate process, including Opening Statement, Rebuttal, Free Debate, Closing Statement, and Judgment. To transcend traditional binary classification, D2D introduces a multi-dimensional evaluation mechanism that assesses each claim across five distinct dimensions: Factuality, Source Reliability, Reasoning Quality, Clarity, and Ethics. Experiments with GPT-4o on two datasets demonstrate significant improvements over baseline methods, and the case study highlight D2D's capability to iteratively refine evidence while improving decision transparency, representing a substantial advancement towards interpretable misinformation detection. The code will be released publicly after the official publication.
>
---
#### [replaced 068] Understanding Bias Reinforcement in LLM Agents Debate
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16814v4](http://arxiv.org/pdf/2503.16814v4)**

> **作者:** Jihwan Oh; Minchan Jeong; Jongwoo Ko; Se-Young Yun
>
> **备注:** 32 pages, 9 figures
>
> **摘要:** Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
>
---
#### [replaced 069] Backdoor Attacks on Dense Retrieval via Public and Unintentional Triggers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.13532v3](http://arxiv.org/pdf/2402.13532v3)**

> **作者:** Quanyu Long; Yue Deng; LeiLei Gan; Wenya Wang; Sinno Jialin Pan
>
> **备注:** Accepted by COLM 2025
>
> **摘要:** Dense retrieval systems have been widely used in various NLP applications. However, their vulnerabilities to potential attacks have been underexplored. This paper investigates a novel attack scenario where the attackers aim to mislead the retrieval system into retrieving the attacker-specified contents. Those contents, injected into the retrieval corpus by attackers, can include harmful text like hate speech or spam. Unlike prior methods that rely on model weights and generate conspicuous, unnatural outputs, we propose a covert backdoor attack triggered by grammar errors. Our approach ensures that the attacked models can function normally for standard queries while covertly triggering the retrieval of the attacker's contents in response to minor linguistic mistakes. Specifically, dense retrievers are trained with contrastive loss and hard negative sampling. Surprisingly, our findings demonstrate that contrastive loss is notably sensitive to grammatical errors, and hard negative sampling can exacerbate susceptibility to backdoor attacks. Our proposed method achieves a high attack success rate with a minimal corpus poisoning rate of only 0.048\%, while preserving normal retrieval performance. This indicates that the method has negligible impact on user experience for error-free queries. Furthermore, evaluations across three real-world defense strategies reveal that the malicious passages embedded within the corpus remain highly resistant to detection and filtering, underscoring the robustness and subtlety of the proposed attack \footnote{Codes of this work are available at https://github.com/ruyue0001/Backdoor_DPR.}.
>
---
#### [replaced 070] Trusted Knowledge Extraction for Operations and Maintenance Intelligence
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22935v2](http://arxiv.org/pdf/2507.22935v2)**

> **作者:** Kathleen P. Mealey; Jonathan A. Karr Jr.; Priscila Saboia Moreira; Paul R. Brenner; Charles F. Vardeman II
>
> **摘要:** Deriving operational intelligence from organizational data repositories is a key challenge due to the dichotomy of data confidentiality vs data integration objectives, as well as the limitations of Natural Language Processing (NLP) tools relative to the specific knowledge structure of domains such as operations and maintenance. In this work, we discuss Knowledge Graph construction and break down the Knowledge Extraction process into its Named Entity Recognition, Coreference Resolution, Named Entity Linking, and Relation Extraction functional components. We then evaluate sixteen NLP tools in concert with or in comparison to the rapidly advancing capabilities of Large Language Models (LLMs). We focus on the operational and maintenance intelligence use case for trusted applications in the aircraft industry. A baseline dataset is derived from a rich public domain US Federal Aviation Administration dataset focused on equipment failures or maintenance requirements. We assess the zero-shot performance of NLP and LLM tools that can be operated within a controlled, confidential environment (no data is sent to third parties). Based on our observation of significant performance limitations, we discuss the challenges related to trusted NLP and LLM tools as well as their Technical Readiness Level for wider use in mission-critical industries such as aviation. We conclude with recommendations to enhance trust and provide our open-source curated dataset to support further baseline testing and evaluation.
>
---
#### [replaced 071] DecisionFlow: Advancing Large Language Model as Principled Decision Maker
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21397v2](http://arxiv.org/pdf/2505.21397v2)**

> **作者:** Xiusi Chen; Shanyong Wang; Cheng Qian; Hongru Wang; Peixuan Han; Heng Ji
>
> **备注:** EMNLP 2025 Findings; 25 pages, 15 figures
>
> **摘要:** In high-stakes domains such as healthcare and finance, effective decision-making demands not just accurate outcomes but transparent and explainable reasoning. However, current language models often lack the structured deliberation needed for such tasks, instead generating decisions and justifications in a disconnected, post-hoc manner. To address this, we propose DecisionFlow, a novel decision modeling framework that guides models to reason over structured representations of actions, attributes, and constraints. Rather than predicting answers directly from prompts, DecisionFlow builds a semantically grounded decision space and infers a latent utility function to evaluate trade-offs in a transparent, utility-driven manner. This process produces decisions tightly coupled with interpretable rationales reflecting the model's reasoning. Empirical results on two high-stakes benchmarks show that DecisionFlow not only achieves up to 30% accuracy gains over strong prompting baselines but also enhances alignment in outcomes. Our work is a critical step toward integrating symbolic reasoning with LLMs, enabling more accountable, explainable, and reliable LLM decision support systems. Code and data are at https://github.com/xiusic/DecisionFlow.
>
---
#### [replaced 072] HeteroTune: Efficient Federated Learning for Large Heterogeneous Models
- **分类: cs.LG; cs.CL; cs.CV; cs.DC; 68T07; I.2.11**

- **链接: [http://arxiv.org/pdf/2411.16796v2](http://arxiv.org/pdf/2411.16796v2)**

> **作者:** Ruofan Jia; Weiying Xie; Jie Lei; Jitao Ma; Haonan Qin; Leyuan Fang
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** While large pre-trained models have achieved impressive performance across AI tasks, their deployment in privacy-sensitive and distributed environments remains challenging. Federated learning (FL) offers a viable solution by enabling decentralized fine-tuning without data sharing, but real-world applications face significant obstacles due to heterogeneous client resources in compute and memory. To address this, we propose HeteroTune, a novel federated fine-tuning paradigm for large, heterogeneous models operating under limited communication and computation budgets. The core of our method lies in a novel architecture, DeMA (Dense Mixture of Adapters), which enables flexible and efficient aggregation of heterogeneous models by preserving their full representational capacity while facilitating seamless cross-model knowledge fusion. We further introduce CMGA (Cross-Model Gradient Alignment), a lightweight yet effective mechanism that enhances training stability by harmonizing gradient directions across heterogeneous client models during aggregation, mitigating update conflicts and promoting more consistent convergence in federated settings. We provide both theoretical analysis and empirical evidence showing that HeteroTune achieves state-of-the-art performance and efficiency across diverse tasks and model architectures. For example, on LLaMA models, it reduces communication overhead by 99.5%, cuts peak memory usage by ~50%, and improves performance by 4.61%.
>
---
#### [replaced 073] LLM-Forest: Ensemble Learning of LLMs with Graph-Augmented Prompts for Data Imputation
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21520v4](http://arxiv.org/pdf/2410.21520v4)**

> **作者:** Xinrui He; Yikun Ban; Jiaru Zou; Tianxin Wei; Curtiss B. Cook; Jingrui He
>
> **摘要:** Missing data imputation is a critical challenge in various domains, such as healthcare and finance, where data completeness is vital for accurate analysis. Large language models (LLMs), trained on vast corpora, have shown strong potential in data generation, making them a promising tool for data imputation. However, challenges persist in designing effective prompts for a finetuning-free process and in mitigating biases and uncertainty in LLM outputs. To address these issues, we propose a novel framework, LLM-Forest, which introduces a "forest" of few-shot prompt learning LLM "trees" with their outputs aggregated via confidence-based weighted voting based on LLM self-assessment, inspired by the ensemble learning (Random Forest). This framework is established on a new concept of bipartite information graphs to identify high-quality relevant neighboring entries with both feature and value granularity. Extensive experiments on 9 real-world datasets demonstrate the effectiveness and efficiency of LLM-Forest.
>
---
#### [replaced 074] X-Teaming: Multi-Turn Jailbreaks and Defenses with Adaptive Multi-Agents
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.13203v2](http://arxiv.org/pdf/2504.13203v2)**

> **作者:** Salman Rahman; Liwei Jiang; James Shiffer; Genglin Liu; Sheriff Issaka; Md Rizwan Parvez; Hamid Palangi; Kai-Wei Chang; Yejin Choi; Saadia Gabriel
>
> **摘要:** Multi-turn interactions with language models (LMs) pose critical safety risks, as harmful intent can be strategically spread across exchanges. Yet, the vast majority of prior work has focused on single-turn safety, while adaptability and diversity remain among the key challenges of multi-turn red-teaming. To address these challenges, we present X-Teaming, a scalable framework that systematically explores how seemingly harmless interactions escalate into harmful outcomes and generates corresponding attack scenarios. X-Teaming employs collaborative agents for planning, attack optimization, and verification, achieving state-of-the-art multi-turn jailbreak effectiveness and diversity with success rates up to 98.1% across representative leading open-weight and closed-source models. In particular, X-Teaming achieves a 96.2% attack success rate against the latest Claude 3.7 Sonnet model, which has been considered nearly immune to single-turn attacks. Building on X-Teaming, we introduce XGuard-Train, an open-source multi-turn safety training dataset that is 20x larger than the previous best resource, comprising 30K interactive jailbreaks, designed to enable robust multi-turn safety alignment for LMs. Our work offers essential tools and insights for mitigating sophisticated conversational attacks, advancing the multi-turn safety of LMs.
>
---
#### [replaced 075] Evaluating $n$-Gram Novelty of Language Models Using Rusty-DAWG
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.13069v4](http://arxiv.org/pdf/2406.13069v4)**

> **作者:** William Merrill; Noah A. Smith; Yanai Elazar
>
> **备注:** To appear at EMNLP 2024
>
> **摘要:** How novel are texts generated by language models (LMs) relative to their training corpora? In this work, we investigate the extent to which modern LMs generate $n$-grams from their training data, evaluating both (i) the probability LMs assign to complete training $n$-grams and (ii) $n$-novelty, the proportion of $n$-grams generated by an LM that did not appear in the training data (for arbitrarily large $n$). To enable arbitrary-length $n$-gram search over a corpus in constant time w.r.t. corpus size, we develop Rusty-DAWG, a novel search tool inspired by indexing of genomic data. We compare the novelty of LM-generated text to human-written text and explore factors that affect generation novelty, focusing on the Pythia models. We find that, for $n > 4$, LM-generated text is less novel than human-written text, though it is more novel for smaller $n$. Larger LMs and more constrained decoding strategies both decrease novelty. Finally, we show that LMs complete $n$-grams with lower loss if they are more frequent in the training data. Overall, our results reveal factors influencing the novelty of LM-generated text, and we release Rusty-DAWG to facilitate further pretraining data research.
>
---
#### [replaced 076] Detecting Knowledge Boundary of Vision Large Language Models by Sampling-Based Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18023v3](http://arxiv.org/pdf/2502.18023v3)**

> **作者:** Zhuo Chen; Xinyu Wang; Yong Jiang; Zhen Zhang; Xinyu Geng; Pengjun Xie; Fei Huang; Kewei Tu
>
> **备注:** EMNLP2025 Main Conference
>
> **摘要:** Despite the advancements made in Vision Large Language Models (VLLMs), like text Large Language Models (LLMs), they have limitations in addressing questions that require real-time information or are knowledge-intensive. Indiscriminately adopting Retrieval Augmented Generation (RAG) techniques is an effective yet expensive way to enable models to answer queries beyond their knowledge scopes. To mitigate the dependence on retrieval and simultaneously maintain, or even improve, the performance benefits provided by retrieval, we propose a method to detect the knowledge boundary of VLLMs, allowing for more efficient use of techniques like RAG. Specifically, we propose a method with two variants that fine-tune a VLLM on an automatically constructed dataset for boundary identification. Experimental results on various types of Visual Question Answering datasets show that our method successfully depicts a VLLM's knowledge boundary, based on which we are able to reduce indiscriminate retrieval while maintaining or improving the performance. In addition, we show that the knowledge boundary identified by our method for one VLLM can be used as a surrogate boundary for other VLLMs. Code will be released at https://github.com/Chord-Chen-30/VLLM-KnowledgeBoundary
>
---
#### [replaced 077] Evaluating Contrast Localizer for Identifying Causal Units in Social & Mathematical Tasks in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08276v3](http://arxiv.org/pdf/2508.08276v3)**

> **作者:** Yassine Jamaa; Badr AlKhamissi; Satrajit Ghosh; Martin Schrimpf
>
> **备注:** Accepted at the Interplay of Model Behavior and Model Internals Workshop co-located with COLM 2025
>
> **摘要:** This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units.
>
---
#### [replaced 078] Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18556v2](http://arxiv.org/pdf/2505.18556v2)**

> **作者:** Jun Zhuang; Haibo Jin; Ye Zhang; Zhengjian Kang; Wenbin Zhang; Gaby G. Dagher; Haohan Wang
>
> **备注:** Accepted for EMNLP'25 Findings. TL;DR: We propose a new two-stage intent-based prompt-refinement framework, IntentPrompt, that aims to explore the vulnerability of LLMs' content moderation guardrails by refining prompts into benign-looking declarative forms via intent manipulation for red-teaming purposes
>
> **摘要:** Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.
>
---
#### [replaced 079] A Factorized Probabilistic Model of the Semantics of Vague Temporal Adverbials Relative to Different Event Types
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.01311v2](http://arxiv.org/pdf/2505.01311v2)**

> **作者:** Svenja Kenneweg; Jörg Deigmöller; Julian Eggert; Philipp Cimiano
>
> **备注:** Camera-ready version of the paper accepted at the 2025 Annual Meeting of the Cognitive Science Society (CogSci 2025). Published proceedings version: https://escholarship.org/uc/item/623599f4
>
> **摘要:** Vague temporal adverbials, such as recently, just, and a long time ago, describe the temporal distance between a past event and the utterance time but leave the exact duration underspecified. In this paper, we introduce a factorized model that captures the semantics of these adverbials as probabilistic distributions. These distributions are composed with event-specific distributions to yield a contextualized meaning for an adverbial applied to a specific event. We fit the model's parameters using existing data capturing judgments of native speakers regarding the applicability of these vague temporal adverbials to events that took place a given time ago. Comparing our approach to a non-factorized model based on a single Gaussian distribution for each pair of event and temporal adverbial, we find that while both models have similar predictive power, our model is preferable in terms of Occam's razor, as it is simpler and has better extendability.
>
---
#### [replaced 080] OpenHuEval: Evaluating Large Language Model on Hungarian Specifics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21500v2](http://arxiv.org/pdf/2503.21500v2)**

> **作者:** Haote Yang; Xingjian Wei; Jiang Wu; Noémi Ligeti-Nagy; Jiaxing Sun; Yinfan Wang; Zijian Győző Yang; Junyuan Gao; Jingchao Wang; Bowen Jiang; Shasha Wang; Nanjun Yu; Zihao Zhang; Shixin Hong; Hongwei Liu; Wei Li; Songyang Zhang; Dahua Lin; Lijun Wu; Gábor Prószéky; Conghui He
>
> **摘要:** We introduce OpenHuEval, the first benchmark for LLMs focusing on the Hungarian language and specifics. OpenHuEval is constructed from a vast collection of Hungarian-specific materials sourced from multiple origins. In the construction, we incorporated the latest design principles for evaluating LLMs, such as using real user queries from the internet, emphasizing the assessment of LLMs' generative capabilities, and employing LLM-as-judge to enhance the multidimensionality and accuracy of evaluations. Ultimately, OpenHuEval encompasses eight Hungarian-specific dimensions, featuring five tasks and 3953 questions. Consequently, OpenHuEval provides the comprehensive, in-depth, and scientifically accurate assessment of LLM performance in the context of the Hungarian language and its specifics. We evaluated current mainstream LLMs, including both traditional LLMs and recently developed Large Reasoning Models. The results demonstrate the significant necessity for evaluation and model optimization tailored to the Hungarian language and specifics. We also established the framework for analyzing the thinking processes of LRMs with OpenHuEval, revealing intrinsic patterns and mechanisms of these models in non-English languages, with Hungarian serving as a representative example. We will release OpenHuEval at https://github.com/opendatalab/OpenHuEval .
>
---
#### [replaced 081] Missing Melodies: AI Music Generation and its "Nearly" Complete Omission of the Global South
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.04100v3](http://arxiv.org/pdf/2412.04100v3)**

> **作者:** Atharva Mehta; Shivam Chauhan; Monojit Choudhury
>
> **备注:** Submitted to CACM, 12 pages, 2 figures
>
> **摘要:** Recent advances in generative AI have sparked renewed interest and expanded possibilities for music generation. However, the performance and versatility of these systems across musical genres are heavily influenced by the availability of training data. We conducted an extensive analysis of over one million hours of audio datasets used in AI music generation research and manually reviewed more than 200 papers from eleven prominent AI and music conferences and organizations (AAAI, ACM, EUSIPCO, EURASIP, ICASSP, ICML, IJCAI, ISMIR, NeurIPS, NIME, SMC) to identify a critical gap in the fair representation and inclusion of the musical genres of the Global South in AI research. Our findings reveal a stark imbalance: approximately 86% of the total dataset hours and over 93% of researchers focus primarily on music from the Global North. However, around 40% of these datasets include some form of non-Western music, genres from the Global South account for only 14.6% of the data. Furthermore, approximately 51% of the papers surveyed concentrate on symbolic music generation, a method that often fails to capture the cultural nuances inherent in music from regions such as South Asia, the Middle East, and Africa. As AI increasingly shapes the creation and dissemination of music, the significant underrepresentation of music genres in datasets and research presents a serious threat to global musical diversity. We also propose some important steps to mitigate these risks and foster a more inclusive future for AI-driven music generation.
>
---
#### [replaced 082] ImF: Implicit Fingerprint for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21805v3](http://arxiv.org/pdf/2503.21805v3)**

> **作者:** Jiaxuan Wu; Wanli Peng; Hang Fu; Yiming Xue; Juan Wen
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.
>
---
#### [replaced 083] Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia: A Proof-of-Concept Study
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.17054v2](http://arxiv.org/pdf/2409.17054v2)**

> **作者:** Nur Ahmad Khatim; Azmul Asmar Irfan; Mansur M. Arief
>
> **摘要:** One of the critical issues contributing to inefficiency in Puskesmas (Indonesian community health centers) is the time-consuming nature of documenting doctor-patient interactions. Doctors must conduct thorough consultations and manually transcribe detailed notes into ePuskesmas electronic health records (EHR), which creates substantial administrative burden to already overcapacitated physicians. This paper presents a proof-of-concept framework using large language models (LLMs) to automate real-time transcription and summarization of doctor-patient conversations in Bahasa Indonesia. Our system combines Whisper model for transcription with GPT-3.5 for medical summarization, implemented as a browser extension that automatically populates ePuskesmas forms. Through controlled roleplay experiments with medical validation, we demonstrate the technical feasibility of processing detailed 300+ seconds trimmed consultations in under 30 seconds while maintaining clinical accuracy. This work establishes the foundation for AI-assisted clinical documentation in resource-constrained healthcare environments. However, concerns have also been raised regarding privacy compliance and large-scale clinical evaluation addressing language and cultural biases for LLMs.
>
---
#### [replaced 084] GoalfyMax: A Protocol-Driven Multi-Agent System for Intelligent Experience Entities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09497v2](http://arxiv.org/pdf/2507.09497v2)**

> **作者:** Siyi Wu; Zeyu Wang; Xinyuan Song; Zhengpeng Zhou; Lifan Sun; Tianyu Shi
>
> **备注:** The author information is incorrect, some contributors are not included, and the submission has not been approved by all authors
>
> **摘要:** Modern enterprise environments demand intelligent systems capable of handling complex, dynamic, and multi-faceted tasks with high levels of autonomy and adaptability. However, traditional single-purpose AI systems often lack sufficient coordination, memory reuse, and task decomposition capabilities, limiting their scalability in realistic settings. To address these challenges, we present \textbf{GoalfyMax}, a protocol-driven framework for end-to-end multi-agent collaboration. GoalfyMax introduces a standardized Agent-to-Agent (A2A) communication layer built on the Model Context Protocol (MCP), allowing independent agents to coordinate through asynchronous, protocol-compliant interactions. It incorporates the Experience Pack (XP) architecture, a layered memory system that preserves both task rationales and execution traces, enabling structured knowledge retention and continual learning. Moreover, our system integrates advanced features including multi-turn contextual dialogue, long-short term memory modules, and dynamic safety validation, supporting robust, real-time strategy adaptation. Empirical results on complex task orchestration benchmarks and case study demonstrate that GoalfyMax achieves superior adaptability, coordination, and experience reuse compared to baseline frameworks. These findings highlight its potential as a scalable, future-ready foundation for multi-agent intelligent systems.
>
---
#### [replaced 085] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05587v2](http://arxiv.org/pdf/2506.05587v2)**

> **作者:** Junjie Xing; Yeye He; Mengyu Zhou; Haoyu Dong; Shi Han; Lingjiao Chen; Dongmei Zhang; Surajit Chaudhuri; H. V. Jagadish
>
> **备注:** Included additional benchmark results covering 24 LLMs
>
> **摘要:** Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area. In this work, we introduce MMTU, a large-scale benchmark with over 30K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI o4-mini and DeepSeek R1 score only around 60%, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at https://github.com/MMTU-Benchmark/MMTU and https://huggingface.co/datasets/MMTU-benchmark/MMTU.
>
---
#### [replaced 086] Mitigating Jailbreaks with Intent-Aware LLMs
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12072v2](http://arxiv.org/pdf/2508.12072v2)**

> **作者:** Wei Jie Yeo; Ranjan Satapathy; Erik Cambria
>
> **摘要:** Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.
>
---
#### [replaced 087] TombRaider: Entering the Vault of History to Jailbreak Large Language Models
- **分类: cs.CR; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2501.18628v2](http://arxiv.org/pdf/2501.18628v2)**

> **作者:** Junchen Ding; Jiahao Zhang; Yi Liu; Ziqi Ding; Gelei Deng; Yuekang Li
>
> **备注:** Main Conference of EMNLP
>
> **摘要:** Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes. Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.
>
---
#### [replaced 088] CoLMbo: Speaker Language Model for Descriptive Profiling
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.09375v2](http://arxiv.org/pdf/2506.09375v2)**

> **作者:** Massa Baali; Shuo Han; Syed Abdul Hannan; Purusottam Samal; Karanveer Singh; Soham Deshmukh; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner. This paper introduces CoLMbo, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to adapt dynamically to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach not only enhances traditional speaker profiling but also excels in zero-shot scenarios across diverse datasets, marking a significant advancement in the field of speaker recognition.
>
---
#### [replaced 089] IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection
- **分类: cs.CL; cs.AI; cs.CV; 68T50; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.16258v2](http://arxiv.org/pdf/2505.16258v2)**

> **作者:** Aashish Anantha Ramakrishnan; Aadarsh Anantha Ramakrishnan; Dongwon Lee
>
> **备注:** Accepted in the COLM First Workshop on Pragmatic Reasoning in Language Models (PragLM), Montreal, Canada, October 2025, https://sites.google.com/berkeley.edu/praglm
>
> **摘要:** Interpreting figurative language such as sarcasm across multi-modal inputs presents unique challenges, often requiring task-specific fine-tuning and extensive reasoning steps. However, current Chain-of-Thought approaches do not efficiently leverage the same cognitive processes that enable humans to identify sarcasm. We present IRONIC, an in-context learning framework that leverages Multi-modal Coherence Relations to analyze referential, analogical and pragmatic image-text linkages. Our experiments show that IRONIC achieves state-of-the-art performance on zero-shot Multi-modal Sarcasm Detection across different baselines. This demonstrates the need for incorporating linguistic and cognitive insights into the design of multi-modal reasoning strategies. Our code is available at: https://github.com/aashish2000/IRONIC
>
---
#### [replaced 090] AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.05140v2](http://arxiv.org/pdf/2506.05140v2)**

> **作者:** Chih-Kai Yang; Neo Ho; Yi-Jyun Lee; Hung-yi Lee
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements.
>
---
#### [replaced 091] A2HCoder: An LLM-Driven Coding Agent for Hierarchical Algorithm-to-HDL Translation
- **分类: cs.CL; cs.AR; cs.PL**

- **链接: [http://arxiv.org/pdf/2508.10904v2](http://arxiv.org/pdf/2508.10904v2)**

> **作者:** Jie Lei; Ruofan Jia; J. Andrew Zhang; Hao Zhang
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** In wireless communication systems, stringent requirements such as ultra-low latency and power consumption have significantly increased the demand for efficient algorithm-to-hardware deployment. However, a persistent and substantial gap remains between algorithm design and hardware implementation. Bridging this gap traditionally requires extensive domain expertise and time-consuming manual development, due to fundamental mismatches between high-level programming languages like MATLAB and hardware description languages (HDLs) such as Verilog-in terms of memory access patterns, data processing manners, and datatype representations. To address this challenge, we propose A2HCoder: a Hierarchical Algorithm-to-HDL Coding Agent, powered by large language models (LLMs), designed to enable agile and reliable algorithm-to-hardware translation. A2HCoder introduces a hierarchical framework that enhances both robustness and interpretability while suppressing common hallucination issues in LLM-generated code. In the horizontal dimension, A2HCoder decomposes complex algorithms into modular functional blocks, simplifying code generation and improving consistency. In the vertical dimension, instead of relying on end-to-end generation, A2HCoder performs step-by-step, fine-grained translation, leveraging external toolchains such as MATLAB and Vitis HLS for debugging and circuit-level synthesis. This structured process significantly mitigates hallucinations and ensures hardware-level correctness. We validate A2HCoder through a real-world deployment case in the 5G wireless communication domain, demonstrating its practicality, reliability, and deployment efficiency.
>
---
#### [replaced 092] EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.04424v2](http://arxiv.org/pdf/2502.04424v2)**

> **作者:** He Hu; Yucheng Zhou; Lianzhong You; Hongbo Xu; Qianning Wang; Zheng Lian; Fei Richard Yu; Fei Ma; Laizhong Cui
>
> **摘要:** With the integration of Multimodal large language models (MLLMs) into robotic systems and various AI applications, embedding emotional intelligence (EI) capabilities into these models is essential for enabling robots to effectively address human emotional needs and interact seamlessly in real-world scenarios. Existing static, text-based, or text-image benchmarks overlook the multimodal complexities of real-world interactions and fail to capture the dynamic, multimodal nature of emotional expressions, making them inadequate for evaluating MLLMs' EI. Based on established psychological theories of EI, we build EmoBench-M, a novel benchmark designed to evaluate the EI capability of MLLMs across 13 valuation scenarios from three key dimensions: foundational emotion recognition, conversational emotion understanding, and socially complex emotion analysis. Evaluations of both open-source and closed-source MLLMs on EmoBench-M reveal a significant performance gap between them and humans, highlighting the need to further advance their EI capabilities. All benchmark resources, including code and datasets, are publicly available at https://emo-gml.github.io/.
>
---
#### [replaced 093] Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.00451v3](http://arxiv.org/pdf/2502.00451v3)**

> **作者:** Aishik Mandal; Tanmoy Chakraborty; Iryna Gurevych
>
> **备注:** 18 pages, 2 figures, Accepted in Nature Computational Science
>
> **摘要:** Mental health disorders create profound personal and societal burdens, yet conventional diagnostics are resource-intensive and limit accessibility. Advances in artificial intelligence, particularly natural language processing and multimodal methods, offer promise for detecting and addressing mental disorders, but raise critical privacy risks. This paper examines these challenges and proposes solutions, including anonymization, synthetic data, and privacy-preserving training, while outlining frameworks for privacy-utility trade-offs, aiming to advance reliable, privacy-aware AI tools that support clinical decision-making and improve mental health outcomes.
>
---
#### [replaced 094] Seeing Sarcasm Through Different Eyes: Analyzing Multimodal Sarcasm Perception in Large Vision-Language Models
- **分类: cs.CL; cs.MM; cs.SI**

- **链接: [http://arxiv.org/pdf/2503.12149v3](http://arxiv.org/pdf/2503.12149v3)**

> **作者:** Junjie Chen; Xuyang Liu; Subin Huang; Linfeng Zhang; Hang Yu
>
> **摘要:** With the advent of large vision-language models (LVLMs) demonstrating increasingly human-like abilities, a pivotal question emerges: do different LVLMs interpret multimodal sarcasm differently, and can a single model grasp sarcasm from multiple perspectives like humans? To explore this, we introduce an analytical framework using systematically designed prompts on existing multimodal sarcasm datasets. Evaluating 12 state-of-the-art LVLMs over 2,409 samples, we examine interpretive variations within and across models, focusing on confidence levels, alignment with dataset labels, and recognition of ambiguous "neutral" cases. We further validate our findings on a diverse 100-sample mini-benchmark, incorporating multiple datasets, expanded prompt variants, and representative commercial LVLMs. Our findings reveal notable discrepancies -- across LVLMs and within the same model under varied prompts. While classification-oriented prompts yield higher internal consistency, models diverge markedly when tasked with interpretive reasoning. These results challenge binary labeling paradigms by highlighting sarcasm's subjectivity. We advocate moving beyond rigid annotation schemes toward multi-perspective, uncertainty-aware modeling, offering deeper insights into multimodal sarcasm comprehension. Our code and data are available at: https://github.com/CoderChen01/LVLMSarcasmAnalysis
>
---
#### [replaced 095] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15075v5](http://arxiv.org/pdf/2505.15075v5)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** The first version of this paper mistakenly included a prompt injection phrase, which was inappropriate and unprofessional. Although we corrected the version on arXiv and withdrew from the conference, my co-authors and university strongly request a full withdrawal. Given the situation, I no longer have the authority to manage this paper, and withdrawing it from arXiv is the most responsible action
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
#### [replaced 096] A Factuality and Diversity Reconciled Decoding Method for Knowledge-Grounded Dialogue Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.05718v2](http://arxiv.org/pdf/2407.05718v2)**

> **作者:** Chenxu Yang; Zheng Lin; Chong Tian; Liang Pang; Lanrui Wang; Zhengyang Tong; Qirong Ho; Yanan Cao; Weiping Wang
>
> **摘要:** Grounding external knowledge can enhance the factuality of responses in dialogue generation. However, excessive emphasis on it might result in the lack of engaging and diverse expressions. Through the introduction of randomness in sampling, current approaches can increase the diversity. Nevertheless, such sampling method could undermine the factuality in dialogue generation. In this study, to discover a solution for advancing creativity without relying on questionable randomness and to subtly reconcile the factuality and diversity within the source-grounded paradigm, a novel method named DoGe is proposed. DoGe can dynamically alternate between the utilization of internal parameter knowledge and external source knowledge based on the model's factual confidence. Extensive experiments on three widely-used datasets show that DoGe can not only enhance response diversity but also maintain factuality, and it significantly surpasses other various decoding strategy baselines.
>
---
#### [replaced 097] Does GPT-4 surpass human performance in linguistic pragmatics?
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2312.09545v2](http://arxiv.org/pdf/2312.09545v2)**

> **作者:** Ljubisa Bojic; Predrag Kovacevic; Milan Cabarkapa
>
> **备注:** 19 pages, 1 figure, 2 tables
>
> **摘要:** As Large Language Models (LLMs) become increasingly integrated into everyday life as general purpose multimodal AI systems, their capabilities to simulate human understanding are under examination. This study investigates LLMs ability to interpret linguistic pragmatics, which involves context and implied meanings. Using Grice communication principles, we evaluated both LLMs (GPT-2, GPT-3, GPT-3.5, GPT-4, and Bard) and human subjects (N = 147) on dialogue-based tasks. Human participants included 71 primarily Serbian students and 76 native English speakers from the United States. Findings revealed that LLMs, particularly GPT-4, outperformed humans. GPT4 achieved the highest score of 4.80, surpassing the best human score of 4.55. Other LLMs performed well: GPT 3.5 scored 4.10, Bard 3.75, and GPT-3 3.25. GPT-2 had the lowest score of 1.05. The average LLM score was 3.39, exceeding the human cohorts averages of 2.80 (Serbian students) and 2.34 (U.S. participants). In the ranking of all 155 subjects (including LLMs and humans), GPT-4 secured the top position, while the best human ranked second. These results highlight significant progress in LLMs ability to simulate understanding of linguistic pragmatics. Future studies should confirm these findings with more dialogue-based tasks and diverse participants. This research has important implications for advancing general-purpose AI models in various communication-centered tasks, including potential application in humanoid robots in the future.
>
---
#### [replaced 098] TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.23266v2](http://arxiv.org/pdf/2410.23266v2)**

> **作者:** Ziyao Shangguan; Chuhan Li; Yuxuan Ding; Yanan Zheng; Yilun Zhao; Tesca Fitzgerald; Arman Cohan
>
> **摘要:** Existing benchmarks often highlight the remarkable performance achieved by state-of-the-art Multimodal Foundation Models (MFMs) in leveraging temporal context for video understanding. However, how well do the models truly perform visual temporal reasoning? Our study of existing benchmarks shows that this capability of MFMs is likely overestimated as many questions can be solved by using a single, few, or out-of-order frames. To systematically examine current visual temporal reasoning tasks, we propose three principles with corresponding metrics: (1) Multi-Frame Gain, (2) Frame Order Sensitivity, and (3) Frame Information Disparity. Following these principles, we introduce TOMATO, Temporal Reasoning Multimodal Evaluation, a novel benchmark crafted to rigorously assess MFMs' temporal reasoning capabilities in video understanding. TOMATO comprises 1,484 carefully curated, human-annotated questions spanning six tasks (i.e., action count, direction, rotation, shape & trend, velocity & frequency, and visual cues), applied to 1,417 videos, including 805 self-recorded and -generated videos, that encompass human-centric, real-world, and simulated scenarios. Our comprehensive evaluation reveals a human-model performance gap of 57.3% with the best-performing model. Moreover, our in-depth analysis uncovers more fundamental limitations beyond this gap in current MFMs. While they can accurately recognize events in isolated frames, they fail to interpret these frames as a continuous sequence. We believe TOMATO will serve as a crucial testbed for evaluating the next-generation MFMs and as a call to the community to develop AI systems capable of comprehending human world dynamics through the video modality.
>
---
#### [replaced 099] Versatile Framework for Song Generation with Prompt-based Control
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.19062v5](http://arxiv.org/pdf/2504.19062v5)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Ruiqi Li; Jingyu Lu; Rongjie Huang; Ruiyuan Zhang; Zhiqing Hong; Ziyue Jiang; Zhou Zhao
>
> **备注:** Accepted by Findings of EMNLP 2025
>
> **摘要:** Song generation focuses on producing controllable high-quality songs based on various prompts. However, existing methods struggle to generate vocals and accompaniments with prompt-based control and proper alignment. Additionally, they fall short in supporting various tasks. To address these challenges, we introduce VersBand, a multi-task song generation framework for synthesizing high-quality, aligned songs with prompt-based control. VersBand comprises these primary models: 1) VocalBand, a decoupled model, leverages the flow-matching method for generating singing styles, pitches, and mel-spectrograms, allowing fast, high-quality vocal generation with style control. 2) AccompBand, a flow-based transformer model, incorporates the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control. This model allows for generating controllable, high-quality accompaniments aligned with vocals. 3) Two generation models, LyricBand for lyrics and MelodyBand for melodies, contribute to the comprehensive multi-task song generation system, allowing for extensive control based on multiple prompts. Experimental results show that VersBand outperforms baseline models across multiple song generation tasks using objective and subjective metrics. Demos and codes are available at https://aaronz345.github.io/VersBandDemo and https://github.com/AaronZ345/VersBand.
>
---
#### [replaced 100] Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2308.15022v4](http://arxiv.org/pdf/2308.15022v4)**

> **作者:** Qingyue Wang; Yanhe Fu; Yanan Cao; Shuai Wang; Zhiliang Tian; Liang Ding
>
> **备注:** This paper has been accepted by Neurocomputing
>
> **摘要:** Recently, large language models (LLMs), such as GPT-4, stand out remarkable conversational abilities, enabling them to engage in dynamic and contextually relevant dialogues across a wide range of topics. However, given a long conversation, these chatbots fail to recall past information and tend to generate inconsistent responses. To address this, we propose to recursively generate summaries/ memory using large language models (LLMs) to enhance long-term memory ability. Specifically, our method first stimulates LLMs to memorize small dialogue contexts and then recursively produce new memory using previous memory and following contexts. Finally, the chatbot can easily generate a highly consistent response with the help of the latest memory. We evaluate our method on both open and closed LLMs, and the experiments on the widely-used public dataset show that our method can generate more consistent responses in a long-context conversation. Also, we show that our strategy could nicely complement both long-context (e.g., 8K and 16K) and retrieval-enhanced LLMs, bringing further long-term dialogue performance. Notably, our method is a potential solution to enable the LLM to model the extremely long context. The code and scripts are released.
>
---
#### [replaced 101] Large Language Models Meet NLP: A Survey
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.12819v2](http://arxiv.org/pdf/2405.12819v2)**

> **作者:** Libo Qin; Qiguang Chen; Xiachong Feng; Yang Wu; Yongheng Zhang; Yinghui Li; Min Li; Wanxiang Che; Philip S. Yu
>
> **备注:** The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-025-50472-3}
>
> **摘要:** While large language models (LLMs) like ChatGPT have shown impressive capabilities in Natural Language Processing (NLP) tasks, a systematic investigation of their potential in this field remains largely unexplored. This study aims to address this gap by exploring the following questions: (1) How are LLMs currently applied to NLP tasks in the literature? (2) Have traditional NLP tasks already been solved with LLMs? (3) What is the future of the LLMs for NLP? To answer these questions, we take the first step to provide a comprehensive overview of LLMs in NLP. Specifically, we first introduce a unified taxonomy including (1) parameter-frozen paradigm and (2) parameter-tuning paradigm to offer a unified perspective for understanding the current progress of LLMs in NLP. Furthermore, we summarize the new frontiers and the corresponding challenges, aiming to inspire further groundbreaking advancements. We hope this work offers valuable insights into the potential and limitations of LLMs, while also serving as a practical guide for building effective LLMs in NLP.
>
---
#### [replaced 102] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2505.10924v3](http://arxiv.org/pdf/2505.10924v3)**

> **作者:** Ada Chen; Yongjiang Wu; Junyuan Zhang; Jingyu Xiao; Shu Yang; Jen-tse Huang; Kun Wang; Wenxuan Wang; Shuai Wang
>
> **摘要:** Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents.
>
---
#### [replaced 103] Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks
- **分类: cs.AI; cs.CL; 68T05; I.2.6; I.2.8; H.3.3**

- **链接: [http://arxiv.org/pdf/2504.08525v4](http://arxiv.org/pdf/2504.08525v4)**

> **作者:** Ye Ye
>
> **备注:** 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
>
> **摘要:** Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. A reference implementation of the core TME components is available at https://github.com/biubiutomato/TME-Agent, including basic examples and structured memory integration. While the current implementation uses a tree-based structure, TME is designed to be graph-aware, supporting reusable substeps, converging task paths, and shared dependencies. This lays the groundwork for future DAG-based memory architectures.
>
---
#### [replaced 104] Orthogonal Finetuning for Direct Preference Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.14836v3](http://arxiv.org/pdf/2409.14836v3)**

> **作者:** Chenxu Yang; Ruipeng Jia; Naibin Gu; Zheng Lin; Siyuan Chen; Chao Pang; Weichong Yin; Yu Sun; Hua Wu; Weiping Wang
>
> **摘要:** DPO is an effective preference optimization algorithm. However, the DPO-tuned models tend to overfit on the dispreferred samples, manifested as overly long generations lacking diversity. While recent regularization approaches have endeavored to alleviate this issue by modifying the objective function, they achieved that at the cost of alignment performance degradation. In this paper, we innovatively incorporate regularization from the perspective of weight updating to curb alignment overfitting. Through the pilot experiment, we discovered that there exists a positive correlation between overfitting and the hyperspherical energy fluctuation. Hence, we introduce orthogonal finetuning for DPO via a weight-Rotated Preference Optimization (RoPO) method, which merely conducts rotational and magnitude-stretching updates on the weight parameters to maintain the hyperspherical energy invariant, thereby preserving the knowledge encoded in the angle between neurons. Extensive experiments demonstrate that our model aligns perfectly with human preferences while retaining the original expressive capacity using only 0.0086% of the trainable parameters, suggesting an effective regularization against overfitting. Specifically, RoPO outperforms DPO by up to 10 points on MT-Bench and by up to 2.8 points on AlpacaEval 2, while enhancing the generation diversity by an average of 6 points.
>
---
#### [replaced 105] AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark
- **分类: cs.DB; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06724v3](http://arxiv.org/pdf/2412.06724v3)**

> **作者:** Lan Li; Liri Fang; Bertram Ludäscher; Vetle I. Torvik
>
> **备注:** EMNLP Findings, 2025
>
> **摘要:** Data cleaning is a time-consuming and error-prone manual process, even with modern workflow tools such as OpenRefine. We present AutoDCWorkflow, an LLM-based pipeline for automatically generating data-cleaning workflows. The pipeline takes a raw table and a data analysis purpose, and generates a sequence of OpenRefine operations designed to produce a minimal, clean table sufficient to address the purpose. Six operations correspond to common data quality issues, including format inconsistencies, type errors, and duplicates. To evaluate AutoDCWorkflow, we create a benchmark with metrics assessing answers, data, and workflow quality for 142 purposes using 96 tables across six topics. The evaluation covers three key dimensions: (1) Purpose Answer: can the cleaned table produce a correct answer? (2) Column (Value): how closely does it match the ground truth table? (3) Workflow (Operations): to what extent does the generated workflow resemble the human-curated ground truth? Experiments show that Llama 3.1, Mistral, and Gemma 2 significantly enhance data quality, outperforming the baseline across all metrics. Gemma 2-27B consistently generates high-quality tables and answers, while Gemma 2-9B excels in producing workflows that closely resemble human-annotated versions.
>
---
#### [replaced 106] More Women, Same Stereotypes: Unpacking the Gender Bias Paradox in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15904v3](http://arxiv.org/pdf/2503.15904v3)**

> **作者:** Evan Chen; Run-Jun Zhan; Yan-Bai Lin; Hung-Hsuan Chen
>
> **摘要:** Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases. This study introduces a novel evaluation framework to uncover gender biases in LLMs: using free-form storytelling to surface biases embedded within the models. A systematic analysis of ten prominent LLMs shows a consistent pattern of overrepresenting female characters across occupations, likely due to supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Paradoxically, despite this overrepresentation, the occupational gender distributions produced by these LLMs align more closely with human stereotypes than with real-world labor data. This highlights the challenge and importance of implementing balanced mitigation measures to promote fairness and prevent the establishment of potentially new biases. We release the prompts and LLM-generated stories at GitHub.
>
---
#### [replaced 107] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13357v2](http://arxiv.org/pdf/2507.13357v2)**

> **作者:** Atharva Bhargude; Ishan Gonehal; Dave Yoon; Kaustubh Vinnakota; Chandler Haney; Aaron Sandoval; Kevin Zhu
>
> **备注:** Published at ACL 2025 SRW, 9 pages, 3 figures
>
> **摘要:** Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93, surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.
>
---
#### [replaced 108] LETToT: Label-Free Evaluation of Large Language Models On Tourism Using Expert Tree-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.11280v2](http://arxiv.org/pdf/2508.11280v2)**

> **作者:** Ruiyan Qi; Congding Wen; Weibo Zhou; Jiwei Li; Shangsong Liang; Lingbo Li
>
> **摘要:** Evaluating large language models (LLMs) in specific domain like tourism remains challenging due to the prohibitive cost of annotated benchmarks and persistent issues like hallucinations. We propose $\textbf{L}$able-Free $\textbf{E}$valuation of LLM on $\textbf{T}$ourism using Expert $\textbf{T}$ree-$\textbf{o}$f-$\textbf{T}$hought (LETToT), a framework that leverages expert-derived reasoning structures-instead of labeled data-to access LLMs in tourism. First, we iteratively refine and validate hierarchical ToT components through alignment with generic quality dimensions and expert feedback. Results demonstrate the effectiveness of our systematically optimized expert ToT with 4.99-14.15\% relative quality gains over baselines. Second, we apply LETToT's optimized expert ToT to evaluate models of varying scales (32B-671B parameters), revealing: (1) Scaling laws persist in specialized domains (DeepSeek-V3 leads), yet reasoning-enhanced smaller models (e.g., DeepSeek-R1-Distill-Llama-70B) close this gap; (2) For sub-72B models, explicit reasoning architectures outperform counterparts in accuracy and conciseness ($p<0.05$). Our work established a scalable, label-free paradigm for domain-specific LLM evaluation, offering a robust alternative to conventional annotated benchmarks.
>
---
#### [replaced 109] sudoLLM: On Multi-role Alignment of Language Models
- **分类: cs.CL; cs.CR; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.14607v2](http://arxiv.org/pdf/2505.14607v2)**

> **作者:** Soumadeep Saha; Akshay Chaturvedi; Joy Mahapatra; Utpal Garain
>
> **备注:** Accepted to EMNLP 2025 (findings)
>
> **摘要:** User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.
>
---

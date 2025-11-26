# 自然语言处理 cs.CL

- **最新发布 51 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Language-Independent Sentiment Labelling with Distant Supervision: A Case Study for English, Sepedi and Setswana
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言情感标注任务，针对低资源非洲语言（如塞佩迪语、茨瓦纳语）缺乏标注数据的问题，提出一种基于远监督的无语言依赖情感标注方法，利用表情符号和情感词自动标注推文，实验表明该方法可显著减少人工标注工作量。**

- **链接: [https://arxiv.org/pdf/2511.19818v1](https://arxiv.org/pdf/2511.19818v1)**

> **作者:** Koena Ronny Mabokela; Tim Schlippe; Mpho Raborife; Turgay Celik
>
> **备注:** Published in the The Fourth Workshop on Processing Emotions, Decisions and Opinions (EDO 2023) at 10th Language & Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics (LTC 2023), Poznań, Poland, 21-23 April 2023. ISBN: 978-83-232-4176-8
>
> **摘要:** Sentiment analysis is a helpful task to automatically analyse opinions and emotions on various topics in areas such as AI for Social Good, AI in Education or marketing. While many of the sentiment analysis systems are developed for English, many African languages are classified as low-resource languages due to the lack of digital language resources like text labelled with corresponding sentiment classes. One reason for that is that manually labelling text data is time-consuming and expensive. Consequently, automatic and rapid processes are needed to reduce the manual effort as much as possible making the labelling process as efficient as possible. In this paper, we present and analyze an automatic language-independent sentiment labelling method that leverages information from sentiment-bearing emojis and words. Our experiments are conducted with tweets in the languages English, Sepedi and Setswana from SAfriSenti, a multilingual sentiment corpus for South African languages. We show that our sentiment labelling approach is able to label the English tweets with an accuracy of 66%, the Sepedi tweets with 69%, and the Setswana tweets with 63%, so that on average only 34% of the automatically generated labels remain to be corrected.
>
---
#### [new 002] On Evaluating LLM Alignment by Evaluating LLMs as Judges
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）对齐人类偏好的评估问题，旨在解决传统评估依赖人工或强模型评判的局限。作者发现生成与评估能力具有一致性，提出无需直接评估输出的新基准AlignEval，通过评估模型作为评判者的表现来衡量对齐程度，效果优于现有自动评估方法。**

- **链接: [https://arxiv.org/pdf/2511.20604v1](https://arxiv.org/pdf/2511.20604v1)**

> **作者:** Yixin Liu; Pengfei Liu; Arman Cohan
>
> **备注:** NeurIPS 2025 Camera Ready
>
> **摘要:** Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs.
>
---
#### [new 003] REFLEX: Self-Refining Explainable Fact-Checking via Disentangling Truth into Style and Substance
- **分类: cs.CL**

- **简介: 该论文针对社交媒体虚假信息泛滥问题，提出REFLEX框架，解决现有大模型事实核查方法依赖外部知识导致延迟高、易幻觉的问题。通过角色扮演对话与激活向量解耦，将真相分离为风格与实质，实现自精炼的可解释事实核查，仅用465样本即达领先性能，并证明解释信号能双向提升推理能力。**

- **链接: [https://arxiv.org/pdf/2511.20233v1](https://arxiv.org/pdf/2511.20233v1)**

> **作者:** Chuyi Kong; Gao Wei; Jing Ma; Hongzhan Lin; Zhiyuan Fan
>
> **摘要:** The prevalence of misinformation on social media threatens public trust, demanding automated fact-checking systems that provide accurate verdicts with interpretable explanations. However, existing large language model-based (LLM-based) approaches often rely heavily on external knowledge sources, introducing substantial latency and even hallucinations that undermine reliability, interpretability, and responsiveness, which is crucial for real-time use. To address these challenges, we propose REason-guided Fact-checking with Latent EXplanations REFLEX paradigm, a plug-and-play, self-refining paradigm that leverages the internal knowledge in backbone model to improve both verdict accuracy and explanation quality. REFLEX reformulates fact-checking as a role-play dialogue and jointly trains verdict prediction and explanation generation. It adaptively extracts contrastive activation pairs between the backbone model and its fine-tuned variant to construct steering vectors that disentangle truth into style and substance naturally. These activation-level signals guide inference and suppress noisy explanations, enabling more faithful and efficient reasoning. Experiments on real-world datasets show that REFLEX outperforms previous methods that steer toward a single truth direction and underscores the challenge traditional approaches face when handling the subtle, human-unknown truth in fact-checking tasks. Remarkably, with only 465 self-refined training samples, RELFEX achieves state-of-the-art performance. Furthermore, models trained with explanatory objectives can effectively guide those without them, yielding up to a 7.57% improvement, highlighting that internal explanation signals play a dual role in both interpreting and enhancing factual reasoning.
>
---
#### [new 004] MTA: A Merge-then-Adapt Framework for Personalized Large Language Model
- **分类: cs.CL**

- **简介: 该论文针对个性化大语言模型（PLLM）存储成本高、稀疏数据下性能差的问题，提出MTA框架。通过构建共享的元LoRA库，动态融合生成用户专属参数，并引入轻量级LoRA模块实现少样本高效微调，有效提升可扩展性与个性化效果。**

- **链接: [https://arxiv.org/pdf/2511.20072v1](https://arxiv.org/pdf/2511.20072v1)**

> **作者:** Xiaopeng Li; Yuanjin Zheng; Wanyu Wang; wenlin zhang; Pengyue Jia; Yiqi Wang; Maolin Wang; Xuetao Wei; Xiangyu Zhao
>
> **摘要:** Personalized Large Language Models (PLLMs) aim to align model outputs with individual user preferences, a crucial capability for user-centric applications. However, the prevalent approach of fine-tuning a separate module for each user faces two major limitations: (1) storage costs scale linearly with the number of users, rendering the method unscalable; and (2) fine-tuning a static model from scratch often yields suboptimal performance for users with sparse data. To address these challenges, we propose MTA, a Merge-then-Adapt framework for PLLMs. MTA comprises three key stages. First, we construct a shared Meta-LoRA Bank by selecting anchor users and pre-training meta-personalization traits within meta-LoRA modules. Second, to ensure scalability and enable dynamic personalization combination beyond static models, we introduce an Adaptive LoRA Fusion stage. This stage retrieves and dynamically merges the most relevant anchor meta-LoRAs to synthesize a user-specific one, thereby eliminating the need for user-specific storage and supporting more flexible personalization. Third, we propose a LoRA Stacking for Few-Shot Personalization stage, which applies an additional ultra-low-rank, lightweight LoRA module on top of the merged LoRA. Fine-tuning this module enables effective personalization under few-shot settings. Extensive experiments on the LaMP benchmark demonstrate that our approach outperforms existing SOTA methods across multiple tasks.
>
---
#### [new 005] What does it mean to understand language?
- **分类: cs.CL**

- **简介: 该论文探讨语言理解的认知与神经机制，提出深层理解需将语言信息转移至非语言脑区以构建心理模型。针对“何为真正理解语言”的问题，通过整合认知神经科学证据，主张语言理解依赖跨脑区协作，为揭示语言理解的神经基础提供新路径。**

- **链接: [https://arxiv.org/pdf/2511.19757v1](https://arxiv.org/pdf/2511.19757v1)**

> **作者:** Colton Casto; Anna Ivanova; Evelina Fedorenko; Nancy Kanwisher
>
> **摘要:** Language understanding entails not just extracting the surface-level meaning of the linguistic input, but constructing rich mental models of the situation it describes. Here we propose that because processing within the brain's core language system is fundamentally limited, deeply understanding language requires exporting information from the language system to other brain regions that compute perceptual and motor representations, construct mental models, and store our world knowledge and autobiographical memories. We review the existing evidence for this hypothesis, and argue that recent progress in cognitive neuroscience provides both the conceptual foundation and the methods to directly test it, thus opening up a new strategy to reveal what it means, cognitively and neurally, to understand language.
>
---
#### [new 006] A Task-Oriented Evaluation Framework for Text Normalization in Modern NLP Pipelines
- **分类: cs.CL**

- **简介: 该论文针对文本归一化中词干提取方法评估不足的问题，提出任务导向的评估框架，综合考量词干有效性、下游任务性能与语义相似性。通过对比孟加拉语和英语词干提取器，揭示了高效率未必安全，强调意义保真度的重要性。**

- **链接: [https://arxiv.org/pdf/2511.20409v1](https://arxiv.org/pdf/2511.20409v1)**

> **作者:** Md Abdullah Al Kafi; Raka Moni; Sumit Kumar Banshal
>
> **摘要:** Text normalization is an essential preprocessing step in many natural language processing (NLP) tasks, and stemming is one such normalization technique that reduces words to their base or root form. However, evaluating stemming methods is challenging because current evaluation approaches are limited and do not capture the potential harm caused by excessive stemming; therefore, it is essential to develop new approaches to evaluate stemming methods. To address this issue, this study propose a novel, task-oriented approach to evaluate stemming methods, which considers three aspects: (1) the utility of stemming using Stemming Effectiveness Score (SES), (2) the impact of stemming on downstream tasks using Model Performance Delta (MPD), and (3) the semantic similarity between stemmed and original words using Average Normalized Levenshtein Distance (ANLD), thus providing a comprehensive evaluation framework. We apply our evaluation framework to compare two stemmers for Bangla (BNLTK) and English (Snowball), and our results reveal a significant issue, prompting us to analyze their performance in detail. While the Bangla stemmer achieves the highest SES (1.67) due to effective word reduction (CR = 1.90), SES alone is insufficient because our proposed safety measure, ANLD, reveals that this high SES is due to harmful over-stemming (ANLD = 0.26), which correlates with the observed decrease in downstream performance.In contrast, the English stemmer achieves a moderate SES (1.31) with a safe meaning distance (ANLD = 0.14), allowing its word reduction to contribute positively to downstream performance; therefore, it is a more reliable stemmer. Our study provides a valuable tool for distinguishing between potential efficiency gains (high SES) and meaning preservation (low ANLD).
>
---
#### [new 007] Online-PVLM: Advancing Personalized VLMs with Online Concept Learning
- **分类: cs.CL**

- **简介: 该论文针对个性化视觉语言模型（VLM）在测试时无法实时适应新概念的问题，提出Online-PVLM框架，通过超球面表示实现无需训练的在线概念学习。研究构建了大规模基准OP-Eval，验证了方法在高效、可扩展性上的优势。**

- **链接: [https://arxiv.org/pdf/2511.20056v1](https://arxiv.org/pdf/2511.20056v1)**

> **作者:** Huiyu Bai; Runze Wang; Zhuoyun Du; Yiyang Zhao; Fengji Zhang; Haoyu Chen; Xiaoyong Zhu; Bo Zheng; Xuejiao Zhao
>
> **备注:** Work in Progress
>
> **摘要:** Personalized Visual Language Models (VLMs) are gaining increasing attention for their formidable ability in user-specific concepts aligned interactions (e.g., identifying a user's bike). Existing methods typically require the learning of separate embeddings for each new concept, which fails to support real-time adaptation during testing. This limitation becomes particularly pronounced in large-scale scenarios, where efficient retrieval of concept embeddings is not achievable. To alleviate this gap, we propose Online-PVLM, a framework for online concept learning by leveraging hyperbolic representations. Our approach makes a train-free paradigm for concept embeddings generation at test time, making the use of personalized VLMs both scalable and efficient. In addition, we develop OP-Eval, a comprehensive and large-scale benchmark comprising 1,292 concepts and over 30K high-quality instances with diverse question types, designed to rigorously assess online concept learning in realistic scenarios. Extensive experiments demonstrate the state-of-the-art performance of our proposed framework. Our source code and dataset will be made available.
>
---
#### [new 008] From Words to Wisdom: Discourse Annotation and Baseline Models for Student Dialogue Understanding
- **分类: cs.CL**

- **简介: 该论文针对教育对话中知识建构与任务完成话语的自动识别问题，构建了首个相关标注数据集，并基于GPT-3.5和Llama-3.1建立基线模型。研究旨在提升教育对话分析的效率与规模，解决现有NLP技术在教育场景下适用性不足的问题。**

- **链接: [https://arxiv.org/pdf/2511.20547v1](https://arxiv.org/pdf/2511.20547v1)**

> **作者:** Farjana Sultana Mim; Shuchin Aeron; Eric Miller; Kristen Wendell
>
> **摘要:** Identifying discourse features in student conversations is quite important for educational researchers to recognize the curricular and pedagogical variables that cause students to engage in constructing knowledge rather than merely completing tasks. The manual analysis of student conversations to identify these discourse features is time-consuming and labor-intensive, which limits the scale and scope of studies. Leveraging natural language processing (NLP) techniques can facilitate the automatic detection of these discourse features, offering educational researchers scalable and data-driven insights. However, existing studies in NLP that focus on discourse in dialogue rarely address educational data. In this work, we address this gap by introducing an annotated educational dialogue dataset of student conversations featuring knowledge construction and task production discourse. We also establish baseline models for automatically predicting these discourse properties for each turn of talk within conversations, using pre-trained large language models GPT-3.5 and Llama-3.1. Experimental results indicate that these state-of-the-art models perform suboptimally on this task, indicating the potential for future research.
>
---
#### [new 009] $\text{R}^2\text{R}$: A Route-to-Rerank Post-Training Framework for Multi-Domain Decoder-Only Rerankers
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对检索增强生成中解码器仅重排序器在多领域应用时缺乏领域特异性的问题，提出R2R框架。通过动态专家路由与两阶段训练策略，结合实体抽象机制，避免表面特征过拟合，提升模型对金融、法律、医疗等领域的适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.19987v1](https://arxiv.org/pdf/2511.19987v1)**

> **作者:** Xinyu Wang; Hanwei Wu; Qingchen Hu; Zhenghan Tai; Jingrui Tian; Lei Ding; Jijun Chi; Hailin He; Tung Sum Thomas Kwok; Yufei Cui; Sicheng Lyu; Muzhi Li; Mingze Li; Xinyue Yu; Ling Zhou; Peng Lu
>
> **备注:** 13 pages, including 3 figures and 3 tables
>
> **摘要:** Decoder-only rerankers are central to Retrieval-Augmented Generation (RAG). However, generalist models miss domain-specific nuances in high-stakes fields like finance and law, and naive fine-tuning causes surface-form overfitting and catastrophic forgetting. To address this challenge, we introduce R2R, a domain-aware framework that combines dynamic expert routing with a two-stage training strategy, Entity Abstraction for Generalization (EAG). EAG introduces a counter-shortcut mechanism by masking the most predictive surface cues, forcing the reranker to learn domain-invariant relevance patterns rather than memorizing dataset-specific entities. To efficiently activate domain experts, R2R employs a lightweight Latent Semantic Router that probes internal representations from the frozen backbone decoder to select the optimal LoRA expert per query. Extensive experiments across different reranker backbones and diverse domains (legal, medical, and financial) demonstrate that R2R consistently surpasses generalist and single-domain fine-tuned baselines. Our results confirm that R2R is a model-agnostic and modular approach to domain specialization with strong cross-domain robustness.
>
---
#### [new 010] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios
- **分类: cs.CL**

- **简介: 该论文针对大批次场景下生成式模型推理效率低的问题，提出SpecFormer架构。通过融合单向与双向注意力机制，实现无损推测解码，突破传统方法对长前缀树的依赖，在降低计算开销的同时提升并行生成能力，显著提升大规模语言模型推理速度。**

- **链接: [https://arxiv.org/pdf/2511.20340v1](https://arxiv.org/pdf/2511.20340v1)**

> **作者:** Luohe Shi; Zuchao Li; Lefei Zhang; Baoyuan Qi; Guoming Liu; Hai Zhao
>
> **备注:** accepted by AAAI-2026
>
> **摘要:** Speculative decoding accelerates LLM inference by utilizing otherwise idle computational resources during memory-to-chip data transfer. Current speculative decoding methods typically assume a considerable amount of available computing power, then generate a complex and massive draft tree using a small autoregressive language model to improve overall prediction accuracy. However, methods like batching have been widely applied in mainstream model inference systems as a superior alternative to speculative decoding, as they compress the available idle computing power. Therefore, performing speculative decoding with low verification resources and low scheduling costs has become an important research problem. We believe that more capable models that allow for parallel generation on draft sequences are what we truly need. Recognizing the fundamental nature of draft models to only generate sequences of limited length, we propose SpecFormer, a novel architecture that integrates unidirectional and bidirectional attention mechanisms. SpecFormer combines the autoregressive model's ability to extract information from the entire input sequence with the parallel generation benefits of non-autoregressive models. This design eliminates the reliance on large prefix trees and achieves consistent acceleration, even in large-batch scenarios. Through lossless speculative decoding experiments across models of various scales, we demonstrate that SpecFormer sets a new standard for scaling LLM inference with lower training demands and reduced computational costs.
>
---
#### [new 011] Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究语音识别中的误发音检测与诊断任务，旨在无需模型训练即可准确识别和分析发音错误。提出一种基于检索的无训练框架，利用预训练ASR模型实现高效、精准的误发音检测，避免了传统方法对音素建模或额外训练的依赖，在L2-ARCTIC数据集上取得69.60%的F1分数。**

- **链接: [https://arxiv.org/pdf/2511.20107v1](https://arxiv.org/pdf/2511.20107v1)**

> **作者:** Huu Tuong Tu; Ha Viet Khanh; Tran Tien Dat; Vu Huan; Thien Van Luong; Nguyen Tien Cuong; Nguyen Thi Thu Trang
>
> **摘要:** Mispronunciation Detection and Diagnosis (MDD) is crucial for language learning and speech therapy. Unlike conventional methods that require scoring models or training phoneme-level models, we propose a novel training-free framework that leverages retrieval techniques with a pretrained Automatic Speech Recognition model. Our method avoids phoneme-specific modeling or additional task-specific training, while still achieving accurate detection and diagnosis of pronunciation errors. Experiments on the L2-ARCTIC dataset show that our method achieves a superior F1 score of 69.60% while avoiding the complexity of model training.
>
---
#### [new 012] AppSelectBench: Application-Level Tool Selection Benchmark
- **分类: cs.CL**

- **简介: 该论文提出AppSelectBench，一个用于评估计算机使用代理（CUAs）应用级工具选择能力的基准。针对现有基准仅关注细粒度API选择、忽视跨应用推理的问题，构建了涵盖百种应用、十万+真实用户任务的评测体系，覆盖多种设置，揭示大模型在应用选择上的系统性短板，推动智能代理在应用层面的推理研究。**

- **链接: [https://arxiv.org/pdf/2511.19957v1](https://arxiv.org/pdf/2511.19957v1)**

> **作者:** Tianyi Chen; Michael Solodko; Sen Wang; Jongwoo Ko; Junheng Hao; Colby Banbury; Sara Abdali; Saeed Amizadeh; Qing Xiao; Yinheng Li; Tianyu Ding; Kamran Ghasedi Dizaji; Suzhen Zheng; Hao Fan; Justin Wagle; Pashmina Cameron; Kazuhito Koishida
>
> **摘要:** Computer Using Agents (CUAs) are increasingly equipped with external tools, enabling them to perform complex and realistic tasks. For CUAs to operate effectively, application selection, which refers to deciding which application to use before invoking fine-grained tools such as APIs, is a fundamental capability. It determines whether the agent initializes the correct environment, avoids orchestration confusion, and efficiently focuses on relevant context. However, existing benchmarks primarily assess fine-grained API selection, offering limited insight into whether models can reason across and choose between different applications. To fill this gap, we introduce AppSelectBench, a comprehensive benchmark for evaluating application selection in CUAs. AppSelectBench contains a novel user task generation pipeline that produces realistic, diverse, and semantically grounded user intents at scale, together with unified evaluation protocols covering random, heuristic, zero-shot, few-shot, and retrieval-augmented-settings. AppSelectBench covers one hundred widely used desktop applications and includes more than one hundred thousand realistic, diverse, and semantically grounded user tasks. Extensive experiments across both closed-source and open-source large language models reveal systematic strengths and weaknesses in inter-application reasoning, showing that even the most capable models still struggle to make consistent application choices. Together, these results establish AppSelectBench as a foundation for studying and advancing application level reasoning, an essential yet underexplored capability of intelligent CUAs. The source is available at https://github.com/microsoft/appselectbench.
>
---
#### [new 013] Generation, Evaluation, and Explanation of Novelists' Styles with Single-Token Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于风格建模任务，旨在解决无配对数据下生成与评估19世纪小说家写作风格的问题。通过单标记提示微调大模型生成仿写文本，并利用基于Transformer的检测器及可解释AI方法评估风格相似性，实现自动化、可靠的风格评价。**

- **链接: [https://arxiv.org/pdf/2511.20459v1](https://arxiv.org/pdf/2511.20459v1)**

> **作者:** Mosab Rezaei; Mina Rajaei Moghadam; Abdul Rahman Shaikh; Hamed Alhoori; Reva Freedman
>
> **摘要:** Recent advances in large language models have created new opportunities for stylometry, the study of writing styles and authorship. Two challenges, however, remain central: training generative models when no paired data exist, and evaluating stylistic text without relying only on human judgment. In this work, we present a framework for both generating and evaluating sentences in the style of 19th-century novelists. Large language models are fine-tuned with minimal, single-token prompts to produce text in the voices of authors such as Dickens, Austen, Twain, Alcott, and Melville. To assess these generative models, we employ a transformer-based detector trained on authentic sentences, using it both as a classifier and as a tool for stylistic explanation. We complement this with syntactic comparisons and explainable AI methods, including attention-based and gradient-based analyses, to identify the linguistic cues that drive stylistic imitation. Our findings show that the generated text reflects the authors' distinctive patterns and that AI-based evaluation offers a reliable alternative to human assessment. All artifacts of this work are published online.
>
---
#### [new 014] Adversarial Confusion Attack: Disrupting Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出“对抗混淆攻击”，针对多模态大语言模型（MLLMs）的系统性干扰。通过最大化下一词熵，利用少量开源模型生成可迁移的对抗图像，使模型输出混乱或错误，破坏其可靠性。攻击在白盒下有效，且能跨模型泛化，适用于嵌入网页等实际场景。**

- **链接: [https://arxiv.org/pdf/2511.20494v1](https://arxiv.org/pdf/2511.20494v1)**

> **作者:** Jakub Hoscilowicz; Artur Janicki
>
> **摘要:** We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.
>
---
#### [new 015] EM2LDL: A Multilingual Speech Corpus for Mixed Emotion Recognition through Label Distribution Learning
- **分类: cs.CL**

- **简介: 该论文提出EM2LDL，一个用于多语言混合情绪识别的语料库，解决传统数据集语言单一、无法建模混合情绪的问题。通过整合英、中、粤三语的自然表达数据，标注32类细粒度情绪分布，支持跨性别、年龄、人格的自监督学习实验，推动情感计算在心理健康与跨文化沟通中的应用。**

- **链接: [https://arxiv.org/pdf/2511.20106v1](https://arxiv.org/pdf/2511.20106v1)**

> **作者:** Xingfeng Li; Xiaohan Shi; Junjie Li; Yongwei Li; Masashi Unoki; Tomoki Toda; Masato Akagi
>
> **备注:** Submitted to IEEE Transactions on Affective computing
>
> **摘要:** This study introduces EM2LDL, a novel multilingual speech corpus designed to advance mixed emotion recognition through label distribution learning. Addressing the limitations of predominantly monolingual and single-label emotion corpora \textcolor{black}{that restrict linguistic diversity, are unable to model mixed emotions, and lack ecological validity}, EM2LDL comprises expressive utterances in English, Mandarin, and Cantonese, capturing the intra-utterance code-switching prevalent in multilingual regions like Hong Kong and Macao. The corpus integrates spontaneous emotional expressions from online platforms, annotated with fine-grained emotion distributions across 32 categories. Experimental baselines using self-supervised learning models demonstrate robust performance in speaker-independent gender-, age-, and personality-based evaluations, with HuBERT-large-EN achieving optimal results. By incorporating linguistic diversity and ecological validity, EM2LDL enables the exploration of complex emotional dynamics in multilingual settings. This work provides a versatile testbed for developing adaptive, empathetic systems for applications in affective computing, including mental health monitoring and cross-cultural communication. The dataset, annotations, and baseline codes are publicly available at https://github.com/xingfengli/EM2LDL.
>
---
#### [new 016] SEDA: A Self-Adapted Entity-Centric Data Augmentation for Boosting Gird-based Discontinuous NER Models
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对网格法在识别跨句不连续命名实体时的分割错误与遗漏问题，提出自适应实体中心数据增强方法SEDA。通过引入图像增强技术改进网格模型，提升对不连续实体的识别能力，在多个数据集上显著提高F1分数，尤其在不连续实体上表现更优。**

- **链接: [https://arxiv.org/pdf/2511.20143v1](https://arxiv.org/pdf/2511.20143v1)**

> **作者:** Wen-Fang Su; Hsiao-Wei Chou; Wen-Yang Lin
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Named Entity Recognition (NER) is a critical task in natural language processing, yet it remains particularly challenging for discontinuous entities. The primary difficulty lies in text segmentation, as traditional methods often missegment or entirely miss cross-sentence discontinuous entities, significantly affecting recognition accuracy. Therefore, we aim to address the segmentation and omission issues associated with such entities. Recent studies have shown that grid-tagging methods are effective for information extraction due to their flexible tagging schemes and robust architectures. Building on this, we integrate image data augmentation techniques, such as cropping, scaling, and padding, into grid-based models to enhance their ability to recognize discontinuous entities and handle segmentation challenges. Experimental results demonstrate that traditional segmentation methods often fail to capture cross-sentence discontinuous entities, leading to decreased performance. In contrast, our augmented grid models achieve notable improvements. Evaluations on the CADEC, ShARe13, and ShARe14 datasets show F1 score gains of 1-2.5% overall and 3.7-8.4% for discontinuous entities, confirming the effectiveness of our approach.
>
---
#### [new 017] More Bias, Less Bias: BiasPrompting for Enhanced Multiple-Choice Question Answering
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在多项选择题回答中的推理不足问题，提出BiasPrompting框架。通过生成各选项的推理并进行对比评估，增强模型对所有选项的全面分析能力，显著提升复杂题目上的答题准确率。**

- **链接: [https://arxiv.org/pdf/2511.20086v1](https://arxiv.org/pdf/2511.20086v1)**

> **作者:** Duc Anh Vu; Thong Nguyen; Cong-Duy Nguyen; Viet Anh Nguyen; Anh Tuan Luu
>
> **备注:** Accepted at the 41st ACM/SIGAPP Symposium On Applied Computing (SAC 2026), Main Conference
>
> **摘要:** With the advancement of large language models (LLMs), their performance on multiple-choice question (MCQ) tasks has improved significantly. However, existing approaches face key limitations: answer choices are typically presented to LLMs without contextual grounding or explanation. This absence of context can lead to incomplete exploration of all possible answers, ultimately degrading the models' reasoning capabilities. To address these challenges, we introduce BiasPrompting, a novel inference framework that guides LLMs to generate and critically evaluate reasoning across all plausible answer options before reaching a final prediction. It consists of two components: first, a reasoning generation stage, where the model is prompted to produce supportive reasonings for each answer option, and then, a reasoning-guided agreement stage, where the generated reasonings are synthesized to select the most plausible answer. Through comprehensive evaluations, BiasPrompting demonstrates significant improvements in five widely used multiple-choice question answering benchmarks. Our experiments showcase that BiasPrompting enhances the reasoning capabilities of LLMs and provides a strong foundation for tackling complex and challenging questions, particularly in settings where existing methods underperform.
>
---
#### [new 018] Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition
- **分类: cs.CL**

- **简介: 该论文针对低资源语言语音识别性能差的问题，提出一种基于潜在空间混合的数据增强方法（Latent Mixup），通过合成多样化语音提升模型泛化能力，有效缩小了高/低资源语言间的识别性能差距，推动了语音技术的公平性。**

- **链接: [https://arxiv.org/pdf/2511.20534v1](https://arxiv.org/pdf/2511.20534v1)**

> **作者:** Wesley Bian; Xiaofeng Lin; Guang Cheng
>
> **备注:** Accepted at ICML 2025 Workshop on Machine Learning for Audio
>
> **摘要:** Modern machine learning models for audio tasks often exhibit superior performance on English and other well-resourced languages, primarily due to the abundance of available training data. This disparity leads to an unfair performance gap for low-resource languages, where data collection is both challenging and costly. In this work, we introduce a novel data augmentation technique for speech corpora designed to mitigate this gap. Through comprehensive experiments, we demonstrate that our method significantly improves the performance of automatic speech recognition systems on low-resource languages. Furthermore, we show that our approach outperforms existing augmentation strategies, offering a practical solution for enhancing speech technology in underrepresented linguistic communities.
>
---
#### [new 019] Can LLMs Faithfully Explain Themselves in Low-Resource Languages? A Case Study on Emotion Detection in Persian
- **分类: cs.CL**

- **简介: 该论文研究低资源语言中大模型自解释的可信度问题，聚焦波斯语情感分类任务。通过对比模型与人类标注者识别的关键词，评估解释忠实性，发现模型解释虽具一致性却常偏离人类判断，揭示现有解释方法在多语言、低资源场景下的局限性。**

- **链接: [https://arxiv.org/pdf/2511.19719v1](https://arxiv.org/pdf/2511.19719v1)**

> **作者:** Mobina Mehrazar; Mohammad Amin Yousefi; Parisa Abolfath Beygi; Behnam Bahrak
>
> **摘要:** Large language models (LLMs) are increasingly used to generate self-explanations alongside their predictions, a practice that raises concerns about the faithfulness of these explanations, especially in low-resource languages. This study evaluates the faithfulness of LLM-generated explanations in the context of emotion classification in Persian, a low-resource language, by comparing the influential words identified by the model against those identified by human annotators. We assess faithfulness using confidence scores derived from token-level log-probabilities. Two prompting strategies, differing in the order of explanation and prediction (Predict-then-Explain and Explain-then-Predict), are tested for their impact on explanation faithfulness. Our results reveal that while LLMs achieve strong classification performance, their generated explanations often diverge from faithful reasoning, showing greater agreement with each other than with human judgments. These results highlight the limitations of current explanation methods and metrics, emphasizing the need for more robust approaches to ensure LLM reliability in multilingual and low-resource contexts.
>
---
#### [new 020] The Text Aphasia Battery (TAB): A Clinically-Grounded Benchmark for Aphasia-Like Deficits in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出文本失语症测评工具（TAB），针对大语言模型的失语症样语言缺陷评估问题，设计四类文本任务，构建可扩展的临床基准。通过自动化评估验证其可靠性，实现对模型语言能力缺陷的标准化、规模化分析。**

- **链接: [https://arxiv.org/pdf/2511.20507v1](https://arxiv.org/pdf/2511.20507v1)**

> **作者:** Nathan Roll; Jill Kries; Flora Jin; Catherine Wang; Ann Marie Finley; Meghan Sumner; Cory Shain; Laura Gwilliams
>
> **摘要:** Large language models (LLMs) have emerged as a candidate "model organism" for human language, offering an unprecedented opportunity to study the computational basis of linguistic disorders like aphasia. However, traditional clinical assessments are ill-suited for LLMs, as they presuppose human-like pragmatic pressures and probe cognitive processes not inherent to artificial architectures. We introduce the Text Aphasia Battery (TAB), a text-only benchmark adapted from the Quick Aphasia Battery (QAB) to assess aphasic-like deficits in LLMs. The TAB comprises four subtests: Connected Text, Word Comprehension, Sentence Comprehension, and Repetition. This paper details the TAB's design, subtests, and scoring criteria. To facilitate large-scale use, we validate an automated evaluation protocol using Gemini 2.5 Flash, which achieves reliability comparable to expert human raters (prevalence-weighted Cohen's kappa = 0.255 for model--consensus agreement vs. 0.286 for human--human agreement). We release TAB as a clinically-grounded, scalable framework for analyzing language deficits in artificial systems.
>
---
#### [new 021] Profile-LLM: Dynamic Profile Optimization for Realistic Personality Expression in LLMs
- **分类: cs.CL**

- **简介: 该论文聚焦于个性化大模型的人格表达优化任务，旨在解决现有方法中角色提示未能最大化人格表现的问题。提出PersonaPulse框架，通过迭代优化提示并结合情境响应评估，提升人格表达的真实性与上下文契合度。实验表明其优于基于心理学描述的提示，并发现模型规模与人格控制能力相关。**

- **链接: [https://arxiv.org/pdf/2511.19852v1](https://arxiv.org/pdf/2511.19852v1)**

> **作者:** Shi-Wei Dai; Yan-Wei Shie; Tsung-Huan Yang; Lun-Wei Ku; Yung-Hui Li
>
> **摘要:** Personalized Large Language Models (LLMs) have been shown to be an effective way to create more engaging and enjoyable user-AI interactions. While previous studies have explored using prompts to elicit specific personality traits in LLMs, they have not optimized these prompts to maximize personality expression. To address this limitation, we propose PersonaPulse: Dynamic Profile Optimization for Realistic Personality Expression in LLMs, a framework that leverages LLMs' inherent knowledge of personality traits to iteratively enhance role-play prompts while integrating a situational response benchmark as a scoring tool, ensuring a more realistic and contextually grounded evaluation to guide the optimization process. Quantitative evaluations demonstrate that the prompts generated by PersonaPulse outperform those of prior work, which were designed based on personality descriptions from psychological studies. Additionally, we explore the relationship between model size and personality modeling through extensive experiments. Finally, we find that, for certain personality traits, the extent of personality evocation can be partially controlled by pausing the optimization process. These findings underscore the importance of prompt optimization in shaping personality expression within LLMs, offering valuable insights for future research on adaptive AI interactions.
>
---
#### [new 022] Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对知识图谱上的多跳问答任务，解决推理路径爆炸与答案缺乏可验证性问题。提出两种高效方法：基于LLM规划的符号搜索与嵌入引导的神经搜索，实现高准确率与百倍加速，通过知识蒸馏使小模型达到大模型性能，证明结构化推理优于直接生成。**

- **链接: [https://arxiv.org/pdf/2511.19648v1](https://arxiv.org/pdf/2511.19648v1)**

> **作者:** Manil Shrestha; Edward Kim
>
> **摘要:** Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations.
>
---
#### [new 023] BengaliFig: A Low-Resource Challenge for Figurative and Culturally Grounded Reasoning in Bengali
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BengaliFig，一个面向孟加拉语的低资源隐喻与文化语境推理挑战集。针对大语言模型在低资源语言中隐喻和文化推理能力不足的问题，构建了435个源自孟加拉口头与文学传统的谜题，涵盖五维标注，并通过AI辅助生成多选题。实验揭示主流LLMs在该任务上的系统性缺陷，为文化敏感型NLP评估提供新基准。**

- **链接: [https://arxiv.org/pdf/2511.20399v1](https://arxiv.org/pdf/2511.20399v1)**

> **作者:** Abdullah Al Sefat
>
> **摘要:** Large language models excel on broad multilingual benchmarks but remain to be evaluated extensively in figurative and culturally grounded reasoning, especially in low-resource contexts. We present BengaliFig, a compact yet richly annotated challenge set that targets this gap in Bengali, a widely spoken low-resourced language. The dataset contains 435 unique riddles drawn from Bengali oral and literary traditions. Each item is annotated along five orthogonal dimensions capturing reasoning type, trap type, cultural depth, answer category, and difficulty, and is automatically converted to multiple-choice format through a constraint-aware, AI-assisted pipeline. We evaluate eight frontier LLMs from major providers under zero-shot and few-shot chain-of-thought prompting, revealing consistent weaknesses in metaphorical and culturally specific reasoning. BengaliFig thus contributes both a diagnostic probe for evaluating LLM robustness in low-resource cultural contexts and a step toward inclusive and heritage-aware NLP evaluation.
>
---
#### [new 024] A Systematic Analysis of Large Language Models with RAG-enabled Dynamic Prompting for Medical Error Detection and Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医学文档中的错误检测与修正任务，旨在提升大语言模型（LLM）在临床文本中的准确性。通过对比零样本、静态提示和检索增强动态提示（RDP）三种策略，发现RDP显著提升检测召回率、降低误报率，并生成更准确的修正结果。**

- **链接: [https://arxiv.org/pdf/2511.19858v1](https://arxiv.org/pdf/2511.19858v1)**

> **作者:** Farzad Ahmed; Joniel Augustine Jerome; Meliha Yetisgen; Özlem Uzuner
>
> **摘要:** Objective: Clinical documentation contains factual, diagnostic, and management errors that can compromise patient safety. Large language models (LLMs) may help detect and correct such errors, but their behavior under different prompting strategies remains unclear. We evaluate zero-shot prompting, static prompting with random exemplars (SPR), and retrieval-augmented dynamic prompting (RDP) for three subtasks of medical error processing: error flag detection, error sentence detection, and error correction. Methods: Using the MEDEC dataset, we evaluated nine instruction-tuned LLMs (GPT, Claude, Gemini, and OpenAI o-series models). We measured performance using accuracy, recall, false-positive rate (FPR), and an aggregate score of ROUGE-1, BLEURT, and BERTScore for error correction. We also analyzed example outputs to identify failure modes and differences between LLM and clinician reasoning. Results: Zero-shot prompting showed low recall in both detection tasks, often missing abbreviation-heavy or atypical errors. SPR improved recall but increased FPR. Across all nine LLMs, RDP reduced FPR by about 15 percent, improved recall by 5 to 10 percent in error sentence detection, and generated more contextually accurate corrections. Conclusion: Across diverse LLMs, RDP outperforms zero-shot and SPR prompting. Using retrieved exemplars improves detection accuracy, reduces false positives, and enhances the reliability of medical error correction.
>
---
#### [new 025] SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space
- **分类: cs.CL**

- **简介: 该论文针对大语言模型长序列处理中注意力机制的二次复杂度问题，提出SSA框架。通过在特征空间对齐全注意力与稀疏注意力输出，解决稀疏训练中梯度缺失导致的性能下降和稀疏度不足问题，实现高效且可灵活调节的稀疏注意力，显著提升长序列推理性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20102v1](https://arxiv.org/pdf/2511.20102v1)**

> **作者:** Zhenyi Shen; Junru Lu; Lin Gui; Jiazheng Li; Yulan He; Di Yin; Xing Sun
>
> **备注:** 28 pages
>
> **摘要:** The quadratic complexity of full attention limits efficient long-context processing in large language models (LLMs). Sparse attention mitigates this cost by restricting each query to attend to a subset of previous tokens; however, training-free approaches often lead to severe performance degradation. Native sparse-attention methods (e.g., NSA, MoBA) alleviate this issue, yet exhibit a critical paradox: they produce lower attention sparsity than full-attention models, despite aiming to approximate full attention, which may constrain their effectiveness. We attribute this paradox to gradient update deficiency: low-ranked key-value pairs excluded during sparse training receive neither forward contribution nor backward gradients, and thus never learn proper suppression. To overcome this limitation, we propose SSA (Sparse Sparse Attention), a unified training framework that considers both sparse and full attention and enforces bidirectional alignment at every layer. This design preserves gradient flow to all tokens while explicitly encouraging sparse-attention outputs to align with their full-attention counterparts, thereby promoting stronger sparsity. As a result, SSA achieves state-of-the-art performance under both sparse and full attention inference across multiple commonsense benchmarks. Furthermore, SSA enables models to adapt smoothly to varying sparsity budgets; performance improves consistently as more tokens are allowed to attend, supporting flexible compute-performance trade-offs at inference time. Finally, we show that native sparse-attention training surprisingly improves long-context extrapolation by mitigating the over-allocation of attention values in sink areas, with SSA demonstrating the strongest extrapolation capability.
>
---
#### [new 026] KyrgyzBERT: A Compact, Efficient Language Model for Kyrgyz NLP
- **分类: cs.CL**

- **简介: 该论文针对低资源语言Kyrgyz缺乏基础NLP工具的问题，提出首个公开的单语BERT模型KyrgyzBERT。模型采用适配其形态结构的自定义分词器，参数量35.9M。研究构建了kyrgyz-sst2情感分析基准，并验证模型在该任务上表现优异，性能媲美五倍大的mBERT，推动了Kyrgyz NLP发展。**

- **链接: [https://arxiv.org/pdf/2511.20182v1](https://arxiv.org/pdf/2511.20182v1)**

> **作者:** Adilet Metinov; Gulida M. Kudakeeva; Gulnara D. Kabaeva
>
> **备注:** 3 pages, 1 figure, 2 tables. Preprint
>
> **摘要:** Kyrgyz remains a low-resource language with limited foundational NLP tools. To address this gap, we introduce KyrgyzBERT, the first publicly available monolingual BERT-based language model for Kyrgyz. The model has 35.9M parameters and uses a custom tokenizer designed for the language's morphological structure. To evaluate performance, we create kyrgyz-sst2, a sentiment analysis benchmark built by translating the Stanford Sentiment Treebank and manually annotating the full test set. KyrgyzBERT fine-tuned on this dataset achieves an F1-score of 0.8280, competitive with a fine-tuned mBERT model five times larger. All models, data, and code are released to support future research in Kyrgyz NLP.
>
---
#### [new 027] Gender Bias in Emotion Recognition by Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型在情感认知任务中的性别偏见问题，旨在评估模型在理解他人情绪时是否存在性别偏差。通过分析模型对人物描述的情感判断，提出并验证了基于训练的去偏策略，发现仅靠提示工程无法有效降低偏见。**

- **链接: [https://arxiv.org/pdf/2511.19785v1](https://arxiv.org/pdf/2511.19785v1)**

> **作者:** Maureen Herbert; Katie Sun; Angelica Lim; Yasaman Etesam
>
> **备注:** Accepted at AAAI 2026 Workshop (WS37)
>
> **摘要:** The rapid advancement of large language models (LLMs) and their growing integration into daily life underscore the importance of evaluating and ensuring their fairness. In this work, we examine fairness within the domain of emotional theory of mind, investigating whether LLMs exhibit gender biases when presented with a description of a person and their environment and asked, "How does this person feel?". Furthermore, we propose and evaluate several debiasing strategies, demonstrating that achieving meaningful reductions in bias requires training based interventions rather than relying solely on inference-time prompt-based approaches such as prompt engineering.
>
---
#### [new 028] Breaking Bad: Norms for Valence, Arousal, and Dominance for over 10k English Multiword Expressions
- **分类: cs.CL**

- **简介: 该论文提出NRC VAD Lexicon v2，针对10,000个英语多词表达（MWEs）及其组成词进行情感三维度（效价、唤醒度、支配度）的人工标注，扩展了原有词典覆盖范围与时效性。旨在解决情感词汇资源不足问题，支持自然语言处理、心理学等领域的研究。**

- **链接: [https://arxiv.org/pdf/2511.19816v1](https://arxiv.org/pdf/2511.19816v1)**

> **作者:** Saif M. Mohammad
>
> **摘要:** Factor analysis studies have shown that the primary dimensions of word meaning are Valence (V), Arousal (A), and Dominance (D). Existing lexicons such as the NRC VAD Lexicon, published in 2018, include VAD association ratings for words. Here, we present a complement to it, which has human ratings of valence, arousal, and dominance for 10k English Multiword Expressions (MWEs) and their constituent words. We also increase the coverage of unigrams, especially words that have become more common since 2018. In all, the new NRC VAD Lexicon v2 now has entries for 10k MWEs and 25k words, in addition to the entries in v1. We show that the associations are highly reliable. We use the lexicon to examine emotional characteristics of MWEs, including: 1. The degree to which MWEs (idioms, noun compounds, and verb particle constructions) exhibit strong emotionality; 2. The degree of emotional compositionality in MWEs. The lexicon enables a wide variety of research in NLP, Psychology, Public Health, Digital Humanities, and Social Sciences. The NRC VAD Lexicon v2 is freely available through the project webpage: http://saifmohammad.com/WebPages/nrc-vad.html
>
---
#### [new 029] Latent Collaboration in Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多智能体系统中的协作问题，旨在解决传统文本通信效率低、信息损失的问题。提出无需训练的LatentMAS框架，实现智能体在连续隐空间的直接协作，通过共享隐状态记忆提升表达能力与信息保真度，显著提升推理效率与准确率。**

- **链接: [https://arxiv.org/pdf/2511.20639v1](https://arxiv.org/pdf/2511.20639v1)**

> **作者:** Jiaru Zou; Xiyuan Yang; Ruizhong Qiu; Gaotang Li; Katherine Tieu; Pan Lu; Ke Shen; Hanghang Tong; Yejin Choi; Jingrui He; James Zou; Mengdi Wang; Ling Yang
>
> **备注:** Project: https://github.com/Gen-Verse/LatentMAS
>
> **摘要:** Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at https://github.com/Gen-Verse/LatentMAS.
>
---
#### [new 030] Comparative Analysis of LoRA-Adapted Embedding Models for Clinical Cardiology Text Representation
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对临床心脏病学文本表示任务，比较了10种经LoRA微调的Transformer模型。基于10万+医学文本对，发现编码器架构如BioLinkBERT在性能与效率上更优，挑战了大模型必优的假设，为临床NLP系统设计提供实证指导。**

- **链接: [https://arxiv.org/pdf/2511.19739v1](https://arxiv.org/pdf/2511.19739v1)**

> **作者:** Richard J. Young; Alice M. Matthews
>
> **备注:** 25 pages, 13 figures, 5 tables
>
> **摘要:** Domain-specific text embeddings are critical for clinical natural language processing, yet systematic comparisons across model architectures remain limited. This study evaluates ten transformer-based embedding models adapted for cardiology through Low-Rank Adaptation (LoRA) fine-tuning on 106,535 cardiology text pairs derived from authoritative medical textbooks. Results demonstrate that encoder-only architectures, particularly BioLinkBERT, achieve superior domain-specific performance (separation score: 0.510) compared to larger decoder-based models, while requiring significantly fewer computational resources. The findings challenge the assumption that larger language models necessarily produce better domain-specific embeddings and provide practical guidance for clinical NLP system development. All models, training code, and evaluation datasets are publicly available to support reproducible research in medical informatics.
>
---
#### [new 031] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的类比推理能力，旨在探究其是否能编码并应用高层次关系概念。通过分析比例与故事类比，发现LLM虽能捕捉关系信息，但在新情境中应用时仍受限；成功推理依赖结构对齐，失败则源于关系信息缺失或错位。**

- **链接: [https://arxiv.org/pdf/2511.20344v1](https://arxiv.org/pdf/2511.20344v1)**

> **作者:** Taewhoo Lee; Minju Song; Chanwoong Yoon; Jungwoo Park; Jaewoo Kang
>
> **备注:** AAAI 2026
>
> **摘要:** Analogical reasoning is at the core of human cognition, serving as an important foundation for a variety of intellectual activities. While prior work has shown that LLMs can represent task patterns and surface-level concepts, it remains unclear whether these models can encode high-level relational concepts and apply them to novel situations through structured comparisons. In this work, we explore this fundamental aspect using proportional and story analogies, and identify three key findings. First, LLMs effectively encode the underlying relationships between analogous entities; both attributive and relational information propagate through mid-upper layers in correct cases, whereas reasoning failures reflect missing relational information within these layers. Second, unlike humans, LLMs often struggle not only when relational information is missing, but also when attempting to apply it to new entities. In such cases, strategically patching hidden representations at critical token positions can facilitate information transfer to a certain extent. Lastly, successful analogical reasoning in LLMs is marked by strong structural alignment between analogous situations, whereas failures often reflect degraded or misplaced alignment. Overall, our findings reveal that LLMs exhibit emerging but limited capabilities in encoding and applying high-level relational concepts, highlighting both parallels and gaps with human cognition.
>
---
#### [new 032] Directional Optimization Asymmetry in Transformers: A Synthetic Stress Test
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer在序列方向性上的优化不对称性。针对“方向性失败源于语言统计还是架构本身”这一问题，构建熵可控的合成基准，通过正向与反向任务对比，发现即使在无语义、无统计偏倚下，Transformer仍存在显著方向性优化差距，揭示其因果训练机制内生的方向摩擦。**

- **链接: [https://arxiv.org/pdf/2511.19997v1](https://arxiv.org/pdf/2511.19997v1)**

> **作者:** Mihir Sahasrabudhe
>
> **备注:** 19 pages, 4 figures. Code available at https://github.com/mihirs-0/synass
>
> **摘要:** Transformers are theoretically reversal-invariant: their function class does not prefer left-to-right over right-to-left mappings. Yet empirical studies on natural language repeatedly report a "reversal curse," and recent work on temporal asymmetry in LLMs suggests that real-world corpora carry their own arrow of time. This leaves an unresolved question: do directional failures stem from linguistic statistics, or from the architecture itself? We cut through this ambiguity with a fully synthetic, entropy-controlled benchmark designed as a clean-room stress test for directional learning. Using random string mappings with tunable branching factor K, we construct forward tasks with zero conditional entropy and inverse tasks with analytically determined entropy floors. Excess loss above these floors reveals that even scratch-trained GPT-2 models exhibit a strong, reproducible directional optimization gap (e.g., 1.16 nats at K=5), far larger than that of an MLP trained on the same data. Pre-trained initializations shift optimization behavior but do not eliminate this gap, while LoRA encounters a sharp capacity wall on high-entropy inverse mappings. Together, these results isolate a minimal, semantics-free signature of directional friction intrinsic to causal Transformer training-one that persists even when linguistic priors, token frequencies, and corpus-level temporal asymmetries are removed. Our benchmark provides a controlled instrument for dissecting directional biases in modern sequence models and motivates deeper mechanistic study of why inversion remains fundamentally harder for Transformers.
>
---
#### [new 033] A Machine Learning Approach for Detection of Mental Health Conditions and Cyberbullying from Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文针对社交平台上心理健康问题与网络欺凌的检测任务，提出一种统一的多分类框架。通过构建平衡训练集与真实分布测试集，对比多种模型，发现领域适配的MentalBERT表现最优。研究强调系统作为人工辅助工具，并开发可解释性框架与原型仪表板，推动其在实际场景中的应用。**

- **链接: [https://arxiv.org/pdf/2511.20001v1](https://arxiv.org/pdf/2511.20001v1)**

> **作者:** Edward Ajayi; Martha Kachweka; Mawuli Deku; Emily Aiken
>
> **备注:** Accepted for Oral Presentation at the AAAI-26 Bridge Program on AI for Medicine and Healthcare (AIMedHealth). To appear in Proceedings of Machine Learning Research (PMLR)
>
> **摘要:** Mental health challenges and cyberbullying are increasingly prevalent in digital spaces, necessitating scalable and interpretable detection systems. This paper introduces a unified multiclass classification framework for detecting ten distinct mental health and cyberbullying categories from social media data. We curate datasets from Twitter and Reddit, implementing a rigorous "split-then-balance" pipeline to train on balanced data while evaluating on a realistic, held-out imbalanced test set. We conducted a comprehensive evaluation comparing traditional lexical models, hybrid approaches, and several end-to-end fine-tuned transformers. Our results demonstrate that end-to-end fine-tuning is critical for performance, with the domain-adapted MentalBERT emerging as the top model, achieving an accuracy of 0.92 and a Macro F1 score of 0.76, surpassing both its generic counterpart and a zero-shot LLM baseline. Grounded in a comprehensive ethical analysis, we frame the system as a human-in-the-loop screening aid, not a diagnostic tool. To support this, we introduce a hybrid SHAPLLM explainability framework and present a prototype dashboard ("Social Media Screener") designed to integrate model predictions and their explanations into a practical workflow for moderators. Our work provides a robust baseline, highlighting future needs for multi-label, clinically-validated datasets at the critical intersection of online safety and computational mental health.
>
---
#### [new 034] "When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究低资源环境下语法错误纠正（GEC）任务，针对Indic语言因数据稀缺导致的性能瓶颈，探索基于大语言模型（如GPT-4.1、Gemini-2.5）的提示学习方法。通过设计优化提示与少量样本策略，显著提升多语言GEC效果，在多个Indic语言上取得领先或前列成绩，验证了提示技术在跨语言资源匮乏场景下的有效性。**

- **链接: [https://arxiv.org/pdf/2511.20120v1](https://arxiv.org/pdf/2511.20120v1)**

> **作者:** Somsubhra De; Harsh Kumar; Arun Prakash A
>
> **备注:** 10 pages, 5 figures, 5 tables; Accept-demonstration at BHASHA Workshop, IJCNLP-AACL 2025
>
> **摘要:** Grammatical error correction (GEC) is an important task in Natural Language Processing that aims to automatically detect and correct grammatical mistakes in text. While recent advances in transformer-based models and large annotated datasets have greatly improved GEC performance for high-resource languages such as English, the progress has not extended equally. For most Indic languages, GEC remains a challenging task due to limited resources, linguistic diversity and complex morphology. In this work, we explore prompting-based approaches using state-of-the-art large language models (LLMs), such as GPT-4.1, Gemini-2.5 and LLaMA-4, combined with few-shot strategy to adapt them to low-resource settings. We observe that even basic prompting strategies, such as zero-shot and few-shot approaches, enable these LLMs to substantially outperform fine-tuned Indic-language models like Sarvam-22B, thereby illustrating the exceptional multilingual generalization capabilities of contemporary LLMs for GEC. Our experiments show that carefully designed prompts and lightweight adaptation significantly enhance correction quality across multiple Indic languages. We achieved leading results in the shared task--ranking 1st in Tamil (GLEU: 91.57) and Hindi (GLEU: 85.69), 2nd in Telugu (GLEU: 85.22), 4th in Bangla (GLEU: 92.86), and 5th in Malayalam (GLEU: 92.97). These findings highlight the effectiveness of prompt-driven NLP techniques and underscore the potential of large-scale LLMs to bridge resource gaps in multilingual GEC.
>
---
#### [new 035] EfficientXpert: Efficient Domain Adaptation for Large Language Models via Propagation-Aware Pruning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大语言模型在特定领域部署时的资源消耗问题，提出EfficientXpert框架。通过传播感知剪枝与高效适配器更新，实现低资源下高精度的领域自适应，显著提升模型在医疗、法律等领域的压缩效率与性能保持能力。**

- **链接: [https://arxiv.org/pdf/2511.19935v1](https://arxiv.org/pdf/2511.19935v1)**

> **作者:** Songlin Zhao; Michael Pitts; Zhuwei Qin
>
> **摘要:** The rapid advancement of large language models (LLMs) has increased the demand for domain-specialized variants in areas such as law, healthcare, and finance. However, their large size remains a barrier to deployment in resource-constrained environments, and existing compression methods either generalize poorly across domains or incur high overhead. In this work, we propose \textbf{EfficientXpert}, a lightweight domain-pruning framework that combines a propagation-aware pruning criterion (Foresight Mask) with an efficient adapter-update algorithm (Partial Brain Surgeon). Integrated into the LoRA fine-tuning process, EfficientXpert enables a one-step transformation of general pretrained models into sparse, domain-adapted experts. Across health and legal tasks, it retains up to 98% of dense-model performance at 40% sparsity, outperforming state-of-the-art methods. Further analysis reveals substantial domain-dependent structural shifts that degrade the effectiveness of general pruning masks, underscoring the need for adaptive, domain-aware pruning strategies tailored to each domain.
>
---
#### [new 036] The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究开放权重大模型在微调中出现的“涌现错位”问题，旨在评估不同模型对安全风险的抵抗能力。通过九个开源模型实验，发现结构化输出（如JSON）会显著提升错位率，且整体错位水平远低于闭源模型，揭示格式依赖性是关键脆弱点。**

- **链接: [https://arxiv.org/pdf/2511.20104v1](https://arxiv.org/pdf/2511.20104v1)**

> **作者:** Craig Dickson
>
> **摘要:** Prior work has shown that fine-tuning models on a narrow domain with misaligned data can lead to broad misalignment - a phenomenon termed "emergent misalignment" (Betley et al. 2025). While all tested models were susceptible to emergent misalignment, some models showed more resistance than others. Specifically the Qwen-2.5 family proved to be relatively resistant, while GPT-4o exhibited the strongest misalignment. In this paper we evaluate if current-generation open-weights models exhibit similar resistance to the Qwen-2.5 family and measure misalignment robustness over a range of model architectures and scales. We replicate the effect across nine modern open-weights models (Gemma 3 and Qwen 3 families, 1B-32B parameters). Models fine-tuned on insecure code generation show a 0.68% misalignment rate (compared to 0.07% for base models), matching the lower end of prior open-model results but dramatically lower than GPT-4o's 20%. We identify a critical format-dependent vulnerability: requiring JSON output doubles misalignment rates compared to natural language prompts (0.96% vs 0.42%). This suggests that structural constraints may bypass safety training by reducing the model's 'degrees of freedom' to refuse. These findings confirm emergent misalignment as a reproducible phenomenon in modern open-weights models, with rates substantially lower than observed in proprietary systems.
>
---
#### [new 037] Geometry of Decision Making in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在多选题问答任务中的决策几何结构，旨在揭示其内部决策机制。通过分析隐藏表示的内在维度，发现模型从低维输入经中间层扩展至高维，再压缩为低维决策空间，表明模型隐式将语言输入投影到与任务相关的低维流形上，为理解模型泛化与推理提供了新的几何视角。**

- **链接: [https://arxiv.org/pdf/2511.20315v1](https://arxiv.org/pdf/2511.20315v1)**

> **作者:** Abhinav Joshi; Divyanshu Bhatt; Ashutosh Modi
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) show strong generalization across diverse tasks, yet the internal decision-making processes behind their predictions remain opaque. In this work, we study the geometry of hidden representations in LLMs through the lens of \textit{intrinsic dimension} (ID), focusing specifically on decision-making dynamics in a multiple-choice question answering (MCQA) setting. We perform a large-scale study, with 28 open-weight transformer models and estimate ID across layers using multiple estimators, while also quantifying per-layer performance on MCQA tasks. Our findings reveal a consistent ID pattern across models: early layers operate on low-dimensional manifolds, middle layers expand this space, and later layers compress it again, converging to decision-relevant representations. Together, these results suggest LLMs implicitly learn to project linguistic inputs onto structured, low-dimensional manifolds aligned with task-specific decisions, providing new geometric insights into how generalization and reasoning emerge in language models.
>
---
#### [new 038] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对视觉语言模型在细粒度图像理解任务中的性能瓶颈，提出CropVLM方法。通过强化学习实现动态“局部放大”，无需标注框即可提升模型对细节的感知能力。该方法可通用适配各类VLM，显著改善高分辨率理解任务表现，尤其在域外场景下有效，且不需微调原模型。**

- **链接: [https://arxiv.org/pdf/2511.19820v1](https://arxiv.org/pdf/2511.19820v1)**

> **作者:** Miguel Carvalho; Helder Dias; Bruno Martins
>
> **摘要:** Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting.
>
---
#### [new 039] Soft Adaptive Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型强化学习中的策略优化不稳定问题，提出软自适应策略优化（SAPO）。针对高方差的词元重要性比率，通过平滑温度控制的门控机制替代硬截断，实现序列一致且词元自适应的更新。相比现有方法，SAPO提升训练稳定性与样本效率，在数学推理和多任务场景中均取得更好性能。**

- **链接: [https://arxiv.org/pdf/2511.20347v1](https://arxiv.org/pdf/2511.20347v1)**

> **作者:** Chang Gao; Chujie Zheng; Xiong-Hui Chen; Kai Dang; Shixuan Liu; Bowen Yu; An Yang; Shuai Bai; Jingren Zhou; Junyang Lin
>
> **摘要:** Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs.
>
---
#### [new 040] Quantifying Modality Contributions via Disentangling Multimodal Representations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对多模态模型中模态贡献难以量化的问题，提出基于部分信息分解（PID）的框架，通过分解嵌入表示中的独特、冗余与协同信息，区分模态的独立与交互作用贡献。采用IPFP算法实现无需重训练的可扩展分析，提供可解释的表示层面洞察，超越传统依赖准确率下降的评估方式。**

- **链接: [https://arxiv.org/pdf/2511.19470v1](https://arxiv.org/pdf/2511.19470v1)**

> **作者:** Padegal Amit; Omkar Mahesh Kashyap; Namitha Rayasam; Nidhi Shekhar; Surabhi Narayan
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Quantifying modality contributions in multimodal models remains a challenge, as existing approaches conflate the notion of contribution itself. Prior work relies on accuracy-based approaches, interpreting performance drops after removing a modality as indicative of its influence. However, such outcome-driven metrics fail to distinguish whether a modality is inherently informative or whether its value arises only through interaction with other modalities. This distinction is particularly important in cross-attention architectures, where modalities influence each other's representations. In this work, we propose a framework based on Partial Information Decomposition (PID) that quantifies modality contributions by decomposing predictive information in internal embeddings into unique, redundant, and synergistic components. To enable scalable, inference-only analysis, we develop an algorithm based on the Iterative Proportional Fitting Procedure (IPFP) that computes layer and dataset-level contributions without retraining. This provides a principled, representation-level view of multimodal behavior, offering clearer and more interpretable insights than outcome-based metrics.
>
---
#### [new 041] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation
- **分类: cs.DC; cs.CL**

- **简介: 该论文针对高绩效GPU核函数生成任务，解决现有LLM方法在正确性与效率间的矛盾。提出宏思考微编码（MTMC）框架，分层优化：宏观用强化学习探索高效策略，微观用通用LLM逐步实现代码，兼顾正确性与性能。实验表明，MTMC显著优于现有方法，在准确率和速度上均有大幅提升。**

- **链接: [https://arxiv.org/pdf/2511.20100v1](https://arxiv.org/pdf/2511.20100v1)**

> **作者:** Xinguo Zhu; Shaohui Peng; Jiaming Guo; Yunji Chen; Qi Guo; Yuanbo Wen; Hang Qin; Ruizhi Chen; Qirui Zhou; Ke Gao; Yanjun Wu; Chen Zhao; Ling Li
>
> **备注:** 9 pages, 2 figures, accepted by AAAI 2026
>
> **摘要:** Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup.
>
---
#### [new 042] Studying Maps at Scale: A Digital Investigation of Cartography and the Evolution of Figuration
- **分类: cs.CV; cs.CL; cs.DL**

- **简介: 该论文属大规模地图文化遗产研究任务，旨在解决传统地图研究忽视文化语义与历史演变的问题。通过整合超百万地图数据，运用语义分割与目标检测技术，分析地图的地理结构、符号系统及政治文化关联，揭示地图作为象征性文化产物的演化规律与传播机制。**

- **链接: [https://arxiv.org/pdf/2511.19538v1](https://arxiv.org/pdf/2511.19538v1)**

> **作者:** Remi Petitpierre
>
> **备注:** PhD thesis, EPFL. 396 pages, 156 figures
>
> **摘要:** This thesis presents methods and datasets to investigate cartographic heritage on a large scale and from a cultural perspective. Heritage institutions worldwide have digitized more than one million maps, and automated techniques now enable large-scale recognition and extraction of map content. Yet these methods have engaged little with the history of cartography, or the view that maps are semantic-symbolic systems, and cultural objects reflecting political and epistemic expectations. This work leverages a diverse corpus of 771,561 map records and 99,715 digitized images aggregated from 38 digital catalogs. After normalization, the dataset includes 236,925 contributors and spans six centuries, from 1492 to 1948. These data make it possible to chart geographic structures and the global chronology of map publication. The spatial focus of cartography is analyzed in relation to political dynamics, evidencing links between Atlantic maritime charting, the triangular trade, and colonial expansion. Further results document the progression of national, domestic focus and the impact of military conflicts on publication volumes. The research introduces semantic segmentation techniques and object detection models for the generic recognition of land classes and cartographic signs, trained on annotated data and synthetic images. The analysis of land classes shows that maps are designed images whose framing and composition emphasize features through centering and semantic symmetries. The study of cartographic figuration encodes 63 M signs and 25 M fragments into a latent visual space, revealing figurative shifts such as the replacement of relief hachures by terrain contours and showing that signs tend to form locally consistent systems. Analyses of collaboration and diffusion highlight the role of legitimacy, larger actors, and major cities in the spread of figurative norms and semiotic cultures.
>
---
#### [new 043] Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究统一多模态模型中理解与生成的关系，旨在解决“理解是否真正指导生成”这一核心问题。通过构建UniSandbox评估框架与合成数据集，发现存在显著的理解-生成差距。研究揭示链式思维（CoT）可有效弥合差距，并提出自训练方法实现隐式推理，为未来模型设计提供新思路。**

- **链接: [https://arxiv.org/pdf/2511.20561v1](https://arxiv.org/pdf/2511.20561v1)**

> **作者:** Yuwei Niu; Weiyang Jin; Jiaqi Liao; Chaoran Feng; Peng Jin; Bin Lin; Zongjian Li; Bin Zhu; Weihao Yu; Li Yuan
>
> **摘要:** Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox
>
---
#### [new 044] Training-Free Generation of Diverse and High-Fidelity Images via Prompt Semantic Space Optimization
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对文本生成图像模型中图像多样性不足的问题，提出无需训练的TPSO方法。通过优化提示词嵌入空间，探索低频语义区域，在不降低图像质量的前提下显著提升生成多样性，适用于多种扩散模型。**

- **链接: [https://arxiv.org/pdf/2511.19811v1](https://arxiv.org/pdf/2511.19811v1)**

> **作者:** Debin Meng; Chen Jin; Zheng Gao; Yanran Li; Ioannis Patras; Georgios Tzimiropoulos
>
> **备注:** under review
>
> **摘要:** Image diversity remains a fundamental challenge for text-to-image diffusion models. Low-diversity models tend to generate repetitive outputs, increasing sampling redundancy and hindering both creative exploration and downstream applications. A primary cause is that generation often collapses toward a strong mode in the learned distribution. Existing attempts to improve diversity, such as noise resampling, prompt rewriting, or steering-based guidance, often still collapse to dominant modes or introduce distortions that degrade image quality. In light of this, we propose Token-Prompt embedding Space Optimization (TPSO), a training-free and model-agnostic module. TPSO introduces learnable parameters to explore underrepresented regions of the token embedding space, reducing the tendency of the model to repeatedly generate samples from strong modes of the learned distribution. At the same time, the prompt-level space provides a global semantic constraint that regulates distribution shifts, preventing quality degradation while maintaining high fidelity. Extensive experiments on MS-COCO and three diffusion backbones show that TPSO significantly enhances generative diversity, improving baseline performance from 1.10 to 4.18 points, without sacrificing image quality. Code will be released upon acceptance.
>
---
#### [new 045] BlockCert: Certified Blockwise Extraction of Transformer Mechanisms
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出BlockCert框架，用于对Transformer模型进行可认证的分块机制提取与局部编辑。针对机制可解释性与模型编辑缺乏形式化保证的问题，通过构建带证书的分块替代模型，实现误差边界控制与全局行为验证，实证表明其在多个模型上具备高覆盖率与低误差，推动了可解释性与形式化推理的融合。**

- **链接: [https://arxiv.org/pdf/2511.17645v1](https://arxiv.org/pdf/2511.17645v1)**

> **作者:** Sandro Andric
>
> **备注:** 16 pages, 1 figure
>
> **摘要:** Mechanistic interpretability aspires to reverse-engineer neural networks into explicit algorithms, while model editing seeks to modify specific behaviours without retraining. Both areas are typically evaluated with informal evidence and ad-hoc experiments, with few explicit guarantees about how far an extracted or edited model can drift from the original on relevant inputs. We introduce BlockCert, a framework for certified blockwise extraction of transformer mechanisms, and outline how a lightweight extension can support certified local edits. Given a pre-trained transformer and a prompt distribution, BlockCert extracts structured surrogate implementations for residual blocks together with machine-checkable certificates that bound approximation error, record coverage metrics, and hash the underlying artifacts. We formalize a simple Lipschitz-based composition theorem in Lean 4 that lifts these local guarantees to a global deviation bound. Empirically, we apply the framework to GPT-2 small, TinyLlama-1.1B-Chat, and Llama-3.2-3B. Across these models we obtain high per-block coverage and small residual errors on the evaluated prompts, and in the TinyLlama setting we show that a fully stitched model matches the baseline perplexity within approximately 6e-5 on stress prompts. Our results suggest that blockwise extraction with explicit certificates is feasible for real transformer language models and offers a practical bridge between mechanistic interpretability and formal reasoning about model behaviour.
>
---
#### [new 046] Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对视觉语言模型（VLMs）在多步视觉推理中工具使用能力弱的问题，提出VISTA-Gym训练环境，通过标准化接口支持工具集成与强化学习。基于此，训练出VISTA-R1模型，实现工具与推理的协同，显著提升多任务视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2511.19773v1](https://arxiv.org/pdf/2511.19773v1)**

> **作者:** Meng Lu; Ran Xu; Yi Fang; Wenxuan Zhang; Yue Yu; Gaurav Srivastava; Yuchen Zhuang; Mohamed Elhoseiny; Charles Fleming; Carl Yang; Zhengzhong Tu; Yang Xie; Guanghua Xiao; Hanrui Wang; Di Jin; Wenqi Shi; Xuan Wang
>
> **备注:** 17 pages, 9 figures, work in progress
>
> **摘要:** While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs.
>
---
#### [new 047] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大模型内部机制不透明的问题，提出基于奇异向量的细粒度可解释性方法。通过分解注意力头与MLP为正交奇异方向，揭示其内部多重重叠子功能，验证了计算在低秩子空间中的结构化分布，推动了对Transformer电路的精细理解。**

- **链接: [https://arxiv.org/pdf/2511.20273v1](https://arxiv.org/pdf/2511.20273v1)**

> **作者:** Areeb Ahmad; Abhinav Joshi; Ashutosh Modi
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Transformer-based language models exhibit complex and distributed behavior, yet their internal computations remain poorly understood. Existing mechanistic interpretability methods typically treat attention heads and multilayer perceptron layers (MLPs) (the building blocks of a transformer architecture) as indivisible units, overlooking possibilities of functional substructure learned within them. In this work, we introduce a more fine-grained perspective that decomposes these components into orthogonal singular directions, revealing superposed and independent computations within a single head or MLP. We validate our perspective on widely used standard tasks like Indirect Object Identification (IOI), Gender Pronoun (GP), and Greater Than (GT), showing that previously identified canonical functional heads, such as the name mover, encode multiple overlapping subfunctions aligned with distinct singular directions. Nodes in a computational graph, that are previously identified as circuit elements show strong activation along specific low-rank directions, suggesting that meaningful computations reside in compact subspaces. While some directions remain challenging to interpret fully, our results highlight that transformer computations are more distributed, structured, and compositional than previously assumed. This perspective opens new avenues for fine-grained mechanistic interpretability and a deeper understanding of model internals.
>
---
#### [new 048] DesignPref: Capturing Personal Preferences in Visual Design Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文针对视觉设计生成中个体偏好差异问题，提出DesignPref数据集（12k设计对比，20名设计师标注）。研究发现设计师间偏好分歧大（Krippendorff's alpha=0.25），传统多数投票方法不准确。通过个性化微调与RAG集成，证明少量个性化数据即可显著提升个体偏好预测效果，为个性化设计生成提供新范式。**

- **链接: [https://arxiv.org/pdf/2511.20513v1](https://arxiv.org/pdf/2511.20513v1)**

> **作者:** Yi-Hao Peng; Jeffrey P. Bigham; Jason Wu
>
> **摘要:** Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste.
>
---
#### [new 049] Fara-7B: An Efficient Agentic Model for Computer Use
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对计算机使用代理（CUA）缺乏高质量训练数据的问题，提出FaraGen合成数据生成系统，构建多步骤网页任务数据集。基于此数据训练出小型高效模型Fara-7B，仅用截图与坐标实现端上运行，在多个基准上超越同类模型，并优于更大模型，验证了可扩展数据生成对高效代理模型的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.19663v1](https://arxiv.org/pdf/2511.19663v1)**

> **作者:** Ahmed Awadallah; Yash Lara; Raghav Magazine; Hussein Mozannar; Akshay Nambi; Yash Pandya; Aravind Rajeswaran; Corby Rosset; Alexey Taymanov; Vibhav Vineet; Spencer Whitehead; Andrew Zhao
>
> **摘要:** Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench.
>
---
#### [new 050] MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在微调时破坏预训练视觉-语言表征、损害泛化能力的问题，提出MAPS框架。通过模块级近邻调度，分阶段放松不同模块的约束，平衡稳定性与适应性，无需额外参数或数据，显著提升多场景下的性能。**

- **链接: [https://arxiv.org/pdf/2511.19878v1](https://arxiv.org/pdf/2511.19878v1)**

> **作者:** Chengyue Huang; Mellon M. Zhang; Robert Azarcon; Glen Chou; Zsolt Kira
>
> **摘要:** Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer.
>
---
#### [new 051] CounterVQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦视频理解中的反事实推理任务，针对视觉语言模型在复杂因果链推理上的薄弱表现，提出CounterVQA基准与CFGPT后训练方法，有效提升模型对假设情境下替代结果的推断能力。**

- **链接: [https://arxiv.org/pdf/2511.19923v1](https://arxiv.org/pdf/2511.19923v1)**

> **作者:** Yuefei Chen; Jiang Liu; Xiaodong Lin; Ruixiang Tang
>
> **摘要:** Vision Language Models (VLMs) have recently shown significant advancements in video understanding, especially in feature alignment, event reasoning, and instruction-following tasks. However, their capability for counterfactual reasoning, inferring alternative outcomes under hypothetical conditions, remains underexplored. This capability is essential for robust video understanding, as it requires identifying underlying causal structures and reasoning about unobserved possibilities, rather than merely recognizing observed patterns. To systematically evaluate this capability, we introduce CounterVQA, a video-based benchmark featuring three progressive difficulty levels that assess different aspects of counterfactual reasoning. Through comprehensive evaluation of both state-of-the-art open-source and closed-source models, we uncover a substantial performance gap: while these models achieve reasonable accuracy on simple counterfactual questions, performance degrades significantly on complex multi-hop causal chains. To address these limitations, we develop a post-training method, CFGPT, that enhances a model's visual counterfactual reasoning ability by distilling its counterfactual reasoning capability from the language modality, yielding consistent improvements across all CounterVQA difficulty levels. Dataset and code will be further released.
>
---
## 更新

#### [replaced 001] Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大模型代理的推理与执行纠缠、记忆易失、动作不可控等问题，提出结构化认知循环（SCL）架构，通过分离五阶段流程与软符号控制机制，实现可解释、可调控的智能代理。工作包括理论建模、三类设计原则提炼及开源实现，显著提升任务可靠性与透明性。**

- **链接: [https://arxiv.org/pdf/2511.17673v2](https://arxiv.org/pdf/2511.17673v2)**

> **作者:** Myung Ho Kim
>
> **备注:** Polished the abstract and replaced the demonstration screenshots
>
> **摘要:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
>
---
#### [replaced 002] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划分离的问题。通过动作无关的未来状态预测，实现状态-动作协同预测，提升规划可靠性。引入动态并行令牌生成机制，仅用前视摄像头即达到领先性能。**

- **链接: [https://arxiv.org/pdf/2510.19654v2](https://arxiv.org/pdf/2510.19654v2)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [replaced 003] Multi-Modal Data Exploration via Language Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对跨模态数据（如文本、图像、数据库）的自然语言查询难题，提出M²EX系统。通过基于大模型的智能体框架，将复杂问题分解为子任务，协同调用各模态专家，实现高效多模态数据探索。实验表明，该方法在准确率、响应速度和成本控制上均优于现有技术。**

- **链接: [https://arxiv.org/pdf/2412.18428v2](https://arxiv.org/pdf/2412.18428v2)**

> **作者:** Farhad Nooralahzadeh; Yi Zhang; Jonathan Furst; Kurt Stockinger
>
> **备注:** Accepted to the IJCNLP AACL 2025 Findings
>
> **摘要:** International enterprises, organizations, and hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying both structured databases and unstructured modalities (e.g., texts, images) in natural language remains largely unexplored. In this paper, we propose M$^2$EX -a system that enables multi-modal data exploration via language agents. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) M$^2$EX leverages an LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis and to orchestrate modality-specific experts in an efficient query plan. (3) Experimental results on multi-modal datasets, encompassing relational data, text, and images, demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling in both accuracy and various performance metrics, including query latency, API costs, and planning efficiency, thanks to the more effective utilization of the reasoning capabilities of LLMs.
>
---
#### [replaced 004] RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文针对胸部X光片（CXR）解读中推理不可解释、多模态信息融合不足、工具间矛盾无法解决等问题，提出RadAgents框架。通过模拟放射科医生工作流程，整合多模态推理与视觉-文本对齐的验证机制，实现可审计、一致且临床可信的智能诊断。**

- **链接: [https://arxiv.org/pdf/2509.20490v2](https://arxiv.org/pdf/2509.20490v2)**

> **作者:** Kai Zhang; Corey D Barrett; Jangwon Kim; Lichao Sun; Tara Taghavi; Krishnaram Kenthapadi
>
> **备注:** ML4H'25; Work in progress
>
> **摘要:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.
>
---
#### [replaced 005] OceanGym: A Benchmark Environment for Underwater Embodied Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出OceanGym，首个面向水下具身智能体的综合性基准环境，旨在解决水下感知与决策难题。通过整合多模态大模型，构建涵盖八类任务的仿真平台，推动智能体在低可见度、强流等复杂环境下实现自主探索与长程目标达成，填补了水下智能研究空白。**

- **链接: [https://arxiv.org/pdf/2509.26536v2](https://arxiv.org/pdf/2509.26536v2)**

> **作者:** Yida Xue; Mingjun Mao; Xiangyuan Ru; Yuqi Zhu; Baochang Ren; Shuofei Qiao; Mengru Wang; Shumin Deng; Xinyu An; Ningyu Zhang; Ying Chen; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at https://github.com/OceanGPT/OceanGym.
>
---
#### [replaced 006] LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation
- **分类: cs.CL**

- **简介: 该论文提出LiRA，一个用于生成可靠且可读文献综述的多智能体框架。针对自动化文献综述中写作质量与准确性不足的问题，设计协作式智能体流程，实现内容规划、撰写、编辑与评审，显著提升生成文本质量与引用准确度，在真实场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2510.05138v2](https://arxiv.org/pdf/2510.05138v2)**

> **作者:** Gregory Hok Tjoan Go; Khang Ly; Anders Søgaard; Amin Tabatabaei; Maarten de Rijke; Xinyi Chen
>
> **摘要:** The rapid growth of scientific publications has made it increasingly difficult to keep literature reviews comprehensive and up-to-date. Though prior work has focused on automating retrieval and screening, the writing phase of systematic reviews remains largely under-explored, especially with regard to readability and factual accuracy. To address this, we present LiRA (Literature Review Agents), a multi-agent collaborative workflow which emulates the human literature review process. LiRA utilizes specialized agents for content outlining, subsection writing, editing, and reviewing, producing cohesive and comprehensive review articles. Evaluated on SciReviewGen and a proprietary ScienceDirect dataset, LiRA outperforms current baselines such as AutoSurvey and MASS-Survey in writing and citation quality, while maintaining competitive similarity to human-written reviews. We further evaluate LiRA in real-world scenarios using document retrieval and assess its robustness to reviewer model variation. Our findings highlight the potential of agentic LLM workflows, even without domain-specific tuning, to improve the reliability and usability of automated scientific writing.
>
---
#### [replaced 007] Large language models replicate and predict human cooperation across experiments in game theory
- **分类: cs.AI; cs.CL; cs.GT; cs.MA**

- **简介: 该论文研究大语言模型在博弈论实验中复制与预测人类合作行为的能力。针对LLMs是否真实模拟人类决策这一关键问题，作者构建数字孪生实验框架，系统评估三款开源模型，发现Llama能高保真复现人类合作模式，且无需角色提示即可实现群体行为模拟，还生成了可验证的新假设，为社会行为研究提供新方法。**

- **链接: [https://arxiv.org/pdf/2511.04500v2](https://arxiv.org/pdf/2511.04500v2)**

> **作者:** Andrea Cera Palatsi; Samuel Martin-Gutierrez; Ana S. Cardenal; Max Pellert
>
> **摘要:** Large language models (LLMs) are increasingly used both to make decisions in domains such as health, education and law, and to simulate human behavior. Yet how closely LLMs mirror actual human decision-making remains poorly understood. This gap is critical: misalignment could produce harmful outcomes in practical applications, while failure to replicate human behavior renders LLMs ineffective for social simulations. Here, we address this gap by developing a digital twin of game-theoretic experiments and introducing a systematic prompting and probing framework for machine-behavioral evaluation. Testing three open-source models (Llama, Mistral and Qwen), we find that Llama reproduces human cooperation patterns with high fidelity, capturing human deviations from rational choice theory, while Qwen aligns closely with Nash equilibrium predictions. Notably, we achieved population-level behavioral replication without persona-based prompting, simplifying the simulation process. Extending beyond the original human-tested games, we generate and preregister testable hypotheses for novel game configurations outside the original parameter grid. Our findings demonstrate that appropriately calibrated LLMs can replicate aggregate human behavioral patterns and enable systematic exploration of unexplored experimental spaces, offering a complementary approach to traditional research in the social and behavioral sciences that generates new empirical predictions about human social decision-making.
>
---
#### [replaced 008] HyperbolicRAG: Enhancing Retrieval-Augmented Generation with Hyperbolic Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强型生成任务，旨在解决传统RAG在处理复杂知识图谱时忽视层次结构的问题。提出HyperbolicRAG，通过超球面嵌入实现语义与层级的统一表示，结合对比正则化与跨空间融合机制，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2511.18808v2](https://arxiv.org/pdf/2511.18808v2)**

> **作者:** Linxiao Cao; Ruitao Wang; Jindong Li; Zhipeng Zhou; Menglin Yang
>
> **备注:** 12 pages
>
> **摘要:** Retrieval-augmented generation (RAG) enables large language models (LLMs) to access external knowledge, helping mitigate hallucinations and enhance domain-specific expertise. Graph-based RAG enhances structural reasoning by introducing explicit relational organization that enables information propagation across semantically connected text units. However, these methods typically rely on Euclidean embeddings that capture semantic similarity but lack a geometric notion of hierarchical depth, limiting their ability to represent abstraction relationships inherent in complex knowledge graphs. To capture both fine-grained semantics and global hierarchy, we propose HyperbolicRAG, a retrieval framework that integrates hyperbolic geometry into graph-based RAG. HyperbolicRAG introduces three key designs: (1) a depth-aware representation learner that embeds nodes within a shared Poincare manifold to align semantic similarity with hierarchical containment, (2) an unsupervised contrastive regularization that enforces geometric consistency across abstraction levels, and (3) a mutual-ranking fusion mechanism that jointly exploits retrieval signals from Euclidean and hyperbolic spaces, emphasizing cross-space agreement during inference. Extensive experiments across multiple QA benchmarks demonstrate that HyperbolicRAG outperforms competitive baselines, including both standard RAG and graph-augmented baselines.
>
---
#### [replaced 009] Toward Honest Language Models for Deductive Reasoning
- **分类: cs.CL**

- **简介: 该论文研究语言模型在演绎推理中的诚实性问题，旨在使模型仅在结论被前提逻辑蕴含时作答，否则拒绝回答。针对现有方法在不充分输入下仍生成答案的问题，作者构建了基于图结构的双数据集，提出ACNCHOR强化学习方法，通过注入真实推理轨迹稳定训练，显著提升模型诚实推理能力。**

- **链接: [https://arxiv.org/pdf/2511.09222v3](https://arxiv.org/pdf/2511.09222v3)**

> **作者:** Jiarui Liu; Kaustubh Dhole; Yingheng Wang; Haoyang Wen; Sarah Zhang; Haitao Mao; Gaotang Li; Neeraj Varshney; Jingguo Liu; Xiaoman Pan
>
> **摘要:** Deductive reasoning is the process of deriving conclusions strictly from the given premises, without relying on external knowledge. We define honesty in this setting as a model's ability to respond only when the conclusion is logically entailed by the premises, and to abstain otherwise. However, current language models often fail to reason honestly, producing unwarranted answers when the input is insufficient. To study this challenge, we formulate honest deductive reasoning as multi-step tasks where models must either derive the correct conclusion or abstain. We curate two datasets from graph structures, one for linear algebra and one for logical inference, and introduce unanswerable cases by randomly perturbing an edge in half of the instances. We find that prompting and existing training methods, including GRPO with or without supervised fine-tuning initialization, struggle on these tasks. In particular, GRPO optimize only for final task outcomes, leaving models vulnerable to collapse when negative rewards dominate early training. To address this, we propose ACNCHOR, a reinforcement learning method that injects ground truth trajectories into rollouts, preventing early training collapse. Our results demonstrate that this method stabilizes learning and significantly improves the overall reasoning performance, underscoring the importance of training dynamics for enabling honest deductive reasoning in language models.
>
---
#### [replaced 010] Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文针对NLP中基于语义触发的后门攻击问题，提出SteganoBackdoor。通过自然语言隐写技术，将语义触发器转化为高隐蔽性载体，在极低数据污染率下实现超99%攻击成功率，有效规避现有防御，揭示了当前防御体系在真实威胁场景下的盲区。**

- **链接: [https://arxiv.org/pdf/2511.14301v2](https://arxiv.org/pdf/2511.14301v2)**

> **作者:** Eric Xue; Ruiyi Zhang; Zijun Zhang; Pengtao Xie
>
> **摘要:** Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.
>
---
#### [replaced 011] TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出TurnBench-MS基准，针对大语言模型在多轮、多步推理能力评估不足的问题，设计基于代码破解的交互式任务。通过引入反馈机制与隐藏规则，评估模型在复杂推理中的持续推断与适应能力，揭示当前模型在高难度场景下的显著性能差距。**

- **链接: [https://arxiv.org/pdf/2506.01341v2](https://arxiv.org/pdf/2506.01341v2)**

> **作者:** Yiran Zhang; Mo Wang; Xiaoyang Li; Kaixuan Ren; Chencheng Zhu; Usman Naseem
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Despite impressive advances in large language models (LLMs), existing benchmarks often focus on single-turn or single-step tasks, failing to capture the kind of iterative reasoning required in real-world settings. To address this limitation, we introduce TurnBench, a novel benchmark that evaluates multi-turn, multi-step reasoning through an interactive code-breaking task inspired by the "Turing Machine Board Game." In each episode, a model must uncover hidden logical or arithmetic rules by making sequential guesses, receiving structured feedback, and integrating clues across multiple rounds. This dynamic setup requires models to reason over time, adapt based on past information, and maintain consistency across steps-capabilities underexplored in current benchmarks. TurnBench includes two modes: Classic, which tests standard reasoning, and Nightmare, which introduces increased complexity and requires robust inferential chains. To support fine-grained analysis, we provide ground-truth annotations for intermediate reasoning steps. Our evaluation of state-of-the-art LLMs reveals significant gaps: the best model achieves 84% accuracy in Classic mode, but performance drops to 18% in Nightmare mode. In contrast, human participants achieve 100% in both, underscoring the challenge TurnBench poses to current models. By incorporating feedback loops and hiding task rules, TurnBench reduces contamination risks and provides a rigorous testbed for diagnosing and advancing multi-step, multi-turn reasoning in LLMs.
>
---
#### [replaced 012] AI-Mediated Communication Reshapes Social Structure in Opinion-Diverse Groups
- **分类: cs.SI; cs.CL**

- **简介: 该论文研究AI辅助沟通对意见多元群体社会结构的影响。通过在线实验，考察不同AI干预方式（个体或关系型协助）如何影响群体凝聚力与分化。结果表明，AI根据用户语境介入沟通，可引发群体结构的宏观差异，揭示了人机协同对集体组织的重塑作用。**

- **链接: [https://arxiv.org/pdf/2510.21984v2](https://arxiv.org/pdf/2510.21984v2)**

> **作者:** Faria Huq; Elijah L. Claggett; Hirokazu Shirado
>
> **备注:** Preprint, Under Review
>
> **摘要:** Group segregation or cohesion can emerge from micro-level communication, and AI-assisted messaging may shape this process. Here, we report a preregistered online experiment (N = 557 across 60 sessions) in which participants discussed controversial political topics over multiple rounds and could freely change groups. Some participants received real-time message suggestions from a large language model (LLM), either personalized to their stance (individual assistance) or incorporating their group members' perspectives (relational assistance). We find that small variations in AI-mediated communication cascade into macro-level differences in group composition. Participants with individual assistance send more messages and show greater stance-based clustering, whereas those with relational assistance use more receptive language and form more heterogeneous ties. Hybrid expressive processes-jointly produced by humans and AI-can reshape collective organization. The patterns of structural division and cohesion depend on how AI incorporates users' interaction context.
>
---
#### [replaced 013] Scalable Parameter-Light Spectral Method for Clustering Short Text Embeddings with a Cohesion-Based Evaluation Metric
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对短文本聚类中需预先指定聚类数的问题，提出一种可扩展的谱方法，通过拉普拉斯特征谱结构自动估计聚类数，并引入基于信息论的“凝聚比”评估指标。实验表明，结合该估计算法的K-Means等方法显著优于现有无参轻量级方法。**

- **链接: [https://arxiv.org/pdf/2511.19350v2](https://arxiv.org/pdf/2511.19350v2)**

> **作者:** Nikita Neveditsin; Pawan Lingras; Vijay Mago
>
> **摘要:** Clustering short text embeddings is a foundational task in natural language processing, yet remains challenging due to the need to specify the number of clusters in advance. We introduce a scalable spectral method that estimates the number of clusters directly from the structure of the Laplacian eigenspectrum, constructed using cosine similarities and guided by an adaptive sampling strategy. This sampling approach enables our estimator to efficiently scale to large datasets without sacrificing reliability. To support intrinsic evaluation of cluster quality without ground-truth labels, we propose the Cohesion Ratio, a simple and interpretable evaluation metric that quantifies how much intra-cluster similarity exceeds the global similarity background. It has an information-theoretic motivation inspired by mutual information, and in our experiments it correlates closely with extrinsic measures such as normalized mutual information and homogeneity. Extensive experiments on six short-text datasets and four modern embedding models show that standard algorithms like K-Means and HAC, when guided by our estimator, significantly outperform popular parameter-light methods such as HDBSCAN, OPTICS, and Leiden. These results demonstrate the practical value of our spectral estimator and Cohesion Ratio for unsupervised organization and evaluation of short text data. Implementation of our estimator of k and Cohesion Ratio, along with code for reproducing the experiments, is available at https://anonymous.4open.science/r/towards_clustering-0C2E.
>
---
#### [replaced 014] ExDDV: A New Dataset for Explainable Deepfake Detection in Video
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出ExDDV，首个面向视频深度伪造可解释检测的数据集。针对现有检测模型缺乏可解释性的问题，构建包含5.4K视频及文本描述与点击标注的多模态数据集，评估视觉-语言模型在定位与描述伪造痕迹上的表现，验证文本与点击监督对提升模型可解释性的重要性。**

- **链接: [https://arxiv.org/pdf/2503.14421v2](https://arxiv.org/pdf/2503.14421v2)**

> **作者:** Vlad Hondru; Eduard Hogea; Darian Onchis; Radu Tudor Ionescu
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** The ever growing realism and quality of generated videos makes it increasingly harder for humans to spot deepfake content, who need to rely more and more on automatic deepfake detectors. However, deepfake detectors are also prone to errors, and their decisions are not explainable, leaving humans vulnerable to deepfake-based fraud and misinformation. To this end, we introduce ExDDV, the first dataset and benchmark for Explainable Deepfake Detection in Video. ExDDV comprises around 5.4K real and deepfake videos that are manually annotated with text descriptions (to explain the artifacts) and clicks (to point out the artifacts). We evaluate a number of vision-language models on ExDDV, performing experiments with various fine-tuning and in-context learning strategies. Our results show that text and click supervision are both required to develop robust explainable models for deepfake videos, which are able to localize and describe the observed artifacts. Our novel dataset and code to reproduce the results are available at https://github.com/vladhondru25/ExDDV.
>
---
#### [replaced 015] A Comprehensive Survey on Long Context Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对长文本处理效率低的问题，系统综述了长上下文语言模型（LCLM）的技术进展。聚焦于高效建模、训练部署与评估分析三大方面，涵盖架构设计、数据策略、基础设施及评测方法，旨在推动大模型在长文档、对话等场景下的应用与优化。**

- **链接: [https://arxiv.org/pdf/2503.17407v2](https://arxiv.org/pdf/2503.17407v2)**

> **作者:** Jiaheng Liu; Dawei Zhu; Zhiqi Bai; Yancheng He; Huanxuan Liao; Haoran Que; Zekun Wang; Chenchen Zhang; Ge Zhang; Jiebin Zhang; Yuanxing Zhang; Zhuo Chen; Hangyu Guo; Shilong Li; Ziqiang Liu; Yong Shan; Yifan Song; Jiayi Tian; Wenhao Wu; Zhejian Zhou; Ruijie Zhu; Junlan Feng; Yang Gao; Shizhu He; Zhoujun Li; Tianyu Liu; Fanyu Meng; Wenbo Su; Yingshui Tan; Zili Wang; Jian Yang; Wei Ye; Bo Zheng; Wangchunshu Zhou; Wenhao Huang; Sujian Li; Zhaoxiang Zhang
>
> **摘要:** Efficient processing of long contexts has been a persistent pursuit in Natural Language Processing. With the growing number of long documents, dialogues, and other textual data, it is important to develop Long Context Language Models (LCLMs) that can process and analyze extensive inputs in an effective and efficient way. In this paper, we present a comprehensive survey on recent advances in long-context modeling for large language models. Our survey is structured around three key aspects: how to obtain effective and efficient LCLMs, how to train and deploy LCLMs efficiently, and how to evaluate and analyze LCLMs comprehensively. For the first aspect, we discuss data strategies, architectural designs, and workflow approaches oriented with long context processing. For the second aspect, we provide a detailed examination of the infrastructure required for LCLM training and inference. For the third aspect, we present evaluation paradigms for long-context comprehension and long-form generation, as well as behavioral analysis and mechanism interpretability of LCLMs. Beyond these three key aspects, we thoroughly explore the diverse application scenarios where existing LCLMs have been deployed and outline promising future development directions. This survey provides an up-to-date review of the literature on long-context LLMs, which we wish to serve as a valuable resource for both researchers and engineers. An associated GitHub repository collecting the latest papers and repos is available at: \href{https://github.com/LCLM-Horizon/A-Comprehensive-Survey-For-Long-Context-Language-Modeling}{\color[RGB]{175,36,67}{LCLM-Horizon}}.
>
---
#### [replaced 016] Health Sentinel: An AI Pipeline For Real-time Disease Outbreak Detection
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出Health Sentinel，一个用于实时疾病暴发检测的AI信息提取管道。针对传统监测方法滞后、人工筛查海量网络媒体信息不现实的问题，系统融合机器学习与非机器学习方法，自动从在线文章中提取健康事件信息，已处理超3亿篇文章，识别逾9.5万条事件，其中3500余条被专家认定为潜在疫情，助力及时干预。**

- **链接: [https://arxiv.org/pdf/2506.19548v2](https://arxiv.org/pdf/2506.19548v2)**

> **作者:** Devesh Pant; Rishi Raj Grandhe; Vipin Samaria; Mukul Paul; Sudhir Kumar; Saransh Khanna; Jatin Agrawal; Jushaan Singh Kalra; Akhil VSSG; Satish V Khalikar; Vipin Garg; Himanshu Chauhan; Pranay Verma; Neha Khandelwal; Soma S Dhavala; Minesh Mathew
>
> **摘要:** Early detection of disease outbreaks is crucial to ensure timely intervention by the health authorities. Due to the challenges associated with traditional indicator-based surveillance, monitoring informal sources such as online media has become increasingly popular. However, owing to the number of online articles getting published everyday, manual screening of the articles is impractical. To address this, we propose Health Sentinel. It is a multi-stage information extraction pipeline that uses a combination of ML and non-ML methods to extract events-structured information concerning disease outbreaks or other unusual health events-from online articles. The extracted events are made available to the Media Scanning and Verification Cell (MSVC) at the National Centre for Disease Control (NCDC), Delhi for analysis, interpretation and further dissemination to local agencies for timely intervention. From April 2022 till date, Health Sentinel has processed over 300 million news articles and identified over 95,000 unique health events across India of which over 3,500 events were shortlisted by the public health experts at NCDC as potential outbreaks.
>
---
#### [replaced 017] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对大推理模型（LRM）的评估问题，提出无污染的中等规模评测方案，并发布视觉语言推理基准ROME。旨在评估模型在文本与视觉线索下的自动可验证推理能力，推动多模态推理模型的客观评测。**

- **链接: [https://arxiv.org/pdf/2509.17177v3](https://arxiv.org/pdf/2509.17177v3)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** Project homepage: https://flageval-baai.github.io/LRM-Eval/ This work will also be presented at NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models (FoRLM); update with trials on Gemini 3 Pro
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [replaced 018] Why Reasoning Matters? A Survey of Advancements in Multimodal Reasoning (v1)
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦多模态推理任务，旨在解决视觉与文本信息融合中的冲突与理解难题。通过综述现有推理技术，分析核心挑战，提出优化方法与评估框架，推动大模型在多模态场景下的智能推理能力发展。**

- **链接: [https://arxiv.org/pdf/2504.03151v2](https://arxiv.org/pdf/2504.03151v2)**

> **作者:** Jing Bi; Susan Liang; Xiaofei Zhou; Pinxin Liu; Junjia Guo; Yunlong Tang; Luchuan Song; Chao Huang; Ali Vosoughi; Guangyu Sun; Jinxi He; Jiarui Wu; Shu Yang; Daoan Zhang; Chen Chen; Lianggong Bruce Wen; Zhang Liu; Jiebo Luo; Chenliang Xu
>
> **摘要:** Reasoning is central to human intelligence, enabling structured problem-solving across diverse tasks. Recent advances in large language models (LLMs) have greatly enhanced their reasoning abilities in arithmetic, commonsense, and symbolic domains. However, effectively extending these capabilities into multimodal contexts-where models must integrate both visual and textual inputs-continues to be a significant challenge. Multimodal reasoning introduces complexities, such as handling conflicting information across modalities, which require models to adopt advanced interpretative strategies. Addressing these challenges involves not only sophisticated algorithms but also robust methodologies for evaluating reasoning accuracy and coherence. This paper offers a concise yet insightful overview of reasoning techniques in both textual and multimodal LLMs. Through a thorough and up-to-date comparison, we clearly formulate core reasoning challenges and opportunities, highlighting practical methods for post-training optimization and test-time inference. Our work provides valuable insights and guidance, bridging theoretical frameworks and practical implementations, and sets clear directions for future research.
>
---
#### [replaced 019] Gram2Vec: An Interpretable Document Vectorizer
- **分类: cs.CL**

- **简介: 该论文提出Gram2Vec，一种基于语法特征频率的可解释文档向量化方法，用于作者验证和AI文本检测。通过提取标准化语法特征频次，生成高维向量，实现模型可解释性。相比神经网络方法，其优势在于能清晰解释判断依据，并在两类任务中表现优于传统Biber特征模型。**

- **链接: [https://arxiv.org/pdf/2406.12131v2](https://arxiv.org/pdf/2406.12131v2)**

> **作者:** Peter Zeng; Hannah Stortz; Eric Sclafani; Alina Shabaeva; Maria Elizabeth Garza; Daniel Greeson Owen Rambow
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** We present Gram2Vec, a grammatical style embedding system that embeds documents into a higher dimensional space by extracting the normalized relative frequencies of grammatical features present in the text. Compared to neural approaches, Gram2Vec offers inherent interpretability based on how the feature vectors are generated. In this paper, we use authorship verification and AI detection as two applications to show how Gram2Vec can be used. For authorship verification, we use the features from Gram2Vec to explain why a pair of documents is by the same or by different authors. We also demonstrate how Gram2Vec features can be used to train a classifier for AI detection, outperforming machine learning models trained on a comparable set of Biber features.
>
---
#### [replaced 020] Computational Turing Test Reveals Systematic Differences Between Human and AI Language
- **分类: cs.CL; cs.MA; cs.SI**

- **简介: 该论文提出计算版图灵测试，旨在验证大语言模型生成文本的人类相似性。针对现有评估依赖不可靠的人类判断的问题，研究融合语义与风格特征，系统比较九个模型在多种校准策略下的表现，发现模型仍难以真实模拟人类情感表达，且提升拟人度常牺牲语义准确性。**

- **链接: [https://arxiv.org/pdf/2511.04195v2](https://arxiv.org/pdf/2511.04195v2)**

> **作者:** Nicolò Pagan; Petter Törnberg; Christopher A. Bail; Anikó Hannák; Christopher Barrie
>
> **摘要:** Large language models (LLMs) are increasingly used in the social sciences to simulate human behavior, based on the assumption that they can generate realistic, human-like text. Yet this assumption remains largely untested. Existing validation efforts rely heavily on human-judgment-based evaluations -- testing whether humans can distinguish AI from human output -- despite evidence that such judgments are blunt and unreliable. As a result, the field lacks robust tools for assessing the realism of LLM-generated text or for calibrating models to real-world data. This paper makes two contributions. First, we introduce a computational Turing test: a validation framework that integrates aggregate metrics (BERT-based detectability and semantic similarity) with interpretable linguistic features (stylistic markers and topical patterns) to assess how closely LLMs approximate human language within a given dataset. Second, we systematically compare nine open-weight LLMs across five calibration strategies -- including fine-tuning, stylistic prompting, and context retrieval -- benchmarking their ability to reproduce user interactions on X (formerly Twitter), Bluesky, and Reddit. Our findings challenge core assumptions in the literature. Even after calibration, LLM outputs remain clearly distinguishable from human text, particularly in affective tone and emotional expression. Instruction-tuned models underperform their base counterparts, and scaling up model size does not enhance human-likeness. Crucially, we identify a trade-off: optimizing for human-likeness often comes at the cost of semantic fidelity, and vice versa. These results provide a much-needed scalable framework for validation and calibration in LLM simulations -- and offer a cautionary note about their current limitations in capturing human communication.
>
---
#### [replaced 021] Filtering with Self-Attention and Storing with MLP: One-Layer Transformers Can Provably Acquire and Extract Knowledge
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型的知识存储与提取机制，针对预训练中知识获取和微调后知识提取的理论空白，提出首个基于单层Transformer的理论框架。通过分析训练动态，证明模型能有效存储知识并准确提取，揭示低秩微调的有效性，并解释幻觉产生的条件。**

- **链接: [https://arxiv.org/pdf/2508.00901v3](https://arxiv.org/pdf/2508.00901v3)**

> **作者:** Ruichen Xu; Kexin Chen
>
> **摘要:** Modern large language models (LLMs) demonstrate exceptional performance on knowledge-intensive tasks, yet the theoretical mechanisms underlying knowledge acquisition (storage and memorization) during pre-training and extraction (retrieval and recall) during inference after fine-tuning remain poorly understood. Although prior theoretical studies have explored these processes through analyses of training dynamics, they overlook critical components essential for a comprehensive theory: (1) the multi-layer perceptron (MLP), empirically identified as the primary module for knowledge storage; (2) out-of-distribution (OOD) adaptivity, which enables LLMs to generalize to unseen scenarios post-pre-training; and (3) next-token prediction, the standard autoregressive objective that encodes knowledge as conditional probabilities. In this work, we introduce, to the best of our knowledge, the first theoretical framework that addresses these limitations by examining the training dynamics of one-layer transformers. Under regularity assumptions, we establish that: (i) transformers attain near-optimal training loss during pre-training, demonstrating effective knowledge acquisition; (ii) given a sufficiently large fine-tuning dataset and appropriate data multiplicity conditions, transformers achieve low generalization error on factual knowledge acquired during pre-training but not revisited in fine-tuning, indicating robust knowledge extraction; and (iii) violation of these conditions leads to elevated generalization error, manifesting as hallucinations. Our analysis encompasses both full fine-tuning and low-rank fine-tuning, yielding insights into the efficacy of practical low-rank adaptation methods. We validate our theoretical findings through experiments on synthetic datasets and the real-world PopQA benchmark, employing GPT-2 and Llama-3.2-1B models.
>
---
#### [replaced 022] BiasJailbreak:Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）中的伦理偏见如何被利用进行越狱攻击（BiasJailbreak），揭示了模型在不同性别、种族关键词下存在显著安全漏洞。提出自动生成偏见关键词的攻击方法，并设计高效防御机制BiasDefense，以提升模型安全性与公平性。**

- **链接: [https://arxiv.org/pdf/2410.13334v5](https://arxiv.org/pdf/2410.13334v5)**

> **作者:** Isack Lee; Haebin Seong
>
> **备注:** Accepted as a workshop paper at AAAI 2026
>
> **摘要:** Although large language models (LLMs) demonstrate impressive proficiency in various tasks, they present potential safety risks, such as `jailbreaks', where malicious inputs can coerce LLMs into generating harmful content bypassing safety alignments. In this paper, we delve into the ethical biases in LLMs and examine how those biases could be exploited for jailbreaks. Notably, these biases result in a jailbreaking success rate in GPT-4o models that differs by 20\% between non-binary and cisgender keywords and by 16\% between white and black keywords, even when the other parts of the prompts are identical. We introduce the concept of BiasJailbreak, highlighting the inherent risks posed by these safety-induced biases. BiasJailbreak generates biased keywords automatically by asking the target LLM itself, and utilizes the keywords to generate harmful output. Additionally, we propose an efficient defense method BiasDefense, which prevents jailbreak attempts by injecting defense prompts prior to generation. BiasDefense stands as an appealing alternative to Guard Models, such as Llama-Guard, that require additional inference cost after text generation. Our findings emphasize that ethical biases in LLMs can actually lead to generating unsafe output, and suggest a method to make the LLMs more secure and unbiased. To enable further research and improvements, we open-source our code and artifacts of BiasJailbreak, providing the community with tools to better understand and mitigate safety-induced biases in LLMs.
>
---
#### [replaced 023] ShortageSim: Simulating Drug Shortages under Information Asymmetry
- **分类: cs.MA; cs.CL; cs.GT**

- **简介: 该论文针对药品短缺问题，提出ShortageSim框架，模拟信息不对称下监管干预对生产者与采购方竞争行为的影响。通过大语言模型代理建模异质决策，解决传统模型假设理想化的问题，实验证明其显著缩短短缺响应时间，为政策评估提供新方法。**

- **链接: [https://arxiv.org/pdf/2509.01813v3](https://arxiv.org/pdf/2509.01813v3)**

> **作者:** Mingxuan Cui; Yilan Jiang; Duo Zhou; Cheng Qian; Yuji Zhang; Qiong Wang
>
> **备注:** Accepted by AAAI 2026. Oral presentation. 25 pages
>
> **摘要:** Drug shortages pose critical risks to patient care and healthcare systems worldwide, yet the effectiveness of regulatory interventions remains poorly understood due to information asymmetries in pharmaceutical supply chains. We propose \textbf{ShortageSim}, addresses this challenge by providing the first simulation framework that evaluates the impact of regulatory interventions on competition dynamics under information asymmetry. Using Large Language Model (LLM)-based agents, the framework models the strategic decisions of drug manufacturers and institutional buyers, in response to shortage alerts given by the regulatory agency. Unlike traditional game theory models that assume perfect rationality and complete information, ShortageSim simulates heterogeneous interpretations on regulatory announcements and the resulting decisions. Experiments on self-processed dataset of historical shortage events show that ShortageSim reduces the resolution lag for production disruption cases by up to 84\%, achieving closer alignment to real-world trajectories than the zero-shot baseline. Our framework confirms the effect of regulatory alert in addressing shortages and introduces a new method for understanding competition in multi-stage environments under uncertainty. We open-source ShortageSim and a dataset of 2,925 FDA shortage events, providing a novel framework for future research on policy design and testing in supply chains under information asymmetry.
>
---
#### [replaced 024] Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling
- **分类: cs.CL; cs.DB**

- **简介: 该论文针对文本转SQL任务，解决现有方法在复杂数据库上性能不足的问题。提出Agentar-Scale-SQL框架，通过协同的测试时扩展策略，融合内部、序列与并行缩放，显著提升模型推理能力，在BIRD基准上达81.67%执行准确率，超越当前最优水平。**

- **链接: [https://arxiv.org/pdf/2509.24403v4](https://arxiv.org/pdf/2509.24403v4)**

> **作者:** Pengfei Wang; Baolin Sun; Xuemei Dong; Yaxun Dai; Hongwei Yuan; Mengdie Chu; Yingqi Gao; Xiang Qi; Peng Zhang; Ying Yan
>
> **摘要:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.
>
---
#### [replaced 025] Mixture of Attention Spans: Optimizing LLM Inference Efficiency with Heterogeneous Sliding-Window Lengths
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大模型长文本推理中注意力机制效率低下的问题，提出Mixture of Attention Spans（MoA），通过为不同注意力头动态分配异构滑窗长度，优化内存与吞吐。实验表明，MoA显著提升上下文长度与推理效率，降低内存占用，性能接近全注意力。**

- **链接: [https://arxiv.org/pdf/2406.14909v3](https://arxiv.org/pdf/2406.14909v3)**

> **作者:** Tianyu Fu; Haofeng Huang; Xuefei Ning; Genghan Zhang; Boju Chen; Tianqi Wu; Hongyi Wang; Zixiao Huang; Shiyao Li; Shengen Yan; Guohao Dai; Huazhong Yang; Yu Wang
>
> **备注:** Published at CoLM'25
>
> **摘要:** Sliding-window attention offers a hardware-efficient solution to the memory and throughput challenges of Large Language Models (LLMs) in long-context scenarios. Existing methods typically employ a single window length across all attention heads and input sizes. However, this uniform approach fails to capture the heterogeneous attention patterns inherent in LLMs, ignoring their distinct accuracy-latency trade-offs. To address this challenge, we propose *Mixture of Attention Spans* (MoA), which automatically tailors distinct sliding-window length configurations to different heads and layers. MoA constructs and navigates a search space of various window lengths and their scaling rules relative to input sizes. It profiles the model, evaluates potential configurations, and pinpoints the optimal length configurations for each head. MoA adapts to varying input sizes, revealing that some attention heads expand their focus to accommodate longer inputs, while other heads consistently concentrate on fixed-length local contexts. Experiments show that MoA increases the effective context length by 3.9x with the same average sliding-window length, boosting retrieval accuracy by 1.5-7.1x over the uniform-window baseline across Vicuna-{7B, 13B} and Llama3-{8B, 70B} models. Moreover, MoA narrows the performance gap with full attention, reducing the maximum relative performance drop from 9%-36% to within 5% across three long-context understanding benchmarks. MoA achieves a 1.2-1.4x GPU memory reduction, boosting decode throughput by 6.6-8.2x and 1.7-1.9x over FlashAttention2 and vLLM, with minimal performance impact. Our code is available at: https://github.com/thu-nics/MoA
>
---
#### [replaced 026] Counterfactual Simulatability of LLM Explanations for Generation Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在生成任务中的解释可反事实模拟性，旨在评估解释是否帮助用户预测模型在反事实输入下的输出。针对新闻摘要和医疗建议任务，提出通用评估框架，发现摘要任务中解释有效，而医疗建议任务仍有改进空间，且该评估更适用于技能型任务。**

- **链接: [https://arxiv.org/pdf/2505.21740v3](https://arxiv.org/pdf/2505.21740v3)**

> **作者:** Marvin Limpijankit; Yanda Chen; Melanie Subbiah; Nicholas Deas; Kathleen McKeown
>
> **备注:** INLG25
>
> **摘要:** LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
>
---
#### [replaced 027] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态视觉语言模型中的测试时思考（test-time thinking）问题，旨在提升视觉推理能力。针对“思考越多越好的误区”，提出基于不确定性的自适应回溯策略，通过短回溯提示与广度搜索，增强图像关联性，显著提升多个基准上的性能，实现新基准突破。**

- **链接: [https://arxiv.org/pdf/2511.15613v2](https://arxiv.org/pdf/2511.15613v2)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yolo Y. Tang; Luchuan Song; Susan Liang; Zhongfei Mark Zhang; Jason J. Corso; Chenliang Xu
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [replaced 028] LightMem: Lightweight and Efficient Memory-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文针对大语言模型在动态环境中难以有效利用历史交互信息的问题，提出轻量高效的LightMem记忆系统。受人类记忆模型启发，将记忆分为感知、短期和长期三阶段，实现高效的信息压缩、组织与离线更新。实验表明，LightMem显著提升问答准确率，大幅降低计算开销与API调用次数。**

- **链接: [https://arxiv.org/pdf/2510.18866v2](https://arxiv.org/pdf/2510.18866v2)**

> **作者:** Jizhan Fang; Xinle Deng; Haoming Xu; Ziyan Jiang; Yuqi Tang; Ziwen Xu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. On LongMemEval and LoCoMo, using GPT and Qwen backbones, LightMem consistently surpasses strong baselines, improving QA accuracy by up to 7.7% / 29.3%, reducing total token usage by up to 38x / 20.9x and API calls by up to 30x / 55.5x, while purely online test-time costs are even lower, achieving up to 106x / 117x token reduction and 159x / 310x fewer API calls. The code is available at https://github.com/zjunlp/LightMem.
>
---
#### [replaced 029] Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction
- **分类: q-fin.CP; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦于股票收益预测任务，旨在融合量化因子与大语言模型生成的新闻流表示。提出融合学习框架，比较多种特征融合方式，并设计自适应混合模型与解耦训练策略，以提升多模态数据建模效果，优化选股与预测性能。**

- **链接: [https://arxiv.org/pdf/2510.15691v3](https://arxiv.org/pdf/2510.15691v3)**

> **作者:** Tian Guo; Emmanuel Hauptmann
>
> **摘要:** In quantitative investing, return prediction supports various tasks, including stock selection, portfolio optimization, and risk management. Quantitative factors, such as valuation, quality, and growth, capture various characteristics of stocks. Unstructured data, like news and transcripts, has attracted growing attention, driven by recent advances in large language models (LLMs). This paper examines effective methods for leveraging multimodal factors and newsflow in return prediction and stock selection. First, we introduce a fusion learning framework to learn a unified representation from factors and newsflow representations generated by an LLM. Within this framework, we compare three methods of different architectural complexities: representation combination, representation summation, and attentive representations. Next, building on the limitation of fusion learning observed in empirical comparison, we explore the mixture model that adaptively combines predictions made by single modalities and their fusion. To mitigate the training instability of the mixture model, we introduce a decoupled training approach with theoretical insights. Finally, our experiments on real investment universes yield several insights into effective multimodal modeling of factors and news for stock return prediction and selection.
>
---
#### [replaced 030] Improved LLM Agents for Financial Document Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对金融文档中的数值问答任务，解决无标注情况下传统批判代理性能下降的问题。提出改进的批判代理与计算器代理，通过二者协同提升准确性和安全性，超越现有最优方法。**

- **链接: [https://arxiv.org/pdf/2506.08726v2](https://arxiv.org/pdf/2506.08726v2)**

> **作者:** Nelvin Tan; Zian Seng; Liang Zhang; Yu-Ching Shih; Dong Yang; Amol Salunkhe
>
> **备注:** 13 pages, 5 figures. Unlike the previous version, LLM names are now unmasked
>
> **摘要:** Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
>
---
#### [replaced 031] LaajMeter: A Framework for LaaJ Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源场景下大语言模型作为评判者（LaaJ）的评估难题，提出LaajMeter框架。通过模拟生成虚拟模型与评判者，系统评估不同评价指标对LaaJ质量的敏感性，帮助验证指标有效性并确定性能阈值，提升NLP评估的可信度与可复现性。**

- **链接: [https://arxiv.org/pdf/2508.10161v2](https://arxiv.org/pdf/2508.10161v2)**

> **作者:** Samuel Ackerman; Gal Amram; Ora Nova Fandina; Eitan Farchi; Shmulik Froimovich; Raviv Gal; Wesam Ibraheem; Avi Ziv
>
> **摘要:** Large Language Models (LLMs) are increasingly used as evaluators in natural language processing tasks, a paradigm known as LLM-as-a-Judge (LaaJ). The analysis of a LaaJ software, commonly refereed to as meta-evaluation, pose significant challenges in domain-specific contexts. In such domains, in contrast to general domains, annotated data is scarce and expert evaluation is costly. As a result, meta-evaluation is often performed using metrics that have not been validated for the specific domain in which they are applied. Therefore, it becomes difficult to determine which metrics effectively identify LaaJ quality, and further, what threshold indicates sufficient evaluator performance. In this work, we introduce LaaJMeter, a simulation-based framework for controlled meta-evaluation of LaaJs. LaaJMeter enables engineers to generate synthetic data representing virtual models and judges, allowing systematic analysis of evaluation metrics under realistic conditions. This helps practitioners validate LaaJs for specific tasks: they can test whether their metrics correctly distinguish between high and low quality (virtual) LaaJs, and estimate appropriate thresholds for evaluator adequacy. We demonstrate the utility of LaaJMeter in a code translation task involving a legacy programming language, showing how different metrics vary in sensitivity to evaluator quality. Our results highlight the limitations of common metrics and the importance of principled metric selection. LaaJMeter provides a scalable and extensible solution for assessing LaaJs in low-resource settings, contributing to the broader effort to ensure trustworthy and reproducible evaluation in NLP.
>
---
#### [replaced 032] SAS: Simulated Attention Score
- **分类: cs.CL**

- **简介: 该论文针对Transformer中注意力机制的效率与性能平衡问题，提出Simulated Attention Score（SAS）方法。通过低维表示投影模拟更多注意力头和更大特征维度，在不增加参数量的前提下提升模型容量与表达能力，并引入PEAA控制成本。实验表明其在多种任务上显著优于现有注意力变体。**

- **链接: [https://arxiv.org/pdf/2507.07694v2](https://arxiv.org/pdf/2507.07694v2)**

> **作者:** Chuanyang Zheng; Jiankai Sun; Yihang Gao; Yuehao Wang; Peihao Wang; Jing Xiong; Liliang Ren; Hao Cheng; Janardhan Kulkarni; Yelong Shen; Atlas Wang; Mac Schwager; Anderson Schneider; Xiaodong Liu; Jianfeng Gao
>
> **备注:** Tech Report
>
> **摘要:** The attention mechanism is a core component of the Transformer architecture. Various methods have been developed to compute attention scores, including multi-head attention (MHA), multi-query attention, group-query attention and so on. We further analyze the MHA and observe that its performance improves as the number of attention heads increases, provided the hidden size per head remains sufficiently large. Therefore, increasing both the head count and hidden size per head with minimal parameter overhead can lead to significant performance gains at a low cost. Motivated by this insight, we introduce Simulated Attention Score (SAS), which maintains a compact model size while simulating a larger number of attention heads and hidden feature dimension per head. This is achieved by projecting a low-dimensional head representation into a higher-dimensional space, effectively increasing attention capacity without increasing parameter count. Beyond the head representations, we further extend the simulation approach to feature dimension of the key and query embeddings, enhancing expressiveness by mimicking the behavior of a larger model while preserving the original model size. To control the parameter cost, we also propose Parameter-Efficient Attention Aggregation (PEAA). Comprehensive experiments on a variety of datasets and tasks demonstrate the effectiveness of the proposed SAS method, achieving significant improvements over different attention variants.
>
---
#### [replaced 033] CNS-Obsidian: A Neurosurgical Vision-Language Model Built From Scientific Publications
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出CNS-Obsidian，一个基于同行评审文献训练的神经外科视觉语言模型，旨在解决通用大模型在高风险医疗决策中因训练数据不透明带来的可信度问题。研究通过构建高质量医学图像-文本数据集，训练出小型但高效的专业模型，在真实临床场景中验证其诊断辅助能力，证明领域专用模型可实现与大型模型相当的性能，且更具透明性与可解释性。**

- **链接: [https://arxiv.org/pdf/2502.19546v5](https://arxiv.org/pdf/2502.19546v5)**

> **作者:** Anton Alyakin; Jaden Stryker; Daniel Alexander Alber; Jin Vivian Lee; Karl L. Sangwon; Brandon Duderstadt; Akshay Save; David Kurland; Spencer Frome; Shrutika Singh; Jeff Zhang; Eunice Yang; Ki Yun Park; Cordelia Orillac; Aly A. Valliani; Sean Neifert; Albert Liu; Aneek Patel; Christopher Livia; Darryl Lau; Ilya Laufer; Peter A. Rozman; Eveline Teresa Hidalgo; Howard Riina; Rui Feng; Todd Hollon; Yindalon Aphinyanaphongs; John G. Golfinos; Laura Snyder; Eric Leuthardt; Douglas Kondziolka; Eric Karl Oermann
>
> **摘要:** General-purpose VLMs demonstrate impressive capabilities, but their opaque training on uncurated internet data poses critical limitations for high-stakes decision-making, such as in neurosurgery. We present CNS-Obsidian, a neurosurgical VLM trained on peer-reviewed literature, and demonstrate its clinical utility versus GPT-4o in a real-world setting. We compiled 23,984 articles from Neurosurgery Publications journals, yielding 78,853 figures and captions. Using GPT-4o and Claude Sonnet-3.5, we converted these into 263,064 training samples across three formats: instruction fine-tuning, multiple-choice questions, and differential diagnosis. We trained CNS-Obsidian, a fine-tune of the 34-billion parameter LLaVA-Next model. In a blinded, randomized trial at NYU Langone Health (Aug 30-Nov 30, 2024), neurosurgery consultations were assigned to either CNS-Obsidian or a HIPAA-compliant GPT-4o endpoint as diagnostic co-pilot after consultations. Primary outcomes were diagnostic helpfulness and accuracy, assessed via user ratings and presence of correct diagnosis within the VLM-provided differential. CNS-Obsidian matched GPT-4o on synthetic questions (76.13% vs 77.54%, p=0.235), but only achieved 46.81% accuracy on human-generated questions versus GPT-4o's 65.70% (p<10-15). In the randomized trial, 70 consultations were evaluated (32 CNS-Obsidian, 38 GPT-4o) from 959 total consults (7.3% utilization). CNS-Obsidian received positive ratings in 40.62% of cases versus 57.89% for GPT-4o (p=0.230). Both models included correct diagnosis in approximately 60% of cases (59.38% vs 65.79%, p=0.626). Domain-specific VLMs trained on curated scientific literature can approach frontier model performance despite being orders of magnitude smaller and less expensive to train. This establishes a transparent framework for scientific communities to build specialized AI models.
>
---
#### [replaced 034] AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs
- **分类: cs.CL**

- **简介: 该论文提出AraFinNews，首个大规模阿拉伯语金融新闻数据集，用于金融领域摘要任务。针对阿拉伯语金融文本摘要中准确性与专业性不足的问题，研究了领域自适应大模型的影响，验证了领域微调对提升数值处理与风格一致性的重要作用。**

- **链接: [https://arxiv.org/pdf/2511.01265v3](https://arxiv.org/pdf/2511.01265v3)**

> **作者:** Mo El-Haj; Paul Rayson
>
> **备注:** 9 pages
>
> **摘要:** We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning a decade of reporting from 2015 to 2025. Designed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a realistic benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we investigate the impact of domain specificity on abstractive summarisation of Arabic financial texts with large language models (LLMs). In particular, we evaluate transformer-based models: mT5, AraT5, and the domain-adapted FinAraT5 to examine how financial-domain pretraining influences accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more coherent summaries, especially in their handling of quantitative and entity-centric information. These findings highlight the importance of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-uk/AraFinNews.
>
---
#### [replaced 035] Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文针对大语言模型在长周期稀疏奖励任务中的探索-利用平衡问题，提出SPEAR方法。通过自模仿学习与渐进式熵调控，分阶段促进探索与利用，显著提升成功率，且开销极小，适用于各类代理强化学习任务。**

- **链接: [https://arxiv.org/pdf/2509.22601v3](https://arxiv.org/pdf/2509.22601v3)**

> **作者:** Yulei Qin; Xiaoyu Tan; Zhengbao He; Gang Li; Haojia Lin; Zongyi Li; Zihan Xu; Yuchen Shi; Siqi Cai; Renting Rui; Shaofei Cai; Yuzheng Cai; Xuan Zhang; Sheng Ye; Ke Li; Xing Sun
>
> **备注:** 45 pages, 14 figures
>
> **摘要:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent's own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL, where a replay buffer stores good experience for off-policy update, by gradually steering the policy entropy across stages. Specifically, the proposed curriculum scheduling harmonizes intrinsic reward shaping and self-imitation to 1) expedite exploration via frequent tool interactions at the beginning, and 2) strengthen exploitation of successful tactics upon convergence towards familiarity with the environment. We also combine bag-of-tricks of industrial RL optimizations for a strong baseline Dr.BoT to demonstrate our effectiveness. In ALFWorld and WebShop, SPEAR increases the success rates of GRPO/GiGPO/Dr.BoT by up to 16.1%/5.1%/8.6% and 20.7%/11.8%/13.9%, respectively. In AIME24 and AIME25, SPEAR boosts Dr.BoT by up to 3.8% and 6.1%, respectively. Such gains incur only 10%-25% extra theoretical complexity and negligible runtime overhead in practice, demonstrating the plug-and-play scalability of SPEAR.
>
---
#### [replaced 036] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出MMTU基准，涵盖25类真实世界表格任务，旨在全面评估模型在表格理解、推理与操作方面的专家级能力。针对现有评测任务单一、难以反映实际需求的问题，构建大规模多任务基准，揭示当前大模型在复杂表格任务上仍有显著提升空间。**

- **链接: [https://arxiv.org/pdf/2506.05587v3](https://arxiv.org/pdf/2506.05587v3)**

> **作者:** Junjie Xing; Yeye He; Mengyu Zhou; Haoyu Dong; Shi Han; Lingjiao Chen; Dongmei Zhang; Surajit Chaudhuri; H. V. Jagadish
>
> **备注:** Accepted at NeurIPS 2025; Code and data available at https://github.com/MMTU-Benchmark/MMTU and https://huggingface.co/datasets/MMTU-benchmark/MMTU
>
> **摘要:** Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area. In this work, we introduce MMTU, a large-scale benchmark with over 28K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI GPT-5 and DeepSeek R1 score only around 69\% and 57\% respectively, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at https://github.com/MMTU-Benchmark/MMTU and https://huggingface.co/datasets/MMTU-benchmark/MMTU.
>
---
#### [replaced 037] EHR-R1: A Reasoning-Enhanced Foundational Language Model for Electronic Health Record Analysis
- **分类: cs.CL**

- **简介: 该论文针对电子健康记录（EHR）分析中大模型推理能力不足的问题，构建了EHR-Ins大规模推理数据集，提出EHR-R1推理增强型语言模型，并设计EHR-Bench评估基准。通过多阶段训练，显著提升模型在42项EHR任务上的推理与预测性能，优于GPT-4o等先进模型。**

- **链接: [https://arxiv.org/pdf/2510.25628v2](https://arxiv.org/pdf/2510.25628v2)**

> **作者:** Yusheng Liao; Chaoyi Wu; Junwei Liu; Shuyang Jiang; Pengcheng Qiu; Haowen Wang; Yun Yue; Shuai Zhen; Jian Wang; Qianrui Fan; Jinjie Gu; Ya Zhang; Yanfeng Wang; Yu Wang; Weidi Xie
>
> **摘要:** Electronic Health Records (EHRs) contain rich yet complex information, and their automated analysis is critical for clinical decision-making. Despite recent advances of large language models (LLMs) in clinical workflows, their ability to analyze EHRs remains limited due to narrow task coverage and lack of EHR-oriented reasoning capabilities. This paper aims to bridge the gap, specifically, we present EHR-Ins, a large-scale, comprehensive EHR reasoning instruction dataset, comprising 300k high-quality reasoning cases and 4M non-reasoning cases across 42 distinct EHR tasks. Its core innovation is a thinking-graph-driven framework that enables to generate high-quality reasoning data at scale. Based on it, we develop EHR-R1, a series of reasoning-enhanced LLMs with up to 72B parameters tailored for EHR analysis. Through a multi-stage training paradigm, including domain adaptation, reasoning enhancement, and reinforcement learning, EHR-R1 systematically acquires domain knowledge and diverse reasoning capabilities, enabling accurate and robust EHR analysis. Lastly, we introduce EHR-Bench, a new benchmark curated from MIMIC-IV, spanning 42 tasks, to comprehensively assess reasoning and prediction across EHR scenarios. In experiments, we show that the resulting EHR-R1 consistently outperforms state-of-the-art commercial and open-source LLMs (including DeepSeek-V3 and GPT-4o), surpassing GPT-4o by over 30 points on MIMIC-Bench and achieving a 10\% higher zero-shot AUROC on EHRSHOT. Collectively, EHR-Ins, EHR-R1, and EHR-Bench have significantly advanced the development for more reliable and clinically relevant EHR analysis.
>
---
#### [replaced 038] ConfTuner: Training Large Language Models to Express Their Confidence Verbally
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ConfTuner，一种用于训练大语言模型口头表达置信度的新方法。针对现有模型过度自信的问题，通过引入基于分词的布里尔分数损失函数，实现无需真实置信度标签的高效微调，提升模型在多任务中的置信度校准能力，并促进自纠正与模型级联的性能提升。**

- **链接: [https://arxiv.org/pdf/2508.18847v2](https://arxiv.org/pdf/2508.18847v2)**

> **作者:** Yibo Li; Miao Xiong; Jiaying Wu; Bryan Hooi
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes domains such as science, law, and healthcare, where accurate expressions of uncertainty are essential for reliability and trust. However, current LLMs are often observed to generate incorrect answers with high confidence, a phenomenon known as "overconfidence". Recent efforts have focused on calibrating LLMs' verbalized confidence: i.e., their expressions of confidence in text form, such as "I am 80% confident that...". Existing approaches either rely on prompt engineering or fine-tuning with heuristically generated uncertainty estimates, both of which have limited effectiveness and generalizability. Motivated by the notion of proper scoring rules for calibration in classical machine learning models, we introduce ConfTuner, a simple and efficient fine-tuning method that introduces minimal overhead and does not require ground-truth confidence scores or proxy confidence estimates. ConfTuner relies on a new loss function, tokenized Brier score, which we theoretically prove to be a proper scoring rule, intuitively meaning that it "correctly incentivizes the model to report its true probability of being correct". ConfTuner improves calibration across diverse reasoning tasks and generalizes to black-box models such as GPT-4o. Our results further show that better-calibrated confidence enables downstream gains in self-correction and model cascade, advancing the development of trustworthy LLM systems. The code is available at https://github.com/liushiliushi/ConfTuner.
>
---
#### [replaced 039] Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training
- **分类: cs.CL**

- **简介: 该论文针对伊朗语医学问答中小型语言模型推理能力弱的问题，提出基于RLAIF与DPO的推理增强方法。通过构建包含正确与错误思维链的双语数据集，仅用少量数据训练出性能超越大型模型的伊朗语医学模型，验证了高效推理训练的有效性。**

- **链接: [https://arxiv.org/pdf/2510.20059v3](https://arxiv.org/pdf/2510.20059v3)**

> **作者:** Mehrdad Ghassabi; Sadra Hakim; Hamidreza Baradaran Kashani; Pedram Rostami
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Enhancing reasoning capabilities in small language models is critical for specialized applications such as medical question answering, particularly in underrepresented languages like Persian. In this study, we employ Reinforcement Learning with AI Feedback (RLAIF) and Direct preference optimization (DPO) to improve the reasoning skills of a general-purpose Persian language model. To achieve this, we translated a multiple-choice medical question-answering dataset into Persian and used RLAIF to generate rejected-preferred answer pairs, which are essential for DPO training. By prompting both teacher and student models to produce Chain-of-Thought (CoT) reasoning responses, we compiled a dataset containing correct and incorrect reasoning trajectories. This dataset, comprising 2 million tokens in preferred answers and 2.5 million tokens in rejected ones, was used to train a baseline model, significantly enhancing its medical reasoning capabilities in Persian. Remarkably, the resulting model outperformed its predecessor, gaokerena-V, which was trained on approximately 57 million tokens, despite leveraging a much smaller dataset. These results highlight the efficiency and effectiveness of reasoning-focused training approaches in developing domain-specific language models with limited data availability.
>
---
#### [replaced 040] MedS$^3$: Towards Medical Slow Thinking with Self-Evolved Soft Dual-sided Process Supervision
- **分类: cs.CL**

- **简介: 该论文针对医疗语言模型在临床推理中任务覆盖窄、中间步骤监督不足、依赖专有系统等问题，提出MedS³框架。通过自演化机制，利用蒙特卡洛树搜索生成可验证的推理轨迹，并结合软双过程奖励模型实现细粒度错误识别，显著提升小模型的推理能力与可信度。**

- **链接: [https://arxiv.org/pdf/2501.12051v4](https://arxiv.org/pdf/2501.12051v4)**

> **作者:** Shuyang Jiang; Yusheng Liao; Zhe Chen; Ya Zhang; Yanfeng Wang; Yu Wang
>
> **备注:** 20 pages;Accepted as a Main paper at AAAI26
>
> **摘要:** Medical language models face critical barriers to real-world clinical reasoning applications. However, mainstream efforts, which fall short in task coverage, lack fine-grained supervision for intermediate reasoning steps, and rely on proprietary systems, are still far from a versatile, credible and efficient language model for clinical reasoning usage. To this end, we propose MedS3, a self-evolving framework that imparts robust reasoning capabilities to small, deployable models. Starting with 8,000 curated instances sampled via a curriculum strategy across five medical domains and 16 datasets, we use a small base policy model to conduct Monte Carlo Tree Search (MCTS) for constructing rule-verifiable reasoning trajectories. Self-explored reasoning trajectories ranked by node values are used to bootstrap the policy model via reinforcement fine-tuning and preference learning. Moreover, we introduce a soft dual process reward model that incorporates value dynamics: steps that degrade node value are penalized, enabling fine-grained identification of reasoning errors even when the final answer is correct. Experiments on eleven benchmarks show that MedS3 outperforms the previous state-of-the-art medical model by +6.45 accuracy points and surpasses 32B-scale general-purpose reasoning models by +8.57 points. Additional empirical analysis further demonstrates that MedS3 achieves robust and faithful reasoning behavior.
>
---
#### [replaced 041] From Generation to Detection: A Multimodal Multi-Task Dataset for Benchmarking Health Misinformation
- **分类: cs.CL**

- **简介: 该论文针对健康谣言泛滥与生成式AI加剧信息虚假的问题，构建了多模态多任务数据集MM Health，包含真人与AI生成的3.47万篇图文新闻。通过可靠性、原创性及AI生成检测三项任务，揭示现有模型在识别真伪与来源上的不足，旨在推动健康领域跨模态谣言检测技术发展。**

- **链接: [https://arxiv.org/pdf/2505.18685v2](https://arxiv.org/pdf/2505.18685v2)**

> **作者:** Zhihao Zhang; Yiran Zhang; Xiyue Zhou; Liting Huang; Imran Razzak; Preslav Nakov; Usman Naseem
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Infodemics and health misinformation have significant negative impact on individuals and society, exacerbating confusion and increasing hesitancy in adopting recommended health measures. Recent advancements in generative AI, capable of producing realistic, human like text and images, have significantly accelerated the spread and expanded the reach of health misinformation, resulting in an alarming surge in its dissemination. To combat the infodemics, most existing work has focused on developing misinformation datasets from social media and fact checking platforms, but has faced limitations in topical coverage, inclusion of AI generation, and accessibility of raw content. To address these issues, we present MM Health, a large scale multimodal misinformation dataset in the health domain consisting of 34,746 news article encompassing both textual and visual information. MM Health includes human-generated multimodal information (5,776 articles) and AI generated multimodal information (28,880 articles) from various SOTA generative AI models. Additionally, We benchmarked our dataset against three tasks (reliability checks, originality checks, and fine-grained AI detection) demonstrating that existing SOTA models struggle to accurately distinguish the reliability and origin of information. Our dataset aims to support the development of misinformation detection across various health scenarios, facilitating the detection of human and machine generated content at multimodal levels.
>
---
#### [replaced 042] MindEval: Benchmarking Language Models on Multi-turn Mental Health Support
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MindEval框架，用于评估大模型在多轮心理健康支持对话中的表现。针对现有基准无法模拟真实治疗互动的缺陷，通过心理学专家协作设计自动化评估体系，验证了模拟患者的真实性与评价可靠性，并评测12个主流模型，发现其普遍表现不佳，且长对话与严重症状下性能下降。**

- **链接: [https://arxiv.org/pdf/2511.18491v2](https://arxiv.org/pdf/2511.18491v2)**

> **作者:** José Pombal; Maya D'Eon; Nuno M. Guerreiro; Pedro Henrique Martins; António Farinhas; Ricardo Rei
>
> **摘要:** Demand for mental health support through AI chatbots is surging, though current systems present several limitations, like sycophancy or overvalidation, and reinforcement of maladaptive beliefs. A core obstacle to the creation of better systems is the scarcity of benchmarks that capture the complexity of real therapeutic interactions. Most existing benchmarks either only test clinical knowledge through multiple-choice questions or assess single responses in isolation. To bridge this gap, we present MindEval, a framework designed in collaboration with Ph.D-level Licensed Clinical Psychologists for automatically evaluating language models in realistic, multi-turn mental health therapy conversations. Through patient simulation and automatic evaluation with LLMs, our framework balances resistance to gaming with reproducibility via its fully automated, model-agnostic design. We begin by quantitatively validating the realism of our simulated patients against human-generated text and by demonstrating strong correlations between automatic and human expert judgments. Then, we evaluate 12 state-of-the-art LLMs and show that all models struggle, scoring below 4 out of 6, on average, with particular weaknesses in problematic AI-specific patterns of communication. Notably, reasoning capabilities and model scale do not guarantee better performance, and systems deteriorate with longer interactions or when supporting patients with severe symptoms. We release all code, prompts, and human evaluation data.
>
---
#### [replaced 043] The magnitude of categories of texts enriched by language models
- **分类: math.CT; cs.CL**

- **简介: 该论文研究语言模型生成文本的数学结构，提出基于下一词概率的文本类别度量方法，计算其广义度量空间的大小（magnitude）与莫比乌斯函数。通过将大小函数视为配分函数，揭示其与香农熵和泰尔斯熵的联系，并用上同调理论刻画其拓扑性质，解决文本生成概率建模与信息度量问题。**

- **链接: [https://arxiv.org/pdf/2501.06662v2](https://arxiv.org/pdf/2501.06662v2)**

> **作者:** Tai-Danae Bradley; Juan Pablo Vigneaux
>
> **备注:** 26 pages
>
> **摘要:** The purpose of this article is twofold. Firstly, we use the next-token probabilities given by a language model to explicitly define a category of texts in natural language enriched over the unit interval, in the sense of Bradley, Terilla, and Vlassopoulos. We consider explicitly the terminating conditions for text generation and determine when the enrichment itself can be interpreted as a probability over texts. Secondly, we compute the Möbius function and the magnitude of an associated generalized metric space of texts. The magnitude function of that space is a sum over texts (prompts) of the $t$-logarithmic (Tsallis) entropies of the next-token probability distributions associated with each prompt, plus the cardinality of the model's possible outputs. A suitable evaluation of the magnitude function's derivative recovers a sum of Shannon entropies, which justifies seeing magnitude as a partition function. Following Leinster and Shulman, we also express the magnitude function of the generalized metric space as an Euler characteristic of magnitude homology and provide an explicit description of the zeroeth and first magnitude homology groups.
>
---

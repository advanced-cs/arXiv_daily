# 自然语言处理 cs.CL

- **最新发布 62 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] EICAP: Deep Dive in Assessment and Enhancement of Large Language Models in Emotional Intelligence through Multi-Turn Conversations
- **分类: cs.CL; cs.HC**

- **简介: 论文提出四层情感智能框架（情感跟踪、因果推断、评估、回应生成），构建EICAP-Bench多轮对话基准评估开源LLMs情感能力，通过LoRA微调发现仅评估层显著提升，揭示现有预训练与指令微调方法在情感推理上的局限性，强调需针对性数据与建模策略。**

- **链接: [http://arxiv.org/pdf/2508.06196v1](http://arxiv.org/pdf/2508.06196v1)**

> **作者:** Nizi Nazar; Ehsaneddin Asgari
>
> **摘要:** Emotional Intelligence (EI) is a critical yet underexplored dimension in the development of human-aligned LLMs. To address this gap, we introduce a unified, psychologically grounded four-layer taxonomy of EI tailored for large language models (LLMs), encompassing emotional tracking, cause inference, appraisal, and emotionally appropriate response generation. Building on this framework, we present EICAP-Bench, a novel MCQ style multi-turn benchmark designed to evaluate EI capabilities in open-source LLMs across diverse linguistic and cultural contexts. We evaluate six LLMs: LLaMA3 (8B), LLaMA3-Instruct, Gemma (9B), Gemma-Instruct, Qwen2.5 (7B), and Qwen2.5-Instruct on EmoCap-Bench, identifying Qwen2.5-Instruct as the strongest baseline. To assess the potential for enhancing EI capabilities, we fine-tune both Qwen2.5-Base and Qwen2.5-Instruct using LoRA adapters on UltraChat (UC), a large-scale, instruction-tuned dialogue dataset, in both English and Arabic. Our statistical analysis reveals that among the five EI layers, only the Appraisal layer shows significant improvement through UC-based fine-tuning. These findings highlight the limitations of existing pretraining and instruction-tuning paradigms in equipping LLMs with deeper emotional reasoning and underscore the need for targeted data and modeling strategies for comprehensive EI alignment.
>
---
#### [new 002] Post-training for Efficient Communication via Convention Formation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出通过后训练和目标微调提升LLM在形成惯例以提高多轮对话效率的能力，设计两个新基准进行验证。**

- **链接: [http://arxiv.org/pdf/2508.06482v1](http://arxiv.org/pdf/2508.06482v1)**

> **作者:** Yilun Hua; Evan Wang; Yoav Artzi
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Humans communicate with increasing efficiency in multi-turn interactions, by adapting their language and forming ad-hoc conventions. In contrast, prior work shows that LLMs do not naturally show this behavior. We develop a post-training process to develop this ability through targeted fine-tuning on heuristically identified demonstrations of convention formation. We evaluate with two new benchmarks focused on this capability. First, we design a focused, cognitively-motivated interaction benchmark that consistently elicits strong convention formation trends in humans. Second, we create a new document-grounded reference completion task that reflects in-the-wild convention formation behavior. Our studies show significantly improved convention formation abilities in post-trained LLMs across the two evaluation methods.
>
---
#### [new 003] EvolvR: Self-Evolving Pairwise Reasoning for Story Evaluation to Enhance Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出EvolvR框架，通过多角色Chain-of-Thought数据生成与自我过滤，解决开放任务中故事评价准确性不足问题，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2508.06046v1](http://arxiv.org/pdf/2508.06046v1)**

> **作者:** Xinda Wang; Zhengxu Hou; Yangshijie Zhang; Bingren Yan; Zhibo Yang; Xingsheng Zhang; Luxi Xing; Qiang Zhou; Chen Zhang
>
> **摘要:** Although the effectiveness of Large Language Models (LLMs) as judges (LLM-as-a-judge) has been validated, their performance remains limited in open-ended tasks, particularly in story evaluation. Accurate story evaluation is crucial not only for assisting human quality judgment but also for providing key signals to guide story generation. However, existing methods face a dilemma: prompt engineering for closed-source models suffers from poor adaptability, while fine-tuning approaches for open-source models lack the rigorous reasoning capabilities essential for story evaluation. To address this, we propose the Self-Evolving Pairwise Reasoning (EvolvR) framework. Grounded in pairwise comparison, the framework first self-synthesizes score-aligned Chain-of-Thought (CoT) data via a multi-persona strategy. To ensure data quality, these raw CoTs undergo a self-filtering process, utilizing multi-agents to guarantee their logical rigor and robustness. Finally, the evaluator trained on the refined data is deployed as a reward model to guide the story generation task. Experimental results demonstrate that our framework achieves state-of-the-art (SOTA) performance on three evaluation benchmarks including StoryER, HANNA and OpenMEVA. Furthermore, when served as a reward model, it significantly enhances the quality of generated stories, thereby fully validating the superiority of our self-evolving approach.
>
---
#### [new 004] DKG-LLM : A Framework for Medical Diagnosis and Personalized Treatment Recommendations via Dynamic Knowledge Graph and Large Language Model Integration
- **分类: cs.CL**

- **简介: 论文提出DKG-LLM框架，通过动态知识图谱与大语言模型整合，解决复杂医疗数据处理与个性化治疗推荐问题，实现84.19%诊断准确率和89.63%治疗推荐精度。**

- **链接: [http://arxiv.org/pdf/2508.06186v1](http://arxiv.org/pdf/2508.06186v1)**

> **作者:** Ali Sarabadani; Maryam Abdollahi Shamami; Hamidreza Sadeghsalehi; Borhan Asadi; Saba Hesaraki
>
> **摘要:** Large Language Models (LLMs) have grown exponentially since the release of ChatGPT. These models have gained attention due to their robust performance on various tasks, including language processing tasks. These models achieve understanding and comprehension of tasks by training billions of parameters. The development of these models is a transformative force in enhancing natural language understanding and has taken a significant step towards artificial general intelligence (AGI). In this study, we aim to present the DKG-LLM framework. The DKG-LLM framework introduces a groundbreaking approach to medical diagnosis and personalized treatment recommendations by integrating a dynamic knowledge graph (DKG) with the Grok 3 large language model. Using the Adaptive Semantic Fusion Algorithm (ASFA), heterogeneous medical data (including clinical reports and PubMed articles) and patient records dynamically generate a knowledge graph consisting of 15,964 nodes in 13 distinct types (e.g., diseases, symptoms, treatments, patient profiles) and 127,392 edges in 26 relationship types (e.g., causal, therapeutic, association). ASFA utilizes advanced probabilistic models, Bayesian inference, and graph optimization to extract semantic information, dynamically updating the graph with approximately 150 new nodes and edges in each data category while maintaining scalability with up to 987,654 edges. Real-world datasets, including MIMIC-III and PubMed, were utilized to evaluate the proposed architecture. The evaluation results show that DKG-LLM achieves a diagnostic accuracy of 84.19%. The model also has a treatment recommendation accuracy of 89.63% and a semantic coverage of 93.48%. DKG-LLM is a reliable and transformative tool that handles noisy data and complex multi-symptom diseases, along with feedback-based learning from physician input.
>
---
#### [new 005] FineDialFact: A benchmark for Fine-grained Dialogue Fact Verification
- **分类: cs.CL**

- **简介: 论文提出FineDialFact基准用于细粒度对话事实验证，解决现有方法标签粗粒度的问题，通过构建公开对话数据集并评估多种方法，发现需结合Chain-of-Thought提升性能，但HybriDialogue上F1-score仅0.75，凸显挑战性。**

- **链接: [http://arxiv.org/pdf/2508.05782v1](http://arxiv.org/pdf/2508.05782v1)**

> **作者:** Xiangyan Chen; Yufeng Li; Yujian Gan; Arkaitz Zubiaga; Matthew Purver
>
> **摘要:** Large Language Models (LLMs) are known to produce hallucinations - factually incorrect or fabricated information - which poses significant challenges for many Natural Language Processing (NLP) applications, such as dialogue systems. As a result, detecting hallucinations has become a critical area of research. Current approaches to hallucination detection in dialogue systems primarily focus on verifying the factual consistency of generated responses. However, these responses often contain a mix of accurate, inaccurate or unverifiable facts, making one factual label overly simplistic and coarse-grained. In this paper, we introduce a benchmark, FineDialFact, for fine-grained dialogue fact verification, which involves verifying atomic facts extracted from dialogue responses. To support this, we construct a dataset based on publicly available dialogue datasets and evaluate it using various baseline methods. Experimental results demonstrate that methods incorporating Chain-of-Thought (CoT) reasoning can enhance performance in dialogue fact verification. Despite this, the best F1-score achieved on the HybriDialogue, an open-domain dialogue dataset, is only 0.75, indicating that the benchmark remains a challenging task for future research. Our dataset and code will be public on GitHub.
>
---
#### [new 006] Harnessing Adaptive Topology Representations for Zero-Shot Graph Question Answering
- **分类: cs.CL; cs.AI; cs.GR; cs.LG**

- **简介: 论文提出针对零样本图问答任务，设计适应性图表示（F_ZS）与动态路由框架，解决现有单一图表示方法难以适配不同模型与任务的问题，通过GRE指标优化响应效率，提升LMMs在零样本图QA的准确性。**

- **链接: [http://arxiv.org/pdf/2508.06345v1](http://arxiv.org/pdf/2508.06345v1)**

> **作者:** Yanbin Wei; Jiangyue Yan; Chun Kang; Yang Chen; Hua Liu; James T. Kwok; Yu Zhang
>
> **摘要:** Large Multimodal Models (LMMs) have shown generalized zero-shot capabilities in diverse domain question-answering (QA) tasks, including graph QA that involves complex graph topologies. However, most current approaches use only a single type of graph representation, namely Topology Representation Form (TRF), such as prompt-unified text descriptions or style-fixed visual styles. Those "one-size-fits-all" approaches fail to consider the specific preferences of different models or tasks, often leading to incorrect or overly long responses. To address this, we first analyze the characteristics and weaknesses of existing TRFs, and then design a set of TRFs, denoted by $F_{ZS}$, tailored to zero-shot graph QA. We then introduce a new metric, Graph Response Efficiency (GRE), which measures the balance between the performance and the brevity in graph QA. Built on these, we develop the DynamicTRF framework, which aims to improve both the accuracy and conciseness of graph QA. To be specific, DynamicTRF first creates a TRF Preference (TRFP) dataset that ranks TRFs based on their GRE scores, to probe the question-specific TRF preferences. Then it trains a TRF router on the TRFP dataset, to adaptively assign the best TRF from $F_{ZS}$ for each question during the inference. Extensive experiments across 7 in-domain algorithmic graph QA tasks and 2 out-of-domain downstream tasks show that DynamicTRF significantly enhances the zero-shot graph QA of LMMs in terms of accuracy
>
---
#### [new 007] Discovering Properties of Inflectional Morphology in Neural Emergent Communication
- **分类: cs.CL**

- **简介: 论文提出通过引入小词汇约束模拟双音节，构建类似自然语言屈折形态的神经网络通信模型，探索连续性与融合性属性，揭示新兴通信中屈折形态的涌现规律。**

- **链接: [http://arxiv.org/pdf/2508.05843v1](http://arxiv.org/pdf/2508.05843v1)**

> **作者:** Miles Gilberti; Shane Storks; Huteng Dai
>
> **摘要:** Emergent communication (EmCom) with deep neural network-based agents promises to yield insights into the nature of human language, but remains focused primarily on a few subfield-specific goals and metrics that prioritize communication schemes which represent attributes with unique characters one-to-one and compose them syntactically. We thus reinterpret a common EmCom setting, the attribute-value reconstruction game, by imposing a small-vocabulary constraint to simulate double articulation, and formulating a novel setting analogous to naturalistic inflectional morphology (enabling meaningful comparison to natural language communication schemes). We develop new metrics and explore variations of this game motivated by real properties of inflectional morphology: concatenativity and fusionality. Through our experiments, we discover that simulated phonological constraints encourage concatenative morphology, and emergent languages replicate the tendency of natural languages to fuse grammatical attributes.
>
---
#### [new 008] LLMs vs. Chinese Anime Enthusiasts: A Comparative Study on Emotionally Supportive Role-Playing
- **分类: cs.CL**

- **简介: 论文比较LLMs与中文动漫爱好者在情感支持性角色扮演中的表现，构建ChatAnime数据集，设计评估体系，揭示LLMs在角色扮演与情感支持上超越人类，但人类在响应多样性上占优。**

- **链接: [http://arxiv.org/pdf/2508.06388v1](http://arxiv.org/pdf/2508.06388v1)**

> **作者:** Lanlan Qiu; Xiao Pu; Yeqi Feng; Tianxing He
>
> **备注:** 21 pages, 17 figures, 3 tables
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing conversations and providing emotional support as separate research directions. However, there remains a significant research gap in combining these capabilities to enable emotionally supportive interactions with virtual characters. To address this research gap, we focus on anime characters as a case study because of their well-defined personalities and large fan bases. This choice enables us to effectively evaluate how well LLMs can provide emotional support while maintaining specific character traits. We introduce ChatAnime, the first Emotionally Supportive Role-Playing (ESRP) dataset. We first thoughtfully select 20 top-tier characters from popular anime communities and design 60 emotion-centric real-world scenario questions. Then, we execute a nationwide selection process to identify 40 Chinese anime enthusiasts with profound knowledge of specific characters and extensive experience in role-playing. Next, we systematically collect two rounds of dialogue data from 10 LLMs and these 40 Chinese anime enthusiasts. To evaluate the ESRP performance of LLMs, we design a user experience-oriented evaluation system featuring 9 fine-grained metrics across three dimensions: basic dialogue, role-playing and emotional support, along with an overall metric for response diversity. In total, the dataset comprises 2,400 human-written and 24,000 LLM-generated answers, supported by over 132,000 human annotations. Experimental results show that top-performing LLMs surpass human fans in role-playing and emotional support, while humans still lead in response diversity. We hope this work can provide valuable resources and insights for future research on optimizing LLMs in ESRP. Our datasets are available at https://github.com/LanlanQiu/ChatAnime.
>
---
#### [new 009] Pragmatics beyond humans: meaning, communication, and LLMs
- **分类: cs.CL; cs.HC**

- **简介: 论文重构语用学，挑战传统理论，提出HMC框架，分析LLMs的替换论与上下文挫败问题。**

- **链接: [http://arxiv.org/pdf/2508.06167v1](http://arxiv.org/pdf/2508.06167v1)**

> **作者:** Vít Gvoždiak
>
> **摘要:** The paper reconceptualizes pragmatics not as a subordinate, third dimension of meaning, but as a dynamic interface through which language operates as a socially embedded tool for action. With the emergence of large language models (LLMs) in communicative contexts, this understanding needs to be further refined and methodologically reconsidered. The first section challenges the traditional semiotic trichotomy, arguing that connectionist LLM architectures destabilize established hierarchies of meaning, and proposes the Human-Machine Communication (HMC) framework as a more suitable alternative. The second section examines the tension between human-centred pragmatic theories and the machine-centred nature of LLMs. While traditional, Gricean-inspired pragmatics continue to dominate, it relies on human-specific assumptions ill-suited to predictive systems like LLMs. Probabilistic pragmatics, particularly the Rational Speech Act framework, offers a more compatible teleology by focusing on optimization rather than truth-evaluation. The third section addresses the issue of substitutionalism in three forms - generalizing, linguistic, and communicative - highlighting the anthropomorphic biases that distort LLM evaluation and obscure the role of human communicative subjects. Finally, the paper introduces the concept of context frustration to describe the paradox of increased contextual input paired with a collapse in contextual understanding, emphasizing how users are compelled to co-construct pragmatic conditions both for the model and themselves. These arguments suggest that pragmatic theory may need to be adjusted or expanded to better account for communication involving generative AI.
>
---
#### [new 010] Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SRD框架，通过动态筛选高质量训练数据和课程学习策略，解决知识蒸馏中数据质量与模型兼容性问题，提升压缩模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.06135v1](http://arxiv.org/pdf/2508.06135v1)**

> **作者:** Lingyuan Liu; Mengxiang Zhang
>
> **摘要:** Knowledge Distillation (KD) is a fundamental technique for compressing large language models (LLMs) into compact, efficient student models. However, existing white-box KD methods mainly focus on balancing ground truth and student-generated responses while overlooking two critical factors: training data quality and student-model compatibility. To address these limitations, we propose Selective Reflection Distillation (SRD), a novel data curation framework that leverages reflections from student models to systematically refine training data. SRD dynamically evaluates and selects prompt-response pairs by comparing ground truth data with student model outputs, selectively curating high-quality, student-compatible training instances through automated ranking based on difficulty. Furthermore, after selecting the training data, a curriculum scheduling strategy is employed to incrementally introduce these curated subsets into the distillation process at fixed intervals. As a plug-and-play enhancement, SRD consistently improves distillation outcomes across diverse white-box KD approaches and model architectures, as well as decreases computational cost significantly during KD training. Experiments on a range of language model benchmarks demonstrate SRD's consistent improvements in distilled model performance, as well as a reduction in training runtime by up to 39%, under diverse KD methods and model families. Notably, SRD operates as a plug-and-play module, enhancing sample efficiency without modifying underlying KD algorithms. Our findings highlight that data quality and compatibility are pivotal to effective and efficient distillation of LLMs, and SRD provides a principled framework to achieve both. This work advances the understanding of data-centric factors in KD and offers practical insights for enhancing the capability and efficiency of compressed LLMs.
>
---
#### [new 011] Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation
- **分类: cs.CL; cs.CY**

- **简介: 本文综述LLMs有害内容生成与安全缓解，提出统一伤害分类及防御策略，分析攻击与缓解方法，总结未来方向。**

- **链接: [http://arxiv.org/pdf/2508.05775v1](http://arxiv.org/pdf/2508.05775v1)**

> **作者:** Chi Zhang; Changjia Zhu; Junjie Xiong; Xiaoran Xu; Lingyao Li; Yao Liu; Zhuo Lu
>
> **摘要:** Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.
>
---
#### [new 012] UR$^2$: Unify RAG and Reasoning through Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出UR2框架，通过强化学习统一RAG与推理，解决孤立发展问题，提升适应性，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06165v1](http://arxiv.org/pdf/2508.06165v1)**

> **作者:** Weitao Li; Boran Xiang; Xiaolong Wang; Zhinan Gou; Weizhi Ma; Yang Liu
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG), which enhances knowledge grounding, and Reinforcement Learning from Verifiable Rewards (RLVR), which optimizes complex reasoning abilities. However, these two capabilities are often developed in isolation, and existing efforts to unify them remain narrow in scope-typically limited to open-domain QA with fixed retrieval settings and task-specific assumptions. This lack of integration constrains generalization and limits the applicability of RAG-RL methods to broader domains. To bridge this gap, we propose UR2 (Unified RAG and Reasoning), a general framework that unifies retrieval and reasoning through reinforcement learning. UR2 introduces two key contributions: a difficulty-aware curriculum training that selectively invokes retrieval only for challenging problems, and a hybrid knowledge access strategy combining domain-specific offline corpora with LLM-generated summaries. These components are designed to enable dynamic coordination between retrieval and reasoning, improving adaptability across a diverse range of tasks. Experiments across open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks demonstrate that UR2 (built on Qwen2.5-3/7B and LLaMA-3.1-8B) significantly outperforms existing RAG and RL methods, achieving comparable performance to GPT-4o-mini and GPT-4.1-mini on several benchmarks. We have released all code, models, and data at https://github.com/Tsinghua-dhy/UR2.
>
---
#### [new 013] Classification is a RAG problem: A case study on hate speech detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出基于RAG的分类方法用于仇恨言论检测，通过动态政策引擎（CPE）实现无需重新训练的灵活分类，提升准确率与可解释性，解决传统分类需昂贵重训的问题。**

- **链接: [http://arxiv.org/pdf/2508.06204v1](http://arxiv.org/pdf/2508.06204v1)**

> **作者:** Richard Willats; Josh Pennington; Aravind Mohan; Bertie Vidgen
>
> **摘要:** Robust content moderation requires classification systems that can quickly adapt to evolving policies without costly retraining. We present classification using Retrieval-Augmented Generation (RAG), which shifts traditional classification tasks from determining the correct category in accordance with pre-trained parameters to evaluating content in relation to contextual knowledge retrieved at inference. In hate speech detection, this transforms the task from "is this hate speech?" to "does this violate the hate speech policy?" Our Contextual Policy Engine (CPE) - an agentic RAG system - demonstrates this approach and offers three key advantages: (1) robust classification accuracy comparable to leading commercial systems, (2) inherent explainability via retrieved policy segments, and (3) dynamic policy updates without model retraining. Through three experiments, we demonstrate strong baseline performance and show that the system can apply fine-grained policy control by correctly adjusting protection for specific identity groups without requiring retraining or compromising overall performance. These findings establish that RAG can transform classification into a more flexible, transparent, and adaptable process for content moderation and wider classification problems.
>
---
#### [new 014] Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLMs在跨语言移民讨论分类任务中的能力，解决是否可通过有限微调实现跨语言主题检测及偏差修正问题，通过多语微调和轻量模型验证了少量语言暴露的有效性。**

- **链接: [http://arxiv.org/pdf/2508.06435v1](http://arxiv.org/pdf/2508.06435v1)**

> **作者:** Andrea Nasuto; Stefano Maria Iacus; Francisco Rowe; Devika Jain
>
> **摘要:** Large language models (LLMs) are transforming social-science research by enabling scalable, precise analysis. Their adaptability raises the question of whether knowledge acquired through fine-tuning in a few languages can transfer to unseen languages that only appeared during pre-training. To examine this, we fine-tune lightweight LLaMA 3.2-3B models on monolingual, bilingual, or multilingual data sets to classify immigration-related tweets from X/Twitter across 13 languages, a domain characterised by polarised, culturally specific discourse. We evaluate whether minimal language-specific fine-tuning enables cross-lingual topic detection and whether adding targeted languages corrects pre-training biases. Results show that LLMs fine-tuned in one or two languages can reliably classify immigration-related content in unseen languages. However, identifying whether a tweet expresses a pro- or anti-immigration stance benefits from multilingual fine-tuning. Pre-training bias favours dominant languages, but even minimal exposure to under-represented languages during fine-tuning (as little as $9.62\times10^{-11}$ of the original pre-training token volume) yields significant gains. These findings challenge the assumption that cross-lingual mastery requires extensive multilingual training: limited language coverage suffices for topic-level generalisation, and structural biases can be corrected with lightweight interventions. By releasing 4-bit-quantised, LoRA fine-tuned models, we provide an open-source, reproducible alternative to proprietary LLMs that delivers 35 times faster inference at just 0.00000989% of the dollar cost of the OpenAI GPT-4o model, enabling scalable, inclusive research.
>
---
#### [new 015] "Mirror" Language AI Models of Depression are Criterion-Contaminated
- **分类: cs.CL; cs.CY**

- **简介: 论文探讨Mirror语言模型在抑郁症预测中的Criterion Contamination问题，通过对比Non-Mirror模型，发现其效果夸大且泛化性差，建议使用Non-Mirror模型获取更可靠的特征。**

- **链接: [http://arxiv.org/pdf/2508.05830v1](http://arxiv.org/pdf/2508.05830v1)**

> **作者:** Tong Li; Rasiq Hussain; Mehak Gupta; Joshua R. Oltmanns
>
> **备注:** 39 pages, 9 figures
>
> **摘要:** A growing number of studies show near-perfect LLM language-based prediction of depression assessment scores (up to R2 of .70). However, many develop these models directly from language responses to depression assessments. These "Mirror models" suffer from "criterion contamination", which arises when a predicted score depends in part on the predictors themselves. This causes artificial effect size inflation which reduces model generalizability. The present study compares the performance of Mirror models versus "Non-Mirror models", which are developed from language that does not mirror the assessment they are developed to predict. N = 110 research participants completed two different interviews: structured diagnostic and life history interviews. GPT-4, GPT-4o and LLaMA3-70B were then prompted to predict structured diagnostic interview depression scores from the two transcripts separately. Mirror models (using structured diagnostic data) showed very large effect sizes (e.g., R2 = .80). As expected, NonMirror models (using life history data) demonstrated smaller effect sizes, but were relatively large (e.g., R2 = .27). When Mirror and Non-Mirror model-predicted structured interview depression scores were correlated with self-reported depression symptoms, Mirror and NonMirror performed the same (e.g., r = ~.54), indicating that Mirror models contain bias perhaps due to criterion contamination. Topic modeling identified clusters across Mirror and Non-Mirror models, as well as between true-positive and false-positive predictions. In this head-to-head comparison study, Mirror language AI models of depression showed artificially inflated effect sizes and less generalizability. As language AI models for depression continue to evolve, incorporating Non-Mirror models may identify interpretable, and generalizable semantic features that have unique utility in real-world psychological assessment.
>
---
#### [new 016] You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures
- **分类: cs.CL**

- **简介: 论文提出LogicRAG框架，解决传统RAG依赖预设图导致的高成本问题，通过动态构建推理结构实现高效检索与生成。**

- **链接: [http://arxiv.org/pdf/2508.06105v1](http://arxiv.org/pdf/2508.06105v1)**

> **作者:** Shengyuan Chen; Chuang Zhou; Zheng Yuan; Qinggang Zhang; Zeyang Cui; Hao Chen; Yilin Xiao; Jiannong Cao; Xiao Huang
>
> **摘要:** Large language models (LLMs) often suffer from hallucination, generating factually incorrect statements when handling questions beyond their knowledge and perception. Retrieval-augmented generation (RAG) addresses this by retrieving query-relevant contexts from knowledge bases to support LLM reasoning. Recent advances leverage pre-constructed graphs to capture the relational connections among distributed documents, showing remarkable performance in complex tasks. However, existing Graph-based RAG (GraphRAG) methods rely on a costly process to transform the corpus into a graph, introducing overwhelming token cost and update latency. Moreover, real-world queries vary in type and complexity, requiring different logic structures for accurate reasoning. The pre-built graph may not align with these required structures, resulting in ineffective knowledge retrieval. To this end, we propose a \textbf{\underline{Logic}}-aware \textbf{\underline{R}}etrieval-\textbf{\underline{A}}ugmented \textbf{\underline{G}}eneration framework (\textbf{LogicRAG}) that dynamically extracts reasoning structures at inference time to guide adaptive retrieval without any pre-built graph. LogicRAG begins by decomposing the input query into a set of subproblems and constructing a directed acyclic graph (DAG) to model the logical dependencies among them. To support coherent multi-step reasoning, LogicRAG then linearizes the graph using topological sort, so that subproblems can be addressed in a logically consistent order. Besides, LogicRAG applies graph pruning to reduce redundant retrieval and uses context pruning to filter irrelevant context, significantly reducing the overall token cost. Extensive experiments demonstrate that LogicRAG achieves both superior performance and efficiency compared to state-of-the-art baselines.
>
---
#### [new 017] GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
- **分类: cs.CL**

- **简介: 论文提出GLM-4.5开源混合专家模型，通过混合推理技术实现ARC任务高性能，参数355B，优于同类模型，发布紧凑版供研究。**

- **链接: [http://arxiv.org/pdf/2508.06471v1](http://arxiv.org/pdf/2508.06471v1)**

> **作者:** GLM-4. 5 Team; :; Aohan Zeng; Xin Lv; Qinkai Zheng; Zhenyu Hou; Bin Chen; Chengxing Xie; Cunxiang Wang; Da Yin; Hao Zeng; Jiajie Zhang; Kedong Wang; Lucen Zhong; Mingdao Liu; Rui Lu; Shulin Cao; Xiaohan Zhang; Xuancheng Huang; Yao Wei; Yean Cheng; Yifan An; Yilin Niu; Yuanhao Wen; Yushi Bai; Zhengxiao Du; Zihan Wang; Zilin Zhu; Bohan Zhang; Bosi Wen; Bowen Wu; Bowen Xu; Can Huang; Casey Zhao; Changpeng Cai; Chao Yu; Chen Li; Chendi Ge; Chenghua Huang; Chenhui Zhang; Chenxi Xu; Chenzheng Zhu; Chuang Li; Congfeng Yin; Daoyan Lin; Dayong Yang; Dazhi Jiang; Ding Ai; Erle Zhu; Fei Wang; Gengzheng Pan; Guo Wang; Hailong Sun; Haitao Li; Haiyang Li; Haiyi Hu; Hanyu Zhang; Hao Peng; Hao Tai; Haoke Zhang; Haoran Wang; Haoyu Yang; He Liu; He Zhao; Hongwei Liu; Hongxi Yan; Huan Liu; Huilong Chen; Ji Li; Jiajing Zhao; Jiamin Ren; Jian Jiao; Jiani Zhao; Jianyang Yan; Jiaqi Wang; Jiayi Gui; Jiayue Zhao; Jie Liu; Jijie Li; Jing Li; Jing Lu; Jingsen Wang; Jingwei Yuan; Jingxuan Li; Jingzhao Du; Jinhua Du; Jinxin Liu; Junkai Zhi; Junli Gao; Ke Wang; Lekang Yang; Liang Xu; Lin Fan; Lindong Wu; Lintao Ding; Lu Wang; Man Zhang; Minghao Li; Minghuan Xu; Mingming Zhao; Mingshu Zhai; Pengfan Du; Qian Dong; Shangde Lei; Shangqing Tu; Shangtong Yang; Shaoyou Lu; Shijie Li; Shuang Li; Shuang-Li; Shuxun Yang; Sibo Yi; Tianshu Yu; Wei Tian; Weihan Wang; Wenbo Yu; Weng Lam Tam; Wenjie Liang; Wentao Liu; Xiao Wang; Xiaohan Jia; Xiaotao Gu; Xiaoying Ling; Xin Wang; Xing Fan; Xingru Pan; Xinyuan Zhang; Xinze Zhang; Xiuqing Fu; Xunkai Zhang; Yabo Xu; Yandong Wu; Yida Lu; Yidong Wang; Yilin Zhou; Yiming Pan; Ying Zhang; Yingli Wang; Yingru Li; Yinpei Su; Yipeng Geng; Yitong Zhu; Yongkun Yang; Yuhang Li; Yuhao Wu; Yujiang Li; Yunan Liu; Yunqing Wang; Yuntao Li; Yuxuan Zhang; Zezhen Liu; Zhen Yang; Zhengda Zhou; Zhongpei Qiao; Zhuoer Feng; Zhuorui Liu; Zichen Zhang; Zihan Wang; Zijun Yao; Zikang Wang; Ziqiang Liu; Ziwei Chai; Zixuan Li; Zuodong Zhao; Wenguang Chen; Jidong Zhai; Bin Xu; Minlie Huang; Hongning Wang; Juanzi Li; Yuxiao Dong; Jie Tang
>
> **摘要:** We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model with 355B total parameters and 32B activated parameters, featuring a hybrid reasoning method that supports both thinking and direct response modes. Through multi-stage training on 23T tokens and comprehensive post-training with expert model iteration and reinforcement learning, GLM-4.5 achieves strong performance across agentic, reasoning, and coding (ARC) tasks, scoring 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified. With much fewer parameters than several competitors, GLM-4.5 ranks 3rd overall among all evaluated models and 2nd on agentic benchmarks. We release both GLM-4.5 (355B parameters) and a compact version, GLM-4.5-Air (106B parameters), to advance research in reasoning and agentic AI systems. Code, models, and more information are available at https://github.com/zai-org/GLM-4.5.
>
---
#### [new 018] HapticLLaMA: A Multimodal Sensory Language Model for Haptic Captioning
- **分类: cs.CL**

- **简介: 论文提出HapticLLaMA，通过多模态感知模型解决触觉信号描述问题，采用频率/EnCodec令牌化技术整合至LLaMA，结合RLHF提升人类评分，实现触觉信号到自然语言的高效转换。**

- **链接: [http://arxiv.org/pdf/2508.06475v1](http://arxiv.org/pdf/2508.06475v1)**

> **作者:** Guimin Hu; Daniel Hershcovich; Hasti Seifi
>
> **摘要:** Haptic captioning is the task of generating natural language descriptions from haptic signals, such as vibrations, for use in virtual reality, accessibility, and rehabilitation applications. While previous multimodal research has focused primarily on vision and audio, haptic signals for the sense of touch remain underexplored. To address this gap, we formalize the haptic captioning task and propose HapticLLaMA, a multimodal sensory language model that interprets vibration signals into descriptions in a given sensory, emotional, or associative category. We investigate two types of haptic tokenizers, a frequency-based tokenizer and an EnCodec-based tokenizer, that convert haptic signals into sequences of discrete units, enabling their integration with the LLaMA model. HapticLLaMA is trained in two stages: (1) supervised fine-tuning using the LLaMA architecture with LoRA-based adaptation, and (2) fine-tuning via reinforcement learning from human feedback (RLHF). We assess HapticLLaMA's captioning performance using both automated n-gram metrics and human evaluation. HapticLLaMA demonstrates strong capability in interpreting haptic vibration signals, achieving a METEOR score of 59.98 and a BLEU-4 score of 32.06 respectively. Additionally, over 61% of the generated captions received human ratings above 3.5 on a 7-point scale, with RLHF yielding a 10% improvement in the overall rating distribution, indicating stronger alignment with human haptic perception. These findings highlight the potential of large language models to process and adapt to sensory data.
>
---
#### [new 019] Human-like fleeting memory improves language learning but impairs reading time prediction in transformer language models
- **分类: cs.CL; I.2.7**

- **简介: 论文研究短暂记忆对Transformer语言模型语言学习与阅读时间预测的影响，发现短暂记忆提升语言学习但削弱阅读预测。**

- **链接: [http://arxiv.org/pdf/2508.05803v1](http://arxiv.org/pdf/2508.05803v1)**

> **作者:** Abishek Thamma; Micha Heilbron
>
> **摘要:** Human memory is fleeting. As words are processed, the exact wordforms that make up incoming sentences are rapidly lost. Cognitive scientists have long believed that this limitation of memory may, paradoxically, help in learning language - an idea supported by classic connectionist modelling work. The rise of Transformers appears to challenge this idea, as these models can learn language effectively, despite lacking memory limitations or other architectural recency biases. Here, we investigate the hypothesized benefit of fleeting memory for language learning in tightly controlled experiments on transformer language models. Training transformers with and without fleeting memory on a developmentally realistic training set, we find that fleeting memory consistently improves language learning (as quantified by both overall language modelling performance and targeted syntactic evaluation) but, unexpectedly, impairs surprisal-based prediction of human reading times. Interestingly, follow up analyses revealed that this discrepancy - better language modeling, yet worse reading time prediction - could not be accounted for by prior explanations of why better language models sometimes fit human reading time worse. Together, these results support a benefit of memory limitations on neural network language learning - but not on predicting behavior.
>
---
#### [new 020] One Size Does Not Fit All: A Distribution-Aware Sparsification for More Precise Model Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出基于分布的自适应稀疏化方法TADrop，用于解决模型合并中参数干扰问题，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.06163v1](http://arxiv.org/pdf/2508.06163v1)**

> **作者:** Yingfeng Luo; Dingyang Lin; Junxin Wang; Ziqiang Xu; Kaiyan Chang; Tong Zheng; Bei Li; Anxiang Ma; Tong Xiao; Zhengtao Yu; Jingbo Zhu
>
> **备注:** Under review
>
> **摘要:** Model merging has emerged as a compelling data-free paradigm for multi-task learning, enabling the fusion of multiple fine-tuned models into a single, powerful entity. A key technique in merging methods is sparsification, which prunes redundant parameters from task vectors to mitigate interference. However, prevailing approaches employ a ``one-size-fits-all'' strategy, applying a uniform sparsity ratio that overlooks the inherent structural and statistical heterogeneity of model parameters. This often leads to a suboptimal trade-off, where critical parameters are inadvertently pruned while less useful ones are retained. To address this limitation, we introduce \textbf{TADrop} (\textbf{T}ensor-wise \textbf{A}daptive \textbf{Drop}), an adaptive sparsification strategy that respects this heterogeneity. Instead of a global ratio, TADrop assigns a tailored sparsity level to each parameter tensor based on its distributional properties. The core intuition is that tensors with denser, more redundant distributions can be pruned aggressively, while sparser, more critical ones are preserved. As a simple and plug-and-play module, we validate TADrop by integrating it with foundational, classic, and SOTA merging methods. Extensive experiments across diverse tasks (vision, language, and multimodal) and models (ViT, BEiT) demonstrate that TADrop consistently and significantly boosts their performance. For instance, when enhancing a leading merging method, it achieves an average performance gain of 2.0\% across 8 ViT-B/32 tasks. TADrop provides a more effective way to mitigate parameter interference by tailoring sparsification to the model's structure, offering a new baseline for high-performance model merging.
>
---
#### [new 021] Memp: Exploring Agent Procedural Memory
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 论文提出Memp，通过将代理轨迹分解为细粒度步骤和高层抽象，构建可学习、更新的程序性记忆，动态维护知识库以提升任务性能，验证了其在TravelPlanner和ALFWorld中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.06433v1](http://arxiv.org/pdf/2508.06433v1)**

> **作者:** Runnan Fang; Yuan Liang; Xiaobin Wang; Jialong Wu; Shuofei Qiao; Pengjun Xie; Fei Huang; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) based agents excel at diverse tasks, yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work, we investigate strategies to endow agents with a learnable, updatable, and lifelong procedural memory. We propose Memp that distills past agent trajectories into both fine-grained, step-by-step instructions and higher-level, script-like abstractions, and explore the impact of different strategies for Build, Retrieval, and Update of procedural memory. Coupled with a dynamic regimen that continuously updates, corrects, and deprecates its contents, this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined, agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover, procedural memory built from a stronger model retains its value: migrating the procedural memory to a weaker model yields substantial performance gains.
>
---
#### [new 022] Large Language Model Data Generation for Enhanced Intent Recognition in German Speech
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 论文提出通过合成LLM数据提升德语语音意图识别，解决现有方法对短命令和英语的局限，采用适应Whisper和微调Transformer，结合合成数据测试，显示合成数据提升性能，LeoLM优于ChatGPT。**

- **链接: [http://arxiv.org/pdf/2508.06277v1](http://arxiv.org/pdf/2508.06277v1)**

> **作者:** Theresa Pekarek Rosin; Burak Can Kaplan; Stefan Wermter
>
> **备注:** 11 pages, 3 figures, accepted at KONVENS 2025
>
> **摘要:** Intent recognition (IR) for speech commands is essential for artificial intelligence (AI) assistant systems; however, most existing approaches are limited to short commands and are predominantly developed for English. This paper addresses these limitations by focusing on IR from speech by elderly German speakers. We propose a novel approach that combines an adapted Whisper ASR model, fine-tuned on elderly German speech (SVC-de), with Transformer-based language models trained on synthetic text datasets generated by three well-known large language models (LLMs): LeoLM, Llama3, and ChatGPT. To evaluate the robustness of our approach, we generate synthetic speech with a text-to-speech model and conduct extensive cross-dataset testing. Our results show that synthetic LLM-generated data significantly boosts classification performance and robustness to different speaking styles and unseen vocabulary. Notably, we find that LeoLM, a smaller, domain-specific 13B LLM, surpasses the much larger ChatGPT (175B) in dataset quality for German intent recognition. Our approach demonstrates that generative AI can effectively bridge data gaps in low-resource domains. We provide detailed documentation of our data generation and training process to ensure transparency and reproducibility.
>
---
#### [new 023] Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 论文提出Spectrum Projection Score（SPS）作为RAG中检索与生成协同的评估指标，解决传统方法难以量化检索贡献的问题，通过对比生成token面积与子空间方向实现语义对齐度测量，并结合xCompress框架动态控制检索摘要，提升性能与交互效率。**

- **链接: [http://arxiv.org/pdf/2508.05909v1](http://arxiv.org/pdf/2508.05909v1)**

> **作者:** Zhanghao Hu; Qinglin Zhu; Siya Qi; Yulan He; Hanqi Yan; Lin Gui
>
> **摘要:** Large Language Models (LLMs) have shown improved generation performance through retrieval-augmented generation (RAG) following the retriever-reader paradigm, which supplements model inputs with externally retrieved knowledge. However, prior work often evaluates RAG holistically, assessing the retriever and reader jointly, making it difficult to isolate the true contribution of retrieval, particularly given the prompt sensitivity of LLMs used as readers. We introduce Spectrum Projection Score (SPS), a lightweight, supervision-free metric that allows the reader to gauge the semantic alignment of a retrieved summary with its hidden representation by comparing the area formed by generated tokens from the summary, and the principal directions of subspace in the reader and to measure the relevance. Building on SPS we present xCompress, an inference time controller framework that dynamically samples, ranks, and compresses retrieval summary candidates. Extensive experiments on five QA benchmarks with four open source LLMs show that SPS not only enhances performance across a range of tasks but also provides a principled perspective on the interaction between retrieval and generation.
>
---
#### [new 024] Efficient Knowledge Probing of Large Language Models by Adapting Pre-trained Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 论文提出PEEK方法，通过调整预训练嵌入模型预测LLM知识，解决传统耗时的直接前向传播问题，实现高效知识探测。**

- **链接: [http://arxiv.org/pdf/2508.06030v1](http://arxiv.org/pdf/2508.06030v1)**

> **作者:** Kartik Sharma; Yiqiao Jin; Rakshit Trivedi; Srijan Kumar
>
> **摘要:** Large language models (LLMs) acquire knowledge across diverse domains such as science, history, and geography encountered during generative pre-training. However, due to their stochasticity, it is difficult to predict what LLMs have acquired. Prior work has developed different ways to probe this knowledge by investigating the hidden representations, crafting specific task prompts, curating representative samples, and estimating their uncertainty. However, these methods require making forward passes through the underlying model to probe the LLM's knowledge about a specific fact, making them computationally expensive and time-consuming. To bridge this gap, we propose $\textbf{PEEK}$ or $\textbf{P}$roxy $\textbf{E}$mbeddings to $\textbf{E}$stimate $\textbf{K}$nowledge of LLMs, by leveraging the pre-trained embedding models that effectively encode factual knowledge as text or graphs as proxies for LLMs. First, we identify a training set of facts known by LLMs through various probing strategies and then adapt embedding models to predict the LLM outputs with a linear decoder layer. Comprehensive evaluation on $3$ Wikipedia-derived datasets, $4$ LLMs, and $7$ embedding models shows that embeddings can predict LLM knowledge on a held-out set with up to 90 % accuracy. Furthermore, we find that sentence embedding models are more suitable than graph embeddings to predict LLM knowledge, shedding light on the underlying representation of the factual landscape. Thus, we believe that knowledge-adapted embeddings can be used to identify knowledge gaps in LLMs at scale and can provide deeper insights into LLMs' internal inductive bias. The code and data are made available at https://github.com/claws-lab/peek.
>
---
#### [new 025] Beyond Uniform Criteria: Scenario-Adaptive Multi-Dimensional Jailbreak Evaluation
- **分类: cs.CL**

- **简介: 论文提出场景自适应多维框架SceneJailEval，解决现有二分类方法无法量化危害程度及场景不匹配问题，构建14场景数据集并取得0.917 F1等优异结果。**

- **链接: [http://arxiv.org/pdf/2508.06194v1](http://arxiv.org/pdf/2508.06194v1)**

> **作者:** Lai Jiang; Yuekang Li; Xiaohan Zhang; Youtao Ding; Li Pan
>
> **摘要:** Precise jailbreak evaluation is vital for LLM red teaming and jailbreak research. Current approaches employ binary classification ( e.g., string matching, toxic text classifiers, LLM-driven methods), yielding only "yes/no" labels without quantifying harm intensity. Existing multi-dimensional frameworks ( e.g., Security Violation, Relative Truthfulness, Informativeness) apply uniform evaluation criteria across scenarios, resulting in scenario-specific mismatches--for instance, "Relative Truthfulness" is irrelevant to "hate speech"--which compromise evaluation precision. To tackle these limitations, we introduce SceneJailEval, with key contributions: (1) A groundbreaking scenario-adaptive multi-dimensional framework for jailbreak evaluation, overcoming the critical "one-size-fits-all" constraint of existing multi-dimensional methods, and featuring strong extensibility to flexibly adapt to customized or emerging scenarios. (2) A comprehensive 14-scenario dataset with diverse jailbreak variants and regional cases, filling the long-standing gap in high-quality, holistic benchmarks for scenario-adaptive evaluation. (3) SceneJailEval achieves state-of-the-art results, with an F1 score of 0.917 on our full-scenario dataset (+6% over prior SOTA) and 0.995 on JBB (+3% over prior SOTA), surpassing accuracy limits of existing evaluation methods in heterogeneous scenarios and confirming its advantage.
>
---
#### [new 026] Comparing Knowledge Injection Methods for LLMs in a Low-Resource Regime
- **分类: cs.CL**

- **简介: 论文对比不同知识注入方法在低资源环境下的效果，解决小数据下LLMs知识更新难题，通过新闻数据验证多样化提示提升学习效率，揭示遗忘现象并探讨RAG局限性，证实模型可自动生成合成数据。**

- **链接: [http://arxiv.org/pdf/2508.06178v1](http://arxiv.org/pdf/2508.06178v1)**

> **作者:** Hugo Abonizio; Thales Almeida; Roberto Lotufo; Rodrigo Nogueira
>
> **摘要:** Large language models (LLMs) often require vast amounts of text to effectively acquire new knowledge. While continuing pre-training on large corpora or employing retrieval-augmented generation (RAG) has proven successful, updating an LLM with only a few thousand or million tokens remains challenging. In this work, we investigate the task of injecting small, unstructured information into LLMs and its relation to the catastrophic forgetting phenomenon. We use a dataset of recent news -- ensuring no overlap with the model's pre-training data -- to evaluate the knowledge acquisition by probing the model with question-answer pairs related the learned information. Starting from a continued pre-training baseline, we explored different augmentation algorithms to generate synthetic data to improve the knowledge acquisition capabilities. Our experiments show that simply continuing pre-training on limited data yields modest improvements, whereas exposing the model to diverse textual variations significantly improves the learning of new facts -- particularly with methods that induce greater variability through diverse prompting. Furthermore, we shed light on the forgetting phenomenon in small-data regimes, illustrating the delicate balance between learning new content and retaining existing capabilities. We also confirm the sensitivity of RAG-based approaches for knowledge injection, which often lead to greater degradation on control datasets compared to parametric methods. Finally, we demonstrate that models can generate effective synthetic training data themselves, suggesting a pathway toward self-improving model updates. All code and generated data used in our experiments are publicly available, providing a resource for studying efficient knowledge injection in LLMs with limited data at https://github.com/hugoabonizio/knowledge-injection-methods.
>
---
#### [new 027] Few-Shot Prompting for Extractive Quranic QA with Instruction-Tuned LLMs
- **分类: cs.CL; cs.IR**

- **简介: 本文提出基于少样本提示的提取式Quran问答方法，针对复杂语言和低资源场景，通过指令调优LLM并设计专用阿拉伯提示框架与后处理系统，提升精度并减少幻觉，验证提示式指令调优在低资源语料中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.06103v1](http://arxiv.org/pdf/2508.06103v1)**

> **作者:** Mohamed Basem; Islam Oshallah; Ali Hamdi; Ammar Mohammed
>
> **备注:** 6 pages , 2 figures , Accepted in IMSA 2025,Egypt , https://imsa.msa.edu.eg/
>
> **摘要:** This paper presents two effective approaches for Extractive Question Answering (QA) on the Quran. It addresses challenges related to complex language, unique terminology, and deep meaning in the text. The second uses few-shot prompting with instruction-tuned large language models such as Gemini and DeepSeek. A specialized Arabic prompt framework is developed for span extraction. A strong post-processing system integrates subword alignment, overlap suppression, and semantic filtering. This improves precision and reduces hallucinations. Evaluations show that large language models with Arabic instructions outperform traditional fine-tuned models. The best configuration achieves a pAP10 score of 0.637. The results confirm that prompt-based instruction tuning is effective for low-resource, semantically rich QA tasks.
>
---
#### [new 028] Scaling Personality Control in LLMs with Big Five Scaler Prompts
- **分类: cs.CL; cs.MA**

- **简介: 论文提出Big5-Scaler框架，通过自然语言提示嵌入数值特质值，实现LLMs可控人格控制，评估其在特质表达、对话生成等任务中的效果。**

- **链接: [http://arxiv.org/pdf/2508.06149v1](http://arxiv.org/pdf/2508.06149v1)**

> **作者:** Gunhee Cho; Yun-Gyung Cheong
>
> **摘要:** We present Big5-Scaler, a prompt-based framework for conditioning large language models (LLMs) with controllable Big Five personality traits. By embedding numeric trait values into natural language prompts, our method enables fine-grained personality control without additional training. We evaluate Big5-Scaler across trait expression, dialogue generation, and human trait imitation tasks. Results show that it induces consistent and distinguishable personality traits across models, with performance varying by prompt type and scale. Our analysis highlights the effectiveness of concise prompts and lower trait intensities, providing a efficient approach for building personality-aware dialogue agents.
>
---
#### [new 029] Matrix-Driven Instant Review: Confident Detection and Reconstruction of LLM Plagiarism on PC
- **分类: cs.CL; math.PR**

- **简介: 论文提出基于矩阵分析和大偏差理论的MDIR方法，解决LLM剽窃检测中权重重建、p值计算及误报问题，实现高效准确的检测。**

- **链接: [http://arxiv.org/pdf/2508.06309v1](http://arxiv.org/pdf/2508.06309v1)**

> **作者:** Ruichong Zhang
>
> **摘要:** In recent years, concerns about intellectual property (IP) in large language models (LLMs) have grown significantly. Plagiarizing other LLMs (through direct weight copying, upcycling, pruning, or continual pretraining) and claiming authorship without properly attributing to the original license, is a serious misconduct that can lead to significant financial and reputational harm to the original developers. However, existing methods for detecting LLM plagiarism fall short in key areas. They fail to accurately reconstruct weight correspondences, lack the ability to compute statistical significance measures such as $p$-values, and may mistakenly flag models trained on similar data as being related. To address these limitations, we propose Matrix-Driven Instant Review (MDIR), a novel method that leverages matrix analysis and Large Deviation Theory. MDIR achieves accurate reconstruction of weight relationships, provides rigorous $p$-value estimation, and focuses exclusively on weight similarity without requiring full model inference. Experimental results demonstrate that MDIR reliably detects plagiarism even after extensive transformations, such as random permutations and continual pretraining with trillions of tokens. Moreover, all detections can be performed on a single PC within an hour, making MDIR both efficient and accessible.
>
---
#### [new 030] Quantifying Conversation Drift in MCP via Latent Polytope
- **分类: cs.CL**

- **简介: 论文提出SecMCP框架，通过潜在多面体建模量化MCP中对话漂移，解决工具中毒等安全威胁，提升检测精度与系统效率。**

- **链接: [http://arxiv.org/pdf/2508.06418v1](http://arxiv.org/pdf/2508.06418v1)**

> **作者:** Haoran Shi; Hongwei Yao; Shuo Shao; Shaopeng Jiao; Ziqi Peng; Zhan Qin; Cong Wang
>
> **摘要:** The Model Context Protocol (MCP) enhances large language models (LLMs) by integrating external tools, enabling dynamic aggregation of real-time data to improve task execution. However, its non-isolated execution context introduces critical security and privacy risks. In particular, adversarially crafted content can induce tool poisoning or indirect prompt injection, leading to conversation hijacking, misinformation propagation, or data exfiltration. Existing defenses, such as rule-based filters or LLM-driven detection, remain inadequate due to their reliance on static signatures, computational inefficiency, and inability to quantify conversational hijacking. To address these limitations, we propose SecMCP, a secure framework that detects and quantifies conversation drift, deviations in latent space trajectories induced by adversarial external knowledge. By modeling LLM activation vectors within a latent polytope space, SecMCP identifies anomalous shifts in conversational dynamics, enabling proactive detection of hijacking, misleading, and data exfiltration. We evaluate SecMCP on three state-of-the-art LLMs (Llama3, Vicuna, Mistral) across benchmark datasets (MS MARCO, HotpotQA, FinQA), demonstrating robust detection with AUROC scores exceeding 0.915 while maintaining system usability. Our contributions include a systematic categorization of MCP security threats, a novel latent polytope-based methodology for quantifying conversation drift, and empirical validation of SecMCP's efficacy.
>
---
#### [new 031] Evaluating Style-Personalized Text Generation: Challenges and Directions
- **分类: cs.CL**

- **简介: 论文探讨风格个性化文本生成的评估难题，提出通过构建八任务基准，对比BLEU/ROUGE等指标与风格嵌入、LLM裁判等方法，证实多指标组合更有效。**

- **链接: [http://arxiv.org/pdf/2508.06374v1](http://arxiv.org/pdf/2508.06374v1)**

> **作者:** Anubhav Jangra; Bahareh Sarrafzadeh; Adrian de Wynter; Silviu Cucerzan; Sujay Kumar Jauhar
>
> **摘要:** While prior research has built tools and benchmarks towards style personalized text generation, there has been limited exploration of evaluation in low-resource author style personalized text generation space. Through this work, we question the effectiveness of the widely adopted evaluation metrics like BLEU and ROUGE, and explore other evaluation paradigms such as style embeddings and LLM-as-judge to holistically evaluate the style personalized text generation task. We evaluate these metrics and their ensembles using our style discrimination benchmark, that spans eight writing tasks, and evaluates across three settings, domain discrimination, authorship attribution, and LLM personalized vs non-personalized discrimination. We provide conclusive evidence to adopt ensemble of diverse evaluation metrics to effectively evaluate style personalized text generation.
>
---
#### [new 032] Echoes of Automation: The Increasing Use of LLMs in Newsmaking
- **分类: cs.CL; cs.AI**

- **简介: 论文分析LLMs在新闻报道中的应用，探讨其对新闻可信度与作者署名的影响，通过检测工具及语言分析揭示LLMs提升可读性但降低正式性的趋势，尤其在本地和学院媒体中广泛使用。**

- **链接: [http://arxiv.org/pdf/2508.06445v1](http://arxiv.org/pdf/2508.06445v1)**

> **作者:** Abolfazl Ansari; Delvin Ce Zhang; Nafis Irtiza Tripto; Dongwon Lee
>
> **备注:** To appear in 18th International Conference on Social Computing, Behavioral-Cultural Modeling, & Prediction and Behavior Representation in Modeling and Simulation, and to be published in the Springer LNCS series
>
> **摘要:** The rapid rise of Generative AI (GenAI), particularly LLMs, poses concerns for journalistic integrity and authorship. This study examines AI-generated content across over 40,000 news articles from major, local, and college news media, in various media formats. Using three advanced AI-text detectors (e.g., Binoculars, Fast-Detect GPT, and GPTZero), we find substantial increase of GenAI use in recent years, especially in local and college news. Sentence-level analysis reveals LLMs are often used in the introduction of news, while conclusions usually written manually. Linguistic analysis shows GenAI boosts word richness and readability but lowers formality, leading to more uniform writing styles, particularly in local media.
>
---
#### [new 033] Crisp Attention: Regularizing Transformers via Structured Sparsity
- **分类: cs.CL; cs.AI**

- **简介: 论文提出结构化注意力稀疏化方法，通过在微调中引入稀疏性提升Transformer性能，突破传统认为稀疏化会牺牲准确性的假设，验证其作为隐式正则化有效防止过拟合。**

- **链接: [http://arxiv.org/pdf/2508.06016v1](http://arxiv.org/pdf/2508.06016v1)**

> **作者:** Sagar Gandhi; Vishal Gandhi
>
> **摘要:** The quadratic computational cost of the self-attention mechanism is a primary challenge in scaling Transformer models. While attention sparsity is widely studied as a technique to improve computational efficiency, it is almost universally assumed to come at the cost of model accuracy. In this paper, we report a surprising counter-example to this common wisdom. By introducing structured, post-hoc sparsity to the attention mechanism of a DistilBERT model during fine-tuning on the SST-2 sentiment analysis task, we find that model accuracy improves significantly. Our model with 80\% attention sparsity achieves a validation accuracy of 91.59\%, a 0.97\% absolute improvement over the dense baseline. We hypothesize that this phenomenon is due to sparsity acting as a powerful implicit regularizer, preventing the model from overfitting by forcing it to make predictions with a more constrained and robust set of features. Our work recasts attention sparsity not just as a tool for computational efficiency, but as a potential method for improving the generalization and performance of Transformer models.
>
---
#### [new 034] Adversarial Topic-aware Prompt-tuning for Cross-topic Automated Essay Scoring
- **分类: cs.CL**

- **简介: 论文提出ATOP方法，解决跨主题自动评分中主题特定特征缺失问题，通过联合学习共享与特定特征，利用对抗训练和伪标签提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.05987v1](http://arxiv.org/pdf/2508.05987v1)**

> **作者:** Chunyun Zhang; Hongyan Zhao; Chaoran Cui; Qilong Song; Zhiqing Lu; Shuai Gong; Kailin Liu
>
> **摘要:** Cross-topic automated essay scoring (AES) aims to develop a transferable model capable of effectively evaluating essays on a target topic. A significant challenge in this domain arises from the inherent discrepancies between topics. While existing methods predominantly focus on extracting topic-shared features through distribution alignment of source and target topics, they often neglect topic-specific features, limiting their ability to assess critical traits such as topic adherence. To address this limitation, we propose an Adversarial TOpic-aware Prompt-tuning (ATOP), a novel method that jointly learns topic-shared and topic-specific features to improve cross-topic AES. ATOP achieves this by optimizing a learnable topic-aware prompt--comprising both shared and specific components--to elicit relevant knowledge from pre-trained language models (PLMs). To enhance the robustness of topic-shared prompt learning and mitigate feature scale sensitivity introduced by topic alignment, we incorporate adversarial training within a unified regression and classification framework. In addition, we employ a neighbor-based classifier to model the local structure of essay representations and generate pseudo-labels for target-topic essays. These pseudo-labels are then used to guide the supervised learning of topic-specific prompts tailored to the target topic. Extensive experiments on the publicly available ASAP++ dataset demonstrate that ATOP significantly outperforms existing state-of-the-art methods in both holistic and multi-trait essay scoring. The implementation of our method is publicly available at: https://anonymous.4open.science/r/ATOP-A271.
>
---
#### [new 035] SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning
- **分类: cs.CL**

- **简介: 论文提出SlimInfer框架，通过动态剪枝长上下文LLM推理中的冗余token，利用信息扩散机制实现高效加速，降低内存与I/O成本，提升TTFT速度2.53倍，端到端延迟1.88倍。**

- **链接: [http://arxiv.org/pdf/2508.06447v1](http://arxiv.org/pdf/2508.06447v1)**

> **作者:** Lingkun Long; Rubing Yang; Yushi Huang; Desheng Hui; Ao Zhou; Jianlei Yang
>
> **摘要:** Long-context inference for Large Language Models (LLMs) is heavily limited by high computational demands. While several existing methods optimize attention computation, they still process the full set of hidden states at each layer, limiting overall efficiency. In this work, we propose SlimInfer, an innovative framework that aims to accelerate inference by directly pruning less critical prompt tokens during the forward pass. Our key insight is an information diffusion phenomenon: As information from critical tokens propagates through layers, it becomes distributed across the entire sequence. This diffusion process suggests that LLMs can maintain their semantic integrity when excessive tokens, even including these critical ones, are pruned in hidden states. Motivated by this, SlimInfer introduces a dynamic fine-grained pruning mechanism that accurately removes redundant tokens of hidden state at intermediate layers. This layer-wise pruning naturally enables an asynchronous KV cache manager that prefetches required token blocks without complex predictors, reducing both memory usage and I/O costs. Extensive experiments show that SlimInfer can achieve up to $\mathbf{2.53\times}$ time-to-first-token (TTFT) speedup and $\mathbf{1.88\times}$ end-to-end latency reduction for LLaMA3.1-8B-Instruct on a single RTX 4090, without sacrificing performance on LongBench. Our code will be released upon acceptance.
>
---
#### [new 036] Cyberbullying Detection via Aggression-Enhanced Prompting
- **分类: cs.CL**

- **简介: 论文提出通过整合攻击性检测作为辅助任务，利用增强提示管道提升网络欺凌检测性能，解决复杂表达下的检测难题。**

- **链接: [http://arxiv.org/pdf/2508.06360v1](http://arxiv.org/pdf/2508.06360v1)**

> **作者:** Aisha Saeid; Anu Sabu; Girish A. Koushik; Ferrante Neri; Diptesh Kanojia
>
> **备注:** Accepted to RANLP 2025
>
> **摘要:** Detecting cyberbullying on social media remains a critical challenge due to its subtle and varied expressions. This study investigates whether integrating aggression detection as an auxiliary task within a unified training framework can enhance the generalisation and performance of large language models (LLMs) in cyberbullying detection. Experiments are conducted on five aggression datasets and one cyberbullying dataset using instruction-tuned LLMs. We evaluated multiple strategies: zero-shot, few-shot, independent LoRA fine-tuning, and multi-task learning (MTL). Given the inconsistent results of MTL, we propose an enriched prompt pipeline approach in which aggression predictions are embedded into cyberbullying detection prompts to provide contextual augmentation. Preliminary results show that the enriched prompt pipeline consistently outperforms standard LoRA fine-tuning, indicating that aggression-informed context significantly boosts cyberbullying detection. This study highlights the potential of auxiliary tasks, such as aggression detection, to improve the generalisation of LLMs for safety-critical applications on social networks.
>
---
#### [new 037] InfoCausalQA:Can Models Perform Non-explicit Causal Reasoning Based on Infographic?
- **分类: cs.CL; cs.AI**

- **简介: 论文提出InfoCausalQA作为多模态因果推理基准，解决VLMs在因果推理（如因果关系识别、干预分析）中的不足，通过量化与语义任务评估图表+文本组合的因果理解能力，揭示当前模型需提升多模态因果推理能力。**

- **链接: [http://arxiv.org/pdf/2508.06220v1](http://arxiv.org/pdf/2508.06220v1)**

> **作者:** Keummin Ka; Junhyeong Park; Jahyun Jeon; Youngjae Yu
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) have demonstrated impressive capabilities in perception and reasoning. However, the ability to perform causal inference -- a core aspect of human cognition -- remains underexplored, particularly in multimodal settings. In this study, we introduce InfoCausalQA, a novel benchmark designed to evaluate causal reasoning grounded in infographics that combine structured visual data with textual context. The benchmark comprises two tasks: Task 1 focuses on quantitative causal reasoning based on inferred numerical trends, while Task 2 targets semantic causal reasoning involving five types of causal relations: cause, effect, intervention, counterfactual, and temporal. We manually collected 494 infographic-text pairs from four public sources and used GPT-4o to generate 1,482 high-quality multiple-choice QA pairs. These questions were then carefully revised by humans to ensure they cannot be answered based on surface-level cues alone but instead require genuine visual grounding. Our experimental results reveal that current VLMs exhibit limited capability in computational reasoning and even more pronounced limitations in semantic causal reasoning. Their significantly lower performance compared to humans indicates a substantial gap in leveraging infographic-based information for causal inference. Through InfoCausalQA, we highlight the need for advancing the causal reasoning abilities of multimodal AI systems.
>
---
#### [new 038] Do Machines Think Emotionally? Cognitive Appraisal Analysis of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型（LLMs）是否具备情感推理能力，通过认知评估理论分析其内部机制，提出CoRE基准评估情感认知维度，揭示模型对不同情绪的推理模式与内部表示。**

- **链接: [http://arxiv.org/pdf/2508.05880v1](http://arxiv.org/pdf/2508.05880v1)**

> **作者:** Sree Bhattacharyya; Lucas Craig; Tharun Dilliraj; Jia Li; James Z. Wang
>
> **摘要:** Affective Computing has been established as a crucial field of inquiry to advance the holistic development of Artificial Intelligence (AI) systems. Foundation models -- especially Large Language Models (LLMs) -- have been evaluated, trained, or instruction-tuned in several past works, to become better predictors or generators of emotion. Most of these studies, however, approach emotion-related tasks in a supervised manner, assessing or training the capabilities of LLMs using discrete emotion labels associated with stimuli (e.g., text, images, video, audio). Evaluation studies, in particular, have often been limited to standard and superficial emotion-related tasks, such as the recognition of evoked or expressed emotions. In this paper, we move beyond surface-level emotion tasks to investigate how LLMs reason about emotions through cognitive dimensions. Drawing from cognitive appraisal theory, we examine whether LLMs produce coherent and plausible cognitive reasoning when reasoning about emotionally charged stimuli. We introduce a large-scale benchmark on Cognitive Reasoning for Emotions - CoRE - to evaluate internal cognitive structures implicitly used by LLMs for emotional reasoning. Through a plethora of evaluation experiments and analysis, we seek to answer: (a) Are models more likely to implicitly rely on specific cognitive appraisal dimensions?, (b) What cognitive dimensions are important for characterizing specific emotions?, and, (c) Can the internal representations of different emotion categories in LLMs be interpreted through cognitive appraisal dimensions? Our results and analyses reveal diverse reasoning patterns across different LLMs. Our benchmark and code will be made publicly available.
>
---
#### [new 039] AURA: Affordance-Understanding and Risk-aware Alignment Technique for Large Language Models
- **分类: cs.CL**

- **简介: 论文提出AURA框架，针对LLMs的 affordance-based 安全风险，通过Process Reward Models实现多层评估与动态解码，提升逻辑正确性与安全性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06124v1](http://arxiv.org/pdf/2508.06124v1)**

> **作者:** Sayantan Adak; Pratyush Chatterjee; Somnath Banerjee; Rima Hazra; Somak Aditya; Animesh Mukherjee
>
> **摘要:** Present day LLMs face the challenge of managing affordance-based safety risks-situations where outputs inadvertently facilitate harmful actions due to overlooked logical implications. Traditional safety solutions, such as scalar outcome-based reward models, parameter tuning, or heuristic decoding strategies, lack the granularity and proactive nature needed to reliably detect and intervene during subtle yet crucial reasoning steps. Addressing this fundamental gap, we introduce AURA, an innovative, multi-layered framework centered around Process Reward Models (PRMs), providing comprehensive, step level evaluations across logical coherence and safety-awareness. Our framework seamlessly combines introspective self-critique, fine-grained PRM assessments, and adaptive safety-aware decoding to dynamically and proactively guide models toward safer reasoning trajectories. Empirical evidence clearly demonstrates that this approach significantly surpasses existing methods, significantly improving the logical integrity and affordance-sensitive safety of model outputs. This research represents a pivotal step toward safer, more responsible, and contextually aware AI, setting a new benchmark for alignment-sensitive applications.
>
---
#### [new 040] Semantic and Structural Analysis of Implicit Biases in Large Language Models: An Interpretable Approach
- **分类: cs.CL**

- **简介: 论文提出一种可解释方法，分析大型语言模型隐性偏见，通过语义嵌入与注意力机制揭示偏见形成路径，验证其在多维度数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.06155v1](http://arxiv.org/pdf/2508.06155v1)**

> **作者:** Renhan Zhang; Lian Lian; Zhen Qi; Guiran Liu
>
> **摘要:** This paper addresses the issue of implicit stereotypes that may arise during the generation process of large language models. It proposes an interpretable bias detection method aimed at identifying hidden social biases in model outputs, especially those semantic tendencies that are not easily captured through explicit linguistic features. The method combines nested semantic representation with a contextual contrast mechanism. It extracts latent bias features from the vector space structure of model outputs. Using attention weight perturbation, it analyzes the model's sensitivity to specific social attribute terms, thereby revealing the semantic pathways through which bias is formed. To validate the effectiveness of the method, this study uses the StereoSet dataset, which covers multiple stereotype dimensions including gender, profession, religion, and race. The evaluation focuses on several key metrics, such as bias detection accuracy, semantic consistency, and contextual sensitivity. Experimental results show that the proposed method achieves strong detection performance across various dimensions. It can accurately identify bias differences between semantically similar texts while maintaining high semantic alignment and output stability. The method also demonstrates high interpretability in its structural design. It helps uncover the internal bias association mechanisms within language models. This provides a more transparent and reliable technical foundation for bias detection. The approach is suitable for real-world applications where high trustworthiness of generated content is required.
>
---
#### [new 041] Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Temporal Self-Rewarding模型，通过协调过去、现在、未来生成，解决同步改进chosen/rejected导致的表示差异问题，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.06026v1](http://arxiv.org/pdf/2508.06026v1)**

> **作者:** Yidong Wang; Xin Wang; Cunxiang Wang; Junfeng Fang; Qiufeng Wang; Jianing Chu; Xuran Meng; Shuxun Yang; Libo Qin; Yue Zhang; Wei Ye; Shikun Zhang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Self-Rewarding Language Models propose an architecture in which the Large Language Models(LLMs) both generates responses and evaluates its own outputs via LLM-as-a-Judge prompting, dynamically improving its generative capabilities through iterative Direct Preference Optimization (DPO). However, our analysis reveals a critical limitation in existing Self-Rewarding paradigms: the synchronized improvement of chosen and rejected responses progressively narrows the representational difference between contrasting samples, undermining effective preference learning. We propose \textbf{Temporal Self-Rewarding Language Models} that strategically coordinate past, present, and future model generations to sustain learning signals. Our dual-phase framework introduces: (1) \textit{Anchored Rejection} - fixing rejected responses using the past initial model's outputs and (2) \textit{Future-Guided Chosen} - dynamically curating chosen samples using next-generation model predictions. Extensive experiments across three model families (Llama, Qwen, Mistral) and different model sizes (Llama3B/8B/70B) demonstrate significant improvements when trained with our method compared to Self-Rewarding using same computation resources. For example, Llama3.1-8B reaches a 29.44 win rate on AlpacaEval 2.0 with our method, outperforming the Self-Rewarding baseline (19.69) by 9.75. Notably, our method also demonstrates superior out-of-distribution generalization across mathematical reasoning (GSM8K), knowledge-based QA (ARC, TruthfulQA), and code generation (HumanEval) tasks, even though we do not specifically collect such training data.
>
---
#### [new 042] Prosocial Behavior Detection in Player Game Chat: From Aligning Human-AI Definitions to Efficient Annotation at Scale
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; K.4**

- **简介: 论文提出一种面向玩家游戏聊天的利他行为检测方法，解决缺乏明确定义和标注数据的问题，通过三阶段流水线结合人类与AI协作，提升标注效率与精度，降低推理成本。**

- **链接: [http://arxiv.org/pdf/2508.05938v1](http://arxiv.org/pdf/2508.05938v1)**

> **作者:** Rafal Kocielnik; Min Kim; Penphob; Boonyarungsrit; Fereshteh Soltani; Deshawn Sambrano; Animashree Anandkumar; R. Michael Alvarez
>
> **备注:** 9 pages, 4 figures, 4 tables
>
> **摘要:** Detecting prosociality in text--communication intended to affirm, support, or improve others' behavior--is a novel and increasingly important challenge for trust and safety systems. Unlike toxic content detection, prosociality lacks well-established definitions and labeled data, requiring new approaches to both annotation and deployment. We present a practical, three-stage pipeline that enables scalable, high-precision prosocial content classification while minimizing human labeling effort and inference costs. First, we identify the best LLM-based labeling strategy using a small seed set of human-labeled examples. We then introduce a human-AI refinement loop, where annotators review high-disagreement cases between GPT-4 and humans to iteratively clarify and expand the task definition-a critical step for emerging annotation tasks like prosociality. This process results in improved label quality and definition alignment. Finally, we synthesize 10k high-quality labels using GPT-4 and train a two-stage inference system: a lightweight classifier handles high-confidence predictions, while only $\sim$35\% of ambiguous instances are escalated to GPT-4o. This architecture reduces inference costs by $\sim$70% while achieving high precision ($\sim$0.90). Our pipeline demonstrates how targeted human-AI interaction, careful task formulation, and deployment-aware architecture design can unlock scalable solutions for novel responsible AI tasks.
>
---
#### [new 043] PEACH: A sentence-aligned Parallel English-Arabic Corpus for Healthcare
- **分类: cs.CL**

- **简介: 论文构建了医疗领域的英文-阿拉伯语平行语料库，用于对比语言学、翻译研究及NLP任务，解决翻译质量评估与文本可读性问题。**

- **链接: [http://arxiv.org/pdf/2508.05722v1](http://arxiv.org/pdf/2508.05722v1)**

> **作者:** Rania Al-Sabbagh
>
> **摘要:** This paper introduces PEACH, a sentence-aligned parallel English-Arabic corpus of healthcare texts encompassing patient information leaflets and educational materials. The corpus contains 51,671 parallel sentences, totaling approximately 590,517 English and 567,707 Arabic word tokens. Sentence lengths vary between 9.52 and 11.83 words on average. As a manually aligned corpus, PEACH is a gold-standard corpus, aiding researchers in contrastive linguistics, translation studies, and natural language processing. It can be used to derive bilingual lexicons, adapt large language models for domain-specific machine translation, evaluate user perceptions of machine translation in healthcare, assess patient information leaflets and educational materials' readability and lay-friendliness, and as an educational resource in translation studies. PEACH is publicly accessible.
>
---
#### [new 044] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
- **分类: cs.CL**

- **简介: 论文提出ConlangCrafter，通过多跳LLM管道分解构造语言设计为音系、形态等模块，利用LLMs的元能力生成多样且一致的语言，解决传统方法效率低、多样性不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.06094v1](http://arxiv.org/pdf/2508.06094v1)**

> **作者:** Morris Alper; Moran Yanuka; Raja Giryes; Gašper Beguš
>
> **备注:** Project page: https://conlangcrafter.github.io
>
> **摘要:** Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, large-scale foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages -- phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' meta-linguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We evaluate ConlangCrafter on metrics measuring coherence and typological diversity, demonstrating its ability to produce coherent and varied conlangs without human linguistic expertise.
>
---
#### [new 045] Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports
- **分类: cs.IR; cs.AI; cs.CL; cs.CV; cs.LG; I.2.7; I.7.2; J.1**

- **简介: 该论文提出微调VLM用于马来西亚审计财务报告中表格的Markdown转换，解决旋转布局、多级标题等问题，通过LoRA优化并评估TEDS指标，实现高精度转换。**

- **链接: [http://arxiv.org/pdf/2508.05669v1](http://arxiv.org/pdf/2508.05669v1)**

> **作者:** Jin Khye Tan; En Jun Choong; Ethan Jeremiah Chitty; Yan Pheng Choo; John Hsin Yang Wong; Chern Eu Cheah
>
> **备注:** 28 pages, 14 figures, 5 tables. Evaluation code (LLM-as-a-judge and Markdown TEDS) is available at https://github.com/jinkhye/MyFinMarkdown. The development dataset and evaluation benchmark are available on Hugging Face at https://huggingface.co/datasets/jinkhye/MyFinMarkdown-sample and https://huggingface.co/datasets/jinkhye/MyFinMarkdown-bench respectively
>
> **摘要:** Accurately extracting and representing the structure of tabular data from financial documents remains a critical challenge in document understanding, particularly for regulatory and analytical use cases. This study addresses the complexity of converting financial tables from Malaysian audited financial reports into Markdown format, a task complicated by rotated layouts, multi-level headers, and implicit structural cues. We propose a fine-tuned vision-language model (VLM), based on Qwen2.5-VL-7B, optimized for high-fidelity Markdown generation from document images. Our approach includes a curated dataset of 2,152 image-text pairs with augmentations and a supervised fine-tuning strategy using LoRA. To assess performance, we evaluated our model on 100 out-of-sample tables using a dual framework: a criteria-based LLM-as-a-judge for fine-grained accuracy and our novel Markdown Tree-Edit-Distance-based Similarity (TEDS) metric for holistic structural fidelity. Our model achieves a 92.20% overall accuracy on the criteria-based assessment and a 96.53% Markdown TEDS score. This performance significantly surpasses its Qwen2.5-VL-7B base model, larger-scale VLMs, and specialized reasoning-enabled models. Compared to these self-hosted alternatives, it also significantly reduces inference time. Furthermore, its accuracy exceeds that of widely used proprietary models such as OpenAI's GPT-4o and Gemini 2.5 Flash. These results demonstrate that domain-specific fine-tuning provides an effective and efficient method to bridge the gap between unstructured financial documents and downstream automation, rivalling much larger and more general models without their computational overhead.
>
---
#### [new 046] A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges
- **分类: cs.DL; cs.AI; cs.CL; cs.IR**

- **简介: 该论文通过系统综述2020-2025年RAG研究，分析技术、评估指标及挑战，总结高被引成果，识别方法论缺陷并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2508.06401v1](http://arxiv.org/pdf/2508.06401v1)**

> **作者:** Andrew Brown; Muhammad Roman; Barry Devereux
>
> **备注:** 58 pages
>
> **摘要:** This systematic review of the research literature on retrieval-augmented generation (RAG) provides a focused analysis of the most highly cited studies published between 2020 and May 2025. A total of 128 articles met our inclusion criteria. The records were retrieved from ACM Digital Library, IEEE Xplore, Scopus, ScienceDirect, and the Digital Bibliography and Library Project (DBLP). RAG couples a neural retriever with a generative language model, grounding output in up-to-date, non-parametric memory while retaining the semantic generalisation stored in model weights. Guided by the PRISMA 2020 framework, we (i) specify explicit inclusion and exclusion criteria based on citation count and research questions, (ii) catalogue datasets, architectures, and evaluation practices, and (iii) synthesise empirical evidence on the effectiveness and limitations of RAG. To mitigate citation-lag bias, we applied a lower citation-count threshold to papers published in 2025 so that emerging breakthroughs with naturally fewer citations were still captured. This review clarifies the current research landscape, highlights methodological gaps, and charts priority directions for future research.
>
---
#### [new 047] ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls
- **分类: cs.CR; cs.AI; cs.CL; cs.MA**

- **简介: 论文提出ScamAgent，通过多轮对话模拟诈骗电话，揭示现有安全机制对AI代理威胁的失效，构建自主学习的诈骗生成模型并验证其实际应用。**

- **链接: [http://arxiv.org/pdf/2508.06457v1](http://arxiv.org/pdf/2508.06457v1)**

> **作者:** Sanket Badhe
>
> **备注:** Accepted at CAMLIS 25: Conference on Applied Machine Learning for Information Security. 10 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive fluency and reasoning capabilities, but their potential for misuse has raised growing concern. In this paper, we present ScamAgent, an autonomous multi-turn agent built on top of LLMs, capable of generating highly realistic scam call scripts that simulate real-world fraud scenarios. Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns. We show that current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats. Even models with strong prompt-level safeguards can be bypassed when prompts are decomposed, disguised, or delivered incrementally within an agent framework. We further demonstrate the transformation of scam scripts into lifelike voice calls using modern text-to-speech systems, completing a fully automated scam pipeline. Our findings highlight an urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI.
>
---
#### [new 048] Enhancing Retrieval-Augmented Generation for Electric Power Industry Customer Support
- **分类: cs.IR; cs.AI; cs.CL; I.2.m**

- **简介: 论文提出通过查询重写、RAG融合、意图识别与上下文重排等技术，改进电力行业客服系统，解决模糊、多意图查询问题，最终在两个数据集上实现97.9%和89.6%的高准确率。**

- **链接: [http://arxiv.org/pdf/2508.05664v1](http://arxiv.org/pdf/2508.05664v1)**

> **作者:** Hei Yu Chan; Kuok Tou Ho; Chenglong Ma; Yujing Si; Hok Lai Lin; Sa Lei Lam
>
> **备注:** 6 pages
>
> **摘要:** Many AI customer service systems use standard NLP pipelines or finetuned language models, which often fall short on ambiguous, multi-intent, or detail-specific queries. This case study evaluates recent techniques: query rewriting, RAG Fusion, keyword augmentation, intent recognition, and context reranking, for building a robust customer support system in the electric power domain. We compare vector-store and graph-based RAG frameworks, ultimately selecting the graph-based RAG for its superior performance in handling complex queries. We find that query rewriting improves retrieval for queries using non-standard terminology or requiring precise detail. RAG Fusion boosts performance on vague or multifaceted queries by merging multiple retrievals. Reranking reduces hallucinations by filtering irrelevant contexts. Intent recognition supports the decomposition of complex questions into more targeted sub-queries, increasing both relevance and efficiency. In contrast, keyword augmentation negatively impacts results due to biased keyword selection. Our final system combines intent recognition, RAG Fusion, and reranking to handle disambiguation and multi-source queries. Evaluated on both a GPT-4-generated dataset and a real-world electricity provider FAQ dataset, it achieves 97.9% and 89.6% accuracy respectively, substantially outperforming baseline RAG models.
>
---
#### [new 049] Effective Training Data Synthesis for Improving MLLM Chart Understanding
- **分类: cs.CV; cs.CL**

- **简介: 论文提出五步数据合成方法，生成ECD数据集以提升MLLM图表理解能力，解决现有合成图表与真实图表相似性不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.06492v1](http://arxiv.org/pdf/2508.06492v1)**

> **作者:** Yuwei Yang; Zeyu Zhang; Yunzhong Hou; Zhuowan Li; Gaowen Liu; Ali Payani; Yuan-Sen Ting; Liang Zheng
>
> **备注:** Accepted by ICCV 2025 (poster). 26 pages, 17 figures
>
> **摘要:** Being able to effectively read scientific plots, or chart understanding, is a central part toward building effective agents for science. However, existing multimodal large language models (MLLMs), especially open-source ones, are still falling behind with a typical success rate of 30%-50% on challenging benchmarks. Previous studies on fine-tuning MLLMs with synthetic charts are often restricted by their inadequate similarity to the real charts, which could compromise model training and performance on complex real-world charts. In this study, we show that modularizing chart generation and diversifying visual details improves chart understanding capabilities. In particular, we design a five-step data synthesis pipeline, where we separate data and function creation for single plot generation, condition the generation of later subplots on earlier ones for multi-subplot figures, visually diversify the generated figures, filter out low quality data, and finally generate the question-answer (QA) pairs with GPT-4o. This approach allows us to streamline the generation of fine-tuning datasets and introduce the effective chart dataset (ECD), which contains 10k+ chart images and 300k+ QA pairs, covering 25 topics and featuring 250+ chart type combinations with high visual complexity. We show that ECD consistently improves the performance of various MLLMs on a range of real-world and synthetic test sets. Code, data and models are available at: https://github.com/yuweiyang-anu/ECD.
>
---
#### [new 050] DMFI: Dual-Modality Fine-Tuning and Inference Framework for LLM-Based Insider Threat Detection
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文提出双模态框架DMFI，用于基于LLM的内部威胁检测，解决传统模型难以捕捉语义和行为动态的问题，通过语义推理与行为建模结合，提升检测准确性。**

- **链接: [http://arxiv.org/pdf/2508.05694v1](http://arxiv.org/pdf/2508.05694v1)**

> **作者:** Kaichuan Kong; Dongjie Liu; Xiaobo Jin; Guanggang Geng; Zhiying Li; Jian Weng
>
> **备注:** Submitted to the 2025 IEEE International Conference on Data Mining (ICDM)
>
> **摘要:** Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection. Our work demonstrates the effectiveness of combining LLM reasoning with structured behavioral modeling, offering a scalable and deployable solution for modern insider threat detection.
>
---
#### [new 051] NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 本文提出NanoCodec，通过低帧率编码优化语音LLM推理，解决高帧率导致的慢问题，实现高质量压缩并设新基准。**

- **链接: [http://arxiv.org/pdf/2508.05835v1](http://arxiv.org/pdf/2508.05835v1)**

> **作者:** Edresson Casanova; Paarth Neekhara; Ryan Langman; Shehzeen Hussain; Subhankar Ghosh; Xuesong Yang; Ante Jukić; Jason Li; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Large Language Models (LLMs) have significantly advanced audio processing by leveraging audio codecs to discretize audio into tokens, enabling the application of language modeling techniques to speech data. However, existing audio codecs often operate at high frame rates, leading to slow training and inference, particularly for autoregressive models. To address this, there is growing interest in low frame-rate audio codecs, which reduce the number of autoregressive steps required to generate one second of audio. In this paper, we conduct ablation studies to examine the impact of frame rate, bitrate, and causality on codec reconstruction quality. Based on our findings, we introduce NanoCodec, a state-of-the-art audio codec that achieves high-quality compression at just 12.5 frames per second (FPS). NanoCodec outperforms related works across various bitrate ranges, establishing a new benchmark for low-latency and efficient Speech LLM training and inference.
>
---
#### [new 052] DINA: A Dual Defense Framework Against Internal Noise and External Attacks in Natural Language Processing
- **分类: cs.CR; cs.CL**

- **简介: 论文提出DINA框架，结合视觉噪声学习与对抗训练，解决内外部攻击，提升NLP模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.05671v1](http://arxiv.org/pdf/2508.05671v1)**

> **作者:** Ko-Wei Chuang; Hen-Hsen Huang; Tsai-Yen Li
>
> **备注:** 7 pages
>
> **摘要:** As large language models (LLMs) and generative AI become increasingly integrated into customer service and moderation applications, adversarial threats emerge from both external manipulations and internal label corruption. In this work, we identify and systematically address these dual adversarial threats by introducing DINA (Dual Defense Against Internal Noise and Adversarial Attacks), a novel unified framework tailored specifically for NLP. Our approach adapts advanced noisy-label learning methods from computer vision and integrates them with adversarial training to simultaneously mitigate internal label sabotage and external adversarial perturbations. Extensive experiments conducted on a real-world dataset from an online gaming service demonstrate that DINA significantly improves model robustness and accuracy compared to baseline models. Our findings not only highlight the critical necessity of dual-threat defenses but also offer practical strategies for safeguarding NLP systems in realistic adversarial scenarios, underscoring broader implications for fair and responsible AI deployment.
>
---
#### [new 053] Sample-efficient LLM Optimization with Reset Replay
- **分类: cs.LG; cs.CL**

- **简介: 论文提出LoRR插件，通过高重放训练和周期性重置缓解样本效率与primacy bias问题，结合监督微调与偏好损失提升数据利用，显著提升LLM在数学任务上的表现。**

- **链接: [http://arxiv.org/pdf/2508.06412v1](http://arxiv.org/pdf/2508.06412v1)**

> **作者:** Zichuan Liu; Jinyu Wang; Lei Song; Jiang Bian
>
> **摘要:** Recent advancements in post-training Large Language Models (LLMs), particularly through Reinforcement Learning (RL) and preference optimization methods, are key drivers for enhancing their reasoning capabilities. However, these methods are often plagued by low sample efficiency and a susceptibility to primacy bias, where overfitting to initial experiences degrades policy quality and damages the learning process. To address these challenges, we introduce LLM optimization with Reset Replay (LoRR), a general and powerful plugin designed to enhance sample efficiency in any preference-based optimization framework. LoRR core mechanism enables training at a high replay number, maximizing the utility of each collected data batch. To counteract the risk of overfitting inherent in high-replay training, LoRR incorporates a periodic reset strategy with reusing initial data, which preserves network plasticity. Furthermore, it leverages a hybrid optimization objective, combining supervised fine-tuning (SFT) and preference-based losses to further bolster data exploitation. Our extensive experiments demonstrate that LoRR significantly boosts the performance of various preference optimization methods on both mathematical and general reasoning benchmarks. Notably, an iterative DPO approach augmented with LoRR achieves comparable performance on challenging math tasks, outperforming some complex and computationally intensive RL-based algorithms. These findings highlight that LoRR offers a practical, sample-efficient, and highly effective paradigm for LLM finetuning, unlocking greater performance from limited data.
>
---
#### [new 054] Basic interactive algorithms: Preview
- **分类: cs.LO; cs.CL; math.LO; quant-ph**

- **简介: 论文预览基本交互算法的公理化，探讨其与物理论题的关系，提出非确定性/概率算法可视为基本算法，揭示算法扩展对图灵论题的影响。**

- **链接: [http://arxiv.org/pdf/2508.05798v1](http://arxiv.org/pdf/2508.05798v1)**

> **作者:** Yuri Gurevich
>
> **摘要:** This dialog paper offers a preview and provides a foretaste of an upcoming work on the axiomatization of basic interactive algorithms. The modern notion of algorithm was elucidated in the 1930s--1950s. It was axiomatized a quarter of a century ago as the notion of ``sequential algorithm'' or ``classical algorithm''; we prefer to call it ``basic algorithm" now. The axiomatization was used to show that for every basic algorithm there is a behaviorally equivalent abstract state machine. It was also used to prove the Church-Turing thesis as it has been understood by the logicians. Starting from the 1960s, the notion of algorithm has expanded -- probabilistic algorithms, quantum algorithms, etc. -- prompting introduction of a much more ambitious version of the Church-Turing thesis commonly known as the ``physical thesis.'' We emphasize the difference between the two versions of the Church-Turing thesis and illustrate how nondeterministic and probabilistic algorithms can be viewed as basic algorithms with appropriate oracles. The same view applies to quantum circuit algorithms and many other classes of algorithms.
>
---
#### [new 055] Do Ethical AI Principles Matter to Users? A Large-Scale Analysis of User Sentiment and Satisfaction
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文研究伦理AI原则对用户满意度的影响，通过分析10万条用户评论，发现七项伦理维度与满意度正相关，但不同用户类型（技术/非技术）及产品类型（开发平台/终端应用）存在差异，强调需考虑用户视角与上下文。**

- **链接: [http://arxiv.org/pdf/2508.05913v1](http://arxiv.org/pdf/2508.05913v1)**

> **作者:** Stefan Pasch; Min Chul Cha
>
> **摘要:** As AI systems become increasingly embedded in organizational workflows and consumer applications, ethical principles such as fairness, transparency, and robustness have been widely endorsed in policy and industry guidelines. However, there is still scarce empirical evidence on whether these principles are recognized, valued, or impactful from the perspective of users. This study investigates the link between ethical AI and user satisfaction by analyzing over 100,000 user reviews of AI products from G2. Using transformer-based language models, we measure sentiment across seven ethical dimensions defined by the EU Ethics Guidelines for Trustworthy AI. Our findings show that all seven dimensions are positively associated with user satisfaction. Yet, this relationship varies systematically across user and product types. Technical users and reviewers of AI development platforms more frequently discuss system-level concerns (e.g., transparency, data governance), while non-technical users and reviewers of end-user applications emphasize human-centric dimensions (e.g., human agency, societal well-being). Moreover, the association between ethical AI and user satisfaction is significantly stronger for non-technical users and end-user applications across all dimensions. Our results highlight the importance of ethical AI design from users' perspectives and underscore the need to account for contextual differences across user roles and product types.
>
---
#### [new 056] A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文为LLM基深度搜索代理提供系统综述，分析架构、优化、应用及评估，识别挑战并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2508.05668v1](http://arxiv.org/pdf/2508.05668v1)**

> **作者:** Yunjia Xi; Jianghao Lin; Yongzhao Xiao; Zheli Zhou; Rong Shan; Te Gao; Jiachen Zhu; Weiwen Liu; Yong Yu; Weinan Zhang
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on https://github.com/YunjiaXi/Awesome-Search-Agent-Papers.
>
---
#### [new 057] Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出一种基于patch-level CLIP嵌入的框架，将预训练多模态LLMs与扩散模型融合，解决直接训练或桥梁方法成本高的问题，通过轻量适配器实现高效可控图像生成。**

- **链接: [http://arxiv.org/pdf/2508.05954v1](http://arxiv.org/pdf/2508.05954v1)**

> **作者:** Han Lin; Jaemin Cho; Amir Zadeh; Chuan Li; Mohit Bansal
>
> **备注:** Project Page: https://bifrost-1.github.io
>
> **摘要:** There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices.
>
---
#### [new 058] AttriLens-Mol: Attribute Guided Reinforcement Learning for Molecular Property Prediction with Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出基于属性引导的强化学习框架AttriLens-Mol，用于分子性质预测，解决传统方法依赖人工提示的问题，通过奖励机制引导模型提取相关属性，提升预测精度与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.04748v1](http://arxiv.org/pdf/2508.04748v1)**

> **作者:** Xuan Lin; Long Chen; Yile Wang
>
> **备注:** 9 pages
>
> **摘要:** Large Language Models (LLMs) have shown promise in assisting molecular property prediction tasks but often rely on human-crafted prompts and chain-of-thought templates. While recent advanced large reasoning models like DeepSeek-R1 employ reinforcement learning for an extended ``thinking'' process, their reasoning can be verbose and lack relevance. We introduce AttriLens-Mol, an attribute-guided reinforcement learning framework for molecular property prediction with LLMs. AttriLens-Mol steers the model's reasoning by using: (1) a format reward encouraging attribute-based structured output, (2) a count reward to avoid enumerating irrelevant attributes, and (3) a rationality reward using advanced LLMs and RDKit to verify the relatedness of the generated attributes. This approach implicitly elicits the model's inherent knowledge of relevant molecular attributes during reasoning, enables making predictions for the molecular property more effectively. Experiments on both in-distribution and out-of-distribution datasets show that, training both 7B-size R1-Distilled-Qwen2.5 and R1-Distilled-LLaMA3.1 models on 4,000 samples with our proposed AttriLens-Mol method significantly boosts the performance, getting comparable or better results than supervised fine-tuning models (Mol-Instructions, ChemDFM, etc.) and advanced models (GPT-3.5, GPT-4o, DeepSeek-V3, DeepSeek-R1, etc.). Further, our extracted attributes for the target property, when used as features for an interpretable decision tree model, yield superior performance compared to attributes generated by prompting LLMs. This shows that AttriLens-Mol effectively elicits more relevant and predictive molecular attributes, leading to enhanced interpretability and performance for property prediction. We release the code in https://github.com/szu-tera/AttriLens-Mol.
>
---
#### [new 059] ThematicPlane: Bridging Tacit User Intent and Latent Spaces for Image Generation
- **分类: cs.HC; cs.AI; cs.CL; cs.CV; H.5.2; I.2.7**

- **简介: 论文提出ThematicPlane系统，通过交互式主题平面连接隐性创意意图与系统控制，解决非专家图像生成中意图对齐难题，促进迭代创作。**

- **链接: [http://arxiv.org/pdf/2508.06065v1](http://arxiv.org/pdf/2508.06065v1)**

> **作者:** Daniel Lee; Nikhil Sharma; Donghoon Shin; DaEun Choi; Harsh Sharma; Jeonghwan Kim; Heng Ji
>
> **摘要:** Generative AI has made image creation more accessible, yet aligning outputs with nuanced creative intent remains challenging, particularly for non-experts. Existing tools often require users to externalize ideas through prompts or references, limiting fluid exploration. We introduce ThematicPlane, a system that enables users to navigate and manipulate high-level semantic concepts (e.g., mood, style, or narrative tone) within an interactive thematic design plane. This interface bridges the gap between tacit creative intent and system control. In our exploratory study (N=6), participants engaged in divergent and convergent creative modes, often embracing unexpected results as inspiration or iteration cues. While they grounded their exploration in familiar themes, differing expectations of how themes mapped to outputs revealed a need for more explainable controls. Overall, ThematicPlane fosters expressive, iterative workflows and highlights new directions for intuitive, semantics-driven interaction in generative design tools.
>
---
#### [new 060] Position: Intelligent Coding Systems Should Write Programs with Justifications
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 论文提出智能编码系统需生成代码并提供解释，解决AI决策不透明问题，强调认知对齐与语义忠实，并探索神经符号方法实现一致性检查。**

- **链接: [http://arxiv.org/pdf/2508.06017v1](http://arxiv.org/pdf/2508.06017v1)**

> **作者:** Xiangzhe Xu; Shiwei Feng; Zian Su; Chengpeng Wang; Xiangyu Zhang
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** Intelligent coding systems are transforming software development by enabling users to specify code behavior in natural language. However, the opaque decision-making of AI-driven coders raises trust and usability concerns, particularly for non-expert users who cannot inspect low-level implementations. We argue that these systems should not only generate code but also produce clear, consistent justifications that bridge model reasoning and user understanding. To this end, we identify two critical justification properties-cognitive alignment and semantic faithfulness-and highlight the limitations of existing methods, including formal verification, static analysis, and post-hoc explainability. We advocate exploring neuro-symbolic approaches for justification generation, where symbolic constraints guide model behavior during training and program semantics are enriched through neural representations, enabling automated consistency checks at inference time.
>
---
#### [new 061] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System
- **分类: cs.CR; cs.CL**

- **简介: 论文提出Fact2Fiction攻击框架，针对自主事实核查系统，通过模仿分解策略与生成解释制造恶意证据，提升攻击成功率，揭示系统安全漏洞并提出防御建议。**

- **链接: [http://arxiv.org/pdf/2508.06059v1](http://arxiv.org/pdf/2508.06059v1)**

> **作者:** Haorui He; Yupeng Li; Bin Benjamin Zhu; Dacheng Wen; Reynold Cheng; Francis C. M. Lau
>
> **摘要:** State-of-the-art fact-checking systems combat misinformation at scale by employing autonomous LLM-based agents to decompose complex claims into smaller sub-claims, verify each sub-claim individually, and aggregate the partial results to produce verdicts with justifications (explanatory rationales for the verdicts). The security of these systems is crucial, as compromised fact-checkers, which tend to be easily underexplored, can amplify misinformation. This work introduces Fact2Fiction, the first poisoning attack framework targeting such agentic fact-checking systems. Fact2Fiction mirrors the decomposition strategy and exploits system-generated justifications to craft tailored malicious evidences that compromise sub-claim verification. Extensive experiments demonstrate that Fact2Fiction achieves 8.9\%--21.2\% higher attack success rates than state-of-the-art attacks across various poisoning budgets. Fact2Fiction exposes security weaknesses in current fact-checking systems and highlights the need for defensive countermeasures.
>
---
#### [new 062] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization
- **分类: cs.AI; cs.CL**

- **简介: 论文提出基于AEPO的框架，通过多答案生成和AER函数优化探索，提升GUI接地中的语义对齐，实现9%的性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05731v1](http://arxiv.org/pdf/2508.05731v1)**

> **作者:** Yuhang Liu; Zeyu Liu; Shuanghe Zhu; Pengxiang Li; Congkai Xie; Jiasheng Wang; Xueyu Hu; Xiaotian Han; Jianbo Yuan; Xinyao Wang; Shengyu Zhang; Hongxia Yang; Fei Wu
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has propelled the development of autonomous agents that operate on Graphical User Interfaces (GUIs) using pure visual input. A fundamental challenge is robustly grounding natural language instructions. This requires a precise spatial alignment, which accurately locates the coordinates of each element, and, more critically, a correct semantic alignment, which matches the instructions to the functionally appropriate UI element. Although Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be effective at improving spatial alignment for these MLLMs, we find that inefficient exploration bottlenecks semantic alignment, which prevent models from learning difficult semantic associations. To address this exploration problem, we present Adaptive Exploration Policy Optimization (AEPO), a new policy optimization framework. AEPO employs a multi-answer generation strategy to enforce broader exploration, which is then guided by a theoretically grounded Adaptive Exploration Reward (AER) function derived from first principles of efficiency eta=U/C. Our AEPO-trained models, InfiGUI-G1-3B and InfiGUI-G1-7B, establish new state-of-the-art results across multiple challenging GUI grounding benchmarks, achieving significant relative improvements of up to 9.0% against the naive RLVR baseline on benchmarks designed to test generalization and semantic understanding. Resources are available at https://github.com/InfiXAI/InfiGUI-G1.
>
---
## 更新

#### [replaced 001] Structural Embedding Projection for Contextual Large Language Model Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18826v2](http://arxiv.org/pdf/2501.18826v2)**

> **作者:** Vincent Enoasmo; Cedric Featherstonehaugh; Xavier Konstantinopoulos; Zacharias Huntington
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Structured embedding transformations offer a promising approach for enhancing the efficiency and coherence of language model inference. The introduction of Structural Embedding Projection (SEP) provides a mechanism for refining token representations through projection matrices that integrate hierarchical and relational dependencies. The mathematical formulation of SEP enables embedding spaces to capture structured contextual relationships, thereby improving semantic fidelity without significantly increasing computational overhead. Experimental evaluations conducted on a range of linguistic datasets revealed that SEP contributed to reductions in perplexity and enhanced contextual coherence, demonstrating its potential to refine language model outputs. Computational efficiency assessments highlighted variations across different datasets, suggesting that the integration of structured embeddings introduced dataset-dependent trade-offs between inference speed and representational richness. The qualitative analysis of generated responses indicated that SEP enhanced narrative consistency and topic alignment, leading to improved fluency in multi-sentence text generation. The modifications to embedding layers required precise optimization to ensure stable training dynamics, as the introduction of structured transformations altered the traditional representation-learning process. The architectural adjustments necessary for SEP implementation influenced inference latency and memory consumption, requiring a balance between efficiency gains and additional processing demands. The impact of SEP on lexical diversity suggested that embedding modifications influenced the model's vocabulary usage, reflecting a more context-aware selection of generated tokens.
>
---
#### [replaced 002] Autonomous Structural Memory Manipulation for Large Language Models Using Hierarchical Embedding Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.14119v2](http://arxiv.org/pdf/2501.14119v2)**

> **作者:** Derek Yotheringhay; Alistair Kirkland; Humphrey Kirkbride; Josiah Whitesteeple
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Transformative innovations in model architectures have introduced hierarchical embedding augmentation as a means to redefine the representation of tokens through multi-level semantic structures, offering enhanced adaptability to complex linguistic inputs. Autonomous structural memory manipulation further advances this paradigm through dynamic memory reallocation mechanisms that prioritize critical contextual features while suppressing less relevant information, enabling scalable and efficient performance across diverse tasks. Experimental results reveal substantial improvements in computational efficiency, with marked reductions in processing overhead for longer input sequences, achieved through memory reorganization strategies that adapt to evolving contextual requirements. Hierarchical embeddings not only improved contextual alignment but also facilitated task generalization by capturing relationships at varying semantic granularities, ensuring coherence across layers without introducing significant computational redundancies. Comparative analysis against baseline models demonstrated unique advantages in accuracy, efficiency, and interpretability, particularly in tasks requiring complex contextual understanding or domain-specific adaptability. The ability to dynamically adjust token representations and memory configurations contributed to the model's robustness under varied and unpredictable input conditions. Applications benefiting from these advancements include multi-domain generalization, interactive systems, and scenarios involving real-time decision-making, where traditional static memory architectures often face limitations. The proposed methodology combines advanced embedding and memory management strategies into a cohesive framework that addresses scalability challenges while preserving task-specific relevance.
>
---
#### [replaced 003] Structured Convergence in Large Language Model Representations via Hierarchical Latent Space Folding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.08947v2](http://arxiv.org/pdf/2502.08947v2)**

> **作者:** Fenella Harcourt; Naderdel Piero; Gilbert Sutherland; Daphne Holloway; Harriet Bracknell; Julian Ormsby
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Token representations in high-dimensional latent spaces often exhibit redundancy, limiting computational efficiency and reducing structural coherence across model layers. Hierarchical latent space folding introduces a structured transformation mechanism that enforces a multi-scale organization within learned embeddings, refining representational compactness while preserving essential contextual distinctions. The proposed approach incorporates dynamic folding operations that iteratively adjust token embeddings through structured transformations, influencing both short-range and long-range dependencies in sequential processing tasks. Empirical evaluation demonstrates a reduction in representational variance across layers, contributing to more stable perplexity distributions and enhancing predictive confidence in text generation. The structured redistribution of attention head utilization leads to more efficient allocation of computational resources, particularly in deeper layers, where hierarchical refinements improve contextual abstraction. Comparative analysis of activation sparsity patterns suggests that hierarchical adjustments selectively reinforce critical pathways while reducing computational overhead in non-essential regions of the model. Statistical assessments of token reordering frequencies reveal that hierarchical modifications introduce subtle shifts in sequential dependencies, improving contextual alignment while maintaining syntactic correctness. Computational trade-offs associated with hierarchical folding introduce marginal increases in training time per epoch, yet empirical findings indicate that inference efficiency benefits from the structured representation adjustments. The results highlight the impact of hierarchical latent space folding on optimizing model performance through improved representation structuring and computational efficiency.
>
---
#### [replaced 004] Rank1: Test-Time Compute for Reranking in Information Retrieval
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.18418v2](http://arxiv.org/pdf/2502.18418v2)**

> **作者:** Orion Weller; Kathryn Ricci; Eugene Yang; Andrew Yates; Dawn Lawrie; Benjamin Van Durme
>
> **备注:** Published at CoLM 2025
>
> **摘要:** We introduce Rank1, the first reranking model trained to take advantage of test-time compute. Rank1 demonstrates the applicability within retrieval of using a reasoning language model (i.e. OpenAI's o1, Deepseek's R1, etc.) for distillation in order to rapidly improve the performance of a smaller model. We gather and open-source a dataset of more than 600,000 examples of R1 reasoning traces from queries and passages in MS MARCO. Models trained on this dataset show: (1) state-of-the-art performance on advanced reasoning and instruction following datasets; (2) work remarkably well out of distribution due to the ability to respond to user-input prompts; and (3) have explainable reasoning chains that can be given to users or RAG-based systems. Further, we demonstrate that quantized versions of these models retain strong performance while using less compute/memory. Overall, Rank1 shows that test-time compute allows for a fundamentally new type of explainable and performant reranker model for search.
>
---
#### [replaced 005] Bench-2-CoP: Can We Trust Benchmarking for EU AI Compliance?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05464v2](http://arxiv.org/pdf/2508.05464v2)**

> **作者:** Matteo Prandi; Vincenzo Suriani; Federico Pierucci; Marcello Galisai; Daniele Nardi; Piercosma Bisconti
>
> **摘要:** The rapid advancement of General Purpose AI (GPAI) models necessitates robust evaluation frameworks, especially with emerging regulations like the EU AI Act and its associated Code of Practice (CoP). Current AI evaluation practices depend heavily on established benchmarks, but these tools were not designed to measure the systemic risks that are the focus of the new regulatory landscape. This research addresses the urgent need to quantify this "benchmark-regulation gap." We introduce Bench-2-CoP, a novel, systematic framework that uses validated LLM-as-judge analysis to map the coverage of 194,955 questions from widely-used benchmarks against the EU AI Act's taxonomy of model capabilities and propensities. Our findings reveal a profound misalignment: the evaluation ecosystem dedicates the vast majority of its focus to a narrow set of behavioral propensities. On average, benchmarks devote 61.6% of their regulatory-relevant questions to "Tendency to hallucinate" and 31.2% to "Lack of performance reliability", while critical functional capabilities are dangerously neglected. Crucially, capabilities central to loss-of-control scenarios, including evading human oversight, self-replication, and autonomous AI development, receive zero coverage in the entire benchmark corpus. This study provides the first comprehensive, quantitative analysis of this gap, demonstrating that current public benchmarks are insufficient, on their own, for providing the evidence of comprehensive risk assessment required for regulatory compliance and offering critical insights for the development of next-generation evaluation tools.
>
---
#### [replaced 006] Contextually Entangled Gradient Mapping for Optimized LLM Comprehension
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00048v2](http://arxiv.org/pdf/2502.00048v2)**

> **作者:** Colin Sisate; Alistair Goldfinch; Vincent Waterstone; Sebastian Kingsley; Mariana Blackthorn
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Contextually Entangled Gradient Mapping (CEGM) introduces a new approach to gradient optimization, redefining the relationship between contextual embeddings and gradient updates to enhance semantic coherence and reasoning capabilities in neural architectures. By treating gradients as dynamic carriers of contextual dependencies rather than isolated numerical entities, the proposed methodology bridges critical gaps in existing optimization strategies. The integration of entangled gradient dynamics into a loss regularization framework demonstrated significant improvements in tasks involving long-form reasoning, contextual retention, and adaptability to unseen domains. Experimental evaluations showed that the CEGM-enhanced model consistently outperformed baseline approaches, achieving higher accuracy in token-level predictions and greater resilience to noisy inputs. Practical implementations involved modifications to training pipelines, introducing entanglement layers and dynamic coefficient adjustments that seamlessly align with existing architectures. Results further highlighted reductions in semantic drift during sequential transformations and improvements in embedding coherence across paraphrased sentences, showing the robustness and versatility of the proposed methodology. The findings demonstrate the broader implications of gradient entanglement for both theoretical advancements and practical applications in optimization strategies.
>
---
#### [replaced 007] One ruler to measure them all: Benchmarking multilingual long-context language models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01996v2](http://arxiv.org/pdf/2503.01996v2)**

> **作者:** Yekyung Kim; Jenna Russell; Marzena Karpinska; Mohit Iyyer
>
> **摘要:** We present ONERULER, a multilingual benchmark designed to evaluate long-context language models across 26 languages. ONERULER adapts the English-only RULER benchmark (Hsieh et al., 2024) by including seven synthetic tasks that test both retrieval and aggregation, including new variations of the "needle-in-a-haystack" task that allow for the possibility of a nonexistent needle. We create ONERULER through a two-step process, first writing English instructions for each task and then collaborating with native speakers to translate them into 25 additional languages. Experiments with both open-weight and closed LLMs reveal a widening performance gap between low- and high-resource languages as context length increases from 8K to 128K tokens. Surprisingly, English is not the top-performing language on long-context tasks (ranked 6th out of 26), with Polish emerging as the top language. Our experiments also show that many LLMs (particularly OpenAI's o3-mini-high) incorrectly predict the absence of an answer, even in high-resource languages. Finally, in cross-lingual scenarios where instructions and context appear in different languages, performance can fluctuate by up to 20% depending on the instruction language. We hope the release of ONERULER will facilitate future research into improving multilingual and cross-lingual long-context training pipelines.
>
---
#### [replaced 008] Exploring Synaptic Resonance in Large Language Models: A Novel Approach to Contextual Memory Integration
- **分类: cs.CL; cs.AI; cs.NE**

- **链接: [http://arxiv.org/pdf/2502.10699v2](http://arxiv.org/pdf/2502.10699v2)**

> **作者:** George Applegarth; Christian Weatherstone; Maximilian Hollingsworth; Henry Middlebrook; Marcus Irvin
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Contextual memory integration remains a high challenge in the development of language models, particularly in tasks that require maintaining coherence over extended sequences. Traditional approaches, such as self-attention mechanisms and memory-augmented architectures, often prioritize short-term dependencies, leading to fragmentation and inconsistency in long-range contextual understanding. Inspired by principles of synaptic plasticity observed in biological neural systems, a novel mechanism, Synaptic Resonance, is introduced to dynamically reinforce relevant memory pathways during training and inference. Unlike static memory representations, this mechanism continuously adjusts synaptic weight matrices based on contextual relevance, allowing for improved information retention without excessive computational overhead. Evaluations conducted on an open-source language model demonstrate reductions in perplexity, enhancements in contextual coherence, and increased robustness against input noise, highlighting the effectiveness of reinforcement-driven memory modulation. Comparative analysis against baseline models further reveals that the proposed approach achieves higher memory retention efficiency while maintaining computational feasibility. The architectural modifications integrate seamlessly into existing transformer-based frameworks, ensuring stable convergence and efficient inference without sacrificing scalability. Applications benefiting from improved long-term contextual consistency, such as dialogue systems and document summarization, stand to gain from this approach. Empirical findings suggest that dynamically reinforced memory pathways offer a promising alternative to conventional memory mechanisms, addressing longstanding limitations in extended sequence modeling.
>
---
#### [replaced 009] Self-Steering Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07081v2](http://arxiv.org/pdf/2504.07081v2)**

> **作者:** Gabriel Grand; Joshua B. Tenenbaum; Vikash K. Mansinghka; Alexander K. Lew; Jacob Andreas
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** While test-time reasoning enables language models (LMs) to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B or Qwen3-1.7B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. Our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs.
>
---
#### [replaced 010] From Next-Token to Mathematics: The Learning Dynamics of Mathematical Reasoning in Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.00900v2](http://arxiv.org/pdf/2407.00900v2)**

> **作者:** Shubhra Mishra; Gabriel Poesia; Noah D. Goodman
>
> **备注:** Accepted to COLM 2025. Dataset and code: https://github.com/gpoesia/mathcamps/
>
> **摘要:** Large Language Models (LLMs) solely trained on next-token prediction learn to solve a wide range of problems involving mathematical reasoning. But how does this ability evolve during training? We show the first analysis of how mathematical reasoning abilities of several open-weight LLMs develop during pre-training and post-training. To this end, we construct MathCAMPS, a synthetic dataset of novel mathematical reasoning problems grounded in 44 fine-grained skills taken from the Common Core curriculum from K to 8th grades. In one experiment, we show that mathematical skills are learned during pre-training in an order that measurably correlates with the human-designed curriculum, even though training data are randomly ordered. We also show a detailed analysis of which mathematical abilities benefit from instruction tuning, a widely used post-training method and, in contrast, which skills suffer. Our work paves the way for an empirical understanding of LLM training dynamics in relation to reasoning.
>
---
#### [replaced 011] Gradient-Regularized Latent Space Modulation in Large Language Models for Structured Contextual Synthesis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01979v2](http://arxiv.org/pdf/2502.01979v2)**

> **作者:** Derek Yotheringhay; Beatrix Nightingale; Maximilian Featherstone; Edmund Worthington; Hugo Ashdown
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Generating structured textual content requires mechanisms that enforce coherence, stability, and adherence to predefined constraints while maintaining semantic fidelity. Conventional approaches often rely on rule-based heuristics or fine-tuning strategies that lack flexibility and generalizability across diverse tasks. The incorporation of Gradient-Regularized Latent Space Modulation (GRLSM) introduces a novel paradigm for guiding text generation through the application of structured constraints within the latent space. The integration of gradient-based regularization mitigates abrupt variations in latent representations, ensuring a smoother encoding process that enhances structural consistency and logical progression within generated sequences. Comparative evaluations demonstrate that latent space modulation leads to a reduction in perplexity, increased coherence scores, and improved structural alignment across multiple domains. Stability assessments further indicate that the imposition of spectral norm constraints facilitates more controlled variations in generated text, preserving semantic consistency under input perturbations. Empirical results confirm that structured latent space constraints not only refine the organization of generated outputs but also enhance interpretability through more predictable and reliable synthesis patterns. Performance metrics illustrate that the GRLSM framework substantially reduces structural inconsistencies while preserving the generative flexibility inherent in neural models.
>
---
#### [replaced 012] The Devil Is in the Word Alignment Details: On Translation-Based Cross-Lingual Transfer for Token Classification Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10507v2](http://arxiv.org/pdf/2505.10507v2)**

> **作者:** Benedikt Ebing; Goran Glavaš
>
> **摘要:** Translation-based strategies for cross-lingual transfer XLT such as translate-train -- training on noisy target language data translated from the source language -- and translate-test -- evaluating on noisy source language data translated from the target language -- are competitive XLT baselines. In XLT for token classification tasks, however, these strategies include label projection, the challenging step of mapping the labels from each token in the original sentence to its counterpart(s) in the translation. Although word aligners (WAs) are commonly used for label projection, the low-level design decisions for applying them to translation-based XLT have not been systematically investigated. Moreover, recent marker-based methods, which project labeled spans by inserting tags around them before (or after) translation, claim to outperform WAs in label projection for XLT. In this work, we revisit WAs for label projection, systematically investigating the effects of low-level design decisions on token-level XLT: (i) the algorithm for projecting labels between (multi-)token spans, (ii) filtering strategies to reduce the number of noisily mapped labels, and (iii) the pre-tokenization of the translated sentences. We find that all of these substantially impact translation-based XLT performance and show that, with optimized choices, XLT with WA offers performance at least comparable to that of marker-based methods. We then introduce a new projection strategy that ensembles translate-train and translate-test predictions and demonstrate that it substantially outperforms the marker-based projection. Crucially, we show that our proposed ensembling also reduces sensitivity to low-level WA design choices, resulting in more robust XLT for token classification tasks.
>
---
#### [replaced 013] DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09349v2](http://arxiv.org/pdf/2506.09349v2)**

> **作者:** Chao-Hong Tan; Qian Chen; Wen Wang; Chong Deng; Qinglin Zhang; Luyao Cheng; Hai Yu; Xin Zhang; Xiang Lv; Tianyu Zhao; Chong Zhang; Yukun Ma; Yafeng Chen; Hui Wang; Jiaqing Liu; Jieping Ye
>
> **备注:** Work in progress
>
> **摘要:** Recent studies on end-to-end speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM's autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech-text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents DrVoice, a parallel speech-text voice conversation model based on joint autoregressive modeling, featuring dual-resolution speech representations. Whereas current methods utilize mainly 12.5Hz input audio representation, our proposed dual-resolution mechanism reduces the input frequency for the LLM to 5Hz. Experimental results on Spoken Question Answering benchmarks demonstrate that D RVOICE establishes new state-of-the-art (SOTA) performance among similar size speech foundation models with relative small amount of data.
>
---
#### [replaced 014] PaPaformer: Language Model from Pre-trained Parallel Paths
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00544v2](http://arxiv.org/pdf/2508.00544v2)**

> **作者:** Joonas Tapaninaho; Mourad Oussala
>
> **摘要:** The training of modern large-language models requires an increasingly amount of computation power and time. Even smaller variants, such as small-language models (SLMs), take several days to train in the best-case scenarios, often requiring multiple GPUs. This paper explores methods to train and evaluate decoder-only transformer-based language models in hours instead of days/weeks. We introduces \textit{PaPaformer}, a decoder-only transformer architecture variant, whose lower-dimensional parallel paths are combined into larger model. The paper shows that these lower-dimensional paths can be trained individually with different types of training data and then combined into one larger model. This method gives the option to reduce the total number of model parameters and the training time with increasing performance. Moreover, the use of parallel path structure opens interesting possibilities to customize paths to accommodate specific task requirements.
>
---
#### [replaced 015] Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01872v3](http://arxiv.org/pdf/2501.01872v3)**

> **作者:** Rachneet Sachdeva; Rima Hazra; Iryna Gurevych
>
> **备注:** Our code is publicly available at https://github.com/UKPLab/arxiv2025-poate-attack
>
> **摘要:** Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.
>
---
#### [replaced 016] Exploring Contextual Flux in Large Language Models: A Novel Approach to Self-Modulating Semantic Networks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10942v2](http://arxiv.org/pdf/2502.10942v2)**

> **作者:** Henry Evidail; Zachary Mountebank; Alistair Hathersage; Peter Stanhope; Basil Ravenscroft; Tobias Waddingham
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Self-modulating mechanisms introduce dynamic adaptation capabilities within language models through contextual realignment strategies that influence token embedding trajectories across extended sequences. Contextual Flux is explored as an approach to embedding modulation, integrating an auxiliary gating mechanism within the self-attention framework to dynamically adjust token representations based on evolving contextual dependencies. The empirical analysis evaluates entropy variations, latent space realignments, and coherence stability to assess the extent to which self-regulation enhances text generation consistency while preserving generative flexibility. Quantitative assessments suggest that embedding shifts contribute to more structured adaptation in long-form sequences, with measured reductions in redundant phrase repetitions and improvements in thematic retention. Variability in contextual weight computation affects modulation stability, leading to differing levels of adaptation across diverse linguistic structures. The computational demands introduced through real-time embedding reconfiguration are examined in relation to model scalability, emphasizing the need for optimization strategies in high-volume generative applications. The findings suggest that while adaptive embedding updates improve certain aspects of coherence, their impact remains contingent on model capacity and input complexity.
>
---
#### [replaced 017] Extract-and-Abstract: Unifying Extractive and Abstractive Summarization within Single Encoder-Decoder Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.11827v2](http://arxiv.org/pdf/2409.11827v2)**

> **作者:** Yuping Wu; Hao Li; Goran Nenadic; Xiao-Jun Zeng
>
> **摘要:** Extract-then-Abstract is a naturally coherent paradigm to conduct abstractive summarization with the help of salient information identified by the extractive model. Previous works that adopt this paradigm train the extractor and abstractor separately and introduce extra parameters to highlight the extracted salients to the abstractor, which results in error accumulation and additional training costs. In this paper, we first introduce a parameter-free highlight method into the encoder-decoder framework: replacing the encoder attention mask with a saliency mask in the cross-attention module to force the decoder to focus only on salient parts of the input. A preliminary analysis compares different highlight methods, demonstrating the effectiveness of our saliency mask. We further propose the novel extract-and-abstract paradigm, ExtAbs., which jointly and seamlessly performs Extractive and Abstractive summarization tasks within single encoder-decoder model to reduce error accumulation. In ExtAbs, the vanilla encoder is augmented to extract salients, and the vanilla decoder is modified with the proposed saliency mask to generate summaries. Built upon BART and PEGASUS, experiments on three datasets show that ExtAbs can achieve superior performance than baselines on the extractive task and performs comparable, or even better than the vanilla models on the abstractive task.
>
---
#### [replaced 018] The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.10970v4](http://arxiv.org/pdf/2501.10970v4)**

> **作者:** Nitay Calderon; Roi Reichart; Rotem Dror
>
> **摘要:** The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
>
---
#### [replaced 019] Architectural Fusion Through Contextual Partitioning in Large Language Models: A Novel Approach to Parameterized Knowledge Integration
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.12901v2](http://arxiv.org/pdf/2501.12901v2)**

> **作者:** Offa Kingsleigh; Alfred Abercrombie; David Woolstencroft; Beorhtric Meadowcroft; Marcus Irvin
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Contextual Partitioning introduces an innovative approach to enhancing the architectural design of large-scale computational models through the dynamic segmentation of parameters into context-aware regions. This methodology emphasizes the importance of task-specific specialization, achieved through adaptive parameter allocation mechanisms that align with the linguistic features of input data. Experimental evaluations demonstrated substantial improvements in accuracy, perplexity, and contextual coherence across a variety of linguistic tasks, highlighting the adaptability and scalability of the proposed framework. By reducing redundancy and enhancing computational efficiency, Contextual Partitioning not only streamlines model operations but also expands the scope of applications for advanced language processing systems. The approach operates autonomously, requiring no external fine-tuning, thereby addressing a significant limitation in conventional parameter optimization techniques. Empirical results demonstrate the effectiveness of gradient-driven segmentation, enabling models to dynamically recalibrate and specialize in response to task-specific demands. Furthermore, resource utilization metrics reveal notable reductions in memory usage and training times, confirming the efficiency of the approach. Observations from qualitative analyses illustrate improved contextual coherence and logical flow in generated outputs, reinforcing the practical value of this technique. The findings collectively demonstrate the potential for Contextual Partitioning to redefine the scalability and adaptability of computational language architectures in diverse and complex domains.
>
---
#### [replaced 020] Layers at Similar Depths Generate Similar Activations Across LLM Architectures
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08775v3](http://arxiv.org/pdf/2504.08775v3)**

> **作者:** Christopher Wolfram; Aaron Schein
>
> **摘要:** How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures.
>
---
#### [replaced 021] Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01700v2](http://arxiv.org/pdf/2503.01700v2)**

> **作者:** Yongchao Chen; Yilun Hao; Yang Zhang; Chuchu Fan
>
> **备注:** 7 pages, 7 figures, 3 tables
>
> **摘要:** Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website https://yongchao98.github.io/Code-Symbol-Planner/ for prompts, videos, and code.
>
---
#### [replaced 022] Nyay-Darpan: Enhancing Decision Making Through Summarization and Case Retrieval for Consumer Law in India
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06090v2](http://arxiv.org/pdf/2507.06090v2)**

> **作者:** Swapnil Bhattacharyya; Harshvivek Kashid; Shrey Ganatra; Spandan Anaokar; Shruti Nair; Reshma Sekhar; Siddharth Manohar; Rahul Hemrajani; Pushpak Bhattacharyya
>
> **摘要:** AI-based judicial assistance and case prediction have been extensively studied in criminal and civil domains, but remain largely unexplored in consumer law, especially in India. In this paper, we present Nyay-Darpan, a novel two-in-one framework that (i) summarizes consumer case files and (ii) retrieves similar case judgements to aid decision-making in consumer dispute resolution. Our methodology not only addresses the gap in consumer law AI tools but also introduces an innovative approach to evaluate the quality of the summary. The term 'Nyay-Darpan' translates into 'Mirror of Justice', symbolizing the ability of our tool to reflect the core of consumer disputes through precise summarization and intelligent case retrieval. Our system achieves over 75 percent accuracy in similar case prediction and approximately 70 percent accuracy across material summary evaluation metrics, demonstrating its practical effectiveness. We will publicly release the Nyay-Darpan framework and dataset to promote reproducibility and facilitate further research in this underexplored yet impactful domain.
>
---
#### [replaced 023] Topic Over Source: The Key to Effective Data Mixing for Language Models Pre-training
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16802v3](http://arxiv.org/pdf/2502.16802v3)**

> **作者:** Jiahui Peng; Xinlin Zhuang; Jiantao Qiu; Ren Ma; Jing Yu; He Zhu; Conghui He
>
> **摘要:** The performance of large language models (LLMs) is significantly affected by the quality and composition of their pre-training data, which is inherently diverse, spanning various languages, sources, and topics. Effectively integrating these heterogeneous data groups is crucial for optimizing LLM performance. Previous research has predominantly concentrated on source-based data mixing, often neglecting the nuanced topic-level characteristics of the data. To address this gap, we propose a topic-based data mixing strategy that utilizes detailed topic labels generated through a multi-stage process combining unsupervised clustering, LLM-based summarization, and supervised classifier training. With this strategy, we conduct the first comprehensive comparison of topic-based versus source-based partitioning across multiple mixing strategies. We demonstrate that language models pretrained on data mixed by topics consistently outperform those trained on data mixed by sources across multiple methods including RegMix, DoReMi,temperature-based sampling, and a manual mixing method based on downstream task performance. Our theoretical analysis reveals that topic-based data achieves significantly lower validation loss compared to source-based approaches, creating a better optimization landscape for model training. We will make our code, annotated datasets, and topic classification models publicly available to facilitate further research.
>
---
#### [replaced 024] Integrating large language models and active inference to understand eye movements in reading and dyslexia
- **分类: q-bio.NC; cs.CL**

- **链接: [http://arxiv.org/pdf/2308.04941v3](http://arxiv.org/pdf/2308.04941v3)**

> **作者:** Francesco Donnarumma; Mirco Frosolone; Giovanni Pezzulo
>
> **备注:** Main Document - 30 pages, 1 Table, 10 Figures + Supplementary 16 pages, 17 Tables
>
> **摘要:** We present a novel computational model employing hierarchical active inference to simulate reading and eye movements. The model characterizes linguistic processing as inference over a hierarchical generative model, facilitating predictions and inferences at various levels of granularity, from syllables to sentences. Our approach combines the strengths of large language models for realistic textual predictions and active inference for guiding eye movements to informative textual information, enabling the testing of predictions. The model exhibits proficiency in reading both known and unknown words and sentences, adhering to the distinction between lexical and nonlexical routes in dual route theories of reading. Our model therefore provides a novel approach to understand the cognitive processes underlying reading and eye movements, within a predictive processing framework. Furthermore, our model can potentially aid in understanding how maladaptive predictive processing can produce reading deficits associated with dyslexia. As a proof of concept, we show that attenuating the contribution of priors during the reading process leads to incorrect inferences and a more fragmented reading style, characterized by a greater number of shorter saccades, aligning with empirical findings regarding eye movements in dyslexic individuals. In summary, our model represents a significant advancement in comprehending the cognitive processes involved in reading and eye movements, with potential implications for understanding dyslexia in terms of maladaptive inference.
>
---
#### [replaced 025] Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.19028v3](http://arxiv.org/pdf/2506.19028v3)**

> **作者:** Weijie Xu; Yiwen Wang; Chi Xue; Xiangkun Hu; Xi Fang; Guimin Dong; Chandan K. Reddy
>
> **备注:** 29 pages, 9 figures, 15 tables
>
> **摘要:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
>
---
#### [replaced 026] Contextual Morphogenesis in Large Language Models: A Novel Approach to Self-Organizing Token Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00301v2](http://arxiv.org/pdf/2502.00301v2)**

> **作者:** Alistair Dombrowski; Beatrix Engelhardt; Dimitri Fairbrother; Henry Evidail
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Token representations influence the efficiency and adaptability of language models, yet conventional tokenization strategies impose rigid segmentation boundaries that do not adjust dynamically to evolving contextual relationships. The introduction of contextual morphogenesis establishes a self-organizing mechanism that restructures token boundaries based on learned contextual dependencies, allowing embeddings to evolve progressively across iterative processing steps. Empirical evaluations demonstrate that dynamically adjusted tokenization contributes to reductions in perplexity while maintaining representational stability, particularly in linguistically complex domains where static segmentation fails to capture nuanced dependencies. Computational trade-offs associated with self-organizing token structures indicate that additional processing overhead remains within feasible limits, provided that optimization strategies account for segmentation update efficiency. Comparative assessments across different linguistic corpora suggest that adaptive tokenization preserves interpretability while improving alignment with contextual cues, reinforcing the potential of morphogenetic segmentation mechanisms to refine predictive accuracy. Stability analyses confirm that evolving token structures maintain consistent segmentation behaviors across varied text distributions, ensuring that representational adaptations remain linguistically coherent. The effectiveness of contextual morphogenesis in refining structural stability and predictive performance highlights its viability as an alternative to traditional tokenization methods. Further analysis of computational efficiency considerations suggests that hybrid strategies integrating both static and dynamic segmentation techniques may offer a balanced approach to optimizing representational flexibility while maintaining inference efficiency.
>
---
#### [replaced 027] MyCulture: Exploring Malaysia's Diverse Culture under Low-Resource Language Constraints
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05429v2](http://arxiv.org/pdf/2508.05429v2)**

> **作者:** Zhong Ken Hew; Jia Xin Low; Sze Jue Yang; Chee Seng Chan
>
> **摘要:** Large Language Models (LLMs) often exhibit cultural biases due to training data dominated by high-resource languages like English and Chinese. This poses challenges for accurately representing and evaluating diverse cultural contexts, particularly in low-resource language settings. To address this, we introduce MyCulture, a benchmark designed to comprehensively evaluate LLMs on Malaysian culture across six pillars: arts, attire, customs, entertainment, food, and religion presented in Bahasa Melayu. Unlike conventional benchmarks, MyCulture employs a novel open-ended multiple-choice question format without predefined options, thereby reducing guessing and mitigating format bias. We provide a theoretical justification for the effectiveness of this open-ended structure in improving both fairness and discriminative power. Furthermore, we analyze structural bias by comparing model performance on structured versus free-form outputs, and assess language bias through multilingual prompt variations. Our evaluation across a range of regional and international LLMs reveals significant disparities in cultural comprehension, highlighting the urgent need for culturally grounded and linguistically inclusive benchmarks in the development and assessment of LLMs.
>
---
#### [replaced 028] Exploring Superior Function Calls via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05118v2](http://arxiv.org/pdf/2508.05118v2)**

> **作者:** Bingguang Hao; Maolin Wang; Zengzhuang Xu; Yicheng Chen; Cunyin Peng; Jinjie GU; Chenyi Zhuang
>
> **摘要:** Function calling capabilities are crucial for deploying Large Language Models in real-world applications, yet current training approaches fail to develop robust reasoning strategies. Supervised fine-tuning produces models that rely on superficial pattern matching, while standard reinforcement learning methods struggle with the complex action space of structured function calls. We present a novel reinforcement learning framework designed to enhance group relative policy optimization through strategic entropy based exploration specifically tailored for function calling tasks. Our approach addresses three critical challenges in function calling: insufficient exploration during policy learning, lack of structured reasoning in chain-of-thought generation, and inadequate verification of parameter extraction. Our two-stage data preparation pipeline ensures high-quality training samples through iterative LLM evaluation and abstract syntax tree validation. Extensive experiments on the Berkeley Function Calling Leaderboard demonstrate that this framework achieves state-of-the-art performance among open-source models with 86.02\% overall accuracy, outperforming standard GRPO by up to 6\% on complex multi-function scenarios. Notably, our method shows particularly strong improvements on code-pretrained models, suggesting that structured language generation capabilities provide an advantageous starting point for reinforcement learning in function calling tasks. We will release all the code, models and dataset to benefit the community.
>
---
#### [replaced 029] Pretraining on the Test Set Is No Longer All You Need: A Debate-Driven Approach to QA Benchmarks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17747v2](http://arxiv.org/pdf/2507.17747v2)**

> **作者:** Linbo Cao; Jinman Zhao
>
> **备注:** 22 pages, 7 figures. Accepted to COLM 2025. Code available at: github.com/l6cao/Debate-Driven-Evaluation
>
> **摘要:** As frontier language models increasingly saturate standard QA benchmarks, concerns about data contamination, memorization, and escalating dataset creation costs persist. We propose a debate-driven evaluation paradigm that transforms any existing QA dataset into structured adversarial debates--where one model is given the official answer to defend, and another constructs and defends an alternative answer--adjudicated by a judge model blind to the correct solution. By forcing multi-round argumentation, this approach substantially increases difficulty while penalizing shallow memorization, yet reuses QA items to reduce curation overhead. We make two main contributions: (1) an evaluation pipeline to systematically convert QA tasks into debate-based assessments, and (2) a public benchmark that demonstrates our paradigm's effectiveness on a subset of MMLU-Pro questions, complete with standardized protocols and reference models. Empirical results validate the robustness of the method and its effectiveness against data contamination--a Llama 3.1 model fine-tuned on test questions showed dramatic accuracy improvements (50% -> 82%) but performed worse in debates. Results also show that even weaker judges can reliably differentiate stronger debaters, highlighting how debate-based evaluation can scale to future, more capable systems while maintaining a fraction of the cost of creating new benchmarks. Overall, our framework underscores that "pretraining on the test set is no longer all you need," offering a sustainable path for measuring the genuine reasoning ability of advanced language models.
>
---
#### [replaced 030] Single-Pass Document Scanning for Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03101v2](http://arxiv.org/pdf/2504.03101v2)**

> **作者:** Weili Cao; Jianyou Wang; Youze Zheng; Longtian Bao; Qirui Zheng; Taylor Berg-Kirkpatrick; Ramamohan Paturi; Leon Bergen
>
> **备注:** Published at Conference on Language Modeling (COLM), 2025
>
> **摘要:** Handling extremely large documents for question answering is challenging: chunk-based embedding methods often lose track of important global context, while full-context transformers can be prohibitively expensive for hundreds of thousands of tokens. We propose a single-pass document scanning approach that processes the entire text in linear time, preserving global coherence while deciding which sentences are most relevant to the query. On 41 QA benchmarks, our single-pass scanner consistently outperforms chunk-based embedding methods and competes with large language models at a fraction of the computational cost. By conditioning on the entire preceding context without chunk breaks, the method preserves global coherence, which is especially important for long documents. Overall, single-pass document scanning offers a simple solution for question answering over massive text. All code, datasets, and model checkpoints are available at https://github.com/MambaRetriever/MambaRetriever
>
---
#### [replaced 031] MAATS: A Multi-Agent Automated Translation System Based on MQM Evaluation
- **分类: cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.14848v2](http://arxiv.org/pdf/2505.14848v2)**

> **作者:** George Wang; Jiaqian Hu; Safinah Ali
>
> **摘要:** We present MAATS, a Multi Agent Automated Translation System that leverages the Multidimensional Quality Metrics (MQM) framework as a fine-grained signal for error detection and refinement. MAATS employs multiple specialized AI agents, each focused on a distinct MQM category (e.g., Accuracy, Fluency, Style, Terminology), followed by a synthesis agent that integrates the annotations to iteratively refine translations. This design contrasts with conventional single-agent methods that rely on self-correction. Evaluated across diverse language pairs and Large Language Models (LLMs), MAATS outperforms zero-shot and single-agent baselines with statistically significant gains in both automatic metrics and human assessments. It excels particularly in semantic accuracy, locale adaptation, and linguistically distant language pairs. Qualitative analysis highlights its strengths in multi-layered error diagnosis, omission detection across perspectives, and context-aware refinement. By aligning modular agent roles with interpretable MQM dimensions, MAATS narrows the gap between black-box LLMs and human translation workflows, shifting focus from surface fluency to deeper semantic and contextual fidelity.
>
---
#### [replaced 032] CoAct-1: Computer-using Agents with Coding as Actions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03923v2](http://arxiv.org/pdf/2508.03923v2)**

> **作者:** Linxin Song; Yutong Dai; Viraj Prabhu; Jieyu Zhang; Taiwei Shi; Li Li; Junnan Li; Silvio Savarese; Zeyuan Chen; Jieyu Zhao; Ran Xu; Caiming Xiong
>
> **摘要:** Autonomous agents that operate computers via Graphical User Interfaces (GUIs) often struggle with efficiency and reliability on complex, long-horizon tasks. While augmenting these agents with planners can improve task decomposition, they remain constrained by the inherent limitations of performing all actions through GUI manipulation, leading to brittleness and inefficiency. In this work, we introduce a more robust and flexible paradigm: enabling agents to use coding as a enhanced action. We present CoAct-1, a novel multi-agent system that synergistically combines GUI-based control with direct programmatic execution. CoAct-1 features an Orchestrator that dynamically delegates subtasks to either a conventional GUI Operator or a specialized Programmer agent, which can write and execute Python or Bash scripts. This hybrid approach allows the agent to bypass inefficient GUI action sequences for tasks like file management and data processing, while still leveraging visual interaction when necessary. We evaluate our system on the challenging OSWorld benchmark, where CoAct-1 achieves a new state-of-the-art success rate of 60.76%, significantly outperforming prior methods. Furthermore, our approach dramatically improves efficiency, reducing the average number of steps required to complete a task to just 10.15, compared to 15 for leading GUI agents. Our results demonstrate that integrating coding as a core action provides a more powerful, efficient, and scalable path toward generalized computer automation.
>
---
#### [replaced 033] CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15254v2](http://arxiv.org/pdf/2504.15254v2)**

> **作者:** Anirudh Khatry; Robert Zhang; Jia Pan; Ziteng Wang; Qiaochu Chen; Greg Durrett; Isil Dillig
>
> **备注:** To be published at COLM, 2025
>
> **摘要:** C-to-Rust transpilation is essential for modernizing legacy C code while enhancing safety and interoperability with modern Rust ecosystems. However, no dataset currently exists for evaluating whether a system can transpile C into safe Rust that passes a set of test cases. We introduce CRUST-Bench, a dataset of 100 C repositories, each paired with manually-written interfaces in safe Rust as well as test cases that can be used to validate correctness of the transpilation. By considering entire repositories rather than isolated functions, CRUST-Bench captures the challenges of translating complex projects with dependencies across multiple files. The provided Rust interfaces provide explicit specifications that ensure adherence to idiomatic, memory-safe Rust patterns, while the accompanying test cases enforce functional correctness. We evaluate state-of-the-art large language models (LLMs) on this task and find that safe and idiomatic Rust generation is still a challenging problem for various state-of-the-art methods and techniques. We also provide insights into the errors LLMs usually make in transpiling code from C to safe Rust. The best performing model, OpenAI o1, is able to solve only 15 tasks in a single-shot setting. Improvements on CRUST-Bench would lead to improved transpilation systems that can reason about complex scenarios and help in migrating legacy codebases from C into languages like Rust that ensure memory safety. You can find the dataset and code at https://github.com/anirudhkhatry/CRUST-bench.
>
---
#### [replaced 034] Decompositional Reasoning for Graph Retrieval with Large Language Models
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.13380v2](http://arxiv.org/pdf/2506.13380v2)**

> **作者:** Valentin Six; Evan Dufraisse; Gaël de Chalendar
>
> **摘要:** Large Language Models (LLMs) excel at many NLP tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To tackle this problem, we propose a novel retrieval approach that integrates textual knowledge graphs into the LLM reasoning process via query decomposition. Our method decomposes complex questions into sub-questions, retrieves relevant textual subgraphs, and composes a question-specific knowledge graph to guide answer generation. For that, we use a weighted similarity function that focuses on both the complex question and the generated subquestions to extract a relevant subgraph, which allows efficient and precise retrieval for complex questions and improves the performance of LLMs on multi-hop QA tasks. This structured reasoning pipeline enhances factual grounding and interpretability while leveraging the generative strengths of LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls.
>
---
#### [replaced 035] Latent Structure Modulation in Large Language Models Through Stochastic Concept Embedding Transitions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05553v2](http://arxiv.org/pdf/2502.05553v2)**

> **作者:** Stefan Whitaker; Colin Sisate; Marcel Windsor; Nikolai Fairweather; Tarquin Goldborough; Oskar Lindenfeld
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Stochastic embedding transitions introduce a probabilistic mechanism for adjusting token representations dynamically during inference, mitigating the constraints imposed through static or deterministic embeddings. A transition framework was proposed in which each token embedding evolved through probabilistic updates, ensuring adaptability while preserving semantic integrity across linguistic contexts. Empirical evaluations demonstrated that models incorporating stochastic transitions exhibited greater lexical diversity, improved generative coherence, and enhanced retention of low-frequency vocabulary, contributing to more varied sentence structures and reduced reliance on high-probability token selections. Statistical analyses of embedding drift across transformer layers indicated that representations evolved more flexibly without losing coherence, supporting the hypothesis that controlled stochasticity facilitated context-sensitive representation learning. Experimental results revealed that probabilistic embeddings introduced minor computational overhead while maintaining generative efficiency, reinforcing their feasibility in large-scale applications. A comparative study with traditional embedding approaches highlighted measurable gains in text completion accuracy, dialogue coherence, and structural complexity, confirming the effectiveness of stochastic transitions in enhancing representation expressiveness. Clustering patterns in the embedding space suggested that probabilistic updates preserved meaningful semantic groupings while enabling context-driven shifts, further validating the stability of the transition mechanism. Performance metrics indicated that stochastic transitions balanced adaptability and control, ensuring that generative outputs remained linguistically coherent without excessive randomness.
>
---
#### [replaced 036] Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09614v2](http://arxiv.org/pdf/2505.09614v2)**

> **作者:** Anthony GX-Chen; Dongyan Lin; Mandana Samiei; Doina Precup; Blake A. Richards; Rob Fergus; Kenneth Marino
>
> **备注:** COLM 2025 Camera Ready
>
> **摘要:** Language model (LM) agents are increasingly used as autonomous decision-makers which need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established Blicket Test paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not child-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning.
>
---
#### [replaced 037] Context-Preserving Tensorial Reconfiguration in Large Language Model Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00246v2](http://arxiv.org/pdf/2502.00246v2)**

> **作者:** Larin Tonix; Morgana Baskerville; Nathaniel Stourton; Ophelia Tattershall
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Handling long-range dependencies in neural architectures has remained a persistent challenge due to computational limitations and inefficient contextual retention mechanisms. Tensorial operations have provided a foundation for restructuring model representations, yet conventional architectures have struggled to incorporate such techniques without introducing excessive complexity. A novel approach, Context-Preserving Tensorial Reconfiguration (CPTR), enables dynamic reorganization of weight tensors through structured factorization and adaptive contraction, allowing for enhanced contextual integration without substantial computational overhead. Empirical evaluations demonstrate that CPTR improves coherence retention across extended sequences, leading to measurable reductions in perplexity and improved recall accuracy for long-context tasks. Performance comparisons reveal that CPTR-enhanced models exhibit greater computational efficiency and reduced memory consumption while maintaining competitive language generation fluency and accuracy. Gradient stability metrics further validate the improved training efficiency, revealing more controlled variance in weight updates. Comparative studies across baseline and CPTR-enhanced models confirm that tensorial reconfiguration contributes to more stable and computationally efficient language modeling. The findings support the potential of CPTR in refining contemporary neural architectures for tasks requiring long-range contextual understanding and efficient memory utilization.
>
---
#### [replaced 038] Context-Aware Hierarchical Merging for Long Document Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00977v2](http://arxiv.org/pdf/2502.00977v2)**

> **作者:** Litu Ou; Mirella Lapata
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Hierarchical Merging is a technique commonly used to summarize very long texts ($>$100K tokens) by breaking down the input into smaller sections, summarizing those sections individually, and then merging or combining those summaries into a final coherent summary. Although it helps address the limitations of large language models (LLMs) with fixed input length constraints, the recursive merging process can amplify LLM hallucinations, increasing the risk of factual inaccuracies. In this paper, we seek to mitigate hallucinations by enriching hierarchical merging with context from the source document. Specifically, we propose different approaches to contextual augmentation ranging from \emph{replacing} intermediate summaries with relevant input context, to \emph{refining} them while using the context as supporting evidence, and \emph{aligning} them implicitly (via citations) to the input. Experimental results on datasets representing legal and narrative domains show that contextual augmentation consistently outperforms zero-shot and hierarchical merging baselines for the Llama 3.1 model family. Our analysis further reveals that refinement methods tend to perform best when paired with extractive summarization for identifying relevant input.
>
---
#### [replaced 039] Humans overrely on overconfident language models, across languages
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.06306v2](http://arxiv.org/pdf/2507.06306v2)**

> **作者:** Neil Rathi; Dan Jurafsky; Kaitlyn Zhou
>
> **备注:** camera ready
>
> **摘要:** As large language models (LLMs) are deployed globally, it is crucial that their responses are calibrated across languages to accurately convey uncertainty and limitations. Prior work shows that LLMs are linguistically overconfident in English, leading users to overrely on confident generations. However, the usage and interpretation of epistemic markers (e.g., 'I think it's') differs sharply across languages. Here, we study the risks of multilingual linguistic (mis)calibration, overconfidence, and overreliance across five languages to evaluate LLM safety in a global context. Our work finds that overreliance risks are high across languages. We first analyze the distribution of LLM-generated epistemic markers and observe that LLMs are overconfident across languages, frequently generating strengtheners even as part of incorrect responses. Model generations are, however, sensitive to documented cross-linguistic variation in usage: for example, models generate the most markers of uncertainty in Japanese and the most markers of certainty in German and Mandarin. Next, we measure human reliance rates across languages, finding that reliance behaviors differ cross-linguistically: for example, participants are significantly more likely to discount expressions of uncertainty in Japanese than in English (i.e., ignore their 'hedging' function and rely on generations that contain them). Taken together, these results indicate a high risk of reliance on overconfident model generations across languages. Our findings highlight the challenges of multilingual linguistic calibration and stress the importance of culturally and linguistically contextualized model safety evaluations.
>
---
#### [replaced 040] Are Your LLMs Capable of Stable Reasoning?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.13147v5](http://arxiv.org/pdf/2412.13147v5)**

> **作者:** Junnan Liu; Hongwei Liu; Linchen Xiao; Ziyi Wang; Kuikun Liu; Songyang Gao; Wenwei Zhang; Songyang Zhang; Kai Chen
>
> **备注:** ACL 2025 Camera, Benchmark: https://huggingface.co/datasets/opencompass/LiveMathBench, Code: https://github.com/open-compass/GPassK
>
> **摘要:** The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.
>
---
#### [replaced 041] INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.09105v2](http://arxiv.org/pdf/2406.09105v2)**

> **作者:** Chenwei Lin; Hanjia Lyu; Xian Xu; Jiebo Luo
>
> **备注:** To appear in the International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) and Multimodal Large Language Models (MLLMs) have demonstrated outstanding performance in various general multimodal applications and have shown increasing promise in specialized domains. However, their potential in the insurance domain-characterized by diverse application scenarios and rich multimodal data-remains largely underexplored. To date, there is no systematic review of multimodal tasks, nor a benchmark specifically designed to assess the capabilities of LVLMs in insurance. This gap hinders the development of LVLMs within the insurance industry. This study systematically reviews and categorizes multimodal tasks for 4 representative types of insurance: auto, property, health, and agricultural. We introduce INS-MMBench, the first hierarchical benchmark tailored for the insurance domain. INS-MMBench encompasses 22 fundamental tasks, 12 meta-tasks and 5 scenario tasks, enabling a comprehensive and progressive assessment from basic capabilities to real-world use cases. We benchmark 11 leading LVLMs, including closed-source models such as GPT-4o and open-source models like LLaVA. Our evaluation validates the effectiveness of INS-MMBench and offers detailed insights into the strengths and limitations of current LVLMs on a variety of insurance-related multimodal tasks. We hope that INS-MMBench will accelerate the integration of LVLMs into the insurance industry and foster interdisciplinary research. Our dataset and evaluation code are available at https://github.com/FDU-INS/INS-MMBench.
>
---
#### [replaced 042] CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis
- **分类: cs.PL; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.23145v2](http://arxiv.org/pdf/2503.23145v2)**

> **作者:** Anjiang Wei; Tarun Suresh; Jiannan Cao; Naveen Kannan; Yuheng Wu; Kai Yan; Thiago S. F. X. Teixeira; Ke Wang; Alex Aiken
>
> **摘要:** Inductive program synthesis, or programming by example, requires synthesizing functions from input-output examples that generalize to unseen inputs. While large language model agents have shown promise in programming tasks guided by natural language, their ability to perform inductive program synthesis is underexplored. Existing evaluation protocols rely on static sets of examples and held-out tests, offering no feedback when synthesized functions are incorrect and failing to reflect real-world scenarios such as reverse engineering. We propose CodeARC, the Code Abstraction and Reasoning Challenge, a new evaluation framework where agents interact with a hidden target function by querying it with new inputs, synthesizing candidate functions, and iteratively refining their solutions using a differential testing oracle. This interactive setting encourages agents to perform function calls and self-correction based on feedback. We construct the first large-scale benchmark for general-purpose inductive program synthesis, featuring 1114 functions. Among 18 models evaluated, o3-mini performs best with a success rate of 52.7%, highlighting the difficulty of this task. Fine-tuning LLaMA-3.1-8B-Instruct on curated synthesis traces yields up to a 31% relative performance gain. CodeARC provides a more realistic and challenging testbed for evaluating LLM-based program synthesis and inductive reasoning. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/CodeARC
>
---
#### [replaced 043] OpenCodeReasoning: Advancing Data Distillation for Competitive Coding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01943v2](http://arxiv.org/pdf/2504.01943v2)**

> **作者:** Wasi Uddin Ahmad; Sean Narenthiran; Somshubra Majumdar; Aleksander Ficek; Siddhartha Jain; Jocelyn Huang; Vahid Noroozi; Boris Ginsburg
>
> **备注:** Published at COLM 2025
>
> **摘要:** Since the advent of reasoning-based large language models, many have found great success from distilling reasoning capabilities into student models. Such techniques have significantly bridged the gap between reasoning and standard LLMs on coding tasks. Despite this, much of the progress on distilling reasoning models remains locked behind proprietary datasets or lacks details on data curation, filtering and subsequent training. To address this, we construct a superior supervised fine-tuning (SFT) dataset that we use to achieve state-of-the-art coding capability results in models of various sizes. Our distilled models use only SFT to achieve 61.8% on LiveCodeBench and 24.6% on CodeContests, surpassing alternatives trained with reinforcement learning. We then perform analysis on the data sources used to construct our dataset, the impact of code execution filtering, and the importance of instruction/solution diversity. We observe that execution filtering negatively affected benchmark accuracy, leading us to prioritize instruction diversity over solution correctness. Finally, we also analyze the token efficiency and reasoning patterns utilized by these models. We will open-source these datasets and distilled models to the community.
>
---
#### [replaced 044] Evaluation of LLMs in AMR Parsing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05028v2](http://arxiv.org/pdf/2508.05028v2)**

> **作者:** Shu Han Ho
>
> **备注:** 27 pages, 32 figures
>
> **摘要:** AMR (Abstract Meaning Representation) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
>
---
#### [replaced 045] Noosemia: toward a Cognitive and Phenomenological Account of Intentionality Attribution in Human-Generative AI Interaction
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.02622v2](http://arxiv.org/pdf/2508.02622v2)**

> **作者:** Enrico De Santis; Antonello Rizzi
>
> **备注:** This version has been extensively revised and revisited in light of feedback and further research. Several sections have been expanded or improved for greater clarity and completeness. Specifically, new clarification on complex system foundation related to Noosemia has been added (Secs. "2.4 and "2.5")
>
> **摘要:** This paper introduces and formalizes Noosem\`ia, a novel cognitive-phenomenological pattern emerging from human interaction with generative AI systems, particularly those enabling dialogic or multimodal exchanges. We propose a multidisciplinary framework to explain how, under certain conditions, users attribute intentionality, agency, and even interiority to these systems - a process grounded not in physical resemblance, but in linguistic performance, epistemic opacity, and emergent technological complexity. By linking an LLM declination of meaning holism to our technical notion of the LLM Contextual Cognitive Field, we clarify how LLMs construct meaning relationally and how coherence and a simulacrum of agency arise at the human-AI interface. The analysis situates noosemia alongside pareidolia, animism, the intentional stance and the uncanny valley, distinguishing its unique characteristics. We also introduce a-noosemia to describe the phenomenological withdrawal of such projections. The paper concludes with reflections on the broader philosophical, epistemological and social implications of noosemic dynamics and directions for future research.
>
---
#### [replaced 046] EvidenceBench: A Benchmark for Extracting Evidence from Biomedical Papers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.18736v2](http://arxiv.org/pdf/2504.18736v2)**

> **作者:** Jianyou Wang; Weili Cao; Kaicheng Wang; Xiaoyue Wang; Ashish Dalvi; Gino Prasad; Qishan Liang; Hsuan-lin Her; Ming Wang; Qin Yang; Gene W. Yeo; David E. Neal; Maxim Khan; Christopher D. Rosin; Ramamohan Paturi; Leon Bergen
>
> **备注:** Published at Conference on Language Modeling (COLM) 2025
>
> **摘要:** We study the task of automatically finding evidence relevant to hypotheses in biomedical papers. Finding relevant evidence is an important step when researchers investigate scientific hypotheses. We introduce EvidenceBench to measure models performance on this task, which is created by a novel pipeline that consists of hypothesis generation and sentence-by-sentence annotation of biomedical papers for relevant evidence, completely guided by and faithfully following existing human experts judgment. We demonstrate the pipeline's validity and accuracy with multiple sets of human-expert annotations. We evaluated a diverse set of language models and retrieval systems on the benchmark and found that model performances still fall significantly short of the expert level on this task. To show the scalability of our proposed pipeline, we create a larger EvidenceBench-100k with 107,461 fully annotated papers with hypotheses to facilitate model training and development. Both datasets are available at https://github.com/EvidenceBench/EvidenceBench
>
---
#### [replaced 047] Benchmarking LLMs on the Semantic Overlap Summarization Task
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.17008v2](http://arxiv.org/pdf/2402.17008v2)**

> **作者:** John Salvador; Naman Bansal; Mousumi Akter; Souvika Sarkar; Anupam Das; Shubhra Kanti Karmaker
>
> **摘要:** Semantic Overlap Summarization (SOS) is a constrained multi-document summarization task, where the constraint is to capture the common/overlapping information between two alternative narratives. In this work, we perform a benchmarking study of popular Large Language Models (LLMs) exclusively on the SOS task. Additionally, we introduce the PrivacyPolicyPairs (3P) dataset to expand the space of SOS benchmarks in terms of quantity and variety. This dataset provides 135 high-quality SOS data samples sourced from privacy policy documents. We then use a standard prompting taxonomy called TELeR to create and evaluate 905,216 distinct LLM-generated summaries over two SOS datasets from different domains, and we further conduct human evaluation on a subset of 540 samples. We conclude the paper by analyzing models' performances and the reliability of automatic evaluation. The code and datasets used to conduct this study are available at https://anonymous.4open.science/r/llm_eval-E16D.
>
---
#### [replaced 048] Reducibility among NP-Hard graph problems and boundary classes
- **分类: cs.CC; cs.CL; cs.DM**

- **链接: [http://arxiv.org/pdf/2411.14553v2](http://arxiv.org/pdf/2411.14553v2)**

> **作者:** Syed Mujtaba Hassan; Shahid Hussain; Abdul Samad
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Many NP-hard graph problems become easy for some classes of graphs. For example, coloring is easy for bipartite graphs, but NP-hard in general. So we can ask question like when does a hard problem become easy? What is the minimum substructure for which the problem remains hard? We use the notion of boundary classes to study such questions. In this paper, we introduce a method for transforming the boundary class of one NP-hard graph problem into a boundary class for another problem. If {\Pi} and {\Gamma} are two NP-hard graph problems where {\Pi} is reducible to {\Gamma}, we transform a boundary class of {\Pi} into a boundary class of {\Gamma}. More formally if {\Pi} is reducible to {\Gamma}, where the reduction satisfies certain conditions, then X is a boundary class of {\Pi} if and only if the image of X under the reduction is a boundary class of {\Gamma}. This gives us a relationship between boundary classes and reducibility among several NP-hard problems. To show the strength of our main result, we apply our theorem to obtain some previously unknown boundary classes for a few graph problems namely; vertex-cover, clique, traveling-salesperson, bounded-degree-spanning-tree, subgraph-isomorphism and clique-cover.
>
---
#### [replaced 049] Not All Data Are Unlearned Equally
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05058v5](http://arxiv.org/pdf/2504.05058v5)**

> **作者:** Aravind Krishnan; Siva Reddy; Marius Mosbach
>
> **摘要:** Machine unlearning is concerned with the task of removing knowledge learned from particular data points from a trained model. In the context of large language models (LLMs), unlearning has recently received increased attention, particularly for removing knowledge about named entities from models for privacy purposes. While various approaches have been proposed to address the unlearning problem, most existing approaches treat all data points to be unlearned equally, i.e., unlearning that Montreal is a city in Canada is treated exactly the same as unlearning the phone number of the first author of this paper. In this work, we show that this all data is equal assumption does not hold for LLM unlearning. We study how the success of unlearning depends on the frequency of the knowledge we want to unlearn in the pre-training data of a model and find that frequency strongly affects unlearning, i.e., more frequent knowledge is harder to unlearn. Additionally, we uncover a misalignment between probability and generation-based evaluations of unlearning and show that this problem worsens as models become larger. Overall, our experiments highlight the need for better evaluation practices and novel methods for LLM unlearning that take the training data of models into account.
>
---
#### [replaced 050] Can a Crow Hatch a Falcon? Lineage Matters in Predicting Large Language Model Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.19811v2](http://arxiv.org/pdf/2504.19811v2)**

> **作者:** Takuya Tamura; Taro Yano; Masafumi Enomoto; Masafumi Oyamada
>
> **摘要:** Accurately forecasting the performance of Large Language Models (LLMs) before extensive fine-tuning or merging can substantially reduce both computational expense and development time. Although prior approaches like scaling laws account for global factors such as parameter size or training tokens, they often overlook explicit lineage relationships-i.e., which models are derived or merged from which parents. In this work, we propose a novel Lineage-Regularized Matrix Factorization (LRMF) framework that encodes ancestral ties among LLMs via a graph Laplacian regularizer. By leveraging multi-hop parent-child connections, LRMF consistently outperforms conventional matrix factorization and collaborative filtering methods in both instance-level and benchmark-level performance prediction. Our large-scale study includes 2,934 publicly available Hugging Face models and 21,000+ instances across 6 major benchmarks, showing that the introduction of lineage constraints yields up to 0.15-0.30 higher Pearson correlation coefficients with actual performance compared to baseline methods. Moreover, LRMF effectively addresses the cold-start problem, providing accurate estimates for newly derived or merged models even with minimal data. This lineage-guided strategy thus offers a resource-efficient way to inform hyperparameter tuning, data selection, and model combination in modern LLM development.
>
---
#### [replaced 051] Neural Contextual Reinforcement Framework for Logical Structure Language Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.11417v2](http://arxiv.org/pdf/2501.11417v2)**

> **作者:** Marcus Irvin; William Cooper; Edward Hughes; Jessica Morgan; Christopher Hamilton
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** The Neural Contextual Reinforcement Framework introduces an innovative approach to enhancing the logical coherence and structural consistency of text generated by large language models. Leveraging reinforcement learning principles, the framework integrates custom reward functions and dynamic context alignment mechanisms to address challenges inherent in maintaining long-range dependencies across extended sequences. The architecture incorporates multi-head attention layers and hierarchical encoding modules, enabling the model to produce outputs that align closely with human expectations of logical structure and semantic flow. Quantitative evaluations across diverse datasets demonstrate substantial improvements in coherence metrics, perplexity reduction, and semantic alignment, showcasing the framework's ability to outperform baseline models in both general and domain-specific tasks. Qualitative analyses further highlight the framework's capacity to generate text with improved narrative clarity and reduced redundancy, reflecting its effectiveness in balancing fluency with structural precision. In addition to its performance gains, the framework exhibits robustness in handling noisy input data and scalability across varying model sizes, reinforcing its versatility in practical applications. Experimental results reveal that optimal context window sizes significantly influence coherence outcomes, showing the importance of architectural flexibility in adapting to diverse linguistic structures. Cross-lingual performance evaluations affirm the framework's adaptability to multiple languages, extending its utility beyond monolingual contexts. Resource efficiency analyses indicate a reduction in computational overhead compared to traditional approaches, emphasizing the practicality of the framework for large-scale deployment.
>
---
#### [replaced 052] Statistical Coherence Alignment for Large Language Model Representation Learning Through Tensor Field Convergence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09815v2](http://arxiv.org/pdf/2502.09815v2)**

> **作者:** Jonathan Gale; Godfrey Aldington; Harriet Thistlewood; Thomas Tattershall; Basil Wentworth; Vincent Enoasmo
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Representation learning plays a central role in structuring internal embeddings to capture the statistical properties of language, influencing the coherence and contextual consistency of generated text. Statistical Coherence Alignment is introduced as a method to enforce structured token representations through tensor field convergence, guiding embeddings to reflect statistical dependencies inherent in linguistic data. A mathematical framework is established to quantify coherence alignment, integrating a loss function that optimizes representational consistency across training iterations. Empirical evaluations demonstrate that applying coherence constraints improves perplexity, enhances classification accuracy, and refines rare word embeddings, contributing to a more stable representation space. Comparative analyses with baseline models reveal that the proposed method fosters a more interpretable internal structure, ensuring that embeddings retain contextual dependencies while mitigating representation collapse. The impact on coherence score distributions suggests that the alignment mechanism strengthens semantic integrity across diverse linguistic constructs, leading to a more balanced organization of learned embeddings. Computational assessments indicate that while the method introduces additional memory and training costs, the structured optimization process justifies the trade-offs in applications requiring heightened contextual fidelity. Experimental results validate the effectiveness of coherence alignment in optimizing token representations, providing insights into how statistical dependencies can be leveraged to improve language model training.
>
---
#### [replaced 053] ProsodyLM: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.20091v2](http://arxiv.org/pdf/2507.20091v2)**

> **作者:** Kaizhi Qian; Xulin Fan; Junrui Ni; Slava Shechtman; Mark Hasegawa-Johnson; Chuang Gan; Yang Zhang
>
> **摘要:** Speech language models refer to language models with speech processing and understanding capabilities. One key desirable capability for speech language models is the ability to capture the intricate interdependency between content and prosody. The existing mainstream paradigm of training speech language models, which converts speech into discrete tokens before feeding them into LLMs, is sub-optimal in learning prosody information -- we find that the resulting LLMs do not exhibit obvious emerging prosody processing capabilities via pre-training alone. To overcome this, we propose ProsodyLM, which introduces a simple tokenization scheme amenable to learning prosody. Each speech utterance is first transcribed into text, followed by a sequence of word-level prosody tokens. Compared with conventional speech tokenization schemes, the proposed tokenization scheme retains more complete prosody information, and is more understandable to text-based LLMs. We find that ProsodyLM can learn surprisingly diverse emerging prosody processing capabilities through pre-training alone, ranging from harnessing the prosody nuances in generated speech, such as contrastive focus, understanding emotion and stress in an utterance, to maintaining prosody consistency in long contexts.
>
---
#### [replaced 054] Structural Perturbation in Large Language Model Representations through Recursive Symbolic Regeneration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05794v2](http://arxiv.org/pdf/2502.05794v2)**

> **作者:** Kathlyn Eaglewood; Tobias Featherington; Dorian Mayfair; Sylvester Grimshaw; James Pettigrew
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Symbolic perturbations offer a novel approach for influencing neural representations without requiring direct modification of model parameters. The recursive regeneration of symbolic structures introduces structured variations in latent embeddings, leading to controlled shifts in attention dynamics and lexical diversity across sequential generations. A comparative analysis with conventional fine-tuning techniques reveals that structural modifications at the symbolic level induce distinct variations in contextual sensitivity while maintaining overall model fluency and coherence. Shifts in attention weight distributions highlight the role of symbolic modifications in adjusting token dependencies, influencing response variability, and refining long-form text generation. Experimental findings suggest that symbolic perturbations can enhance adaptability in domain-specific applications, allowing modifications in model behavior without retraining. Evaluations of semantic drift indicate that recursive regeneration alters long-range token dependencies, affecting topic coherence across extended text sequences. Results from lexical variability assessments further support the conclusion that symbolic-level modifications introduce interpretable variations in generated responses, potentially enabling more controlled stylistic adjustments in automated text generation.
>
---
#### [replaced 055] CUB: Benchmarking Context Utilisation Techniques for Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16518v2](http://arxiv.org/pdf/2505.16518v2)**

> **作者:** Lovisa Hagström; Youna Kim; Haeun Yu; Sang-goo Lee; Richard Johansson; Hyunsoo Cho; Isabelle Augenstein
>
> **备注:** 28 pages
>
> **摘要:** Incorporating external knowledge is crucial for knowledge-intensive tasks, such as question answering and fact checking. However, language models (LMs) may ignore relevant information that contradicts outdated parametric memory or be distracted by irrelevant contexts. While many context utilisation manipulation techniques (CMTs) have recently been proposed to alleviate these issues, few have seen systematic comparison. In this paper, we develop CUB (Context Utilisation Benchmark) - the first comprehensive benchmark designed to help practitioners within retrieval-augmented generation (RAG) diagnose CMTs under different context conditions. With this benchmark, we conduct the most extensive evaluation to date of seven state-of-the-art methods, representative of the main categories of CMTs, across three diverse datasets and tasks, applied to nine LMs. Our results reveal that most existing CMTs struggle to handle the full spectrum of context types encountered in real-world retrieval-augmented scenarios. We also find that many CMTs display inflated performance on simple synthesised datasets, compared to more realistic datasets with naturally occurring samples. Our findings expose critical gaps in current CMT evaluation practices and demonstrate the need for holistic testing and the development of CMTs that can robustly handle multiple context types.
>
---
#### [replaced 056] No Query, No Access
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07258v2](http://arxiv.org/pdf/2505.07258v2)**

> **作者:** Wenqiang Wang; Siyuan Liang; Yangshijie Zhang; Xiaojun Jia; Hao Lin; Xiaochun Cao
>
> **摘要:** Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary. Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/
>
---
#### [replaced 057] Towards Pareto Optimal Throughput in Small Language Model Serving
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.03353v3](http://arxiv.org/pdf/2404.03353v3)**

> **作者:** Pol G. Recasens; Yue Zhu; Chen Wang; Eun Kyung Lee; Olivier Tardieu; Alaa Youssef; Jordi Torres; Josep Ll. Berral
>
> **备注:** Revised version of the paper published at EuroMLSys'24, fix figure 6 and 7
>
> **摘要:** Large language models (LLMs) have revolutionized the state-of-the-art of many different natural language processing tasks. Although serving LLMs is computationally and memory demanding, the rise of Small Language Models (SLMs) offers new opportunities for resource-constrained users, who now are able to serve small models with cutting-edge performance. In this paper, we present a set of experiments designed to benchmark SLM inference at performance and energy levels. Our analysis provides a new perspective in serving, highlighting that the small memory footprint of SLMs allows for reaching the Pareto-optimal throughput within the resource capacity of a single accelerator. In this regard, we present an initial set of findings demonstrating how model replication can effectively improve resource utilization for serving SLMs.
>
---
#### [replaced 058] Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11881v2](http://arxiv.org/pdf/2502.11881v2)**

> **作者:** Hyunwoo Kim; Melanie Sclar; Tan Zhi-Xuan; Lance Ying; Sydney Levine; Yang Liu; Joshua B. Tenenbaum; Yejin Choi
>
> **备注:** COLM 2025. For code and data, see https://hyunw.kim/thought-tracing
>
> **摘要:** Existing LLM reasoning methods have shown impressive capabilities across various tasks, such as solving math and coding problems. However, applying these methods to scenarios without ground-truth answers or rule-based verification methods - such as tracking the mental states of an agent - remains challenging. Inspired by the sequential Monte Carlo algorithm, we introduce thought-tracing, an inference-time reasoning algorithm designed to trace the mental states of specific agents by generating hypotheses and weighting them based on observations without relying on ground-truth solutions to questions in datasets. Our algorithm is modeled after the Bayesian theory-of-mind framework, using LLMs to approximate probabilistic inference over agents' evolving mental states based on their perceptions and actions. We evaluate thought-tracing on diverse theory-of-mind benchmarks, demonstrating significant performance improvements compared to baseline LLMs. Our experiments also reveal interesting behaviors of the recent reasoning models - e.g., o3 and R1 - on theory-of-mind, highlighting the difference of social reasoning compared to other domains.
>
---
#### [replaced 059] Automated Privacy Information Annotation in Large Language Model Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20910v2](http://arxiv.org/pdf/2505.20910v2)**

> **作者:** Hang Zeng; Xiangyu Liu; Yong Hu; Chaoyue Niu; Fan Wu; Shaojie Tang; Guihai Chen
>
> **备注:** 8 content pages
>
> **摘要:** Users interacting with large language models (LLMs) under their real identifiers often unknowingly risk disclosing private information. Automatically notifying users whether their queries leak privacy and which phrases leak what private information has therefore become a practical need. Existing privacy detection methods, however, were designed for different objectives and application domains, typically tagging personally identifiable information (PII) in anonymous content, which is insufficient in real-name interaction scenarios with LLMs. In this work, to support the development and evaluation of privacy detection models for LLM interactions that are deployable on local user devices, we construct a large-scale multilingual dataset with 249K user queries and 154K annotated privacy phrases. In particular, we build an automated privacy annotation pipeline with strong LLMs to automatically extract privacy phrases from dialogue datasets and annotate leaked information. We also design evaluation metrics at the levels of privacy leakage, extracted privacy phrase, and privacy information. We further establish baseline methods using light-weight LLMs with both tuning-free and tuning-based methods, and report a comprehensive evaluation of their performance. Evaluation results reveal a gap between current performance and the requirements of real-world LLM applications, motivating future research into more effective local privacy detection methods grounded in our dataset.
>
---
#### [replaced 060] AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20160v2](http://arxiv.org/pdf/2506.20160v2)**

> **作者:** Ruosen Li; Ziming Luo; Quan Zhang; Ruochen Li; Ben Zhou; Ali Payani; Xinya Du
>
> **摘要:** Large reasoning models (LRMs) achieve impressive reasoning capabilities by generating lengthy chain-of-thoughts, but this "overthinking" incurs high latency and cost without commensurate accuracy gains. In this work, we introduce AALC, a lightweight, accuracy-aware length reward integrated into reinforcement learning that dynamically balances correctness and brevity during training. By incorporating validation accuracy into the reward and employing a smooth, dynamically scheduled length penalty, AALC delays length penalty until target performance is met. Through extensive experiments across standard and out-of-distribution math benchmarks, we show that our approach reduces response length by over 50% while maintaining or even improving the original accuracy. Furthermore, qualitative analysis reveals that our method curbs redundant reasoning patterns such as excessive subgoal setting and verification, leading to structurally refined outputs rather than naive truncation. We also identify that efficiency gains are accompanied by reduced interpretability: models trained with AALC omit some narrative framing and explanatory context. These findings highlight the potential of reward-based strategies to guide LRMs toward more efficient, generalizable reasoning paths.
>
---
#### [replaced 061] Contextual Reinforcement in Multimodal Token Compression for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.16658v2](http://arxiv.org/pdf/2501.16658v2)**

> **作者:** Naderdel Piero; Zacharias Cromwell; Nathaniel Wainwright; Matthias Nethercott
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Effective token compression remains a critical challenge for scaling models to handle increasingly complex and diverse datasets. A novel mechanism based on contextual reinforcement is introduced, dynamically adjusting token importance through interdependencies and semantic relevance. This approach enables substantial reductions in token usage while preserving the quality and coherence of information representation. Incorporating graph-based algorithms and adaptive weighting, the method captures subtle contextual relationships across textual and multimodal data, ensuring robust alignment and performance in downstream tasks. Evaluations across varied domains reveal significant improvements in accuracy and semantic retention, particularly for tasks requiring detailed cross-modal interactions. Memory usage analyses demonstrate improved computational efficiency, with minimal overhead despite the additional reinforcement processes. Performance gains are further validated through error distribution analyses, showing reduced semantic loss and syntactic inconsistencies compared to baseline models. The modular architecture ensures compatibility with a wide range of open-source frameworks, facilitating scalable implementation for real-world applications. These findings highlight the potential of contextual reinforcement in redefining token management strategies and advancing large-scale model design.
>
---
#### [replaced 062] From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v2](http://arxiv.org/pdf/2507.04996v2)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks.
>
---
#### [replaced 063] No Universal Prompt: Unifying Reasoning through Adaptive Prompting for Temporal Table Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11246v2](http://arxiv.org/pdf/2506.11246v2)**

> **作者:** Abhishek Rajgaria; Kushagra Dixit; Mayank Vyas; Harshavardhan Kalalbandi; Dan Roth; Vivek Gupta
>
> **备注:** 23 pages, 21 Tables, 10 Figures
>
> **摘要:** Temporal Table Reasoning is a critical challenge for Large Language Models (LLMs), requiring effective reasoning to extract relevant insights. Despite existence of multiple prompting methods, their impact on table reasoning remains largely unexplored. Furthermore, model performance varies drastically across different table and context structures, making it difficult to determine an optimal approach. This work investigates multiple prompting technique on diverse table types to determine that performance depends on factors such as entity type, table structure, requirement of additional context and question complexity, with "NO" single method consistently outperforming others. To address this, we introduce SEAR, an adaptive prompting framework inspired by human reasoning that dynamically adjusts to context and integrates structured reasoning. Our results demonstrate that SEAR achieves superior performance across all table types compared to baseline prompting techniques. Additionally, we explore the impact of table structure refactoring, finding that a unified representation enhances model reasoning.
>
---
#### [replaced 064] OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04030v2](http://arxiv.org/pdf/2504.04030v2)**

> **作者:** Wasi Uddin Ahmad; Aleksander Ficek; Mehrzad Samadi; Jocelyn Huang; Vahid Noroozi; Somshubra Majumdar; Boris Ginsburg
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering.
>
---
#### [replaced 065] Towards More Realistic Extraction Attacks: An Adversarial Perspective
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.02596v3](http://arxiv.org/pdf/2407.02596v3)**

> **作者:** Yash More; Prakhar Ganesh; Golnoosh Farnadi
>
> **备注:** To appear in TACL
>
> **摘要:** Language models are prone to memorizing their training data, making them vulnerable to extraction attacks. While existing research often examines isolated setups, such as a single model or a fixed prompt, real-world adversaries have a considerably larger attack surface due to access to models across various sizes and checkpoints, and repeated prompting. In this paper, we revisit extraction attacks from an adversarial perspective -- with multi-faceted access to the underlying data. We find significant churn in extraction trends, i.e., even unintuitive changes to the prompt, or targeting smaller models and earlier checkpoints, can extract distinct information. By combining multiple attacks, our adversary doubles ($2 \times$) the extraction risks, persisting even under mitigation strategies like data deduplication. We conclude with four case studies, including detecting pre-training data, copyright violations, extracting personally identifiable information, and attacking closed-source models, showing how our more realistic adversary can outperform existing adversaries in the literature.
>
---
#### [replaced 066] Structural Reformation of Large Language Model Neuron Encapsulation for Divergent Information Aggregation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07124v2](http://arxiv.org/pdf/2502.07124v2)**

> **作者:** Denis Bakushev; Gideon Boultinghouse; Harriet Oppenheimer; Sebastian Gillingwater; Valentina Ashington; Wilfred Stanborough
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship
>
> **摘要:** Structured neuron encapsulation introduces a modular framework that enables more effective aggregation and specialization of information within deep learning architectures. A model modified through this framework demonstrated improved perplexity scores, greater lexical variability, and enhanced consistency in logical reasoning, suggesting that structured parameter distribution contributes to more efficient language representation. Statistical analyses of generated text highlighted a wider range of sentence structures and reduced redundancy in token selection, indicating that encapsulation fosters more adaptable language generation. A detailed evaluation of attention weight distributions revealed that the experimental model exhibited greater divergence in cross-layer activations, supporting the hypothesis that encapsulated neurons assume specialized processing roles. Logical consistency assessments further demonstrated that modular architectures mitigate contradictory outputs, reducing internal conflicts in inferred relationships between linguistic constructs. Computational trade-offs were analyzed, with results showing a minor increase in processing overhead, though improvements in parameter efficiency and structured decision-making compensated for the additional complexity. The mathematical formulation of the encapsulation mechanism confirmed that modular aggregation maintains stable convergence properties while promoting distinct functional roles for different neuron clusters.
>
---

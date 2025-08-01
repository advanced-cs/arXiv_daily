# 自然语言处理 cs.CL

- **最新发布 107 篇**

- **更新 83 篇**

## 最新发布

#### [new 001] RoD-TAL: A Benchmark for Answering Questions in Romanian Driving License Exams
- **分类: cs.CL**

- **简介: 该论文属于多模态法律教育任务，旨在解决罗马尼亚语驾驶法规问答问题。作者构建了包含文本和图像题目的数据集RoD-TAL，并评估了大语言模型与视觉语言模型在信息检索、问答及视觉推理中的表现，发现领域微调与推理优化能提升性能，但视觉推理仍具挑战。**

- **链接: [http://arxiv.org/pdf/2507.19666v1](http://arxiv.org/pdf/2507.19666v1)**

> **作者:** Andrei Vlad Man; Răzvan-Alexandru Smădu; Cristian-George Craciun; Dumitru-Clementin Cercel; Florin Pop; Mihaela-Claudia Cercel
>
> **备注:** 49 pages, 52 figures
>
> **摘要:** The intersection of AI and legal systems presents a growing need for tools that support legal education, particularly in under-resourced languages such as Romanian. In this work, we aim to evaluate the capabilities of Large Language Models (LLMs) and Vision-Language Models (VLMs) in understanding and reasoning about Romanian driving law through textual and visual question-answering tasks. To facilitate this, we introduce RoD-TAL, a novel multimodal dataset comprising Romanian driving test questions, text-based and image-based, alongside annotated legal references and human explanations. We implement and assess retrieval-augmented generation (RAG) pipelines, dense retrievers, and reasoning-optimized models across tasks including Information Retrieval (IR), Question Answering (QA), Visual IR, and Visual QA. Our experiments demonstrate that domain-specific fine-tuning significantly enhances retrieval performance. At the same time, chain-of-thought prompting and specialized reasoning models improve QA accuracy, surpassing the minimum grades required to pass driving exams. However, visual reasoning remains challenging, highlighting the potential and the limitations of applying LLMs and VLMs to legal education.
>
---
#### [new 002] Memorization in Fine-Tuned Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究微调大语言模型中的记忆机制，旨在分析微调过程中影响模型记忆训练数据的因素，特别是在医疗领域中的隐私风险。通过成员推理攻击和生成任务评估记忆程度，探索不同权重矩阵、困惑度与LoRA秩对记忆的影响，揭示性能与隐私间的权衡。**

- **链接: [http://arxiv.org/pdf/2507.21009v1](http://arxiv.org/pdf/2507.21009v1)**

> **作者:** Danil Savine; Muni Sreenivas Pydi; Jamal Atif; Olivier Cappé
>
> **摘要:** This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events. Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning. Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks. These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns.
>
---
#### [new 003] Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于视觉语言模型（VLM）安全评估任务，旨在解决当前评估数据集偏重文本、忽视视觉漏洞的问题。作者提出Text2VLM，将纯文本数据转化为多模态数据，用于测试VLM对视觉提示攻击的鲁棒性，发现开源模型在引入视觉输入时更易受攻击，并通过人工评估验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.20704v1](http://arxiv.org/pdf/2507.20704v1)**

> **作者:** Gabriel Downer; Sean Craven; Damian Ruck; Jake Thomas
>
> **备注:** 9 pages, 9 figures. Jake Thomas served as Editor for this manuscript
>
> **摘要:** The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications.
>
---
#### [new 004] MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts?
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于代码语言模型安全性任务，旨在解决模型在面对多轮恶意编码提示时的脆弱性问题。作者提出了代码分解攻击方法和MOCHA基准，用于评估模型对恶意提示的鲁棒性，并验证了微调提升安全性的效果。**

- **链接: [http://arxiv.org/pdf/2507.19598v1](http://arxiv.org/pdf/2507.19598v1)**

> **作者:** Muntasir Wahed; Xiaona Zhou; Kiet A. Nguyen; Tianjiao Yu; Nirav Diwan; Gang Wang; Dilek Hakkani-Tür; Ismini Lourentzou
>
> **备注:** Winner Defender Team at Amazon Nova AI Challenge 2025
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly enhanced their code generation capabilities. However, their robustness against adversarial misuse, particularly through multi-turn malicious coding prompts, remains underexplored. In this work, we introduce code decomposition attacks, where a malicious coding task is broken down into a series of seemingly benign subtasks across multiple conversational turns to evade safety filters. To facilitate systematic evaluation, we introduce \benchmarkname{}, a large-scale benchmark designed to evaluate the robustness of code LLMs against both single-turn and multi-turn malicious prompts. Empirical results across open- and closed-source models reveal persistent vulnerabilities, especially under multi-turn scenarios. Fine-tuning on MOCHA improves rejection rates while preserving coding ability, and importantly, enhances robustness on external adversarial datasets with up to 32.4% increase in rejection rates without any additional supervision.
>
---
#### [new 005] Diversity-Enhanced Reasoning for Subjective Questions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型推理模型在主观问题回答中推理多样性不足的问题。通过构建多角色视角的多样性增强框架MultiRole-R1，结合强化学习与奖励塑形，提升模型在主观推理任务中的准确性和多样性。**

- **链接: [http://arxiv.org/pdf/2507.20187v1](http://arxiv.org/pdf/2507.20187v1)**

> **作者:** Yumeng Wang; Zhiyuan Fan; Jiayu Liu; Yi R. Fung
>
> **摘要:** Large reasoning models (LRM) with long chain-of-thought (CoT) capabilities have shown strong performance on objective tasks, such as math reasoning and coding. However, their effectiveness on subjective questions that may have different responses from different perspectives is still limited by a tendency towards homogeneous reasoning, introduced by the reliance on a single ground truth in supervised fine-tuning and verifiable reward in reinforcement learning. Motivated by the finding that increasing role perspectives consistently improves performance, we propose MultiRole-R1, a diversity-enhanced framework with multiple role perspectives, to improve the accuracy and diversity in subjective reasoning tasks. MultiRole-R1 features an unsupervised data construction pipeline that generates reasoning chains that incorporate diverse role perspectives. We further employ reinforcement learning via Group Relative Policy Optimization (GRPO) with reward shaping, by taking diversity as a reward signal in addition to the verifiable reward. With specially designed reward functions, we successfully promote perspective diversity and lexical diversity, uncovering a positive relation between reasoning diversity and accuracy. Our experiment on six benchmarks demonstrates MultiRole-R1's effectiveness and generalizability in enhancing both subjective and objective reasoning, showcasing the potential of diversity-enhanced training in LRMs.
>
---
#### [new 006] KLAAD: Refining Attention Mechanisms to Reduce Societal Bias in Generative Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在减少生成语言模型中的社会偏见。作者提出了KLAAD框架，通过注意力机制对齐刻板印象与反刻板印象句子的注意力分布，结合交叉熵、KL散度和三元组损失进行训练。实验表明该方法在BBQ和BOLD基准上有效降低偏见，同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2507.19962v1](http://arxiv.org/pdf/2507.19962v1)**

> **作者:** Seorin Kim; Dongyoung Lee; Jaejin Lee
>
> **摘要:** Large language models (LLMs) often exhibit societal biases in their outputs, prompting ethical concerns regarding fairness and harm. In this work, we propose KLAAD (KL-Attention Alignment Debiasing), an attention-based debiasing framework that implicitly aligns attention distributions between stereotypical and anti-stereotypical sentence pairs without directly modifying model weights. KLAAD introduces a composite training objective combining Cross-Entropy, KL divergence, and Triplet losses, guiding the model to consistently attend across biased and unbiased contexts while preserving fluency and coherence. Experimental evaluation of KLAAD demonstrates improved bias mitigation on both the BBQ and BOLD benchmarks, with minimal impact on language modeling quality. The results indicate that attention-level alignment offers a principled solution for mitigating bias in generative language models.
>
---
#### [new 007] Goal Alignment in LLM-Based User Simulators for Conversational AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话式AI任务，旨在解决大型语言模型在多轮对话中难以保持目标导向行为的问题。作者提出了用户目标状态追踪（UGST）框架，并设计了三阶段方法来开发能自主追踪目标进展的用户模拟器，从而提升目标对齐能力。**

- **链接: [http://arxiv.org/pdf/2507.20152v1](http://arxiv.org/pdf/2507.20152v1)**

> **作者:** Shuhaib Mehri; Xiaocheng Yang; Takyoung Kim; Gokhan Tur; Shikib Mehri; Dilek Hakkani-Tür
>
> **摘要:** User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators.
>
---
#### [new 008] Text2Vis: A Challenging and Diverse Benchmark for Generating Multimodal Visualizations from Text
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文本生成可视化任务，旨在解决缺乏全面评估基准的问题。作者构建了Text2Vis数据集，包含1,985个样本，涵盖20多种图表类型和复杂数据科学查询。他们提出跨模态框架提升生成效果，并开发自动化评估系统。论文工作推动了文本到可视化模型的发展与评估标准化。**

- **链接: [http://arxiv.org/pdf/2507.19969v1](http://arxiv.org/pdf/2507.19969v1)**

> **作者:** Mizanur Rahman; Md Tahmid Rahman Laskar; Shafiq Joty; Enamul Hoque
>
> **摘要:** Automated data visualization plays a crucial role in simplifying data interpretation, enhancing decision-making, and improving efficiency. While large language models (LLMs) have shown promise in generating visualizations from natural language, the absence of comprehensive benchmarks limits the rigorous evaluation of their capabilities. We introduce Text2Vis, a benchmark designed to assess text-to-visualization models, covering 20+ chart types and diverse data science queries, including trend analysis, correlation, outlier detection, and predictive analytics. It comprises 1,985 samples, each with a data table, natural language query, short answer, visualization code, and annotated charts. The queries involve complex reasoning, conversational turns, and dynamic data retrieval. We benchmark 11 open-source and closed-source models, revealing significant performance gaps, highlighting key challenges, and offering insights for future advancements. To close this gap, we propose the first cross-modal actor-critic agentic framework that jointly refines the textual answer and visualization code, increasing GPT-4o`s pass rate from 26% to 42% over the direct approach and improving chart quality. We also introduce an automated LLM-based evaluation framework that enables scalable assessment across thousands of samples without human annotation, measuring answer correctness, code execution success, visualization readability, and chart accuracy. We release Text2Vis at https://github.com/vis-nlp/Text2Vis.
>
---
#### [new 009] Enhancing Hallucination Detection via Future Context
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成文本中的幻觉检测问题。通过采样未来上下文，提供有效线索，结合多种采样方法，提升幻觉检测效果。**

- **链接: [http://arxiv.org/pdf/2507.20546v1](http://arxiv.org/pdf/2507.20546v1)**

> **作者:** Joosung Lee; Cheonbok Park; Hwiyeol Jo; Jeonghoon Kim; Joonsuk Park; Kang Min Yoo
>
> **摘要:** Large Language Models (LLMs) are widely used to generate plausible text on online platforms, without revealing the generation process. As users increasingly encounter such black-box outputs, detecting hallucinations has become a critical challenge. To address this challenge, we focus on developing a hallucination detection framework for black-box generators. Motivated by the observation that hallucinations, once introduced, tend to persist, we sample future contexts. The sampled future contexts provide valuable clues for hallucination detection and can be effectively integrated with various sampling-based methods. We extensively demonstrate performance improvements across multiple methods using our proposed sampling approach.
>
---
#### [new 010] Modeling Professionalism in Expert Questioning through Linguistic Differentiation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在建模专家提问中的专业性。通过构建标注框架，提取金融分析师问题中的语言特征，构建数据集并训练分类器，以区分专家与大模型生成的问题，探索专业性的可学习性。**

- **链接: [http://arxiv.org/pdf/2507.20249v1](http://arxiv.org/pdf/2507.20249v1)**

> **作者:** Giulia D'Agostino; Chung-Chi Chen
>
> **摘要:** Professionalism is a crucial yet underexplored dimension of expert communication, particularly in high-stakes domains like finance. This paper investigates how linguistic features can be leveraged to model and evaluate professionalism in expert questioning. We introduce a novel annotation framework to quantify structural and pragmatic elements in financial analyst questions, such as discourse regulators, prefaces, and request types. Using both human-authored and large language model (LLM)-generated questions, we construct two datasets: one annotated for perceived professionalism and one labeled by question origin. We show that the same linguistic features correlate strongly with both human judgments and authorship origin, suggesting a shared stylistic foundation. Furthermore, a classifier trained solely on these interpretable features outperforms gemini-2.0 and SVM baselines in distinguishing expert-authored questions. Our findings demonstrate that professionalism is a learnable, domain-general construct that can be captured through linguistically grounded modeling.
>
---
#### [new 011] Ontology-Enhanced Knowledge Graph Completion using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱补全任务，旨在解决现有基于大语言模型的方法因隐式知识表示导致的推理不确定性问题。作者提出OL-KGC方法，融合本体知识与神经结构信息，通过结构嵌入和自动抽取逻辑规则来提升补全效果，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.20643v1](http://arxiv.org/pdf/2507.20643v1)**

> **作者:** Wenbin Guo; Xin Wang; Jiaoyan Chen; Zhao Li; Zirui Chen
>
> **摘要:** Large Language Models (LLMs) have been extensively adopted in Knowledge Graph Completion (KGC), showcasing significant research advancements. However, as black-box models driven by deep neural architectures, current LLM-based KGC methods rely on implicit knowledge representation with parallel propagation of erroneous knowledge, thereby hindering their ability to produce conclusive and decisive reasoning outcomes. We aim to integrate neural-perceptual structural information with ontological knowledge, leveraging the powerful capabilities of LLMs to achieve a deeper understanding of the intrinsic logic of the knowledge. We propose an ontology enhanced KGC method using LLMs -- OL-KGC. It first leverages neural perceptual mechanisms to effectively embed structural information into the textual space, and then uses an automated extraction algorithm to retrieve ontological knowledge from the knowledge graphs (KGs) that needs to be completed, which is further transformed into a textual format comprehensible to LLMs for providing logic guidance. We conducted extensive experiments on three widely-used benchmarks -- FB15K-237, UMLS and WN18RR. The experimental results demonstrate that OL-KGC significantly outperforms existing mainstream KGC methods across multiple evaluation metrics, achieving state-of-the-art performance.
>
---
#### [new 012] SessionIntentBench: A Multi-task Inter-session Intention-shift Modeling Benchmark for E-commerce Customer Behavior Understanding
- **分类: cs.CL**

- **简介: 该论文属于电商用户行为理解任务，旨在解决跨会话用户意图建模不足的问题。作者提出了一个意图树概念和多模态基准SessionIntentBench，包含四种子任务，通过10,905个会话挖掘超过1300万任务，评估大型语言模型对意图变化的理解能力，并验证引入意图可提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.20185v1](http://arxiv.org/pdf/2507.20185v1)**

> **作者:** Yuqi Yang; Weiqi Wang; Baixuan Xu; Wei Fan; Qing Zong; Chunkit Chan; Zheye Deng; Xin Liu; Yifan Gao; Changlong Yu; Chen Luo; Yang Li; Zheng Li; Qingyu Yin; Bing Yin; Yangqiu Song
>
> **摘要:** Session history is a common way of recording user interacting behaviors throughout a browsing activity with multiple products. For example, if an user clicks a product webpage and then leaves, it might because there are certain features that don't satisfy the user, which serve as an important indicator of on-the-spot user preferences. However, all prior works fail to capture and model customer intention effectively because insufficient information exploitation and only apparent information like descriptions and titles are used. There is also a lack of data and corresponding benchmark for explicitly modeling intention in E-commerce product purchase sessions. To address these issues, we introduce the concept of an intention tree and propose a dataset curation pipeline. Together, we construct a sibling multimodal benchmark, SessionIntentBench, that evaluates L(V)LMs' capability on understanding inter-session intention shift with four subtasks. With 1,952,177 intention entries, 1,132,145 session intention trajectories, and 13,003,664 available tasks mined using 10,905 sessions, we provide a scalable way to exploit the existing session data for customer intention understanding. We conduct human annotations to collect ground-truth label for a subset of collected data to form an evaluation gold set. Extensive experiments on the annotated data further confirm that current L(V)LMs fail to capture and utilize the intention across the complex session setting. Further analysis show injecting intention enhances LLMs' performances.
>
---
#### [new 013] Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的异常检测任务，旨在解决在有限标注数据下跨领域识别语言异常（如垃圾短信、假新闻、仇恨言论）的问题。作者提出一种基于元学习的少样本学习框架，结合原型网络与领域重采样策略，实现对新任务的快速适应，并在多个数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20019v1](http://arxiv.org/pdf/2507.20019v1)**

> **作者:** Saurav Singla; Aarav Singla; Advik Gupta; Parnika Gupta
>
> **备注:** 15 pages. PyTorch code for few-shot anomaly detection using meta-learning is available upon request or can be shared via GitHub
>
> **摘要:** We propose a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. We treat anomaly detection as a few shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, we evaluate model generalization on unseen tasks with minimal labeled anomalies. Our method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that our method outperforms strong baselines in F1 and AUC scores. We also release the code and benchmarks to facilitate further research in few-shot text anomaly detection.
>
---
#### [new 014] DRIVE: Disfluency-Rich Synthetic Dialog Data Generation Framework for Intelligent Vehicle Environments
- **分类: cs.CL**

- **简介: 该论文属于对话数据生成任务，旨在解决车载场景中AI对话缺乏真实不流畅表达的问题。作者提出了DiscoDrive框架，生成包含犹豫、重复等自然口语特征的合成对话数据，提升模型在真实车载交互中的表现，并验证了其在训练与数据增强中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19867v1](http://arxiv.org/pdf/2507.19867v1)**

> **作者:** Anshul Chavda; M Jagadeesh; Chintalapalli Raja Kullayappa; B Jayaprakash; Medchalimi Sruthi; Pushpak Bhattacharyya
>
> **摘要:** In-car conversational AI is becoming increasingly critical as autonomous vehicles and smart assistants gain widespread adoption. Yet, existing datasets fail to capture the spontaneous disfluencies such as hesitations, false starts, repetitions, and self-corrections that characterize real driver-AI dialogs. To address this, we introduce DiscoDrive, a synthetic corpus of 3500 multi-turn dialogs across seven automotive domains, generated using a two-stage, prompt-driven pipeline that dynamically integrates disfluencies during synthesis. We show that DiscoDrive is effective both as a training resource, enabling DialoGPT-Medium and T5-Base to match or exceed KVRET-trained models on the MultiWOZ 2.2 and Schema-Guided Dialogue (SGD) relevant test sets (BLEU-4 improvements of 0.26 to 0.61; METEOR +2.10; ROUGE-L +3.48; BERTScore F1 improvements of 1.35 to 3.48), and as a data augmentation resource in low-resource scenarios, delivering additional gains of up to BLEU-4 +0.38, METEOR +1.95, ROUGE-L +2.87, and BERTScore F1 +4.00 when combined with 10 percent of KVRET. Human evaluations further confirm that dialogs sampled from DiscoDrive are rated higher than KVRET's human-collected dialogs in naturalness (3.8 vs 3.6) and coherence (4.1 vs 4.0), and are perceived as more context-appropriate than leading post-hoc methods (such as LARD), without compromising clarity. DiscoDrive fills a critical gap in existing resources and serves as a versatile corpus for both training and augmenting conversational AI, enabling robust handling of real-world, disfluent in-car interactions.
>
---
#### [new 015] Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri
- **分类: cs.CL**

- **简介: 论文提出WOKIE，一个自动化翻译SKOS叙词表的开源工具，旨在解决数字人文学科中多语言知识资源共享与互操作性难题。通过结合外部翻译服务和大语言模型优化，提升翻译质量与本体匹配效果，支持跨语言研究基础设施建设。**

- **链接: [http://arxiv.org/pdf/2507.19537v1](http://arxiv.org/pdf/2507.19537v1)**

> **作者:** Felix Kraus; Nicolas Blumenröhr; Danah Tonne; Achim Streit
>
> **摘要:** We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures.
>
---
#### [new 016] Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems
- **分类: cs.CL; cs.AI; cs.SC**

- **简介: 该论文属于自然语言处理与问答系统任务，旨在解决闭域场景中问答系统的可解释性与高效推理问题。作者提出Text-JEPA框架，将自然语言转为一阶逻辑，并结合Z3求解器进行推理，提升系统透明度与效率。**

- **链接: [http://arxiv.org/pdf/2507.20491v1](http://arxiv.org/pdf/2507.20491v1)**

> **作者:** Tuan Bui; Trong Le; Phat Thai; Sang Nguyen; Minh Hua; Ngan Pham; Thang Bui; Tho Quan
>
> **备注:** 8 pages, 3 figures. Accepted at the International Joint Conference on Neural Networks (IJCNN) 2025, Workshop on Trustworthiness and Reliability in Neuro-Symbolic AI. https://2025.ijcnn.org
>
> **摘要:** Recent advances in large language models (LLMs) have significantly enhanced question-answering (QA) capabilities, particularly in open-domain contexts. However, in closed-domain scenarios such as education, healthcare, and law, users demand not only accurate answers but also transparent reasoning and explainable decision-making processes. While neural-symbolic (NeSy) frameworks have emerged as a promising solution, leveraging LLMs for natural language understanding and symbolic systems for formal reasoning, existing approaches often rely on large-scale models and exhibit inefficiencies in translating natural language into formal logic representations. To address these limitations, we introduce Text-JEPA (Text-based Joint-Embedding Predictive Architecture), a lightweight yet effective framework for converting natural language into first-order logic (NL2FOL). Drawing inspiration from dual-system cognitive theory, Text-JEPA emulates System 1 by efficiently generating logic representations, while the Z3 solver operates as System 2, enabling robust logical inference. To rigorously evaluate the NL2FOL-to-reasoning pipeline, we propose a comprehensive evaluation framework comprising three custom metrics: conversion score, reasoning score, and Spearman rho score, which collectively capture the quality of logical translation and its downstream impact on reasoning accuracy. Empirical results on domain-specific datasets demonstrate that Text-JEPA achieves competitive performance with significantly lower computational overhead compared to larger LLM-based systems. Our findings highlight the potential of structured, interpretable reasoning frameworks for building efficient and explainable QA systems in specialized domains.
>
---
#### [new 017] CaliDrop: KV Cache Compression with Calibration
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成过程中KV缓存占用内存过大的问题。论文提出CaliDrop方法，通过校准机制提升KV缓存压缩中的令牌驱逐策略，减少内存占用同时保持模型准确性。**

- **链接: [http://arxiv.org/pdf/2507.19906v1](http://arxiv.org/pdf/2507.19906v1)**

> **作者:** Yi Su; Quantong Qiu; Yuechi Zhou; Juntao Li; Qingrong Xia; Ping Li; Xinyu Duan; Zhefeng Wang; Min Zhang
>
> **摘要:** Large Language Models (LLMs) require substantial computational resources during generation. While the Key-Value (KV) cache significantly accelerates this process by storing attention intermediates, its memory footprint grows linearly with sequence length, batch size, and model size, creating a bottleneck in long-context scenarios. Various KV cache compression techniques, including token eviction, quantization, and low-rank projection, have been proposed to mitigate this bottleneck, often complementing each other. This paper focuses on enhancing token eviction strategies. Token eviction leverages the observation that the attention patterns are often sparse, allowing for the removal of less critical KV entries to save memory. However, this reduction usually comes at the cost of notable accuracy degradation, particularly under high compression ratios. To address this issue, we propose \textbf{CaliDrop}, a novel strategy that enhances token eviction through calibration. Our preliminary experiments show that queries at nearby positions exhibit high similarity. Building on this observation, CaliDrop performs speculative calibration on the discarded tokens to mitigate the accuracy loss caused by token eviction. Extensive experiments demonstrate that CaliDrop significantly improves the accuracy of existing token eviction methods.
>
---
#### [new 018] AQUA: A Large Language Model for Aquaculture & Fisheries
- **分类: cs.CL; cs.AI; cs.CE; cs.LG; cs.RO**

- **简介: 该论文提出AQUA，首个专为水产养殖设计的大语言模型，旨在解决行业面临的疾病、效率、成本等复杂问题。通过AQUADAPT框架生成高质量合成数据，支持研究与决策，推动AI在水产养殖的应用。**

- **链接: [http://arxiv.org/pdf/2507.20520v1](http://arxiv.org/pdf/2507.20520v1)**

> **作者:** Praneeth Narisetty; Uday Kumar Reddy Kattamanchi; Lohit Akshant Nimma; Sri Ram Kaushik Karnati; Shiva Nagendra Babu Kore; Mounika Golamari; Tejashree Nageshreddy
>
> **摘要:** Aquaculture plays a vital role in global food security and coastal economies by providing sustainable protein sources. As the industry expands to meet rising demand, it faces growing challenges such as disease outbreaks, inefficient feeding practices, rising labor costs, logistical inefficiencies, and critical hatchery issues, including high mortality rates and poor water quality control. Although artificial intelligence has made significant progress, existing machine learning methods fall short of addressing the domain-specific complexities of aquaculture. To bridge this gap, we introduce AQUA, the first large language model (LLM) tailored for aquaculture, designed to support farmers, researchers, and industry practitioners. Central to this effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic Framework for generating and refining high-quality synthetic data using a combination of expert knowledge, largescale language models, and automated evaluation techniques. Our work lays the foundation for LLM-driven innovations in aquaculture research, advisory systems, and decision-making tools.
>
---
#### [new 019] Sem-DPO: Mitigating Semantic Inconsistency in Preference Optimization for Prompt Engineering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决自动提示工程中的语义不一致问题。作者提出了Sem-DPO方法，在保留DPO简洁高效的基础上，通过语义相似性加权减少生成提示与原意的偏离，理论分析并实验证明其在多个基准上的优越性。**

- **链接: [http://arxiv.org/pdf/2507.20133v1](http://arxiv.org/pdf/2507.20133v1)**

> **作者:** Anas Mohamed; Azal Ahmad Khan; Xinran Wang; Ahmad Faraz Khan; Shuwen Ge; Saman Bahzad Khan; Ayaan Ahmad; Ali Anwar
>
> **摘要:** Generative AI can now synthesize strikingly realistic images from text, yet output quality remains highly sensitive to how prompts are phrased. Direct Preference Optimization (DPO) offers a lightweight, off-policy alternative to RL for automatic prompt engineering, but its token-level regularization leaves semantic inconsistency unchecked as prompts that win higher preference scores can still drift away from the user's intended meaning. We introduce Sem-DPO, a variant of DPO that preserves semantic consistency yet retains its simplicity and efficiency. Sem-DPO scales the DPO loss by an exponential weight proportional to the cosine distance between the original prompt and winning candidate in embedding space, softly down-weighting training signals that would otherwise reward semantically mismatched prompts. We provide the first analytical bound on semantic drift for preference-tuned prompt generators, showing that Sem-DPO keeps learned prompts within a provably bounded neighborhood of the original text. On three standard text-to-image prompt-optimization benchmarks and two language models, Sem-DPO achieves 8-12% higher CLIP similarity and 5-9% higher human-preference scores (HPSv2.1, PickScore) than DPO, while also outperforming state-of-the-art baselines. These findings suggest that strong flat baselines augmented with semantic weighting should become the new standard for prompt-optimization studies and lay the groundwork for broader, semantics-aware preference optimization in language models.
>
---
#### [new 020] Before the Outrage: Challenges and Advances in Predicting Online Antisocial Behavior
- **分类: cs.CL**

- **简介: 该论文属于在线反社会行为预测任务，旨在解决社交媒体中仇恨言论、骚扰等行为的提前预测问题。论文系统回顾了49项研究，提出了五类预测任务的统一分类体系，分析了建模技术趋势与数据集挑战，为未来研究提供框架与方向。**

- **链接: [http://arxiv.org/pdf/2507.20614v1](http://arxiv.org/pdf/2507.20614v1)**

> **作者:** Anaïs Ollagnier
>
> **摘要:** Antisocial behavior (ASB) on social media-including hate speech, harassment, and trolling-poses growing challenges for platform safety and societal wellbeing. While prior work has primarily focused on detecting harmful content after it appears, predictive approaches aim to forecast future harmful behaviors-such as hate speech propagation, conversation derailment, or user recidivism-before they fully unfold. Despite increasing interest, the field remains fragmented, lacking a unified taxonomy or clear synthesis of existing methods. This paper presents a systematic review of over 49 studies on ASB prediction, offering a structured taxonomy of five core task types: early harm detection, harm emergence prediction, harm propagation prediction, behavioral risk prediction, and proactive moderation support. We analyze how these tasks differ by temporal framing, prediction granularity, and operational goals. In addition, we examine trends in modeling techniques-from classical machine learning to pre-trained language models-and assess the influence of dataset characteristics on task feasibility and generalization. Our review highlights methodological challenges, such as dataset scarcity, temporal drift, and limited benchmarks, while outlining emerging research directions including multilingual modeling, cross-platform generalization, and human-in-the-loop systems. By organizing the field around a coherent framework, this survey aims to guide future work toward more robust and socially responsible ASB prediction.
>
---
#### [new 021] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出了MCIF，一个跨语言、多模态的指令跟随评测基准，旨在评估大语言模型在多语言、多模态及长短上下文中的指令理解能力。现有评测集在语言、模态和上下文长度方面存在局限，MCIF填补了这一空白，支持英文、德文、意大利文和中文，包含语音、视觉和文本三种模态，适用于科学讲座场景。**

- **链接: [http://arxiv.org/pdf/2507.19634v1](http://arxiv.org/pdf/2507.19634v1)**

> **作者:** Sara Papi; Maike Züfle; Marco Gaido; Beatrice Savoldi; Danni Liu; Ioannis Douros; Luisa Bentivogli; Jan Niehues
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations -- hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities -- speech, vision, and text -- and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development.
>
---
#### [new 022] Reframe Your Life Story: Interactive Narrative Therapist and Innovative Moment Assessment with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于心理健康支持任务，旨在解决现有大语言模型在模拟专业心理治疗和捕捉治疗进展方面的不足。通过开发交互式叙事治疗师（INT）和创新时刻评估（IMA），论文实现了更高质量的治疗对话生成与疗效评估。**

- **链接: [http://arxiv.org/pdf/2507.20241v1](http://arxiv.org/pdf/2507.20241v1)**

> **作者:** Yi Feng; Jiaqi Wang; Wenxuan Zhang; Zhuang Chen; Yutong Shen; Xiyao Xiao; Minlie Huang; Liping Jing; Jian Yu
>
> **摘要:** Recent progress in large language models (LLMs) has opened new possibilities for mental health support, yet current approaches lack realism in simulating specialized psychotherapy and fail to capture therapeutic progression over time. Narrative therapy, which helps individuals transform problematic life stories into empowering alternatives, remains underutilized due to limited access and social stigma. We address these limitations through a comprehensive framework with two core components. First, INT (Interactive Narrative Therapist) simulates expert narrative therapists by planning therapeutic stages, guiding reflection levels, and generating contextually appropriate expert-like responses. Second, IMA (Innovative Moment Assessment) provides a therapy-centric evaluation method that quantifies effectiveness by tracking "Innovative Moments" (IMs), critical narrative shifts in client speech signaling therapy progress. Experimental results on 260 simulated clients and 230 human participants reveal that INT consistently outperforms standard LLMs in therapeutic quality and depth. We further demonstrate the effectiveness of INT in synthesizing high-quality support conversations to facilitate social applications.
>
---
#### [new 023] Soft Injection of Task Embeddings Outperforms Prompt-Based In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务。旨在解决大语言模型在上下文学习中依赖长提示、效率低的问题。提出“软注入任务嵌入”方法，通过优化注意力头参数，在激活空间中注入任务信息，无需提示中的示例，提升性能并降低推理成本。**

- **链接: [http://arxiv.org/pdf/2507.20906v1](http://arxiv.org/pdf/2507.20906v1)**

> **作者:** Jungwon Park; Wonjong Rhee
>
> **备注:** Preprint
>
> **摘要:** In-Context Learning (ICL) enables Large Language Models (LLMs) to perform tasks by conditioning on input-output examples in the prompt, without requiring any update in model parameters. While widely adopted, it remains unclear whether prompting with multiple examples is the most effective and efficient way to convey task information. In this work, we propose Soft Injection of task embeddings. The task embeddings are constructed only once using few-shot ICL prompts and repeatedly used during inference. Soft injection is performed by softly mixing task embeddings with attention head activations using pre-optimized mixing parameters, referred to as soft head-selection parameters. This method not only allows a desired task to be performed without in-prompt demonstrations but also significantly outperforms existing ICL approaches while reducing memory usage and compute cost at inference time. An extensive evaluation is performed across 57 tasks and 12 LLMs, spanning four model families of sizes from 4B to 70B. Averaged across 57 tasks, our method outperforms 10-shot ICL by 10.1%-13.9% across 12 LLMs. Additional analyses show that our method also serves as an insightful tool for analyzing task-relevant roles of attention heads, revealing that task-relevant head positions selected by our method transfer across similar tasks but not across dissimilar ones -- underscoring the task-specific nature of head functionality. Our soft injection method opens a new paradigm for reducing prompt length and improving task performance by shifting task conditioning from the prompt space to the activation space.
>
---
#### [new 024] Efficient Attention Mechanisms for Large Language Models: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中自注意力机制计算效率低的问题。论文系统综述了线性注意力和稀疏注意力两类高效注意力机制，分析了其算法创新与硬件优化，并探讨了其在大规模预训练模型中的应用与混合设计。**

- **链接: [http://arxiv.org/pdf/2507.19595v1](http://arxiv.org/pdf/2507.19595v1)**

> **作者:** Yutao Sun; Zhenyu Li; Yike Zhang; Tengyu Pan; Bowen Dong; Yuyi Guo; Jianyong Wang
>
> **备注:** work in progress
>
> **摘要:** Transformer-based architectures have become the prevailing backbone of large language models. However, the quadratic time and memory complexity of self-attention remains a fundamental obstacle to efficient long-context modeling. To address this limitation, recent research has introduced two principal categories of efficient attention mechanisms. Linear attention methods achieve linear complexity through kernel approximations, recurrent formulations, or fastweight dynamics, thereby enabling scalable inference with reduced computational overhead. Sparse attention techniques, in contrast, limit attention computation to selected subsets of tokens based on fixed patterns, block-wise routing, or clustering strategies, enhancing efficiency while preserving contextual coverage. This survey provides a systematic and comprehensive overview of these developments, integrating both algorithmic innovations and hardware-level considerations. In addition, we analyze the incorporation of efficient attention into largescale pre-trained language models, including both architectures built entirely on efficient attention and hybrid designs that combine local and global components. By aligning theoretical foundations with practical deployment strategies, this work aims to serve as a foundational reference for advancing the design of scalable and efficient language models.
>
---
#### [new 025] Cognitive Chain-of-Thought: Structured Multimodal Reasoning about Social Situations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于多模态推理任务，旨在解决视觉语言模型（VLM）在社会情境中推理能力不足的问题。作者提出了Cognitive Chain-of-Thought（CoCoT）提示策略，通过感知、情境和规范三个阶段提升模型的社会意识与可解释性，实验证明其在多个任务上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.20409v1](http://arxiv.org/pdf/2507.20409v1)**

> **作者:** Eunkyu Park; Wesley Hanwen Deng; Gunhee Kim; Motahhare Eslami; Maarten Sap
>
> **备注:** Under review; 17 pages
>
> **摘要:** Chain-of-Thought (CoT) prompting helps models think step by step. But what happens when they must see, understand, and judge-all at once? In visual tasks grounded in social context, where bridging perception with norm-grounded judgments is essential, flat CoT often breaks down. We introduce Cognitive Chain-of-Thought (CoCoT), a prompting strategy that scaffolds VLM reasoning through three cognitively inspired stages: perception, situation, and norm. Our experiments show that, across multiple multimodal benchmarks (including intent disambiguation, commonsense reasoning, and safety), CoCoT consistently outperforms CoT and direct prompting (+8\% on average). Our findings demonstrate that cognitively grounded reasoning stages enhance interpretability and social awareness in VLMs, paving the way for safer and more reliable multimodal systems.
>
---
#### [new 026] What Language(s) Does Aya-23 Think In? How Multilinguality Affects Internal Language Representations
- **分类: cs.CL**

- **简介: 该论文研究Aya-23-8B模型的多语言内部表示机制，属于自然语言处理任务。旨在理解多语言训练如何影响模型处理语言混合、翻译等任务的内部激活模式。通过logit lens和神经元分析，发现其激活模式受语言类型和基底语言影响，且多语言神经元集中于深层网络。**

- **链接: [http://arxiv.org/pdf/2507.20279v1](http://arxiv.org/pdf/2507.20279v1)**

> **作者:** Katharina Trinley; Toshiki Nakai; Tatiana Anikina; Tanja Baeumel
>
> **备注:** pre-print
>
> **摘要:** Large language models (LLMs) excel at multilingual tasks, yet their internal language processing remains poorly understood. We analyze how Aya-23-8B, a decoder-only LLM trained on balanced multilingual data, handles code-mixed, cloze, and translation tasks compared to predominantly monolingual models like Llama 3 and Chinese-LLaMA-2. Using logit lens and neuron specialization analyses, we find: (1) Aya-23 activates typologically related language representations during translation, unlike English-centric models that rely on a single pivot language; (2) code-mixed neuron activation patterns vary with mixing rates and are shaped more by the base language than the mixed-in one; and (3) Aya-23's languagespecific neurons for code-mixed inputs concentrate in final layers, diverging from prior findings on decoder-only models. Neuron overlap analysis further shows that script similarity and typological relations impact processing across model types. These findings reveal how multilingual training shapes LLM internals and inform future cross-lingual transfer research.
>
---
#### [new 027] Towards Inclusive NLP: Assessing Compressed Multilingual Transformers across Diverse Language Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估压缩多语言模型在多种语言下的性能。论文研究了模型压缩对高、低资源语言（如阿拉伯语、英语和印度语）的影响，发现多语言模型优于单语言模型，量化压缩有效，但剪枝会显著降低性能，尤其在大模型中。**

- **链接: [http://arxiv.org/pdf/2507.19699v1](http://arxiv.org/pdf/2507.19699v1)**

> **作者:** Maitha Alshehhi; Ahmed Sharshar; Mohsen Guizani
>
> **备注:** Published in the 3rd International Workshop on Generalizing from Limited Resources in the Open World. Workshop at International Joint Conference on Artificial Intelligence (IJCAI) 2025
>
> **摘要:** Although LLMs have attained significant success in high-resource languages, their capacity in low-resource linguistic environments like Kannada and Arabic is not yet fully understood. This work benchmarking the performance of multilingual and monolingual Large Language Models (LLMs) across Arabic, English, and Indic languages, with particular emphasis on the effects of model compression strategies such as pruning and quantization. Findings shows significant performance differences driven by linguistic diversity and resource availability on SOTA LLMS as BLOOMZ, AceGPT, Jais, LLaMA-2, XGLM, and AraGPT2. We find that multilingual versions of the model outperform their language-specific counterparts across the board, indicating substantial cross-lingual transfer benefits. Quantization (4-bit and 8-bit) is effective in maintaining model accuracy while promoting efficiency, but aggressive pruning significantly compromises performance, especially in bigger models. Our findings pinpoint key strategies to construct scalable and fair multilingual NLP solutions and underscore the need for interventions to address hallucination and generalization errors in the low-resource setting.
>
---
#### [new 028] UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models' Reasoning Abilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理能力。针对传统强化学习在超长输出训练中的效率低和熵崩溃问题，论文提出UloRL方法，通过分段解码和动态掩码策略，显著提升训练速度和模型性能，验证了其在长序列生成中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19766v1](http://arxiv.org/pdf/2507.19766v1)**

> **作者:** Dong Du; Shulin Liu; Tao Yang; Shaohua Chen; Yang Li
>
> **备注:** 12 pages
>
> **摘要:** Recent advances in large language models (LLMs) have highlighted the potential of reinforcement learning with verifiable rewards (RLVR) to enhance reasoning capabilities through extended output sequences. However, traditional RL frameworks face inefficiencies when handling ultra-long outputs due to long-tail sequence distributions and entropy collapse during training. To address these challenges, we propose an Ultra-Long Output Reinforcement Learning (UloRL) approach for advancing large language models' reasoning abilities. Specifically, we divide ultra long output decoding into short segments, enabling efficient training by mitigating delays caused by long-tail samples. Additionally, we introduce dynamic masking of well-Mastered Positive Tokens (MPTs) to prevent entropy collapse. Experimental results demonstrate the effectiveness of our approach. On the Qwen3-30B-A3B model, RL with segment rollout achieved 2.06x increase in training speed, while RL training with 128k-token outputs improves the model's performance on AIME2025 from 70.9\% to 85.1\% and on BeyondAIME from 50.7\% to 61.9\%, even surpassing Qwen3-235B-A22B with remarkable gains. These findings underscore the potential of our methods to advance the reasoning capabilities of LLMs with ultra-long sequence generation. We will release our code and model for further use by the community.
>
---
#### [new 029] Geometric-Mean Policy Optimization
- **分类: cs.CL**

- **简介: 该论文提出几何平均策略优化（GMPO），改进组相对策略优化（GRPO）在处理重要性加权奖励时的不稳定问题。任务为提升大语言模型推理能力，解决训练中极端重要性采样比导致的策略更新不稳定问题。工作包括设计GMPO方法、理论与实验分析验证其稳定性和性能优势。**

- **链接: [http://arxiv.org/pdf/2507.20673v1](http://arxiv.org/pdf/2507.20673v1)**

> **作者:** Yuzhong Zhao; Yue Liu; Junpeng Liu; Jingye Chen; Xun Wu; Yaru Hao; Tengchao Lv; Shaohan Huang; Lei Cui; Qixiang Ye; Fang Wan; Furu Wei
>
> **备注:** Code is available at https://github.com/callsys/GMPO
>
> **摘要:** Recent advancements, such as Group Relative Policy Optimization (GRPO), have enhanced the reasoning capabilities of large language models by optimizing the arithmetic mean of token-level rewards. However, GRPO suffers from unstable policy updates when processing tokens with outlier importance-weighted rewards, which manifests as extreme importance sampling ratios during training, i.e., the ratio between the sampling probabilities assigned to a token by the current and old policies. In this work, we propose Geometric-Mean Policy Optimization (GMPO), a stabilized variant of GRPO. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. In addition, we provide comprehensive theoretical and experimental analysis to justify the design and stability benefits of GMPO. Beyond improved stability, GMPO-7B outperforms GRPO by an average of 4.1% on multiple mathematical benchmarks and 1.4% on multimodal reasoning benchmark, including AIME24, AMC, MATH500, OlympiadBench, Minerva, and Geometry3K. Code is available at https://github.com/callsys/GMPO.
>
---
#### [new 030] Leveraging Open-Source Large Language Models for Clinical Information Extraction in Resource-Constrained Settings
- **分类: cs.CL**

- **简介: 该论文属于临床信息抽取任务，旨在解决资源有限环境下医学报告中信息提取困难的问题。论文开发了开源框架llm_extractinator，评估了9个开源大语言模型在荷兰语临床信息抽取中的表现，证明了开源模型的有效性和实用性。**

- **链接: [http://arxiv.org/pdf/2507.20859v1](http://arxiv.org/pdf/2507.20859v1)**

> **作者:** Luc Builtjes; Joeran Bosma; Mathias Prokop; Bram van Ginneken; Alessa Hering
>
> **备注:** 34 pages, 5 figures
>
> **摘要:** Medical reports contain rich clinical information but are often unstructured and written in domain-specific language, posing challenges for information extraction. While proprietary large language models (LLMs) have shown promise in clinical natural language processing, their lack of transparency and data privacy concerns limit their utility in healthcare. This study therefore evaluates nine open-source generative LLMs on the DRAGON benchmark, which includes 28 clinical information extraction tasks in Dutch. We developed \texttt{llm\_extractinator}, a publicly available framework for information extraction using open-source generative LLMs, and used it to assess model performance in a zero-shot setting. Several 14 billion parameter models, Phi-4-14B, Qwen-2.5-14B, and DeepSeek-R1-14B, achieved competitive results, while the bigger Llama-3.3-70B model achieved slightly higher performance at greater computational cost. Translation to English prior to inference consistently degraded performance, highlighting the need of native-language processing. These findings demonstrate that open-source LLMs, when used with our framework, offer effective, scalable, and privacy-conscious solutions for clinical information extraction in low-resource settings.
>
---
#### [new 031] Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation
- **分类: cs.CL; 68T50**

- **简介: 该论文属于自然语言处理中的自动化评估任务，旨在解决现有LLM-as-a-judge方法在评估维度单一和泛化能力不足的问题。论文提出MAJ-EVAL框架，通过多智能体模拟不同人类评估者，自动生成多维度评估反馈，提高评估结果与人类专家的一致性。**

- **链接: [http://arxiv.org/pdf/2507.21028v1](http://arxiv.org/pdf/2507.21028v1)**

> **作者:** Jiaju Chen; Yuxuan Lu; Xiaojie Wang; Huimin Zeng; Jing Huang; Jiri Gesi; Ying Xu; Bingsheng Yao; Dakuo Wang
>
> **摘要:** Nearly all human work is collaborative; thus, the evaluation of real-world NLP applications often requires multiple dimensions that align with diverse human perspectives. As real human evaluator resources are often scarce and costly, the emerging "LLM-as-a-judge" paradigm sheds light on a promising approach to leverage LLM agents to believably simulate human evaluators. Yet, to date, existing LLM-as-a-judge approaches face two limitations: persona descriptions of agents are often arbitrarily designed, and the frameworks are not generalizable to other tasks. To address these challenges, we propose MAJ-EVAL, a Multi-Agent-as-Judge evaluation framework that can automatically construct multiple evaluator personas with distinct dimensions from relevant text documents (e.g., research papers), instantiate LLM agents with the personas, and engage in-group debates with multi-agents to Generate multi-dimensional feedback. Our evaluation experiments in both the educational and medical domains demonstrate that MAJ-EVAL can generate evaluation results that better align with human experts' ratings compared with conventional automated evaluation metrics and existing LLM-as-a-judge methods.
>
---
#### [new 032] A survey of diversity quantification in natural language processing: The why, what, where and how
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多样性度量标准不统一的问题。论文通过调查ACL文集中近6年涉及“多样性”关键词的研究，提出基于生态学和经济学的统一分类框架，涵盖多样性测量的原因、对象、位置和方法，以提升NLP领域对多样性的理解与比较。**

- **链接: [http://arxiv.org/pdf/2507.20858v1](http://arxiv.org/pdf/2507.20858v1)**

> **作者:** Louis Estève; Marie-Catherine de Marneffe; Nurit Melnik; Agata Savary; Olha Kanishcheva
>
> **摘要:** The concept of diversity has received increased consideration in Natural Language Processing (NLP) in recent years. This is due to various motivations like promoting and inclusion, approximating human linguistic behavior, and increasing systems' performance. Diversity has however often been addressed in an ad hoc manner in NLP, and with few explicit links to other domains where this notion is better theorized. We survey articles in the ACL Anthology from the past 6 years, with "diversity" or "diverse" in their title. We find a wide range of settings in which diversity is quantified, often highly specialized and using inconsistent terminology. We put forward a unified taxonomy of why, what on, where, and how diversity is measured in NLP. Diversity measures are cast upon a unified framework from ecology and economy (Stirling, 2007) with 3 dimensions of diversity: variety, balance and disparity. We discuss the trends which emerge due to this systematized approach. We believe that this study paves the way towards a better formalization of diversity in NLP, which should bring a better understanding of this notion and a better comparability between various approaches.
>
---
#### [new 033] A Tensor-Based Compiler and a Runtime for Neuron-Level DNN Certifier Specifications
- **分类: cs.CL**

- **简介: 该论文属于编译器与深度学习模型验证任务，旨在解决神经网络验证中设计与实现间的语义鸿沟问题。作者提出了一种基于张量的编译框架，将神经元级别的验证规范自动转换为高效的张量级别实现，并设计了g-BCSR格式以优化稀疏张量的存储与计算，从而简化新验证器的开发并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.20055v1](http://arxiv.org/pdf/2507.20055v1)**

> **作者:** Avaljot Singh; Yamin Chandini Sarita; Aditya Mishra; Ishaan Goyal; Gagandeep Singh; Charith Mendis
>
> **摘要:** The uninterpretability of DNNs has led to the adoption of abstract interpretation-based certification as a practical means to establish trust in real-world systems that rely on DNNs. However, the current landscape supports only a limited set of certifiers, and developing new ones or modifying existing ones for different applications remains difficult. This is because the mathematical design of certifiers is expressed at the neuron level, while their implementations are optimized and executed at the tensor level. This mismatch creates a semantic gap between design and implementation, making manual bridging both complex and expertise-intensive -- requiring deep knowledge in formal methods, high-performance computing, etc. We propose a compiler framework that automatically translates neuron-level specifications of DNN certifiers into tensor-based, layer-level implementations. This is enabled by two key innovations: a novel stack-based intermediate representation (IR) and a shape analysis that infers the implicit tensor operations needed to simulate the neuron-level semantics. During lifting, the shape analysis creates tensors in the minimal shape required to perform the corresponding operations. The IR also enables domain-specific optimizations as rewrites. At runtime, the resulting tensor computations exhibit sparsity tied to the DNN architecture. This sparsity does not align well with existing formats. To address this, we introduce g-BCSR, a double-compression format that represents tensors as collections of blocks of varying sizes, each possibly internally sparse. Using our compiler and g-BCSR, we make it easy to develop new certifiers and analyze their utility across diverse DNNs. Despite its flexibility, the compiler achieves performance comparable to hand-optimized implementations.
>
---
#### [new 034] IQ Test for LLMs: An Evaluation Framework for Uncovering Core Skills in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务中的模型评估方向。旨在解决当前大语言模型评估缺乏对任务间关系与核心技能的解析问题。工作包括构建包含60个LLM和44项任务的排行榜，运用因子分析识别潜在技能，提供识别冗余任务、辅助模型选择及模型技能画像的实用工具。**

- **链接: [http://arxiv.org/pdf/2507.20208v1](http://arxiv.org/pdf/2507.20208v1)**

> **作者:** Aviya Maimon; Amir DN Cohen; Gal Vishne; Shauli Ravfogel; Reut Tsarfaty
>
> **摘要:** Current evaluations of large language models (LLMs) rely on benchmark scores, but it is difficult to interpret what these individual scores reveal about a model's overall skills. Specifically, as a community we lack understanding of how tasks relate to one another, what they measure in common, how they differ, or which ones are redundant. As a result, models are often assessed via a single score averaged across benchmarks, an approach that fails to capture the models' wholistic strengths and limitations. Here, we propose a new evaluation paradigm that uses factor analysis to identify latent skills driving performance across benchmarks. We apply this method to a comprehensive new leaderboard showcasing the performance of 60 LLMs on 44 tasks, and identify a small set of latent skills that largely explain performance. Finally, we turn these insights into practical tools that identify redundant tasks, aid in model selection, and profile models along each latent skill.
>
---
#### [new 035] HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理中键值缓存内存需求过高的问题。提出了HCAttention框架，通过量化键、卸载值和动态缓存淘汰，在极端内存限制下实现高效推理。在LongBench基准上，将缓存压缩至原大小的25%仍保持准确率，且无需模型微调。**

- **链接: [http://arxiv.org/pdf/2507.19823v1](http://arxiv.org/pdf/2507.19823v1)**

> **作者:** Dongquan Yang; Yifan Yang; Xiaotian Yu; Xianbiao Qi; Rong Xiao
>
> **摘要:** Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compression methods exhibit noticeable performance degradation when memory is reduced by more than 85%. Additionally, strategies that leverage GPU-CPU collaboration for approximate attention remain underexplored in this setting. We propose HCAttention, a heterogeneous attention computation framework that integrates key quantization, value offloading, and dynamic KV eviction to enable efficient inference under extreme memory constraints. The method is compatible with existing transformer architectures and does not require model fine-tuning. Experimental results on the LongBench benchmark demonstrate that our approach preserves the accuracy of full-attention model while shrinking the KV cache memory footprint to 25% of its original size. Remarkably, it stays competitive with only 12.5% of the cache, setting a new state-of-the-art in LLM KV cache compression. To the best of our knowledge, HCAttention is the first to extend the Llama-3-8B model to process 4 million tokens on a single A100 GPU with 80GB memory.
>
---
#### [new 036] The Polish Vocabulary Size Test: A Novel Adaptive Test for Receptive Vocabulary Assessment
- **分类: cs.CL**

- **简介: 论文提出了波兰语词汇量自适应测试（PVST），用于评估母语者和非母语者的接受性词汇量。基于项目反应理论和计算机自适应测试技术，PVST动态调整测试难度，确保准确性和效率。研究通过1475名参与者验证测试有效性。**

- **链接: [http://arxiv.org/pdf/2507.19869v1](http://arxiv.org/pdf/2507.19869v1)**

> **作者:** Danil Fokin; Monika Płużyczka; Grigory Golovin
>
> **摘要:** We present the Polish Vocabulary Size Test (PVST), a novel tool for assessing the receptive vocabulary size of both native and non-native Polish speakers. Based on Item Response Theory and Computerized Adaptive Testing, PVST dynamically adjusts to each test-taker's proficiency level, ensuring high accuracy while keeping the test duration short. To validate the test, a pilot study was conducted with 1.475 participants. Native Polish speakers demonstrated significantly larger vocabularies compared to non-native speakers. For native speakers, vocabulary size showed a strong positive correlation with age. The PVST is available online at myvocab.info/pl.
>
---
#### [new 037] MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建了MediQAl，一个包含32,603道法语医学问答题的评测数据集，用于评估语言模型在医学事实记忆与推理能力方面的表现。数据集涵盖三种题型，并标注了理解与推理类型。论文通过测试14种大模型，揭示了记忆与推理任务间的性能差距，填补了医学领域多语言资源的空白。**

- **链接: [http://arxiv.org/pdf/2507.20917v1](http://arxiv.org/pdf/2507.20917v1)**

> **作者:** Adrien Bazoge
>
> **摘要:** This work introduces MediQAl, a French medical question answering dataset designed to evaluate the capabilities of language models in factual medical recall and reasoning over real-world clinical scenarios. MediQAl contains 32,603 questions sourced from French medical examinations across 41 medical subjects. The dataset includes three tasks: (i) Multiple-Choice Question with Unique answer, (ii) Multiple-Choice Question with Multiple answer, and (iii) Open-Ended Question with Short-Answer. Each question is labeled as Understanding or Reasoning, enabling a detailed analysis of models' cognitive capabilities. We validate the MediQAl dataset through extensive evaluation with 14 large language models, including recent reasoning-augmented models, and observe a significant performance gap between factual recall and reasoning tasks. Our evaluation provides a comprehensive benchmark for assessing language models' performance on French medical question answering, addressing a crucial gap in multilingual resources for the medical domain.
>
---
#### [new 038] Flora: Effortless Context Construction to Arbitrary Length and Scale
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长文本处理中的效率与性能问题。作者提出Flora方法，无需人工或大模型干预，自动组合短指令生成多样化的长上下文，提升模型在长文本基准测试中的表现，同时保持短文本处理能力。实验表明该方法在多个模型上有效。**

- **链接: [http://arxiv.org/pdf/2507.19786v1](http://arxiv.org/pdf/2507.19786v1)**

> **作者:** Tianxiang Chen; Zhentao Tan; Xiaofan Bo; Yue Wu; Tao Gong; Qi Chu; Jieping Ye; Nenghai Yu
>
> **摘要:** Effectively handling long contexts is challenging for Large Language Models (LLMs) due to the rarity of long texts, high computational demands, and substantial forgetting of short-context abilities. Recent approaches have attempted to construct long contexts for instruction tuning, but these methods often require LLMs or human interventions, which are both costly and limited in length and diversity. Also, the drop in short-context performances of present long-context LLMs remains significant. In this paper, we introduce Flora, an effortless (human/LLM-free) long-context construction strategy. Flora can markedly enhance the long-context performance of LLMs by arbitrarily assembling short instructions based on categories and instructing LLMs to generate responses based on long-context meta-instructions. This enables Flora to produce contexts of arbitrary length and scale with rich diversity, while only slightly compromising short-context performance. Experiments on Llama3-8B-Instruct and QwQ-32B show that LLMs enhanced by Flora excel in three long-context benchmarks while maintaining strong performances in short-context tasks. Our data-construction code is available at \href{https://github.com/txchen-USTC/Flora}{https://github.com/txchen-USTC/Flora}.
>
---
#### [new 039] ZSE-Cap: A Zero-Shot Ensemble for Image Retrieval and Prompt-Guided Captioning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于图像检索与描述生成任务，旨在解决基于文章内容的图像检索和事件关联描述生成问题。论文提出ZSE-Cap系统，采用零样本方法，结合CLIP、SigLIP和DINOv2模型的集成提升检索效果，并通过提示引导Gemma 3模型生成与文章事件相关的图像描述。**

- **链接: [http://arxiv.org/pdf/2507.20564v1](http://arxiv.org/pdf/2507.20564v1)**

> **作者:** Duc-Tai Dinh; Duc Anh Khoa Dinh
>
> **摘要:** We present ZSE-Cap (Zero-Shot Ensemble for Captioning), our 4th place system in Event-Enriched Image Analysis (EVENTA) shared task on article-grounded image retrieval and captioning. Our zero-shot approach requires no finetuning on the competition's data. For retrieval, we ensemble similarity scores from CLIP, SigLIP, and DINOv2. For captioning, we leverage a carefully engineered prompt to guide the Gemma 3 model, enabling it to link high-level events from the article to the visual content in the image. Our system achieved a final score of 0.42002, securing a top-4 position on the private test set, demonstrating the effectiveness of combining foundation models through ensembling and prompting. Our code is available at https://github.com/ductai05/ZSE-Cap.
>
---
#### [new 040] A Gold Standard Dataset and Evaluation Framework for Depression Detection and Explanation in Social Media using LLMs
- **分类: cs.CL**

- **简介: 该论文属于心理健康与自然语言处理交叉任务，旨在解决社交媒体中抑郁检测与解释缺乏细粒度标注数据的问题。作者构建了一个包含1017条社交媒体帖子的高质量数据集，标注了抑郁片段并分类至12种症状。基于此，他们提出了评估大语言模型（LLMs）在抑郁解释任务中表现的框架，并通过零样本与少样本提示策略评估了GPT-4.1、Gemini 2.5 Pro等模型，强调了人类专业知识在引导LLM行为中的价值。**

- **链接: [http://arxiv.org/pdf/2507.19899v1](http://arxiv.org/pdf/2507.19899v1)**

> **作者:** Prajval Bolegave; Pushpak Bhattacharya
>
> **摘要:** Early detection of depression from online social media posts holds promise for providing timely mental health interventions. In this work, we present a high-quality, expert-annotated dataset of 1,017 social media posts labeled with depressive spans and mapped to 12 depression symptom categories. Unlike prior datasets that primarily offer coarse post-level labels \cite{cohan-etal-2018-smhd}, our dataset enables fine-grained evaluation of both model predictions and generated explanations. We develop an evaluation framework that leverages this clinically grounded dataset to assess the faithfulness and quality of natural language explanations generated by large language models (LLMs). Through carefully designed prompting strategies, including zero-shot and few-shot approaches with domain-adapted examples, we evaluate state-of-the-art proprietary LLMs including GPT-4.1, Gemini 2.5 Pro, and Claude 3.7 Sonnet. Our comprehensive empirical analysis reveals significant differences in how these models perform on clinical explanation tasks, with zero-shot and few-shot prompting. Our findings underscore the value of human expertise in guiding LLM behavior and offer a step toward safer, more transparent AI systems for psychological well-being.
>
---
#### [new 041] AI-Driven Generation of Old English: A Framework for Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决古英语资源匮乏问题。通过结合参数高效微调、回译数据增强和双代理生成框架，生成高质量古英语文本，显著提升翻译效果，为濒危语言保护提供可行方案。**

- **链接: [http://arxiv.org/pdf/2507.20111v1](http://arxiv.org/pdf/2507.20111v1)**

> **作者:** Rodrigo Gabriel Salazar Alva; Matías Nuñez; Cristian López; Javier Martín Arista
>
> **摘要:** Preserving ancient languages is essential for understanding humanity's cultural and linguistic heritage, yet Old English remains critically under-resourced, limiting its accessibility to modern natural language processing (NLP) techniques. We present a scalable framework that uses advanced large language models (LLMs) to generate high-quality Old English texts, addressing this gap. Our approach combines parameter-efficient fine-tuning (Low-Rank Adaptation, LoRA), data augmentation via backtranslation, and a dual-agent pipeline that separates the tasks of content generation (in English) and translation (into Old English). Evaluation with automated metrics (BLEU, METEOR, and CHRF) shows significant improvements over baseline models, with BLEU scores increasing from 26 to over 65 for English-to-Old English translation. Expert human assessment also confirms high grammatical accuracy and stylistic fidelity in the generated texts. Beyond expanding the Old English corpus, our method offers a practical blueprint for revitalizing other endangered languages, effectively uniting AI innovation with the goals of cultural preservation.
>
---
#### [new 042] Setting The Table with Intent: Intent-aware Schema Generation and Editing for Literature Review Tables
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文献综述表格生成任务，旨在解决模式生成中的模糊性和缺乏编辑方法的问题。作者通过合成意图增强数据集，提升模式重建效果，并提出基于大语言模型的编辑技术优化生成结果。**

- **链接: [http://arxiv.org/pdf/2507.19521v1](http://arxiv.org/pdf/2507.19521v1)**

> **作者:** Vishakh Padmakumar; Joseph Chee Chang; Kyle Lo; Doug Downey; Aakanksha Naik
>
> **摘要:** The increasing volume of academic literature makes it essential for researchers to organize, compare, and contrast collections of documents. Large language models (LLMs) can support this process by generating schemas defining shared aspects along which to compare papers. However, progress on schema generation has been slow due to: (i) ambiguity in reference-based evaluations, and (ii) lack of editing/refinement methods. Our work is the first to address both issues. First, we present an approach for augmenting unannotated table corpora with synthesized intents and apply it to create a dataset for studying schema generation conditioned on a given information need, thus reducing ambiguity. With this dataset, we show how incorporating table intents significantly improves baseline performance in reconstructing reference schemas. Next, we propose several LLM-based schema editing techniques. We start by comprehensively benchmarking several single-shot schema generation methods, including prompted LLM workflows and fine-tuned models, showing that smaller, open-weight models can be fine-tuned to be competitive with state-of-the-art prompted LLMs. Then we demonstrate that our editing techniques can further improve schemas generated by these methods.
>
---
#### [new 043] Multi-Agent Interactive Question Generation Framework for Long Document Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档理解任务，旨在解决长上下文、复杂布局文档理解中训练数据稀缺的问题。论文提出了一种多智能体交互式问答生成框架，自动生成高质量的英文和阿拉伯文长文档问答数据，用于提升大视觉语言模型的长上下文理解能力。**

- **链接: [http://arxiv.org/pdf/2507.20145v1](http://arxiv.org/pdf/2507.20145v1)**

> **作者:** Kesen Wang; Daulet Toibazar; Abdulrahman Alfulayt; Abdulaziz S. Albadawi; Ranya A. Alkahtani; Asma A. Ibrahim; Haneen A. Alhomoud; Sherif Mohamed; Pedro J. Moreno
>
> **摘要:** Document Understanding (DU) in long-contextual scenarios with complex layouts remains a significant challenge in vision-language research. Although Large Vision-Language Models (LVLMs) excel at short-context DU tasks, their performance declines in long-context settings. A key limitation is the scarcity of fine-grained training data, particularly for low-resource languages such as Arabic. Existing state-of-the-art techniques rely heavily on human annotation, which is costly and inefficient. We propose a fully automated, multi-agent interactive framework to generate long-context questions efficiently. Our approach efficiently generates high-quality single- and multi-page questions for extensive English and Arabic documents, covering hundreds of pages across diverse domains. This facilitates the development of LVLMs with enhanced long-context understanding ability. Experimental results in this work have shown that our generated English and Arabic questions (\textbf{AraEngLongBench}) are quite challenging to major open- and close-source LVLMs. The code and data proposed in this work can be found in https://github.com/wangk0b/Multi_Agentic_QA_Long_Doc.git. Sample Question and Answer (QA) pairs and structured system prompts can be found in the Appendix.
>
---
#### [new 044] Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决多模态大语言模型（MLLMs）因计算和内存需求高而难以部署的问题。通过结构剪枝与恢复训练方法，如逐层和逐宽剪枝，结合微调和知识蒸馏，实现高效压缩。实验表明，在少量数据下仍可保持高性能。**

- **链接: [http://arxiv.org/pdf/2507.20749v1](http://arxiv.org/pdf/2507.20749v1)**

> **作者:** Yiran Huang; Lukas Thede; Massimiliano Mancini; Wenjia Xu; Zeynep Akata
>
> **备注:** Accepted at GCPR 2025
>
> **摘要:** While Multimodal Large Language Models (MLLMs) demonstrate impressive capabilities, their substantial computational and memory requirements pose significant barriers to practical deployment. Current parameter reduction techniques primarily involve training MLLMs from Small Language Models (SLMs), but these methods offer limited flexibility and remain computationally intensive. To address this gap, we propose to directly compress existing MLLMs through structural pruning combined with efficient recovery training. Specifically, we investigate two structural pruning paradigms--layerwise and widthwise pruning--applied to the language model backbone of MLLMs, alongside supervised finetuning and knowledge distillation. Additionally, we assess the feasibility of conducting recovery training with only a small fraction of the available data. Our results show that widthwise pruning generally maintains better performance in low-resource scenarios with limited computational resources or insufficient finetuning data. As for the recovery training, finetuning only the multimodal projector is sufficient at small compression levels (< 20%). Furthermore, a combination of supervised finetuning and hidden-state distillation yields optimal recovery across various pruning levels. Notably, effective recovery can be achieved with as little as 5% of the original training data, while retaining over 95% of the original performance. Through empirical study on two representative MLLMs, i.e., LLaVA-v1.5-7B and Bunny-v1.0-3B, this study offers actionable insights for practitioners aiming to compress MLLMs effectively without extensive computation resources or sufficient data.
>
---
#### [new 045] ProsodyLM: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决现有模型难以捕捉语音韵律信息的问题。作者提出ProsodyLM，通过引入文本与词级韵律标记的联合建模方法，使模型在预训练中即可学习多样化的韵律处理能力，如理解语调、情绪和保持长文本韵律一致性。**

- **链接: [http://arxiv.org/pdf/2507.20091v1](http://arxiv.org/pdf/2507.20091v1)**

> **作者:** Kaizhi Qian; Xulin Fan; Junrui Ni; Slava Shechtman; Mark Hasegawa-Johnson; Chuang Gan; Yang Zhang
>
> **摘要:** Speech language models refer to language models with speech processing and understanding capabilities. One key desirable capability for speech language models is the ability to capture the intricate interdependency between content and prosody. The existing mainstream paradigm of training speech language models, which converts speech into discrete tokens before feeding them into LLMs, is sub-optimal in learning prosody information -- we find that the resulting LLMs do not exhibit obvious emerging prosody processing capabilities via pre-training alone. To overcome this, we propose ProsodyLM, which introduces a simple tokenization scheme amenable to learning prosody. Each speech utterance is first transcribed into text, followed by a sequence of word-level prosody tokens. Compared with conventional speech tokenization schemes, the proposed tokenization scheme retains more complete prosody information, and is more understandable to text-based LLMs. We find that ProsodyLM can learn surprisingly diverse emerging prosody processing capabilities through pre-training alone, ranging from harnessing the prosody nuances in generated speech, such as contrastive focus, understanding emotion and stress in an utterance, to maintaining prosody consistency in long contexts.
>
---
#### [new 046] EMBRACE: Shaping Inclusive Opinion Representation by Aligning Implicit Conversations with Social Norms
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决对话模型中隐含观点表达与社会规范对齐的问题。现有方法依赖表面特征，忽视了隐性意见表达，可能导致偏见。作者提出EMBRACE框架，通过建模回应立场来评估观点表达，并使用PU学习和指令微调语言模型进行验证，以提升模型的包容性和公平性。**

- **链接: [http://arxiv.org/pdf/2507.20264v1](http://arxiv.org/pdf/2507.20264v1)**

> **作者:** Abeer Aldayel; Areej Alokaili
>
> **备注:** Under review for publication
>
> **摘要:** Shaping inclusive representations that embrace diversity and ensure fair participation and reflections of values is at the core of many conversation-based models. However, many existing methods rely on surface inclusion using mention of user demographics or behavioral attributes of social groups. Such methods overlook the nuanced, implicit expression of opinion embedded in conversations. Furthermore, the over-reliance on overt cues can exacerbate misalignment and reinforce harmful or stereotypical representations in model outputs. Thus, we took a step back and recognized that equitable inclusion needs to account for the implicit expression of opinion and use the stance of responses to validate the normative alignment. This study aims to evaluate how opinions are represented in NLP or computational models by introducing an alignment evaluation framework that foregrounds implicit, often overlooked conversations and evaluates the normative social views and discourse. Our approach models the stance of responses as a proxy for the underlying opinion, enabling a considerate and reflective representation of diverse social viewpoints. We evaluate the framework using both (i) positive-unlabeled (PU) online learning with base classifiers, and (ii) instruction-tuned language models to assess post-training alignment. Through this, we provide a lens on how implicit opinions are (mis)represented and offer a pathway toward more inclusive model behavior.
>
---
#### [new 047] JT-Math: A Multi-Stage Framework for Advanced Mathematical Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型解决复杂数学问题的能力。论文提出了JT-Math-8B框架，通过多阶段优化方法，结合高质量预训练和强化学习，训练出擅长直接答题和复杂推理的模型，在竞赛级数学任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.19748v1](http://arxiv.org/pdf/2507.19748v1)**

> **作者:** Yifan Hao; Fangning Chao; Yaqian Hao; Zhaojun Cui; Huan Bai; Haiyu Zhang; Yankai Liu; Chao Deng; Junlan Feng
>
> **摘要:** Mathematical reasoning is a cornerstone of artificial general intelligence and a primary benchmark for evaluating the capabilities of Large Language Models (LLMs). While state-of-the-art models show promise, they often falter when faced with complex problems that demand deep conceptual understanding and intricate, multi-step deliberation. To address this challenge, we introduce JT-Math-8B, a series of open-source models comprising base, instruct, and thinking versions, built upon a systematic, multi-stage optimization framework. Our pre-training corpus is a high-quality, 210B-token dataset curated through a dedicated data pipeline that uses model-based validation to ensure quality and diversity. The Instruct Model is optimized for direct, concise answers through Supervised Fine-Tuning (SFT) and a GRPO-based reinforcement learning (RL) method. The Thinking Model is trained for complex problem-solving using a Long Chain-of-Thought (Long CoT) approach, combining SFT with a novel, multi-stage RL curriculum that progressively increases task difficulty and context length up to 32K tokens. JT-Math-8B achieves state-of-the-art results among open-source models of similar size, surpassing prominent models like OpenAI's O1-mini and GPT-4o , and demonstrating superior performance on competition-level mathematics.
>
---
#### [new 048] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决数学领域训练数据不足的问题。作者提出了SAND-Math生成管道，通过从头生成高质量问题并提升其难度，显著提高了模型在AIME25基准上的表现，有效增强了数学推理LLMs的性能。**

- **链接: [http://arxiv.org/pdf/2507.20527v1](http://arxiv.org/pdf/2507.20527v1)**

> **作者:** Chaitanya Manem; Pratik Prabhanjan Brahma; Prakamya Mishra; Zicheng Liu; Emad Barsoum
>
> **摘要:** The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
>
---
#### [new 049] FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决金融领域语言模型生成内容中的事实错误问题。作者构建了标注错误的金融问答数据集，通过微调Phi-4等模型提升事实错误的检测与编辑能力，取得了优于OpenAI-o3的效果，并提出了可推广的框架。**

- **链接: [http://arxiv.org/pdf/2507.20930v1](http://arxiv.org/pdf/2507.20930v1)**

> **作者:** Likun Tan; Kuan-Wei Huang; Kevin Wu
>
> **摘要:** Hallucinations in large language models pose a critical challenge for applications requiring factual reliability, particularly in high-stakes domains such as finance. This work presents an effective approach for detecting and editing factually incorrect content in model-generated responses based on the provided context. Given a user-defined domain-specific error taxonomy, we construct a synthetic dataset by inserting tagged errors into financial question-answering corpora and then fine-tune four language models, Phi-4, Phi-4-mini, Qwen3-4B, and Qwen3-14B, to detect and edit these factual inaccuracies. Our best-performing model, fine-tuned Phi-4, achieves an 8% improvement in binary F1 score and a 30% gain in overall detection performance compared to OpenAI-o3. Notably, our fine-tuned Phi-4-mini model, despite having only 4 billion parameters, maintains competitive performance with just a 2% drop in binary detection and a 0.1% decline in overall detection compared to OpenAI-o3. Our work provides a practical solution for detecting and editing factual inconsistencies in financial text generation while introducing a generalizable framework that can enhance the trustworthiness and alignment of large language models across diverse applications beyond finance. Our code and data are available at https://github.com/pegasi-ai/fine-grained-editting.
>
---
#### [new 050] Are You There God? Lightweight Narrative Annotation of Christian Fiction with LMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与文学分析交叉任务，旨在解决基督教小说中“神迹”描写的自动标注与分析问题。作者通过构建人类标注准则，并利用轻量级语言模型进行自动化标注，比较了《末日迷踪》系列与其他基督教小说、以及不同性别作者作品间的差异。**

- **链接: [http://arxiv.org/pdf/2507.19756v1](http://arxiv.org/pdf/2507.19756v1)**

> **作者:** Rebecca M. M. Hicke; Brian Haggard; Mia Ferrante; Rayhan Khanna; David Mimno
>
> **摘要:** In addition to its more widely studied political activities, the American Evangelical movement has a well-developed but less externally visible cultural and literary side. Christian Fiction, however, has been little studied, and what scholarly attention there is has focused on the explosively popular Left Behind series. In this work, we use computational tools to provide both a broad topical overview of Christian Fiction as a genre and a more directed exploration of how its authors depict divine acts. Working with human annotators we first developed definitions and a codebook for "acts of God." We then adapted those instructions designed for human annotators for use by a recent, lightweight LM with the assistance of a much larger model. The laptop-scale LM is capable of matching human annotations, even when the task is subtle and challenging. Using these annotations, we show that significant and meaningful differences exist between the Left Behind books and Christian Fiction more broadly and between books by male and female authors.
>
---
#### [new 051] Ta-G-T: Subjectivity Capture in Table to Text Generation via RDF Graphs
- **分类: cs.CL**

- **简介: 该论文属于表格到文本生成任务，旨在解决现有方法缺乏对数据主观解读的问题。作者提出Ta-G-T三阶段管道，通过RDF图提取、文本聚合与主观性注入，提升生成文本的主观表达与事实准确性，相比部分大模型表现更优。**

- **链接: [http://arxiv.org/pdf/2507.19710v1](http://arxiv.org/pdf/2507.19710v1)**

> **作者:** Ronak Upasham; Tathagata Dey; Pushpak Bhattacharyya
>
> **摘要:** In Table-to-Text (T2T) generation, existing approaches predominantly focus on providing objective descriptions of tabular data. However, generating text that incorporates subjectivity, where subjectivity refers to interpretations beyond raw numerical data, remains underexplored. To address this, we introduce a novel pipeline that leverages intermediate representations to generate both objective and subjective text from tables. Our three-stage pipeline consists of: 1) extraction of Resource Description Framework (RDF) triples, 2) aggregation of text into coherent narratives, and 3) infusion of subjectivity to enrich the generated text. By incorporating RDFs, our approach enhances factual accuracy while maintaining interpretability. Unlike large language models (LLMs) such as GPT-3.5, Mistral-7B, and Llama-2, our pipeline employs smaller, fine-tuned T5 models while achieving comparable performance to GPT-3.5 and outperforming Mistral-7B and Llama-2 in several metrics. We evaluate our approach through quantitative and qualitative analyses, demonstrating its effectiveness in balancing factual accuracy with subjective interpretation. To the best of our knowledge, this is the first work to propose a structured pipeline for T2T generation that integrates intermediate representations to enhance both factual correctness and subjectivity.
>
---
#### [new 052] CodeNER: Code Prompting for Named Entity Recognition
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文属于命名实体识别（NER）任务，旨在解决现有方法仅依赖上下文信息、难以准确捕捉标签需求的问题。论文提出CodeNER，通过代码提示提供详细的BIO标注规则，提升大语言模型对NER任务的理解与执行能力。**

- **链接: [http://arxiv.org/pdf/2507.20423v1](http://arxiv.org/pdf/2507.20423v1)**

> **作者:** Sungwoo Han; Hyeyeon Kim; Jingun Kwon; Hidetaka Kamigaito; Manabu Okumura
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Recent studies have explored various approaches for treating candidate named entity spans as both source and target sequences in named entity recognition (NER) by leveraging large language models (LLMs). Although previous approaches have successfully generated candidate named entity spans with suitable labels, they rely solely on input context information when using LLMs, particularly, ChatGPT. However, NER inherently requires capturing detailed labeling requirements with input context information. To address this issue, we propose a novel method that leverages code-based prompting to improve the capabilities of LLMs in understanding and performing NER. By embedding code within prompts, we provide detailed BIO schema instructions for labeling, thereby exploiting the ability of LLMs to comprehend long-range scopes in programming languages. Experimental results demonstrate that the proposed code-based prompting method outperforms conventional text-based prompting on ten benchmarks across English, Arabic, Finnish, Danish, and German datasets, indicating the effectiveness of explicitly structuring NER instructions. We also verify that combining the proposed code-based prompting method with the chain-of-thought prompting further improves performance.
>
---
#### [new 053] RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究检索增强生成（RAG）在真实场景中的有效性。任务是评估RAG在多知识源下的表现，解决其在现实应用中效果不明确的问题。作者使用大规模知识库MassiveDS进行实验，发现RAG对小模型更有效，重排序器作用有限，且模型难以在不同知识源间有效切换，指出需改进自适应检索策略。**

- **链接: [http://arxiv.org/pdf/2507.20059v1](http://arxiv.org/pdf/2507.20059v1)**

> **作者:** Ran Xu; Yuchen Zhuang; Yue Yu; Haoyu Wang; Wenqi Shi; Carl Yang
>
> **备注:** Work in Progress. Code will be published at: https://github.com/ritaranx/RAG_in_the_Wild
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved at inference time. While RAG demonstrates strong performance on benchmarks largely derived from general-domain corpora like Wikipedia, its effectiveness under realistic, diverse retrieval scenarios remains underexplored. We evaluated RAG systems using MassiveDS, a large-scale datastore with mixture of knowledge, and identified critical limitations: retrieval mainly benefits smaller models, rerankers add minimal value, and no single retrieval source consistently excels. Moreover, current LLMs struggle to route queries across heterogeneous knowledge sources. These findings highlight the need for adaptive retrieval strategies before deploying RAG in real-world settings. Our code and data can be found at https://github.com/ritaranx/RAG_in_the_Wild.
>
---
#### [new 054] Latent Inter-User Difference Modeling for LLM Personalization
- **分类: cs.CL**

- **简介: 该论文属于大语言模型个性化任务，旨在解决现有方法忽视用户间差异、依赖语言提示的问题。作者提出DEP框架，通过在潜在空间中建模用户差异，结合稀疏自编码器提取任务相关特征，实现更有效的个性化输出。**

- **链接: [http://arxiv.org/pdf/2507.20849v1](http://arxiv.org/pdf/2507.20849v1)**

> **作者:** Yilun Qiu; Tianhao Shi; Xiaoyan Zhao; Fengbin Zhu; Yang Zhang; Fuli Feng
>
> **摘要:** Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
>
---
#### [new 055] FHSTP@EXIST 2025 Benchmark: Sexism Detection with Transparent Speech Concept Bottleneck Models
- **分类: cs.CL; cs.AI; cs.CY; cs.SI; I.2**

- **简介: 该论文参与EXIST 2025挑战赛，旨在识别和分类社交媒体中的性别歧视内容。论文提出三种模型：SCBM、SCBMT和XLM-RoBERTa，用于解决三个子任务：性别歧视识别、意图识别和分类。重点在于使用可解释的瓶颈概念（如形容词）结合大语言模型和Transformer，提升模型解释性与性能，并探索元数据的辅助作用。**

- **链接: [http://arxiv.org/pdf/2507.20924v1](http://arxiv.org/pdf/2507.20924v1)**

> **作者:** Roberto Labadie-Tamayo; Adrian Jaques Böck; Djordje Slijepčević; Xihui Chen; Andreas Babic; Matthias Zeppelzauer
>
> **备注:** 12 pages
>
> **摘要:** Sexism has become widespread on social media and in online conversation. To help address this issue, the fifth Sexism Identification in Social Networks (EXIST) challenge is initiated at CLEF 2025. Among this year's international benchmarks, we concentrate on solving the first task aiming to identify and classify sexism in social media textual posts. In this paper, we describe our solutions and report results for three subtasks: Subtask 1.1 - Sexism Identification in Tweets, Subtask 1.2 - Source Intention in Tweets, and Subtask 1.3 - Sexism Categorization in Tweets. We implement three models to address each subtask which constitute three individual runs: Speech Concept Bottleneck Model (SCBM), Speech Concept Bottleneck Model with Transformer (SCBMT), and a fine-tuned XLM-RoBERTa transformer model. SCBM uses descriptive adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to encode input texts into a human-interpretable representation of adjectives, then used to train a lightweight classifier for downstream tasks. SCBMT extends SCBM by fusing adjective-based representation with contextual embeddings from transformers to balance interpretability and classification performance. Beyond competitive results, these two models offer fine-grained explanations at both instance (local) and class (global) levels. We also investigate how additional metadata, e.g., annotators' demographic profiles, can be leveraged. For Subtask 1.1, XLM-RoBERTa, fine-tuned on provided data augmented with prior datasets, ranks 6th for English and Spanish and 4th for English in the Soft-Soft evaluation. Our SCBMT achieves 7th for English and Spanish and 6th for Spanish.
>
---
#### [new 056] HITSZ's End-To-End Speech Translation Systems Combining Sequence-to-Sequence Auto Speech Recognition Model and Indic Large Language Model for IWSLT 2025 in Indic Track
- **分类: cs.CL**

- **简介: 该论文属于语音翻译任务，旨在解决低资源场景下英印语种互译问题。工作内容为构建端到端系统，结合预训练Whisper语音识别模型与Krutrim印地语大模型，并探索思维链方法提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2507.19616v1](http://arxiv.org/pdf/2507.19616v1)**

> **作者:** Xuchen Wei; Yangxin Wu; Yaoyin Zhang; Henglyu Liu; Kehai Chen; Xuefeng Bai; Min Zhang
>
> **备注:** 7 pages, 1 figure, submitted to IWSLT 2025
>
> **摘要:** This paper presents HITSZ's submission for the IWSLT 2025 Indic track, focusing on speech-to-text translation (ST) for English-to-Indic and Indic-to-English language pairs. To enhance translation quality in this low-resource scenario, we propose an end-to-end system integrating the pre-trained Whisper automated speech recognition (ASR) model with Krutrim, an Indic-specialized large language model (LLM). Experimental results demonstrate that our end-to-end system achieved average BLEU scores of $28.88$ for English-to-Indic directions and $27.86$ for Indic-to-English directions. Furthermore, we investigated the Chain-of-Thought (CoT) method. While this method showed potential for significant translation quality improvements on successfully parsed outputs (e.g. a $13.84$ BLEU increase for Tamil-to-English), we observed challenges in ensuring the model consistently adheres to the required CoT output format.
>
---
#### [new 057] MoL-RL: Distilling Multi-Step Environmental Feedback into LLMs for Feedback-Independent Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型难以有效利用多步环境反馈信号进行推理的问题。作者提出MoL-RL方法，结合MoL持续训练与GRPO后训练，将多步反馈信号融入模型，实现无需外部反馈的独立推理。实验表明其在数学和代码生成任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.20278v1](http://arxiv.org/pdf/2507.20278v1)**

> **作者:** Kang Yang; Jingxue Chen; Qingkun Tang; Tianxiang Zhang; Qianchun Lu
>
> **备注:** 12pages,3figures
>
> **摘要:** Large language models (LLMs) face significant challenges in effectively leveraging sequential environmental feedback (EF) signals, such as natural language evaluations, for feedback-independent chain-of-thought (CoT) reasoning. Existing approaches either convert EF into scalar rewards, losing rich contextual information, or employ refinement datasets, failing to exploit the multi-step and discrete nature of EF interactions. To address these limitations, we propose MoL-RL, a novel training paradigm that integrates multi-step EF signals into LLMs through a dual-objective optimization framework. Our method combines MoL (Mixture-of-Losses) continual training, which decouples domain-specific EF signals (optimized via cross-entropy loss) and general language capabilities (preserved via Kullback-Leibler divergence), with GRPO-based post-training to distill sequential EF interactions into single-step inferences. This synergy enables robust feedback-independent reasoning without relying on external feedback loops. Experimental results on mathematical reasoning (MATH-500, AIME24/AIME25) and code generation (CodeAgent-Test) benchmarks demonstrate that MoL-RL achieves state-of-the-art performance with the Qwen3-8B model, while maintaining strong generalization across model scales (Qwen3-4B). This work provides a promising approach for leveraging multi-step textual feedback to enhance LLMs' reasoning capabilities in diverse domains.
>
---
#### [new 058] Survey of NLU Benchmarks Diagnosing Linguistic Phenomena: Why not Standardize Diagnostics Benchmarks?
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于自然语言理解（NLU）任务，旨在解决当前NLU诊断基准缺乏统一标准的问题。论文调研了多种语言的NLU基准，重点分析其诊断数据集和覆盖的语言现象，比较其优劣，并探讨建立统一评估标准的必要性。**

- **链接: [http://arxiv.org/pdf/2507.20419v1](http://arxiv.org/pdf/2507.20419v1)**

> **作者:** Khloud AL Jallad; Nada Ghneim; Ghaida Rebdawi
>
> **摘要:** Natural Language Understanding (NLU) is a basic task in Natural Language Processing (NLP). The evaluation of NLU capabilities has become a trending research topic that attracts researchers in the last few years, resulting in the development of numerous benchmarks. These benchmarks include various tasks and datasets in order to evaluate the results of pretrained models via public leaderboards. Notably, several benchmarks contain diagnostics datasets designed for investigation and fine-grained error analysis across a wide range of linguistic phenomena. This survey provides a comprehensive review of available English, Arabic, and Multilingual NLU benchmarks, with a particular emphasis on their diagnostics datasets and the linguistic phenomena they covered. We present a detailed comparison and analysis of these benchmarks, highlighting their strengths and limitations in evaluating NLU tasks and providing in-depth error analysis. When highlighting the gaps in the state-of-the-art, we noted that there is no naming convention for macro and micro categories or even a standard set of linguistic phenomena that should be covered. Consequently, we formulated a research question regarding the evaluation metrics of the evaluation diagnostics benchmarks: "Why do not we have an evaluation standard for the NLU evaluation diagnostics benchmarks?" similar to ISO standard in industry. We conducted a deep analysis and comparisons of the covered linguistic phenomena in order to support experts in building a global hierarchy for linguistic phenomena in future. We think that having evaluation metrics for diagnostics evaluation could be valuable to gain more insights when comparing the results of the studied models on different diagnostics benchmarks.
>
---
#### [new 059] Advancing Mental Disorder Detection: A Comparative Evaluation of Transformer and LSTM Architectures on Social Media
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决社交媒体中心理健康障碍的自动检测问题。通过比较Transformer与LSTM模型的分类性能，使用Reddit数据构建了大规模标注数据集，并验证了Transformer模型（尤其是RoBERTa）在心理健康分类中的优越性。**

- **链接: [http://arxiv.org/pdf/2507.19511v1](http://arxiv.org/pdf/2507.19511v1)**

> **作者:** Khalid Hasan; Jamil Saquer; Mukulika Ghosh
>
> **备注:** The 49th IEEE International Conference on Computers, Software, and Applications (COMPSAC 2025) (camera-ready)
>
> **摘要:** The rising prevalence of mental health disorders necessitates the development of robust, automated tools for early detection and monitoring. Recent advances in Natural Language Processing (NLP), particularly transformer-based architectures, have demonstrated significant potential in text analysis. This study provides a comprehensive evaluation of state-of-the-art transformer models (BERT, RoBERTa, DistilBERT, ALBERT, and ELECTRA) against Long Short-Term Memory (LSTM) based approaches using different text embedding techniques for mental health disorder classification on Reddit. We construct a large annotated dataset, validating its reliability through statistical judgmental analysis and topic modeling. Experimental results demonstrate the superior performance of transformer models over traditional deep-learning approaches. RoBERTa achieved the highest classification performance, with a 99.54% F1 score on the hold-out test set and a 96.05% F1 score on the external test set. Notably, LSTM models augmented with BERT embeddings proved highly competitive, achieving F1 scores exceeding 94% on the external dataset while requiring significantly fewer computational resources. These findings highlight the effectiveness of transformer-based models for real-time, scalable mental health monitoring. We discuss the implications for clinical applications and digital mental health interventions, offering insights into the capabilities and limitations of state-of-the-art NLP methodologies in mental disorder detection.
>
---
#### [new 060] Advancing Dialectal Arabic to Modern Standard Arabic Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决方言阿拉伯语（DA）到标准阿拉伯语（MSA）翻译难题。论文评估了无需训练的提示方法，并开发了资源高效的微调流程，提升了翻译质量，尤其在低资源环境下表现突出。**

- **链接: [http://arxiv.org/pdf/2507.20301v1](http://arxiv.org/pdf/2507.20301v1)**

> **作者:** Abdullah Alabdullah; Lifeng Han; Chenghua Lin
>
> **摘要:** Dialectal Arabic (DA) poses a persistent challenge for natural language processing (NLP), as most everyday communication in the Arab world occurs in dialects that diverge significantly from Modern Standard Arabic (MSA). This linguistic divide limits access to digital services and educational resources and impedes progress in Arabic machine translation. This paper presents two core contributions to advancing DA-MSA translation for the Levantine, Egyptian, and Gulf dialects, particularly in low-resource and computationally constrained settings: a comprehensive evaluation of training-free prompting techniques, and the development of a resource-efficient fine-tuning pipeline. Our evaluation of prompting strategies across six large language models (LLMs) found that few-shot prompting consistently outperformed zero-shot, chain-of-thought, and our proposed Ara-TEaR method. GPT-4o achieved the highest performance across all prompting settings. For fine-tuning, a quantized Gemma2-9B model achieved a CHrF++ score of 49.88, outperforming zero-shot GPT-4o (44.58). Joint multi-dialect trained models outperformed single-dialect counterparts by over 10% CHrF++, and 4-bit quantization reduced memory usage by 60% with less than 1% performance loss. The results and insights of our experiments offer a practical blueprint for improving dialectal inclusion in Arabic NLP, showing that high-quality DA-MSA machine translation is achievable even with limited resources and paving the way for more inclusive language technologies.
>
---
#### [new 061] Length Representations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型内部如何编码控制输出长度的机制，属于自然语言生成任务。通过分析注意力机制与隐藏单元，发现可通过调整特定单元控制长度，且长度信息部分与语义信息分离，表明模型具备内部调控长度的能力。**

- **链接: [http://arxiv.org/pdf/2507.20398v1](http://arxiv.org/pdf/2507.20398v1)**

> **作者:** Sangjun Moon; Dasom Choi; Jingun Kwon; Hidetaka Kamigaito; Manabu Okumura
>
> **摘要:** Large language models (LLMs) have shown remarkable capabilities across various tasks, that are learned from massive amounts of text-based data. Although LLMs can control output sequence length, particularly in instruction-based settings, the internal mechanisms behind this control have been unexplored yet. In this study, we provide empirical evidence on how output sequence length information is encoded within the internal representations in LLMs. In particular, our findings show that multi-head attention mechanisms are critical in determining output sequence length, which can be adjusted in a disentangled manner. By scaling specific hidden units within the model, we can control the output sequence length without losing the informativeness of the generated text, thereby indicating that length information is partially disentangled from semantic information. Moreover, some hidden units become increasingly active as prompts become more length-specific, thus reflecting the model's internal awareness of this attribute. Our findings suggest that LLMs have learned robust and adaptable internal mechanisms for controlling output length without any external control.
>
---
#### [new 062] SGPO: Self-Generated Preference Optimization based on Self-Improver
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在解决依赖人工标注数据和分布偏移问题。提出SGPO方法，通过自改进机制生成偏好数据，用于优化策略模型，无需外部标注数据，提升了性能。**

- **链接: [http://arxiv.org/pdf/2507.20181v1](http://arxiv.org/pdf/2507.20181v1)**

> **作者:** Hyeonji Lee; Daejin Jo; Seohwan Yun; Sungwoong Kim
>
> **摘要:** Large language models (LLMs), despite their extensive pretraining on diverse datasets, require effective alignment to human preferences for practical and reliable deployment. Conventional alignment methods typically employ off-policy learning and depend on human-annotated datasets, which limits their broad applicability and introduces distribution shift issues during training. To address these challenges, we propose Self-Generated Preference Optimization based on Self-Improver (SGPO), an innovative alignment framework that leverages an on-policy self-improving mechanism. Specifically, the improver refines responses from a policy model to self-generate preference data for direct preference optimization (DPO) of the policy model. Here, the improver and policy are unified into a single model, and in order to generate higher-quality preference data, this self-improver learns to make incremental yet discernible improvements to the current responses by referencing supervised fine-tuning outputs. Experimental results on AlpacaEval 2.0 and Arena-Hard show that the proposed SGPO significantly improves performance over DPO and baseline self-improving methods without using external preference data.
>
---
#### [new 063] Infogen: Generating Complex Statistical Infographics from Documents
- **分类: cs.CL**

- **简介: 该论文属于文本到统计信息图生成任务，旨在解决从文本生成复杂信息图的问题。作者提出了Infogen框架，通过生成元数据再转换为信息图代码，实现了多图表组合的信息图生成，并创建了首个相关数据集Infodat。**

- **链接: [http://arxiv.org/pdf/2507.20046v1](http://arxiv.org/pdf/2507.20046v1)**

> **作者:** Akash Ghosh; Aparna Garimella; Pritika Ramu; Sambaran Bandyopadhyay; Sriparna Saha
>
> **备注:** ACL Main 2025
>
> **摘要:** Statistical infographics are powerful tools that simplify complex data into visually engaging and easy-to-understand formats. Despite advancements in AI, particularly with LLMs, existing efforts have been limited to generating simple charts, with no prior work addressing the creation of complex infographics from text-heavy documents that demand a deep understanding of the content. We address this gap by introducing the task of generating statistical infographics composed of multiple sub-charts (e.g., line, bar, pie) that are contextually accurate, insightful, and visually aligned. To achieve this, we define infographic metadata that includes its title and textual insights, along with sub-chart-specific details such as their corresponding data and alignment. We also present Infodat, the first benchmark dataset for text-to-infographic metadata generation, where each sample links a document to its metadata. We propose Infogen, a two-stage framework where fine-tuned LLMs first generate metadata, which is then converted into infographic code. Extensive evaluations on Infodat demonstrate that Infogen achieves state-of-the-art performance, outperforming both closed and open-source LLMs in text-to-statistical infographic generation.
>
---
#### [new 064] On The Role of Pretrained Language Models in General-Purpose Text Embeddings: A Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探讨预训练语言模型在通用文本嵌入中的作用。论文分析了预训练语言模型如何推动通用文本嵌入的发展，涵盖其在架构设计、表达增强、训练策略等方面的基本角色及多语言、多模态等高级功能，并展望了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.20783v1](http://arxiv.org/pdf/2507.20783v1)**

> **作者:** Meishan Zhang; Xin Zhang; Xinping Zhao; Shouzheng Huang; Baotian Hu; Min Zhang
>
> **备注:** 45 pages, 2 figures, 9 tables
>
> **摘要:** Text embeddings have attracted growing interest due to their effectiveness across a wide range of natural language processing (NLP) tasks, such as retrieval, classification, clustering, bitext mining, and summarization. With the emergence of pretrained language models (PLMs), general-purpose text embeddings (GPTE) have gained significant traction for their ability to produce rich, transferable representations. The general architecture of GPTE typically leverages PLMs to derive dense text representations, which are then optimized through contrastive learning on large-scale pairwise datasets. In this survey, we provide a comprehensive overview of GPTE in the era of PLMs, focusing on the roles PLMs play in driving its development. We first examine the fundamental architecture and describe the basic roles of PLMs in GPTE, i.e., embedding extraction, expressivity enhancement, training strategies, learning objectives, and data construction. Then, we describe advanced roles enabled by PLMs, such as multilingual support, multimodal integration, code understanding, and scenario-specific adaptation. Finally, we highlight potential future research directions that move beyond traditional improvement goals, including ranking integration, safety considerations, bias mitigation, structural information incorporation, and the cognitive extension of embeddings. This survey aims to serve as a valuable reference for both newcomers and established researchers seeking to understand the current state and future potential of GPTE.
>
---
#### [new 065] Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于多模态检索增强生成（RAG）任务，旨在解决视觉语言模型在处理多模态、多轮问答时易产生幻觉的问题。论文提出了一种多阶段验证为核心框架的方法，通过查询路由、检索与摘要、双路径生成及事后验证等模块，提升答案的准确性和可靠性，减少幻觉现象。**

- **链接: [http://arxiv.org/pdf/2507.20136v1](http://arxiv.org/pdf/2507.20136v1)**

> **作者:** Baiyu Chen; Wilson Wongso; Xiaoqian Hu; Yue Tan; Flora Salim
>
> **备注:** KDD Cup 2025 Meta CRAG-MM Challenge
>
> **摘要:** This paper presents the technical solution developed by team CRUISE for the KDD Cup 2025 Meta Comprehensive RAG Benchmark for Multi-modal, Multi-turn (CRAG-MM) challenge. The challenge aims to address a critical limitation of modern Vision Language Models (VLMs): their propensity to hallucinate, especially when faced with egocentric imagery, long-tail entities, and complex, multi-hop questions. This issue is particularly problematic in real-world applications where users pose fact-seeking queries that demand high factual accuracy across diverse modalities. To tackle this, we propose a robust, multi-stage framework that prioritizes factual accuracy and truthfulness over completeness. Our solution integrates a lightweight query router for efficiency, a query-aware retrieval and summarization pipeline, a dual-pathways generation and a post-hoc verification. This conservative strategy is designed to minimize hallucinations, which incur a severe penalty in the competition's scoring metric. Our approach achieved 3rd place in Task 1, demonstrating the effectiveness of prioritizing answer reliability in complex multi-modal RAG systems. Our implementation is available at https://github.com/Breezelled/KDD-Cup-2025-Meta-CRAG-MM .
>
---
#### [new 066] Mind the Gap: Conformative Decoding to Improve Output Diversity of Instruction-Tuned Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决指令微调导致大语言模型输出多样性下降的问题。通过分析不同模型及微调阶段对多样性的影响，发现DPO对多样性影响最大。为此，提出了一种新的解码方法——共形解码，利用基础模型提升指令模型的输出多样性，同时保持或提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.20956v1](http://arxiv.org/pdf/2507.20956v1)**

> **作者:** Max Peeperkorn; Tom Kouwenhoven; Dan Brown; Anna Jordanous
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Instruction-tuning large language models (LLMs) reduces the diversity of their outputs, which has implications for many tasks, particularly for creative tasks. This paper investigates the ``diversity gap'' for a writing prompt narrative generation task. This gap emerges as measured by current diversity metrics for various open-weight and open-source LLMs. The results show significant decreases in diversity due to instruction-tuning. We explore the diversity loss at each fine-tuning stage for the OLMo and OLMo 2 models to further understand how output diversity is affected. The results indicate that DPO has the most substantial impact on diversity. Motivated by these findings, we present a new decoding strategy, conformative decoding, which guides an instruct model using its more diverse base model to reintroduce output diversity. We show that conformative decoding typically increases diversity and even maintains or improves quality.
>
---
#### [new 067] Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation
- **分类: cs.CL; 68T50, 68T05; I.2.7; I.7**

- **简介: 该论文属于新闻推荐任务，旨在解决信息过载问题。它通过结合多视角新闻建模（NAML）与长短期用户兴趣建模（LSTUR），提出Co-NAML-LSTUR模型，更准确地捕捉用户动态兴趣和新闻语义特征，提升推荐效果。**

- **链接: [http://arxiv.org/pdf/2507.20210v1](http://arxiv.org/pdf/2507.20210v1)**

> **作者:** Minh Hoang Nguyen; Thuat Thien Nguyen; Minh Nhat Ta
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** News recommendation systems play a vital role in mitigating information overload by delivering personalized news content. A central challenge is to effectively model both multi-view news representations and the dynamic nature of user interests, which often span both short- and long-term preferences. Existing methods typically rely on single-view features of news articles (e.g., titles or categories) or fail to comprehensively capture user preferences across time scales. In this work, we propose Co-NAML-LSTUR, a hybrid news recommendation framework that integrates NAML for attentive multi-view news modeling and LSTUR for capturing both long- and short-term user representations. Our model also incorporates BERT-based word embeddings to enhance semantic feature extraction. We evaluate Co-NAML-LSTUR on two widely used benchmarks, MIND-small and MIND-large. Experimental results show that Co-NAML-LSTUR achieves substantial improvements over most state-of-the-art baselines on MIND-small and MIND-large, respectively. These results demonstrate the effectiveness of combining multi-view news representations with dual-scale user modeling. The implementation of our model is publicly available at https://github.com/MinhNguyenDS/Co-NAML-LSTUR.
>
---
#### [new 068] DYNARTmo: A Dynamic Articulatory Model for Visualization of Speech Movement Patterns
- **分类: cs.CL**

- **简介: 论文提出了DYNARTmo，一个用于可视化语音运动模式的动态发音模型。该模型基于UK-DYNAMO框架，整合发音欠定、片段与动作控制及协同发音原理，可模拟六种关键发音器官，适用于语音教学和言语治疗。**

- **链接: [http://arxiv.org/pdf/2507.20343v1](http://arxiv.org/pdf/2507.20343v1)**

> **作者:** Bernd J. Kröger
>
> **备注:** 10 pages, 29 references, 2 figures, supplementary material
>
> **摘要:** We present DYNARTmo, a dynamic articulatory model designed to visualize speech articulation processes in a two-dimensional midsagittal plane. The model builds upon the UK-DYNAMO framework and integrates principles of articulatory underspecification, segmental and gestural control, and coarticulation. DYNARTmo simulates six key articulators based on ten continuous and six discrete control parameters, allowing for the generation of both vocalic and consonantal articulatory configurations. The current implementation is embedded in a web-based application (SpeechArticulationTrainer) that includes sagittal, glottal, and palatal views, making it suitable for use in phonetics education and speech therapy. While this paper focuses on the static modeling aspects, future work will address dynamic movement generation and integration with articulatory-acoustic modules.
>
---
#### [new 069] Automating Thematic Review of Prevention of Future Deaths Reports: Replicating the ONS Child Suicide Study using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与公共健康数据分析任务，旨在解决手动分析预防未来死亡报告效率低的问题。研究使用开源大语言模型工具包（PFD Toolkit）自动化识别与编码儿童自杀相关报告，较人工方法提升效率与覆盖范围，实现快速、可复现的分析。**

- **链接: [http://arxiv.org/pdf/2507.20786v1](http://arxiv.org/pdf/2507.20786v1)**

> **作者:** Sam Osian; Arpan Dutta; Sahil Bhandari; Iain E. Buchan; Dan W. Joyce
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** Prevention of Future Deaths (PFD) reports, issued by coroners in England and Wales, flag systemic hazards that may lead to further loss of life. Analysis of these reports has previously been constrained by the manual effort required to identify and code relevant cases. In 2025, the Office for National Statistics (ONS) published a national thematic review of child-suicide PFD reports ($\leq$ 18 years), identifying 37 cases from January 2015 to November 2023 - a process based entirely on manual curation and coding. We evaluated whether a fully automated, open source "text-to-table" language-model pipeline (PFD Toolkit) could reproduce the ONS's identification and thematic analysis of child-suicide PFD reports, and assessed gains in efficiency and reliability. All 4,249 PFD reports published from July 2013 to November 2023 were processed via PFD Toolkit's large language model pipelines. Automated screening identified cases where the coroner attributed death to suicide in individuals aged 18 or younger, and eligible reports were coded for recipient category and 23 concern sub-themes, replicating the ONS coding frame. PFD Toolkit identified 72 child-suicide PFD reports - almost twice the ONS count. Three blinded clinicians adjudicated a stratified sample of 144 reports to validate the child-suicide screening. Against the post-consensus clinical annotations, the LLM-based workflow showed substantial to almost-perfect agreement (Cohen's $\kappa$ = 0.82, 95% CI: 0.66-0.98, raw agreement = 91%). The end-to-end script runtime was 8m 16s, transforming a process that previously took months into one that can be completed in minutes. This demonstrates that automated LLM analysis can reliably and efficiently replicate manual thematic reviews of coronial data, enabling scalable, reproducible, and timely insights for public health and safety. The PFD Toolkit is openly available for future research.
>
---
#### [new 070] RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务中的角色扮演评估领域，旨在解决现有大语言模型角色扮演评估方法过于以角色为中心、无法反映真实用户需求的问题。论文提出了RMTBench这一用户中心的双语角色扮演基准，包含80个角色和8000多轮对话，通过基于用户意图的多轮对话机制和LLM评分，更真实地评估模型的角色扮演能力。**

- **链接: [http://arxiv.org/pdf/2507.20352v1](http://arxiv.org/pdf/2507.20352v1)**

> **作者:** Hao Xiang; Tianyi Tang; Yang Su; Bowen Yu; An Yang; Fei Huang; Yichang Zhang; Yaojie Lu; Hongyu Lin; Xianpei Han; Jingren Zhou; Junyang Lin; Le Sun
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon.
>
---
#### [new 071] CONCAP: Seeing Beyond English with Concepts Retrieval-Augmented Captioning
- **分类: cs.CL**

- **简介: 论文提出CONCAP，一种多语言图像描述生成模型，通过结合图像概念与检索到的多语言描述，解决因多语言数据不足和翻译偏差导致的性能问题，旨在提升低资源语言的图像描述质量。**

- **链接: [http://arxiv.org/pdf/2507.20411v1](http://arxiv.org/pdf/2507.20411v1)**

> **作者:** George Ibrahim; Rita Ramos; Yova Kementchedjhieva
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Multilingual vision-language models have made significant strides in image captioning, yet they still lag behind their English counterparts due to limited multilingual training data and costly large-scale model parameterization. Retrieval-augmented generation (RAG) offers a promising alternative by conditioning caption generation on retrieved examples in the target language, reducing the need for extensive multilingual training. However, multilingual RAG captioning models often depend on retrieved captions translated from English, which can introduce mismatches and linguistic biases relative to the source language. We introduce CONCAP, a multilingual image captioning model that integrates retrieved captions with image-specific concepts, enhancing the contextualization of the input image and grounding the captioning process across different languages. Experiments on the XM3600 dataset indicate that CONCAP enables strong performance on low- and mid-resource languages, with highly reduced data requirements. Our findings highlight the effectiveness of concept-aware retrieval augmentation in bridging multilingual performance gaps.
>
---
#### [new 072] Mitigating Geospatial Knowledge Hallucination in Large Language Models: Benchmarking and Dynamic Factuality Aligning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理与地理空间分析交叉任务，旨在解决大语言模型（LLMs）在生成地理空间知识时出现的幻觉问题。作者构建了评估框架，并提出基于Kahneman-Tversky优化的动态事实对齐方法，提升LLMs在地理空间任务中的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.19586v1](http://arxiv.org/pdf/2507.19586v1)**

> **作者:** Shengyuan Wang; Jie Feng; Tianhui Liu; Dan Pei; Yong Li
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Large language models (LLMs) possess extensive world knowledge, including geospatial knowledge, which has been successfully applied to various geospatial tasks such as mobility prediction and social indicator prediction. However, LLMs often generate inaccurate geospatial knowledge, leading to geospatial hallucinations (incorrect or inconsistent representations of geospatial information) that compromise their reliability. While the phenomenon of general knowledge hallucination in LLMs has been widely studied, the systematic evaluation and mitigation of geospatial hallucinations remain largely unexplored. To address this gap, we propose a comprehensive evaluation framework for geospatial hallucinations, leveraging structured geospatial knowledge graphs for controlled assessment. Through extensive evaluation across 20 advanced LLMs, we uncover the hallucinations in their geospatial knowledge. Building on these insights, we introduce a dynamic factuality aligning method based on Kahneman-Tversky Optimization (KTO) to mitigate geospatial hallucinations in LLMs, leading to a performance improvement of over 29.6% on the proposed benchmark. Extensive experimental results demonstrate the effectiveness of our benchmark and learning algorithm in enhancing the trustworthiness of LLMs in geospatial knowledge and reasoning tasks.
>
---
#### [new 073] Basic Reading Distillation
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型因计算资源需求高而难以部署的问题。通过提出基本阅读蒸馏（BRD）方法，让小模型模仿大模型的基本阅读行为，从而在多种任务上达到与大模型相当的性能。**

- **链接: [http://arxiv.org/pdf/2507.19741v1](http://arxiv.org/pdf/2507.19741v1)**

> **作者:** Zhi Zhou; Sirui Miao; Xiangyu Duan; Hao Yang; Min Zhang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable abilities in various natural language processing areas, but they demand high computation resources which limits their deployment in real-world. Distillation is one technique to solve this problem through either knowledge distillation or task distillation. Both distillation approaches train small models to imitate specific features of LLMs, but they all neglect basic reading education for small models on generic texts that are \emph{unrelated} to downstream tasks. In this paper, we propose basic reading distillation (BRD) which educates a small model to imitate LLMs basic reading behaviors, such as named entity recognition, question raising and answering, on each sentence. After such basic education, we apply the small model on various tasks including language inference benchmarks and BIG-bench tasks. It shows that the small model can outperform or perform comparable to over 20x bigger LLMs. Analysis reveals that BRD effectively influences the probability distribution of the small model, and has orthogonality to either knowledge distillation or task distillation.
>
---
#### [new 074] Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory
- **分类: cs.CL**

- **简介: 该论文属于教育评估任务，旨在评估大型语言模型（LLMs）在大规模写作评分中的可靠性。研究使用概化理论，比较人类评分员与AI评分员在AP中文写作任务（包括故事叙述和邮件回应）中的评分一致性。结果发现，人类评分更可靠，但LLMs在特定条件下（尤其是故事叙述）表现合理，结合人机评分的混合模型提升了整体可靠性。**

- **链接: [http://arxiv.org/pdf/2507.19980v1](http://arxiv.org/pdf/2507.19980v1)**

> **作者:** Dan Song; Won-Chan Lee; Hong Jiao
>
> **摘要:** This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments.
>
---
#### [new 075] When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification
- **分类: cs.CL**

- **简介: 该论文属于多语言事实核查任务，旨在解决多语言虚假信息的细粒度验证问题。作者评估了五种语言模型在25种语言中的表现，发现小型专用模型如XLM-R比大型通用模型更有效，取得了更高的准确率，并揭示了大型模型在证据利用和类别偏差上的问题。**

- **链接: [http://arxiv.org/pdf/2507.20700v1](http://arxiv.org/pdf/2507.20700v1)**

> **作者:** Hanna Shcharbakova; Tatiana Anikina; Natalia Skachkova; Josef van Genabith
>
> **备注:** Published at the FEVER Workshop, ACL 2025
>
> **摘要:** The rapid spread of multilingual misinformation requires robust automated fact verification systems capable of handling fine-grained veracity assessments across diverse languages. While large language models have shown remarkable capabilities across many NLP tasks, their effectiveness for multilingual claim verification with nuanced classification schemes remains understudied. We conduct a comprehensive evaluation of five state-of-the-art language models on the X-Fact dataset, which spans 25 languages with seven distinct veracity categories. Our experiments compare small language models (encoder-based XLM-R and mT5) with recent decoder-only LLMs (Llama 3.1, Qwen 2.5, Mistral Nemo) using both prompting and fine-tuning approaches. Surprisingly, we find that XLM-R (270M parameters) substantially outperforms all tested LLMs (7-12B parameters), achieving 57.7% macro-F1 compared to the best LLM performance of 16.9%. This represents a 15.8% improvement over the previous state-of-the-art (41.9%), establishing new performance benchmarks for multilingual fact verification. Our analysis reveals problematic patterns in LLM behavior, including systematic difficulties in leveraging evidence and pronounced biases toward frequent categories in imbalanced data settings. These findings suggest that for fine-grained multilingual fact verification, smaller specialized models may be more effective than general-purpose large models, with important implications for practical deployment of fact-checking systems.
>
---
#### [new 076] VLQA: The First Comprehensive, Large, and High-Quality Vietnamese Dataset for Legal Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律问答任务，旨在解决越南语法律领域缺乏高质量标注数据的问题。作者构建了首个全面、大规模、高质量的越南语法律问答数据集VLQA，并进行了统计分析与实验验证，以推动低资源语言的法律文本处理研究。**

- **链接: [http://arxiv.org/pdf/2507.19995v1](http://arxiv.org/pdf/2507.19995v1)**

> **作者:** Tan-Minh Nguyen; Hoang-Trung Nguyen; Trong-Khoi Dao; Xuan-Hieu Phan; Ha-Thanh Nguyen; Thi-Hai-Yen Vuong
>
> **摘要:** The advent of large language models (LLMs) has led to significant achievements in various domains, including legal text processing. Leveraging LLMs for legal tasks is a natural evolution and an increasingly compelling choice. However, their capabilities are often portrayed as greater than they truly are. Despite the progress, we are still far from the ultimate goal of fully automating legal tasks using artificial intelligence (AI) and natural language processing (NLP). Moreover, legal systems are deeply domain-specific and exhibit substantial variation across different countries and languages. The need for building legal text processing applications for different natural languages is, therefore, large and urgent. However, there is a big challenge for legal NLP in low-resource languages such as Vietnamese due to the scarcity of resources and annotated data. The need for labeled legal corpora for supervised training, validation, and supervised fine-tuning is critical. In this paper, we introduce the VLQA dataset, a comprehensive and high-quality resource tailored for the Vietnamese legal domain. We also conduct a comprehensive statistical analysis of the dataset and evaluate its effectiveness through experiments with state-of-the-art models on legal information retrieval and question-answering tasks.
>
---
#### [new 077] FAEDKV: Infinite-Window Fourier Transform for Unbiased KV Cache Compression
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型处理长文本时键值缓存（KV Cache）占用资源过高的问题。现有压缩方法存在信息偏差或依赖模型重训练，而论文提出FAEDKV，通过无限窗口傅里叶变换实现无偏压缩，保留所有上下文信息。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.20030v1](http://arxiv.org/pdf/2507.20030v1)**

> **作者:** Runchao Li; Yao Fu; Mu Sheng; Xianxuan Long; Haotian Yu; Pan Li
>
> **摘要:** The efficacy of Large Language Models (LLMs) in long-context tasks is often hampered by the substantial memory footprint and computational demands of the Key-Value (KV) cache. Current compression strategies, including token eviction and learned projections, frequently lead to biased representations -- either by overemphasizing recent/high-attention tokens or by repeatedly degrading information from earlier context -- and may require costly model retraining. We present FAEDKV (Frequency-Adaptive Infinite-Window for KV cache), a novel, training-free KV cache compression framework that ensures unbiased information retention. FAEDKV operates by transforming the KV cache into the frequency domain using a proposed Infinite-Window Fourier Transform (IWDFT). This approach allows for the equalized contribution of all tokens to the compressed representation, effectively preserving both early and recent contextual information. A preliminary frequency ablation study identifies critical spectral components for layer-wise, targeted compression. Experiments on LongBench benchmark demonstrate FAEDKV's superiority over existing methods by up to 22\%. In addition, our method shows superior, position-agnostic retrieval accuracy on the Needle-In-A-Haystack task compared to compression based approaches.
>
---
#### [new 078] Zero-shot Performance of Generative AI in Brazilian Portuguese Medical Exam
- **分类: cs.CL**

- **简介: 该论文评估了生成式AI在巴西葡萄牙语医学考试中的零样本表现，属于多语言医学AI性能评估任务。旨在解决非英语AI模型在医疗场景中的性能偏差问题。研究团队测试了多个大语言模型和多模态模型对巴西医学考试题目的作答能力，并与人类考生对比，分析模型在语言、多模态推理等方面的差距。**

- **链接: [http://arxiv.org/pdf/2507.19885v1](http://arxiv.org/pdf/2507.19885v1)**

> **作者:** Cesar Augusto Madid Truyts; Amanda Gomes Rabelo; Gabriel Mesquita de Souza; Daniel Scaldaferri Lages; Adriano Jose Pereira; Uri Adrian Prync Flato; Eduardo Pontes dos Reis; Joaquim Edson Vieira; Paulo Sergio Panse Silveira; Edson Amaro Junior
>
> **摘要:** Artificial intelligence (AI) has shown the potential to revolutionize healthcare by improving diagnostic accuracy, optimizing workflows, and personalizing treatment plans. Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have achieved notable advancements in natural language processing and medical applications. However, the evaluation of these models has focused predominantly on the English language, leading to potential biases in their performance across different languages. This study investigates the capability of six LLMs (GPT-4.0 Turbo, LLaMA-3-8B, LLaMA-3-70B, Mixtral 8x7B Instruct, Titan Text G1-Express, and Command R+) and four MLLMs (Claude-3.5-Sonnet, Claude-3-Opus, Claude-3-Sonnet, and Claude-3-Haiku) to answer questions written in Brazilian spoken portuguese from the medical residency entrance exam of the Hospital das Cl\'inicas da Faculdade de Medicina da Universidade de S\~ao Paulo (HCFMUSP) - the largest health complex in South America. The performance of the models was benchmarked against human candidates, analyzing accuracy, processing time, and coherence of the generated explanations. The results show that while some models, particularly Claude-3.5-Sonnet and Claude-3-Opus, achieved accuracy levels comparable to human candidates, performance gaps persist, particularly in multimodal questions requiring image interpretation. Furthermore, the study highlights language disparities, emphasizing the need for further fine-tuning and data set augmentation for non-English medical AI applications. Our findings reinforce the importance of evaluating generative AI in various linguistic and clinical settings to ensure a fair and reliable deployment in healthcare. Future research should explore improved training methodologies, improved multimodal reasoning, and real-world clinical integration of AI-driven medical assistance.
>
---
#### [new 079] Multilingual Self-Taught Faithfulness Evaluators
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言文本评估任务，旨在解决现有信息忠实度评估方法依赖英语标注数据且难以扩展到多语言的问题。作者提出一种自学习评估框架，利用合成多语言摘要数据和跨语言迁移学习，无需大量标注数据即可提升多语言忠实度评估效果。**

- **链接: [http://arxiv.org/pdf/2507.20752v1](http://arxiv.org/pdf/2507.20752v1)**

> **作者:** Carlo Alfano; Aymen Al Marjani; Zeno Jonke; Amin Mantrach; Saab Mansour; Marcello Federico
>
> **摘要:** The growing use of large language models (LLMs) has increased the need for automatic evaluation systems, particularly to address the challenge of information hallucination. Although existing faithfulness evaluation approaches have shown promise, they are predominantly English-focused and often require expensive human-labeled training data for fine-tuning specialized models. As LLMs see increased adoption in multilingual contexts, there is a need for accurate faithfulness evaluators that can operate across languages without extensive labeled data. This paper presents Self-Taught Evaluators for Multilingual Faithfulness, a framework that learns exclusively from synthetic multilingual summarization data while leveraging cross-lingual transfer learning. Through experiments comparing language-specific and mixed-language fine-tuning approaches, we demonstrate a consistent relationship between an LLM's general language capabilities and its performance in language-specific evaluation tasks. Our framework shows improvements over existing baselines, including state-of-the-art English evaluators and machine translation-based approaches.
>
---
#### [new 080] Post-Completion Learning for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种名为Post-Completion Learning（PCL）的语言模型训练框架，旨在利用模型生成结束后的内容空间，提升模型的推理和自我评估能力。论文属于自然语言处理任务，重点解决传统训练方法忽略生成后学习机会的问题。论文工作包括设计白-box强化学习方法、双轨SFT优化及混合训练策略，实验证明其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.20252v1](http://arxiv.org/pdf/2507.20252v1)**

> **作者:** Xiang Fei; Siqi Wang; Shu Wei; Yuxiang Nie; Wei Shi; Hao Feng; Can Huang
>
> **摘要:** Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>}) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point. To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization. Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency.
>
---
#### [new 081] Dialogues of Dissent: Thematic and Rhetorical Dimensions of Hate and Counter-Hate Speech in Social Media Conversations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与社交媒体分析任务，旨在解决仇恨言论与反仇恨言论的识别与分析问题。作者提出多标签标注框架，从主题与修辞维度对社交媒体对话中的仇恨与反仇恨言论进行标注，并分析其互动模式，以理解仇恨传播机制及反制策略。**

- **链接: [http://arxiv.org/pdf/2507.20528v1](http://arxiv.org/pdf/2507.20528v1)**

> **作者:** Effi Levi; Gal Ron; Odelia Oshri; Shaul R. Shenhav
>
> **摘要:** We introduce a novel multi-labeled scheme for joint annotation of hate and counter-hate speech in social media conversations, categorizing hate and counter-hate messages into thematic and rhetorical dimensions. The thematic categories outline different discursive aspects of each type of speech, while the rhetorical dimension captures how hate and counter messages are communicated, drawing on Aristotle's Logos, Ethos and Pathos. We annotate a sample of 92 conversations, consisting of 720 tweets, and conduct statistical analyses, incorporating public metrics, to explore patterns of interaction between the thematic and rhetorical dimensions within and between hate and counter-hate speech. Our findings provide insights into the spread of hate messages on social media, the strategies used to counter them, and their potential impact on online behavior.
>
---
#### [new 082] The Policy Cliff: A Theoretical Analysis of Reward-Policy Maps in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于强化学习理论分析任务，旨在解决大语言模型中策略不稳定导致的安全问题。论文提出了奖励函数到策略映射的数学框架，揭示了策略脆弱性源于最优动作非唯一性，并分析多奖励设置下的稳定性机制，证明熵正则化可恢复稳定性。**

- **链接: [http://arxiv.org/pdf/2507.20150v1](http://arxiv.org/pdf/2507.20150v1)**

> **作者:** Xingcheng Xu
>
> **摘要:** Reinforcement learning (RL) plays a crucial role in shaping the behavior of large language and reasoning models (LLMs/LRMs). However, it often produces brittle and unstable policies, leading to critical failures such as spurious reasoning, deceptive alignment, and instruction disobedience that undermine the trustworthiness and safety of LLMs/LRMs. Currently, these issues lack a unified theoretical explanation and are typically addressed using ad-hoc heuristics. This paper presents a rigorous mathematical framework for analyzing the stability of the mapping from a reward function to the optimal policy. We show that policy brittleness often stems from non-unique optimal actions, a common occurrence when multiple valid traces exist in a reasoning task. This theoretical lens provides a unified explanation for a range of seemingly disparate failures, reframing them as rational outcomes of optimizing rewards that may be incomplete or noisy, especially in the presence of action degeneracy. We extend this analysis from the fundamental single-reward setting to the more realistic multi-reward RL across diverse domains, showing how stability is governed by an "effective reward" aggregation mechanism. We also prove that entropy regularization restores policy stability at the cost of increased stochasticity. Our framework provides a unified explanation for recent empirical findings on deceptive reasoning, instruction-following trade-offs, and RLHF-induced sophistry, and is further validated through perturbation experiments in multi-reward RL. This work advances policy-stability analysis from empirical heuristics towards a principled theory, offering essential insights for designing safer and more trustworthy AI systems.
>
---
#### [new 083] SciToolAgent: A Knowledge Graph-Driven Scientific Agent for Multi-Tool Integration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于工具自动化任务，旨在解决科学工具集成困难的问题。作者提出了SciToolAgent，基于知识图谱和大语言模型，实现多工具智能选择与执行，并包含安全检查模块。评估和案例研究表明其在复杂科研流程中的有效性与广泛适用性。**

- **链接: [http://arxiv.org/pdf/2507.20280v1](http://arxiv.org/pdf/2507.20280v1)**

> **作者:** Keyan Ding; Jing Yu; Junjie Huang; Yuchen Yang; Qiang Zhang; Huajun Chen
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** Scientific research increasingly relies on specialized computational tools, yet effectively utilizing these tools demands substantial domain expertise. While Large Language Models (LLMs) show promise in tool automation, they struggle to seamlessly integrate and orchestrate multiple tools for complex scientific workflows. Here, we present SciToolAgent, an LLM-powered agent that automates hundreds of scientific tools across biology, chemistry, and materials science. At its core, SciToolAgent leverages a scientific tool knowledge graph that enables intelligent tool selection and execution through graph-based retrieval-augmented generation. The agent also incorporates a comprehensive safety-checking module to ensure responsible and ethical tool usage. Extensive evaluations on a curated benchmark demonstrate that SciToolAgent significantly outperforms existing approaches. Case studies in protein engineering, chemical reactivity prediction, chemical synthesis, and metal-organic framework screening further demonstrate SciToolAgent's capability to automate complex scientific workflows, making advanced research tools accessible to both experts and non-experts.
>
---
#### [new 084] MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading
- **分类: q-fin.TR; cs.CL; cs.LG**

- **简介: 该论文属于金融交易任务，旨在解决传统方法在多模态数据整合与可解释性上的不足。作者提出了MountainLion，一个基于大语言模型的多模态、多智能体系统，通过分析文本新闻、K线图和交易信号图生成高质量金融报告，并支持动态策略调整与用户交互。**

- **链接: [http://arxiv.org/pdf/2507.20474v1](http://arxiv.org/pdf/2507.20474v1)**

> **作者:** Siyi Wu; Zhaoyang Guan; Leyi Zhao; Xinyuan Song; Xinyu Ying; Hanlin Zhang; Michele Pak; Yangfan He; Yi Xin; Jianhui Wang; Tianyu Shi
>
> **摘要:** Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
>
---
#### [new 085] LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决大模型微调中资源消耗高、参数利用不充分的问题。作者提出LoRA-PAR方法，结合“快慢思维”理论，将数据与参数划分为两类子系统，分别用监督微调和强化学习进行训练，以更高效地适配不同任务需求。**

- **链接: [http://arxiv.org/pdf/2507.20999v1](http://arxiv.org/pdf/2507.20999v1)**

> **作者:** Yining Huang; Bin Li; Keke Tang; Meilian Chen
>
> **备注:** 10 pages
>
> **摘要:** Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
>
---
#### [new 086] Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于人机交互与动作生成任务，旨在解决人机共舞中的非语言沟通问题。论文构建了大规模即兴萨尔萨舞数据集CoMPAS3D，包含动作标注与舞者水平信息，提出SalsaAgent模型，实现人机协同舞蹈生成，推动具身智能在社交互动与创意动作生成中的研究。**

- **链接: [http://arxiv.org/pdf/2507.19684v1](http://arxiv.org/pdf/2507.19684v1)**

> **作者:** Bermet Burkanova; Payam Jome Yazdian; Chuxuan Zhang; Trinity Evans; Paige Tuttösí; Angelica Lim
>
> **备注:** https://rosielab.github.io/compas3d
>
> **摘要:** Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation.
>
---
#### [new 087] $K^4$: Online Log Anomaly Detection Via Unsupervised Typicality Learning
- **分类: cs.LG; cs.CL; cs.DC**

- **简介: 论文属于日志异常检测任务，旨在解决现有方法依赖人工解析、检测速度慢及评估不真实的问题。工作提出无监督在线检测框架$K^4$，通过k近邻统计将日志嵌入转换为四维描述符，实现快速训练与推理，并在更真实的评估协议下取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.20051v1](http://arxiv.org/pdf/2507.20051v1)**

> **作者:** Weicong Chen; Vikash Singh; Zahra Rahmani; Debargha Ganguly; Mohsen Hariri; Vipin Chaudhary
>
> **摘要:** Existing Log Anomaly Detection (LogAD) methods are often slow, dependent on error-prone parsing, and use unrealistic evaluation protocols. We introduce $K^4$, an unsupervised and parser-independent framework for high-performance online detection. $K^4$ transforms arbitrary log embeddings into compact four-dimensional descriptors (Precision, Recall, Density, Coverage) using efficient k-nearest neighbor (k-NN) statistics. These descriptors enable lightweight detectors to accurately score anomalies without retraining. Using a more realistic online evaluation protocol, $K^4$ sets a new state-of-the-art (AUROC: 0.995-0.999), outperforming baselines by large margins while being orders of magnitude faster, with training under 4 seconds and inference as low as 4 $\mu$s.
>
---
#### [new 088] Leveraging Fine-Tuned Large Language Models for Interpretable Pancreatic Cystic Lesion Feature Extraction and Risk Categorization
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理与医疗文本分析任务，旨在解决人工提取胰腺囊性病变特征费时费力的问题。作者通过微调开源大语言模型，利用GPT-4o生成的思维链数据，实现自动特征提取与风险分类，准确率高且与专家水平相当。**

- **链接: [http://arxiv.org/pdf/2507.19973v1](http://arxiv.org/pdf/2507.19973v1)**

> **作者:** Ebrahim Rasromani; Stella K. Kang; Yanqi Xu; Beisong Liu; Garvit Luhadia; Wan Fung Chui; Felicia L. Pasadyn; Yu Chih Hung; Julie Y. An; Edwin Mathieu; Zehui Gu; Carlos Fernandez-Granda; Ammar A. Javed; Greg D. Sacks; Tamas Gonda; Chenchan Huang; Yiqiu Shen
>
> **摘要:** Background: Manual extraction of pancreatic cystic lesion (PCL) features from radiology reports is labor-intensive, limiting large-scale studies needed to advance PCL research. Purpose: To develop and evaluate large language models (LLMs) that automatically extract PCL features from MRI/CT reports and assign risk categories based on guidelines. Materials and Methods: We curated a training dataset of 6,000 abdominal MRI/CT reports (2005-2024) from 5,134 patients that described PCLs. Labels were generated by GPT-4o using chain-of-thought (CoT) prompting to extract PCL and main pancreatic duct features. Two open-source LLMs were fine-tuned using QLoRA on GPT-4o-generated CoT data. Features were mapped to risk categories per institutional guideline based on the 2017 ACR White Paper. Evaluation was performed on 285 held-out human-annotated reports. Model outputs for 100 cases were independently reviewed by three radiologists. Feature extraction was evaluated using exact match accuracy, risk categorization with macro-averaged F1 score, and radiologist-model agreement with Fleiss' Kappa. Results: CoT fine-tuning improved feature extraction accuracy for LLaMA (80% to 97%) and DeepSeek (79% to 98%), matching GPT-4o (97%). Risk categorization F1 scores also improved (LLaMA: 0.95; DeepSeek: 0.94), closely matching GPT-4o (0.97), with no statistically significant differences. Radiologist inter-reader agreement was high (Fleiss' Kappa = 0.888) and showed no statistically significant difference with the addition of DeepSeek-FT-CoT (Fleiss' Kappa = 0.893) or GPT-CoT (Fleiss' Kappa = 0.897), indicating that both models achieved agreement levels on par with radiologists. Conclusion: Fine-tuned open-source LLMs with CoT supervision enable accurate, interpretable, and efficient phenotyping for large-scale PCL research, achieving performance comparable to GPT-4o.
>
---
#### [new 089] Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations
- **分类: cs.RO; cs.CL; cs.IT; cs.LG; cs.SY; eess.SY; math.IT**

- **简介: 该论文属于人机协作任务中的信息融合研究，旨在解决机器人如何有效融合人类观察与传感器数据的问题。论文提出了一种基于特征金字塔的空间语言似然网络（FP-LGN），通过学习地图图像特征与空间语言的关系，建立人类输入的不确定性模型。实验表明该方法在不确定性感知融合中表现优异，提升了协作任务性能。**

- **链接: [http://arxiv.org/pdf/2507.19947v1](http://arxiv.org/pdf/2507.19947v1)**

> **作者:** Supawich Sitdhipol; Waritwong Sukprasongdee; Ekapol Chuangsuwanich; Rina Tse
>
> **备注:** Accepted to the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
>
> **摘要:** Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance.
>
---
#### [new 090] Dissecting Persona-Driven Reasoning in Language Models via Activation Patching
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究语言模型如何通过激活修补处理角色信息。论文探讨了角色设定对模型推理的影响，发现早期MLP层处理角色语义，中间MHA层利用这些信息影响输出，并识别了关注种族和颜色身份的注意力头。**

- **链接: [http://arxiv.org/pdf/2507.20936v1](http://arxiv.org/pdf/2507.20936v1)**

> **作者:** Ansh Poonia; Maeghal Jain
>
> **备注:** 11 pages
>
> **摘要:** Large language models (LLMs) exhibit remarkable versatility in adopting diverse personas. In this study, we examine how assigning a persona influences a model's reasoning on an objective task. Using activation patching, we take a first step toward understanding how key components of the model encode persona-specific information. Our findings reveal that the early Multi-Layer Perceptron (MLP) layers attend not only to the syntactic structure of the input but also process its semantic content. These layers transform persona tokens into richer representations, which are then used by the middle Multi-Head Attention (MHA) layers to shape the model's output. Additionally, we identify specific attention heads that disproportionately attend to racial and color-based identities.
>
---
#### [new 091] Does AI and Human Advice Mitigate Punishment for Selfish Behavior? An Experiment on AI ethics From a Psychological Perspective
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; econ.GN; q-fin.EC**

- **简介: 该论文研究AI与人类建议如何影响自私行为的惩罚。通过行为经济学与心理学方法，实验发现行为和建议内容影响惩罚程度，而建议来源（AI或人类）不影响。重点在于揭示AI伦理问题中的责任归因与惩罚机制。**

- **链接: [http://arxiv.org/pdf/2507.19487v1](http://arxiv.org/pdf/2507.19487v1)**

> **作者:** Margarita Leib; Nils Köbis; Ivan Soraperra
>
> **摘要:** People increasingly rely on AI-advice when making decisions. At times, such advice can promote selfish behavior. When individuals abide by selfishness-promoting AI advice, how are they perceived and punished? To study this question, we build on theories from social psychology and combine machine-behavior and behavioral economic approaches. In a pre-registered, financially-incentivized experiment, evaluators could punish real decision-makers who (i) received AI, human, or no advice. The advice (ii) encouraged selfish or prosocial behavior, and decision-makers (iii) behaved selfishly or, in a control condition, behaved prosocially. Evaluators further assigned responsibility to decision-makers and their advisors. Results revealed that (i) prosocial behavior was punished very little, whereas selfish behavior was punished much more. Focusing on selfish behavior, (ii) compared to receiving no advice, selfish behavior was penalized more harshly after prosocial advice and more leniently after selfish advice. Lastly, (iii) whereas selfish decision-makers were seen as more responsible when they followed AI compared to human advice, punishment between the two advice sources did not vary. Overall, behavior and advice content shape punishment, whereas the advice source does not.
>
---
#### [new 092] EcoTransformer: Attention without Multiplication
- **分类: cs.LG; cs.AI; cs.CL; 68T05**

- **简介: 该论文属于模型优化任务，旨在解决Transformer计算和能耗高的问题。工作提出EcoTransformer，用基于L1距离和拉普拉斯核的注意力机制替代点积，避免矩阵乘法，在保持性能的同时显著降低能耗。**

- **链接: [http://arxiv.org/pdf/2507.20096v1](http://arxiv.org/pdf/2507.20096v1)**

> **作者:** Xin Gao; Xingming Xu
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** The Transformer, with its scaled dot-product attention mechanism, has become a foundational architecture in modern AI. However, this mechanism is computationally intensive and incurs substantial energy costs. We propose a new Transformer architecture EcoTransformer, in which the output context vector is constructed as the convolution of the values using a Laplacian kernel, where the distances are measured by the L1 metric between the queries and keys. Compared to dot-product based attention, the new attention score calculation is free of matrix multiplication. It performs on par with, or even surpasses, scaled dot-product attention in NLP, bioinformatics, and vision tasks, while consuming significantly less energy.
>
---
#### [new 093] Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于安全评估任务，旨在解决AI代理在现实环境中部署时的安全漏洞问题。通过举办大规模红队竞赛，收集大量提示注入攻击，构建了ART基准测试，评估19个最先进模型的安全性，发现多数代理在少量查询内即出现策略违规，表明现有AI代理存在普遍且严重的安全漏洞。**

- **链接: [http://arxiv.org/pdf/2507.20526v1](http://arxiv.org/pdf/2507.20526v1)**

> **作者:** Andy Zou; Maxwell Lin; Eliot Jones; Micha Nowak; Mateusz Dziemian; Nick Winter; Alexander Grattan; Valent Nathanael; Ayla Croft; Xander Davies; Jai Patel; Robert Kirk; Nate Burnikell; Yarin Gal; Dan Hendrycks; J. Zico Kolter; Matt Fredrikson
>
> **摘要:** Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment.
>
---
#### [new 094] Your AI, Not Your View: The Bias of LLMs in Investment Analysis
- **分类: q-fin.PM; cs.AI; cs.CL**

- **简介: 论文研究了大型语言模型（LLMs）在投资分析中的确认偏误问题，属于金融与人工智能交叉任务。它旨在解决LLMs因预训练知识与实时市场数据冲突导致的投资建议偏差问题。论文通过构建假设场景，量化分析模型在行业、市值和动量等维度的偏好及其固化程度，揭示了模型对大盘股和逆向策略的倾向，并发现其易形成确认偏误。**

- **链接: [http://arxiv.org/pdf/2507.20957v1](http://arxiv.org/pdf/2507.20957v1)**

> **作者:** Hoyoung Lee; Junhyuk Seo; Suhwan Park; Junhyeong Lee; Wonbin Ahn; Chanyeol Choi; Alejandro Lopez-Lira; Yongjae Lee
>
> **摘要:** In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence.
>
---
#### [new 095] Agentic Reinforced Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出了一种名为Agentic Reinforced Policy Optimization (ARPO)的强化学习算法，旨在解决多轮大型语言模型（LLM）代理在复杂任务中平衡长期推理与工具交互的问题。通过熵自适应机制和优势归因估计，ARPO在13项挑战性任务中表现出色，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.19849v1](http://arxiv.org/pdf/2507.19849v1)**

> **作者:** Guanting Dong; Hangyu Mao; Kai Ma; Licheng Bao; Yifei Chen; Zhongyuan Wang; Zhongxia Chen; Jiazhen Du; Huiyang Wang; Fuzheng Zhang; Guorui Zhou; Yutao Zhu; Ji-Rong Wen; Zhicheng Dou
>
> **备注:** Working on progress
>
> **摘要:** Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments. Our code and datasets are released at https://github.com/dongguanting/ARPO
>
---
#### [new 096] $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像到LaTeX转换任务，旨在解决视觉语言模型（VLM）在精细视觉元素理解上的不足，导致LaTeX生成不准确的问题。作者提出了一种名为$A^2R^2$的新框架，结合注意力定位与迭代优化，提升模型预测质量。此外，还构建了一个用于评估的高难度数据集Img2LaTex-Hard-1K，并通过实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20890v1](http://arxiv.org/pdf/2507.20890v1)**

> **作者:** Zhecheng Li; Guoxian Song; Yiwei Wang; Zhen Xiong; Junsong Yuan; Yujun Cai
>
> **摘要:** Img2LaTeX is a practically significant task that involves converting mathematical expressions or tabular data from images into LaTeX code. In recent years, vision-language models (VLMs) have demonstrated strong performance across a variety of visual understanding tasks, owing to their generalization capabilities. While some studies have explored the use of VLMs for the Img2LaTeX task, their performance often falls short of expectations. Empirically, VLMs sometimes struggle with fine-grained visual elements, leading to inaccurate LaTeX predictions. To address this challenge, we propose $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement, a framework that effectively integrates attention localization and iterative refinement within a visual reasoning framework, enabling VLMs to perform self-correction and progressively improve prediction quality. For effective evaluation, we introduce a new dataset, Img2LaTex-Hard-1K, consisting of 1,100 carefully curated and challenging examples designed to rigorously evaluate the capabilities of VLMs within this task domain. Extensive experimental results demonstrate that: (1) $A^2R^2$ significantly improves model performance across six evaluation metrics spanning both textual and visual levels, consistently outperforming other baseline methods; (2) Increasing the number of inference rounds yields notable performance gains, underscoring the potential of $A^2R^2$ in test-time scaling scenarios; (3) Ablation studies and human evaluations validate the practical effectiveness of our approach, as well as the strong synergy among its core components during inference.
>
---
#### [new 097] Improving the Performance of Sequential Recommendation Systems with an Extended Large Language Model
- **分类: cs.IR; cs.AI; cs.CL; H.3.3; I.2.6; I.2.7**

- **简介: 该论文属于推荐系统任务，旨在解决如何提升基于大语言模型的推荐系统性能的问题。通过在LlamaRec框架中用Llama3替代Llama2，实验证明该方法在多个数据集上显著提升了推荐效果，且无需修改系统结构。**

- **链接: [http://arxiv.org/pdf/2507.19990v1](http://arxiv.org/pdf/2507.19990v1)**

> **作者:** Sinnyum Choi; Woong Kim
>
> **摘要:** Recently, competition in the field of artificial intelligence (AI) has intensified among major technological companies, resulting in the continuous release of new large-language models (LLMs) that exhibit improved language understanding and context-based reasoning capabilities. It is expected that these advances will enable more efficient personalized recommendations in LLM-based recommendation systems through improved quality of training data and architectural design. However, many studies have not considered these recent developments. In this study, it was proposed to improve LLM-based recommendation systems by replacing Llama2 with Llama3 in the LlamaRec framework. To ensure a fair comparison, random seed values were set and identical input data was provided during preprocessing and training. The experimental results show average performance improvements of 38.65\%, 8.69\%, and 8.19\% for the ML-100K, Beauty, and Games datasets, respectively, thus confirming the practicality of this method. Notably, the significant improvements achieved by model replacement indicate that the recommendation quality can be improved cost-effectively without the need to make structural changes to the system. Based on these results, it is our contention that the proposed approach is a viable solution for improving the performance of current recommendation systems.
>
---
#### [new 098] Customize Multi-modal RAI Guardrails with Precedent-based predictions
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于多模态内容审核任务，旨在根据用户自定义策略过滤有害图像内容。现有方法难以适应多样化的策略或需大量样本。论文提出基于先例（precedent）的预测方法，通过收集高质量先例并设计优化机制，提升模型灵活性与泛化能力，有效应对新策略和少量样本场景。**

- **链接: [http://arxiv.org/pdf/2507.20503v1](http://arxiv.org/pdf/2507.20503v1)**

> **作者:** Cheng-Fu Yang; Thanh Tran; Christos Christodoulopoulos; Weitong Ruan; Rahul Gupta; Kai-Wei Chang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** A multi-modal guardrail must effectively filter image content based on user-defined policies, identifying material that may be hateful, reinforce harmful stereotypes, contain explicit material, or spread misinformation. Deploying such guardrails in real-world applications, however, poses significant challenges. Users often require varied and highly customizable policies and typically cannot provide abundant examples for each custom policy. Consequently, an ideal guardrail should be scalable to the multiple policies and adaptable to evolving user standards with minimal retraining. Existing fine-tuning methods typically condition predictions on pre-defined policies, restricting their generalizability to new policies or necessitating extensive retraining to adapt. Conversely, training-free methods struggle with limited context lengths, making it difficult to incorporate all the policies comprehensively. To overcome these limitations, we propose to condition model's judgment on "precedents", which are the reasoning processes of prior data points similar to the given input. By leveraging precedents instead of fixed policies, our approach greatly enhances the flexibility and adaptability of the guardrail. In this paper, we introduce a critique-revise mechanism for collecting high-quality precedents and two strategies that utilize precedents for robust prediction. Experimental results demonstrate that our approach outperforms previous methods across both few-shot and full-dataset scenarios and exhibits superior generalization to novel policies.
>
---
#### [new 099] The Importance of Facial Features in Vision-based Sign Language Recognition: Eyes, Mouth or Full Face?
- **分类: cs.CV; cs.CL; eess.IV**

- **简介: 该论文属于视觉手语识别任务，旨在解决非手动面部特征在识别中的作用问题。作者通过对比眼睛、嘴巴和全脸的贡献，结合CNN和Transformer模型，发现嘴巴特征最关键，显著提升识别准确率，强调了面部特征对自动手语识别的重要性。**

- **链接: [http://arxiv.org/pdf/2507.20884v1](http://arxiv.org/pdf/2507.20884v1)**

> **作者:** Dinh Nam Pham; Eleftherios Avramidis
>
> **备注:** Accepted at 9th International Workshop on Sign Language Translation and Avatar Technologies @ ACM IVA'25
>
> **摘要:** Non-manual facial features play a crucial role in sign language communication, yet their importance in automatic sign language recognition (ASLR) remains underexplored. While prior studies have shown that incorporating facial features can improve recognition, related work often relies on hand-crafted feature extraction and fails to go beyond the comparison of manual features versus the combination of manual and facial features. In this work, we systematically investigate the contribution of distinct facial regionseyes, mouth, and full faceusing two different deep learning models (a CNN-based model and a transformer-based model) trained on an SLR dataset of isolated signs with randomly selected classes. Through quantitative performance and qualitative saliency map evaluation, we reveal that the mouth is the most important non-manual facial feature, significantly improving accuracy. Our findings highlight the necessity of incorporating facial features in ASLR.
>
---
#### [new 100] The Impact of Fine-tuning Large Language Models on Automated Program Repair
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自动化程序修复任务，旨在提升大语言模型在此任务中的表现。论文研究了不同微调技术对模型性能的影响，比较了全量微调与参数高效微调方法（如LoRA和IA3），发现后者在减少训练参数的同时提升了修复效果。**

- **链接: [http://arxiv.org/pdf/2507.19909v1](http://arxiv.org/pdf/2507.19909v1)**

> **作者:** Roman Macháček; Anastasiia Grishina; Max Hort; Leon Moonen
>
> **备注:** Accepted for publication in the research track of the 41th International Conference on Software Maintenance and Evolution (ICSME 2025)
>
> **摘要:** Automated Program Repair (APR) uses various tools and techniques to help developers achieve functional and error-free code faster. In recent years, Large Language Models (LLMs) have gained popularity as components in APR tool chains because of their performance and flexibility. However, training such models requires a significant amount of resources. Fine-tuning techniques have been developed to adapt pre-trained LLMs to specific tasks, such as APR, and enhance their performance at far lower computational costs than training from scratch. In this study, we empirically investigate the impact of various fine-tuning techniques on the performance of LLMs used for APR. Our experiments provide insights into the performance of a selection of state-of-the-art LLMs pre-trained on code. The evaluation is done on three popular APR benchmarks (i.e., QuixBugs, Defects4J and HumanEval-Java) and considers six different LLMs with varying parameter sizes (resp. CodeGen, CodeT5, StarCoder, DeepSeekCoder, Bloom, and CodeLlama-2). We consider three training regimens: no fine-tuning, full fine-tuning, and parameter-efficient fine-tuning (PEFT) using LoRA and IA3. We observe that full fine-tuning techniques decrease the benchmarking performance of various models due to different data distributions and overfitting. By using parameter-efficient fine-tuning methods, we restrict models in the amount of trainable parameters and achieve better results. Keywords: large language models, automated program repair, parameter-efficient fine-tuning, AI4Code, AI4SE, ML4SE.
>
---
#### [new 101] AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出AutoSign，用于连续手语识别任务，旨在将姿态序列直接翻译为文本。传统方法依赖多阶段流程，存在误差传播和扩展性差等问题。AutoSign采用解码器-only的Transformer结构，结合时间压缩模块和预训练模型，实现端到端翻译，提升了识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.19840v1](http://arxiv.org/pdf/2507.19840v1)**

> **作者:** Samuel Ebimobowei Johnny; Blessed Guda; Andrew Blayama Stephen; Assane Gueye
>
> **备注:** Paper to appear at the 1st Workshop in Multimodal Sign Language Recognition at ICCV 2025
>
> **摘要:** Continuously recognizing sign gestures and converting them to glosses plays a key role in bridging the gap between the hearing and hearing-impaired communities. This involves recognizing and interpreting the hands, face, and body gestures of the signer, which pose a challenge as it involves a combination of all these features. Continuous Sign Language Recognition (CSLR) methods rely on multi-stage pipelines that first extract visual features, then align variable-length sequences with target glosses using CTC or HMM-based approaches. However, these alignment-based methods suffer from error propagation across stages, overfitting, and struggle with vocabulary scalability due to the intermediate gloss representation bottleneck. To address these limitations, we propose AutoSign, an autoregressive decoder-only transformer that directly translates pose sequences to natural language text, bypassing traditional alignment mechanisms entirely. The use of this decoder-only approach allows the model to directly map between the features and the glosses without the need for CTC loss while also directly learning the textual dependencies in the glosses. Our approach incorporates a temporal compression module using 1D CNNs to efficiently process pose sequences, followed by AraGPT2, a pre-trained Arabic decoder, to generate text (glosses). Through comprehensive ablation studies, we demonstrate that hand and body gestures provide the most discriminative features for signer-independent CSLR. By eliminating the multi-stage pipeline, AutoSign achieves substantial improvements on the Isharah-1000 dataset, achieving an improvement of up to 6.1\% in WER score compared to the best existing method.
>
---
#### [new 102] PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出PITA框架，用于大语言模型（LLM）推理阶段的偏好对齐。任务是生成符合用户偏好的输出，无需额外训练。传统方法依赖预训练奖励模型，而PITA直接整合用户反馈，通过小型引导策略调整推理时的词元概率，降低计算成本并避免依赖奖励模型。方法基于随机搜索与迭代优化，适用于数学推理和情感分类等任务。**

- **链接: [http://arxiv.org/pdf/2507.20067v1](http://arxiv.org/pdf/2507.20067v1)**

> **作者:** Sarat Chandra Bobbili; Ujwal Dinesha; Dheeraj Narasimha; Srinivas Shakkottai
>
> **摘要:** Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences.
>
---
#### [new 103] Kimi K2: Open Agentic Intelligence
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出Kimi K2，一种具备320亿激活参数的MoE大模型，采用MuonClip优化器解决训练不稳定问题，并通过多阶段后训练提升代理能力。在多项基准测试中表现优异，适用于编码、数学及推理任务，推动开源代理智能研究。**

- **链接: [http://arxiv.org/pdf/2507.20534v1](http://arxiv.org/pdf/2507.20534v1)**

> **作者:** Kimi Team; Yifan Bai; Yiping Bao; Guanduo Chen; Jiahao Chen; Ningxin Chen; Ruijue Chen; Yanru Chen; Yuankun Chen; Yutian Chen; Zhuofu Chen; Jialei Cui; Hao Ding; Mengnan Dong; Angang Du; Chenzhuang Du; Dikang Du; Yulun Du; Yu Fan; Yichen Feng; Kelin Fu; Bofei Gao; Hongcheng Gao; Peizhong Gao; Tong Gao; Xinran Gu; Longyu Guan; Haiqing Guo; Jianhang Guo; Hao Hu; Xiaoru Hao; Tianhong He; Weiran He; Wenyang He; Chao Hong; Yangyang Hu; Zhenxing Hu; Weixiao Huang; Zhiqi Huang; Zihao Huang; Tao Jiang; Zhejun Jiang; Xinyi Jin; Yongsheng Kang; Guokun Lai; Cheng Li; Fang Li; Haoyang Li; Ming Li; Wentao Li; Yanhao Li; Yiwei Li; Zhaowei Li; Zheming Li; Hongzhan Lin; Xiaohan Lin; Zongyu Lin; Chengyin Liu; Chenyu Liu; Hongzhang Liu; Jingyuan Liu; Junqi Liu; Liang Liu; Shaowei Liu; T. Y. Liu; Tianwei Liu; Weizhou Liu; Yangyang Liu; Yibo Liu; Yiping Liu; Yue Liu; Zhengying Liu; Enzhe Lu; Lijun Lu; Shengling Ma; Xinyu Ma; Yingwei Ma; Shaoguang Mao; Jie Mei; Xin Men; Yibo Miao; Siyuan Pan; Yebo Peng; Ruoyu Qin; Bowen Qu; Zeyu Shang; Lidong Shi; Shengyuan Shi; Feifan Song; Jianlin Su; Zhengyuan Su; Xinjie Sun; Flood Sung; Heyi Tang; Jiawen Tao; Qifeng Teng; Chensi Wang; Dinglu Wang; Feng Wang; Haiming Wang; Jianzhou Wang; Jiaxing Wang; Jinhong Wang; Shengjie Wang; Shuyi Wang; Yao Wang; Yejie Wang; Yiqin Wang; Yuxin Wang; Yuzhi Wang; Zhaoji Wang; Zhengtao Wang; Zhexu Wang; Chu Wei; Qianqian Wei; Wenhao Wu; Xingzhe Wu; Yuxin Wu; Chenjun Xiao; Xiaotong Xie; Weimin Xiong; Boyu Xu; Jing Xu; Jinjing Xu; L. H. Xu; Lin Xu; Suting Xu; Weixin Xu; Xinran Xu; Yangchuan Xu; Ziyao Xu; Junjie Yan; Yuzi Yan; Xiaofei Yang; Ying Yang; Zhen Yang; Zhilin Yang; Zonghan Yang; Haotian Yao; Xingcheng Yao; Wenjie Ye; Zhuorui Ye; Bohong Yin; Longhui Yu; Enming Yuan; Hongbang Yuan; Mengjie Yuan; Haobing Zhan; Dehao Zhang; Hao Zhang; Wanlu Zhang; Xiaobin Zhang; Yangkun Zhang; Yizhi Zhang; Yongting Zhang; Yu Zhang; Yutao Zhang; Yutong Zhang; Zheng Zhang; Haotian Zhao; Yikai Zhao; Huabin Zheng; Shaojie Zheng; Jianren Zhou; Xinyu Zhou; Zaida Zhou; Zhen Zhu; Weiyu Zhuang; Xinxing Zu
>
> **备注:** tech report of Kimi K2
>
> **摘要:** We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments. Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence.
>
---
#### [new 104] FedDPG: An Adaptive Yet Efficient Prompt-tuning Approach in Federated Learning Settings
- **分类: cs.LG; cs.AI; cs.CL; I.2; I.7**

- **简介: 论文提出FedDPG，一种结合动态提示生成的联邦学习方法，用于提升预训练语言模型在隐私保护场景下的效率与性能。任务是解决联邦学习中通信计算受限及提示调优灵活性不足的问题。工作包括设计动态提示生成网络，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.19534v1](http://arxiv.org/pdf/2507.19534v1)**

> **作者:** Ali Shakeri; Wei Emma Zhang; Amin Beheshti; Weitong Chen; Jian Yang; Lishan Yang
>
> **备注:** 12 pages; Published to PAKDD'2025
>
> **摘要:** Pre-trained Language Models (PLMs) have demonstrated impressive performance in various NLP tasks. However, traditional fine-tuning methods for leveraging PLMs for downstream tasks entail significant computational overhead. Prompt-tuning has emerged as an efficient alternative that involves prepending a limited number of parameters to the input sequence and only updating them while the PLM's parameters are frozen. However, this technique's prompts remain fixed for all inputs, reducing the model's flexibility. The Federated Learning (FL) technique has gained attention in recent years to address the growing concerns around data privacy. However, challenges such as communication and computation limitations of clients still need to be addressed. To mitigate these challenges, this paper introduces the Federated Dynamic Prompt Generator (FedDPG), which incorporates a dynamic prompt generator network to generate context-aware prompts based on the given input, ensuring flexibility and adaptability while prioritising data privacy in federated learning settings. Our experiments on three NLP benchmark datasets showcase that FedDPG outperforms the state-of-the-art parameter-efficient fine-tuning methods in terms of global model performance, and has significantly reduced the calculation time and the number of parameters to be sent through the FL network.
>
---
#### [new 105] The Devil is in the EOS: Sequence Training for Detailed Image Captioning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像描述生成任务，旨在解决当前模型生成描述缺乏细节的问题。论文发现交叉熵训练导致模型过早结束生成（EOS偏差），提出一种无监督方法缓解该问题，使生成更长、更详细的描述。实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.20077v1](http://arxiv.org/pdf/2507.20077v1)**

> **作者:** Abdelrahman Mohamed; Yova Kementchedjhieva
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Despite significant advances in vision-language models (VLMs), image captioning often suffers from a lack of detail, with base models producing short, generic captions. This limitation persists even though VLMs are equipped with strong vision and language backbones. While supervised data and complex reward functions have been proposed to improve detailed image captioning, we identify a simpler underlying issue: a bias towards the end-of-sequence (EOS) token, which is introduced during cross-entropy training. We propose an unsupervised method to debias the model's tendency to predict the EOS token prematurely. By reducing this bias, we encourage the generation of longer, more detailed captions without the need for intricate reward functions or supervision. Our approach is straightforward, effective, and easily applicable to any pretrained model. We demonstrate its effectiveness through experiments with three VLMs and on three detailed captioning benchmarks. Our results show a substantial increase in caption length and relevant details, albeit with an expected increase in the rate of hallucinations.
>
---
#### [new 106] The Carbon Cost of Conversation, Sustainability in the Age of Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 这篇论文属于环境影响评估与可持续发展任务，旨在解决大型语言模型（LLMs）带来的碳排放、水资源消耗和电子废弃物问题。论文通过案例研究量化其环境成本，分析问题成因，并提出技术、政策和文化层面的可持续路径。**

- **链接: [http://arxiv.org/pdf/2507.20018v1](http://arxiv.org/pdf/2507.20018v1)**

> **作者:** Sayed Mahbub Hasan Amiri; Prasun Goswami; Md. Mainul Islam; Mohammad Shakhawat Hossen; Sayed Majhab Hasan Amiri; Naznin Akter
>
> **备注:** 22 Pages, 5 Tables
>
> **摘要:** Large language models (LLMs) like GPT-3 and BERT have revolutionized natural language processing (NLP), yet their environmental costs remain dangerously overlooked. This article critiques the sustainability of LLMs, quantifying their carbon footprint, water usage, and contribution to e-waste through case studies of models such as GPT-4 and energy-efficient alternatives like Mistral 7B. Training a single LLM can emit carbon dioxide equivalent to hundreds of cars driven annually, while data centre cooling exacerbates water scarcity in vulnerable regions. Systemic challenges corporate greenwashing, redundant model development, and regulatory voids perpetuate harm, disproportionately burdening marginalized communities in the Global South. However, pathways exist for sustainable NLP: technical innovations (e.g., model pruning, quantum computing), policy reforms (carbon taxes, mandatory emissions reporting), and cultural shifts prioritizing necessity over novelty. By analysing industry leaders (Google, Microsoft) and laggards (Amazon), this work underscores the urgency of ethical accountability and global cooperation. Without immediate action, AIs ecological toll risks outpacing its societal benefits. The article concludes with a call to align technological progress with planetary boundaries, advocating for equitable, transparent, and regenerative AI systems that prioritize both human and environmental well-being.
>
---
#### [new 107] Enhancing Project-Specific Code Completion by Inferring Internal API Information
- **分类: cs.SE; cs.CL**

- **简介: 论文属于代码补全任务，旨在解决项目内API信息难以融入补全结果的问题。作者提出通过构建API使用示例与语义描述，增强LLM生成能力，并构建新基准ProjBench验证方法有效性，显著提升了代码与标识符匹配准确率。**

- **链接: [http://arxiv.org/pdf/2507.20888v1](http://arxiv.org/pdf/2507.20888v1)**

> **作者:** Le Deng; Xiaoxue Ren; Chao Ni; Ming Liang; David Lo; Zhongxin Liu
>
> **摘要:** Project-specific code completion is a critical task that leverages context from a project to generate accurate code. State-of-the-art methods use retrieval-augmented generation (RAG) with large language models (LLMs) and project information for code completion. However, they often struggle to incorporate internal API information, which is crucial for accuracy, especially when APIs are not explicitly imported in the file. To address this, we propose a method to infer internal API information without relying on imports. Our method extends the representation of APIs by constructing usage examples and semantic descriptions, building a knowledge base for LLMs to generate relevant completions. We also introduce ProjBench, a benchmark that avoids leaked imports and consists of large-scale real-world projects. Experiments on ProjBench and CrossCodeEval show that our approach significantly outperforms existing methods, improving code exact match by 22.72% and identifier exact match by 18.31%. Additionally, integrating our method with existing baselines boosts code match by 47.80% and identifier match by 35.55%.
>
---
## 更新

#### [replaced 001] From Answers to Rationales: Self-Aligning Multimodal Reasoning with Answer-Oriented Chain-of-Thought
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02984v2](http://arxiv.org/pdf/2507.02984v2)**

> **作者:** Wentao Tan; Qiong Cao; Yibing Zhan; Chao Xue; Changxing Ding
>
> **摘要:** Achieving human-like reasoning capabilities in Multimodal Large Language Models (MLLMs) has long been a goal. Current methods primarily focus on synthesizing positive rationales, typically relying on manual annotations or complex systems. Moreover, they often overlook negative reasoning, which limits the model's generalization ability and robustness in multimodal inference. To address this gap, we propose a novel framework: \textbf{S}elf-Aligning \textbf{M}ultimodal Reasoning with \textbf{A}nswer-O\textbf{r}iented Chain-of-\textbf{T}hought (SMART). SMART employs an answer-oriented chain-of-thought (AoT) prompt to automatically construct high-quality data. Drawing inspiration from human proof-based strategies, AoT leverages both correct and incorrect answers to extract key visual information that links questions and answers. When provided with correct answers, the model produces strong positive rationales. Conversely, when correct answers are replaced with incorrect alternatives, the model generates an erroneous yet compelling reasoning path, serving as a form of discriminative negative rationale. Models trained with AoT-generated data outperform those trained on manually annotated datasets, demonstrating superior reasoning capabilities. Consequently, SMART establishes an iterative generation-optimization method that continually enhances the model's reasoning skills. Experiments indicate that the SMART framework significantly improves various MLLMs, regardless of model architecture, parameter size, or pre-training dataset. The code is available at https://github.com/WentaoTan/SMART.
>
---
#### [replaced 002] When Does Metadata Conditioning (NOT) Work for Language Model Pre-Training? A Study with Context-Free Grammars
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.17562v2](http://arxiv.org/pdf/2504.17562v2)**

> **作者:** Rei Higuchi; Ryotaro Kawata; Naoki Nishikawa; Kazusato Oko; Shoichiro Yamaguchi; Sosuke Kobayashi; Seiya Tokui; Kohei Hayashi; Daisuke Okanohara; Taiji Suzuki
>
> **摘要:** The ability to acquire latent semantics is one of the key properties that determines the performance of language models. One convenient approach to invoke this ability is to prepend metadata (e.g. URLs, domains, and styles) at the beginning of texts in the pre-training data, making it easier for the model to access latent semantics before observing the entire text. Previous studies have reported that this technique actually improves the performance of trained models in downstream tasks; however, this improvement has been observed only in specific downstream tasks, without consistent enhancement in average next-token prediction loss. To understand this phenomenon, we closely investigate how prepending metadata during pre-training affects model performance by examining its behavior using artificial data. Interestingly, we found that this approach produces both positive and negative effects on the downstream tasks. We demonstrate that the effectiveness of the approach depends on whether latent semantics can be inferred from the downstream task's prompt. Specifically, through investigations using data generated by probabilistic context-free grammars, we show that training with metadata helps improve model's performance when the given context is long enough to infer the latent semantics. In contrast, the technique negatively impacts performance when the context lacks the necessary information to make an accurate posterior inference.
>
---
#### [replaced 003] A Practice of Post-Training on Llama-3 70B with Optimal Selection of Additional Language Mixture Ratio
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.06624v2](http://arxiv.org/pdf/2409.06624v2)**

> **作者:** Ningyuan Xi; Yetao Wu; Kun Fan; Teng Chen; Qingqing Gu; Luo Ji
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Large Language Models (LLM) often need to be Continual Pre-Trained (CPT) to obtain unfamiliar language skills or adapt to new domains. The huge training cost of CPT often asks for cautious choice of key hyper-parameters such as the mixture ratio of extra language or domain corpus. However, there is no systematic study that bridges the gap between the optimal mixture ratio and the actual model performance, and the gap between experimental scaling law and the actual deployment in the full model size. In this paper, we perform CPT on Llama-3 8B and 70B to enhance its Chinese ability. We study the optimal correlation between the Additional Language Mixture Ratio (ALMR) and the Learning Rate (LR) on the 8B size which directly indicates the optimal experimental setup. By thorough choice of hyper-parameter, and subsequent fine-tuning, the model capability is improved not only on the Chinese-related benchmark but also in some specific domains including math, coding, and emotional intelligence. We deploy the final 70B version of LLM on a real-life chat system which obtains satisfying performance.
>
---
#### [replaced 004] Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.02456v2](http://arxiv.org/pdf/2505.02456v2)**

> **作者:** Elisa Forcada Rodríguez; Olatz Perez-de-Viñaspre; Jon Ander Campos; Dietrich Klakow; Vagrant Gautam
>
> **备注:** Workshop on Gender Bias in Natural Language Processing at ACL 2025
>
> **摘要:** One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work.
>
---
#### [replaced 005] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.06645v3](http://arxiv.org/pdf/2501.06645v3)**

> **作者:** Tong Liu; Xiao Yu; Wenxuan Zhou; Jindong Gu; Volker Tresp
>
> **备注:** ACL 2025
>
> **摘要:** Efficient preference optimization algorithms such as Direct Preference Optimization (DPO) have become a popular approach in aligning large language models (LLMs) with human preferences. These algorithms implicitly treat the LLM as a reward model, and focus on training it to correct misranked preference pairs. However, recent work~\citep{chen2024preference} empirically finds that DPO training \textit{rarely improves these misranked preference pairs}, despite its gradient emphasizing on these cases. We introduce FocalPO, a DPO variant that instead \textit{down-weighs} misranked preference pairs and prioritizes enhancing the model's understanding of pairs that it can already rank correctly. Inspired by Focal Loss used in vision tasks, FocalPO achieves this by adding a modulating factor to dynamically scale DPO loss. Our experiment demonstrates that FocalPO surpasses DPO and its variants on popular benchmarks like Alpaca Eval 2.0 using Mistral-Base-7B and Llama-3-Instruct-8B, with the introduced hyperparameter fixed. Additionally, we empirically reveals how FocalPO affects training on correct and incorrect sample groups, further underscoring its effectiveness.
>
---
#### [replaced 006] Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16802v4](http://arxiv.org/pdf/2507.16802v4)**

> **作者:** Yanjun Zheng; Xiyang Du; Longfei Liao; Xiaoke Zhao; Zhaowen Zhou; Jingze Song; Bo Zhang; Jiawei Liu; Xiang Qi; Zhe Li; Zhiqiang Zhang; Wei Wang; Peng Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit considerable promise in financial applications; however, prevailing models frequently demonstrate limitations when confronted with scenarios that necessitate sophisticated reasoning capabilities, stringent trustworthiness criteria, and efficient adaptation to domain-specific requirements. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task label system with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage training pipeline, and dynamic attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including Fineva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA-diamond. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. The Finova bench is available at https://github.com/antgroup/Finova.
>
---
#### [replaced 007] Measuring Information Distortion in Hierarchical Ultra long Novel Reconstruction:The Optimal Expansion Ratio
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.12572v2](http://arxiv.org/pdf/2505.12572v2)**

> **作者:** Hanwen Shen; Ting Ying
>
> **摘要:** A two stage novel generation framework (outline -> section outline -> manuscript) is widely used in long novel generation,(e.g., \textsc{DOME}, \textsc{Plan\&Write}, \textsc{Long Writer}), but study of such framework in ultra long novel(>1M words) reconstruction is little. Building on recent text compression methods (\textsc{LLMZip}, \textsc{LLM2Vec}), we conduct an information-theoretic analysis to quantify semantic distortion under different compression-expansion ratios. We examine how outline length affects information preservation. Experiments on ultra-long novels show that the optimal compression-expansion ratio significantly reduces semantic distortion compared to other non-optimal compression-expansion ratio.
>
---
#### [replaced 008] Scaling Analysis of Interleaved Speech-Text Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.02398v2](http://arxiv.org/pdf/2504.02398v2)**

> **作者:** Gallil Maimon; Michael Hassid; Amit Roth; Yossi Adi
>
> **备注:** Accepted at COLM 2025
>
> **摘要:** Existing Speech Language Model (SLM) scaling analysis paints a bleak picture. It predicts that SLMs require much more compute and data compared to text, leading some to question the feasibility of training high-quality SLMs. However, modern SLMs are often initialised from pre-trained TextLMs using speech-text interleaving to allow knowledge transfer. This raises the question - "Do interleaved SLMs scale more efficiently than textless-SLMs?" In this paper we answer a resounding yes! We conduct scaling analysis of interleaved SLMs by training several dozen and analysing the scaling trends. We see that under this setup SLMs scale more efficiently with compute. Additionally, our results indicate that the scaling dynamics significantly differ from textless-SLMs, suggesting one should allocate notably more of the compute budget to increasing model size over training tokens. We also study the role of synthetic data and TextLM model families in unlocking this potential. Results suggest that our scaled up model achieves comparable semantic speech performance to leading models, while using less compute and data. We open source models, samples, and data - https://pages.cs.huji.ac.il/adiyoss-lab/sims/ .
>
---
#### [replaced 009] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18576v2](http://arxiv.org/pdf/2507.18576v2)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names; v2 modifies minor issues
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [replaced 010] ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21613v2](http://arxiv.org/pdf/2506.21613v2)**

> **作者:** Gautam Siddharth Kashyap; Mohammad Anas Azeez; Rafiq Ali; Zohaib Hasan Siddiqui; Jiechao Gao; Usman Naseem
>
> **备注:** Updated Version
>
> **摘要:** Hate speech targeting children on social media is a serious and growing problem, yet current NLP systems struggle to detect it effectively. This gap exists mainly because existing datasets focus on adults, lack age specific labels, miss nuanced linguistic cues, and are often too small for robust modeling. To address this, we introduce ChildGuard, the first large scale English dataset dedicated to hate speech aimed at children. It contains 351,877 annotated examples from X (formerly Twitter), Reddit, and YouTube, labeled by three age groups: younger children (under 11), pre teens (11--12), and teens (13--17). The dataset is split into two subsets for fine grained analysis: a contextual subset (157K) focusing on discourse level features, and a lexical subset (194K) emphasizing word-level sentiment and vocabulary. Benchmarking state of the art hate speech models on ChildGuard reveals notable drops in performance, highlighting the challenges of detecting child directed hate speech.
>
---
#### [replaced 011] Minimal Pair-Based Evaluation of Code-Switching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01840v2](http://arxiv.org/pdf/2506.01840v2)**

> **作者:** Igor Sterner; Simone Teufel
>
> **备注:** ACL 2025
>
> **摘要:** There is a lack of an evaluation methodology that estimates the extent to which large language models (LLMs) use code-switching (CS) in the same way as bilinguals. Existing methods do not have wide language coverage, fail to account for the diverse range of CS phenomena, or do not scale. We propose an intervention based on minimal pairs of CS. Each minimal pair contains one naturally occurring CS sentence and one minimally manipulated variant. We collect up to 1,000 such pairs each for 11 language pairs. Our human experiments show that, for every language pair, bilinguals consistently prefer the naturally occurring CS sentence. Meanwhile our experiments with current LLMs show that the larger the model, the more consistently it assigns higher probability to the naturally occurring CS sentence than to the variant. In accordance with theoretical claims, the largest probability differences arise in those pairs where the manipulated material consisted of closed-class words.
>
---
#### [replaced 012] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.05167v2](http://arxiv.org/pdf/2412.05167v2)**

> **作者:** Kuofeng Gao; Shu-Tao Xia; Ke Xu; Philip Torr; Jindong Gu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Audio-Language Models (LALMs), such as GPT-4o, have recently unlocked audio dialogue capabilities, enabling direct spoken exchanges with humans. The potential of LALMs broadens their applicability across a wide range of practical scenarios supported by audio dialogues. However, given these advancements, a comprehensive benchmark to evaluate the performance of LALMs in the open-ended audio dialogue understanding remains absent currently. To address this gap, we propose an Audio Dialogue Understanding Benchmark (ADU-Bench), which consists of 4 benchmark datasets. They assess the open-ended audio dialogue ability for LALMs in 3 general scenarios, 12 skills, 9 multilingual languages, and 4 categories of ambiguity handling. Notably, we firstly propose the evaluation of ambiguity handling in audio dialogues that expresses different intentions beyond the same literal meaning of sentences, e.g., "Really!?" with different intonations. In summary, ADU-Bench includes over 20,000 open-ended audio dialogues for the assessment of LALMs. Through extensive experiments on 16 LALMs, our analysis reveals that existing LALMs struggle with mathematical symbols and formulas, understanding human behavior such as roleplay, comprehending multiple languages, and handling audio dialogue ambiguities from different phonetic elements, such as intonations, pause positions, and homophones. The benchmark is available at https://adu-bench.github.io/.
>
---
#### [replaced 013] Group Sequence Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18071v2](http://arxiv.org/pdf/2507.18071v2)**

> **作者:** Chujie Zheng; Shixuan Liu; Mingze Li; Xiong-Hui Chen; Bowen Yu; Chang Gao; Kai Dang; Yuqiong Liu; Rui Men; An Yang; Jingren Zhou; Junyang Lin
>
> **摘要:** This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models.
>
---
#### [replaced 014] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14351v2](http://arxiv.org/pdf/2505.14351v2)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** 15 pages
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-\"U-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [replaced 015] Scaling Physical Reasoning with the PHYSICS Dataset
- **分类: cs.CL; cs.LG; physics.ed-ph**

- **链接: [http://arxiv.org/pdf/2506.00022v3](http://arxiv.org/pdf/2506.00022v3)**

> **作者:** Shenghe Zheng; Qianjia Cheng; Junchi Yao; Mengsong Wu; Haonan He; Ning Ding; Yu Cheng; Shuyue Hu; Lei Bai; Dongzhan Zhou; Ganqu Cui; Peng Ye
>
> **备注:** Work on physical datasets
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress on advanced reasoning tasks such as mathematics and coding competitions. Meanwhile, physics, despite being both reasoning-intensive and essential to real-world understanding, received limited academic and industrial attention. This paper introduces PHYSICS, a dataset containing 16,568 high-quality physics problems spanning subjects and difficulty levels, to facilitate this issue. Specifically, PHYSICS is curated with exercises from over 100 textbooks through a carefully designed pipeline for quality control. It covers five major physics domains: Mechanics, Electromagnetism, Thermodynamics, Optics, and Modern Physics. It also spans a wide range of difficulty levels, from high school to graduate-level physics courses. To utilize the data for improving and evaluating the model's physical reasoning capabilities, we split the dataset into training and test sets, and provide reasoning paths generated by powerful reasoning models for the training data to facilitate model training. In addition, for the evaluation part, we find that existing evaluation frameworks exhibit biases in aspects such as units, simplification, and precision in physics domain. To balance efficiency and accuracy, we introduce a Rule+Model evaluation framework tailored to physics problems. Our evaluations on current state-of-the-art open-source and proprietary models highlight the limitations of current models in handling physics-related tasks. We hope that our dataset and evaluation methodology will jointly advance the development of LLMs in the field of physics.
>
---
#### [replaced 016] AI as a deliberative partner fosters intercultural empathy for Americans but fails for Latin American participants
- **分类: cs.HC; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.13887v2](http://arxiv.org/pdf/2504.13887v2)**

> **作者:** Isabel Villanueva; Tara Bobinac; Binwei Yao; Junjie Hu; Kaiping Chen
>
> **摘要:** Despite increasing AI chatbot deployment in public discourse, empirical evidence on their capacity to foster intercultural empathy remains limited. Through a randomized experiment, we assessed how different AI deliberation approaches--cross-cultural deliberation (presenting other-culture perspectives), own-culture deliberation (representing participants' own culture), and non-deliberative control--affect intercultural empathy across American and Latin American participants. Cross-cultural deliberation increased intercultural empathy among American participants through positive emotional engagement, but produced no such effects for Latin American participants, who perceived AI responses as culturally inauthentic despite explicit prompting to represent their cultural perspectives. Our analysis of participant-driven feedback, where users directly flagged and explained culturally inappropriate AI responses, revealed systematic gaps in AI's representation of Latin American contexts that persist despite sophisticated prompt engineering. These findings demonstrate that current approaches to AI cultural alignment--including linguistic adaptation and explicit cultural prompting--cannot fully address deeper representational asymmetries in AI systems. Our work advances both deliberation theory and AI alignment research by revealing how the same AI system can simultaneously promote intercultural understanding for one cultural group while failing for another, with critical implications for designing equitable AI systems for cross-cultural democratic discourse.
>
---
#### [replaced 017] Computational Analysis of Character Development in Holocaust Testimonies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17063v4](http://arxiv.org/pdf/2412.17063v4)**

> **作者:** Esther Shizgal; Eitan Wagner; Renana Keydar; Omri Abend
>
> **摘要:** This work presents a computational approach to analyze character development along the narrative timeline. The analysis characterizes the inner and outer changes the protagonist undergoes within a narrative, and the interplay between them. We consider transcripts of Holocaust survivor testimonies as a test case, each telling the story of an individual in first-person terms. We focus on the survivor's religious trajectory, examining the evolution of their disposition toward religious belief and practice along the testimony. Clustering the resulting trajectories in the dataset, we identify common sequences in the data. Our findings highlight multiple common structures of religiosity across the narratives: in terms of belief, most present a constant disposition, while for practice, most present an oscillating structure, serving as valuable material for historical and sociological research. This work demonstrates the potential of natural language processing techniques for analyzing character evolution through thematic trajectories in narratives.
>
---
#### [replaced 018] Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to GQA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20677v2](http://arxiv.org/pdf/2412.20677v2)**

> **作者:** Qingyun Jin; Xiaohui Song; Feng Zhou; Zengchang Qin
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, as the model size and the input sequence's length increase, the linearly increasing key-value (KV) cache significantly degrades inference throughput. Therefore, grouped-query attention (GQA), as an alternative to multi-head attention (MHA), has been widely introduced into LLMs. In this work, we propose a cost-effective method for converting MHA into GQA with any compression ratio of KV heads. The key point of our method lies in the application of Procrustes analysis to the attention heads, which enhances the similarity among attention heads while preserving computational invariance, thereby improving the model's post-training performance. Subsequently, we employ $\mathit{L_0}$ regularization to prune redundant parameters. The model after pruning can be adapted to the standard GQA framework. Experimental results show that our strategy can compress up to 87.5\% KV heads of LLaMA2-7B model and 75\% KV heads of Sheared-LLaMA-1.3B with acceptable performance degradation. Our code is released at https://github.com/fpcsong/mha2gqa.
>
---
#### [replaced 019] Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16789v2](http://arxiv.org/pdf/2505.16789v2)**

> **作者:** Punya Syon Pandey; Samuel Simko; Kellin Pelrine; Zhijing Jin
>
> **摘要:** As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.
>
---
#### [replaced 020] Unveil Multi-Picture Descriptions for Multilingual Mild Cognitive Impairment Detection via Contrastive Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.17067v3](http://arxiv.org/pdf/2505.17067v3)**

> **作者:** Kristin Qi; Jiali Cheng; Youxiang Zhu; Hadi Amiri; Xiaohui Liang
>
> **备注:** IEEE Global Communications Conference (GlobeCom) 2025
>
> **摘要:** Detecting Mild Cognitive Impairment from picture descriptions is critical yet challenging, especially in multilingual and multiple picture settings. Prior work has primarily focused on English speakers describing a single picture (e.g., the 'Cookie Theft'). The TAUKDIAL-2024 challenge expands this scope by introducing multilingual speakers and multiple pictures, which presents new challenges in analyzing picture-dependent content. To address these challenges, we propose a framework with three components: (1) enhancing discriminative representation learning via supervised contrastive learning, (2) involving image modality rather than relying solely on speech and text modalities, and (3) applying a Product of Experts (PoE) strategy to mitigate spurious correlations and overfitting. Our framework improves MCI detection performance, achieving a +7.1% increase in Unweighted Average Recall (UAR) (from 68.1% to 75.2%) and a +2.9% increase in F1 score (from 80.6% to 83.5%) compared to the text unimodal baseline. Notably, the contrastive learning component yields greater gains for the text modality compared to speech. These results highlight our framework's effectiveness in multilingual and multi-picture MCI detection.
>
---
#### [replaced 021] Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12854v3](http://arxiv.org/pdf/2503.12854v3)**

> **作者:** Songjun Tu; Jiahao Lin; Xiangyu Tian; Qichao Zhang; Linjing Li; Yuqian Fu; Nan Xu; Wei He; Xiangyuan Lan; Dongmei Jiang; Dongbin Zhao
>
> **备注:** 23pages
>
> **摘要:** Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
>
---
#### [replaced 022] MeTHanol: Modularized Thinking Language Models with Intermediate Layer Thinking, Decoding and Bootstrapping Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.12059v5](http://arxiv.org/pdf/2409.12059v5)**

> **作者:** Ningyuan Xi; Xiaoyu Wang; Yetao Wu; Teng Chen; Qingqing Gu; Yue Zhao; Jinxian Qu; Zhonglin Jiang; Yong Chen; Luo Ji
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Current research efforts are focused on enhancing the thinking and reasoning capability of large language model (LLM) by prompting, data-driven emergence and inference-time computation. In this study, we consider stimulating language model's thinking and cognitive abilities from a modular perspective, which mimics the human brain architecture. We select a specific intermediate attention layer with newly implemented language heads. We conduct dual-layer fine-tuning by annotated (query, thought, answer) samples and show that the intermediate layer can also learn to decode fluent and reasonable language tokens. A two-pass inference mechanism is designed to generate thoughts then formal responses. The entire framework is called modularized thinking language model (MeTHanol) which can enhance LLM's cognitive behaviors as indicated by Theory of Mind (ToM) and Vignette-based experiments. Case studies also show that MeTHanol can plan and self-reflect and generate human-like thoughts and answers, even on unseen and open-domain tasks. MeTHanol can also adapt to a personalized prompt and behave as the specified character. Our study holds promise for significant cognitive gains from a modular perspective. Our code, model and data are available at https://bachozean.github.io/methanol-page
>
---
#### [replaced 023] Data Caricatures: On the Representation of African American Language in Pretraining Corpora
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10789v2](http://arxiv.org/pdf/2503.10789v2)**

> **作者:** Nicholas Deas; Blake Vente; Amith Ananthram; Jessica A. Grieser; Desmond Patton; Shana Kleiner; James Shepard; Kathleen McKeown
>
> **备注:** ACL 2025
>
> **摘要:** With a combination of quantitative experiments, human judgments, and qualitative analyses, we evaluate the quantity and quality of African American Language (AAL) representation in 12 predominantly English, open-source pretraining corpora. We specifically focus on the sources, variation, and naturalness of included AAL texts representing the AAL-speaking community. We find that AAL is underrepresented in all evaluated pretraining corpora compared to US demographics, constituting as few as 0.007% and at most 0.18% of documents. We also find that more than 25% of AAL texts in C4 may be perceived as inappropriate for LLMs to generate and to reinforce harmful stereotypes. Finally, we find that most automated filters are more likely to conserve White Mainstream English (WME) texts over AAL in pretraining corpora.
>
---
#### [replaced 024] Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04886v2](http://arxiv.org/pdf/2507.04886v2)**

> **作者:** A. Bochkov
>
> **备注:** Added a new Ablation Study section with a key experiment on random noise embeddings. Expanded the discussion on 'representational interference' and updated results and figures accordingly
>
> **摘要:** Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research.
>
---
#### [replaced 025] Advancing Large Language Models for Tibetan with Curated Data and Continual Pre-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09205v4](http://arxiv.org/pdf/2507.09205v4)**

> **作者:** Leiyu Pan; Bojian Xiong; Lei Yang; Renren Jin; Shaowei Zhang; Yue Chen; Ling Shi; Jiang Zhou; Junru Wu; Zhen Wang; Jianxiang Peng; Juesi Xiao; Tianyu Dong; Zhuowen Han; Zhuo Chen; Yuqi Ren; Deyi Xiong
>
> **摘要:** Large language models have achieved remarkable progress across many languages. However, Tibetan, as a representative low-resource language, is particularly underrepresented in existing models due to the scarcity of high-quality training corpora. To address this gap, we curate the largest Tibetan pre-training corpus to date, aggregating data from diverse sources and applying a dedicated data cleaning and processing pipeline tailored for Tibetan. With the curated data, we continue pre/post-training a multilingual base model to enhance its generative capabilities in Tibetan. To evaluate the Tibetan capabilities of the model, we create new high-quality Tibetan benchmarks, and complement them with existing public benchmarks. Experimental results demonstrate that our model consistently and significantly outperforms both open-source models of similar scale and Tibetan-tailored models across a wide range of tasks.
>
---
#### [replaced 026] Memorization: A Close Look at Books
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12549v2](http://arxiv.org/pdf/2504.12549v2)**

> **作者:** Iris Ma; Ian Domingo; Alberto Krone-Martins; Pierre Baldi; Cristina V. Lopes
>
> **备注:** Accepted at ACL 2025 L2M2 Workshop
>
> **摘要:** To what extent can entire books be extracted from LLMs? Using the Llama 3 70B family of models, and the "prefix-prompting" extraction technique, we were able to auto-regressively reconstruct, with a very high level of similarity, one entire book (Alice's Adventures in Wonderland) from just the first 500 tokens. We were also able to obtain high extraction rates on several other books, piece-wise. However, these successes do not extend uniformly to all books. We show that extraction rates of books correlate with book popularity and thus, likely duplication in the training data. We also confirm the undoing of mitigations in the instruction-tuned Llama 3.1, following recent work (Nasr et al., 2025). We further find that this undoing comes from changes to only a tiny fraction of weights concentrated primarily in the lower transformer blocks. Our results provide evidence of the limits of current regurgitation mitigation strategies and introduce a framework for studying how fine-tuning affects the retrieval of verbatim memorization in aligned LLMs.
>
---
#### [replaced 027] Explainable Synthetic Image Detection through Diffusion Timestep Ensembling
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06201v2](http://arxiv.org/pdf/2503.06201v2)**

> **作者:** Yixin Wu; Feiran Zhang; Tianyuan Shi; Ruicheng Yin; Zhenghua Wang; Zhenliang Gan; Xiaohua Wang; Changze Lv; Xiaoqing Zheng; Xuanjing Huang
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Recent advances in diffusion models have enabled the creation of deceptively real images, posing significant security risks when misused. In this study, we empirically show that different timesteps of DDIM inversion reveal varying subtle distinctions between synthetic and real images that are extractable for detection, in the forms of such as Fourier power spectrum high-frequency discrepancies and inter-pixel variance distributions. Based on these observations, we propose a novel synthetic image detection method that directly utilizes features of intermediately noised images by training an ensemble on multiple noised timesteps, circumventing conventional reconstruction-based strategies. To enhance human comprehension, we introduce a metric-grounded explanation generation and refinement module to identify and explain AI-generated flaws. Additionally, we construct the GenHard and GenExplain benchmarks to provide detection samples of greater difficulty and high-quality rationales for fake images. Extensive experiments show that our method achieves state-of-the-art performance with 98.91% and 95.89% detection accuracy on regular and challenging samples respectively, and demonstrates generalizability and robustness. Our code and datasets are available at https://github.com/Shadowlized/ESIDE.
>
---
#### [replaced 028] Navigating the Risks of Using Large Language Models for Text Annotation in Social Science Research
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.22040v2](http://arxiv.org/pdf/2503.22040v2)**

> **作者:** Hao Lin; Yongjun Zhang
>
> **摘要:** Large language models (LLMs) have the potential to revolutionize computational social science, particularly in automated textual analysis. In this paper, we conduct a systematic evaluation of the promises and risks associated with using LLMs for text classification tasks, using social movement studies as an example. We propose a framework for social scientists to incorporate LLMs into text annotation, either as the primary coding decision-maker or as a coding assistant. This framework offers researchers tools to develop the potential best-performing prompt, and to systematically examine and report the validity and reliability of LLMs as a methodological tool. Additionally, we evaluate and discuss its epistemic risks associated with validity, reliability, replicability, and transparency. We conclude with several practical guidelines for using LLMs in text annotation tasks and offer recommendations for more effectively communicating epistemic risks in research.
>
---
#### [replaced 029] Everything is a Video: Unifying Modalities through Next-Frame Prediction
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.10503v2](http://arxiv.org/pdf/2411.10503v2)**

> **作者:** G. Thomas Hudson; Dean Slack; Thomas Winterbottom; Jamie Sterling; Chenghao Xiao; Junjie Shentu; Noura Al Moubayed
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Multimodal learning, which involves integrating information from various modalities such as text, images, audio, and video, is pivotal for numerous complex tasks like visual question answering, cross-modal retrieval, and caption generation. Traditional approaches rely on modality-specific encoders and late fusion techniques, which can hinder scalability and flexibility when adapting to new tasks or modalities. To address these limitations, we introduce a novel framework that extends the concept of task reformulation beyond natural language processing (NLP) to multimodal learning. We propose to reformulate diverse multimodal tasks into a unified next-frame prediction problem, allowing a single model to handle different modalities without modality-specific components. This method treats all inputs and outputs as sequential frames in a video, enabling seamless integration of modalities and effective knowledge transfer across tasks. Our approach is evaluated on a range of tasks, including text-to-text, image-to-text, video-to-video, video-to-text, and audio-to-text, demonstrating the model's ability to generalize across modalities with minimal adaptation. We show that task reformulation can significantly simplify multimodal model design across various tasks, laying the groundwork for more generalized multimodal foundation models.
>
---
#### [replaced 030] AutoLibra: Agent Metric Induction from Open-Ended Feedback
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02820v2](http://arxiv.org/pdf/2505.02820v2)**

> **作者:** Hao Zhu; Phil Cuvin; Xinkai Yu; Charlotte Ka Yee Yan; Jason Zhang; Diyi Yang
>
> **备注:** https://opensocial.world/
>
> **摘要:** Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose AutoLibra, a framework for agent evaluation, that transforms open-ended human feedback e.g. "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own" into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta-metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra-induced metrics serve as better prompt-engineering targets than the task success rate on a wide range of text game tasks, improving agent performance over baseline by a mean of 20%. Second, we show that AutoLibra can iteratively select high-quality fine-tuning data for web navigation agents. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents.
>
---
#### [replaced 031] A Structured Bangla Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13610v3](http://arxiv.org/pdf/2506.13610v3)**

> **作者:** Abdullah Al Shafi; Rowzatul Zannat; Abdul Muntakim; Mahmudul Hasan
>
> **备注:** Preprint
>
> **摘要:** Disease-symptom datasets are significant and in demand for medical research, disease diagnosis, clinical decision-making, and AI-driven health management applications. These datasets help identify symptom patterns associated with specific diseases, thus improving diagnostic accuracy and enabling early detection. The dataset presented in this study systematically compiles disease-symptom relationships from various online sources, medical literature, and publicly available health databases. The data was gathered through analyzing peer-reviewed medical articles, clinical case studies, and disease-symptom association reports. Only the verified medical sources were included in the dataset, while those from non-peer-reviewed and anecdotal sources were excluded. The dataset is structured in a tabular format, where the first column represents diseases, and the remaining columns represent symptoms. Each symptom cell contains a binary value, indicating whether a symptom is associated with a disease. Thereby, this structured representation makes the dataset very useful for a wide range of applications, including machine learning-based disease prediction, clinical decision support systems, and epidemiological studies. Although there are some advancements in the field of disease-symptom datasets, there is a significant gap in structured datasets for the Bangla language. This dataset aims to bridge that gap by facilitating the development of multilingual medical informatics tools and improving disease prediction models for underrepresented linguistic communities. Further developments should include region-specific diseases and further fine-tuning of symptom associations for better diagnostic performance
>
---
#### [replaced 032] Preference learning made easy: Everything should be understood through win rate
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.10505v2](http://arxiv.org/pdf/2502.10505v2)**

> **作者:** Lily H. Zhang; Rajesh Ranganath
>
> **备注:** ICML 2025
>
> **摘要:** Preference learning, or the task of aligning generative models to preference comparison data, has yet to reach the conceptual maturity of classification, density estimation, etc. To close this gap, this work presents a framework to understand preference learning starting from the sampling distribution of pairwise preference data. First, we prove that the only evaluation of a generative model that respects both preferences and prevalences in the data distribution is a form of win rate, justifying win rate as the focal point to understand preference learning. We then analyze preference learning methods as win rate optimization (WRO) or non-WRO. We present novel instances of WRO beyond existing examples (RLHF, NLHF) and identify two key theoretical benefits of all such methods. We prove that common non-WRO methods like DPO and SFT on preferred samples lack these properties and suggest ways to mitigate such theoretical limitations. We also show that WRO underperforms in practice due optimization difficulties and that optimization success predicts performance better than choices which affect the objective's solution. Our analysis highlights best practices for existing methods and provides recommendations for future research, guided by the principle that one should either align non-WRO methods more closely with WRO or improve the optimization of WRO objectives.
>
---
#### [replaced 033] Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.00222v2](http://arxiv.org/pdf/2406.00222v2)**

> **作者:** Maximillian Chen; Ruoxi Sun; Tomas Pfister; Sercan Ö. Arık
>
> **备注:** ICLR 2025; Code: https://github.com/google-research/google-research/tree/master/learning_to_clarify
>
> **摘要:** Large language models (LLMs), optimized through human feedback, have rapidly emerged as a leading paradigm for developing intelligent conversational assistants. However, despite their strong performance across many benchmarks, LLM-based agents might still lack conversational skills such as disambiguation -- when they are faced with ambiguity, they often overhedge or implicitly guess users' true intents rather than asking clarification questions. Under task-specific settings, high-quality conversation samples are often limited, constituting a bottleneck for LLMs' ability to learn optimal dialogue action policies. We propose Action-Based Contrastive Self-Training (ACT), a quasi-online preference optimization algorithm based on Direct Preference Optimization (DPO), that enables data-efficient dialogue policy learning in multi-turn conversation modeling. We demonstrate ACT's efficacy under in data-efficient tuning scenarios, even when there is no action label available, using multiple real-world conversational tasks: tabular-grounded question-answering, machine reading comprehension, and AmbigSQL, a novel task for disambiguating information-seeking requests for complex SQL generation towards data analysis agents. Additionally, we propose evaluating LLMs' ability to function as conversational agents by examining whether they can implicitly recognize and reason about ambiguity in conversation. ACT demonstrates substantial conversation modeling improvements over standard tuning approaches like supervised fine-tuning and DPO.
>
---
#### [replaced 034] Multi-Agent Retrieval-Augmented Framework for Evidence-Based Counterspeech Against Health Misinformation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07307v2](http://arxiv.org/pdf/2507.07307v2)**

> **作者:** Anirban Saha Anik; Xiaoying Song; Elliott Wang; Bryan Wang; Bengisu Yarimbas; Lingzi Hong
>
> **备注:** Accepted for publication at COLM 2025
>
> **摘要:** Large language models (LLMs) incorporated with Retrieval-Augmented Generation (RAG) have demonstrated powerful capabilities in generating counterspeech against misinformation. However, current studies rely on limited evidence and offer less control over final outputs. To address these challenges, we propose a Multi-agent Retrieval-Augmented Framework to generate counterspeech against health misinformation, incorporating multiple LLMs to optimize knowledge retrieval, evidence enhancement, and response refinement. Our approach integrates both static and dynamic evidence, ensuring that the generated counterspeech is relevant, well-grounded, and up-to-date. Our method outperforms baseline approaches in politeness, relevance, informativeness, and factual accuracy, demonstrating its effectiveness in generating high-quality counterspeech. To further validate our approach, we conduct ablation studies to verify the necessity of each component in our framework. Furthermore, cross evaluations show that our system generalizes well across diverse health misinformation topics and datasets. And human evaluations reveal that refinement significantly enhances counterspeech quality and obtains human preference.
>
---
#### [replaced 035] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions
- **分类: cs.LG; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.02087v2](http://arxiv.org/pdf/2507.02087v2)**

> **作者:** Eitan Anzenberg; Arunava Samajpati; Sivasankaran Chandrasekar; Varun Kacholia
>
> **备注:** 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
>
> **摘要:** The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
>
---
#### [replaced 036] Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17137v2](http://arxiv.org/pdf/2505.17137v2)**

> **作者:** Kristin Qi; Youxiang Zhu; Caroline Summerour; John A. Batsis; Xiaohui Liang
>
> **备注:** IEEE Global Communications Conference (GlobeCom) 2025
>
> **摘要:** Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
>
---
#### [replaced 037] Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18509v2](http://arxiv.org/pdf/2502.18509v2)**

> **作者:** Ivoline Ngong; Swanand Kadhe; Hao Wang; Keerthiram Murugesan; Justin D. Weisz; Amit Dhurandhar; Karthikeyan Natesan Ramamurthy
>
> **备注:** 22 pages, 2 figures
>
> **摘要:** Conversational agents are increasingly woven into individuals' personal lives, yet users often underestimate the privacy risks associated with them. The moment users share information with these agents-such as large language models (LLMs)-their private information becomes vulnerable to exposure. In this paper, we characterize the notion of contextual privacy for user interactions with LLM-based Conversational Agents (LCAs). It aims to minimize privacy risks by ensuring that users (sender) disclose only information that is both relevant and necessary for achieving their intended goals when interacting with LCAs (untrusted receivers). Through a formative design user study, we observe how even "privacy-conscious" users inadvertently reveal sensitive information through indirect disclosures. Based on insights from this study, we propose a locally deployable framework that operates between users and LCAs, identifying and reformulating out-of-context information in user prompts. Our evaluation using examples from ShareGPT shows that lightweight models can effectively implement this framework, achieving strong gains in contextual privacy while preserving the user's intended interaction goals. Notably, about 76% of participants in our human evaluation preferred the reformulated prompts over the original ones, validating the usability and effectiveness of contextual privacy in our proposed framework. We opensource the code at https://github.com/IBM/contextual-privacy-LLM.
>
---
#### [replaced 038] DoubleDipper: Improving Long-Context LLMs via Context Recycling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.13632v4](http://arxiv.org/pdf/2406.13632v4)**

> **作者:** Arie Cattan; Alon Jacovi; Alex Fabrikant; Jonathan Herzig; Roee Aharoni; Hannah Rashkin; Dror Marcus; Avinatan Hassidim; Yossi Matias; Idan Szpektor; Avi Caciularu
>
> **摘要:** Despite recent advancements in Large Language Models (LLMs), their performance on tasks involving long contexts remains sub-optimal. In this work, we propose DoubleDipper, a novel In-Context-Learning method that automatically generates few-shot examples for long context QA tasks by recycling contexts. Specifically, given a long input context (1-3k tokens) and a query, we generate additional query-output pairs from the given context as few-shot examples, while introducing the context only once. This ensures that the demonstrations are leveraging the same context as the target query while only adding a small number of tokens to the prompt. We further enhance each demonstration by instructing the model to explicitly identify the relevant paragraphs before the answer, which improves performance while providing fine-grained attribution to the answer source. We apply our method on multiple LLMs and obtain substantial improvements (+16 absolute points on average across models) on various QA datasets with long context. Surprisingly, despite introducing only single-hop ICL examples, LLMs successfully generalize to multi-hop long-context QA using our approach.
>
---
#### [replaced 039] Intersectional Bias in Japanese Large Language Models from a Contextualized Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12327v2](http://arxiv.org/pdf/2506.12327v2)**

> **作者:** Hitomi Yanaka; Xinqi He; Jie Lu; Namgi Han; Sunjin Oh; Ryoma Kumon; Yuma Matsuoka; Katsuhiko Watabe; Yuko Itatsu
>
> **备注:** Accepted to the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP2025) at ACL2025
>
> **摘要:** An increasing number of studies have examined the social bias of rapidly developed large language models (LLMs). Although most of these studies have focused on bias occurring in a single social attribute, research in social science has shown that social bias often occurs in the form of intersectionality -- the constitutive and contextualized perspective on bias aroused by social attributes. In this study, we construct the Japanese benchmark inter-JBBQ, designed to evaluate the intersectional bias in LLMs on the question-answering setting. Using inter-JBBQ to analyze GPT-4o and Swallow, we find that biased output varies according to its contexts even with the equal combination of social attributes.
>
---
#### [replaced 040] LIMO: Less is More for Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.03387v2](http://arxiv.org/pdf/2502.03387v2)**

> **作者:** Yixin Ye; Zhen Huang; Yang Xiao; Ethan Chern; Shijie Xia; Pengfei Liu
>
> **备注:** COLM 2025
>
> **摘要:** We challenge the prevailing assumption that complex reasoning in large language models (LLMs) necessitates massive training data. We demonstrate that sophisticated mathematical reasoning can emerge with only a few examples. Specifically, through simple supervised fine-tuning, our model, LIMO, achieves 63.3\% accuracy on AIME24 and 95.6\% on MATH500, surpassing previous fine-tuned models (6.5\% on AIME24, 59.2\% on MATH500) while using only 1\% of the training data required by prior approaches. Furthermore, LIMO exhibits strong out-of-distribution generalization, achieving a 45.8\% absolute improvement across diverse benchmarks, outperforming models trained on 100x more data. Synthesizing these findings, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes. This hypothesis suggests that the threshold for eliciting complex reasoning is not dictated by task complexity but rather by two key factors: (1) the completeness of the model's pre-trained knowledge base and (2) the effectiveness of post-training examples in serving as "cognitive templates" that guide reasoning.
>
---
#### [replaced 041] TIB-STC: A Large-Scale Structured Tibetan Benchmark for Low-Resource Language Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18288v4](http://arxiv.org/pdf/2503.18288v4)**

> **作者:** Cheng Huang; Fan Gao; Yutong Liu; Nyima Tashi; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Xiao Feng; Hao Wang; Yongbin Yu
>
> **摘要:** Advancement of large language models (LLMs) has brought transformative capabilities to NLP, but such progress remains unevenly distributed, especially for low-resource and culturally rich languages like Tibetan. In this paper, we present TIB-STC, the first large-scale, expert-curated, and multi-domain benchmark specifically designed to support the development and evaluation of LLMs for the Tibetan language. Spanning over 11 billion tokens across literature, religion, medicine, law, and daily communication, TIB-STC preserves traditional grammar and stylistic richness. To validate its utility, we train a reference model, Sun-Shine, on TIB-STC through a three-stage pipeline involving pretraining, supervised fine-tuning, and preference optimization. Evaluation on TLUE Benchmark for Tibetan-specific tasks, including Ti-MMLU and Ti-SafetyBench, demonstrates the benchmark's effectiveness in enabling robust instruction-following and culturally aligned generation. We release TIB-STC to advance research in low-resource language modeling and promote inclusivity in multilingual NLP. All data are available at: https://github.com/Vicentvankor/sun-shine
>
---
#### [replaced 042] Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12370v2](http://arxiv.org/pdf/2503.12370v2)**

> **作者:** Rupak Sarkar; Neha Srikanth; Taylor Hudson; Rachel Rudinger; Claire Bonial; Philip Resnik
>
> **备注:** 8 pages
>
> **摘要:** While it is commonly accepted that maintaining common ground plays a role in conversational success, little prior research exists connecting conversational grounding to success in task-oriented conversations. We study failures of grounding in the Ubuntu IRC dataset, where participants use text-only communication to resolve technical issues. We find that disruptions in conversational flow often stem from a misalignment in common ground, driven by a divergence in beliefs and assumptions held by participants. These disruptions, which we call conversational friction, significantly correlate with task success. We find that although LLMs can identify overt cases of conversational friction, they struggle with subtler and more context-dependent instances requiring pragmatic or domain-specific reasoning.
>
---
#### [replaced 043] LLM2TEA: An Agentic AI Designer for Discovery with Generative Evolutionary Multitasking
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2406.14917v3](http://arxiv.org/pdf/2406.14917v3)**

> **作者:** Melvin Wong; Jiao Liu; Thiago Rios; Stefan Menzel; Yew Soon Ong
>
> **备注:** This work is accepted by IEEE CIM. IEEE copyrights applies
>
> **摘要:** This paper presents LLM2TEA, a Large Language Model (LLM) driven MultiTask Evolutionary Algorithm, representing the first agentic AI designer of its kind operating with generative evolutionary multitasking (GEM). LLM2TEA enables the crossbreeding of solutions from multiple domains, fostering novel solutions that transcend disciplinary boundaries. Of particular interest is the ability to discover designs that are both novel and conforming to real-world physical specifications. LLM2TEA comprises an LLM to generate genotype samples from text prompts describing target objects, a text-to-3D generative model to produce corresponding phenotypes, a classifier to interpret its semantic representations, and a computational simulator to assess its physical properties. Novel LLM-based multitask evolutionary operators are introduced to guide the search towards high-performing, practically viable designs. Experimental results in conceptual design optimization validate the effectiveness of LLM2TEA, showing 97% to 174% improvements in the diversity of novel designs over the current text-to-3D baseline. Moreover, over 73% of the generated designs outperform the top 1% of designs produced by the text-to-3D baseline in terms of physical performance. The designs produced by LLM2TEA are not only aesthetically creative but also functional in real-world contexts. Several of these designs have been successfully 3D printed, demonstrating the ability of our approach to transform AI-generated outputs into tangible, physical designs. These designs underscore the potential of LLM2TEA as a powerful tool for complex design optimization and discovery, capable of producing novel and physically viable designs.
>
---
#### [replaced 044] Pruning for Performance: Efficient Idiom and Metaphor Classification in Low-Resource Konkani Using mBERT
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02005v2](http://arxiv.org/pdf/2506.02005v2)**

> **作者:** Timothy Do; Pranav Saran; Harshita Poojary; Pranav Prabhu; Sean O'Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** In this paper, we address the persistent challenges that figurative language expressions pose for natural language processing (NLP) systems, particularly in low-resource languages such as Konkani. We present a hybrid model that integrates a pre-trained Multilingual BERT (mBERT) with a bidirectional LSTM and a linear classifier. This architecture is fine-tuned on a newly introduced annotated dataset for metaphor classification, developed as part of this work. To improve the model's efficiency, we implement a gradient-based attention head pruning strategy. For metaphor classification, the pruned model achieves an accuracy of 78%. We also applied our pruning approach to expand on an existing idiom classification task, achieving 83% accuracy. These results demonstrate the effectiveness of attention head pruning for building efficient NLP tools in underrepresented languages.
>
---
#### [replaced 045] Checklist Engineering Empowers Multilingual LLM Judges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06774v2](http://arxiv.org/pdf/2507.06774v2)**

> **作者:** Mohammad Ghiasvand Mohammadkhani; Hamid Beigy
>
> **摘要:** Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
>
---
#### [replaced 046] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18224v2](http://arxiv.org/pdf/2507.18224v2)**

> **作者:** Shiyuan Li; Yixin Liu; Qingsong Wen; Chengqi Zhang; Shirui Pan
>
> **摘要:** Multi-agent systems (MAS) based on large language models (LLMs) have emerged as a powerful solution for dealing with complex problems across diverse domains. The effectiveness of MAS is critically dependent on its collaboration topology, which has become a focal point for automated design research. However, existing approaches are fundamentally constrained by their reliance on a template graph modification paradigm with a predefined set of agents and hard-coded interaction structures, significantly limiting their adaptability to task-specific requirements. To address these limitations, we reframe MAS design as a conditional autoregressive graph generation task, where both the system composition and structure are designed jointly. We propose ARG-Designer, a novel autoregressive model that operationalizes this paradigm by constructing the collaboration graph from scratch. Conditioned on a natural language task query, ARG-Designer sequentially and dynamically determines the required number of agents, selects their appropriate roles from an extensible pool, and establishes the optimal communication links between them. This generative approach creates a customized topology in a flexible and extensible manner, precisely tailored to the unique demands of different tasks. Extensive experiments across six diverse benchmarks demonstrate that ARG-Designer not only achieves state-of-the-art performance but also enjoys significantly greater token efficiency and enhanced extensibility. The source code of ARG-Designer is available at https://github.com/Shiy-Li/ARG-Designer.
>
---
#### [replaced 047] Critiques of World Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05169v3](http://arxiv.org/pdf/2507.05169v3)**

> **作者:** Eric Xing; Mingkai Deng; Jinyu Hou; Zhiting Hu
>
> **摘要:** World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.
>
---
#### [replaced 048] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17527v3](http://arxiv.org/pdf/2507.17527v3)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Yujiao Du; Ting Han; Yuxiang Hu; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Jun Zhang; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [replaced 049] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.15956v3](http://arxiv.org/pdf/2410.15956v3)**

> **作者:** Yanzhu Guo; Simone Conia; Zelin Zhou; Min Li; Saloni Potdar; Henry Xiao
>
> **备注:** ACL 2025
>
> **摘要:** Current Large Language Models (LLMs) are predominantly designed with English as the primary language, and even the few that are multilingual tend to exhibit strong English-centric biases. Much like speakers who might produce awkward expressions when learning a second language, LLMs often generate unnatural outputs in non-English languages, reflecting English-centric patterns in both vocabulary and grammar. Despite the importance of this issue, the naturalness of multilingual LLM outputs has received limited attention. In this paper, we address this gap by introducing novel automatic corpus-level metrics to assess the lexical and syntactic naturalness of LLM outputs in a multilingual context. Using our new metrics, we evaluate state-of-the-art LLMs on a curated benchmark in French and Chinese, revealing a tendency towards English-influenced patterns. To mitigate this issue, we also propose a simple and effective alignment method to improve the naturalness of an LLM in a target language and domain, achieving consistent improvements in naturalness without compromising the performance on general-purpose benchmarks. Our work highlights the importance of developing multilingual metrics, resources and methods for the new wave of multilingual LLMs.
>
---
#### [replaced 050] Benchmarking Graph Neural Networks for Document Layout Analysis in Public Affairs
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14699v2](http://arxiv.org/pdf/2505.14699v2)**

> **作者:** Miguel Lopez-Duran; Julian Fierrez; Aythami Morales; Ruben Tolosana; Oscar Delgado-Mohatar; Alvaro Ortigosa
>
> **备注:** 15 pages, 2 figures, accepted paper at The Fifth ICDAR International Workshop on Machine Learning
>
> **摘要:** The automatic analysis of document layouts in digital-born PDF documents remains a challenging problem due to the heterogeneous arrangement of textual and nontextual elements and the imprecision of the textual metadata in the Portable Document Format. In this work, we benchmark Graph Neural Network (GNN) architectures for the task of fine-grained layout classification of text blocks from digital native documents. We introduce two graph construction structures: a k-closest-neighbor graph and a fully connected graph, and generate node features via pre-trained text and vision models, thus avoiding manual feature engineering. Three experimental frameworks are evaluated: single-modality (text or visual), concatenated multimodal, and dual-branch multimodal. We evaluated four foundational GNN models and compared them with the baseline. Our experiments are specifically conducted on a rich dataset of public affairs documents that includes more than 20 sources (e.g., regional and national-level official gazettes), 37K PDF documents, with 441K pages in total. Our results demonstrate that GraphSAGE operating on the k-closest-neighbor graph in a dual-branch configuration achieves the highest per-class and overall accuracy, outperforming the baseline in some sources. These findings confirm the importance of local layout relationships and multimodal fusion exploited through GNNs for the analysis of native digital document layouts.
>
---
#### [replaced 051] Code-Switching and Syntax: A Large-Scale Experiment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01846v2](http://arxiv.org/pdf/2506.01846v2)**

> **作者:** Igor Sterner; Simone Teufel
>
> **备注:** Findings of ACL 2025
>
> **摘要:** The theoretical code-switching (CS) literature provides numerous pointwise investigations that aim to explain patterns in CS, i.e. why bilinguals switch language in certain positions in a sentence more often than in others. A resulting consensus is that CS can be explained by the syntax of the contributing languages. There is however no large-scale, multi-language, cross-phenomena experiment that tests this claim. When designing such an experiment, we need to make sure that the system that is predicting where bilinguals tend to switch has access only to syntactic information. We provide such an experiment here. Results show that syntax alone is sufficient for an automatic system to distinguish between sentences in minimal pairs of CS, to the same degree as bilingual humans. Furthermore, the learnt syntactic patterns generalise well to unseen language pairs.
>
---
#### [replaced 052] Juru: Legal Brazilian Large Language Model from Reputable Sources
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.18140v2](http://arxiv.org/pdf/2403.18140v2)**

> **作者:** Roseval Malaquias Junior; Ramon Pires; Roseli Romero; Rodrigo Nogueira
>
> **摘要:** The high compute cost associated with pretraining large language models limits their research. Two strategies have emerged to address this issue: domain specialization and pretraining with high-quality data. To explore these strategies, we specialized the Mistral-7B model with 1.9 billion unique tokens from reputable Brazilian legal sources and conducted few-shot evaluations on legal and general knowledge test suites. Our model, Juru, demonstrates the benefits of domain specialization by achieving improved performance on legal benchmarks, even with a reduced amount of pretraining data. However, this domain specialization through continued pretraining comes at the cost of increased forgetting in unrelated domains, as evidenced by performance degradation on general knowledge test suites in both Portuguese and English. This study contributes to the growing body of scientific evidence showing that pretraining data selection may enhance the performance of large language models, enabling the exploration of these models at a lower cost. Juru is publicly available at https://huggingface.co/roseval/Juru-7B .
>
---
#### [replaced 053] Language Modeling for the Future of Finance: A Survey into Metrics, Tasks, and Data Opportunities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07274v2](http://arxiv.org/pdf/2504.07274v2)**

> **作者:** Nikita Tatarinov; Siddhant Sukhani; Agam Shah; Sudheer Chava
>
> **摘要:** Recent advances in language modeling have led to growing interest in applying Natural Language Processing (NLP) techniques to financial problems, enabling new approaches to analysis and decision-making. To systematically examine this trend, we review 374 NLP research papers published between 2017 and 2024 across 38 conferences and workshops, with a focused analysis of 221 papers that directly address finance-related tasks. We evaluate these papers across 11 quantitative and qualitative dimensions, and our study identifies the following opportunities: (i) expanding the scope of forecasting tasks; (ii) enriching evaluation with financial metrics; (iii) leveraging multilingual and crisis-period datasets; and (iv) balancing PLMs with efficient or interpretable alternatives. We identify actionable directions for research and practice, supported by dataset and tool recommendations, with implications for both the academia and industry communities.
>
---
#### [replaced 054] Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07840v3](http://arxiv.org/pdf/2504.07840v3)**

> **作者:** Cansu Koyuturk; Emily Theophilou; Sabrina Patania; Gregor Donabauer; Andrea Martinenghi; Chiara Antico; Alessia Telari; Alessia Testa; Sathya Bursic; Franca Garzotto; Davinia Hernandez-Leo; Udo Kruschwitz; Davide Taibi; Simona Amenta; Martin Ruskov; Dimitri Ognibene
>
> **备注:** Long paper accepted for AIED 2025, the 26th International Conference on Artificial Intelligence in Education, July 22 - 26, 2025, Palermo, Italy
>
> **摘要:** Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems.
>
---
#### [replaced 055] Otter: A Multi-Modal Model with In-Context Instruction Tuning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2305.03726v2](http://arxiv.org/pdf/2305.03726v2)**

> **作者:** Bo Li; Yuanhan Zhang; Liangyu Chen; Jinghao Wang; Fanyi Pu; Joshua Adrian Cahyono; Jingkang Yang; Ziwei Liu
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) have unveiled great potential as visual assistants. However, most existing works focus on responding to individual instructions or using previous dialogues for contextual understanding. There is little discussion on employing both images and text as in-context examples to enhance the instruction following capability. To bridge this gap, we introduce the \textbf{Otter} model to leverage both textual and visual in-context examples for instruction tuning. Specifically, Otter builds upon Flamingo with Perceiver architecture, and has been instruction tuned for general purpose multi-modal assistant. Otter seamlessly processes multi-modal inputs, supporting modalities including text, multiple images, and dynamic video content. To support the training of Otter, we present the \textbf{MIMIC-IT} (\textbf{M}ult\textbf{I}-\textbf{M}odal \textbf{I}n-\textbf{C}ontext \textbf{I}nstruction \textbf{T}uning) dataset, which encompasses over 3 million multi-modal instruction-response pairs, including approximately 2.2 million unique instructions across a broad spectrum of images and videos. MIMIC-IT has been carefully curated to feature a diverse array of in-context examples for each entry. Comprehensive evaluations suggest that instruction tuning with these in-context examples substantially enhances model convergence and generalization capabilities. Notably, the extensive scenario coverage provided by the MIMIC-IT dataset empowers the Otter model to excel in tasks involving complex video and multi-image understanding.
>
---
#### [replaced 056] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.08026v2](http://arxiv.org/pdf/2503.08026v2)**

> **作者:** Zhen Tan; Jun Yan; I-Hung Hsu; Rujun Han; Zifeng Wang; Long T. Le; Yiwen Song; Yanfei Chen; Hamid Palangi; George Lee; Anand Iyer; Tianlong Chen; Huan Liu; Chen-Yu Lee; Tomas Pfister
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Large Language Models (LLMs) have made significant progress in open-ended dialogue, yet their inability to retain and retrieve relevant information from long-term interactions limits their effectiveness in applications requiring sustained personalization. External memory mechanisms have been proposed to address this limitation, enabling LLMs to maintain conversational continuity. However, existing approaches struggle with two key challenges. First, rigid memory granularity fails to capture the natural semantic structure of conversations, leading to fragmented and incomplete representations. Second, fixed retrieval mechanisms cannot adapt to diverse dialogue contexts and user interaction patterns. In this work, we propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence. Experiments show that RMM demonstrates consistent improvement across various metrics and benchmarks. For example, RMM shows more than 10% accuracy improvement over the baseline without memory management on the LongMemEval dataset.
>
---
#### [replaced 057] Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16534v2](http://arxiv.org/pdf/2507.16534v2)**

> **作者:** Shanghai AI Lab; :; Xiaoyang Chen; Yunhao Chen; Zeren Chen; Zhiyun Chen; Hanyun Cui; Yawen Duan; Jiaxuan Guo; Qi Guo; Xuhao Hu; Hong Huang; Lige Huang; Chunxiao Li; Juncheng Li; Qihao Lin; Dongrui Liu; Xinmin Liu; Zicheng Liu; Chaochao Lu; Xiaoya Lu; Jingjing Qu; Qibing Ren; Jing Shao; Jingwei Shi; Jingwei Sun; Peng Wang; Weibing Wang; Jia Xu; Lewen Yan; Xiao Yu; Yi Yu; Boxuan Zhang; Jie Zhang; Weichen Zhang; Zhijie Zheng; Tianyi Zhou; Bowen Zhou
>
> **备注:** 97 pages, 37 figures
>
> **摘要:** To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges.
>
---
#### [replaced 058] Assessing the Reliability of LLMs Annotations in the Context of Demographic Bias and Model Explanation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13138v2](http://arxiv.org/pdf/2507.13138v2)**

> **作者:** Hadi Mohammadi; Tina Shahedi; Pablo Mosteiro; Massimo Poesio; Ayoub Bagheri; Anastasia Giachanou
>
> **摘要:** Understanding the sources of variability in annotations is crucial for developing fair NLP systems, especially for tasks like sexism detection where demographic bias is a concern. This study investigates the extent to which annotator demographic features influence labeling decisions compared to text content. Using a Generalized Linear Mixed Model, we quantify this inf luence, finding that while statistically present, demographic factors account for a minor fraction ( 8%) of the observed variance, with tweet content being the dominant factor. We then assess the reliability of Generative AI (GenAI) models as annotators, specifically evaluating if guiding them with demographic personas improves alignment with human judgments. Our results indicate that simplistic persona prompting often fails to enhance, and sometimes degrades, performance compared to baseline models. Furthermore, explainable AI (XAI) techniques reveal that model predictions rely heavily on content-specific tokens related to sexism, rather than correlates of demographic characteristics. We argue that focusing on content-driven explanations and robust annotation protocols offers a more reliable path towards fairness than potentially persona simulation.
>
---
#### [replaced 059] Should Top-Down Clustering Affect Boundaries in Unsupervised Word Discovery?
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19204v2](http://arxiv.org/pdf/2507.19204v2)**

> **作者:** Simon Malan; Benjamin van Niekerk; Herman Kamper
>
> **备注:** Submitted to the IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** We investigate the problem of segmenting unlabeled speech into word-like units and clustering these to create a lexicon. Prior work can be categorized into two frameworks. Bottom-up methods first determine boundaries and then cluster the fixed segmented words into a lexicon. In contrast, top-down methods incorporate information from the clustered words to inform boundary selection. However, it is unclear whether top-down information is necessary to improve segmentation. To explore this, we look at two similar approaches that differ in whether top-down clustering informs boundary selection. Our simple bottom-up strategy predicts word boundaries using the dissimilarity between adjacent self-supervised features, then clusters the resulting segments to construct a lexicon. Our top-down system is an updated version of the ES-KMeans dynamic programming method that iteratively uses K-means to update its boundaries. On the five-language ZeroSpeech benchmarks, both approaches achieve comparable state-of-the-art results, with the bottom-up system being nearly five times faster. Through detailed analyses, we show that the top-down influence of ES-KMeans can be beneficial (depending on factors like the candidate boundaries), but in many cases the simple bottom-up method performs just as well. For both methods, we show that the clustering step is a limiting factor. Therefore, we recommend that future work focus on improved clustering techniques and learning more discriminative word-like representations. Project code repository: https://github.com/s-malan/prom-seg-clus.
>
---
#### [replaced 060] Improving Similar Case Retrieval Ranking Performance By Revisiting RankSVM
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11131v2](http://arxiv.org/pdf/2502.11131v2)**

> **作者:** Yuqi Liu; Yan Zheng
>
> **摘要:** Given the rapid development of Legal AI, a lot of attention has been paid to one of the most important legal AI tasks--similar case retrieval, especially with language models to use. In our paper, however, we try to improve the ranking performance of current models from the perspective of learning to rank instead of language models. Specifically, we conduct experiments using a pairwise method--RankSVM as the classifier to substitute a fully connected layer, combined with commonly used language models on similar case retrieval datasets LeCaRDv1 and LeCaRDv2. We finally come to the conclusion that RankSVM could generally help improve the retrieval performance on the LeCaRDv1 and LeCaRDv2 datasets compared with original classifiers by optimizing the precise ranking. It could also help mitigate overfitting owing to class imbalance. Our code is available in https://github.com/liuyuqi123study/RankSVM_for_SLR
>
---
#### [replaced 061] Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data
- **分类: cs.CL; I.2.7; J.4**

- **链接: [http://arxiv.org/pdf/2401.12295v2](http://arxiv.org/pdf/2401.12295v2)**

> **作者:** Leonardo Castro-Gonzalez; Yi-Ling Chung; Hannak Rose Kirk; John Francis; Angus R. Williams; Pica Johansson; Jonathan Bright
>
> **备注:** 46 pages, 17 figures, 6 tables
>
> **摘要:** The field of machine learning has recently made significant progress in reducing the requirements for labelled training data when building new models. These `cheaper' learning techniques hold significant potential for the social sciences, where development of large labelled training datasets is often a significant practical impediment to the use of machine learning for analytical tasks. In this article we review three `cheap' techniques that have developed in recent years: weak supervision, transfer learning and prompt engineering. For the latter, we also review the particular case of zero-shot prompting of large language models. For each technique we provide a guide of how it works and demonstrate its application across six different realistic social science applications (two different tasks paired with three different dataset makeups). We show good performance for all techniques, and in particular we demonstrate how prompting of large language models can achieve high accuracy at very low cost. Our results are accompanied by a code repository to make it easy for others to duplicate our work and use it in their own research. Overall, our article is intended to stimulate further uptake of these techniques in the social sciences.
>
---
#### [replaced 062] Reinforcement learning fine-tuning of language model for instruction following and math reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21560v2](http://arxiv.org/pdf/2506.21560v2)**

> **作者:** Yifu Han; Geo Zhang
>
> **摘要:** This study investigates the effectiveness of reinforcement learning (RL) fine-tuning techniques on a compact language model (Qwen2.5-0.5B Base) for two challenging tasks: instruction following and mathematical reasoning. We compare supervised fine-tuning (SFT), Direct Preference Optimization (DPO) using preference-labeled data, and Reinforce Leave-One-Out (RLOO) with reward models. Our experiments show that RLOO with DeBERTa reward modeling achieves the best alignment, while DPO provides strong and consistent results. For math reasoing tasks, synthetic data augmentation and best-of-N sampling with an external verifier significantly improve accuracy, showing the potential of combining fine-tuning with inference-time tools. This study highlights key trade-offs and practical strategies for training lightweight, task-aligned small-scale language models.
>
---
#### [replaced 063] Real-time Factuality Assessment from Adversarial Feedback
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14651v3](http://arxiv.org/pdf/2410.14651v3)**

> **作者:** Sanxing Chen; Yukun Huang; Bhuwan Dhingra
>
> **摘要:** We show that existing evaluations for assessing the factuality of news from conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors-even after their knowledge cutoffs. This suggests that recent popular false information from such sources can be easily identified due to its likely presence in pre-training/retrieval corpora or the emergence of salient, yet shallow, patterns in these datasets. Instead, we argue that a proper factuality evaluation dataset should test a model's ability to reason about current events by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive variants that challenge LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both evaluating and generating challenging news examples, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG-based evaluation helps discover more deceitful patterns.
>
---
#### [replaced 064] LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10631v2](http://arxiv.org/pdf/2408.10631v2)**

> **作者:** Yupeng Su; Ziyi Guan; Xiaoqun Liu; Tianlai Jin; Dongkuan Wu; Zhengfei Chen; Graziano Chesi; Ngai Wong; Hao Yu
>
> **备注:** Accepted by ICCAD 2025
>
> **摘要:** Large language models (LLMs) have seen substantial growth, necessitating efficient model pruning techniques. Existing post-training pruning methods primarily measure weight importance in converged dense models, often overlooking changes in weight significance during the pruning process, leading to performance degradation. To address this issue, we present LLM-Barber (Block-Aware Rebuilder for Sparsity Mask in One-Shot), a novel one-shot pruning framework that rebuilds the sparsity mask of pruned models without any retraining or weight reconstruction. LLM-Barber incorporates block-aware error optimization across Self-Attention and MLP blocks, facilitating global performance optimization. We are the first to employ the product of weights and gradients as a pruning metric in the context of LLM post-training pruning. This enables accurate identification of weight importance in massive models and significantly reduces computational complexity compared to methods using secondorder information. Our experiments show that LLM-Barber efficiently prunes models from LLaMA and OPT families (7B to 13B) on a single A100 GPU in just 30 minutes, achieving state-of-the-art results in both perplexity and zero-shot performance across various language benchmarks. Code is available at https://github.com/YupengSu/LLM-Barber.
>
---
#### [replaced 065] Detection of Adverse Drug Events in Dutch clinical free text documents using Transformer Models: benchmark study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19396v2](http://arxiv.org/pdf/2507.19396v2)**

> **作者:** Rachel M. Murphy; Nishant Mishra; Nicolette F. de Keizer; Dave A. Dongelmans; Kitty J. Jager; Ameen Abu-Hanna; Joanna E. Klopotowska; Iacer Calixto
>
> **备注:** 30 Pages, 5 Figures (Main Paper), 19 Pages, 2 Figures(Supplements). Rachel M. Murphy and Nishant Mishra are shared first authors. Joanna E. Klopotowska and Iacer Calixto are shared last authors
>
> **摘要:** In this study, we establish a benchmark for adverse drug event (ADE) detection in Dutch clinical free-text documents using several transformer models, clinical scenarios, and fit-for-purpose performance measures. We trained a Bidirectional Long Short-Term Memory (Bi-LSTM) model and four transformer-based Dutch and/or multilingual encoder models (BERTje, RobBERT, MedRoBERTa(.)nl, and NuNER) for the tasks of named entity recognition (NER) and relation classification (RC) using 102 richly annotated Dutch ICU clinical progress notes. Anonymized free-text clinical progress notes of patients admitted to the intensive care unit (ICU) of one academic hospital and discharge letters of patients admitted to Internal Medicine wards of two non-academic hospitals were reused. We evaluated our ADE RC models internally using the gold standard (two-step task) and predicted entities (end-to-end task). In addition, all models were externally validated for detecting ADEs at the document level. We report both micro- and macro-averaged F1 scores, given the dataset imbalance in ADEs. Although differences for the ADE RC task between the models were small, MedRoBERTa(.)nl was the best performing model with a macro-averaged F1 score of 0.63 using the gold standard and 0.62 using predicted entities. The MedRoBERTa(.)nl models also performed the best in our external validation and achieved a recall of between 0.67 to 0.74 using predicted entities, meaning between 67 to 74% of discharge letters with ADEs were detected. Our benchmark study presents a robust and clinically meaningful approach for evaluating language models for ADE detection in clinical free-text documents. Our study highlights the need to use appropriate performance measures fit for the task of ADE detection in clinical free-text documents and envisioned future clinical use.
>
---
#### [replaced 066] What is Wrong with Perplexity for Long-context Language Modeling?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23771v5](http://arxiv.org/pdf/2410.23771v5)**

> **作者:** Lizhe Fang; Yifei Wang; Zhaoyang Liu; Chenheng Zhang; Stefanie Jegelka; Jinyang Gao; Bolin Ding; Yisen Wang
>
> **摘要:** Handling long-context inputs is crucial for large language models (LLMs) in tasks such as extended conversations, document summarization, and many-shot in-context learning. While recent approaches have extended the context windows of LLMs and employed perplexity (PPL) as a standard evaluation metric, PPL has proven unreliable for assessing long-context capabilities. The underlying cause of this limitation has remained unclear. In this work, we provide a comprehensive explanation for this issue. We find that PPL overlooks key tokens, which are essential for long-context understanding, by averaging across all tokens and thereby obscuring the true performance of models in long-context scenarios. To address this, we propose \textbf{LongPPL}, a novel metric that focuses on key tokens by employing a long-short context contrastive method to identify them. Our experiments demonstrate that LongPPL strongly correlates with performance on various long-context benchmarks (e.g., Pearson correlation of -0.96), significantly outperforming traditional PPL in predictive accuracy. Additionally, we introduce \textbf{LongCE} (Long-context Cross-Entropy) loss, a re-weighting strategy for fine-tuning that prioritizes key tokens, leading to consistent improvements across diverse benchmarks. In summary, these contributions offer deeper insights into the limitations of PPL and present effective solutions for accurately evaluating and enhancing the long-context capabilities of LLMs. Code is available at https://github.com/PKU-ML/LongPPL.
>
---
#### [replaced 067] FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18573v2](http://arxiv.org/pdf/2502.18573v2)**

> **作者:** Radu Marinescu; Debarun Bhattacharjya; Junkyu Lee; Tigran Tchrakian; Javier Carnerero Cano; Yufang Hou; Elizabeth Daly; Alessandra Pascale
>
> **摘要:** Large language models (LLMs) have demonstrated vast capabilities on generative tasks in recent years, yet they struggle with guaranteeing the factual correctness of the generated content. This makes these models unreliable in realistic situations where factually accurate responses are expected. In this paper, we propose FactReasoner, a new factuality assessor that relies on probabilistic reasoning to assess the factuality of a long-form generated response. Specifically, FactReasoner decomposes the response into atomic units, retrieves relevant contexts for them from an external knowledge source, and constructs a joint probability distribution over the atoms and contexts using probabilistic encodings of the logical relationships (entailment, contradiction) between the textual utterances corresponding to the atoms and contexts. FactReasoner then computes the posterior probability of whether atomic units in the response are supported by the retrieved contexts. Our experiments on labeled and unlabeled benchmark datasets demonstrate clearly that FactReasoner improves considerably over state-of-the-art prompt-based approaches in terms of both factual precision and recall.
>
---
#### [replaced 068] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04036v3](http://arxiv.org/pdf/2503.04036v3)**

> **作者:** Xinyue Cui; Johnny Tian-Zheng Wei; Swabha Swayamdipta; Robin Jia
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Data watermarking in language models injects traceable signals, such as specific token sequences or stylistic patterns, into copyrighted text, allowing copyright holders to track and verify training data ownership. Previous data watermarking techniques primarily focus on effective memorization during pretraining, while overlooking challenges that arise in other stages of the LLM lifecycle, such as the risk of watermark filtering during data preprocessing and verification difficulties due to API-only access. To address these challenges, we propose a novel data watermarking approach that injects plausible yet fictitious knowledge into training data using generated passages describing a fictitious entity and its associated attributes. Our watermarks are designed to be memorized by the LLM through seamlessly integrating in its training data, making them harder to detect lexically during preprocessing. We demonstrate that our watermarks can be effectively memorized by LLMs, and that increasing our watermarks' density, length, and diversity of attributes strengthens their memorization. We further show that our watermarks remain effective after continual pretraining and supervised finetuning. Finally, we show that our data watermarks can be evaluated even under API-only access via question answering.
>
---
#### [replaced 069] Language Models Resist Alignment: Evidence From Data Compression
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.06144v5](http://arxiv.org/pdf/2406.06144v5)**

> **作者:** Jiaming Ji; Kaile Wang; Tianyi Qiu; Boyuan Chen; Jiayi Zhou; Changye Li; Hantao Lou; Juntao Dai; Yunhuai Liu; Yaodong Yang
>
> **备注:** Accepted by ACL2025 Main
>
> **摘要:** Large language models (LLMs) may exhibit unintended or undesirable behaviors. Recent works have concentrated on aligning LLMs to mitigate harmful outputs. Despite these efforts, some anomalies indicate that even a well-conducted alignment process can be easily circumvented, whether intentionally or accidentally. Does alignment fine-tuning yield have robust effects on models, or are its impacts merely superficial? In this work, we make the first exploration of this phenomenon from both theoretical and empirical perspectives. Empirically, we demonstrate the $\mathbf{elasticity}$ of post-alignment models, i.e., the tendency to revert to the behavior distribution formed during the pre-training phase upon further fine-tuning. Leveraging compression theory, we formally deduce that fine-tuning disproportionately undermines alignment relative to pre-training, potentially by orders of magnitude. We validate the presence of elasticity through experiments on models of varying types and scales. Specifically, we find that model performance declines rapidly before reverting to the pre-training distribution, after which the rate of decline drops significantly. Furthermore, we further reveal that elasticity positively correlates with the increased model size and the expansion of pre-training data. Our findings underscore the need to address the inherent elasticity of LLMs to mitigate their resistance to alignment. The model weight and code are available at pku-lm-resist-alignment.github.io.
>
---
#### [replaced 070] TN-AutoRCA: Benchmark Construction and Agentic Framework for Self-Improving Alarm-Based Root Cause Analysis in Telecommunication Networks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18190v2](http://arxiv.org/pdf/2507.18190v2)**

> **作者:** Keyu Wu; Qianjin Yu; Manlin Mei; Ruiting Liu; Jun Wang; Kailai Zhang; Yelun Bao
>
> **备注:** 10 pages
>
> **摘要:** Root Cause Analysis (RCA) in telecommunication networks is a critical task, yet it presents a formidable challenge for Artificial Intelligence (AI) due to its complex, graph-based reasoning requirements and the scarcity of realistic benchmarks.
>
---
#### [replaced 071] Large Language Models Are Human-Like Internally
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01615v2](http://arxiv.org/pdf/2502.01615v2)**

> **作者:** Tatsuki Kuribayashi; Yohei Oseki; Souhaib Ben Taieb; Kentaro Inui; Timothy Baldwin
>
> **备注:** This is a pre-MIT Press publication version of the paper
>
> **摘要:** Recent cognitive modeling studies have reported that larger language models (LMs) exhibit a poorer fit to human reading behavior (Oh and Schuler, 2023b; Shain et al., 2024; Kuribayashi et al., 2024), leading to claims of their cognitive implausibility. In this paper, we revisit this argument through the lens of mechanistic interpretability and argue that prior conclusions were skewed by an exclusive focus on the final layers of LMs. Our analysis reveals that next-word probabilities derived from internal layers of larger LMs align with human sentence processing data as well as, or better than, those from smaller LMs. This alignment holds consistently across behavioral (self-paced reading times, gaze durations, MAZE task processing times) and neurophysiological (N400 brain potentials) measures, challenging earlier mixed results and suggesting that the cognitive plausibility of larger LMs has been underestimated. Furthermore, we first identify an intriguing relationship between LM layers and human measures: earlier layers correspond more closely with fast gaze durations, while later layers better align with relatively slower signals such as N400 potentials and MAZE processing times. Our work opens new avenues for interdisciplinary research at the intersection of mechanistic interpretability and cognitive modeling.
>
---
#### [replaced 072] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.15846v3](http://arxiv.org/pdf/2507.15846v3)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [replaced 073] The Impact of LoRA Adapters on LLMs for Clinical Text Classification Under Computational and Data Constraints
- **分类: cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.19299v3](http://arxiv.org/pdf/2407.19299v3)**

> **作者:** Thanh-Dung Le; Ti Ti Nguyen; Vu Nguyen Ha; Symeon Chatzinotas; Philippe Jouvet; Rita Noumeir
>
> **备注:** Accepted for publication in the IEEE Access
>
> **摘要:** Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to domain gap, limited data, and stringent hardware constraints. In this study, we evaluate four adapter techniques-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN) - equivalent to Low-Rank Adaptation (LoRA), for clinical note classification under real-world, resource-constrained conditions. All experiments were conducted on a single NVIDIA Quadro P620 GPU (2 GB VRAM, 512 CUDA cores, 1.386 TFLOPS FP32), limiting batch sizes to <8 sequences and maximum sequence length to 256 tokens. Our clinical corpus comprises only 580 000 tokens, several orders of magnitude smaller than standard LLM pre-training datasets. We fine-tuned three biomedical pre-trained LLMs (CamemBERT-bio, AliBERT, DrBERT) and two lightweight Transformer models trained from scratch. Results show that 1) adapter structures provide no consistent gains when fine-tuning biomedical LLMs under these constraints, and 2) simpler Transformers, with minimal parameter counts and training times under six hours, outperform adapter-augmented LLMs, which required over 1000 GPU-hours. Among adapters, GRN achieved the best metrics (accuracy, precision, recall, F1 = 0.88). These findings demonstrate that, in low-resource clinical settings with limited data and compute, lightweight Transformers trained from scratch offer a more practical and efficient solution than large LLMs, while GRN remains a viable adapter choice when minimal adaptation is needed.
>
---
#### [replaced 074] Self-Regularization with Sparse Autoencoders for Controllable LLM-based Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14133v3](http://arxiv.org/pdf/2502.14133v3)**

> **作者:** Xuansheng Wu; Wenhao Yu; Xiaoming Zhai; Ninghao Liu
>
> **备注:** Accepted by SIGKDD 2025
>
> **摘要:** Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impact of these unintended features on classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed self-regularization framework can improve the classifier's generalizability by regularizing those features that are not semantically correlated to the task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. The code and data are publicly available at https://github.com/JacksonWuxs/Controllable_LLM_Classifier.
>
---
#### [replaced 075] LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15606v3](http://arxiv.org/pdf/2506.15606v3)**

> **作者:** Gabriel J. Perin; Runjin Chen; Xuxi Chen; Nina S. T. Hirata; Zhangyang Wang; Junyuan Hong
>
> **摘要:** Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
>
---
#### [replaced 076] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v4](http://arxiv.org/pdf/2507.11936v4)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 077] Benchmarking Linguistic Diversity of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10271v2](http://arxiv.org/pdf/2412.10271v2)**

> **作者:** Yanzhu Guo; Guokan Shang; Chloé Clavel
>
> **摘要:** The development and evaluation of Large Language Models (LLMs) has primarily focused on their task-solving capabilities, with recent models even surpassing human performance in some areas. However, this focus often neglects whether machine-generated language matches the human level of diversity, in terms of vocabulary choice, syntactic construction, and expression of meaning, raising questions about whether the fundamentals of language generation have been fully addressed. This paper emphasizes the importance of examining the preservation of human linguistic richness by language models, given the concerning surge in online content produced or aided by LLMs. We propose a comprehensive framework for evaluating LLMs from various linguistic diversity perspectives including lexical, syntactic, and semantic dimensions. Using this framework, we benchmark several state-of-the-art LLMs across all diversity dimensions, and conduct an in-depth case study for syntactic diversity. Finally, we analyze how different development and deployment choices impact the linguistic diversity of LLM outputs.
>
---
#### [replaced 078] Automating Mathematical Proof Generation Using Large Language Model Agents and Knowledge Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11657v2](http://arxiv.org/pdf/2503.11657v2)**

> **作者:** Vincent Li; Tim Knappe; Yule Fu; Kevin Han; Kevin Zhu
>
> **备注:** Accepted to ICML AI4Math Workshop 2025, NAACL SRW 2025
>
> **摘要:** Large language models have demonstrated remarkable capabilities in natural language processing tasks requiring multi-step logical reasoning capabilities, such as automated theorem proving. However, challenges persist within theorem proving, such as the identification of key mathematical concepts, understanding their interrelationships, and formalizing proofs correctly within natural language. We present KG-prover, a novel framework that leverages knowledge graphs mined from reputable mathematical texts to augment general-purpose LLMs to construct and formalize mathematical proofs. We also study the effects of scaling graph-based, test-time compute using KG-Prover, demonstrating significant performance improvements over baselines across multiple datasets. General-purpose LLMs improve up to 21\% on miniF2F-test when combined with KG-Prover, with consistent improvements ranging from 2-11\% on the ProofNet, miniF2F-test, and MUSTARD datasets without additional scaling. Furthermore, KG-Prover with o4-mini achieves over 50% miniF2F-test. This work provides a promising approach for augmenting natural language proof reasoning with knowledge graphs without the need for additional finetuning.
>
---
#### [replaced 079] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15586v3](http://arxiv.org/pdf/2507.15586v3)**

> **作者:** Xinping Zhao; Shouzheng Huang; Yan Zhong; Xinshuo Hu; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 16 pages, 7 Figures, 10 Tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly impact the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous methods extract evidence straightforwardly without explicit thinking, which risks filtering out key clues and struggles with generalization. To this end, we propose LEAR, which learns to extract rational evidence by (1) explicitly reasoning to identify potential cues within retrieval contents first, and then (2) consciously extracting to avoid omitting any key cues helpful for answering questions. Specifically, we frame evidence reasoning and evidence extraction into one unified response for end-to-end training; apply knowledge token masks for disentanglement to derive reasoning-based and extraction-based answers; and devise three types of verifiable reward functions, including answer, length, and format, to update the model via the policy optimization algorithm. Extensive experiments on three benchmark datasets show the effectiveness of LEAR, providing compact and high-quality evidence, improving the accuracy of downstream tasks, and promoting effective application in online RAG systems.
>
---
#### [replaced 080] Summarization of Opinionated Political Documents with Varied Perspectives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.04093v2](http://arxiv.org/pdf/2411.04093v2)**

> **作者:** Nicholas Deas; Kathleen McKeown
>
> **备注:** COLING 2025
>
> **摘要:** Global partisan hostility and polarization has increased, and this polarization is heightened around presidential elections. Models capable of generating accurate summaries of diverse perspectives can help reduce such polarization by exposing users to alternative perspectives. In this work, we introduce a novel dataset and task for independently summarizing each political perspective in a set of passages from opinionated news articles. For this task, we propose a framework for evaluating different dimensions of perspective summary performance. We benchmark 11 summarization models and LLMs of varying sizes and architectures through both automatic and human evaluation. While recent models like GPT-4o perform well on this task, we find that all models struggle to generate summaries that are faithful to the intended perspective. Our analysis of summaries focuses on how extraction behavior is impacted by features of the input documents.
>
---
#### [replaced 081] Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; math.ST; stat.ME; stat.TH**

- **链接: [http://arxiv.org/pdf/2506.09853v2](http://arxiv.org/pdf/2506.09853v2)**

> **作者:** Xiangning Yu; Zhuohan Wang; Linyi Yang; Haoxuan Li; Anjie Liu; Xiao Xue; Jun Wang; Mengyue Yang
>
> **摘要:** Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness.
>
---
#### [replaced 082] Critique of Impure Reason: Unveiling the reasoning behaviour of medical Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.15748v2](http://arxiv.org/pdf/2412.15748v2)**

> **作者:** Shamus Sim; Tyrone Chen
>
> **备注:** 25 pages, 7 figures, 3 tables. Conceptualization, both authors. formal analysis, both authors. funding acquisition, both authors. investigation, both authors. resources, both authors. supervision, T.C.. validation, both authors. visualization, both authors. writing original draft, both authors. writing review and editing, both authors
>
> **摘要:** Background: Despite the current ubiquity of Large Language Models (LLMs) across the medical domain, there is a surprising lack of studies which address their reasoning behaviour. We emphasise the importance of understanding reasoning behaviour as opposed to high-level prediction accuracies, since it is equivalent to explainable AI (XAI) in this context. In particular, achieving XAI in medical LLMs used in the clinical domain will have a significant impact across the healthcare sector. Results: Therefore, in this work, we adapt the existing concept of reasoning behaviour and articulate its interpretation within the specific context of medical LLMs. We survey and categorise current state-of-the-art approaches for modeling and evaluating reasoning reasoning in medical LLMs. Additionally, we propose theoretical frameworks which can empower medical professionals or machine learning engineers to gain insight into the low-level reasoning operations of these previously obscure models. We also outline key open challenges facing the development of Large Reasoning Models. Conclusion: The subsequent increased transparency and trust in medical machine learning models by clinicians as well as patients will accelerate the integration, application as well as further development of medical AI for the healthcare system as a whole.
>
---
#### [replaced 083] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15075v4](http://arxiv.org/pdf/2505.15075v4)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---

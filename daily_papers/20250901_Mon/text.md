# 自然语言处理 cs.CL

- **最新发布 48 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] Mapping Toxic Comments Across Demographics: A Dataset from German Public Broadcasting
- **分类: cs.CL; cs.CY**

- **简介: 该论文构建首个包含年龄标注的德国毒性评论数据集，解决现有数据缺乏人口统计信息的问题，通过人工与LLM标注分析年龄差异，支持年龄意识内容审核系统开发。**

- **链接: [http://arxiv.org/pdf/2508.21084v1](http://arxiv.org/pdf/2508.21084v1)**

> **作者:** Jan Fillies; Michael Peter Hoffmann; Rebecca Reichel; Roman Salzwedel; Sven Bodemer; Adrian Paschke
>
> **备注:** The paper has been accepted to the EMNLP 2025 main track
>
> **摘要:** A lack of demographic context in existing toxic speech datasets limits our understanding of how different age groups communicate online. In collaboration with funk, a German public service content network, this research introduces the first large-scale German dataset annotated for toxicity and enriched with platform-provided age estimates. The dataset includes 3,024 human-annotated and 30,024 LLM-annotated anonymized comments from Instagram, TikTok, and YouTube. To ensure relevance, comments were consolidated using predefined toxic keywords, resulting in 16.7\% labeled as problematic. The annotation pipeline combined human expertise with state-of-the-art language models, identifying key categories such as insults, disinformation, and criticism of broadcasting fees. The dataset reveals age-based differences in toxic speech patterns, with younger users favoring expressive language and older users more often engaging in disinformation and devaluation. This resource provides new opportunities for studying linguistic variation across demographics and supports the development of more equitable and age-aware content moderation systems.
>
---
#### [new 002] Going over Fine Web with a Fine-Tooth Comb: Technical Report of Indexing Fine Web for Problematic Content Search and Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出基于ElasticSearch的框架，用于高效索引和分析LLM训练数据（如FineWeb-2），解决数据质量、安全及伦理问题，实现快速查询与有害内容检索，提升AI系统安全性与可问责性。**

- **链接: [http://arxiv.org/pdf/2508.21788v1](http://arxiv.org/pdf/2508.21788v1)**

> **作者:** Inés Altemir Marinas; Anastasiia Kucherenko; Andrei Kucharavy
>
> **摘要:** Large language models (LLMs) rely heavily on web-scale datasets like Common Crawl, which provides over 80\% of training data for some modern models. However, the indiscriminate nature of web crawling raises challenges in data quality, safety, and ethics. Despite the critical importance of training data quality, prior research on harmful content has been limited to small samples due to computational constraints. This project presents a framework for indexing and analyzing LLM training datasets using an ElasticSearch-based pipeline. We apply it to SwissAI's FineWeb-2 corpus (1.5TB, four languages), achieving fast query performance--most searches in milliseconds, all under 2 seconds. Our work demonstrates real-time dataset analysis, offering practical tools for safer, more accountable AI systems.
>
---
#### [new 003] HSFN: Hierarchical Selection for Fake News Detection building Heterogeneous Ensemble
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出HSFN方法，针对虚假新闻检测任务，解决集成学习中分类器多样性不足问题。通过分层聚类与层次选择策略，优化异构分类器组合，提升检测准确率。**

- **链接: [http://arxiv.org/pdf/2508.21482v1](http://arxiv.org/pdf/2508.21482v1)**

> **作者:** Sara B. Coutinho; Rafael M. O. Cruz; Francimaria R. S. Nascimento; George D. C. Cavalcanti
>
> **备注:** Accepted by IEEE International Conference on Systems, Man, and Cybernetics (SMC) - IEEE SMC 2025
>
> **摘要:** Psychological biases, such as confirmation bias, make individuals particularly vulnerable to believing and spreading fake news on social media, leading to significant consequences in domains such as public health and politics. Machine learning-based fact-checking systems have been widely studied to mitigate this problem. Among them, ensemble methods are particularly effective in combining multiple classifiers to improve robustness. However, their performance heavily depends on the diversity of the constituent classifiers-selecting genuinely diverse models remains a key challenge, especially when models tend to learn redundant patterns. In this work, we propose a novel automatic classifier selection approach that prioritizes diversity, also extended by performance. The method first computes pairwise diversity between classifiers and applies hierarchical clustering to organize them into groups at different levels of granularity. A HierarchySelect then explores these hierarchical levels to select one pool of classifiers per level, each representing a distinct intra-pool diversity. The most diverse pool is identified and selected for ensemble construction from these. The selection process incorporates an evaluation metric reflecting each classifiers's performance to ensure the ensemble also generalises well. We conduct experiments with 40 heterogeneous classifiers across six datasets from different application domains and with varying numbers of classes. Our method is compared against the Elbow heuristic and state-of-the-art baselines. Results show that our approach achieves the highest accuracy on two of six datasets. The implementation details are available on the project's repository: https://github.com/SaraBCoutinho/HSFN .
>
---
#### [new 004] A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers
- **分类: cs.CL; cs.AI**

- **简介: 该综述系统梳理科学大语言模型（Sci-LLMs）的发展，分析科学数据复杂性带来的挑战，提出统一数据分类和跨模态知识模型，评估基准数据集，探讨半自动标注与专家验证等解决方案，并倡导闭环系统推动科学发现。**

- **链接: [http://arxiv.org/pdf/2508.21148v1](http://arxiv.org/pdf/2508.21148v1)**

> **作者:** Ming Hu; Chenglong Ma; Wei Li; Wanghan Xu; Jiamin Wu; Jucheng Hu; Tianbin Li; Guohang Zhuang; Jiaqi Liu; Yingzhou Lu; Ying Chen; Chaoyang Zhang; Cheng Tan; Jie Ying; Guocheng Wu; Shujian Gao; Pengcheng Chen; Jiashi Lin; Haitao Wu; Lulu Chen; Fengxiang Wang; Yuanyuan Zhang; Xiangyu Zhao; Feilong Tang; Encheng Su; Junzhi Ning; Xinyao Liu; Ye Du; Changkai Ji; Cheng Tang; Huihui Xu; Ziyang Chen; Ziyan Huang; Jiyao Liu; Pengfei Jiang; Yizhou Wang; Chen Tang; Jianyu Wu; Yuchen Ren; Siyuan Yan; Zhonghua Wang; Zhongxing Xu; Shiyan Su; Shangquan Sun; Runkai Zhao; Zhisheng Zhang; Yu Liu; Fudi Wang; Yuanfeng Ji; Yanzhou Su; Hongming Shan; Chunmei Feng; Jiahao Xu; Jiangtao Yan; Wenhao Tang; Diping Song; Lihao Liu; Yanyan Huang; Lequan Yu; Bin Fu; Shujun Wang; Xiaomeng Li; Xiaowei Hu; Yun Gu; Ben Fei; Zhongying Deng; Benyou Wang; Yuewen Cao; Minjie Shen; Haodong Duan; Jie Xu; Yirong Chen; Fang Yan; Hongxia Hao; Jielan Li; Jiajun Du; Yanbo Wang; Imran Razzak; Chi Zhang; Lijun Wu; Conghui He; Zhaohui Lu; Jinhai Huang; Yihao Liu; Fenghua Ling; Yuqiang Li; Aoran Wang; Qihao Zheng; Nanqing Dong; Tianfan Fu; Dongzhan Zhou; Yan Lu; Wenlong Zhang; Jin Ye; Jianfei Cai; Wanli Ouyang; Yu Qiao; Zongyuan Ge; Shixiang Tang; Junjun He; Chunfeng Song; Lei Bai; Bowen Zhou
>
> **摘要:** Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery.
>
---
#### [new 005] AllSummedUp: un framework open-source pour comparer les metriques d'evaluation de resume
- **分类: cs.CL; cs.AI**

- **简介: 论文提出AllSummedUp开源框架，比较六种摘要评估指标，揭示文献性能与实际实验的差异，指出高人类一致性指标的计算成本与不稳定性，并倡导更稳健的评估协议以提升自动摘要评估的可靠性。**

- **链接: [http://arxiv.org/pdf/2508.21389v1](http://arxiv.org/pdf/2508.21389v1)**

> **作者:** Tanguy Herserant; Vincent Guigue
>
> **备注:** in French language
>
> **摘要:** This paper investigates reproducibility challenges in automatic text summarization evaluation. Based on experiments conducted across six representative metrics ranging from classical approaches like ROUGE to recent LLM-based methods (G-Eval, SEval-Ex), we highlight significant discrepancies between reported performances in the literature and those observed in our experimental setting. We introduce a unified, open-source framework, applied to the SummEval dataset and designed to support fair and transparent comparison of evaluation metrics. Our results reveal a structural trade-off: metrics with the highest alignment with human judgments tend to be computationally intensive and less stable across runs. Beyond comparative analysis, this study highlights key concerns about relying on LLMs for evaluation, stressing their randomness, technical dependencies, and limited reproducibility. We advocate for more robust evaluation protocols including exhaustive documentation and methodological standardization to ensure greater reliability in automatic summarization assessment.
>
---
#### [new 006] Challenges and Applications of Large Language Models: A Comparison of GPT and DeepSeek family of models
- **分类: cs.CL; cs.AI; cs.LG; 68T50, 68T07; I.2.7; I.2.6; H.3.3**

- **简介: 该论文通过对比GPT与DeepSeek模型，综述LLMs的16项挑战，分析闭源与开源模型的权衡，探讨其在不同领域的应用适配性，旨在指导LLMs的开发与选型。**

- **链接: [http://arxiv.org/pdf/2508.21377v1](http://arxiv.org/pdf/2508.21377v1)**

> **作者:** Shubham Sharma; Sneha Tuli; Narendra Badam
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) are transforming AI across industries, but their development and deployment remain complex. This survey reviews 16 key challenges in building and using LLMs and examines how these challenges are addressed by two state-of-the-art models with unique approaches: OpenAI's closed source GPT-4o (May 2024 update) and DeepSeek-V3-0324 (March 2025), a large open source Mixture-of-Experts model. Through this comparison, we showcase the trade-offs between closed source models (robust safety, fine-tuned reliability) and open source models (efficiency, adaptability). We also explore LLM applications across different domains (from chatbots and coding tools to healthcare and education), highlighting which model attributes are best suited for each use case. This article aims to guide AI researchers, developers, and decision-makers in understanding current LLM capabilities, limitations, and best practices.
>
---
#### [new 007] Quantifying Label-Induced Bias in Large Language Model Self- and Cross-Evaluations
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在自评与交叉评估中因标签诱导产生的偏差问题。通过对比真实与虚假标签对模型评分的影响，揭示标签对判断的显著扭曲作用，提出需采用盲评或多模型评估以确保公平性。**

- **链接: [http://arxiv.org/pdf/2508.21164v1](http://arxiv.org/pdf/2508.21164v1)**

> **作者:** Muskan Saraf; Sajjad Rezvani Boroujeni; Justin Beaudry; Hossein Abedi; Tom Bush
>
> **摘要:** Large language models (LLMs) are increasingly used to evaluate outputs, yet their judgments may be influenced. This study examines bias in self- and cross-model evaluations by ChatGPT, Gemini, and Claude under four conditions: no labels, true labels, and two false-label scenarios. Blog posts authored by each model were evaluated by all three using both overall preference voting and quality ratings for Coherence, Informativeness, and Conciseness, with all scores expressed as percentages for direct comparison. Results reveal striking asymmetries: the "Claude" label consistently boosts scores, while the "Gemini" label consistently depresses them, regardless of actual content. False labels frequently reversed rankings, producing shifts of up to 50 percentage points in preference votes and up to 12 percentage points in converted quality ratings. Gemini's self-scores collapsed under true labels, while Claude's self-preference intensified. These findings show that perceived model identity can heavily distort high-level judgments and subtly influence detailed quality ratings, underscoring the need for blind or multimodel evaluation protocols to ensure fairness in LLM benchmarking.
>
---
#### [new 008] Not All Parameters Are Created Equal: Smart Isolation Boosts Fine-Tuning Performance
- **分类: cs.CL**

- **简介: 论文提出CPI-FT框架，通过识别并隔离任务核心参数，结合参数融合与冻结策略，解决多任务微调中的任务干扰与灾难性遗忘问题。**

- **链接: [http://arxiv.org/pdf/2508.21741v1](http://arxiv.org/pdf/2508.21741v1)**

> **作者:** Yao Wang; Di Liang; Minlong Peng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Supervised fine-tuning (SFT) is a pivotal approach to adapting large language models (LLMs) for downstream tasks; however, performance often suffers from the ``seesaw phenomenon'', where indiscriminate parameter updates yield progress on certain tasks at the expense of others. To address this challenge, we propose a novel \emph{Core Parameter Isolation Fine-Tuning} (CPI-FT) framework. Specifically, we first independently fine-tune the LLM on each task to identify its core parameter regions by quantifying parameter update magnitudes. Tasks with similar core regions are then grouped based on region overlap, forming clusters for joint modeling. We further introduce a parameter fusion technique: for each task, core parameters from its individually fine-tuned model are directly transplanted into a unified backbone, while non-core parameters from different tasks are smoothly integrated via Spherical Linear Interpolation (SLERP), mitigating destructive interference. A lightweight, pipelined SFT training phase using mixed-task data is subsequently employed, while freezing core regions from prior tasks to prevent catastrophic forgetting. Extensive experiments on multiple public benchmarks demonstrate that our approach significantly alleviates task interference and forgetting, consistently outperforming vanilla multi-task and multi-stage fine-tuning baselines.
>
---
#### [new 009] Improving Aviation Safety Analysis: Automated HFACS Classification Using Reinforcement Learning with Group Relative Policy Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出自动化HFACS分类框架，利用GRPO优化LLM解决传统方法可扩展性与一致性问题，通过合成数据提升准确率，优于GPT-5等模型，并提出新评估指标。**

- **链接: [http://arxiv.org/pdf/2508.21201v1](http://arxiv.org/pdf/2508.21201v1)**

> **作者:** Arash Ahmadi; Sarah Sharif; Yaser Banad
>
> **摘要:** Analyzing the human factors behind aviation accidents is crucial for preventing future incidents, yet traditional methods using the Human Factors Analysis and Classification System (HFACS) are limited by scalability and consistency. To address this, we introduce an automated HFACS classification framework for aviation safety analysis that utilizes Reinforcement Learning with Group Relative Policy Optimization (GRPO) to fine-tune a Llama-3.1 8B language model. Our approach incorporates a multi-component reward system tailored for aviation safety analysis and integrates synthetic data generation to overcome class imbalance in accident datasets. The resulting GRPO-optimized model achieved noticeable performance gains, including a 350% increase in exact match accuracy (from 0.0400 to 0.1800) and an improved partial match accuracy of 0.8800. Significantly, our specialized model outperforms state-of-the-art LLMs (Large Language Models), including GPT-5-mini and Gemini-2.5-fiash, on key metrics. This research also proposes exact match accuracy in multi-label HFACS classification problem as a new benchmarking methodology to evaluate the advanced reasoning capabilities of language models. Ultimately, our work validates that smaller, domain-optimized models can provide a computationally efficient and better solution for critical safety analysis. This approach makes powerful, low-latency deployment on resource-constrained edge devices feasible.
>
---
#### [new 010] Med-RewardBench: Benchmarking Reward Models and Judges for Medical Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Med-RewardBench基准，专门评估医疗多模态大模型的奖励模型与法官，解决现有基准忽视临床准确性与相关性的不足，通过专家标注数据与多维度评估，分析32个模型表现并开发改进基线模型。**

- **链接: [http://arxiv.org/pdf/2508.21430v1](http://arxiv.org/pdf/2508.21430v1)**

> **作者:** Meidan Ding; Jipeng Zhang; Wenxuan Wang; Cheng-Yi Li; Wei-Chieh Fang; Hsin-Yu Wu; Haiqin Zhong; Wenting Chen; Linlin Shen
>
> **备注:** 19 pages, 5 figures, 3 tables
>
> **摘要:** Multimodal large language models (MLLMs) hold significant potential in medical applications, including disease diagnosis and clinical decision-making. However, these tasks require highly accurate, context-sensitive, and professionally aligned responses, making reliable reward models and judges critical. Despite their importance, medical reward models (MRMs) and judges remain underexplored, with no dedicated benchmarks addressing clinical requirements. Existing benchmarks focus on general MLLM capabilities or evaluate models as solvers, neglecting essential evaluation dimensions like diagnostic accuracy and clinical relevance. To address this, we introduce Med-RewardBench, the first benchmark specifically designed to evaluate MRMs and judges in medical scenarios. Med-RewardBench features a multimodal dataset spanning 13 organ systems and 8 clinical departments, with 1,026 expert-annotated cases. A rigorous three-step process ensures high-quality evaluation data across six clinically critical dimensions. We evaluate 32 state-of-the-art MLLMs, including open-source, proprietary, and medical-specific models, revealing substantial challenges in aligning outputs with expert judgment. Additionally, we develop baseline models that demonstrate substantial performance improvements through fine-tuning.
>
---
#### [new 011] Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Middo框架，通过闭环动态数据优化解决LLM微调中静态数据质量不足的问题，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.21589v1](http://arxiv.org/pdf/2508.21589v1)**

> **作者:** Zinan Tang; Xin Gao; Qizhi Pei; Zhuoshi Pan; Mengzhang Cai; Jiang Wu; Conghui He; Lijun Wu
>
> **备注:** Accepted by EMNLP 2025 (main)
>
> **摘要:** Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon.
>
---
#### [new 012] Normality and the Turing Test
- **分类: cs.CL; cs.AI**

- **简介: 论文通过“正常性”概念重新分析图灵测试，论证其应基于统计平均的正常智能而非异常表现，指出大型语言模型因追求例外智能而难以通过测试，挑战图灵测试对人类认知的理解基础。**

- **链接: [http://arxiv.org/pdf/2508.21382v1](http://arxiv.org/pdf/2508.21382v1)**

> **作者:** Alexandre Kabbach
>
> **摘要:** This paper proposes to revisit the Turing test through the concept of normality. Its core argument is that the statistical interpretation of the normal--understood as the average both in the normative and mathematical sense of the term--proves useful for understanding the Turing test in at least two ways. First, in the sense that the Turing test targets normal/average rather than exceptional human intelligence, so that successfully passing the test requires building machines that "make mistakes" and display imperfect behavior just like normal/average humans. Second, in the sense that the Turing test is a statistical test where judgments of intelligence are never carried out by a single "average" judge (understood as non-expert) but always by a full jury. As such, the notion of "average human interrogator" that Turing talks about in his original paper should be understood primarily as referring to a mathematical abstraction made of the normalized aggregate of individual judgments of multiple judges. In short, this paper argues that the Turing test is a test of normal intelligence as assessed by a normal judge characterizing the average judgment of a pool of human interrogators. Its conclusions are twofold. First, it argues that large language models such as ChatGPT are unlikely to pass the Turing test as those models precisely target exceptional rather than normal/average human intelligence. As such, they constitute models of what it proposes to call artificial smartness rather than artificial intelligence per se. Second, it argues that the core question of whether the Turing test can contribute anything to the understanding of human cognition is that of whether the human mind is really reducible to the normal/average mind--a question which largely extends beyond the Turing test itself and questions the conceptual underpinnings of the normalist paradigm it belongs to.
>
---
#### [new 013] A Survey on Current Trends and Recent Advances in Text Anonymization
- **分类: cs.CL; cs.AI**

- **简介: 该论文为文本匿名化领域的综述，旨在解决隐私保护与数据可用性平衡问题。通过总结基础方法、LLM影响、领域挑战及评估框架，指导未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.21587v1](http://arxiv.org/pdf/2508.21587v1)**

> **作者:** Tobias Deußer; Lorenz Sparrenberg; Armin Berger; Max Hahnbück; Christian Bauckhage; Rafet Sifa
>
> **备注:** Accepted at IEEE DSAA 2025
>
> **摘要:** The proliferation of textual data containing sensitive personal information across various domains requires robust anonymization techniques to protect privacy and comply with regulations, while preserving data usability for diverse and crucial downstream tasks. This survey provides a comprehensive overview of current trends and recent advances in text anonymization techniques. We begin by discussing foundational approaches, primarily centered on Named Entity Recognition, before examining the transformative impact of Large Language Models, detailing their dual role as sophisticated anonymizers and potent de-anonymization threats. The survey further explores domain-specific challenges and tailored solutions in critical sectors such as healthcare, law, finance, and education. We investigate advanced methodologies incorporating formal privacy models and risk-aware frameworks, and address the specialized subfield of authorship anonymization. Additionally, we review evaluation frameworks, comprehensive metrics, benchmarks, and practical toolkits for real-world deployment of anonymization solutions. This review consolidates current knowledge, identifies emerging trends and persistent challenges, including the evolving privacy-utility trade-off, the need to address quasi-identifiers, and the implications of LLM capabilities, and aims to guide future research directions for both academics and practitioners in this field.
>
---
#### [new 014] Is this chart lying to me? Automating the detection of misleading visualizations
- **分类: cs.CL; cs.CV; cs.GR**

- **简介: 论文提出Misviz和Misviz-synth数据集，用于自动检测误导性图表，解决数据不足问题，评估多模型性能，发现任务挑战性，推动该领域发展。**

- **链接: [http://arxiv.org/pdf/2508.21675v1](http://arxiv.org/pdf/2508.21675v1)**

> **作者:** Jonathan Tonglet; Jan Zimny; Tinne Tuytelaars; Iryna Gurevych
>
> **备注:** Preprint under review. Code and data available at: https://github.com/UKPLab/arxiv2025-misviz
>
> **摘要:** Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also release Misviz-synth, a synthetic dataset of 81,814 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and fine-tuned classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code.
>
---
#### [new 015] Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework
- **分类: cs.CL**

- **简介: 论文提出反事实评估框架，测试自动审稿系统检测研究逻辑错误的能力。发现逻辑错误不影响其输出，提出改进建议并公开数据集。**

- **链接: [http://arxiv.org/pdf/2508.21422v1](http://arxiv.org/pdf/2508.21422v1)**

> **作者:** Nils Dycke; Iryna Gurevych
>
> **摘要:** Large Language Models (LLMs) have great potential to accelerate and support scholarly peer review and are increasingly used as fully automatic review generators (ARGs). However, potential biases and systematic errors may pose significant risks to scientific integrity; understanding the specific capabilities and limitations of state-of-the-art ARGs is essential. We focus on a core reviewing skill that underpins high-quality peer review: detecting faulty research logic. This involves evaluating the internal consistency between a paper's results, interpretations, and claims. We present a fully automated counterfactual evaluation framework that isolates and tests this skill under controlled conditions. Testing a range of ARG approaches, we find that, contrary to expectation, flaws in research logic have no significant effect on their output reviews. Based on our findings, we derive three actionable recommendations for future work and release our counterfactual dataset and evaluation framework publicly.
>
---
#### [new 016] Personality Matters: User Traits Predict LLM Preferences in Multi-Turn Collaborative Tasks
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究用户性格如何影响其在多轮协作任务中对LLM的偏好，通过实验分析不同性格类型对GPT-4和Claude 3.5的偏好差异，揭示传统评估忽略的模型特性差异。**

- **链接: [http://arxiv.org/pdf/2508.21628v1](http://arxiv.org/pdf/2508.21628v1)**

> **作者:** Sarfaroz Yunusov; Kaige Chen; Kazi Nishat Anwar; Ali Emami
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** As Large Language Models (LLMs) increasingly integrate into everyday workflows, where users shape outcomes through multi-turn collaboration, a critical question emerges: do users with different personality traits systematically prefer certain LLMs over others? We conducted a study with 32 participants evenly distributed across four Keirsey personality types, evaluating their interactions with GPT-4 and Claude 3.5 across four collaborative tasks: data analysis, creative writing, information retrieval, and writing assistance. Results revealed significant personality-driven preferences: Rationals strongly preferred GPT-4, particularly for goal-oriented tasks, while idealists favored Claude 3.5, especially for creative and analytical tasks. Other personality types showed task-dependent preferences. Sentiment analysis of qualitative feedback confirmed these patterns. Notably, aggregate helpfulness ratings were similar across models, showing how personality-based analysis reveals LLM differences that traditional evaluations miss.
>
---
#### [new 017] TrInk: Ink Generation with Transformer Network
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TrInk模型，基于Transformer实现手写体生成，解决文本到笔画点的对齐与风格一致性问题。通过改进交叉注意力机制和评估体系，显著降低字符与词错误率。**

- **链接: [http://arxiv.org/pdf/2508.21098v1](http://arxiv.org/pdf/2508.21098v1)**

> **作者:** Zezhong Jin; Shubhang Desai; Xu Chen; Biyi Fang; Zhuoyi Huang; Zhe Li; Chong-Xin Gan; Xiao Tu; Man-Wai Mak; Yan Lu; Shujie Liu
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** In this paper, we propose TrInk, a Transformer-based model for ink generation, which effectively captures global dependencies. To better facilitate the alignment between the input text and generated stroke points, we introduce scaled positional embeddings and a Gaussian memory mask in the cross-attention module. Additionally, we design both subjective and objective evaluation pipelines to comprehensively assess the legibility and style consistency of the generated handwriting. Experiments demonstrate that our Transformer-based model achieves a 35.56\% reduction in character error rate (CER) and an 29.66% reduction in word error rate (WER) on the IAM-OnDB dataset compared to previous methods. We provide an demo page with handwriting samples from TrInk and baseline models at: https://akahello-a11y.github.io/trink-demo/
>
---
#### [new 018] Reasoning-Intensive Regression
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出推理密集型回归（RiR）任务，解决文本中细微数值属性推断问题。针对有限数据与计算资源，提出MENTAT方法，结合提示优化与神经集成学习，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2508.21762v1](http://arxiv.org/pdf/2508.21762v1)**

> **作者:** Diane Tchuindjo; Omar Khattab
>
> **摘要:** AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e. deducing subtle numerical properties from text. Unlike standard language regression tasks, e.g. for sentiment or similarity, RiR often appears instead in ad-hoc problems like rubric-based scoring or domain-specific retrieval, where much deeper analysis of text is required while only limited task-specific training data and computation are available. We cast three realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR.
>
---
#### [new 019] Enhancing Robustness of Autoregressive Language Models against Orthographic Attacks via Pixel-based Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于像素的生成语言模型，通过将文本转为图像嵌入，解决自回归模型对多语言拼写攻击的脆弱性，提升鲁棒性及多语言兼容性。**

- **链接: [http://arxiv.org/pdf/2508.21206v1](http://arxiv.org/pdf/2508.21206v1)**

> **作者:** Han Yang; Jian Lan; Yihong Liu; Hinrich Schütze; Thomas Seidl
>
> **摘要:** Autoregressive language models are vulnerable to orthographic attacks, where input text is perturbed with characters from multilingual alphabets, leading to substantial performance degradation. This vulnerability primarily stems from the out-of-vocabulary issue inherent in subword tokenizers and their embeddings. To address this limitation, we propose a pixel-based generative language model that replaces the text-based embeddings with pixel-based representations by rendering words as individual images. This design provides stronger robustness to noisy inputs, while an extension of compatibility to multilingual text across diverse writing systems. We evaluate the proposed method on the multilingual LAMBADA dataset, WMT24 dataset and the SST-2 benchmark, demonstrating both its resilience to orthographic noise and its effectiveness in multilingual settings.
>
---
#### [new 020] L3Cube-MahaSTS: A Marathi Sentence Similarity Dataset and Models
- **分类: cs.CL; cs.LG**

- **简介: 论文提出MahaSTS数据集及MahaSBERT-STS-v2模型，解决马拉地语句子相似度任务中的数据与模型短缺问题，通过人工标注和结构化监督提升低资源场景下的性能。**

- **链接: [http://arxiv.org/pdf/2508.21569v1](http://arxiv.org/pdf/2508.21569v1)**

> **作者:** Aishwarya Mirashi; Ananya Joshi; Raviraj Joshi
>
> **摘要:** We present MahaSTS, a human-annotated Sentence Textual Similarity (STS) dataset for Marathi, along with MahaSBERT-STS-v2, a fine-tuned Sentence-BERT model optimized for regression-based similarity scoring. The MahaSTS dataset consists of 16,860 Marathi sentence pairs labeled with continuous similarity scores in the range of 0-5. To ensure balanced supervision, the dataset is uniformly distributed across six score-based buckets spanning the full 0-5 range, thus reducing label bias and enhancing model stability. We fine-tune the MahaSBERT model on this dataset and benchmark its performance against other alternatives like MahaBERT, MuRIL, IndicBERT, and IndicSBERT. Our experiments demonstrate that MahaSTS enables effective training for sentence similarity tasks in Marathi, highlighting the impact of human-curated annotations, targeted fine-tuning, and structured supervision in low-resource settings. The dataset and model are publicly shared at https://github.com/l3cube-pune/MarathiNLP
>
---
#### [new 021] CoBA: Counterbias Text Augmentation for Mitigating Various Spurious Correlations via Semantic Triples
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CoBA框架，通过语义三元组分解与修改生成对抗性数据，减少模型对错误相关性的依赖，提升下游任务性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.21083v1](http://arxiv.org/pdf/2508.21083v1)**

> **作者:** Kyohoon Jin; Juhwan Choi; Jungmin Yun; Junho Lee; Soojin Jang; Youngbin Kim
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Deep learning models often learn and exploit spurious correlations in training data, using these non-target features to inform their predictions. Such reliance leads to performance degradation and poor generalization on unseen data. To address these limitations, we introduce a more general form of counterfactual data augmentation, termed counterbias data augmentation, which simultaneously tackles multiple biases (e.g., gender bias, simplicity bias) and enhances out-of-distribution robustness. We present CoBA: CounterBias Augmentation, a unified framework that operates at the semantic triple level: first decomposing text into subject-predicate-object triples, then selectively modifying these triples to disrupt spurious correlations. By reconstructing the text from these adjusted triples, CoBA generates counterbias data that mitigates spurious patterns. Through extensive experiments, we demonstrate that CoBA not only improves downstream task performance, but also effectively reduces biases and strengthens out-of-distribution resilience, offering a versatile and robust solution to the challenges posed by spurious correlations.
>
---
#### [new 022] PiCSAR: Probabilistic Confidence Selection And Ranking
- **分类: cs.CL; cs.AI**

- **简介: 论文提出PiCSAR方法，用于提升大模型推理任务的准确性。通过联合对数似然分解推理与答案信心，无需真实答案即可评分候选解，减少样本量并提升性能。**

- **链接: [http://arxiv.org/pdf/2508.21787v1](http://arxiv.org/pdf/2508.21787v1)**

> **作者:** Joshua Ong Jun Leang; Zheng Zhao; Aryo Pradipta Gema; Sohee Yang; Wai-Chung Kwan; Xuanli He; Wenda Li; Pasquale Minervini; Eleonora Giunchiglia; Shay B. Cohen
>
> **摘要:** Best-of-n sampling improves the accuracy of large language models (LLMs) and large reasoning models (LRMs) by generating multiple candidate solutions and selecting the one with the highest reward. The key challenge for reasoning tasks is designing a scoring function that can identify correct reasoning chains without access to ground-truth answers. We propose Probabilistic Confidence Selection And Ranking (PiCSAR): a simple, training-free method that scores each candidate generation using the joint log-likelihood of the reasoning and final answer. The joint log-likelihood of the reasoning and final answer naturally decomposes into reasoning confidence and answer confidence. PiCSAR achieves substantial gains across diverse benchmarks (+10.18 on MATH500, +9.81 on AIME2025), outperforming baselines with at least 2x fewer samples in 16 out of 20 comparisons. Our analysis reveals that correct reasoning chains exhibit significantly higher reasoning and answer confidence, justifying the effectiveness of PiCSAR.
>
---
#### [new 023] Granite Embedding R2 Models
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出Granite Embedding R2模型，针对企业密集检索任务，解决长文本、多领域检索的效率与准确性问题。通过扩展上下文长度、优化架构（含22层/12层模型）及数据治理，实现19-44%的速度提升和SOTA性能，开源支持企业级部署。**

- **链接: [http://arxiv.org/pdf/2508.21085v1](http://arxiv.org/pdf/2508.21085v1)**

> **作者:** Parul Awasthy; Aashka Trivedi; Yulong Li; Meet Doshi; Riyaz Bhat; Vignesh P; Vishwajeet Kumar; Yushu Yang; Bhavani Iyer; Abraham Daniels; Rudra Murthy; Ken Barker; Martin Franz; Madison Lee; Todd Ward; Salim Roukos; David Cox; Luis Lastras; Jaydeep Sen; Radu Florian
>
> **摘要:** We introduce the Granite Embedding R2 models, a comprehensive family of high-performance English encoder-based embedding models engineered for enterprise-scale dense retrieval applications. Building upon our first-generation release, these models deliver substantial improvements, including 16x expanded context length (8,192 tokens), state-of-the-art performance across diverse retrieval domains - text, code, long-document search, multi-turn conversational, and tabular data - and measurable speed advantages of 19-44\% over leading competitors while maintaining superior accuracy. Our release encompasses both bi-encoder and cross-encoder architectures, featuring a highly effective 22-layer retriever model and its efficient 12-layer counterpart, alongside a high-quality reranker model, all trained exclusively on enterprise-appropriate data with comprehensive governance oversight. The models demonstrate exceptional versatility across standard benchmarks, IBM-developed evaluation suites, and real-world enterprise use cases, establishing new performance standards for open-source embedding models. In an era where retrieval speed and accuracy are paramount for competitive advantage, the Granite R2 models deliver a compelling combination of cutting-edge performance, enterprise-ready licensing, and transparent data provenance that organizations require for mission-critical deployments. All models are publicly available under the Apache 2.0 license at https://huggingface.co/collections/ibm-granite, enabling unrestricted research and commercial use.
>
---
#### [new 024] How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations
- **分类: cs.CL**

- **简介: 该论文研究锚定效应对LLM价格谈判的影响，通过实验评估其表现，分析推理与个性因素，发现推理模型更少受锚定效应影响，为LLM安全应用提供依据。**

- **链接: [http://arxiv.org/pdf/2508.21137v1](http://arxiv.org/pdf/2508.21137v1)**

> **作者:** Yoshiki Takenami; Yin Jou Huang; Yugo Murawaki; Chenhui Chu
>
> **备注:** work in progress
>
> **摘要:** Cognitive biases, well-studied in humans, can also be observed in LLMs, affecting their reliability in real-world applications. This paper investigates the anchoring effect in LLM-driven price negotiations. To this end, we instructed seller LLM agents to apply the anchoring effect and evaluated negotiations using not only an objective metric but also a subjective metric. Experimental results show that LLMs are influenced by the anchoring effect like humans. Additionally, we investigated the relationship between the anchoring effect and factors such as reasoning and personality. It was shown that reasoning models are less prone to the anchoring effect, suggesting that the long chain of thought mitigates the effect. However, we found no significant correlation between personality traits and susceptibility to the anchoring effect. These findings contribute to a deeper understanding of cognitive biases in LLMs and to the realization of safe and responsible application of LLMs in society.
>
---
#### [new 025] Beyond the Surface: Probing the Ideological Depth of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究LLM的意识形态深度，解决表面响应易操控性与深层立场稳定性的问题。通过测量可操控性与稀疏自编码器分析，发现模型意识形态深度可量化，干预核心特征可引发系统性推理变化。**

- **链接: [http://arxiv.org/pdf/2508.21448v1](http://arxiv.org/pdf/2508.21448v1)**

> **作者:** Shariar Kabir; Kevin Esterling; Yue Dong
>
> **摘要:** Large Language Models (LLMs) have demonstrated pronounced ideological leanings, yet the stability and depth of these positions remain poorly understood. Surface-level responses can often be manipulated through simple prompt engineering, calling into question whether they reflect a coherent underlying ideology. This paper investigates the concept of "ideological depth" in LLMs, defined as the robustness and complexity of their internal political representations. We employ a dual approach: first, we measure the "steerability" of two well-known open-source LLMs using instruction prompting and activation steering. We find that while some models can easily switch between liberal and conservative viewpoints, others exhibit resistance or an increased rate of refusal, suggesting a more entrenched ideological structure. Second, we probe the internal mechanisms of these models using Sparse Autoencoders (SAEs). Preliminary analysis reveals that models with lower steerability possess more distinct and abstract ideological features. Our evaluations reveal that one model can contain 7.3x more political features than another model of similar size. This allows targeted ablation of a core political feature in an ideologically "deep" model, leading to consistent, logical shifts in its reasoning across related topics, whereas the same intervention in a "shallow" model results in an increase in refusal outputs. Our findings suggest that ideological depth is a quantifiable property of LLMs and that steerability serves as a valuable window into their latent political architecture.
>
---
#### [new 026] BLUEX Revisited: Enhancing Benchmark Coverage with Automatic Captioning
- **分类: cs.CL; cs.AI**

- **简介: 该论文更新BLUEX数据集，通过自动图像标题生成增强多语言评估基准，解决LLM预训练数据污染问题，提升文本模型可访问性，并评估模型利用视觉上下文的能力。**

- **链接: [http://arxiv.org/pdf/2508.21294v1](http://arxiv.org/pdf/2508.21294v1)**

> **作者:** João Guilherme Alves Santos; Giovana Kerche Bonás; Thales Sales Almeida
>
> **备注:** 12 pages, 5 figures, 2 tables
>
> **摘要:** With the growing capabilities of Large Language Models (LLMs), there is an increasing need for robust evaluation methods, especially in multilingual and non-English contexts. We present an updated version of the BLUEX dataset, now including 2024-2025 exams and automatically generated image captions using state-of-the-art models, enhancing its relevance for data contamination studies in LLM pretraining. Captioning strategies increase accessibility to text-only models by more than 40%, producing 1,422 usable questions, more than doubling the number in the original BLUEX. We evaluated commercial and open-source LLMs and their ability to leverage visual context through captions.
>
---
#### [new 027] Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在提升小语言模型的创意写作能力，针对现有方法计算成本高、创意不足的问题，提出两种AI反馈强化学习策略，其中基于原则的LLM-as-a-Judge方法在生成质量、效率和数据依赖性上表现更优。**

- **链接: [http://arxiv.org/pdf/2508.21476v1](http://arxiv.org/pdf/2508.21476v1)**

> **作者:** Xiaolong Wei; Bo Lu; Xingyu Zhang; Zhejun Zhao; Dongdong Shen; Long Xia; Dawei Yin
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at https://github.com/weixiaolong94-hub/Igniting-Creative-Writing-in-Small-Language-Models.
>
---
#### [new 028] Do Self-Supervised Speech Models Exhibit the Critical Period Effects in Language Acquisition?
- **分类: cs.CL**

- **简介: 该论文通过对比实验研究自监督语音模型是否呈现语言关键期效应，训练不同L2起始和L1结束时间的模型，评估语音辨别能力，发现S3Ms未表现出关键期效应，反而延迟L2起始提升L2表现，延迟L1结束导致L1遗忘。**

- **链接: [http://arxiv.org/pdf/2508.21210v1](http://arxiv.org/pdf/2508.21210v1)**

> **作者:** Yurie Koga; Shunsuke Kando; Yusuke Miyao
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** This paper investigates whether the Critical Period (CP) effects in human language acquisition are observed in self-supervised speech models (S3Ms). CP effects refer to greater difficulty in acquiring a second language (L2) with delayed L2 exposure onset, and greater retention of their first language (L1) with delayed L1 exposure offset. While previous work has studied these effects using textual language models, their presence in speech models remains underexplored despite the central role of spoken language in human language acquisition. We train S3Ms with varying L2 training onsets and L1 training offsets on child-directed speech and evaluate their phone discrimination performance. We find that S3Ms do not exhibit clear evidence of either CP effects in terms of phonological acquisition. Notably, models with delayed L2 exposure onset tend to perform better on L2 and delayed L1 exposure offset leads to L1 forgetting.
>
---
#### [new 029] Efficient Code Embeddings from Code Generation Models
- **分类: cs.CL; cs.AI; cs.IR; 68T50; I.2.7**

- **简介: 该论文提出基于代码生成模型的高效代码嵌入方法，解决多语言代码检索、技术问答及语义相似性识别问题，通过自回归模型预训练与最后token池化生成嵌入，验证小模型的高性能。**

- **链接: [http://arxiv.org/pdf/2508.21290v1](http://arxiv.org/pdf/2508.21290v1)**

> **作者:** Daria Kryvosheieva; Saba Sturua; Michael Günther; Scott Martens; Han Xiao
>
> **备注:** 9 pages, table and evaluations 5-9
>
> **摘要:** jina-code-embeddings is a novel code embedding model suite designed to retrieve code from natural language queries, perform technical question-answering, and identify semantically similar code snippets across programming languages. It makes innovative use of an autoregressive backbone pre-trained on both text and code, generating embeddings via last-token pooling. We outline the training recipe and demonstrate state-of-the-art performance despite the relatively small size of the models, validating this approach to code embedding model construction.
>
---
#### [new 030] QZhou-Embedding Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出QZhou-Embedding模型，基于Qwen2.5-7B-Instruct，通过多任务框架、数据合成与两阶段训练提升文本嵌入质量，在MTEB/CMTEB等基准上取得SOTA，解决检索、聚类等任务。**

- **链接: [http://arxiv.org/pdf/2508.21632v1](http://arxiv.org/pdf/2508.21632v1)**

> **作者:** Peng Yu; En Xu; Bin Chen; Haibiao Chen; Yinfei Xu
>
> **摘要:** We present QZhou-Embedding, a general-purpose contextual text embedding model with exceptional text representation capabilities. Built upon the Qwen2.5-7B-Instruct foundation model, we designed a unified multi-task framework comprising specialized data transformation and training strategies. The data transformation scheme enables the incorporation of more diverse textual training datasets, while the task-specific training strategies enhance model learning efficiency. We developed a data synthesis pipeline leveraging LLM API, incorporating techniques such as paraphrasing, augmentation, and hard negative example generation to improve the semantic richness and sample difficulty of the training set. Additionally, we employ a two-stage training strategy, comprising initial retrieval-focused pretraining followed by full-task fine-tuning, enabling the embedding model to extend its capabilities based on robust retrieval performance. Our model achieves state-of-the-art results on the MTEB and CMTEB benchmarks, ranking first on both leaderboards (August 27 2025), and simultaneously achieves state-of-the-art performance on tasks including reranking, clustering, etc. Our findings demonstrate that higher-quality, more diverse data is crucial for advancing retrieval model performance, and that leveraging LLMs generative capabilities can further optimize data quality for embedding model breakthroughs. Our model weights are released on HuggingFace under Apache 2.0 license. For reproducibility, we provide evaluation code and instructions on GitHub.
>
---
#### [new 031] Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DMP管道，通过消除自一致性方法中的冗余生成，解决大模型幻觉检测中计算成本高的问题，实现3倍速度提升且保持检测性能。**

- **链接: [http://arxiv.org/pdf/2508.21228v1](http://arxiv.org/pdf/2508.21228v1)**

> **作者:** Weizhi Gao; Xiaorui Liu; Feiyi Wang; Dan Lu; Junqi Yin
>
> **备注:** 14 pages, under review
>
> **摘要:** Large language models (LLMs) have demonstrated impressive performance in both research and real-world applications, but they still struggle with hallucination. Existing hallucination detection methods often perform poorly on sentence-level generation or rely heavily on domain-specific knowledge. While self-consistency approaches help address these limitations, they incur high computational costs due to repeated generation. In this paper, we conduct the first study on identifying redundancy in self-consistency methods, manifested as shared prefix tokens across generations, and observe that non-exact-answer tokens contribute minimally to the semantic content. Based on these insights, we propose a novel Decoding Memory Pipeline (DMP) that accelerates generation through selective inference and annealed decoding. Being orthogonal to the model, dataset, decoding strategy, and self-consistency baseline, our DMP consistently improves the efficiency of multi-response generation and holds promise for extension to alignment and reasoning tasks. Extensive experiments show that our method achieves up to a 3x speedup without sacrificing AUROC performance.
>
---
#### [new 032] BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文提出BED-LLM框架，通过贝叶斯实验设计优化LLM在多轮对话中的信息获取能力，解决传统方法适应性差的问题。创新点包括EIG估计器、非上下文依赖的条件策略及查询生成方法，实验验证其在用户偏好推断等任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.21184v1](http://arxiv.org/pdf/2508.21184v1)**

> **作者:** Deepro Choudhury; Sinead Williamson; Adam Goliński; Ning Miao; Freddie Bickford Smith; Michael Kirchhof; Yizhe Zhang; Tom Rainforth
>
> **摘要:** We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
>
---
#### [new 033] Can Multimodal LLMs Solve the Basic Perception Problems of Percept-V?
- **分类: cs.CL; cs.CV**

- **简介: 该论文评估多模态大语言模型（MLLMs）在基础视觉感知任务中的表现，解决其在简单感知任务上的性能评估问题。作者构建了Percept-V数据集，测试GPT-4o、Gemini等模型，发现模型性能随任务复杂度显著下降。**

- **链接: [http://arxiv.org/pdf/2508.21143v1](http://arxiv.org/pdf/2508.21143v1)**

> **作者:** Samrajnee Ghosh; Naman Agarwal; Hemanshu Garg; Chinmay Mittal; Mausam; Parag Singla
>
> **摘要:** The reasoning abilities of Multimodal Large Language Models (MLLMs) have garnered a lot of attention in recent times, with advances made in frontiers like coding, mathematics, and science. However, very limited experiments have been done to assess their performance in simple perception tasks performed over uncontaminated, generated images containing basic shapes and structures. To address this issue, the paper introduces a dataset, Percept-V, containing a total of 7200 program-generated images equally divided into 30 categories, each testing a combination of visual perception skills. Unlike previously proposed datasets, Percept-V comprises very basic tasks of varying complexity that test the perception abilities of MLLMs. This dataset is then tested on state-of-the-art MLLMs like GPT-4o, Gemini, and Claude as well as Large Reasoning Models (LRMs) like OpenAI o4-mini and DeepSeek R1 to gauge their performance. Contrary to the evidence that MLLMs excel in many complex tasks, our experiments show a significant drop in the models' performance with increasing problem complexity across all categories. An analysis of the performances also reveals that the tested MLLMs exhibit a similar trend in accuracy across categories, testing a particular cognitive skill and find some skills to be more difficult than others.
>
---
#### [new 034] Discovering Semantic Subdimensions through Disentangled Conceptual Representations
- **分类: cs.CL**

- **简介: 该论文提出DCSRM模型，通过解耦词嵌入识别细粒度语义子维度，验证其神经合理性，解决现有方法语义表示过于宽泛的问题。**

- **链接: [http://arxiv.org/pdf/2508.21436v1](http://arxiv.org/pdf/2508.21436v1)**

> **作者:** Yunhao Zhang; Shaonan Wang; Nan Lin; Xinyi Dong; Chong Li; Chengqing Zong
>
> **摘要:** Understanding the core dimensions of conceptual semantics is fundamental to uncovering how meaning is organized in language and the brain. Existing approaches often rely on predefined semantic dimensions that offer only broad representations, overlooking finer conceptual distinctions. This paper proposes a novel framework to investigate the subdimensions underlying coarse-grained semantic dimensions. Specifically, we introduce a Disentangled Continuous Semantic Representation Model (DCSRM) that decomposes word embeddings from large language models into multiple sub-embeddings, each encoding specific semantic information. Using these sub-embeddings, we identify a set of interpretable semantic subdimensions. To assess their neural plausibility, we apply voxel-wise encoding models to map these subdimensions to brain activation. Our work offers more fine-grained interpretable semantic subdimensions of conceptual meaning. Further analyses reveal that semantic dimensions are structured according to distinct principles, with polarity emerging as a key factor driving their decomposition into subdimensions. The neural correlates of the identified subdimensions support their cognitive and neuroscientific plausibility.
>
---
#### [new 035] Quantum-Enhanced Natural Language Generation: A Multi-Model Framework with Hybrid Quantum-Classical Architectures
- **分类: quant-ph; cs.CL; cs.LG**

- **简介: 论文评估量子文本生成模型与传统Transformer/MLP架构，比较五种模型在多种数据集上的表现，分析其在不同指标下的性能差异，探讨量子模型在特定场景下的竞争力。**

- **链接: [http://arxiv.org/pdf/2508.21332v1](http://arxiv.org/pdf/2508.21332v1)**

> **作者:** Chi-Sheng Chen; En-Jui Kuo
>
> **摘要:** This paper presents a comprehensive evaluation of quantum text generation models against traditional Transformer/MLP architectures, addressing the growing interest in quantum computing applications for natural language processing. We conduct systematic experiments comparing five distinct models: Transformer (baseline), Quantum Kernel Self-Attention Network (QKSAN), Quantum RWKV (QRWKV), and Quantum Attention Sequence Architecture (QASA) across five diverse datasets including simple sentences, short stories, quantum phrases, haiku poetry, and proverbs. Our evaluation employs multiple metrics including perplexity, BLEU scores, vocabulary diversity, repetition rates, and fluency measures to assess different aspects of text generation quality. The experimental results reveal that while traditional Transformer models maintain overall superiority with the lowest average perplexity (1.21) and highest BLEU-1 score (0.2895), quantum-inspired models demonstrate competitive performance in specific scenarios. Notably, QKSAN achieves a competitive BLEU-1 score of 0.2800 while maintaining zero repetition rates, and QRWKV demonstrates perfect vocabulary diversity (Distinct-1 = 1.000) in certain tasks.
>
---
#### [new 036] Why Stop at Words? Unveiling the Bigger Picture through Line-Level OCR
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出行级OCR方法，解决传统字符分割错误和上下文不足问题，构建数据集并验证，提升准确率5.4%和效率4倍。**

- **链接: [http://arxiv.org/pdf/2508.21693v1](http://arxiv.org/pdf/2508.21693v1)**

> **作者:** Shashank Vempati; Nishit Anand; Gaurav Talebailkar; Arpan Garai; Chetan Arora
>
> **备注:** 11 pages. Project Website: https://nishitanand.github.io/line-level-ocr-website
>
> **摘要:** Conventional optical character recognition (OCR) techniques segmented each character and then recognized. This made them prone to error in character segmentation, and devoid of context to exploit language models. Advances in sequence to sequence translation in last decade led to modern techniques first detecting words and then inputting one word at a time to a model to directly output full words as sequence of characters. This allowed better utilization of language models and bypass error-prone character segmentation step. We observe that the above transition in style has moved the bottleneck in accuracy to word segmentation. Hence, in this paper, we propose a natural and logical progression from word level OCR to line-level OCR. The proposal allows to bypass errors in word detection, and provides larger sentence context for better utilization of language models. We show that the proposed technique not only improves the accuracy but also efficiency of OCR. Despite our thorough literature survey, we did not find any public dataset to train and benchmark such shift from word to line-level OCR. Hence, we also contribute a meticulously curated dataset of 251 English page images with line-level annotations. Our experimentation revealed a notable end-to-end accuracy improvement of 5.4%, underscoring the potential benefits of transitioning towards line-level OCR, especially for document images. We also report a 4 times improvement in efficiency compared to word-based pipelines. With continuous improvements in large language models, our methodology also holds potential to exploit such advances. Project Website: https://nishitanand.github.io/line-level-ocr-website
>
---
#### [new 037] Morae: Proactively Pausing UI Agents for User Choices
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文提出Morae系统，解决现有UI代理缺乏用户决策参与的问题。通过自动识别任务中的决策点并暂停，结合大模型解析用户意图与界面信息，让用户明确选择，提升任务完成率与偏好匹配度。**

- **链接: [http://arxiv.org/pdf/2508.21456v1](http://arxiv.org/pdf/2508.21456v1)**

> **作者:** Yi-Hao Peng; Dingzeyu Li; Jeffrey P. Bigham; Amy Pavel
>
> **备注:** ACM UIST 2025
>
> **摘要:** User interface (UI) agents promise to make inaccessible or complex UIs easier to access for blind and low-vision (BLV) users. However, current UI agents typically perform tasks end-to-end without involving users in critical choices or making them aware of important contextual information, thus reducing user agency. For example, in our field study, a BLV participant asked to buy the cheapest available sparkling water, and the agent automatically chose one from several equally priced options, without mentioning alternative products with different flavors or better ratings. To address this problem, we introduce Morae, a UI agent that automatically identifies decision points during task execution and pauses so that users can make choices. Morae uses large multimodal models to interpret user queries alongside UI code and screenshots, and prompt users for clarification when there is a choice to be made. In a study over real-world web tasks with BLV participants, Morae helped users complete more tasks and select options that better matched their preferences, as compared to baseline agents, including OpenAI Operator. More broadly, this work exemplifies a mixed-initiative approach in which users benefit from the automation of UI agents while being able to express their preferences.
>
---
#### [new 038] Summarize-Exemplify-Reflect: Data-driven Insight Distillation Empowers LLMs for Few-shot Tabular Classification
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对少样本表格分类任务中结构数据变异性带来的挑战，提出InsightTab框架，通过数据驱动的洞察蒸馏（规则总结、战略示例、反思学习）提升LLM分类性能，实验证明方法有效且优于现有技术。**

- **链接: [http://arxiv.org/pdf/2508.21561v1](http://arxiv.org/pdf/2508.21561v1)**

> **作者:** Yifei Yuan; Jiatong Li; Weijia Zhang; Mohammad Aliannejadi; Evangelos Kanoulas; Renjun Hu
>
> **备注:** EMNLP 25 Findings
>
> **摘要:** Recent studies show the promise of large language models (LLMs) for few-shot tabular classification but highlight challenges due to the variability in structured data. To address this, we propose distilling data into actionable insights to enable robust and effective classification by LLMs. Drawing inspiration from human learning processes, we introduce InsightTab, an insight distillation framework guided by principles of divide-and-conquer, easy-first, and reflective learning. Our approach integrates rule summarization, strategic exemplification, and insight reflection through deep collaboration between LLMs and data modeling techniques. The obtained insights enable LLMs to better align their general knowledge and capabilities with the particular requirements of specific tabular tasks. We extensively evaluate InsightTab on nine datasets. The results demonstrate consistent improvement over state-of-the-art methods. Ablation studies further validate the principle-guided distillation process, while analyses emphasize InsightTab's effectiveness in leveraging labeled data and managing bias.
>
---
#### [new 039] Model-Task Alignment Drives Distinct RL Outcomes
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究模型-任务对齐对强化学习结果的影响，发现强对齐使反直觉现象（如单例训练效果佳）出现，而标准RL在挑战性任务中更稳健。**

- **链接: [http://arxiv.org/pdf/2508.21188v1](http://arxiv.org/pdf/2508.21188v1)**

> **作者:** Haoze Wu; Cheng Wang; Wenshuo Zhao; Junxian He
>
> **摘要:** Recent advances in applying reinforcement learning (RL) to large language models (LLMs) have led to substantial progress. In particular, a series of remarkable yet often counterintuitive phenomena have been reported in LLMs, exhibiting patterns not typically observed in traditional RL settings. For example, notable claims include that a single training example can match the performance achieved with an entire dataset, that the reward signal does not need to be very accurate, and that training solely with negative samples can match or even surpass sophisticated reward-based methods. However, the precise conditions under which these observations hold - and, critically, when they fail - remain unclear. In this work, we identify a key factor that differentiates RL observations: whether the pretrained model already exhibits strong Model-Task Alignment, as measured by pass@k accuracy on the evaluated task. Through a systematic and comprehensive examination of a series of counterintuitive claims, supported by rigorous experimental validation across different model architectures and task domains, our findings show that while standard RL training remains consistently robust across settings, many of these counterintuitive results arise only when the model and task already exhibit strong model-task alignment. In contrast, these techniques fail to drive substantial learning in more challenging regimes, where standard RL methods remain effective.
>
---
#### [new 040] CrossTL: A Universal Programming Language Translator with Unified Intermediate Representation
- **分类: cs.PL; cs.CL; cs.GR; 68N20, 68N15, 68W10; D.3.4; D.3.2; D.1.3**

- **简介: 该论文提出CrossTL，解决多语言双向翻译复杂度高的问题。通过统一中间表示CrossGL，实现CUDA、Rust等语言间的高效转换，采用模块化架构支持扩展，验证了通用代码翻译的可行性，推动语言无关编程发展。**

- **链接: [http://arxiv.org/pdf/2508.21256v1](http://arxiv.org/pdf/2508.21256v1)**

> **作者:** Nripesh Niketan; Vaatsalya Shrivastva
>
> **备注:** 15 Pages, 5 Figures, 1 Table. Introduces CrossTL, a universal programming language translator enabling bidirectional translation between 8 programming languages (CUDA, HIP, Metal, DirectX HLSL, OpenGL GLSL, Vulkan SPIR-V, Rust, Mojo) through a unified intermediate representation called CrossGL. Includes comprehensive evaluation with complex real-world examples
>
> **摘要:** We present CrossTL, a universal programming language translator enabling bidirectional translation between multiple languages through a unified intermediate representation called CrossGL. Traditional approaches require separate translators for each language pair, leading to exponential complexity growth. CrossTL uses a single universal IR to facilitate translations between CUDA, HIP, Metal, DirectX HLSL, OpenGL GLSL, Vulkan SPIR-V, Rust, and Mojo, with Slang support in development. Our system consists of: language-specific lexers/parsers converting source code to ASTs, bidirectional CrossGL translation modules implementing ToCrossGLConverter classes for importing code and CodeGen classes for target generation, and comprehensive backend implementations handling full translation pipelines. We demonstrate effectiveness through comprehensive evaluation across programming domains, achieving successful compilation and execution across all supported backends. The universal IR design enables adding new languages with minimal effort, requiring only language-specific frontend/backend components. Our contributions include: (1) a unified IR capturing semantics of multiple programming paradigms, (2) a modular architecture enabling extensibility, (3) a comprehensive framework supporting GPU compute, graphics programming, and systems languages, and (4) empirical validation demonstrating practical viability of universal code translation. CrossTL represents a significant step toward language-agnostic programming, enabling write-once, deploy-everywhere development.
>
---
#### [new 041] Database Normalization via Dual-LLM Self-Refinement
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 论文提出Miffie框架，利用双模型自优化实现自动化数据库规范化，解决手动操作耗时易错问题。通过生成与验证模块协同迭代，结合定制提示，实现高准确率且高效处理复杂模式。**

- **链接: [http://arxiv.org/pdf/2508.17693v1](http://arxiv.org/pdf/2508.17693v1)**

> **作者:** Eunjae Jo; Nakyung Lee; Gyuyeong Kim
>
> **备注:** 5 pages
>
> **摘要:** Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy.
>
---
#### [new 042] Normalisation of SWIFT Message Counterparties with Feature Extraction and Clustering
- **分类: cs.LG; cs.CL**

- **简介: 论文提出混合方法（字符串相似性、主题建模、聚类与规则）解决SWIFT交易对手方聚类问题，克服自然语言处理局限，提升识别精度，减少人工审查。**

- **链接: [http://arxiv.org/pdf/2508.21081v1](http://arxiv.org/pdf/2508.21081v1)**

> **作者:** Thanasis Schoinas; Benjamin Guinard; Diba Esbati; Richard Chalk
>
> **摘要:** Short text clustering is a known use case in the text analytics community. When the structure and content falls in the natural language domain e.g. Twitter posts or instant messages, then natural language techniques can be used, provided texts are of sufficient length to allow for use of (pre)trained models to extract meaningful information, such as part-of-speech or topic annotations. However, natural language models are not suitable for clustering transaction counterparties, as they are found in bank payment messaging systems, such as SWIFT. The manually typed tags are typically physical or legal entity details, which lack sentence structure, while containing all the variations and noise that manual entry introduces. This leaves a gap in an investigator or counter-fraud professional's toolset when looking to augment their knowledge of payment flow originator and beneficiary entities and trace funds and assets. A gap that vendors traditionally try to close with fuzzy matching tools. With these considerations in mind, we are proposing a hybrid string similarity, topic modelling, hierarchical clustering and rule-based pipeline to facilitate clustering of transaction counterparties, also catering for unknown number of expected clusters. We are also devising metrics to supplement the evaluation of the approach, based on the well-known measures of precision and recall. Testing on a real-life labelled dataset demonstrates significantly improved performance over a baseline rule-based ('keyword') approach. The approach retains most of the interpretability found in rule-based systems, as the former adds an additional level of cluster refinement to the latter. The resulting workflow reduces the need for manual review. When only a subset of the population needs to be investigated, such as in sanctions investigations, the approach allows for better control of the risks of missing entity variations.
>
---
#### [new 043] Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding
- **分类: cs.AI; cs.CL; I.2.7; I.2.11; I.2.6**

- **简介: 该论文研究通过认知脚手架（符号结构+记忆机制）提升LLM教学能力，对比消融实验验证架构对认知行为的影响，证明架构设计可增强LLM的适应性推理与教学策略。**

- **链接: [http://arxiv.org/pdf/2508.21204v1](http://arxiv.org/pdf/2508.21204v1)**

> **作者:** Vanessa Figueiredo
>
> **摘要:** We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs.
>
---
#### [new 044] From Canonical to Complex: Benchmarking LLM Capabilities in Undergraduate Thermodynamics
- **分类: physics.ed-ph; cs.CL; physics.chem-ph**

- **简介: 该论文开发UTQA基准，评估LLM在热力学教学中的推理能力。针对LLM自主教学的可靠性问题，通过50道题测试其原理理解与图像推理能力，发现当前模型在复杂场景和视觉绑定任务中表现不足，尚不适合独立教学。**

- **链接: [http://arxiv.org/pdf/2508.21452v1](http://arxiv.org/pdf/2508.21452v1)**

> **作者:** Anna Geißler; Luca-Sophie Bien; Friedrich Schöppler; Tobias Hertel
>
> **备注:** Benchmark downloadable at https://huggingface.co/datasets/herteltm/UTQA
>
> **摘要:** Large language models (LLMs) are increasingly considered as tutoring aids in science education. Yet their readiness for unsupervised use in undergraduate instruction remains uncertain, as reliable teaching requires more than fluent recall: it demands consistent, principle-grounded reasoning. Thermodynamics, with its compact laws and subtle distinctions between state and path functions, reversibility, and entropy, provides an ideal testbed for evaluating such capabilities. Here we present UTQA, a 50-item undergraduate thermodynamics question answering benchmark, covering ideal-gas processes, reversibility, and diagram interpretation. No leading 2025-era model exceeded our 95\% competence threshold: the best LLMs achieved 82\% accuracy, with text-only items performing better than image reasoning tasks, which often fell to chance levels. Prompt phrasing and syntactic complexity showed modest to little correlation with performance. The gap concentrates in finite-rate/irreversible scenarios and in binding visual features to thermodynamic meaning, indicating that current LLMs are not yet suitable for unsupervised tutoring in this domain.
>
---
#### [new 045] Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文评估不同表格到文本序列化方法对LLM在贷款审批任务中的性能与公平性影响，比较零样本与上下文学习效果，发现特定格式提升性能但加剧公平差距，强调数据表示对模型可靠性的重要性。**

- **链接: [http://arxiv.org/pdf/2508.21512v1](http://arxiv.org/pdf/2508.21512v1)**

> **作者:** Israel Abebe Azime; Deborah D. Kanubala; Tejumade Afonja; Mario Fritz; Isabel Valera; Dietrich Klakow; Philipp Slusallek
>
> **摘要:** Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making.
>
---
#### [new 046] Designing Smarter Conversational Agents for Kids: Lessons from Cognitive Work and Means-Ends Analyses
- **分类: cs.HC; cs.CL; I.2.1; H.5.2**

- **简介: 论文设计智能儿童对话代理，通过认知工作分析和结构化提示框架，提升儿童与CA的互动效果。研究巴西儿童使用场景，提出分层对话树、个性化档案及家长审核内容，构建首个儿童-CA信息流框架。**

- **链接: [http://arxiv.org/pdf/2508.21209v1](http://arxiv.org/pdf/2508.21209v1)**

> **作者:** Vanessa Figueiredo
>
> **摘要:** This paper presents two studies on how Brazilian children (ages 9--11) use conversational agents (CAs) for schoolwork, discovery, and entertainment, and how structured scaffolds can enhance these interactions. In Study 1, a seven-week online investigation with 23 participants (children, parents, teachers) employed interviews, observations, and Cognitive Work Analysis to map children's information-processing flows, the role of more knowledgeable others, functional uses, contextual goals, and interaction patterns to inform conversation-tree design. We identified three CA functions: School, Discovery, Entertainment, and derived ``recipe'' scaffolds mirroring parent-child support. In Study 2, we prompted GPT-4o-mini on 1,200 simulated child-CA exchanges, comparing conversation-tree recipes based on structured-prompting to an unstructured baseline. Quantitative evaluation of readability, question count/depth/diversity, and coherence revealed gains for the recipe approach. Building on these findings, we offer design recommendations: scaffolded conversation-trees, child-dedicated profiles for personalized context, and caregiver-curated content. Our contributions include the first CWA application with Brazilian children, an empirical framework of child-CA information flows, and an LLM-scaffolding ``recipe'' (i.e., structured-prompting) for effective, scaffolded learning.
>
---
#### [new 047] Stairway to Fairness: Connecting Group and Individual Fairness
- **分类: cs.IR; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究推荐系统中群体公平与个体公平的关系，解决两者评估标准不统一导致的对比困难问题。通过对比分析评估指标，发现高群体公平可能损害个体公平，为实践提供平衡两者的参考。**

- **链接: [http://arxiv.org/pdf/2508.21334v1](http://arxiv.org/pdf/2508.21334v1)**

> **作者:** Theresia Veronika Rampisela; Maria Maistro; Tuukka Ruotsalo; Falk Scholer; Christina Lioma
>
> **备注:** Accepted to RecSys 2025 (short paper)
>
> **摘要:** Fairness in recommender systems (RSs) is commonly categorised into group fairness and individual fairness. However, there is no established scientific understanding of the relationship between the two fairness types, as prior work on both types has used different evaluation measures or evaluation objectives for each fairness type, thereby not allowing for a proper comparison of the two. As a result, it is currently not known how increasing one type of fairness may affect the other. To fill this gap, we study the relationship of group and individual fairness through a comprehensive comparison of evaluation measures that can be used for both fairness types. Our experiments with 8 runs across 3 datasets show that recommendations that are highly fair for groups can be very unfair for individuals. Our finding is novel and useful for RS practitioners aiming to improve the fairness of their systems. Our code is available at: https://github.com/theresiavr/stairway-to-fairness.
>
---
#### [new 048] AHELM: A Holistic Evaluation of Audio-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AHELM基准，整合多数据集和标准化评估，全面评测音频语言模型在10个关键方面（如公平性、安全性等），解决现有评估碎片化问题，测试14个模型并揭示性能差异。**

- **链接: [http://arxiv.org/pdf/2508.21376v1](http://arxiv.org/pdf/2508.21376v1)**

> **作者:** Tony Lee; Haoqin Tu; Chi Heem Wong; Zijun Wang; Siwei Yang; Yifan Mai; Yuyin Zhou; Cihang Xie; Percy Liang
>
> **摘要:** Evaluations of audio-language models (ALMs) -- multimodal models that take interleaved audio and text as input and output text -- are hindered by the lack of standardized benchmarks; most benchmarks measure only one or two capabilities and omit evaluative aspects such as fairness or safety. Furthermore, comparison across models is difficult as separate evaluations test a limited number of models and use different prompting methods and inference parameters. To address these shortfalls, we introduce AHELM, a benchmark that aggregates various datasets -- including 2 new synthetic audio-text datasets called PARADE, which evaluates the ALMs on avoiding stereotypes, and CoRe-Bench, which measures reasoning over conversational audio through inferential multi-turn question answering -- to holistically measure the performance of ALMs across 10 aspects we have identified as important to the development and usage of ALMs: audio perception, knowledge, reasoning, emotion detection, bias, fairness, multilinguality, robustness, toxicity, and safety. We also standardize the prompts, inference parameters, and evaluation metrics to ensure equitable comparisons across models. We test 14 open-weight and closed-API ALMs from 3 developers and 3 additional simple baseline systems each consisting of an automatic speech recognizer and a language model. Our results show that while Gemini 2.5 Pro ranks top in 5 out of 10 aspects, it exhibits group unfairness ($p=0.01$) on ASR tasks whereas most of the other models do not. We also find that the baseline systems perform reasonably well on AHELM, with one ranking 5th overall despite having only speech-to-text capabilities. For transparency, all raw prompts, model generations, and outputs are available on our website at https://crfm.stanford.edu/helm/audio/v1.0.0. AHELM is intended to be a living benchmark and new datasets and models will be added over time.
>
---
## 更新

#### [replaced 001] Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12800v3](http://arxiv.org/pdf/2508.12800v3)**

> **作者:** Yong Deng; Guoqing Wang; Zhenzhe Ying; Xiaofeng Wu; Jinzhen Lin; Wenwen Xiong; Yuqin Dai; Shuo Yang; Zhanwei Zhang; Qiwen Wang; Yang Qin; Yuan Wang; Quanxing Zha; Sunhao Dai; Changhua Meng
>
> **摘要:** Large language models (LLMs) exhibit remarkable problem-solving abilities, but struggle with complex tasks due to static internal knowledge. Retrieval-Augmented Generation (RAG) enhances access to external information, yet remains limited in multi-hop reasoning and strategic search due to rigid workflows. Recent advancements in agentic deep research empower LLMs to autonomously reason, search, and synthesize information. However, current approaches relying on outcome-based reinforcement learning (RL) face critical issues such as conflicting gradients and reward sparsity, limiting performance gains and training efficiency. To address these, we first propose Atomic Thought, a novel LLM thinking paradigm that decomposes reasoning into fine-grained functional units. These units are supervised by Reasoning Reward Models (RRMs), which provide Atomic Thought Rewards (ATR) for fine-grained guidance. Building on this, we propose Atom-Searcher, a novel RL framework for agentic deep research that integrates Atomic Thought and ATR. Atom-Searcher uses a curriculum-inspired reward schedule, prioritizing process-level ATR early and transitioning to outcome rewards, accelerating convergence on effective reasoning paths. Experiments on seven benchmarks show consistent improvements over the state-of-the-art. Key advantages include: (1) Atom-Searcher scales computation at test-time. (2) Atomic Thought provides supervision anchors for RRMs, bridging deep research tasks and RRMs. (3) Atom-Searcher exhibits more interpretable, human-like reasoning patterns.
>
---
#### [replaced 002] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.19828v2](http://arxiv.org/pdf/2508.19828v2)**

> **作者:** Sikuan Yan; Xiufeng Yang; Zuchao Huang; Ercong Nie; Zifeng Ding; Zonggen Li; Xiaowen Ma; Hinrich Schütze; Volker Tresp; Yunpu Ma
>
> **备注:** work in progress
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking any learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns to perform structured memory operations, including adding, updating, deleting, or taking no operation on memory entries; and an Answer Agent that selects the most relevant entries and reasons over them to produce an answer. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management and utilization with minimal supervision. With as few as 152 question-answer pairs and a corresponding temporal memory bank for training, Memory-R1 outperforms the strongest existing baseline and demonstrates strong generalization across diverse question types and LLM backbones. Beyond presenting an effective approach, this work provides insights into how RL can unlock more agentic, memory-aware behavior in LLMs, pointing toward richer, more persistent reasoning systems.
>
---
#### [replaced 003] Uncovering the Bigger Picture: Comprehensive Event Understanding Via Diverse News Retrieval
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.19758v2](http://arxiv.org/pdf/2508.19758v2)**

> **作者:** Yixuan Tang; Yuanyuan Shi; Yiqun Sun; Anthony Kum Hoe Tung
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Access to diverse perspectives is essential for understanding real-world events, yet most news retrieval systems prioritize textual relevance, leading to redundant results and limited viewpoint exposure. We propose NEWSCOPE, a two-stage framework for diverse news retrieval that enhances event coverage by explicitly modeling semantic variation at the sentence level. The first stage retrieves topically relevant content using dense retrieval, while the second stage applies sentence-level clustering and diversity-aware re-ranking to surface complementary information. To evaluate retrieval diversity, we introduce three interpretable metrics, namely Average Pairwise Distance, Positive Cluster Coverage, and Information Density Ratio, and construct two paragraph-level benchmarks: LocalNews and DSGlobal. Experiments show that NEWSCOPE consistently outperforms strong baselines, achieving significantly higher diversity without compromising relevance. Our results demonstrate the effectiveness of fine-grained, interpretable modeling in mitigating redundancy and promoting comprehensive event understanding. The data and code are available at https://github.com/tangyixuan/NEWSCOPE.
>
---
#### [replaced 004] Don't lie to your friends: Learning what you know from collaborative self-play
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14481v3](http://arxiv.org/pdf/2503.14481v3)**

> **作者:** Jacob Eisenstein; Reza Aghajani; Adam Fisch; Dheeru Dua; Fantine Huot; Mirella Lapata; Vicky Zayats; Jonathan Berant
>
> **备注:** CoLM 2025 camera-ready version
>
> **摘要:** To be helpful assistants, AI agents must be aware of their own capabilities and limitations. This includes knowing when to answer from parametric knowledge versus using tools, when to trust tool outputs, and when to abstain or hedge. Such capabilities are hard to teach through supervised fine-tuning because they require constructing examples that reflect the agent's specific capabilities. We therefore propose a radically new approach to teaching agents what they know: \emph{collaborative self-play}. We construct multi-agent collaborations in which the group is rewarded for collectively arriving at correct answers. The desired meta-knowledge emerges from the incentives built into the structure of the interaction. We focus on small societies of agents that have access to heterogeneous tools (corpus-specific retrieval), and therefore must collaborate to maximize their success while minimizing their effort. Experiments show that group-level rewards for multi-agent communities can induce policies that \emph{transfer} to improve tool use and selective prediction in settings where individual agents are deployed in isolation.
>
---
#### [replaced 005] WebInject: Prompt Injection Attack to Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11717v3](http://arxiv.org/pdf/2505.11717v3)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **备注:** EMNLP 2025 main
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.
>
---
#### [replaced 006] TrustGeoGen: Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15780v2](http://arxiv.org/pdf/2504.15780v2)**

> **作者:** Daocheng Fu; Jianlong Chen; Renqiu Xia; Zijun Chen; Qi Liu; Yuan Feng; Hongbin Zhou; Renrui Zhang; Shiyang Feng; Peng Gao; Hongyuan Zha; Junchi Yan; Botian Shi; Yu Qiao; Bo Zhang
>
> **摘要:** Mathematical geometric problem solving (GPS) demands verifiable logical coherence and multimodal reasoning capabilities. While large language models (LLMs) have shown rapid progress in GPS, their advancement is hindered by the lack of reliable benchmarks and systematic methodologies. A critical challenge is the inherent hallucination in LLMs, which leads to synthetic GPS datasets that are often noisy, unverified, and self-contradictory. To address this, we introduce TrustGeoGen, a data engine that generates formally verified geometric problems to establish a principled and trustworthy benchmark. Our engine integrates four key innovations: 1) Multimodal Alignment, which synchronizes the generation of diagrams, text, and step-by-step solutions; 2) Formal Verification, ensuring all reasoning paths are rule-compliant; 3) Connection Thinking, bridging formal deduction with human-like logical steps; and 4) our \textit{GeoExplore} series algorithms, which produce diverse problem variants with multiple solutions and self-reflective backtracking. Using this engine, we create the GeoTrust-200K dataset and the corresponding GeoTrust-test benchmark, both with guaranteed cross-modal integrity. Experiments reveal that state-of-the-art models achieve only 45.83\% accuracy on GeoTrust-test, highlighting its significant challenge. Furthermore, training on our synthesized data substantially improves model performance on GPS tasks, with strong generalization to out-of-domain (OOD) benchmarks. Our code and data are available at https://github.com/Alpha-Innovator/TrustGeoGen
>
---
#### [replaced 007] ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.00631v2](http://arxiv.org/pdf/2412.00631v2)**

> **作者:** Yang Wu; Huayi Zhang; Yizheng Jiao; Lin Ma; Xiaozhong Liu; Jinhong Yu; Dongyu Zhang; Dezhi Yu; Wei Xu
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Instruction tuning has underscored the significant potential of large language models (LLMs) in producing more human controllable and effective outputs in various domains. In this work, we focus on the data selection problem for task-specific instruction tuning of LLMs. Prevailing methods primarily rely on the crafted similarity metrics to select training data that aligns with the test data distribution. The goal is to minimize instruction tuning loss on the test data, ultimately improving performance on the target task. However, it has been widely observed that instruction tuning loss (i.e., cross-entropy loss for next token prediction) in LLMs often fails to exhibit a monotonic relationship with actual task performance. This misalignment undermines the effectiveness of current data selection methods for task-specific instruction tuning. To address this issue, we introduce ROSE, a novel Reward-Oriented inStruction data sElection method which leverages pairwise preference loss as a reward signal to optimize data selection for task-specific instruction tuning. Specifically, ROSE adapts an influence formulation to approximate the influence of training data points relative to a few-shot preference validation set to select the most task-related training data points. Experimental results show that by selecting just 5\% of the training data using ROSE, our approach can achieve competitive results compared to fine-tuning with the full training dataset, and it surpasses other state-of-the-art data selection methods for task-specific instruction tuning. Our qualitative analysis further confirms the robust generalizability of our method across multiple benchmark datasets and diverse model architectures.
>
---
#### [replaced 008] DeepTrans: Deep Reasoning Translation via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10187v2](http://arxiv.org/pdf/2504.10187v2)**

> **作者:** Jiaan Wang; Fandong Meng; Jie Zhou
>
> **备注:** Accepted by Transactions of the Association for Computational Linguistics (TACL)
>
> **摘要:** Recently, deep reasoning LLMs (e.g., OpenAI o1 and DeepSeek-R1) have shown promising performance in various downstream tasks. Free translation is an important and interesting task in the multilingual world, which requires going beyond word-for-word translation. However, the task is still under-explored in deep reasoning LLMs. In this paper, we introduce DeepTrans, a deep reasoning translation model that learns free translation via reinforcement learning (RL). Specifically, we carefully build a reward model with pre-defined scoring criteria on both the translation results and the thought processes. The reward model teaches DeepTrans how to think and free-translate the given sentences during RL. Besides, our RL training does not need any labeled translations, avoiding the human-intensive annotation or resource-intensive data synthesis. Experimental results show the effectiveness of DeepTrans. Using Qwen2.5-7B as the backbone, DeepTrans improves performance by 16.3% in literature translation, and outperforms strong deep reasoning LLMs. Moreover, we summarize the failures and interesting findings during our RL exploration. We hope this work could inspire other researchers in free translation.
>
---
#### [replaced 009] Interpretable Mnemonic Generation for Kanji Learning via Expectation-Maximization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05137v2](http://arxiv.org/pdf/2507.05137v2)**

> **作者:** Jaewook Lee; Alexander Scarlatos; Andrew Lan
>
> **备注:** The Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Learning Japanese vocabulary is a challenge for learners from Roman alphabet backgrounds due to script differences. Japanese combines syllabaries like hiragana with kanji, which are logographic characters of Chinese origin. Kanji are also complicated due to their complexity and volume. Keyword mnemonics are a common strategy to aid memorization, often using the compositional structure of kanji to form vivid associations. Despite recent efforts to use large language models (LLMs) to assist learners, existing methods for LLM-based keyword mnemonic generation function as a black box, offering limited interpretability. We propose a generative framework that explicitly models the mnemonic construction process as driven by a set of common rules, and learn them using a novel Expectation-Maximization-type algorithm. Trained on learner-authored mnemonics from an online platform, our method learns latent structures and compositional rules, enabling interpretable and systematic mnemonics generation. Experiments show that our method performs well in the cold-start setting for new learners while providing insight into the mechanisms behind effective mnemonic creation.
>
---
#### [replaced 010] Toxicity Begets Toxicity: Unraveling Conversational Chains in Political Podcasts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.12640v2](http://arxiv.org/pdf/2501.12640v2)**

> **作者:** Naquee Rizwan; Nayandeep Deb; Sarthak Roy; Vishwajeet Singh Solanki; Kiran Garimella; Animesh Mukherjee
>
> **摘要:** Tackling toxic behavior in digital communication continues to be a pressing concern for both academics and industry professionals. While significant research has explored toxicity on platforms like social networks and discussion boards, podcasts despite their rapid rise in popularity remain relatively understudied in this context. This work seeks to fill that gap by curating a dataset of political podcast transcripts and analyzing them with a focus on conversational structure. Specifically, we investigate how toxicity surfaces and intensifies through sequences of replies within these dialogues, shedding light on the organic patterns by which harmful language can escalate across conversational turns. Warning: Contains potentially abusive/toxic contents.
>
---
#### [replaced 011] BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15689v2](http://arxiv.org/pdf/2506.15689v2)**

> **作者:** Liulu He; Shenli Zheng; Karwei Sun; Yijiang Liu; Yufei Zhao; Chongkang Tan; Huanrui Yang; Yuan Du; Li Du
>
> **摘要:** Rotations have become essential to state-of-the-art quantization pipelines for large language models (LLMs) by effectively smoothing outliers in weights and activations. However, further optimizing the rotation parameters offers only limited performance gains and introduces significant training overhead: due to rotation parameter sharing, full-model must be loaded simultaneously to enable backpropagation, resulting in substantial memory consumption and limited practical utility. In this work, we identify two fundamental limitations of current rotational quantization methods: (i) rotation fails to align channel means, resulting in wider quantization bounds and increased rounding errors; and (ii) rotation makes the activation distribution more Gaussian-like, increasing energy loss caused by clipping errors. To address these issues, we introduce \textbf{BASE-Q}, a simple yet powerful approach that combines bias correction and asymmetric scaling to effectively reduce rounding and clipping errors. Furthermore, BASE-Q enables blockwise optimization, eliminating the need for memory-intensive full-model backpropagation. Extensive experiments on various LLMs and benchmarks demonstrate the effectiveness of BASE-Q, narrowing the accuracy gap to full-precision models by 50.5\%, 42.9\%, and 29.2\% compared to QuaRot, SpinQuant, and OSTQuant, respectively. The code will be released soon.
>
---
#### [replaced 012] Revealing Fine-Grained Values and Opinions in Large Language Models
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.19238v3](http://arxiv.org/pdf/2406.19238v3)**

> **作者:** Dustin Wright; Arnav Arora; Nadav Borenstein; Srishti Yadav; Serge Belongie; Isabelle Augenstein
>
> **备注:** Findings of EMNLP 2024; 28 pages, 20 figures, 7 tables
>
> **摘要:** Uncovering latent values and opinions embedded in large language models (LLMs) can help identify biases and mitigate potential harm. Recently, this has been approached by prompting LLMs with survey questions and quantifying the stances in the outputs towards morally and politically charged statements. However, the stances generated by LLMs can vary greatly depending on how they are prompted, and there are many ways to argue for or against a given position. In this work, we propose to address this by analysing a large and robust dataset of 156k LLM responses to the 62 propositions of the Political Compass Test (PCT) generated by 6 LLMs using 420 prompt variations. We perform coarse-grained analysis of their generated stances and fine-grained analysis of the plain text justifications for those stances. For fine-grained analysis, we propose to identify tropes in the responses: semantically similar phrases that are recurrent and consistent across different prompts, revealing natural patterns in the text that a given LLM is prone to produce. We find that demographic features added to prompts significantly affect outcomes on the PCT, reflecting bias, as well as disparities between the results of tests when eliciting closed-form vs. open domain responses. Additionally, patterns in the plain text rationales via tropes show that similar justifications are repeatedly generated across models and prompts even with disparate stances.
>
---
#### [replaced 013] Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17464v3](http://arxiv.org/pdf/2505.17464v3)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **备注:** Accepted by EMNLP2025 (Main Conference)
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present Hydra, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. Hydra handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, Hydra uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, Hydra fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that Hydra achieves overall state-of-the-art results on all benchmarks with GPT-3.5, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, Hydra enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo. The source code is available on https://stevetantan.github.io/Hydra/.
>
---
#### [replaced 014] UI-Bench: A Benchmark for Evaluating Design Capabilities of AI Text-to-App Tools
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20410v2](http://arxiv.org/pdf/2508.20410v2)**

> **作者:** Sam Jung; Agustin Garcinuno; Spencer Mateega
>
> **摘要:** AI text-to-app tools promise high quality applications and websites in minutes, yet no public benchmark rigorously verifies those claims. We introduce UI-Bench, the first large-scale benchmark that evaluates visual excellence across competing AI text-to-app tools through expert pairwise comparison. Spanning 10 tools, 30 prompts, 300 generated sites, and 4,000+ expert judgments, UI-Bench ranks systems with a TrueSkill-derived model that yields calibrated confidence intervals. UI-Bench establishes a reproducible standard for advancing AI-driven web design. We release (i) the complete prompt set, (ii) an open-source evaluation framework, and (iii) a public leaderboard. The generated sites rated by participants will be released soon. View the UI-Bench leaderboard at https://uibench.ai/leaderboard.
>
---
#### [replaced 015] Strategic resource allocation in memory encoding: An efficiency principle shaping language processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14728v2](http://arxiv.org/pdf/2503.14728v2)**

> **作者:** Weijie Xu; Richard Futrell
>
> **备注:** manuscript under review
>
> **摘要:** How is the limited capacity of working memory efficiently used to support human linguistic behaviors? In this paper, we propose Strategic Resource Allocation (SRA) as an efficiency principle for memory encoding in sentence processing. The idea is that working memory resources are dynamically and strategically allocated to prioritize novel and unexpected information. From a resource-rational perspective, we argue that SRA is the principled solution to a computational problem posed by two functional assumptions about working memory, namely its limited capacity and its noisy representation. Specifically, working memory needs to minimize the retrieval error of past inputs under the constraint of limited memory resources, an optimization problem whose solution is to allocate more resources to encode more surprising inputs with higher precision. One of the critical consequences of SRA is that surprising inputs are encoded with enhanced representations, and therefore are less susceptible to memory decay and interference. Empirically, through naturalistic corpus data, we find converging evidence for SRA in the context of dependency locality from both production and comprehension, where non-local dependencies with less predictable antecedents are associated with reduced locality effect. However, our results also reveal considerable cross-linguistic variability, suggesting the need for a closer examination of how SRA, as a domain-general memory efficiency principle, interacts with language-specific phrase structures. SRA highlights the critical role of representational uncertainty in understanding memory encoding. It also reimages the effects of surprisal and entropy on processing difficulty from the perspective of efficient memory encoding.
>
---
#### [replaced 016] Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.19028v4](http://arxiv.org/pdf/2506.19028v4)**

> **作者:** Weijie Xu; Yiwen Wang; Chi Xue; Xiangkun Hu; Xi Fang; Guimin Dong; Chandan K. Reddy
>
> **备注:** 29 pages, 9 figures, 15 tables
>
> **摘要:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
>
---
#### [replaced 017] Towards Understanding Camera Motions in Any Video
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.15376v2](http://arxiv.org/pdf/2504.15376v2)**

> **作者:** Zhiqiu Lin; Siyuan Cen; Daniel Jiang; Jay Karhade; Hewei Wang; Chancharik Mitra; Tiffany Ling; Yuhan Huang; Sifan Liu; Mingyu Chen; Rushikesh Zawar; Xue Bai; Yilun Du; Chuang Gan; Deva Ramanan
>
> **备注:** Project site: https://linzhiqiu.github.io/papers/camerabench/
>
> **摘要:** We introduce CameraBench, a large-scale dataset and benchmark designed to assess and improve camera motion understanding. CameraBench consists of ~3,000 diverse internet videos, annotated by experts through a rigorous multi-stage quality control process. One of our contributions is a taxonomy of camera motion primitives, designed in collaboration with cinematographers. We find, for example, that some motions like "follow" (or tracking) require understanding scene content like moving subjects. We conduct a large-scale human study to quantify human annotation performance, revealing that domain expertise and tutorial-based training can significantly enhance accuracy. For example, a novice may confuse zoom-in (a change of intrinsics) with translating forward (a change of extrinsics), but can be trained to differentiate the two. Using CameraBench, we evaluate Structure-from-Motion (SfM) and Video-Language Models (VLMs), finding that SfM models struggle to capture semantic primitives that depend on scene content, while VLMs struggle to capture geometric primitives that require precise estimation of trajectories. We then fine-tune a generative VLM on CameraBench to achieve the best of both worlds and showcase its applications, including motion-augmented captioning, video question answering, and video-text retrieval. We hope our taxonomy, benchmark, and tutorials will drive future efforts towards the ultimate goal of understanding camera motions in any video.
>
---
#### [replaced 018] A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreement
- **分类: cs.CL; cs.AI; cs.HC; 68T50 (Primary) 68T37 (Secondary); I.2.7; I.2.1**

- **链接: [http://arxiv.org/pdf/2411.04090v3](http://arxiv.org/pdf/2411.04090v3)**

> **作者:** Guillermo Villate-Castillo; Javier Del Ser; Borja Sanz
>
> **备注:** 35 pages, 1 figure
>
> **摘要:** Content moderation typically combines the efforts of human moderators and machine learning models. However, these systems often rely on data where significant disagreement occurs during moderation, reflecting the subjective nature of toxicity perception. Rather than dismissing this disagreement as noise, we interpret it as a valuable signal that highlights the inherent ambiguity of the content,an insight missed when only the majority label is considered. In this work, we introduce a novel content moderation framework that emphasizes the importance of capturing annotation disagreement. Our approach uses multitask learning, where toxicity classification serves as the primary task and annotation disagreement is addressed as an auxiliary task. Additionally, we leverage uncertainty estimation techniques, specifically Conformal Prediction, to account for both the ambiguity in comment annotations and the model's inherent uncertainty in predicting toxicity and disagreement.The framework also allows moderators to adjust thresholds for annotation disagreement, offering flexibility in determining when ambiguity should trigger a review. We demonstrate that our joint approach enhances model performance, calibration, and uncertainty estimation, while offering greater parameter efficiency and improving the review process in comparison to single-task methods.
>
---
#### [replaced 019] SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17178v3](http://arxiv.org/pdf/2507.17178v3)**

> **作者:** Zhiqiang Liu; Enpei Niu; Yin Hua; Mengshu Sun; Lei Liang; Huajun Chen; Wen Zhang
>
> **备注:** EMNLP 2025
>
> **摘要:** Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/zjukg/SKA-Bench.
>
---
#### [replaced 020] Blind Spot Navigation in Large Language Model Reasoning with Thought Space Explorer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.24155v3](http://arxiv.org/pdf/2410.24155v3)**

> **作者:** Jinghan Zhang; Fengran Mo; Tharindu Cyril Weerasooriya; Yeyang Zhou; Xinyue Ye; Dongjie Wang; Yanjie Fu; Kunpeng Liu
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated their potential in handling complex reasoning tasks, which are usually achieved by constructing a thought chain to guide the model in solving the problem with multi-step thinking. However, existing methods often remain confined to previously explored solution spaces and thus overlook the critical blind spot within LLMs' cognitive range. To address these issues, we introduce the ``Thought Space Explorer'' (TSE), a novel framework to expand and optimize thought structures to guide LLMs to explore their blind spots of thinking. By generating new reasoning steps and branches based on the original thought structure with various designed strategies, TSE broadens the thought exploration view and alleviates the impact of blind spots for LLM reasoning. Experimental results on multiple levels of reasoning tasks demonstrate the efficacy of TSE by surpassing various baseline methods. We also conduct extensive analysis to understand how structured and expansive thought can contribute to unleashing the potential of LLM reasoning capabilities.
>
---
#### [replaced 021] German4All -- A Dataset and Model for Readability-Controlled Paraphrasing in German
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17973v2](http://arxiv.org/pdf/2508.17973v2)**

> **作者:** Miriam Anschütz; Thanh Mai Pham; Eslam Nasrallah; Maximilian Müller; Cristian-George Craciun; Georg Groh
>
> **备注:** Accepted to INLG 2025
>
> **摘要:** The ability to paraphrase texts across different complexity levels is essential for creating accessible texts that can be tailored toward diverse reader groups. Thus, we introduce German4All, the first large-scale German dataset of aligned readability-controlled, paragraph-level paraphrases. It spans five readability levels and comprises over 25,000 samples. The dataset is automatically synthesized using GPT-4 and rigorously evaluated through both human and LLM-based judgments. Using German4All, we train an open-source, readability-controlled paraphrasing model that achieves state-of-the-art performance in German text simplification, enabling more nuanced and reader-specific adaptations. We opensource both the dataset and the model to encourage further research on multi-level paraphrasing
>
---
#### [replaced 022] E2LLM: Encoder Elongated Large Language Models for Long-Context Understanding and Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.06679v2](http://arxiv.org/pdf/2409.06679v2)**

> **作者:** Zihan Liao; Jun Wang; Hang Yu; Lingxiao Wei; Jianguo Li; Jun Wang; Wei Zhang
>
> **备注:** Accept by EMNLP'25
>
> **摘要:** Processing long contexts is increasingly important for Large Language Models (LLMs) in tasks like multi-turn dialogues, code generation, and document summarization. This paper addresses the challenges of achieving high long-context performance, low computational complexity, and compatibility with pretrained models -- collectively termed the ``impossible triangle''. We introduce E2LLM (Encoder Elongated Large Language Models), a novel approach that effectively navigates this paradox. E2LLM divides long contexts into chunks, compresses each into soft prompts using a pretrained text encoder, and aligns these representations with a decoder-only LLM via an adapter. To enhance the LLM's reasoning with these soft prompts, we employ two training objectives: encoder output reconstruction and long-context instruction fine-tuning. Extensive experiments reveal that E2LLM not only outperforms 8 state-of-the-art (SOTA) methods in effectiveness and efficiency for document summarization and question answering, but also achieves the best performance on LongBench v2 among models of comparable size.
>
---
#### [replaced 023] Testing Conviction: An Argumentative Framework for Measuring LLM Political Stability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17052v2](http://arxiv.org/pdf/2504.17052v2)**

> **作者:** Shariar Kabir; Kevin Esterling; Yue Dong
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) increasingly shape political discourse, yet exhibit inconsistent responses when challenged. While prior research categorizes LLMs as left- or right-leaning based on single-prompt responses, a critical question remains: Do these classifications reflect stable ideologies or superficial mimicry? Existing methods cannot distinguish between genuine ideological alignment and performative text generation. To address this, we propose a framework for evaluating ideological depth through (1) argumentative consistency and (2) uncertainty quantification. Testing 12 LLMs on 19 economic policies from the Political Compass Test, we classify responses as stable or performative ideological positioning. Results show 95% of left-leaning models and 89% of right-leaning models demonstrate behavior consistent with our classifications across different experimental conditions. Furthermore, semantic entropy strongly validates our classifications (AUROC=0.78), revealing uncertainty's relationship to ideological consistency. Our findings demonstrate that ideological stability is topic-dependent and challenge the notion of monolithic LLM ideologies, and offer a robust way to distinguish genuine alignment from performative behavior.
>
---
#### [replaced 024] Inducing Programmatic Skills for Agentic Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06821v2](http://arxiv.org/pdf/2504.06821v2)**

> **作者:** Zora Zhiruo Wang; Apurva Gandhi; Graham Neubig; Daniel Fried
>
> **摘要:** To succeed in common digital tasks such as web navigation, agents must carry out a variety of specialized tasks such as searching for products or planning a travel route. To tackle these tasks, agents can bootstrap themselves by learning task-specific skills online through interaction with the web environment. In this work, we demonstrate that programs are an effective representation for skills. We propose agent skill induction (ASI), which allows agents to adapt themselves by inducing, verifying, and utilizing program-based skills on the fly. We start with an evaluation on the WebArena agent benchmark and show that ASI outperforms the static baseline agent and its text-skill counterpart by 23.5% and 11.3% in success rate, mainly thanks to the programmatic verification guarantee during the induction phase. ASI also improves efficiency by reducing 10.7-15.3% of the steps over baselines, by composing primitive actions (e.g., click) into higher-level skills (e.g., search product). We then highlight the efficacy of ASI in remaining efficient and accurate under scaled-up web activities. Finally, we examine the generalizability of induced skills when transferring between websites, and find that ASI can effectively reuse common skills, while also updating incompatible skills to versatile website changes.
>
---
#### [replaced 025] THEME: Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics
- **分类: q-fin.PM; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.16936v2](http://arxiv.org/pdf/2508.16936v2)**

> **作者:** Hoyoung Lee; Wonbin Ahn; Suhwan Park; Jaehoon Lee; Minjae Kim; Sungdong Yoo; Taeyoon Lim; Woohyung Lim; Yongjae Lee
>
> **备注:** Accepted at ACM International Conference on Information and Knowledge Management (CIKM)
>
> **摘要:** Thematic investing, which aims to construct portfolios aligned with structural trends, remains a challenging endeavor due to overlapping sector boundaries and evolving market dynamics. A promising direction is to build semantic representations of investment themes from textual data. However, despite their power, general-purpose LLM embedding models are not well-suited to capture the nuanced characteristics of financial assets, since the semantic representation of investment assets may differ fundamentally from that of general financial text. To address this, we introduce THEME, a framework that fine-tunes embeddings using hierarchical contrastive learning. THEME aligns themes and their constituent stocks using their hierarchical relationship, and subsequently refines these embeddings by incorporating stock returns. This process yields representations effective for retrieving thematically aligned assets with strong return potential. Empirical results demonstrate that THEME excels in two key areas. For thematic asset retrieval, it significantly outperforms leading large language models. Furthermore, its constructed portfolios demonstrate compelling performance. By jointly modeling thematic relationships from text and market dynamics from returns, THEME generates stock embeddings specifically tailored for a wide range of practical investment applications.
>
---
#### [replaced 026] KG-CQR: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval
- **分类: cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2508.20417v2](http://arxiv.org/pdf/2508.20417v2)**

> **作者:** Chi Minh Bui; Ngoc Mai Thieu; Van Vinh Nguyen; Jason J. Jung; Khac-Hoai Nam Bui
>
> **备注:** Accepted at Main EMNLP 2025
>
> **摘要:** The integration of knowledge graphs (KGs) with large language models (LLMs) offers significant potential to improve the retrieval phase of retrieval-augmented generation (RAG) systems. In this study, we propose KG-CQR, a novel framework for Contextual Query Retrieval (CQR) that enhances the retrieval phase by enriching the contextual representation of complex input queries using a corpus-centric KG. Unlike existing methods that primarily address corpus-level context loss, KG-CQR focuses on query enrichment through structured relation representations, extracting and completing relevant KG subgraphs to generate semantically rich query contexts. Comprising subgraph extraction, completion, and contextual generation modules, KG-CQR operates as a model-agnostic pipeline, ensuring scalability across LLMs of varying sizes without additional training. Experimental results on RAGBench and MultiHop-RAG datasets demonstrate KG-CQR's superior performance, achieving a 4-6% improvement in mAP and a 2-3% improvement in Recall@25 over strong baseline models. Furthermore, evaluations on challenging RAG tasks such as multi-hop question answering show that, by incorporating KG-CQR, the performance consistently outperforms the existing baseline in terms of retrieval effectiveness
>
---
#### [replaced 027] Active Domain Knowledge Acquisition with 100-Dollar Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17202v2](http://arxiv.org/pdf/2508.17202v2)**

> **作者:** Yang Wu; Raha Moraffah; Rujing Yao; Jinhong Yu; Zhimin Tao; Xiaozhong Liu
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
>
---
#### [replaced 028] L3Cube-MahaEmotions: A Marathi Emotion Recognition Dataset with Synthetic Annotations using CoTR prompting and Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00863v2](http://arxiv.org/pdf/2506.00863v2)**

> **作者:** Nidhi Kowtal; Raviraj Joshi
>
> **摘要:** Emotion recognition in low-resource languages like Marathi remains challenging due to limited annotated data. We present L3Cube-MahaEmotions, a high-quality Marathi emotion recognition dataset with 11 fine-grained emotion labels. The training data is synthetically annotated using large language models (LLMs), while the validation and test sets are manually labeled to serve as a reliable gold-standard benchmark. Building on the MahaSent dataset, we apply the Chain-of-Translation (CoTR) prompting technique, where Marathi sentences are translated into English and emotion labeled via a single prompt. GPT-4 and Llama3-405B were evaluated, with GPT-4 selected for training data annotation due to superior label quality. We evaluate model performance using standard metrics and explore label aggregation strategies (e.g., Union, Intersection). While GPT-4 predictions outperform fine-tuned BERT models, BERT-based models trained on synthetic labels fail to surpass GPT-4. This highlights both the importance of high-quality human-labeled data and the inherent complexity of emotion recognition. An important finding of this work is that generic LLMs like GPT-4 and Llama3-405B generalize better than fine-tuned BERT for complex low-resource emotion recognition tasks. The dataset and model are shared publicly at https://github.com/l3cube-pune/MarathiNLP
>
---
#### [replaced 029] MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21773v3](http://arxiv.org/pdf/2504.21773v3)**

> **作者:** Junsheng Huang; Zhitao He; Yucheng Huang; Sandeep Polisetty; Qingyun Wang; Yi. R; Fung
>
> **备注:** We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
>
> **摘要:** The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
>
---
#### [replaced 030] Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15266v4](http://arxiv.org/pdf/2504.15266v4)**

> **作者:** Vaishnavh Nagarajan; Chen Henry Wu; Charles Ding; Aditi Raghunathan
>
> **备注:** ICML 2025 (oral)
>
> **摘要:** We design a suite of minimal algorithmic tasks that are a loose abstraction of open-ended real-world tasks. This allows us to cleanly and controllably quantify the creative limits of the present-day language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how next-token learning is myopic; multi-token approaches, namely teacherless training and diffusion models, comparatively excel in producing diverse and original output. Secondly, to elicit randomness without hurting coherence, we find that injecting noise at the input layer (dubbed seed-conditioning) works surprisingly as well as (and in some conditions, better than) temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing open-ended creative skills, and offers new arguments for going beyond next-token learning and temperature sampling. We make part of the code available under https://github.com/chenwu98/algorithmic-creativity
>
---
#### [replaced 031] FedSEA-LLaMA: A Secure, Efficient and Adaptive Federated Splitting Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.DC**

- **链接: [http://arxiv.org/pdf/2505.15683v2](http://arxiv.org/pdf/2505.15683v2)**

> **作者:** Zishuai Zhang; Hainan zhang; Weihua Li; Qinnan zhang; jin Dong; Yongxin Tong; Zhiming Zheng
>
> **摘要:** Private data holds promise for improving LLMs due to its high quality, but its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based federated split models are proposed, which offload most model parameters to the server (or distributed clients) while retaining only a small portion on the client to ensure data privacy. Despite this design, they still face three challenges: 1) Peer-to-peer key encryption struggles to secure transmitted vectors effectively; 2) The auto-regressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) Fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FedSEA-LLaMA, a Secure, Efficient, and Adaptive Federated splitting framework based on LLaMA2. First, we inject Gaussian noise into forward-pass hidden states to enable secure end-to-end vector transmission. Second, we employ attention-mask compression and KV cache collaboration to reduce communication costs, accelerating training and inference. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements. Experiments on natural language understanding, summarization, and conversational QA tasks show that FedSEA-LLaMA maintains performance comparable to centralized LLaMA2 and achieves up to 8x speedups in training and inference. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FedSEA-LLaMA in security and adaptability.
>
---
#### [replaced 032] Refusal Tokens: A Simple Way to Calibrate Refusals in Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06748v2](http://arxiv.org/pdf/2412.06748v2)**

> **作者:** Neel Jain; Aditya Shrivastava; Chenyang Zhu; Daben Liu; Alfy Samuel; Ashwinee Panda; Anoop Kumar; Micah Goldblum; Tom Goldstein
>
> **备注:** 20 pages
>
> **摘要:** A key component of building safe and reliable language models is enabling the models to appropriately refuse to follow certain instructions or answer certain questions. We may want models to output refusal messages for various categories of user queries, for example, ill-posed questions, instructions for committing illegal acts, or queries which require information past the model's knowledge horizon. Engineering models that refuse to answer such questions is complicated by the fact that an individual may want their model to exhibit varying levels of sensitivity for refusing queries of various categories, and different users may want different refusal rates. The current default approach involves training multiple models with varying proportions of refusal messages from each category to achieve the desired refusal rates, which is computationally expensive and may require training a new model to accommodate each user's desired preference over refusal rates. To address these challenges, we propose refusal tokens, one such token for each refusal category or a single refusal token, which are prepended to the model's responses during training. We then show how to increase or decrease the probability of generating the refusal token for each category during inference to steer the model's refusal behavior. Refusal tokens enable controlling a single model's refusal rates without the need of any further fine-tuning, but only by selectively intervening during generation.
>
---
#### [replaced 033] Continuous Language Model Interpolation for Dynamic and Controllable Text Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.07117v2](http://arxiv.org/pdf/2404.07117v2)**

> **作者:** Sara Kangaslahti; David Alvarez-Melis
>
> **备注:** 20 pages, 22 figures
>
> **摘要:** As large language models (LLMs) have gained popularity for a variety of use cases, making them adaptable and controllable has become increasingly important, especially for user-facing applications. While the existing literature on LLM adaptation primarily focuses on finding a model (or models) that optimizes a single predefined objective, here we focus on the challenging case where the model must dynamically adapt to diverse -- and often changing -- user preferences. For this, we leverage adaptation methods based on linear weight interpolation, casting them as continuous multi-domain interpolators that produce models with specific prescribed generation characteristics on-the-fly. Specifically, we use low-rank updates to fine-tune a base model to various different domains, yielding a set of anchor models with distinct generation profiles. Then, we use the weight updates of these anchor models to parametrize the entire (infinite) class of models contained within their convex hull. We empirically show that varying the interpolation weights yields predictable and consistent change in the model outputs with respect to all of the controlled attributes. We find that there is little entanglement between most attributes and identify and discuss the pairs of attributes for which this is not the case. Our results suggest that linearly interpolating between the weights of fine-tuned models facilitates predictable, fine-grained control of model outputs with respect to multiple stylistic characteristics simultaneously.
>
---
#### [replaced 034] Exploring Selective Retrieval-Augmentation for Long-Tail Legal Text Classification
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.19997v3](http://arxiv.org/pdf/2508.19997v3)**

> **作者:** Boheng Mao
>
> **摘要:** Legal text classification is a fundamental NLP task in the legal domain. Benchmark datasets in this area often exhibit a long-tail label distribution, where many labels are underrepresented, leading to poor model performance on rare classes. This paper explores Selective Retrieval-Augmentation (SRA) as a proof-of-concept approach to this problem. SRA focuses on augmenting samples belonging to low-frequency labels in the training set, preventing the introduction of noise for well-represented classes, and requires no changes to the model architecture. Retrieval is performed only from the training data to ensure there is no potential information leakage, removing the need for external corpora simultaneously. SRA is tested on two legal text classification benchmark datasets with long-tail distributions: LEDGAR (single-label) and UNFAIR-ToS (multi-label). Results show that SRA achieves consistent gains in both micro-F1 and macro-F1 over LexGLUE baselines.
>
---
#### [replaced 035] Transforming Wearable Data into Personal Health Insights using Large Language Model Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.06464v3](http://arxiv.org/pdf/2406.06464v3)**

> **作者:** Mike A. Merrill; Akshay Paruchuri; Naghmeh Rezaei; Geza Kovacs; Javier Perez; Yun Liu; Erik Schenck; Nova Hammerquist; Jake Sunshine; Shyam Tailor; Kumar Ayush; Hao-Wei Su; Qian He; Cory Y. McLean; Mark Malhotra; Shwetak Patel; Jiening Zhan; Tim Althoff; Daniel McDuff; Xin Liu
>
> **备注:** 53 pages, 7 main figures, 2 main tables, accepted to Nature Communications
>
> **摘要:** Deriving personalized insights from popular wearable trackers requires complex numerical reasoning that challenges standard LLMs, necessitating tool-based approaches like code generation. Large language model (LLM) agents present a promising yet largely untapped solution for this analysis at scale. We introduce the Personal Health Insights Agent (PHIA), a system leveraging multistep reasoning with code generation and information retrieval to analyze and interpret behavioral health data. To test its capabilities, we create and share two benchmark datasets with over 4000 health insights questions. A 650-hour human expert evaluation shows that PHIA significantly outperforms a strong code generation baseline, achieving 84% accuracy on objective, numerical questions and, for open-ended ones, earning 83% favorable ratings while being twice as likely to achieve the highest quality rating. This work can advance behavioral health by empowering individuals to understand their data, enabling a new era of accessible, personalized, and data-driven wellness for the wider population.
>
---
#### [replaced 036] Trust but Verify! A Survey on Verification Design for Test-time Scaling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16665v2](http://arxiv.org/pdf/2508.16665v2)**

> **作者:** V Venktesh; Mandeep Rathee; Avishek Anand
>
> **备注:** 18 pages
>
> **摘要:** Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing a verifier. The verifiers serve as reward models that help score the candidate outputs from the decoding process to diligently explore the vast solution space and select the best outcome. This paradigm commonly termed has emerged as a superior approach owing to parameter free scaling at inference time and high performance gains. The verifiers could be prompt-based, fine-tuned as a discriminative or generative model to verify process paths, outcomes or both. Despite their widespread adoption, there is no detailed collection, clear categorization and discussion of diverse verification approaches and their training mechanisms. In this survey, we cover the diverse approaches in the literature and present a unified view of verifier training, types and their utility in test-time scaling. Our repository can be found at https://github.com/elixir-research-group/Verifierstesttimescaling.github.io.
>
---
#### [replaced 037] Retrieval-Augmented Machine Translation with Unstructured Knowledge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04342v2](http://arxiv.org/pdf/2412.04342v2)**

> **作者:** Jiaan Wang; Fandong Meng; Yingxue Zhang; Jie Zhou
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Retrieval-augmented generation (RAG) introduces additional information to enhance large language models (LLMs). In machine translation (MT), previous work typically retrieves in-context examples from paired MT corpora, or domain-specific knowledge from knowledge graphs, to enhance MT models. However, a large amount of world knowledge is organized in unstructured documents, and might not be fully paired across different languages. In this paper, we study retrieval-augmented MT using unstructured documents. Specifically, we build RAGtrans, the first benchmark to train and evaluate LLMs' retrieval-augmented MT ability. RAGtrans contains 169K MT samples collected via GPT-4o and human translators. Besides, documents from various languages are also provided to supply the knowledge to these samples. Based on RAGtrans, we further propose a multi-task training method to teach LLMs how to use information from multilingual documents during their translation. The method uses existing multilingual corpora to create auxiliary training objectives without additional labeling requirements. Extensive experiments show that the method improves LLMs by 1.6-3.1 BLEU and 1.0-2.0 COMET scores in En-Zh, and 1.7-2.9 BLEU and 2.1-2.7 COMET scores in En-De. We also conclude the critical difficulties that current LLMs face with this task.
>
---

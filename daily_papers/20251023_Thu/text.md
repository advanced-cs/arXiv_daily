# 自然语言处理 cs.CL

- **最新发布 98 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] Multi-Faceted Evaluation of Tool-Augmented Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文针对工具增强型对话系统的评估难题，提出TRACE基准与SCOPE框架，旨在自动发现多轮对话中的复杂错误模式。解决现有方法仅关注用户满意度或工具调用正确性、忽略误判工具结果等问题，显著提升对误导性用户反馈场景的评估能力。**

- **链接: [http://arxiv.org/pdf/2510.19186v1](http://arxiv.org/pdf/2510.19186v1)**

> **作者:** Zhaoyi Joey Hou; Tanya Shourya; Yingfan Wang; Shamik Roy; Vinayshekhar Bannihatti Kumar; Rashmi Gangadharaiah
>
> **备注:** The first two authors contributed equally. Manuscript under submission
>
> **摘要:** Evaluating conversational AI systems that use external tools is challenging, as errors can arise from complex interactions among user, agent, and tools. While existing evaluation methods assess either user satisfaction or agents' tool-calling capabilities, they fail to capture critical errors in multi-turn tool-augmented dialogues-such as when agents misinterpret tool results yet appear satisfactory to users. We introduce TRACE, a benchmark of systematically synthesized tool-augmented conversations covering diverse error cases, and SCOPE, an evaluation framework that automatically discovers diverse error patterns and evaluation rubrics in tool-augmented dialogues. Experiments show SCOPE significantly outperforms the baseline, particularly on challenging cases where user satisfaction signals are misleading.
>
---
#### [new 002] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型推理中因“学习悬崖”导致的难以突破难题的问题，提出Scaf-GRPO框架。通过诊断学习停滞并渐进注入提示，引导模型自主解题。实验表明，该方法显著提升数学推理能力，使Qwen2.5-Math-7B在AIME24上pass@1提升44.3%。**

- **链接: [http://arxiv.org/pdf/2510.19807v1](http://arxiv.org/pdf/2510.19807v1)**

> **作者:** Xichen Zhang; Sitong Wu; Yinghao Zhu; Haoru Tan; Shaozuo Yu; Ziyi He; Jiaya Jia
>
> **备注:** Code: https://github.com/dvlab-research/Scaf-GRPO
>
> **摘要:** Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM.
>
---
#### [new 003] Tibetan Language and AI: A Comprehensive Survey of Resources, Methods and Challenges
- **分类: cs.CL**

- **简介: 该论文聚焦藏语人工智能，系统梳理其文本与语音资源、NLP任务及大模型进展，分析数据稀缺、拼写不一等挑战，提出跨语言迁移与社区共建等解决方案，旨在推动低资源语言AI研究，构建可持续生态。**

- **链接: [http://arxiv.org/pdf/2510.19144v1](http://arxiv.org/pdf/2510.19144v1)**

> **作者:** Cheng Huang; Nyima Tashi; Fan Gao; Yutong Liu; Jiahao Li; Hao Tian; Siyang Jiang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Jin Zhang; Xiao Feng; Hao Wang; Jie Tang; Guojie Tang; Xiangxiang Wang; Jia Zhang; Tsengdar Lee; Yongbin Yu
>
> **摘要:** Tibetan, one of the major low-resource languages in Asia, presents unique linguistic and sociocultural characteristics that pose both challenges and opportunities for AI research. Despite increasing interest in developing AI systems for underrepresented languages, Tibetan has received limited attention due to a lack of accessible data resources, standardized benchmarks, and dedicated tools. This paper provides a comprehensive survey of the current state of Tibetan AI in the AI domain, covering textual and speech data resources, NLP tasks, machine translation, speech recognition, and recent developments in LLMs. We systematically categorize existing datasets and tools, evaluate methods used across different tasks, and compare performance where possible. We also identify persistent bottlenecks such as data sparsity, orthographic variation, and the lack of unified evaluation metrics. Additionally, we discuss the potential of cross-lingual transfer, multi-modal learning, and community-driven resource creation. This survey aims to serve as a foundational reference for future work on Tibetan AI research and encourages collaborative efforts to build an inclusive and sustainable AI ecosystem for low-resource languages.
>
---
#### [new 004] Improving Topic Modeling of Social Media Short Texts with Rephrasing: A Case Study of COVID-19 Related Tweets
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对社交媒体短文本主题建模中因语言简略、非正式和噪声导致的主题不清晰问题，提出模型无关的TM-Rephrase框架，利用大语言模型将口语化推文重写为正式语言，提升主题一致性、唯一性和多样性，显著改善了LDA等算法的效果。**

- **链接: [http://arxiv.org/pdf/2510.18908v1](http://arxiv.org/pdf/2510.18908v1)**

> **作者:** Wangjiaxuan Xin; Shuhua Yin; Shi Chen; Yaorong Ge
>
> **摘要:** Social media platforms such as Twitter (now X) provide rich data for analyzing public discourse, especially during crises such as the COVID-19 pandemic. However, the brevity, informality, and noise of social media short texts often hinder the effectiveness of traditional topic modeling, producing incoherent or redundant topics that are often difficult to interpret. To address these challenges, we have developed \emph{TM-Rephrase}, a model-agnostic framework that leverages large language models (LLMs) to rephrase raw tweets into more standardized and formal language prior to topic modeling. Using a dataset of 25,027 COVID-19-related Twitter posts, we investigate the effects of two rephrasing strategies, general- and colloquial-to-formal-rephrasing, on multiple topic modeling methods. Results demonstrate that \emph{TM-Rephrase} improves three metrics measuring topic modeling performance (i.e., topic coherence, topic uniqueness, and topic diversity) while reducing topic redundancy of most topic modeling algorithms, with the colloquial-to-formal strategy yielding the greatest performance gains and especially for the Latent Dirichlet Allocation (LDA) algorithm. This study contributes to a model-agnostic approach to enhancing topic modeling in public health related social media analysis, with broad implications for improved understanding of public discourse in health crisis as well as other important domains.
>
---
#### [new 005] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文提出从混合语言历史文献中检测拉丁文片段的多模态任务，针对复杂版式文档，构建了724页标注数据集，评估大模型性能。研究首次系统分析了主流模型在该任务中的能力与局限，证明了现代模型实现可靠拉丁文检测的可行性。**

- **链接: [http://arxiv.org/pdf/2510.19585v1](http://arxiv.org/pdf/2510.19585v1)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Under review. Both the dataset and code will be published
>
> **摘要:** This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task.
>
---
#### [new 006] JointCQ: Improving Factual Hallucination Detection with Joint Claim and Query Generation
- **分类: cs.CL**

- **简介: 该论文针对大模型事实幻觉检测任务，解决现有方法在声明提取和查询生成阶段存在的上下文丢失与查询不精准问题。提出JointCQ框架，联合生成声明与查询，通过筛选训练数据并微调模型，提升下游检索与验证效果，显著改善幻觉检测性能。**

- **链接: [http://arxiv.org/pdf/2510.19310v1](http://arxiv.org/pdf/2510.19310v1)**

> **作者:** Fan Xu; Huixuan Zhang; Zhenliang Zhang; Jiahao Wang; Xiaojun Wan
>
> **摘要:** Current large language models (LLMs) often suffer from hallucination issues, i,e, generating content that appears factual but is actually unreliable. A typical hallucination detection pipeline involves response decomposition (i.e., claim extraction), query generation, evidence collection (i.e., search or retrieval), and claim verification. However, existing methods exhibit limitations in the first two stages, such as context loss during claim extraction and low specificity in query generation, resulting in degraded performance across the hallucination detection pipeline. In this work, we introduce JointCQ https://github.com/pku0xff/JointCQ, a joint claim-and-query generation framework designed to construct an effective and efficient claim-query generator. Our framework leverages elaborately designed evaluation criteria to filter synthesized training data, and finetunes a language model for joint claim extraction and query generation, providing reliable and informative inputs for downstream search and verification. Experimental results demonstrate that our method outperforms previous methods on multiple open-domain QA hallucination detection benchmarks, advancing the goal of more trustworthy and transparent language model systems.
>
---
#### [new 007] Local Obfuscation by GLINER for Impartial Context Aware Lineage: Development and evaluation of PII Removal system
- **分类: cs.CL**

- **简介: 该论文针对临床文本中敏感信息（PII）的隐私保护问题，提出本地部署的LOGICAL系统。基于微调的GLiNER模型实现高效、精准的PII识别与脱敏，优于现有API服务与大模型方案，在低资源环境下保障数据安全，支持研究与AI开发。**

- **链接: [http://arxiv.org/pdf/2510.19346v1](http://arxiv.org/pdf/2510.19346v1)**

> **作者:** Prakrithi Shivaprakash; Lekhansh Shukla; Animesh Mukherjee; Prabhat Chand; Pratima Murthy
>
> **备注:** 30 pages, 15 main text and 15 supplementary material
>
> **摘要:** Removing Personally Identifiable Information (PII) from clinical notes in Electronic Health Records (EHRs) is essential for research and AI development. While Large Language Models (LLMs) are powerful, their high computational costs and the data privacy risks of API-based services limit their use, especially in low-resource settings. To address this, we developed LOGICAL (Local Obfuscation by GLINER for Impartial Context-Aware Lineage), an efficient, locally deployable PII removal system built on a fine-tuned Generalist and Lightweight Named Entity Recognition (GLiNER) model. We used 1515 clinical documents from a psychiatric hospital's EHR system. We defined nine PII categories for removal. A modern-gliner-bi-large-v1.0 model was fine-tuned on 2849 text instances and evaluated on a test set of 376 instances using character-level precision, recall, and F1-score. We compared its performance against Microsoft Azure NER, Microsoft Presidio, and zero-shot prompting with Gemini-Pro-2.5 and Llama-3.3-70B-Instruct. The fine-tuned GLiNER model achieved superior performance, with an overall micro-average F1-score of 0.980, significantly outperforming Gemini-Pro-2.5 (F1-score: 0.845). LOGICAL correctly sanitised 95% of documents completely, compared to 64% for the next-best solution. The model operated efficiently on a standard laptop without a dedicated GPU. However, a 2% entity-level false negative rate underscores the need for human-in-the-loop validation across all tested systems. Fine-tuned, specialised transformer models like GLiNER offer an accurate, computationally efficient, and secure solution for PII removal from clinical notes. This "sanitisation at the source" approach is a practical alternative to resource-intensive LLMs, enabling the creation of de-identified datasets for research and AI development while preserving data privacy, particularly in resource-constrained environments.
>
---
#### [new 008] Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对文本摘要中多目标优化难题，提出基于超体积优化（HVO）的强化学习方法，动态平衡一致性、连贯性、相关性和流畅性。通过引导模型逼近帕累托前沿，实现更均衡的摘要生成。实验表明，该方法优于现有方法，7B模型性能接近GPT-4。**

- **链接: [http://arxiv.org/pdf/2510.19325v1](http://arxiv.org/pdf/2510.19325v1)**

> **作者:** Junjie Song; Yiwen Liu; Dapeng Li; Yin Sun; Shukun Fu; Siqi Chen; Yuji Cao
>
> **摘要:** Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at https://github.com/ai4business-LiAuto/HVO.git
>
---
#### [new 009] VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对计算机操作智能体训练中高质量标注数据稀缺的问题，提出VideoAgentTrek框架，通过自动解析海量未标注屏幕视频，提取GUI交互动作与参数。利用逆动力学模型实现动作定位与内容识别，生成152万条交互数据，显著提升任务成功率与步准确率，实现了无需人工标注的规模化预训练。**

- **链接: [http://arxiv.org/pdf/2510.19488v1](http://arxiv.org/pdf/2510.19488v1)**

> **作者:** Dunjie Lu; Yiheng Xu; Junli Wang; Haoyuan Wu; Xinyuan Wang; Zekun Wang; Junlin Yang; Hongjin Su; Jixuan Chen; Junda Chen; Yuchen Mao; Jingren Zhou; Junyang Lin; Binyuan Hui; Tao Yu
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VideoAgentTrek, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos at web scale, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop Video2Action, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries and context, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text with high fidelity. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps automatically. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation.
>
---
#### [new 010] From Memorization to Generalization: Fine-Tuning Large Language Models for Biomedical Term-to-Identifier Normalization
- **分类: cs.CL; I.2**

- **简介: 该论文研究大语言模型在生物医学术语标准化任务中的表现，旨在提升术语到标准标识符的映射准确性。通过对比微调前后模型在不同术语体系上的记忆与泛化能力，发现标识符流行度和词汇化程度是影响微调效果的关键因素。**

- **链接: [http://arxiv.org/pdf/2510.19036v1](http://arxiv.org/pdf/2510.19036v1)**

> **作者:** Suswitha Pericharla; Daniel B. Hier; Tayo Obafemi-Ajayi
>
> **备注:** Submitted for publication to BMC BioData Mining
>
> **摘要:** Effective biomedical data integration depends on automated term normalization, the mapping of natural language biomedical terms to standardized identifiers. This linking of terms to identifiers is essential for semantic interoperability. Large language models (LLMs) show promise for this task but perform unevenly across terminologies. We evaluated both memorization (training-term performance) and generalization (validation-term performance) across multiple biomedical ontologies. Fine-tuning Llama 3.1 8B revealed marked differences by terminology. GO mappings showed strong memorization gains (up to 77% improvement in term-to-identifier accuracy), whereas HPO showed minimal improvement. Generalization occurred only for protein-gene (GENE) mappings (13.9% gain), while fine-tuning for HPO and GO yielded negligible transfer. Baseline accuracy varied by model scale, with GPT-4o outperforming both Llama variants for all terminologies. Embedding analyses showed tight semantic alignment between gene symbols and protein names but weak alignment between terms and identifiers for GO or HPO, consistent with limited lexicalization. Fine-tuning success depended on two interacting factors: identifier popularity and lexicalization. Popular identifiers were more likely encountered during pretraining, enhancing memorization. Lexicalized identifiers, such as gene symbols, enabled semantic generalization. By contrast, arbitrary identifiers in GO and HPO constrained models to rote learning. These findings provide a predictive framework for when fine-tuning enhances factual recall versus when it fails due to sparse or non-lexicalized identifiers.
>
---
#### [new 011] DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference
- **分类: cs.CL**

- **简介: 该论文针对推理型大模型冗余思考问题，提出DiffAdapt框架。通过分析思维链熵分布，识别题目难易度并动态匹配高效推理策略，无需微调主模型，仅用轻量探针分类，显著降低token消耗22.4%，在保持高精度的同时实现计算效率提升。**

- **链接: [http://arxiv.org/pdf/2510.19669v1](http://arxiv.org/pdf/2510.19669v1)**

> **作者:** Xiang Liu; Xuming Hu; Xiaowen Chu; Eunsol Choi
>
> **摘要:** Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
>
---
#### [new 012] Interpretable Question Answering with Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于知识图谱的可解释问答系统，不依赖大模型RAG。通过文档预处理生成QA对，构建知识图谱并用嵌入与模糊技术检索、重排、改写边信息，最终生成答案。旨在提升问答可解释性与准确性，解决了传统方法依赖大模型带来的不可解释性问题。**

- **链接: [http://arxiv.org/pdf/2510.19181v1](http://arxiv.org/pdf/2510.19181v1)**

> **作者:** Kartikeya Aneja; Manasvi Srivastava; Subhayan Das; Nagender Aneja
>
> **摘要:** This paper presents a question answering system that operates exclusively on a knowledge graph retrieval without relying on retrieval augmented generation (RAG) with large language models (LLMs). Instead, a small paraphraser model is used to paraphrase the entity relationship edges retrieved from querying the knowledge graph. The proposed pipeline is divided into two main stages. The first stage involves pre-processing a document to generate sets of question-answer (QA) pairs. The second stage converts these QAs into a knowledge graph from which graph-based retrieval is performed using embeddings and fuzzy techniques. The graph is queried, re-ranked, and paraphrased to generate a final answer. This work includes an evaluation using LLM-as-a-judge on the CRAG benchmark, which resulted in accuracies of 71.9% and 54.4% using LLAMA-3.2 and GPT-3.5-Turbo, respectively.
>
---
#### [new 013] Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
- **分类: cs.CL**

- **简介: 该论文聚焦于评估大语言模型在英韩双语对话中的社会推理能力，旨在识别说话者间的人际关系（如恋人、朋友）。研究构建了1000条电影对白组成的SCRIPTS数据集，标注概率性关系标签。实验发现，现有模型在韩语上表现显著下降，且常错误选择“不太可能”关系，思维链提示效果有限并可能放大偏见。**

- **链接: [http://arxiv.org/pdf/2510.19028v1](http://arxiv.org/pdf/2510.19028v1)**

> **作者:** Eunsu Kim; Junyeong Park; Juhyun Oh; Kiwoong Park; Seyoung Song; A. Seza Dogruoz; Najoung Kim; Alice Oh
>
> **摘要:** As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models.
>
---
#### [new 014] HAD: HAllucination Detection Language Models Based on a Comprehensive Hallucination Taxonomy
- **分类: cs.CL**

- **简介: 该论文针对自然语言生成中的幻觉问题，提出基于11类幻觉的综合分类体系与HAD模型。模型实现幻觉检测、定位与修正一体化，利用9万条合成数据训练，并在2248样本的HADTest上验证，显著优于现有方法，在多个基准上达到最先进水平。**

- **链接: [http://arxiv.org/pdf/2510.19318v1](http://arxiv.org/pdf/2510.19318v1)**

> **作者:** Fan Xu; Xinyu Hu; Zhenghan Yu; Li Lin; Xu Zhang; Yang Zhang; Wei Zhou; Jinjie Gu; Xiaojun Wan
>
> **摘要:** The increasing reliance on natural language generation (NLG) models, particularly large language models, has raised concerns about the reliability and accuracy of their outputs. A key challenge is hallucination, where models produce plausible but incorrect information. As a result, hallucination detection has become a critical task. In this work, we introduce a comprehensive hallucination taxonomy with 11 categories across various NLG tasks and propose the HAllucination Detection (HAD) models https://github.com/pku0xff/HAD, which integrate hallucination detection, span-level identification, and correction into a single inference process. Trained on an elaborate synthetic dataset of about 90K samples, our HAD models are versatile and can be applied to various NLG tasks. We also carefully annotate a test set for hallucination detection, called HADTest, which contains 2,248 samples. Evaluations on in-domain and out-of-domain test sets show that our HAD models generally outperform the existing baselines, achieving state-of-the-art results on HaluEval, FactCHD, and FaithBench, confirming their robustness and versatility.
>
---
#### [new 015] ToMMeR -- Efficient Entity Mention Detection from Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对命名实体识别中的提及检测任务，提出轻量级模型ToMMeR（<300K参数），从大语言模型早期层高效提取实体提及。在13个基准上实现93%零样本召回率，结合判别器保证高精度。研究表明，实体表示自然涌现于早期层，且可低开销恢复。**

- **链接: [http://arxiv.org/pdf/2510.19410v1](http://arxiv.org/pdf/2510.19410v1)**

> **作者:** Victor Morand; Nadi Tomeh; Josiane Mothe; Benjamin Piwowarski
>
> **备注:** Code is available at https://github.com/VictorMorand/llm2ner
>
> **摘要:** Identifying which text spans refer to entities -- mention detection -- is both foundational for information extraction and a known performance bottleneck. We introduce ToMMeR, a lightweight model (<300K parameters) probing mention detection capabilities from early LLM layers. Across 13 NER benchmarks, ToMMeR achieves 93\% recall zero-shot, with over 90\% precision using an LLM as a judge showing that ToMMeR rarely produces spurious predictions despite high recall. Cross-model analysis reveals that diverse architectures (14M-15B parameters) converge on similar mention boundaries (DICE >75\%), confirming that mention detection emerges naturally from language modeling. When extended with span classification heads, ToMMeR achieves near SOTA NER performance (80-87\% F1 on standard benchmarks). Our work provides evidence that structured entity representations exist in early transformer layers and can be efficiently recovered with minimal parameters.
>
---
#### [new 016] Slot Filling as a Reasoning Task for SpeechLLMs
- **分类: cs.CL**

- **简介: 该论文将槽位填充任务视为推理过程，提出在语音大模型中引入链式思维框架，通过构建推理数据集并采用监督微调提升性能。研究对比了不同文本基础模型的效果，发现混合模式的语音大模型优于单一模式，且专用逻辑推理模型作为基础效果较差。**

- **链接: [http://arxiv.org/pdf/2510.19326v1](http://arxiv.org/pdf/2510.19326v1)**

> **作者:** Kadri Hacioglu; Manjunath K E; Andreas Stolcke
>
> **摘要:** We propose integration of reasoning into speech large language models (speechLLMs) for the end-to-end slot-filling task. Inspired by the recent development of reasoning LLMs, we use a chain-of-thought framework to decompose the slot-filling task into multiple reasoning steps, create a reasoning dataset and apply the supervised fine-tuning strategy to a speechLLM. We distinguish between regular and reasoning speechLLMs and experiment with different types and sizes of LLMs as their text foundation models. We demonstrate performance improvements by introducing reasoning (intermediate) steps. However, we show that a reasoning textual LLM developed mainly for math, logic and coding domains might be inferior as a foundation model for a reasoning speechLLM. We further show that hybrid speechLLMs, built on a hybrid text foundation LLM and fine-tuned to preserve both direct and reasoning modes of operation, have better performance than those fine-tuned employing only one mode of operation.
>
---
#### [new 017] Machine Text Detectors are Membership Inference Attacks
- **分类: cs.CL**

- **简介: 该论文揭示了机器文本检测与成员推理攻击在方法上的可迁移性，证明二者共享最优度量。通过理论分析与大规模实验，验证了跨任务性能高度相关，并提出统一评估框架MINT，促进两领域协作与公平比较。**

- **链接: [http://arxiv.org/pdf/2510.19492v1](http://arxiv.org/pdf/2510.19492v1)**

> **作者:** Ryuto Koike; Liam Dugan; Masahiro Kaneko; Chris Callison-Burch; Naoaki Okazaki
>
> **摘要:** Although membership inference attacks (MIAs) and machine-generated text detection target different goals, identifying training samples and synthetic texts, their methods often exploit similar signals based on a language model's probability distribution. Despite this shared methodological foundation, the two tasks have been independently studied, which may lead to conclusions that overlook stronger methods and valuable insights developed in the other task. In this work, we theoretically and empirically investigate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. For our theoretical contribution, we prove that the metric that achieves the asymptotically highest performance on both tasks is the same. We unify a large proportion of the existing literature in the context of this optimal metric and hypothesize that the accuracy with which a given method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments, including 7 state-of-the-art MIA methods and 5 state-of-the-art machine text detectors across 13 domains and 10 generators, demonstrate very strong rank correlation (rho > 0.6) in cross-task performance. We notably find that Binoculars, originally designed for machine text detection, achieves state-of-the-art performance on MIA benchmarks as well, demonstrating the practical impact of the transferability. Our findings highlight the need for greater cross-task awareness and collaboration between the two research communities. To facilitate cross-task developments and fair evaluations, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, with implementation of 15 recent methods from both tasks.
>
---
#### [new 018] Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对NLP模型在处理风格化字体时的脆弱性，提出风格攻击伪装（SAD）方法。旨在解决人类可读文本与模型误判之间的感知差异问题。通过设计轻量与强力两种攻击形式，在情感分析、机器翻译及多模态任务中验证了攻击有效性，揭示了风格化文本对模型安全的潜在威胁。**

- **链接: [http://arxiv.org/pdf/2510.19641v1](http://arxiv.org/pdf/2510.19641v1)**

> **作者:** Yangshijie Zhang; Xinda Wang; Jialin Liu; Wenqiang Wang; Zhicong Ma; Xingxing Jia
>
> **摘要:** With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation.
>
---
#### [new 019] AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型推理加速中的推测解码（SD）任务，解决传统知识蒸馏与SD目标不匹配的问题。提出AdaSPEC方法，通过选择性过滤难拟合令牌，提升小模型对易拟合令牌的对齐度，显著提高令牌接受率，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.19779v1](http://arxiv.org/pdf/2510.19779v1)**

> **作者:** Yuezhou Hu; Jiaxin Guo; Xinyu Feng; Tuo Zhao
>
> **摘要:** Speculative Decoding (SD) accelerates large language model inference by employing a small draft model to generate predictions, which are then verified by a larger target model. The effectiveness of SD hinges on the alignment between these models, which is typically enhanced by Knowledge Distillation (KD). However, conventional KD methods aim to minimize the KL divergence between the draft and target models across all tokens, a goal that is misaligned with the true objective of SD, which is to maximize token acceptance rate. Therefore, draft models often struggle to fully assimilate the target model's knowledge due to capacity constraints, leading to suboptimal performance. To address this challenge, we propose AdaSPEC, a novel method that incorporates selective token filtering into the KD process. AdaSPEC utilizes a reference model to identify and filter out difficult-to-fit tokens, enabling the distillation of a draft model that better aligns with the target model on simpler tokens. This approach improves the overall token acceptance rate without compromising generation quality. We evaluate AdaSPEC across diverse tasks, including arithmetic reasoning, instruction-following, coding, and summarization, using model configurations of 31M/1.4B and 350M/2.7B parameters. Our results demonstrate that AdaSPEC consistently outperforms the state-of-the-art DistillSpec method, achieving higher acceptance rates across all tasks (up to 15\%). The code is publicly available at https://github.com/yuezhouhu/adaspec.
>
---
#### [new 020] Dynamic Evaluation for Oversensitivity in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大模型过度敏感问题，提出动态评估框架，通过生成模型特定的挑战性数据集，构建可演化基准OVERBENCH。解决了静态基准因模型迭代失效的问题，实现对防御性误判的持续监测与漏洞发现。**

- **链接: [http://arxiv.org/pdf/2510.19005v1](http://arxiv.org/pdf/2510.19005v1)**

> **作者:** Sophia Xiao Pu; Sitao Cheng; Xin Eric Wang; William Yang Wang
>
> **备注:** EMNLP-Findings 2025
>
> **摘要:** Oversensitivity occurs when language models defensively reject prompts that are actually benign. This behavior not only disrupts user interactions but also obscures the boundary between harmful and harmless content. Existing benchmarks rely on static datasets that degrade overtime as models evolve, leading to data contamination and diminished evaluative power. To address this, we develop a framework that dynamically generates model-specific challenging datasets, capturing emerging defensive patterns and aligning with each model's unique behavior. Building on this approach, we construct OVERBENCH, a benchmark that aggregates these datasets across diverse LLM families, encompassing 450,000 samples from 25 models. OVERBENCH provides a dynamic and evolving perspective on oversensitivity, allowing for continuous monitoring of defensive triggers as models advance, highlighting vulnerabilities that static datasets overlook.
>
---
#### [new 021] Lookahead Routing for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型多模型系统中的路由效率问题，提出Lookahead框架。通过预测潜在输出的隐表示来“前瞻”模型响应，实现更精准的路由决策，避免全量推理。在多个基准上验证，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.19506v1](http://arxiv.org/pdf/2510.19506v1)**

> **作者:** Canbin Huang; Tianyuan Shi; Yuhua Zhu; Ruijun Chen; Xiaojun Quan
>
> **摘要:** Large language model (LLM) routers improve the efficiency of multi-model systems by directing each query to the most appropriate model while leveraging the diverse strengths of heterogeneous LLMs. Most existing approaches frame routing as a classification problem based solely on the input query. While this reduces overhead by avoiding inference across all models, it overlooks valuable information that could be gleaned from potential outputs and fails to capture implicit intent or contextual nuances that often emerge only during response generation. These limitations can result in suboptimal routing decisions, particularly for complex or ambiguous queries that require deeper semantic understanding. To address this challenge, we propose Lookahead, a routing framework that "foresees" potential model outputs by predicting their latent representations and uses these predictions to guide model selection, thus enabling more informed routing without full inference. Within this framework, we implement two approaches based on causal and masked language models. Empirical evaluations across seven public benchmarks - spanning instruction following, mathematical reasoning, and code generation - show that Lookahead consistently outperforms existing routing baselines, achieving an average performance gain of 7.7% over the state-of-the-art. Our code is available at https://github.com/huangcb01/lookahead-routing.
>
---
#### [new 022] ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大工具集下LLM工具调用时上下文窗口受限的问题，提出ToolDreamer框架。通过LLM生成假设性工具描述（TD），使检索器更精准匹配查询与工具，提升检索效果。方法适用于稀疏与密集检索模型，无需额外训练，有效减轻LLM推理负担。**

- **链接: [http://arxiv.org/pdf/2510.19791v1](http://arxiv.org/pdf/2510.19791v1)**

> **作者:** Saptarshi Sengupta; Zhengyu Zhou; Jun Araki; Xingbo Wang; Bingqing Wang; Suhang Wang; Zhe Feng
>
> **摘要:** Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window.
>
---
#### [new 023] Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search
- **分类: cs.CL**

- **简介: 该论文针对长时程智能体搜索中因上下文限制导致的效率低下问题，提出SLIM框架。通过分离搜索与浏览工具并周期性摘要轨迹，有效控制上下文长度，显著减少工具调用次数，提升搜索效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.18939v1](http://arxiv.org/pdf/2510.18939v1)**

> **作者:** Howard Yen; Ashwin Paranjape; Mengzhou Xia; Thejas Venkatesh; Jack Hessel; Danqi Chen; Yuhao Zhang
>
> **备注:** Code and data are available here: https://github.com/howard-yen/SLIM
>
> **摘要:** Long-horizon agentic search requires iteratively exploring the web over long trajectories and synthesizing information across many sources, and is the foundation for enabling powerful applications like deep research systems. In this work, we show that popular agentic search frameworks struggle to scale to long trajectories primarily due to context limitations-they accumulate long, noisy content, hit context window and tool budgets, or stop early. Then, we introduce SLIM (Simple Lightweight Information Management), a simple framework that separates retrieval into distinct search and browse tools, and periodically summarizes the trajectory, keeping context concise while enabling longer, more focused searches. On long-horizon tasks, SLIM achieves comparable performance at substantially lower cost and with far fewer tool calls than strong open-source baselines across multiple base models. Specifically, with o3 as the base model, SLIM achieves 56% on BrowseComp and 31% on HLE, outperforming all open-source frameworks by 8 and 4 absolute points, respectively, while incurring 4-6x fewer tool calls. Finally, we release an automated fine-grained trajectory analysis pipeline and error taxonomy for characterizing long-horizon agentic search frameworks; SLIM exhibits fewer hallucinations than prior systems. We hope our analysis framework and simple tool design inform future long-horizon agents.
>
---
#### [new 024] That's Deprecated! Understanding, Detecting, and Steering Knowledge Conflicts in Language Models for Code Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型在代码生成中面对参数知识与提示冲突时的行为。针对知识冲突的检测与调控问题，提出通用框架、评估方法及数据集，实现80.65%冲突检测准确率，并通过激活层调控提升12.6%引导成功率，强调模型规模、任务与方向的平衡。**

- **链接: [http://arxiv.org/pdf/2510.19116v1](http://arxiv.org/pdf/2510.19116v1)**

> **作者:** Jaesung Bae; Cameron Churchwell; Mitchell Hermon; Tsun-An Hsieh; Jocelyn Xu; Yekaterina Yegorova; Mark Hasegawa-Johnson; Heng Ji
>
> **摘要:** This paper investigates how large language models (LLMs) behave when faced with discrepancies between their parametric knowledge and conflicting information contained in a prompt. Building on prior question-answering (QA) research, we extend the investigation of knowledge conflicts to the realm of code generation. We propose a domain-agnostic framework for constructing and interpreting such conflicts, along with a novel evaluation method and dataset tailored to code conflict scenarios. Our experiments indicate that sufficiently large LLMs encode the notion of a knowledge conflict in their parameters, enabling us to detect knowledge conflicts with up to \textbf{80.65\%} accuracy. Building on these insights, we show that activation-level steering can achieve up to a \textbf{12.6\%} improvement in steering success over a random baseline. However, effectiveness depends critically on balancing model size, task domain, and steering direction. The experiment code and data will be made publicly available after acceptance.
>
---
#### [new 025] "You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations
- **分类: cs.CL**

- **简介: 该论文属于人工智能在招聘评估中的应用研究，旨在检验大语言模型（LLMs）能否通过科技公司标准的工程师招聘测评。研究采用真实招聘题库，让先进LLMs作答并对比官方答案，发现所有模型均未达标，揭示了当前LLMs在专业能力评估中存在显著不足。**

- **链接: [http://arxiv.org/pdf/2510.19167v1](http://arxiv.org/pdf/2510.19167v1)**

> **作者:** Dingjie Fu; Dianxing Shi
>
> **备注:** Technical Report, 14 pages, 8 figures
>
> **摘要:** With the proliferation of the internet and the rapid advancement of Artificial Intelligence, leading technology companies face an urgent annual demand for a considerable number of software and algorithm engineers. To efficiently and effectively identify high-potential candidates from thousands of applicants, these firms have established a multi-stage selection process, which crucially includes a standardized hiring evaluation designed to assess job-specific competencies. Motivated by the demonstrated prowess of Large Language Models (LLMs) in coding and reasoning tasks, this paper investigates a critical question: Can LLMs successfully pass these hiring evaluations? To this end, we conduct a comprehensive examination of a widely used professional assessment questionnaire. We employ state-of-the-art LLMs to generate responses and subsequently evaluate their performance. Contrary to any prior expectation of LLMs being ideal engineers, our analysis reveals a significant inconsistency between the model-generated answers and the company-referenced solutions. Our empirical findings lead to a striking conclusion: All evaluated LLMs fails to pass the hiring evaluation.
>
---
#### [new 026] Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文针对语音模型评估碎片化问题，提出一个三维分类体系，明确“何种模型适用何种评估”。通过梳理现有评测任务，统一评估标准，揭示短板并指导未来基准设计，为语音模型评估提供系统性框架。**

- **链接: [http://arxiv.org/pdf/2510.19509v1](http://arxiv.org/pdf/2510.19509v1)**

> **作者:** Maureen de Seyssel; Eeshan Gunesh Dhekane
>
> **备注:** 57 pages (26 main, 25 appendix, 6 references)
>
> **摘要:** Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the \textbf{evaluation aspect} being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.
>
---
#### [new 027] Are Large Language Models Sensitive to the Motives Behind Communication?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）对沟通动机的敏感性，属于自然语言理解任务。旨在解决LLM能否像人类一样识别信息源动机并据此评估可信度的问题。通过控制实验与真实广告场景测试，发现LLM具备基础动机辨识能力，但需干预提升其在复杂环境中的表现。**

- **链接: [http://arxiv.org/pdf/2510.19687v1](http://arxiv.org/pdf/2510.19687v1)**

> **作者:** Addison J. Wu; Ryan Liu; Kerem Oktar; Theodore R. Sumers; Thomas L. Griffiths
>
> **备注:** NeurIPS 2025
>
> **摘要:** Human communication is motivated: people speak, write, and create content with a particular communicative intent in mind. As a result, information that large language models (LLMs) and AI agents process is inherently framed by humans' intentions and incentives. People are adept at navigating such nuanced information: we routinely identify benevolent or self-serving motives in order to decide what statements to trust. For LLMs to be effective in the real world, they too must critically evaluate content by factoring in the motivations of the source -- for instance, weighing the credibility of claims made in a sales pitch. In this paper, we undertake a comprehensive study of whether LLMs have this capacity for motivational vigilance. We first employ controlled experiments from cognitive science to verify that LLMs' behavior is consistent with rational models of learning from motivated testimony, and find they successfully discount information from biased sources in a human-like manner. We then extend our evaluation to sponsored online adverts, a more naturalistic reflection of LLM agents' information ecosystems. In these settings, we find that LLMs' inferences do not track the rational models' predictions nearly as closely -- partly due to additional information that distracts them from vigilance-relevant considerations. However, a simple steering intervention that boosts the salience of intentions and incentives substantially increases the correspondence between LLMs and the rational model. These results suggest that LLMs possess a basic sensitivity to the motivations of others, but generalizing to novel real-world settings will require further improvements to these models.
>
---
#### [new 028] Do Prompts Reshape Representations? An Empirical Study of Prompting Effects on Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究提示（prompting）对预训练语言模型嵌入表示的影响，属于自然语言处理中的零样本学习任务。旨在探究提示如何改变模型内部表示及其与任务相关性的关系。作者通过一系列探针实验，分析不同提示模板对零样本分类效果的影响，发现提示相关性与表示质量无稳定关联，挑战了“更相关提示带来更好表示”的假设，并探讨了可能原因。**

- **链接: [http://arxiv.org/pdf/2510.19694v1](http://arxiv.org/pdf/2510.19694v1)**

> **作者:** Cesar Gonzalez-Gutierrez; Dirk Hovy
>
> **摘要:** Prompting is a common approach for leveraging LMs in zero-shot settings. However, the underlying mechanisms that enable LMs to perform diverse tasks without task-specific supervision remain poorly understood. Studying the relationship between prompting and the quality of internal representations can shed light on how pre-trained embeddings may support in-context task solving. In this empirical study, we conduct a series of probing experiments on prompt embeddings, analyzing various combinations of prompt templates for zero-shot classification. Our findings show that while prompting affects the quality of representations, these changes do not consistently correlate with the relevance of the prompts to the target task. This result challenges the assumption that more relevant prompts necessarily lead to better representations. We further analyze potential factors that may contribute to this unexpected behavior.
>
---
#### [new 029] Modality Matching Matters: Calibrating Language Distances for Cross-Lingual Transfer in URIEL+
- **分类: cs.CL**

- **简介: 该论文针对跨语言迁移中语言距离度量的局限性，提出结构感知的多模态距离融合框架。针对地理、谱系、类型学三类距离，分别设计加权分布、双曲嵌入和潜在变量模型，并统一为任务无关的综合距离。有效提升多语言NLP任务性能，推动更精准的跨语言知识迁移。**

- **链接: [http://arxiv.org/pdf/2510.19217v1](http://arxiv.org/pdf/2510.19217v1)**

> **作者:** York Hay Ng; Aditya Khan; Xiang Lu; Matteo Salloum; Michael Zhou; Phuong H. Hoang; A. Seza Doğruöz; En-Shiun Annie Lee
>
> **摘要:** Existing linguistic knowledge bases such as URIEL+ provide valuable geographic, genetic and typological distances for cross-lingual transfer but suffer from two key limitations. One, their one-size-fits-all vector representations are ill-suited to the diverse structures of linguistic data, and two, they lack a principled method for aggregating these signals into a single, comprehensive score. In this paper, we address these gaps by introducing a framework for type-matched language distances. We propose novel, structure-aware representations for each distance type: speaker-weighted distributions for geography, hyperbolic embeddings for genealogy, and a latent variables model for typology. We unify these signals into a robust, task-agnostic composite distance. In selecting transfer languages, our representations and composite distances consistently improve performance across a wide range of NLP tasks, providing a more principled and effective toolkit for multilingual research.
>
---
#### [new 030] Contextual Augmentation for Entity Linking using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于实体链接任务，旨在解决传统两步法效率低、效果差的问题。提出一种融合实体识别与消歧的统一框架，利用大语言模型增强实体提及上下文，提升跨域性能，实验表明方法达到当前最优效果。**

- **链接: [http://arxiv.org/pdf/2510.18888v1](http://arxiv.org/pdf/2510.18888v1)**

> **作者:** Daniel Vollmers; Hamada M. Zahera; Diego Moussallem; Axel-Cyrille Ngonga Ngomo
>
> **摘要:** Entity Linking involves detecting and linking entity mentions in natural language texts to a knowledge graph. Traditional methods use a two-step process with separate models for entity recognition and disambiguation, which can be computationally intensive and less effective. We propose a fine-tuned model that jointly integrates entity recognition and disambiguation in a unified framework. Furthermore, our approach leverages large language models to enrich the context of entity mentions, yielding better performance in entity disambiguation. We evaluated our approach on benchmark datasets and compared with several baselines. The evaluation results show that our approach achieves state-of-the-art performance on out-of-domain datasets.
>
---
#### [new 031] M3-SLU: Evaluating Speaker-Attributed Reasoning in Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出M3-SLU基准，用于评估多说话人、多轮语音理解中的说话人归属推理能力。针对当前模型在识别“谁在何时说了什么”上的不足，基于四大语料构建1.2万+标注实例，包含问答与匹配两项任务，提供端到端与级联模型基线，揭示了现有模型在说话人感知理解上的关键缺陷。**

- **链接: [http://arxiv.org/pdf/2510.19358v1](http://arxiv.org/pdf/2510.19358v1)**

> **作者:** Yejin Kwon; Taewoo Kang; Hyunsoo Yoon; Changouk Kim
>
> **备注:** Submitted to LREC 2026. 11 pages, 5 figures
>
> **摘要:** We present M3-SLU, a new multimodal large language model (MLLM) benchmark for evaluating multi-speaker, multi-turn spoken language understanding. While recent models show strong performance in speech and text comprehension, they still struggle with speaker-attributed reasoning, the ability to understand who said what and when in natural conversations. M3-SLU is built from four open corpora (CHiME-6, MELD, MultiDialog, and AMI) and comprises over 12,000 validated instances with paired audio, transcripts, and metadata. It includes two tasks: (1) Speaker-Attributed Question Answering and (2) Speaker Attribution via Utterance Matching. We provide baseline results for both cascaded pipelines and end-to-end MLLMs, evaluated using an LLM-as-Judge and accuracy metrics. Results show that while models can capture what was said, they often fail to identify who said it, revealing a key gap in speaker-aware dialogue understanding. M3-SLU offers as a challenging benchmark to advance research in speaker-aware multimodal understanding.
>
---
#### [new 032] A Graph Signal Processing Framework for Hallucination Detection in Large Language Models
- **分类: cs.CL; cs.LG; eess.SP; stat.ML**

- **简介: 该论文针对大语言模型中的幻觉检测问题，提出基于图信号处理的谱分析框架。将注意力机制建模为动态图，利用频谱特征（如狄利克雷能量、谱熵）识别事实与幻觉，实验表明不同幻觉类型具独特谱模式，检测准确率达88.75%。**

- **链接: [http://arxiv.org/pdf/2510.19117v1](http://arxiv.org/pdf/2510.19117v1)**

> **作者:** Valentin Noël
>
> **备注:** Preprint under review (2025). 11 pages, 7 figures. Code and scripts: to be released
>
> **摘要:** Large language models achieve impressive results but distinguishing factual reasoning from hallucinations remains challenging. We propose a spectral analysis framework that models transformer layers as dynamic graphs induced by attention, with token embeddings as signals on these graphs. Through graph signal processing, we define diagnostics including Dirichlet energy, spectral entropy, and high-frequency energy ratios, with theoretical connections to computational stability. Experiments across GPT architectures suggest universal spectral patterns: factual statements exhibit consistent "energy mountain" behavior with low-frequency convergence, while different hallucination types show distinct signatures. Logical contradictions destabilize spectra with large effect sizes ($g>1.0$), semantic errors remain stable but show connectivity drift, and substitution hallucinations display intermediate perturbations. A simple detector using spectral signatures achieves 88.75% accuracy versus 75% for perplexity-based baselines, demonstrating practical utility. These findings indicate that spectral geometry may capture reasoning patterns and error behaviors, potentially offering a framework for hallucination detection in large language models.
>
---
#### [new 033] MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大模型中MoE架构因专家模块化导致弹性不足的问题，提出MoE-Prism，通过离线拆分专家为细粒度子专家、在线动态调度，实现服务弹性与QoS自适应。显著提升运行点多样性，优化云与边缘场景下的吞吐与延迟表现。**

- **链接: [http://arxiv.org/pdf/2510.19366v1](http://arxiv.org/pdf/2510.19366v1)**

> **作者:** Xinfeng Xia; Jiacheng Liu; Xiaofeng Hou; Peng Tang; Mingxuan Zhang; Wenfeng Wang; Chao Li
>
> **摘要:** Mixture-of-Experts (MoE) models, the state-of-the-art in large-scale AI, achieve high quality by sparsely activating parameters. However, their reliance on routing between a few monolithic experts via a top-k mechanism creates a "quality cliff", offering only a few coarse-grained operating points. This inflexibility forces a difficult trade-off between cost and quality, preventing adaptation to diverse Service Level Objectives (SLOs) and leading to significant resource over-provisioning. This paper introduces MoE-Prism, a model-system co-design that transforms rigid MoE models into elastic services. Our methodology is divided into two phases. First, an \emph{Offline Refactoring Engine} systematically deconstructs monolithic experts into fine-grained "sub-experts." This engine employs a partitioning optimization solver that uses a metaheuristic-based approach to group neurons, preserving functional locality without requiring retraining. Second, an \emph{Online Scheduling Engine} leverages this new elasticity through QoS-aware scheduling. It implements specialized policies to solve complex system problems, including maximizing throughput in cloud deployments and managing latency-optimized offloading for memory-constrained devices. Our evaluation across three different MoE models shows that MoE-Prismprovides over 4 times more distinct, stable operating points than the baseline. This allows an AI service to dynamically improve throughput by up to 19.9\% under a strict latency budget or reduce latency by up to 10.36\% under limited resources. MoE-Prism provides the critical "control knob" to bridge the model-system gap, enabling the next generation of adaptive, efficient, and QoS-aware AI services.
>
---
#### [new 034] Hubble: a Model Suite to Advance the Study of LLM Memorization
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Hubble，一套用于研究大模型记忆现象的开源模型套件。针对大语言模型可能记忆敏感数据的问题，构建了标准与扰动版本模型，通过控制插入文本验证记忆风险与训练策略的关系，揭示了稀释敏感数据和提前暴露可降低记忆风险，为隐私保护提供实证依据。**

- **链接: [http://arxiv.org/pdf/2510.19811v1](http://arxiv.org/pdf/2510.19811v1)**

> **作者:** Johnny Tian-Zheng Wei; Ameya Godbole; Mohammad Aflah Khan; Ryan Wang; Xiaoyuan Zhu; James Flemings; Nitya Kashyap; Krishna P. Gummadi; Willie Neiswanger; Robin Jia
>
> **摘要:** We present Hubble, a suite of fully open-source large language models (LLMs) for the scientific study of LLM memorization. Hubble models come in standard and perturbed variants: standard models are pretrained on a large English corpus, and perturbed models are trained in the same way but with controlled insertion of text (e.g., book passages, biographies, and test sets) designed to emulate key memorization risks. Our core release includes 8 models -- standard and perturbed models with 1B or 8B parameters, pretrained on 100B or 500B tokens -- establishing that memorization risks are determined by the frequency of sensitive data relative to size of the training corpus (i.e., a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus). Our release also includes 6 perturbed models with text inserted at different pretraining phases, showing that sensitive data without continued exposure can be forgotten. These findings suggest two best practices for addressing memorization risks: to dilute sensitive data by increasing the size of the training corpus, and to order sensitive data to appear earlier in training. Beyond these general empirical findings, Hubble enables a broad range of memorization research; for example, analyzing the biographies reveals how readily different types of private information are memorized. We also demonstrate that the randomized insertions in Hubble make it an ideal testbed for membership inference and machine unlearning, and invite the community to further explore, benchmark, and build upon our work.
>
---
#### [new 035] When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于大模型在动态知识演化下的表现问题，提出evolveQA基准，基于真实时间序列数据构建，评估模型对随时间变化的事实的处理能力。通过对比不同知识截止日期的模型性能，揭示了现有模型在时序知识冲突下的显著退化，最高达31%。**

- **链接: [http://arxiv.org/pdf/2510.19172v1](http://arxiv.org/pdf/2510.19172v1)**

> **作者:** Nishanth Sridhar Nakshatri; Shamik Roy; Manoj Ghuhan Arivazhagan; Hanhan Zhou; Vinayshekhar Bannihatti Kumar; Rashmi Gangadharaiah
>
> **备注:** Under submission
>
> **摘要:** LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
>
---
#### [new 036] The Massive Legal Embedding Benchmark (MLEB)
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出大规模法律嵌入基准MLEB，解决法律信息检索数据稀缺、覆盖不全问题。构建涵盖多司法辖区、多文档类型与任务的10个专家标注数据集，填补开源领域空白，推动可复现的法律AI研究。**

- **链接: [http://arxiv.org/pdf/2510.19365v1](http://arxiv.org/pdf/2510.19365v1)**

> **作者:** Umar Butler; Abdur-Rahman Butler; Adrian Lucas Malec
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** We present the Massive Legal Embedding Benchmark (MLEB), the largest, most diverse, and most comprehensive open-source benchmark for legal information retrieval to date. MLEB consists of ten expert-annotated datasets spanning multiple jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore), document types (cases, legislation, regulatory guidance, contracts, and literature), and task types (search, zero-shot classification, and question answering). Seven of the datasets in MLEB were newly constructed in order to fill domain and jurisdictional gaps in the open-source legal information retrieval landscape. We document our methodology in building MLEB and creating the new constituent datasets, and release our code, results, and data openly to assist with reproducible evaluations.
>
---
#### [new 037] CrossNews-UA: A Cross-lingual News Semantic Similarity Benchmark for Ukrainian, Polish, Russian, and English
- **分类: cs.CL**

- **简介: 该论文聚焦跨语言新闻语义相似性任务，针对非英语新闻虚假信息检测难题，提出可扩展的众包标注流程，构建了以乌克兰语为中心的多语言新闻数据集CrossNews-UA，涵盖乌、波、俄、英四语。通过4W标准标注语义相似性，评估多种模型性能，揭示多语言新闻分析挑战。**

- **链接: [http://arxiv.org/pdf/2510.19628v1](http://arxiv.org/pdf/2510.19628v1)**

> **作者:** Daryna Dementieva; Evgeniya Sukhodolskaya; Alexander Fraser
>
> **摘要:** In the era of social networks and rapid misinformation spread, news analysis remains a critical task. Detecting fake news across multiple languages, particularly beyond English, poses significant challenges. Cross-lingual news comparison offers a promising approach to verify information by leveraging external sources in different languages (Chen and Shu, 2024). However, existing datasets for cross-lingual news analysis (Chen et al., 2022a) were manually curated by journalists and experts, limiting their scalability and adaptability to new languages. In this work, we address this gap by introducing a scalable, explainable crowdsourcing pipeline for cross-lingual news similarity assessment. Using this pipeline, we collected a novel dataset CrossNews-UA of news pairs in Ukrainian as a central language with linguistically and contextually relevant languages-Polish, Russian, and English. Each news pair is annotated for semantic similarity with detailed justifications based on the 4W criteria (Who, What, Where, When). We further tested a range of models, from traditional bag-of-words, Transformer-based architectures to large language models (LLMs). Our results highlight the challenges in multilingual news analysis and offer insights into models performance.
>
---
#### [new 038] Transformer-Based Low-Resource Language Translation: A Study on Standard Bengali to Sylheti
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究低资源语言翻译任务，聚焦孟加拉语到锡尔赫特语的翻译。针对数据稀缺问题，比较微调多语言Transformer模型与零样本大语言模型的效果，结果表明微调模型显著优于后者，验证了任务特定适配对少数语言的重要性。**

- **链接: [http://arxiv.org/pdf/2510.18898v1](http://arxiv.org/pdf/2510.18898v1)**

> **作者:** Mangsura Kabir Oni; Tabia Tanzin Prama
>
> **摘要:** Machine Translation (MT) has advanced from rule-based and statistical methods to neural approaches based on the Transformer architecture. While these methods have achieved impressive results for high-resource languages, low-resource varieties such as Sylheti remain underexplored. In this work, we investigate Bengali-to-Sylheti translation by fine-tuning multilingual Transformer models and comparing them with zero-shot large language models (LLMs). Experimental results demonstrate that fine-tuned models significantly outperform LLMs, with mBART-50 achieving the highest translation adequacy and MarianMT showing the strongest character-level fidelity. These findings highlight the importance of task-specific adaptation for underrepresented languages and contribute to ongoing efforts toward inclusive language technologies.
>
---
#### [new 039] PBBQ: A Persian Bias Benchmark Dataset Curated with Human-AI Collaboration for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对波斯语大语言模型中的社会偏见问题，提出PBBQ基准数据集，涵盖16个文化类别，基于250人问卷与专家协作构建，含3.7万条问题。通过评测多个模型，发现其普遍存在偏见并复制人类偏见模式，旨在推动波斯语LLMs的公平性研究与改进。**

- **链接: [http://arxiv.org/pdf/2510.19616v1](http://arxiv.org/pdf/2510.19616v1)**

> **作者:** Farhan Farsi; Shayan Bali; Fatemeh Valeh; Parsa Ghofrani; Alireza Pakniat; Kian Kashfipour; Amir H. Payberah
>
> **摘要:** With the increasing adoption of large language models (LLMs), ensuring their alignment with social norms has become a critical concern. While prior research has examined bias detection in various languages, there remains a significant gap in resources addressing social biases within Persian cultural contexts. In this work, we introduce PBBQ, a comprehensive benchmark dataset designed to evaluate social biases in Persian LLMs. Our benchmark, which encompasses 16 cultural categories, was developed through questionnaires completed by 250 diverse individuals across multiple demographics, in close collaboration with social science experts to ensure its validity. The resulting PBBQ dataset contains over 37,000 carefully curated questions, providing a foundation for the evaluation and mitigation of bias in Persian language models. We benchmark several open-source LLMs, a closed-source model, and Persian-specific fine-tuned models on PBBQ. Our findings reveal that current LLMs exhibit significant social biases across Persian culture. Additionally, by comparing model outputs to human responses, we observe that LLMs often replicate human bias patterns, highlighting the complex interplay between learned representations and cultural stereotypes.Upon acceptance of the paper, our PBBQ dataset will be publicly available for use in future work. Content warning: This paper contains unsafe content.
>
---
#### [new 040] Modeling Turn-Taking with Semantically Informed Gestures
- **分类: cs.CL**

- **简介: 该论文聚焦于多模态对话中的轮次转换预测任务，旨在解决仅依赖语音与语言特征时轮换预测不精准的问题。研究构建了扩展的DnD Gesture++数据集，引入2,663条语义手势标注，并提出融合文本、音频与手势的混合专家模型，实验证明语义手势能有效提升预测性能，凸显其在轮换管理中的互补作用。**

- **链接: [http://arxiv.org/pdf/2510.19350v1](http://arxiv.org/pdf/2510.19350v1)**

> **作者:** Varsha Suresh; M. Hamza Mughal; Christian Theobalt; Vera Demberg
>
> **摘要:** In conversation, humans use multimodal cues, such as speech, gestures, and gaze, to manage turn-taking. While linguistic and acoustic features are informative, gestures provide complementary cues for modeling these transitions. To study this, we introduce DnD Gesture++, an extension of the multi-party DnD Gesture corpus enriched with 2,663 semantic gesture annotations spanning iconic, metaphoric, deictic, and discourse types. Using this dataset, we model turn-taking prediction through a Mixture-of-Experts framework integrating text, audio, and gestures. Experiments show that incorporating semantically guided gestures yields consistent performance gains over baselines, demonstrating their complementary role in multimodal turn-taking.
>
---
#### [new 041] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG
- **分类: cs.CL**

- **简介: 该论文针对多跳检索增强生成（Multi-hop RAG）效率低下的问题，提出TSSS框架。通过模板化推理减少重复生成，并采用基于检索器的确定性终止机制，实现高效稳定推理，在多个数据集上达到先进性能，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2510.19171v1](http://arxiv.org/pdf/2510.19171v1)**

> **作者:** Jihwan Bang; Juntae Lee; Seunghan Yang; Sungha Choi
>
> **备注:** Accepted at NeurIPS 2025 Workshop
>
> **摘要:** Multi-hop retrieval-augmented generation (RAG) is a promising strategy for complex reasoning, yet existing iterative prompting approaches remain inefficient. They often regenerate predictable token sequences at every step and rely on stochastic stopping, leading to excessive token usage and unstable termination. We propose TSSS (Think Straight, Stop Smart), a structured multi-hop RAG framework designed for efficiency. TSSS introduces (i) a template-based reasoning that caches recurring prefixes and anchors sub-queries to the main question, reducing token generation cost while promoting stable reasoning, and (ii) a retriever-based terminator, which deterministically halts reasoning once additional sub-queries collapse into repetition. This separation of structured reasoning and termination control enables both faster inference and more reliable answers. On HotpotQA, 2WikiMultiHop, and MuSiQue, TSSS achieves state-of-the-art accuracy and competitive efficiency among RAG-CoT approaches, highlighting its effectiveness in efficiency-constrained scenarios such as on-device inference.
>
---
#### [new 042] Spatio-temporal Sign Language Representation and Translation
- **分类: cs.CL; cs.CV**

- **简介: 该论文参与WMT-SLT 2022任务，将瑞士德语手语视频翻译为德语文本。针对传统SLT模型忽略时序特征的问题，提出端到端的时空特征表示与翻译模型，统一学习空间与时间信息，提升泛化能力。虽在开发集表现良好，测试集性能显著下降。**

- **链接: [http://arxiv.org/pdf/2510.19413v1](http://arxiv.org/pdf/2510.19413v1)**

> **作者:** Yasser Hamidullah; Josef van Genabith; Cristina España-Bonet
>
> **摘要:** This paper describes the DFKI-MLT submission to the WMT-SLT 2022 sign language translation (SLT) task from Swiss German Sign Language (video) into German (text). State-of-the-art techniques for SLT use a generic seq2seq architecture with customized input embeddings. Instead of word embeddings as used in textual machine translation, SLT systems use features extracted from video frames. Standard approaches often do not benefit from temporal features. In our participation, we present a system that learns spatio-temporal feature representations and translation in a single model, resulting in a real end-to-end architecture expected to better generalize to new data sets. Our best system achieved $5\pm1$ BLEU points on the development set, but the performance on the test dropped to $0.11\pm0.06$ BLEU points.
>
---
#### [new 043] Re:Member: Emotional Question Generation from Personal Memories
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出Re:Member系统，属于情感化语言学习任务。针对第二语言学习中互动性不足的问题，利用用户个人视频生成带有情绪色彩的靶语提问，通过视觉上下文匹配情感语音风格，结合多模态分析与语音合成技术，促进情感回忆与语言实践。**

- **链接: [http://arxiv.org/pdf/2510.19030v1](http://arxiv.org/pdf/2510.19030v1)**

> **作者:** Zackary Rackauckas; Nobuaki Minematsu; Julia Hirschberg
>
> **备注:** Accepted to HCI+NLP at ACL 2025
>
> **摘要:** We present Re:Member, a system that explores how emotionally expressive, memory-grounded interaction can support more engaging second language (L2) learning. By drawing on users' personal videos and generating stylized spoken questions in the target language, Re:Member is designed to encourage affective recall and conversational engagement. The system aligns emotional tone with visual context, using expressive speech styles such as whispers or late-night tones to evoke specific moods. It combines WhisperX-based transcript alignment, 3-frame visual sampling, and Style-BERT-VITS2 for emotional synthesis within a modular generation pipeline. Designed as a stylized interaction probe, Re:Member highlights the role of affect and personal media in learner-centered educational technologies.
>
---
#### [new 044] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints
- **分类: cs.CL**

- **简介: 该论文针对大模型知识更新难题，提出KORE方法，通过知识导向的增强与约束机制，实现新知识高效注入与旧知识有效保留，解决多模态模型因静态权重导致的知识滞后与灾难性遗忘问题。**

- **链接: [http://arxiv.org/pdf/2510.19316v1](http://arxiv.org/pdf/2510.19316v1)**

> **作者:** Kailin Jiang; Hongbo Jiang; Ning Jiang; Zhi Gao; Jinhe Bi; Yuchen Ren; Bin Li; Yuntao Du; Lei Liu; Qing Li
>
> **备注:** project page: https://kore-lmm.github.io/
>
> **摘要:** Large Multimodal Models encode extensive factual knowledge in their pre-trained weights. However, its knowledge remains static and limited, unable to keep pace with real-world developments, which hinders continuous knowledge acquisition. Effective knowledge injection thus becomes critical, involving two goals: knowledge adaptation (injecting new knowledge) and knowledge retention (preserving old knowledge). Existing methods often struggle to learn new knowledge and suffer from catastrophic forgetting. To address this, we propose KORE, a synergistic method of KnOwledge-oRientEd augmentations and constraints for injecting new knowledge into large multimodal models while preserving old knowledge. Unlike general text or image data augmentation, KORE automatically converts individual knowledge items into structured and comprehensive knowledge to ensure that the model accurately learns new knowledge, enabling accurate adaptation. Meanwhile, KORE stores previous knowledge in the covariance matrix of LMM's linear layer activations and initializes the adapter by projecting the original weights into the matrix's null space, defining a fine-tuning direction that minimizes interference with previous knowledge, enabling powerful retention. Extensive experiments on various LMMs, including LLaVA-v1.5-7B, LLaVA-v1.5-13B, and Qwen2.5-VL-7B, show that KORE achieves superior new knowledge injection performance and effectively mitigates catastrophic forgetting.
>
---
#### [new 045] MMAO-Bench: MultiModal All in One Benchmark Reveals Compositional Law between Uni-modal and Omni-modal in OmniModels
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出MMAO-Bench基准，评估多模态大模型在单模态与全模态理解上的能力。旨在揭示单模态与全模态性能间的组合规律，解决现有评估缺乏系统性的问题。通过1880个精心设计样本和多步开放题，发现强模型具协同增益，弱模型受瓶颈制约。**

- **链接: [http://arxiv.org/pdf/2510.18915v1](http://arxiv.org/pdf/2510.18915v1)**

> **作者:** Chen Chen; ZeYang Hu; Fengjiao Chen; Liya Ma; Jiaxing Liu; Xiaoyu Li; Xuezhi Cao
>
> **备注:** 10 pages, 8 figures. Work in progress
>
> **摘要:** Multimodal Large Languages models have been progressing from uni-modal understanding toward unifying visual, audio and language modalities, collectively termed omni models. However, the correlation between uni-modal and omni-modal remains unclear, which requires comprehensive evaluation to drive omni model's intelligence evolution. In this work, we propose a novel, high quality and diversity omni model benchmark, MultiModal All in One Benchmark (MMAO-Bench), which effectively assesses both uni-modal and omni-modal understanding capabilities. The benchmark consists of 1880 human curated samples, across 44 task types, and a innovative multi-step open-ended question type that better assess complex reasoning tasks. Experimental result shows the compositional law between cross-modal and uni-modal performance and the omni-modal capability manifests as a bottleneck effect on weak models, while exhibiting synergistic promotion on strong models.
>
---
#### [new 046] CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation
- **分类: cs.CL; I.2.6; C.2.4; C.3**

- **简介: 该论文提出CoSense-LLM，面向边缘计算场景的多模态感知任务，解决低延迟、低能耗、高隐私下大模型部署难题。通过轻量编码、本地检索增强生成、成本与不确定性感知的调度策略及安全执行机制，实现语义感知与云边协同，保障响应效率与数据隐私。**

- **链接: [http://arxiv.org/pdf/2510.19670v1](http://arxiv.org/pdf/2510.19670v1)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** 19 pages,8 figures
>
> **摘要:** We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
>
---
#### [new 047] Conditions for Catastrophic Forgetting in Multilingual Translation
- **分类: cs.CL**

- **简介: 该论文研究多语言翻译中微调导致的灾难性遗忘问题。通过系统实验发现，模型与数据规模比例是遗忘的关键因素，指令遵循能力比模型架构更重要。参数高效微调无明显优势，而跨语言对齐可缓解遗忘并促进未见语言的正向迁移。**

- **链接: [http://arxiv.org/pdf/2510.19546v1](http://arxiv.org/pdf/2510.19546v1)**

> **作者:** Danni Liu; Jan Niehues
>
> **备注:** Multilingual Representation Learning (MRL) Workshop 2025
>
> **摘要:** Fine-tuning multilingual foundation models on specific languages often induces catastrophic forgetting, degrading performance on languages unseen in fine-tuning. While this phenomenon is widely-documented, the literature presents fragmented results about when forgetting occurs. To address this ambiguity, we conduct a systematic empirical study using machine translation as a testbed to identify the conditions that trigger catastrophic forgetting in multilingual fine-tuning. Through controlled experiments across different model architectures, data scales, and fine-tuning approaches, we reveal that the relative scale between model and data size is a primary determinant of forgetting. Moreover, we demonstrate that a model's instruction-following ability is more critical for retaining multilingual knowledge than its architecture. Contrary to assumptions, parameter-efficient fine-tuning offers no clear advantage over full fine-tuning in mitigating forgetting. Lastly, we show that cross-lingual alignment can mitigate forgetting while also facilitating positive transfer to unseen target languages.
>
---
#### [new 048] When Models Can't Follow: Testing Instruction Adherence Across 256 LLMs
- **分类: cs.CL; cs.LG; I.2.7; I.2.6**

- **简介: 该论文聚焦于大语言模型（LLM）的指令遵循能力评估任务，旨在解决现有基准测试难以诊断具体失败模式的问题。作者构建了20个精心设计的提示构成的紧凑测试套件，对256个模型进行大规模实证分析，揭示了不同模型在格式合规、逻辑顺序等维度的共性缺陷，提出了一种高效、可复现的诊断工具。**

- **链接: [http://arxiv.org/pdf/2510.18892v1](http://arxiv.org/pdf/2510.18892v1)**

> **作者:** Richard J. Young; Brandon Gillins; Alice M. Matthews
>
> **备注:** 21 pages, 3 figures, 5 tables. Comprehensive evaluation of 256 LLMs on instruction-following tasks
>
> **摘要:** Despite widespread deployment of Large Language Models, systematic evaluation of instruction-following capabilities remains challenging. While comprehensive benchmarks exist, focused assessments that quickly diagnose specific instruction adherence patterns are valuable. As newer models may be trained on existing benchmarks, novel evaluation approaches are needed to assess genuine capabilities rather than memorized performance. This paper presents a streamlined evaluation framework using twenty carefully designed prompts to assess LLM instruction-following across diverse task categories. We demonstrate this framework through a large-scale empirical study conducted on October 14, 2025, testing 256 verified working models from 331 available via OpenRouter. To ensure methodological rigor and prevent selection bias, we first verified each model's basic functionality before inclusion. Unlike large-scale benchmarks requiring extensive computational resources, our approach offers a practical diagnostic tool researchers and practitioners can readily apply. Our methodology builds upon verifiable instructions while introducing a compact test suite balancing comprehensiveness with efficiency. Each prompt targets distinct aspects of instruction following, including format compliance, content constraints, logical sequencing, and multi-step task execution. We evaluate models from major providers (OpenAI, Anthropic, Google, Meta, Mistral) and emerging implementations (Qwen, DeepSeek, community models), providing comparative performance analysis. Our findings reveal consistent failure modes and identify specific instruction types posing particular challenges. This work contributes both a practical evaluation tool and one of the most comprehensive empirical analyses of instruction-following capabilities across the contemporary LLM landscape.
>
---
#### [new 049] What is the Best Sequence Length for BABYLM?
- **分类: cs.CL**

- **简介: 该论文研究婴儿语言模型（BabyLM）的最优序列长度。针对固定计算预算下序列长度对模型性能的影响，比较Mamba与OPT模型在不同任务上的表现，发现短序列适合语法泛化，长序列利于形态类比推理，揭示了任务与架构对最佳序列长度的依赖性。**

- **链接: [http://arxiv.org/pdf/2510.19493v1](http://arxiv.org/pdf/2510.19493v1)**

> **作者:** Suchir Salhan; Richard Diehl Martinez; Zébulon Goriely; Paula Buttery
>
> **备注:** Paper Accepted at the 2025 BabyLM Workshop @ EMNLP (Suzhou, China)
>
> **摘要:** Transformer language models typically operate with a fixed-length context window, which has grown in step with large-scale pretraining datasets. In the BabyLM Challenge, however, many past submissions have defaulted to using much shorter sequence lengths. We examine the impact of sequence length on BabyLM pretraining, to answer the simple question: what sequence length should we be using when training Baby LMs? Using 100M-word training data and fixed compute budgets, we compare 125M-parameter Mamba and OPT models, finding that although longer is often better, the optimal length depends on both task and architecture. Shorter sequences are sufficient for grammatical generalization tasks whereas longer contexts benefit morphological analogical reasoning tasks.
>
---
#### [new 050] LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts
- **分类: cs.CL**

- **简介: 该论文针对长上下文推理任务，提出LoongRL方法，通过KeyChain生成高难度长文本问答数据，利用强化学习诱导模型形成“计划-检索-推理-验证”模式，显著提升长程多跳问答与信息检索能力，实现128K长度任务的高效推理。**

- **链接: [http://arxiv.org/pdf/2510.19363v1](http://arxiv.org/pdf/2510.19363v1)**

> **作者:** Siyuan Wang; Gaokai Zhang; Li Lyna Zhang; Ning Shang; Fan Yang; Dongyao Chen; Mao Yang
>
> **摘要:** Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities.
>
---
#### [new 051] SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets
- **分类: cs.CL**

- **简介: 该论文提出SheetBrain，一种用于复杂大表格的神经符号推理框架。针对大语言模型在表格理解与推理中结构把握不准、结果不可靠的问题，设计了理解、执行与验证三模块协同的工作流，支持问答与操作任务。在新构建的SheetBench上显著提升准确率。**

- **链接: [http://arxiv.org/pdf/2510.19247v1](http://arxiv.org/pdf/2510.19247v1)**

> **作者:** Ziwei Wang; Jiayuan Su; Mengyu Zhou; Huaxing Zeng; Mengni Jia; Xiao Lv; Haoyu Dong; Xiaojun Ma; Shi Han; Dongmei Zhang
>
> **摘要:** Understanding and reasoning over complex spreadsheets remain fundamental challenges for large language models (LLMs), which often struggle with accurately capturing the complex structure of tables and ensuring reasoning correctness. In this work, we propose SheetBrain, a neuro-symbolic dual workflow agent framework designed for accurate reasoning over tabular data, supporting both spreadsheet question answering and manipulation tasks. SheetBrain comprises three core modules: an understanding module, which produces a comprehensive overview of the spreadsheet - including sheet summary and query-based problem insight to guide reasoning; an execution module, which integrates a Python sandbox with preloaded table-processing libraries and an Excel helper toolkit for effective multi-turn reasoning; and a validation module, which verifies the correctness of reasoning and answers, triggering re-execution when necessary. We evaluate SheetBrain on multiple public tabular QA and manipulation benchmarks, and introduce SheetBench, a new benchmark targeting large, multi-table, and structurally complex spreadsheets. Experimental results show that SheetBrain significantly improves accuracy on both existing benchmarks and the more challenging scenarios presented in SheetBench. Our code is publicly available at https://github.com/microsoft/SheetBrain.
>
---
#### [new 052] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models
- **分类: cs.CL**

- **简介: 该论文针对大模型时间敏感知识理解能力不足的问题，提出MINED基准，涵盖6维11任务，评估15个LMMs。结果表明多数模型在动态知识更新上表现弱，尤其体育类知识差；通过知识编辑可有效更新单条知识。**

- **链接: [http://arxiv.org/pdf/2510.19457v1](http://arxiv.org/pdf/2510.19457v1)**

> **作者:** Kailin Jiang; Ning Jiang; Yuchen Ren; Yuchen Li; Yifan Gao; Jinhe Bi; Yunpu Ma; Qingqing Liu; Xianhao Wang; Yifan Jia; Hongbo Jiang; Yaocong Hu; Bin Li; Lei Liu; Yuntao Du
>
> **备注:** project page:https://mined-lmm.github.io/
>
> **摘要:** Large Multimodal Models (LMMs) encode rich factual knowledge via cross-modal pre-training, yet their static representations struggle to maintain an accurate understanding of time-sensitive factual knowledge. Existing benchmarks remain constrained by static designs, inadequately evaluating LMMs' ability to understand time-sensitive knowledge. To address this gap, we propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types. Evaluating 15 widely used LMMs on MINED shows that Gemini-2.5-Pro achieves the highest average CEM score of 63.07, while most open-source LMMs still lack time understanding ability. Meanwhile, LMMs perform best on organization knowledge, whereas their performance is weakest on sport. To address these challenges, we investigate the feasibility of updating time-sensitive knowledge in LMMs through knowledge editing methods and observe that LMMs can effectively update knowledge via knowledge editing methods in single editing scenarios.
>
---
#### [new 053] Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection
- **分类: cs.CL; cs.CY**

- **简介: 该论文聚焦于自然语言处理中的仇恨言论检测任务，旨在缓解模型因身份偏见导致的不公平检测问题。通过引入标注者人格化特征，结合浅层与基于RAG的深层人格提示方法，探究同群与异群人格对模型敏感性的影响，验证了人格化可提升检测公平性，但存在局限。**

- **链接: [http://arxiv.org/pdf/2510.19331v1](http://arxiv.org/pdf/2510.19331v1)**

> **作者:** Ewelina Gajewska; Arda Derbent; Jaroslaw A Chudziak; Katarzyna Budzynska
>
> **备注:** This paper has been accepted for the upcoming 59th Hawaii International Conference on System Sciences (HICSS-59), 2026, Hawaii, USA. The final published version will appear in the official conference proceedings
>
> **摘要:** In this paper, we investigate how personalising Large Language Models (Persona-LLMs) with annotator personas affects their sensitivity to hate speech, particularly regarding biases linked to shared or differing identities between annotators and targets. To this end, we employ Google's Gemini and OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona prompting and a deeply contextualised persona development based on Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We analyse the impact of using in-group and out-group annotator personas on the models' detection performance and fairness across diverse social groups. This work bridges psychological insights on group identity with advanced NLP techniques, demonstrating that incorporating socio-demographic attributes into LLMs can address bias in automated hate speech detection. Our results highlight both the potential and limitations of persona-based approaches in reducing bias, offering valuable insights for developing more equitable hate speech detection systems.
>
---
#### [new 054] SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型在复杂推理中“浅层思考”（underthinking）问题，提出SmartSwitch框架。通过监测思维切换并利用奖励模型识别高潜力被放弃的思路，主动回溯并插入深化提示，促进深度探索。实验表明该方法有效提升多模型在数学推理任务中的性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.19767v1](http://arxiv.org/pdf/2510.19767v1)**

> **作者:** Xichen Zhang; Sitong Wu; Haoru Tan; Shaozuo Yu; Yinghao Zhu; Ziyi He; Jiaya Jia
>
> **备注:** Code: https://github.com/dvlab-research/SmartSwitch
>
> **摘要:** The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes.
>
---
#### [new 055] Small Language Models Offer Significant Potential for Science Community
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对科学文献检索中大模型成本高、存在偏见的问题，提出使用小型语言模型（MiniLM）实现高效、低成本的精准信息提取。基于7700万条高质量地学文献句子，通过语义搜索与句级索引，实现快速检索、趋势分析与研究演化追踪，适用于事实核查、教育等场景。**

- **链接: [http://arxiv.org/pdf/2510.18890v1](http://arxiv.org/pdf/2510.18890v1)**

> **作者:** Jian Zhang
>
> **摘要:** Recent advancements in natural language processing, particularly with large language models (LLMs), are transforming how scientists engage with the literature. While the adoption of LLMs is increasing, concerns remain regarding potential information biases and computational costs. Rather than LLMs, I developed a framework to evaluate the feasibility of precise, rapid, and cost-effective information retrieval from extensive geoscience literature using freely available small language models (MiniLMs). A curated corpus of approximately 77 million high-quality sentences, extracted from 95 leading peer-reviewed geoscience journals such as Geophysical Research Letters and Earth and Planetary Science Letters published during years 2000 to 2024, was constructed. MiniLMs enable a computationally efficient approach for extracting relevant domain-specific information from these corpora through semantic search techniques and sentence-level indexing. This approach, unlike LLMs such as ChatGPT-4 that often produces generalized responses, excels at identifying substantial amounts of expert-verified information with established, multi-disciplinary sources, especially for information with quantitative findings. Furthermore, by analyzing emotional tone via sentiment analysis and topical clusters through unsupervised clustering within sentences, MiniLM provides a powerful tool for tracking the evolution of conclusions, research priorities, advancements, and emerging questions within geoscience communities. Overall, MiniLM holds significant potential within the geoscience community for applications such as fact and image retrievals, trend analyses, contradiction analyses, and educational purposes.
>
---
#### [new 056] TheMCPCompany: Creating General-purpose Agents with Task-specific Tools
- **分类: cs.CL**

- **简介: 该论文提出TheMCPCompany基准，用于评估工具调用型通用智能体在真实服务交互任务中的表现。针对当前智能体依赖浏览器、难以高效利用大量专用工具的问题，构建了超1.8万工具的REST API驱动环境，并提供标注的最优工具路径。实验表明，先进模型在简单环境中能有效发现工具，但在复杂企业场景中仍面临挑战，凸显了对更强推理与检索能力的需求。**

- **链接: [http://arxiv.org/pdf/2510.19286v1](http://arxiv.org/pdf/2510.19286v1)**

> **作者:** Reza Esfandiarpoor; Vishwas Suryanarayanan; Stephen H. Bach; Vishal Chowdhary; Anthony Aue
>
> **备注:** Code: https://github.com/Reza-esfandiarpoor/the-mcp-company
>
> **摘要:** Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models.
>
---
#### [new 057] Training-Free Spectral Fingerprints of Voice Processing in Transformers
- **分类: cs.CL; cs.LG; eess.SP; stat.ML**

- **简介: 该论文研究多语言语音处理中变压器架构的隐性计算特征。针对模型因训练偏好产生的架构偏差问题，提出基于注意力图谱的谱分析方法，通过早期层代数连通性变化检测不同模型对语音切换的响应差异，揭示其计算指纹。结果表明，该方法能无训练地诊断模型偏见，验证其在行为相关性和可解释性上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.19131v1](http://arxiv.org/pdf/2510.19131v1)**

> **作者:** Valentin Noël
>
> **备注:** Preprint under review (2025). 12 pages, 8 figures
>
> **摘要:** Different transformer architectures implement identical linguistic computations via distinct connectivity patterns, yielding model imprinted ``computational fingerprints'' detectable through spectral analysis. Using graph signal processing on attention induced token graphs, we track changes in algebraic connectivity (Fiedler value, $\Delta\lambda_2$) under voice alternation across 20 languages and three model families, with a prespecified early window (layers 2--5). Our analysis uncovers clear architectural signatures: Phi-3-Mini shows a dramatic English specific early layer disruption ($\overline{\Delta\lambda_2}_{[2,5]}\!\approx\!-0.446$) while effects in 19 other languages are minimal, consistent with public documentation that positions the model primarily for English use. Qwen2.5-7B displays small, distributed shifts that are largest for morphologically rich languages, and LLaMA-3.2-1B exhibits systematic but muted responses. These spectral signatures correlate strongly with behavioral differences (Phi-3: $r=-0.976$) and are modulated by targeted attention head ablations, linking the effect to early attention structure and confirming functional relevance. Taken together, the findings are consistent with the view that training emphasis can leave detectable computational imprints: specialized processing strategies that manifest as measurable connectivity patterns during syntactic transformations. Beyond voice alternation, the framework differentiates reasoning modes, indicating utility as a simple, training free diagnostic for revealing architectural biases and supporting model reliability analysis.
>
---
#### [new 058] Evaluating LLM Story Generation through Large-scale Network Analysis of Social Structures
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦于评估大语言模型（LLM）的叙事生成能力，提出通过分析故事中角色间的有向社交网络来量化其创作特征。针对传统人工评估难以规模化的问题，研究构建了可扩展的网络分析方法，对比四类LLM与人类写作在社交结构上的差异，发现LLM更倾向生成紧密正向关系网，揭示了其创作中的系统性偏差。**

- **链接: [http://arxiv.org/pdf/2510.18932v1](http://arxiv.org/pdf/2510.18932v1)**

> **作者:** Hiroshi Nonaka; K. E. Perry
>
> **备注:** This paper has 14 pages and 8 figures. To be presented at the NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Evaluating the creative capabilities of large language models (LLMs) in complex tasks often requires human assessments that are difficult to scale. We introduce a novel, scalable methodology for evaluating LLM story generation by analyzing underlying social structures in narratives as signed character networks. To demonstrate its effectiveness, we conduct a large-scale comparative analysis using networks from over 1,200 stories, generated by four leading LLMs (GPT-4o, GPT-4o mini, Gemini 1.5 Pro, and Gemini 1.5 Flash) and a human-written corpus. Our findings, based on network properties like density, clustering, and signed edge weights, show that LLM-generated stories consistently exhibit a strong bias toward tightly-knit, positive relationships, which aligns with findings from prior research using human assessment. Our proposed approach provides a valuable tool for evaluating limitations and tendencies in the creative storytelling of current and future LLMs.
>
---
#### [new 059] Context-aware Fairness Evaluation and Mitigation in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在对话中产生的不公平行为问题，提出一种动态可逆的剪枝框架。通过检测上下文相关的神经元激活，在推理阶段自适应掩码调节其影响，实现细粒度、记忆感知的公平性控制，提升多轮多语言对话的一致性与可控性。**

- **链接: [http://arxiv.org/pdf/2510.18914v1](http://arxiv.org/pdf/2510.18914v1)**

> **作者:** Afrozah Nadeem; Mark Dras; Usman Naseem
>
> **备注:** PrePrint
>
> **摘要:** Large language models often display undesirable behaviors embedded in their internal representations, undermining fairness, inconsistency drift, amplification of harmful content, and the propagation of unwanted patterns during extended dialogue and conversations. Although training-time or data-centric methods attempt to reduce these effects, they are computationally expensive, irreversible once deployed, and slow to adapt to new conversational contexts. Pruning-based methods provide a flexible and transparent way to reduce bias by adjusting the neurons responsible for certain behaviors. However, most existing approaches are static; once a neuron is removed, the model loses the ability to adapt when the conversation or context changes. To address this, we propose a dynamic, reversible, pruning-based framework that detects context-aware neuron activations and applies adaptive masking to modulate their influence during generation. Our inference-time solution provides fine-grained, memory-aware mitigation with knowledge-preserved, more coherent behavior across multilingual single- and multi-turn dialogues, enabling dynamic fairness control in real-world conversational AI.
>
---
#### [new 060] Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型（LLM）的条件化生成任务，解决传统微调方法参数量大的问题。提出Zhyper框架，通过因子分解的超网络，根据文本描述生成上下文感知的LoRA适配器，实现高效参数控制。实验表明，其性能媲美先进方法，参数减少达26倍，并在文化对齐任务中提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19733v1](http://arxiv.org/pdf/2510.19733v1)**

> **作者:** M. H. I. Abdalla; Zhipin Wang; Christian Frey; Steffen Eger; Josif Grabocka
>
> **摘要:** Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values.
>
---
#### [new 061] LLavaCode: Compressed Code Representations for Retrieval-Augmented Code Generation
- **分类: cs.CL**

- **简介: 该论文针对代码补全中上下文过长导致推理慢的问题，提出LLavaCode框架，通过压缩代码为少量语义丰富的单标记向量，实现高效检索增强生成。显著降低时延（TTFT减少20-38%），同时提升生成质量，适用于交互式开发环境。**

- **链接: [http://arxiv.org/pdf/2510.19644v1](http://arxiv.org/pdf/2510.19644v1)**

> **作者:** Daria Cherniuk; Nikita Sukhorukov; Nikita Sushko; Daniil Gusak; Danil Sivtsov; Elena Tutubalina; Evgeny Frolov
>
> **摘要:** Retrieval-augmented generation has emerged as one of the most effective approaches for code completion, particularly when context from a surrounding repository is essential. However, incorporating context significantly extends sequence length, leading to slower inference - a critical limitation for interactive settings such as IDEs. In this work, we introduce LlavaCode, a framework that compresses code into compact, semantically rich representations interpretable by code LLM, enhancing generation quality while reducing the retrieved context to only a few compressed single-token vectors. Using a small projector module we can significantly increase the EM and ES metrics of coding model with negligible latency increase. Our experiments demonstrate that compressed context enables 20-38% reduction in Time-to-First-Token (TTFT) on line completion tasks compared to full-RAG pipelines.
>
---
#### [new 062] The Art of Asking: Multilingual Prompt Optimization for Synthetic Data
- **分类: cs.CL**

- **简介: 该论文针对多语言大模型训练中因翻译驱动提示导致的文化偏差与性能瓶颈问题，提出提示空间优化框架。通过自然度、文化适配性与难度增强三方面改进跨语言提示，在12种语言上显著提升模型表现，实现更鲁棒、全球化的多语言能力。**

- **链接: [http://arxiv.org/pdf/2510.19806v1](http://arxiv.org/pdf/2510.19806v1)**

> **作者:** David Mora; Viraat Aryabumi; Wei-Yin Ko; Sara Hooker; Julia Kreutzer; Marzieh Fadaee
>
> **摘要:** Synthetic data has become a cornerstone for scaling large language models, yet its multilingual use remains bottlenecked by translation-based prompts. This strategy inherits English-centric framing and style and neglects cultural dimensions, ultimately constraining model generalization. We argue that the overlooked prompt space-the very inputs that define training distributions-offers a more powerful lever for improving multilingual performance. We introduce a lightweight framework for prompt-space optimization, where translated prompts are systematically transformed for Naturalness, Cultural Adaptation, and Difficulty Enhancement. Using an off-the-shelf multilingual LLM, we apply these transformations to prompts for 12 languages spanning 7 families. Under identical data conditions, our approaches achieve substantial and consistent downstream improvements over the translation-only baseline: +4.7% on Global-MMLU accuracy, +2.4% on Flores XCometXL and +35.3% wins in preferences on mArenaHard. We establish prompt-space optimization as a simple yet powerful paradigm for building multilingual LLMs that are more robust, culturally grounded, and globally capable.
>
---
#### [new 063] DiSRouter: Distributed Self-Routing for LLM Selections
- **分类: cs.CL**

- **简介: 该论文提出DiSRouter，一种分布式自路由框架，用于大模型选择任务。针对现有中心化路由系统灵活性差、泛化能力弱的问题，通过构建具备自我认知能力的分布式LLM代理网络，实现动态查询路由。采用两阶段自知训练提升模型自我判断能力，显著提升性能与泛化性。**

- **链接: [http://arxiv.org/pdf/2510.19208v1](http://arxiv.org/pdf/2510.19208v1)**

> **作者:** Hang Zheng; Hongshen Xu; Yongkai Lin; Shuai Fan; Lu Chen; Kai Yu
>
> **摘要:** The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems.
>
---
#### [new 064] Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于阅读理解中的难度可控多选题生成任务。针对现有方法无法直接生成多选题且难度控制不精准的问题，提出基于大模型与直接偏好优化的生成方法，提升难度控制准确性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.19265v1](http://arxiv.org/pdf/2510.19265v1)**

> **作者:** Yuto Tomikawa; Masaki Uto
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Difficulty-controllable question generation for reading comprehension has gained significant attention in the field of education as a fundamental tool for adaptive learning support. Although several neural question generation methods have recently succeeded in controlling difficulty, conventional approaches still face two major limitations. First, they cannot directly generate multiple-choice questions, which are the most widely used question type in educational contexts. Second, they are not explicitly trained to optimize the accuracy of difficulty control, leaving room for further improvement in difficulty controllability. To address these limitations, this study proposes a novel difficulty-controllable multiple-choice question generation method for reading comprehension which leverages a large language model trained using a direct preference optimization technique to improve the accuracy of difficulty control.
>
---
#### [new 065] Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究自动语音识别（ASR）与语音翻译（ST）任务中最小贝叶斯风险（MBR）解码的有效性。针对当前主流使用束搜索而MBR在文本生成中表现更优的现象，作者评估了MBR在英语和日语上的ASR/ST性能，发现其在多数场景下优于束搜索，验证了MBR在高精度离线语音任务中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.19471v1](http://arxiv.org/pdf/2510.19471v1)**

> **作者:** Yuu Jinnai
>
> **摘要:** Recent work has shown that sample-based Minimum Bayes Risk (MBR) decoding outperforms beam search in text-to-text generation tasks, such as machine translation, text summarization, and image captioning. On the other hand, beam search is the current practice for speech-to-text tasks such as automatic speech recognition (ASR) and Speech Translation (ST). Given that MBR decoding is effective in text-to-text generation tasks, it is reasonable to expect it to also be effective for speech-to-text tasks. In this paper, we evaluate MBR decoding for ASR and ST tasks on English and Japanese using Whisper and its derivative models. We observe that the accuracy of MBR decoding outperforms that of beam search in most of the experimental settings we have evaluated. The results show that MBR decoding is a promising method for offline ASR and ST tasks that require high accuracy. The code is available at https://github.com/CyberAgentAILab/mbr-for-asr
>
---
#### [new 066] Adapting Multilingual Models to Code-Mixed Tasks via Model Merging
- **分类: cs.CL**

- **简介: 该论文研究代码混杂自然语言处理任务，提出通过模型合并（Model Merging）提升多语言模型适应能力。针对标注数据有限或无标注数据可用的场景，结合持续预训练与模型合并，显著优于全微调和仅预训练方法，并在跨语言迁移中表现更优，为不同数据条件下模型适配提供有效方案。**

- **链接: [http://arxiv.org/pdf/2510.19782v1](http://arxiv.org/pdf/2510.19782v1)**

> **作者:** Prashant Kodali; Vaishnavi Shivkumar; Swarang Joshi; Monojit Choudhary; Ponnurangam Kumaraguru; Manish Shrivastava
>
> **备注:** 9 pages, 5 tables, CODS 2025
>
> **摘要:** We study model merging as a practical alternative to conventional adaptation strategies for code-mixed NLP. Starting from a multilingual base model, we: (i) perform continued pre-training (CPT) on unlabeled code-mixed text to obtain an adapted checkpoint, (ii) merge checkpoint with the base model, and (iii) fine-tune (FT) on the downstream task data. We evaluate our approach for sentence classification (sentiment and hate speech) task in English-Hindi (En-Hi) and English-Spanish (En-Es) using XLM-R and Llama-3.2-1B models. Our results show that merged models consistently outperform full fine-tuning and CPT->FT. We observe gains of 2--5 points in F1 over full fine-tuning and ~1-2 points over CPT->FT, indicating that unlabeled data is leveraged more effectively via merging than via CPT alone. Zero-/few-shot prompting with larger LLMs (e.g., Llama-3.3-70B) lags behind fine-tuned and merged checkpoints, underscoring limits of in-context learning for code-mixed inputs. We further test cross-pair transfer by training on En-Hi and evaluating on En-Ta and En-Ml: merged checkpoints transfer more strongly than monolingual-English baselines (e.g., TV/TIES variants reaching 0.65-0.68 F1 vs 0.61-0.63 for full fine-tuning), suggesting that code-mixed knowledge is a more reliable substrate for low-resource pairs. We conclude with adaptation recipes matched to common data regimes (labeled only; labeled+unlabeled; transfer-only) and discuss limitations and scaling considerations for broader tasks and larger models.
>
---
#### [new 067] ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ProfBench，一个涵盖物理、化学、金融、咨询等专业领域的7000+问答评估基准，旨在解决大模型在专业文档理解与综合生成能力评估中的难题。通过构建低成本、高鲁棒的LLM-Judge系统，实现高效公平的评估，揭示了当前大模型在专业任务上的显著不足及开源与闭源模型间的性能差异。**

- **链接: [http://arxiv.org/pdf/2510.18941v1](http://arxiv.org/pdf/2510.18941v1)**

> **作者:** Zhilin Wang; Jaehun Jung; Ximing Lu; Shizhe Diao; Ellie Evans; Jiaqi Zeng; Pavlo Molchanov; Yejin Choi; Jan Kautz; Yi Dong
>
> **备注:** 23 pages
>
> **摘要:** Evaluating progress in large language models (LLMs) is often constrained by the challenge of verifying responses, limiting assessments to tasks like mathematics, programming, and short-form question-answering. However, many real-world applications require evaluating LLMs in processing professional documents, synthesizing information, and generating comprehensive reports in response to user queries. We introduce ProfBench: a set of over 7000 response-criterion pairs as evaluated by human-experts with professional knowledge across Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA. We build robust and affordable LLM-Judges to evaluate ProfBench rubrics, by mitigating self-enhancement bias and reducing the cost of evaluation by 2-3 orders of magnitude, to make it fair and accessible to the broader community. Our findings reveal that ProfBench poses significant challenges even for state-of-the-art LLMs, with top-performing models like GPT-5-high achieving only 65.9\% overall performance. Furthermore, we identify notable performance disparities between proprietary and open-weight models and provide insights into the role that extended thinking plays in addressing complex, professional-domain tasks. Data: https://huggingface.co/datasets/nvidia/ProfBench and Code: https://github.com/NVlabs/ProfBench
>
---
#### [new 068] Misinformation Detection using Large Language Models with Explainability
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对在线平台虚假信息传播问题，提出一种可解释且高效的虚假信息检测方法。基于RoBERTa和DistilBERT，采用两阶段微调策略，在两个真实数据集上验证有效性。结合LIME与SHAP实现细粒度解释，证明轻量模型在保持性能的同时显著降低计算成本，提升了检测系统的可信度与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.18918v1](http://arxiv.org/pdf/2510.18918v1)**

> **作者:** Jainee Patel; Chintan Bhatt; Himani Trivedi; Thanh Thi Nguyen
>
> **备注:** Accepted for publication in the Proceedings of the 8th International Conference on Algorithms, Computing and Artificial Intelligence (ACAI 2025)
>
> **摘要:** The rapid spread of misinformation on online platforms undermines trust among individuals and hinders informed decision making. This paper shows an explainable and computationally efficient pipeline to detect misinformation using transformer-based pretrained language models (PLMs). We optimize both RoBERTa and DistilBERT using a two-step strategy: first, we freeze the backbone and train only the classification head; then, we progressively unfreeze the backbone layers while applying layer-wise learning rate decay. On two real-world benchmark datasets, COVID Fake News and FakeNewsNet GossipCop, we test the proposed approach with a unified protocol of preprocessing and stratified splits. To ensure transparency, we integrate the Local Interpretable Model-Agnostic Explanations (LIME) at the token level to present token-level rationales and SHapley Additive exPlanations (SHAP) at the global feature attribution level. It demonstrates that DistilBERT achieves accuracy comparable to RoBERTa while requiring significantly less computational resources. This work makes two key contributions: (1) it quantitatively shows that a lightweight PLM can maintain task performance while substantially reducing computational cost, and (2) it presents an explainable pipeline that retrieves faithful local and global justifications without compromising performance. The results suggest that PLMs combined with principled fine-tuning and interpretability can be an effective framework for scalable, trustworthy misinformation detection.
>
---
#### [new 069] Unraveling Emotions with Pre-Trained Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦情感识别任务，针对大语言模型在开放文本中情感分析面临的上下文模糊、语言多样性和复杂情感表达等挑战，比较了微调与提示工程的效果，并探究了提示设计与情感分组对模型性能的影响。实验表明，微调模型表现优异，而结构化提示与情感分组可显著提升LLM性能。**

- **链接: [http://arxiv.org/pdf/2510.19668v1](http://arxiv.org/pdf/2510.19668v1)**

> **作者:** Alejandro Pajón-Sanmartín; Francisco De Arriba-Pérez; Silvia García-Méndez; Fátima Leal; Benedita Malheiro; Juan Carlos Burguillo-Rial
>
> **摘要:** Transformer models have significantly advanced the field of emotion recognition. However, there are still open challenges when exploring open-ended queries for Large Language Models (LLMs). Although current models offer good results, automatic emotion analysis in open texts presents significant challenges, such as contextual ambiguity, linguistic variability, and difficulty interpreting complex emotional expressions. These limitations make the direct application of generalist models difficult. Accordingly, this work compares the effectiveness of fine-tuning and prompt engineering in emotion detection in three distinct scenarios: (i) performance of fine-tuned pre-trained models and general-purpose LLMs using simple prompts; (ii) effectiveness of different emotion prompt designs with LLMs; and (iii) impact of emotion grouping techniques on these models. Experimental tests attain metrics above 70% with a fine-tuned pre-trained model for emotion recognition. Moreover, the findings highlight that LLMs require structured prompt engineering and emotion grouping to enhance their performance. These advancements improve sentiment analysis, human-computer interaction, and understanding of user behavior across various domains.
>
---
#### [new 070] SONAR-SLT: Multilingual Sign Language Translation via Language-Agnostic Sentence Embedding Supervision
- **分类: cs.CL**

- **简介: 该论文聚焦多语言手语翻译任务，解决传统方法依赖单一语种文本监督导致的扩展性差问题。提出基于跨语言多模态嵌入的监督机制，结合多语言目标增强与视频级扰动，实现直接多语言翻译，显著提升低资源场景下的性能。**

- **链接: [http://arxiv.org/pdf/2510.19398v1](http://arxiv.org/pdf/2510.19398v1)**

> **作者:** Yasser Hamidullah; Shakib Yazdani; Cennet Oguz; Josef van Genabith; Cristina España-Bonet
>
> **摘要:** Sign language translation (SLT) is typically trained with text in a single spoken language, which limits scalability and cross-language generalization. Earlier approaches have replaced gloss supervision with text-based sentence embeddings, but up to now, these remain tied to a specific language and modality. In contrast, here we employ language-agnostic, multimodal embeddings trained on text and speech from multiple languages to supervise SLT, enabling direct multilingual translation. To address data scarcity, we propose a coupled augmentation method that combines multilingual target augmentations (i.e. translations into many languages) with video-level perturbations, improving model robustness. Experiments show consistent BLEURT gains over text-only sentence embedding supervision, with larger improvements in low-resource settings. Our results demonstrate that language-agnostic embedding supervision, combined with coupled augmentation, provides a scalable and semantically robust alternative to traditional SLT training.
>
---
#### [new 071] AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型数学推理能力提升难题，提出AgenticMath框架，通过四阶段自动化流程生成高质量数学问答数据。其核心工作是构建基于多智能体的高质数据生成管道，显著提升小规模数据（30-60K）的训练效率与效果，优于传统大规模低质数据方法。**

- **链接: [http://arxiv.org/pdf/2510.19361v1](http://arxiv.org/pdf/2510.19361v1)**

> **作者:** Xianyang Liu; Yilin Liu; Shuai Wang; Hao Cheng; Andrew Estornell; Yuzhi Zhao; Jiaheng Wei
>
> **备注:** Work in progress
>
> **摘要:** The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
>
---
#### [new 072] BLiSS 1.0: Evaluating Bilingual Learner Competence in Second Language Small Language Models
- **分类: cs.CL**

- **简介: 该论文提出BLiSS 1.0，一个评估双语学习者第二语言小模型能力的基准。针对现有评测与人类语言习得模式脱节的问题，构建包含280万条自然学习者语句的三元组数据集，通过“选择性容忍”测试衡量模型对自然错误与人工错误的区分能力。实验表明该能力与语法正确性不同，且受训练范式显著影响，验证了其在评估模型语言习得机制方面的有效性。**

- **链接: [http://arxiv.org/pdf/2510.19419v1](http://arxiv.org/pdf/2510.19419v1)**

> **作者:** Yuan Gao; Suchir Salhan; Andrew Caines; Paula Buttery; Weiwei Sun
>
> **备注:** Accepted Paper at the BabyLM Workshop 2025 @ EMNLP (Presentation in Suzhou, China)
>
> **摘要:** To bridge the gap between performance-oriented benchmarks and the evaluation of cognitively inspired models, we introduce BLiSS 1.0, a Benchmark of Learner Interlingual Syntactic Structure. Our benchmark operationalizes a new paradigm of selective tolerance, testing whether a model finds a naturalistic learner error more plausible than a matched, artificial error within the same sentence. Constructed from over 2.8 million naturalistic learner sentences, BLiSS provides 136,867 controlled triplets (corrected, learner, artificial) for this purpose. Experiments on a diverse suite of models demonstrate that selective tolerance is a distinct capability from standard grammaticality, with performance clustering strongly by training paradigm. This validates BLiSS as a robust tool for measuring how different training objectives impact a model's alignment with the systematic patterns of human language acquisition.
>
---
#### [new 073] From Answers to Guidance: A Proactive Dialogue System for Legal Documents
- **分类: cs.CL**

- **简介: 该论文针对普通民众难以理解复杂法律文本的问题，提出EUDial数据集与LexGuide框架。任务为构建主动式法律对话系统。工作包括构建多轮法律对话数据集，并设计基于层次主题组织的检索增强生成方法，实现结构化、连贯的法律信息引导。**

- **链接: [http://arxiv.org/pdf/2510.19723v1](http://arxiv.org/pdf/2510.19723v1)**

> **作者:** Ashish Chouhan; Michael Gertz
>
> **备注:** 21 pages, 3 figures, 2 tables, 2 prompts
>
> **摘要:** The accessibility of legal information remains a constant challenge, particularly for laypersons seeking to understand and apply complex institutional texts. While the European Union provides open access to legislation, parliamentary responses, and regulatory documents, these resources can be challenging for laypeople to explore. In this paper, we introduce EUDial, a proactive multi-turn dialogue dataset constructed from 204 blogs curated by the Citizens' Enquiries Unit (AskEP) of the European Parliamentary Research Service. EUDial contains 880 dialogue turns (averaging 4.3 turns per dialogue), where each dialogue includes initial questions, structured answers, and follow-up questions. Beyond dataset construction, we propose the LexGuide framework that leverages retrieval-augmented generation with hierarchical topic organization to structure dialogue progression, ensuring both comprehensive coverage of legal aspects and coherence across conversational turns. The results demonstrate that proactive, structured navigation closes the gap between the availability of legal information and citizen comprehension, establishing EUDial and LexGuide as practical resources for advancing proactive legal dialogue systems.
>
---
#### [new 074] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出DuoLens框架，针对多语言文本与代码的机器生成内容检测任务，解决现有检测器计算成本高或准确率低的问题。通过微调编码器型小语言模型（如RoBERTA、CodeBERTa），在保持高精度（AUROC 0.97–0.99）的同时，显著降低延迟与显存占用，并具备强鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.18904v1](http://arxiv.org/pdf/2510.18904v1)**

> **作者:** Shriyansh Agrawal; Aidan Lau; Sanyam Shah; Ahan M R; Kevin Zhu; Sunishchal Dev; Vasu Sharma
>
> **备注:** Accepted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025): 4th Workshop on Deep Learning for Code
>
> **摘要:** The prevalence of Large Language Models (LLMs) for generating multilingual text and source code has only increased the imperative for machine-generated content detectors to be accurate and efficient across domains. Current detectors, predominantly utilizing zero-shot methods, such as Fast DetectGPT or GPTZero, either incur high computational cost or lack sufficient accuracy, often with a trade-off between the two, leaving room for further improvement. To address these gaps, we propose the fine-tuning of encoder-only Small Language Models (SLMs), in particular, the pre-trained models of RoBERTA and CodeBERTa using specialized datasets on source code and other natural language to prove that for the task of binary classification, SLMs outperform LLMs by a huge margin whilst using a fraction of compute. Our encoders achieve AUROC $= 0.97$ to $0.99$ and macro-F1 $0.89$ to $0.94$ while reducing latency by $8$-$12\times$ and peak VRAM by $3$-$5\times$ at $512$-token inputs. Under cross-generator shifts and adversarial transformations (paraphrase, back-translation; code formatting/renaming), performance retains $\geq 92%$ of clean AUROC. We release training and evaluation scripts with seeds and configs; a reproducibility checklist is also included.
>
---
#### [new 075] Learning from the Best, Differently: A Diversity-Driven Rethinking on Data Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型预训练数据选择问题，提出ODiS算法。针对现有方法因维度相关性导致多样性缺失的问题，通过PCA实现多维评分正交化，分别选取各正交维度的高分数据，兼顾质量与多样性。实验表明，该方法显著提升下游性能。**

- **链接: [http://arxiv.org/pdf/2510.18909v1](http://arxiv.org/pdf/2510.18909v1)**

> **作者:** Hongyi He; Xiao Liu; Zhenghao Lin; Mingni Tang; Yi Cheng; Jintao Wang; Wenjie Li; Peng Cheng; Yeyun Gong
>
> **摘要:** High-quality pre-training data is crutial for large language models, where quality captures factual reliability and semantic value, and diversity ensures broad coverage and distributional heterogeneity. Existing approaches typically rely on single or multiple-dimensional score-based selection. However, directly selecting top-scored data often degrades performance, and sampling from a broader range is required to recover results. The above non-monotonicity between dataset scores and downstream benchmark results reveals a fundamental bias: score-based methods collapse correlated dimensions, causing top-scored data to appear high-quality while systematically overlooking diversity. We argue that ensuring diversity requires decomposing correlated metrics into orthogonal feature dimensions, from which the top-scored data can be directly selected. Therefore, we proposed the Orthogonal Diversity-Aware Selection (ODiS) algorithm, which preserves both quality and diversity during data selection. First, ODiS evaluates data from multiple dimensions, covering language quality, knowledge quality, and comprehension difficulty. The multi-dimensional scores are then decorrelated via Principal Component Analysis (PCA), yielding orthogonal evaluation dimensions. For each dimension, a Roberta-based scorer is trained to regress the data onto PCA-projected scores, enabling scalable inference on large corpora. Finally, ODiS constructs the training dataset by selecting top-scored data within each orthogonal dimension, thereby ensuring both quality and diversity. Empirical results show that ODiS-selected data exhibit less than 2\% inter-dimension overlap, confirming orthogonality between dimensions. More importantly, models trained with ODiS-selected data significantly outperform other baselines on downstream benchmarks, highlighting the necessity of orthogonal, diversity-aware data selection for LLMs.
>
---
#### [new 076] When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文针对心理健康领域大语言模型（LLM）评估难题，提出MentalBench-100k与MentalAlign-70k两个大规模基准。通过真实对话数据与多维度评分框架，系统评估LLM生成内容的质量，并引入统计方法量化人工与机器评判的一致性与偏差，解决评估可靠性与可信度问题。**

- **链接: [http://arxiv.org/pdf/2510.19032v1](http://arxiv.org/pdf/2510.19032v1)**

> **作者:** Abeer Badawi; Elahe Rahimi; Md Tahmid Rahman Laskar; Sheri Grach; Lindsay Bertrand; Lames Danok; Jimmy Huang; Frank Rudzicz; Elham Dolatabadi
>
> **摘要:** Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: https://github.com/abeerbadawi/MentalBench/
>
---
#### [new 077] Sign Language Translation with Sentence Embedding Supervision
- **分类: cs.CL**

- **简介: 该论文研究无词典标注的手语翻译任务，旨在解决缺乏大规模人工标注词典的问题。提出用目标语句嵌入作为训练监督信号，无需手动标注，支持多语言，显著提升性能，达到无词典方法的新基准。**

- **链接: [http://arxiv.org/pdf/2510.19367v1](http://arxiv.org/pdf/2510.19367v1)**

> **作者:** Yasser Hamidullah; Josef van Genabith; Cristina España-Bonet
>
> **摘要:** State-of-the-art sign language translation (SLT) systems facilitate the learning process through gloss annotations, either in an end2end manner or by involving an intermediate step. Unfortunately, gloss labelled sign language data is usually not available at scale and, when available, gloss annotations widely differ from dataset to dataset. We present a novel approach using sentence embeddings of the target sentences at training time that take the role of glosses. The new kind of supervision does not need any manual annotation but it is learned on raw textual data. As our approach easily facilitates multilinguality, we evaluate it on datasets covering German (PHOENIX-2014T) and American (How2Sign) sign languages and experiment with mono- and multilingual sentence embeddings and translation systems. Our approach significantly outperforms other gloss-free approaches, setting the new state-of-the-art for data sets where glosses are not available and when no additional SLT datasets are used for pretraining, diminishing the gap between gloss-free and gloss-dependent systems.
>
---
#### [new 078] Blackbox Model Provenance via Palimpsestic Membership Inference
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型版权溯源问题，旨在验证黑箱衍生模型是否源自原始模型。提出基于“擦写记忆”特性的成员推理方法，在查询与观测两种场景下，通过检测输出与训练顺序的相关性，实现对模型来源的统计证明。**

- **链接: [http://arxiv.org/pdf/2510.19796v1](http://arxiv.org/pdf/2510.19796v1)**

> **作者:** Rohith Kuditipudi; Jing Huang; Sally Zhu; Diyi Yang; Christopher Potts; Percy Liang
>
> **摘要:** Suppose Alice trains an open-weight language model and Bob uses a blackbox derivative of Alice's model to produce text. Can Alice prove that Bob is using her model, either by querying Bob's derivative model (query setting) or from the text alone (observational setting)? We formulate this question as an independence testing problem--in which the null hypothesis is that Bob's model or text is independent of Alice's randomized training run--and investigate it through the lens of palimpsestic memorization in language models: models are more likely to memorize data seen later in training, so we can test whether Bob is using Alice's model using test statistics that capture correlation between Bob's model or text and the ordering of training examples in Alice's training run. If Alice has randomly shuffled her training data, then any significant correlation amounts to exactly quantifiable statistical evidence against the null hypothesis, regardless of the composition of Alice's training data. In the query setting, we directly estimate (via prompting) the likelihood Bob's model gives to Alice's training examples and order; we correlate the likelihoods of over 40 fine-tunes of various Pythia and OLMo base models ranging from 1B to 12B parameters with the base model's training data order, achieving a p-value on the order of at most 1e-8 in all but six cases. In the observational setting, we try two approaches based on estimating 1) the likelihood of Bob's text overlapping with spans of Alice's training examples and 2) the likelihood of Bob's text with respect to different versions of Alice's model we obtain by repeating the last phase (e.g., 1%) of her training run on reshuffled data. The second approach can reliably distinguish Bob's text from as little as a few hundred tokens; the first does not involve any retraining but requires many more tokens (several hundred thousand) to achieve high power.
>
---
#### [new 079] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划脱节的问题。通过整合建模与规划，利用无动作未来状态预测提升规划性能，实现协同状态-动作预测。引入动态并行令牌生成机制，仅用前视摄像头即达到领先效果。**

- **链接: [http://arxiv.org/pdf/2510.19654v1](http://arxiv.org/pdf/2510.19654v1)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [new 080] GaLLoP: Gradient-based Sparse Learning on Low-Magnitude Parameters
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GaLLoP，一种基于梯度与参数幅度的稀疏微调方法，旨在高效适配大模型于下游任务。通过选择梯度大、预训练值小的参数进行微调，平衡任务适应性与知识保留，有效缓解灾难性遗忘，提升性能稳定性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19778v1](http://arxiv.org/pdf/2510.19778v1)**

> **作者:** Anand Choudhary; Yasser Sulaıman; Lukas Mauch; Ghouthi Boukli Hacene; Fabien Cardinaux; Antoine Bosselut
>
> **摘要:** Sparse fine-tuning techniques adapt LLMs to downstream tasks by only tuning a sparse subset of model parameters. However, the effectiveness of sparse adaptation depends on optimally selecting the model parameters to be fine-tuned. In this work, we introduce a novel sparse fine-tuning technique named GaLLoP: Gradient-based Sparse Learning on Low-Magnitude Parameters, which fine-tunes only those model parameters which have the largest gradient magnitudes on downstream tasks and the smallest pre-trained magnitudes, intuitively prioritizing parameters that are highly task-relevant, but minimally disruptive to pre-trained knowledge. Our experimentation with LLaMA3 8B and Gemma 2B as base models shows that GaLLoP consistently improves or matches the in-distribution as well as out-of-distribution performance obtained via the usage of other leading parameter-efficient fine-tuning techniques, including LoRA, DoRA, and SAFT. Our analysis demonstrates that GaLLoP mitigates catastrophic forgetting and memorization of task data, as important pre-trained parameters remain unchanged, and stabilizes performance relative to other fine-tuning techniques, robustly generalizing across most random seeds.
>
---
#### [new 081] PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对详细图像描述的评估难题，提出PoSh框架，利用场景图引导大模型作为评判者，实现可复现、可解释的细粒度错误定位。构建新数据集DOCENT，验证PoSh优于现有指标，且能有效指导模型训练，推动视觉语言模型在复杂图像描述上的发展。**

- **链接: [http://arxiv.org/pdf/2510.19060v1](http://arxiv.org/pdf/2510.19060v1)**

> **作者:** Amith Ananthram; Elias Stengel-Eskin; Lorena A. Bradford; Julia Demarest; Adam Purvis; Keith Krut; Robert Stein; Rina Elster Pantalony; Mohit Bansal; Kathleen McKeown
>
> **备注:** 24 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
>
> **摘要:** While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
>
---
#### [new 082] Aligning Multilingual News for Stock Return Prediction
- **分类: q-fin.CP; cs.CL; J.4; I.2.7**

- **简介: 该论文针对多语言新闻在股票预测中因翻译损失语义细节的问题，提出基于最优传输的跨语言句子对齐方法。通过匹配英文与日文财经新闻，提升语义一致性，增强预测模型可解释性与准确性，显著提高交易策略的夏普比率。**

- **链接: [http://arxiv.org/pdf/2510.19203v1](http://arxiv.org/pdf/2510.19203v1)**

> **作者:** Yuntao Wu; Lynn Tao; Ing-Haw Cheng; Charles Martineau; Yoshio Nozawa; John Hull; Andreas Veneris
>
> **备注:** 6 pages, 4 tables, 2 figures, AI for Finance Symposium'25 Workshop at ICAIF'25
>
> **摘要:** News spreads rapidly across languages and regions, but translations may lose subtle nuances. We propose a method to align sentences in multilingual news articles using optimal transport, identifying semantically similar content across languages. We apply this method to align more than 140,000 pairs of Bloomberg English and Japanese news articles covering around 3500 stocks in Tokyo exchange over 2012-2024. Aligned sentences are sparser, more interpretable, and exhibit higher semantic similarity. Return scores constructed from aligned sentences show stronger correlations with realized stock returns, and long-short trading strategies based on these alignments achieve 10\% higher Sharpe ratios than analyzing the full text sample.
>
---
#### [new 083] Benchmarking On-Device Machine Learning on Apple Silicon with MLX
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究在Apple Silicon设备上部署机器学习模型的性能，聚焦于Transformer模型的推理延迟。为解决跨框架迁移难题，提出MLX-Transformers框架，实现从Hugging Face直接加载模型并转换至MLX格式。通过对比MLX与PyTorch在苹果设备与NVIDIA GPU上的表现，验证了MLX在高效、便捷地运行大模型方面的潜力。**

- **链接: [http://arxiv.org/pdf/2510.18921v1](http://arxiv.org/pdf/2510.18921v1)**

> **作者:** Oluwaseun A. Ajayi; Ogundepo Odunayo
>
> **备注:** 19 pages, 6 figures. Presented at the 6th Deep Learning Indaba (DLI 2024), Dakar, Senegal; non-archival presentation. Poster: https://storage.googleapis.com/indaba-public/Oluwaseun_Ajayi%20.pdf
>
> **摘要:** The recent widespread adoption of Large Language Models (LLMs) and machine learning in general has sparked research interest in exploring the possibilities of deploying these models on smaller devices such as laptops and mobile phones. This creates a need for frameworks and approaches that are capable of taking advantage of on-device hardware. The MLX framework was created to address this need. It is a framework optimized for machine learning (ML) computations on Apple silicon devices, facilitating easier research, experimentation, and prototyping. This paper presents a performance evaluation of MLX, focusing on inference latency of transformer models. We compare the performance of different transformer architecture implementations in MLX with their Pytorch counterparts. For this research we create a framework called MLX-transformers which includes different transformer implementations in MLX and downloads the model checkpoints in pytorch and converts it to the MLX format. By leveraging the advanced architecture and capabilities of Apple Silicon, MLX-Transformers enables seamless execution of transformer models directly sourced from Hugging Face, eliminating the need for checkpoint conversion often required when porting models between frameworks. Our study benchmarks different transformer models on two Apple Silicon macbook devices against an NVIDIA CUDA GPU. Specifically, we compare the inference latency performance of models with the same parameter sizes and checkpoints. We evaluate the performance of BERT, RoBERTa, and XLM-RoBERTa models, with the intention of extending future work to include models of different modalities, thus providing a more comprehensive assessment of MLX's capabilities. The results highlight MLX's potential in enabling efficient and more accessible on-device ML applications within Apple's ecosystem.
>
---
#### [new 084] A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在临床试验报告合规性评估中的认知能力，聚焦于CONSORT标准的遵循情况。通过对比不同提示策略下模型的行为与元认知表现，揭示其推理差异与局限，旨在提升医疗AI的可解释性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.19139v1](http://arxiv.org/pdf/2510.19139v1)**

> **作者:** Sohyeon Jeon; Hyung-Chul Lee
>
> **摘要:** Despite the rapid expansion of Large Language Models (LLMs) in healthcare, the ability of these systems to assess clinical trial reporting according to CONSORT standards remains unclear, particularly with respect to their cognitive and reasoning strategies. This study applies a behavioral and metacognitive analytic approach with expert-validated data, systematically comparing two representative LLMs under three prompt conditions. Clear differences emerged in how the models approached various CONSORT items, and prompt types, including shifts in reasoning style, explicit uncertainty, and alternative interpretations shaped response patterns. Our results highlight the current limitations of these systems in clinical compliance automation and underscore the importance of understanding their cognitive adaptations and strategic behavior in developing more explainable and reliable medical AI.
>
---
#### [new 085] ColorAgent: Building A Robust, Personalized, and Interactive OS Agent
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文提出ColorAgent，一个面向操作系统的智能代理，旨在实现长期、稳健的交互与个性化主动服务。针对OS代理难以持续执行复杂任务及缺乏用户感知的问题，研究通过分步强化学习与多智能体框架提升鲁棒性，并引入意图识别与主动交互机制。在安卓基准测试中表现优异，达到新SOTA，但指出当前评估体系需完善。**

- **链接: [http://arxiv.org/pdf/2510.19386v1](http://arxiv.org/pdf/2510.19386v1)**

> **作者:** Ning Li; Qiqiang Lin; Zheng Wu; Xiaoyun Mo; Weiming Zhang; Yin Zhao; Xiangmou Qu; Jiamu Zhou; Jun Wang; Congmin Zheng; Yuanyi Song; Hongjiang Chen; Heyuan Huang; Jihong Wang; Jiaxin Yin; Jingwei Yu; Junwei Liao; Qiuying Peng; Xingyu Lou; Jun Wang; Weiwen Liu; Zhuosheng Zhang; Weinan Zhang
>
> **摘要:** With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security. Our code is available at https://github.com/MadeAgents/mobile-use.
>
---
#### [new 086] Towards Better Health Conversations: The Benefits of Context-seeking
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文研究对话式AI在健康咨询中的应用，旨在解决LLM回答不准确、缺乏个性化的问题。通过四组混合方法研究（N=163），发现主动追问上下文能显著提升交互质量。据此开发“路径引导AI”，在实验中被证明更助人、相关且定制化，验证了主动获取背景信息的有效性。**

- **链接: [http://arxiv.org/pdf/2510.18880v1](http://arxiv.org/pdf/2510.18880v1)**

> **作者:** Rory Sayres; Yuexing Hao; Abbi Ward; Amy Wang; Beverly Freeman; Serena Zhan; Diego Ardila; Jimmy Li; I-Ching Lee; Anna Iurchenko; Siyi Kou; Kartikeya Badola; Jimmy Hu; Bhawesh Kumar; Keith Johnson; Supriya Vijay; Justin Krogue; Avinatan Hassidim; Yossi Matias; Dale R. Webster; Sunny Virmani; Yun Liu; Quang Duong; Mike Schaekermann
>
> **摘要:** Navigating health questions can be daunting in the modern information landscape. Large language models (LLMs) may provide tailored, accessible information, but also risk being inaccurate, biased or misleading. We present insights from 4 mixed-methods studies (total N=163), examining how people interact with LLMs for their own health questions. Qualitative studies revealed the importance of context-seeking in conversational AIs to elicit specific details a person may not volunteer or know to share. Context-seeking by LLMs was valued by participants, even if it meant deferring an answer for several turns. Incorporating these insights, we developed a "Wayfinding AI" to proactively solicit context. In a randomized, blinded study, participants rated the Wayfinding AI as more helpful, relevant, and tailored to their concerns compared to a baseline AI. These results demonstrate the strong impact of proactive context-seeking on conversational dynamics, and suggest design patterns for conversational AI to help navigate health topics.
>
---
#### [new 087] Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对长文本推理中的高计算与存储开销问题，提出Ring-linear模型系列，采用线性注意力与Softmax注意力混合架构，显著降低推理成本。通过优化注意力比例与自研FP8算子库，提升训练效率50%，实现高效稳定且性能领先的长序列推理。**

- **链接: [http://arxiv.org/pdf/2510.19338v1](http://arxiv.org/pdf/2510.19338v1)**

> **作者:** Ling Team; Bin Han; Caizhi Tang; Chen Liang; Donghao Zhang; Fan Yuan; Feng Zhu; Jie Gao; Jingyu Hu; Longfei Li; Meng Li; Mingyang Zhang; Peijie Jiang; Peng Jiao; Qian Zhao; Qingyuan Yang; Wenbo Shen; Xinxing Yang; Yalin Zhang; Yankun Ren; Yao Zhao; Yibo Cao; Yixuan Sun; Yue Zhang; Yuchen Fang; Zibin Lin; Zixuan Cheng; Jun Zhou
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks.
>
---
#### [new 088] Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出AutoPage，一个用于将科研论文自动转化为交互式网页的多智能体协作系统。针对人工构建网页耗时费力的问题，通过分步协同与校验机制，实现高效、低成本（<0.1美元）生成高质量网页，首次构建了相关基准PageBench。**

- **链接: [http://arxiv.org/pdf/2510.19600v1](http://arxiv.org/pdf/2510.19600v1)**

> **作者:** Qianli Ma; Siyu Wang; Yilin Chen; Yinhao Tang; Yixiang Yang; Chang Guo; Bingjie Gao; Zhening Xing; Yanan Sun; Zhipeng Zhang
>
> **摘要:** In the quest for scientific progress, communicating research is as vital as the discovery itself. Yet, researchers are often sidetracked by the manual, repetitive chore of building project webpages to make their dense papers accessible. While automation has tackled static slides and posters, the dynamic, interactive nature of webpages has remained an unaddressed challenge. To bridge this gap, we reframe the problem, arguing that the solution lies not in a single command, but in a collaborative, hierarchical process. We introduce $\textbf{AutoPage}$, a novel multi-agent system that embodies this philosophy. AutoPage deconstructs paper-to-page creation into a coarse-to-fine pipeline from narrative planning to multimodal content generation and interactive rendering. To combat AI hallucination, dedicated "Checker" agents verify each step against the source paper, while optional human checkpoints ensure the final product aligns perfectly with the author's vision, transforming the system from a mere tool into a powerful collaborative assistant. To rigorously validate our approach, we also construct $\textbf{PageBench}$, the first benchmark for this new task. Experiments show AutoPage not only generates high-quality, visually appealing pages but does so with remarkable efficiency in under 15 minutes for less than \$0.1. Code and dataset will be released at $\href{https://mqleet.github.io/AutoPage_ProjectPage/}{Webpage}$.
>
---
#### [new 089] LLM Unlearning with LLM Beliefs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大模型遗忘任务，针对现有方法因“挤压效应”导致敏感内容仅被替换而非真正遗忘的问题。提出基于模型信念的自举框架（BS），通过抑制高置信度生成来消除概率质量重分配，实现更彻底的遗忘并保持模型可用性。**

- **链接: [http://arxiv.org/pdf/2510.19422v1](http://arxiv.org/pdf/2510.19422v1)**

> **作者:** Kemou Li; Qizhou Wang; Yue Wang; Fengpeng Li; Jun Liu; Bo Han; Jiantao Zhou
>
> **摘要:** Large language models trained on vast corpora inherently risk memorizing sensitive or harmful content, which may later resurface in their outputs. Prevailing unlearning methods generally rely on gradient ascent and its variants to lower the probability of specific target responses. However, we find that this strategy induces a critical side effect: probability mass is redistributed into high-likelihood regions, often corresponding to semantically related rephrasings of the targets. We refer to this as the squeezing effect, which explains why many methods yield merely spurious unlearning, a problem further obscured by automated metrics (e.g., ROUGE, truth ratio) that misreport actual success. To address this, we propose a bootstrapping (BS) framework that explicitly links the squeezing effect with the model's own high-confidence generations, namely its model beliefs. Since model beliefs inherently capture the very high-likelihood regions where probability mass is squeezed, incorporating them into the unlearning objective directly counters the squeezing effect. By jointly suppressing both target responses and model beliefs, BS-T (token) attenuates high-probability tokens, whereas BS-S (sequence) removes entire high-confidence generations, together achieving more thorough forgetting while preserving utility. Extensive experiments across diverse benchmarks with various model families confirm the effectiveness of our approach.
>
---
#### [new 090] StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出StutterZero与StutterFormer，首个端到端波形到波形的口吃语音转写与修正模型，直接将口吃语音转为流利语音并生成文本。针对现有方法分离转写与重建导致误差放大问题，通过联合建模提升准确率，在多个基准上显著降低词错误率并提升语义相似度。**

- **链接: [http://arxiv.org/pdf/2510.18938v1](http://arxiv.org/pdf/2510.18938v1)**

> **作者:** Qianheng Xu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems.
>
---
#### [new 091] Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Pico-Banana-400K数据集，用于文本引导图像编辑任务。针对现有数据集规模小、质量低、缺乏真实图像的问题，利用纳米香蕉模型生成40万张高质量编辑对，涵盖单轮、多轮、偏好与指令重写等复杂场景，提升模型训练与评估的多样性与真实性。**

- **链接: [http://arxiv.org/pdf/2510.19808v1](http://arxiv.org/pdf/2510.19808v1)**

> **作者:** Yusu Qian; Eli Bocek-Rivele; Liangchen Song; Jialing Tong; Yinfei Yang; Jiasen Lu; Wenze Hu; Zhe Gan
>
> **摘要:** Recent advances in multimodal models have demonstrated remarkable text-guided image editing capabilities, with systems like GPT-4o and Nano-Banana setting new benchmarks. However, the research community's progress remains constrained by the absence of large-scale, high-quality, and openly accessible datasets built from real images. We introduce Pico-Banana-400K, a comprehensive 400K-image dataset for instruction-based image editing. Our dataset is constructed by leveraging Nano-Banana to generate diverse edit pairs from real photographs in the OpenImages collection. What distinguishes Pico-Banana-400K from previous synthetic datasets is our systematic approach to quality and diversity. We employ a fine-grained image editing taxonomy to ensure comprehensive coverage of edit types while maintaining precise content preservation and instruction faithfulness through MLLM-based quality scoring and careful curation. Beyond single turn editing, Pico-Banana-400K enables research into complex editing scenarios. The dataset includes three specialized subsets: (1) a 72K-example multi-turn collection for studying sequential editing, reasoning, and planning across consecutive modifications; (2) a 56K-example preference subset for alignment research and reward model training; and (3) paired long-short editing instructions for developing instruction rewriting and summarization capabilities. By providing this large-scale, high-quality, and task-rich resource, Pico-Banana-400K establishes a robust foundation for training and benchmarking the next generation of text-guided image editing models.
>
---
#### [new 092] NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对参数高效微调（PEFT）中记忆效率与表达能力的权衡问题，提出NeuroAda方法。通过选择重要参数并引入可更新的旁路连接，在仅更新极少量参数（≤0.02%）的情况下实现高精度微调，显著降低内存占用（最多减少60%），在23+任务上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2510.18940v1](http://arxiv.org/pdf/2510.18940v1)**

> **作者:** Zhi Zhang; Yixian Shen; Congfeng Cao; Ekaterina Shutova
>
> **摘要:** Existing parameter-efficient fine-tuning (PEFT) methods primarily fall into two categories: addition-based and selective in-situ adaptation. The former, such as LoRA, introduce additional modules to adapt the model to downstream tasks, offering strong memory efficiency. However, their representational capacity is often limited, making them less suitable for fine-grained adaptation. In contrast, the latter directly fine-tunes a carefully chosen subset of the original model parameters, allowing for more precise and effective adaptation, but at the cost of significantly increased memory consumption. To reconcile this trade-off, we propose NeuroAda, a novel PEFT method that enables fine-grained model finetuning while maintaining high memory efficiency. Our approach first identifies important parameters (i.e., connections within the network) as in selective adaptation, and then introduces bypass connections for these selected parameters. During finetuning, only the bypass connections are updated, leaving the original model parameters frozen. Empirical results on 23+ tasks spanning both natural language generation and understanding demonstrate that NeuroAda achieves state-of-the-art performance with as little as $\leq \textbf{0.02}\%$ trainable parameters, while reducing CUDA memory usage by up to 60%. We release our code here: https://github.com/FightingFighting/NeuroAda.git.
>
---
#### [new 093] olmOCR 2: Unit Test Rewards for Document OCR
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出olmOCR 2，用于将数字化印刷文档（如PDF）转换为结构化文本。针对复杂布局下OCR精度不足的问题，采用基于可验证奖励的强化学习，利用大量二元单元测试作为奖励信号，训练7B视觉语言模型。通过合成数据生成管道提升测试覆盖度，显著提升数学公式、表格和多栏布局的识别性能。**

- **链接: [http://arxiv.org/pdf/2510.19817v1](http://arxiv.org/pdf/2510.19817v1)**

> **作者:** Jake Poznanski; Luca Soldaini; Kyle Lo
>
> **备注:** https://olmocr.allen.ai/
>
> **摘要:** We present olmOCR 2, the latest in our family of powerful OCR systems for converting digitized print documents, like PDFs, into clean, naturally ordered plain text. olmOCR 2 is powered by olmOCR-2-7B-1025, a specialized, 7B vision language model (VLM) trained using reinforcement learning with verifiable rewards (RLVR), where our rewards are a diverse set of binary unit tests. To scale unit test creation, we develop a pipeline for generating synthetic documents with diverse and challenging layouts, known ground-truth HTML source code, and extracted test cases. We show that RL training on these test cases results in state-of-the-art performance on olmOCR-Bench, our English-language OCR benchmark, with the largest improvements in math formula conversion, table parsing, and multi-column layouts compared to previous versions. We release our model, data and code under permissive open licenses.
>
---
#### [new 094] OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出OpenGuardrails，一个开源的上下文感知AI安全防护平台。针对LLM应用中的内容安全、模型操控攻击和数据泄露问题，构建了统一大模型检测与轻量级命名实体识别结合的防护体系，支持私有化部署，实现多语言下SOTA的安全检测效果。**

- **链接: [http://arxiv.org/pdf/2510.19169v1](http://arxiv.org/pdf/2510.19169v1)**

> **作者:** Thomas Wang; Haowen Li
>
> **摘要:** As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use.
>
---
#### [new 095] [De|Re]constructing VLMs' Reasoning in Counting
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）在计数任务中的推理能力，针对其在对象数量、空间布局和干扰项影响下的表现不佳问题，通过控制实验与层分析揭示错误源于最后一层表示到输出空间的映射错误。提出仅微调输出层的方法，提升准确率最高达21%，并在真实数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.19555v1](http://arxiv.org/pdf/2510.19555v1)**

> **作者:** Simone Alghisi; Gabriel Roccabruna; Massimo Rizzoli; Seyed Mahed Mousavi; Giuseppe Riccardi
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Vision-Language Models (VLMs) have recently gained attention due to their competitive performance on multiple downstream tasks, achieved by following user-input instructions. However, VLMs still exhibit several limitations in visual reasoning, such as difficulties in identifying relations (e.g., spatial, temporal, and among objects), understanding temporal sequences (e.g., frames), and counting objects. In this work, we go beyond score-level benchmark evaluations of VLMs by investigating the underlying causes of their failures and proposing a targeted approach to improve their reasoning capabilities. We study the reasoning skills of seven state-of-the-art VLMs in the counting task under controlled experimental conditions. Our experiments show that VLMs are highly sensitive to the number and type of objects, their spatial arrangement, and the co-occurrence of distractors. A layer-wise analysis reveals that errors are due to incorrect mapping of the last-layer representation into the output space. Our targeted training shows that fine-tuning just the output layer improves accuracy by up to 21%. We corroborate these findings by achieving consistent improvements on real-world datasets.
>
---
#### [new 096] BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型强化学习中的离策略训练不稳定问题，提出BAPO方法。通过动态调整裁剪边界，平衡正负样本贡献，抑制熵衰减，提升优化稳定性与探索能力。在多个基准上实现高效、稳定训练，显著优于现有开源与闭源模型。**

- **链接: [http://arxiv.org/pdf/2510.18927v1](http://arxiv.org/pdf/2510.18927v1)**

> **作者:** Zhiheng Xi; Xin Guo; Yang Nan; Enyu Zhou; Junrui Shen; Wenxiang Chen; Jiaqi Liu; Jixuan Huang; Zhihao Zhang; Honglin Guo; Xun Deng; Zhikai Lei; Miao Zheng; Guoteng Wang; Shuo Zhang; Peng Sun; Rui Zheng; Hang Yan; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Preprint
>
> **摘要:** Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.
>
---
#### [new 097] HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出HSCodeComp，首个面向深度搜索代理的专家级电商基准，用于评估其在复杂层级规则下的推理能力。任务为根据产品描述预测10位海关编码（HSCode），解决现有基准忽略规则应用的问题。基于真实电商数据构建，包含632个专家标注样本，实验揭示当前模型性能远低于人类专家。**

- **链接: [http://arxiv.org/pdf/2510.19631v1](http://arxiv.org/pdf/2510.19631v1)**

> **作者:** Yiqian Yang; Tian Lan; Qianghuai Jia; Li Zhu; Hui Jiang; Hang Zhu; Longyue Wang; Weihua Luo; Kaifu Zhang
>
> **摘要:** Effective deep search agents must not only access open-domain and domain-specific knowledge but also apply complex rules-such as legal clauses, medical manuals and tariff rules. These rules often feature vague boundaries and implicit logic relationships, making precise application challenging for agents. However, this critical capability is largely overlooked by current agent benchmarks. To fill this gap, we introduce HSCodeComp, the first realistic, expert-level e-commerce benchmark designed to evaluate deep search agents in hierarchical rule application. In this task, the deep reasoning process of agents is guided by these rules to predict 10-digit Harmonized System Code (HSCode) of products with noisy but realistic descriptions. These codes, established by the World Customs Organization, are vital for global supply chain efficiency. Built from real-world data collected from large-scale e-commerce platforms, our proposed HSCodeComp comprises 632 product entries spanning diverse product categories, with these HSCodes annotated by several human experts. Extensive experimental results on several state-of-the-art LLMs, open-source, and closed-source agents reveal a huge performance gap: best agent achieves only 46.8% 10-digit accuracy, far below human experts at 95.0%. Besides, detailed analysis demonstrates the challenges of hierarchical rule application, and test-time scaling fails to improve performance further.
>
---
#### [new 098] The Zero-Step Thinking: An Empirical Study of Mode Selection as Harder Early Exit in Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究推理模型中的“零步思考”问题，即在无实际推理前提下提前选择长链或短链思维模式。针对模式选择比早期退出更难的问题，通过实证分析九种基线方法，发现基于提示的方法表现差，依赖内部信息的方法虽优但不稳定，揭示了现有方法在信息有限场景下的不足。**

- **链接: [http://arxiv.org/pdf/2510.19176v1](http://arxiv.org/pdf/2510.19176v1)**

> **作者:** Yuqiao Tan; Shizhu He; Kang Liu; Jun Zhao
>
> **备注:** Accepted by NeurIPS'25 Efficient Reasoning Workshop
>
> **摘要:** Reasoning models have demonstrated exceptional performance in tasks such as mathematics and logical reasoning, primarily due to their ability to engage in step-by-step thinking during the reasoning process. However, this often leads to overthinking, resulting in unnecessary computational overhead. To address this issue, Mode Selection aims to automatically decide between Long-CoT (Chain-of-Thought) or Short-CoT by utilizing either a Thinking or NoThinking mode. Simultaneously, Early Exit determines the optimal stopping point during the iterative reasoning process. Both methods seek to reduce the computational burden. In this paper, we first identify Mode Selection as a more challenging variant of the Early Exit problem, as they share similar objectives but differ in decision timing. While Early Exit focuses on determining the best stopping point for concise reasoning at inference time, Mode Selection must make this decision at the beginning of the reasoning process, relying on pre-defined fake thoughts without engaging in an explicit reasoning process, referred to as zero-step thinking. Through empirical studies on nine baselines, we observe that prompt-based approaches often fail due to their limited classification capabilities when provided with minimal hand-crafted information. In contrast, approaches that leverage internal information generally perform better across most scenarios but still exhibit issues with stability. Our findings indicate that existing methods relying solely on the information provided by models are insufficient for effectively addressing Mode Selection in scenarios with limited information, highlighting the ongoing challenges of this task. Our code is available at https://github.com/Trae1ounG/Zero_Step_Thinking.
>
---
## 更新

#### [replaced 001] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.19878v3](http://arxiv.org/pdf/2503.19878v3)**

> **作者:** Nengbo Wang; Xiaotian Han; Jagdip Singh; Jing Ma; Vipin Chaudhary
>
> **备注:** Accepted at Findings of ACL 2025
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing (NLP), particularly through Retrieval-Augmented Generation (RAG), which enhances LLM capabilities by integrating external knowledge. However, traditional RAG systems face critical limitations, including disrupted contextual integrity due to text chunking, and over-reliance on semantic similarity for retrieval. To address these issues, we propose CausalRAG, a novel framework that incorporates causal graphs into the retrieval process. By constructing and tracing causal relationships, CausalRAG preserves contextual continuity and improves retrieval precision, leading to more accurate and interpretable responses. We evaluate CausalRAG against regular RAG and graph-based RAG approaches, demonstrating its superiority across several metrics. Our findings suggest that grounding retrieval in causal reasoning provides a promising approach to knowledge-intensive tasks.
>
---
#### [replaced 002] Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16062v2](http://arxiv.org/pdf/2510.16062v2)**

> **作者:** Guiyao Tie; Zenghui Yuan; Zeli Zhao; Chaoran Hu; Tianhe Gu; Ruihang Zhang; Sizhe Zhang; Junran Wu; Xiaoyue Tu; Ming Jin; Qingsong Wen; Lixing Chen; Pan Zhou; Lichao Sun
>
> **备注:** 47 pages, 25 figures, 10 tables
>
> **摘要:** Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
>
---
#### [replaced 003] LongCodeBench: Evaluating Coding LLMs at 1M Context Windows
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07897v3](http://arxiv.org/pdf/2505.07897v3)**

> **作者:** Stefano Rando; Luca Romani; Alessio Sampieri; Luca Franco; John Yang; Yuta Kyuragi; Fabio Galasso; Tatsunori Hashimoto
>
> **摘要:** Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5. The LCB dataset is available publicly at https://huggingface.co/datasets/Steefano/LCB and the codebase to replicate the work on this paper at https://github.com/Zteefano/long-code-bench.
>
---
#### [replaced 004] Flow-SLM: Joint Learning of Linguistic and Acoustic Information for Spoken Language Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09350v3](http://arxiv.org/pdf/2508.09350v3)**

> **作者:** Ju-Chieh Chou; Jiawei Zhou; Karen Livescu
>
> **备注:** ASRU 2025. Project page: https://jjery2243542.github.io/flowslm.github.io/
>
> **摘要:** Textless spoken language models (SLMs) are generative models of speech that do not rely on text supervision. Most textless SLMs learn to predict the next semantic token, a discrete representation of linguistic content, and rely on a separate vocoder to add acoustic information to the generated speech. Such models have no access to acoustic context and no built-in control over acoustic details. In this work, we propose to jointly model linguistic and acoustic information by generating semantic tokens and a continuous real-valued representation of the acoustic frame. We use a flow-matching objective to predict the continuous vector conditioned on the semantic tokens. We study the design space of this approach and find that predicting multiple future semantic tokens helps preserve linguistic information. Our approach achieves comparable performance to existing models in terms of linguistic likelihood benchmarks, while providing better acoustic detail in prompted generation.
>
---
#### [replaced 005] Can Large Language Models be Effective Online Opinion Miners?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15695v3](http://arxiv.org/pdf/2505.15695v3)**

> **作者:** Ryang Heo; Yongsik Seo; Junseong Lee; Dongha Lee
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** The surge of user-generated online content presents a wealth of insights into customer preferences and market trends. However, the highly diverse, complex, and context-rich nature of such contents poses significant challenges to traditional opinion mining approaches. To address this, we introduce Online Opinion Mining Benchmark (OOMB), a novel dataset and evaluation protocol designed to assess the ability of large language models (LLMs) to mine opinions effectively from diverse and intricate online environments. OOMB provides extensive (entity, feature, opinion) tuple annotations and a comprehensive opinion-centric summary that highlights key opinion topics within each content, thereby enabling the evaluation of both the extractive and abstractive capabilities of models. Through our proposed benchmark, we conduct a comprehensive analysis of which aspects remain challenging and where LLMs exhibit adaptability, to explore whether they can effectively serve as opinion miners in realistic online scenarios. This study lays the foundation for LLM-based opinion mining and discusses directions for future research in this field.
>
---
#### [replaced 006] NAACL2025 Tutorial: Adaptation of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.03931v3](http://arxiv.org/pdf/2504.03931v3)**

> **作者:** Zixuan Ke; Yifei Ming; Shafiq Joty
>
> **备注:** NAACL2025 Tutorial
>
> **摘要:** This tutorial on adaptation of LLMs is designed to address the growing demand for models that go beyond the static capabilities of generic LLMs by providing an overview of dynamic, domain-specific, and task-adaptive LLM adaptation techniques. While general LLMs have demonstrated strong generalization across a variety of tasks, they often struggle to perform well in specialized domains such as finance, healthcare, and code generation for underrepresented languages. Additionally, their static nature limits their ability to evolve with the changing world, and they are often extremely large in size, making them impractical and costly to deploy at scale. As a result, the adaptation of LLMs has drawn much attention since the birth of LLMs and is of core importance, both for industry, which focuses on serving its targeted users, and academia, which can greatly benefit from small but powerful LLMs. To address this gap, this tutorial aims to provide an overview of the LLM adaptation techniques. We start with an introduction to LLM adaptation, from both the data perspective and the model perspective. We then emphasize how the evaluation metrics and benchmarks are different from other techniques. After establishing the problems, we explore various adaptation techniques. We categorize adaptation techniques into two main families. The first is parametric knowledge adaptation, which focuses on updating the parametric knowledge within LLMs. Additionally, we will discuss real-time adaptation techniques, including model editing, which allows LLMs to be updated dynamically in production environments. The second kind of adaptation is semi-parametric knowledge adaptation, where the goal is to update LLM parameters to better leverage external knowledge or tools through techniques like retrieval-augmented generation (RAG) and agent-based systems.
>
---
#### [replaced 007] Demystifying Domain-adaptive Post-training for Financial LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04961v4](http://arxiv.org/pdf/2501.04961v4)**

> **作者:** Zixuan Ke; Yifei Ming; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **备注:** EMNLP 2025 (Oral, ARR best paper nomination)
>
> **摘要:** Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs
>
---
#### [replaced 008] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19955v3](http://arxiv.org/pdf/2505.19955v3)**

> **作者:** Hui Chen; Miao Xiong; Yujie Lu; Wei Han; Ailin Deng; Yufei He; Jiaying Wu; Yibo Li; Yue Liu; Bryan Hooi
>
> **备注:** 49 pages, 9 figures. Accepted by NeurIPS 2025 D&B Track
>
> **摘要:** Recent advancements in AI agents have demonstrated their growing potential to drive and support scientific discovery. In this work, we introduce MLR-Bench, a comprehensive benchmark for evaluating AI agents on open-ended machine learning research. MLR-Bench includes three key components: (1) 201 research tasks sourced from NeurIPS, ICLR, and ICML workshops covering diverse ML topics; (2) MLR-Judge, an automated evaluation framework combining LLM-based reviewers with carefully designed review rubrics to assess research quality; and (3) MLR-Agent, a modular agent scaffold capable of completing research tasks through four stages: idea generation, proposal formulation, experimentation, and paper writing. Our framework supports both stepwise assessment across these distinct research stages, and end-to-end evaluation of the final research paper. We then use MLR-Bench to evaluate six frontier LLMs and an advanced coding agent, finding that while LLMs are effective at generating coherent ideas and well-structured papers, current coding agents frequently (e.g., in 80% of the cases) produce fabricated or invalidated experimental results--posing a major barrier to scientific reliability. We validate MLR-Judge through human evaluation, showing high agreement with expert reviewers, supporting its potential as a scalable tool for research evaluation. We open-source MLR-Bench to help the community benchmark, diagnose, and improve AI research agents toward trustworthy and transparent scientific discovery.
>
---
#### [replaced 009] GUARD: Guided Unlearning and Retention via Data Attribution for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10946v2](http://arxiv.org/pdf/2506.10946v2)**

> **作者:** Peizhi Niu; Evelyn Ma; Huiting Zhou; Duo Zhou; Huan Zhang; S. Rasoul Etesami; Olgica Milenkovic
>
> **摘要:** Unlearning in large language models is becoming increasingly important due to regulatory compliance, copyright protection, and privacy concerns. However, a key challenge in LLM unlearning is unintended forgetting, where the removal of specific data inadvertently impairs the utility of the model and its retention of valuable, desired information. While prior work has primarily focused on architectural innovations, the influence of data-level factors on unlearning performance remains underexplored. As a result, existing methods often suffer from degraded retention when forgetting high-impact data. To address this problem, we propose GUARD, a novel framework for Guided Unlearning And Retention via Data attribution. At its core, GUARD introduces a lightweight proxy data attribution metric tailored for LLM unlearning, which quantifies the alignment between the Forget and Retain sets while remaining computationally efficient. Building on this, we design a novel unlearning objective that assigns adaptive, nonuniform unlearning weights to samples, inversely proportional to their proxy attribution scores. Through such a reallocation of unlearning power, GUARD mitigates unintended retention loss. We also provide rigorous theoretical guarantees that GUARD significantly improves retention while maintaining forgetting metrics comparable to prior methods. Extensive experiments on the TOFU and MUSE benchmarks across multiple LLM architectures demonstrate that GUARD reduces utility sacrifice on the TOFU Retain Set by up to 194.92 percent in terms of Truth Ratio when forgetting 10 percent of the training data, and improves knowledge retention on the MUSE NEWS Retain Set by 16.20 percent, with comparable or very moderate increases in privacy loss compared to state-of-the-art methods.
>
---
#### [replaced 010] Measuring Data Science Automation: A Survey of Evaluation Tools for AI Assistants and Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08800v2](http://arxiv.org/pdf/2506.08800v2)**

> **作者:** Irene Testini; José Hernández-Orallo; Lorenzo Pacchiardi
>
> **备注:** Published in Transactions of Machine Learning Research (TMLR), 10/2025 https://openreview.net/forum?id=MB0TCLfLn1
>
> **摘要:** Data science aims to extract insights from data to support decision-making processes. Recently, Large Language Models (LLMs) have been increasingly used as assistants for data science, by suggesting ideas, techniques and small code snippets, or for the interpretation of results and reporting. Proper automation of some data-science activities is now promised by the rise of LLM agents, i.e., AI systems powered by an LLM equipped with additional affordances--such as code execution and knowledge bases--that can perform self-directed actions and interact with digital environments. In this paper, we survey the evaluation of LLM assistants and agents for data science. We find (1) a dominant focus on a small subset of goal-oriented activities, largely ignoring data management and exploratory activities; (2) a concentration on pure assistance or fully autonomous agents, without considering intermediate levels of human-AI collaboration; and (3) an emphasis on human substitution, therefore neglecting the possibility of higher levels of automation thanks to task transformation.
>
---
#### [replaced 011] Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15061v2](http://arxiv.org/pdf/2510.15061v2)**

> **作者:** Samuel Paech; Allen Roush; Judah Goldfeder; Ravid Shwartz-Ziv
>
> **备注:** 11 pages + appendices, 16 figures
>
> **摘要:** Widespread LLM adoption has introduced characteristic repetitive phraseology, termed "slop," which degrades output quality and makes AI-generated text immediately recognizable. We present Antislop, a comprehensive framework providing tools to both detect and eliminate these overused patterns. Our approach combines three innovations: (1) The Antislop Sampler, which uses backtracking to suppress unwanted strings at inference time without destroying vocabulary; (2) An automated pipeline that profiles model-specific slop against human baselines and generates training data; (3) Final Token Preference Optimization (FTPO), a novel fine-tuning method that operates on individual tokens, surgically adjusting logits wherever a banned pattern has appeared in an inference trace. We demonstrate that some slop patterns appear over 1,000x more frequently in LLM output than human text. The Antislop Sampler successfully suppresses 8,000+ patterns while maintaining quality, whereas token banning becomes unusable at just 2,000. Most importantly, FTPO achieves 90% slop reduction while maintaining or improving performance in cross-domain evals including GSM8K, MMLU, and creative writing tasks. In contrast, DPO suffers significant degradation in writing quality and lexical diversity despite achieving weaker suppression. We release all code and results under MIT license: https://github.com/sam-paech/auto-antislop.
>
---
#### [replaced 012] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18876v2](http://arxiv.org/pdf/2510.18876v2)**

> **作者:** Haochen Wang; Yuhao Wang; Tao Zhang; Yikang Zhou; Yanwei Li; Jiacong Wang; Jiani Zheng; Ye Tian; Jiahao Meng; Zilong Huang; Guangcan Mai; Anran Wang; Yunhai Tong; Zhuochen Wang; Xiangtai Li; Zhaoxiang Zhang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos.
>
---
#### [replaced 013] Evaluating NLP Embedding Models for Handling Science-Specific Symbolic Expressions in Student Texts
- **分类: cs.CL; cs.AI; physics.ed-ph**

- **链接: [http://arxiv.org/pdf/2505.17950v2](http://arxiv.org/pdf/2505.17950v2)**

> **作者:** Tom Bleckmann; Paul Tschisgale
>
> **摘要:** In recent years, natural language processing (NLP) has become integral to educational data mining, particularly in the analysis of student-generated language products. For research and assessment purposes, so-called embedding models are typically employed to generate numeric representations of text that capture its semantic content for use in subsequent quantitative analyses. Yet when it comes to science-related language, symbolic expressions such as equations and formulas introduce challenges that current embedding models struggle to address. Existing research studies and practical applications often either overlook these challenges or remove symbolic expressions altogether, potentially leading to biased research findings and diminished performance of practical applications. This study therefore explores how contemporary embedding models differ in their capability to process and interpret science-related symbolic expressions. To this end, various embedding models are evaluated using physics-specific symbolic expressions drawn from authentic student responses, with performance assessed via two approaches: 1) similarity-based analyses and 2) integration into a machine learning pipeline. Our findings reveal significant differences in model performance, with OpenAI's GPT-text-embedding-3-large outperforming all other examined models, though its advantage over other models was moderate rather than decisive. Overall, this study underscores the importance for educational data mining researchers and practitioners of carefully selecting NLP embedding models when working with science-related language products that include symbolic expressions. The code and (partial) data are available at https://doi.org/10.17605/OSF.IO/6XQVG.
>
---
#### [replaced 014] Hire Your Anthropologist! Rethinking Culture Benchmarks Through an Anthropological Lens
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.05931v2](http://arxiv.org/pdf/2510.05931v2)**

> **作者:** Mai AlKhamissi; Yunze Xiao; Badr AlKhamissi; Mona Diab
>
> **备注:** 12 pages; 2 figures; First two author contributed equally
>
> **摘要:** Cultural evaluation of large language models has become increasingly important, yet current benchmarks often reduce culture to static facts or homogeneous values. This view conflicts with anthropological accounts that emphasize culture as dynamic, historically situated, and enacted in practice. To analyze this gap, we introduce a four-part framework that categorizes how benchmarks frame culture, such as knowledge, preference, performance, or bias. Using this lens, we qualitatively examine 20 cultural benchmarks and identify six recurring methodological issues, including treating countries as cultures, overlooking within-culture diversity, and relying on oversimplified survey formats. Drawing on established anthropological methods, we propose concrete improvements: incorporating real-world narratives and scenarios, involving cultural communities in design and validation, and evaluating models in context rather than isolation. Our aim is to guide the development of cultural benchmarks that go beyond static recall tasks and more accurately capture the responses of the models to complex cultural situations.
>
---
#### [replaced 015] CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14455v2](http://arxiv.org/pdf/2505.14455v2)**

> **作者:** Chihan Huang; Hao Tang
>
> **摘要:** Although autoregressive models have dominated language modeling in recent years, there has been a growing interest in exploring alternative paradigms to the conventional next-token prediction framework. Diffusion-based language models have emerged as a compelling alternative due to their powerful parallel generation capabilities and inherent editability. However, these models are often constrained by fixed-length generation. A promising direction is to combine the strengths of both paradigms, segmenting sequences into blocks, modeling autoregressive dependencies across blocks while leveraging discrete diffusion to estimate the conditional distribution within each block given the preceding context. Nevertheless, their practical application is often hindered by two key limitations: rigid fixed-length outputs and a lack of flexible control mechanisms. In this work, we address the critical limitations of fixed granularity and weak controllability in current large diffusion language models. We propose CtrlDiff, a dynamic and controllable semi-autoregressive framework that adaptively determines the size of each generation block based on local semantics using reinforcement learning. Furthermore, we introduce a classifier-guided control mechanism tailored to discrete diffusion, which significantly reduces computational overhead while facilitating efficient post-hoc conditioning without retraining. Extensive experiments demonstrate that CtrlDiff sets a new standard among hybrid diffusion models, narrows the performance gap to state-of-the-art autoregressive approaches, and enables effective conditional text generation across diverse tasks.
>
---
#### [replaced 016] Natural Language Processing for Cardiology: A Narrative Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16708v2](http://arxiv.org/pdf/2510.16708v2)**

> **作者:** Kailai Yang; Yan Leng; Xin Zhang; Tianlin Zhang; Paul Thompson; Bernard Keavney; Maciej Tomaszewski; Sophia Ananiadou
>
> **摘要:** Cardiovascular diseases are becoming increasingly prevalent in modern society, with a profound impact on global health and well-being. These Cardiovascular disorders are complex and multifactorial, influenced by genetic predispositions, lifestyle choices, and diverse socioeconomic and clinical factors. Information about these interrelated factors is dispersed across multiple types of textual data, including patient narratives, medical records, and scientific literature. Natural language processing (NLP) has emerged as a powerful approach for analysing such unstructured data, enabling healthcare professionals and researchers to gain deeper insights that may transform the diagnosis, treatment, and prevention of cardiac disorders. This review provides a comprehensive overview of NLP research in cardiology from 2014 to 2025. We systematically searched six literature databases for studies describing NLP applications across a range of cardiovascular diseases. After a rigorous screening process, we identified 265 relevant articles. Each study was analysed across multiple dimensions, including NLP paradigms, cardiology-related tasks, disease types, and data sources. Our findings reveal substantial diversity within these dimensions, reflecting the breadth and evolution of NLP research in cardiology. A temporal analysis further highlights methodological trends, showing a progression from rule-based systems to large language models. Finally, we discuss key challenges and future directions, such as developing interpretable LLMs and integrating multimodal data. To the best of our knowledge, this review represents the most comprehensive synthesis of NLP research in cardiology to date.
>
---
#### [replaced 017] PixelWorld: How Far Are We from Perceiving Everything as Pixels?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19339v3](http://arxiv.org/pdf/2501.19339v3)**

> **作者:** Zhiheng Lyu; Xueguang Ma; Wenhu Chen
>
> **摘要:** Recent agentic language models increasingly need to interact with real-world environments that contain tightly intertwined visual and textual information, often through raw camera pixels rather than separately processed images and tokenized text. This shift highlights the need for a unified perception paradigm. To investigate this idea, we explore Perceive Everything as Pixels (PEAP) and introduce PixelWorld, a benchmark that renders natural-language, tabular, mathematical, and diagrammatic inputs into a shared pixel space. Experiments across multiple benchmarks show that PEAP achieves comparable performance to token-based approaches on semantic understanding tasks, suggesting that vision transformers can partially capture global textual semantics without explicit tokenization. In contrast, reasoning-intensive tasks such as mathematics and code show notable performance degradation, although Chain-of-Thought prompting helps mitigate this gap by compensating for missing symbolic structure. We further find that when visual and textual information are closely integrated, representing everything as pixels simplifies preprocessing and avoids cross-modal misalignment. PixelWorld thus provides a systematic and practical framework for evaluating unified vision--language models and facilitates further exploration of pixel-based multimodal learning.
>
---
#### [replaced 018] Mathematics with large language models as provers and verifiers
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2510.12829v2](http://arxiv.org/pdf/2510.12829v2)**

> **作者:** Hieu Le Duc; Leo Liberti
>
> **摘要:** During 2024 and 2025 the discussion about the theorem-proving capabilities of large language models started reporting interesting success stories, mostly to do with difficult exercises (such as problems from the International Mathematical Olympiad), but also with conjectures [Feldman & Karbasi, arXiv:2509.18383v1] formulated for the purpose of verifying whether the artificial intelligence could prove it. In this paper we report a theorem proving feat achieved by ChatGPT by using a protocol involving different prover and verifier instances of the gpt-5 model working collaboratively. To make sure that the produced proofs do not suffer from hallucinations, the final proof is formally verified by the lean proof assistant, and the conformance of premises and conclusion of the lean code is verified by a human. Our methodology is by no means complete or exact. It was nonetheless able to solve five out of six 2025 IMO problems, and close about a third of the sixty-six number theory conjectures in [Cohen, Journal of Integer Sequences, 2025].
>
---
#### [replaced 019] PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2510.17947v2](http://arxiv.org/pdf/2510.17947v2)**

> **作者:** Neeladri Bhuiya; Madhav Aggarwal; Diptanshu Purwar
>
> **备注:** First two authors have equal author contributions
>
> **摘要:** Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation.
>
---
#### [replaced 020] Improving Metacognition and Uncertainty Communication in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.05126v2](http://arxiv.org/pdf/2510.05126v2)**

> **作者:** Mark Steyvers; Catarina Belem; Padhraic Smyth
>
> **摘要:** Large language models (LLMs) are increasingly used in decision-making contexts, but when they present answers without signaling low confidence, users may unknowingly act on erroneous outputs. Prior work shows that LLMs maintain internal uncertainty signals, yet their expressed confidence is often miscalibrated and poorly discriminates between correct and incorrect answers. We investigate whether supervised fine-tuning can improve models' ability to communicate uncertainty and whether such improvements generalize across tasks and domains. We fine-tune LLMs on datasets spanning general knowledge, mathematics, and open-ended trivia, and evaluate two metacognitive tasks: (1) single-question confidence estimation, where the model assigns a numeric certainty to its answer, and (2) pairwise confidence comparison, where the model selects which of two answers it is more likely to answer correctly. We assess generalization to unseen domains, including medical and legal reasoning. Results show that fine-tuning improves calibration (alignment between stated confidence and accuracy) and discrimination (higher confidence for correct vs. incorrect responses) within and across domains. However, gains are task-specific: training on single-question calibration does not transfer to pairwise comparison, and vice versa. Multitask fine-tuning yields broader gains, lowering calibration error and strengthening discrimination in out-of-domain evaluations. This suggests that uncertainty communication in LLMs is trainable but requires multitask training to generalize effectively.
>
---
#### [replaced 021] metaTextGrad: Automatically optimizing language model optimizers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18524v2](http://arxiv.org/pdf/2505.18524v2)**

> **作者:** Guowei Xu; Mert Yuksekgonul; Carlos Guestrin; James Zou
>
> **备注:** 21 pages, 2 figures
>
> **摘要:** Large language models (LLMs) are increasingly used in learning algorithms, evaluations, and optimization tasks. Recent studies have shown that using LLM-based optimizers to automatically optimize model prompts, demonstrations, predictions themselves, or other components can significantly enhance the performance of AI systems, as demonstrated by frameworks such as DSPy and TextGrad. However, optimizers built on language models themselves are usually designed by humans with manual design choices; optimizers themselves are not optimized. Moreover, these optimizers are general purpose by design, to be useful to a broad audience, and are not tailored for specific tasks. To address these challenges, we propose metaTextGrad, which focuses on designing a meta-optimizer to further enhance existing optimizers and align them to be good optimizers for a given task. Our approach consists of two key components: a meta prompt optimizer and a meta structure optimizer. The combination of these two significantly improves performance across multiple benchmarks, achieving an average absolute performance improvement of up to 6% compared to the best baseline.
>
---
#### [replaced 022] The Coverage Principle: How Pre-Training Enables Post-Training
- **分类: stat.ML; cs.AI; cs.CL; cs.LG; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2510.15020v2](http://arxiv.org/pdf/2510.15020v2)**

> **作者:** Fan Chen; Audrey Huang; Noah Golowich; Sadhika Malladi; Adam Block; Jordan T. Ash; Akshay Krishnamurthy; Dylan J. Foster
>
> **摘要:** Language models demonstrate remarkable abilities when pre-trained on large text corpora and fine-tuned for specific tasks, but how and why pre-training shapes the success of the final model remains poorly understood. Notably, although pre-training success is often quantified by cross-entropy loss, cross-entropy can be a poor predictor of downstream performance. Instead, we provide a theoretical perspective on this relationship through the lens of \emph{coverage}, which quantifies the probability mass the pre-trained model places on high-quality responses and which is necessary and sufficient for post-training and test-time scaling methods such as Best-of-N to succeed. Our main results develop an understanding of \emph{the coverage principle}, a phenomenon whereby next-token prediction (more generally, maximum likelihood) implicitly optimizes toward a model with good coverage. In particular, we uncover a mechanism that explains the power of coverage in predicting downstream performance: \emph{coverage generalizes faster than cross-entropy}, avoiding spurious dependence on problem-dependent parameters such as the sequence length. We also study practical algorithmic interventions with provable benefits for improving coverage, including (i) model/checkpoint selection procedures, (ii) gradient normalization schemes, and (iii) test-time decoding strategies.
>
---
#### [replaced 023] Unveiling Transformer Perception by Exploring Input Manifolds
- **分类: cs.LG; cs.AI; cs.CL; I.2.7; I.6.4**

- **链接: [http://arxiv.org/pdf/2410.06019v2](http://arxiv.org/pdf/2410.06019v2)**

> **作者:** Alessandro Benfenati; Alfio Ferrara; Alessio Marta; Davide Riva; Elisabetta Rocchetti
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** This paper introduces a general method for the exploration of equivalence classes in the input space of Transformer models. The proposed approach is based on sound mathematical theory which describes the internal layers of a Transformer architecture as sequential deformations of the input manifold. Using eigendecomposition of the pullback of the distance metric defined on the output space through the Jacobian of the model, we are able to reconstruct equivalence classes in the input space and navigate across them. Our method enables two complementary exploration procedures: the first retrieves input instances that produce the same class probability distribution as the original instance-thus identifying elements within the same equivalence class-while the second discovers instances that yield a different class probability distribution, effectively navigating toward distinct equivalence classes. Finally, we demonstrate how the retrieved instances can be meaningfully interpreted by projecting their embeddings back into a human-readable format.
>
---
#### [replaced 024] Chiron-o1: Igniting Multimodal Large Language Models towards Generalizable Medical Reasoning via Mentor-Intern Collaborative Search
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16962v2](http://arxiv.org/pdf/2506.16962v2)**

> **作者:** Haoran Sun; Yankai Jiang; Wenjie Lou; Yujie Zhang; Wenjie Li; Lilong Wang; Mianxin Liu; Lei Liu; Xiaosong Wang
>
> **摘要:** Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at https://github.com/manglu097/Chiron-o1
>
---
#### [replaced 025] Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05735v4](http://arxiv.org/pdf/2506.05735v4)**

> **作者:** Rongzhe Wei; Peizhi Niu; Hans Hao-Hsun Hsu; Ruihan Wu; Haoteng Yin; Mohsen Ghassemi; Yifan Li; Vamsi K. Potluru; Eli Chien; Kamalika Chaudhuri; Olgica Milenkovic; Pan Li
>
> **备注:** NeurIPS Camera-Ready Version. Code available at: https://github.com/Graph-COM/Knowledge_Unlearning
>
> **摘要:** Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
>
---
#### [replaced 026] Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05571v2](http://arxiv.org/pdf/2510.05571v2)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Kaiwen Zhou; Zhen Zhang; Yue Fan; Yanan Xie; Peng Qi; Xin Eric Wang
>
> **摘要:** The promotion of academic papers has become an important means of enhancing research visibility. However, existing automated methods struggle limited storytelling, insufficient aesthetic quality, and constrained self-adjustment, making it difficult to achieve efficient and engaging dissemination. At the heart of those challenges is a simple principle: \emph{there is no way to improve it when you cannot evaluate it right}. To address this, we introduce \textbf{EvoPresent}, a self-improvement agent framework that unifies coherent narratives, aesthetic-aware designs, and realistic presentation delivery via virtual characters. Central to EvoPresent is \textbf{PresAesth}, a multi-task reinforcement learning (RL) aesthetic model that provides reliable aesthetic scoring, defect adjustment, and comparative feedback, enabling iterative self-improvement even under limited aesthetic training data. To systematically evaluate the methods, we introduce \textbf{EvoPresent Benchmark}, a comprehensive benchmark comprising: \textit{Presentation Generation Quality}, built on 650 top-tier AI conference papers with multimodal resources (slides, videos and scripts) to assess both content and design; and \textit{Aesthetic Awareness}, consisting of 2,000 slide pairs with varying aesthetic levels, supporting joint training and evaluation on scoring, defect adjustment, and comparison. Our findings highlight that (i) High-quality feedback is essential for agent self-improvement, while initial capability alone does not guarantee effective self-correction. (ii) Automated generation pipelines exhibit a trade-off between visual design and content construction. (iii) Multi-task RL training shows stronger generalization in aesthetic awareness tasks.
>
---
#### [replaced 027] Robustness Assessment and Enhancement of Text Watermarking for Google's SynthID
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20228v2](http://arxiv.org/pdf/2508.20228v2)**

> **作者:** Xia Han; Qi Li; Jianbing Ni; Mohammad Zulkernine
>
> **备注:** Accepted by TrustCom2025
>
> **摘要:** Recent advances in LLM watermarking methods such as SynthID-Text by Google DeepMind offer promising solutions for tracing the provenance of AI-generated text. However, our robustness assessment reveals that SynthID-Text is vulnerable to meaning-preserving attacks, such as paraphrasing, copy-paste modifications, and back-translation, which can significantly degrade watermark detectability. To address these limitations, we propose SynGuard, a hybrid framework that combines the semantic alignment strength of Semantic Information Retrieval (SIR) with the probabilistic watermarking mechanism of SynthID-Text. Our approach jointly embeds watermarks at both lexical and semantic levels, enabling robust provenance tracking while preserving the original meaning. Experimental results across multiple attack scenarios show that SynGuard improves watermark recovery by an average of 11.1\% in F1 score compared to SynthID-Text. These findings demonstrate the effectiveness of semantic-aware watermarking in resisting real-world tampering. All code, datasets, and evaluation scripts are publicly available at: https://github.com/githshine/SynGuard.
>
---
#### [replaced 028] From TOWER to SPIRE: Adding the Speech Modality to a Translation-Specialist LLM
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10620v3](http://arxiv.org/pdf/2503.10620v3)**

> **作者:** Kshitij Ambilduke; Ben Peters; Sonal Sannigrahi; Anil Keshwani; Tsz Kin Lam; Bruno Martins; André F. T. Martins; Marcely Zanon Boito
>
> **备注:** EMNLP 2025 (Findings) camera ready
>
> **摘要:** We introduce Spire, a speech-augmented language model (LM) capable of both translating and transcribing speech input from English into 10 other languages as well as translating text input in both language directions. Spire integrates the speech modality into an existing multilingual LM via speech discretization and continued pre-training using only 42.5K hours of speech. In particular, we adopt the pretraining framework of multilingual LMs and treat discretized speech input as an additional translation language. This approach not only equips the model with speech capabilities, but also preserves its strong text-based performance. We achieve this using significantly less data than existing speech LMs, demonstrating that discretized speech input integration as an additional language is feasible during LM adaptation. We make our code and models available to the community.
>
---
#### [replaced 029] Quantum Natural Language Processing: A Comprehensive Review of Models, Methods, and Applications
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09909v2](http://arxiv.org/pdf/2504.09909v2)**

> **作者:** Farha Nausheen; Khandakar Ahmed; M Imad Khan; Farina Riaz
>
> **摘要:** In recent developments, deep learning methodologies applied to Natural Language Processing (NLP) have revealed a paradox: They improve performance but demand considerable data and resources for their training. Alternatively, quantum computing exploits the principles of quantum mechanics to overcome the computational limitations of current methodologies, thereby establishing an emerging field known as quantum natural language processing (QNLP). This domain holds the potential to attain a quantum advantage in the processing of linguistic structures, surpassing classical models in both efficiency and accuracy. In this paper, it is proposed to categorise QNLP models based on quantum computing principles, architecture, and computational approaches. This paper attempts to provide a survey on how quantum meets language by mapping state-of-the-art in this area, embracing quantum encoding techniques for classical data, QNLP models for prevalent NLP tasks, and quantum optimisation techniques for hyper parameter tuning. The landscape of quantum computing approaches applied to various NLP tasks is summarised by showcasing the specific QNLP methods used, and the popularity of these methods is indicated by their count. From the findings, it is observed that QNLP approaches are still limited to small data sets, with only a few models explored extensively, and there is increasing interest in the application of quantum computing to natural language processing tasks.
>
---
#### [replaced 030] ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.03502v2](http://arxiv.org/pdf/2510.03502v2)**

> **作者:** Ali Khairallah; Arkaitz Zubiaga
>
> **备注:** 47 pages, 15 figures. Dataset available at Zenodo: https://doi.org/10.5281/zenodo.17249602 Codebase available at GitHub: https://github.com/alikhairallah/ALHD-Benchmarking
>
> **摘要:** We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats.
>
---
#### [replaced 031] LV-Eval: A Balanced Long-Context Benchmark with 5 Length Levels Up to 256K
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.05136v3](http://arxiv.org/pdf/2402.05136v3)**

> **作者:** Tao Yuan; Xuefei Ning; Dong Zhou; Zhijie Yang; Shiyao Li; Minghui Zhuang; Zheyue Tan; Zhuyu Yao; Dahua Lin; Boxun Li; Guohao Dai; Shengen Yan; Yu Wang
>
> **摘要:** State-of-the-art large language models (LLMs) are now claiming remarkable supported context lengths of 256k or even more. In contrast, the average context lengths of mainstream benchmarks are insufficient (5k-21k), and they suffer from potential knowledge leakage and inaccurate metrics, resulting in biased evaluation. This paper introduces LV-Eval, a challenging long-context benchmark with five length levels (16k, 32k, 64k, 128k, and 256k) reaching up to 256k words. LV-Eval features two main tasks, single-hop QA and multi-hop QA, comprising 11 bilingual datasets. The design of LV-Eval has incorporated three key techniques, namely confusing facts insertion, keyword and phrase replacement, and keyword-recall-based metric design. The advantages of LV-Eval include controllable evaluation across different context lengths, challenging test instances with confusing facts, mitigated knowledge leakage, and more objective evaluations. We evaluate 15 LLMs on LV-Eval and conduct ablation studies on the benchmarking techniques. The results reveal that: (i) Moonshot-v1 and recent large-scale open-source models, such as Qwen-2.5-72B and Llama-3.1-70B, achieve the highest performance on LV-Eval, particularly at lengths below 64k. (ii) Models exhibit distinct score trends. For example, GLM-4-9B-128k, Yi-6B-200k, and Llama3-8B-1M exhibit a relatively gentle degradation of performance, but their absolute performances may not necessarily be higher than those of LLMs with shorter context lengths. (iii) LLMs' performances can significantly degrade in the presence of confusing information, especially in the pressure test of "needle in a haystack". (iv) Issues related to knowledge leakage and inaccurate metrics introduce bias in evaluation, and these concerns are alleviated in LV-Eval. All datasets and evaluation codes are released at: https://github.com/infinigence/LVEval.
>
---
#### [replaced 032] Learning Linear Attention in Polynomial Time
- **分类: cs.LG; cs.AI; cs.CL; cs.DS**

- **链接: [http://arxiv.org/pdf/2410.10101v3](http://arxiv.org/pdf/2410.10101v3)**

> **作者:** Morris Yau; Ekin Akyürek; Jiayuan Mao; Joshua B. Tenenbaum; Stefanie Jegelka; Jacob Andreas
>
> **摘要:** Previous research has explored the computational expressivity of Transformer models in simulating Boolean circuits or Turing machines. However, the learnability of these simulators from observational data has remained an open question. Our study addresses this gap by providing the first polynomial-time learnability results (specifically strong, agnostic PAC learning) for single-layer Transformers with linear attention. We show that linear attention may be viewed as a linear predictor in a suitably defined RKHS. As a consequence, the problem of learning any linear transformer may be converted into the problem of learning an ordinary linear predictor in an expanded feature space, and any such predictor may be converted back into a multiheaded linear transformer. Moving to generalization, we show how to efficiently identify training datasets for which every empirical risk minimizer is equivalent (up to trivial symmetries) to the linear Transformer that generated the data, thereby guaranteeing the learned model will correctly generalize across all inputs. Finally, we provide examples of computations expressible via linear attention and therefore polynomial-time learnable, including associative memories, finite automata, and a class of Universal Turing Machine (UTMs) with polynomially bounded computation histories. We empirically validate our theoretical findings on three tasks: learning random linear attention networks, key--value associations, and learning to execute finite automata. Our findings bridge a critical gap between theoretical expressivity and learnability of Transformers, and show that flexible and general models of computation are efficiently learnable.
>
---
#### [replaced 033] DeCAL Tokenwise Compression
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08514v2](http://arxiv.org/pdf/2508.08514v2)**

> **作者:** Sameer Panwar
>
> **摘要:** This paper introduces DeCAL, a new method for tokenwise compression. DeCAL uses an encoder-decoder language model pretrained with denoising to learn to produce high-quality, general-purpose compressed representations from the encoder. DeCAL applies small modifications to the encoder, with the emphasis on maximizing compression quality, even at the expense of compute. We show that DeCAL at 2x compression can match uncompressed on several downstream tasks, with usually only a minor dropoff in metrics up to 8x compression, among question-answering, summarization, and multi-vector retrieval tasks. DeCAL offers significant savings where pre-computed dense representations can be utilized, and we believe the approach can be further developed to be more broadly applicable.
>
---
#### [replaced 034] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01735v3](http://arxiv.org/pdf/2410.01735v3)**

> **作者:** Duy Nguyen; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** NeurIPS 2025 camera-ready. First two authors contributed equally. Code: https://github.com/duykhuongnguyen/LASeR-MAB
>
> **摘要:** Reward Models (RMs) are crucial to aligning large language models (LLMs), but the degree to which an RM specialized to one task (e.g. writing) generalizes to new tasks (e.g. math) is often not known a priori, often making using only one fixed RM to train LLMs suboptimal. However, optimizing LLMs with multiple RMs simultaneously can incur a prohibitively high computational cost and lead to conflicting signals from different RMs that may degrade performance. To address these challenges, we introduce LASeR (Learning to Adaptively Select Rewards), which frames reward model selection as a multi-armed bandit problem, efficiently and iteratively training LLMs using multiple RMs by selecting the most well-suited RM for each instance. On commonsense and math reasoning tasks, we show that LASeR boosts iterative LLM training, improving the absolute average accuracy of Llama-3-8B over three datasets by 2.67% over an ensemble of RM scores while also showing superior efficiency (e.g., a 2x speedup). Moreover, on WildChat (open-ended instruction-following tasks), LASeR leads to a 72.69% AlpacaEval win rate over the RM score ensemble baseline. Extending to long-context generation, LASeR improves by 2.96 F1 points (avg.) on single-document QA tasks and 2.97 F1 points on few-shot learning over the RM score ensemble baseline with best-of-n sampling.
>
---
#### [replaced 035] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.24379v3](http://arxiv.org/pdf/2505.24379v3)**

> **作者:** Xiaoyu Wu; Yifei Pang; Terrance Liu; Zhiwei Steven Wu
>
> **备注:** Accepted by Neurips 2025
>
> **摘要:** Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.
>
---
#### [replaced 036] Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.18279v2](http://arxiv.org/pdf/2510.18279v2)**

> **作者:** Yanhong Li; Zixuan Lan; Jiawei Zhou
>
> **备注:** Accepted to EMNLP 2025 Findings ("Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs")
>
> **摘要:** Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.
>
---
#### [replaced 037] Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21589v5](http://arxiv.org/pdf/2508.21589v5)**

> **作者:** Zinan Tang; Xin Gao; Qizhi Pei; Zhuoshi Pan; Mengzhang Cai; Jiang Wu; Conghui He; Lijun Wu
>
> **备注:** Accepted by EMNLP 2025 (Main)
>
> **摘要:** Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
>
---
#### [replaced 038] Using (Not-so) Large Language Models to Generate Simulation Models in a Formal DSL: A Study on Reaction Networks
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01675v2](http://arxiv.org/pdf/2503.01675v2)**

> **作者:** Justin N. Kreikemeyer; Miłosz Jankowski; Pia Wilsdorf; Adelinde M. Uhrmacher
>
> **备注:** 27 pages, 5 figures; supplemental material available at https://doi.org/10.1145/3733719
>
> **摘要:** Formal languages are an integral part of modeling and simulation. They allow the distillation of knowledge into concise simulation models amenable to automatic execution, interpretation, and analysis. However, the arguably most humanly accessible means of expressing models is through natural language, which is not easily interpretable by computers. Here, we evaluate how a Large Language Model (LLM) might be used for formalizing natural language into simulation models. Existing studies only explored using very large LLMs, like the commercial GPT models, without fine-tuning model weights. To close this gap, we show how an open-weights, 7B-parameter Mistral model can be fine-tuned to translate natural language descriptions to reaction network models in a domain-specific language, offering a self-hostable, compute-efficient, and memory efficient alternative. To this end, we develop a synthetic data generator to serve as the basis for fine-tuning and evaluation. Our quantitative evaluation shows that our fine-tuned Mistral model can recover the ground truth simulation model in up to 84.5% of cases. In addition, our small-scale user study demonstrates the model's practical potential for one-time generation as well as interactive modeling in various domains. While promising, in its current form, the fine-tuned small LLM cannot catch up with large LLMs. We conclude that higher-quality training data are required, and expect future small and open-source LLMs to offer new opportunities.
>
---
#### [replaced 039] LoRA vs Full Fine-tuning: An Illusion of Equivalence
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21228v3](http://arxiv.org/pdf/2410.21228v3)**

> **作者:** Reece Shuttleworth; Jacob Andreas; Antonio Torralba; Pratyusha Sharma
>
> **摘要:** Fine-tuning is a crucial paradigm for adapting pre-trained large language models to downstream tasks. Recently, methods like Low-Rank Adaptation (LoRA) have been shown to effectively fine-tune LLMs with an extreme reduction in trainable parameters. But, \emph{are their learned solutions really equivalent?} We study how LoRA and full-finetuning change pre-trained models by analyzing the model's weight matrices through the lens of their spectral properties. We find that LoRA and full fine-tuning yield weight matrices whose singular value decompositions exhibit very different structure: weight matrices trained with LoRA have new, high-ranking singular vectors, which we call \emph{intruder dimensions}, while those trained with full fine-tuning do not. Further, we extend the finding that LoRA forgets less than full fine-tuning and find its forgetting is vastly localized to the intruder dimension -- by causally intervening on the intruder dimensions by changing their associated singular values post-fine-tuning, we show that they cause forgetting. Moreover, scaling them down significantly improves modeling of the pre-training distribution with a minimal drop in downstream task performance. Given this, we should expect accumulating intruder dimensions to be harmful and lead to more forgetting. This will be amplified during continual learning because of sequentially fine-tuning, and we show that LoRA models do accumulate intruder dimensions here tend to perform worse in this setting, emphasizing the practicality of our findings.
>
---
#### [replaced 040] Efficient Interleaved Speech Modeling through Knowledge Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.23670v2](http://arxiv.org/pdf/2506.23670v2)**

> **作者:** Mohammadmahdi Nouriborji; Morteza Rohanian
>
> **摘要:** Current speech language models exceed the size and latency constraints of many deployment environments. We build compact, expressive speech generation models through layer-aligned distillation, matching hidden states, attention maps, and softened logits to compress large multimodal transformers by 3x with minimal loss in performance. We introduce TinyWave, a family of 2B-parameter models for speech-to-speech and interleaved speech-text generation, trained on 50,000 hours of public audio. TinyWave supports (i) speech-only generation using phonetic or expressive tokens and (ii) mixed speech-text continuations. Evaluation on Libri-Light shows TinyWave within 1.4 normalized perplexity points of its teacher. Accuracy on spoken StoryCloze and SALMon reaches 93-97% of the teacher's performance, outperforming size-matched baselines. These models are optimized for deployment on commodity hardware, enabling applications in real-time conversational agents, assistive technologies, and low-resource environments. We release models, training code, and evaluation scripts to support reproducible research on compact, expressive speech generation.
>
---
#### [replaced 041] FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.16439v2](http://arxiv.org/pdf/2510.16439v2)**

> **作者:** Syed Rifat Raiyan; Md Farhan Ishmam; Abdullah Al Imran; Mohammad Ali Moni
>
> **摘要:** Large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. Much of this overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. We address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to preserve the top-k% tokens in their original order, and obtain a sparse frugalized prompt. We evaluate the approach across four NLP tasks: Sentiment Analysis, Commonsense QA, Summarization, and Mathematical Reasoning, using a suite of frontier LLMs. For the first three tasks, a 20% prompt reduction incurs only a marginal loss in task performance, demonstrating that contemporary LLMs can reconstruct elided context from high-salience cues. In contrast, performance on mathematical reasoning deteriorates sharply, reflecting a stronger dependence on complete token continuity. Further analysis with bottom-k% and random-k% tokens reveals asymmetric performance patterns that may suggest potential task contamination effects, wherein models may resort to shallow memorized patterns from pretraining exposure for conventional NLP tasks. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs, and delineate the boundary between tasks tolerant to contextual sparsity and those requiring exhaustive context. Our source code and models are available at: https://github.com/Starscream-11813/Frugal-ICL.
>
---
#### [replaced 042] WikiVideo: Article Generation from Multiple Videos
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00939v2](http://arxiv.org/pdf/2504.00939v2)**

> **作者:** Alexander Martin; Reno Kriz; William Gantt Walden; Kate Sanders; Hannah Recknor; Eugene Yang; Francis Ferraro; Benjamin Van Durme
>
> **备注:** Repo can be found here: https://github.com/alexmartin1722/wikivideo
>
> **摘要:** We introduce the task of grounded article generation with the goal of creating a Wikipedia-style article from multiple diverse videos about real-world events -- from natural disasters to political elections -- where all the information in the article is supported by video evidence. Videos are intuitive sources for retrieval-augmented generation (RAG), but most contemporary RAG workflows focus heavily on text while existing methods for video-based summarization focus on low-level scene understanding rather than high-level event semantics. To close this gap, we introduce WikiVideo, a benchmark consisting of expert-written articles and densely annotated videos that provide evidence for articles' claims, facilitating the integration of video into RAG pipelines and enabling the creation of in-depth content that is grounded in multimodal sources. We further propose Collaborative Article Generation (CAG), a novel interactive method for article creation from multiple videos. CAG leverages an iterative interaction between an r1-style reasoning model and a VideoLLM to draw higher-level inferences about the target event than is possible with VideoLLMs alone, which fixate on low-level visual features. We benchmark state-of-the-art VideoLLMs and CAG in both oracle retrieval and RAG settings and find that CAG consistently outperforms alternative methods, while suggesting intriguing avenues for future work.
>
---
#### [replaced 043] InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13878v2](http://arxiv.org/pdf/2505.13878v2)**

> **作者:** Yanggan Gu; Yuanyi Wang; Zhaoyi Yan; Yiming Zhang; Qi Zhou; Fei Wu; Hongxia Yang
>
> **摘要:** Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks.
>
---
#### [replaced 044] ScholaWrite: A Dataset of End-to-End Scholarly Writing Process
- **分类: cs.HC; cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2502.02904v4](http://arxiv.org/pdf/2502.02904v4)**

> **作者:** Khanh Chi Le; Linghe Wang; Minhwa Lee; Ross Volkov; Luan Tuyen Chau; Dongyeop Kang
>
> **备注:** Equal contribution: Khanh Chi Le, Linghe Wang, Minhwa Lee | project page: https://minnesotanlp.github.io/scholawrite/
>
> **摘要:** Writing is a cognitively demanding activity that requires constant decision-making, heavy reliance on working memory, and frequent shifts between tasks of different goals. To build writing assistants that truly align with writers' cognition, we must capture and decode the complete thought process behind how writers transform ideas into final texts. We present ScholaWrite, the first dataset of end-to-end scholarly writing, tracing the multi-month journey from initial drafts to final manuscripts. We contribute three key advances: (1) a Chrome extension that unobtrusively records keystrokes on Overleaf, enabling the collection of realistic, in-situ writing data; (2) a novel corpus of full scholarly manuscripts, enriched with fine-grained annotations of cognitive writing intentions. The dataset includes \LaTeX-based edits from five computer science preprints, capturing nearly 62K text changes over four months; and (3) analyses and insights into the micro-dynamics of scholarly writing, highlighting gaps between human writing processes and the current capabilities of large language models (LLMs) in providing meaningful assistance. ScholaWrite underscores the value of capturing end-to-end writing data to develop future writing assistants that support, not replace, the cognitive work of scientists.
>
---
#### [replaced 045] GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.18129v2](http://arxiv.org/pdf/2503.18129v2)**

> **作者:** Varvara Krechetova; Denis Kochedykov
>
> **备注:** Github with code and benchmark set: https://github.com/Solirinai/GeoBenchX
>
> **摘要:** This paper establishes a benchmark for evaluating tool-calling capabilities of large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess eight commercial LLMs (Claude Sonnet 3.5 and 4, Claude Haiku 3.5, Gemini 2.0 Flash, Gemini 2.5 Pro Preview, GPT-4o, GPT-4.1 and o4-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks in four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test rejection accuracy. We develop a LLM-as-Judge evaluation framework to compare agent solutions against reference solutions. Results show o4-mini and Claude 3.5 Sonnet achieve the best overall performance, OpenAI's GPT-4.1, GPT-4o and Google's Gemini 2.5 Pro Preview do not fall far behind, but the last two are more efficient in identifying unsolvable tasks. Claude Sonnet 4, due its preference to provide any solution rather than reject a task, proved to be less accurate. We observe significant differences in token usage, with Anthropic models consuming more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources (available at https://github.com/Solirinai/GeoBenchX), providing one more standardized method for the ongoing evaluation of LLMs for GeoAI.
>
---
#### [replaced 046] Reasoning Models Better Express Their Confidence
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14489v2](http://arxiv.org/pdf/2505.14489v2)**

> **作者:** Dongkeun Yoon; Seungone Kim; Sohee Yang; Sunkyoung Kim; Soyeon Kim; Yongil Kim; Eunbi Choi; Yireun Kim; Minjoon Seo
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Despite their strengths, large language models (LLMs) often fail to communicate their confidence accurately, making it difficult to assess when they might be wrong and limiting their reliability. In this work, we demonstrate that reasoning models that engage in extended chain-of-thought (CoT) reasoning exhibit superior performance not only in problem-solving but also in accurately expressing their confidence. Specifically, we benchmark six reasoning models across six datasets and find that they achieve strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings. Our detailed analysis reveals that these gains in calibration stem from the slow thinking behaviors of reasoning models (e.g., exploring alternative approaches and backtracking) which enable them to adjust their confidence dynamically throughout their CoT, making it progressively more accurate. In particular, we find that reasoning models become increasingly better calibrated as their CoT unfolds, a trend not observed in non-reasoning models. Moreover, removing slow thinking behaviors from the CoT leads to a significant drop in calibration. Lastly, we show that non-reasoning models also demonstrate enhanced calibration when simply guided to slow think via in-context learning, fully isolating slow thinking as the source of the calibration gains.
>
---
#### [replaced 047] Who's Asking? Investigating Bias Through the Lens of Disability Framed Queries in LLMs
- **分类: cs.CL; cs.AI; cs.CY; 68T50, 68T07, 68T05; I.2.7; I.2.6; K.4.2**

- **链接: [http://arxiv.org/pdf/2508.15831v2](http://arxiv.org/pdf/2508.15831v2)**

> **作者:** Vishnu Hari; Kalpana Panda; Srikant Panda; Amit Agarwal; Hitesh Laxmichand Patel
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Large Language Models (LLMs) routinely infer users demographic traits from phrasing alone, which can result in biased responses, even when no explicit demographic information is provided. The role of disability cues in shaping these inferences remains largely uncharted. Thus, we present the first systematic audit of disability-conditioned demographic bias across eight state-of-the-art instruction-tuned LLMs ranging from 3B to 72B parameters. Using a balanced template corpus that pairs nine disability categories with six real-world business domains, we prompt each model to predict five demographic attributes - gender, socioeconomic status, education, cultural background, and locality - under both neutral and disability-aware conditions. Across a varied set of prompts, models deliver a definitive demographic guess in up to 97\% of cases, exposing a strong tendency to make arbitrary inferences with no clear justification. Disability context heavily shifts predicted attribute distributions, and domain context can further amplify these deviations. We observe that larger models are simultaneously more sensitive to disability cues and more prone to biased reasoning, indicating that scale alone does not mitigate stereotype amplification. Our findings reveal persistent intersections between ableism and other demographic stereotypes, pinpointing critical blind spots in current alignment strategies. We release our evaluation framework and results to encourage disability-inclusive benchmarking and recommend integrating abstention calibration and counterfactual fine-tuning to curb unwarranted demographic inference. Code and data will be released on acceptance.
>
---
#### [replaced 048] Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01349v4](http://arxiv.org/pdf/2502.01349v4)**

> **作者:** Giorgos Filandrianos; Angeliki Dimitriou; Maria Lymperaiou; Konstantinos Thomas; Giorgos Stamou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
>
---
#### [replaced 049] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks
- **分类: cs.AI; cs.CL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.00890v2](http://arxiv.org/pdf/2508.00890v2)**

> **作者:** Fali Wang; Hui Liu; Zhenwei Dai; Jingying Zeng; Zhiwei Zhang; Zongyu Wu; Chen Luo; Zhen Li; Xianfeng Tang; Qi He; Suhang Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability.
>
---
#### [replaced 050] ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06313v3](http://arxiv.org/pdf/2507.06313v3)**

> **作者:** Kiarash Zahirnia; Zahra Golpayegani; Walid Ahmed; Yang Liu
>
> **摘要:** Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
>
---
#### [replaced 051] Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12631v2](http://arxiv.org/pdf/2508.12631v2)**

> **作者:** Yiqun Zhang; Hao Li; Jianhao Chen; Hangfan Zhang; Peng Ye; Lei Bai; Shuyue Hu
>
> **备注:** This work has been accepted to DAI 2025
>
> **摘要:** Balancing performance and efficiency is a central challenge in large language model (LLM) advancement. GPT-5 addresses this with test-time routing, dynamically assigning queries to either an efficient or a high-capacity model during inference. In this work, we present Avengers-Pro, a test-time routing framework that ensembles LLMs of varying capacities and efficiencies, providing a unified solution for all performance-efficiency tradeoffs. The Avengers-Pro embeds and clusters incoming queries, then routes each to the most suitable model based on a performance-efficiency score. Across 6 challenging benchmarks and 8 leading models -- including GPT-5-medium, Gemini-2.5-pro, and Claude-opus-4.1 -- Avengers-Pro achieves state-of-the-art results: by varying a performance-efficiency trade-off parameter, it can surpass the strongest single model (GPT-5-medium) by +7% in average accuracy. Moreover, it can match the average accuracy of the strongest single model at 27% lower cost, and reach ~90% of that performance at 63% lower cost. Last but not least, it achieves a Pareto frontier, consistently yielding the highest accuracy for any given cost, and the lowest cost for any given accuracy, among all single models. Code is available at https://github.com/ZhangYiqun018/AvengersPro.
>
---
#### [replaced 052] Evaluation Framework for Highlight Explanations of Context Utilisation in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.02629v2](http://arxiv.org/pdf/2510.02629v2)**

> **作者:** Jingyi Sun; Pepa Atanasova; Sagnik Ray Choudhury; Sekh Mainul Islam; Isabelle Augenstein
>
> **摘要:** Context utilisation, the ability of Language Models (LMs) to incorporate relevant information from the provided context when generating responses, remains largely opaque to users, who cannot determine whether models draw from parametric memory or provided context, nor identify which specific context pieces inform the response. Highlight explanations (HEs) offer a natural solution as they can point the exact context pieces and tokens that influenced model outputs. However, no existing work evaluates their effectiveness in accurately explaining context utilisation. We address this gap by introducing the first gold standard HE evaluation framework for context attribution, using controlled test cases with known ground-truth context usage, which avoids the limitations of existing indirect proxy evaluations. To demonstrate the framework's broad applicability, we evaluate four HE methods -- three established techniques and MechLight, a mechanistic interpretability approach we adapt for this task -- across four context scenarios, four datasets, and five LMs. Overall, we find that MechLight performs best across all context scenarios. However, all methods struggle with longer contexts and exhibit positional biases, pointing to fundamental challenges in explanation accuracy that require new approaches to deliver reliable context utilisation explanations at scale.
>
---
#### [replaced 053] Flexible-length Text Infilling for Discrete Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13579v2](http://arxiv.org/pdf/2506.13579v2)**

> **作者:** Andrew Zhang; Anushka Sivakumar; Chiawei Tang; Chris Thomas
>
> **备注:** Major edit of methodology section. Matches EMNLP camera-ready version
>
> **摘要:** Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility.
>
---
#### [replaced 054] Memorization-Compression Cycles Improve Generalization
- **分类: cs.LG; cs.AI; cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.08727v2](http://arxiv.org/pdf/2505.08727v2)**

> **作者:** Fangyuan Yu
>
> **备注:** 12 pages, 6 figures, NeurIPS2025 NEGEL Workshop
>
> **摘要:** We prove theoretically that generalization improves not only through data scaling but also by compressing internal representations. To operationalize this insight, we introduce the Information Bottleneck Language Modeling (IBLM) objective, which reframes language modeling as a constrained optimization problem: minimizing representation entropy subject to optimal prediction performance. Empirically, we observe an emergent memorization-compression cycle during LLM pretraining, evidenced by oscillation positive/negative gradient alignment between cross-entropy and Matrix-Based Entropy (MBE), a measure of representation entropy. This pattern closely mirrors the predictive-compressive trade-off prescribed by IBLM and also parallels the biological alternation between awake learning and sleep consolidation. Motivated by this observation, we propose Gated Phase Transition (GAPT), a training algorithm that adaptively switches between memorization and compression phases. When applied to GPT-2 pretraining on FineWeb dataset, GAPT reduces MBE by 50% and improves cross-entropy by 4.8%. GAPT improves OOD generalizatino by 35% in a pretraining task on arithmetic multiplication. In a setting designed to simulate catastrophic forgetting, GAPT reduces interference by compressing and separating representations, achieving a 97% improvement in separation - paralleling the functional role of sleep consolidation.
>
---
#### [replaced 055] Test-time Prompt Intervention
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02511v2](http://arxiv.org/pdf/2508.02511v2)**

> **作者:** Chenxu Yang; Qingyi Si; Mz Dai; Dingyu Yao; Mingyu Zheng; Minghui Chen; Zheng Lin; Weiping Wang
>
> **备注:** 24 pages, 20 figures, under review
>
> **摘要:** Test-time compute has led to remarkable success in the large language model (LLM) community, particularly for complex tasks, where longer chains of thought (CoTs) are generated to enhance reasoning capabilities. However, growing evidence reveals that such reasoning models often produce CoTs plagued by excessive redundancy, including unnecessary verification steps and repetitive reasoning shifts. The root cause lies in post-training of them that overly rely on outcome reward paradigms, as the data of process reward paradigms, which regulate intermediate reasoning steps, is difficult to construct at scale. To address this, we propose PI, a novel framework for Test-time Prompt Intervention. PI provides an interface to dynamically guide and regulate reasoning paths during inference through timely (When module) and proper (How module) interventions and post-intervention sampling (Which module). This allows human problem-solving expertise and cognitive science principles to be seamlessly integrated into LLMs' reasoning processes, enhancing controllability and interpretability. Extensive experiments across multiple models and datasets demonstrate that PI significantly shortens CoTs while reducing hallucination, yielding more concise and reliable reasoning.
>
---
#### [replaced 056] dInfer: An Efficient Inference Framework for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.08666v3](http://arxiv.org/pdf/2510.08666v3)**

> **作者:** Yuxin Ma; Lun Du; Lanning Wei; Kun Chen; Qian Xu; Kangyu Wang; Guofeng Feng; Guoshan Lu; Lin Liu; Xiaojing Qi; Xinyuan Zhang; Zhen Tao; Haibo Feng; Ziyun Jiang; Ying Xu; Zenan Huang; Yihong Zhuang; Haokai Xu; Jiaqi Hu; Zhenzhong Lan; Junbo Zhao; Jianguo Li; Da Zheng
>
> **摘要:** Diffusion-based large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, leveraging denoising-based generation to enable inherent parallelism. Even more and more open-sourced dLLM models emerge, yet their widespread adoption remains constrained by the lack of a standardized and efficient inference framework. We present dInfer, an efficient and extensible framework for dLLM inference. dInfer decomposes the inference pipeline into four modular components--model, diffusion iteration manager, decoding strategy, and KV-cache manager--and integrates novel algorithms for each component alongside system-level optimizations. Through this combination of algorithmic innovations and system enhancements, dInfer achieves substantial efficiency gains without compromising output quality on LLaDA-MoE. At batch size 1, it surpasses 1,100 tokens per second on HumanEval and averages over 800 tokens per second across six benchmarks on $8\times$ H800 GPUs. Compared to prior systems, dInfer delivers a $10\times$ speedup over Fast-dLLM while maintaining similar model performance. Even compared to the AR model (with a comparable number of activation parameters and performance) QWen2.5-3B, which is highly optimized with the latest vLLM inference engine, dInfer still delivers a $2$-$3\times$ speedup. The implementation of dInfer is open-sourced at https://github.com/inclusionAI/dInfer.
>
---

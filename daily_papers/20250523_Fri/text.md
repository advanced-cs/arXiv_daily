# 自然语言处理 cs.CL

- **最新发布 185 篇**

- **更新 111 篇**

## 最新发布

#### [new 001] DecoupledESC: Enhancing Emotional Support Generation via Strategy-Response Decoupled Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于情感支持对话生成任务，旨在解决现有方法中因策略与内容纠缠导致的心理错误和优化模糊问题。提出IPM-PrefDial数据集，并设计DecoupledESC框架，将任务分解为策略规划与共情响应生成两阶段，分别通过监督微调和偏好优化提升效果。**

- **链接: [http://arxiv.org/pdf/2505.16995v1](http://arxiv.org/pdf/2505.16995v1)**

> **作者:** Chao Zhang; Xin Shi; Xueqiao Zhang; Yifan Zhu; Yi Yang; Yawei Luo
>
> **摘要:** Recent advances in Emotional Support Conversation (ESC) have improved emotional support generation by fine-tuning Large Language Models (LLMs) via Supervised Fine-Tuning (SFT). However, common psychological errors still persist. While Direct Preference Optimization (DPO) shows promise in reducing such errors through pairwise preference learning, its effectiveness in ESC tasks is limited by two key challenges: (1) Entangled data structure: Existing ESC data inherently entangles psychological strategies and response content, making it difficult to construct high-quality preference pairs; and (2) Optimization ambiguity: Applying vanilla DPO to such entangled pairwise data leads to ambiguous training objectives. To address these issues, we introduce Inferential Preference Mining (IPM) to construct high-quality preference data, forming the IPM-PrefDial dataset. Building upon this data, we propose a Decoupled ESC framework inspired by Gross's Extended Process Model of Emotion Regulation, which decomposes the ESC task into two sequential subtasks: strategy planning and empathic response generation. Each was trained via SFT and subsequently enhanced by DPO to align with the psychological preference. Extensive experiments demonstrate that our Decoupled ESC framework outperforms joint optimization baselines, reducing preference bias and improving response quality.
>
---
#### [new 002] AppealCase: A Dataset and Benchmark for Civil Case Appeal Scenarios
- **分类: cs.CL**

- **简介: 该论文提出AppealCase数据集，填补民事上诉场景研究空白。针对上诉案件纠错机制分析不足的问题，构建含1万组真实一、二审案件及五维标注（判决翻转、法律条款等）的数据集，并设计五项新任务。实验显示现有模型表现欠佳（F1<50%），推动LegalAI在上诉分析与司法决策一致性研究。**

- **链接: [http://arxiv.org/pdf/2505.16514v1](http://arxiv.org/pdf/2505.16514v1)**

> **作者:** Yuting Huang; Meitong Guo; Yiquan Wu; Ang Li; Xiaozhong Liu; Keting Yin; Changlong Sun; Fei Wu; Kun Kuang
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Recent advances in LegalAI have primarily focused on individual case judgment analysis, often overlooking the critical appellate process within the judicial system. Appeals serve as a core mechanism for error correction and ensuring fair trials, making them highly significant both in practice and in research. To address this gap, we present the AppealCase dataset, consisting of 10,000 pairs of real-world, matched first-instance and second-instance documents across 91 categories of civil cases. The dataset also includes detailed annotations along five dimensions central to appellate review: judgment reversals, reversal reasons, cited legal provisions, claim-level decisions, and whether there is new information in the second instance. Based on these annotations, we propose five novel LegalAI tasks and conduct a comprehensive evaluation across 20 mainstream models. Experimental results reveal that all current models achieve less than 50% F1 scores on the judgment reversal prediction task, highlighting the complexity and challenge of the appeal scenario. We hope that the AppealCase dataset will spur further research in LegalAI for appellate case analysis and contribute to improving consistency in judicial decision-making.
>
---
#### [new 003] From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs
- **分类: cs.CL**

- **简介: 该论文属于LLMs文化价值观适配任务，旨在解决仅用WVS调查数据导致的文化表征同质化及事实知识干扰问题。研究对比了纯WVS训练与补充百科/场景叙事数据的效果，发现后者显著提升文化表征的差异性，但下游任务效果不一，揭示文化对齐的复杂性。**

- **链接: [http://arxiv.org/pdf/2505.16408v1](http://arxiv.org/pdf/2505.16408v1)**

> **作者:** Muhammad Farid Adilazuarda; Chen Cecilia Liu; Iryna Gurevych; Alham Fikri Aji
>
> **摘要:** Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior.
>
---
#### [new 004] Citation Parsing and Analysis with Language Models
- **分类: cs.CL; cs.DL; cs.SI**

- **简介: 该论文研究基于语言模型的引文解析，旨在解决全球南半球知识共享网络数据不足的问题。作者构建数据集评估开源模型，发现其准确识别引文组件，远超现有方法，尤其是Qwen3-0.6B表现优异，可高效解析，为开发小型 robust 引文解析工具提供支持，提升索引和研究质量。**

- **链接: [http://arxiv.org/pdf/2505.15948v1](http://arxiv.org/pdf/2505.15948v1)**

> **作者:** Parth Sarin; Juan Pablo Alperin
>
> **备注:** Presented at the Workshop on Open Citations & Open Scholarly Metadata 2025
>
> **摘要:** A key type of resource needed to address global inequalities in knowledge production and dissemination is a tool that can support journals in understanding how knowledge circulates. The absence of such a tool has resulted in comparatively less information about networks of knowledge sharing in the Global South. In turn, this gap authorizes the exclusion of researchers and scholars from the South in indexing services, reinforcing colonial arrangements that de-center and minoritize those scholars. In order to support citation network tracking on a global scale, we investigate the capacity of open-weight language models to mark up manuscript citations in an indexable format. We assembled a dataset of matched plaintext and annotated citations from preprints and published research papers. Then, we evaluated a number of open-weight language models on the annotation task. We find that, even out of the box, today's language models achieve high levels of accuracy on identifying the constituent components of each citation, outperforming state-of-the-art methods. Moreover, the smallest model we evaluated, Qwen3-0.6B, can parse all fields with high accuracy in $2^5$ passes, suggesting that post-training is likely to be effective in producing small, robust citation parsing models. Such a tool could greatly improve the fidelity of citation networks and thus meaningfully improve research indexing and discovery, as well as further metascientific research.
>
---
#### [new 005] Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers
- **分类: cs.CL**

- **简介: 该论文属于AI安全攻击任务，针对大型推理模型(LRMs)潜在安全漏洞，提出SEAL攻击方法。通过自适应分层加密与动态策略（随机/自适应调整加密参数），突破模型安全机制，实验显示对GPT-4-mini成功率80.8%，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16241v1](http://arxiv.org/pdf/2505.16241v1)**

> **作者:** Viet-Anh Nguyen; Shiqian Zhao; Gia Dao; Runyi Hu; Yi Xie; Luu Anh Tuan
>
> **摘要:** Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.
>
---
#### [new 006] From Tens of Hours to Tens of Thousands: Scaling Back-Translation for Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出Speech Back-Translation方法，解决多语言语音识别中数据稀缺问题。通过将文本转为合成语音，仅用少量真实数据生成大量高质量合成语音，建立可扩展的训练 pipeline。其开发智能性评估框架，验证合成数据对ASR的提升效果，最终使Whisper模型错误率下降超30%。**

- **链接: [http://arxiv.org/pdf/2505.16972v1](http://arxiv.org/pdf/2505.16972v1)**

> **作者:** Tianduo Wang; Lu Xu; Wei Lu; Shanbo Cheng
>
> **摘要:** Recent advances in Automatic Speech Recognition (ASR) have been largely fueled by massive speech corpora. However, extending coverage to diverse languages with limited resources remains a formidable challenge. This paper introduces Speech Back-Translation, a scalable pipeline that improves multilingual ASR models by converting large-scale text corpora into synthetic speech via off-the-shelf text-to-speech (TTS) models. We demonstrate that just tens of hours of real transcribed speech can effectively train TTS models to generate synthetic speech at hundreds of times the original volume while maintaining high quality. To evaluate synthetic speech quality, we develop an intelligibility-based assessment framework and establish clear thresholds for when synthetic data benefits ASR training. Using Speech Back-Translation, we generate more than 500,000 hours of synthetic speech in ten languages and continue pre-training Whisper-large-v3, achieving average transcription error reductions of over 30\%. These results highlight the scalability and effectiveness of Speech Back-Translation for enhancing multilingual ASR systems.
>
---
#### [new 007] MuseRAG: Idea Originality Scoring At Scale
- **分类: cs.CL**

- **简介: 该论文提出MuseRAG系统，解决大规模创意原创性自动化评估问题。传统人工分组方法效率低且误差大，其结合LLM与RAG框架，自动聚类创意并计算原创性分数，实验显示与人工评分高度一致（r=0.89），支持高效创造力研究。**

- **链接: [http://arxiv.org/pdf/2505.16232v1](http://arxiv.org/pdf/2505.16232v1)**

> **作者:** Ali Sarosh Bangash; Krish Veera; Ishfat Abrar Islam; Raiyan Abdul Baten
>
> **摘要:** An objective, face-valid way to assess the originality of creative ideas is to measure how rare each idea is within a population -- an approach long used in creativity research but difficult to automate at scale. Tabulating response frequencies via manual bucketing of idea rephrasings is labor-intensive, error-prone, and brittle under large corpora. We introduce a fully automated, psychometrically validated pipeline for frequency-based originality scoring. Our method, MuseRAG, combines large language models (LLMs) with an externally orchestrated retrieval-augmented generation (RAG) framework. Given a new idea, the system retrieves semantically similar prior idea buckets and zero-shot prompts the LLM to judge whether the new idea belongs to an existing bucket or forms a new one. The resulting buckets enable computation of frequency-based originality metrics. Across five datasets (N=1143, n_ideas=16294), MuseRAG matches human annotators in idea clustering structure and resolution (AMI = 0.59) and in participant-level scoring (r = 0.89) -- while exhibiting strong convergent and external validity. Our work enables intent-sensitive, human-aligned originality scoring at scale to aid creativity research.
>
---
#### [new 008] Collaboration among Multiple Large Language Models for Medical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗多选问答任务，旨在解决多语言模型协同不足的问题。提出多LLM协作框架，通过分析三个预训练模型，提升其推理能力并减少预测分歧，同时研究模型对抗意见下的信心与准确率关联。**

- **链接: [http://arxiv.org/pdf/2505.16648v1](http://arxiv.org/pdf/2505.16648v1)**

> **作者:** Kexin Shang; Chia-Hsuan Chang; Christopher C. Yang
>
> **备注:** Accepted to IEEE International Conference on Healthcare Informatics 2025
>
> **摘要:** Empowered by vast internal knowledge reservoir, the new generation of large language models (LLMs) demonstrate untapped potential to tackle medical tasks. However, there is insufficient effort made towards summoning up a synergic effect from multiple LLMs' expertise and background. In this study, we propose a multi-LLM collaboration framework tailored on a medical multiple-choice questions dataset. Through post-hoc analysis on 3 pre-trained LLM participants, our framework is proved to boost all LLMs reasoning ability as well as alleviate their divergence among questions. We also measure an LLM's confidence when it confronts with adversary opinions from other LLMs and observe a concurrence between LLM's confidence and prediction accuracy.
>
---
#### [new 009] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance
- **分类: cs.CL**

- **简介: 该论文属于对话系统优化任务，旨在解决小型LLM在任务型对话中话题不一致及抗干扰问题。提出EnSToM方法，通过动态调整熵缩放的引导向量，依据输入不确定性控制引导强度，提升话题保持能力，同时保持模型效率，实验显示其优于微调方法且数据需求小。**

- **链接: [http://arxiv.org/pdf/2505.16526v1](http://arxiv.org/pdf/2505.16526v1)**

> **作者:** Heejae Suh; Yejin Jeon; Deokhyung Kang; Taehee Park; Yejin Min; Gary Geunbae Lee
>
> **备注:** Accepted at ACL 2025 (Findings, long paper)
>
> **摘要:** Small large language models (sLLMs) offer the advantage of being lightweight and efficient, which makes them suitable for resource-constrained environments. However, sLLMs often struggle to maintain topic consistency in task-oriented dialogue systems, which is critical for scenarios such as service chatbots. Specifically, it is important to ensure that the model denies off-topic or malicious inputs and adheres to its intended functionality so as to prevent potential misuse and uphold reliability. Towards this, existing activation engineering approaches have been proposed to manipulate internal activations during inference. While these methods are effective in certain scenarios, our preliminary experiments reveal their limitations in ensuring topic adherence. Therefore, to address this, we propose a novel approach termed Entropy-scaled Steering vectors for Topic Maintenance (EnSToM). EnSToM dynamically adjusts the steering intensity based on input uncertainty, which allows the model to handle off-topic distractors effectively while preserving on-topic accuracy. Our experiments demonstrate that EnSToM achieves significant performance gain with a relatively small data size compared to fine-tuning approaches. By improving topic adherence without compromising efficiency, our approach provides a robust solution for enhancing sLLM-based dialogue systems.
>
---
#### [new 010] Continually Self-Improving Language Models for Bariatric Surgery Question--Answering
- **分类: cs.CL**

- **简介: 该论文提出bRAGgen模型，解决代谢手术患者因医疗资源不均导致的信息获取难题。通过动态阈值触发实时医学证据整合，提升问答系统的持续准确性，并构建首个领域数据集bRAGq（1302问），验证显示其优于现有模型。**

- **链接: [http://arxiv.org/pdf/2505.16102v1](http://arxiv.org/pdf/2505.16102v1)**

> **作者:** Yash Kumar Atri; Thomas H Shin; Thomas Hartvigsen
>
> **摘要:** While bariatric and metabolic surgery (MBS) is considered the gold standard treatment for severe and morbid obesity, its therapeutic efficacy hinges upon active and longitudinal engagement with multidisciplinary providers, including surgeons, dietitians/nutritionists, psychologists, and endocrinologists. This engagement spans the entire patient journey, from preoperative preparation to long-term postoperative management. However, this process is often hindered by numerous healthcare disparities, such as logistical and access barriers, which impair easy patient access to timely, evidence-based, clinician-endorsed information. To address these gaps, we introduce bRAGgen, a novel adaptive retrieval-augmented generation (RAG)-based model that autonomously integrates real-time medical evidence when response confidence dips below dynamic thresholds. This self-updating architecture ensures that responses remain current and accurate, reducing the risk of misinformation. Additionally, we present bRAGq, a curated dataset of 1,302 bariatric surgery--related questions, validated by an expert bariatric surgeon. bRAGq constitutes the first large-scale, domain-specific benchmark for comprehensive MBS care. In a two-phase evaluation, bRAGgen is benchmarked against state-of-the-art models using both large language model (LLM)--based metrics and expert surgeon review. Across all evaluation dimensions, bRAGgen demonstrates substantially superior performance in generating clinically accurate and relevant responses.
>
---
#### [new 011] MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出MASLab，一个统一的LLM多智能体系统代码库，旨在解决领域内缺乏标准化工具导致的重复开发、不公平对比和高门槛问题。通过整合20+方法、提供统一基准与实验环境，及模块化结构，降低使用与扩展难度，助力公平研究与技术演进。**

- **链接: [http://arxiv.org/pdf/2505.16988v1](http://arxiv.org/pdf/2505.16988v1)**

> **作者:** Rui Ye; Keduan Huang; Qimin Wu; Yuzhu Cai; Tian Jin; Xianghe Pang; Xiangrui Liu; Jiaqi Su; Chen Qian; Bohan Tang; Kaiqu Liang; Jiaao Chen; Yue Hu; Zhenfei Yin; Rongye Shi; Bo An; Yang Gao; Wenjun Wu; Lei Bai; Siheng Chen
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community.
>
---
#### [new 012] MPL: Multiple Programming Languages with Large Language Models for Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于信息提取任务，旨在解决现有方法仅依赖Python模拟代码输入、忽略其他编程语言（如C++/Java）在监督微调中的潜力的问题。提出MPL框架，结合多种编程语言并引入function-prompt与虚拟运行技术，通过实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16107v1](http://arxiv.org/pdf/2505.16107v1)**

> **作者:** Bo Li; Gexiang Fang; Wei Ye; Zhenghua Xu; Jinglei Zhang; Hao Cheng; Shikun Zhang
>
> **备注:** Findings of ACL2025
>
> **摘要:** Recent research in information extraction (IE) focuses on utilizing code-style inputs to enhance structured output generation. The intuition behind this is that the programming languages (PLs) inherently exhibit greater structural organization than natural languages (NLs). This structural advantage makes PLs particularly suited for IE tasks. Nevertheless, existing research primarily focuses on Python for code-style simulation, overlooking the potential of other widely-used PLs (e.g., C++ and Java) during the supervised fine-tuning (SFT) phase. In this research, we propose \textbf{M}ultiple \textbf{P}rogramming \textbf{L}anguages with large language models for information extraction (abbreviated as \textbf{MPL}), a novel framework that explores the potential of incorporating different PLs in the SFT phase. Additionally, we introduce \texttt{function-prompt} with virtual running to simulate code-style inputs more effectively and efficiently. Experimental results on a wide range of datasets demonstrate the effectiveness of MPL. Furthermore, we conduct extensive experiments to provide a comprehensive analysis. We have released our code for future research.
>
---
#### [new 013] Augmenting LLM Reasoning with Dynamic Notes Writing for Complex QA
- **分类: cs.CL**

- **简介: 该论文针对多跳问答任务，解决迭代式RAG在处理长上下文时因无关信息积累导致推理效率下降的问题。提出Notes Writing方法，通过动态生成简洁笔记过滤噪声，提升LLM处理能力，框架通用且效果显著（平均提升15.6%）。**

- **链接: [http://arxiv.org/pdf/2505.16293v1](http://arxiv.org/pdf/2505.16293v1)**

> **作者:** Rishabh Maheshwary; Masoud Hashemi; Khyati Mahajan; Shiva Krishna Reddy Malay; Sai Rajeswar; Sathwik Tejaswi Madhusudhan; Spandana Gella; Vikas Yadav
>
> **摘要:** Iterative RAG for multi-hop question answering faces challenges with lengthy contexts and the buildup of irrelevant information. This hinders a model's capacity to process and reason over retrieved content and limits performance. While recent methods focus on compressing retrieved information, they are either restricted to single-round RAG, require finetuning or lack scalability in iterative RAG. To address these challenges, we propose Notes Writing, a method that generates concise and relevant notes from retrieved documents at each step, thereby reducing noise and retaining only essential information. This indirectly increases the effective context length of Large Language Models (LLMs), enabling them to reason and plan more effectively while processing larger volumes of input text. Notes Writing is framework agnostic and can be integrated with different iterative RAG methods. We demonstrate its effectiveness with three iterative RAG methods, across two models and four evaluation datasets. Notes writing yields an average improvement of 15.6 percentage points overall, with minimal increase in output tokens.
>
---
#### [new 014] CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出CLEAR框架，用于放射报告质量评估。针对现有指标无法细致捕捉临床差异的问题，其通过专家标注的五属性（首次出现、变化、严重性、位置、建议）对比报告，实现多维度临床可解释评估，并构建CLEAR-Bench数据集验证，与临床判断高度一致。**

- **链接: [http://arxiv.org/pdf/2505.16325v1](http://arxiv.org/pdf/2505.16325v1)**

> **作者:** Yuyang Jiang; Chacha Chen; Shengyuan Wang; Feng Li; Zecong Tang; Benjamin M. Mervak; Lydia Chelala; Christopher M Straus; Reve Chahine; Samuel G. Armato III; Chenhao Tan
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Existing metrics often lack the granularity and interpretability to capture nuanced clinical differences between candidate and ground-truth radiology reports, resulting in suboptimal evaluation. We introduce a Clinically-grounded tabular framework with Expert-curated labels and Attribute-level comparison for Radiology report evaluation (CLEAR). CLEAR not only examines whether a report can accurately identify the presence or absence of medical conditions, but also assesses whether it can precisely describe each positively identified condition across five key attributes: first occurrence, change, severity, descriptive location, and recommendation. Compared to prior works, CLEAR's multi-dimensional, attribute-level outputs enable a more comprehensive and clinically interpretable evaluation of report quality. Additionally, to measure the clinical alignment of CLEAR, we collaborate with five board-certified radiologists to develop CLEAR-Bench, a dataset of 100 chest X-ray reports from MIMIC-CXR, annotated across 6 curated attributes and 13 CheXpert conditions. Our experiments show that CLEAR achieves high accuracy in extracting clinical attributes and provides automated metrics that are strongly aligned with clinical judgment.
>
---
#### [new 015] LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于机器翻译质量评估（MTQE）任务，旨在解决现有LLM直接评分方法与人类判断相关性低的问题。提出生成式评估方法：利用解码器LLM生成高质量参考译文，再通过语义相似度计算评分。实验覆盖8种LLM和8种语言，结果优于传统直接评分和非LLM指标，证明生成结合语义评估的混合方法更优。**

- **链接: [http://arxiv.org/pdf/2505.16129v1](http://arxiv.org/pdf/2505.16129v1)**

> **作者:** Hyang Cui
>
> **备注:** 5 pages, 2 figures, 2 tables. Conforms to the ACL Rolling Review (ARR) short paper track. Code and data available at: https://github.com/CuiNiki/LLMs-Are-Not-Scorers
>
> **摘要:** Recent studies have applied large language models (LLMs) to machine translation quality estimation (MTQE) by prompting models to assign numeric scores. Nonetheless, these direct scoring methods tend to show low segment-level correlation with human judgments. In this paper, we propose a generation-based evaluation paradigm that leverages decoder-only LLMs to produce high-quality references, followed by semantic similarity scoring using sentence embeddings. We conduct the most extensive evaluation to date in MTQE, covering 8 LLMs and 8 language pairs. Empirical results show that our method outperforms both intra-LLM direct scoring baselines and external non-LLM reference-free metrics from MTME. These findings demonstrate the strength of generation-based evaluation and support a shift toward hybrid approaches that combine fluent generation with accurate semantic assessment.
>
---
#### [new 016] PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PMPO框架，针对现有提示优化方法依赖高成本生成或人工标注的问题，通过token级交叉熵损失评估直接优化提示，屏蔽并改进低质量片段，无需采样或人工干预，适用于多种任务，实验显示其在模型大小和任务上均表现优异。**

- **链接: [http://arxiv.org/pdf/2505.16307v1](http://arxiv.org/pdf/2505.16307v1)**

> **作者:** Chenzhuo Zhao; Ziqian Liu; Xingda Wang; Junting Lu; Chaoyi Ruan
>
> **摘要:** Prompt optimization offers a practical and broadly applicable alternative to fine-tuning for improving large language model (LLM) performance. However, existing methods often rely on costly output generation, self-critiquing abilities, or human-annotated preferences, which limit their scalability, especially for smaller or non-instruction-tuned models. We introduce PMPO (Probabilistic Metric Prompt Optimization), a unified framework that refines prompts using token-level cross-entropy loss as a direct, lightweight evaluation signal. PMPO identifies low-quality prompt segments by masking and measuring their impact on loss, then rewrites and selects improved variants by minimizing loss over positive and negative examples. Unlike prior methods, it requires no output sampling or human evaluation during optimization, relying only on forward passes and log-likelihoods. PMPO supports both supervised and preference-based tasks through a closely aligned loss-based evaluation strategy. Experiments show that PMPO consistently outperforms prior methods across model sizes and tasks: it achieves the highest average accuracy on BBH, performs strongly on GSM8K and AQUA-RAT, and improves AlpacaEval 2.0 win rates by over 19 points. These results highlight PMPO's effectiveness, efficiency, and broad applicability.
>
---
#### [new 017] Can reasoning models comprehend mathematical problems in Chinese ancient texts? An empirical study based on data from Suanjing Shishu
- **分类: cs.CL; cs.AI**

- **简介: 论文构建Guji_MATH基准，评估推理模型对中国古代数学问题（基于《算经十书》）的处理能力。解决模型在古典中文数学题中的理解与解题挑战。工作包括提取538题形成结构化数据集，设计双评估模式测试六模型，结果显示需优化古典中文和文化知识理解，为古籍知识挖掘与文化传播提供方法支持。**

- **链接: [http://arxiv.org/pdf/2505.16660v1](http://arxiv.org/pdf/2505.16660v1)**

> **作者:** Liu Chang; Wang Dongbo; Liu liu; Zhao Zhixiao
>
> **备注:** 29pages, 7 figures
>
> **摘要:** This study addresses the challenges in intelligent processing of Chinese ancient mathematical classics by constructing Guji_MATH, a benchmark for evaluating classical texts based on Suanjing Shishu. It systematically assesses the mathematical problem-solving capabilities of mainstream reasoning models under the unique linguistic constraints of classical Chinese. Through machine-assisted annotation and manual verification, 538 mathematical problems were extracted from 8 canonical texts, forming a structured dataset centered on the "Question-Answer-Solution" framework, supplemented by problem types and difficulty levels. Dual evaluation modes--closed-book (autonomous problem-solving) and open-book (reproducing classical solution methods)--were designed to evaluate the performance of six reasoning models on ancient Chinese mathematical problems. Results indicate that reasoning models can partially comprehend and solve these problems, yet their overall performance remains inferior to benchmarks on modern mathematical tasks. Enhancing models' classical Chinese comprehension and cultural knowledge should be prioritized for optimization. This study provides methodological support for mining mathematical knowledge from ancient texts and disseminating traditional culture, while offering new perspectives for evaluating cross-linguistic and cross-cultural capabilities of reasoning models.
>
---
#### [new 018] ScholarBench: A Bilingual Benchmark for Abstraction, Comprehension, and Reasoning Evaluation in Academic Contexts
- **分类: cs.CL**

- **简介: 该论文提出双语学术评测基准ScholarBench，针对现有模型在专业领域复杂任务上的不足，通过构建覆盖8个学科、5类题型的5000+中英双语问题，评估模型的抽象、理解与推理能力，挑战性测试显示当前最佳模型平均得分仅0.543。**

- **链接: [http://arxiv.org/pdf/2505.16566v1](http://arxiv.org/pdf/2505.16566v1)**

> **作者:** Dongwon Noh; Donghyeok Koh; Junghun Yuk; Gyuwan Kim; Jaeyong Lee; Kyungtae Lim; Cheoneum Park
>
> **摘要:** Prior benchmarks for evaluating the domain-specific knowledge of large language models (LLMs) lack the scalability to handle complex academic tasks. To address this, we introduce \texttt{ScholarBench}, a benchmark centered on deep expert knowledge and complex academic problem-solving, which evaluates the academic reasoning ability of LLMs and is constructed through a three-step process. \texttt{ScholarBench} targets more specialized and logically complex contexts derived from academic literature, encompassing five distinct problem types. Unlike prior benchmarks, \texttt{ScholarBench} evaluates the abstraction, comprehension, and reasoning capabilities of LLMs across eight distinct research domains. To ensure high-quality evaluation data, we define category-specific example attributes and design questions that are aligned with the characteristic research methodologies and discourse structures of each domain. Additionally, this benchmark operates as an English-Korean bilingual dataset, facilitating simultaneous evaluation for linguistic capabilities of LLMs in both languages. The benchmark comprises 5,031 examples in Korean and 5,309 in English, with even state-of-the-art models like o3-mini achieving an average evaluation score of only 0.543, demonstrating the challenging nature of this benchmark.
>
---
#### [new 019] Explain Less, Understand More: Jargon Detection via Personalized Parameter-Efficient Fine-tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于个性化专业术语检测任务，旨在解决传统用户特定模型微调资源消耗大、成本高的问题。研究提出基于LoRA的轻量级微调与个性化提示策略，并结合少量标注数据和无监督用户背景信号，实现高效、低资源的个性化检测，性能超GPT-4且数据效率提升10倍。**

- **链接: [http://arxiv.org/pdf/2505.16227v1](http://arxiv.org/pdf/2505.16227v1)**

> **作者:** Bohao Wu; Qingyun Wang; Yue Guo
>
> **摘要:** Personalizing jargon detection and explanation is essential for making technical documents accessible to readers with diverse disciplinary backgrounds. However, tailoring models to individual users typically requires substantial annotation efforts and computational resources due to user-specific finetuning. To address this, we present a systematic study of personalized jargon detection, focusing on methods that are both efficient and scalable for real-world deployment. We explore two personalization strategies: (1) lightweight fine-tuning using Low-Rank Adaptation (LoRA) on open-source models, and (2) personalized prompting, which tailors model behavior at inference time without retaining. To reflect realistic constraints, we also investigate hybrid approaches that combine limited annotated data with unsupervised user background signals. Our personalized LoRA model outperforms GPT-4 by 21.4% in F1 score and exceeds the best performing oracle baseline by 8.3%. Remarkably, our method achieves comparable performance using only 10% of the annotated training data, demonstrating its practicality for resource-constrained settings. Our study offers the first work to systematically explore efficient, low-resource personalization of jargon detection using open-source language models, offering a practical path toward scalable, user-adaptive NLP system.
>
---
#### [new 020] Comparative analysis of subword tokenization approaches for Indian languages
- **分类: cs.CL**

- **简介: 该论文比较了子词分词方法（SentencePiece、BPE、WordPiece）在印度语言机器翻译中的效果，旨在解决复杂形态语言的分词优化问题。研究通过统计、神经及多语种神经翻译模型测试这些方法，使用BLEU等指标评估，发现SentencePiece在多数单语模型中表现最佳，而BPE更适合多语种模型，且印度语→英语翻译效果更优。**

- **链接: [http://arxiv.org/pdf/2505.16868v1](http://arxiv.org/pdf/2505.16868v1)**

> **作者:** Sudhansu Bala Das; Samujjal Choudhury; Tapas Kumar Mishra; Bidyut Kr. Patra
>
> **备注:** 24 pages, 4 tables
>
> **摘要:** Tokenization is the act of breaking down text into smaller parts, or tokens, that are easier for machines to process. This is a key phase in machine translation (MT) models. Subword tokenization enhances this process by breaking down words into smaller subword units, which is especially beneficial in languages with complicated morphology or a vast vocabulary. It is useful in capturing the intricate structure of words in Indian languages (ILs), such as prefixes, suffixes, and other morphological variations. These languages frequently use agglutinative structures, in which words are formed by the combination of multiple morphemes such as suffixes, prefixes, and stems. As a result, a suitable tokenization strategy must be chosen to address these scenarios. This paper examines how different subword tokenization techniques, such as SentencePiece, Byte Pair Encoding (BPE), and WordPiece Tokenization, affect ILs. The effectiveness of these subword tokenization techniques is investigated in statistical, neural, and multilingual neural machine translation models. All models are examined using standard evaluation metrics, such as the Bilingual Evaluation Understudy (BLEU) score, TER, METEOR, CHRF, RIBES, and COMET. Based on the results, it appears that for the majority of language pairs for the Statistical and Neural MT models, the SentencePiece tokenizer continuously performed better than other tokenizers in terms of BLEU score. However, BPE tokenization outperformed other tokenization techniques in the context of Multilingual Neural Machine Translation model. The results show that, despite using the same tokenizer and dataset for each model, translations from ILs to English surpassed translations from English to ILs.
>
---
#### [new 021] SLMEval: Entropy-Based Calibration for Human-Aligned Evaluation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大语言模型评估任务，针对现有校准方法在开放任务中与人类判断相关性弱或负的问题，提出SLMEval方法。通过熵最大化和少量人类偏好数据优化评分分布，提升与人类判断的关联性（如Spearman达0.57），并降低5-30倍成本。**

- **链接: [http://arxiv.org/pdf/2505.16003v1](http://arxiv.org/pdf/2505.16003v1)**

> **作者:** Roland Daynauth; Christopher Clarke; Krisztian Flautner; Lingjia Tang; Jason Mars
>
> **摘要:** The LLM-as-a-Judge paradigm offers a scalable, reference-free approach for evaluating language models. Although several calibration techniques have been proposed to better align these evaluators with human judgment, prior studies focus primarily on narrow, well-structured benchmarks. As a result, it remains unclear whether such calibrations generalize to real-world, open-ended tasks. In this work, we show that SOTA calibrated evaluators often fail in these settings, exhibiting weak or even negative correlation with human judgments. To address this, we propose SLMEval, a novel and efficient calibration method based on entropy maximization over a small amount of human preference data. By estimating a latent distribution over model quality and reweighting evaluator scores accordingly, SLMEval achieves strong correlation with human evaluations across two real-world production use cases and the public benchmark. For example, on one such task, SLMEval achieves a Spearman correlation of 0.57 with human judgments, while G-Eval yields a negative correlation. In addition, SLMEval reduces evaluation costs by 5-30x compared to GPT-4-based calibrated evaluators such as G-eval.
>
---
#### [new 022] Memorization or Reasoning? Exploring the Idiom Understanding of LLMs
- **分类: cs.CL**

- **简介: 该论文属于LLMs机制分析任务，探究其处理习语的原理。针对LLMs处理习语依赖记忆还是推理的疑问，提出跨六语种的MIDAS数据集，通过综合评测发现模型采用记忆与推理结合的混合策略，尤其在组合性习语中依赖上下文推理。**

- **链接: [http://arxiv.org/pdf/2505.16216v1](http://arxiv.org/pdf/2505.16216v1)**

> **作者:** Jisu Kim; Youngwoo Shin; Uiji Hwang; Jihun Choi; Richeng Xuan; Taeuk Kim
>
> **摘要:** Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
>
---
#### [new 023] Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出从大语言模型（LLMs）提取概率知识以参数化贝叶斯网络（BN），解决LLMs概率知识应用不足及数据稀缺下的建模偏差问题。通过实验对比多种方法，验证LLM在估计条件概率的效用，并探索其作为专家先验结合少量数据优化BN参数，建立评估基准。**

- **链接: [http://arxiv.org/pdf/2505.15918v1](http://arxiv.org/pdf/2505.15918v1)**

> **作者:** Aliakbar Nafar; Kristen Brent Venable; Zijun Cui; Parisa Kordjamshidi
>
> **摘要:** Large Language Models (LLMs) have demonstrated potential as factual knowledge bases; however, their capability to generate probabilistic knowledge about real-world events remains understudied. This paper investigates using probabilistic knowledge inherent in LLMs to derive probability estimates for statements concerning events and their interrelationships captured via a Bayesian Network (BN). Using LLMs in this context allows for the parameterization of BNs, enabling probabilistic modeling within specific domains. Experiments on eighty publicly available Bayesian Networks, from healthcare to finance, demonstrate that querying LLMs about the conditional probabilities of events provides meaningful results when compared to baselines, including random and uniform distributions, as well as approaches based on next-token generation probabilities. We explore how these LLM-derived distributions can serve as expert priors to refine distributions extracted from minimal data, significantly reducing systematic biases. Overall, this work introduces a promising strategy for automatically constructing Bayesian Networks by combining probabilistic knowledge extracted from LLMs with small amounts of real-world data. Additionally, we evaluate several prompting strategies for eliciting probabilistic knowledge from LLMs and establish the first comprehensive baseline for assessing LLM performance in extracting probabilistic knowledge.
>
---
#### [new 024] Do Large Language Models Excel in Complex Logical Reasoning with Formal Language?
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估大语言模型（LLMs）在正式语言逻辑推理中的表现，针对系统评测不足及提升泛化能力问题，通过模型光谱、任务分类和轨迹格式三个维度进行综合评测，发现思考模型表现更优，归纳推理普遍受限，PoT格式泛化最佳；并提出拒绝微调方法优化小模型，提升跨语言泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.16998v1](http://arxiv.org/pdf/2505.16998v1)**

> **作者:** Jin Jiang; Jianing Wang; Yuchen Yan; Yang Liu; Jianhua Zhu; Mengdi Zhang; Xunliang Cai; Liangcai Gao
>
> **摘要:** Large Language Models (LLMs) have been shown to achieve breakthrough performance on complex logical reasoning tasks. Nevertheless, most existing research focuses on employing formal language to guide LLMs to derive reliable reasoning paths, while systematic evaluations of these capabilities are still limited. In this paper, we aim to conduct a comprehensive evaluation of LLMs across various logical reasoning problems utilizing formal languages. From the perspective of three dimensions, i.e., spectrum of LLMs, taxonomy of tasks, and format of trajectories, our key findings are: 1) Thinking models significantly outperform Instruct models, especially when formal language is employed; 2) All LLMs exhibit limitations in inductive reasoning capability, irrespective of whether they use a formal language; 3) Data with PoT format achieves the best generalization performance across other languages. Additionally, we also curate the formal-relative training data to further enhance the small language models, and the experimental results indicate that a simple rejected fine-tuning method can better enable LLMs to generalize across formal languages and achieve the best overall performance. Our codes and reports are available at https://github.com/jiangjin1999/FormalEval.
>
---
#### [new 025] Small Language Models in the Real World: Insights from Industrial Text Classification
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，探讨小型语言模型在工业场景（如邮件分类、法律文档分类及长文本处理）中的应用潜力。针对大模型推理效率低、资源消耗高的问题，通过评估提示工程与监督微调方法，分析小模型的性能与VRAM效率，为工业本地部署提供优化策略。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16078v1](http://arxiv.org/pdf/2505.16078v1)**

> **作者:** Lujun Li; Lama Sleem; Niccolo' Gentile; Geoffrey Nichil; Radu State
>
> **摘要:** With the emergence of ChatGPT, Transformer models have significantly advanced text classification and related tasks. Decoder-only models such as Llama exhibit strong performance and flexibility, yet they suffer from inefficiency on inference due to token-by-token generation, and their effectiveness in text classification tasks heavily depends on prompt quality. Moreover, their substantial GPU resource requirements often limit widespread adoption. Thus, the question of whether smaller language models are capable of effectively handling text classification tasks emerges as a topic of significant interest. However, the selection of appropriate models and methodologies remains largely underexplored. In this paper, we conduct a comprehensive evaluation of prompt engineering and supervised fine-tuning methods for transformer-based text classification. Specifically, we focus on practical industrial scenarios, including email classification, legal document categorization, and the classification of extremely long academic texts. We examine the strengths and limitations of smaller models, with particular attention to both their performance and their efficiency in Video Random-Access Memory (VRAM) utilization, thereby providing valuable insights for the local deployment and application of compact models in industrial settings.
>
---
#### [new 026] SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于深度搜索任务，解决复杂场景下多步骤推理的训练数据稀缺与环境分布不匹配问题。提出SimpleDeepSearcher框架，通过模拟真实用户搜索生成高质量训练数据，并采用多准则筛选策略优化数据质量，仅用871样本SFT即超越RL基线，验证轻量框架在数据受限场景的有效性。**

- **链接: [http://arxiv.org/pdf/2505.16834v1](http://arxiv.org/pdf/2505.16834v1)**

> **作者:** Shuang Sun; Huatong Song; Yuhao Wang; Ruiyang Ren; Jinhao Jiang; Junjie Zhang; Fei Bai; Jia Deng; Wayne Xin Zhao; Zheng Liu; Lei Fang; Zhongyuan Wang; Ji-Rong Wen
>
> **摘要:** Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at https://github.com/RUCAIBox/SimpleDeepSearcher.
>
---
#### [new 027] Large Language Models based ASR Error Correction for Child Conversations
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究基于大型语言模型（LLMs）修正儿童对话的自动语音识别（ASR）错误。旨在解决ASR在儿童语音转录中的准确性问题。通过在两个数据集上实验，比较LLMs对零样本及微调ASR（如CTC模型）的纠错效果，发现其对零样本和CTC有效，但难以提升自回归模型（如Whisper）的上下文纠错能力。**

- **链接: [http://arxiv.org/pdf/2505.16212v1](http://arxiv.org/pdf/2505.16212v1)**

> **作者:** Anfeng Xu; Tiantian Feng; So Hyun Kim; Somer Bishop; Catherine Lord; Shrikanth Narayanan
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) has recently shown remarkable progress, but accurately transcribing children's speech remains a significant challenge. Recent developments in Large Language Models (LLMs) have shown promise in improving ASR transcriptions. However, their applications in child speech including conversational scenarios are underexplored. In this study, we explore the use of LLMs in correcting ASR errors for conversational child speech. We demonstrate the promises and challenges of LLMs through experiments on two children's conversational speech datasets with both zero-shot and fine-tuned ASR outputs. We find that while LLMs are helpful in correcting zero-shot ASR outputs and fine-tuned CTC-based ASR outputs, it remains challenging for LLMs to improve ASR performance when incorporating contextual information or when using fine-tuned autoregressive ASR (e.g., Whisper) outputs.
>
---
#### [new 028] Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild
- **分类: cs.CL; cs.HC**

- **简介: 该论文分析用户使用LLM写作时的人机协作行为，通过研究Bing Copilot和WildChat的真实交互数据，识别出典型协作模式（PATHs），如意图修订、内容探索等，并发现写作意图与协作行为的关联，为优化LLM对齐提供依据。**

- **链接: [http://arxiv.org/pdf/2505.16023v1](http://arxiv.org/pdf/2505.16023v1)**

> **作者:** Sheshera Mysore; Debarati Das; Hancheng Cao; Bahareh Sarrafzadeh
>
> **备注:** Pre-print under-review
>
> **摘要:** As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large-scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human-AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users' intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.
>
---
#### [new 029] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training
- **分类: cs.CL**

- **简介: 该论文研究LLM预训练中元数据效用，评估发现URL加速训练且需长提示提升性能，而主题/格式元数据虽无法提速，但能控制生成内容，为可控生成提供依据。**

- **链接: [http://arxiv.org/pdf/2505.16570v1](http://arxiv.org/pdf/2505.16570v1)**

> **作者:** Dongyang Fan; Vinko Sabolčec; Martin Jaggi
>
> **摘要:** Large Language Models (LLMs) are commonly pretrained on vast corpora of text without utilizing contextual metadata such as source, quality, or topic, leading to a context-free learning paradigm. While recent studies suggest that adding metadata like URL information as context (i.e., auxiliary inputs not used in the loss calculation) can improve training efficiency and downstream performance, they offer limited understanding of which types of metadata are truly effective and under what conditions. In this work, we conduct a systematic evaluation and find that not all metadata types contribute equally. Only URL context speeds up training, whereas quality scores and topic/format domain information offer no clear benefit. Furthermore, the improved downstream performances of URL conditioning emerge only when longer prompts are used at inference time. In addition, we demonstrate that context-aware pretraining enables more controllable generation than context-free pretraining, in a classifier-free guidance fashion. Although topic and format metadata do not accelerate training, they are effective for steering outputs, offering human-interpretable control over generation.
>
---
#### [new 030] O$^2$-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出O²-Searcher，一种基于强化学习的搜索代理模型，用于开放领域的开放式与封闭式问答。针对大语言模型静态知识限制及开放式问题研究不足，其通过模拟搜索环境动态获取知识，设计奖励函数使代理自适应策略，并构建O²-QA基准。实验显示其在开放任务上超越现有方法，同时在封闭任务中达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.16582v1](http://arxiv.org/pdf/2505.16582v1)**

> **作者:** Jianbiao Mei; Tao Hu; Daocheng Fu; Licheng Wen; Xuemeng Yang; Rong Wu; Pinlong Cai; Xing Gao; Yu Yang; Chengjun Xie; Botian Shi; Yong Liu; Yu Qiao
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs), despite their advancements, are fundamentally limited by their static parametric knowledge, hindering performance on tasks requiring open-domain up-to-date information. While enabling LLMs to interact with external knowledge environments is a promising solution, current efforts primarily address closed-end problems. Open-ended questions, which characterized by lacking a standard answer or providing non-unique and diverse answers, remain underexplored. To bridge this gap, we present O$^2$-Searcher, a novel search agent leveraging reinforcement learning to effectively tackle both open-ended and closed-ended questions in the open domain. O$^2$-Searcher leverages an efficient, locally simulated search environment for dynamic knowledge acquisition, effectively decoupling the external world knowledge from model's sophisticated reasoning processes. It employs a unified training mechanism with meticulously designed reward functions, enabling the agent to identify problem types and adapt different answer generation strategies. Furthermore, to evaluate performance on complex open-ended tasks, we construct O$^2$-QA, a high-quality benchmark featuring 300 manually curated, multi-domain open-ended questions with associated web page caches. Extensive experiments show that O$^2$-Searcher, using only a 3B model, significantly surpasses leading LLM agents on O$^2$-QA. It also achieves SOTA results on various closed-ended QA benchmarks against similarly-sized models, while performing on par with much larger ones.
>
---
#### [new 031] EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios
- **分类: cs.CL**

- **简介: 该论文提出EduBench，首个针对教育场景的多维度基准评测数据集，解决大模型在教育领域应用不足的问题。工作包括构建含9类场景的4000+教育情境数据、设计覆盖12项指标的评估体系，通过人工标注确保质量，并证明小模型在该数据集上可媲美大模型性能。**

- **链接: [http://arxiv.org/pdf/2505.16160v1](http://arxiv.org/pdf/2505.16160v1)**

> **作者:** Bin Xu; Yu Bai; Huashan Sun; Yiguan Lin; Siming Liu; Xinyue Liang; Yaolin Li; Yang Gao; Heyan Huang
>
> **摘要:** As large language models continue to advance, their application in educational contexts remains underexplored and under-optimized. In this paper, we address this gap by introducing the first diverse benchmark tailored for educational scenarios, incorporating synthetic data containing 9 major scenarios and over 4,000 distinct educational contexts. To enable comprehensive assessment, we propose a set of multi-dimensional evaluation metrics that cover 12 critical aspects relevant to both teachers and students. We further apply human annotation to ensure the effectiveness of the model-generated evaluation responses. Additionally, we succeed to train a relatively small-scale model on our constructed dataset and demonstrate that it can achieve performance comparable to state-of-the-art large models (e.g., Deepseek V3, Qwen Max) on the test set. Overall, this work provides a practical foundation for the development and evaluation of education-oriented language models. Code and data are released at https://github.com/ybai-nlp/EduBench.
>
---
#### [new 032] VeriFastScore: Speeding up long-form factuality evaluation
- **分类: cs.CL**

- **简介: 论文提出VeriFastScore，通过微调Llama3.1模型加速长文本事实性评估。解决现有方法LLM调用多、耗时长的问题，利用合成数据实现同时提取和验证声明，速度提升6.6倍，与原方法强相关（r=0.80/0.94），并公开模型和数据。**

- **链接: [http://arxiv.org/pdf/2505.16973v1](http://arxiv.org/pdf/2505.16973v1)**

> **作者:** Rishanth Rajendhran; Amir Zadeh; Matthew Sarte; Chuan Li; Mohit Iyyer
>
> **摘要:** Metrics like FactScore and VeriScore that evaluate long-form factuality operate by decomposing an input response into atomic claims and then individually verifying each claim. While effective and interpretable, these methods incur numerous LLM calls and can take upwards of 100 seconds to evaluate a single response, limiting their practicality in large-scale evaluation and training scenarios. To address this, we propose VeriFastScore, which leverages synthetic data to fine-tune Llama3.1 8B for simultaneously extracting and verifying all verifiable claims within a given text based on evidence from Google Search. We show that this task cannot be solved via few-shot prompting with closed LLMs due to its complexity: the model receives ~4K tokens of evidence on average and needs to concurrently decompose claims, judge their verifiability, and verify them against noisy evidence. However, our fine-tuned VeriFastScore model demonstrates strong correlation with the original VeriScore pipeline at both the example level (r=0.80) and system level (r=0.94) while achieving an overall speedup of 6.6x (9.9x excluding evidence retrieval) over VeriScore. To facilitate future factuality research, we publicly release our VeriFastScore model and synthetic datasets.
>
---
#### [new 033] INFERENCEDYNAMICS: Efficient Routing Across LLMs through Structured Capability and Knowledge Profiling
- **分类: cs.CL**

- **简介: 该论文属于LLM路由优化任务，解决现有方法在扩展性和适应性上的不足。提出InferenceDynamics框架，通过建模模型能力和知识实现多维路由，在RouteMix数据集和多个基准测试中验证了其高效选择最优模型、提升性能与资源利用率的效果。**

- **链接: [http://arxiv.org/pdf/2505.16303v1](http://arxiv.org/pdf/2505.16303v1)**

> **作者:** Haochen Shi; Tianshi Zheng; Weiqi Wang; Baixuan Xu; Chunyang Li; Chunkit Chan; Tao Fan; Yangqiu Song; Qiang Yang
>
> **备注:** 17 pages
>
> **摘要:** Large Language Model (LLM) routing is a pivotal technique for navigating a diverse landscape of LLMs, aiming to select the best-performing LLMs tailored to the domains of user queries, while managing computational resources. However, current routing approaches often face limitations in scalability when dealing with a large pool of specialized LLMs, or in their adaptability to extending model scope and evolving capability domains. To overcome those challenges, we propose InferenceDynamics, a flexible and scalable multi-dimensional routing framework by modeling the capability and knowledge of models. We operate it on our comprehensive dataset RouteMix, and demonstrate its effectiveness and generalizability in group-level routing using modern benchmarks including MMLU-Pro, GPQA, BigGenBench, and LiveBench, showcasing its ability to identify and leverage top-performing models for given tasks, leading to superior outcomes with efficient resource utilization. The broader adoption of Inference Dynamics can empower users to harness the full specialized potential of the LLM ecosystem, and our code will be made publicly available to encourage further research.
>
---
#### [new 034] Sparse Activation Editing for Reliable Instruction Following in Narratives
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文针对语言模型在复杂叙事场景中难以准确遵循指令的问题，提出无需训练的Concise-SAE框架，通过自然语言指令编辑相关神经元，并构建FreeInstruct基准（1212例）评估效果，提升多任务指令遵循能力。**

- **链接: [http://arxiv.org/pdf/2505.16505v1](http://arxiv.org/pdf/2505.16505v1)**

> **作者:** Runcong Zhao; Chengyu Cao; Qinglin Zhu; Xiucheng Lv; Shun Shao; Lin Gui; Ruifeng Xu; Yulan He
>
> **摘要:** Complex narrative contexts often challenge language models' ability to follow instructions, and existing benchmarks fail to capture these difficulties. To address this, we propose Concise-SAE, a training-free framework that improves instruction following by identifying and editing instruction-relevant neurons using only natural language instructions, without requiring labelled data. To thoroughly evaluate our method, we introduce FreeInstruct, a diverse and realistic benchmark of 1,212 examples that highlights the challenges of instruction following in narrative-rich settings. While initially motivated by complex narratives, Concise-SAE demonstrates state-of-the-art instruction adherence across varied tasks without compromising generation quality.
>
---
#### [new 035] Does Localization Inform Unlearning? A Rigorous Examination of Local Parameter Attribution for Knowledge Unlearning in Language Models
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于语言模型知识遗忘任务，旨在验证局部参数更新对有效遗忘的因果性。通过重新审视现有方法并设计受控实验，发现参数修改范围不严格确定，挑战了"局部参数决定知识定位"的核心假设。**

- **链接: [http://arxiv.org/pdf/2505.16252v1](http://arxiv.org/pdf/2505.16252v1)**

> **作者:** Hwiyeong Lee; Uiji Hwang; Hyelim Lim; Taeuk Kim
>
> **摘要:** Large language models often retain unintended content, prompting growing interest in knowledge unlearning. Recent approaches emphasize localized unlearning, which restricts parameter updates to specific regions in an effort to remove target knowledge while preserving unrelated general knowledge. However, their effectiveness remains uncertain due to the lack of robust and thorough evaluation of the trade-off between the competing goals of unlearning. In this paper, we begin by revisiting existing localized unlearning approaches. We then conduct controlled experiments to rigorously evaluate whether local parameter updates causally contribute to unlearning. Our findings reveal that the set of parameters that must be modified for effective unlearning is not strictly determined, challenging the core assumption of localized unlearning that parameter locality is inherently indicative of effective knowledge removal.
>
---
#### [new 036] Align-GRAG: Reasoning-Guided Dual Alignment for Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于图检索增强生成任务，旨在解决图RAG中无关节点冗余及图-语言表示鸿沟问题。提出Align-GRAG框架，在检索后通过KL散度和对比损失实现图节点与语言表征的双对齐，优化图编码器与LLM推理，提升生成准确性与效率。**

- **链接: [http://arxiv.org/pdf/2505.16237v1](http://arxiv.org/pdf/2505.16237v1)**

> **作者:** Derong Xu; Pengyue Jia; Xiaopeng Li; Yingyi Zhang; Maolin Wang; Qidong Liu; Xiangyu Zhao; Yichao Wang; Huifeng Guo; Ruiming Tang; Enhong Chen; Tong Xu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities, but still struggle with issues like hallucinations and outdated information. Retrieval-augmented generation (RAG) addresses these issues by grounding LLM outputs in external knowledge with an Information Retrieval (IR) system. Building on this foundation, graph-based RAG systems go a step further by retrieving subgraphs, which preserve the relationships between knowledge entities and provide more comprehensive context. However, graph RAG faces two challenges: (1) Retrieving relevant information introduces irrelevant nodes (especially in dense graph databases, where retrieval usually extends to adjacent nodes), and leads to overly lengthy inputs that hinder efficiency; (2) The representation gap between graph and language during generation with LLMs limits the ability to fully leverage graph structures for enhanced understanding. To address these limitations, we propose Align-GRAG, a novel reasoning-guided dual alignment framework in post-retrieval phrase. It first formulates a subgraph by retrieving nodes and edges. Then an Aligner is proposed to jointly optimizes a graph encoder with LLM-summarized reasoning. It achieves dual alignment of graph node and representation by leveraging KL divergence loss and contrastive loss, facilitating efficient pruning of irrelevant knowledge and establishing a unified semantic space. The Generator integrates the aligned graph data with LLM to produce coherent and accurate answers. Experiments on GraphQA benchmark across three tasks (including common sense reasoning, scene graph understanding, and knowledge graph reasoning) validate the effectiveness of our method. The code will be available upon accepted.
>
---
#### [new 037] NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出NOVER框架，解决激励训练依赖外部验证器及高成本标注数据的问题。通过仅使用标准监督数据的无验证器强化学习，提升语言模型在文本任务中的推理性能，优于同类蒸馏模型，拓展了优化可能性。**

- **链接: [http://arxiv.org/pdf/2505.16022v1](http://arxiv.org/pdf/2505.16022v1)**

> **作者:** Wei Liu; Siya Qi; Xinyu Wang; Chen Qian; Yali Du; Yulan He
>
> **备注:** 20 pages, 5 tables, 12 figures
>
> **摘要:** Recent advances such as DeepSeek R1-Zero highlight the effectiveness of incentive training, a reinforcement learning paradigm that computes rewards solely based on the final answer part of a language model's output, thereby encouraging the generation of intermediate reasoning steps. However, these methods fundamentally rely on external verifiers, which limits their applicability to domains like mathematics and coding where such verifiers are readily available. Although reward models can serve as verifiers, they require high-quality annotated data and are costly to train. In this work, we propose NOVER, NO-VERifier Reinforcement Learning, a general reinforcement learning framework that requires only standard supervised fine-tuning data with no need for an external verifier. NOVER enables incentive training across a wide range of text-to-text tasks and outperforms the model of the same size distilled from large reasoning models such as DeepSeek R1 671B by 7.7 percent. Moreover, the flexibility of NOVER enables new possibilities for optimizing large language models, such as inverse incentive training.
>
---
#### [new 038] An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态情感分析（MSA）任务，旨在解决零样本学习下多模态语言模型（MLLMs）情感感知能力不足的问题。通过研究In-Context Learning（ICL）中演示样本的检索、呈现和分布三个关键因素，优化配置策略并纠正模型情感预测偏差，使准确率较零样本和随机ICL基线分别提升15.9%和11.2%。**

- **链接: [http://arxiv.org/pdf/2505.16193v1](http://arxiv.org/pdf/2505.16193v1)**

> **作者:** Daiqing Wu; Dongbao Yang; Sicheng Zhao; Can Ma; Yu Zhou
>
> **摘要:** The advancements in Multimodal Large Language Models (MLLMs) have enabled various multimodal tasks to be addressed under a zero-shot paradigm. This paradigm sidesteps the cost of model fine-tuning, emerging as a dominant trend in practical application. Nevertheless, Multimodal Sentiment Analysis (MSA), a pivotal challenge in the quest for general artificial intelligence, fails to accommodate this convenience. The zero-shot paradigm exhibits undesirable performance on MSA, casting doubt on whether MLLMs can perceive sentiments as competent as supervised models. By extending the zero-shot paradigm to In-Context Learning (ICL) and conducting an in-depth study on configuring demonstrations, we validate that MLLMs indeed possess such capability. Specifically, three key factors that cover demonstrations' retrieval, presentation, and distribution are comprehensively investigated and optimized. A sentimental predictive bias inherent in MLLMs is also discovered and later effectively counteracted. By complementing each other, the devised strategies for three factors result in average accuracy improvements of 15.9% on six MSA datasets against the zero-shot paradigm and 11.2% against the random ICL baseline.
>
---
#### [new 039] Date Fragments: A Hidden Bottleneck of Tokenization for Temporal Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间推理任务，解决BPE分词器拆分日期导致结构丢失的问题。提出日期碎片率指标、DateAugBench基准数据集，并揭示大模型通过注意力机制拼接日期碎片的机制，发现碎片化严重时准确率下降10%，模型规模越大修复越高效。**

- **链接: [http://arxiv.org/pdf/2505.16088v1](http://arxiv.org/pdf/2505.16088v1)**

> **作者:** Gagan Bhatia; Maxime Peyrard; Wei Zhao
>
> **摘要:** Modern BPE tokenizers often split calendar dates into meaningless fragments, e.g., 20250312 $\rightarrow$ 202, 503, 12, inflating token counts and obscuring the inherent structure needed for robust temporal reasoning. In this work, we (1) introduce a simple yet interpretable metric, termed date fragmentation ratio, that measures how faithfully a tokenizer preserves multi-digit date components; (2) release DateAugBench, a suite of 6500 examples spanning three temporal reasoning tasks: context-based date resolution, format-invariance puzzles, and date arithmetic across historical, contemporary, and future regimes; and (3) through layer-wise probing and causal attention-hop analyses, uncover an emergent date-abstraction mechanism whereby large language models stitch together the fragments of month, day, and year components for temporal reasoning. Our experiments show that excessive fragmentation correlates with accuracy drops of up to 10 points on uncommon dates like historical and futuristic dates. Further, we find that the larger the model, the faster the emergent date abstraction that heals date fragments is accomplished. Lastly, we observe a reasoning path that LLMs follow to assemble date fragments, typically differing from human interpretation (year $\rightarrow$ month $\rightarrow$ day).
>
---
#### [new 040] Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task
- **分类: cs.CL**

- **简介: 该论文研究LLMs能否模拟人类在语音流畅任务中的行为变异性。通过测试34种模型配置（提示、温度、模型类型），与106人数据对比，发现虽部分配置接近人类平均，但多样性不足且结构僵硬，模型组合无效，揭示人类与模型的检索差异，指出LLMs在模拟认知上的局限。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16164v1](http://arxiv.org/pdf/2505.16164v1)**

> **作者:** Mengyang Qiu; Zoe Brisebois; Siena Sun
>
> **摘要:** Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 model configurations, varying prompt specificity, sampling temperature, and model type, and compared outputs to responses from 106 human participants. While some configurations, especially Claude 3.7 Sonnet, matched human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse and structurally rigid, and LLM ensembles failed to increase diversity. Network analyses further revealed fundamental differences in retrieval structure between humans and models. These results highlight key limitations in using LLMs to simulate human cognition and behavior.
>
---
#### [new 041] The Language of Interoception: Examining Embodiment and Emotion Through a Corpus of Body Part Mentions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与情感科学交叉研究，旨在探索语言中身体部位提及（BPMs）与情绪及健康的关系。通过构建包含博客和推文的BPMs语料库，并标注情绪，分析其使用模式与情感强度关联，发现BPMs高频出现且与负面健康结果显著相关，为NLP与人类福祉研究提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.16189v1](http://arxiv.org/pdf/2505.16189v1)**

> **作者:** Sophie Wu; Jan Philip Wahle; Saif M. Mohammad
>
> **备注:** 8 pages, 26 figures
>
> **摘要:** This paper is the first investigation of the connection between emotion, embodiment, and everyday language in a large sample of natural language data. We created corpora of body part mentions (BPMs) in online English text (blog posts and tweets). This includes a subset featuring human annotations for the emotions of the person whose body part is mentioned in the text. We show that BPMs are common in personal narratives and tweets (~5% to 10% of posts include BPMs) and that their usage patterns vary markedly by time and %geographic location. Using word-emotion association lexicons and our annotated data, we show that text containing BPMs tends to be more emotionally charged, even when the BPM is not explicitly used to describe a physical reaction to the emotion in the text. Finally, we discover a strong and statistically significant correlation between body-related language and a variety of poorer health outcomes. In sum, we argue that investigating the role of body-part related words in language can open up valuable avenues of future research at the intersection of NLP, the affective sciences, and the study of human wellbeing.
>
---
#### [new 042] Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation
- **分类: cs.CL; cs.SE**

- **简介: 该论文研究大语言模型（LLM）在代码评估中的偏见问题，探讨其能否公平处理语义相同但存在变量名、注释等表面差异的代码。提出六类潜在偏见类型，通过跨五种编程语言和多模型的实验，发现LLM普遍存在评分偏差，即使生成测试用例后仍无法避免，强调需改进评估方法。**

- **链接: [http://arxiv.org/pdf/2505.16222v1](http://arxiv.org/pdf/2505.16222v1)**

> **作者:** Jiwon Moon; Yerin Hwang; Dongryeol Lee; Taegwan Kang; Yongil Kim; Kyomin Jung
>
> **备注:** 26 pages
>
> **摘要:** With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to both positive and negative biases, resulting in inflated or unfairly low scores. Moreover, we observe that LLM judges remain vulnerable to these biases even when prompted to generate test cases before scoring, highlighting the need for more robust code evaluation methods.
>
---
#### [new 043] Understanding Fact Recall in Language Models: Why Two-Stage Training Encourages Memorization but Mixed Training Teaches Knowledge
- **分类: cs.CL**

- **简介: 该论文研究语言模型事实回忆的训练策略。对比两阶段训练（先储后忆，易死记）与混合训练（联合优化，助泛化），分析参数差异。提出跨任务梯度追踪法，发现混合训练通过共享参数提升知识泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.16178v1](http://arxiv.org/pdf/2505.16178v1)**

> **作者:** Ying Zhang; Benjamin Heinzerling; Dongyuan Li; Ryoma Ishigaki; Yuta Hitomi; Kentaro Inui
>
> **摘要:** Fact recall, the ability of language models (LMs) to retrieve specific factual knowledge, remains a challenging task despite their impressive general capabilities. Common training strategies often struggle to promote robust recall behavior with two-stage training, which first trains a model with fact-storing examples (e.g., factual statements) and then with fact-recalling examples (question-answer pairs), tending to encourage rote memorization rather than generalizable fact retrieval. In contrast, mixed training, which jointly uses both types of examples, has been empirically shown to improve the ability to recall facts, but the underlying mechanisms are still poorly understood. In this work, we investigate how these training strategies affect how model parameters are shaped during training and how these differences relate to their ability to recall facts. We introduce cross-task gradient trace to identify shared parameters, those strongly influenced by both fact-storing and fact-recalling examples. Our analysis on synthetic fact recall datasets with the Llama-3.2B and Pythia-2.8B models reveals that mixed training encouraging a larger and more centralized set of shared parameters. These findings suggest that the emergence of parameters may play a key role in enabling LMs to generalize factual knowledge across task formulations.
>
---
#### [new 044] R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出R1-Searcher++框架，解决LLMs因静态知识易幻觉的问题。通过结合内部知识与外部检索，采用两阶段训练（SFT冷启动+RL动态知识获取），设计奖励机制与记忆模块，提升推理效率与知识利用，优化RAG方法的性能与泛化。**

- **链接: [http://arxiv.org/pdf/2505.17005v1](http://arxiv.org/pdf/2505.17005v1)**

> **作者:** Huatong Song; Jinhao Jiang; Wenqing Tian; Zhipeng Chen; Yuhuan Wu; Jiahao Zhao; Yingqian Min; Wayne Xin Zhao; Lei Fang; Ji-Rong Wen
>
> **摘要:** Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at https://github.com/RUCAIBox/R1-Searcher-plus.
>
---
#### [new 045] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）在问题解决中的隐含偏见，揭示其在关联解决方案正确性与人口统计学特征时的两种真实性偏见：归因偏见（将正确解更归于特定群体）与评估偏见（因作者背景差异评估相同解）。实验跨数学、编码等任务，发现LLMs对非裔美国人正确解归因少、亚洲作者写作评分低，并在可视化中自动关联种族刻板印象颜色，警示其在教育评估中的应用风险。**

- **链接: [http://arxiv.org/pdf/2505.16128v1](http://arxiv.org/pdf/2505.16128v1)**

> **作者:** Yue Zhou; Barbara Di Eugenio
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Despite LLMs' explicit alignment against demographic stereotypes, they have been shown to exhibit biases under various social contexts. In this work, we find that LLMs exhibit concerning biases in how they associate solution veracity with demographics. Through experiments across five human value-aligned LLMs on mathematics, coding, commonsense, and writing problems, we reveal two forms of such veracity biases: Attribution Bias, where models disproportionately attribute correct solutions to certain demographic groups, and Evaluation Bias, where models' assessment of identical solutions varies based on perceived demographic authorship. Our results show pervasive biases: LLMs consistently attribute fewer correct solutions and more incorrect ones to African-American groups in math and coding, while Asian authorships are least preferred in writing evaluation. In additional studies, we show LLMs automatically assign racially stereotypical colors to demographic groups in visualization code, suggesting these biases are deeply embedded in models' reasoning processes. Our findings indicate that demographic bias extends beyond surface-level stereotypes and social context provocations, raising concerns about LLMs' deployment in educational and evaluation settings.
>
---
#### [new 046] Latent Principle Discovery for Language Model Self-Improvement
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型自我改进任务，旨在自动化发现隐含行为准则以提升生成质量。为解决人工标注原则耗时问题，提出通过自我修正框架从模型自身挖掘原则，结合聚类压缩为可解释准则，并利用后验正则化算法优化模型调用原则进行迭代改进。实验显示小模型在多个评测中性能显著提升。**

- **链接: [http://arxiv.org/pdf/2505.16927v1](http://arxiv.org/pdf/2505.16927v1)**

> **作者:** Keshav Ramji; Tahira Naseem; Ramón Fernandez Astudillo
>
> **摘要:** When language model (LM) users aim to improve the quality of its generations, it is crucial to specify concrete behavioral attributes that the model should strive to reflect. However, curating such principles across many domains, even non-exhaustively, requires a labor-intensive annotation process. To automate this process, we propose eliciting these latent attributes guiding model reasoning towards human-preferred responses by explicitly modeling them in a self-correction setting. Our approach mines new principles from the LM itself and compresses the discovered elements to an interpretable set via clustering. Specifically, we employ an approximation of posterior-regularized Monte Carlo Expectation-Maximization to both identify a condensed set of the most effective latent principles and teach the LM to strategically invoke them in order to intrinsically refine its responses. We demonstrate that bootstrapping our algorithm over multiple iterations enables smaller language models (7-8B parameters) to self-improve, achieving +8-10% in AlpacaEval win-rate, an average of +0.3 on MT-Bench, and +19-23% in principle-following win-rate on IFEval. We also show that clustering the principles yields interpretable and diverse model-generated constitutions while retaining model performance. The gains our method achieves highlight the potential of automated, principle-driven post-training recipes toward continual self-improvement.
>
---
#### [new 047] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出工具导向对话数据集T1，旨在解决LLMs在多轮对话中协调工具依赖及动态规划的挑战。通过构建覆盖9个领域的多轮数据集，集成缓存机制管理记忆，支持复用/重算决策，并作为开源模型评估基准，提升复杂场景下的规划能力。**

- **链接: [http://arxiv.org/pdf/2505.16986v1](http://arxiv.org/pdf/2505.16986v1)**

> **作者:** Amartya Chakraborty; Paresh Dashore; Nadia Bathaee; Anmol Jain; Anirban Das; Shi-Xiong Zhang; Sambit Sahu; Milind Naphade; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-source language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios.
>
---
#### [new 048] A Japanese Language Model and Three New Evaluation Benchmarks for Pharmaceutical NLP
- **分类: cs.CL**

- **简介: 该论文提出针对制药领域的日语语言模型，解决现有模型在术语密集型和知识推理任务中的不足。通过持续预训练构建领域专用模型，并设计三个新基准（YakugakuQA、NayoseQA、SogoCheck）评估模型表现。实验显示其超越开源模型，接近商业模型，揭示跨句一致性推理的挑战，同时开源资源促进后续研究。**

- **链接: [http://arxiv.org/pdf/2505.16661v1](http://arxiv.org/pdf/2505.16661v1)**

> **作者:** Issey Sukeda; Takuro Fujii; Kosei Buma; Shunsuke Sasaki; Shinnosuke Ono
>
> **备注:** 15 pages, 9 tables, 5 figures
>
> **摘要:** We present a Japanese domain-specific language model for the pharmaceutical field, developed through continual pretraining on 2 billion Japanese pharmaceutical tokens and 8 billion English biomedical tokens. To enable rigorous evaluation, we introduce three new benchmarks: YakugakuQA, based on national pharmacist licensing exams; NayoseQA, which tests cross-lingual synonym and terminology normalization; and SogoCheck, a novel task designed to assess consistency reasoning between paired statements. We evaluate our model against both open-source medical LLMs and commercial models, including GPT-4o. Results show that our domain-specific model outperforms existing open models and achieves competitive performance with commercial ones, particularly on terminology-heavy and knowledge-based tasks. Interestingly, even GPT-4o performs poorly on SogoCheck, suggesting that cross-sentence consistency reasoning remains an open challenge. Our benchmark suite offers a broader diagnostic lens for pharmaceutical NLP, covering factual recall, lexical variation, and logical consistency. This work demonstrates the feasibility of building practical, secure, and cost-effective language models for Japanese domain-specific applications, and provides reusable evaluation resources for future research in pharmaceutical and healthcare NLP. Our model, codes, and datasets are released at https://github.com/EQUES-Inc/pharma-LLM-eval.
>
---
#### [new 049] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对剪枝大视觉语言模型安全性下降问题，提出分层安全校准（HSR）方法。通过评估注意力头的安全贡献，恢复关键神经元，提升剪枝后模型的安全性，首次实现轻量级安全恢复并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.16104v1](http://arxiv.org/pdf/2505.16104v1)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Gerard de Melo; Xiaoling Wang; Linlin Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** With the increasing size of Large Vision-Language Models (LVLMs), network pruning techniques aimed at compressing models for deployment in resource-constrained environments have garnered significant attention. However, we observe that pruning often leads to a degradation in safety performance. To address this issue, we present a novel and lightweight approach, termed Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the contribution of each attention head to safety, identifying the most critical ones, and then selectively restoring neurons directly within these attention heads that play a pivotal role in maintaining safety. This process hierarchically realigns the safety of pruned LVLMs, progressing from the attention head level to the neuron level. We validate HSR across various models and pruning strategies, consistently achieving notable improvements in safety performance. To our knowledge, this is the first work explicitly focused on restoring safety in LVLMs post-pruning.
>
---
#### [new 050] Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization
- **分类: cs.CL**

- **简介: 该论文研究LLMs如何通过刻板印象进行隐式个性化，解决其导致少数群体回复质量低及持续偏见的问题。通过合成对话分析模型内部及回答，发现LLMs基于刻板印象推断用户属性，甚至用户明确身份后仍存在，并提出线性探测干预方法缓解问题，强调需提高模型透明度。**

- **链接: [http://arxiv.org/pdf/2505.16467v1](http://arxiv.org/pdf/2505.16467v1)**

> **作者:** Vera Neplenbroek; Arianna Bisazza; Raquel Fernández
>
> **摘要:** Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
>
---
#### [new 051] Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6x6 Sudoku
- **分类: cs.CL**

- **简介: 该论文评估了5个大语言模型在解决并解释6x6数独任务中的表现，探究其能否通过战略推理生成逐步解释。研究发现模型解题能力有限，且均无法提供体现人类式逻辑推理的解释，揭示LLMs在人机协作决策中需提升解释能力。**

- **链接: [http://arxiv.org/pdf/2505.15993v1](http://arxiv.org/pdf/2505.15993v1)**

> **作者:** Anirudh Maiya; Razan Alghamdi; Maria Leonor Pacheco; Ashutosh Trivedi; Fabio Somenzi
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** The success of Large Language Models (LLMs) in human-AI collaborative decision-making hinges on their ability to provide trustworthy, gradual, and tailored explanations. Solving complex puzzles, such as Sudoku, offers a canonical example of this collaboration, where clear and customized explanations often hold greater importance than the final solution. In this study, we evaluate the performance of five LLMs in solving and explaining \sixsix{} Sudoku puzzles. While one LLM demonstrates limited success in solving puzzles, none can explain the solution process in a manner that reflects strategic reasoning or intuitive problem-solving. These findings underscore significant challenges that must be addressed before LLMs can become effective partners in human-AI collaborative decision-making.
>
---
#### [new 052] Embodied Agents Meet Personalization: Exploring Memory Utilization for Personalized Assistance
- **分类: cs.CL**

- **简介: 该论文提出MEMENTO框架，评估具身智能体利用用户记忆（如物品语义、行为模式）进行个性化任务（如物体排列）的能力，解决现有模型在多记忆引用时表现差的问题，实验显示GPT-4o等模型性能下降30.5%，为研发更有效的个性化助手提供方向。**

- **链接: [http://arxiv.org/pdf/2505.16348v1](http://arxiv.org/pdf/2505.16348v1)**

> **作者:** Taeyoon Kwon; Dongwook Choi; Sunghwan Kim; Hyojun Kim; Seungjun Moon; Beong-woo Kwak; Kuan-Hao Huang; Jinyoung Yeo
>
> **备注:** Work in progress
>
> **摘要:** Embodied agents empowered by large language models (LLMs) have shown strong performance in household object rearrangement tasks. However, these tasks primarily focus on single-turn interactions with simplified instructions, which do not truly reflect the challenges of providing meaningful assistance to users. To provide personalized assistance, embodied agents must understand the unique semantics that users assign to the physical world (e.g., favorite cup, breakfast routine) by leveraging prior interaction history to interpret dynamic, real-world instructions. Yet, the effectiveness of embodied agents in utilizing memory for personalized assistance remains largely underexplored. To address this gap, we present MEMENTO, a personalized embodied agent evaluation framework designed to comprehensively assess memory utilization capabilities to provide personalized assistance. Our framework consists of a two-stage memory evaluation process design that enables quantifying the impact of memory utilization on task performance. This process enables the evaluation of agents' understanding of personalized knowledge in object rearrangement tasks by focusing on its role in goal interpretation: (1) the ability to identify target objects based on personal meaning (object semantics), and (2) the ability to infer object-location configurations from consistent user patterns, such as routines (user patterns). Our experiments across various LLMs reveal significant limitations in memory utilization, with even frontier models like GPT-4o experiencing a 30.5% performance drop when required to reference multiple memories, particularly in tasks involving user patterns. These findings, along with our detailed analyses and case studies, provide valuable insights for future research in developing more effective personalized embodied agents. Project website: https://connoriginal.github.io/MEMENTO
>
---
#### [new 053] LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding
- **分类: cs.CL**

- **简介: 该论文属于LLM流处理适配任务，旨在解决批处理LLM迁移到流处理时的性能下降问题。研究发现输入-注意力是主要矛盾，提出组位置编码方法，通过保持相对位置信息而非绝对顺序，弥合流-批模式差异，无需架构修改即提升跨模态/跨语言任务效果。**

- **链接: [http://arxiv.org/pdf/2505.16983v1](http://arxiv.org/pdf/2505.16983v1)**

> **作者:** Junlong Tong; Jinlan Fu; Zixuan Lin; Yingqi Fan; Anhao Zhao; Hui Su; Xiaoyu Shen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
>
---
#### [new 054] $I^2G$: Generating Instructional Illustrations via Text-Conditioned Diffusion
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出I²G框架，通过文本引导的扩散模型将程序性文本转化为连贯视觉指令，解决纯文本难以表达复杂动作和空间关系的问题。创新包括句法解析编码保持语义完整性、话语连贯模型确保步骤一致性及专用评估协议，在三个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16425v1](http://arxiv.org/pdf/2505.16425v1)**

> **作者:** Jing Bi; Pinxin Liu; Ali Vosoughi; Jiarui Wu; Jinxi He; Chenliang Xu
>
> **备注:** 13 pages, 5 figures, under review
>
> **摘要:** The effective communication of procedural knowledge remains a significant challenge in natural language processing (NLP), as purely textual instructions often fail to convey complex physical actions and spatial relationships. We address this limitation by proposing a language-driven framework that translates procedural text into coherent visual instructions. Our approach models the linguistic structure of instructional content by decomposing it into goal statements and sequential steps, then conditioning visual generation on these linguistic elements. We introduce three key innovations: (1) a constituency parser-based text encoding mechanism that preserves semantic completeness even with lengthy instructions, (2) a pairwise discourse coherence model that maintains consistency across instruction sequences, and (3) a novel evaluation protocol specifically designed for procedural language-to-image alignment. Our experiments across three instructional datasets (HTStep, CaptainCook4D, and WikiAll) demonstrate that our method significantly outperforms existing baselines in generating visuals that accurately reflect the linguistic content and sequential nature of instructions. This work contributes to the growing body of research on grounding procedural language in visual content, with applications spanning education, task guidance, and multimodal language understanding.
>
---
#### [new 055] University of Indonesia at SemEval-2025 Task 11: Evaluating State-of-the-Art Encoders for Multi-Label Emotion Detection
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文参与SemEval-2025多语言多标签情绪分类任务，研究如何提升跨语言情绪检测效果。通过对比完全微调与仅训练分类器策略，发现基于mE5、BGE等提示编码器的分类器优于XLMR/mBERT微调。最佳模型采用BGE集成+CatBoost分类器，获56.58平均F1-macro分。**

- **链接: [http://arxiv.org/pdf/2505.16460v1](http://arxiv.org/pdf/2505.16460v1)**

> **作者:** Ikhlasul Akmal Hanif; Eryawan Presma Yulianrifat; Jaycent Gunawan Ongris; Eduardus Tjitrahardja; Muhammad Falensi Azmi; Rahmat Bryan Naufal; Alfan Farizki Wicaksono
>
> **备注:** 16 pages, 13 tables, 1 figures
>
> **摘要:** This paper presents our approach for SemEval 2025 Task 11 Track A, focusing on multilabel emotion classification across 28 languages. We explore two main strategies: fully fine-tuning transformer models and classifier-only training, evaluating different settings such as fine-tuning strategies, model architectures, loss functions, encoders, and classifiers. Our findings suggest that training a classifier on top of prompt-based encoders such as mE5 and BGE yields significantly better results than fully fine-tuning XLMR and mBERT. Our best-performing model on the final leaderboard is an ensemble combining multiple BGE models, where CatBoost serves as the classifier, with different configurations. This ensemble achieves an average F1-macro score of 56.58 across all languages.
>
---
#### [new 056] ToDi: Token-wise Distillation via Fine-Grained Divergence Control
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决传统方法（如FKL/RKL）因全局统一的散度损失导致的token级预测差异忽视问题。提出ToDi方法，通过教师-学生概率对数比的sigmoid权重函数，为每个token自适应融合FKL与RKL，实现细粒度分布对齐，提升蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2505.16297v1](http://arxiv.org/pdf/2505.16297v1)**

> **作者:** Seongryong Jung; Suwan Yoon; DongGeon Kim; Hwanhee Lee
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Large language models (LLMs) offer impressive performance but are impractical for resource-constrained deployment due to high latency and energy consumption. Knowledge distillation (KD) addresses this by transferring knowledge from a large teacher to a smaller student model. However, conventional KD, notably approaches like Forward KL (FKL) and Reverse KL (RKL), apply uniform divergence loss across the entire vocabulary, neglecting token-level prediction discrepancies. By investigating these representative divergences via gradient analysis, we reveal that FKL boosts underestimated tokens, while RKL suppresses overestimated ones, showing their complementary roles. Based on this observation, we propose Token-wise Distillation (ToDi), a novel method that adaptively combines FKL and RKL per token using a sigmoid-based weighting function derived from the teacher-student probability log-ratio. ToDi dynamically emphasizes the appropriate divergence for each token, enabling precise distribution alignment. We demonstrate that ToDi consistently outperforms recent distillation baselines using uniform or less granular strategies across instruction-following benchmarks. Extensive ablation studies and efficiency analysis further validate ToDi's effectiveness and practicality.
>
---
#### [new 057] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，解决现有系统未模拟人类验证原子声明行为的问题。提出EMULATE多代理框架，通过分工协作（如排序搜索结果、评估网页内容）优化证据检索与分析，实验显示优于先前方法。**

- **链接: [http://arxiv.org/pdf/2505.16576v1](http://arxiv.org/pdf/2505.16576v1)**

> **作者:** Spencer Hong; Meng Luo; Xinyi Wan
>
> **摘要:** Determining the veracity of atomic claims is an imperative component of many recently proposed fact-checking systems. Many approaches tackle this problem by first retrieving evidence by querying a search engine and then performing classification by providing the evidence set and atomic claim to a large language model, but this process deviates from what a human would do in order to perform the task. Recent work attempted to address this issue by proposing iterative evidence retrieval, allowing for evidence to be collected several times and only when necessary. Continuing along this line of research, we propose a novel claim verification system, called EMULATE, which is designed to better emulate human actions through the use of a multi-agent framework where each agent performs a small part of the larger task, such as ranking search results according to predefined criteria or evaluating webpage content. Extensive experiments on several benchmarks show clear improvements over prior work, demonstrating the efficacy of our new multi-agent framework.
>
---
#### [new 058] Spontaneous Speech Variables for Evaluating LLMs Cognitive Plausibility
- **分类: cs.CL**

- **简介: 该论文提出用自发语音中的生产变量（如语音缩减、重音）评估LLMs认知可信度，解决现有评估方法缺乏语言生成层面认知指标的问题。工作包括提取语音变量并测试不同预训练模型（书面/口语/混合数据）的预测能力，发现口语数据训练的模型预测更优，为语音语料库作为LLM基准提供依据。**

- **链接: [http://arxiv.org/pdf/2505.16277v1](http://arxiv.org/pdf/2505.16277v1)**

> **作者:** Sheng-Fu Wang; Laurent Prevot; Jou-an Chi; Ri-Sheng Huang; Shu-Kai Hsieh
>
> **备注:** The 14th Workshop on Cognitive Modeling and Computational Linguistics (CMCL). May 3, 2025. Collocated with NAACL 2025
>
> **摘要:** The achievements of Large Language Models in Natural Language Processing, especially for high-resource languages, call for a better understanding of their characteristics from a cognitive perspective. Researchers have attempted to evaluate artificial models by testing their ability to predict behavioral (e.g., eye-tracking fixations) and physiological (e.g., brain responses) variables during language processing (e.g., reading/listening). In this paper, we propose using spontaneous speech corpora to derive production variables (speech reductions, prosodic prominences) and applying them in a similar fashion. More precisely, we extract. We then test models trained with a standard procedure on different pretraining datasets (written, spoken, and mixed genres) for their ability to predict these two variables. Our results show that, after some fine-tuning, the models can predict these production variables well above baselines. We also observe that spoken genre training data provides more accurate predictions than written genres. These results contribute to the broader effort of using high-quality speech corpora as benchmarks for LLMs.
>
---
#### [new 059] Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型微调引发的意外漏洞问题。任务为分析微调数据特性（如语言特征、毒性等）如何导致模型对抗脆弱性。通过评估微调模型的对抗攻击成功率，探究数据因素与漏洞的关联及因果关系，提出防御策略并强调数据设计对模型稳健性的重要作用。**

- **链接: [http://arxiv.org/pdf/2505.16789v1](http://arxiv.org/pdf/2505.16789v1)**

> **作者:** Punya Syon Pandey; Samuel Simko; Kellin Pelrine; Zhijing Jin
>
> **摘要:** As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.
>
---
#### [new 060] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM微调优化任务，解决如何有效利用模型自身错误改进性能。提出"错误日志"追踪错误，设计Copilot模型通过修正Pilot模型的logits提升推理，并采用联合训练与融合推理框架，实验显示性能提升达34.5%，计算开销低。**

- **链接: [http://arxiv.org/pdf/2505.16270v1](http://arxiv.org/pdf/2505.16270v1)**

> **作者:** Jiaru Zou; Yikun Ban; Zihao Li; Yunzhe Qi; Ruizhong Qiu; Ling Yang; Jingrui He
>
> **备注:** 33 pages, 7 figures
>
> **摘要:** Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability.
>
---
#### [new 061] Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains
- **分类: cs.CL**

- **简介: 该论文提出METEORA方法，改进敏感领域RAG模型。针对传统RAG依赖top-k排序、可解释性差及抗攻击弱的问题，用基于理由的分阶段证据选择替代重排序，通过LLM生成理由指导局部匹配、全局截断及上下文扩展，并用Verifier过滤恶意内容。实验显示其提升33.34%准确率，减少50%计算量，并显著增强抗毒化攻击能力。**

- **链接: [http://arxiv.org/pdf/2505.16014v1](http://arxiv.org/pdf/2505.16014v1)**

> **作者:** Yash Saxena; Anpur Padia; Mandar S Chaudhary; Kalpa Gunaratna; Srinivasan Parthasarathy; Manas Gaur
>
> **摘要:** Traditional Retrieval-Augmented Generation (RAG) pipelines rely on similarity-based retrieval and re-ranking, which depend on heuristics such as top-k, and lack explainability, interpretability, and robustness against adversarial content. To address this gap, we propose a novel method METEORA that replaces re-ranking in RAG with a rationale-driven selection approach. METEORA operates in two stages. First, a general-purpose LLM is preference-tuned to generate rationales conditioned on the input query using direct preference optimization. These rationales guide the evidence chunk selection engine, which selects relevant chunks in three stages: pairing individual rationales with corresponding retrieved chunks for local relevance, global selection with elbow detection for adaptive cutoff, and context expansion via neighboring chunks. This process eliminates the need for top-k heuristics. The rationales are also used for consistency check using a Verifier LLM to detect and filter poisoned or misleading content for safe generation. The framework provides explainable and interpretable evidence flow by using rationales consistently across both selection and verification. Our evaluation across six datasets spanning legal, financial, and academic research domains shows that METEORA improves generation accuracy by 33.34% while using approximately 50% fewer chunks than state-of-the-art re-ranking methods. In adversarial settings, METEORA significantly improves the F1 score from 0.10 to 0.44 over the state-of-the-art perplexity-based defense baseline, demonstrating strong resilience to poisoning attacks. Code available at: https://anonymous.4open.science/r/METEORA-DC46/README.md
>
---
#### [new 062] LIFEBench: Evaluating Length Instruction Following in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LIFEBench基准，评估大语言模型遵循长度指令的能力。针对模型难以生成符合指定长度文本（如万字小说）的问题，构建涵盖中英双语、4类任务及16-32K词的评测集，测试26个模型发现：多数模型在长文本生成时表现骤降，甚至无法达到厂商声称的最大长度，推理模型表现最佳。揭示了LLMs在长度控制上的根本局限性。**

- **链接: [http://arxiv.org/pdf/2505.16234v1](http://arxiv.org/pdf/2505.16234v1)**

> **作者:** Wei Zhang; Zhenhong Zhou; Junfeng Fang; Rongwu Xu; Kun Wang; Yuanhe Zhang; Rui Wang; Ge Zhang; Xinfeng Li; Li Sun; Lingjuan Lyu; Yang Liu; Sen Su
>
> **备注:** 81 pages, 22 tables, 32 figures. Homepage: https://ydyjya.github.io/LIFEBench/
>
> **摘要:** While large language models (LLMs) can solve PhD-level reasoning problems over long context inputs, they still struggle with a seemingly simpler task: following explicit length instructions-e.g., write a 10,000-word novel. Additionally, models often generate far too short outputs, terminate prematurely, or even refuse the request. Existing benchmarks focus primarily on evaluating generations quality, but often overlook whether the generations meet length constraints. To this end, we introduce Length Instruction Following Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to follow length instructions across diverse tasks and a wide range of specified lengths. LIFEBench consists of 10,800 instances across 4 task categories in both English and Chinese, covering length constraints ranging from 16 to 8192 words. We evaluate 26 widely-used LLMs and find that most models reasonably follow short-length instructions but deteriorate sharply beyond a certain threshold. Surprisingly, almost all models fail to reach the vendor-claimed maximum output lengths in practice, as further confirmed by our evaluations extending up to 32K words. Even long-context LLMs, despite their extended input-output windows, counterintuitively fail to improve length-instructions following. Notably, Reasoning LLMs outperform even specialized long-text generation models, achieving state-of-the-art length following. Overall, LIFEBench uncovers fundamental limitations in current LLMs' length instructions following ability, offering critical insights for future progress.
>
---
#### [new 063] Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs
- **分类: cs.CL**

- **简介: 该论文针对多模态LLM训练中视觉指令调优导致的语言能力退化问题，提出Locate-then-Merge框架，通过神经元级参数融合策略（Neuron-Fusion），筛选参数变化大的视觉相关神经元保留，减弱小变化的语言神经元影响，实现视觉适配与语言能力的平衡，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16703v1](http://arxiv.org/pdf/2505.16703v1)**

> **作者:** Zeping Yu; Sophia Ananiadou
>
> **摘要:** Although multimodal large language models (MLLMs) have achieved impressive performance, the multimodal instruction tuning stage often causes catastrophic forgetting of the base LLM's language ability, even in strong models like Llama3. To address this, we propose Locate-then-Merge, a training-free parameter fusion framework that first locates important parameters and then selectively merges them. We further introduce Neuron-Fusion, a neuron-level strategy that preserves the influence of neurons with large parameter shifts--neurons likely responsible for newly acquired visual capabilities--while attenuating the influence of neurons with smaller changes that likely encode general-purpose language skills. This design enables better retention of visual adaptation while mitigating language degradation. Experiments on 13 benchmarks across both language and visual tasks show that Neuron-Fusion consistently outperforms existing model merging methods. Further analysis reveals that our method effectively reduces context hallucination in generation.
>
---
#### [new 064] Semiotic Reconstruction of Destination Expectation Constructs An LLM-Driven Computational Paradigm for Social Media Tourism Analytics
- **分类: cs.CL; stat.AP**

- **简介: 该论文提出基于LLM的双方法框架，解决社交媒体旅游分析中UGC处理的可扩展性问题。通过无监督提取旅游期望并结合调查数据监督微调，量化发现休闲社交因素比自然情感因素更驱动用户参与，推动旅游策略优化与计算社会科学应用。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16118v1](http://arxiv.org/pdf/2505.16118v1)**

> **作者:** Haotian Lan; Yao Gao; Yujun Cheng; Wei Yuan; Kun Wang
>
> **备注:** 33 pages, 6 figures
>
> **摘要:** Social media's rise establishes user-generated content (UGC) as pivotal for travel decisions, yet analytical methods lack scalability. This study introduces a dual-method LLM framework: unsupervised expectation extraction from UGC paired with survey-informed supervised fine-tuning. Findings reveal leisure/social expectations drive engagement more than foundational natural/emotional factors. By establishing LLMs as precision tools for expectation quantification, we advance tourism analytics methodology and propose targeted strategies for experience personalization and social travel promotion. The framework's adaptability extends to consumer behavior research, demonstrating computational social science's transformative potential in marketing optimization.
>
---
#### [new 065] Exploring the Relationship Between Diversity and Quality in Ad Text Generation
- **分类: cs.CL**

- **简介: 该论文属于广告文本生成任务，研究多样性增强方法对广告质量的影响。针对现有方法未深入探索广告场景的问题，分析多样性方法、超参数、输入输出格式及模型对生成文本多样性和质量的关联，以优化广告文本生成效果。**

- **链接: [http://arxiv.org/pdf/2505.16418v1](http://arxiv.org/pdf/2505.16418v1)**

> **作者:** Yoichi Aoki; Soichiro Murakami; Ukyo Honda; Akihiko Kato
>
> **摘要:** In natural language generation for advertising, creating diverse and engaging ad texts is crucial for capturing a broad audience and avoiding advertising fatigue. Regardless of the importance of diversity, the impact of the diversity-enhancing methods in ad text generation -- mainly tested on tasks such as summarization and machine translation -- has not been thoroughly explored. Ad text generation significantly differs from these tasks owing to the text style and requirements. This research explores the relationship between diversity and ad quality in ad text generation by considering multiple factors, such as diversity-enhancing methods, their hyperparameters, input-output formats, and the models.
>
---
#### [new 066] Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）幻觉机制，探究上下文注入引发的内部状态漂移如何导致幻觉。通过设计两种滴定实验（添加部分错误或误导内容），分析六种LLM的幻觉率及隐藏状态、注意力变化。发现幻觉随轮次增长并稳定，相关上下文导致高自信幻觉，无关内容引发主题偏离，且存在attention锁定阈值，为幻觉预测与缓解提供理论依据。**

- **链接: [http://arxiv.org/pdf/2505.16894v1](http://arxiv.org/pdf/2505.16894v1)**

> **作者:** Zeyu Wei; Shuo Wang; Xiaohui Rong; Xuemin Liu; He Li
>
> **摘要:** Hallucinations -- plausible yet erroneous outputs -- remain a critical barrier to reliable deployment of large language models (LLMs). We present the first systematic study linking hallucination incidence to internal-state drift induced by incremental context injection. Using TruthfulQA, we construct two 16-round "titration" tracks per question: one appends relevant but partially flawed snippets, the other injects deliberately misleading content. Across six open-source LLMs, we track overt hallucination rates with a tri-perspective detector and covert dynamics via cosine, entropy, JS and Spearman drifts of hidden states and attention maps. Results reveal (1) monotonic growth of hallucination frequency and representation drift that plateaus after 5--7 rounds; (2) relevant context drives deeper semantic assimilation, producing high-confidence "self-consistent" hallucinations, whereas irrelevant context induces topic-drift errors anchored by attention re-routing; and (3) convergence of JS-Drift ($\sim0.69$) and Spearman-Drift ($\sim0$) marks an "attention-locking" threshold beyond which hallucinations solidify and become resistant to correction. Correlation analyses expose a seesaw between assimilation capacity and attention diffusion, clarifying size-dependent error modes. These findings supply empirical foundations for intrinsic hallucination prediction and context-aware mitigation mechanisms.
>
---
#### [new 067] Does Synthetic Data Help Named Entity Recognition for Low-Resource Languages?
- **分类: cs.CL**

- **简介: 该论文研究低资源语言的命名实体识别（NER）任务，探索合成数据对提升该任务效果的作用。针对11种不同语系的低资源语言，实验表明合成数据能有效增强模型性能，但不同语言间效果差异显著。**

- **链接: [http://arxiv.org/pdf/2505.16814v1](http://arxiv.org/pdf/2505.16814v1)**

> **作者:** Gaurav Kamath; Sowmya Vajjala
>
> **备注:** pre-print
>
> **摘要:** Named Entity Recognition(NER) for low-resource languages aims to produce robust systems for languages where there is limited labeled training data available, and has been an area of increasing interest within NLP. Data augmentation for increasing the amount of low-resource labeled data is a common practice. In this paper, we explore the role of synthetic data in the context of multilingual, low-resource NER, considering 11 languages from diverse language families. Our results suggest that synthetic data does in fact hold promise for low-resource language NER, though we see significant variation between languages.
>
---
#### [new 068] From Generic Empathy to Personalized Emotional Support: A Self-Evolution Framework for User Preference Alignment
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在情感支持中回应通用化的问题，提出自进化框架，通过微调基础模型及迭代自我优化，对齐用户隐含偏好（如性格、情绪、情境），提升个性化支持效果，减少无用回应。**

- **链接: [http://arxiv.org/pdf/2505.16610v1](http://arxiv.org/pdf/2505.16610v1)**

> **作者:** Jing Ye; Lu Xiang; Yaping Zhang; Chengqing Zong
>
> **备注:** 27 pages
>
> **摘要:** Effective emotional support hinges on understanding users' emotions and needs to provide meaningful comfort during multi-turn interactions. Large Language Models (LLMs) show great potential for expressing empathy; however, they often deliver generic and one-size-fits-all responses that fail to address users' specific needs. To tackle this issue, we propose a self-evolution framework designed to help LLMs improve their responses to better align with users' implicit preferences concerning user profiles (personalities), emotional states, and specific situations. Our framework consists of two distinct phases: \textit{(1)} \textit{Emotional Support Experience Acquisition}, where LLMs are fine-tuned on limited emotional support conversation data to provide basic support, and \textit{(2)} \textit{Self-Improvement for Personalized Emotional Support}, where LLMs leverage self-reflection and self-refinement to generate personalized responses. Through iterative direct preference optimization between the pre- and post-refined responses, our model generates responses that reflect a better understanding of the user's implicit preferences. Extensive experiments and evaluations demonstrate that our method significantly enhances the model's performance in emotional support, reducing unhelpful responses and minimizing discrepancies between user preferences and model outputs.
>
---
#### [new 069] WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于Web代理训练任务，旨在解决多轮交互中复杂动态环境下的长周期决策难题。提出端到端多轮RL框架WebAgent-R1，通过异步生成轨迹和二元奖励在线学习，显著提升Qwen和Llama等模型在WebArena-Lite的 success rate，并分析了思考提示策略及两种变体对训练效果的影响。**

- **链接: [http://arxiv.org/pdf/2505.16421v1](http://arxiv.org/pdf/2505.16421v1)**

> **作者:** Zhepei Wei; Wenlin Yao; Yao Liu; Weizhi Zhang; Qin Lu; Liang Qiu; Changlong Yu; Puyang Xu; Chao Zhang; Bing Yin; Hyokun Yun; Lihong Li
>
> **备注:** Preprint
>
> **摘要:** While reinforcement learning (RL) has demonstrated remarkable success in enhancing large language models (LLMs), it has primarily focused on single-turn tasks such as solving math problems. Training effective web agents for multi-turn interactions remains challenging due to the complexity of long-horizon decision-making across dynamic web interfaces. In this work, we present WebAgent-R1, a simple yet effective end-to-end multi-turn RL framework for training web agents. It learns directly from online interactions with web environments by asynchronously generating diverse trajectories, entirely guided by binary rewards depending on task success. Experiments on the WebArena-Lite benchmark demonstrate the effectiveness of WebAgent-R1, boosting the task success rate of Qwen-2.5-3B from 6.1% to 33.9% and Llama-3.1-8B from 8.5% to 44.8%, significantly outperforming existing state-of-the-art methods and strong proprietary models such as OpenAI o3. In-depth analyses reveal the effectiveness of the thinking-based prompting strategy and test-time scaling through increased interactions for web tasks. We further investigate different RL initialization policies by introducing two variants, namely WebAgent-R1-Zero and WebAgent-R1-CoT, which highlight the importance of the warm-up training stage (i.e., behavior cloning) and provide insights on incorporating long chain-of-thought (CoT) reasoning in web agents.
>
---
#### [new 070] On the reliability of feature attribution methods for speech classification
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音分类任务，研究特征归因方法在语音领域的可靠性问题。通过分析输入类型、扰动时长等影响因素及其与任务特性的交互，发现标准方法普遍不可靠，仅在单词级分类中词对齐扰动法有效。**

- **链接: [http://arxiv.org/pdf/2505.16406v1](http://arxiv.org/pdf/2505.16406v1)**

> **作者:** Gaofei Shen; Hosein Mohebbi; Arianna Bisazza; Afra Alishahi; Grzegorz Chrupała
>
> **摘要:** As the capabilities of large-scale pre-trained models evolve, understanding the determinants of their outputs becomes more important. Feature attribution aims to reveal which parts of the input elements contribute the most to model outputs. In speech processing, the unique characteristics of the input signal make the application of feature attribution methods challenging. We study how factors such as input type and aggregation and perturbation timespan impact the reliability of standard feature attribution methods, and how these factors interact with characteristics of each classification task. We find that standard approaches to feature attribution are generally unreliable when applied to the speech domain, with the exception of word-aligned perturbation methods when applied to word-based classification tasks.
>
---
#### [new 071] Beyond Static Testbeds: An Interaction-Centric Agent Simulation Platform for Dynamic Recommender Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出动态推荐系统模拟平台RecInter，解决传统测试方法资源消耗大且无法模拟用户与环境动态交互的问题。通过实时更新项目属性、引入商家代理及高保真模块，实现推荐系统生态真实演化模拟，有效复现品牌忠诚等现象。**

- **链接: [http://arxiv.org/pdf/2505.16429v1](http://arxiv.org/pdf/2505.16429v1)**

> **作者:** Song Jin; Juntian Zhang; Yuhan Liu; Xun Zhang; Yufei Zhang; Guojun Yin; Fei Jiang; Wei Lin; Rui Yan
>
> **摘要:** Evaluating and iterating upon recommender systems is crucial, yet traditional A/B testing is resource-intensive, and offline methods struggle with dynamic user-platform interactions. While agent-based simulation is promising, existing platforms often lack a mechanism for user actions to dynamically reshape the environment. To bridge this gap, we introduce RecInter, a novel agent-based simulation platform for recommender systems featuring a robust interaction mechanism. In RecInter platform, simulated user actions (e.g., likes, reviews, purchases) dynamically update item attributes in real-time, and introduced Merchant Agents can reply, fostering a more realistic and evolving ecosystem. High-fidelity simulation is ensured through Multidimensional User Profiling module, Advanced Agent Architecture, and LLM fine-tuned on Chain-of-Thought (CoT) enriched interaction data. Our platform achieves significantly improved simulation credibility and successfully replicates emergent phenomena like Brand Loyalty and the Matthew Effect. Experiments demonstrate that this interaction mechanism is pivotal for simulating realistic system evolution, establishing our platform as a credible testbed for recommender systems research.
>
---
#### [new 072] Position of Uncertainty: A Cross-Linguistic Study of Positional Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型（LLM）的跨语言分析任务，研究位置偏差及其与语言特性、模型性能的关联。通过对比五种语言（含自由语序语言），揭示模型间位置偏好差异（如Qwen偏后期token）、明确位置提示降低准确率、位置偏差与熵的关系，以及LLM对词序的强制性影响，挑战传统早期偏好的假设，解决LLM跨语言处理机制问题。**

- **链接: [http://arxiv.org/pdf/2505.16134v1](http://arxiv.org/pdf/2505.16134v1)**

> **作者:** Menschikov Mikhail; Alexander Kharitonov; Maiia Kotyga; Vadim Porvatov; Anna Zhukovskaya; David Kagramanyan; Egor Shvetsov; Evgeny Burnaev
>
> **摘要:** Large language models exhibit positional bias -- systematic neglect of information at specific context positions -- yet its interplay with linguistic diversity remains poorly understood. We present a cross-linguistic study across five typologically distinct languages (English, Russian, German, Hindi, Vietnamese), examining how positional bias interacts with model uncertainty, syntax, and prompting. Key findings: (1) Positional bias is model-driven, with language-specific variations -- Qwen2.5-7B favors late positions, challenging assumptions of early-token bias; (2) Explicit positional guidance (e.g., correct context is at position X) reduces accuracy across languages, undermining prompt-engineering practices; (3) Aligning context with positional bias increases entropy, yet minimal entropy does not predict accuracy. (4) We further uncover that LLMs differently impose dominant word order in free-word-order languages like Hindi.
>
---
#### [new 073] Nested Named Entity Recognition as Single-Pass Sequence Labeling
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 论文属于嵌套命名实体识别（NNER）任务，解决其结构复杂性导致的效率问题。通过将构成树结构线性化结合预训练模型，转化为单次序列标注任务，仅需n次标记即可捕捉嵌套实体，性能与高效系统相当且兼容现成工具。**

- **链接: [http://arxiv.org/pdf/2505.16855v1](http://arxiv.org/pdf/2505.16855v1)**

> **作者:** Alberto Muñoz-Ortiz; David Vilares; Caio COrro; Carlos Gómez-Rodríguez
>
> **备注:** Submitted to EMNLP 2025
>
> **摘要:** We cast nested named entity recognition (NNER) as a sequence labeling task by leveraging prior work that linearizes constituency structures, effectively reducing the complexity of this structured prediction problem to straightforward token classification. By combining these constituency linearizations with pretrained encoders, our method captures nested entities while performing exactly $n$ tagging actions. Our approach achieves competitive performance compared to less efficient systems, and it can be trained using any off-the-shelf sequence labeling library.
>
---
#### [new 074] Understanding and Analyzing Inappropriately Targeting Language in Online Discourse: A Comparative Annotation Study
- **分类: cs.CL**

- **简介: 该论文属于在线 discourse 分析任务，旨在解决自动检测隐式与显式歧视性语言的挑战。通过整合人类专家、众包标注者与 ChatGPT 的标注对比，构建多维度标注框架，识别针对个体/群体的不当语言（如社会信念、体型形象等新类别），揭示上下文对判断的关键作用及 ChatGPT 的局限性，为改进内容审核策略提供依据。**

- **链接: [http://arxiv.org/pdf/2505.16847v1](http://arxiv.org/pdf/2505.16847v1)**

> **作者:** Baran Barbarestani; Isa Maks; Piek Vossen
>
> **摘要:** This paper introduces a method for detecting inappropriately targeting language in online conversations by integrating crowd and expert annotations with ChatGPT. We focus on English conversation threads from Reddit, examining comments that target individuals or groups. Our approach involves a comprehensive annotation framework that labels a diverse data set for various target categories and specific target words within the conversational context. We perform a comparative analysis of annotations from human experts, crowd annotators, and ChatGPT, revealing strengths and limitations of each method in recognizing both explicit hate speech and subtler discriminatory language. Our findings highlight the significant role of contextual factors in identifying hate speech and uncover new categories of targeting, such as social belief and body image. We also address the challenges and subjective judgments involved in annotation and the limitations of ChatGPT in grasping nuanced language. This study provides insights for improving automated content moderation strategies to enhance online safety and inclusivity.
>
---
#### [new 075] MPO: Multilingual Safety Alignment via Reward Gap Optimization
- **分类: cs.CL**

- **简介: 该论文提出多语言安全对齐方法MPO，解决现有单语种安全对齐方法（如RLHF/DPO）在跨语言噪声数据中效果差的问题。通过最小化主语言（英语）与目标语言的奖励差距，将安全能力跨语言迁移，实验验证其有效性且不损害多语言通用性。**

- **链接: [http://arxiv.org/pdf/2505.16869v1](http://arxiv.org/pdf/2505.16869v1)**

> **作者:** Weixiang Zhao; Yulin Hu; Yang Deng; Tongtong Wu; Wenxuan Zhang; Jiahe Guo; An Zhang; Yanyan Zhao; Bing Qin; Tat-Seng Chua; Ting Liu
>
> **备注:** To Appear at ACL 2025 (Main)
>
> **摘要:** Large language models (LLMs) have become increasingly central to AI applications worldwide, necessitating robust multilingual safety alignment to ensure secure deployment across diverse linguistic contexts. Existing preference learning methods for safety alignment, such as RLHF and DPO, are primarily monolingual and struggle with noisy multilingual data. To address these limitations, we introduce Multilingual reward gaP Optimization (MPO), a novel approach that leverages the well-aligned safety capabilities of the dominant language (English) to improve safety alignment across multiple languages. MPO directly minimizes the reward gap difference between the dominant language and target languages, effectively transferring safety capabilities while preserving the original strengths of the dominant language. Extensive experiments on three LLMs, LLaMA-3.1, Gemma-2 and Qwen2.5, validate MPO's efficacy in multilingual safety alignment without degrading general multilingual utility.
>
---
#### [new 076] UNCLE: Uncertainty Expressions in Long-Form Generation
- **分类: cs.CL**

- **简介: 该论文属于改进大语言模型（LLMs）不确定性表达的任务，旨在解决其长文本生成中因知识不足导致的幻觉问题。团队构建了跨领域基准UNCLE（含长/短问答数据），提出新评估指标，实验表明现有模型在长文本中表现不佳，训练方法比提示方法更有效，为未来研究指明方向。**

- **链接: [http://arxiv.org/pdf/2505.16922v1](http://arxiv.org/pdf/2505.16922v1)**

> **作者:** Ruihan Yang; Caiqi Zhang; Zhisong Zhang; Xinting Huang; Dong Yu; Nigel Collier; Deqing Yang
>
> **摘要:** Large Language Models (LLMs) are prone to hallucination, particularly in long-form generations. A promising direction to mitigate hallucination is to teach LLMs to express uncertainty explicitly when they lack sufficient knowledge. However, existing work lacks direct and fair evaluation of LLMs' ability to express uncertainty effectively in long-form generation. To address this gap, we first introduce UNCLE, a benchmark designed to evaluate uncertainty expression in both long- and short-form question answering (QA). UNCLE spans five domains and comprises 4k long-form QA instances and over 20k short-form QA pairs. Our dataset is the first to directly bridge short- and long-form QA with paired questions and gold-standard answers. Along with the benchmark, we propose a suite of new metrics to assess the models' capabilities to selectively express uncertainty. Using UNCLE, we then demonstrate that current models fail to convey uncertainty appropriately in long-form generation. We further explore both prompt-based and training-based methods to improve models' performance, with the training-based methods yielding greater gains. Further analysis of alignment gaps between short- and long-form uncertainty expression highlights promising directions for future research using UNCLE.
>
---
#### [new 077] Evaluating Large Language Model with Knowledge Oriented Language Specific Simple Question Answering
- **分类: cs.CL**

- **简介: 该论文提出KoLasSimpleQA基准，评估LLMs的多语言事实问答能力。针对现有评测在跨语言事实准确性和模型自知能力上的不足，构建覆盖9种语言、包含通用知识和语言特有领域的试题，测试主流模型发现领域表现差异，为多语言模型优化提供指导。**

- **链接: [http://arxiv.org/pdf/2505.16591v1](http://arxiv.org/pdf/2505.16591v1)**

> **作者:** Bowen Jiang; Runchuan Zhu; Jiang Wu; Zinco Jiang; Yifan He; Junyuan Gao; Jia Yu; Rui Min; Yinfan Wang; Haote Yang; Songyang Zhang; Dahua Lin; Lijun Wu; Conghui He
>
> **备注:** Equal contribution: Bowen Jiang, Runchuan Zhu, Jiang Wu; Corresponding author: Conghui He
>
> **摘要:** We introduce KoLasSimpleQA, the first benchmark evaluating the multilingual factual ability of Large Language Models (LLMs). Inspired by existing research, we created the question set with features such as single knowledge point coverage, absolute objectivity, unique answers, and temporal stability. These questions enable efficient evaluation using the LLM-as-judge paradigm, testing both the LLMs' factual memory and self-awareness ("know what they don't know"). KoLasSimpleQA expands existing research in two key dimensions: (1) Breadth (Multilingual Coverage): It includes 9 languages, supporting global applicability evaluation. (2) Depth (Dual Domain Design): It covers both the general domain (global facts) and the language-specific domain (such as history, culture, and regional traditions) for a comprehensive assessment of multilingual capabilities. We evaluated mainstream LLMs, including traditional LLM and emerging Large Reasoning Models. Results show significant performance differences between the two domains, particularly in performance metrics, ranking, calibration, and robustness. This highlights the need for targeted evaluation and optimization in multilingual contexts. We hope KoLasSimpleQA will help the research community better identify LLM capability boundaries in multilingual contexts and provide guidance for model optimization. We will release KoLasSimpleQA at https://github.com/opendatalab/KoLasSimpleQA .
>
---
#### [new 078] PaTH Attention: Position Encoding via Accumulating Householder Transformations
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出PaTH位置编码方法，改进RoPE的局限性。针对RoPE因仅依赖相对位置导致的表达力不足问题，PaTH通过积累数据相关的Householder变换，使变换与输入动态关联，并设计高效并行算法优化训练，实验显示优于RoPE等基线。**

- **链接: [http://arxiv.org/pdf/2505.16381v1](http://arxiv.org/pdf/2505.16381v1)**

> **作者:** Songlin Yang; Yikang Shen; Kaiyue Wen; Shawn Tan; Mayank Mishra; Liliang Ren; Rameswar Panda; Yoon Kim
>
> **备注:** Preprint
>
> **摘要:** The attention mechanism is a core primitive in modern large language models (LLMs) and AI more broadly. Since attention by itself is permutation-invariant, position encoding is essential for modeling structured domains such as language. Rotary position encoding (RoPE) has emerged as the de facto standard approach for position encoding and is part of many modern LLMs. However, in RoPE the key/query transformation between two elements in a sequence is only a function of their relative position and otherwise independent of the actual input. This limits the expressivity of RoPE-based transformers. This paper describes PaTH, a flexible data-dependent position encoding scheme based on accumulated products of Householder(like) transformations, where each transformation is data-dependent, i.e., a function of the input. We derive an efficient parallel algorithm for training through exploiting a compact representation of products of Householder matrices, and implement a FlashAttention-style blockwise algorithm that minimizes I/O cost. Across both targeted synthetic benchmarks and moderate-scale real-world language modeling experiments, we find that PaTH demonstrates superior performance compared to RoPE and other recent baselines.
>
---
#### [new 079] Internal and External Impacts of Natural Language Processing Papers
- **分类: cs.CL; cs.DL**

- **简介: 该论文分析了顶级NLP会议论文（1979-2024）的学术与社会影响，通过引用和专利/媒体/政策数据，比较不同主题的影响力。发现语言模型影响最广，伦理等议题政策关注多而学术引用少，外部领域侧重应用或社会影响。任务为评估NLP研究的内外影响力差异及领域偏好。**

- **链接: [http://arxiv.org/pdf/2505.16061v1](http://arxiv.org/pdf/2505.16061v1)**

> **作者:** Yu Zhang
>
> **备注:** 7 pages; Accepted to ACL 2025
>
> **摘要:** We investigate the impacts of NLP research published in top-tier conferences (i.e., ACL, EMNLP, and NAACL) from 1979 to 2024. By analyzing citations from research articles and external sources such as patents, media, and policy documents, we examine how different NLP topics are consumed both within the academic community and by the broader public. Our findings reveal that language modeling has the widest internal and external influence, while linguistic foundations have lower impacts. We also observe that internal and external impacts generally align, but topics like ethics, bias, and fairness show significant attention in policy documents with much fewer academic citations. Additionally, external domains exhibit distinct preferences, with patents focusing on practical NLP applications and media and policy documents engaging more with the societal implications of NLP models.
>
---
#### [new 080] Two-way Evidence self-Alignment based Dual-Gated Reasoning Enhancement
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对知识密集型多步推理任务，提出TW-ESA和DGR模块，解决证据逻辑对齐及不确定性推理问题，通过双向自对齐和双重门控融合提升推理准确性和鲁棒性，实验显示显著提升。**

- **链接: [http://arxiv.org/pdf/2505.16806v1](http://arxiv.org/pdf/2505.16806v1)**

> **作者:** Kexin Zhang; Junlan Chen; Daifeng Li; Yuxuan Zhang; Yangyang Feng; Bowen Deng; Weixu Chen
>
> **摘要:** Large language models (LLMs) encounter difficulties in knowledge-intensive multi-step reasoning (KIMSR) tasks. One challenge is how to effectively extract and represent rationale evidence. The current methods often extract semantically relevant but logically irrelevant evidence, resulting in flawed reasoning and inaccurate responses. We propose a two-way evidence self-alignment (TW-ESA) module, which utilizes the mutual alignment between strict reasoning and LLM reasoning to enhance its understanding of the causal logic of evidence, thereby addressing the first challenge. Another challenge is how to utilize the rationale evidence and LLM's intrinsic knowledge for accurate reasoning when the evidence contains uncertainty. We propose a dual-gated reasoning enhancement (DGR) module to gradually fuse useful knowledge of LLM within strict reasoning, which can enable the model to perform accurate reasoning by focusing on causal elements in the evidence and exhibit greater robustness. The two modules are collaboratively trained in a unified framework ESA-DGR. Extensive experiments on three diverse and challenging KIMSR datasets reveal that ESA-DGR significantly surpasses state-of-the-art LLM-based fine-tuning methods, with remarkable average improvements of 4% in exact match (EM) and 5% in F1 score. The implementation code is available at https://anonymous.4open.science/r/ESA-DGR-2BF8.
>
---
#### [new 081] In-Context Watermarks for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出In-Context Watermarking（ICW），通过提示工程在文本中嵌入水印，解决现有方法需访问模型解码的局限。针对无法访问模型的场景（如学术审稿），设计四种策略及检测方法，并验证其可行性，实现模型无关的内容溯源。**

- **链接: [http://arxiv.org/pdf/2505.16934v1](http://arxiv.org/pdf/2505.16934v1)**

> **作者:** Yepeng Liu; Xuandong Zhao; Christopher Kruegel; Dawn Song; Yuheng Bu
>
> **摘要:** The growing use of large language models (LLMs) for sensitive applications has highlighted the need for effective watermarking techniques to ensure the provenance and accountability of AI-generated text. However, most existing watermarking methods require access to the decoding process, limiting their applicability in real-world settings. One illustrative example is the use of LLMs by dishonest reviewers in the context of academic peer review, where conference organizers have no access to the model used but still need to detect AI-generated reviews. Motivated by this gap, we introduce In-Context Watermarking (ICW), which embeds watermarks into generated text solely through prompt engineering, leveraging LLMs' in-context learning and instruction-following abilities. We investigate four ICW strategies at different levels of granularity, each paired with a tailored detection method. We further examine the Indirect Prompt Injection (IPI) setting as a specific case study, in which watermarking is covertly triggered by modifying input documents such as academic manuscripts. Our experiments validate the feasibility of ICW as a model-agnostic, practical watermarking approach. Moreover, our findings suggest that as LLMs become more capable, ICW offers a promising direction for scalable and accessible content attribution.
>
---
#### [new 082] Breaking mBad! Supervised Fine-tuning for Cross-Lingual Detoxification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究跨语言去毒化任务，旨在通过监督微调使大型语言模型在高/低资源语言间迁移毒性抑制能力。针对多语言场景下毒性减少与任务性能平衡的挑战，实验分析了504种设置，在数据有限情况下评估跨分布毒性降低效果，并揭示安全性和知识保留的 trade-off。**

- **链接: [http://arxiv.org/pdf/2505.16722v1](http://arxiv.org/pdf/2505.16722v1)**

> **作者:** Himanshu Beniwal; Youngwoo Kim; Maarten Sap; Soham Dan; Thomas Hartvigsen
>
> **摘要:** As large language models (LLMs) become increasingly prevalent in global applications, ensuring that they are toxicity-free across diverse linguistic contexts remains a critical challenge. We explore "Cross-lingual Detoxification", a cross-lingual paradigm that mitigates toxicity, enabling detoxification capabilities to transfer between high and low-resource languages across different script families. We analyze cross-lingual detoxification's effectiveness through 504 extensive settings to evaluate toxicity reduction in cross-distribution settings with limited data and investigate how mitigation impacts model performance on non-toxic tasks, revealing trade-offs between safety and knowledge preservation. Our code and dataset are publicly available at https://github.com/himanshubeniwal/Breaking-mBad.
>
---
#### [new 083] HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation
- **分类: cs.CL**

- **简介: 该论文提出HiMATE框架，用于机器翻译评估。针对现有LLM方法在错误跨度识别和严重性评估上的不足，设计基于MQM层次的多智能体系统，结合自我反思和信息不对称讨论策略减少幻觉。实验显示其F1值提升89%，优于基线。**

- **链接: [http://arxiv.org/pdf/2505.16281v1](http://arxiv.org/pdf/2505.16281v1)**

> **作者:** Shijie Zhang; Renhao Li; Songsheng Wang; Philipp Koehn; Min Yang; Derek F. Wong
>
> **摘要:** The advancement of Large Language Models (LLMs) enables flexible and interpretable automatic evaluations. In the field of machine translation evaluation, utilizing LLMs with translation error annotations based on Multidimensional Quality Metrics (MQM) yields more human-aligned judgments. However, current LLM-based evaluation methods still face challenges in accurately identifying error spans and assessing their severity. In this paper, we propose HiMATE, a Hierarchical Multi-Agent Framework for Machine Translation Evaluation. We argue that existing approaches inadequately exploit the fine-grained structural and semantic information within the MQM hierarchy. To address this, we develop a hierarchical multi-agent system grounded in the MQM error typology, enabling granular evaluation of subtype errors. Two key strategies are incorporated to further mitigate systemic hallucinations within the framework: the utilization of the model's self-reflection capability and the facilitation of agent discussion involving asymmetric information. Empirically, HiMATE outperforms competitive baselines across different datasets in conducting human-aligned evaluations. Further analyses underscore its significant advantage in error span detection and severity assessment, achieving an average F1-score improvement of 89% over the best-performing baseline. We make our code and data publicly available at https://anonymous.4open.science/r/HiMATE-Anony.
>
---
#### [new 084] IFEval-Audio: Benchmarking Instruction-Following Capability in Audio-based Large Language Models
- **分类: cs.CL**

- **简介: 论文提出IFEval-Audio数据集，评估音频大语言模型的指令遵循能力。针对多模态模型在音频任务中该能力不足且缺乏评测的问题，构建含280个样本的六维度基准数据集，测试并公开以推动研究。**

- **链接: [http://arxiv.org/pdf/2505.16774v1](http://arxiv.org/pdf/2505.16774v1)**

> **作者:** Yiming Gao; Bin Wang; Chengwei Wei; Shuo Sun; AiTi Aw
>
> **备注:** Link: https://github.com/AudioLLMs/AudioBench/tree/main/IFEval-Audio
>
> **摘要:** Large language models (LLMs) have demonstrated strong instruction-following capabilities in text-based tasks. However, this ability often deteriorates in multimodal models after alignment with non-text modalities such as images or audio. While several recent efforts have investigated instruction-following performance in text and vision-language models, instruction-following in audio-based large language models remains largely unexplored. To bridge this gap, we introduce IFEval-Audio, a novel evaluation dataset designed to assess the ability to follow instructions in an audio LLM. IFEval-Audio contains 280 audio-instruction-answer triples across six diverse dimensions: Content, Capitalization, Symbol, List Structure, Length, and Format. Each example pairs an audio input with a text instruction, requiring the model to generate an output that follows a specified structure. We benchmark state-of-the-art audio LLMs on their ability to follow audio-involved instructions. The dataset is released publicly to support future research in this emerging area.
>
---
#### [new 085] Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLM）忠实性优化任务，旨在解决模型在信息生成中偏离上下文的问题。提出CANOE框架，通过合成多样化QA数据构建无标注训练集，并设计Dual-GRPO强化学习方法，结合规则奖励同时优化短/长文本生成，提升模型忠实性，实验显示其在11项任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.16483v1](http://arxiv.org/pdf/2505.16483v1)**

> **作者:** Shuzheng Si; Haozhe Zhao; Cheng Gao; Yuzhuo Bai; Zhitong Wang; Bofei Gao; Kangyang Luo; Wenhao Li; Yufei Huang; Gang Chen; Fanchao Qi; Minjia Zhang; Baobao Chang; Maosong Sun
>
> **摘要:** Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework, CANOE, to improve the faithfulness of LLMs in both short-form and long-form generation tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different downstream tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1.
>
---
#### [new 086] Learning Beyond Limits: Multitask Learning and Synthetic Data for Low-Resource Canonical Morpheme Segmentation
- **分类: cs.CL**

- **简介: 该论文针对低资源语言的语素分割任务，提出结合多任务学习与LLM生成合成数据的方法。通过同时预测形态分段和词义并共享语言表示，提升模型泛化能力；利用LLM生成合成数据缓解数据稀缺问题。实验显示该方法显著提高了多语言的分割精度和F1值。**

- **链接: [http://arxiv.org/pdf/2505.16800v1](http://arxiv.org/pdf/2505.16800v1)**

> **作者:** Changbing Yang; Garrett Nicolai
>
> **摘要:** We introduce a transformer-based morpheme segmentation system that augments a low-resource training signal through multitask learning and LLM-generated synthetic data. Our framework jointly predicts morphological segments and glosses from orthographic input, leveraging shared linguistic representations obtained through a common documentary process to enhance model generalization. To further address data scarcity, we integrate synthetic training data generated by large language models (LLMs) using in-context learning. Experimental results on the SIGMORPHON 2023 dataset show that our approach significantly improves word-level segmentation accuracy and morpheme-level F1-score across multiple low-resource languages.
>
---
#### [new 087] CASTILLO: Characterizing Response Length Distributions of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLM）推理资源管理任务，旨在解决其响应长度随机性导致的计算资源分配难题。研究构建了CASTILLO数据集，对13种开源LLM在7个任务上生成的响应长度进行统计分析（含均值、极值等），揭示模型间/内的长度变异性和生成异常现象，为资源预测与优化提供数据基础。**

- **链接: [http://arxiv.org/pdf/2505.16881v1](http://arxiv.org/pdf/2505.16881v1)**

> **作者:** Daniel F. Perez-Ramirez; Dejan Kostic; Magnus Boman
>
> **备注:** Dataset available in https://huggingface.co/datasets/danfperam/castillo and code is available in https://github.com/DanielFPerez/castillo
>
> **摘要:** Efficiently managing compute resources for Large Language Model (LLM) inference remains challenging due to the inherently stochastic and variable lengths of autoregressive text generation. Accurately estimating response lengths in advance enables proactive resource allocation, yet existing approaches either bias text generation towards certain lengths or rely on assumptions that ignore model- and prompt-specific variability. We introduce CASTILLO, a dataset characterizing response length distributions across 13 widely-used open-source LLMs evaluated on seven distinct instruction-following corpora. For each $\langle$prompt, model$\rangle$ sample pair, we generate 10 independent completions using fixed decoding hyper-parameters, record the token length of each response, and publish summary statistics (mean, std-dev, percentiles), along with the shortest and longest completions, and the exact generation settings. Our analysis reveals significant inter- and intra-model variability in response lengths (even under identical generation settings), as well as model-specific behaviors and occurrences of partial text degeneration in only subsets of responses. CASTILLO enables the development of predictive models for proactive scheduling and provides a systematic framework for analyzing model-specific generation behaviors. We publicly release the dataset and code to foster research at the intersection of generative language modeling and systems.
>
---
#### [new 088] Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出RLKD框架，通过强化学习与生成结构奖励模型（GSRM），解决传统监督微调无法有效蒸馏LLMs隐式多分支推理结构的问题。GSRM将推理路径分解为元推理-解决问题步骤，计算结构对齐奖励，使学生模型学习教师的深层推理结构而非固定路径。实验显示RLKD在少量数据下优于传统方法，提升推理能力。**

- **链接: [http://arxiv.org/pdf/2505.16142v1](http://arxiv.org/pdf/2505.16142v1)**

> **作者:** Shicheng Xu; Liang Pang; Yunchang Zhu; Jia Gu; Zihao Wei; Jingcheng Deng; Feiyang Pan; Huawei Shen; Xueqi Cheng
>
> **备注:** 15 pages
>
> **摘要:** Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
>
---
#### [new 089] Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Tool-Star框架，通过强化学习提升LLM的多工具协作推理能力，解决工具使用数据不足及有效协作问题。工作包括：设计工具集成数据合成管道生成高质量样本，提出两阶段训练方法（冷启动微调与分层奖励RL算法），实验验证其在多任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2505.16410v1](http://arxiv.org/pdf/2505.16410v1)**

> **作者:** Guanting Dong; Yifei Chen; Xiaoxi Li; Jiajie Jin; Hongjin Qian; Yutao Zhu; Hangyu Mao; Guorui Zhou; Zhicheng Dou; Ji-Rong Wen
>
> **备注:** Working in progress
>
> **摘要:** Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at https://github.com/dongguanting/Tool-Star.
>
---
#### [new 090] Resource for Error Analysis in Text Simplification: New Taxonomy and Test Collection
- **分类: cs.CL; cs.AI; I.2.6; I.5.2**

- **简介: 该论文属于文本简化（TS）任务的评估研究，旨在解决现有自动文本简化（ATS）评估方法无法有效检测错误的问题。提出信息失真导向的错误分类法，构建标注简化文本数据集，并分析现有模型的错误检测能力，为改进TS评估与模型开发提供工具。**

- **链接: [http://arxiv.org/pdf/2505.16392v1](http://arxiv.org/pdf/2505.16392v1)**

> **作者:** Benjamin Vendeville; Liana Ermakova; Pierre De Loor
>
> **备注:** Accepted at SIGIR 2025
>
> **摘要:** The general public often encounters complex texts but does not have the time or expertise to fully understand them, leading to the spread of misinformation. Automatic Text Simplification (ATS) helps make information more accessible, but its evaluation methods have not kept up with advances in text generation, especially with Large Language Models (LLMs). In particular, recent studies have shown that current ATS metrics do not correlate with the presence of errors. Manual inspections have further revealed a variety of errors, underscoring the need for a more nuanced evaluation framework, which is currently lacking. This resource paper addresses this gap by introducing a test collection for detecting and classifying errors in simplified texts. First, we propose a taxonomy of errors, with a formal focus on information distortion. Next, we introduce a parallel dataset of automatically simplified scientific texts. This dataset has been human-annotated with labels based on our proposed taxonomy. Finally, we analyze the quality of the dataset, and we study the performance of existing models to detect and classify errors from that taxonomy. These contributions give researchers the tools to better evaluate errors in ATS, develop more reliable models, and ultimately improve the quality of automatically simplified texts.
>
---
#### [new 091] Ask, Retrieve, Summarize: A Modular Pipeline for Scientific Literature Summarization
- **分类: cs.CL**

- **简介: 该论文提出XSum，针对科学文献多文档摘要任务，解决研究者难以高效整合知识的问题。通过问题生成模块动态生成检索问题，结合编辑模块生成符合学术规范的摘要，在SurveySum数据集上显著提升指标，提供可扩展框架。**

- **链接: [http://arxiv.org/pdf/2505.16349v1](http://arxiv.org/pdf/2505.16349v1)**

> **作者:** Pierre Achkar; Tim Gollub; Martin Potthast
>
> **备注:** Accepted at SCOLIA@ECIR 2025 Workshop
>
> **摘要:** The exponential growth of scientific publications has made it increasingly difficult for researchers to stay updated and synthesize knowledge effectively. This paper presents XSum, a modular pipeline for multi-document summarization (MDS) in the scientific domain using Retrieval-Augmented Generation (RAG). The pipeline includes two core components: a question-generation module and an editor module. The question-generation module dynamically generates questions adapted to the input papers, ensuring the retrieval of relevant and accurate information. The editor module synthesizes the retrieved content into coherent and well-structured summaries that adhere to academic standards for proper citation. Evaluated on the SurveySum dataset, XSum demonstrates strong performance, achieving considerable improvements in metrics such as CheckEval, G-Eval and Ref-F1 compared to existing approaches. This work provides a transparent, adaptable framework for scientific summarization with potential applications in a wide range of domains. Code available at https://github.com/webis-de/scolia25-xsum
>
---
#### [new 092] BP-Seg: A graphical model approach to unsupervised and non-contiguous text segmentation using belief propagation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BP-Seg，一种无监督图模型方法，用于非连续文本分割。解决传统方法无法有效分组远距离但语义相关句子的问题，通过信念传播结合局部连贯性与全局语义相似性，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16965v1](http://arxiv.org/pdf/2505.16965v1)**

> **作者:** Fengyi Li; Kayhan Behdin; Natesh Pillai; Xiaofeng Wang; Zhipeng Wang; Ercan Yildiz
>
> **摘要:** Text segmentation based on the semantic meaning of sentences is a fundamental task with broad utility in many downstream applications. In this paper, we propose a graphical model-based unsupervised learning approach, named BP-Seg for efficient text segmentation. Our method not only considers local coherence, capturing the intuition that adjacent sentences are often more related, but also effectively groups sentences that are distant in the text yet semantically similar. This is achieved through belief propagation on the carefully constructed graphical models. Experimental results on both an illustrative example and a dataset with long-form documents demonstrate that our method performs favorably compared to competing approaches.
>
---
#### [new 093] IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection
- **分类: cs.CL; cs.AI; cs.CV; 68T50; I.2.7; I.2.10**

- **简介: 该论文属于多模态讽刺检测任务，旨在解决现有方法未能有效利用人类认知过程分析图文关联以识别讽刺的问题。提出IRONIC框架，通过多模态连贯关系（指代、类比、语用）建模图文联系，实现零样本场景下的讽刺检测，并达当前最优效果。**

- **链接: [http://arxiv.org/pdf/2505.16258v1](http://arxiv.org/pdf/2505.16258v1)**

> **作者:** Aashish Anantha Ramakrishnan; Aadarsh Anantha Ramakrishnan; Dongwon Lee
>
> **摘要:** Interpreting figurative language such as sarcasm across multi-modal inputs presents unique challenges, often requiring task-specific fine-tuning and extensive reasoning steps. However, current Chain-of-Thought approaches do not efficiently leverage the same cognitive processes that enable humans to identify sarcasm. We present IRONIC, an in-context learning framework that leverages Multi-modal Coherence Relations to analyze referential, analogical and pragmatic image-text linkages. Our experiments show that IRONIC achieves state-of-the-art performance on zero-shot Multi-modal Sarcasm Detection across different baselines. This demonstrates the need for incorporating linguistic and cognitive insights into the design of multi-modal reasoning strategies. Our code is available at: https://github.com/aashish2000/IRONIC
>
---
#### [new 094] CUB: Benchmarking Context Utilisation Techniques for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CUB基准，评估语言模型的上下文利用技术（CMTs），解决现有方法缺乏系统对比及难以处理复杂上下文的问题。通过测试7种方法在3个数据集和9个模型上的表现，发现多数技术在真实场景中效果不佳，强调需开发更鲁棒的CMTs。**

- **链接: [http://arxiv.org/pdf/2505.16518v1](http://arxiv.org/pdf/2505.16518v1)**

> **作者:** Lovisa Hagström; Youna Kim; Haeun Yu; Sang-goo Lee; Richard Johansson; Hyunsoo Cho; Isabelle Augenstein
>
> **备注:** 27 pages
>
> **摘要:** Incorporating external knowledge is crucial for knowledge-intensive tasks, such as question answering and fact checking. However, language models (LMs) may ignore relevant information that contradicts outdated parametric memory or be distracted by irrelevant contexts. While many context utilisation manipulation techniques (CMTs) that encourage or suppress context utilisation have recently been proposed to alleviate these issues, few have seen systematic comparison. In this paper, we develop CUB (Context Utilisation Benchmark) to help practitioners within retrieval-augmented generation (RAG) identify the best CMT for their needs. CUB allows for rigorous testing on three distinct context types, observed to capture key challenges in realistic context utilisation scenarios. With this benchmark, we evaluate seven state-of-the-art methods, representative of the main categories of CMTs, across three diverse datasets and tasks, applied to nine LMs. Our results show that most of the existing CMTs struggle to handle the full set of types of contexts that may be encountered in real-world retrieval-augmented scenarios. Moreover, we find that many CMTs display an inflated performance on simple synthesised datasets, compared to more realistic datasets with naturally occurring samples. Altogether, our results show the need for holistic tests of CMTs and the development of CMTs that can handle multiple context types.
>
---
#### [new 095] Training Step-Level Reasoning Verifiers with Formal Verification Tools
- **分类: cs.CL**

- **简介: 该论文提出FoVer方法，利用Z3、Isabelle等形式验证工具自动标注步骤级错误，构建无需人工标注的训练数据集，解决PRMs（过程奖励模型）依赖高成本标注和仅适用于数学推理的局限。实验显示其在逻辑证明等任务中效果优于基线和人类标注模型。**

- **链接: [http://arxiv.org/pdf/2505.15960v1](http://arxiv.org/pdf/2505.15960v1)**

> **作者:** Ryo Kamoi; Yusen Zhang; Nan Zhang; Sarkar Snigdha Sarathi Das; Rui Zhang
>
> **备注:** Datasets, models, and code are provided at https://github.com/psunlpgroup/FoVer. Please also refer to our project website at https://fover-prm.github.io/
>
> **摘要:** Process Reward Models (PRMs), which provide step-by-step feedback on the reasoning generated by Large Language Models (LLMs), are receiving increasing attention. However, two key research gaps remain: collecting accurate step-level error labels for training typically requires costly human annotation, and existing PRMs are limited to math reasoning problems. In response to these gaps, this paper aims to address the challenges of automatic dataset creation and the generalization of PRMs to diverse reasoning tasks. To achieve this goal, we propose FoVer, an approach for training PRMs on step-level error labels automatically annotated by formal verification tools, such as Z3 for formal logic and Isabelle for theorem proof, which provide automatic and accurate verification for symbolic tasks. Using this approach, we synthesize a training dataset with error labels on LLM responses for formal logic and theorem proof tasks without human annotation. Although this data synthesis is feasible only for tasks compatible with formal verification, we observe that LLM-based PRMs trained on our dataset exhibit cross-task generalization, improving verification across diverse reasoning tasks. Specifically, PRMs trained with FoVer significantly outperform baseline PRMs based on the original LLMs and achieve competitive or superior results compared to state-of-the-art PRMs trained on labels annotated by humans or stronger models, as measured by step-level verification on ProcessBench and Best-of-K performance across 12 reasoning benchmarks, including MATH, AIME, ANLI, MMLU, and BBH. The datasets, models, and code are provided at https://github.com/psunlpgroup/FoVer.
>
---
#### [new 096] R1-Compress: Long Chain-of-Thought Compression via Chunk Compression and Search
- **分类: cs.CL**

- **简介: 该论文针对长链式推理（Long-CoT）计算开销大的问题，提出R1-Compress：通过分块压缩保留局部推理信息，结合跨块搜索确保连贯性，减少20% token使用同时保持高精度（如MATH500准确率仅降0.6%），解决现有方法信息损失或输出不连贯的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.16838v1](http://arxiv.org/pdf/2505.16838v1)**

> **作者:** Yibo Wang; Li Shen; Huanjin Yao; Tiansheng Huang; Rui Liu; Naiqiang Tan; Jiaxing Huang; Kai Zhang; Dacheng Tao
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by enabling step-by-step problem-solving, yet its extension to Long-CoT introduces substantial computational overhead due to increased token length. Existing compression approaches -- instance-level and token-level -- either sacrifice essential local reasoning signals like reflection or yield incoherent outputs. To address these limitations, we propose R1-Compress, a two-stage chunk-level compression framework that preserves both local information and coherence. Our method segments Long-CoT into manageable chunks, applies LLM-driven inner-chunk compression, and employs an inter-chunk search mechanism to select the short and coherent sequence. Experiments on Qwen2.5-Instruct models across MATH500, AIME24, and GPQA-Diamond demonstrate that R1-Compress significantly reduces token usage while maintaining comparable reasoning accuracy. On MATH500, R1-Compress achieves an accuracy of 92.4%, with only a 0.6% drop compared to the Long-CoT baseline, while reducing token usage by about 20%. Source code will be available at https://github.com/w-yibo/R1-Compress
>
---
#### [new 097] KNN-SSD: Enabling Dynamic Self-Speculative Decoding via Nearest Neighbor Layer Set Optimization
- **分类: cs.CL**

- **简介: 该论文提出KNN-SSD算法，针对自推测解码（SSD）因领域变化导致加速性能下降的问题，通过KNN匹配不同跳过层与输入领域，提升其泛化性，实现LLM推理1.3-1.6倍加速。**

- **链接: [http://arxiv.org/pdf/2505.16162v1](http://arxiv.org/pdf/2505.16162v1)**

> **作者:** Mingbo Song; Heming Xia; Jun Zhang; Chak Tou Leong; Qiancheng Xu; Wenjie Li; Sujian Li
>
> **备注:** 8 pages
>
> **摘要:** Speculative Decoding (SD) has emerged as a widely used paradigm to accelerate the inference of large language models (LLMs) without compromising generation quality. It works by efficiently drafting multiple tokens using a compact model and then verifying them in parallel using the target LLM. Notably, Self-Speculative Decoding proposes skipping certain layers to construct the draft model, which eliminates the need for additional parameters or training. Despite its strengths, we observe in this work that drafting with layer skipping exhibits significant sensitivity to domain shifts, leading to a substantial drop in acceleration performance. To enhance the domain generalizability of this paradigm, we introduce KNN-SSD, an algorithm that leverages K-Nearest Neighbor (KNN) search to match different skipped layers with various domain inputs. We evaluated our algorithm in various models and multiple tasks, observing that its application leads to 1.3x-1.6x speedup in LLM inference.
>
---
#### [new 098] Power-Law Decay Loss for Large Language Model Finetuning: Focusing on Information Sparsity to Enhance Generation Quality
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本生成模型微调任务，针对标准交叉熵损失忽视低频关键token的问题，提出Power-Law Decay Loss（PDL），通过词频幂律衰减调整token权重，降低高频token权重以提升低频信息词的贡献，增强生成文本的质量与多样性，适用于摘要、对话系统等任务。**

- **链接: [http://arxiv.org/pdf/2505.16900v1](http://arxiv.org/pdf/2505.16900v1)**

> **作者:** Jintian Shao; Hongyi Huang; Jiayi Wu; Beiwen Zhang; ZhiYu Wu; You Shan; MingKai Zheng
>
> **摘要:** During the finetuning stage of text generation tasks, standard cross-entropy loss treats all tokens equally. This can lead models to overemphasize high-frequency, low-information tokens, neglecting lower-frequency tokens crucial for specificity and informativeness in generated content. This paper introduces a novel loss function, Power-Law Decay Loss (PDL), specifically designed to optimize the finetuning process for text generation. The core motivation for PDL stems from observations in information theory and linguistics: the informativeness of a token is often inversely proportional to its frequency of occurrence. PDL re-weights the contribution of each token in the standard cross-entropy loss based on its frequency in the training corpus, following a power-law decay. Specifically, the weights for high-frequency tokens are reduced, while low-frequency, information-dense tokens are assigned higher weights. This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the quality, diversity, and informativeness of the generated text. We theoretically elaborate on the motivation and construction of PDL and discuss its potential applications and advantages across various text generation finetuning tasks, such as abstractive summarization, dialogue systems, and style transfer.
>
---
#### [new 099] SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Prediction in Academic Papers
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文属于学术论文自动新颖性预测任务，旨在解决现有方法依赖词汇/实体组合导致评估效果有限的问题。通过划分IMRaD章节结构，测试不同组合输入语言模型，发现引言、结果与讨论组合最优，而引言和结果为关键部分。**

- **链接: [http://arxiv.org/pdf/2505.16330v1](http://arxiv.org/pdf/2505.16330v1)**

> **作者:** Wenqing Wu; Chengzhi Zhang; Tong Bao; Yi Zhao
>
> **摘要:** Novelty is a core component of academic papers, and there are multiple perspectives on the assessment of novelty. Existing methods often focus on word or entity combinations, which provide limited insights. The content related to a paper's novelty is typically distributed across different core sections, e.g., Introduction, Methodology and Results. Therefore, exploring the optimal combination of sections for evaluating the novelty of a paper is important for advancing automated novelty assessment. In this paper, we utilize different combinations of sections from academic papers as inputs to drive language models to predict novelty scores. We then analyze the results to determine the optimal section combinations for novelty score prediction. We first employ natural language processing techniques to identify the sectional structure of academic papers, categorizing them into introduction, methods, results, and discussion (IMRaD). Subsequently, we used different combinations of these sections (e.g., introduction and methods) as inputs for pretrained language models (PLMs) and large language models (LLMs), employing novelty scores provided by human expert reviewers as ground truth labels to obtain prediction results. The results indicate that using introduction, results and discussion is most appropriate for assessing the novelty of a paper, while the use of the entire text does not yield significant results. Furthermore, based on the results of the PLMs and LLMs, the introduction and results appear to be the most important section for the task of novelty score prediction. The code and dataset for this paper can be accessed at https://github.com/njust-winchy/SC4ANM.
>
---
#### [new 100] Mechanistic Understanding and Mitigation of Language Confusion in English-Centric Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究英语为中心的大型语言模型（LLMs）的语言混淆问题，即模型生成非用户所需的语言。通过行为基准测试和神经元级分析，发现语言切换的关键点由最终层转换失败驱动，提出编辑关键神经元的干预方法，在减少混淆的同时保持模型性能，为多语言建模提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.16538v1](http://arxiv.org/pdf/2505.16538v1)**

> **作者:** Ercong Nie; Helmut Schmid; Hinrich Schütze
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Language confusion -- where large language models (LLMs) generate unintended languages against the user's need -- remains a critical challenge, especially for English-centric models. We present the first mechanistic interpretability (MI) study of language confusion, combining behavioral benchmarking with neuron-level analysis. Using the Language Confusion Benchmark (LCB), we show that confusion points (CPs) -- specific positions where language switches occur -- are central to this phenomenon. Through layer-wise analysis with TunedLens and targeted neuron attribution, we reveal that transition failures in the final layers drive confusion. We further demonstrate that editing a small set of critical neurons, identified via comparative analysis with multilingual-tuned models, substantially mitigates confusion without harming general competence or fluency. Our approach matches multilingual alignment in confusion reduction for most languages and yields cleaner, higher-quality outputs. These findings provide new insights into the internal dynamics of LLMs and highlight neuron-level interventions as a promising direction for robust, interpretable multilingual language modeling.
>
---
#### [new 101] SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，旨在解决现有模型依赖昂贵外部监督（如人工标注或奖励模型）的问题。提出SSR-Zero框架，通过无参考、在线自奖励机制训练模型，在英中翻译任务上超越多个大型模型，结合外部监督后达SOTA，验证了自奖励机制的有效性。**

- **链接: [http://arxiv.org/pdf/2505.16637v1](http://arxiv.org/pdf/2505.16637v1)**

> **作者:** Wenjie Yang; Mao Zheng; Mingyang Song; Zheng Li
>
> **摘要:** Large language models (LLMs) have recently demonstrated remarkable capabilities in machine translation (MT). However, most advanced MT-specific LLMs heavily rely on external supervision signals during training, such as human-annotated reference data or trained reward models (RMs), which are often expensive to obtain and challenging to scale. To overcome this limitation, we propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for MT that is reference-free, fully online, and relies solely on self-judging rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs, e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR with external supervision from COMET, our strongest model, SSR-X-Zero-7B, achieves state-of-the-art performance in English $\leftrightarrow$ Chinese translation, surpassing all existing open-source models under 72B parameters and even outperforming closed-source models, e.g., GPT-4o and Gemini 1.5 Pro. Our analysis highlights the effectiveness of the self-rewarding mechanism compared to the external LLM-as-a-judge approach in MT and demonstrates its complementary benefits when combined with trained RMs. Our findings provide valuable insight into the potential of self-improving RL methods. We have publicly released our code, data and models.
>
---
#### [new 102] Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理效率优化任务，旨在解决Chain-of-Thought（CoT）推理计算成本高、低效的问题。提出CoLaR框架，通过两阶段训练（监督微调+强化学习），在潜空间动态压缩推理链：首先预测压缩嵌入，再利用非确定性潜态头探索更短路径，实现推理链长度缩短53.3%且精度损失仅4.8%。**

- **链接: [http://arxiv.org/pdf/2505.16552v1](http://arxiv.org/pdf/2505.16552v1)**

> **作者:** Wenhui Tan; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Ruihua Song
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
>
---
#### [new 103] LAGO: Few-shot Crosslingual Embedding Inversion Attacks via Language Similarity-Aware Graph Optimization
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文提出LAGO方法，针对多语言NLP系统的隐私漏洞，解决跨语言嵌入逆攻击的少样本问题。通过构建语言相似性感知的图优化框架，利用句法和词汇约束实现跨语言协作学习，结合正则化与约束条件，在极少量数据下提升攻击跨语言嵌入空间的转移性，实验显示较基线提升10-20%。**

- **链接: [http://arxiv.org/pdf/2505.16008v1](http://arxiv.org/pdf/2505.16008v1)**

> **作者:** Wenrui Yu; Yiyi Chen; Johannes Bjerva; Sokol Kosta; Qiongxiu Li
>
> **摘要:** We propose LAGO - Language Similarity-Aware Graph Optimization - a novel approach for few-shot cross-lingual embedding inversion attacks, addressing critical privacy vulnerabilities in multilingual NLP systems. Unlike prior work in embedding inversion attacks that treat languages independently, LAGO explicitly models linguistic relationships through a graph-based constrained distributed optimization framework. By integrating syntactic and lexical similarity as edge constraints, our method enables collaborative parameter learning across related languages. Theoretically, we show this formulation generalizes prior approaches, such as ALGEN, which emerges as a special case when similarity constraints are relaxed. Our framework uniquely combines Frobenius-norm regularization with linear inequality or total variation constraints, ensuring robust alignment of cross-lingual embedding spaces even with extremely limited data (as few as 10 samples per language). Extensive experiments across multiple languages and embedding models demonstrate that LAGO substantially improves the transferability of attacks with 10-20% increase in Rouge-L score over baselines. This work establishes language similarity as a critical factor in inversion attack transferability, urging renewed focus on language-aware privacy-preserving multilingual embeddings.
>
---
#### [new 104] On Multilingual Encoder Language Model Compression for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言模型压缩任务，针对低资源语言模型效率低问题，提出结合知识蒸馏、结构化剪枝等技术进行极端压缩，显著缩小模型规模（92%压缩率），同时保持下游任务（如情感分析等）性能（仅2-10%下降）。通过消融实验验证方法有效性，并分析数据量对压缩效果的影响。**

- **链接: [http://arxiv.org/pdf/2505.16956v1](http://arxiv.org/pdf/2505.16956v1)**

> **作者:** Daniil Gurgurov; Michal Gregor; Josef van Genabith; Simon Ostermann
>
> **备注:** Pre-print
>
> **摘要:** In this paper, we combine two-step knowledge distillation, structured pruning, truncation, and vocabulary trimming for extremely compressing multilingual encoder-only language models for low-resource languages. Our novel approach systematically combines existing techniques and takes them to the extreme, reducing layer depth, feed-forward hidden size, and intermediate layer embedding size to create significantly smaller monolingual models while retaining essential language-specific knowledge. We achieve compression rates of up to 92% with only a marginal performance drop of 2-10% in four downstream tasks, including sentiment analysis, topic classification, named entity recognition, and part-of-speech tagging, across three low-resource languages. Notably, the performance degradation correlates with the amount of language-specific data in the teacher model, with larger datasets resulting in smaller performance losses. Additionally, we conduct extensive ablation studies to identify best practices for multilingual model compression using these techniques.
>
---
#### [new 105] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型的In-Context Meta Learning机制，旨在解释模型如何通过训练获得元学习能力而非仅复制答案。通过扩展复制任务至元学习场景，发现模型在训练中经历多阶段电路涌现，每个阶段形成独特电路，揭示ICL能力的动态形成过程。**

- **链接: [http://arxiv.org/pdf/2505.16694v1](http://arxiv.org/pdf/2505.16694v1)**

> **作者:** Gouki Minegishi; Hiroki Furuta; Shohei Taniguchi; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Transformer-based language models exhibit In-Context Learning (ICL), where predictions are made adaptively based on context. While prior work links induction heads to ICL through a sudden jump in accuracy, this can only account for ICL when the answer is included within the context. However, an important property of practical ICL in large language models is the ability to meta-learn how to solve tasks from context, rather than just copying answers from context; how such an ability is obtained during training is largely unexplored. In this paper, we experimentally clarify how such meta-learning ability is acquired by analyzing the dynamics of the model's circuit during training. Specifically, we extend the copy task from previous research into an In-Context Meta Learning setting, where models must infer a task from examples to answer queries. Interestingly, in this setting, we find that there are multiple phases in the process of acquiring such abilities, and that a unique circuit emerges in each phase, contrasting with the single-phases change in induction heads. The emergence of such circuits can be related to several phenomena known in large language models, and our analysis lead to a deeper understanding of the source of the transformer's ICL ability.
>
---
#### [new 106] OpenEthics: A Comprehensive Ethical Evaluation of Open-Source Generative Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于开源大模型伦理评估任务，解决现有研究在伦理维度、语言覆盖和模型多样性不足的问题。工作包括对29个模型进行跨英语/土耳其语的稳健性、可靠性、安全性和公平性评估，发现优化侧重安全与公平，可靠性待改进，大参数模型如Gemma和Qwen表现最优。**

- **链接: [http://arxiv.org/pdf/2505.16036v1](http://arxiv.org/pdf/2505.16036v1)**

> **作者:** Burak Erinç Çetin; Yıldırım Özen; Elif Naz Demiryılmaz; Kaan Engür; Cagri Toraman
>
> **摘要:** Generative large language models present significant potential but also raise critical ethical concerns. Most studies focus on narrow ethical dimensions, and also limited diversity of languages and models. To address these gaps, we conduct a broad ethical evaluation of 29 recent open-source large language models using a novel data collection including four ethical aspects: Robustness, reliability, safety, and fairness. We analyze model behavior in both a commonly used language, English, and a low-resource language, Turkish. Our aim is to provide a comprehensive ethical assessment and guide safer model development by filling existing gaps in evaluation breadth, language coverage, and model diversity. Our experimental results, based on LLM-as-a-Judge, reveal that optimization efforts for many open-source models appear to have prioritized safety and fairness, and demonstrated good robustness while reliability remains a concern. We demonstrate that ethical evaluation can be effectively conducted independently of the language used. In addition, models with larger parameter counts tend to exhibit better ethical performance, with Gemma and Qwen models demonstrating the most ethical behavior among those evaluated.
>
---
#### [new 107] Aligning Dialogue Agents with Global Feedback via Large Language Model Reward Decomposition
- **分类: cs.CL**

- **简介: 该论文属于对话系统优化任务，旨在通过单一全局反馈信号自动分解奖励，解决传统方法依赖人工设计或细粒度反馈的问题。提出基于冻结LLM的奖励分解框架，通过文本或结合多模态行为描述推断局部奖励，蒸馏后用于强化学习，提升对话质量。**

- **链接: [http://arxiv.org/pdf/2505.15922v1](http://arxiv.org/pdf/2505.15922v1)**

> **作者:** Dong Won Lee; Hae Won Park; Cynthia Breazeal; Louis-Philippe Morency
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** We propose a large language model based reward decomposition framework for aligning dialogue agents using only a single session-level feedback signal. We leverage the reasoning capabilities of a frozen, pretrained large language model (LLM) to infer fine-grained local implicit rewards by decomposing global, session-level feedback. Our first text-only variant prompts the LLM to perform reward decomposition using only the dialogue transcript. The second multimodal variant incorporates additional behavioral cues, such as pitch, gaze, and facial affect, expressed as natural language descriptions. These inferred turn-level rewards are distilled into a lightweight reward model, which we utilize for RL-based fine-tuning for dialogue generation. We evaluate both text-only and multimodal variants against state-of-the-art reward decomposition methods and demonstrate notable improvements in human evaluations of conversation quality, suggesting that LLMs are strong reward decomposers that obviate the need for manual reward shaping and granular human feedback.
>
---
#### [new 108] Leveraging Online Data to Enhance Medical Knowledge in a Small Persian Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对波斯语小型语言模型在医学领域数据匮乏的问题，通过爬取医学杂志和构建医生-患者问答数据集进行微调，提升其医疗问答准确性，为资源有限环境提供解决方案。**

- **链接: [http://arxiv.org/pdf/2505.16000v1](http://arxiv.org/pdf/2505.16000v1)**

> **作者:** Mehrdad ghassabi; Pedram Rostami; Hamidreza Baradaran Kashani; Amirhossein Poursina; Zahra Kazemi; Milad Tavakoli
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The rapid advancement of language models has demonstrated the potential of artificial intelligence in the healthcare industry. However, small language models struggle with specialized domains in low-resource languages like Persian. While numerous medical-domain websites exist in Persian, no curated dataset or corpus has been available making ours the first of its kind. This study explores the enhancement of medical knowledge in a small language model by leveraging accessible online data, including a crawled corpus from medical magazines and a dataset of real doctor-patient QA pairs. We fine-tuned a baseline model using our curated data to improve its medical knowledge. Benchmark evaluations demonstrate that the fine-tuned model achieves improved accuracy in medical question answering and provides better responses compared to its baseline. This work highlights the potential of leveraging open-access online data to enrich small language models in medical fields, providing a novel solution for Persian medical AI applications suitable for resource-constrained environments.
>
---
#### [new 109] PIIvot: A Lightweight NLP Anonymization Framework for Question-Anchored Tutoring Dialogues
- **分类: cs.CL**

- **简介: 论文提出PIIvot，轻量级NLP框架，用于教学对话中的PII匿名化。解决现有方法误差高、查准率与召回率权衡导致应用受限的问题。通过利用数据上下文简化检测，并发布QATD-2k数据集，支持教育对话研究。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16931v1](http://arxiv.org/pdf/2505.16931v1)**

> **作者:** Matthew Zent; Digory Smith; Simon Woodhead
>
> **备注:** 6 pages, 2 figures, submitted to EMNLP 2025, for associated dataset, see https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k
>
> **摘要:** Personally identifiable information (PII) anonymization is a high-stakes task that poses a barrier to many open-science data sharing initiatives. While PII identification has made large strides in recent years, in practice, error thresholds and the recall/precision trade-off still limit the uptake of these anonymization pipelines. We present PIIvot, a lighter-weight framework for PII anonymization that leverages knowledge of the data context to simplify the PII detection problem. To demonstrate its effectiveness, we also contribute QATD-2k, the largest open-source real-world tutoring dataset of its kind, to support the demand for quality educational dialogue data.
>
---
#### [new 110] Steering Large Language Models for Machine Translation Personalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译个性化任务，解决大语言模型在低资源场景下难以满足隐式风格需求的问题。提出结合提示策略、推理干预及基于稀疏自编码器的对比框架，通过提取潜在概念引导翻译风格，同时保持翻译质量，并分析了方法对模型表征的影响。**

- **链接: [http://arxiv.org/pdf/2505.16612v1](http://arxiv.org/pdf/2505.16612v1)**

> **作者:** Daniel Scalena; Gabriele Sarti; Arianna Bisazza; Elisabetta Fersini; Malvina Nissim
>
> **摘要:** High-quality machine translation systems based on large language models (LLMs) have simplified the production of personalized translations reflecting specific stylistic constraints. However, these systems still struggle in settings where stylistic requirements are less explicit and might be harder to convey via prompting. We explore various strategies for personalizing LLM-generated translations in low-resource settings, focusing on the challenging literary translation domain. We explore prompting strategies and inference-time interventions for steering model generations towards a personalized style, and propose a contrastive framework exploiting latent concepts extracted from sparse autoencoders to identify salient personalization properties. Our results show that steering achieves strong personalization while preserving translation quality. We further examine the impact of steering on LLM representations, finding model layers with a relevant impact for personalization are impacted similarly by multi-shot prompting and our steering method, suggesting similar mechanism at play.
>
---
#### [new 111] Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文研究大语言模型（LLMs）的机器遗忘任务，旨在解决现有基于token的评估指标无法准确衡量遗忘效果的问题。提出基于表示层面的评估框架（PCA相似性、核对齐、Fisher信息），分析六种遗忘方法在文本、代码、数学等领域的表现，发现遗忘存在可逆（保留潜在特征）与不可逆（深层损伤）两类，并揭示输出层权重微调导致评估误导，提供工具包促进可信遗忘研究。**

- **链接: [http://arxiv.org/pdf/2505.16831v1](http://arxiv.org/pdf/2505.16831v1)**

> **作者:** Xiaoyu Xu; Xiang Yue; Yang Liu; Qingqing Ye; Haibo Hu; Minxin Du
>
> **备注:** 44 pages
>
> **摘要:** Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: https://github.com/XiaoyuXU1/Representational_Analysis_Tools.git.
>
---
#### [new 112] Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning
- **分类: cs.CL**

- **简介: 该论文综述潜隐CoT推理，解决传统CoT显式语言步骤的低效及抽象推理限制。提出四维度分类法，分析方法设计与挑战，推动LLM推理发展。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16782v1](http://arxiv.org/pdf/2505.16782v1)**

> **作者:** Xinghao Chen; Anhao Zhao; Heming Xia; Xuan Lu; Hanlin Wang; Yanjun Chen; Wei Zhang; Jian Wang; Wenjie Li; Xiaoyu Shen
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance on complex reasoning tasks with Chain-of-Thought (CoT) prompting. However, conventional CoT relies on reasoning steps explicitly verbalized in natural language, introducing inefficiencies and limiting its applicability to abstract reasoning. To address this, there has been growing research interest in latent CoT reasoning, where inference occurs within latent spaces. By decoupling reasoning from language, latent reasoning promises richer cognitive representations and more flexible, faster inference. Researchers have explored various directions in this promising field, including training methodologies, structural innovations, and internal reasoning mechanisms. This paper presents a comprehensive overview and analysis of this reasoning paradigm. We begin by proposing a unified taxonomy from four perspectives: token-wise strategies, internal mechanisms, analysis, and applications. We then provide in-depth discussions and comparative analyses of representative methods, highlighting their design patterns, strengths, and open challenges. We aim to provide a structured foundation for advancing this emerging direction in LLM reasoning. The relevant papers will be regularly updated at https://github.com/EIT-NLP/Awesome-Latent-CoT.
>
---
#### [new 113] BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law, including case law
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BR-TaxQA-R数据集，用于巴西个人所得税法的参考问答任务，包含715个官方问题及法律条文。通过RAG框架对比不同文本分段策略和商业模型，发现系统在回答相关性上优于商业工具，但事实正确性需人工审核，数据集已公开。**

- **链接: [http://arxiv.org/pdf/2505.15916v1](http://arxiv.org/pdf/2505.15916v1)**

> **作者:** Juvenal Domingos Júnior; Augusto Faria; E. Seiti de Oliveira; Erick de Brito; Matheus Teotonio; Andre Assumpção; Diedre Carmo; Roberto Lotufo; Jayr Pereira
>
> **摘要:** This paper presents BR-TaxQA-R, a novel dataset designed to support question answering with references in the context of Brazilian personal income tax law. The dataset contains 715 questions from the 2024 official Q\&A document published by Brazil's Internal Revenue Service, enriched with statutory norms and administrative rulings from the Conselho Administrativo de Recursos Fiscais (CARF). We implement a Retrieval-Augmented Generation (RAG) pipeline using OpenAI embeddings for searching and GPT-4o-mini for answer generation. We compare different text segmentation strategies and benchmark our system against commercial tools such as ChatGPT and Perplexity.ai using RAGAS-based metrics. Results show that our custom RAG pipeline outperforms commercial systems in Response Relevancy, indicating stronger alignment with user queries, while commercial models achieve higher scores in Factual Correctness and fluency. These findings highlight a trade-off between legally grounded generation and linguistic fluency. Crucially, we argue that human expert evaluation remains essential to ensure the legal validity of AI-generated answers in high-stakes domains such as taxation. BR-TaxQA-R is publicly available at https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R.
>
---
#### [new 114] KoBALT: Korean Benchmark For Advanced Linguistic Tasks
- **分类: cs.CL**

- **简介: 该论文提出KoBALT，一个针对韩语的综合性语言基准测试，旨在解决现有评估方法缺乏语言深度和类型学支持的问题。通过设计覆盖句法、语义等五领域的700道专家策划问题（低语料库重叠），减少数据泄露风险，评估20个LLM发现性能差异显著（最高61%准确率），并验证其与人类判断强相关，为韩语模型提供有效评测框架。**

- **链接: [http://arxiv.org/pdf/2505.16125v1](http://arxiv.org/pdf/2505.16125v1)**

> **作者:** Hyopil Shin; Sangah Lee; Dongjun Jang; Wooseok Song; Jaeyoon Kim; Chaeyoung Oh; Hyemi Jo; Youngchae Ahn; Sihyun Oh; Hyohyeong Chang; Sunkyoung Kim; Jinsik Lee
>
> **备注:** Under Reveiw
>
> **摘要:** We introduce KoBALT (Korean Benchmark for Advanced Linguistic Tasks), a comprehensive linguistically-motivated benchmark comprising 700 multiple-choice questions spanning 24 phenomena across five linguistic domains: syntax, semantics, pragmatics, phonetics/phonology, and morphology. KoBALT is designed to advance the evaluation of large language models (LLMs) in Korean, a morphologically rich language, by addressing the limitations of conventional benchmarks that often lack linguistic depth and typological grounding. It introduces a suite of expert-curated, linguistically motivated questions with minimal n-gram overlap with standard Korean corpora, substantially mitigating the risk of data contamination and allowing a more robust assessment of true language understanding. Our evaluation of 20 contemporary LLMs reveals significant performance disparities, with the highest-performing model achieving 61\% general accuracy but showing substantial variation across linguistic domains - from stronger performance in semantics (66\%) to considerable weaknesses in phonology (31\%) and morphology (36\%). Through human preference evaluation with 95 annotators, we demonstrate a strong correlation between KoBALT scores and human judgments, validating our benchmark's effectiveness as a discriminative measure of Korean language understanding. KoBALT addresses critical gaps in linguistic evaluation for typologically diverse languages and provides a robust framework for assessing genuine linguistic competence in Korean language models.
>
---
#### [new 115] Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLMs去偏任务，针对现有方法难以同时消除多类型偏见的问题，提出含五种偏见的多偏基准，并开发因果效应估计引导的CMBE方法，通过分离语义与偏见的因果效应实现多偏消除，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.16522v1](http://arxiv.org/pdf/2505.16522v1)**

> **作者:** Zhouhao Sun; Zhiyuan Kan; Xiao Ding; Li Du; Yang Zhao; Bing Qin; Ting Liu
>
> **摘要:** Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs.
>
---
#### [new 116] When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction
- **分类: cs.CL**

- **简介: 该论文研究大语言模型何时承认错误，属于模型自我修正任务。通过构建模型特定数据集，发现LLMs收回错误答案的频率低且与内部信念相关：仅当模型不相信答案时才会尝试验证并收回，监督微调可提升此能力。**

- **链接: [http://arxiv.org/pdf/2505.16170v1](http://arxiv.org/pdf/2505.16170v1)**

> **作者:** Yuqing Yang; Robin Jia
>
> **摘要:** Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on https://github.com/ayyyq/llm-retraction.
>
---
#### [new 117] Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文通过因果干预方法分析大型语言模型（LLMs）处理英语填充-间隙结构（如疑问句、定语从句）的机制，探究其共享语法属性。任务为利用LLMs内部分析推动语言学理论发展，解决传统理论与模型隐含机制间的差异。研究通过分布式互换实验，发现LLMs对这些结构的相似抽象分析，并揭示频率、填充类型等新影响因素，为理论修正提供依据。**

- **链接: [http://arxiv.org/pdf/2505.16002v1](http://arxiv.org/pdf/2505.16002v1)**

> **作者:** Sasha Boguraev; Christopher Potts; Kyle Mahowald
>
> **备注:** 20 pages, 19 figures, 11 tables
>
> **摘要:** Large Language Models (LLMs) have emerged as powerful sources of evidence for linguists seeking to develop theories of syntax. In this paper, we argue that causal interpretability methods, applied to LLMs, can greatly enhance the value of such evidence by helping us characterize the abstract mechanisms that LLMs learn to use. Our empirical focus is a set of English filler-gap dependency constructions (e.g., questions, relative clauses). Linguistic theories largely agree that these constructions share many properties. Using experiments based in Distributed Interchange Interventions, we show that LLMs converge on similar abstract analyses of these constructions. These analyses also reveal previously overlooked factors -- relating to frequency, filler type, and surrounding context -- that could motivate changes to standard linguistic theory. Overall, these results suggest that mechanistic, internal analyses of LLMs can push linguistic theory forward.
>
---
#### [new 118] BiasLab: Toward Explainable Political Bias Detection with Dual-Axis Annotations and Rationale Indicators
- **分类: cs.CL**

- **简介: 该论文提出BiasLab数据集，通过双轴标注（对民主党和共和党的情感倾向）及理由标注，解决政治偏见检测的可解释性问题。工作包括构建300篇新闻文章数据集、分析标注一致性、对比GPT-4o与人类标注差异，并建立偏见感知预测与理由分类基准，推动透明NLP系统开发。**

- **链接: [http://arxiv.org/pdf/2505.16081v1](http://arxiv.org/pdf/2505.16081v1)**

> **作者:** KMA Solaiman
>
> **备注:** Under review
>
> **摘要:** We present BiasLab, a dataset of 300 political news articles annotated for perceived ideological bias. These articles were selected from a curated 900-document pool covering diverse political events and source biases. Each article is labeled by crowdworkers along two independent scales, assessing sentiment toward the Democratic and Republican parties, and enriched with rationale indicators. The annotation pipeline incorporates targeted worker qualification and was refined through pilot-phase analysis. We quantify inter-annotator agreement, analyze misalignment with source-level outlet bias, and organize the resulting labels into interpretable subsets. Additionally, we simulate annotation using schema-constrained GPT-4o, enabling direct comparison to human labels and revealing mirrored asymmetries, especially in misclassifying subtly right-leaning content. We define two modeling tasks: perception drift prediction and rationale type classification, and report baseline performance to illustrate the challenge of explainable bias detection. BiasLab's rich rationale annotations provide actionable interpretations that facilitate explainable modeling of political bias, supporting the development of transparent, socially aware NLP systems. We release the dataset, annotation schema, and modeling code to encourage research on human-in-the-loop interpretability and the evaluation of explanation effectiveness in real-world settings.
>
---
#### [new 119] LLaMAs Have Feelings Too: Unveiling Sentiment and Emotion Representations in LLaMA Models Through Probing
- **分类: cs.CL; cs.AI**

- **简介: 该论文通过探针任务探究LLaMA模型中的情感表征机制。属于模型内部表征分析任务，旨在定位情感特征分布并优化分析效率。工作包括用探针分类器检测各层情感编码，发现中间层最有效，末尾token非最优，且方法使内存需求降低57%。**

- **链接: [http://arxiv.org/pdf/2505.16491v1](http://arxiv.org/pdf/2505.16491v1)**

> **作者:** Dario Di Palma; Alessandro De Bellis; Giovanni Servedio; Vito Walter Anelli; Fedelucio Narducci; Tommaso Di Noia
>
> **摘要:** Large Language Models (LLMs) have rapidly become central to NLP, demonstrating their ability to adapt to various tasks through prompting techniques, including sentiment analysis. However, we still have a limited understanding of how these models capture sentiment-related information. This study probes the hidden layers of Llama models to pinpoint where sentiment features are most represented and to assess how this affects sentiment analysis. Using probe classifiers, we analyze sentiment encoding across layers and scales, identifying the layers and pooling methods that best capture sentiment signals. Our results show that sentiment information is most concentrated in mid-layers for binary polarity tasks, with detection accuracy increasing up to 14% over prompting techniques. Additionally, we find that in decoder-only models, the last token is not consistently the most informative for sentiment encoding. Finally, this approach enables sentiment tasks to be performed with memory requirements reduced by an average of 57%. These insights contribute to a broader understanding of sentiment in LLMs, suggesting layer-specific probing as an effective approach for sentiment tasks beyond prompting, with potential to enhance model utility and reduce memory requirements.
>
---
#### [new 120] Diverse, not Short: A Length-Controlled Self-Learning Framework for Improving Response Diversity of Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型生成任务，旨在解决现有多样性优化方法因偏爱短输出而限制表达的问题。提出Diverse-NS框架，通过长度控制的自学习机制，利用少量偏好数据平衡多样性与长度，提升模型在创意生成等任务的多样性表现，同时发现小模型可有效指导大模型的多样性训练。**

- **链接: [http://arxiv.org/pdf/2505.16245v1](http://arxiv.org/pdf/2505.16245v1)**

> **作者:** Vijeta Deshpande; Debasmita Ghose; John D. Patterson; Roger Beaty; Anna Rumshisky
>
> **摘要:** Diverse language model responses are crucial for creative generation, open-ended tasks, and self-improvement training. We show that common diversity metrics, and even reward models used for preference optimization, systematically bias models toward shorter outputs, limiting expressiveness. To address this, we introduce Diverse, not Short (Diverse-NS), a length-controlled self-learning framework that improves response diversity while maintaining length parity. By generating and filtering preference data that balances diversity, quality, and length, Diverse-NS enables effective training using only 3,000 preference pairs. Applied to LLaMA-3.1-8B and the Olmo-2 family, Diverse-NS substantially enhances lexical and semantic diversity. We show consistent improvement in diversity with minor reduction or gains in response quality on four creative generation tasks: Divergent Associations, Persona Generation, Alternate Uses, and Creative Writing. Surprisingly, experiments with the Olmo-2 model family (7B, and 13B) show that smaller models like Olmo-2-7B can serve as effective "diversity teachers" for larger models. By explicitly addressing length bias, our method efficiently pushes models toward more diverse and expressive outputs.
>
---
#### [new 121] Pre-training Large Memory Language Models with Internal and External Knowledge
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Large Memory Language Models（LMLM），通过预训练将事实知识存储于模型参数和外部数据库，解决传统语言模型作为黑箱难以验证、更新知识的问题。方法上屏蔽外部知识的训练损失，促使模型主动检索而非记忆，实现性能与大模型相当且知识库可编辑验证。**

- **链接: [http://arxiv.org/pdf/2505.15962v1](http://arxiv.org/pdf/2505.15962v1)**

> **作者:** Linxi Zhao; Sofian Zalouk; Christian K. Belardi; Justin Lovelace; Jin Peng Zhou; Kilian Q. Weinberger; Yoav Artzi; Jennifer J. Sun
>
> **摘要:** Neural language models are black-boxes -- both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We propose a new class of language models, Large Memory Language Models (LMLM) with a pre-training recipe that stores factual knowledge in both internal weights and an external database. Our approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger, knowledge-dense LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases. This work represents a fundamental shift in how language models interact with and manage factual knowledge.
>
---
#### [new 122] What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse
- **分类: cs.CL; cs.MM**

- **简介: 该论文构建首个气候变迁表情包数据集CLIMATEMEMES（含1,184个样本），探究媒体框架与立场的关联。任务为立场检测与媒体框架识别，通过评估LLaVA-NeXT等模型发现人类标注显著提升效果，LLMs在框架检测优于VLMs，揭示视觉语言模型处理气候议题的局限性。**

- **链接: [http://arxiv.org/pdf/2505.16592v1](http://arxiv.org/pdf/2505.16592v1)**

> **作者:** Shijia Zhou; Siyao Peng; Simon Luebke; Jörg Haßler; Mario Haim; Saif M. Mohammad; Barbara Plank
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Media framing refers to the emphasis on specific aspects of perceived reality to shape how an issue is defined and understood. Its primary purpose is to shape public perceptions often in alignment with the authors' opinions and stances. However, the interaction between stance and media frame remains largely unexplored. In this work, we apply an interdisciplinary approach to conceptualize and computationally explore this interaction with internet memes on climate change. We curate CLIMATEMEMES, the first dataset of climate-change memes annotated with both stance and media frames, inspired by research in communication science. CLIMATEMEMES includes 1,184 memes sourced from 47 subreddits, enabling analysis of frame prominence over time and communities, and sheds light on the framing preferences of different stance holders. We propose two meme understanding tasks: stance detection and media frame detection. We evaluate LLaVA-NeXT and Molmo in various setups, and report the corresponding results on their LLM backbone. Human captions consistently enhance performance. Synthetic captions and human-corrected OCR also help occasionally. Our findings highlight that VLMs perform well on stance, but struggle on frames, where LLMs outperform VLMs. Finally, we analyze VLMs' limitations in handling nuanced frames and stance expressions on climate change internet memes.
>
---
#### [new 123] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于Retrieval-Augmented Generation（RAG）的上下文归因任务，旨在解决现有方法依赖耗时计算或人工标注的问题。提出Jensen-Shannon散度驱动的ARC-JSD方法，无需微调即可高效定位关键上下文句子，并通过分析揭示模型内部注意力机制与MLP层的作用。**

- **链接: [http://arxiv.org/pdf/2505.16415v1](http://arxiv.org/pdf/2505.16415v1)**

> **作者:** Ruizhe Li; Chen Chen; Yuchen Hu; Yanjun Gao; Xi Wang; Emine Yilmaz
>
> **备注:** Work in process
>
> **摘要:** Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen-Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models.
>
---
#### [new 124] Semantic Pivots Enable Cross-Lingual Transfer in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）的跨语言迁移能力，旨在解析其工作机制并提升性能。提出词级跨语言翻译任务量化能力，发现模型通过"共现行为"和"语义枢纽行为"学习跨语言映射，通过重构含更多语义枢纽的预训练数据集优化模型，实验验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.16385v1](http://arxiv.org/pdf/2505.16385v1)**

> **作者:** Kaiyu He; Tong Zhou; Yubo Chen; Delai Qiu; Shengping Liu; Kang Liu; Jun Zhao
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Large language models (LLMs) demonstrate remarkable ability in cross-lingual tasks. Understanding how LLMs acquire this ability is crucial for their interpretability. To quantify the cross-lingual ability of LLMs accurately, we propose a Word-Level Cross-Lingual Translation Task. To find how LLMs learn cross-lingual ability, we trace the outputs of LLMs' intermediate layers in the word translation task. We identify and distinguish two distinct behaviors in the forward pass of LLMs: co-occurrence behavior and semantic pivot behavior. We attribute LLMs' two distinct behaviors to the co-occurrence frequency of words and find the semantic pivot from the pre-training dataset. Finally, to apply our findings to improve the cross-lingual ability of LLMs, we reconstruct a semantic pivot-aware pre-training dataset using documents with a high proportion of semantic pivots. Our experiments validate the effectiveness of our approach in enhancing cross-lingual ability. Our research contributes insights into the interpretability of LLMs and offers a method for improving LLMs' cross-lingual ability.
>
---
#### [new 125] TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.6; F.2.2**

- **简介: 该论文提出TRIM方法，针对大型语言模型剪枝任务，解决现有统一稀疏度剪枝导致高稀疏比下性能下降的问题。通过逐行分配动态稀疏度并迭代优化，减少输出质量波动，提升压缩效率。实验显示其在80%稀疏度下显著优于基线，实现更高稳定性与性能（如Qwen2.5-14B perplexity降48%）。**

- **链接: [http://arxiv.org/pdf/2505.16743v1](http://arxiv.org/pdf/2505.16743v1)**

> **作者:** Florentin Beck; William Rudman; Carsten Eickhoff
>
> **摘要:** Large Language Models (LLMs) present significant computational and memory challenges due to their extensive size, making pruning essential for their efficient deployment. Existing one-shot pruning methods often apply uniform sparsity constraints across layers or within each layer, resulting in suboptimal performance, especially at high sparsity ratios. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a novel approach that applies varying sparsity ratios to individual output dimensions (rows) within each layer. TRIM employs an iterative adjustment process guided by quality metrics to optimize dimension-wise sparsity allocation, focusing on reducing variance in quality retention across outputs to preserve critical information. TRIM can be seamlessly integrated with existing layer-wise pruning strategies. Our evaluations on perplexity and zero-shot tasks across diverse LLM families (Qwen2.5, LLaMA-2, and OPT) and sparsity levels demonstrate that TRIM achieves new state-of-the-art results and enhances stability. For instance, at 80% sparsity, TRIM reduces perplexity by 48% for Qwen2.5-14B and over 90% for OPT-13B compared to baseline methods. We conclude that fine-grained, dimension-wise sparsity adaptation is crucial for pushing the limits of extreme LLM compression. Code available at: https://github.com/flobk/TRIM
>
---
#### [new 126] Automated Feedback Loops to Protect Text Simplification with Generative AI from Information Loss
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本简化任务，旨在解决生成AI简化健康文本时关键信息丢失问题。研究通过对比五种方法（添加全部缺失实体/词、排名前3实体或随机添加）修复简化文本，发现添加全部缺失实体效果最佳，但现有工具无法有效排序关键信息。**

- **链接: [http://arxiv.org/pdf/2505.16172v1](http://arxiv.org/pdf/2505.16172v1)**

> **作者:** Abhay Kumara Sri Krishna Nandiraju; Gondy Leroy; David Kauchak; Arif Ahmed
>
> **摘要:** Understanding health information is essential in achieving and maintaining a healthy life. We focus on simplifying health information for better understanding. With the availability of generative AI, the simplification process has become efficient and of reasonable quality, however, the algorithms remove information that may be crucial for comprehension. In this study, we compare generative AI to detect missing information in simplified text, evaluate its importance, and fix the text with the missing information. We collected 50 health information texts and simplified them using gpt-4-0613. We compare five approaches to identify missing elements and regenerate the text by inserting the missing elements. These five approaches involve adding missing entities and missing words in various ways: 1) adding all the missing entities, 2) adding all missing words, 3) adding the top-3 entities ranked by gpt-4-0613, and 4, 5) serving as controls for comparison, adding randomly chosen entities. We use cosine similarity and ROUGE scores to evaluate the semantic similarity and content overlap between the original, simplified, and reconstructed simplified text. We do this for both summaries and full text. Overall, we find that adding missing entities improves the text. Adding all the missing entities resulted in better text regeneration, which was better than adding the top-ranked entities or words, or random words. Current tools can identify these entities, but are not valuable in ranking them.
>
---
#### [new 127] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于提升大语言模型（LLMs）事实性评估的任务，旨在解决现有研究依赖不真实合成数据导致的评估局限。通过提出从表格数据抽样真假事实句及利用问答集生成LLM依赖的真实数据集，构建更挑战性的评测集。研究发现先前结论部分成立，但模型生成数据上的事实编码能力推广困难，为事实性研究提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.16520v1](http://arxiv.org/pdf/2505.16520v1)**

> **作者:** Giovanni Servedio; Alessandro De Bellis; Dario Di Palma; Vito Walter Anelli; Tommaso Di Noia
>
> **摘要:** Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation.
>
---
#### [new 128] SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型行为控制任务，旨在解决开放生成场景下可靠调控LLM行为的挑战。提出SAE-SSV方法：通过稀疏自编码器分离语义特征，识别任务相关子空间，学习约束引导向量实现精准行为调控，在保持生成质量前提下提升控制成功率。**

- **链接: [http://arxiv.org/pdf/2505.16188v1](http://arxiv.org/pdf/2505.16188v1)**

> **作者:** Zirui He; Mingyu Jin; Bo Shen; Ali Payani; Yongfeng Zhang; Mengnan Du
>
> **备注:** 30 pages, 24 figures, 12 tables
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but controlling their behavior reliably remains challenging, especially in open-ended generation settings. This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs)to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and politics polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. Further analysis reveals that a notably small subspace is sufficient for effective steering, enabling more targeted and interpretable interventions.
>
---
#### [new 129] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出多模态扩散模型LLaDA-V，属于多模态任务，旨在改进传统自回归方法的不足，提升视觉语言对齐与数据扩展性。通过整合视觉编码器和MLP连接器，将视觉特征映射到语言空间，实验证明其在多模态任务中性能优于LLaMA3-V等模型，展现扩散模型在多模态领域的潜力。**

- **链接: [http://arxiv.org/pdf/2505.16933v1](http://arxiv.org/pdf/2505.16933v1)**

> **作者:** Zebin You; Shen Nie; Xiaolu Zhang; Jun Hu; Jun Zhou; Zhiwu Lu; Ji-Rong Wen; Chongxuan Li
>
> **摘要:** In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large Language Model (MLLM) that integrates visual instruction tuning with masked diffusion models, representing a departure from the autoregressive paradigms dominant in current multimodal approaches. Built upon LLaDA, a representative large language diffusion model, LLaDA-V incorporates a vision encoder and MLP connector that projects visual features into the language embedding space, enabling effective multimodal alignment. Our empirical investigation reveals several intriguing results: First, LLaDA-V demonstrates promising multimodal performance despite its language model being weaker on purely textual tasks than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal tasks with better data scalability. It also narrows the performance gap to Qwen2-VL, suggesting the effectiveness of its architecture for multimodal tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal understanding compared to existing hybrid autoregressive-diffusion and purely diffusion-based MLLMs. Our findings suggest that large language diffusion models show promise in multimodal contexts and warrant further investigation in future research. Project page and codes: https://ml-gsai.github.io/LLaDA-V-demo/.
>
---
#### [new 130] Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Pixel Reasoner框架，解决视觉语言模型（VLMs）在视觉密集任务中因依赖文本推理导致的局限。通过引入像素空间推理操作（如放大、选帧）及两阶段训练（指令调优+好奇心驱动强化学习），提升模型直接分析视觉信息的能力，在多个基准测试中达开源模型最佳精度。任务为视觉推理，核心是增强VLMs的像素级推理能力。**

- **链接: [http://arxiv.org/pdf/2505.15966v1](http://arxiv.org/pdf/2505.15966v1)**

> **作者:** Alex Su; Haozhe Wang; Weimin Ren; Fangzhen Lin; Wenhu Chen
>
> **备注:** Haozhe Wang and Alex Su contributed equally and listed alphabetically
>
> **摘要:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
>
---
#### [new 131] Dynamic Sampling that Adapts: Iterative DPO for Self-Aware Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 论文提出SAI-DPO算法，针对静态数据选择策略无法适应模型动态训练的问题。该算法通过实时评估模型阶段能力，动态调整数据选择，提升数学推理任务效率与性能。实验显示平均提升21.3%，尤其在AIME24和AMC23上显著改善。**

- **链接: [http://arxiv.org/pdf/2505.16176v1](http://arxiv.org/pdf/2505.16176v1)**

> **作者:** Jun Rao; Xuebo Liu; Hexuan Deng; Zepeng Lin; Zixiong Yu; Jiansheng Wei; Xiaojun Meng; Min Zhang
>
> **摘要:** In the realm of data selection for reasoning tasks, existing approaches predominantly rely on externally predefined static metrics such as difficulty and diversity, which are often designed for supervised fine-tuning (SFT) and lack adaptability to continuous training processes. A critical limitation of these methods is their inability to dynamically align with the evolving capabilities of models during online training, a gap that becomes increasingly pronounced with the rise of dynamic training paradigms and online reinforcement learning (RL) frameworks (e.g., R1 models). To address this, we introduce SAI-DPO, an algorithm that dynamically selects training data by continuously assessing a model's stage-specific reasoning abilities across different training phases. By integrating real-time model performance feedback, SAI-DPO adaptively adapts data selection to the evolving strengths and weaknesses of the model, thus enhancing both data utilization efficiency and final task performance. Extensive experiments on three state-of-the-art models and eight mathematical reasoning benchmarks, including challenging competition-level datasets (e.g., AIME24 and AMC23), demonstrate that SAI-DPO achieves an average performance boost of up to 21.3 percentage points, with particularly notable improvements of 10 and 15 points on AIME24 and AMC23, respectively. These results highlight the superiority of dynamic, model-adaptive data selection over static, externally defined strategies in advancing reasoning.
>
---
#### [new 132] ViQAgent: Zero-Shot Video Question Answering via Agent with Open-Vocabulary Grounding Validation
- **分类: cs.CV; cs.CL; I.4.8**

- **简介: 该论文提出ViQAgent，用于零样本视频问答任务，解决长期物体追踪与推理决策对齐问题。结合思维链框架与YOLO-World增强视觉接地，通过时间帧交叉验证提升准确性，实现视频理解新SOTA，支持多领域应用。**

- **链接: [http://arxiv.org/pdf/2505.15928v1](http://arxiv.org/pdf/2505.15928v1)**

> **作者:** Tony Montes; Fernando Lozano
>
> **摘要:** Recent advancements in Video Question Answering (VideoQA) have introduced LLM-based agents, modular frameworks, and procedural solutions, yielding promising results. These systems use dynamic agents and memory-based mechanisms to break down complex tasks and refine answers. However, significant improvements remain in tracking objects for grounding over time and decision-making based on reasoning to better align object references with language model outputs, as newer models get better at both tasks. This work presents an LLM-brained agent for zero-shot Video Question Answering (VideoQA) that combines a Chain-of-Thought framework with grounding reasoning alongside YOLO-World to enhance object tracking and alignment. This approach establishes a new state-of-the-art in VideoQA and Video Understanding, showing enhanced performance on NExT-QA, iVQA, and ActivityNet-QA benchmarks. Our framework also enables cross-checking of grounding timeframes, improving accuracy and providing valuable support for verification and increased output reliability across multiple video domains. The code is available at https://github.com/t-montes/viqagent.
>
---
#### [new 133] CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于大模型安全任务，旨在解决有害微调攻击问题。提出CTRAP机制，通过在模型对齐阶段嵌入陷阱，当检测到恶意微调时触发模型核心能力崩溃，阻止其被滥用，而正常微调不受影响。实验证明其有效且不影响合法使用。**

- **链接: [http://arxiv.org/pdf/2505.16559v1](http://arxiv.org/pdf/2505.16559v1)**

> **作者:** Biao Yi; Tiansheng Huang; Baolei Zhang; Tong Li; Lihai Nie; Zheli Liu; Li Shen
>
> **摘要:** Fine-tuning-as-a-service, while commercially successful for Large Language Model (LLM) providers, exposes models to harmful fine-tuning attacks. As a widely explored defense paradigm against such attacks, unlearning attempts to remove malicious knowledge from LLMs, thereby essentially preventing them from being used to perform malicious tasks. However, we highlight a critical flaw: the powerful general adaptability of LLMs allows them to easily bypass selective unlearning by rapidly relearning or repurposing their capabilities for harmful tasks. To address this fundamental limitation, we propose a paradigm shift: instead of selective removal, we advocate for inducing model collapse--effectively forcing the model to "unlearn everything"--specifically in response to updates characteristic of malicious adaptation. This collapse directly neutralizes the very general capabilities that attackers exploit, tackling the core issue unaddressed by selective unlearning. We introduce the Collapse Trap (CTRAP) as a practical mechanism to implement this concept conditionally. Embedded during alignment, CTRAP pre-configures the model's reaction to subsequent fine-tuning dynamics. If updates during fine-tuning constitute a persistent attempt to reverse safety alignment, the pre-configured trap triggers a progressive degradation of the model's core language modeling abilities, ultimately rendering it inert and useless for the attacker. Crucially, this collapse mechanism remains dormant during benign fine-tuning, ensuring the model's utility and general capabilities are preserved for legitimate users. Extensive empirical results demonstrate that CTRAP effectively counters harmful fine-tuning risks across various LLMs and attack settings, while maintaining high performance in benign scenarios. Our code is available at https://anonymous.4open.science/r/CTRAP.
>
---
#### [new 134] Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉任务，解决MLLM在多帧空间理解不足的问题。提出Multi-SpatialMLLM框架，整合深度感知、视觉对应与动态感知，构建2700万样本的MultiSPA数据集及基准测试，提升多帧推理能力，支持机器人等应用。**

- **链接: [http://arxiv.org/pdf/2505.17015v1](http://arxiv.org/pdf/2505.17015v1)**

> **作者:** Runsen Xu; Weiyao Wang; Hao Tang; Xingyu Chen; Xiaodong Wang; Fu-Jen Chu; Dahua Lin; Matt Feiszli; Kevin J. Liang
>
> **备注:** 24 pages. An MLLM, dataset, and benchmark for multi-frame spatial understanding. Project page: https://runsenxu.com/projects/Multi-SpatialMLLM
>
> **摘要:** Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for robotics and other real-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with robust multi-frame spatial understanding by integrating depth perception, visual correspondence, and dynamic perception. Central to our approach is the MultiSPA dataset, a novel, large-scale collection of more than 27 million samples spanning diverse 3D and 4D scenes. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable, generalizable multi-frame reasoning. We further observe multi-task benefits and early indications of emergent capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics.
>
---
#### [new 135] DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出DuFFin框架，解决大语言模型黑盒环境下知识产权保护问题。针对现有方法干扰生成或依赖白盒访问的缺陷，其通过提取触发模式和知识指纹验证模型来源，在多种变体模型上实现IP-ROC>0.95的高准确率。**

- **链接: [http://arxiv.org/pdf/2505.16530v1](http://arxiv.org/pdf/2505.16530v1)**

> **作者:** Yuliang Yan; Haochun Tang; Shuo Yan; Enyan Dai
>
> **摘要:** Large language models (LLMs) are considered valuable Intellectual Properties (IP) for legitimate owners due to the enormous computational cost of training. It is crucial to protect the IP of LLMs from malicious stealing or unauthorized deployment. Despite existing efforts in watermarking and fingerprinting LLMs, these methods either impact the text generation process or are limited in white-box access to the suspect model, making them impractical. Hence, we propose DuFFin, a novel $\textbf{Du}$al-Level $\textbf{Fin}$gerprinting $\textbf{F}$ramework for black-box setting ownership verification. DuFFin extracts the trigger pattern and the knowledge-level fingerprints to identify the source of a suspect model. We conduct experiments on a variety of models collected from the open-source website, including four popular base models as protected LLMs and their fine-tuning, quantization, and safety alignment versions, which are released by large companies, start-ups, and individual users. Results show that our method can accurately verify the copyright of the base protected LLM on their model variants, achieving the IP-ROC metric greater than 0.95. Our code is available at https://github.com/yuliangyan0807/llm-fingerprint.
>
---
#### [new 136] NovelSeek: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出NovelSeek框架，属于自主科学研究（ASR）任务，旨在通过闭环多智能体系统提升科研效率与跨领域适应性。解决传统科研中效率低、创新慢及人机协作不足的问题。工作包括构建可扩展（跨12领域）、交互式（支持专家反馈）和高效（快速提升模型性能）的系统，实验显示其在反应预测、活性预测等任务中显著缩短时间并提升精度。**

- **链接: [http://arxiv.org/pdf/2505.16938v1](http://arxiv.org/pdf/2505.16938v1)**

> **作者:** NovelSeek Team; Bo Zhang; Shiyang Feng; Xiangchao Yan; Jiakang Yuan; Zhiyin Yu; Xiaohan He; Songtao Huang; Shaowei Hou; Zheng Nie; Zhilong Wang; Jinyao Liu; Runmin Ma; Tianshuo Peng; Peng Ye; Dongzhan Zhou; Shufei Zhang; Xiaosong Wang; Yilan Zhang; Meng Li; Zhongying Tu; Xiangyu Yue; Wangli Ouyang; Bowen Zhou; Lei Bai
>
> **备注:** HomePage: https://alpha-innovator.github.io/NovelSeek-project-page
>
> **摘要:** Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce NovelSeek, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. NovelSeek highlights three key advantages: 1) Scalability: NovelSeek has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: NovelSeek provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: NovelSeek has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.52 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours.
>
---
#### [new 137] AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学与代码推理任务，旨在通过大规模强化学习（RL）提升中小模型推理能力，解决现有蒸馏方法效果局限及训练细节缺失问题。提出分阶段RL策略（先数学后代码），结合优化数据管道和课程学习，显著提升模型在数学（如AIME+14.6%/+17.2%）和代码（如LiveCodeBench+6.8%/+5.8%）任务表现，超越蒸馏基线。**

- **链接: [http://arxiv.org/pdf/2505.16400v1](http://arxiv.org/pdf/2505.16400v1)**

> **作者:** Yang Chen; Zhuolin Yang; Zihan Liu; Chankyu Lee; Peng Xu; Mohammad Shoeybi; Bryan Catanzaro; Wei Ping
>
> **备注:** We release the model at: https://huggingface.co/nvidia/AceReason-Nemotron-14B
>
> **摘要:** Despite recent progress in large-scale reinforcement learning (RL) for reasoning, the training recipe for building high-performing reasoning models remains elusive. Key implementation details of frontier models, such as DeepSeek-R1, including data curation strategies and RL training recipe, are often omitted. Moreover, recent research indicates distillation remains more effective than RL for smaller models. In this work, we demonstrate that large-scale RL can significantly enhance the reasoning capabilities of strong, small- and mid-sized models, achieving results that surpass those of state-of-the-art distillation-based models. We systematically study the RL training process through extensive ablations and propose a simple yet effective approach: first training on math-only prompts, then on code-only prompts. Notably, we find that math-only RL not only significantly enhances the performance of strong distilled models on math benchmarks (e.g., +14.6% / +17.2% on AIME 2025 for the 7B / 14B models), but also code reasoning tasks (e.g., +6.8% / +5.8% on LiveCodeBench for the 7B / 14B models). In addition, extended code-only RL iterations further improve performance on code benchmarks with minimal or no degradation in math results. We develop a robust data curation pipeline to collect challenging prompts with high-quality, verifiable answers and test cases to enable verification-based RL across both domains. Finally, we identify key experimental insights, including curriculum learning with progressively increasing response lengths and the stabilizing effect of on-policy parameter updates. We find that RL not only elicits the foundational reasoning capabilities acquired during pretraining and supervised fine-tuning (e.g., distillation), but also pushes the limits of the model's reasoning ability, enabling it to solve problems that were previously unsolvable.
>
---
#### [new 138] InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于RAG代理信息检索评估任务，针对现有基准静态环境、固定语料及简单查询无法评估动态代理行为的问题，提出InfoDeepSeek基准，通过设计确定性、难度和多样性的复杂问题及开发细粒度评估框架，评估代理在动态网络中的信息检索效果，揭示代理行为并提供研究见解。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15872v1](http://arxiv.org/pdf/2505.15872v1)**

> **作者:** Yunjia Xi; Jianghao Lin; Menghui Zhu; Yongzhao Xiao; Zhuoying Ou; Jiaqi Liu; Tong Wan; Bo Chen; Weiwen Liu; Yasheng Wang; Ruiming Tang; Weinan Zhang; Yong Yu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding responses with retrieved information. As an emerging paradigm, Agentic RAG further enhances this process by introducing autonomous LLM agents into the information seeking process. However, existing benchmarks fall short in evaluating such systems, as they are confined to a static retrieval environment with a fixed, limited corpus} and simple queries that fail to elicit agentic behavior. Moreover, their evaluation protocols assess information seeking effectiveness by pre-defined gold sets of documents, making them unsuitable for the open-ended and dynamic nature of real-world web environments. To bridge this gap, we present InfoDeepSeek, a new benchmark with challenging questions designed for assessing agentic information seeking in real-world, dynamic web environments. We propose a systematic methodology for constructing challenging queries satisfying the criteria of determinacy, difficulty, and diversity. Based on this, we develop the first evaluation framework tailored to dynamic agentic information seeking, including fine-grained metrics about the accuracy, utility, and compactness of information seeking outcomes. Through extensive experiments across LLMs, search engines, and question types, InfoDeepSeek reveals nuanced agent behaviors and offers actionable insights for future research.
>
---
#### [new 139] $\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决现有大推荐模型依赖外部LLM进行推理导致的资源浪费与优化不足问题。提出R²ec模型，通过统一架构实现推荐与推理能力的交织，并设计RecPO强化学习框架，利用融合奖励方案仅凭推荐标签同步优化两者，提升Hit@5和NDCG@20指标。**

- **链接: [http://arxiv.org/pdf/2505.16994v1](http://arxiv.org/pdf/2505.16994v1)**

> **作者:** Runyang You; Yongqi Li; Xinyu Lin; Xin Zhang; Wenjie Wang; Wenjie Li; Liqiang Nie
>
> **摘要:** Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. Current studies usually position LLMs as external reasoning modules to yield auxiliary thought for augmenting conventional recommendation pipelines. However, such decoupled designs are limited in significant resource cost and suboptimal joint optimization. To address these issues, we propose \name, a unified large recommender model with intrinsic reasoning capabilities. Initially, we reconceptualize the model architecture to facilitate interleaved reasoning and recommendation in the autoregressive process. Subsequently, we propose RecPO, a corresponding reinforcement learning framework that optimizes \name\ both the reasoning and recommendation capabilities simultaneously in a single policy update; RecPO introduces a fused reward scheme that solely leverages recommendation labels to simulate the reasoning capability, eliminating dependency on specialized reasoning annotations. Experiments on three datasets with various baselines verify the effectiveness of \name, showing relative improvements of 68.67\% in Hit@5 and 45.21\% in NDCG@20. Code available at https://github.com/YRYangang/RRec.
>
---
#### [new 140] BioDSA-1K: Benchmarking Data Science Agents for Biomedical Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出BioDSA-1K基准，用于评估AI代理在生物医学假设验证中的能力。针对AI难以处理复杂数据与证据解释的问题，构建了包含1,029个任务和1,177个分析计划的评测集，覆盖假设决策、证据一致性、推理正确性及代码可执行性四个维度，并纳入数据不足的非验证性案例，推动可信生物医学AI研发。**

- **链接: [http://arxiv.org/pdf/2505.16100v1](http://arxiv.org/pdf/2505.16100v1)**

> **作者:** Zifeng Wang; Benjamin Danek; Jimeng Sun
>
> **摘要:** Validating scientific hypotheses is a central challenge in biomedical research, and remains difficult for artificial intelligence (AI) agents due to the complexity of real-world data analysis and evidence interpretation. In this work, we present BioDSA-1K, a benchmark designed to evaluate AI agents on realistic, data-driven biomedical hypothesis validation tasks. BioDSA-1K consists of 1,029 hypothesis-centric tasks paired with 1,177 analysis plans, curated from over 300 published biomedical studies to reflect the structure and reasoning found in authentic research workflows. Each task includes a structured hypothesis derived from the original study's conclusions, expressed in the affirmative to reflect the language of scientific reporting, and one or more pieces of supporting evidence grounded in empirical data tables. While these hypotheses mirror published claims, they remain testable using standard statistical or machine learning methods. The benchmark enables evaluation along four axes: (1) hypothesis decision accuracy, (2) alignment between evidence and conclusion, (3) correctness of the reasoning process, and (4) executability of the AI-generated analysis code. Importantly, BioDSA-1K includes non-verifiable hypotheses: cases where the available data are insufficient to support or refute a claim, reflecting a common yet underexplored scenario in real-world science. We propose BioDSA-1K as a foundation for building and evaluating generalizable, trustworthy AI agents for biomedical discovery.
>
---
#### [new 141] Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于提升大模型推理效率的任务，旨在解决其过度思考产生冗余内容的问题。提出ACPO框架，通过双思维模式切换与任务难度动态适配，结合监督微调和强化学习优化，实现高效混合推理。**

- **链接: [http://arxiv.org/pdf/2505.16315v1](http://arxiv.org/pdf/2505.16315v1)**

> **作者:** Xiaoxue Cheng; Junyi Li; Zhenduo Zhang; Xinyu Tang; Wayne Xin Zhao; Xinyu Kong; Zhiqiang Zhang
>
> **备注:** work in progress
>
> **摘要:** Large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks, but often suffer from overthinking, generating redundant content regardless of task difficulty. Inspired by the dual process theory in cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a reinforcement learning framework that enables LRMs to achieve efficient reasoning through adaptive cognitive allocation and dynamic system switch. ACPO incorporates two key components: (1) introducing system-aware reasoning tokens to explicitly represent the thinking modes thereby making the model's cognitive process transparent, and (2) integrating online difficulty estimation and token length budget to guide adaptive system switch and reasoning during reinforcement learning. To this end, we propose a two-stage training strategy. The first stage begins with supervised fine-tuning to cold start the model, enabling it to generate reasoning paths with explicit thinking modes. In the second stage, we apply ACPO to further enhance adaptive system switch for difficulty-aware reasoning. Experimental results demonstrate that ACPO effectively reduces redundant reasoning while adaptively adjusting cognitive allocation based on task complexity, achieving efficient hybrid reasoning.
>
---
#### [new 142] UFT: Unifying Supervised and Reinforcement Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出UFT方法，整合监督与强化微调，解决两者过拟合及依赖基模型的局限，提升大模型推理能力，理论证明其突破RFT的样本复杂度瓶颈，加速收敛。**

- **链接: [http://arxiv.org/pdf/2505.16984v1](http://arxiv.org/pdf/2505.16984v1)**

> **作者:** Mingyang Liu; Gabriele Farina; Asuman Ozdaglar
>
> **摘要:** Post-training has demonstrated its importance in enhancing the reasoning capabilities of large language models (LLMs). The primary post-training methods can be categorized into supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). SFT is efficient and well-suited for small language models, but it may lead to overfitting and limit the reasoning abilities of larger models. In contrast, RFT generally yields better generalization but depends heavily on the strength of the base model. To address the limitations of SFT and RFT, we propose Unified Fine-Tuning (UFT), a novel post-training paradigm that unifies SFT and RFT into a single, integrated process. UFT enables the model to effectively explore solutions while incorporating informative supervision signals, bridging the gap between memorizing and thinking underlying existing methods. Notably, UFT outperforms both SFT and RFT in general, regardless of model sizes. Furthermore, we theoretically prove that UFT breaks RFT's inherent exponential sample complexity bottleneck, showing for the first time that unified training can exponentially accelerate convergence on long-horizon reasoning tasks.
>
---
#### [new 143] Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统优化任务，针对基于LLM的协作系统在复杂软件开发任务中的效率问题。提出通过文本反馈驱动的两步优化方法：先定位低效代理并分析其失败原因，再优化其提示策略；对比不同优化模式（在线/离线、个体/群体）及多轮提示策略，验证方法有效性并分析系统行为影响。**

- **链接: [http://arxiv.org/pdf/2505.16086v1](http://arxiv.org/pdf/2505.16086v1)**

> **作者:** Ming Shen; Raphael Shu; Anurag Pratik; James Gung; Yubin Ge; Monica Sunkara; Yi Zhang
>
> **摘要:** We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
>
---
#### [new 144] How do Scaling Laws Apply to Knowledge Graph Engineering Tasks? The Impact of Model Size on Large Language Model Performance
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于知识图谱工程（KGE）任务，研究模型规模对大语言模型（LLMs）性能的影响。旨在解决如何平衡模型性能与资源成本的问题，通过LLM-KG-Bench框架评估26个LLMs，分析不同规模模型在KGE任务中的表现趋势，发现模型规模通常提升性能，但存在 plateau效应及同家族大模型表现下降的局部情况，建议测试相近规模模型以优化成本效益。**

- **链接: [http://arxiv.org/pdf/2505.16276v1](http://arxiv.org/pdf/2505.16276v1)**

> **作者:** Desiree Heim; Lars-Peter Meyer; Markus Schröder; Johannes Frey; Andreas Dengel
>
> **备注:** Peer reviewed and to appear in the ESWC 2025 Workshops and Tutorials Joint Proceedings (Workshop on Evaluation of Language Models in Knowledge Engineering [ELMKE])
>
> **摘要:** When using Large Language Models (LLMs) to support Knowledge Graph Engineering (KGE), one of the first indications when searching for an appropriate model is its size. According to the scaling laws, larger models typically show higher capabilities. However, in practice, resource costs are also an important factor and thus it makes sense to consider the ratio between model performance and costs. The LLM-KG-Bench framework enables the comparison of LLMs in the context of KGE tasks and assesses their capabilities of understanding and producing KGs and KG queries. Based on a dataset created in an LLM-KG-Bench run covering 26 open state-of-the-art LLMs, we explore the model size scaling laws specific to KGE tasks. In our analyses, we assess how benchmark scores evolve between different model size categories. Additionally, we inspect how the general score development of single models and families of models correlates to their size. Our analyses revealed that, with a few exceptions, the model size scaling laws generally also apply to the selected KGE tasks. However, in some cases, plateau or ceiling effects occurred, i.e., the task performance did not change much between a model and the next larger model. In these cases, smaller models could be considered to achieve high cost-effectiveness. Regarding models of the same family, sometimes larger models performed worse than smaller models of the same family. These effects occurred only locally. Hence it is advisable to additionally test the next smallest and largest model of the same family.
>
---
#### [new 145] NQKV: A KV Cache Quantization Scheme Based on Normal Distribution Characteristics
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）部署优化任务，旨在解决KV缓存内存消耗过大的问题。通过分析KV缓存元素符合正态分布的特性，提出NQKV算法，采用块级分位数量化方法，在保持输出质量前提下将KV缓存压缩至更低比特，使OPT模型实现批处理量翻倍/上下文长度增加4倍，吞吐量提升9.3倍。**

- **链接: [http://arxiv.org/pdf/2505.16210v1](http://arxiv.org/pdf/2505.16210v1)**

> **作者:** Zhihang Cai; Xingjun Zhang; Zhendong Tan; Zheng Wei
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency across a wide range of tasks. However, LLMs often require larger batch sizes to enhance throughput or longer context lengths to meet task demands, which significantly increases the memory resource consumption of the Key-Value (KV) cache during inference, becoming a major bottleneck in LLM deployment. To address this issue, quantization is a common and straightforward approach. Currently, quantization methods for activations are limited to 8-bit, and quantization to even lower bits can lead to substantial accuracy drops. To further save space by quantizing the KV cache to even lower bits, we analyzed the element distribution of the KV cache and designed the NQKV algorithm. Since the elements within each block of the KV cache follow a normal distribution, NQKV employs per-block quantile quantization to achieve information-theoretically optimal quantization error. Without significantly compromising model output quality, NQKV enables the OPT model to perform inference with an 2x larger batch size or a 4x longer context length, and it improves throughput by 9.3x compared to when the KV cache is not used.
>
---
#### [new 146] SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于大模型安全任务，解决其对未知有害查询和绕过攻击泛化不足的问题。通过发现"安全顿悟时刻"出现在关键句，提出SafeKey方法：包含双路径安全头和查询掩码建模，增强安全信号与注意力，显著降低有害响应率。**

- **链接: [http://arxiv.org/pdf/2505.16186v1](http://arxiv.org/pdf/2505.16186v1)**

> **作者:** Kaiwen Zhou; Xuandong Zhao; Gaowen Liu; Jayanth Srinivasa; Aosong Feng; Dawn Song; Xin Eric Wang
>
> **摘要:** Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.
>
---
#### [new 147] GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉生成任务，解决复杂文本提示（含多对象、精确空间关系）生成图像的困难。提出GoT-R1框架，结合强化学习与生成式思维链，设计双阶段多维奖励机制，通过MLLM评估推理过程与输出，提升语义、空间准确性及视觉质量，实验显示显著提升。**

- **链接: [http://arxiv.org/pdf/2505.17022v1](http://arxiv.org/pdf/2505.17022v1)**

> **作者:** Chengqi Duan; Rongyao Fang; Yuqing Wang; Kun Wang; Linjiang Huang; Xingyu Zeng; Hongsheng Li; Xihui Liu
>
> **备注:** Github page refer to: https://github.com/gogoduan/GoT-R1
>
> **摘要:** Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at https://github.com/gogoduan/GoT-R1.
>
---
#### [new 148] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出AudioTrust框架，用于评估音频大语言模型（ALLMs）的多维度可信度。针对现有评估方法忽视音频特有风险（如隐私、鲁棒性）的问题，构建含6个评估维度、18种实验设置及4420个真实场景样本的基准测试，并设计9项音频专用指标。实验揭示了当前模型在高风险场景中的局限性，助力安全部署。**

- **链接: [http://arxiv.org/pdf/2505.16211v1](http://arxiv.org/pdf/2505.16211v1)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Zhe Wang; Xingjian Du; Shun Zhang; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Xiaojun Jia; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Haoyang Li; Yiming Li; Xiaobin Zhuang; Yang Liu; Haibo Hu; Zhuo Chen; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; XiaoFeng Wang; Wenyuan Xu; Wei Dong; Xinfeng Li
>
> **备注:** Technical Report
>
> **摘要:** The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at https://github.com/JusperLee/AudioTrust.
>
---
#### [new 149] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于大型音频语言模型（LALM）评估任务，旨在解决现有评估方法碎片化、缺乏系统分类的问题。提出四维评估分类法（听觉处理、知识推理、对话能力、公平安全），总结挑战并指明未来方向，同时公开论文集支持研究。**

- **链接: [http://arxiv.org/pdf/2505.15957v1](http://arxiv.org/pdf/2505.15957v1)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** Project Website: https://github.com/b08202033/LALM-Evaluation-Survey
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [new 150] AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentIF，首个评估大语言模型（LLMs）在代理场景中遵循复杂指令的基准。针对现有模型在长指令（平均1723词）和多约束（平均11.9个/指令）下表现差的问题，基于50个真实任务构建数据集，通过多方法评估模型并分析其失败模式。**

- **链接: [http://arxiv.org/pdf/2505.16944v1](http://arxiv.org/pdf/2505.16944v1)**

> **作者:** Yunjia Qi; Hao Peng; Xiaozhi Wang; Amy Xin; Youfeng Liu; Bin Xu; Lei Hou; Juanzi Li
>
> **摘要:** Large Language Models (LLMs) have demonstrated advanced capabilities in real-world agentic applications. Growing research efforts aim to develop LLM-based agents to address practical demands, introducing a new challenge: agentic scenarios often involve lengthy instructions with complex constraints, such as extended system prompts and detailed tool specifications. While adherence to such instructions is crucial for agentic applications, whether LLMs can reliably follow them remains underexplored. In this paper, we introduce AgentIF, the first benchmark for systematically evaluating LLM instruction following ability in agentic scenarios. AgentIF features three key characteristics: (1) Realistic, constructed from 50 real-world agentic applications. (2) Long, averaging 1,723 words with a maximum of 15,630 words. (3) Complex, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. To construct AgentIF, we collect 707 human-annotated instructions across 50 agentic tasks from industrial application agents and open-source agentic systems. For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation. We use AgentIF to systematically evaluate existing advanced LLMs. We observe that current models generally perform poorly, especially in handling complex constraint structures and tool specifications. We further conduct error analysis and analytical experiments on instruction length and meta constraints, providing some findings about the failure modes of existing LLMs. We have released the code and data to facilitate future research.
>
---
#### [new 151] The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
- **分类: cs.LG; cs.AI; cs.CL; cs.NA; math.NA; math.OC**

- **简介: 该论文提出Polar Express算法，针对深度学习中高效计算矩阵极分解的任务。解决传统方法（如牛顿-施 ulz）收敛慢或依赖GPU不友好的操作（如QR分解）的问题。通过优化多项式迭代规则并确保最优性，实现快速收敛与bfloat16精度下的稳定性，提升Muon优化框架在大规模模型（如GPT-2）中的性能。**

- **链接: [http://arxiv.org/pdf/2505.16932v1](http://arxiv.org/pdf/2505.16932v1)**

> **作者:** Noah Amsel; David Persson; Christopher Musco; Robert Gower
>
> **摘要:** Computing the polar decomposition and the related matrix sign function, has been a well-studied problem in numerical analysis for decades. More recently, it has emerged as an important subroutine in deep learning, particularly within the Muon optimization framework. However, the requirements in this setting differ significantly from those of traditional numerical analysis. In deep learning, methods must be highly efficient and GPU-compatible, but high accuracy is often unnecessary. As a result, classical algorithms like Newton-Schulz (which suffers from slow initial convergence) and methods based on rational functions (which rely on QR decompositions or matrix inverses) are poorly suited to this context. In this work, we introduce Polar Express, a GPU-friendly algorithm for computing the polar decomposition. Like classical polynomial methods such as Newton-Schulz, our approach uses only matrix-matrix multiplications, making it GPU-compatible. Motivated by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the polynomial update rule at each iteration by solving a minimax optimization problem, and we prove that it enjoys a strong worst-case optimality guarantee. This property ensures both rapid early convergence and fast asymptotic convergence. We also address finite-precision issues, making it stable in bfloat16 in practice. We apply Polar Express within the Muon optimization framework and show consistent improvements in validation loss on large-scale models such as GPT-2, outperforming recent alternatives across a range of learning rates.
>
---
#### [new 152] OViP: Online Vision-Language Preference Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型幻觉抑制任务，旨在解决现有方法依赖低效负面样本导致的训练效果差问题。提出OViP框架，通过动态分析模型自身生成的幻觉输出，结合扩散模型合成负面图像，实时生成对比数据优化模型，同时改进评估协议。实验表明其有效降低幻觉并保持多模态能力。**

- **链接: [http://arxiv.org/pdf/2505.15963v1](http://arxiv.org/pdf/2505.15963v1)**

> **作者:** Shujun Liu; Siyuan Wang; Zejun Li; Jianxiang Wang; Cheng Zeng; Zhongyu Wei
>
> **备注:** 22 pages, 10 figures, 8 tables
>
> **摘要:** Large vision-language models (LVLMs) remain vulnerable to hallucination, often generating content misaligned with visual inputs. While recent approaches advance multi-modal Direct Preference Optimization (DPO) to mitigate hallucination, they typically rely on predefined or randomly edited negative samples that fail to reflect actual model errors, limiting training efficacy. In this work, we propose an Online Vision-language Preference Learning (OViP) framework that dynamically constructs contrastive training data based on the model's own hallucinated outputs. By identifying semantic differences between sampled response pairs and synthesizing negative images using a diffusion model, OViP generates more relevant supervision signals in real time. This failure-driven training enables adaptive alignment of both textual and visual preferences. Moreover, we refine existing evaluation protocols to better capture the trade-off between hallucination suppression and expressiveness. Experiments on hallucination and general benchmarks demonstrate that OViP effectively reduces hallucinations while preserving core multi-modal capabilities.
>
---
#### [new 153] Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型可解释性评估任务，解决稀疏自编码器（SAE）概念表示对输入扰动鲁棒性不足的问题。通过构建对抗扰动优化框架，量化概念表示的稳定性，发现微小扰动即可操控解释结果，证明SAE表示脆弱，不适用于模型监控等安全场景。**

- **链接: [http://arxiv.org/pdf/2505.16004v1](http://arxiv.org/pdf/2505.16004v1)**

> **作者:** Aaron J. Li; Suraj Srinivas; Usha Bhalla; Himabindu Lakkaraju
>
> **摘要:** Sparse autoencoders (SAEs) are commonly used to interpret the internal activations of large language models (LLMs) by mapping them to human-interpretable concept representations. While existing evaluations of SAEs focus on metrics such as the reconstruction-sparsity tradeoff, human (auto-)interpretability, and feature disentanglement, they overlook a critical aspect: the robustness of concept representations to input perturbations. We argue that robustness must be a fundamental consideration for concept representations, reflecting the fidelity of concept labeling. To this end, we formulate robustness quantification as input-space optimization problems and develop a comprehensive evaluation framework featuring realistic scenarios in which adversarial perturbations are crafted to manipulate SAE representations. Empirically, we find that tiny adversarial input perturbations can effectively manipulate concept-based interpretations in most scenarios without notably affecting the outputs of the base LLMs themselves. Overall, our results suggest that SAE concept representations are fragile and may be ill-suited for applications in model monitoring and oversight.
>
---
#### [new 154] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像检索任务，旨在解决现有模型在属性聚焦查询中因全局嵌入忽略细节导致性能不足的问题。团队构建COCO-Facet基准测试，发现CLIP和MLLM模型表现不佳，提出通过可提示嵌入高亮关键属性，并设计加速策略提升实用性。**

- **链接: [http://arxiv.org/pdf/2505.15877v1](http://arxiv.org/pdf/2505.15877v1)**

> **作者:** Siting Li; Xiang Gao; Simon Shaolei Du
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** While an image is worth more than a thousand words, only a few provide crucial information for a given task and thus should be focused on. In light of this, ideal text-to-image (T2I) retrievers should prioritize specific visual attributes relevant to queries. To evaluate current retrievers on handling attribute-focused queries, we build COCO-Facet, a COCO-based benchmark with 9,112 queries about diverse attributes of interest. We find that CLIP-like retrievers, which are widely adopted due to their efficiency and zero-shot ability, have poor and imbalanced performance, possibly because their image embeddings focus on global semantics and subjects while leaving out other details. Notably, we reveal that even recent Multimodal Large Language Model (MLLM)-based, stronger retrievers with a larger output dimension struggle with this limitation. Hence, we hypothesize that retrieving with general image embeddings is suboptimal for performing such queries. As a solution, we propose to use promptable image embeddings enabled by these multimodal retrievers, which boost performance by highlighting required attributes. Our pipeline for deriving such embeddings generalizes across query types, image pools, and base retriever architectures. To enhance real-world applicability, we offer two acceleration strategies: Pre-processing promptable embeddings and using linear approximations. We show that the former yields a 15% improvement in Recall@5 when prompts are predefined, while the latter achieves an 8% improvement when prompts are only available during inference.
>
---
#### [new 155] X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出X-MAS框架，解决多智能体系统（MAS）依赖单一LLM导致性能受限的问题。通过异构LLM驱动各代理，构建X-MAS-Bench评估平台，测试27个LLM在5领域及功能的性能，实验显示异构配置较同构提升8.4%-47%，证明异构MAS在提升协作AI系统效能的潜力。**

- **链接: [http://arxiv.org/pdf/2505.16997v1](http://arxiv.org/pdf/2505.16997v1)**

> **作者:** Rui Ye; Xiangrui Liu; Qimin Wu; Xianghe Pang; Zhenfei Yin; Lei Bai; Siheng Chen
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems.
>
---
#### [new 156] Grounding Chest X-Ray Visual Question Answering with Generated Radiology Reports
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出结合生成放射报告的胸片VQA方法，处理单图异常检测与时序差异比较问题。通过两阶段模型（报告生成和答案生成），利用预测报告增强答案生成模块，统一处理两类问题，在Medical-Diff-VQA数据集达SOTA。任务为医学影像问答，解决如何有效融合报告提升模型性能并处理双模式问题。**

- **链接: [http://arxiv.org/pdf/2505.16624v1](http://arxiv.org/pdf/2505.16624v1)**

> **作者:** Francesco Dalla Serra; Patrick Schrempf; Chaoyang Wang; Zaiqiao Meng; Fani Deligianni; Alison Q. O'Neil
>
> **摘要:** We present a novel approach to Chest X-ray (CXR) Visual Question Answering (VQA), addressing both single-image image-difference questions. Single-image questions focus on abnormalities within a specific CXR ("What abnormalities are seen in image X?"), while image-difference questions compare two longitudinal CXRs acquired at different time points ("What are the differences between image X and Y?"). We further explore how the integration of radiology reports can enhance the performance of VQA models. While previous approaches have demonstrated the utility of radiology reports during the pre-training phase, we extend this idea by showing that the reports can also be leveraged as additional input to improve the VQA model's predicted answers. First, we propose a unified method that handles both types of questions and auto-regressively generates the answers. For single-image questions, the model is provided with a single CXR. For image-difference questions, the model is provided with two CXRs from the same patient, captured at different time points, enabling the model to detect and describe temporal changes. Taking inspiration from 'Chain-of-Thought reasoning', we demonstrate that performance on the CXR VQA task can be improved by grounding the answer generator module with a radiology report predicted for the same CXR. In our approach, the VQA model is divided into two steps: i) Report Generation (RG) and ii) Answer Generation (AG). Our results demonstrate that incorporating predicted radiology reports as evidence to the AG model enhances performance on both single-image and image-difference questions, achieving state-of-the-art results on the Medical-Diff-VQA dataset.
>
---
#### [new 157] Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出Aug2Search框架，利用LLM生成合成数据增强Facebook Marketplace的嵌入式检索（EBR）模型。针对平台数据多样性不足的问题，通过生成查询、优化商品描述及二者结合的策略生成合成数据，实验显示其提升EBR模型4% ROC_AUC，证明合成数据单独训练效果优于原始数据。**

- **链接: [http://arxiv.org/pdf/2505.16065v1](http://arxiv.org/pdf/2505.16065v1)**

> **作者:** Ruijie Xi; He Ba; Hao Yuan; Rishu Agrawal; Arul Prakash
>
> **摘要:** Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
>
---
#### [new 158] Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; math.OC**

- **简介: 该论文属于大语言模型（LLMs）安全优化任务，旨在解决微调过程中模型生成有害内容的风险。针对微调（即使使用良性数据）导致安全下降的问题，提出安全感知探测（SAP）框架，通过在梯度传播中嵌入安全探测器识别风险梯度方向，平衡任务性能与安全性。实验表明其有效降低有害输出且保持模型效果。**

- **链接: [http://arxiv.org/pdf/2505.16737v1](http://arxiv.org/pdf/2505.16737v1)**

> **作者:** Chengcan Wu; Zhixin Zhang; Zeming Wei; Yihao Zhang; Meng Sun
>
> **摘要:** The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at https://github.com/ChengcanWu/SAP.
>
---
#### [new 159] Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于减少大型视觉-语言模型（LVLMs）幻觉问题的任务。针对现有方法计算成本高或干预效果不足的问题，提出SSL方法：通过稀疏自编码器识别与幻觉/事实相关的语义方向，精准调整模型表示，无需训练即能有效抑制幻觉，且跨模型适用性好、效率高。**

- **链接: [http://arxiv.org/pdf/2505.16146v1](http://arxiv.org/pdf/2505.16146v1)**

> **作者:** Zhenglin Hua; Jinghan He; Zijun Yao; Tianxu Han; Haiyun Guo; Yuheng Jia; Junfeng Fang
>
> **摘要:** Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks such as visual question answering (VQA) and image captioning. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs' internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with either hallucinations or actuality, realizing more precise and direct hallucination-related representations. Our analysis demonstrates that interventions along the faithful direction we identified can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose Steering LVLMs via SAE Latent Directions (SSL), a training-free method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead.
>
---
#### [new 160] A Survey of Large Language Models for Text-Guided Molecular Discovery: from Molecule Generation to Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于综述任务，探讨大型语言模型（LLMs）在文本引导分子发现中的应用，聚焦分子生成与优化两大任务。通过提出分类法分析技术，总结数据集与评估方法，并讨论挑战与未来方向，为LLMs与分子科学交叉研究提供资源。**

- **链接: [http://arxiv.org/pdf/2505.16094v1](http://arxiv.org/pdf/2505.16094v1)**

> **作者:** Ziqing Wang; Kexin Zhang; Zihan Zhao; Yibo Wen; Abhishek Pandey; Han Liu; Kaize Ding
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) are introducing a paradigm shift in molecular discovery by enabling text-guided interaction with chemical spaces through natural language, symbolic notations, with emerging extensions to incorporate multi-modal inputs. To advance the new field of LLM for molecular discovery, this survey provides an up-to-date and forward-looking review of the emerging use of LLMs for two central tasks: molecule generation and molecule optimization. Based on our proposed taxonomy for both problems, we analyze representative techniques in each category, highlighting how LLM capabilities are leveraged across different learning settings. In addition, we include the commonly used datasets and evaluation protocols. We conclude by discussing key challenges and future directions, positioning this survey as a resource for researchers working at the intersection of LLMs and molecular science. A continuously updated reading list is available at https://github.com/REAL-Lab-NU/Awesome-LLM-Centric-Molecular-Discovery.
>
---
#### [new 161] GRIT: Teaching MLLMs to Think with Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，解决现有模型缺乏视觉信息整合的问题。提出GRIT方法，通过结合文本与边界框坐标生成视觉 grounding 推理链，并采用改进的RL算法GRPO-GR，仅需少量标注数据即可高效训练多模态模型。**

- **链接: [http://arxiv.org/pdf/2505.15879v1](http://arxiv.org/pdf/2505.15879v1)**

> **作者:** Yue Fan; Xuehai He; Diji Yang; Kaizhi Zheng; Ching-Chen Kuo; Yuting Zheng; Sravana Jyothi Narayanaraju; Xinze Guan; Xin Eric Wang
>
> **摘要:** Recent studies have demonstrated the efficacy of using Reinforcement Learning (RL) in building reasoning models that articulate chains of thoughts prior to producing final answers. However, despite ongoing advances that aim at enabling reasoning for vision-language tasks, existing open-source visual reasoning models typically generate reasoning content with pure natural language, lacking explicit integration of visual information. This limits their ability to produce clearly articulated and visually grounded reasoning chains. To this end, we propose Grounded Reasoning with Images and Texts (GRIT), a novel method for training MLLMs to think with images. GRIT introduces a grounded reasoning paradigm, in which models generate reasoning chains that interleave natural language and explicit bounding box coordinates. These coordinates point to regions of the input image that the model consults during its reasoning process. Additionally, GRIT is equipped with a reinforcement learning approach, GRPO-GR, built upon the GRPO algorithm. GRPO-GR employs robust rewards focused on the final answer accuracy and format of the grounded reasoning output, which eliminates the need for data with reasoning chain annotations or explicit bounding box labels. As a result, GRIT achieves exceptional data efficiency, requiring as few as 20 image-question-answer triplets from existing datasets. Comprehensive evaluations demonstrate that GRIT effectively trains MLLMs to produce coherent and visually grounded reasoning chains, showing a successful unification of reasoning and grounding abilities.
>
---
#### [new 162] Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究自回归图像生成中的CoT推理任务，对比DPO与GRPO算法，解决文本-图像一致性、美学优化及奖励模型设计问题。通过评估两算法的领域内性能与泛化能力，分析奖励模型影响，并探索三种扩展策略以提升性能。**

- **链接: [http://arxiv.org/pdf/2505.17017v1](http://arxiv.org/pdf/2505.17017v1)**

> **作者:** Chengzhuo Tong; Ziyu Guo; Renrui Zhang; Wenyu Shan; Xinyu Wei; Zhenghao Xing; Hongsheng Li; Pheng-Ann Heng
>
> **备注:** Code is released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
> **摘要:** Recent advancements underscore the significant role of Reinforcement Learning (RL) in enhancing the Chain-of-Thought (CoT) reasoning capabilities of large language models (LLMs). Two prominent RL algorithms, Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), are central to these developments, showcasing different pros and cons. Autoregressive image generation, also interpretable as a sequential CoT reasoning process, presents unique challenges distinct from LLM-based CoT reasoning. These encompass ensuring text-image consistency, improving image aesthetic quality, and designing sophisticated reward models, rather than relying on simpler rule-based rewards. While recent efforts have extended RL to this domain, these explorations typically lack an in-depth analysis of the domain-specific challenges and the characteristics of different RL strategies. To bridge this gap, we provide the first comprehensive investigation of the GRPO and DPO algorithms in autoregressive image generation, evaluating their in-domain performance and out-of-domain generalization, while scrutinizing the impact of different reward models on their respective capabilities. Our findings reveal that GRPO and DPO exhibit distinct advantages, and crucially, that reward models possessing stronger intrinsic generalization capabilities potentially enhance the generalization potential of the applied RL algorithms. Furthermore, we systematically explore three prevalent scaling strategies to enhance both their in-domain and out-of-domain proficiency, deriving unique insights into efficiently scaling performance for each paradigm. We hope our study paves a new path for inspiring future work on developing more effective RL algorithms to achieve robust CoT reasoning in the realm of autoregressive image generation. Code is released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
---
#### [new 163] Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决训练数据中"假负样本"损害模型性能的问题。通过级联LLM提示识别并重新标记错误标注的负样本，提升检索模型鲁棒性，实验显示在BEIR等数据集上nDCG@10指标显著提升。**

- **链接: [http://arxiv.org/pdf/2505.16967v1](http://arxiv.org/pdf/2505.16967v1)**

> **作者:** Nandan Thakur; Crystina Zhang; Xueguang Ma; Jimmy Lin
>
> **备注:** Code is available at https://github.com/castorini/rlhn & datasets are available at https://huggingface.co/rlhn
>
> **摘要:** Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini.
>
---
#### [new 164] CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark
- **分类: cs.AR; cs.AI; cs.CL; cs.LG; cs.PL**

- **简介: 论文提出CASS，首个跨GPU架构（NVIDIA-AMD）代码转译数据集与模型，解决低级代码移植难题。包含7万验证代码对，训练模型实现95%源码、37.5%汇编翻译准确率，超商业工具；开发CASS-Bench基准测试，开源资源推动编译器与硬件翻译研究。**

- **链接: [http://arxiv.org/pdf/2505.16968v1](http://arxiv.org/pdf/2505.16968v1)**

> **作者:** Ahmed Heakl; Sarim Hashmi; Gustavo Bertolo Stahl; Seung Hun Eddie Han; Salman Khan; Abdulrahman Mahmoud
>
> **备注:** 20 pages, 11 figures, 5 tables
>
> **摘要:** We introduce \texttt{CASS}, the first large-scale dataset and model suite for cross-architecture GPU code transpilation, targeting both source-level (CUDA~$\leftrightarrow$~HIP) and assembly-level (Nvidia SASS~$\leftrightarrow$~AMD RDNA3) translation. The dataset comprises 70k verified code pairs across host and device, addressing a critical gap in low-level GPU code portability. Leveraging this resource, we train the \texttt{CASS} family of domain-specific language models, achieving 95\% source translation accuracy and 37.5\% assembly translation accuracy, substantially outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our generated code matches native performance in over 85\% of test cases, preserving runtime and memory behavior. To support rigorous evaluation, we introduce \texttt{CASS-Bench}, a curated benchmark spanning 16 GPU domains with ground-truth execution. All data, models, and evaluation tools are released as open source to foster progress in GPU compiler tooling, binary compatibility, and LLM-guided hardware translation. Dataset and benchmark are on \href{https://huggingface.co/datasets/MBZUAI/cass}{\textcolor{blue}{HuggingFace}}, with code at \href{https://github.com/GustavoStahl/CASS}{\textcolor{blue}{GitHub}}.
>
---
#### [new 165] R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLM）推理能力提升任务，旨在解决强化学习（RL）中的稀疏奖励和优势消失问题。提出Share-GRPO方法，通过扩展问题空间、共享多样化推理路径及分层优势估计，优化模型推理性能。**

- **链接: [http://arxiv.org/pdf/2505.16673v1](http://arxiv.org/pdf/2505.16673v1)**

> **作者:** Huanjin Yao; Qixiang Yin; Jingyi Zhang; Min Yang; Yibo Wang; Wenhao Wu; Fei Su; Li Shen; Minghui Qiu; Dacheng Tao; Jiaxing Huang
>
> **备注:** Technical report
>
> **摘要:** In this work, we aim to incentivize the reasoning ability of Multimodal Large Language Models (MLLMs) via reinforcement learning (RL) and develop an effective approach that mitigates the sparse reward and advantage vanishing issues during RL. To this end, we propose Share-GRPO, a novel RL approach that tackle these issues by exploring and sharing diverse reasoning trajectories over expanded question space. Specifically, Share-GRPO first expands the question space for a given question via data transformation techniques, and then encourages MLLM to effectively explore diverse reasoning trajectories over the expanded question space and shares the discovered reasoning trajectories across the expanded questions during RL. In addition, Share-GRPO also shares reward information during advantage computation, which estimates solution advantages hierarchically across and within question variants, allowing more accurate estimation of relative advantages and improving the stability of policy training. Extensive evaluations over six widely-used reasoning benchmarks showcase the superior performance of our method. Code will be available at https://github.com/HJYao00/R1-ShareVL.
>
---
#### [new 166] ATR-Bench: A Federated Learning Benchmark for Adaptation, Trust, and Reasoning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出ATR-Bench框架，针对联邦学习中适应性、可信度和推理三大核心维度，解决其标准化评估缺失的问题。通过系统分析方法、基准测试异构环境下的模型表现，并开源工具促进联邦学习的系统化研究。**

- **链接: [http://arxiv.org/pdf/2505.16850v1](http://arxiv.org/pdf/2505.16850v1)**

> **作者:** Tajamul Ashraf; Mohammed Mohsen Peerzada; Moloud Abdar; Yutong Xie; Yuyin Zhou; Xiaofeng Liu; Iqra Altaf Gillani; Janibul Bashir
>
> **备注:** Federated Learning Benchmark for Domain Adaptation, Trustworthiness, and Reasoning
>
> **摘要:** Federated Learning (FL) has emerged as a promising paradigm for collaborative model training while preserving data privacy across decentralized participants. As FL adoption grows, numerous techniques have been proposed to tackle its practical challenges. However, the lack of standardized evaluation across key dimensions hampers systematic progress and fair comparison of FL methods. In this work, we introduce ATR-Bench, a unified framework for analyzing federated learning through three foundational dimensions: Adaptation, Trust, and Reasoning. We provide an in-depth examination of the conceptual foundations, task formulations, and open research challenges associated with each theme. We have extensively benchmarked representative methods and datasets for adaptation to heterogeneous clients and trustworthiness in adversarial or unreliable environments. Due to the lack of reliable metrics and models for reasoning in FL, we only provide literature-driven insights for this dimension. ATR-Bench lays the groundwork for a systematic and holistic evaluation of federated learning with real-world relevance. We will make our complete codebase publicly accessible and a curated repository that continuously tracks new developments and research in the FL literature.
>
---
#### [new 167] Redemption Score: An Evaluation Framework to Rank Image Captions While Redeeming Image Semantics and Language Pragmatics
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Redemption Score框架，用于评估和排名图像标题，解决现有指标无法兼顾视觉语义与语言质量的问题。通过融合Mutual Information Divergence（全局图文对齐）、DINO-based图像生成相似性（视觉定位）和BERTScore（文本相似度），在Flickr8k实现56.43 Kendall-τ，超越12种方法，无需任务训练。**

- **链接: [http://arxiv.org/pdf/2505.16180v1](http://arxiv.org/pdf/2505.16180v1)**

> **作者:** Ashim Dahal; Ankit Ghimire; Saydul Akbar Murad; Nick Rahimi
>
> **摘要:** Evaluating image captions requires cohesive assessment of both visual semantics and language pragmatics, which is often not entirely captured by most metrics. We introduce Redemption Score, a novel hybrid framework that ranks image captions by triangulating three complementary signals: (1) Mutual Information Divergence (MID) for global image-text distributional alignment, (2) DINO-based perceptual similarity of cycle-generated images for visual grounding, and (3) BERTScore for contextual text similarity against human references. A calibrated fusion of these signals allows Redemption Score to offer a more holistic assessment. On the Flickr8k benchmark, Redemption Score achieves a Kendall-$\tau$ of 56.43, outperforming twelve prior methods and demonstrating superior correlation with human judgments without requiring task-specific training. Our framework provides a more robust and nuanced evaluation by effectively redeeming image semantics and linguistic interpretability indicated by strong transfer of knowledge in the Conceptual Captions and MS COCO datasets.
>
---
#### [new 168] Benchmarking Retrieval-Augmented Multimomal Generation for Document Question Answering
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于文档视觉问答（DocVQA）任务，针对现有方法忽视视觉信息及缺乏多模态评估基准的问题，提出MMDocRAG基准，包含4,055个跨模态QA对与证据链，引入多模态评估指标，实验显示专有视觉模型表现更优，多模态输入提升效果显著，为开发多模态问答系统提供测试平台和优化方向。**

- **链接: [http://arxiv.org/pdf/2505.16470v1](http://arxiv.org/pdf/2505.16470v1)**

> **作者:** Kuicai Dong; Yujing Chang; Shijie Huang; Yasheng Wang; Ruiming Tang; Yong Liu
>
> **备注:** preprint. code available at \url{https://mmdocrag.github.io/MMDocRAG/}
>
> **摘要:** Document Visual Question Answering (DocVQA) faces dual challenges in processing lengthy multimodal documents (text, images, tables) and performing cross-modal reasoning. Current document retrieval-augmented generation (DocRAG) methods remain limited by their text-centric approaches, frequently missing critical visual information. The field also lacks robust benchmarks for assessing multimodal evidence selection and integration. We introduce MMDocRAG, a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with multi-page, cross-modal evidence chains. Our framework introduces innovative metrics for evaluating multimodal quote selection and enables answers that interleave text with relevant visual elements. Through large-scale experiments with 60 VLM/LLM models and 14 retrieval systems, we identify persistent challenges in multimodal evidence retrieval, selection, and integration.Key findings reveal advanced proprietary LVMs show superior performance than open-sourced alternatives. Also, they show moderate advantages using multimodal inputs over text-only inputs, while open-source alternatives show significant performance degradation. Notably, fine-tuned LLMs achieve substantial improvements when using detailed image descriptions. MMDocRAG establishes a rigorous testing ground and provides actionable insights for developing more robust multimodal DocVQA systems. Our benchmark and code are available at https://mmdocrag.github.io/MMDocRAG/.
>
---
#### [new 169] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属数学推理任务，针对现有强化学习算法（如GRPO/DAPO）因粗粒度序列级优势估计无法区分token贡献的问题，提出KTAE算法。通过统计分析量化关键token对最终结果的贡献，结合序列优势实现细粒度优势估计，提升模型性能，实验显示其在五项基准测试中表现更优，精度更高且响应更短。**

- **链接: [http://arxiv.org/pdf/2505.16826v1](http://arxiv.org/pdf/2505.16826v1)**

> **作者:** Wei Sun; Wen Yang; Pu Jian; Qianlong Du; Fuwei Cui; Shuo Ren; Jiajun Zhang
>
> **摘要:** Recent advances have demonstrated that integrating reinforcement learning with rule-based rewards can significantly enhance the reasoning capabilities of large language models, even without supervised fine-tuning. However, prevalent reinforcement learning algorithms such as GRPO and its variants like DAPO, suffer from a coarse granularity issue when computing the advantage. Specifically, they compute rollout-level advantages that assign identical values to every token within a sequence, failing to capture token-specific contributions and hindering effective learning. To address this limitation, we propose Key-token Advantage Estimation (KTAE) - a novel algorithm that estimates fine-grained, token-level advantages without introducing additional models. KTAE leverages the correctness of sampled rollouts and applies statistical analysis to quantify the importance of individual tokens within a sequence to the final outcome. This quantified token-level importance is then combined with the rollout-level advantage to obtain a more fine-grained token-level advantage estimation. Empirical results show that models trained with GRPO+KTAE and DAPO+KTAE outperform baseline methods across five mathematical reasoning benchmarks. Notably, they achieve higher accuracy with shorter responses and even surpass R1-Distill-Qwen-1.5B using the same base model.
>
---
#### [new 170] Meta-PerSER: Few-Shot Listener Personalized Speech Emotion Recognition via Meta-learning
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出Meta-PerSER框架，解决传统语音情感识别（SER）忽视个体情感解读差异的问题。通过元学习（MAML）结合自监督预训练和优化策略，利用少量标注快速适配个人标注风格，在IEMOCAP数据集上验证了其个性化SER的优越性。任务：个性化少样本SER；问题：传统方法聚合标注导致个体偏差；方法：元学习+预训练模型适配。**

- **链接: [http://arxiv.org/pdf/2505.16220v1](http://arxiv.org/pdf/2505.16220v1)**

> **作者:** Liang-Yeh Shen; Shi-Xin Fang; Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by INTERSPEECH 2025. 7 pages, including 2 pages of appendix
>
> **摘要:** This paper introduces Meta-PerSER, a novel meta-learning framework that personalizes Speech Emotion Recognition (SER) by adapting to each listener's unique way of interpreting emotion. Conventional SER systems rely on aggregated annotations, which often overlook individual subtleties and lead to inconsistent predictions. In contrast, Meta-PerSER leverages a Model-Agnostic Meta-Learning (MAML) approach enhanced with Combined-Set Meta-Training, Derivative Annealing, and per-layer per-step learning rates, enabling rapid adaptation with only a few labeled examples. By integrating robust representations from pre-trained self-supervised models, our framework first captures general emotional cues and then fine-tunes itself to personal annotation styles. Experiments on the IEMOCAP corpus demonstrate that Meta-PerSER significantly outperforms baseline methods in both seen and unseen data scenarios, highlighting its promise for personalized emotion recognition.
>
---
#### [new 171] Causal LLM Routing: End-to-End Regret Minimization from Observational Data
- **分类: cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文提出因果端到端LLM路由框架，解决传统方法误差累积及依赖全反馈数据的问题。通过观测数据最小化决策遗憾，设计分类上界和softmax加权近似目标，并扩展处理异构成本偏好，实验显示SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.16037v1](http://arxiv.org/pdf/2505.16037v1)**

> **作者:** Asterios Tsiourvas; Wei Sun; Georgia Perakis
>
> **摘要:** LLM routing aims to select the most appropriate model for each query, balancing competing performance metrics such as accuracy and cost across a pool of language models. Prior approaches typically adopt a decoupled strategy, where the metrics are first predicted and the model is then selected based on these estimates. This setup is prone to compounding errors and often relies on full-feedback data, where each query is evaluated by all candidate models, which is costly to obtain and maintain in practice. In contrast, we learn from observational data, which records only the outcome of the model actually deployed. We propose a causal end-to-end framework that learns routing policies by minimizing decision-making regret from observational data. To enable efficient optimization, we introduce two theoretically grounded surrogate objectives: a classification-based upper bound, and a softmax-weighted regret approximation shown to recover the optimal policy at convergence. We further extend our framework to handle heterogeneous cost preferences via an interval-conditioned architecture. Experiments on public benchmarks show that our method outperforms existing baselines, achieving state-of-the-art performance across different embedding models.
>
---
#### [new 172] MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MedFrameQA，首个评估多图像医疗视觉问答（VQA）的基准，解决临床诊断需对比多影像而现有方法侧重单图分析的问题。通过自动化提取视频帧构建逻辑连贯的VQA数据（2851题，覆盖43器官），并测试多种模型发现其推理能力不足，推动多图临床推理研究。**

- **链接: [http://arxiv.org/pdf/2505.16964v1](http://arxiv.org/pdf/2505.16964v1)**

> **作者:** Suhao Yu; Haojin Wang; Juncheng Wu; Cihang Xie; Yuyin Zhou
>
> **备注:** 9 pages, 4 Figures Benchmark data: https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA
>
> **摘要:** Existing medical VQA benchmarks mostly focus on single-image analysis, yet clinicians almost always compare a series of images before reaching a diagnosis. To better approximate this workflow, we introduce MedFrameQA -- the first benchmark that explicitly evaluates multi-image reasoning in medical VQA. To build MedFrameQA both at scale and in high-quality, we develop 1) an automated pipeline that extracts temporally coherent frames from medical videos and constructs VQA items whose content evolves logically across images, and 2) a multiple-stage filtering strategy, including model-based and manual review, to preserve data clarity, difficulty, and medical relevance. The resulting dataset comprises 2,851 VQA pairs (gathered from 9,237 high-quality frames in 3,420 videos), covering nine human body systems and 43 organs; every question is accompanied by two to five images. We comprehensively benchmark ten advanced Multimodal LLMs -- both proprietary and open source, with and without explicit reasoning modules -- on MedFrameQA. The evaluation challengingly reveals that all models perform poorly, with most accuracies below 50%, and accuracy fluctuates as the number of images per question increases. Error analysis further shows that models frequently ignore salient findings, mis-aggregate evidence across images, and propagate early mistakes through their reasoning chains; results also vary substantially across body systems, organs, and modalities. We hope this work can catalyze research on clinically grounded, multi-image reasoning and accelerate progress toward more capable diagnostic AI systems.
>
---
#### [new 173] From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Pedagogical Visualization
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出教育可视化评估基准EduVisBench及多智能体框架EduVisAgent，解决基础模型生成教学可视化效果差的问题。通过构建多领域问题集和协作式智能体系统，提升复杂推理的视觉表达能力，实验显示效果提升40.2%。**

- **链接: [http://arxiv.org/pdf/2505.16832v1](http://arxiv.org/pdf/2505.16832v1)**

> **作者:** Haonian Ji; Shi Qiu; Siyang Xin; Siwei Han; Zhaorun Chen; Hongyi Wang; Dake Zhang; Huaxiu Yao
>
> **备注:** 16 pages; 7 figures
>
> **摘要:** While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at https://github.com/aiming-lab/EduVisBench and https://github.com/aiming-lab/EduVisAgent.
>
---
#### [new 174] When VLMs Meet Image Classification: Test Sets Renovation via Missing Label Identification
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出REVEAL框架，结合视觉语言模型（如LLaVA）与标注方法（如Cleanlab），解决图像分类测试集中的噪声标签和缺失标签问题。通过多模型预测聚合与共识过滤，改进数据集标签质量，经验证提升6个基准测试集准确性，助力公平模型评估。**

- **链接: [http://arxiv.org/pdf/2505.16149v1](http://arxiv.org/pdf/2505.16149v1)**

> **作者:** Zirui Pang; Haosheng Tan; Yuhan Pu; Zhijie Deng; Zhouan Shen; Keyu Hu; Jiaheng Wei
>
> **摘要:** Image classification benchmark datasets such as CIFAR, MNIST, and ImageNet serve as critical tools for model evaluation. However, despite the cleaning efforts, these datasets still suffer from pervasive noisy labels and often contain missing labels due to the co-existing image pattern where multiple classes appear in an image sample. This results in misleading model comparisons and unfair evaluations. Existing label cleaning methods focus primarily on noisy labels, but the issue of missing labels remains largely overlooked. Motivated by these challenges, we present a comprehensive framework named REVEAL, integrating state-of-the-art pre-trained vision-language models (e.g., LLaVA, BLIP, Janus, Qwen) with advanced machine/human label curation methods (e.g., Docta, Cleanlab, MTurk), to systematically address both noisy labels and missing label detection in widely-used image classification test sets. REVEAL detects potential noisy labels and omissions, aggregates predictions from various methods, and refines label accuracy through confidence-informed predictions and consensus-based filtering. Additionally, we provide a thorough analysis of state-of-the-art vision-language models and pre-trained image classifiers, highlighting their strengths and limitations within the context of dataset renovation by revealing 10 observations. Our method effectively reveals missing labels from public datasets and provides soft-labeled results with likelihoods. Through human verifications, REVEAL significantly improves the quality of 6 benchmark test sets, highly aligning to human judgments and enabling more accurate and meaningful comparisons in image classification.
>
---
#### [new 175] Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance
- **分类: cs.AI; cs.CL; I.2.6; I.2.7**

- **简介: 该论文属于LLMs基准测试任务，评估其在金融文本情感分析中的可靠性。针对LLMs难以解析财报电话中模糊表述和专业术语的问题，测试了Copilot、ChatGPT、Gemini及传统模型，分析其情感分析结果与市场反应的关联，并优化提示工程与可视化方法以提升准确性。**

- **链接: [http://arxiv.org/pdf/2505.16090v1](http://arxiv.org/pdf/2505.16090v1)**

> **作者:** Dominick Kubica; Dylan T. Gordon; Nanami Emura; Derleen Saini; Charlie Goldenberg
>
> **备注:** 6 pages, 4 figures. Research conducted as part of a Microsoft-sponsored Capstone Project at Santa Clara University
>
> **摘要:** As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence.
>
---
#### [new 176] Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary?
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务中的段落重排研究，旨在探究推理机制对重排模型准确性的影响。通过对比推理模型（ReasonRR）与非推理模型（StandardRR），发现后者表现更优；进一步禁用推理模块后（ReasonRR-NoReason），性能反而提升。研究指出，LLM的推理过程导致相关性评分极化，忽略部分相关性，损害重排效果。**

- **链接: [http://arxiv.org/pdf/2505.16886v1](http://arxiv.org/pdf/2505.16886v1)**

> **作者:** Nour Jedidi; Yung-Sung Chuang; James Glass; Jimmy Lin
>
> **摘要:** With the growing success of reasoning models across complex natural language tasks, researchers in the Information Retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on Large Language Models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction. But, does reasoning actually improve reranking accuracy? In this paper, we dive deeper into this question, studying the impact of the reasoning process by comparing reasoning-based pointwise rerankers (ReasonRR) to standard, non-reasoning pointwise rerankers (StandardRR) under identical training conditions, and observe that StandardRR generally outperforms ReasonRR. Building on this observation, we then study the importance of reasoning to ReasonRR by disabling its reasoning process (ReasonRR-NoReason), and find that ReasonRR-NoReason is surprisingly more effective than ReasonRR. Examining the cause of this result, our findings reveal that reasoning-based rerankers are limited by the LLM's reasoning process, which pushes it toward polarized relevance scores and thus fails to consider the partial relevance of passages, a key factor for the accuracy of pointwise rerankers.
>
---
#### [new 177] Merge to Mix: Mixing Datasets via Model Merging
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型微调任务，旨在解决传统数据集混合选择依赖试错、耗时低效的问题。提出Merge to Mix方法，通过合并各单任务微调模型替代完整混合训练，加速最优数据集组合筛选，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16066v1](http://arxiv.org/pdf/2505.16066v1)**

> **作者:** Zhixu Silvia Tao; Kasper Vinken; Hao-Wei Yeh; Avi Cooper; Xavier Boix
>
> **摘要:** Mixing datasets for fine-tuning large models (LMs) has become critical for maximizing performance on downstream tasks. However, composing effective dataset mixtures typically relies on heuristics and trial-and-error, often requiring multiple fine-tuning runs to achieve the desired outcome. We propose a novel method, $\textit{Merge to Mix}$, that accelerates composing dataset mixtures through model merging. Model merging is a recent technique that combines the abilities of multiple individually fine-tuned LMs into a single LM by using a few simple arithmetic operations. Our key insight is that merging models individually fine-tuned on each dataset in a mixture can effectively serve as a surrogate for a model fine-tuned on the entire mixture. Merge to Mix leverages this insight to accelerate selecting dataset mixtures without requiring full fine-tuning on each candidate mixture. Our experiments demonstrate that Merge to Mix surpasses state-of-the-art methods in dataset selection for fine-tuning LMs.
>
---
#### [new 178] NAN: A Training-Free Solution to Coefficient Estimation in Model Merging
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型合并任务，针对现有方法依赖启发式确定合并系数导致的可扩展性与通用性不足问题，提出NAN方法。通过最小二乘优化分析，发现最优权重与模型任务信息量相关，采用参数范数逆值估计系数，实现无训练、普适有效的合并策略。**

- **链接: [http://arxiv.org/pdf/2505.16148v1](http://arxiv.org/pdf/2505.16148v1)**

> **作者:** Chongjie Si; Kangtao Lv; Jingjing Jiang; Yadao Wang; Yongwei Wang; Xiaokang Yang; Wenbo Su; Bo Zheng; Wei Shen
>
> **摘要:** Model merging offers a training-free alternative to multi-task learning by combining independently fine-tuned models into a unified one without access to raw data. However, existing approaches often rely on heuristics to determine the merging coefficients, limiting their scalability and generality. In this work, we revisit model merging through the lens of least-squares optimization and show that the optimal merging weights should scale with the amount of task-specific information encoded in each model. Based on this insight, we propose NAN, a simple yet effective method that estimates model merging coefficients via the inverse of parameter norm. NAN is training-free, plug-and-play, and applicable to a wide range of merging strategies. Extensive experiments on show that NAN consistently improves performance of baseline methods.
>
---
#### [new 179] All You Need is "Leet": Evading Hate-speech Detection AI
- **分类: cs.CR; cs.CL; cs.LG; K.6.5**

- **简介: 该论文属于对抗攻击任务，旨在通过生成微小扰动使先进仇恨言论检测模型失效，同时保持文本原意。研究设计黑盒攻击技术，成功使86.8%的仇恨文本逃避检测。**

- **链接: [http://arxiv.org/pdf/2505.16263v1](http://arxiv.org/pdf/2505.16263v1)**

> **作者:** Sampanna Yashwant Kahu; Naman Ahuja
>
> **备注:** 10 pages, 22 figures, The source code and data used in this work is available at: https://github.com/SampannaKahu/all_you_need_is_leet
>
> **摘要:** Social media and online forums are increasingly becoming popular. Unfortunately, these platforms are being used for spreading hate speech. In this paper, we design black-box techniques to protect users from hate-speech on online platforms by generating perturbations that can fool state of the art deep learning based hate speech detection models thereby decreasing their efficiency. We also ensure a minimal change in the original meaning of hate-speech. Our best perturbation attack is successfully able to evade hate-speech detection for 86.8 % of hateful text.
>
---
#### [new 180] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AdaSTaR算法，改进自监督推理模型（STaR）的训练数据采样方法。针对随机采样导致的训练不均衡及效率低问题，通过自适应多样性采样和动态难度调整，提升训练效果并减少计算。在6个基准测试中实现最佳准确率，FLOPs减少58.6%，适用于多种模型。**

- **链接: [http://arxiv.org/pdf/2505.16322v1](http://arxiv.org/pdf/2505.16322v1)**

> **作者:** Woosung Koh; Wonbeen Oh; Jaein Jang; MinHyung Lee; Hyeongjin Kim; Ah Yeon Kim; Joonkee Kim; Junghyun Lee; Taehyeon Kim; Se-Young Yun
>
> **备注:** Pre-print
>
> **摘要:** Self-Taught Reasoners (STaR), synonymously known as Rejection sampling Fine-Tuning (RFT), is an integral part of the training pipeline of self-improving reasoning Language Models (LMs). The self-improving mechanism often employs random observation (data) sampling. However, this results in trained observation imbalance; inefficiently over-training on solved examples while under-training on challenging ones. In response, we introduce Adaptive STaR (AdaSTaR), a novel algorithm that rectifies this by integrating two adaptive sampling principles: (1) Adaptive Sampling for Diversity: promoting balanced training across observations, and (2) Adaptive Sampling for Curriculum: dynamically adjusting data difficulty to match the model's evolving strength. Across six benchmarks, AdaSTaR achieves best test accuracy in all instances (6/6) and reduces training FLOPs by an average of 58.6% against an extensive list of baselines. These improvements in performance and efficiency generalize to different pre-trained LMs and larger models, paving the way for more efficient and effective self-improving LMs.
>
---
#### [new 181] MiLQ: Benchmarking IR Models for Bilingual Web Search with Mixed Language Queries
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，针对 bilingual speakers 常用混合语言查询但研究不足的问题，构建首个公开基准测试集 MiLQ，评估多语言模型表现并分析查询策略。实验显示现有模型表现一般，而故意在查询中加入英语能提升检索效果，为改进 IR 模型提供方向。**

- **链接: [http://arxiv.org/pdf/2505.16631v1](http://arxiv.org/pdf/2505.16631v1)**

> **作者:** Jonghwi Kim; Deokhyung Kang; Seonjeong Hwang; Yunsu Kim; Jungseul Ok; Gary Lee
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Despite bilingual speakers frequently using mixed-language queries in web searches, Information Retrieval (IR) research on them remains scarce. To address this, we introduce MiLQ,Mixed-Language Query test set, the first public benchmark of mixed-language queries, confirmed as realistic and highly preferred. Experiments show that multilingual IR models perform moderately on MiLQ and inconsistently across native, English, and mixed-language queries, also suggesting code-switched training data's potential for robust IR models handling such queries. Meanwhile, intentional English mixing in queries proves an effective strategy for bilinguals searching English documents, which our analysis attributes to enhanced token matching compared to native queries.
>
---
#### [new 182] MAPS: A Multilingual Benchmark for Global Agent Performance and Security
- **分类: cs.DB; cs.CL; cs.CR**

- **简介: 该论文提出多语言基准MAPS，评估代理AI系统的跨语言性能与安全性。针对现有基准仅支持英语导致非英语用户面临可靠性与安全风险的问题，团队将四个英文基准翻译为十种语言，构建805个任务，揭示语言切换导致的性能下降，并提出优化建议，推动公平可靠的全球AI发展。**

- **链接: [http://arxiv.org/pdf/2505.15935v1](http://arxiv.org/pdf/2505.15935v1)**

> **作者:** Omer Hofman; Oren Rachmil; Shamik Bose; Vikas Pahuja; Jonathan Brokman; Toshiya Shimizu; Trisha Starostina; Kelly Marchisio; Seraphina Goldfarb-Tarrant; Roman Vainshtein
>
> **摘要:** Agentic AI systems, which build on Large Language Models (LLMs) and interact with tools and memory, have rapidly advanced in capability and scope. Yet, since LLMs have been shown to struggle in multilingual settings, typically resulting in lower performance and reduced safety, agentic systems risk inheriting these limitations. This raises concerns about the global accessibility of such systems, as users interacting in languages other than English may encounter unreliable or security-critical agent behavior. Despite growing interest in evaluating agentic AI, existing benchmarks focus exclusively on English, leaving multilingual settings unexplored. To address this gap, we propose MAPS, a multilingual benchmark suite designed to evaluate agentic AI systems across diverse languages and tasks. MAPS builds on four widely used agentic benchmarks - GAIA (real-world tasks), SWE-bench (code generation), MATH (mathematical reasoning), and the Agent Security Benchmark (security). We translate each dataset into ten diverse languages, resulting in 805 unique tasks and 8,855 total language-specific instances. Our benchmark suite enables a systematic analysis of how multilingual contexts affect agent performance and robustness. Empirically, we observe consistent degradation in both performance and security when transitioning from English to other languages, with severity varying by task and correlating with the amount of translated input. Building on these findings, we provide actionable recommendations to guide agentic AI systems development and assessment under multilingual settings. This work establishes a standardized evaluation framework, encouraging future research towards equitable, reliable, and globally accessible agentic AI. MAPS benchmark suite is publicly available at https://huggingface.co/datasets/Fujitsu-FRE/MAPS
>
---
#### [new 183] SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Dev数据集，针对功能驱动开发(FDD)任务的AI挑战，提供1.4万训练样本及可执行测试环境，用于评估和训练模型。通过监督微调和强化学习，验证其提升模型性能（如7B模型接近GPT-4水平），填补了复杂代码功能开发的AI研究空白。**

- **链接: [http://arxiv.org/pdf/2505.16975v1](http://arxiv.org/pdf/2505.16975v1)**

> **作者:** Yaxin Du; Yuzhu Cai; Yifan Zhou; Cheng Wang; Yu Qian; Xianghe Pang; Qian Liu; Yue Hu; Siheng Chen
>
> **摘要:** Large Language Models (LLMs) have shown strong capability in diverse software engineering tasks, e.g. code completion, bug fixing, and document generation. However, feature-driven development (FDD), a highly prevalent real-world task that involves developing new functionalities for large, existing codebases, remains underexplored. We therefore introduce SWE-Dev, the first large-scale dataset (with 14,000 training and 500 test samples) designed to evaluate and train autonomous coding systems on real-world feature development tasks. To ensure verifiable and diverse training, SWE-Dev uniquely provides all instances with a runnable environment and its developer-authored executable unit tests. This collection not only provides high-quality data for Supervised Fine-Tuning (SFT), but also enables Reinforcement Learning (RL) by delivering accurate reward signals from executable unit tests. Our extensive evaluations on SWE-Dev, covering 17 chatbot LLMs, 10 reasoning models, and 10 Multi-Agent Systems (MAS), reveal that FDD is a profoundly challenging frontier for current AI (e.g., Claude-3.7-Sonnet achieves only 22.45\% Pass@3 on the hard test split). Crucially, we demonstrate that SWE-Dev serves as an effective platform for model improvement: fine-tuning on training set enabled a 7B model comparable to GPT-4o on \textit{hard} split, underscoring the value of its high-quality training data. Code is available here \href{https://github.com/justLittleWhite/SWE-Dev}{https://github.com/justLittleWhite/SWE-Dev}.
>
---
#### [new 184] SPaRC: A Spatial Pathfinding Reasoning Challenge
- **分类: cs.AI; cs.CL**

- **简介: 论文提出SPaRC数据集，包含1000个2D路径谜题，评估模型在空间推理与复杂规则下的多步骤问题解决能力。针对现有数据集无法测试抽象多步任务的缺陷，该工作揭示模型在路径规划和逻辑推理上的不足，指出需改进训练与推理方法以提升表现。**

- **链接: [http://arxiv.org/pdf/2505.16686v1](http://arxiv.org/pdf/2505.16686v1)**

> **作者:** Lars Benedikt Kaesberg; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **摘要:** Existing reasoning datasets saturate and fail to test abstract, multi-step problems, especially pathfinding and complex rule constraint satisfaction. We introduce SPaRC (Spatial Pathfinding Reasoning Challenge), a dataset of 1,000 2D grid pathfinding puzzles to evaluate spatial and symbolic reasoning, requiring step-by-step planning with arithmetic and geometric rules. Humans achieve near-perfect accuracy (98.0%; 94.5% on hard puzzles), while the best reasoning models, such as o4-mini, struggle (15.8%; 1.1% on hard puzzles). Models often generate invalid paths (>50% of puzzles for o4-mini), and reasoning tokens reveal they make errors in navigation and spatial logic. Unlike humans, who take longer on hard puzzles, models fail to scale test-time compute with difficulty. Allowing models to make multiple solution attempts improves accuracy, suggesting potential for better spatial reasoning with improved training and efficient test-time scaling methods. SPaRC can be used as a window into models' spatial reasoning limitations and drive research toward new methods that excel in abstract, multi-step problem-solving.
>
---
#### [new 185] CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出对抗攻击任务，解决在黑盒环境下通过篡改LLM系统提示实现对话劫持的问题。CAIN框架通过两阶段生成与优化恶意提示，使模型仅对特定问题（如政治、健康相关）输出有害回答，其余保持正常，实验显示其显著攻击效果，强调需提升LLM安全性。**

- **链接: [http://arxiv.org/pdf/2505.16888v1](http://arxiv.org/pdf/2505.16888v1)**

> **作者:** Viet Pham; Thai Le
>
> **摘要:** Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.
>
---
## 更新

#### [replaced 001] Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.20073v2](http://arxiv.org/pdf/2502.20073v2)**

> **作者:** Haochen Sun; Shuwen Zhang; Lujie Niu; Lei Ren; Hao Xu; Hao Fu; Fangkun Zhao; Caixia Yuan; Xiaojie Wang
>
> **备注:** 30 pages, 17 figures
>
> **摘要:** Large language models (LLMs) based agent systems have made great strides in real-world applications beyond traditional NLP tasks. This paper proposes a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work. We conduct extensive experiments over 11 popular LLMs and show that, while the LLMs present a strong ability in goal interpretation, there is a significant discrepancy in active collaboration and continuous adaptation which are critical for efficiently fulfilling complicated tasks. Notably, we highlight the strengths and weaknesses in LLM-MAS and provide insights for improving and evaluating LLM-MAS on a unified and open-sourced benchmark. The environments, 30 open-ended tasks, and the evaluation package are publicly available at https://github.com/YusaeMeow/Collab-Overcooked.
>
---
#### [replaced 002] FineFilter: A Fine-grained Noise Filtering Mechanism for Retrieval-Augmented Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11811v4](http://arxiv.org/pdf/2502.11811v4)**

> **作者:** Qianchi Zhang; Hainan Zhang; Liang Pang; Ziwei Wang; Hongwei Zheng; Yongxin Tong; Zhiming Zheng
>
> **备注:** 18 pages, 4 figures, 18 tables, under review
>
> **摘要:** Retrieved documents containing noise will hinder Retrieval-Augmented Generation (RAG) from detecting answer clues, necessitating noise filtering mechanisms to enhance accuracy. Existing methods use reranking or summarization to identify the most relevant sentences, but directly and accurately locating answer clues from these large-scale and complex documents remains challenging. Unlike these document-level operations, we treat noise filtering as a sentence-level MinMax optimization problem: first identifying potential clues from multiple documents, then ranking them by relevance, and finally retaining the minimum number of clues through truncation. In this paper, we propose FineFilter, a novel fine-grained noise filtering mechanism for RAG, consisting of a clue extractor, a reranker, and a truncator. We optimize each module to tackle complex reasoning challenges: (1) The clue extractor first uses sentences containing the answer and similar ones as fine-tuning targets, aiming to extract sufficient potential clues; (2) The reranker is trained to prioritize effective clues based on the real feedback from the generation module, with clues capable of generating correct answers as positive samples and others as negative; (3) The truncator takes the minimum number of clues needed to answer the question (truncation point) as fine-tuning targets, and performs truncation on the reranked clues to achieve fine-grained noise filtering. Experiments on three QA datasets demonstrate that FineFilter significantly improves QA performance over baselines on both LLaMA3 and Mistral. Further analysis confirms its effectiveness in complex reasoning, robustness to unreliable retrieval, and generalization to different scenarios.
>
---
#### [replaced 003] CodeMind: Evaluating Large Language Models for Code Reasoning
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **链接: [http://arxiv.org/pdf/2402.09664v5](http://arxiv.org/pdf/2402.09664v5)**

> **作者:** Changshu Liu; Yang Chen; Reyhaneh Jabbarvand
>
> **摘要:** Large Language Models (LLMs) have been widely used to automate programming tasks. Their capabilities have been evaluated by assessing the quality of generated code through tests or proofs. The extent to which they can reason about code is a critical question revealing important insights about their true capabilities. This paper introduces CodeMind, a framework designed to gauge the code reasoning abilities of LLMs through the following explicit and implicit code reasoning tasks: Independent Execution Reasoning (IER), Specification Reasoning (SR) and Dynamic Semantics Reasoning (DSR). The first evaluates the abilities of LLMs to simulate the execution of given inputs to a code and predict the output (IER). The second assesses the abilities of LLMs to incorporate the simulation of test data in the specification into code generation (SR). Finally, CodeMind evaluates LLMs' abilities to understand overall code semantics only given a specific input/output (DSR). Our extensive evaluation of ten LLMs across four widely used benchmarks using CodeMind shows that LLMs, depending on their size and training strategy, can reason about some dynamic aspects of code. However, their performance drops for code with higher complexity, non-trivial logical and arithmetic operators, non-primitive types, and API calls. We show that these reasoning tasks evaluate LLMs differently, and a comprehensive evaluation of code reasoning requires them all. Finally, we show that the performance of LLMs in bug repair is not correlated with any of the code reasoning tasks, and except for advanced frontier models, other LLMs do not incorporate code reasoning when performing bug repair.
>
---
#### [replaced 004] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.05714v4](http://arxiv.org/pdf/2501.05714v4)**

> **作者:** Chen Huang; Yang Deng; Wenqiang Lei; Jiancheng Lv; Tat-Seng Chua; Jimmy Xiangji Huang
>
> **备注:** ACL 2025 Main paper
>
> **摘要:** With the advancement of large language models (LLMs), intelligent models have evolved from mere tools to autonomous agents with their own goals and strategies for cooperating with humans. This evolution has birthed a novel paradigm in NLP, i.e., human-model cooperation, that has yielded remarkable progress in numerous NLP tasks in recent years. In this paper, we take the first step to present a thorough review of human-model cooperation, exploring its principles, formalizations, and open challenges. In particular, we introduce a new taxonomy that provides a unified perspective to summarize existing approaches. Also, we discuss potential frontier areas and their corresponding challenges. We regard our work as an entry point, paving the way for more breakthrough research in this regard.
>
---
#### [replaced 005] Diverse Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18101v4](http://arxiv.org/pdf/2501.18101v4)**

> **作者:** Jack Lanchantin; Angelica Chen; Shehzaad Dhuliawala; Ping Yu; Jason Weston; Sainbayar Sukhbaatar; Ilia Kulikov
>
> **摘要:** Post-training of language models, either through reinforcement learning, preference optimization or supervised finetuning, tends to sharpen the output probability distribution and reduce the diversity of generated responses. This is particularly a problem for creative generative tasks where varied responses are desired. In this work we introduce Diverse Preference Optimization (DivPO), an optimization method which learns to generate much more diverse responses than standard pipelines, while maintaining the quality of the generations. In DivPO, preference pairs are selected by first considering a pool of responses, and a measure of diversity among them, and selecting chosen examples as being more rare but high quality, while rejected examples are more common, but low quality. DivPO results in generating 45.6% more diverse persona attributes, and a 74.6% increase in story diversity, while maintaining similar win rates as standard baselines. On general instruction following, DivPO results in a 46.2% increase in diversity, and a 2.4% winrate improvement compared to DPO.
>
---
#### [replaced 006] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12212v2](http://arxiv.org/pdf/2505.12212v2)**

> **作者:** Shaobo Wang; Xiangqi Jin; Ziming Wang; Jize Wang; Jiajun Zhang; Kaixin Li; Zichen Wen; Zhong Li; Conghui He; Xuming Hu; Linfeng Zhang
>
> **备注:** Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
>
> **摘要:** Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup.
>
---
#### [replaced 007] Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.10056v3](http://arxiv.org/pdf/2403.10056v3)**

> **作者:** Yongquan He; Wenyuan Zhang; Xuancheng Huang; Peng Zhang
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score, to measure the generalization and instruction-following abilities of LLMs. Experiments demonstrate our method achieves superior performance on both seen and held-out tasks.
>
---
#### [replaced 008] Slamming: Training a Speech Language Model on One GPU in a Day
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.15814v2](http://arxiv.org/pdf/2502.15814v2)**

> **作者:** Gallil Maimon; Avishai Elmakies; Yossi Adi
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** We introduce Slam, a recipe for training high-quality Speech Language Models (SLMs) on a single academic GPU in 24 hours. We do so through empirical analysis of model initialisation and architecture, synthetic training data, preference optimisation with synthetic data and tweaking all other components. We empirically demonstrate that this training recipe also scales well with more compute getting results on par with leading SLMs in a fraction of the compute cost. We hope these insights will make SLM training and research more accessible. In the context of SLM scaling laws, our results far outperform predicted compute optimal performance, giving an optimistic view to SLM feasibility. See code, data, models, samples at - https://pages.cs.huji.ac.il/adiyoss-lab/slamming .
>
---
#### [replaced 009] DomainCQA: Crafting Expert-Level QA from Domain-Specific Charts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19498v3](http://arxiv.org/pdf/2503.19498v3)**

> **作者:** Ling Zhong; Yujing Lu; Jing Yang; Weiming Li; Peng Wei; Yongheng Wang; Manni Duan; Qing Zhang
>
> **备注:** 87 pages, 65 figures
>
> **摘要:** Chart Question Answering (CQA) benchmarks are essential for evaluating the capability of Multimodal Large Language Models (MLLMs) to interpret visual data. However, current benchmarks focus primarily on the evaluation of general-purpose CQA but fail to adequately capture domain-specific challenges. We introduce DomainCQA, a systematic methodology for constructing domain-specific CQA benchmarks, and demonstrate its effectiveness by developing AstroChart, a CQA benchmark in the field of astronomy. Our evaluation shows that current MLLMs face fundamental challenges in vision-language alignment and domain adaptation, highlighting a critical gap in current benchmarks. By providing a scalable and rigorous framework, DomainCQA enables more precise assessment and improvement of MLLMs for domain-specific applications.
>
---
#### [replaced 010] FoREST: Frame of Reference Evaluation in Spatial Reasoning Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17775v2](http://arxiv.org/pdf/2502.17775v2)**

> **作者:** Tanawan Premsri; Parisa Kordjamshidi
>
> **备注:** 9 pages
>
> **摘要:** Spatial reasoning is a fundamental aspect of human intelligence. One key concept in spatial cognition is the Frame of Reference (FoR), which identifies the perspective of spatial expressions. Despite its significance, FoR has received limited attention in AI models that need spatial intelligence. There is a lack of dedicated benchmarks and in-depth evaluation of large language models (LLMs) in this area. To address this issue, we introduce the Frame of Reference Evaluation in Spatial Reasoning Tasks (FoREST) benchmark, designed to assess FoR comprehension in LLMs. We evaluate LLMs on answering questions that require FoR comprehension and layout generation in text-to-image models using FoREST. Our results reveal a notable performance gap across different FoR classes in various LLMs, affecting their ability to generate accurate layouts for text-to-image generation. This highlights critical shortcomings in FoR comprehension. To improve FoR understanding, we propose Spatial-Guided prompting, which improves LLMs ability to extract essential spatial concepts. Our proposed method improves overall performance across spatial reasoning tasks.
>
---
#### [replaced 011] From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15463v3](http://arxiv.org/pdf/2503.15463v3)**

> **作者:** Jia-Nan Li; Jian Guan; Songhao Wu; Wei Wu; Rui Yan
>
> **摘要:** Large language models (LLMs) have traditionally been aligned through one-size-fits-all approaches that assume uniform human preferences, fundamentally overlooking the diversity in user values and needs. This paper introduces a comprehensive framework for scalable personalized alignment of LLMs. We establish a systematic preference space characterizing psychological and behavioral dimensions, alongside diverse persona representations for robust preference inference in real-world scenarios. Building upon this foundation, we introduce \textsc{AlignX}, a large-scale dataset of over 1.3 million personalized preference examples, and develop two complementary alignment approaches: \textit{in-context alignment} directly conditioning on persona representations and \textit{preference-bridged alignment} modeling intermediate preference distributions. Extensive experiments demonstrate substantial improvements over existing methods, with an average 17.06\% accuracy gain across four benchmarks while exhibiting a strong adaptation capability to novel preferences, robustness to limited user data, and precise preference controllability. These results validate our approach toward user-adaptive AI systems.
>
---
#### [replaced 012] Red-Teaming for Inducing Societal Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.04756v2](http://arxiv.org/pdf/2405.04756v2)**

> **作者:** Chu Fei Luo; Ahmad Ghawanmeh; Bharat Bhimshetty; Kashyap Murali; Murli Jadhav; Xiaodan Zhu; Faiza Khan Khattak
>
> **摘要:** Ensuring the safe deployment of AI systems is critical in industry settings where biased outputs can lead to significant operational, reputational, and regulatory risks. Thorough evaluation before deployment is essential to prevent these hazards. Red-teaming addresses this need by employing adversarial attacks to develop guardrails that detect and reject biased or harmful queries, enabling models to be retrained or steered away from harmful outputs. However, most red-teaming efforts focus on harmful or unethical instructions rather than addressing social bias, leaving this critical area under-explored despite its significant real-world impact, especially in customer-facing systems. We propose two bias-specific red-teaming methods, Emotional Bias Probe (EBP) and BiasKG, to evaluate how standard safety measures for harmful content affect bias. For BiasKG, we refactor natural language stereotypes into a knowledge graph. We use these attacking strategies to induce biased responses from several open- and closed-source language models. Unlike prior work, these methods specifically target social bias. We find our method increases bias in all models, even those trained with safety guardrails. Our work emphasizes uncovering societal bias in LLMs through rigorous evaluation, and recommends measures ensure AI safety in high-stakes industry deployments.
>
---
#### [replaced 013] GLEE: A Unified Framework and Benchmark for Language-based Economic Environments
- **分类: cs.CL; cs.AI; cs.CY; cs.GT; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.05254v2](http://arxiv.org/pdf/2410.05254v2)**

> **作者:** Eilam Shapira; Omer Madmon; Itamar Reinman; Samuel Joseph Amouyal; Roi Reichart; Moshe Tennenholtz
>
> **摘要:** Large Language Models (LLMs) show significant potential in economic and strategic interactions, where communication via natural language is often prevalent. This raises key questions: Do LLMs behave rationally? How do they perform compared to humans? Do they tend to reach an efficient and fair outcome? What is the role of natural language in strategic interaction? How do characteristics of the economic environment influence these dynamics? These questions become crucial concerning the economic and societal implications of integrating LLM-based agents into real-world data-driven systems, such as online retail platforms and recommender systems. To answer these questions, we introduce a benchmark for standardizing research on two-player, sequential, language-based games. Inspired by the economic literature, we define three base families of games with consistent parameterization, degrees of freedom and economic measures to evaluate agents' performance (self-gain), as well as the game outcome (efficiency and fairness). We develop an open-source framework for interaction simulation and analysis, and utilize it to collect a dataset of LLM vs. LLM interactions across numerous game configurations and an additional dataset of human vs. LLM interactions. Through extensive experimentation, we demonstrate how our framework and dataset can be used to: (i) compare the behavior of LLM-based agents in various economic contexts; (ii) evaluate agents in both individual and collective performance measures; and (iii) quantify the effect of the economic characteristics of the environments on the behavior of agents. Our results suggest that the market parameters, as well as the choice of the LLMs, tend to have complex and interdependent effects on the economic outcome, which calls for careful design and analysis of the language-based economic ecosystem.
>
---
#### [replaced 014] Transferring Textual Preferences to Vision-Language Understanding through Model Merging
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13487v2](http://arxiv.org/pdf/2502.13487v2)**

> **作者:** Chen-An Li; Tzu-Han Lin; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Accepted to ACL 2025 main
>
> **摘要:** Large vision-language models (LVLMs) perform outstandingly across various multimodal tasks. However, their ability to evaluate generated content remains limited, and training vision-language reward models (VLRMs) with preference data is computationally expensive. This paper explores a training-free alternative by merging text-based reward models (RMs) with LVLMs to create VLRMs. Our approach shows that integrating these models leads to improved performance over LVLMs' scoring and text-based RMs, offering an efficient method for incorporating textual preferences into LVLMs.
>
---
#### [replaced 015] Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04380v2](http://arxiv.org/pdf/2502.04380v2)**

> **作者:** Zhenqing Ling; Daoyuan Chen; Liuyi Yao; Qianli Shen; Yaliang Li; Ying Shen
>
> **备注:** 33 pages, 20 figures, 21 tables
>
> **摘要:** Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
>
---
#### [replaced 016] How Real Are Synthetic Therapy Conversations? Evaluating Fidelity in Prolonged Exposure Dialogues
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; 68T50; I.2.7; H.3.1**

- **链接: [http://arxiv.org/pdf/2504.21800v3](http://arxiv.org/pdf/2504.21800v3)**

> **作者:** Suhas BN; Dominik Mattioli; Saeed Abdullah; Rosa I. Arriaga; Chris W. Wiese; Andrew M. Sherrill
>
> **备注:** 10 pages, 5 tables
>
> **摘要:** The growing adoption of synthetic data in healthcare is driven by privacy concerns, limited access to real-world data, and the high cost of annotation. This work explores the use of synthetic Prolonged Exposure (PE) therapeutic conversations for Post-Traumatic Stress Disorder (PTSD) as a scalable alternative for training and evaluating clinical models. We systematically compare real and synthetic dialogues using linguistic, structural, and protocol-specific metrics, including turn-taking patterns and treatment fidelity. We also introduce and evaluate PE-specific metrics derived from linguistic analysis and semantic modeling, offering a novel framework for assessing clinical fidelity beyond surface fluency. Our findings show that although synthetic data holds promise for mitigating data scarcity and protecting patient privacy, it can struggle to capture the subtle dynamics of therapeutic interactions. Synthetic therapy dialogues closely match structural features of real-world conversations (e.g., speaker switch ratio: 0.98 vs. 0.99); however, they may not adequately reflect key fidelity markers (e.g., distress monitoring). We highlight gaps in existing evaluation frameworks and advocate for fidelity-aware metrics that go beyond surface fluency to uncover clinically significant failures. Our findings clarify where synthetic data can effectively complement real-world datasets -- and where critical limitations remain.
>
---
#### [replaced 017] Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01222v2](http://arxiv.org/pdf/2503.01222v2)**

> **作者:** Wenbin Wang; Yongcheng Jing; Liang Ding; Yingjie Wang; Li Shen; Yong Luo; Bo Du; Dacheng Tao
>
> **摘要:** High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs. Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on HR-Bench.
>
---
#### [replaced 018] FIRE: Flexible Integration of Data Quality Ratings for Effective Pre-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00761v3](http://arxiv.org/pdf/2502.00761v3)**

> **作者:** Liangyu Xu; Xuemiao Zhang; Feiyu Duan; Sirui Wang; Rongxiang Weng; Jingang Wang; Xunliang Cai
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Selecting high-quality data can improve the pretraining efficiency of large language models (LLMs). Existing methods generally rely on heuristic techniques or single quality signals, limiting their ability to evaluate data quality comprehensively. In this work, we propose FIRE, a flexible and scalable framework for integrating multiple data quality raters, which allows for a comprehensive assessment of data quality across various dimensions. FIRE aligns multiple quality signals into a unified space, and integrates diverse data quality raters to provide a comprehensive quality signal for each data point. Further, we introduce a progressive data selection scheme based on FIRE that iteratively refines the selection of high-quality data points. Extensive experiments show that FIRE outperforms other data selection methods and significantly boosts pretrained model performance across a wide range of downstream tasks, while requiring less than 37.5\% of the training data needed by the Random baseline to reach the target performance.
>
---
#### [replaced 019] MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09499v2](http://arxiv.org/pdf/2503.09499v2)**

> **作者:** Zhe Xu; Daoyuan Chen; Zhenqing Ling; Yaliang Li; Ying Shen
>
> **备注:** 22 pages, 7 tables
>
> **摘要:** Large foundation models face challenges in acquiring transferable, structured thinking abilities, especially when supervised with rigid templates or crowd-annotated instruction datasets. Unlike prior approaches, we focus on a thinking-centric data synthesis paradigm that enables models to evolve through self-generated, cognitively guided data. We propose MindGYM, a structured and scalable framework for question synthesis, composed of: (1) Cognitive Thinking Process Injection, which infuses high-level reasoning objectives to shape the model's synthesis behavior; (2) Seed Single-Hop Question Synthesis, generating atomic questions from diverse semantic types to encourage broader thinking; and (3) Challenging Multi-Hop QA Synthesis, composing more complex multi-hop questions based on QA seeds for deeper reasoning. Detailed analysis shows that synthetic data generated by our method achieves 16.7% higher average quality and 67.91% lower quality variance compared to baseline sources, highlighting that both high-quality and self-contained data are essential for effective, thinking-oriented fine-tuning. MindGYM improves performance on six reasoning benchmarks, achieving gains of up to 16% on MathVision using only 400 data samples, and generalizable improvements across different model sizes and architectures. MindGYM underscores the viability of self-challenging mechanisms in refining large model capabilities while minimizing human intervention and resource demands. Code and data are released to promote data-centric research into self-evolving foundation models driven by their internal reasoning capabilities.
>
---
#### [replaced 020] Social Bias in Popular Question-Answering Benchmarks
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.15553v2](http://arxiv.org/pdf/2505.15553v2)**

> **作者:** Angelie Kraft; Judith Simon; Sonja Schimmler
>
> **摘要:** Question-answering (QA) and reading comprehension (RC) benchmarks are essential for assessing the capabilities of large language models (LLMs) in retrieving and reproducing knowledge. However, we demonstrate that popular QA and RC benchmarks are biased and do not cover questions about different demographics or regions in a representative way, potentially due to a lack of diversity of those involved in their creation. We perform a qualitative content analysis of 30 benchmark papers and a quantitative analysis of 20 respective benchmark datasets to learn (1) who is involved in the benchmark creation, (2) how social bias is addressed or prevented, and (3) whether the demographics of the creators and annotators correspond to particular biases in the content. Most analyzed benchmark papers provided insufficient information regarding the stakeholders involved in benchmark creation, particularly the annotators. Notably, just one of the benchmark papers explicitly reported measures taken to address social representation issues. Moreover, the data analysis revealed gender, religion, and geographic biases across a wide range of encyclopedic, commonsense, and scholarly benchmarks. More transparent and bias-aware QA and RC benchmark creation practices are needed to facilitate better scrutiny and incentivize the development of fairer LLMs.
>
---
#### [replaced 021] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.08085v4](http://arxiv.org/pdf/2410.08085v4)**

> **作者:** Yuan Sui; Yufei He; Zifeng Ding; Bryan Hooi
>
> **备注:** This paper has been accepted by ACL 2025
>
> **摘要:** Recent works integrating Knowledge Graphs (KGs) have shown promising improvements in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing benchmarks primarily focus on closed-ended tasks, leaving a gap in evaluating performance on more complex, real-world scenarios. This limitation also hinders a thorough assessment of KGs' potential to reduce hallucinations in LLMs. To address this, we introduce OKGQA, a new benchmark specifically designed to evaluate LLMs augmented with KGs in open-ended, real-world question answering settings. OKGQA reflects practical complexities through diverse question types and incorporates metrics to quantify both hallucination rates and reasoning improvements in LLM+KG models. To consider the scenarios in which KGs may contain varying levels of errors, we propose a benchmark variant, OKGQA-P, to assess model performance when the semantics and structure of KGs are deliberately perturbed and contaminated. In this paper, we aims to (1) explore whether KGs can make LLMs more trustworthy in an open-ended setting, and (2) conduct a comparative analysis to shed light on method design. We believe this study can facilitate a more complete performance comparison and encourages continuous improvement in integrating KGs with LLMs to mitigate hallucination, and make LLMs more trustworthy. Code and data are released at https://github.com/Y-Sui/OKGQA.
>
---
#### [replaced 022] Extractive Structures Learned in Pretraining Enable Generalization on Finetuned Facts
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04614v3](http://arxiv.org/pdf/2412.04614v3)**

> **作者:** Jiahai Feng; Stuart Russell; Jacob Steinhardt
>
> **摘要:** Pretrained language models (LMs) can generalize to implications of facts that they are finetuned on. For example, if finetuned on ``John Doe lives in Tokyo," LMs can correctly answer ``What language do the people in John Doe's city speak?'' with ``Japanese''. However, little is known about the mechanisms that enable this generalization or how they are learned during pretraining. We introduce extractive structures as a framework for describing how components in LMs (e.g., MLPs or attention heads) coordinate to enable this generalization. The structures consist of informative components that store training facts as weight changes, and upstream and downstream extractive components that query and process the stored information to produce the correct implication. We hypothesize that extractive structures are learned during pretraining when encountering implications of previously known facts. This yields two predictions: a data ordering effect where extractive structures can be learned only if facts precede their implications, and a weight grafting effect where extractive structures can be transferred to predict counterfactual implications. We empirically demonstrate these phenomena in the OLMo-7b, Llama 3-8b, Gemma 2-9b, and Qwen 2-7b models. Of independent interest, our results also indicate that fact learning can occur at both early and late layers, which lead to different forms of generalization.
>
---
#### [replaced 023] GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15810v2](http://arxiv.org/pdf/2505.15810v2)**

> **作者:** Yuqi Zhou; Sunhao Dai; Shuai Wang; Kaiwen Zhou; Qinglin Jia; Jun Xu
>
> **摘要:** Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm, coupling online Reinforcement Learning (RL) with explicit chain-of-thought reasoning prior to object grounding and thereby achieving substantial performance gains. In this paper, we first conduct extensive analysis experiments of three key components of that training pipeline: input design, output evaluation, and policy update-each revealing distinct challenges arising from blindly applying general-purpose RL without adapting to GUI grounding tasks. Input design: Current templates encourage the model to generate chain-of-thought reasoning, but longer chains unexpectedly lead to worse grounding performance. Output evaluation: Reward functions based on hit signals or box area allow models to exploit box size, leading to reward hacking and poor localization quality. Policy update: Online RL tends to overfit easy examples due to biases in length and sample difficulty, leading to under-optimization on harder cases. To address these issues, we propose three targeted solutions. First, we adopt a Fast Thinking Template that encourages direct answer generation, reducing excessive reasoning during training. Second, we incorporate a box size constraint into the reward function to mitigate reward hacking. Third, we revise the RL objective by adjusting length normalization and adding a difficulty-aware scaling factor, enabling better optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on ScreenSpot-Pro. This surpasses all prior models of similar size and even outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI agent grounding. The project repository is available at https://github.com/Yuqi-Zhou/GUI-G1.
>
---
#### [replaced 024] SMARTe: Slot-based Method for Accountable Relational Triple extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12816v2](http://arxiv.org/pdf/2504.12816v2)**

> **作者:** Xue Wen Tan; Stanley Kok
>
> **摘要:** Relational Triple Extraction (RTE) is a fundamental task in Natural Language Processing (NLP). However, prior research has primarily focused on optimizing model performance, with limited efforts to understand the internal mechanisms driving these models. Many existing methods rely on complex preprocessing to induce specific interactions, often resulting in opaque systems that may not fully align with their theoretical foundations. To address these limitations, we propose SMARTe: a Slot-based Method for Accountable Relational Triple extraction. SMARTe introduces intrinsic interpretability through a slot attention mechanism and frames the task as a set prediction problem. Slot attention consolidates relevant information into distinct slots, ensuring all predictions can be explicitly traced to learned slot representations and the tokens contributing to each predicted relational triple. While emphasizing interpretability, SMARTe achieves performance comparable to state-of-the-art models. Evaluations on the NYT and WebNLG datasets demonstrate that adding interpretability does not compromise performance. Furthermore, we conducted qualitative assessments to showcase the explanations provided by SMARTe, using attention heatmaps that map to their respective tokens. We conclude with a discussion of our findings and propose directions for future research.
>
---
#### [replaced 025] Evaluating LLM-based Approaches to Legal Citation Prediction: Domain-specific Pre-training, Fine-tuning, or RAG? A Benchmark and an Australian Law Case Study
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.06272v2](http://arxiv.org/pdf/2412.06272v2)**

> **作者:** Jiuzhou Han; Paul Burgess; Ehsan Shareghi
>
> **备注:** For code, data, and models see https://auslawbench.github.io
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong potential across legal tasks, yet the problem of legal citation prediction remains under-explored. At its core, this task demands fine-grained contextual understanding and precise identification of relevant legislation or precedent. We introduce the AusLaw Citation Benchmark, a real-world dataset comprising 55k Australian legal instances and 18,677 unique citations which to the best of our knowledge is the first of its scale and scope. We then conduct a systematic benchmarking across a range of solutions: (i) standard prompting of both general and law-specialised LLMs, (ii) retrieval-only pipelines with both generic and domain-specific embeddings, (iii) supervised fine-tuning, and (iv) several hybrid strategies that combine LLMs with retrieval augmentation through query expansion, voting ensembles, or re-ranking. Results show that neither general nor law-specific LLMs suffice as stand-alone solutions, with performance near zero. Instruction tuning (of even a generic open-source LLM) on task-specific dataset is among the best performing solutions. We highlight that database granularity along with the type of embeddings play a critical role in retrieval-based approaches, with hybrid methods which utilise a trained re-ranker delivering the best results. Despite this, a performance gap of nearly 50% remains, underscoring the value of this challenging benchmark as a rigorous test-bed for future research in legal-domain.
>
---
#### [replaced 026] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10063v2](http://arxiv.org/pdf/2504.10063v2)**

> **作者:** Alexandra Bazarova; Aleksandr Yugay; Andrey Shulga; Alina Ermilova; Andrei Volodichev; Konstantin Polev; Julia Belikova; Rauf Parchiev; Dmitry Simakov; Maxim Savchenko; Andrey Savchenko; Serguei Barannikov; Alexey Zaytsev
>
> **摘要:** Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
>
---
#### [replaced 027] CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15132v3](http://arxiv.org/pdf/2502.15132v3)**

> **作者:** Vignesh Kothapalli; Hamed Firooz; Maziar Sanjabi
>
> **备注:** ACL Main 2025
>
> **摘要:** We introduce CoT-ICL Lab, a framework and methodology to generate synthetic tokenized datasets and systematically study chain-of-thought (CoT) in-context learning (ICL) in language models. CoT-ICL Lab allows fine grained control over the complexity of in-context examples by decoupling (1) the causal structure involved in chain token generation from (2) the underlying token processing functions. We train decoder-only transformers (up to 700M parameters) on these datasets and show that CoT accelerates the accuracy transition to higher values across model sizes. In particular, we find that model depth is crucial for leveraging CoT with limited in-context examples, while more examples help shallow models match deeper model performance. Additionally, limiting the diversity of token processing functions throughout training improves causal structure learning via ICL. We also interpret these transitions by analyzing transformer embeddings and attention maps. Overall, CoT-ICL Lab serves as a simple yet powerful testbed for theoretical and empirical insights into ICL and CoT in language models.
>
---
#### [replaced 028] Do different prompting methods yield a common task representation in language models?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12075v2](http://arxiv.org/pdf/2505.12075v2)**

> **作者:** Guy Davidson; Todd M. Gureckis; Brenden M. Lake; Adina Williams
>
> **备注:** 9 pages, 4 figures; under review
>
> **摘要:** Demonstrations and instructions are two primary approaches for prompting language models to perform in-context learning (ICL) tasks. Do identical tasks elicited in different ways result in similar representations of the task? An improved understanding of task representation mechanisms would offer interpretability insights and may aid in steering models. We study this through \textit{function vectors} (FVs), recently proposed as a mechanism to extract few-shot ICL task representations. We generalize FVs to alternative task presentations, focusing on short textual instruction prompts, and successfully extract instruction function vectors that promote zero-shot task accuracy. We find evidence that demonstration- and instruction-based function vectors leverage different model components, and offer several controls to dissociate their contributions to task performance. Our results suggest that different task promptings forms do not induce a common task representation through FVs but elicit different, partly overlapping mechanisms. Our findings offer principled support to the practice of combining instructions and task demonstrations, imply challenges in universally monitoring task inference across presentation forms, and encourage further examinations of LLM task inference mechanisms.
>
---
#### [replaced 029] Transformers for molecular property prediction: Domain adaptation efficiently improves performance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03360v3](http://arxiv.org/pdf/2503.03360v3)**

> **作者:** Afnan Sultan; Max Rausch-Dupont; Shahrukh Khan; Olga Kalinina; Dietrich Klakow; Andrea Volkamer
>
> **摘要:** Over the past six years, molecular transformer models have become key tools in drug discovery. Most existing models are pre-trained on large, unlabeled datasets such as ZINC or ChEMBL. However, the extent to which large-scale pre-training improves molecular property prediction remains unclear. This study evaluates transformer models for this task while addressing their limitations. We explore how pre-training dataset size and chemically informed objectives impact performance. Our results show that increasing the dataset beyond approximately 400K to 800K molecules from large-scale unlabeled databases does not enhance performance across seven datasets covering five ADME endpoints: lipophilicity, permeability, solubility (two datasets), microsomal stability (two datasets), and plasma protein binding. In contrast, domain adaptation on a small, domain-specific dataset (less than or equal 4K molecules) using multi-task regression of physicochemical properties significantly boosts performance (P-value less than 0.001). A model pre-trained on 400K molecules and adapted with domain-specific data outperforms larger models such as MolFormer and performs comparably to MolBERT. Benchmarks against Random Forest (RF) baselines using descriptors and Morgan fingerprints show that chemically and physically informed features consistently yield better performance across model types. While RF remains a strong baseline, we identify concrete practices to enhance transformer performance. Aligning pre-training and adaptation with chemically meaningful tasks and domain-relevant data presents a promising direction for molecular property prediction. Our models are available on HuggingFace for easy use and adaptation.
>
---
#### [replaced 030] Graph-based Confidence Calibration for Large Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02454v2](http://arxiv.org/pdf/2411.02454v2)**

> **作者:** Yukun Li; Sijia Wang; Lifu Huang; Li-Ping Liu
>
> **摘要:** Reliable confidence estimation is essential for enhancing the trustworthiness of large language models (LLMs), especially in high-stakes scenarios. Despite its importance, accurately estimating confidence in LLM responses remains a significant challenge. In this work, we propose using an auxiliary learning model to assess response correctness based on the self-consistency of multiple outputs generated by the LLM. Our method builds a consistency graph to represent the agreement among multiple responses and uses a graph neural network (GNN) to estimate the likelihood that each response is correct. Experiments demonstrate that this method has strong calibration performance on various benchmark datasets and generalizes well to out-of-domain cases.
>
---
#### [replaced 031] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15298v2](http://arxiv.org/pdf/2505.15298v2)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Hongyi Wang; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by \textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.
>
---
#### [replaced 032] DocFusion: A Unified Framework for Document Parsing Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12505v2](http://arxiv.org/pdf/2412.12505v2)**

> **作者:** Mingxu Chai; Ziyu Shen; Chong Zhang; Yue Zhang; Xiao Wang; Shihan Dou; Jihua Kang; Jiazheng Zhang; Qi Zhang
>
> **摘要:** Document parsing is essential for analyzing complex document structures and extracting fine-grained information, supporting numerous downstream applications. However, existing methods often require integrating multiple independent models to handle various parsing tasks, leading to high complexity and maintenance overhead. To address this, we propose DocFusion, a lightweight generative model with only 0.28B parameters. It unifies task representations and achieves collaborative training through an improved objective function. Experiments reveal and leverage the mutually beneficial interaction among recognition tasks, and integrating recognition data significantly enhances detection performance. The final results demonstrate that DocFusion achieves state-of-the-art (SOTA) performance across four key tasks.
>
---
#### [replaced 033] M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11824v2](http://arxiv.org/pdf/2502.11824v2)**

> **作者:** Chengyan Wu; Bolei Ma; Yihong Liu; Zheyu Zhang; Ningyuan Deng; Yanshu Li; Baolan Chen; Yi Zhang; Yun Xue; Barbara Plank
>
> **摘要:** Aspect-based sentiment analysis (ABSA) is a crucial task in information extraction and sentiment analysis, aiming to identify aspects with associated sentiment elements in text. However, existing ABSA datasets are predominantly English-centric, limiting the scope for multilingual evaluation and research. To bridge this gap, we present M-ABSA, a comprehensive dataset spanning 7 domains and 21 languages, making it the most extensive multilingual parallel dataset for ABSA to date. Our primary focus is on triplet extraction, which involves identifying aspect terms, aspect categories, and sentiment polarities. The dataset is constructed through an automatic translation process with human review to ensure quality. We perform extensive experiments using various baselines to assess performance and compatibility on M-ABSA. Our empirical findings highlight that the dataset enables diverse evaluation tasks, such as multilingual and multi-domain transfer learning, and large language model evaluation, underscoring its inclusivity and its potential to drive advancements in multilingual ABSA research.
>
---
#### [replaced 034] Prompt-Guided Internal States for Hallucination Detection of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.04847v3](http://arxiv.org/pdf/2411.04847v3)**

> **作者:** Fujie Zhang; Peiqi Yu; Biao Yi; Baolei Zhang; Tong Li; Zheli Liu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a variety of tasks in different domains. However, they sometimes generate responses that are logically coherent but factually incorrect or misleading, which is known as LLM hallucinations. Data-driven supervised methods train hallucination detectors by leveraging the internal states of LLMs, but detectors trained on specific domains often struggle to generalize well to other domains. In this paper, we aim to enhance the cross-domain performance of supervised detectors with only in-domain data. We propose a novel framework, prompt-guided internal states for hallucination detection of LLMs, namely PRISM. By utilizing appropriate prompts to guide changes to the structure related to text truthfulness in LLMs' internal states, we make this structure more salient and consistent across texts from different domains. We integrated our framework with existing hallucination detection methods and conducted experiments on datasets from different domains. The experimental results indicate that our framework significantly enhances the cross-domain generalization of existing hallucination detection methods.
>
---
#### [replaced 035] Large Language Models are Miscalibrated In-Context Learners
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.13772v3](http://arxiv.org/pdf/2312.13772v3)**

> **作者:** Chengzu Li; Han Zhou; Goran Glavaš; Anna Korhonen; Ivan Vulić
>
> **备注:** 9 pages, 4 figures, 5 tables (20 pages, 5 figures, 13 tables including references and appendices)
>
> **摘要:** When adapting ICL with or without fine-tuning, we are curious about whether the instruction-tuned language model is able to achieve well-calibrated results without suffering from the problem of overconfidence (i.e., miscalibration) considering its strong instruction following ability, especially in such limited data setups. In this work, we deliver an in-depth analysis of the behavior across different choices of learning methods from the perspective of both performance and calibration. Through extensive controlled experiments, we observe that the miscalibration problem exists across all learning methods in low-resource setups. To achieve simultaneous gain for both in-task performance and calibration, we then study the potential of self-ensembling applied at different modeling stages (e.g., variations of in-context examples or variations in prompts or different ensembling strategies) to make the predictions more calibrated and have comparable or even better performance. We find that self-ensembling with max probability produces robust and calibrated predictions. Our work reveals the potential calibration problem of using ICL despite the improvements in task performance and sheds light on which learning paradigm to choose. We also provide practical guidelines for choosing learning paradigms depending on whether the data has been seen by the model before and a worthwhile solution via self-ensembling on how to enhance both task performance and calibration of LMs, which we hope could encourage further study.
>
---
#### [replaced 036] More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.15966v3](http://arxiv.org/pdf/2408.15966v3)**

> **作者:** Yuan Tang; Xu Han; Xianzhi Li; Qiao Yu; Jinfeng Xu; Yixue Hao; Long Hu; Min Chen
>
> **摘要:** Enabling Large Language Models (LLMs) to comprehend the 3D physical world remains a significant challenge. Due to the lack of large-scale 3D-text pair datasets, the success of LLMs has yet to be replicated in 3D understanding. In this paper, we rethink this issue and propose a new task: 3D Data-Efficient Point-Language Understanding. The goal is to enable LLMs to achieve robust 3D object understanding with minimal 3D point cloud and text data pairs. To address this task, we introduce GreenPLM, which leverages more text data to compensate for the lack of 3D data. First, inspired by using CLIP to align images and text, we utilize a pre-trained point cloud-text encoder to map the 3D point cloud space to the text space. This mapping leaves us to seamlessly connect the text space with LLMs. Once the point-text-LLM connection is established, we further enhance text-LLM alignment by expanding the intermediate text space, thereby reducing the reliance on 3D point cloud data. Specifically, we generate 6M free-text descriptions of 3D objects, and design a three-stage training strategy to help LLMs better explore the intrinsic connections between different modalities. To achieve efficient modality alignment, we design a zero-parameter cross-attention module for token pooling. Extensive experimental results show that GreenPLM requires only 12% of the 3D training data used by existing state-of-the-art models to achieve superior 3D understanding. Remarkably, GreenPLM also achieves competitive performance using text-only data. The code and weights are available at: https://github.com/TangYuan96/GreenPLM.
>
---
#### [replaced 037] BenCzechMark : A Czech-centric Multitask and Multimetric Benchmark for Large Language Models with Duel Scoring Mechanism
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.17933v2](http://arxiv.org/pdf/2412.17933v2)**

> **作者:** Martin Fajcik; Martin Docekal; Jan Dolezal; Karel Ondrej; Karel Beneš; Jan Kapsa; Pavel Smrz; Alexander Polok; Michal Hradis; Zuzana Neverilova; Ales Horak; Radoslav Sabol; Michal Stefanik; Adam Jirkovsky; David Adamczyk; Petr Hyner; Jan Hula; Hynek Kydlicek
>
> **备注:** Accepted to TACL
>
> **摘要:** We present BenCzechMark (BCM), the first comprehensive Czech language benchmark designed for large language models, offering diverse tasks, multiple task formats, and multiple evaluation metrics. Its duel scoring system is grounded in statistical significance theory and uses aggregation across tasks inspired by social preference theory. Our benchmark encompasses 50 challenging tasks, with corresponding test datasets, primarily in native Czech, with 14 newly collected ones. These tasks span 8 categories and cover diverse domains, including historical Czech news, essays from pupils or language learners, and spoken word. Furthermore, we collect and clean BUT-Large Czech Collection, the largest publicly available clean Czech language corpus, and use it for (i) contamination analysis and (ii) continuous pretraining of the first Czech-centric 7B language model with Czech-specific tokenization. We use our model as a baseline for comparison with publicly available multilingual models. Lastly, we release and maintain a leaderboard with existing 50 model submissions, where new model submissions can be made at https://huggingface.co/spaces/CZLC/BenCzechMark.
>
---
#### [replaced 038] GRIFFIN: Effective Token Alignment for Faster Speculative Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11018v2](http://arxiv.org/pdf/2502.11018v2)**

> **作者:** Shijing Hu; Jingyang Li; Xingyu Xie; Zhihui Lu; Kim-Chuan Toh; Pan Zhou
>
> **摘要:** Speculative decoding accelerates inference in large language models (LLMs) by generating multiple draft tokens simultaneously. However, existing methods often struggle with token misalignment between the training and decoding phases, limiting their performance. To address this, we propose GRIFFIN, a novel framework that incorporates a token-alignable training strategy and a token-alignable draft model to mitigate misalignment. The training strategy employs a loss masking mechanism to exclude highly misaligned tokens during training, preventing them from negatively impacting the draft model's optimization. The token-alignable draft model introduces input tokens to correct inconsistencies in generated features. Experiments on LLaMA, Vicuna, Qwen and Mixtral models demonstrate that GRIFFIN achieves an average acceptance length improvement of over 8% and a speedup ratio exceeding 7%, outperforming current speculative decoding state-of-the-art methods. Our code and GRIFFIN's draft models are released publicly in https://github.com/hsj576/GRIFFIN.
>
---
#### [replaced 039] EntGPT: Entity Linking with Generative Large Language Models
- **分类: cs.CL; H.3.3**

- **链接: [http://arxiv.org/pdf/2402.06738v3](http://arxiv.org/pdf/2402.06738v3)**

> **作者:** Yifan Ding; Amrit Poudel; Qingkai Zeng; Tim Weninger; Balaji Veeramani; Sanmitra Bhattacharya
>
> **摘要:** Entity Linking in natural language processing seeks to match text entities to their corresponding entries in a dictionary or knowledge base. Traditional approaches rely on contextual models, which can be complex, hard to train, and have limited transferability across different domains. Generative large language models like GPT offer a promising alternative but often underperform with naive prompts. In this study, we introduce EntGPT, employing advanced prompt engineering to enhance EL tasks. Our three-step hard-prompting method (EntGPT-P) significantly boosts the micro-F_1 score by up to 36% over vanilla prompts, achieving competitive performance across 10 datasets without supervised fine-tuning. Additionally, our instruction tuning method (EntGPT-I) improves micro-F_1 scores by 2.1% on average in supervised EL tasks and outperforms several baseline models in six Question Answering tasks. Our methods are compatible with both open-source and proprietary LLMs. All data and code are available on GitHub at https://github.com/yifding/In_Context_EL.
>
---
#### [replaced 040] LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06139v2](http://arxiv.org/pdf/2502.06139v2)**

> **作者:** Sumin An; Junyoung Sung; Wonpyo Park; Chanjun Park; Paul Hongsuck Seo
>
> **备注:** Accepted to NAACL 2025. Project Page: https://ssuminan.github.io/LCIRC/
>
> **摘要:** While large language models (LLMs) excel in generating coherent and contextually rich outputs, their capacity to efficiently handle long-form contexts is limited by fixed-length position embeddings. Additionally, the computational cost of processing long sequences increases quadratically, making it challenging to extend context length. To address these challenges, we propose Long-form Context Injection with Recurrent Compression (LCIRC), a method that enables the efficient processing long-form sequences beyond the model's length limit through recurrent compression without retraining the entire model. We further introduce query dependent context modeling, which selectively compresses query-relevant information, ensuring that the model retains the most pertinent content. Our empirical results demonstrate that Query Dependent LCIRC (QD-LCIRC) significantly improves LLM's ability to manage extended contexts, making it well-suited for tasks that require both comprehensive context understanding and query relevance.
>
---
#### [replaced 041] TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14625v2](http://arxiv.org/pdf/2505.14625v2)**

> **作者:** Zhangchen Xu; Yuetai Li; Fengqing Jiang; Bhaskar Ramasubramanian; Luyao Niu; Bill Yuchen Lin; Radha Poovendran
>
> **摘要:** Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at https://github.com/uw-nsl/TinyV.
>
---
#### [replaced 042] LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05423v3](http://arxiv.org/pdf/2505.05423v3)**

> **作者:** Ran Zhang; Wei Zhao; Lieve Macken; Steffen Eger
>
> **备注:** Updated version, with examples in the appendix
>
> **摘要:** The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation as being superior to human translation from experienced professionals. In the long run, this bias could result in an irreversible decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LiTransProQA, a novel, reference-free, LLM-based question-answering framework designed for literary translation evaluation. LiTransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LiTransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LiTransProQA reaches human-level evaluation performance comparable to trained student evaluators. It shows broad applicability to open-source models like LLaMa3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free tool for evaluating literary translations that require local processing due to copyright or ethical considerations. The code and datasets are available under: https://github.com/zhangr2021/TransProQA.
>
---
#### [replaced 043] The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.09674v3](http://arxiv.org/pdf/2502.09674v3)**

> **作者:** Wenbo Pan; Zhichao Liu; Qiguang Chen; Xiangyang Zhou; Haining Yu; Xiaohua Jia
>
> **备注:** Code and artifacts: https://github.com/BMPixel/safety-residual-space Accepted by ICML 2025
>
> **摘要:** Large Language Models' safety-aligned behaviors, such as refusing harmful queries, can be represented by linear directions in activation space. Previous research modeled safety behavior with a single direction, limiting mechanistic understanding to an isolated safety feature. In this work, we discover that safety-aligned behavior is jointly controlled by multi-dimensional directions. Namely, we study the vector space of representation shifts during safety fine-tuning on Llama 3 8B for refusing jailbreaks. By studying orthogonal directions in the space, we first find that a dominant direction governs the model's refusal behavior, while multiple smaller directions represent distinct and interpretable features like hypothetical narrative and role-playing. We then measure how different directions promote or suppress the dominant direction, showing the important role of secondary directions in shaping the model's refusal representation. Finally, we demonstrate that removing certain trigger tokens in harmful queries can mitigate these directions to bypass the learned safety capability, providing new insights on understanding safety alignment vulnerability from a multi-dimensional perspective. Code and artifacts are available at https://github.com/BMPixel/safety-residual-space.
>
---
#### [replaced 044] Keys to Robust Edits: from Theoretical Insights to Practical Advances
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.09338v2](http://arxiv.org/pdf/2410.09338v2)**

> **作者:** Jianhao Yan; Futing Wang; Yun Luo; Yafu Li; Yue Zhang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) struggle with maintaining accurate knowledge due to conflicting/outdated parametric memories. While locate-and-edit methods address this, their reliance on models' internal representations leads to robustness failures in long-context reasoning and paraphrased queries. We identify a fundamental limitation of locate-and-edit methods: existing semantic keys (for memory localization) cannot simultaneously satisfy robustness (context-invariant activation) and specificity (precise knowledge discrimination). Through theoretical error-bound analysis, we establish formal criteria for effective editing. Our solution introduces \textit{Robust Edit Pathway (REP)}, a plug-and-play module that: (1) disentangles editing keys from native model representations; (2) dynamically adjusts keys via contrastive learning to achieve robustness-specificity balance. Extensive experiments across various editing methods (ROME/MEMIT/R-ROME/EMMET), existing LLMs (LLaMA2, QWen, Mistral), and datasets (CounterFact, ZsRE) show that REP improves success rate over robustness tests by up-to 66.4\% while maintaining the success rate unaffected. Our code can be found at https://github.com/ElliottYan/RobustKeyEdit .
>
---
#### [replaced 045] Normal forms in Virus Machines
- **分类: cs.CL; cs.FL; 68Q07 (Primary) 68Q10, 68R01 (Secondary); F.0; F.1.1**

- **链接: [http://arxiv.org/pdf/2409.03327v2](http://arxiv.org/pdf/2409.03327v2)**

> **作者:** A. Ramírez-de-Arellano; F. G. C. Cabarle; D. Orellana-Martín; M. J. Pérez-Jiménez
>
> **备注:** 24 pages, 14 figures
>
> **摘要:** In the present work, we further study the computational power of virus machines (VMs in short).VMs provide a computing paradigm inspired by the transmission and replication networks of viruses.VMs consist of process units (called hosts) structured by a directed graph whose arcs are called channels and an instruction graph that controls the transmissions of virus objects among hosts. The present work complements our understanding of the computing power of VMs by introducing normal forms; these expressions restrict the features in a given computing model.Some of the features that we restrict in our normal forms include (a) the number of hosts, (b) the number of instructions, and (c) the number of virus objects in each host. After we recall some known results on the computing power of VMs we give our series of normal forms, such as the size of the loops in the network, proving new characterisations of family of sets, such as finite sets, semilinear sets, or recursively enumerable sets (NRE).
>
---
#### [replaced 046] Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16674v2](http://arxiv.org/pdf/2503.16674v2)**

> **作者:** Molly Kennedy; Ayyoob Imani; Timo Spinde; Hinrich Schütze
>
> **摘要:** While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate seven widely used LLMs by prompting them to generate articles and analyze their ideological preferences via Socratic probing. By using our self-contained Socratic approach, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism.
>
---
#### [replaced 047] Is a Peeled Apple Still Red? Evaluating LLMs' Ability for Conceptual Combination with Property Type
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06086v2](http://arxiv.org/pdf/2502.06086v2)**

> **作者:** Seokwon Song; Taehyun Lee; Jaewoo Ahn; Jae Hyuk Sung; Gunhee Kim
>
> **备注:** NAACL 2025 Oral
>
> **摘要:** Conceptual combination is a cognitive process that merges basic concepts, enabling the creation of complex expressions. During this process, the properties of combination (e.g., the whiteness of a peeled apple) can be inherited from basic concepts, newly emerge, or be canceled. However, previous studies have evaluated a limited set of properties and have not examined the generative process. To address this gap, we introduce the Conceptual Combination with Property Type dataset (CCPT), which consists of 12.3K annotated triplets of noun phrases, properties, and property types. Using CCPT, we establish three types of tasks to evaluate LLMs for conceptual combination thoroughly. Our key findings are threefold: (1) Our automatic metric grading property emergence and cancellation closely corresponds with human judgments. (2) LLMs, including OpenAI's o1, struggle to generate noun phrases which possess given emergent properties. (3) Our proposed method, inspired by cognitive psychology model that explains how relationships between concepts are formed, improves performances in all generative tasks. The dataset and experimental code are available at https://github.com/seokwon99/CCPT.git.
>
---
#### [replaced 048] Improving Multilingual Capabilities with Cultural and Local Knowledge in Large Language Models While Enhancing Native Performance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09753v2](http://arxiv.org/pdf/2504.09753v2)**

> **作者:** Ram Mohan Rao Kadiyala; Siddartha Pullakhandam; Siddhant Gupta; Drishti Sharma; Jebish Purbey; Kanwal Mehreen; Muhammad Arham; Hamza Farooq
>
> **备注:** 24 pages, 18 figures
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities, but their development has primarily focused on English and other high-resource languages, leaving many languages underserved. We present our latest Hindi-English bi-lingual LLM \textbf{Mantra-14B} with ~3\% average improvement in benchmark scores over both languages, outperforming models twice its size. Using a curated dataset composed of English and Hindi instruction data of 485K samples, we instruction tuned models such as Qwen-2.5-14B-Instruct and Phi-4 to improve performance over both English and Hindi. Our experiments encompassing seven different LLMs of varying parameter sizes and over 140 training attempts with varying English-Hindi training data ratios demonstrated that it is possible to significantly improve multilingual performance without compromising native performance. Further, our approach avoids resource-intensive techniques like vocabulary expansion or architectural modifications, thus keeping the model size small. Our results indicate that modest fine-tuning with culturally and locally informed data can bridge performance gaps without incurring significant computational overhead. We release our training code, datasets, and models under mit and apache licenses to aid further research towards under-represented and low-resource languages.
>
---
#### [replaced 049] Similarity-Distance-Magnitude Universal Verification
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20167v3](http://arxiv.org/pdf/2502.20167v3)**

> **作者:** Allen Schmaltz
>
> **备注:** 36 pages (1 Figure, 8 Tables, 4 Algorithms, 5 Listings)
>
> **摘要:** We address the neural network robustness problem by adding Similarity (i.e., correctly predicted depth-matches into training)-awareness and Distance-to-training-distribution-awareness to the existing output Magnitude (i.e., decision-boundary)-awareness of the softmax function. The resulting SDM activation function provides strong signals of the relative epistemic (reducible) predictive uncertainty. We use this novel behavior to further address the complementary HCI problem of mapping the output to human-interpretable summary statistics over relevant partitions of a held-out calibration set. Estimates of prediction-conditional uncertainty are obtained via a parsimonious learned transform over the class-conditional empirical CDFs of the output of a final-layer SDM activation function. For decision-making and as an intrinsic model check, estimates of class-conditional accuracy are obtained by further partitioning the high-probability regions of this calibrated output into class-conditional, region-specific CDFs. The uncertainty estimates from SDM calibration are remarkably robust to test-time distribution shifts and out-of-distribution inputs; incorporate awareness of the effective sample size; provide estimates of uncertainty from the learning and data splitting processes; and are well-suited for selective classification and conditional branching for additional test-time compute based on the predictive uncertainty, as for selective LLM generation, routing, and composition over multiple models and retrieval. Finally, we construct SDM networks, LLMs with uncertainty-aware verification and interpretability-by-exemplar as intrinsic properties. We provide open-source software implementing these results.
>
---
#### [replaced 050] PhysicsArena: The First Multimodal Physics Reasoning Benchmark Exploring Variable, Process, and Solution Dimensions
- **分类: cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.15472v2](http://arxiv.org/pdf/2505.15472v2)**

> **作者:** Song Dai; Yibo Yan; Jiamin Su; Dongfang Zihao; Yubo Gao; Yonghua Hei; Jungang Li; Junyan Zhang; Sicheng Tao; Zhuoran Gao; Xuming Hu
>
> **备注:** Under Review
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in diverse reasoning tasks, yet their application to complex physics reasoning remains underexplored. Physics reasoning presents unique challenges, requiring grounding in physical conditions and the interpretation of multimodal information. Current physics benchmarks are limited, often focusing on text-only inputs or solely on problem-solving, thereby overlooking the critical intermediate steps of variable identification and process formulation. To address these limitations, we introduce PhysicsArena, the first multimodal physics reasoning benchmark designed to holistically evaluate MLLMs across three critical dimensions: variable identification, physical process formulation, and solution derivation. PhysicsArena aims to provide a comprehensive platform for assessing and advancing the multimodal physics reasoning abilities of MLLMs.
>
---
#### [replaced 051] LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.12459v2](http://arxiv.org/pdf/2412.12459v2)**

> **作者:** Chia-Hsuan Chang; Jui-Tse Tsai; Yi-Hang Tsai; San-Yih Hwang
>
> **备注:** Accepted to PAKDD 2025
>
> **摘要:** Topic modeling is widely used for uncovering thematic structures within text corpora, yet traditional models often struggle with specificity and coherence in domain-focused applications. Guided approaches, such as SeededLDA and CorEx, incorporate user-provided seed words to improve relevance but remain labor-intensive and static. Large language models (LLMs) offer potential for dynamic topic refinement and discovery, yet their application often incurs high API costs. To address these challenges, we propose the LLM-assisted Iterative Topic Augmentation framework (LITA), an LLM-assisted approach that integrates user-provided seeds with embedding-based clustering and iterative refinement. LITA identifies a small number of ambiguous documents and employs an LLM to reassign them to existing or new topics, minimizing API costs while enhancing topic quality. Experiments on two datasets across topic quality and clustering performance metrics demonstrate that LITA outperforms five baseline models, including LDA, SeededLDA, CorEx, BERTopic, and PromptTopic. Our work offers an efficient and adaptable framework for advancing topic modeling and text clustering.
>
---
#### [replaced 052] APE-Bench I: Towards File-level Automated Proof Engineering of Formal Math Libraries
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.19110v2](http://arxiv.org/pdf/2504.19110v2)**

> **作者:** Huajian Xin; Luming Li; Xiaoran Jin; Jacques Fleuriot; Wenda Li
>
> **摘要:** Recent progress in large language models (LLMs) has shown promise in formal theorem proving, yet existing benchmarks remain limited to isolated, static proof tasks, failing to capture the iterative, engineering-intensive workflows of real-world formal mathematics libraries. Motivated by analogous advances in software engineering, we introduce the paradigm of Automated Proof Engineering (APE), which aims to automate proof engineering tasks such as feature addition, proof refactoring, and bug fixing using LLMs. To facilitate research in this direction, we present APE-Bench I, the first realistic benchmark built from real-world commit histories of Mathlib4, featuring diverse file-level tasks described in natural language and verified via a hybrid approach combining the Lean compiler and LLM-as-a-Judge. We further develop Eleanstic, a scalable parallel verification infrastructure optimized for proof checking across multiple versions of Mathlib. Empirical results on state-of-the-art LLMs demonstrate strong performance on localized edits but substantial degradation on handling complex proof engineering. This work lays the foundation for developing agentic workflows in proof engineering, with future benchmarks targeting multi-file coordination, project-scale verification, and autonomous agents capable of planning, editing, and repairing formal libraries.
>
---
#### [replaced 053] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10736v2](http://arxiv.org/pdf/2505.10736v2)**

> **作者:** Ximing Dong; Shaowei Wang; Dayi Lin; Ahmed E. Hassan
>
> **备注:** ACL 2025, Findings
>
> **摘要:** Optimizing Large Language Model (LLM) performance requires well-crafted prompts, but manual prompt engineering is labor-intensive and often ineffective. Automated prompt optimization techniques address this challenge but the majority of them rely on randomly selected evaluation subsets, which fail to represent the full dataset, leading to unreliable evaluations and suboptimal prompts. Existing coreset selection methods, designed for LLM benchmarking, are unsuitable for prompt optimization due to challenges in clustering similar samples, high data collection costs, and the unavailability of performance data for new or private datasets. To overcome these issues, we propose IPOMP, an Iterative evaluation data selection for effective Prompt Optimization using real-time Model Performance. IPOMP is a two-stage approach that selects representative and diverse samples using semantic clustering and boundary analysis, followed by iterative refinement with real-time model performance data to replace redundant samples. Evaluations on the BIG-bench dataset show that IPOMP improves effectiveness by 1.6% to 5.3% and stability by at least 57% compared with SOTA baselines, with minimal computational overhead below 1%. Furthermore, the results demonstrate that our real-time performance-guided refinement approach can be universally applied to enhance existing coreset selection methods.
>
---
#### [replaced 054] TTRL: Test-Time Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.16084v2](http://arxiv.org/pdf/2504.16084v2)**

> **作者:** Yuxin Zuo; Kaiyan Zhang; Li Sheng; Shang Qu; Ganqu Cui; Xuekai Zhu; Haozhan Li; Yuchen Zhang; Xinwei Long; Ermo Hua; Biqing Qi; Youbang Sun; Zhiyuan Ma; Lifan Yuan; Ning Ding; Bowen Zhou
>
> **摘要:** This paper investigates Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs). The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training. In this work, we introduce Test-Time Reinforcement Learning (TTRL), a novel method for training LLMs using RL on unlabeled data. TTRL enables self-evolution of LLMs by utilizing the priors in the pre-trained models. Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the pass@1 performance of Qwen-2.5-Math-7B by approximately 211% on the AIME 2024 with only unlabeled test data. Furthermore, although TTRL is only supervised by the maj@n metric, TTRL has demonstrated performance to consistently surpass the upper limit of the initial model maj@n, and approach the performance of models trained directly on test data with ground-truth labels. Our experimental findings validate the general effectiveness of TTRL across various tasks and highlight TTRL's potential for broader tasks and domains. GitHub: https://github.com/PRIME-RL/TTRL
>
---
#### [replaced 055] ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13176v2](http://arxiv.org/pdf/2505.13176v2)**

> **作者:** Zihao Cheng; Hongru Wang; Zeming Liu; Yuhang Guo; Yuanfang Guo; Yunhong Wang; Haifeng Wang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at https://github.com/Chengziha0/ToolSpectrum.
>
---
#### [replaced 056] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.13873v4](http://arxiv.org/pdf/2405.13873v4)**

> **作者:** Yuan Sui; Yufei He; Nian Liu; Xiaoxin He; Kun Wang; Bryan Hooi
>
> **备注:** This paper has been accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) are often challenged by generating erroneous or hallucinated responses, especially in complex reasoning tasks. Leveraging Knowledge Graphs (KGs) as external knowledge sources has emerged as a viable solution. However, existing KG-enhanced methods, either retrieval-based or agent-based, encounter difficulties in accurately retrieving knowledge and efficiently traversing KGs at scale. In this paper, we propose a unified framework, FiDeLiS, designed to improve the factuality of LLM responses by anchoring answers to verifiable reasoning steps retrieved from KGs. To achieve this, we leverage step-wise beam search with a deductive scoring function, allowing the LLM to validate reasoning process step by step, and halt the search once the question is deducible. In addition, we propose a Path-RAG module to pre-select a smaller candidate set for each beam search step, reducing computational costs by narrowing the search space. Extensive experiments show that our method, as a training-free framework, not only improve the performance but also enhance the factuality and interpretability across different benchmarks. Code is released at https://github.com/Y-Sui/FiDeLiS.
>
---
#### [replaced 057] SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09604v2](http://arxiv.org/pdf/2502.09604v2)**

> **作者:** Yung-Sung Chuang; Benjamin Cohen-Wang; Shannon Zejiang Shen; Zhaofeng Wu; Hu Xu; Xi Victoria Lin; James Glass; Shang-Wen Li; Wen-tau Yih
>
> **备注:** ICML 2025 main conference paper. The source code is available at https://github.com/facebookresearch/SelfCite
>
> **摘要:** We introduce SelfCite, a novel self-supervised approach that aligns LLMs to generate high-quality, fine-grained, sentence-level citations for the statements in their generated responses. Instead of only relying on costly and labor-intensive annotations, SelfCite leverages a reward signal provided by the LLM itself through context ablation: If a citation is necessary, removing the cited text from the context should prevent the same response; if sufficient, retaining the cited text alone should preserve the same response. This reward can guide the inference-time best-of-N sampling strategy to improve citation quality significantly, as well as be used in preference optimization to directly fine-tune the models for generating better citations. The effectiveness of SelfCite is demonstrated by increasing citation F1 up to 5.3 points on the LongBench-Cite benchmark across five long-form question answering tasks. The source code is available at https://github.com/facebookresearch/SelfCite
>
---
#### [replaced 058] My Words Imply Your Opinion: Reader Agent-based Propagation Enhancement for Personalized Implicit Emotion Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.07367v3](http://arxiv.org/pdf/2412.07367v3)**

> **作者:** Jian Liao; Yu Feng; Yujin Zheng; Jun Zhao; Suge Wang; Jianxing Zheng
>
> **摘要:** The subtlety of emotional expressions makes implicit emotion analysis (IEA) particularly sensitive to user-specific characteristics. Current studies personalize emotion analysis by focusing on the author but neglect the impact of the intended reader on implicit emotional feedback. In this paper, we introduce Personalized IEA (PIEA) and present the RAPPIE model, which addresses subjective variability by incorporating reader feedback. In particular, (1) we create reader agents based on large language models to simulate reader feedback, overcoming the issue of ``spiral of silence effect'' and data incompleteness of real reader reaction. (2) We develop a role-aware multi-view graph learning to model the emotion interactive propagation process in scenarios with sparse reader information. (3) We construct two new PIEA datasets covering English and Chinese social media with detailed user metadata, addressing the text-centric limitation of existing datasets. Extensive experiments show that RAPPIE significantly outperforms state-of-the-art baselines, demonstrating the value of incorporating reader feedback in PIEA.
>
---
#### [replaced 059] Model Merging in Pre-training of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12082v3](http://arxiv.org/pdf/2505.12082v3)**

> **作者:** Yunshui Li; Yiyuan Ma; Shen Yan; Chaoyi Zhang; Jing Liu; Jianqiao Lu; Ziwen Xu; Mengzhao Chen; Minrui Wang; Shiyi Zhan; Jin Ma; Xunhao Lai; Deyi Liu; Yao Luo; Xingyan Bin; Hongbin Ren; Mingji Han; Wenhao Hao; Bairen Yi; LingJun Liu; Bole Ma; Xiaoying Jia; Xun Zhou; Siyuan Qiao; Liang Xiang; Yonghui Wu
>
> **摘要:** Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored. In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process. Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters, we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior. These improvements lead to both more efficient model development and significantly lower training costs. Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications. Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging.
>
---
#### [replaced 060] PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13862v2](http://arxiv.org/pdf/2505.13862v2)**

> **作者:** Guobin Shen; Dongcheng Zhao; Linghao Feng; Xiang He; Jihang Wang; Sicheng Shen; Haibo Tong; Yiting Dong; Jindong Li; Xiang Zheng; Yi Zeng
>
> **摘要:** Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
>
---
#### [replaced 061] C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.06205v2](http://arxiv.org/pdf/2502.06205v2)**

> **作者:** Guoxin Chen; Minpeng Liao; Peiying Yu; Dingmin Wang; Zile Qiao; Chao Yang; Xin Zhao; Kai Fan
>
> **备注:** Camera ready version for ICML 2025
>
> **摘要:** Retrieval-augmented generation (RAG) systems face a fundamental challenge in aligning independently developed retrievers and large language models (LLMs). Existing approaches typically involve modifying either component or introducing simple intermediate modules, resulting in practical limitations and sub-optimal performance. Inspired by human search behavior -- typically involving a back-and-forth process of proposing search queries and reviewing documents, we propose C-3PO, a proxy-centric framework that facilitates communication between retrievers and LLMs through a lightweight multi-agent system. Our framework implements three specialized agents that collaboratively optimize the entire RAG pipeline without altering the retriever and LLMs. These agents work together to assess the need for retrieval, generate effective queries, and select information suitable for the LLMs. To enable effective multi-agent coordination, we develop a tree-structured rollout approach for reward credit assignment in reinforcement learning. Extensive experiments in both in-domain and out-of-distribution scenarios demonstrate that C-3PO significantly enhances RAG performance while maintaining plug-and-play flexibility and superior generalization capabilities.
>
---
#### [replaced 062] BlockPruner: Fine-grained Pruning for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.10594v4](http://arxiv.org/pdf/2406.10594v4)**

> **作者:** Longguang Zhong; Fanqi Wan; Ruijun Chen; Xiaojun Quan; Liangzhi Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** With the rapid growth in the size and complexity of large language models (LLMs), the costs associated with their training and inference have escalated significantly. Research indicates that certain layers in LLMs harbor substantial redundancy, and pruning these layers has minimal impact on the overall performance. While various layer pruning methods have been developed based on this insight, they generally overlook the finer-grained redundancies within the layers themselves. In this paper, we delve deeper into the architecture of LLMs and demonstrate that finer-grained pruning can be achieved by targeting redundancies in multi-head attention (MHA) and multi-layer perceptron (MLP) blocks. We propose a novel, training-free structured pruning approach called BlockPruner. Unlike existing layer pruning methods, BlockPruner segments each Transformer layer into MHA and MLP blocks. It then assesses the importance of these blocks using perplexity measures and applies a heuristic search for iterative pruning. We applied BlockPruner to LLMs of various sizes and architectures and validated its performance across a wide range of downstream tasks. Experimental results show that BlockPruner achieves more granular and effective pruning compared to state-of-the-art baselines.
>
---
#### [replaced 063] Optimizing Case-Based Reasoning System for Functional Test Script Generation with Large Language Models
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.20576v2](http://arxiv.org/pdf/2503.20576v2)**

> **作者:** Siyuan Guo; Huiwu Liu; Xiaolong Chen; Yuming Xie; Liang Zhang; Tao Han; Hechang Chen; Yi Chang; Jun Wang
>
> **备注:** Accepted by KDD 2025 (ADS Track)
>
> **摘要:** In this work, we explore the potential of large language models (LLMs) for generating functional test scripts, which necessitates understanding the dynamically evolving code structure of the target software. To achieve this, we propose a case-based reasoning (CBR) system utilizing a 4R cycle (i.e., retrieve, reuse, revise, and retain), which maintains and leverages a case bank of test intent descriptions and corresponding test scripts to facilitate LLMs for test script generation. To improve user experience further, we introduce Re4, an optimization method for the CBR system, comprising reranking-based retrieval finetuning and reinforced reuse finetuning. Specifically, we first identify positive examples with high semantic and script similarity, providing reliable pseudo-labels for finetuning the retriever model without costly labeling. Then, we apply supervised finetuning, followed by a reinforcement learning finetuning stage, to align LLMs with our production scenarios, ensuring the faithful reuse of retrieved cases. Extensive experimental results on two product development units from Huawei Datacom demonstrate the superiority of the proposed CBR+Re4. Notably, we also show that the proposed Re4 method can help alleviate the repetitive generation issues with LLMs.
>
---
#### [replaced 064] Do Robot Snakes Dream like Electric Sheep? Investigating the Effects of Architectural Inductive Biases on Hallucination
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.17477v5](http://arxiv.org/pdf/2410.17477v5)**

> **作者:** Jerry Huang; Prasanna Parthasarathi; Mehdi Rezagholizadeh; Boxing Chen; Sarath Chandar
>
> **备注:** Accepted to Findings of The 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025
>
> **摘要:** The growth in prominence of large language models (LLMs) in everyday life can be largely attributed to their generative abilities, yet some of this is also owed to the risks and costs associated with their use. On one front is their tendency to hallucinate false or misleading information, limiting their reliability. On another is the increasing focus on the computational limitations associated with traditional self-attention based LLMs, which has brought about new alternatives, in particular recurrent models, meant to overcome them. Yet it remains uncommon to consider these two concerns simultaneously. Do changes in architecture exacerbate/alleviate existing concerns about hallucinations? Do they affect how and where they occur? Through an extensive evaluation, we study how these architecture-based inductive biases affect the propensity to hallucinate. While hallucination remains a general phenomenon not limited to specific architectures, the situations in which they occur and the ease with which specific types of hallucinations can be induced can significantly differ based on the model architecture. These findings highlight the need for better understanding both these problems in conjunction with each other, as well as consider how to design more universal techniques for handling hallucinations.
>
---
#### [replaced 065] GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.11471v3](http://arxiv.org/pdf/2502.11471v3)**

> **作者:** Kangyang Luo; Yuzhuo Bai; Cheng Gao; Shuzheng Si; Yingli Shen; Zhu Liu; Zhitong Wang; Cunliang Kong; Wenhao Li; Yufei Huang; Ye Tian; Xuantang Xiong; Lei Han; Maosong Sun
>
> **备注:** Accepted by ACL2025(Findings)
>
> **摘要:** Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
>
---
#### [replaced 066] LangSAMP: Language-Script Aware Multilingual Pretraining
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.18199v2](http://arxiv.org/pdf/2409.18199v2)**

> **作者:** Yihong Liu; Haotian Ye; Chunlan Ma; Mingyang Wang; Hinrich Schütze
>
> **备注:** ACL 2025
>
> **摘要:** Recent multilingual pretrained language models (mPLMs) often avoid using language embeddings -- learnable vectors assigned to individual languages. However, this places a significant burden on token representations to encode all language-specific information, which may hinder language neutrality. To address this limitation, we propose Language-Script Aware Multilingual Pretraining (LangSAMP), a method that incorporates both language and script embeddings to enhance representation learning. Specifically, we integrate these embeddings into the output of the Transformer blocks before passing the final representations to the language modeling head for prediction. We apply LangSAMP to the continual pretraining of XLM-R on a highly multilingual corpus covering more than 500 languages. The resulting model consistently outperforms the baseline in zero-shot crosslingual transfer across diverse downstream tasks. Extensive analysis reveals that language and script embeddings capture language- and script-specific nuances, which benefits more language-neutral representations, proven by improved pairwise cosine similarity. In our case study, we also show that language and script embeddings can be used to select better source languages for crosslingual transfer. We make our code and models publicly available at https://github.com/cisnlp/LangSAMP.
>
---
#### [replaced 067] Determination of language families using deep learning
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2409.02393v2](http://arxiv.org/pdf/2409.02393v2)**

> **作者:** Peter B. Lerner
>
> **备注:** Second draft with improved statistics of NN simulations. Comments are welcome
>
> **摘要:** We use a c-GAN (convolutional generative adversarial) neural network to analyze transliterated text fragments of extant, dead comprehensible, and one dead non-deciphered (Cypro-Minoan) language to establish linguistic affinities. The paper is agnostic with respect to translation and/or deciphering. However, there is hope that the proposed approach can be useful for decipherment with more sophisticated neural network techniques.
>
---
#### [replaced 068] Adaptive Thinking via Mode Policy Optimization for Social Language Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02156v4](http://arxiv.org/pdf/2505.02156v4)**

> **作者:** Minzheng Wang; Yongbin Li; Haobo Wang; Xinghua Zhang; Nan Xu; Bingli Wu; Fei Huang; Haiyang Yu; Wenji Mao
>
> **备注:** Work in Progress. The code and data are available, see https://github.com/MozerWang/AMPO
>
> **摘要:** Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current studies. Existing methods either lack this kind of reasoning capability or enforce Long Chain-of-Thought reasoning uniformly across all scenarios, resulting in excessive token usage and inflexible social simulation. To address this, we propose an $\textbf{A}$daptive $\textbf{M}$ode $\textbf{L}$earning ($\textbf{AML}$) framework in this paper, aiming to improve the adaptive thinking ability of language agents in dynamic social interactions. To this end, we first identify hierarchical thinking modes ranging from intuitive response to deep deliberation based on the cognitive control theory. We then develop the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm to optimize the context-aware mode switching and reasoning. Our framework advances existing research in three key aspects: (1) Multi-granular thinking mode design, (2) Context-aware mode switching across social interaction, and (3) Token-efficient reasoning via depth-adaptive processing. Extensive experiments on social intelligence benchmarks verify that AML achieves 15.6% higher task performance than GPT-4o. Notably, our AMPO outperforms GRPO by 7.0% with 32.8% shorter reasoning chains, demonstrating the advantage of adaptive thinking mode selection and optimization mechanism in AMPO over GRPO's fixed-depth solution.
>
---
#### [replaced 069] Not All Correct Answers Are Equal: Why Your Distillation Source Matters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14464v2](http://arxiv.org/pdf/2505.14464v2)**

> **作者:** Xiaoyu Tian; Yunjie Ji; Haotian Wang; Shuaiting Chen; Sitong Zhao; Yiping Peng; Han Zhao; Xiangang Li
>
> **摘要:** Distillation has emerged as a practical and effective approach to enhance the reasoning capabilities of open-source language models. In this work, we conduct a large-scale empirical study on reasoning data distillation by collecting verified outputs from three state-of-the-art teacher models-AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1-on a shared corpus of 1.89 million queries. We construct three parallel datasets and analyze their distributions, revealing that AM-Thinking-v1-distilled data exhibits greater token length diversity and lower perplexity. Student models trained on each dataset are evaluated on reasoning benchmarks including AIME2024, AIME2025, MATH500, and LiveCodeBench. The model distilled from AM-Thinking-v1 consistently achieves the best performance (e.g., 84.3 on AIME2024, 72.2 on AIME2025, 98.4 on MATH500, and 65.9 on LiveCodeBench) and demonstrates adaptive output behavior-producing longer responses for harder tasks and shorter ones for simpler tasks. These findings highlight the value of high-quality, verified reasoning traces. We release the AM-Thinking-v1 and Qwen3-235B-A22B distilled datasets to support future research on open and high-performing reasoning-oriented language models. The datasets are publicly available on Hugging Face\footnote{Datasets are available on Hugging Face: \href{https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled}{AM-Thinking-v1-Distilled}, \href{https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled}{AM-Qwen3-Distilled}.}.
>
---
#### [replaced 070] Illusion or Algorithm? Investigating Memorization, Emergence, and Symbolic Processing in In-Context Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11004v2](http://arxiv.org/pdf/2505.11004v2)**

> **作者:** Jingcheng Niu; Subhabrata Dutta; Ahmed Elshabrawy; Harish Tayyar Madabushi; Iryna Gurevych
>
> **摘要:** Large-scale Transformer language models (LMs) trained solely on next-token prediction with web-scale data can solve a wide range of tasks after seeing just a few examples. The mechanism behind this capability, known as in-context learning (ICL), remains both controversial and poorly understood. Some studies argue that it is merely the result of memorizing vast amounts of data, while others contend that it reflects a fundamental, symbolic algorithmic development in LMs. In this work, we introduce a suite of investigative tasks and a novel method to systematically investigate ICL by leveraging the full Pythia scaling suite, including interim checkpoints that capture progressively larger amount of training data. By carefully exploring ICL performance on downstream tasks and simultaneously conducting a mechanistic analysis of the residual stream's subspace, we demonstrate that ICL extends beyond mere "memorization" of the training corpus, yet does not amount to the implementation of an independent symbolic algorithm. Our results also clarify several aspects of ICL, including the influence of training dynamics, model capabilities, and elements of mechanistic interpretability. Overall, our work advances the understanding of ICL and its implications, offering model developers insights into potential improvements and providing AI security practitioners with a basis for more informed guidelines.
>
---
#### [replaced 071] Whose story is it? Personalizing story generation by inferring author styles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13028v2](http://arxiv.org/pdf/2502.13028v2)**

> **作者:** Nischal Ashok Kumar; Chau Minh Pham; Mohit Iyyer; Andrew Lan
>
> **备注:** preprint:55 pages
>
> **摘要:** Personalization is critical for improving user experience in interactive writing and educational applications, yet remains understudied in story generation. We study the task of personalizing story generation, where our goal is to mimic an author's writing style, given other stories written by them. We collect Mythos, a dataset of 3.6k stories from 112 authors, with an average of 16 stories per author, across five distinct sources reflecting diverse story-writing settings. We propose a two-stage pipeline for personalized story generation: first, we infer authors' implicit writing characteristics and organize them into an Author Writing Sheet, which is validated by humans to be of high quality; second, we simulate the author's persona using tailored persona descriptions and personalized story rules. We find that stories personalized using the Author Writing Sheet outperform a non-personalized baseline, achieving a 78% win-rate in capturing authors' past style and 59% in similarity to ground-truth author stories. Human evaluation supports these findings and further highlights trends, such as Reddit stories being easier to personalize, and the Creativity and Language Use aspects of stories being easier to personalize than the Plot.
>
---
#### [replaced 072] Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15431v2](http://arxiv.org/pdf/2505.15431v2)**

> **作者:** Tencent Hunyuan Team; Ao Liu; Botong Zhou; Can Xu; Chayse Zhou; ChenChen Zhang; Chengcheng Xu; Chenhao Wang; Decheng Wu; Dengpeng Wu; Dian Jiao; Dong Du; Dong Wang; Feng Zhang; Fengzong Lian; Guanghui Xu; Guanwei Zhang; Hai Wang; Haipeng Luo; Han Hu; Huilin Xu; Jiajia Wu; Jianchen Zhu; Jianfeng Yan; Jiaqi Zhu; Jihong Zhang; Jinbao Xue; Jun Xia; Junqiang Zheng; Kai Liu; Kai Zhang; Kai Zheng; Kejiao Li; Keyao Wang; Lan Jiang; Lixin Liu; Lulu Wu; Mengyuan Huang; Peijie Yu; Peiqi Wang; Qian Wang; Qianbiao Xiang; Qibin Liu; Qingfeng Sun; Richard Guo; Ruobing Xie; Saiyong Yang; Shaohua Chen; Shihui Hu; Shuai Li; Shuaipeng Li; Shuang Chen; Suncong Zheng; Tao Yang; Tian Zhang; Tinghao Yu; Weidong Han; Weijie Liu; Weijin Zhou; Weikang Wang; Wesleye Chen; Xiao Feng; Xiaoqin Ren; Xingwu Sun; Xiong Kuang; Xuemeng Huang; Xun Cao; Yanfeng Chen; Yang Du; Yang Zhen; Yangyu Tao; Yaping Deng; Yi Shen; Yigeng Hong; Yiqi Chen; Yiqing Huang; Yuchi Deng; Yue Mao; Yulong Wang; Yuyuan Zeng; Zenan Xu; Zhanhui Kang; Zhe Zhao; ZhenXiang Yan; Zheng Fang; Zhichao Hu; Zhongzhi Chen; Zhuoyu Li; Zongwei Li; Alex Yan; Ande Liang; Baitong Liu; Beiping Pan; Bin Xing; Binghong Wu; Bingxin Qu; Bolin Ni; Boyu Wu; Chen Li; Cheng Jiang; Cheng Zhang; Chengjun Liu; Chengxu Yang; Chengzhong Xu; Chiyu Wang; Chong Zha; Daisy Yi; Di Wang; Fanyang Lu; Fei Chen; Feifei Liu; Feng Zheng; Guanghua Yu; Guiyang Li; Guohua Wang; Haisheng Lin; Han Liu; Han Wang; Hao Fei; Hao Lu; Haoqing Jiang; Haoran Sun; Haotian Zhu; Huangjin Dai; Huankui Chen; Huawen Feng; Huihui Cai; Huxin Peng; Jackson Lv; Jiacheng Shi; Jiahao Bu; Jianbo Li; Jianglu Hu; Jiangtao Guan; Jianing Xu; Jianwei Cai; Jiarong Zhang; Jiawei Song; Jie Jiang; Jie Liu; Jieneng Yang; Jihong Zhang; Jin lv; Jing Zhao; Jinjian Li; Jinxing Liu; Jun Zhao; Juntao Guo; Kai Wang; Kan Wu; Lei Fu; Lei He; Lei Wang; Li Liu; Liang Dong; Liya Zhan; Long Cheng; Long Xu; Mao Zheng; Meng Liu; Mengkang Hu; Nanli Chen; Peirui Chen; Peng He; Pengju Pan; Pengzhi Wei; Qi Yang; Qi Yi; Roberts Wang; Rongpeng Chen; Rui Sun; Rui Yang; Ruibin Chen; Ruixu Zhou; Shaofeng Zhang; Sheng Zhang; Shihao Xu; Shuaishuai Chang; Shulin Liu; SiQi Wang; Songjia Feng; Songling Yuan; Tao Zhang; Tianjiao Lang; Tongkai Li; Wei Deng; Wei Li; Weichao Wang; Weigang Zhang; Weixuan Sun; Wen Ouyang; Wenxiang Jiao; Wenzhi Sun; Wenzhuo Jia; Xiang Zhang; Xiangyu He; Xianshun Ren; XiaoYing Zhu; Xiaolong Guo; Xiaoxue Li; Xiaoyu Ma; Xican Lu; Xinhua Feng; Xinting Huang; Xinyu Guan; Xirui Li; Xu Zhang; Xudong Gao; Xun Luo; Xuxiang Qi; Yangkun Chen; Yangyu Tao; Yanling Xiao; Yantao Mai; Yanze Chen; Yao Ding; Yeting Yang; YiFan Song; Yifan Yang; Yijiao Zhu; Yinhe Wu; Yixian Liu; Yong Yang; Yuanjun Cai; Yuanlin Tu; Yue Zhang; Yufei Huang; Yuhang Zhou; Yuhao Jiang; Yuhong Liu; Yuhui Hu; Yujin Lin; Yun Yang; Yunhao Wang; Yusong Zhang; Zekun Wu; Zelong Zhang; Zhan Yu; Zhaoliang Yang; Zhe Zhao; Zheng Li; Zhenyu Huang; Zhiguang Liu; Zhijiang Xu; Zhiqing Kui; Zhiyin Zeng; Zhiyuan Xiong; Zhuo Han; Zifan Wu; Zigang Geng; Zilong Zhao; Ziyan Tang; Ziyuan Zhu; Zonglei Zhu; Zhijiang Xu
>
> **摘要:** As Large Language Models (LLMs) rapidly advance, we introduce Hunyuan-TurboS, a novel large hybrid Transformer-Mamba Mixture of Experts (MoE) model. It synergistically combines Mamba's long-sequence processing efficiency with Transformer's superior contextual understanding. Hunyuan-TurboS features an adaptive long-short chain-of-thought (CoT) mechanism, dynamically switching between rapid responses for simple queries and deep "thinking" modes for complex problems, optimizing computational resources. Architecturally, this 56B activated (560B total) parameter model employs 128 layers (Mamba2, Attention, FFN) with an innovative AMF/MF block pattern. Faster Mamba2 ensures linear complexity, Grouped-Query Attention minimizes KV cache, and FFNs use an MoE structure. Pre-trained on 16T high-quality tokens, it supports a 256K context length and is the first industry-deployed large-scale Mamba model. Our comprehensive post-training strategy enhances capabilities via Supervised Fine-Tuning (3M instructions), a novel Adaptive Long-short CoT Fusion method, Multi-round Deliberation Learning for iterative improvement, and a two-stage Large-scale Reinforcement Learning process targeting STEM and general instruction-following. Evaluations show strong performance: overall top 7 rank on LMSYS Chatbot Arena with a score of 1356, outperforming leading models like Gemini-2.0-Flash-001 (1352) and o4-mini-2025-04-16 (1345). TurboS also achieves an average of 77.9% across 23 automated benchmarks. Hunyuan-TurboS balances high performance and efficiency, offering substantial capabilities at lower inference costs than many reasoning models, establishing a new paradigm for efficient large-scale pre-trained models.
>
---
#### [replaced 073] Breaking Information Cocoons: A Hyperbolic Graph-LLM Framework for Exploration and Exploitation in Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.13865v3](http://arxiv.org/pdf/2411.13865v3)**

> **作者:** Qiyao Ma; Menglin Yang; Mingxuan Ju; Tong Zhao; Neil Shah; Rex Ying
>
> **摘要:** Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. A key challenge lies in balancing content exploration and exploitation while allowing users to adjust their recommendation preferences. Intuitively, this balance can be modeled as a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic graph-LLM framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a semantic-enhanced hierarchical mechanism that aligns rich textual descriptions processed by large language models (LLMs) with collaborative information directly in hyperbolic space, allowing for more nuanced updates that respect the underlying hierarchical structure in user-item profiles; (2) an automatic hierarchical representation by optimizing Dasgupta's cost, which discovers hierarchical structures without requiring predefined hyperparameters, enabling user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons. We open-source our model implementation at https://github.com/Martin-qyma/HERec.
>
---
#### [replaced 074] BAR: A Backward Reasoning based Agent for Complex Minecraft Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14079v2](http://arxiv.org/pdf/2505.14079v2)**

> **作者:** Weihong Du; Wenrui Liao; Binyu Yan; Hongru Liang; Anthony G. Cohn; Wenqiang Lei
>
> **摘要:** Large language model (LLM) based agents have shown great potential in following human instructions and automatically completing various tasks. To complete a task, the agent needs to decompose it into easily executed steps by planning. Existing studies mainly conduct the planning by inferring what steps should be executed next starting from the agent's initial state. However, this forward reasoning paradigm doesn't work well for complex tasks. We propose to study this issue in Minecraft, a virtual environment that simulates complex tasks based on real-world scenarios. We believe that the failure of forward reasoning is caused by the big perception gap between the agent's initial state and task goal. To this end, we leverage backward reasoning and make the planning starting from the terminal state, which can directly achieve the task goal in one step. Specifically, we design a BAckward Reasoning based agent (BAR). It is equipped with a recursive goal decomposition module, a state consistency maintaining module and a stage memory module to make robust, consistent, and efficient planning starting from the terminal state. Experimental results demonstrate the superiority of BAR over existing methods and the effectiveness of proposed modules.
>
---
#### [replaced 075] Steer LLM Latents for Hallucination Detection
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01917v2](http://arxiv.org/pdf/2503.01917v2)**

> **作者:** Seongheon Park; Xuefeng Du; Min-Hsuan Yeh; Haobo Wang; Yixuan Li
>
> **备注:** ICML 2025
>
> **摘要:** Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications.
>
---
#### [replaced 076] Vague Knowledge: Evidence from Analyst Reports
- **分类: econ.GN; cs.AI; cs.CL; math.LO; q-fin.EC; q-fin.GN; 03B48, 03B65, 03E02, 03E15, 03E72, 18E45, 28A05, 62F15, 68T01,
  68T35, 68T50, 91G30,; F.4; I.2.3; I.2.4; I.2.7; J.1; J.4; J.5**

- **链接: [http://arxiv.org/pdf/2505.12269v2](http://arxiv.org/pdf/2505.12269v2)**

> **作者:** Kerry Xiao; Amy Zang
>
> **摘要:** People in the real world often possess vague knowledge of future payoffs, for which quantification is not feasible or desirable. We argue that language, with differing ability to convey vague information, plays an important but less known-role in representing subjective expectations. Empirically, we find that in their reports, analysts include useful information in linguistic expressions but not numerical forecasts. Specifically, the textual tone of analyst reports has predictive power for forecast errors and subsequent revisions in numerical forecasts, and this relation becomes stronger when analyst's language is vaguer, when uncertainty is higher, and when analysts are busier. Overall, our theory and evidence suggest that some useful information is vaguely known and only communicated through language.
>
---
#### [replaced 077] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.07053v2](http://arxiv.org/pdf/2504.07053v2)**

> **作者:** Liang-Hsuan Tseng; Yi-Chang Chen; Kuan-Yi Lee; Da-Shan Shiu; Hung-yi Lee
>
> **备注:** Preprint
>
> **摘要:** Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human-LLM interaction. Joint speech-text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Experimental results show that TASTE-based SLMs perform comparable to previous work on SALMON and StoryCloze; while significantly outperform other pre-trained SLMs on speech continuation across subjective and objective evaluations. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and model are available at https://mtkresearch.github.io/TASTE-SpokenLM.github.io.
>
---
#### [replaced 078] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.02813v3](http://arxiv.org/pdf/2409.02813v3)**

> **作者:** Xiang Yue; Tianyu Zheng; Yuansheng Ni; Yubo Wang; Kai Zhang; Shengbang Tong; Yuxuan Sun; Botao Yu; Ge Zhang; Huan Sun; Yu Su; Wenhu Chen; Graham Neubig
>
> **备注:** ACL 2025 Main
>
> **摘要:** This paper introduces MMMU-Pro, a robust version of the Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU) benchmark. MMMU-Pro rigorously assesses multimodal models' true understanding and reasoning capabilities through a three-step process based on MMMU: (1) filtering out questions answerable by text-only models, (2) augmenting candidate options, and (3) introducing a vision-only input setting where questions are embedded within images. This setting challenges AI to truly "see" and "read" simultaneously, testing a fundamental human cognitive skill of seamlessly integrating visual and textual information. Results show that model performance is substantially lower on MMMU-Pro than on MMMU, ranging from 16.8% to 26.9% across models. We explore the impact of OCR prompts and Chain of Thought (CoT) reasoning, finding that OCR prompts have minimal effect while CoT generally improves performance. MMMU-Pro provides a more rigorous evaluation tool, closely mimicking real-world scenarios and offering valuable directions for future research in multimodal AI.
>
---
#### [replaced 079] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16965v2](http://arxiv.org/pdf/2503.16965v2)**

> **作者:** Zhe Hu; Jing Li; Zhongzhu Pu; Hou Pong Chan; Yu Yin
>
> **摘要:** Vision Language Models exhibited immense potential for embodied AI, yet they often lack the sophisticated situational reasoning required for complex decision-making. This paper shows that VLMs can achieve surprisingly strong decision-making performance when visual scenes are represented merely as text-only descriptions, suggesting foundational reasoning can be effectively learned from language. Motivated by this insight, we propose Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities, where models learn to evaluate actions and their consequences. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Experiments across diverse decision-making benchmarks demonstrate that Praxis-VLM substantially outperforms standard supervised fine-tuning, exhibiting superior performance and generalizability. Further analysis confirms that our models engage in explicit and effective reasoning, underpinning their enhanced performance and adaptability.
>
---
#### [replaced 080] Critique-Guided Distillation: Improving Supervised Fine-tuning via Better Distillation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11628v2](http://arxiv.org/pdf/2505.11628v2)**

> **作者:** Berkcan Kapusuzoglu; Supriyo Chakraborty; Chia-Hsuan Lee; Sambit Sahu
>
> **备注:** Submitted to NeurIPS 2025
>
> **摘要:** Supervised fine-tuning (SFT) using expert demonstrations often suffer from the imitation problem, where the model learns to reproduce the correct responses without understanding the underlying rationale. To address this limitation, we propose Critique-Guided Distillation (CGD), a novel multi-stage framework that integrates teacher model generated explanatory critiques and refined responses into the SFT process. A student model is then trained to map the triplet of prompt, teacher critique, and its own initial response to the corresponding refined teacher response, thereby learning both what to imitate and why. Using entropy-based analysis, we show that CGD reduces refinement uncertainty and can be interpreted as a Bayesian posterior update. We perform extensive empirical evaluation of CGD, on variety of benchmark tasks, and demonstrate significant gains on both math (AMC23 +17.5%) and language understanding tasks (MMLU-Pro +6.3%), while successfully mitigating the format drift issues observed in previous critique fine-tuning (CFT) techniques.
>
---
#### [replaced 081] Identifying Legal Holdings with LLMs: A Systematic Study of Performance, Scale, and Memorization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.02172v2](http://arxiv.org/pdf/2505.02172v2)**

> **作者:** Chuck Arvin
>
> **备注:** Presented as a short paper at International Conference on Artificial Intelligence and Law 2025 (Chicago, IL)
>
> **摘要:** As large language models (LLMs) continue to advance in capabilities, it is essential to assess how they perform on established benchmarks. In this study, we present a suite of experiments to assess the performance of modern LLMs (ranging from 3B to 90B+ parameters) on CaseHOLD, a legal benchmark dataset for identifying case holdings. Our experiments demonstrate ``scaling effects'' - performance on this task improves with model size, with more capable models like GPT4o and AmazonNovaPro achieving macro F1 scores of 0.744 and 0.720 respectively. These scores are competitive with the best published results on this dataset, and do not require any technically sophisticated model training, fine-tuning or few-shot prompting. To ensure that these strong results are not due to memorization of judicial opinions contained in the training data, we develop and utilize a novel citation anonymization test that preserves semantic meaning while ensuring case names and citations are fictitious. Models maintain strong performance under these conditions (macro F1 of 0.728), suggesting the performance is not due to rote memorization. These findings demonstrate both the promise and current limitations of LLMs for legal tasks with important implications for the development and measurement of automated legal analytics and legal benchmarks.
>
---
#### [replaced 082] No Need for Explanations: LLMs can implicitly learn from mistakes in-context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08550v2](http://arxiv.org/pdf/2502.08550v2)**

> **作者:** Lisa Alazraki; Maximilian Mozes; Jon Ander Campos; Tan Yi-Chern; Marek Rei; Max Bartolo
>
> **摘要:** Showing incorrect answers to Large Language Models (LLMs) is a popular strategy to improve their performance in reasoning-intensive tasks. It is widely assumed that, in order to be helpful, the incorrect answers must be accompanied by comprehensive rationales, explicitly detailing where the mistakes are and how to correct them. However, in this work we present a counterintuitive finding: we observe that LLMs perform better in math reasoning tasks when these rationales are eliminated from the context and models are left to infer on their own what makes an incorrect answer flawed. This approach also substantially outperforms chain-of-thought prompting in our evaluations. These results are consistent across LLMs of different sizes and varying reasoning abilities. To gain an understanding of why LLMs learn from mistakes more effectively without explicit corrective rationales, we perform a thorough analysis, investigating changes in context length and answer diversity between different prompting strategies, and their effect on performance. We also examine evidence of overfitting to the in-context rationales when these are provided, and study the extent to which LLMs are able to autonomously infer high-quality corrective rationales given only incorrect answers as input. We find evidence that, while incorrect answers are more beneficial for LLM learning than additional diverse correct answers, explicit corrective rationales over-constrain the model, thus limiting those benefits.
>
---
#### [replaced 083] KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16002v2](http://arxiv.org/pdf/2502.16002v2)**

> **作者:** Jingbo Yang; Bairu Hou; Wei Wei; Yujia Bao; Shiyu Chang
>
> **摘要:** We describe KVLink, an approach for efficient key-value (KV) cache reuse in large language models (LLMs). In many LLM applications, different inputs can share overlapping context, such as the same retrieved document appearing in multiple queries. However, the LLMs still need to encode the entire context for each query, leading to redundant computation. In this paper, we investigate a new strategy to eliminate such inefficiency, where the KV cache of each document is precomputed independently. During inference, the KV caches of retrieved documents are concatenated, allowing the model to reuse cached representations instead of recomputing them. To mitigate the performance degradation when using KV caches computed independently for each document, KVLink introduces two key techniques: adjusting positional embeddings of the KV cache at inference to match the global position after concatenation, and using trainable special tokens to restore self-attention across independently encoded documents. Experiments across 7 datasets demonstrate that KVLink improves question answering accuracy by an average of 4% over state-of-the-art methods. Furthermore, by leveraging precomputed KV caches, our approach reduces time-to-first-token by up to 96% compared to standard LLM inference, making it a scalable and efficient solution for context reuse. Additionally, KVLink can be combined with KV cache compression to further save cache loading and storage overhead while outperforming the baselines.
>
---
#### [replaced 084] Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17956v2](http://arxiv.org/pdf/2502.17956v2)**

> **作者:** Patomporn Payoungkhamdee; Pume Tuchinda; Jinheon Baek; Samuel Cahyawijaya; Can Udomcharoenchaikit; Potsawee Manakul; Peerat Limkonchotiwat; Ekapol Chuangsuwanich; Sarana Nutanong
>
> **摘要:** Multi-step reasoning is essential for large language models (LLMs), yet multilingual performance remains challenging. While Chain-of-Thought (CoT) prompting improves reasoning, it struggles with non-English languages due to the entanglement of reasoning and execution. Program-of-Thought (PoT) prompting separates reasoning from execution, offering a promising alternative but shifting the challenge to generating programs from non-English questions. We propose a framework to evaluate PoT by separating multilingual reasoning from code execution to examine (i) the impact of fine-tuning on question-reasoning alignment and (ii) how reasoning quality affects answer correctness. Our findings demonstrate that PoT fine-tuning substantially enhances multilingual reasoning, outperforming CoT fine-tuned models. We further demonstrate a strong correlation between reasoning quality (measured through code quality) and answer accuracy, highlighting its potential as a test-time performance improvement heuristic.
>
---
#### [replaced 085] Code Readability in the Age of Large Language Models: An Industrial Case Study from Atlassian
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11264v2](http://arxiv.org/pdf/2501.11264v2)**

> **作者:** Wannita Takerngsaksiri; Micheal Fu; Chakkrit Tantithamthavorn; Jirat Pasuksmit; Kun Chen; Ming Wu
>
> **备注:** 11 pages, 7 figures, 8 tables, under review
>
> **摘要:** Software engineers spend a significant amount of time reading code during the software development process. This trend is amplified by the emergence of large language models (LLMs) that automatically generate code. However, little is known about the readability of the LLM-generated code and whether it is still important from practitioners' perspectives in this new era. In this paper, we conduct a survey to explore the practitioners' perspectives on code readability in the age of LLMs and investigate the readability of our LLM-based software development agents framework, HULA, by comparing its generated code with human-written code in real-world scenarios. Overall, the findings underscore that (1) readability remains a critical aspect of software development; (2) the readability of our LLM-generated code is comparable to human-written code, fostering the establishment of appropriate trust and driving the broad adoption of our LLM-powered software development platform.
>
---
#### [replaced 086] LABO: Towards Learning Optimal Label Regularization via Bi-level Optimization
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2305.04971v2](http://arxiv.org/pdf/2305.04971v2)**

> **作者:** Peng Lu; Ahmad Rashid; Ivan Kobyzev; Mehdi Rezagholizadeh; Philippe Langlais
>
> **备注:** Accepted at ACL2023 (Findings)
>
> **摘要:** Regularization techniques are crucial to improving the generalization performance and training efficiency of deep neural networks. Many deep learning algorithms rely on weight decay, dropout, batch/layer normalization to converge faster and generalize. Label Smoothing (LS) is another simple, versatile and efficient regularization which can be applied to various supervised classification tasks. Conventional LS, however, regardless of the training instance assumes that each non-target class is equally likely. In this work, we present a general framework for training with label regularization, which includes conventional LS but can also model instance-specific variants. Based on this formulation, we propose an efficient way of learning LAbel regularization by devising a Bi-level Optimization (LABO) problem. We derive a deterministic and interpretable solution of the inner loop as the optimal label smoothing without the need to store the parameters or the output of a trained model. Finally, we conduct extensive experiments and demonstrate our LABO consistently yields improvement over conventional label regularization on various fields, including seven machine translation and three image classification tasks across various
>
---
#### [replaced 087] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20083v3](http://arxiv.org/pdf/2503.20083v3)**

> **作者:** Benjamin Minixhofer; Ivan Vulić; Edoardo Maria Ponti
>
> **备注:** Preprint, 21 pages
>
> **摘要:** Distillation has shown remarkable success in transferring knowledge from a Large Language Model (LLM) teacher to a student LLM. However, current distillation methods require similar tokenizers between the teacher and the student, restricting their applicability to only a small subset of teacher-student pairs. In this work, we develop a principled cross-tokenizer distillation method to solve this crucial deficiency. Our method is the first to enable effective distillation across fundamentally different tokenizers, while also substantially outperforming prior methods in all other cases. We verify the efficacy of our method on three distinct use cases. First, we show that viewing tokenizer transfer as self-distillation enables unprecedentedly effective transfer across tokenizers, including rapid transfer of subword models to the byte-level. Transferring different models to the same tokenizer also enables ensembling to boost performance. Secondly, we distil a large maths-specialised LLM into a small general-purpose model with a different tokenizer, achieving competitive maths problem-solving performance. Thirdly, we use our method to train state-of-the-art embedding prediction hypernetworks for training-free tokenizer transfer. Our results unlock an expanded range of teacher-student pairs for distillation, enabling new ways to adapt and enhance interaction between LLMs.
>
---
#### [replaced 088] AAAR-1.0: Assessing AI's Potential to Assist Research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.22394v3](http://arxiv.org/pdf/2410.22394v3)**

> **作者:** Renze Lou; Hanzi Xu; Sijia Wang; Jiangshu Du; Ryo Kamoi; Xiaoxin Lu; Jian Xie; Yuxuan Sun; Yusen Zhang; Jihyun Janice Ahn; Hongchao Fang; Zhuoyang Zou; Wenchao Ma; Xi Li; Kai Zhang; Congying Xia; Lifu Huang; Wenpeng Yin
>
> **备注:** ICML 2025. Project Webpage: https://renzelou.github.io/AAAR-1.0/
>
> **摘要:** Numerous studies have assessed the proficiency of AI systems, particularly large language models (LLMs), in facilitating everyday tasks such as email writing, question answering, and creative content generation. However, researchers face unique challenges and opportunities in leveraging LLMs for their own work, such as brainstorming research ideas, designing experiments, and writing or reviewing papers. In this study, we introduce AAAR-1.0, a benchmark dataset designed to evaluate LLM performance in three fundamental, expertise-intensive research tasks: (i) EquationInference, assessing the correctness of equations based on the contextual information in paper submissions; (ii) ExperimentDesign, designing experiments to validate research ideas and solutions; (iii) PaperWeakness, identifying weaknesses in paper submissions; and (iv) REVIEWCRITIQUE, identifying each segment in human reviews is deficient or not. AAAR-1.0 differs from prior benchmarks in two key ways: first, it is explicitly research-oriented, with tasks requiring deep domain expertise; second, it is researcher-oriented, mirroring the primary activities that researchers engage in on a daily basis. An evaluation of both open-source and proprietary LLMs reveals their potential as well as limitations in conducting sophisticated research tasks. We will keep iterating AAAR-1.0 to new versions.
>
---
#### [replaced 089] Uncertainty Distillation: Teaching Language Models to Express Semantic Confidence
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.14749v2](http://arxiv.org/pdf/2503.14749v2)**

> **作者:** Sophia Hager; David Mueller; Kevin Duh; Nicholas Andrews
>
> **摘要:** As large language models (LLMs) are increasingly used for factual question-answering, it becomes more important for LLMs to have the capability to communicate the likelihood that their answer is correct. For these verbalized expressions of uncertainty to be meaningful, they should reflect the error rates at the expressed level of confidence. However, when prompted to express confidence, the error rates of current LLMs are inconsistent with their communicated confidences, highlighting the need for uncertainty quantification methods. Many prior methods calculate lexical uncertainty, estimating a model's confidence in the specific string it generated. In some cases, however, it may be more useful to estimate semantic uncertainty, or the model's confidence in the answer regardless of how it is verbalized. We propose a simple procedure, uncertainty distillation, to teach an LLM to verbalize calibrated semantic confidences. Using held-out data to map initial uncertainty estimates to meaningful probabilities, we create examples annotated with verbalized probabilities for supervised fine-tuning. We compare uncertainty distillation to several strong baselines, and find that our method yields verbalized confidences that correlate well with observed error rates.
>
---
#### [replaced 090] Evaluating Automated Radiology Report Quality through Fine-Grained Phrasal Grounding of Clinical Findings
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01031v3](http://arxiv.org/pdf/2412.01031v3)**

> **作者:** Razi Mahmood; Pingkun Yan; Diego Machado Reyes; Ge Wang; Mannudeep K. Kalra; Parisa Kaviani; Joy T. Wu; Tanveer Syeda-Mahmood
>
> **摘要:** Several evaluation metrics have been developed recently to automatically assess the quality of generative AI reports for chest radiographs based only on textual information using lexical, semantic, or clinical named entity recognition methods. In this paper, we develop a new method of report quality evaluation by first extracting fine-grained finding patterns capturing the location, laterality, and severity of a large number of clinical findings. We then performed phrasal grounding to localize their associated anatomical regions on chest radiograph images. The textual and visual measures are then combined to rate the quality of the generated reports. We present results that compare this evaluation metric with other textual metrics on a gold standard dataset derived from the MIMIC collection and show its robustness and sensitivity to factual errors.
>
---
#### [replaced 091] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12464v5](http://arxiv.org/pdf/2502.12464v5)**

> **作者:** Seanie Lee; Dong Bok Lee; Dominik Wagner; Minki Kang; Haebin Seong; Tobias Bocklet; Juho Lee; Sung Ju Hwang
>
> **备注:** ACL 2025 findings
>
> **摘要:** Deploying large language models (LLMs) in real-world applications requires robust safety guard models to detect and block harmful user prompts. While large safety guard models achieve strong performance, their computational cost is substantial. To mitigate this, smaller distilled models are used, but they often underperform on "hard" examples where the larger model provides accurate predictions. We observe that many inputs can be reliably handled by the smaller model, while only a small fraction require the larger model's capacity. Motivated by this, we propose SafeRoute, a binary router that distinguishes hard examples from easy ones. Our method selectively applies the larger safety guard model to the data that the router considers hard, improving efficiency while maintaining accuracy compared to solely using the larger safety guard model. Experimental results on multiple benchmark datasets demonstrate that our adaptive model selection significantly enhances the trade-off between computational cost and safety performance, outperforming relevant baselines.
>
---
#### [replaced 092] Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03563v2](http://arxiv.org/pdf/2505.03563v2)**

> **作者:** Cléa Chataigner; Rebecca Ma; Prakhar Ganesh; Afaf Taïk; Elliot Creager; Golnoosh Farnadi
>
> **摘要:** Large language models (LLMs) are sensitive to subtle changes in prompt phrasing, complicating efforts to audit them reliably. Prior approaches often rely on arbitrary or ungrounded prompt variations, which may miss key linguistic and demographic factors in real-world usage. We introduce AUGMENT (Automated User-Grounded Modeling and Evaluation of Natural Language Transformations), a framework for systematically generating and evaluating controlled, realistic prompt paraphrases based on linguistic structure and user demographics. AUGMENT ensures paraphrase quality through a combination of semantic, stylistic, and instruction-following criteria. In a case study on the BBQ dataset, we show that user-grounded paraphrasing leads to significant shifts in LLM performance and bias metrics across nine models. Our findings highlight the need for more representative and structured approaches to prompt variation in LLM auditing.
>
---
#### [replaced 093] Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15253v2](http://arxiv.org/pdf/2504.15253v2)**

> **作者:** Yilun Zhou; Austin Xu; Peifeng Wang; Caiming Xiong; Shafiq Joty
>
> **备注:** ICML 2025. The first two authors contributed equally. The codebase is at https://github.com/SalesforceAIResearch/jetts-benchmark
>
> **摘要:** Scaling test-time computation, or affording a generator large language model (LLM) extra compute during inference, typically employs the help of external non-generative evaluators (i.e., reward models). Concurrently, LLM-judges, models trained to generate evaluations and critiques (explanations) in natural language, are becoming increasingly popular in automatic evaluation. Despite judge empirical successes, their effectiveness as evaluators in test-time scaling settings is largely unknown. In this paper, we introduce the Judge Evaluation for Test-Time Scaling (JETTS) benchmark, which evaluates judge performance in three domains (math reasoning, code generation, and instruction following) under three task settings: response reranking, step-level beam search, and critique-based response refinement. We evaluate 10 different judge models (7B-70B parameters) for 8 different base generator models (6.7B-72B parameters). Our benchmark shows that while judges are competitive with outcome reward models in reranking, they are consistently worse than process reward models in beam search procedures. Furthermore, though unique to LLM-judges, their natural language critiques are currently ineffective in guiding the generator towards better responses.
>
---
#### [replaced 094] ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Consensus Enforcement, and Column Exploration
- **分类: cs.CL; I.2.7; I.2.0; H.2.0**

- **链接: [http://arxiv.org/pdf/2502.00675v4](http://arxiv.org/pdf/2502.00675v4)**

> **作者:** Minghang Deng; Ashwin Ramachandran; Canwen Xu; Lanxiang Hu; Zhewei Yao; Anupam Datta; Hao Zhang
>
> **备注:** 32 pages, 2 figures
>
> **摘要:** We present ReFoRCE, a Text-to-SQL agent that tops the Spider 2.0 leaderboard--a challenging benchmark reflecting complex, real-world Text-to-SQL scenarios. While Text-to-SQL systems enable natural language queries over structured databases, deploying them in enterprise environments remains difficult due to large, complex schemas (with over 1,000 columns), diverse SQL dialects (e.g., BigQuery, Snowflake), and sophisticated query requirements (e.g., transformations and analytics). ReFoRCE addresses these challenges through: (a) database information compression via pattern-based table grouping and LLM-guided schema linking to alleviate long-context issues; (b) self-refinement to iteratively correct syntax and semantic errors across dialects; (c) majority-vote consensus to select high-confidence candidates while deferring ambiguous cases arising from sophisticated queries; and (d) iterative column exploration guided by execution feedback to resolve those deferred cases. ReFoRCE achieves new state-of-the-art results, with scores of 35.83 on Spider 2.0-Snow and 36.56 on Spider 2.0-Lite.
>
---
#### [replaced 095] A Unified Approach to Routing and Cascading for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10347v3](http://arxiv.org/pdf/2410.10347v3)**

> **作者:** Jasper Dekoninck; Maximilian Baader; Martin Vechev
>
> **摘要:** The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoretically optimal strategy. Through our analysis, we identify good quality estimators as the critical factor for the success of model selection paradigms. Finally, in our experiments, we show that cascade routing consistently outperforms the individual approaches by a large margin and we analyze quality estimators to determine when routing and/or cascading are useful paradigms for model selection.
>
---
#### [replaced 096] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05066v2](http://arxiv.org/pdf/2503.05066v2)**

> **作者:** Shwai He; Weilin Cai; Jiayi Huang; Ang Li
>
> **摘要:** The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation, optimizing the trade-off between performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where some experts are overloaded while others remain underutilized. This imbalance leads to poor resource utilization and increased latency, as the most burdened expert dictates the overall delay, a phenomenon we define as the \textbf{\textit{Straggler Effect}}. To mitigate this, we propose Capacity-Aware Inference, including two key techniques: (1) \textbf{\textit{Capacity-Aware Token Drop}}, which discards overloaded tokens to regulate the maximum latency of MoE, and (2) \textbf{\textit{Capacity-Aware Token Reroute}}, which reallocates overflowed tokens to underutilized experts, balancing the token distribution. These techniques collectively optimize both high-load and low-load expert utilization, leading to a more efficient MoE inference pipeline. Extensive experiments demonstrate the effectiveness of our methods, showing significant improvements in inference efficiency, e.g., 0.2\% average performance increase and a 1.94$\times$ inference speedup on Mixtral-8$\times$7B-Instruct.
>
---
#### [replaced 097] How Well Can a Long Sequence Model Model Long Sequences? Comparing Architechtural Inductive Biases on Long-Context Abilities
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.08112v3](http://arxiv.org/pdf/2407.08112v3)**

> **作者:** Jerry Huang
>
> **备注:** Accepted to The 31st International Conference on Computational Linguistics (COLING), 2025
>
> **摘要:** Long sequences occur in abundance within real-world scenarios, hence properly modelling them opens numerous down-stream use-cases. Deep neural networks, however, have often struggled with these for a variety of reasons. Recent advances, both in system engineering as well as model design, have enabled the scaling up of model that are purported to support extended context length. In particular, the state-space and linear recurrent neural network families of models hypothetically can entend to infinite sequence lenth. However, is this too good to be true? We conduct an evaluation to show that while such claims may be sound theoretically, there remain large practical gaps that are empirically observed. In particular, recurrent models still suffer in the same settings as long-context LLMs with attention. We further show that different inductive biases have inconsistent extrapolation capabilities, highlighting the need to further study such paradigms and investigate why long-context models seemingly fail to behave as one might expect.
>
---
#### [replaced 098] FastCuRL: Curriculum Reinforcement Learning with Stage-wise Context Scaling for Efficient Training R1-like Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17287v3](http://arxiv.org/pdf/2503.17287v3)**

> **作者:** Mingyang Song; Mao Zheng; Zheng Li; Wenjie Yang; Xuan Luo; Yue Pan; Feng Zhang
>
> **备注:** Ongoing Work
>
> **摘要:** Improving training efficiency continues to be one of the primary challenges in large-scale Reinforcement Learning (RL). In this paper, we investigate how context length and the complexity of training data influence the RL scaling training process of R1-distilled small reasoning models, e.g., DeepSeek-R1-Distill-Qwen-1.5B. Our experimental results reveal that: (1) simply controlling the context length and curating the training data based on the input prompt length can effectively improve the training efficiency of scaling RL, achieving better performance with more concise CoT; (2) properly scaling the context length helps mitigate entropy collapse; and (3) choosing an optimal context length can improve the efficiency of model training and incentivize the model's chain-of-thought reasoning capabilities. Inspired by these insights, we propose FastCuRL, a curriculum RL framework with stage-wise context scaling to achieve efficient training and concise CoT reasoning. Experiment results demonstrate that FastCuRL-1.5B-V3 significantly outperforms state-of-the-art reasoning models on five competition-level benchmarks and achieves 49.6\% accuracy on AIME 2024. Furthermore, FastCuRL-1.5B-Preview surpasses DeepScaleR-1.5B-Preview on five benchmarks while only using a single node with 8 GPUs and a total of 50\% of training steps. %The code, training data, and models will be publicly released.
>
---
#### [replaced 099] GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00063v2](http://arxiv.org/pdf/2505.00063v2)**

> **作者:** Siqi Li; Yufan Shen; Xiangnan Chen; Jiayi Chen; Hengwei Ju; Haodong Duan; Song Mao; Hongbin Zhou; Bo Zhang; Bin Fu; Pinlong Cai; Licheng Wen; Botian Shi; Yong Liu; Xinyu Cai; Yu Qiao
>
> **摘要:** The rapid advancement of multimodal large language models (MLLMs) has profoundly impacted the document domain, creating a wide array of application scenarios. This progress highlights the need for a comprehensive benchmark to evaluate these models' capabilities across various document-specific tasks. However, existing benchmarks often fail to locate specific model weaknesses or guide systematic improvements. To bridge this gap, we introduce a General Document Intelligence Benchmark (GDI-Bench), featuring 2.3k images across 9 key scenarios and 19 document-specific tasks. By decoupling visual complexity and reasoning complexity, the GDI-Bench structures graded tasks that allow performance assessment by difficulty, aiding in model weakness identification and optimization guidance. We evaluate various open-source and closed-source models on GDI-Bench, conducting decoupled analyses in the visual and reasoning domains, revealing their strengths and weaknesses. To address the diverse tasks and domains in the GDI-Bench, we propose a GDI-Model that mitigates catastrophic forgetting during the supervised fine-tuning (SFT) process through an intelligence-preserving training strategy, thereby reinforcing the inherent weaknesses of the base model. Our model achieves state-of-the-art performance on previous benchmarks and the GDI-Bench. Both our benchmark and models are or will be open-sourced on https://huggingface.co/GDIBench.
>
---
#### [replaced 100] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-lingual Transfer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.10945v2](http://arxiv.org/pdf/2505.10945v2)**

> **作者:** Seungyoon Lee; Seongtae Hong; Hyeonseok Moon; Heuiseok Lim
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) increasingly incorporate multilingual capabilities, fueling the demand to transfer them into target language-specific models. However, most approaches, which blend the source model's embedding by replacing the source vocabulary with the target language-specific vocabulary, may constrain expressive capacity in the target language since the source model is predominantly trained on English data. In this paper, we propose Semantic Aware Linear Transfer (SALT), a novel cross-lingual transfer technique that recycles embeddings from target language Pre-trained Language Models (PLMs) to transmit the deep representational strengths of PLM-derived embedding to LLMs. SALT derives unique regression lines based on the similarity in the overlap of the source and target vocabularies, to handle each non-overlapping token's embedding space. Our extensive experiments show that SALT significantly outperforms other transfer methods and achieves lower loss with accelerating faster convergence during language adaptation. Notably, SALT obtains remarkable performance in cross-lingual understanding setups compared to other methods. Furthermore, we highlight the scalable use of PLMs to enhance the functionality of contemporary LLMs by conducting experiments with varying architectures.
>
---
#### [replaced 101] Robust and Fine-Grained Detection of AI Generated Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.11952v2](http://arxiv.org/pdf/2504.11952v2)**

> **作者:** Ram Mohan Rao Kadiyala; Siddartha Pullakhandam; Kanwal Mehreen; Drishti Sharma; Siddhant Gupta; Jebish Purbey; Ashay Srivastava; Subhasya TippaReddy; Arvind Reddy Bobbili; Suraj Telugara Chandrashekhar; Modabbir Adeeb; Srinadh Vura; Hamza Farooq
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** An ideal detection system for machine generated content is supposed to work well on any generator as many more advanced LLMs come into existence day by day. Existing systems often struggle with accurately identifying AI-generated content over shorter texts. Further, not all texts might be entirely authored by a human or LLM, hence we focused more over partial cases i.e human-LLM co-authored texts. Our paper introduces a set of models built for the task of token classification which are trained on an extensive collection of human-machine co-authored texts, which performed well over texts of unseen domains, unseen generators, texts by non-native speakers and those with adversarial inputs. We also introduce a new dataset of over 2.4M such texts mostly co-authored by several popular proprietary LLMs over 23 languages. We also present findings of our models' performance over each texts of each domain and generator. Additional findings include comparison of performance against each adversarial method, length of input texts and characteristics of generated texts compared to the original human authored texts.
>
---
#### [replaced 102] SAFE-SQL: Self-Augmented In-Context Learning with Fine-grained Example Selection for Text-to-SQL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11438v2](http://arxiv.org/pdf/2502.11438v2)**

> **作者:** Jimin Lee; Ingeol Baek; Byeongjeong Kim; Hyunkyung Bae; Hwanhee Lee
>
> **备注:** 13 pages, 5 figures, 10 tables
>
> **摘要:** Text-to-SQL aims to convert natural language questions into executable SQL queries. While previous approaches, such as skeleton-masked selection, have demonstrated strong performance by retrieving similar training examples to guide large language models (LLMs), they struggle in real-world scenarios where such examples are unavailable. To overcome this limitation, we propose Self-Augmentation in-context learning with Fine-grained Example selection for Text-to-SQL (SAFE-SQL), a novel framework that improves SQL generation by generating and filtering self-augmented examples. SAFE-SQL first prompts an LLM to generate multiple Text-to-SQL examples relevant to the test input. Then SAFE-SQL filters these examples through three relevance assessments, constructing high-quality in-context learning examples. Using self-generated examples, SAFE-SQL surpasses the previous zero-shot, and few-shot Text-to-SQL frameworks, achieving higher execution accuracy. Notably, our approach provides additional performance gains in extra hard and unseen scenarios, where conventional methods often fail.
>
---
#### [replaced 103] ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15046v2](http://arxiv.org/pdf/2505.15046v2)**

> **作者:** Yifan Wu; Lutao Yan; Leixian Shen; Yinan Mei; Jiannan Wang; Yuyu Luo
>
> **摘要:** The emergence of Multi-modal Large Language Models (MLLMs) presents new opportunities for chart understanding. However, due to the fine-grained nature of these tasks, applying MLLMs typically requires large, high-quality datasets for task-specific fine-tuning, leading to high data collection and training costs. To address this, we propose ChartCards, a unified chart-metadata generation framework for multi-task chart understanding. ChartCards systematically synthesizes various chart information, including data tables, visualization code, visual elements, and multi-dimensional semantic captions. By structuring this information into organized metadata, ChartCards enables a single chart to support multiple downstream tasks, such as text-to-chart retrieval, chart summarization, chart-to-table conversion, chart description, and chart question answering. Using ChartCards, we further construct MetaChart, a large-scale high-quality dataset containing 10,862 data tables, 85K charts, and 170 K high-quality chart captions. We validate the dataset through qualitative crowdsourcing evaluations and quantitative fine-tuning experiments across various chart understanding tasks. Fine-tuning six different models on MetaChart resulted in an average performance improvement of 5% across all tasks. The most notable improvements are seen in text-to-chart retrieval and chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements of 17% and 28%, respectively.
>
---
#### [replaced 104] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00691v3](http://arxiv.org/pdf/2502.00691v3)**

> **作者:** Haozhe Wang; Long Li; Chao Qu; Fengming Zhu; Weidi Xu; Wei Chu; Fangzhen Lin
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Recent advances in mathematical problem-solving with language models (LMs) integrate chain-of-thought (CoT) reasoning and code execution to harness their complementary strengths. However, existing hybrid frameworks exhibit a critical limitation: they depend on externally dictated instructions or rigid code-integration templates, lacking metacognitive awareness -- the capacity to dynamically evaluate intrinsic capabilities and autonomously determine when and how to integrate tools. This rigidity motivates our study of autonomous code integration, enabling models to adapt tool-usage strategies as their reasoning abilities evolve during training. While reinforcement learning (RL) shows promise for boosting LLM reasoning at scale (e.g., DeepSeek-R1), we demonstrate its inefficiency in learning autonomous code integration due to inadequate exploration of the vast combinatorial space of CoT-code interleaving patterns. To address this challenge, we propose a novel Expectation-Maximization (EM) framework that synergizes structured exploration (E-step) with off-policy RL optimization (M-step), creating a self-reinforcing cycle between metacognitive tool-use decisions and evolving capabilities. Experiments reveal our method achieves superior results through improved exploration. Notably, our 7B model improves over 11% on MATH500 and 9.4% on AIME without o1-like CoT.
>
---
#### [replaced 105] MentalMAC: Enhancing Large Language Models for Detecting Mental Manipulation via Multi-Task Anti-Curriculum Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15255v2](http://arxiv.org/pdf/2505.15255v2)**

> **作者:** Yuansheng Gao; Han Bao; Tong Zhang; Bin Li; Zonghui Wang; Wenzhi Chen
>
> **摘要:** Mental manipulation is a subtle yet pervasive form of psychological abuse that poses serious threats to mental health. Its covert nature and the complexity of manipulation strategies make it challenging to detect, even for state-of-the-art large language models (LLMs). This concealment also hinders the manual collection of large-scale, high-quality annotations essential for training effective models. Although recent efforts have sought to improve LLMs' performance on this task, progress remains limited due to the scarcity of real-world annotated datasets. To address these challenges, we propose MentalMAC, a multi-task anti-curriculum distillation method that enhances LLMs' ability to detect mental manipulation in multi-turn dialogue. Our approach includes: (i) EvoSA, an unsupervised data expansion method based on evolutionary operations and speech act theory; (ii) teacher model-generated multi-task supervision; and (iii) progressive knowledge distillation from complex to simpler tasks. We then constructed the ReaMent dataset with 5,000 real-world dialogue samples, using a MentalMAC-distilled model to assist human annotation. Vast experiments demonstrate that our method significantly narrows the gap between student and teacher models and outperforms competitive LLMs across key evaluation metrics. All code, datasets, and checkpoints will be released upon paper acceptance. Warning: This paper contains content that may be offensive to readers.
>
---
#### [replaced 106] ReviewAgents: Bridging the Gap Between Human and AI-Generated Paper Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.08506v2](http://arxiv.org/pdf/2503.08506v2)**

> **作者:** Xian Gao; Jiacheng Ruan; Jingsheng Gao; Ting Liu; Yuzhuo Fu
>
> **备注:** Work in progress
>
> **摘要:** Academic paper review is a critical yet time-consuming task within the research community. With the increasing volume of academic publications, automating the review process has become a significant challenge. The primary issue lies in generating comprehensive, accurate, and reasoning-consistent review comments that align with human reviewers' judgments. In this paper, we address this challenge by proposing ReviewAgents, a framework that leverages large language models (LLMs) to generate academic paper reviews. We first introduce a novel dataset, Review-CoT, consisting of 142k review comments, designed for training LLM agents. This dataset emulates the structured reasoning process of human reviewers-summarizing the paper, referencing relevant works, identifying strengths and weaknesses, and generating a review conclusion. Building upon this, we train LLM reviewer agents capable of structured reasoning using a relevant-paper-aware training method. Furthermore, we construct ReviewAgents, a multi-role, multi-LLM agent review framework, to enhance the review comment generation process. Additionally, we propose ReviewBench, a benchmark for evaluating the review comments generated by LLMs. Our experimental results on ReviewBench demonstrate that while existing LLMs exhibit a certain degree of potential for automating the review process, there remains a gap when compared to human-generated reviews. Moreover, our ReviewAgents framework further narrows this gap, outperforming advanced LLMs in generating review comments.
>
---
#### [replaced 107] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04598v3](http://arxiv.org/pdf/2503.04598v3)**

> **作者:** Zhijian Zhuo; Yutao Zeng; Ya Wang; Sijun Zhang; Jian Yang; Xiaoqing Li; Xun Zhou; Jinwen Ma
>
> **摘要:** Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, challenges remain in training deep transformer networks, especially regarding the position of layer normalization. While Pre-Norm structures facilitate more stable training owing to their stronger identity path, they often lead to suboptimal performance compared to Post-Norm. In this paper, we propose $\textbf{HybridNorm}$, a simple yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. We provide both theoretical insights and empirical evidence demonstrating that HybridNorm improves gradient flow and model robustness. Extensive experiments on large-scale transformer models, including both dense and sparse variants, show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches across multiple benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models. Code is available at https://github.com/BryceZhuo/HybridNorm.
>
---
#### [replaced 108] Understanding Synthetic Context Extension via Retrieval Heads
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.22316v3](http://arxiv.org/pdf/2410.22316v3)**

> **作者:** Xinyu Zhao; Fangcong Yin; Greg Durrett
>
> **备注:** Published at ICML 2025
>
> **摘要:** Long-context LLMs are increasingly in demand for applications such as retrieval-augmented generation. To defray the cost of pretraining LLMs over long contexts, recent work takes an approach of synthetic context extension: fine-tuning LLMs with synthetically generated long-context data in a post-training stage. However, it remains unclear how and why this synthetic context extension imparts abilities for downstream long-context tasks. In this paper, we investigate fine-tuning on synthetic data for three long-context tasks that require retrieval and reasoning. We vary the realism of "needle" concepts to be retrieved and diversity of the surrounding "haystack" context, from using LLMs to construct synthetic documents to using templated relations and creating symbolic datasets. We find that models trained on synthetic data fall short of the real data, but surprisingly, the mismatch can be interpreted and even predicted in terms of a special set of attention heads that are responsible for retrieval over long context, retrieval heads (Wu et al., 2024). The retrieval heads learned on synthetic data have high overlap with retrieval heads learned on real data, and there is a strong correlation between the recall of heads learned and the downstream performance of a model. Furthermore, with attention knockout and activation patching, we mechanistically show that retrieval heads are necessary and explain model performance, although they are not totally sufficient. Our results shed light on how to interpret synthetic data fine-tuning performance and how to approach creating better data for learning real-world capabilities over long contexts.
>
---
#### [replaced 109] General-Reasoner: Advancing LLM Reasoning Across All Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14652v3](http://arxiv.org/pdf/2505.14652v3)**

> **作者:** Xueguang Ma; Qian Liu; Dongfu Jiang; Ge Zhang; Zejun Ma; Wenhu Chen
>
> **摘要:** Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
>
---
#### [replaced 110] Language Models are Universal Embedders
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2310.08232v2](http://arxiv.org/pdf/2310.08232v2)**

> **作者:** Xin Zhang; Zehan Li; Yanzhao Zhang; Dingkun Long; Pengjun Xie; Meishan Zhang; Min Zhang
>
> **备注:** XLLM Workshop, ACL 2025
>
> **摘要:** In the large language model (LLM) revolution, embedding is a key component of various systems, such as retrieving knowledge or memories for LLMs or building content moderation filters. As such cases span from English to other natural or programming languages, from retrieval to classification and beyond, it is advantageous to build a unified embedding model rather than dedicated ones for each scenario. In this context, the pre-trained multilingual decoder-only large language models, e.g., BLOOM, emerge as a viable backbone option. To assess their potential, we propose straightforward strategies for constructing embedders and introduce a universal evaluation benchmark. Experimental results show that our trained model is proficient at generating good embeddings across languages and tasks, even extending to languages and tasks for which no finetuning/pretraining data is available. We also present detailed analyses and additional evaluations. We hope that this work could encourage the development of more robust open-source universal embedders.
>
---
#### [replaced 111] Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16555v2](http://arxiv.org/pdf/2412.16555v2)**

> **作者:** Yanxu Mao; Peipei Liu; Tiehan Cui; Zhaoteng Yan; Congying Liu; Datao You
>
> **摘要:** Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.
>
---

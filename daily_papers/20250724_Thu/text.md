# 自然语言处理 cs.CL

- **最新发布 47 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在提升多语言对话场景下的语音识别准确率。论文提出了一种基于大语言模型（LLM）的编码器-适配器-LLM架构，并采用多阶段训练策略。实验结果显示其在挑战赛中取得了优异成绩。**

- **链接: [http://arxiv.org/pdf/2507.17288v1](http://arxiv.org/pdf/2507.17288v1)**

> **作者:** Miaomiao Gao; Xiaoxiao Xiang; Yiwen Guo
>
> **摘要:** This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking.
>
---
#### [new 002] Pretraining on the Test Set Is No Longer All You Need: A Debate-Driven Approach to QA Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型在标准问答（QA）基准测试中因数据污染和记忆效应导致的评估不可靠问题。作者提出了一种基于辩论的评估方法，通过构建对抗性辩论并由裁判模型评判，提升评估难度并减少记忆影响。论文贡献包括将QA任务转化为辩论评估的流程，以及一个验证该方法有效性的公开基准测试。**

- **链接: [http://arxiv.org/pdf/2507.17747v1](http://arxiv.org/pdf/2507.17747v1)**

> **作者:** Linbo Cao; Jinman Zhao
>
> **备注:** 22 pages, 7 figures. Accepted to COLM 2025. Code available at: github.com/l6cao/Debate-Driven-Evaluation
>
> **摘要:** As frontier language models increasingly saturate standard QA benchmarks, concerns about data contamination, memorization, and escalating dataset creation costs persist. We propose a debate-driven evaluation paradigm that transforms any existing QA dataset into structured adversarial debates--where one model is given the official answer to defend, and another constructs and defends an alternative answer--adjudicated by a judge model blind to the correct solution. By forcing multi-round argumentation, this approach substantially increases difficulty while penalizing shallow memorization, yet reuses QA items to reduce curation overhead. We make two main contributions: (1) an evaluation pipeline to systematically convert QA tasks into debate-based assessments, and (2) a public benchmark that demonstrates our paradigm's effectiveness on a subset of MMLU-Pro questions, complete with standardized protocols and reference models. Empirical results validate the robustness of the method and its effectiveness against data contamination--a Llama 3.1 model fine-tuned on test questions showed dramatic accuracy improvements (50% -> 82%) but performed worse in debates. Results also show that even weaker judges can reliably differentiate stronger debaters, highlighting how debate-based evaluation can scale to future, more capable systems while maintaining a fraction of the cost of creating new benchmarks. Overall, our framework underscores that "pretraining on the test set is no longer all you need," offering a sustainable path for measuring the genuine reasoning ability of advanced language models.
>
---
#### [new 003] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究如何高效扩展Mixture-of-Experts（MoE）语言模型。它旨在解决MoE模型配置与计算效率之间的预测问题。论文提出了“效率杠杆”（EL）指标，并通过训练300多个模型总结出EL的缩放规律，验证了专家激活比和计算预算的影响。最终设计并训练了Ling-mini-beta模型以验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.17702v1](http://arxiv.org/pdf/2507.17702v1)**

> **作者:** Changxin Tian; Kunlong Chen; Jia Liu; Ziqi Liu; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large Language Models (LLMs) efficiently by decoupling total parameters from computational cost. However, this decoupling creates a critical challenge: predicting the model capacity of a given MoE configurations (e.g., expert activation ratio and granularity) remains an unresolved problem. To address this gap, we introduce Efficiency Leverage (EL), a metric quantifying the computational advantage of an MoE model over a dense equivalent. We conduct a large-scale empirical study, training over 300 models up to 28B parameters, to systematically investigate the relationship between MoE architectural configurations and EL. Our findings reveal that EL is primarily driven by the expert activation ratio and the total compute budget, both following predictable power laws, while expert granularity acts as a non-linear modulator with a clear optimal range. We integrate these discoveries into a unified scaling law that accurately predicts the EL of an MoE architecture based on its configuration. To validate our derived scaling laws, we designed and trained Ling-mini-beta, a pilot model for Ling-2.0 series with only 0.85B active parameters, alongside a 6.1B dense model for comparison. When trained on an identical 1T high-quality token dataset, Ling-mini-beta matched the performance of the 6.1B dense model while consuming over 7x fewer computational resources, thereby confirming the accuracy of our scaling laws. This work provides a principled and empirically-grounded foundation for the scaling of efficient MoE models.
>
---
#### [new 004] SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决现有大语言模型在结构化知识理解评估方面的不足。作者构建了SKA-Bench基准，涵盖KG、Table等多种结构化知识形式，并通过多阶段流程生成测试实例，细粒度评估模型的噪声鲁棒性、顺序不敏感性等能力，发现当前模型仍面临挑战。**

- **链接: [http://arxiv.org/pdf/2507.17178v1](http://arxiv.org/pdf/2507.17178v1)**

> **作者:** Zhiqiang Liu; Enpei Niu; Yin Hua; Mengshu Sun; Lei Liang; Huajun Chen; Wen Zhang
>
> **摘要:** Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/Lza12a/SKA-Bench.
>
---
#### [new 005] The Pluralistic Moral Gap: Understanding Judgment and Value Differences between Humans and Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究人类与大语言模型（LLMs）在道德判断上的差异，属于道德对齐任务。作者构建了一个包含1618个道德困境的数据集，并提出“多元道德差距”概念，发现LLMs在人类共识高时才能较好对齐，但在价值观多样性上不足。为解决该问题，作者提出动态道德剖面（DMP）方法，提升模型对人类道德判断的对齐与价值多样性。**

- **链接: [http://arxiv.org/pdf/2507.17216v1](http://arxiv.org/pdf/2507.17216v1)**

> **作者:** Giuseppe Russo; Debora Nozza; Paul Röttger; Dirk Hovy
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** People increasingly rely on Large Language Models (LLMs) for moral advice, which may influence humans' decisions. Yet, little is known about how closely LLMs align with human moral judgments. To address this, we introduce the Moral Dilemma Dataset, a benchmark of 1,618 real-world moral dilemmas paired with a distribution of human moral judgments consisting of a binary evaluation and a free-text rationale. We treat this problem as a pluralistic distributional alignment task, comparing the distributions of LLM and human judgments across dilemmas. We find that models reproduce human judgments only under high consensus; alignment deteriorates sharply when human disagreement increases. In parallel, using a 60-value taxonomy built from 3,783 value expressions extracted from rationales, we show that LLMs rely on a narrower set of moral values than humans. These findings reveal a pluralistic moral gap: a mismatch in both the distribution and diversity of values expressed. To close this gap, we introduce Dynamic Moral Profiling (DMP), a Dirichlet-based sampling method that conditions model outputs on human-derived value profiles. DMP improves alignment by 64.3% and enhances value diversity, offering a step toward more pluralistic and human-aligned moral guidance from LLMs.
>
---
#### [new 006] Obscured but Not Erased: Evaluating Nationality Bias in LLMs via Name-Based Bias Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型中的国籍偏见问题。通过替换显性国籍标签为文化相关名字，构建新的基准测试数据，研究模型在不同情境下的偏见程度与准确性。结果显示，小模型偏见更严重且准确性更低，偏见在模糊情境中持续存在，揭示了AI系统在全球应用中的潜在问题。**

- **链接: [http://arxiv.org/pdf/2507.16989v1](http://arxiv.org/pdf/2507.16989v1)**

> **作者:** Giulio Pelosio; Devesh Batra; Noémie Bovey; Robert Hankache; Cristovao Iglesias; Greig Cowan; Raad Khraishi
>
> **摘要:** Large Language Models (LLMs) can exhibit latent biases towards specific nationalities even when explicit demographic markers are not present. In this work, we introduce a novel name-based benchmarking approach derived from the Bias Benchmark for QA (BBQ) dataset to investigate the impact of substituting explicit nationality labels with culturally indicative names, a scenario more reflective of real-world LLM applications. Our novel approach examines how this substitution affects both bias magnitude and accuracy across a spectrum of LLMs from industry leaders such as OpenAI, Google, and Anthropic. Our experiments show that small models are less accurate and exhibit more bias compared to their larger counterparts. For instance, on our name-based dataset and in the ambiguous context (where the correct choice is not revealed), Claude Haiku exhibited the worst stereotypical bias scores of 9%, compared to only 3.5% for its larger counterpart, Claude Sonnet, where the latter also outperformed it by 117.7% in accuracy. Additionally, we find that small models retain a larger portion of existing errors in these ambiguous contexts. For example, after substituting names for explicit nationality references, GPT-4o retains 68% of the error rate versus 76% for GPT-4o-mini, with similar findings for other model providers, in the ambiguous context. Our research highlights the stubborn resilience of biases in LLMs, underscoring their profound implications for the development and deployment of AI systems in diverse, global contexts.
>
---
#### [new 007] A Hybrid Early-Exit Algorithm for Large Language Models Based on Space Alignment Decoding (SPADE)
- **分类: cs.CL; cs.PF**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理成本高的问题。通过提出SPADE解码方法，对齐中间层与输出层表示，结合基于熵的置信度评估，实现早期退出，减少计算开销，同时保持输出质量。**

- **链接: [http://arxiv.org/pdf/2507.17618v1](http://arxiv.org/pdf/2507.17618v1)**

> **作者:** Bowen Zheng; Ming Ma; Zhongqiao Lin; Tianming Yang
>
> **摘要:** Large language models are computationally expensive due to their deep structures. Prior research has shown that intermediate layers contain sufficient information to generate accurate answers, leading to the development of early-exit algorithms that reduce inference costs by terminating computation at earlier layers. However, these methods often suffer from poor performance due to misalignment between intermediate and output layer representations that lead to decoding inaccuracy. To address these challenges, we propose SPADE (SPace Alignment DEcoding), a novel decoding method that aligns intermediate layer representations with the output layer by propagating a minimally reduced sequence consisting of only the start token and the answer token. We further optimize the early-exit decision-making process by training a linear approximation of SPADE that computes entropy-based confidence metrics. Putting them together, we create a hybrid early-exit algorithm that monitors confidence levels and stops inference at intermediate layers while using SPADE to generate high-quality outputs. This approach significantly reduces inference costs without compromising accuracy, offering a scalable and efficient solution for deploying large language models in real-world applications.
>
---
#### [new 008] AI-based Clinical Decision Support for Primary Care: A Real-World Study
- **分类: cs.CL**

- **简介: 该论文属于医疗AI应用任务，旨在解决基层医疗中临床决策和文档错误问题。研究团队在肯尼亚内罗毕的Penda Health诊所网络中引入AI Consult工具，评估其对减少诊断和治疗错误的效果。通过对比有无AI支持的39,849次就诊数据，发现AI Consult显著降低了错误率，并提升了医生对护理质量的满意度。**

- **链接: [http://arxiv.org/pdf/2507.16947v1](http://arxiv.org/pdf/2507.16947v1)**

> **作者:** Robert Korom; Sarah Kiptinness; Najib Adan; Kassim Said; Catherine Ithuli; Oliver Rotich; Boniface Kimani; Irene King'ori; Stellah Kamau; Elizabeth Atemba; Muna Aden; Preston Bowman; Michael Sharman; Rebecca Soskin Hicks; Rebecca Distler; Johannes Heidecke; Rahul K. Arora; Karan Singhal
>
> **备注:** Blog: https://openai.com/index/ai-clinical-copilot-penda-health/
>
> **摘要:** We evaluate the impact of large language model-based clinical decision support in live care. In partnership with Penda Health, a network of primary care clinics in Nairobi, Kenya, we studied AI Consult, a tool that serves as a safety net for clinicians by identifying potential documentation and clinical decision-making errors. AI Consult integrates into clinician workflows, activating only when needed and preserving clinician autonomy. We conducted a quality improvement study, comparing outcomes for 39,849 patient visits performed by clinicians with or without access to AI Consult across 15 clinics. Visits were rated by independent physicians to identify clinical errors. Clinicians with access to AI Consult made relatively fewer errors: 16% fewer diagnostic errors and 13% fewer treatment errors. In absolute terms, the introduction of AI Consult would avert diagnostic errors in 22,000 visits and treatment errors in 29,000 visits annually at Penda alone. In a survey of clinicians with AI Consult, all clinicians said that AI Consult improved the quality of care they delivered, with 75% saying the effect was "substantial". These results required a clinical workflow-aligned AI Consult implementation and active deployment to encourage clinician uptake. We hope this study demonstrates the potential for LLM-based clinical decision support tools to reduce errors in real-world settings and provides a practical framework for advancing responsible adoption.
>
---
#### [new 009] MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言推理评估任务，旨在解决当前大语言模型在多语言和文化背景下的推理能力评估不足问题。论文构建了包含法语、西班牙语和中文的原生推理问题基准MultiNRC，涵盖语言、文化、数学等类别，并提供英文对照翻译。通过评估14个主流大语言模型，发现模型在多语言推理上仍有显著不足，尤其在文化相关任务上表现较差。**

- **链接: [http://arxiv.org/pdf/2507.17476v1](http://arxiv.org/pdf/2507.17476v1)**

> **作者:** Alexander R. Fabbri; Diego Mares; Jorge Flores; Meher Mankikar; Ernesto Hernandez; Dean Lee; Bing Liu; Chen Xing
>
> **摘要:** Although recent Large Language Models (LLMs) have shown rapid improvement on reasoning benchmarks in English, the evaluation of such LLMs' multilingual reasoning capability across diverse languages and cultural contexts remains limited. Existing multilingual reasoning benchmarks are typically constructed by translating existing English reasoning benchmarks, biasing these benchmarks towards reasoning problems with context in English language/cultures. In this work, we introduce the Multilingual Native Reasoning Challenge (MultiNRC), a benchmark designed to assess LLMs on more than 1,000 native, linguistic and culturally grounded reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. For cultural/tradition reasoning and math reasoning with cultural relevance, we also provide English equivalent translations of the multilingual questions by manual translation from native speakers fluent in English. This set of English equivalents can provide a direct comparison of LLM reasoning capacity in other languages vs. English on the same reasoning questions. We systematically evaluate current 14 leading LLMs covering most LLM families on MultiNRC and its English equivalent set. The results show that (1) current LLMs are still not good at native multilingual reasoning, with none scoring above 50% on MultiNRC; (2) LLMs exhibit distinct strengths and weaknesses in handling linguistic, cultural, and logical reasoning tasks; (3) Most models perform substantially better in math reasoning in English compared to in original languages (+10%), indicating persistent challenges with culturally grounded knowledge.
>
---
#### [new 010] Synthetic Voice Data for Automatic Speech Recognition in African Languages
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于语音识别任务，旨在解决非洲语言因缺乏语音数据导致的ASR技术难以应用的问题。作者通过合成语音数据（LLM生成文本、TTS合成语音、ASR微调）进行实验，结果显示合成数据可显著提升低资源语言识别效果，且成本极低。论文还分析了合成数据的评估与改进空间。**

- **链接: [http://arxiv.org/pdf/2507.17578v1](http://arxiv.org/pdf/2507.17578v1)**

> **作者:** Brian DeRenzi; Anna Dixon; Mohamed Aymane Farhi; Christian Resch
>
> **备注:** 29 pages incl. appendix, 8 tables, 5 figures. Authors are listed in alphabetical order
>
> **摘要:** Speech technology remains out of reach for most of the over 2300 languages in Africa. We present the first systematic assessment of large-scale synthetic voice corpora for African ASR. We apply a three-step process: LLM-driven text creation, TTS voice synthesis, and ASR fine-tuning. Eight out of ten languages for which we create synthetic text achieved readability scores above 5 out of 7. We evaluated ASR improvement for three (Hausa, Dholuo, Chichewa) and created more than 2,500 hours of synthetic voice data at below 1% of the cost of real data. Fine-tuned Wav2Vec-BERT-2.0 models trained on 250h real and 250h synthetic Hausa matched a 500h real-data-only baseline, while 579h real and 450h to 993h synthetic data created the best performance. We also present gender-disaggregated ASR performance evaluation. For very low-resource languages, gains varied: Chichewa WER improved about 6.5% relative with a 1:2 real-to-synthetic ratio; a 1:1 ratio for Dholuo showed similar improvements on some evaluation data, but not on others. Investigating intercoder reliability, ASR errors and evaluation datasets revealed the need for more robust reviewer protocols and more accurate evaluation data. All data and models are publicly released to invite further work to improve synthetic data for African languages.
>
---
#### [new 011] TyDi QA-WANA: A Benchmark for Information-Seeking Question Answering in Languages of West Asia and North Africa
- **分类: cs.CL**

- **简介: 该论文属于信息检索型问答任务，旨在解决跨语言问答中文化相关性和长文本理解问题。作者构建了包含28K多语言样本的数据集TyDi QA-WANA，覆盖西亚和北非10种语言，每个问题配有一个可能含答案的完整文章。数据直接用各语言采集，避免翻译偏差，并提供基线模型与开源代码促进后续研究。**

- **链接: [http://arxiv.org/pdf/2507.17709v1](http://arxiv.org/pdf/2507.17709v1)**

> **作者:** Parker Riley; Siamak Shakeri; Waleed Ammar; Jonathan H. Clark
>
> **摘要:** We present TyDi QA-WANA, a question-answering dataset consisting of 28K examples divided among 10 language varieties of western Asia and northern Africa. The data collection process was designed to elicit information-seeking questions, where the asker is genuinely curious to know the answer. Each question in paired with an entire article that may or may not contain the answer; the relatively large size of the articles results in a task suitable for evaluating models' abilities to utilize large text contexts in answering questions. Furthermore, the data was collected directly in each language variety, without the use of translation, in order to avoid issues of cultural relevance. We present performance of two baseline models, and release our code and data to facilitate further improvement by the research community.
>
---
#### [new 012] Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于多语言知识图谱问答（KGQA）任务，旨在解决如何将多语言自然语言问题转化为SPARQL查询的问题。作者提出了mKGQAgent框架，通过模块化设计和LLM代理工作流，实现问题解析、实体链接与查询优化，提升了多语言环境下KGQA的准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.16971v1](http://arxiv.org/pdf/2507.16971v1)**

> **作者:** Aleksandr Perevalov; Andreas Both
>
> **备注:** During the final evaluation on the DBpedia- and Corporate-based KGQA benchmarks within the Text2SPARQL challenge 2025, our approach took first place among the other participants
>
> **摘要:** Accessing knowledge via multilingual natural-language interfaces is one of the emerging challenges in the field of information retrieval and related ones. Structured knowledge stored in knowledge graphs can be queried via a specific query language (e.g., SPARQL). Therefore, one needs to transform natural-language input into a query to fulfill an information need. Prior approaches mostly focused on combining components (e.g., rule-based or neural-based) that solve downstream tasks and come up with an answer at the end. We introduce mKGQAgent, a human-inspired framework that breaks down the task of converting natural language questions into SPARQL queries into modular, interpretable subtasks. By leveraging a coordinated LLM agent workflow for planning, entity linking, and query refinement - guided by an experience pool for in-context learning - mKGQAgent efficiently handles multilingual KGQA. Evaluated on the DBpedia- and Corporate-based KGQA benchmarks within the Text2SPARQL challenge 2025, our approach took first place among the other participants. This work opens new avenues for developing human-like reasoning systems in multilingual semantic parsing.
>
---
#### [new 013] Who Attacks, and Why? Using LLMs to Identify Negative Campaigning in 18M Tweets across 19 Countries
- **分类: cs.CL**

- **简介: 该论文属于政治传播分析任务，旨在解决负面竞选信息的跨语言分类问题。研究使用零样本大语言模型对1800万条多国推文进行分析，发现执政党较少使用负面信息，而极端和民粹政党尤其右翼更倾向负面竞选。**

- **链接: [http://arxiv.org/pdf/2507.17636v1](http://arxiv.org/pdf/2507.17636v1)**

> **作者:** Victor Hartman; Petter Törnberg
>
> **摘要:** Negative campaigning is a central feature of political competition, yet empirical research has been limited by the high cost and limited scalability of existing classification methods. This study makes two key contributions. First, it introduces zero-shot Large Language Models (LLMs) as a novel approach for cross-lingual classification of negative campaigning. Using benchmark datasets in ten languages, we demonstrate that LLMs achieve performance on par with native-speaking human coders and outperform conventional supervised machine learning approaches. Second, we leverage this novel method to conduct the largest cross-national study of negative campaigning to date, analyzing 18 million tweets posted by parliamentarians in 19 European countries between 2017 and 2022. The results reveal consistent cross-national patterns: governing parties are less likely to use negative messaging, while ideologically extreme and populist parties -- particularly those on the radical right -- engage in significantly higher levels of negativity. These findings advance our understanding of how party-level characteristics shape strategic communication in multiparty systems. More broadly, the study demonstrates the potential of LLMs to enable scalable, transparent, and replicable research in political communication across linguistic and cultural contexts.
>
---
#### [new 014] Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging
- **分类: cs.CL**

- **简介: 该论文研究主观因素对论证强度的影响，任务是分析情感、叙事和模糊限制语与论证质量的关系。为解决缺乏大规模分析的问题，作者进行了回归分析，并评估了自动化标注方法。结果揭示了这些主观因素对客观和主观论证质量的不同影响。**

- **链接: [http://arxiv.org/pdf/2507.17409v1](http://arxiv.org/pdf/2507.17409v1)**

> **作者:** Carlotta Quensel; Neele Falk; Gabriella Lapesa
>
> **备注:** Accepted to the 12th Workshop on Argument Mining (ArgMining) 2025
>
> **摘要:** In assessing argument strength, the notions of what makes a good argument are manifold. With the broader trend towards treating subjectivity as an asset and not a problem in NLP, new dimensions of argument quality are studied. Although studies on individual subjective features like personal stories exist, there is a lack of large-scale analyses of the relation between these features and argument strength. To address this gap, we conduct regression analysis to quantify the impact of subjective factors $-$ emotions, storytelling, and hedging $-$ on two standard datasets annotated for objective argument quality and subjective persuasion. As such, our contribution is twofold: at the level of contributed resources, as there are no datasets annotated with all studied dimensions, this work compares and evaluates automated annotation methods for each subjective feature. At the level of novel insights, our regression analysis uncovers different patterns of impact of subjective features on the two facets of argument strength encoded in the datasets. Our results show that storytelling and hedging have contrasting effects on objective and subjective argument quality, while the influence of emotions depends on their rhetoric utilization rather than the domain.
>
---
#### [new 015] Each to Their Own: Exploring the Optimal Embedding in RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG中单一嵌入模型性能受限的问题。作者提出了两种融合多嵌入模型的方法：Mixture-Embedding RAG和Confident RAG，其中后者通过多模型生成与置信度选择提升了效果，平均提升10%和5%。**

- **链接: [http://arxiv.org/pdf/2507.17442v1](http://arxiv.org/pdf/2507.17442v1)**

> **作者:** Shiting Chen; Zijian Zhao; Jinsong Chen
>
> **摘要:** Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domain-specific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Confident RAG generates responses multiple times using different embedding models and then selects the responses with the highest confidence level, demonstrating average improvements of approximately 10% and 5% over vanilla LLMs and RAG, respectively. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play approach for various domains. We will release our code upon publication.
>
---
#### [new 016] Multi-Label Classification with Generative AI Models in Healthcare: A Case Study of Suicidality and Risk Factors
- **分类: cs.CL; cs.IR; q-bio.QM**

- **简介: 该论文属于多标签分类任务，旨在解决自杀相关因素的早期识别问题。研究使用生成式大语言模型（如GPT-3.5和GPT-4.5）从精神科电子健康记录中提取自杀意念、自杀尝试等风险因素，提出了端到端的生成式多标签分类流程，并通过新评估方法分析模型表现与错误模式。**

- **链接: [http://arxiv.org/pdf/2507.17009v1](http://arxiv.org/pdf/2507.17009v1)**

> **作者:** Ming Huang; Zehan Li; Yan Hu; Wanjing Wang; Andrew Wen; Scott Lane; Salih Selek; Lokesh Shahani; Rodrigo Machado-Vieira; Jair Soares; Hua Xu; Hongfang Liu
>
> **摘要:** Suicide remains a pressing global health crisis, with over 720,000 deaths annually and millions more affected by suicide ideation (SI) and suicide attempts (SA). Early identification of suicidality-related factors (SrFs), including SI, SA, exposure to suicide (ES), and non-suicidal self-injury (NSSI), is critical for timely intervention. While prior studies have applied AI to detect SrFs in clinical notes, most treat suicidality as a binary classification task, overlooking the complexity of cooccurring risk factors. This study explores the use of generative large language models (LLMs), specifically GPT-3.5 and GPT-4.5, for multi-label classification (MLC) of SrFs from psychiatric electronic health records (EHRs). We present a novel end to end generative MLC pipeline and introduce advanced evaluation methods, including label set level metrics and a multilabel confusion matrix for error analysis. Finetuned GPT-3.5 achieved top performance with 0.94 partial match accuracy and 0.91 F1 score, while GPT-4.5 with guided prompting showed superior performance across label sets, including rare or minority label sets, indicating a more balanced and robust performance. Our findings reveal systematic error patterns, such as the conflation of SI and SA, and highlight the models tendency toward cautious over labeling. This work not only demonstrates the feasibility of using generative AI for complex clinical classification tasks but also provides a blueprint for structuring unstructured EHR data to support large scale clinical research and evidence based medicine.
>
---
#### [new 017] A Unifying Scheme for Extractive Content Selection Tasks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的抽取式内容选择任务。旨在解决不同任务间方法割裂的问题，提出统一框架IGCS，通过指令引导语言模型进行内容选择。工作包括构建统一基准IGCSbench、合成数据集及验证迁移学习效果，优化推理与评估方法。**

- **链接: [http://arxiv.org/pdf/2507.16922v1](http://arxiv.org/pdf/2507.16922v1)**

> **作者:** Shmuel Amar; Ori Shapira; Aviv Slobodkin; Ido Dagan
>
> **摘要:** A broad range of NLP tasks involve selecting relevant text spans from given source texts. Despite this shared objective, such \textit{content selection} tasks have traditionally been studied in isolation, each with its own modeling approaches, datasets, and evaluation metrics. In this work, we propose \textit{instruction-guided content selection (IGCS)} as a beneficial unified framework for such settings, where the task definition and any instance-specific request are encapsulated as instructions to a language model. To promote this framework, we introduce \igcsbench{}, the first unified benchmark covering diverse content selection tasks. Further, we create a large generic synthetic dataset that can be leveraged for diverse content selection tasks, and show that transfer learning with these datasets often boosts performance, whether dedicated training for the targeted task is available or not. Finally, we address generic inference time issues that arise in LLM-based modeling of content selection, assess a generic evaluation metric, and overall propose the utility of our resources and methods for future content selection models. Models and datasets available at https://github.com/shmuelamar/igcs.
>
---
#### [new 018] AI Telephone Surveying: Automating Quantitative Data Collection with an AI Interviewer
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于人机交互与调查研究任务，旨在解决传统电话调查效率低、互动性差的问题。作者构建了一个基于大语言模型、语音识别与合成技术的AI电话调查系统，具备自然对话、适应性强的特点，符合研究规范。通过试点调查与后续人工调查对比，验证了系统的有效性，关注完成率、退出率和满意度三项指标。**

- **链接: [http://arxiv.org/pdf/2507.17718v1](http://arxiv.org/pdf/2507.17718v1)**

> **作者:** Danny D. Leybzon; Shreyas Tirumala; Nishant Jain; Summer Gillen; Michael Jackson; Cameron McPhee; Jennifer Schmidt
>
> **摘要:** With the rise of voice-enabled artificial intelligence (AI) systems, quantitative survey researchers have access to a new data-collection mode: AI telephone surveying. By using AI to conduct phone interviews, researchers can scale quantitative studies while balancing the dual goals of human-like interactivity and methodological rigor. Unlike earlier efforts that used interactive voice response (IVR) technology to automate these surveys, voice AI enables a more natural and adaptive respondent experience as it is more robust to interruptions, corrections, and other idiosyncrasies of human speech. We built and tested an AI system to conduct quantitative surveys based on large language models (LLM), automatic speech recognition (ASR), and speech synthesis technologies. The system was specifically designed for quantitative research, and strictly adhered to research best practices like question order randomization, answer order randomization, and exact wording. To validate the system's effectiveness, we deployed it to conduct two pilot surveys with the SSRS Opinion Panel and followed-up with a separate human-administered survey to assess respondent experiences. We measured three key metrics: the survey completion rates, break-off rates, and respondent satisfaction scores. Our results suggest that shorter instruments and more responsive AI interviewers may contribute to improvements across all three metrics studied.
>
---
#### [new 019] From Feedback to Checklists: Grounded Evaluation of AI-Generated Clinical Notes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI评估任务，旨在解决AI生成临床病历的质量评估难题。现有方法依赖专家评审，成本高且主观性强。作者提出一种新方法，利用真实用户反馈生成结构化检查清单，结合大语言模型进行自动化评估。实验表明，该方法在覆盖性、多样性及预测医生评分方面优于基线方法，并能有效识别低质量病历。**

- **链接: [http://arxiv.org/pdf/2507.17717v1](http://arxiv.org/pdf/2507.17717v1)**

> **作者:** Karen Zhou; John Giorgi; Pranav Mani; Peng Xu; Davis Liang; Chenhao Tan
>
> **摘要:** AI-generated clinical notes are increasingly used in healthcare, but evaluating their quality remains a challenge due to high subjectivity and limited scalability of expert review. Existing automated metrics often fail to align with real-world physician preferences. To address this, we propose a pipeline that systematically distills real user feedback into structured checklists for note evaluation. These checklists are designed to be interpretable, grounded in human feedback, and enforceable by LLM-based evaluators. Using deidentified data from over 21,000 clinical encounters, prepared in accordance with the HIPAA safe harbor standard, from a deployed AI medical scribe system, we show that our feedback-derived checklist outperforms baseline approaches in our offline evaluations in coverage, diversity, and predictive power for human ratings. Extensive experiments confirm the checklist's robustness to quality-degrading perturbations, significant alignment with clinician preferences, and practical value as an evaluation methodology. In offline research settings, the checklist can help identify notes likely to fall below our chosen quality thresholds.
>
---
#### [new 020] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务中的模型评估与反馈优化。旨在解决通过传统标注方式获取高质量成对偏好数据困难的问题，尤其在事实性、数学和代码任务中。论文提出了一种结合外部验证工具（如网络搜索和代码执行）的代理系统，以提升反馈质量，并在多个数据集上进行了实验验证。**

- **链接: [http://arxiv.org/pdf/2507.17015v1](http://arxiv.org/pdf/2507.17015v1)**

> **作者:** Arduin Findeis; Floris Weers; Guoli Yin; Ke Ye; Ruoming Pang; Tom Gunter
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at https://github.com/apple/ml-agent-evaluator.
>
---
#### [new 021] Harnessing RLHF for Robust Unanswerability Recognition and Trustworthy Response Generation in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在面对无法回答的问题时易产生幻觉的问题。论文提出了SALU方法，通过多任务学习和结合人类反馈的强化学习，使模型能够自主判断问题是否可回答，并在不可回答时主动 abstain，从而提升模型的可靠性和事实性。**

- **链接: [http://arxiv.org/pdf/2507.16951v1](http://arxiv.org/pdf/2507.16951v1)**

> **作者:** Shuyuan Lin; Lei Duan; Philip Hughes; Yuxuan Sheng
>
> **摘要:** Conversational Information Retrieval (CIR) systems, while offering intuitive access to information, face a significant challenge: reliably handling unanswerable questions to prevent the generation of misleading or hallucinated content. Traditional approaches often rely on external classifiers, which can introduce inconsistencies with the core generative Large Language Models (LLMs). This paper introduces Self-Aware LLM for Unanswerability (SALU), a novel approach that deeply integrates unanswerability detection directly within the LLM's generative process. SALU is trained using a multi-task learning framework for both standard Question Answering (QA) and explicit abstention generation for unanswerable queries. Crucially, it incorporates a confidence-score-guided reinforcement learning with human feedback (RLHF) phase, which explicitly penalizes hallucinated responses and rewards appropriate abstentions, fostering intrinsic self-awareness of knowledge boundaries. Through extensive experiments on our custom-built C-IR_Answerability dataset, SALU consistently outperforms strong baselines, including hybrid LLM-classifier systems, in overall accuracy for correctly answering or abstaining from questions. Human evaluation further confirms SALU's superior reliability, achieving high scores in factuality, appropriate abstention, and, most importantly, a dramatic reduction in hallucination, demonstrating its ability to robustly "know when to say 'I don't know'."
>
---
#### [new 022] Evolutionary Feature-wise Thresholding for Binary Representation of NLP Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决如何高效将NLP嵌入转换为二进制表示的问题。作者提出一种基于坐标搜索的优化框架，为每个特征寻找最优阈值，提升了二进制编码的准确性与效率。实验表明该方法优于传统固定阈值法，适用于多种机器学习场景。**

- **链接: [http://arxiv.org/pdf/2507.17025v1](http://arxiv.org/pdf/2507.17025v1)**

> **作者:** Soumen Sinha; Shahryar Rahnamayan; Azam Asilian Bidgoli
>
> **摘要:** Efficient text embedding is crucial for large-scale natural language processing (NLP) applications, where storage and computational efficiency are key concerns. In this paper, we explore how using binary representations (barcodes) instead of real-valued features can be used for NLP embeddings derived from machine learning models such as BERT. Thresholding is a common method for converting continuous embeddings into binary representations, often using a fixed threshold across all features. We propose a Coordinate Search-based optimization framework that instead identifies the optimal threshold for each feature, demonstrating that feature-specific thresholds lead to improved performance in binary encoding. This ensures that the binary representations are both accurate and efficient, enhancing performance across various features. Our optimal barcode representations have shown promising results in various NLP applications, demonstrating their potential to transform text representation. We conducted extensive experiments and statistical tests on different NLP tasks and datasets to evaluate our approach and compare it to other thresholding methods. Binary embeddings generated using using optimal thresholds found by our method outperform traditional binarization methods in accuracy. This technique for generating binary representations is versatile and can be applied to any features, not just limited to NLP embeddings, making it useful for a wide range of domains in machine learning applications.
>
---
#### [new 023] WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文属于机器学习任务，旨在解决大语言模型预训练中学习率调度和模型合并的问题。作者提出WSM框架，通过理论分析建立学习率衰减与模型合并的联系，实验证明其优于传统方法，在多个基准上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2507.17634v1](http://arxiv.org/pdf/2507.17634v1)**

> **作者:** Changxin Tian; Jiapeng Wang; Qian Zhao; Kunlong Chen; Jia Liu; Ziqi Liu; Jiaxin Mao; Wayne Xin Zhao; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Recent advances in learning rate (LR) scheduling have demonstrated the effectiveness of decay-free approaches that eliminate the traditional decay phase while maintaining competitive performance. Model merging techniques have emerged as particularly promising solutions in this domain. We present Warmup-Stable and Merge (WSM), a general framework that establishes a formal connection between learning rate decay and model merging. WSM provides a unified theoretical foundation for emulating various decay strategies-including cosine decay, linear decay and inverse square root decay-as principled model averaging schemes, while remaining fully compatible with diverse optimization methods. Through extensive experiments, we identify merge duration-the training window for checkpoint aggregation-as the most critical factor influencing model performance, surpassing the importance of both checkpoint interval and merge quantity. Our framework consistently outperforms the widely-adopted Warmup-Stable-Decay (WSD) approach across multiple benchmarks, achieving significant improvements of +3.5% on MATH, +2.9% on HumanEval, and +5.5% on MMLU-Pro. The performance advantages extend to supervised fine-tuning scenarios, highlighting WSM's potential for long-term model refinement.
>
---
#### [new 024] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音的同声传译任务，旨在解决现有自动同声传译系统在翻译质量、实时性、多说话人混淆和语音克隆延迟等方面的问题。作者提出了Seed-LiveInterpret 2.0，通过双工语音理解和生成框架、大规模预训练与强化学习，实现了低延迟、高保真和语音克隆的端到端同声传译系统。**

- **链接: [http://arxiv.org/pdf/2507.17527v1](http://arxiv.org/pdf/2507.17527v1)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Ting Han; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [new 025] FinGAIA: An End-to-End Benchmark for Evaluating AI Agents in Finance
- **分类: cs.CL**

- **简介: 该论文提出了FinGAIA，一个用于评估金融领域AI代理的端到端基准。任务是衡量AI代理在证券、基金等七个金融子领域的多步、多工具协作能力。论文构建了407项任务，评估10个主流AI代理，发现最佳表现者ChatGPT准确率为48.9%，仍远低于专家水平，并总结了失败模式，指明未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.17186v1](http://arxiv.org/pdf/2507.17186v1)**

> **作者:** Lingfeng Zeng; Fangqi Lou; Zixuan Wang; Jiajie Xu; Jinyi Niu; Mengping Li; Yifan Dong; Qi Qi; Wei Zhang; Ziwei Yang; Jun Han; Ruilun Feng; Ruiqi Hu; Lejie Zhang; Zhengbo Feng; Yicheng Ren; Xin Guo; Zhaowei Liu; Dongpo Cheng; Weige Cai; Liwen Zhang
>
> **摘要:** The booming development of AI agents presents unprecedented opportunities for automating complex tasks across various domains. However, their multi-step, multi-tool collaboration capabilities in the financial sector remain underexplored. This paper introduces FinGAIA, an end-to-end benchmark designed to evaluate the practical abilities of AI agents in the financial domain. FinGAIA comprises 407 meticulously crafted tasks, spanning seven major financial sub-domains: securities, funds, banking, insurance, futures, trusts, and asset management. These tasks are organized into three hierarchical levels of scenario depth: basic business analysis, asset decision support, and strategic risk management. We evaluated 10 mainstream AI agents in a zero-shot setting. The best-performing agent, ChatGPT, achieved an overall accuracy of 48.9\%, which, while superior to non-professionals, still lags financial experts by over 35 percentage points. Error analysis has revealed five recurring failure patterns: Cross-modal Alignment Deficiency, Financial Terminological Bias, Operational Process Awareness Barrier, among others. These patterns point to crucial directions for future research. Our work provides the first agent benchmark closely related to the financial domain, aiming to objectively assess and promote the development of agents in this crucial field. Partial data is available at https://github.com/SUFE-AIFLM-Lab/FinGAIA.
>
---
#### [new 026] Millions of $\text{GeAR}$-s: Extending GraphRAG to Millions of Documents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决大规模文档下的检索增强生成（RAG）问题。作者扩展了图结构的RAG方法GeAR，尝试将其应用于包含数百万文档的SIGIR 2025 LiveRAG挑战赛中，评估其性能与局限性。**

- **链接: [http://arxiv.org/pdf/2507.17399v1](http://arxiv.org/pdf/2507.17399v1)**

> **作者:** Zhili Shen; Chenxin Diao; Pascual Merita; Pavlos Vougiouklis; Jeff Z. Pan
>
> **备注:** Accepted by SIGIR 2025 LiveRAG Challenge Program
>
> **摘要:** Recent studies have explored graph-based approaches to retrieval-augmented generation, leveraging structured or semi-structured information -- such as entities and their relations extracted from documents -- to enhance retrieval. However, these methods are typically designed to address specific tasks, such as multi-hop question answering and query-focused summarisation, and therefore, there is limited evidence of their general applicability across broader datasets. In this paper, we aim to adapt a state-of-the-art graph-based RAG solution: $\text{GeAR}$ and explore its performance and limitations on the SIGIR 2025 LiveRAG Challenge.
>
---
#### [new 027] Megrez2 Technical Report
- **分类: cs.CL**

- **简介: 论文提出Megrez2，一种轻量高效的语言模型架构，旨在优化设备端部署。它通过跨层专家共享机制减少参数量，并采用预门控路由提升推理效率。论文展示了Megrez2-Preview模型，在多种任务中表现优异，兼顾准确性与资源效率。**

- **链接: [http://arxiv.org/pdf/2507.17728v1](http://arxiv.org/pdf/2507.17728v1)**

> **作者:** Boxun Li; Yadong Li; Zhiyuan Li; Congyi Liu; Weilin Liu; Guowei Niu; Zheyue Tan; Haiyang Xu; Zhuyu Yao; Tao Yuan; Dong Zhou; Yueqing Zhuang; Bo Zhao; Guohao Dai; Yu Wang
>
> **摘要:** We present Megrez2, a novel lightweight and high-performance language model architecture optimized for device native deployment. Megrez2 introduces a novel cross-layer expert sharing mechanism, which significantly reduces total parameter count by reusing expert modules across adjacent transformer layers while maintaining most of the model's capacity. It also incorporates pre-gated routing, enabling memory-efficient expert loading and faster inference. As the first instantiation of the Megrez2 architecture, we introduce the Megrez2-Preview model, which is pre-trained on a 5-trillion-token corpus and further enhanced through supervised fine-tuning and reinforcement learning with verifiable rewards. With only 3B activated and 7.5B stored parameters, Megrez2-Preview demonstrates competitive or superior performance compared to larger models on a wide range of tasks, including language understanding, instruction following, mathematical reasoning, and code generation. These results highlight the effectiveness of the Megrez2 architecture to achieve a balance between accuracy, efficiency, and deployability, making it a strong candidate for real-world, resource-constrained applications.
>
---
#### [new 028] CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings
- **分类: cs.CL**

- **简介: 该论文属于医学文本生成任务，旨在解决自动生成放射科报告中临床结论不可靠的问题。论文提出CLARIFID框架，通过模仿专家两步流程，优化诊断准确性，结合多视角图像并强化推理逻辑，提高了报告的临床有效性。**

- **链接: [http://arxiv.org/pdf/2507.17234v1](http://arxiv.org/pdf/2507.17234v1)**

> **作者:** Kyeongkyu Lee; Seonghwan Yoon; Hongki Lim
>
> **摘要:** Automatic generation of radiology reports has the potential to alleviate radiologists' significant workload, yet current methods struggle to deliver clinically reliable conclusions. In particular, most prior approaches focus on producing fluent text without effectively ensuring the factual correctness of the reports and often rely on single-view images, limiting diagnostic comprehensiveness. We propose CLARIFID, a novel framework that directly optimizes diagnostic correctness by mirroring the two-step workflow of experts. Specifically, CLARIFID (1) learns the logical flow from Findings to Impression through section-aware pretraining, (2) is fine-tuned with Proximal Policy Optimization in which the CheXbert F1 score of the Impression section serves as the reward, (3) enforces reasoning-aware decoding that completes "Findings" before synthesizing the "Impression", and (4) fuses multiple chest X-ray views via a vision-transformer-based multi-view encoder. During inference, we apply a reasoning-aware next-token forcing strategy followed by report-level re-ranking, ensuring that the model first produces a comprehensive Findings section before synthesizing the Impression and thereby preserving coherent clinical reasoning. Experimental results on the MIMIC-CXR dataset demonstrate that our method achieves superior clinical efficacy and outperforms existing baselines on both standard NLG metrics and clinically aware scores.
>
---
#### [new 029] CogDual: Enhancing Dual Cognition of LLMs via Reinforcement Learning with Implicit Rule-Based Rewards
- **分类: cs.CL**

- **简介: 论文提出CogDual，通过强化学习优化大语言模型的双认知机制，解决角色扮演任务中角色行为与认知机制不匹配的问题，提升了角色一致性和上下文对齐能力。**

- **链接: [http://arxiv.org/pdf/2507.17147v1](http://arxiv.org/pdf/2507.17147v1)**

> **作者:** Cheng Liu; Yifei Lu; Fanghua Ye; Jian Li; Xingyu Chen; Feiliang Ren; Zhaopeng Tu; Xiaolong Li
>
> **摘要:** Role-Playing Language Agents (RPLAs) have emerged as a significant application direction for Large Language Models (LLMs). Existing approaches typically rely on prompt engineering or supervised fine-tuning to enable models to imitate character behaviors in specific scenarios, but often neglect the underlying \emph{cognitive} mechanisms driving these behaviors. Inspired by cognitive psychology, we introduce \textbf{CogDual}, a novel RPLA adopting a \textit{cognize-then-respond } reasoning paradigm. By jointly modeling external situational awareness and internal self-awareness, CogDual generates responses with improved character consistency and contextual alignment. To further optimize the performance, we employ reinforcement learning with two general-purpose reward schemes designed for open-domain text generation. Extensive experiments on the CoSER benchmark, as well as Cross-MR and LifeChoice, demonstrate that CogDual consistently outperforms existing baselines and generalizes effectively across diverse role-playing tasks.
>
---
#### [new 030] Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain
- **分类: cs.CL; cs.AI; I.2.7; J.m**

- **简介: 该论文属于多语言问答任务，旨在解决农业领域中信息获取的语言障碍问题。通过生成多语言农业合成数据集并微调语言模型，提升了模型在事实准确性、相关性和农业共识方面的表现，为多语言低资源场景下的农业咨询服务提供了有效方法。**

- **链接: [http://arxiv.org/pdf/2507.16974v1](http://arxiv.org/pdf/2507.16974v1)**

> **作者:** Rishemjit Kaur; Arshdeep Singh Bhankhar; Surangika Ranathunga; Jashanpreet Singh Salh; Sudhir Rajput; Vidhi; Kashish Mahendra; Bhavika Berwal; Ritesh Kumar
>
> **备注:** 15 pages, 9 tables, Appendix A-K
>
> **摘要:** Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities.
>
---
#### [new 031] BoSS: Beyond-Semantic Speech
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音技术任务，旨在解决当前语音系统无法捕捉超越语义的隐含信息问题。提出了“超越语义语音”（BoSS）概念，构建了语音交互系统能力等级框架（L1-L5），并基于认知相关理论和机器学习分析语音的多维特征，揭示了当前模型在理解超越语义信号上的不足。**

- **链接: [http://arxiv.org/pdf/2507.17563v1](http://arxiv.org/pdf/2507.17563v1)**

> **作者:** Qing Wang; Zehan Li; Hang Lv; Hongjie Chen; Yaodong Song; Jian Kang; Jie Lian; Jie Li; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Human communication involves more than explicit semantics, with implicit signals and contextual cues playing a critical role in shaping meaning. However, modern speech technologies, such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) often fail to capture these beyond-semantic dimensions. To better characterize and benchmark the progression of speech intelligence, we introduce Spoken Interaction System Capability Levels (L1-L5), a hierarchical framework illustrated the evolution of spoken dialogue systems from basic command recognition to human-like social interaction. To support these advanced capabilities, we propose Beyond-Semantic Speech (BoSS), which refers to the set of information in speech communication that encompasses but transcends explicit semantics. It conveys emotions, contexts, and modifies or extends meanings through multidimensional features such as affective cues, contextual dynamics, and implicit semantics, thereby enhancing the understanding of communicative intentions and scenarios. We present a formalized framework for BoSS, leveraging cognitive relevance theories and machine learning models to analyze temporal and contextual speech dynamics. We evaluate BoSS-related attributes across five different dimensions, reveals that current spoken language models (SLMs) are hard to fully interpret beyond-semantic signals. These findings highlight the need for advancing BoSS research to enable richer, more context-aware human-machine communication.
>
---
#### [new 032] Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私安全任务，旨在评估大语言模型（LLMs）在处理表格数据时面临的成员推理攻击（MIA）风险。论文提出Tab-MIA基准数据集，包含五种数据集和六种编码格式，用于系统评估不同编码下模型的隐私泄露情况。研究发现，LLMs在少量训练轮次后仍易受MIA攻击，AUROC得分高达90%。**

- **链接: [http://arxiv.org/pdf/2507.17259v1](http://arxiv.org/pdf/2507.17259v1)**

> **作者:** Eyal German; Sagiv Antebi; Daniel Samira; Asaf Shabtai; Yuval Elovici
>
> **摘要:** Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs.
>
---
#### [new 033] Dual-branch Prompting for Multimodal Machine Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态机器翻译（MMT）任务，旨在解决现有方法依赖成对图文输入且易受视觉噪声影响的问题。论文提出D2P-MMT框架，利用扩散模型生成重构图像，结合源文本进行双分支提示学习，并引入分布对齐损失提升模型鲁棒性与翻译性能。**

- **链接: [http://arxiv.org/pdf/2507.17588v1](http://arxiv.org/pdf/2507.17588v1)**

> **作者:** Jie Wang; Zhendong Yang; Liansong Zong; Xiaobo Zhang; Dexian Wang; Ji Zhang
>
> **摘要:** Multimodal Machine Translation (MMT) typically enhances text-only translation by incorporating aligned visual features. Despite the remarkable progress, state-of-the-art MMT approaches often rely on paired image-text inputs at inference and are sensitive to irrelevant visual noise, which limits their robustness and practical applicability. To address these issues, we propose D2P-MMT, a diffusion-based dual-branch prompting framework for robust vision-guided translation. Specifically, D2P-MMT requires only the source text and a reconstructed image generated by a pre-trained diffusion model, which naturally filters out distracting visual details while preserving semantic cues. During training, the model jointly learns from both authentic and reconstructed images using a dual-branch prompting strategy, encouraging rich cross-modal interactions. To bridge the modality gap and mitigate training-inference discrepancies, we introduce a distributional alignment loss that enforces consistency between the output distributions of the two branches. Extensive experiments on the Multi30K dataset demonstrate that D2P-MMT achieves superior translation performance compared to existing state-of-the-art approaches.
>
---
#### [new 034] A Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-Augmented Generation in Large Language Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决现有RAG方法忽略文档间语义关联的问题。作者提出QMKGF方法，通过构建多路径知识图谱并融合查询感知的注意力模型，提升生成内容的准确性和语义一致性。实验表明其在多个数据集上表现优异，尤其在HotpotQA上ROUGE-1得分显著提升。**

- **链接: [http://arxiv.org/pdf/2507.16826v1](http://arxiv.org/pdf/2507.16826v1)**

> **作者:** Qikai Wei; Huansheng Ning; Chunlong Han; Jianguo Ding
>
> **摘要:** Retrieval Augmented Generation (RAG) has gradually emerged as a promising paradigm for enhancing the accuracy and factual consistency of content generated by large language models (LLMs). However, existing RAG studies primarily focus on retrieving isolated segments using similarity-based matching methods, while overlooking the intrinsic connections between them. This limitation hampers performance in RAG tasks. To address this, we propose QMKGF, a Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval Augmented Generation. First, we design prompt templates and employ general-purpose LLMs to extract entities and relations, thereby generating a knowledge graph (KG) efficiently. Based on the constructed KG, we introduce a multi-path subgraph construction strategy that incorporates one-hop relations, multi-hop relations, and importance-based relations, aiming to improve the semantic relevance between the retrieved documents and the user query. Subsequently, we designed a query-aware attention reward model that scores subgraph triples based on their semantic relevance to the query. Then, we select the highest score subgraph and enrich subgraph with additional triples from other subgraphs that are highly semantically relevant to the query. Finally, the entities, relations, and triples within the updated subgraph are utilised to expand the original query, thereby enhancing its semantic representation and improving the quality of LLMs' generation. We evaluate QMKGF on the SQuAD, IIRC, Culture, HotpotQA, and MuSiQue datasets. On the HotpotQA dataset, our method achieves a ROUGE-1 score of 64.98\%, surpassing the BGE-Rerank approach by 9.72 percentage points (from 55.26\% to 64.98\%). Experimental results demonstrate the effectiveness and superiority of the QMKGF approach.
>
---
#### [new 035] SiLQ: Simple Large Language Model Quantization-Aware Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型量化后精度下降的问题。作者提出了一种简单、端到端的量化感知训练方法（SiLQ），在增加不到0.1%训练预算的情况下，显著提升量化模型性能，适用于多种模型架构且无需额外操作。**

- **链接: [http://arxiv.org/pdf/2507.16933v1](http://arxiv.org/pdf/2507.16933v1)**

> **作者:** Steven K. Esser; Jeffrey L. McKinstry; Deepika Bablani; Rathinakumar Appuswamy; Dharmendra S. Modha
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Large language models can be quantized to reduce inference time latency, model size, and energy consumption, thereby delivering a better user experience at lower cost. A challenge exists to deliver quantized models with minimal loss of accuracy in reasonable time, and in particular to do so without requiring mechanisms incompatible with specialized inference accelerators. Here, we demonstrate a simple, end-to-end quantization-aware training approach that, with an increase in total model training budget of less than 0.1%, outperforms the leading published quantization methods by large margins on several modern benchmarks, with both base and instruct model variants. The approach easily generalizes across different model architectures, can be applied to activations, cache, and weights, and requires the introduction of no additional operations to the model other than the quantization itself.
>
---
#### [new 036] URPO: A Unified Reward & Policy Optimization Framework for Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统分离式策略与奖励模型导致的复杂流程与性能瓶颈问题。论文提出URPO框架，通过统一奖励与策略优化，在单一模型中融合指令执行与奖励生成，简化训练流程并提升效果。实验表明其在多个评估指标上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.17515v1](http://arxiv.org/pdf/2507.17515v1)**

> **作者:** Songshuo Lu; Hua Wang; Zhi Chen; Yaohua Tang
>
> **摘要:** Large-scale alignment pipelines typically pair a policy model with a separately trained reward model whose parameters remain frozen during reinforcement learning (RL). This separation creates a complex, resource-intensive pipeline and suffers from a performance ceiling due to a static reward signal. We propose a novel framework, Unified Reward & Policy Optimization (URPO), that unifies instruction-following ("player") and reward modeling ("referee") within a single model and a single training phase. Our method recasts all alignment data-including preference pairs, verifiable reasoning, and open-ended instructions-into a unified generative format optimized by a single Group-Relative Policy Optimization (GRPO) loop. This enables the model to learn from ground-truth preferences and verifiable logic while simultaneously generating its own rewards for open-ended tasks. Experiments on the Qwen2.5-7B model demonstrate URPO's superiority. Our unified model significantly outperforms a strong baseline using a separate generative reward model, boosting the instruction-following score on AlpacaEval from 42.24 to 44.84 and the composite reasoning average from 32.66 to 35.66. Furthermore, URPO cultivates a superior internal evaluator as a byproduct of training, achieving a RewardBench score of 85.15 and surpassing the dedicated reward model it replaces (83.55). By eliminating the need for a separate reward model and fostering a co-evolutionary dynamic between generation and evaluation, URPO presents a simpler, more efficient, and more effective path towards robustly aligned language models.
>
---
#### [new 037] Segmentation-free Goodness of Pronunciation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音评估任务，旨在解决传统发音评估方法依赖语音分割的问题。作者提出了无需分割的发音质量评估方法GOP-SA和GOP-AF，利用CTC模型提升评估准确性，并通过实验验证其有效性，取得了发音评估的最新性能表现。**

- **链接: [http://arxiv.org/pdf/2507.16838v1](http://arxiv.org/pdf/2507.16838v1)**

> **作者:** Xinwei Cao; Zijian Fan; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer aided language learning (CALL) systems. Within MDD, phoneme-level pronunciation assessment is key to helping L2 learners improve their pronunciation. However, most systems are based on a form of goodness of pronunciation (GOP) which requires pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general alignment-free method that takes all possible alignments of the target phoneme into account (GOP-AF). We give a theoretical account of our definition of GOP-AF, an implementation that solves potential numerical issues as well as a proper normalization which makes the method applicable with acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and Speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-AF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the Speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.
>
---
#### [new 038] Weak Supervision Techniques towards Enhanced ASR Models in Industry-level CRM Systems
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决通用语音识别模型难以适应工业CRM系统中行业特定语音识别需求的问题。作者提出了一种微调行业专用ASR模型的方法，显著提升了其在实际工业应用中的性能，并已落地应用。**

- **链接: [http://arxiv.org/pdf/2507.16843v1](http://arxiv.org/pdf/2507.16843v1)**

> **作者:** Zhongsheng Wang; Sijie Wang; Jia Wang; Yung-I Liang; Yuxi Zhang; Jiamou Liu
>
> **备注:** Accepted by ICONIP 2024
>
> **摘要:** In the design of customer relationship management (CRM) systems, accurately identifying customer types and offering personalized services are key to enhancing customer satisfaction and loyalty. However, this process faces the challenge of discerning customer voices and intentions, and general pre-trained automatic speech recognition (ASR) models make it difficult to effectively address industry-specific speech recognition tasks. To address this issue, we innovatively proposed a solution for fine-tuning industry-specific ASR models, which significantly improved the performance of the fine-tuned ASR models in industry applications. Experimental results show that our method substantially improves the crucial auxiliary role of the ASR model in industry CRM systems, and this approach has also been adopted in actual industrial applications.
>
---
#### [new 039] ReMeREC: Relation-aware and Multi-entity Referring Expression Comprehension
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多实体指代表达理解任务，旨在解决现有方法在复杂场景中忽略实体间关系导致的定位不准问题。论文构建了包含关系标注的ReMeX数据集，并提出ReMeREC框架，通过TMP模块动态识别实体，结合EIR模块建模实体间关系，提升多实体定位与关系预测性能。**

- **链接: [http://arxiv.org/pdf/2507.16877v1](http://arxiv.org/pdf/2507.16877v1)**

> **作者:** Yizhi Hu; Zezhao Tian; Xingqun Qi; Chen Su; Bingkun Yang; Junhui Yin; Muyi Sun; Man Zhang; Zhenan Sun
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Referring Expression Comprehension (REC) aims to localize specified entities or regions in an image based on natural language descriptions. While existing methods handle single-entity localization, they often ignore complex inter-entity relationships in multi-entity scenes, limiting their accuracy and reliability. Additionally, the lack of high-quality datasets with fine-grained, paired image-text-relation annotations hinders further progress. To address this challenge, we first construct a relation-aware, multi-entity REC dataset called ReMeX, which includes detailed relationship and textual annotations. We then propose ReMeREC, a novel framework that jointly leverages visual and textual cues to localize multiple entities while modeling their inter-relations. To address the semantic ambiguity caused by implicit entity boundaries in language, we introduce the Text-adaptive Multi-entity Perceptron (TMP), which dynamically infers both the quantity and span of entities from fine-grained textual cues, producing distinctive representations. Additionally, our Entity Inter-relationship Reasoner (EIR) enhances relational reasoning and global scene understanding. To further improve language comprehension for fine-grained prompts, we also construct a small-scale auxiliary dataset, EntityText, generated using large language models. Experiments on four benchmark datasets show that ReMeREC achieves state-of-the-art performance in multi-entity grounding and relation prediction, outperforming existing approaches by a large margin.
>
---
#### [new 040] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决真实场景中缺乏明确奖励信号的问题。作者提出“Rubrics as Rewards”框架，利用结构化评分标准作为可解释奖励信号，通过GRPO算法训练语言模型。实验表明，该方法在健康任务中比传统方法表现更优，且更贴合人类偏好。**

- **链接: [http://arxiv.org/pdf/2507.17746v1](http://arxiv.org/pdf/2507.17746v1)**

> **作者:** Anisha Gunjal; Anthony Wang; Elaine Lau; Vaskar Nath; Bing Liu; Sean Hendryx
>
> **摘要:** Extending Reinforcement Learning with Verifiable Rewards (RLVR) to real-world tasks often requires balancing objective and subjective evaluation criteria. However, many such tasks lack a single, unambiguous ground truth-making it difficult to define reliable reward signals for post-training language models. While traditional preference-based methods offer a workaround, they rely on opaque reward functions that are difficult to interpret and prone to spurious correlations. We introduce $\textbf{Rubrics as Rewards}$ (RaR), a framework that uses structured, checklist-style rubrics as interpretable reward signals for on-policy training with GRPO. Our best RaR method yields up to a $28\%$ relative improvement on HealthBench-1k compared to simple Likert-based approaches, while matching or surpassing the performance of reward signals derived from expert-written references. By treating rubrics as structured reward signals, we show that RaR enables smaller-scale judge models to better align with human preferences and sustain robust performance across model scales.
>
---
#### [new 041] DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理与深度学习优化任务，旨在解决Transformer模型难以使用动量SGD优化器训练的问题。论文提出DNT模型，通过深度归一化技术平衡梯度分布，使其适用于动量SGD训练，同时保持与AdamW相当的性能。**

- **链接: [http://arxiv.org/pdf/2507.17501v1](http://arxiv.org/pdf/2507.17501v1)**

> **作者:** Xianbiao Qi; Marco Chen; Wenjie Xiao; Jiaquan Ye; Yelin He; Chun-Guang Li; Zhouchen Lin
>
> **备注:** We have introduced a novel architecture, Deeply Normalized Transformer (DNT), which enables efficient training with vanilla momentum SGDW (mSGDW), achieving performance on par with AdamW-optimized Transformers
>
> **摘要:** Transformers have become the de facto backbone of modern deep learning, yet their training typically demands an advanced optimizer with adaptive learning rate like AdamW, rather than a momentum SGDW (mSGDW). Previous works show that it is mainly due to a heavy-tailed distribution of the gradients. In this paper, we introduce a Deeply Normalized Transformer (DNT), which is meticulously engineered to overcome this limitation enabling seamless training with vanilla mSGDW while yielding comparable performance to the Transformers trained via AdamW. To be specific, in DNT, we strategically integrate normalization techniques at proper positions in the Transformers to effectively modulate the Jacobian matrices of each layer, balance the influence of weights, activations, and their interactions, and thus enable the distributions of gradients concentrated. We provide both theoretical justifications of the normalization technique used in our DNT and extensive empirical evaluation on two popular Transformer architectures to validate that: a) DNT outperforms its counterparts (\ie, ViT and GPT), and b) DNT can be effectively trained with vanilla mSGDW.
>
---
#### [new 042] A Highly Clean Recipe Dataset with Ingredient States Annotation for State Probing Task
- **分类: cs.MM; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在理解烹饪过程中难以追踪食材状态变化的问题。作者构建了一个带有食材状态标注的日本食谱数据集，并设计了三项新任务，用于评估模型对烹饪流程中食材状态转换的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.17232v1](http://arxiv.org/pdf/2507.17232v1)**

> **作者:** Mashiro Toyooka; Kiyoharu Aizawa; Yoko Yamakata
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Large Language Models (LLMs) are trained on a vast amount of procedural texts, but they do not directly observe real-world phenomena. In the context of cooking recipes, this poses a challenge, as intermediate states of ingredients are often omitted, making it difficult for models to track ingredient states and understand recipes accurately. In this paper, we apply state probing, a method for evaluating a language model's understanding of the world, to the domain of cooking. We propose a new task and dataset for evaluating how well LLMs can recognize intermediate ingredient states during cooking procedures. We first construct a new Japanese recipe dataset with clear and accurate annotations of ingredient state changes, collected from well-structured and controlled recipe texts. Using this dataset, we design three novel tasks to evaluate whether LLMs can track ingredient state transitions and identify ingredients present at intermediate steps. Our experiments with widely used LLMs, such as Llama3.1-70B and Qwen2.5-72B, show that learning ingredient state knowledge improves their understanding of cooking processes, achieving performance comparable to commercial LLMs.
>
---
#### [new 043] Pixels, Patterns, but No Poetry: To See The World like Humans
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态人工智能任务，旨在解决当前多模态大语言模型（MLLMs）在人类水平感知上的不足。论文提出了一种新的感知导向基准“图灵视觉测试”（TET），通过四项合成图像任务评估MLLM的感知能力。实验表明，现有MLLM在这些任务上表现差，而微调视觉模块可提升性能，说明问题在于视觉模块的泛化能力，而非语言推理部分。**

- **链接: [http://arxiv.org/pdf/2507.16863v1](http://arxiv.org/pdf/2507.16863v1)**

> **作者:** Hongcheng Gao; Zihao Huang; Lin Xu; Jingyi Tang; Xinhao Li; Yue Liu; Haoyang Li; Taihang Hu; Minhua Lin; Xinlong Yang; Ge Wu; Balong Bi; Hongyu Chen; Wentao Zhang
>
> **摘要:** Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work.
>
---
#### [new 044] Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音交互系统评估任务，旨在解决如何选择语音识别、大语言模型和语音合成组件以优化AI面试系统性能的问题。论文通过分析30万次面试数据，采用自动化评估框架比较不同技术组合的效果，并提出实用建议。**

- **链接: [http://arxiv.org/pdf/2507.16835v1](http://arxiv.org/pdf/2507.16835v1)**

> **作者:** Nima Yazdani; Ali Ansari; Aruj Mahajan; Amirhossein Afsharrad; Seyed Shahabeddin Mousavi
>
> **摘要:** Voice-based conversational AI systems increasingly rely on cascaded architectures combining speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. However, systematic evaluation of different component combinations in production settings remains understudied. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data from over 300,000 AI-conducted job interviews. We develop an automated evaluation framework using LLM-as-a-Judge to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of four production configurations reveals that Google STT paired with GPT-4.1 significantly outperforms alternatives in both conversational and technical quality metrics. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversational AI systems and contribute a validated evaluation methodology for voice-based interactions.
>
---
#### [new 045] Towards Robust Speech Recognition for Jamaican Patois Music Transcription
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决牙买加帕托斯音乐转录中识别效果差的问题。作者构建了40多小时手动标注的帕托斯音乐数据集，用于微调先进语音识别模型，并探索Whisper模型在该语言上的性能扩展规律。**

- **链接: [http://arxiv.org/pdf/2507.16834v1](http://arxiv.org/pdf/2507.16834v1)**

> **作者:** Jordan Madden; Matthew Stone; Dimitri Johnson; Daniel Geddez
>
> **摘要:** Although Jamaican Patois is a widely spoken language, current speech recognition systems perform poorly on Patois music, producing inaccurate captions that limit accessibility and hinder downstream applications. In this work, we take a data-centric approach to this problem by curating more than 40 hours of manually transcribed Patois music. We use this dataset to fine-tune state-of-the-art automatic speech recognition (ASR) models, and use the results to develop scaling laws for the performance of Whisper models on Jamaican Patois audio. We hope that this work will have a positive impact on the accessibility of Jamaican Patois music and the future of Jamaican Patois language modeling.
>
---
#### [new 046] Disaster Informatics after the COVID-19 Pandemic: Bibliometric and Topic Analysis based on Large-scale Academic Literature
- **分类: cs.SI; cs.AI; cs.CL; cs.DL**

- **简介: 该论文属于信息科学与灾害管理交叉领域的分析任务，旨在探讨新冠疫情后灾害信息学的研究趋势与变化。通过大规模文献数据与生成式AI等技术，论文分析了国家、机构、作者的活跃度与合作模式，识别了研究主题演化与重点转移，揭示了疫情对公共卫生相关研究的推动作用及跨领域协作趋势。**

- **链接: [http://arxiv.org/pdf/2507.16820v1](http://arxiv.org/pdf/2507.16820v1)**

> **作者:** Ngan Tran; Haihua Chen; Ana Cleveland; Yuhan Zhou
>
> **备注:** 36 pages, 14 figures, 5 tables
>
> **摘要:** This study presents a comprehensive bibliometric and topic analysis of the disaster informatics literature published between January 2020 to September 2022. Leveraging a large-scale corpus and advanced techniques such as pre-trained language models and generative AI, we identify the most active countries, institutions, authors, collaboration networks, emergent topics, patterns among the most significant topics, and shifts in research priorities spurred by the COVID-19 pandemic. Our findings highlight (1) countries that were most impacted by the COVID-19 pandemic were also among the most active, with each country having specific research interests, (2) countries and institutions within the same region or share a common language tend to collaborate, (3) top active authors tend to form close partnerships with one or two key partners, (4) authors typically specialized in one or two specific topics, while institutions had more diverse interests across several topics, and (5) the COVID-19 pandemic has influenced research priorities in disaster informatics, placing greater emphasis on public health. We further demonstrate that the field is converging on multidimensional resilience strategies and cross-sectoral data-sharing collaborations or projects, reflecting a heightened awareness of global vulnerability and interdependency. Collecting and quality assurance strategies, data analytic practices, LLM-based topic extraction and summarization approaches, and result visualization tools can be applied to comparable datasets or solve similar analytic problems. By mapping out the trends in disaster informatics, our analysis offers strategic insights for policymakers, practitioners, and scholars aiming to enhance disaster informatics capacities in an increasingly uncertain and complex risk landscape.
>
---
#### [new 047] TransLPRNet: Lite Vision-Language Network for Single/Dual-line Chinese License Plate Recognition
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于车牌识别任务，旨在解决复杂环境下中英文车牌识别准确率低的问题。作者提出了TransLPRNet模型，结合轻量级视觉编码器和文本解码器，并构建了双行车牌数据集。此外，引入透视校正网络（PTN）提升识别稳定性与精度，实现了高速度和高准确率。**

- **链接: [http://arxiv.org/pdf/2507.17335v1](http://arxiv.org/pdf/2507.17335v1)**

> **作者:** Guangzhu Xu; Zhi Ke; Pengcheng Zuo; Bangjun Lei
>
> **摘要:** License plate recognition in open environments is widely applicable across various domains; however, the diversity of license plate types and imaging conditions presents significant challenges. To address the limitations encountered by CNN and CRNN-based approaches in license plate recognition, this paper proposes a unified solution that integrates a lightweight visual encoder with a text decoder, within a pre-training framework tailored for single and double-line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we constructed a single/double-line license plate dataset by synthesizing images, applying texture mapping onto real scenes, and blending them with authentic license plate images. Furthermore, to enhance the system's recognition accuracy, we introduce a perspective correction network (PTN) that employs license plate corner coordinate regression as an implicit variable, supervised by license plate view classification information. This network offers improved stability, interpretability, and low annotation costs. The proposed algorithm achieves an average recognition accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with processing speeds reaching up to 167 frames per second, indicating strong practical applicability.
>
---
## 更新

#### [replaced 001] A Survey of Event Causality Identification: Taxonomy, Challenges, Assessment, and Prospects
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.10371v4](http://arxiv.org/pdf/2411.10371v4)**

> **作者:** Qing Cheng; Zefan Zeng; Xingchen Hu; Yuehang Si; Zhong Liu
>
> **摘要:** Event Causality Identification (ECI) has emerged as a pivotal task in natural language processing (NLP), aimed at automatically detecting causal relationships between events in text. In this comprehensive survey, we systematically elucidate the foundational principles and technical frameworks of ECI, proposing a novel classification framework to categorize and clarify existing methods. {We discuss associated challenges, provide quantitative evaluations, and outline future directions for this dynamic and rapidly evolving field. We first delineate key definitions, problem formalization, and evaluation protocols of ECI. Our classification framework organizes ECI methods based on two primary tasks: Sentence-level Event Causality Identification (SECI) and Document-level Event Causality Identification (DECI). For SECI, we review methods including feature pattern-based matching, machine learning-based classification, deep semantic encoding, prompt-based fine-tuning, and causal knowledge pre-training, alongside common data augmentation strategies. For DECI, we focus on techniques such as deep semantic encoding, event graph reasoning, and prompt-based fine-tuning. We dedicate specific discussions to advancements in multi-lingual and cross-lingual ECI as well as zero-shot ECI leveraging Large Language Models (LLMs). Furthermore, we analyze the strengths, limitations, and unresolved challenges of each method. Extensive quantitative evaluations are conducted on four benchmark datasets to assess various ECI methods. Finally, we explore future research directions.
>
---
#### [replaced 002] A Mathematical Theory of Discursive Networks
- **分类: cs.CL; cs.LG; 68T01, 60J10, 91D30, 05C82, 68T50, 68W20, 94A15; I.2.7; I.2.11; G.3**

- **链接: [http://arxiv.org/pdf/2507.06565v5](http://arxiv.org/pdf/2507.06565v5)**

> **作者:** Juan B. Gutiérrez
>
> **备注:** 42 pages, 4 figures, 4 tables, 3 algorithm, 61 references
>
> **摘要:** Large language models (LLMs) turn writing into a live exchange between humans and software. We characterize this new medium as a discursive network that treats people and LLMs as equal nodes and tracks how their statements circulate. We define the generation of erroneous information as invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. We develop a general mathematical model of discursive networks that shows that a network governed only by drift and self-repair stabilizes at a modest error rate. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source Flaws-of-Others (FOO) algorithm: a configurable loop in which any set of agents critique one another while a harmonizer merges their verdicts. We identify an ethical transgression, epithesis, that occurs when humans fail to engage in the discursive network. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from connecting imperfect ones into networks that enforce mutual accountability.
>
---
#### [replaced 003] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12988v2](http://arxiv.org/pdf/2502.12988v2)**

> **作者:** Zixiao Wang; Duzhen Zhang; Ishita Agrawal; Shen Gao; Le Song; Xiuying Chen
>
> **备注:** 19 pages, 3 figures, ACL 2025 Findings
>
> **摘要:** Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought processes of a character. Using Lu Xun, a renowned Chinese writer, as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope that this work inspires future research on deep character persona simulation LLM.
>
---
#### [replaced 004] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15586v2](http://arxiv.org/pdf/2507.15586v2)**

> **作者:** Xinping Zhao; Shouzheng Huang; Yan Zhong; Xinshuo Hu; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 16 pages, 7 Figures, 10 Tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly impact the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous methods extract evidence straightforwardly without explicit thinking, which risks filtering out key clues and struggles with generalization. To this end, we propose LEAR, which learns to extract rational evidence by (1) explicitly reasoning to identify potential cues within retrieval contents first, and then (2) consciously extracting to avoid omitting any key cues helpful for answering questions. Specifically, we frame evidence reasoning and evidence extraction into one unified response for end-to-end training; apply knowledge token masks for disentanglement to derive reasoning-based and extraction-based answers; and devise three types of verifiable reward functions, including answer, length, and format, to update the model via the policy optimization algorithm. Extensive experiments on three benchmark datasets show the effectiveness of LEAR, providing compact and high-quality evidence, improving the accuracy of downstream tasks, and promoting effective application in online RAG systems.
>
---
#### [replaced 005] From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01424v2](http://arxiv.org/pdf/2503.01424v2)**

> **作者:** Zekun Zhou; Xiaocheng Feng; Lei Huang; Xiachong Feng; Ziyun Song; Ruihan Chen; Liang Zhao; Weitao Ma; Yuxuan Gu; Baoxin Wang; Dayong Wu; Guoping Hu; Ting Liu; Bing Qin
>
> **摘要:** Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at https://github.com/zkzhou126/AI-for-Research.
>
---
#### [replaced 006] Advancing Large Language Models for Tibetan with Curated Data and Continual Pre-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09205v3](http://arxiv.org/pdf/2507.09205v3)**

> **作者:** Leiyu Pan; Bojian Xiong; Lei Yang; Renren Jin; Shaowei Zhang; Yue Chen; Ling Shi; Jiang Zhou; Junru Wu; Zhen Wang; Jianxiang Peng; Juesi Xiao; Tianyu Dong; Zhuowen Han; Zhuo Chen; Yuqi Ren; Deyi Xiong
>
> **摘要:** Large language models have achieved remarkable progress across many languages. However, Tibetan, as a representative low-resource language, is particularly underrepresented in existing models due to the scarcity of high-quality training corpora. To address this gap, we curate the largest Tibetan pre-training corpus to date, aggregating data from diverse sources and applying a dedicated data cleaning and processing pipeline tailored for Tibetan. With the curated data, we continue pre/post-training a multilingual base model to enhance its generative capabilities in Tibetan. To evaluate the Tibetan capabilities of the model, we create new high-quality Tibetan benchmarks, and complement them with existing public benchmarks. Experimental results demonstrate that our model consistently and significantly outperforms both open-source models of similar scale and Tibetan-tailored models across a wide range of tasks.
>
---
#### [replaced 007] Impact of Stickers on Multimodal Sentiment and Intent in Social Media: A New Task, Dataset and Baseline
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.08427v2](http://arxiv.org/pdf/2405.08427v2)**

> **作者:** Yuanchen Shi; Biao Ma; Longyin Zhang; Fang Kong
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Stickers are increasingly used in social media to express sentiment and intent. Despite their significant impact on sentiment analysis and intent recognition, little research has been conducted in this area. To address this gap, we propose a new task: \textbf{M}ultimodal chat \textbf{S}entiment \textbf{A}nalysis and \textbf{I}ntent \textbf{R}ecognition involving \textbf{S}tickers (MSAIRS). Additionally, we introduce a novel multimodal dataset containing Chinese chat records and stickers excerpted from several mainstream social media platforms. Our dataset includes paired data with the same text but different stickers, the same sticker but different contexts, and various stickers consisting of the same images with different texts, allowing us to better understand the impact of stickers on chat sentiment and intent. We also propose an effective multimodal joint model, MMSAIR, featuring differential vector construction and cascaded attention mechanisms for enhanced multimodal fusion. Our experiments demonstrate the necessity and effectiveness of jointly modeling sentiment and intent, as they mutually reinforce each other's recognition accuracy. MMSAIR significantly outperforms traditional models and advanced MLLMs, demonstrating the challenge and uniqueness of sticker interpretation in social media. Our dataset and code are available on https://github.com/FakerBoom/MSAIRS-Dataset.
>
---
#### [replaced 008] Resona: Improving Context Copying in Linear Recurrence Models with Retrieval
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22913v3](http://arxiv.org/pdf/2503.22913v3)**

> **作者:** Xinyu Wang; Linrui Ma; Jerry Huang; Peng Lu; Prasanna Parthasarathi; Xiao-Wen Chang; Boxing Chen; Yufei Cui
>
> **备注:** Accepted at the Second Conference on Language Modeling
>
> **摘要:** Recent shifts in the space of large language model (LLM) research have shown an increasing focus on novel architectures to compete with prototypical Transformer-based models that have long dominated this space. Linear recurrent models have proven to be a viable competitor due to their computational efficiency. However, such models still demonstrate a sizable gap compared to Transformers in terms of in-context learning among other tasks that require recalling information from a context. In this work, we introduce Resona, a simple and scalable framework for augmenting linear recurrent models with retrieval. Resona augments models with the ability to integrate retrieved information from the provided input context, enabling tailored behavior to diverse task requirements. Experiments on a variety of linear recurrent models demonstrate that Resona-augmented models observe significant performance gains on a variety of synthetic as well as real-world natural language tasks, highlighting its ability to act as a general purpose method to improve the in-context learning and language modeling abilities of linear recurrent LLMs.
>
---
#### [replaced 009] LEGO Co-builder: Exploring Fine-Grained Vision-Language Modeling for Multimodal LEGO Assembly Assistants
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05515v2](http://arxiv.org/pdf/2507.05515v2)**

> **作者:** Haochen Huang; Jiahuan Pei; Mohammad Aliannejadi; Xin Sun; Moonisa Ahsan; Chuang Yu; Zhaochun Ren; Pablo Cesar; Junxiao Wang
>
> **备注:** This version has been anonymized for double-blind review
>
> **摘要:** Vision-language models (VLMs) are facing the challenges of understanding and following multimodal assembly instructions, particularly when fine-grained spatial reasoning and precise object state detection are required. In this work, we explore LEGO Co-builder, a hybrid benchmark combining real-world LEGO assembly logic with programmatically generated multimodal scenes. The dataset captures stepwise visual states and procedural instructions, allowing controlled evaluation of instruction-following, object detection, and state detection. We introduce a unified framework and assess leading VLMs such as GPT-4o, Gemini, and Qwen-VL, under zero-shot and fine-tuned settings. Our results reveal that even advanced models like GPT-4o struggle with fine-grained assembly tasks, with a maximum F1 score of just 40.54\% on state detection, highlighting gaps in fine-grained visual understanding. We release the benchmark, codebase, and generation pipeline to support future research on multimodal assembly assistants grounded in real-world workflows.
>
---
#### [replaced 010] Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04457v4](http://arxiv.org/pdf/2505.04457v4)**

> **作者:** Shigeki Karita; Yuma Koizumi; Heiga Zen; Haruko Ishikawa; Robin Scheibler; Michiel Bacchiani
>
> **备注:** Accepted to IEEE WASPAA2025
>
> **摘要:** Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaveFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators.
>
---
#### [replaced 011] Tiny language models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14871v2](http://arxiv.org/pdf/2507.14871v2)**

> **作者:** Ronit D. Gross; Yarden Tzach; Tal Halevi; Ella Koresh; Ido Kanter
>
> **备注:** 23 pages, 1 figure and 12 tables, The data and code that support the findings of this study are openly available in a GitHub repository at https://github.com/Rg32601/Tiny-Language-Models
>
> **摘要:** A prominent achievement of natural language processing (NLP) is its ability to understand and generate meaningful human language. This capability relies on complex feedforward transformer block architectures pre-trained on large language models (LLMs). However, LLM pre-training is currently feasible only for a few dominant companies due to the immense computational resources required, limiting broader research participation. This creates a critical need for more accessible alternatives. In this study, we explore whether tiny language models (TLMs) exhibit the same key qualitative features of LLMs. We demonstrate that TLMs exhibit a clear performance gap between pre-trained and non-pre-trained models across classification tasks, indicating the effectiveness of pre-training, even at a tiny scale. The performance gap increases with the size of the pre-training dataset and with greater overlap between tokens in the pre-training and classification datasets. Furthermore, the classification accuracy achieved by a pre-trained deep TLM architecture can be replicated through a soft committee of multiple, independently pre-trained shallow architectures, enabling low-latency TLMs without affecting classification accuracy. Our results are based on pre-training BERT-6 and variants of BERT-1 on subsets of the Wikipedia dataset and evaluating their performance on FewRel, AGNews, and DBPedia classification tasks. Future research on TLM is expected to further illuminate the mechanisms underlying NLP, especially given that its biologically inspired models suggest that TLMs may be sufficient for children or adolescents to develop language. The data and code that support the findings of this study are openly available on https://github.com/Rg32601/Tiny-Language-Models .
>
---
#### [replaced 012] SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20682v3](http://arxiv.org/pdf/2410.20682v3)**

> **作者:** Eunwon Kim; Chanho Park; Buru Chang
>
> **摘要:** Shared memories between two individuals strengthen their bond and are crucial for facilitating their ongoing conversations. This study aims to make long-term dialogue more engaging by leveraging these shared memories. To this end, we introduce a new long-term dialogue dataset named SHARE, constructed from movie scripts, which are a rich source of shared memories among various relationships. Our dialogue dataset contains the summaries of persona information and events of two individuals, as explicitly revealed in their conversation, along with implicitly extractable shared memories. We also introduce EPISODE, a long-term dialogue framework based on SHARE that utilizes shared experiences between individuals. Through experiments using SHARE, we demonstrate that shared memories between two individuals make long-term dialogues more engaging and sustainable, and that EPISODE effectively manages shared memories during dialogue. Our dataset and code are available at https://github.com/e1kim/SHARE.
>
---
#### [replaced 013] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.14534v2](http://arxiv.org/pdf/2507.14534v2)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [replaced 014] Pseudo-Autoregressive Neural Codec Language Models for Efficient Zero-Shot Text-to-Speech Synthesis
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10352v2](http://arxiv.org/pdf/2504.10352v2)**

> **作者:** Yifan Yang; Shujie Liu; Jinyu Li; Yuxuan Hu; Haibin Wu; Hui Wang; Jianwei Yu; Lingwei Meng; Haiyang Sun; Yanqing Liu; Yan Lu; Kai Yu; Xie Chen
>
> **备注:** Accepted in ACM MM 2025
>
> **摘要:** Recent zero-shot text-to-speech (TTS) systems face a common dilemma: autoregressive (AR) models suffer from slow generation and lack duration controllability, while non-autoregressive (NAR) models lack temporal modeling and typically require complex designs. In this paper, we introduce a novel pseudo-autoregressive (PAR) codec language modeling approach that unifies AR and NAR modeling. Combining explicit temporal modeling from AR with parallel generation from NAR, PAR generates dynamic-length spans at fixed time steps. Building on PAR, we propose PALLE, a two-stage TTS system that leverages PAR for initial generation followed by NAR refinement. In the first stage, PAR progressively generates speech tokens along the time dimension, with each step predicting all positions in parallel but only retaining the left-most span. In the second stage, low-confidence tokens are iteratively refined in parallel, leveraging the global contextual information.Experiments demonstrate that PALLE, trained on LibriTTS, outperforms state-of-the-art systems trained on large-scale data, including F5-TTS, E2-TTS, and MaskGCT, on the LibriSpeech test-clean set in terms of speech quality, speaker similarity, and intelligibility, while achieving up to ten times faster inference speed. Audio samples are available at https://microsoft.com/research/project/vall-e-x/palle.
>
---
#### [replaced 015] Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22334v2](http://arxiv.org/pdf/2505.22334v2)**

> **作者:** Lai Wei; Yuting Li; Kaipeng Zheng; Chen Wang; Yue Wang; Linghe Kong; Lichao Sun; Weiran Huang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at https://github.com/waltonfuture/RL-with-Cold-Start.
>
---
#### [replaced 016] A Diagrammatic Calculus for a Functional Model of Natural Language Semantics
- **分类: cs.CL; cs.PL; J.5; D.3.1; D.3.3**

- **链接: [http://arxiv.org/pdf/2507.00782v2](http://arxiv.org/pdf/2507.00782v2)**

> **作者:** Matthieu Pierre Boyer
>
> **备注:** 15 pages plus one page appendix, submission to CSL 2026
>
> **摘要:** In this paper, we study a functional programming approach to natural language semantics, allowing us to increase the expressiveness of a more traditional denotation style. We will formalize a category based type and effect system to represent the semantic difference between syntactically equivalent expressions. We then construct a diagrammatic calculus to model parsing and handling of effects, providing a method to efficiently compute the denotations for sentences.
>
---
#### [replaced 017] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15910v3](http://arxiv.org/pdf/2502.15910v3)**

> **作者:** Zheyuan Liu; Guangyao Dou; Xiangchi Yuan; Chunhui Zhang; Zhaoxuan Tan; Meng Jiang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Generative models such as Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) trained on massive datasets can lead them to memorize and inadvertently reveal sensitive information, raising ethical and privacy concerns. While some prior works have explored this issue in the context of LLMs, it presents a unique challenge for MLLMs due to the entangled nature of knowledge across modalities, making comprehensive unlearning more difficult. To address this challenge, we propose Modality Aware Neuron Unlearning (MANU), a novel unlearning framework for MLLMs designed to selectively clip neurons based on their relative importance to the targeted forget data, curated for different modalities. Specifically, MANU consists of two stages: important neuron selection and selective pruning. The first stage identifies and collects the most influential neurons across modalities relative to the targeted forget knowledge, while the second stage is dedicated to pruning those selected neurons. MANU effectively isolates and removes the neurons that contribute most to the forget data within each modality, while preserving the integrity of retained knowledge. Our experiments conducted across various MLLM architectures illustrate that MANU can achieve a more balanced and comprehensive unlearning in each modality without largely affecting the overall model utility.
>
---
#### [replaced 018] ORANSight-2.0: Foundational LLMs for O-RAN
- **分类: cs.CL; cs.AI; cs.LG; cs.NI**

- **链接: [http://arxiv.org/pdf/2503.05200v2](http://arxiv.org/pdf/2503.05200v2)**

> **作者:** Pranshav Gajjar; Vijay K. Shah
>
> **摘要:** Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative to develop specialized foundational LLMs tailored for O-RAN. Built on 18 models spanning five open-source LLM frameworks -- Mistral, Qwen, Llama, Phi, and Gemma -- ORANSight-2.0 fine-tunes models ranging from 1B to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance in O-RAN-specific tasks. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG)-based instruction-tuning framework that employs two LLM agents -- a Mistral-based Question Generator and a Qwen-based Answer Generator -- to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack.
>
---
#### [replaced 019] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18079v3](http://arxiv.org/pdf/2505.18079v3)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** V3 draft. Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code has been released in https://github.com/microsoft/DeepVideoDiscovery.
>
---
#### [replaced 020] Adaptive Graph Pruning for Multi-Agent Communication
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.02951v3](http://arxiv.org/pdf/2506.02951v3)**

> **作者:** Boyi Li; Zhonghan Zhao; Der-Horng Lee; Gaoang Wang
>
> **备注:** ECAI 2025
>
> **摘要:** Large Language Model (LLM) based multi-agent systems have shown remarkable performance in various tasks, especially when enhanced through collaborative communication. However, current methods often rely on a fixed number of agents and static communication structures, limiting their ability to adapt to varying task complexities. In this paper, we propose Adaptive Graph Pruning (AGP), a novel task-adaptive multi-agent collaboration framework that jointly optimizes agent quantity (hard-pruning) and communication topology (soft-pruning). Specifically, our method employs a two-stage training strategy: firstly, independently training soft-pruning networks for different agent quantities to determine optimal agent-quantity-specific complete graphs and positional masks across specific tasks; and then jointly optimizing hard-pruning and soft-pruning within a maximum complete graph to dynamically configure the number of agents and their communication topologies per task. Extensive experiments demonstrate that our approach is: (1) High-performing, achieving state-of-the-art results across six benchmarks and consistently generalizes across multiple mainstream LLM architectures, with a increase in performance of $2.58\%\sim 9.84\%$; (2) Task-adaptive, dynamically constructing optimized communication topologies tailored to specific tasks, with an extremely high performance in all three task categories (general reasoning, mathematical reasoning, and code generation); (3) Token-economical, having fewer training steps and token consumption at the same time, with a decrease in token consumption of $90\%+$; and (4) Training-efficient, achieving high performance with very few training steps compared with other methods. The performance will surpass the existing baselines after about ten steps of training under six benchmarks.
>
---
#### [replaced 021] Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16802v2](http://arxiv.org/pdf/2507.16802v2)**

> **作者:** Yanjun Zheng; Xiyang Du; Longfei Liao; Xiaoke Zhao; Zhaowen Zhou; Bo Zhang; Jiawei Liu; Xiang Qi; Zhe Li; Zhiqiang Zhang; Wei Wang; Peng Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit considerable promise in financial applications; however, prevailing models frequently demonstrate limitations when confronted with scenarios that necessitate sophisticated reasoning capabilities, stringent trustworthiness criteria, and efficient adaptation to domain-specific requirements. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task label system with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage training pipeline, and dynamic attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including Fineva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA-diamond. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. The Finova bench is available at https://github.com/antgroup/Finova.
>
---
#### [replaced 022] Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16799v2](http://arxiv.org/pdf/2507.16799v2)**

> **作者:** Xiaoyu Zhan; Xinyu Fu; Hao Sun; Yuanqi Li; Jie Guo; Yanwen Guo
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
>
---
#### [replaced 023] Is text normalization relevant for classifying medieval charters?
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2408.16446v2](http://arxiv.org/pdf/2408.16446v2)**

> **作者:** Florian Atzenhofer-Baumgartner; Tamás Kovács
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in LNCS volume 15178 and is available online at https://doi.org/10.1007/978-3-031-72440-4_12
>
> **摘要:** This study examines the impact of historical text normalization on the classification of medieval charters, specifically focusing on document dating and locating. Using a data set of Middle High German charters from a digital archive, we evaluate various classifiers, including traditional and transformer-based models, with and without normalization. Our results indicate that the given normalization minimally improves locating tasks but reduces accuracy for dating, implying that original texts contain crucial features that normalization may obscure. We find that support vector machines and gradient boosting outperform other models, questioning the efficiency of transformers for this use case. Results suggest a selective approach to historical text normalization, emphasizing the significance of preserving some textual characteristics that are critical for classification tasks in document analysis.
>
---
#### [replaced 024] GTA: Grouped-head latenT Attention
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17286v2](http://arxiv.org/pdf/2506.17286v2)**

> **作者:** Luoyang Sun; Cheng Deng; Jiwen Jiang; Xinjian Wu; Haifeng Zhang; Lei Chen; Lionel Ni; Jun Wang
>
> **摘要:** Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint.
>
---
#### [replaced 025] Visualising Policy-Reward Interplay to Inform Zeroth-Order Preference Optimisation of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03460v2](http://arxiv.org/pdf/2503.03460v2)**

> **作者:** Alessio Galatolo; Zhenbang Dai; Katie Winkle; Meriem Beloucif
>
> **备注:** ACL25 Findings
>
> **摘要:** Fine-tuning Large Language Models (LLMs) with first-order methods like back-propagation is computationally intensive. Zeroth-Order (ZO) optimisation uses function evaluations instead of gradients, reducing memory usage, but suffers from slow convergence in high-dimensional models. As a result, ZO research in LLMs has mostly focused on classification, overlooking more complex generative tasks. In this paper, we introduce ZOPrO, a novel ZO algorithm designed for Preference Optimisation in LLMs. We begin by analysing the interplay between policy and reward models during traditional (first-order) Preference Optimisation, uncovering patterns in their relative updates. Guided by these insights, we adapt Simultaneous Perturbation Stochastic Approximation (SPSA) with a targeted sampling strategy to accelerate convergence. Through experiments on summarisation, machine translation, and conversational assistants, we demonstrate that our method consistently enhances reward signals while achieving convergence times comparable to first-order methods. While it falls short of some state-of-the-art methods, our work is the first to apply Zeroth-Order methods to Preference Optimisation in LLMs, going beyond classification tasks and paving the way for a largely unexplored research direction. Code and visualisations are available at https://github.com/alessioGalatolo/VisZOPrO
>
---
#### [replaced 026] An Efficient and Precise Training Data Construction Framework for Process-supervised Reward Model in Mathematical Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02382v2](http://arxiv.org/pdf/2503.02382v2)**

> **作者:** Wei Sun; Qianlong Du; Fuwei Cui; Jiajun Zhang
>
> **摘要:** Enhancing the mathematical reasoning capabilities of Large Language Models (LLMs) is of great scientific and practical significance. Researchers typically employ process-supervised reward models (PRMs) to guide the reasoning process, effectively improving the models' reasoning abilities. However, existing methods for constructing process supervision training data, such as manual annotation and per-step Monte Carlo estimation, are often costly or suffer from poor quality. To address these challenges, this paper introduces a framework called EpicPRM, which annotates each intermediate reasoning step based on its quantified contribution and uses an adaptive binary search algorithm to enhance both annotation precision and efficiency. Using this approach, we efficiently construct a high-quality process supervision training dataset named Epic50k, consisting of 50k annotated intermediate steps. Compared to other publicly available datasets, the PRM trained on Epic50k demonstrates significantly superior performance. Getting Epic50k at https://github.com/xiaolizh1/EpicPRM.
>
---
#### [replaced 027] LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15606v2](http://arxiv.org/pdf/2506.15606v2)**

> **作者:** Gabriel J. Perin; Runjin Chen; Xuxi Chen; Nina S. T. Hirata; Zhangyang Wang; Junyuan Hong
>
> **摘要:** Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
>
---
#### [replaced 028] Large Language Models in Argument Mining: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16383v3](http://arxiv.org/pdf/2506.16383v3)**

> **作者:** Hao Li; Viktor Schlegel; Yizheng Sun; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Work draft
>
> **摘要:** Argument Mining (AM), a critical subfield of Natural Language Processing (NLP), focuses on extracting argumentative structures from text. The advent of Large Language Models (LLMs) has profoundly transformed AM, enabling advanced in-context learning, prompt-based generation, and robust cross-domain adaptability. This survey systematically synthesizes recent advancements in LLM-driven AM. We provide a concise review of foundational theories and annotation frameworks, alongside a meticulously curated catalog of datasets. A key contribution is our comprehensive taxonomy of AM subtasks, elucidating how contemporary LLM techniques -- such as prompting, chain-of-thought reasoning, and retrieval augmentation -- have reconfigured their execution. We further detail current LLM architectures and methodologies, critically assess evaluation practices, and delineate pivotal challenges including long-context reasoning, interpretability, and annotation bottlenecks. Conclusively, we highlight emerging trends and propose a forward-looking research agenda for LLM-based computational argumentation, aiming to strategically guide researchers in this rapidly evolving domain.
>
---
#### [replaced 029] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02832v3](http://arxiv.org/pdf/2503.02832v3)**

> **作者:** Songming Zhang; Xue Zhang; Tong Zhang; Bojie Hu; Yufeng Chen; Jinan Xu
>
> **备注:** ACL 2025 Main Conference, code available at: https://github.com/songmzhang/AlignDistil
>
> **摘要:** In modern large language models (LLMs), LLM alignment is of crucial importance and is typically achieved through methods such as reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO). However, in most existing methods for LLM alignment, all tokens in the response are optimized using a sparse, response-level reward or preference annotation. The ignorance of token-level rewards may erroneously punish high-quality tokens or encourage low-quality tokens, resulting in suboptimal performance and slow convergence speed. To address this issue, we propose AlignDistil, an RLHF-equivalent distillation method for token-level reward optimization. Specifically, we introduce the reward learned by DPO into the RLHF objective and theoretically prove the equivalence between this objective and a token-level distillation process, where the teacher distribution linearly combines the logits from the DPO model and a reference model. On this basis, we further bridge the accuracy gap between the reward from the DPO model and the pure reward model, by building a contrastive DPO reward with a normal and a reverse DPO model. Moreover, to avoid under- and over-optimization on different tokens, we design a token adaptive logit extrapolation mechanism to construct an appropriate teacher distribution for each token. Experimental results demonstrate the superiority of our AlignDistil over existing methods and showcase fast convergence due to its token-level distributional reward optimization.
>
---
#### [replaced 030] WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16199v2](http://arxiv.org/pdf/2507.16199v2)**

> **作者:** Zipeng Ling; Yuehao Tang; Shuliang Liu; Junqi Yang; Shenghong Fu; Yao Wan; Kejia Huang; Chen Huang; Zhichao Hou; Xuming Hu
>
> **摘要:** Large Language Models (LLMs) frequently output the label Unknown, yet current evaluations focus almost exclusively on whether such answers are honest rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon Vague Perception. And thus we introduce a framework that quantifies the proportion of Unknown responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct Known or correct Unknown with valid reasoning. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the potential reasoning ability of LLMs and providing a new perspective on solving the Vague Perception phenomenon.
>
---
#### [replaced 031] Speech as a Multimodal Digital Phenotype for Multi-Task LLM-based Mental Health Prediction
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.23822v3](http://arxiv.org/pdf/2505.23822v3)**

> **作者:** Mai Ali; Christopher Lucasius; Tanmay P. Patel; Madison Aitken; Jacob Vorstman; Peter Szatmari; Marco Battaglia; Deepa Kundur
>
> **备注:** 6 pages, 1 figure, 3 tables. The corresponding author is Mai Ali (maia dot ali at mail dot utoronto dot ca). Christopher Lucasius and Tanmay P. Patel contributed equally
>
> **摘要:** Speech is a noninvasive digital phenotype that can offer valuable insights into mental health conditions, but it is often treated as a single modality. In contrast, we propose the treatment of patient speech data as a trimodal multimedia data source for depression detection. This study explores the potential of large language model-based architectures for speech-based depression prediction in a multimodal regime that integrates speech-derived text, acoustic landmarks, and vocal biomarkers. Adolescent depression presents a significant challenge and is often comorbid with multiple disorders, such as suicidal ideation and sleep disturbances. This presents an additional opportunity to integrate multi-task learning (MTL) into our study by simultaneously predicting depression, suicidal ideation, and sleep disturbances using the multimodal formulation. We also propose a longitudinal analysis strategy that models temporal changes across multiple clinical interactions, allowing for a comprehensive understanding of the conditions' progression. Our proposed approach, featuring trimodal, longitudinal MTL is evaluated on the Depression Early Warning dataset. It achieves a balanced accuracy of 70.8%, which is higher than each of the unimodal, single-task, and non-longitudinal methods.
>
---
#### [replaced 032] Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.13926v2](http://arxiv.org/pdf/2501.13926v2)**

> **作者:** Ziyu Guo; Renrui Zhang; Chengzhuo Tong; Zhizheng Zhao; Rui Huang; Haoquan Zhang; Manyuan Zhang; Jiaming Liu; Shanghang Zhang; Peng Gao; Hongsheng Li; Pheng-Ann Heng
>
> **备注:** Journal Version. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied to verifying and reinforcing image generation scenarios. In this paper, we provide the first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. We focus on three techniques: scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance. Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image, which is the first to incorporate reflection in autoregressive image generation. Using our investigated reasoning strategies, we enhance a baseline model, Show-o, to achieve superior results, with a significant +24% improvement on the GenEval benchmark, surpassing Stable Diffusion 3 by +15%. We hope our study provides unique insights and paves a new path for integrating CoT reasoning with autoregressive image generation. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT
>
---
#### [replaced 033] OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17352v2](http://arxiv.org/pdf/2503.17352v2)**

> **作者:** Yihe Deng; Hritik Bansal; Fan Yin; Nanyun Peng; Wei Wang; Kai-Wei Chang
>
> **备注:** 23 pages, 11 figures, 8 tables
>
> **摘要:** We introduce OpenVLThinker, one of the first open-source large vision-language models (LVLMs) to exhibit sophisticated chain-of-thought reasoning, achieving notable performance gains on challenging visual reasoning tasks. While text-based reasoning models (e.g., Deepseek R1) show promising results in text-only tasks, distilling their reasoning into LVLMs via supervised fine-tuning (SFT) often results in performance degradation due to imprecise visual grounding. Conversely, purely reinforcement learning (RL)-based methods face a large search space, hindering the emergence of reflective behaviors in smaller models (e.g., 7B LVLMs). Surprisingly, alternating between SFT and RL ultimately results in significant performance improvements after a few iterations. Our analysis reveals that the base model rarely exhibits reasoning behaviors initially, but SFT effectively surfaces these latent actions and narrows the RL search space, accelerating the development of reasoning capabilities. Each subsequent RL stage further refines the model's reasoning skills, producing higher-quality SFT data for continued self-improvement. OpenVLThinker-7B consistently advances performance across six benchmarks demanding mathematical and general reasoning, notably improving MathVista by 3.8%, EMMA by 2.4%, and HallusionBench by 1.6%. Beyond demonstrating the synergy between SFT and RL for complex reasoning tasks, our findings provide early evidence towards achieving R1-style reasoning in multimodal contexts. The code, model and data are held at https://github.com/yihedeng9/OpenVLThinker.
>
---
#### [replaced 034] Modeling Public Perceptions of Science in Media
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.16622v2](http://arxiv.org/pdf/2506.16622v2)**

> **作者:** Jiaxin Pei; Dustin Wright; Isabelle Augenstein; David Jurgens
>
> **摘要:** Effectively engaging the public with science is vital for fostering trust and understanding in our scientific community. Yet, with an ever-growing volume of information, science communicators struggle to anticipate how audiences will perceive and interact with scientific news. In this paper, we introduce a computational framework that models public perception across twelve dimensions, such as newsworthiness, importance, and surprisingness. Using this framework, we create a large-scale science news perception dataset with 10,489 annotations from 2,101 participants from diverse US and UK populations, providing valuable insights into public responses to scientific information across domains. We further develop NLP models that predict public perception scores with a strong performance. Leveraging the dataset and model, we examine public perception of science from two perspectives: (1) Perception as an outcome: What factors affect the public perception of scientific information? (2) Perception as a predictor: Can we use the estimated perceptions to predict public engagement with science? We find that individuals' frequency of science news consumption is the driver of perception, whereas demographic factors exert minimal influence. More importantly, through a large-scale analysis and carefully designed natural experiment on Reddit, we demonstrate that the estimated public perception of scientific information has direct connections with the final engagement pattern. Posts with more positive perception scores receive significantly more comments and upvotes, which is consistent across different scientific information and for the same science, but are framed differently. Overall, this research underscores the importance of nuanced perception modeling in science communication, offering new pathways to predict public interest and engagement with scientific content.
>
---
#### [replaced 035] Language Detection by Means of the Minkowski Norm: Identification Through Character Bigrams and Frequency Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16284v2](http://arxiv.org/pdf/2507.16284v2)**

> **作者:** Paul-Andrei Pogăcean; Sanda-Maria Avram
>
> **摘要:** The debate surrounding language identification has gained renewed attention in recent years, especially with the rapid evolution of AI-powered language models. However, the non-AI-based approaches to language identification have been overshadowed. This research explores a mathematical implementation of an algorithm for language determinism by leveraging monograms and bigrams frequency rankings derived from established linguistic research. The datasets used comprise texts varying in length, historical period, and genre, including short stories, fairy tales, and poems. Despite these variations, the method achieves over 80\% accuracy on texts shorter than 150 characters and reaches 100\% accuracy for longer texts. These results demonstrate that classical frequency-based approaches remain effective and scalable alternatives to AI-driven models for language detection.
>
---
#### [replaced 036] Towards Detecting Persuasion on Social Media: From Model Development to Insights on Persuasion Strategies
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.13844v2](http://arxiv.org/pdf/2503.13844v2)**

> **作者:** Elyas Meguellati; Stefano Civelli; Pietro Bernardelle; Shazia Sadiq; Irwin King; Gianluca Demartini
>
> **摘要:** Political advertising plays a pivotal role in shaping public opinion and influencing electoral outcomes, often through subtle persuasive techniques embedded in broader propaganda strategies. Detecting these persuasive elements is crucial for enhancing voter awareness and ensuring transparency in democratic processes. This paper presents an integrated approach that bridges model development and real-world application through two interconnected studies. First, we introduce a lightweight model for persuasive text detection that achieves state-of-the-art performance in Subtask 3 of SemEval 2023 Task 3 while requiring significantly fewer computational resources and training data than existing methods. Second, we demonstrate the model's practical utility by collecting the Australian Federal Election 2022 Facebook Ads (APA22) dataset, partially annotating a subset for persuasion, and fine-tuning the model to adapt from mainstream news to social media content. We then apply the fine-tuned model to label the remainder of the APA22 dataset, revealing distinct patterns in how political campaigns leverage persuasion through different funding strategies, word choices, demographic targeting, and temporal shifts in persuasion intensity as election day approaches. Our findings not only underscore the necessity of domain-specific modeling for analyzing persuasion on social media but also show how uncovering these strategies can enhance transparency, inform voters, and promote accountability in digital campaigns.
>
---
#### [replaced 037] Multi-Level Explanations for Generative Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.14459v2](http://arxiv.org/pdf/2403.14459v2)**

> **作者:** Lucas Monteiro Paes; Dennis Wei; Hyo Jin Do; Hendrik Strobelt; Ronny Luss; Amit Dhurandhar; Manish Nagireddy; Karthikeyan Natesan Ramamurthy; Prasanna Sattigeri; Werner Geyer; Soumya Ghosh
>
> **备注:** Accepted as an oral presentation at ACL 2025. Code available at https://github.com/IBM/ICX360
>
> **摘要:** Despite the increasing use of large language models (LLMs) for context-grounded tasks like summarization and question-answering, understanding what makes an LLM produce a certain response is challenging. We propose Multi-Level Explanations for Generative Language Models (MExGen), a technique to provide explanations for context-grounded text generation. MExGen assigns scores to parts of the context to quantify their influence on the model's output. It extends attribution methods like LIME and SHAP to LLMs used in context-grounded tasks where (1) inference cost is high, (2) input text is long, and (3) the output is text. We conduct a systematic evaluation, both automated and human, of perturbation-based attribution methods for summarization and question answering. The results show that our framework can provide more faithful explanations of generated output than available alternatives, including LLM self-explanations. We open-source code for MExGen as part of the ICX360 toolkit: https://github$.$com/IBM/ICX360.
>
---
#### [replaced 038] Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15239v2](http://arxiv.org/pdf/2506.15239v2)**

> **作者:** Jaione Bengoetxea; Itziar Gonzalez-Dios; Rodrigo Agerri
>
> **摘要:** In this paper, we evaluate the capacity of current language technologies to understand Basque and Spanish language varieties. We use Natural Language Inference (NLI) as a pivot task and introduce a novel, manually-curated parallel dataset in Basque and Spanish, along with their respective variants. Our empirical analysis of crosslingual and in-context learning experiments using encoder-only and decoder-based Large Language Models (LLMs) shows a performance drop when handling linguistic variation, especially in Basque. Error analysis suggests that this decline is not due to lexical overlap, but rather to the linguistic variation itself. Further ablation experiments indicate that encoder-only models particularly struggle with Western Basque, which aligns with linguistic theory that identifies peripheral dialects (e.g., Western) as more distant from the standard. All data and code are publicly available.
>
---
#### [replaced 039] Fairness Evaluation of Large Language Models in Academic Library Reference Services
- **分类: cs.CL; cs.AI; cs.DL**

- **链接: [http://arxiv.org/pdf/2507.04224v2](http://arxiv.org/pdf/2507.04224v2)**

> **作者:** Haining Wang; Jason Clark; Yueru Yan; Star Bradley; Ruiyang Chen; Yiqiong Zhang; Hengyi Fu; Zuoyu Tian
>
> **摘要:** As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We found no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrated nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services.
>
---
#### [replaced 040] MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23404v4](http://arxiv.org/pdf/2505.23404v4)**

> **作者:** Mingyu Yu; Wei Wang; Yanjie Wei; Sujuan Qin; Fei Gao; Wenmin Li
>
> **摘要:** Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.
>
---
#### [replaced 041] 3LM: Bridging Arabic, STEM, and Code through Benchmarking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15850v2](http://arxiv.org/pdf/2507.15850v2)**

> **作者:** Basma El Amel Boussaha; Leen AlQadi; Mugariya Farooq; Shaikha Alsuwaidi; Giulia Campesan; Ahmed Alzubaidi; Mohammed Alyafeai; Hakim Hacid
>
> **摘要:** Arabic is one of the most widely spoken languages in the world, yet efforts to develop and evaluate Large Language Models (LLMs) for Arabic remain relatively limited. Most existing Arabic benchmarks focus on linguistic, cultural, or religious content, leaving a significant gap in domains like STEM and code which are increasingly relevant for real-world LLM applications. To help bridge this gap, we present 3LM, a suite of three benchmarks designed specifically for Arabic. The first is a set of STEM-related question-answer pairs, naturally sourced from Arabic textbooks and educational worksheets. The second consists of synthetically generated STEM questions, created using the same sources. The third benchmark focuses on code generation, built through a careful translation of two widely used code benchmarks, incorporating a human-in-the-loop process with several rounds of review to ensure high-quality and faithful translations. We release all three benchmarks publicly to support the growth of Arabic LLM research in these essential but underrepresented areas.
>
---
#### [replaced 042] Cautious Next Token Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03038v2](http://arxiv.org/pdf/2507.03038v2)**

> **作者:** Yizhou Wang; Lingzhi Zhang; Yue Bai; Mang Tik Chiu; Zhengmian Hu; Mingyuan Zhang; Qihua Dong; Yu Yin; Sohrab Amirghodsi; Yun Fu
>
> **备注:** ACL 2025
>
> **摘要:** Next token prediction paradigm has been prevailing for autoregressive models in the era of LLMs. The current default sampling choice for popular LLMs is temperature scaling together with nucleus sampling to balance diversity and coherence. Nevertheless, such approach leads to inferior performance in various NLP tasks when the model is not certain about testing questions. To this end, we propose a brand new training-free decoding strategy, dubbed as Cautious Next Token Prediction (CNTP). In the decoding process, if the model has comparatively high prediction entropy at a certain step, we sample multiple trials starting from the step independently and stop when encountering any punctuation. Then we select the trial with the lowest perplexity score viewed as the most probable and reliable trial path given the model's capacity. The trial number is negatively correlated with the prediction confidence, i.e., the less confident the model is, the more trials it should sample. This is consistent with human beings' behaviour: when feeling uncertain or unconfident, one tends to think more creatively, exploring multiple thinking paths, to cautiously select the path one feels most confident about. Extensive experiments on both LLMs and MLLMs show that our proposed CNTP approach outperforms existing standard decoding strategies consistently by a clear margin. Moreover, the integration of CNTP with self consistency can further improve over vanilla self consistency. We believe our proposed CNTP has the potential to become one of the default choices for LLM decoding. Code is available at https://github.com/wyzjack/CNTP.
>
---
#### [replaced 043] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14900v2](http://arxiv.org/pdf/2507.14900v2)**

> **作者:** Chongxuan Huang; Yongshi Ye; Biao Fu; Qifeng Su; Xiaodong Shi
>
> **备注:** ACL main 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable multilingual capabilities, however, how to evaluate cross-lingual alignment remains underexplored. Existing alignment benchmarks primarily focus on sentence embeddings, but prior research has shown that neural models tend to induce a non-smooth representation space, which impact of semantic alignment evaluation on low-resource languages. Inspired by neuroscientific findings that similar information activates overlapping neuronal regions, we propose a novel Neuron State-Based Cross-Lingual Alignment (NeuronXA) to assess the cross-lingual a lignment capabilities of LLMs, which offers a more semantically grounded approach to assess cross-lingual alignment. We evaluate NeuronXA on several prominent multilingual LLMs (LLaMA, Qwen, Mistral, GLM, and OLMo) across two transfer tasks and three multilingual benchmarks. The results demonstrate that with only 100 parallel sentence pairs, NeuronXA achieves a Pearson correlation of 0.9556 with downstream tasks performance and 0.8514 with transferability. These findings demonstrate NeuronXA's effectiveness in assessing both cross-lingual alignment and transferability, even with a small dataset. This highlights its potential to advance cross-lingual alignment research and to improve the semantic understanding of multilingual LLMs.
>
---

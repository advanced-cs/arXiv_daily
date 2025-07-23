# 自然语言处理 cs.CL

- **最新发布 62 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型（LLMs）生成写作反馈的能力。为解决LLMs在识别写作问题和提供恰当反馈方面的局限性，作者构建了一个包含1300个故意引入写作问题的故事测试集，并通过自动与人工评估分析模型表现。论文揭示了当前LLMs在写作反馈任务中的优势与不足。**

- **链接: [http://arxiv.org/pdf/2507.16007v1](http://arxiv.org/pdf/2507.16007v1)**

> **作者:** Hannah Rashkin; Elizabeth Clark; Fantine Huot; Mirella Lapata
>
> **备注:** ACL 2025 main conference
>
> **摘要:** Can LLMs provide support to creative writers by giving meaningful writing feedback? In this paper, we explore the challenges and limitations of model-generated writing feedback by defining a new task, dataset, and evaluation frameworks. To study model performance in a controlled manner, we present a novel test set of 1,300 stories that we corrupted to intentionally introduce writing issues. We study the performance of commonly used LLMs in this task with both automatic and human evaluation metrics. Our analysis shows that current models have strong out-of-the-box behavior in many respects -- providing specific and mostly accurate writing feedback. However, models often fail to identify the biggest writing issue in the story and to correctly decide when to offer critical vs. positive feedback.
>
---
#### [new 002] Efficient RL for optimizing conversation level outcomes with an LLM-based tutor
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统与教育技术任务，旨在解决LLM在多轮对话中优化目标与长期教学效果不匹配的问题。论文提出通过低维潜在状态表示学生状态，并基于此优化长期策略，以提升LLM在数学辅导任务中的多轮对话效果。方法更轻量且有效。**

- **链接: [http://arxiv.org/pdf/2507.16252v1](http://arxiv.org/pdf/2507.16252v1)**

> **作者:** Hyunji Nam; Omer Gottesman; Amy Zhang; Dean Foster; Emma Brunskill; Lyle Ungar
>
> **备注:** 9 pages
>
> **摘要:** Large language models (LLMs) built on existing reinforcement learning with human feedback (RLHF) frameworks typically optimize responses based on immediate turn-level human preferences. However, this approach falls short in multi-turn dialogue settings, such as online math tutoring. We propose a method to enhance LLM-based tutors by representing the dialogue history with a lower-dimensional latent state representation of a student and optimizing a long-term policy to determine high-level actions based on the latent state. The goal is to better align the tutor's behavior with the long-term objective of guiding the student towards solving a target math problem on their own. Our model is lightweight, requiring less computational resources than prior work of training the tutor policy end-to-end to directly output the tutor's next utterance. Our experiment results demonstrate that these modifications lead to improved long-term outcomes compared to prompting in LLM-simulated tutoring tasks.
>
---
#### [new 003] Introducing Quality Estimation to Machine Translation Post-editing Workflow: An Empirical Study on Its Usefulness
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于机器翻译后编辑任务，旨在研究质量估计（QE）在英中翻译中的实用性。论文探讨了QE对后编辑效率和学生译者认知的影响，并分析其与机器翻译质量及译者专业水平的交互作用，发现QE能显著提升编辑效率，但其准确性对流程有重要影响。**

- **链接: [http://arxiv.org/pdf/2507.16515v1](http://arxiv.org/pdf/2507.16515v1)**

> **作者:** Siqi Liu; Guangrong Dai; Dechao Li
>
> **备注:** 11 pages, 5 figures, 2 tables. To be published in the Proceedings of the 20th Machine Translation Summit (MT Summit 2025; Geneva, Switzerland)
>
> **摘要:** This preliminary study investigates the usefulness of sentence-level Quality Estimation (QE) in English-Chinese Machine Translation Post-Editing (MTPE), focusing on its impact on post-editing speed and student translators' perceptions. It also explores the interaction effects between QE and MT quality, as well as between QE and translation expertise. The findings reveal that QE significantly reduces post-editing time. The examined interaction effects were not significant, suggesting that QE consistently improves MTPE efficiency across medium- and high-quality MT outputs and among student translators with varying levels of expertise. In addition to indicating potentially problematic segments, QE serves multiple functions in MTPE, such as validating translators' evaluations of MT quality and enabling them to double-check translation outputs. However, interview data suggest that inaccurate QE may hinder post-editing processes. This research provides new insights into the strengths and limitations of QE, facilitating its more effective integration into MTPE workflows to enhance translators' productivity.
>
---
#### [new 004] Language Detection by Means of the Minkowski Norm: Identification Through Character Bigrams and Frequency Analysis
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在解决短文本语言检测问题。通过分析字符单字和双字频率，结合Minkowski范数构建数学算法，实现非AI语言识别。实验表明，该方法在短文本上准确率超80%，长文本和古文中达100%，验证了传统频率方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16284v1](http://arxiv.org/pdf/2507.16284v1)**

> **作者:** Paul-Andrei Pogăcean; Sanda-Maria Avram
>
> **摘要:** The debate surrounding language identification has gained renewed attention in recent years, especially with the rapid evolution of AI-powered language models. However, the non-AI-based approaches to language identification have been overshadowed. This research explores a mathematical implementation of an algorithm for language determinism by leveraging monograms and bigrams frequency rankings derived from established linguistic research. The datasets used comprise texts varying in length, historical period, and genre, including short stories, fairy tales, and poems. Despite these variations, the method achieves over 80\% accuracy on texts shorter than 150 characters and reaches 100\% accuracy for longer texts and older writings. These results demonstrate that classical frequency-based approaches remain effective and scalable alternatives to AI-driven models for language detection.
>
---
#### [new 005] Towards Automated Regulatory Compliance Verification in Financial Auditing with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于金融审计中的监管合规验证任务，旨在解决现有AI系统无法准确验证财务报告是否符合法律要求的问题。论文通过比较开源与闭源大语言模型（如Llama-2和GPT系列）在PwC提供的数据集上的表现，发现Llama-2在检测不合规内容方面优于闭源模型，而GPT-4在多语言场景中表现最佳。**

- **链接: [http://arxiv.org/pdf/2507.16642v1](http://arxiv.org/pdf/2507.16642v1)**

> **作者:** Armin Berger; Lars Hillebrand; David Leonhard; Tobias Deußer; Thiago Bell Felix de Oliveira; Tim Dilmaghani; Mohamed Khaled; Bernd Kliem; Rüdiger Loitz; Christian Bauckhage; Rafet Sifa
>
> **备注:** Accepted and published at BigData 2023, 10 pages, 3 figures, 5 tables
>
> **摘要:** The auditing of financial documents, historically a labor-intensive process, stands on the precipice of transformation. AI-driven solutions have made inroads into streamlining this process by recommending pertinent text passages from financial reports to align with the legal requirements of accounting standards. However, a glaring limitation remains: these systems commonly fall short in verifying if the recommended excerpts indeed comply with the specific legal mandates. Hence, in this paper, we probe the efficiency of publicly available Large Language Models (LLMs) in the realm of regulatory compliance across different model configurations. We place particular emphasis on comparing cutting-edge open-source LLMs, such as Llama-2, with their proprietary counterparts like OpenAI's GPT models. This comparative analysis leverages two custom datasets provided by our partner PricewaterhouseCoopers (PwC) Germany. We find that the open-source Llama-2 70 billion model demonstrates outstanding performance in detecting non-compliance or true negative occurrences, beating all their proprietary counterparts. Nevertheless, proprietary models such as GPT-4 perform the best in a broad variety of scenarios, particularly in non-English contexts.
>
---
#### [new 006] Pixels to Principles: Probing Intuitive Physics Understanding in Multimodal Language Models
- **分类: cs.CL**

- **简介: 论文评估了多模态大语言模型在直觉物理任务中的表现，发现模型在区分物理合理与不合理场景上存在困难。通过分析模型嵌入，发现视觉编码器能捕捉物理线索，但语言模型未能有效利用，导致推理失败。论文指出视觉-语言对齐是关键问题，为改进多模态模型提供方向。**

- **链接: [http://arxiv.org/pdf/2507.16572v1](http://arxiv.org/pdf/2507.16572v1)**

> **作者:** Mohamad Ballout; Serwan Jassim; Elia Bruni
>
> **摘要:** This paper presents a systematic evaluation of state-of-the-art multimodal large language models (MLLMs) on intuitive physics tasks using the GRASP and IntPhys 2 datasets. We assess the open-source models InternVL 2.5, Qwen 2.5 VL, LLaVA-OneVision, and the proprietary Gemini 2.0 Flash Thinking, finding that even the latest models struggle to reliably distinguish physically plausible from implausible scenarios. To go beyond performance metrics, we conduct a probing analysis of model embeddings, extracting intermediate representations at key processing stages to examine how well task-relevant information is preserved. Our results show that, depending on task difficulty, a critical vision-language misalignment can emerge: vision encoders successfully capture physical plausibility cues, but this information is not effectively utilized by the language model, leading to failures in reasoning. This misalignment suggests that the primary limitation of MLLMs in intuitive physics tasks is not the vision component but the ineffective integration of visual and linguistic information. Our findings highlight vision-language alignment as a key area for improvement, offering insights for future MLLMs development.
>
---
#### [new 007] WakenLLM: A Fine-Grained Benchmark for Evaluating LLM Reasoning Potential and Reasoning Process Stability
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型（LLM）的推理能力和推理过程稳定性。它试图解决模型频繁输出“Unknown”标签的问题，区分是因输入本身不确定还是模型能力不足所致。论文提出了一个细粒度基准（WakenLLM），用以量化模型能力不足导致的“Unknown”响应，并探索通过引导提升模型推理性能的可能性。**

- **链接: [http://arxiv.org/pdf/2507.16199v1](http://arxiv.org/pdf/2507.16199v1)**

> **作者:** Zipeng Ling; Yuehao Tang; Shuliang Liu; Junqi Yang; Shenghong Fu; Yao Wan; Kejia Huang; Zhichao Hou; Xuming Hu
>
> **摘要:** Large Language Models (LLMs) frequently output the label \emph{Unknown}, yet current evaluations focus almost exclusively on whether such answers are \emph{honest} rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon \emph{Vague Perception}. And thus we introduce a framework that quantifies the proportion of \emph{Unknown} responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct (\emph{Known}) or intrinsically indeterminate outcomes. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the true reasoning ability of LLMs and providing a new perspective on solving the \emph{Vague Perception} phenomenon.
>
---
#### [new 008] Dutch CrowS-Pairs: Adapting a Challenge Dataset for Measuring Social Biases in Language Models for Dutch
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的社会偏见检测任务，旨在解决语言模型中的社会偏见问题。研究者改编了英文的CrowS-Pairs数据集，创建了荷兰语版本（Dutch CrowS-Pairs），涵盖9类偏见，用于评估荷兰语模型的偏见水平。他们测试了多个荷兰语模型，并分析了不同语言模型间的偏见差异及角色设定对偏见的影响。**

- **链接: [http://arxiv.org/pdf/2507.16442v1](http://arxiv.org/pdf/2507.16442v1)**

> **作者:** Elza Strazda; Gerasimos Spanakis
>
> **备注:** 10 pages, accepted at RANLP 2025 data and code here: https://github.com/jerryspan/Dutch-CrowS-Pairs
>
> **摘要:** Warning: This paper contains explicit statements of offensive stereotypes which might be upsetting. Language models are prone to exhibiting biases, further amplifying unfair and harmful stereotypes. Given the fast-growing popularity and wide application of these models, it is necessary to ensure safe and fair language models. As of recent considerable attention has been paid to measuring bias in language models, yet the majority of studies have focused only on English language. A Dutch version of the US-specific CrowS-Pairs dataset for measuring bias in Dutch language models is introduced. The resulting dataset consists of 1463 sentence pairs that cover bias in 9 categories, such as Sexual orientation, Gender and Disability. The sentence pairs are composed of contrasting sentences, where one of the sentences concerns disadvantaged groups and the other advantaged groups. Using the Dutch CrowS-Pairs dataset, we show that various language models, BERTje, RobBERT, multilingual BERT, GEITje and Mistral-7B exhibit substantial bias across the various bias categories. Using the English and French versions of the CrowS-Pairs dataset, bias was evaluated in English (BERT and RoBERTa) and French (FlauBERT and CamemBERT) language models, and it was shown that English models exhibit the most bias, whereas Dutch models the least amount of bias. Additionally, results also indicate that assigning a persona to a language model changes the level of bias it exhibits. These findings highlight the variability of bias across languages and contexts, suggesting that cultural and linguistic factors play a significant role in shaping model biases.
>
---
#### [new 009] mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言知识图谱构建（mKGC）任务，旨在通过跨语言迁移为资源匮乏语言（如提格利尼亚语和阿姆哈拉语）预测缺失的实体和链接。作者提出mRAKL方法，将任务转化为问答问题，并采用基于检索增强生成（RAG）的模型进行求解。实验表明，与无上下文设置相比，使用BM25检索器的RAG方法提升了性能。**

- **链接: [http://arxiv.org/pdf/2507.16011v1](http://arxiv.org/pdf/2507.16011v1)**

> **作者:** Hellina Hailu Nigatu; Min Li; Maartje ter Hoeve; Saloni Potdar; Sarah Chasins
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Knowledge Graphs represent real-world entities and the relationships between them. Multilingual Knowledge Graph Construction (mKGC) refers to the task of automatically constructing or predicting missing entities and links for knowledge graphs in a multilingual setting. In this work, we reformulate the mKGC task as a Question Answering (QA) task and introduce mRAKL: a Retrieval-Augmented Generation (RAG) based system to perform mKGC. We achieve this by using the head entity and linking relation in a question, and having our model predict the tail entity as an answer. Our experiments focus primarily on two low-resourced languages: Tigrinya and Amharic. We experiment with using higher-resourced languages Arabic and English for cross-lingual transfer. With a BM25 retriever, we find that the RAG-based approach improves performance over a no-context setting. Further, our ablation studies show that with an idealized retrieval system, mRAKL improves accuracy by 4.92 and 8.79 percentage points for Tigrinya and Amharic, respectively.
>
---
#### [new 010] Efficient Compositional Multi-tasking for On-device Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决设备端大语言模型的文本组合多任务处理问题。现有方法多针对单一任务合并，难以应对多任务同时执行的情况。为此，论文提出了一个包含四个组合任务的基准，并设计了适用于资源受限场景的高效方法Learnable Calibration，推动复杂场景下多任务处理能力的发展。**

- **链接: [http://arxiv.org/pdf/2507.16083v1](http://arxiv.org/pdf/2507.16083v1)**

> **作者:** Ondrej Bohdal; Mete Ozay; Jijoong Moon; Kyeng-Hun Lee; Hyeonmok Ko; Umberto Michieli
>
> **摘要:** Adapter parameters provide a mechanism to modify the behavior of machine learning models and have gained significant popularity in the context of large language models (LLMs) and generative AI. These parameters can be merged to support multiple tasks via a process known as task merging. However, prior work on merging in LLMs, particularly in natural language processing, has been limited to scenarios where each test example addresses only a single task. In this paper, we focus on on-device settings and study the problem of text-based compositional multi-tasking, where each test example involves the simultaneous execution of multiple tasks. For instance, generating a translated summary of a long text requires solving both translation and summarization tasks concurrently. To facilitate research in this setting, we propose a benchmark comprising four practically relevant compositional tasks. We also present an efficient method (Learnable Calibration) tailored for on-device applications, where computational resources are limited, emphasizing the need for solutions that are both resource-efficient and high-performing. Our contributions lay the groundwork for advancing the capabilities of LLMs in real-world multi-tasking scenarios, expanding their applicability to complex, resource-constrained use cases.
>
---
#### [new 011] Interpretable Topic Extraction and Word Embedding Learning using row-stochastic DEDICOM
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决主题抽取与词嵌入学习的可解释性问题。通过引入行随机DEDICOM矩阵分解方法，对文本语料的逐点互信息矩阵进行建模，实现词汇中潜在主题聚类的识别并同时学习可解释的词向量表示。研究还提出了一种高效的约束DEDICOM训练方法，并对其主题建模与词嵌入效果进行了定性评估。**

- **链接: [http://arxiv.org/pdf/2507.16695v1](http://arxiv.org/pdf/2507.16695v1)**

> **作者:** Lars Hillebrand; David Biesner; Christian Bauckhage; Rafet Sifa
>
> **备注:** Accepted and published at CD-MAKE 2020, 20 pages, 8 tables, 8 figures
>
> **摘要:** The DEDICOM algorithm provides a uniquely interpretable matrix factorization method for symmetric and asymmetric square matrices. We employ a new row-stochastic variation of DEDICOM on the pointwise mutual information matrices of text corpora to identify latent topic clusters within the vocabulary and simultaneously learn interpretable word embeddings. We introduce a method to efficiently train a constrained DEDICOM algorithm and a qualitative evaluation of its topic modeling and word embedding performance.
>
---
#### [new 012] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成虚假内容的检测问题。通过分析模型内部状态变化，提出新指标ICR Score及检测方法ICR Probe，有效识别并解释虚假生成内容。**

- **链接: [http://arxiv.org/pdf/2507.16488v1](http://arxiv.org/pdf/2507.16488v1)**

> **作者:** Zhenliang Zhang; Xinyu Hu; Huixuan Zhang; Junzhe Zhang; Xiaojun Wan
>
> **备注:** Accepted to ACL 2025 (Main Conference)
>
> **摘要:** Large language models (LLMs) excel at various natural language processing tasks, but their tendency to generate hallucinations undermines their reliability. Existing hallucination detection methods leveraging hidden states predominantly focus on static and isolated representations, overlooking their dynamic evolution across layers, which limits efficacy. To address this limitation, we shift the focus to the hidden state update process and introduce a novel metric, the ICR Score (Information Contribution to Residual Stream), which quantifies the contribution of modules to the hidden states' update. We empirically validate that the ICR Score is effective and reliable in distinguishing hallucinations. Building on these insights, we propose a hallucination detection method, the ICR Probe, which captures the cross-layer evolution of hidden states. Experimental results show that the ICR Probe achieves superior performance with significantly fewer parameters. Furthermore, ablation studies and case analyses offer deeper insights into the underlying mechanism of this method, improving its interpretability.
>
---
#### [new 013] GG-BBQ: German Gender Bias Benchmark for Question Answering
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决德语大语言模型中的性别偏见问题。作者基于英文数据集翻译构建了德语性别偏见测试集GG-BBQ，并通过人工修正提升翻译质量。最终评估多个德语模型，发现均存在性别偏见。**

- **链接: [http://arxiv.org/pdf/2507.16410v1](http://arxiv.org/pdf/2507.16410v1)**

> **作者:** Shalaka Satheesh; Katrin Klug; Katharina Beckh; Héctor Allende-Cid; Sebastian Houben; Teena Hassan
>
> **备注:** Accepted to the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP), taking place on August 1st 2025, as part of ACL 2025 in Vienna
>
> **摘要:** Within the context of Natural Language Processing (NLP), fairness evaluation is often associated with the assessment of bias and reduction of associated harm. In this regard, the evaluation is usually carried out by using a benchmark dataset, for a task such as Question Answering, created for the measurement of bias in the model's predictions along various dimensions, including gender identity. In our work, we evaluate gender bias in German Large Language Models (LLMs) using the Bias Benchmark for Question Answering by Parrish et al. (2022) as a reference. Specifically, the templates in the gender identity subset of this English dataset were machine translated into German. The errors in the machine translated templates were then manually reviewed and corrected with the help of a language expert. We find that manual revision of the translation is crucial when creating datasets for gender bias evaluation because of the limitations of machine translation from English to a language such as German with grammatical gender. Our final dataset is comprised of two subsets: Subset-I, which consists of group terms related to gender identity, and Subset-II, where group terms are replaced with proper names. We evaluate several LLMs used for German NLP on this newly created dataset and report the accuracy and bias scores. The results show that all models exhibit bias, both along and against existing social stereotypes.
>
---
#### [new 014] Beyond Isolated Dots: Benchmarking Structured Table Construction as Deep Knowledge Extraction
- **分类: cs.CL**

- **简介: 该论文提出了一项名为Arranged and Organized Extraction Benchmark (AOE)的新任务，旨在解决大型语言模型（LLMs）在从复杂文档中提取并结构化信息能力不足的问题。论文通过构建一个包含11个多样化任务的双语基准测试，评估LLMs将碎片化信息重构为有序表格的能力。**

- **链接: [http://arxiv.org/pdf/2507.16271v1](http://arxiv.org/pdf/2507.16271v1)**

> **作者:** Tianyun Zhong; Guozhao Mo; Yanjiang Liu; Yihan Chen; Lingdi Kong; Xuanang Chen; Yaojie Lu; Hongyu Lin; Ben He; Le Sun
>
> **摘要:** With the emergence of large language models (LLMs), there is an expectation that LLMs can effectively extract explicit information from complex real-world documents (e.g., papers, reports). However, most LLMs generate paragraph-style answers that are chaotic, disorganized, and untraceable. To bridge this gap, we introduce the Arranged and Organized Extraction Benchmark (AOE), a new bilingual benchmark with data and documents of varying lengths designed to systematically evaluate the ability of LLMs to comprehend fragmented documents and reconstruct isolated information into one organized table. Unlike conventional text-to-table tasks, which rely on fixed schema and narrow task domains, AOE includes 11 carefully crafted tasks across three diverse domains, requiring models to generate context-specific schema tailored to varied input queries. In the experiment, we evaluated both open-source and closed-source state-of-the-art LLMs. The results show that even the most advanced models struggled significantly. The benchmark is available at https://huggingface.co/datasets/tianyumyum/AOE.
>
---
#### [new 015] Advancing Risk and Quality Assurance: A RAG Chatbot for Improved Regulatory Compliance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于风险与质量保障任务，旨在解决高度监管行业中员工处理大量合规查询效率低下的问题。论文提出一种结合检索增强生成（RAG）、大语言模型和混合搜索的聊天机器人系统，提升监管合规查询的准确性和效率，并通过实际部署和超参数分析验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.16711v1](http://arxiv.org/pdf/2507.16711v1)**

> **作者:** Lars Hillebrand; Armin Berger; Daniel Uedelhoven; David Berghaus; Ulrich Warning; Tim Dilmaghani; Bernd Kliem; Thomas Schmid; Rüdiger Loitz; Rafet Sifa
>
> **备注:** Accepted and published at BigData 2024, 3 pages, 3 tables, 2 figures
>
> **摘要:** Risk and Quality (R&Q) assurance in highly regulated industries requires constant navigation of complex regulatory frameworks, with employees handling numerous daily queries demanding accurate policy interpretation. Traditional methods relying on specialized experts create operational bottlenecks and limit scalability. We present a novel Retrieval Augmented Generation (RAG) system leveraging Large Language Models (LLMs), hybrid search and relevance boosting to enhance R&Q query processing. Evaluated on 124 expert-annotated real-world queries, our actively deployed system demonstrates substantial improvements over traditional RAG approaches. Additionally, we perform an extensive hyperparameter analysis to compare and evaluate multiple configuration setups, delivering valuable insights to practitioners.
>
---
#### [new 016] Deep Researcher with Test-Time Diffusion
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决现有深度研究代理在生成复杂长篇报告时性能不足的问题。作者提出TTD-DR框架，将报告生成视为扩散过程，通过初步草稿与检索机制动态结合，迭代优化内容，提升报告质量与连贯性。**

- **链接: [http://arxiv.org/pdf/2507.16075v1](http://arxiv.org/pdf/2507.16075v1)**

> **作者:** Rujun Han; Yanfei Chen; Zoey CuiZhu; Lesly Miculicich; Guan Sun; Yuanjun Bi; Weiming Wen; Hui Wan; Chunfeng Wen; Solène Maître; George Lee; Vishy Tirumalashetty; Emily Xue; Zizhao Zhang; Salem Haykal; Burak Gokturk; Tomas Pfister; Chen-Yu Lee
>
> **摘要:** Deep research agents, powered by Large Language Models (LLMs), are rapidly advancing; yet, their performance often plateaus when generating complex, long-form research reports using generic test-time scaling algorithms. Drawing inspiration from the iterative nature of human research, which involves cycles of searching, reasoning, and revision, we propose the Test-Time Diffusion Deep Researcher (TTD-DR). This novel framework conceptualizes research report generation as a diffusion process. TTD-DR initiates this process with a preliminary draft, an updatable skeleton that serves as an evolving foundation to guide the research direction. The draft is then iteratively refined through a "denoising" process, which is dynamically informed by a retrieval mechanism that incorporates external information at each step. The core process is further enhanced by a self-evolutionary algorithm applied to each component of the agentic workflow, ensuring the generation of high-quality context for the diffusion process. This draft-centric design makes the report writing process more timely and coherent while reducing information loss during the iterative search process. We demonstrate that our TTD-DR achieves state-of-the-art results on a wide array of benchmarks that require intensive search and multi-hop reasoning, significantly outperforming existing deep research agents.
>
---
#### [new 017] MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于科学推理任务，旨在解决缺乏高质量科学推理数据的问题。作者构建了TextbookReasoning和MegaScience两个数据集，包含百万级科学问题与答案，并开发评估系统验证模型性能。训练结果显示，使用MegaScience微调的模型表现优于官方指令模型，尤其对大规模模型效果更佳，推动了科学推理研究发展。**

- **链接: [http://arxiv.org/pdf/2507.16812v1](http://arxiv.org/pdf/2507.16812v1)**

> **作者:** Run-Ze Fan; Zengzhi Wang; Pengfei Liu
>
> **备注:** 39 pages; Github: https://github.com/GAIR-NLP/MegaScience; HF: https://huggingface.co/MegaScience
>
> **摘要:** Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the open-source community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present TextbookReasoning, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce MegaScience, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MegaScience, which significantly outperform the corresponding official instruct models in average performance. In addition, MegaScience exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning. We release our data curation pipeline, evaluation system, datasets, and seven trained models to the community to advance scientific reasoning research.
>
---
#### [new 018] Learning Text Styles: A Study on Transfer, Attribution, and Verification
- **分类: cs.CL**

- **简介: 该论文研究文本风格的计算理解与操控，涉及文本风格迁移、作者归属与验证任务。旨在解决风格迁移中保持内容不变、通过风格特征识别作者，以及判断两文本是否同源的问题。工作包括：基于大模型的参数高效适配、对比解耦风格特征及基于指令的可解释验证微调。**

- **链接: [http://arxiv.org/pdf/2507.16530v1](http://arxiv.org/pdf/2507.16530v1)**

> **作者:** Zhiqiang Hu
>
> **备注:** PhD thesis
>
> **摘要:** This thesis advances the computational understanding and manipulation of text styles through three interconnected pillars: (1) Text Style Transfer (TST), which alters stylistic properties (e.g., sentiment, formality) while preserving content; (2)Authorship Attribution (AA), identifying the author of a text via stylistic fingerprints; and (3) Authorship Verification (AV), determining whether two texts share the same authorship. We address critical challenges in these areas by leveraging parameter-efficient adaptation of large language models (LLMs), contrastive disentanglement of stylistic features, and instruction-based fine-tuning for explainable verification.
>
---
#### [new 019] FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents
- **分类: cs.CL**

- **简介: 该论文提出FinResearchBench，属于金融研究代理评估任务，旨在解决现有评估框架不足的问题。它通过逻辑树分析，自动评估代理在7类金融任务中的表现，覆盖70个典型问题，提供全面可靠评价。**

- **链接: [http://arxiv.org/pdf/2507.16248v1](http://arxiv.org/pdf/2507.16248v1)**

> **作者:** Run Sun; Zuo Bai; Wentao Zhang; Yuxiang Zhang; Li Zhao; Shan Sun; Zhengwen Qiu
>
> **摘要:** Recently, AI agents are rapidly evolving in intelligence and widely used in professional research applications, such as STEM, software development, finance, etc. Among these AI agents, deep research agent is a key category as it can perform long-horizon tasks and solve problems of greater complexity. However, there are few evaluation frameworks and benchmarks that systematically and automatically investigate the capabilities of these research agents. Furthermore, financial research problems have distinct complexity and subtlety. To fill in the gap, we propose FinResearchBench, which is a logic tree based Agent-as-a-Judge and targets specifically for the financial research agents. It provides a comprehensive and automatic assessment of the research agents across 7 key types of tasks in the financial research domain. The contributions of this work are two-folded: (1) the first and innovative Agent-as-a-Judge system that extracts the logic tree of the research outcome and uses it as the intermediate information to present a comprehensive, reliable and robust evaluation; (2) finance oriented that it covers 70 typical financial research questions, spreading across 7 frequently encountered types of tasks in the domain.
>
---
#### [new 020] The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在模拟不同社会人口群体观点时的表现。论文系统研究了不同提示策略对15个交叉群体模拟效果的影响，发现模型在边缘群体上表现较差，但特定提示方法能改善效果，并为设计社会人口提示提供了实用指导。**

- **链接: [http://arxiv.org/pdf/2507.16076v1](http://arxiv.org/pdf/2507.16076v1)**

> **作者:** Marlene Lutz; Indira Sen; Georg Ahnert; Elisa Rogers; Markus Strohmaier
>
> **摘要:** Persona prompting is increasingly used in large language models (LLMs) to simulate views of various sociodemographic groups. However, how a persona prompt is formulated can significantly affect outcomes, raising concerns about the fidelity of such simulations. Using five open-source LLMs, we systematically examine how different persona prompt strategies, specifically role adoption formats and demographic priming strategies, influence LLM simulations across 15 intersectional demographic groups in both open- and closed-ended tasks. Our findings show that LLMs struggle to simulate marginalized groups, particularly nonbinary, Hispanic, and Middle Eastern identities, but that the choice of demographic priming and role adoption strategy significantly impacts their portrayal. Specifically, we find that prompting in an interview-style format and name-based priming can help reduce stereotyping and improve alignment. Surprisingly, smaller models like OLMo-2-7B outperform larger ones such as Llama-3.3-70B. Our findings offer actionable guidance for designing sociodemographic persona prompts in LLM-based simulation studies.
>
---
#### [new 021] Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长视野推理中的上下文限制问题。作者提出了线程推理模型（TIM）和推理运行系统（TIMRUN），通过递归分解问题、构建推理树，并优化GPU内存使用，实现超越上下文限制的高效推理。**

- **链接: [http://arxiv.org/pdf/2507.16784v1](http://arxiv.org/pdf/2507.16784v1)**

> **作者:** Hongyin Luo; Nathaniel Morgan; Tina Li; Derek Zhao; Ai Vy Ngo; Philip Schroeder; Lijie Yang; Assaf Ben-Kish; Jack O'Brien; James Glass
>
> **备注:** Research preview
>
> **摘要:** To break the context limits of large language models (LLMs) that bottleneck reasoning accuracy and efficiency, we propose the Thread Inference Model (TIM), a family of LLMs trained for recursive and decompositional problem solving, and TIMRUN, an inference runtime enabling long-horizon structured reasoning beyond context limits. Together, TIM hosted on TIMRUN supports virtually unlimited working memory and multi-hop tool calls within a single language model inference, overcoming output limits, positional-embedding constraints, and GPU-memory bottlenecks. Performance is achieved by modeling natural language as reasoning trees measured by both length and depth instead of linear sequences. The reasoning trees consist of tasks with thoughts, recursive subtasks, and conclusions based on the concept we proposed in Schroeder et al, 2025. During generation, we maintain a working memory that retains only the key-value states of the most relevant context tokens, selected by a rule-based subtask-pruning mechanism, enabling reuse of positional embeddings and GPU memory pages throughout reasoning. Experimental results show that our system sustains high inference throughput, even when manipulating up to 90% of the KV cache in GPU memory. It also delivers accurate reasoning on mathematical tasks and handles information retrieval challenges that require long-horizon reasoning and multi-hop tool use.
>
---
#### [new 022] Unpacking Ambiguity: The Interaction of Polysemous Discourse Markers and Non-DM Signals
- **分类: cs.CL**

- **简介: 该论文研究英语中歧义连词（DMs）与非连词信号的共现关系及消歧机制，探讨体裁对模式的影响。任务为自然语言处理中的语篇分析，旨在揭示DM多义性与非DM信号的互动规律。采用eRST框架，通过相关性和回归分析验证假设。**

- **链接: [http://arxiv.org/pdf/2507.16748v1](http://arxiv.org/pdf/2507.16748v1)**

> **作者:** Jingni Wu; Amir Zeldes
>
> **摘要:** Discourse markers (DMs) like 'but' or 'then' are crucial for creating coherence in discourse, yet they are often replaced by or co-occur with non-DMs ('in the morning' can mean the same as 'then'), and both can be ambiguous ('since' can refer to time or cause). The interaction mechanism between such signals remains unclear but pivotal for their disambiguation. In this paper we investigate the relationship between DM polysemy and co-occurrence of non-DM signals in English, as well as the influence of genre on these patterns. Using the framework of eRST, we propose a graded definition of DM polysemy, and conduct correlation and regression analyses to examine whether polysemous DMs are accompanied by more numerous and diverse non-DM signals. Our findings reveal that while polysemous DMs do co-occur with more diverse non-DMs, the total number of co-occurring signals does not necessarily increase. Moreover, genre plays a significant role in shaping DM-signal interactions.
>
---
#### [new 023] PromptAL: Sample-Aware Dynamic Soft Prompts for Few-Shot Active Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于主动学习任务，旨在解决少样本场景下经验分布偏离目标分布导致决策边界不优、样本选择不佳的问题。作者提出了PromptAL框架，利用未标注数据构建动态软提示，优化模型预测分布与决策边界，并结合不确定性与多样性选择高质量样本，提升了少样本主动学习效果。**

- **链接: [http://arxiv.org/pdf/2507.16424v1](http://arxiv.org/pdf/2507.16424v1)**

> **作者:** Hui Xiang; Jinqiao Shi; Ting Zhang; Xiaojie Zhao; Yong Liu; Yong Ma
>
> **摘要:** Active learning (AL) aims to optimize model training and reduce annotation costs by selecting the most informative samples for labeling. Typically, AL methods rely on the empirical distribution of labeled data to define the decision boundary and perform uncertainty or diversity estimation, subsequently identifying potential high-quality samples. In few-shot scenarios, the empirical distribution often diverges significantly from the target distribution, causing the decision boundary to shift away from its optimal position. However, existing methods overlook the role of unlabeled samples in enhancing the empirical distribution to better align with the target distribution, resulting in a suboptimal decision boundary and the selection of samples that inadequately represent the target distribution. To address this, we propose a hybrid AL framework, termed \textbf{PromptAL} (Sample-Aware Dynamic Soft \textbf{Prompts} for Few-Shot \textbf{A}ctive \textbf{L}earning). This framework accounts for the contribution of each unlabeled data point in aligning the current empirical distribution with the target distribution, thereby optimizing the decision boundary. Specifically, PromptAL first leverages unlabeled data to construct sample-aware dynamic soft prompts that adjust the model's predictive distribution and decision boundary. Subsequently, based on the adjusted decision boundary, it integrates uncertainty estimation with both global and local diversity to select high-quality samples that more accurately represent the target distribution. Experimental results on six in-domain and three out-of-domain datasets show that PromptAL achieves superior performance over nine baselines. Our codebase is openly accessible.
>
---
#### [new 024] BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言背景下，孟加拉区域方言中的仇恨言论检测问题。现有数据集忽视了非正式、文化丰富的方言表达，导致检测能力有限。论文构建了首个涵盖巴里萨尔、诺阿卡利和吉大港方言的多方言孟加拉仇恨言论数据集BIDWESH，共标注9,183条数据，包含仇恨类型与目标群体信息，为开发方言敏感的NLP工具提供支持。**

- **链接: [http://arxiv.org/pdf/2507.16183v1](http://arxiv.org/pdf/2507.16183v1)**

> **作者:** Azizul Hakim Fayaz; MD. Shorif Uddin; Rayhan Uddin Bhuiyan; Zakia Sultana; Md. Samiul Islam; Bidyarthi Paul; Tashreef Muhammad; Shahriar Manzoor
>
> **摘要:** Hate speech on digital platforms has become a growing concern globally, especially in linguistically diverse countries like Bangladesh, where regional dialects play a major role in everyday communication. Despite progress in hate speech detection for standard Bangla, Existing datasets and systems fail to address the informal and culturally rich expressions found in dialects such as Barishal, Noakhali, and Chittagong. This oversight results in limited detection capability and biased moderation, leaving large sections of harmful content unaccounted for. To address this gap, this study introduces BIDWESH, the first multi-dialectal Bangla hate speech dataset, constructed by translating and annotating 9,183 instances from the BD-SHS corpus into three major regional dialects. Each entry was manually verified and labeled for hate presence, type (slander, gender, religion, call to violence), and target group (individual, male, female, group), ensuring linguistic and contextual accuracy. The resulting dataset provides a linguistically rich, balanced, and inclusive resource for advancing hate speech detection in Bangla. BIDWESH lays the groundwork for the development of dialect-sensitive NLP tools and contributes significantly to equitable and context-aware content moderation in low-resource language settings.
>
---
#### [new 025] iShumei-Chinchunmei at SemEval-2025 Task 4: A balanced forgetting and retention multi-task framework using effective unlearning loss
- **分类: cs.CL**

- **简介: 该论文参与SemEval-2025任务4，解决大型语言模型遗忘敏感内容问题。提出“有效遗忘损失”以平衡遗忘与保留能力，结合多任务框架提升遗忘效率与控制性。最终在比赛中排名第五。**

- **链接: [http://arxiv.org/pdf/2507.16263v1](http://arxiv.org/pdf/2507.16263v1)**

> **作者:** Yujian Sun; Tian Li
>
> **摘要:** As the Large Language Model (LLM) gains widespread adoption, increasing attention has been given to the challenge of making LLM forget non-compliant data memorized during its pre-training. Machine Unlearning focuses on efficiently erasing sensitive information from LLM under limited computational resources. To advance research in this area, SemEval 2025 Task 4: "Unlearning Sensitive Content from Large Language Models" introduces three unlearning datasets and establishes a benchmark by evaluating both forgetting effectiveness and the preservation of standard capabilities. In this work, we propose a more controllable forgetting loss, Effective Unlearning Loss, and explore its integration with various techniques to achieve more efficient and controlled unlearning. Our system ultimately ranked 5th on the competition leaderboard.
>
---
#### [new 026] Enhancing Hindi NER in Low Context: A Comparative study of Transformer-based models with vs. without Retrieval Augmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的命名实体识别（NER）任务，旨在解决低语境下印地语NER性能不足的问题。研究通过对比基于Transformer的模型（如MuRIL、XLM-R、Llama系列、GPT3.5）在是否引入检索增强（RA）情况下的表现，发现RA能显著提升NER效果，尤其在低语境和资源有限的语言场景中。**

- **链接: [http://arxiv.org/pdf/2507.16002v1](http://arxiv.org/pdf/2507.16002v1)**

> **作者:** Sumit Singh; Rohit Mishra; Uma Shanker Tiwary
>
> **摘要:** One major challenge in natural language processing is named entity recognition (NER), which identifies and categorises named entities in textual input. In order to improve NER, this study investigates a Hindi NER technique that makes use of Hindi-specific pretrained encoders (MuRIL and XLM-R) and Generative Models ( Llama-2-7B-chat-hf (Llama2-7B), Llama-2-70B-chat-hf (Llama2-70B), Llama-3-70B-Instruct (Llama3-70B) and GPT3.5-turbo), and augments the data with retrieved data from external relevant contexts, notably from Wikipedia. We have fine-tuned MuRIL, XLM-R and Llama2-7B with and without RA. However, Llama2-70B, lama3-70B and GPT3.5-turbo are utilised for few-shot NER generation. Our investigation shows that the mentioned language models (LMs) with Retrieval Augmentation (RA) outperform baseline methods that don't incorporate RA in most cases. The macro F1 scores for MuRIL and XLM-R are 0.69 and 0.495, respectively, without RA and increase to 0.70 and 0.71, respectively, in the presence of RA. Fine-tuned Llama2-7B outperforms Llama2-7B by a significant margin. On the other hand the generative models which are not fine-tuned also perform better with augmented data. GPT3.5-turbo adopted RA well; however, Llama2-70B and llama3-70B did not adopt RA with our retrieval context. The findings show that RA significantly improves performance, especially for low-context data. This study adds significant knowledge about how best to use data augmentation methods and pretrained models to enhance NER performance, particularly in languages with limited resources.
>
---
#### [new 027] eSapiens's DEREK Module: Deep Extraction & Reasoning Engine for Knowledge with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出eSapiens的DEREK模块，用于企业文档问答的检索增强生成系统。该模块解决多格式文档中精准问答与可追溯性问题，采用混合索引、查询优化、重排序及验证机制，提升准确率与可信度，适用于法律、金融等高风险领域。**

- **链接: [http://arxiv.org/pdf/2507.15863v1](http://arxiv.org/pdf/2507.15863v1)**

> **作者:** Isaac Shi; Zeyuan Li; Fan Liu; Wenli Wang; Lewei He; Yang Yang; Tianyu Shi
>
> **备注:** 8 pages;1 figure;5 tables
>
> **摘要:** We present the DEREK (Deep Extraction & Reasoning Engine for Knowledge) Module, a secure and scalable Retrieval-Augmented Generation pipeline designed specifically for enterprise document question answering. Designed and implemented by eSapiens, the system ingests heterogeneous content (PDF, Office, web), splits it into 1,000-token overlapping chunks, and indexes them in a hybrid HNSW+BM25 store. User queries are refined by GPT-4o, retrieved via combined vector+BM25 search, reranked with Cohere, and answered by an LLM using CO-STAR prompt engineering. A LangGraph verifier enforces citation overlap, regenerating answers until every claim is grounded. On four LegalBench subsets, 1000-token chunks improve Recall@50 by approximately 1 pp and hybrid+rerank boosts Precision@10 by approximately 7 pp; the verifier raises TRACe Utilization above 0.50 and limits unsupported statements to less than 3%. All components run in containers, enforce end-to-end TLS 1.3 and AES-256. These results demonstrate that the DEREK module delivers accurate, traceable, and production-ready document QA with minimal operational overhead. The module is designed to meet enterprise demands for secure, auditable, and context-faithful retrieval, providing a reliable baseline for high-stakes domains such as legal and finance.
>
---
#### [new 028] Small Edits, Big Consequences: Telling Good from Bad Robustness in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在研究大语言模型对输入扰动的敏感性。作者构建了三种微小提示扰动，测试模型在代码生成中的鲁棒性与敏感性，发现模型对词义变化不敏感，建议改进评估与训练方法以更好区分无害噪声与语义变化。**

- **链接: [http://arxiv.org/pdf/2507.15868v1](http://arxiv.org/pdf/2507.15868v1)**

> **作者:** Altynbek Ismailov; Salia Asanova
>
> **摘要:** Large language models (LLMs) now write code in settings where misreading a single word can break safety or cost money, yet we still expect them to overlook stray typos. To probe where useful robustness ends and harmful insensitivity begins, we compile 50 LeetCode problems and craft three minimal prompt perturbations that should vary in importance: (i) progressive underspecification deleting 10 % of words per step; (ii) lexical flip swapping a pivotal quantifier ("max" to "min"); and (iii) jargon inflation replacing a common noun with an obscure technical synonym. Six frontier models, including three "reasoning-tuned" versions, solve each mutated prompt, and their Python outputs are checked against the original test suites to reveal whether they reused the baseline solution or adapted. Among 11 853 generations we observe a sharp double asymmetry. Models remain correct in 85 % of cases even after 90 % of the prompt is missing, showing over-robustness to underspecification, yet only 54 % react to a single quantifier flip that reverses the task, with reasoning-tuned variants even less sensitive than their bases. Jargon edits lie in between, passing through 56 %. Current LLMs thus blur the line between harmless noise and meaning - changing edits, often treating both as ignorable. Masking salient anchors such as function names can force re - evaluation. We advocate evaluation and training protocols that reward differential sensitivity: stay steady under benign noise but adapt - or refuse - when semantics truly change.
>
---
#### [new 029] Towards Compute-Optimal Many-Shot In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在上下文学习中的高效推理问题。论文提出了两种选择演示样本的策略，结合相似性与聚类方法，在保证性能的同时显著降低推理成本并支持计算缓存。**

- **链接: [http://arxiv.org/pdf/2507.16217v1](http://arxiv.org/pdf/2507.16217v1)**

> **作者:** Shahriar Golchin; Yanfei Chen; Rujun Han; Manan Gandhi; Tianli Yu; Swaroop Mishra; Mihai Surdeanu; Rishabh Agarwal; Chen-Yu Lee; Tomas Pfister
>
> **备注:** Final version; accepted at COLM 2025
>
> **摘要:** Long-context large language models (LLMs) are able to process inputs containing up to several million tokens. In the scope of in-context learning (ICL), this translates into using hundreds/thousands of demonstrations in the input prompt, enabling many-shot ICL. In practice, a fixed set of demonstrations is often selected at random in many-shot settings due to (1) high inference costs, (2) the benefits of caching and reusing computations, and (3) the similar performance offered by this strategy compared to others when scaled. In this work, we propose two straightforward strategies for demonstration selection in many-shot ICL that improve performance with minimal computational overhead. Our first method combines a small number of demonstrations, selected based on their similarity to each test sample, with a disproportionately larger set of random demonstrations that are cached. The second strategy improves the first by replacing random demonstrations with those selected using centroids derived from test sample representations via k-means clustering. Our experiments with Gemini Pro and Flash across several datasets indicate that our strategies consistently outperform random selection and surpass or match the most performant selection approach while supporting caching and reducing inference cost by up to an order of magnitude. We also show that adjusting the proportion of demonstrations selected based on different criteria can balance performance and inference cost in many-shot ICL.
>
---
#### [new 030] Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny
- **分类: cs.CL**

- **简介: 该论文属于形式化软件验证任务，旨在解决基于自然语言的强化学习大模型验证不可靠、难以扩展的问题。论文提出Re:Form方法，利用形式语言Dafny，通过自动化数据构建和强化学习设计，减少对人工先验的依赖，提升代码生成与验证的可靠性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.16331v1](http://arxiv.org/pdf/2507.16331v1)**

> **作者:** Chuanhao Yan; Fengdi Che; Xuhan Huang; Xu Xu; Xin Li; Yizhi Li; Xingwei Qu; Jingzhe Shi; Zhuangzhuang He; Chenghua Lin; Yaodong Yang; Binhang Yuan; Hang Zhao; Yu Qiao; Bowen Zhou; Jie Fu
>
> **摘要:** Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
>
---
#### [new 031] Do Large Language Models Have a Planning Theory of Mind? Evidence from MindGames: a Multi-Step Persuasion Task
- **分类: cs.CL**

- **简介: 该论文属于多步骤说服任务，旨在检验大语言模型是否具备规划心智理论（PToM）能力。研究通过MindGames任务要求模型推断对话者的信念和欲望以进行说服，并发现人类表现优于o1-preview模型。论文对比了人类与LLM在需要心智推理与仅需规划的任务中的表现差异，揭示了当前LLM在社会推理方面与人类仍存在差距。**

- **链接: [http://arxiv.org/pdf/2507.16196v1](http://arxiv.org/pdf/2507.16196v1)**

> **作者:** Jared Moore; Ned Cooper; Rasmus Overmark; Beba Cibralic; Nick Haber; Cameron R. Jones
>
> **备注:** To appear in COLM, 2025
>
> **摘要:** Recent evidence suggests Large Language Models (LLMs) display Theory of Mind (ToM) abilities. Most ToM experiments place participants in a spectatorial role, wherein they predict and interpret other agents' behavior. However, human ToM also contributes to dynamically planning action and strategically intervening on others' mental states. We present MindGames: a novel `planning theory of mind' (PToM) task which requires agents to infer an interlocutor's beliefs and desires to persuade them to alter their behavior. Unlike previous evaluations, we explicitly evaluate use cases of ToM. We find that humans significantly outperform o1-preview (an LLM) at our PToM task (11% higher; $p=0.006$). We hypothesize this is because humans have an implicit causal model of other agents (e.g., they know, as our task requires, to ask about people's preferences). In contrast, o1-preview outperforms humans in a baseline condition which requires a similar amount of planning but minimal mental state inferences (e.g., o1-preview is better than humans at planning when already given someone's preferences). These results suggest a significant gap between human-like social reasoning and LLM abilities.
>
---
#### [new 032] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型中的性别偏见评估问题。论文构建了五个德语性别偏见评估数据集，基于已有性别偏见概念，通过多种方法进行性别偏见研究，揭示了德语中男性职业术语的歧义及中性名词对性别认知的影响，强调需开发适用于不同语言的偏见评估框架。**

- **链接: [http://arxiv.org/pdf/2507.16557v1](http://arxiv.org/pdf/2507.16557v1)**

> **作者:** Kristin Gnadt; David Thulke; Simone Kopeinik; Ralf Schlüter
>
> **备注:** Accepted at the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP) at ACL 2025
>
> **摘要:** In recent years, various methods have been proposed to evaluate gender bias in large language models (LLMs). A key challenge lies in the transferability of bias measurement methods initially developed for the English language when applied to other languages. This work aims to contribute to this research strand by presenting five German datasets for gender bias evaluation in LLMs. The datasets are grounded in well-established concepts of gender bias and are accessible through multiple methodologies. Our findings, reported for eight multilingual LLM models, reveal unique challenges associated with gender bias in German, including the ambiguous interpretation of male occupational terms and the influence of seemingly neutral nouns on gender perception. This work contributes to the understanding of gender bias in LLMs across languages and underscores the necessity for tailored evaluation frameworks.
>
---
#### [new 033] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文提出PICACO方法，属于上下文对齐任务，旨在解决大语言模型在单一提示中难以平衡多元价值观导致的对齐不足问题。通过优化元指令，提升模型对多种价值的理解与响应一致性，实现更优的价值对齐。**

- **链接: [http://arxiv.org/pdf/2507.16679v1](http://arxiv.org/pdf/2507.16679v1)**

> **作者:** Han Jiang; Dongyao Zhu; Zhihua Wei; Xiaoyuan Yi; Ziang Xiao; Xing Xie
>
> **摘要:** In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
>
---
#### [new 034] Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于金融领域大模型任务，旨在解决现有模型在金融场景中推理能力不足、可信度低和适应效率差的问题。作者基于Qwen3开发了Agentar-Fin-R1系列模型，结合高质量金融任务分类与可信度保障框架，提升训练效率与实际部署能力，验证了其在金融与通用推理任务上的卓越性能。**

- **链接: [http://arxiv.org/pdf/2507.16802v1](http://arxiv.org/pdf/2507.16802v1)**

> **作者:** Yanjun Zheng; Xiyang Du; Longfei Liao; Xiaoke Zhao; Zhaowen Zhou; Bo Zhang; Jiawei Liu; Xiang Qi; Zhe Li; Zhiqiang Zhang; Wang Wei; Peng Zhang
>
> **摘要:** Large Language Models (LLMs) demonstrate tremendous potential in the financial domain, yet existing models often fall short in scenarios demanding robust reasoning capabilities, stringent trustworthiness requirements, and efficient adaptation to task-specific needs. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task taxonomy with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage learning processes, and detailed attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including FinEva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications.
>
---
#### [new 035] Combining Language and Topic Models for Hierarchical Text Classification
- **分类: cs.CL; cs.LG; I.2.7; I.2.6**

- **简介: 该论文属于层次文本分类（HTC）任务，旨在通过结合预训练语言模型（PLM）和主题模型提取特征，提升分类性能。作者使用卷积层和标签注意力机制融合两类特征，但在三个基准数据集上实验发现，加入主题模型特征反而降低了分类效果，表明其在HTC中未必有益。**

- **链接: [http://arxiv.org/pdf/2507.16490v1](http://arxiv.org/pdf/2507.16490v1)**

> **作者:** Jaco du Toit; Marcel Dunaiski
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** Hierarchical text classification (HTC) is a natural language processing task which has the objective of categorising text documents into a set of classes from a predefined structured class hierarchy. Recent HTC approaches use various techniques to incorporate the hierarchical class structure information with the natural language understanding capabilities of pre-trained language models (PLMs) to improve classification performance. Furthermore, using topic models along with PLMs to extract features from text documents has been shown to be an effective approach for multi-label text classification tasks. The rationale behind the combination of these feature extractor models is that the PLM captures the finer-grained contextual and semantic information while the topic model obtains high-level representations which consider the corpus of documents as a whole. In this paper, we use a HTC approach which uses a PLM and a topic model to extract features from text documents which are used to train a classification model. Our objective is to determine whether the combination of the features extracted from the two models is beneficial to HTC performance in general. In our approach, the extracted features are passed through separate convolutional layers whose outputs are combined and passed to a label-wise attention mechanisms which obtains label-specific document representations by weighing the most important features for each class separately. We perform comprehensive experiments on three HTC benchmark datasets and show that using the features extracted from the topic model generally decreases classification performance compared to only using the features obtained by the PLM. In contrast to previous work, this shows that the incorporation of features extracted from topic models for text classification tasks should not be assumed beneficial.
>
---
#### [new 036] Learning without training: The implicit dynamics of in-context learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）在推理时通过上下文学习新模式的能力。任务是理解其背后机制，提出Transformer块通过自注意力层和MLP层隐式修改权重，实现上下文学习。工作包括理论分析与实验验证，解释LLM可在无权重更新下利用提示中的示例学习。**

- **链接: [http://arxiv.org/pdf/2507.16003v1](http://arxiv.org/pdf/2507.16003v1)**

> **作者:** Benoit Dherin; Michael Munn; Hanna Mazzawi; Michael Wunder; Javier Gonzalvo
>
> **摘要:** One of the most striking features of Large Language Models (LLM) is their ability to learn in context. Namely at inference time an LLM is able to learn new patterns without any additional weight update when these patterns are presented in the form of examples in the prompt, even if these patterns were not seen during training. The mechanisms through which this can happen are still largely unknown. In this work, we show that the stacking of a self-attention layer with an MLP, allows the transformer block to implicitly modify the weights of the MLP layer according to the context. We argue through theory and experimentation that this simple mechanism may be the reason why LLMs can learn in context and not only during training. Specifically, we show under mild simplifying assumptions how a transformer block implicitly transforms a context into a low-rank weight-update of the MLP layer.
>
---
#### [new 037] Self-Contradiction as Self-Improvement: Mitigating the Generation-Understanding Gap in MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态大语言模型（MLLM）任务，旨在解决生成与理解能力不统一导致的自相矛盾问题。作者提出“Nonunified score”量化该问题，并探索通过后训练方法（如SFT、DPO）利用模型自身理解能力来改进生成能力，最终实现生成与理解的协同提升，并提出渐进式课程策略优化训练过程。**

- **链接: [http://arxiv.org/pdf/2507.16663v1](http://arxiv.org/pdf/2507.16663v1)**

> **作者:** Yujin Han; Hao Chen; Andi Han; Zhiheng Wang; Xinyu Lin; Yingya Zhang; Shiwei Zhang; Difan Zou
>
> **备注:** 19 pages, 9 figures, 3 tables
>
> **摘要:** Despite efforts to unify multimodal generation and understanding tasks in a single model, we show these MLLMs exhibit self-contradiction where generation produces images deemed misaligned with input prompts based on the model's own understanding. We define a Nonunified score that quantifies such self-contradiction. Our empirical results reveal that the self-contradiction mainly arises from weak generation that fails to align with prompts, rather than misunderstanding. This capability asymmetry indicates the potential of leveraging self-contradiction for self-improvement, where the stronger model understanding guides the weaker generation to mitigate the generation-understanding gap. Applying standard post-training methods (e.g., SFT, DPO) with such internal supervision successfully improves both generation and unification. We discover a co-improvement effect on both generation and understanding when only fine-tuning the generation branch, a phenomenon known in pre-training but underexplored in post-training. Our analysis shows improvements stem from better detection of false positives that are previously incorrectly identified as prompt-aligned. Theoretically, we show the aligned training dynamics between generation and understanding allow reduced prompt-misaligned generations to also improve mismatch detection in the understanding branch. Additionally, the framework reveals a potential risk of co-degradation under poor supervision-an overlooked phenomenon that is empirically validated in our experiments. Notably, we find intrinsic metrics like Nonunified score cannot distinguish co-degradation from co-improvement, which highlights the necessity of data quality check. Finally, we propose a curriculum-based strategy based on our findings that gradually introduces harder samples as the model improves, leading to better unification and improved MLLM generation and understanding.
>
---
#### [new 038] LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs
- **分类: cs.CL**

- **简介: 该论文提出了LingBench++，一个基于国际语言学奥林匹克竞赛任务的多步推理和跨文化推理基准及推理框架。论文旨在解决现有基准仅关注最终答案准确性的问题，通过提供结构化推理轨迹、逐步评估协议和丰富的类型学元数据，推动语言模型在语言学、文化和认知层面的深入理解与推理能力。**

- **链接: [http://arxiv.org/pdf/2507.16809v1](http://arxiv.org/pdf/2507.16809v1)**

> **作者:** Da-Chen Lian; Ri-Sheng Huang; Pin-Er Chen; Chunki Lim; You-Kuan Lin; Guan-Yu Tseng; Zi-Cheng Yang; Shu-Kai Hsieh
>
> **备注:** 41 pages, 17 figures, 10 tables
>
> **摘要:** We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
>
---
#### [new 039] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文介绍了Step-Audio 2，一种用于音频理解和语音对话的多模态大模型。它通过集成音频编码器和强化学习，提升语音识别和对话响应能力，并结合检索增强生成减少幻觉。模型训练于海量语音数据，适用于多种对话场景。**

- **链接: [http://arxiv.org/pdf/2507.16632v1](http://arxiv.org/pdf/2507.16632v1)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** This paper presents Step-Audio~2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [new 040] P-CoT: A Pedagogically-motivated Participatory Chain-of-Thought Prompting for Phonological Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的语音推理能力。通过设计基于教学理论的参与式思维链提示（P-CoT），在语音生成、音位转换等任务上取得显著提升，最高达52%。论文评估了12个模型，验证了P-CoT的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16656v1](http://arxiv.org/pdf/2507.16656v1)**

> **作者:** Dongjun Jang; Youngchae Ahn; Hyopil Shin
>
> **摘要:** This study explores the potential of phonological reasoning within text-based large language models (LLMs). Utilizing the PhonologyBench benchmark, we assess tasks like rhyme word generation, g2p conversion, and syllable counting. Our evaluations across 12 LLMs reveal that while few-shot learning offers inconsistent gains, the introduction of a novel Pedagogically-motivated Participatory Chain-of-Thought (P-CoT) prompt, which is anchored in educational theories like scaffolding and discovery learning, consistently enhances performance. This method leverages structured guidance to activate latent phonological abilities, achieving up to 52% improvement and even surpassing human baselines in certain tasks. Future work could aim to optimize P-CoT prompts for specific models or explore their application across different linguistic domains.
>
---
#### [new 041] Towards Enforcing Company Policy Adherence in Agentic Workflows
- **分类: cs.CL**

- **简介: 该论文旨在解决大型语言模型代理在企业流程自动化中难以遵循复杂公司政策的问题。通过构建一个分两阶段的框架，包括离线阶段将政策文档编译为可验证的工具使用守卫代码，以及运行时阶段通过这些守卫确保合规性，从而实现对企业政策的强制遵循。论文属于企业流程自动化与合规性控制任务。**

- **链接: [http://arxiv.org/pdf/2507.16459v1](http://arxiv.org/pdf/2507.16459v1)**

> **作者:** Naama Zwerdling; David Boaz; Ella Rabinovich; Guy Uziel; David Amid; Ateret Anaby-Tavor
>
> **备注:** 11 pages
>
> **摘要:** Large Language Model (LLM) agents hold promise for a flexible and scalable alternative to traditional business process automation, but struggle to reliably follow complex company policies. In this study we introduce a deterministic, transparent, and modular framework for enforcing business policy adherence in agentic workflows. Our method operates in two phases: (1) an offline buildtime stage that compiles policy documents into verifiable guard code associated with tool use, and (2) a runtime integration where these guards ensure compliance before each agent action. We demonstrate our approach on the challenging $\tau$-bench Airlines domain, showing encouraging preliminary results in policy enforcement, and further outline key challenges for real-world deployments.
>
---
#### [new 042] Adversarial Demonstration Learning for Low-resource NER Using Dual Similarity
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于低资源命名实体识别（NER）任务，旨在解决示范学习中示例选择和模型训练的问题。作者提出结合语义与特征相似性的双相似性方法选择示例，并引入对抗示范训练，提升模型对示范的参考能力，从而提高低资源场景下的NER性能。**

- **链接: [http://arxiv.org/pdf/2507.15864v1](http://arxiv.org/pdf/2507.15864v1)**

> **作者:** Guowen Yuan; Tien-Hsuan Wu; Lianghao Xia; Ben Kao
>
> **摘要:** We study the problem of named entity recognition (NER) based on demonstration learning in low-resource scenarios. We identify two issues in demonstration construction and model training. Firstly, existing methods for selecting demonstration examples primarily rely on semantic similarity; We show that feature similarity can provide significant performance improvement. Secondly, we show that the NER tagger's ability to reference demonstration examples is generally inadequate. We propose a demonstration and training approach that effectively addresses these issues. For the first issue, we propose to select examples by dual similarity, which comprises both semantic similarity and feature similarity. For the second issue, we propose to train an NER model with adversarial demonstration such that the model is forced to refer to the demonstrations when performing the tagging task. We conduct comprehensive experiments in low-resource NER tasks, and the results demonstrate that our method outperforms a range of methods.
>
---
#### [new 043] Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent
- **分类: cs.CL**

- **简介: 该论文提出Test-Time-Matching（TTM），旨在解决基于大语言模型的角色扮演中角色沉浸感不足和风格控制问题。通过解耦角色特征为性格、记忆和语言风格，实现无需训练的三阶段生成流程。评估显示其在生成一致性角色对话方面效果优异。**

- **链接: [http://arxiv.org/pdf/2507.16799v1](http://arxiv.org/pdf/2507.16799v1)**

> **作者:** Xiaoyu Zhan; Xinyu Fu; Hao Sun; Yuanqi Li; Jie Guo; Yanwen Guo
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
>
---
#### [new 044] The Ever-Evolving Science Exam
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学理解评估任务，旨在解决现有科学基准的数据泄漏风险和评估效率低的问题。作者构建了包含10万题的非公开题库EESE-Pool，并从中定期更新500题构成EESE，实现高效、防泄漏的模型评估。实验表明该方法能有效区分模型在科学领域的能力差异。**

- **链接: [http://arxiv.org/pdf/2507.16514v1](http://arxiv.org/pdf/2507.16514v1)**

> **作者:** Junying Wang; Zicheng Zhang; Yijin Guo; Farong Wen; Ye Shen; Yingji Liang; Yalun Wu; Wenzhe Li; Chunyi Li; Zijian Chen; Qi Jia; Guangtao Zhai
>
> **备注:** 20 pages
>
> **摘要:** As foundation models grow rapidly in capability and deployment, evaluating their scientific understanding becomes increasingly critical. Existing science benchmarks have made progress towards broad **Range**, wide **Reach**, and high **Rigor**, yet they often face two major challenges: **data leakage risks** that compromise benchmarking validity, and **evaluation inefficiency** due to large-scale testing. To address these issues, we introduce the **Ever-Evolving Science Exam (EESE)**, a dynamic benchmark designed to reliably assess scientific capabilities in foundation models. Our approach consists of two components: 1) a non-public **EESE-Pool** with over 100K expertly constructed science instances (question-answer pairs) across 5 disciplines and 500+ subfields, built through a multi-stage pipeline ensuring **Range**, **Reach**, and **Rigor**, 2) a periodically updated 500-instance subset **EESE**, sampled and validated to enable leakage-resilient, low-overhead evaluations. Experiments on 32 open- and closed-source models demonstrate that EESE effectively differentiates the strengths and weaknesses of models in scientific fields and cognitive dimensions. Overall, EESE provides a robust, scalable, and forward-compatible solution for science benchmark design, offering a realistic measure of how well foundation models handle science questions. The project page is at: https://github.com/aiben-ch/EESE.
>
---
#### [new 045] RAVine: Reality-Aligned Evaluation for Agentic Search
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决现有评估框架与智能搜索系统目标不匹配的问题。提出了RAVine，一种更贴近真实场景的评估框架，优化多点查询与长答案的评估，提升细粒度评价准确性，并关注搜索工具交互与效率。**

- **链接: [http://arxiv.org/pdf/2507.16725v1](http://arxiv.org/pdf/2507.16725v1)**

> **作者:** Yilong Xu; Xiang Long; Zhi Zheng; Jinhua Gao
>
> **摘要:** Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine -- a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model's interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at https://github.com/SwordFaith/RAVine.
>
---
#### [new 046] SpeLLM: Character-Level Multi-Head Decoding
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLM）因输出投影层随词汇量线性增长而导致的扩展瓶颈问题。作者提出SpeLLM方法，通过多头解码实现字符级预测，解耦输入与输出词汇，降低运行成本并提升对低资源语言和领域的支持能力。**

- **链接: [http://arxiv.org/pdf/2507.16323v1](http://arxiv.org/pdf/2507.16323v1)**

> **作者:** Amit Ben-Artzy; Roy Schwartz
>
> **摘要:** Scaling LLM vocabulary is often used to reduce input sequence length and alleviate attention's quadratic cost. Yet, current LLM architectures impose a critical bottleneck to this procedure: the output projection layer scales linearly with vocabulary size, rendering substantial expansion impractical. We propose SpeLLM, a method that decouples input and output vocabularies by predicting character-level strings through multiple output heads. In SpeLLM, each of the $k$ linear heads predicts a single character simultaneously, enabling the model to represent a much larger output space using smaller, independent linear heads. We present a self-distillation approach for converting a standard LLM to a SpeLLM. Our experiments with four pre-trained LLMs show their SpeLLM variants achieve competitive performance on downstream tasks while reducing runtime by 5.1% on average across models. Our approach provides a potential avenue for reducing LLM costs, while increasing support for underrepresented languages and domains.
>
---
#### [new 047] AutoMeet: a proof-of-concept study of genAI to automate meetings in automotive engineering
- **分类: cs.CL**

- **简介: 该论文属于知识管理任务，旨在解决汽车工程领域会议记录与信息检索效率低的问题。作者开发了一个基于生成式人工智能（genAI）的端到端会议自动化处理系统 AutoMeet，实现会议录音、生成纪要及通过聊天机器人进行搜索的功能，并通过实际应用收集用户反馈，评估其技术可行性与组织伦理影响。**

- **链接: [http://arxiv.org/pdf/2507.16054v1](http://arxiv.org/pdf/2507.16054v1)**

> **作者:** Simon Baeuerle; Max Radyschevski; Ulrike Pado
>
> **摘要:** In large organisations, knowledge is mainly shared in meetings, which takes up significant amounts of work time. Additionally, frequent in-person meetings produce inconsistent documentation -- official minutes, personal notes, presentations may or may not exist. Shared information therefore becomes hard to retrieve outside of the meeting, necessitating lengthy updates and high-frequency meeting schedules. Generative Artificial Intelligence (genAI) models like Large Language Models (LLMs) exhibit an impressive performance on spoken and written language processing. This motivates a practical usage of genAI for knowledge management in engineering departments: using genAI for transcribing meetings and integrating heterogeneous additional information sources into an easily usable format for ad-hoc searches. We implement an end-to-end pipeline to automate the entire meeting documentation workflow in a proof-of-concept state: meetings are recorded and minutes are created by genAI. These are further made easily searchable through a chatbot interface. The core of our work is to test this genAI-based software tooling in a real-world engineering department and collect extensive survey data on both ethical and technical aspects. Direct feedback from this real-world setup points out both opportunities and risks: a) users agree that the effort for meetings could be significantly reduced with the help of genAI models, b) technical aspects are largely solved already, c) organizational aspects are crucial for a successful ethical usage of such a system.
>
---
#### [new 048] Experience is the Best Teacher: Grounding VLMs for Robotics through Self-Generated Memory
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人与视觉语言模型（VLM）结合的任务，旨在解决VLM在真实机器人上泛化能力差的问题。论文提出了ExpTeach框架，通过自生成经验记忆实现VLM的现实世界适应，结合反思机制与长期记忆检索，显著提升了机器人在多任务中的成功率。**

- **链接: [http://arxiv.org/pdf/2507.16713v1](http://arxiv.org/pdf/2507.16713v1)**

> **作者:** Guowei Lan; Kaixian Qu; René Zurbrügg; Changan Chen; Christopher E. Mower; Haitham Bou-Ammar; Marco Hutter
>
> **摘要:** Vision-language models (VLMs) have been widely adopted in robotics to enable autonomous planning. However, grounding VLMs, originally trained on internet data, to diverse real-world robots remains a challenge. This paper presents ExpTeach, a framework that grounds VLMs to physical robots by building a self-generated memory of real-world experiences. In ExpTeach, the VLM autonomously plans actions, verifies outcomes, reflects on failures, and adapts robot behaviors in a closed loop. The self-generated experiences during this process are then summarized into a long-term memory, enabling retrieval of learned knowledge to guide future tasks via retrieval-augmented generation (RAG). Additionally, ExpTeach enhances the spatial understanding of VLMs with an on-demand image annotation module. In experiments, we show that reflection improves success rates from 36% to 84% on four challenging robotic tasks and observe the emergence of intelligent object interactions, including creative tool use. Across extensive tests on 12 real-world scenarios (including eight unseen ones), we find that grounding with long-term memory boosts single-trial success rates from 22% to 80%, demonstrating the effectiveness and generalizability of ExpTeach.
>
---
#### [new 049] Scaling Linear Attention with Sparse State Expansion
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer在长上下文场景中计算效率低和内存消耗大的问题。论文提出稀疏状态扩展（SSE）方法，通过稀疏更新和状态扩展机制，提升线性注意力的上下文压缩能力和模型性能，在语言建模、上下文检索和数学推理任务中取得良好效果。**

- **链接: [http://arxiv.org/pdf/2507.16577v1](http://arxiv.org/pdf/2507.16577v1)**

> **作者:** Yuqi Pan; Yongqi An; Zheng Li; Yuhong Chou; Ruijie Zhu; Xiaohui Wang; Mingxuan Wang; Jinqiao Wang; Guoqi Li
>
> **摘要:** The Transformer architecture, despite its widespread success, struggles with long-context scenarios due to quadratic computation and linear memory growth. While various linear attention variants mitigate these efficiency constraints by compressing context into fixed-size states, they often degrade performance in tasks such as in-context retrieval and reasoning. To address this limitation and achieve more effective context compression, we propose two key innovations. First, we introduce a row-sparse update formulation for linear attention by conceptualizing state updating as information classification. This enables sparse state updates via softmax-based top-$k$ hard classification, thereby extending receptive fields and reducing inter-class interference. Second, we present Sparse State Expansion (SSE) within the sparse framework, which expands the contextual state into multiple partitions, effectively decoupling parameter size from state capacity while maintaining the sparse classification paradigm. Our design, supported by efficient parallelized implementations, yields effective classification and discriminative state representations. We extensively validate SSE in both pure linear and hybrid (SSE-H) architectures across language modeling, in-context retrieval, and mathematical reasoning benchmarks. SSE demonstrates strong retrieval performance and scales favorably with state size. Moreover, after reinforcement learning (RL) training, our 2B SSE-H model achieves state-of-the-art mathematical reasoning performance among small reasoning models, scoring 64.7 on AIME24 and 51.3 on AIME25, significantly outperforming similarly sized open-source Transformers. These results highlight SSE as a promising and efficient architecture for long-context modeling.
>
---
#### [new 050] MMS Player: an open source software for parametric data-driven animation of Sign Language avatars
- **分类: cs.GR; cs.CL**

- **简介: 该论文属于计算机视觉与人机交互任务，旨在解决手语动画生成问题。作者开发了开源软件MMS-Player，能从新型手语表示格式MMS生成参数化、数据驱动的手语虚拟人动画，并提供多种调用方式及输出格式。**

- **链接: [http://arxiv.org/pdf/2507.16463v1](http://arxiv.org/pdf/2507.16463v1)**

> **作者:** Fabrizio Nunnari; Shailesh Mishra; Patrick Gebhard
>
> **摘要:** This paper describes the MMS-Player, an open source software able to synthesise sign language animations from a novel sign language representation format called MMS (MultiModal Signstream). The MMS enhances gloss-based representations by adding information on parallel execution of signs, timing, and inflections. The implementation consists of Python scripts for the popular Blender 3D authoring tool and can be invoked via command line or HTTP API. Animations can be rendered as videos or exported in other popular 3D animation exchange formats. The software is freely available under GPL-3.0 license at https://github.com/DFKI-SignLanguage/MMS-Player.
>
---
#### [new 051] Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于风险分析任务，旨在识别和评估前沿人工智能模型带来的重大风险。依据“AI-$45^\circ$法”与E-T-C框架，研究对七大风险领域进行评估，判断模型处于绿、黄、红风险区域。结果显示，当前模型尚未突破红线，但部分领域如说服与操控风险较高。论文呼吁加强风险管理与集体应对。**

- **链接: [http://arxiv.org/pdf/2507.16534v1](http://arxiv.org/pdf/2507.16534v1)**

> **作者:** Shanghai AI Lab; :; Xiaoyang Chen; Yunhao Chen; Zeren Chen; Zhiyun Chen; Hanyun Cui; Yawen Duan; Jiaxuan Guo; Qi Guo; Xuhao Hu; Hong Huang; Lige Huang; Chunxiao Li; Juncheng Li; Qihao Lin; Dongrui Liu; Xinmin Liu; Zicheng Liu; Chaochao Lu; Xiaoya Lu; Jingjing Qu; Qibing Ren; Jing Shao; Jingwei Shi; Jingwei Sun; Peng Wang; Weibing Wang; Jia Xu; Lewen Yan; Xiao Yu; Yi Yu; Boxuan Zhang; Jie Zhang; Weichen Zhang; Zhijie Zheng; Tianyi Zhou; Bowen Zhou
>
> **备注:** 97 pages, 37 figures
>
> **摘要:** To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges.
>
---
#### [new 052] Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态文档理解任务，旨在解决现有模型在长文档处理上的不足。作者构建了Document Haystack基准，包含长达200页的复杂文档，并插入文本或图文“针”测试模型检索能力，共包含400种文档变体和8250个问题，提供自动化评估框架，用于评估视觉语言模型的长文档理解性能。**

- **链接: [http://arxiv.org/pdf/2507.15882v1](http://arxiv.org/pdf/2507.15882v1)**

> **作者:** Goeric Huybrechts; Srikanth Ronanki; Sai Muralidhar Jayanthi; Jack Fitzgerald; Srinivasan Veeravanallur
>
> **摘要:** The proliferation of multimodal Large Language Models has significantly advanced the ability to analyze and understand complex data inputs from different modalities. However, the processing of long documents remains under-explored, largely due to a lack of suitable benchmarks. To address this, we introduce Document Haystack, a comprehensive benchmark designed to evaluate the performance of Vision Language Models (VLMs) on long, visually complex documents. Document Haystack features documents ranging from 5 to 200 pages and strategically inserts pure text or multimodal text+image "needles" at various depths within the documents to challenge VLMs' retrieval capabilities. Comprising 400 document variants and a total of 8,250 questions, it is supported by an objective, automated evaluation framework. We detail the construction and characteristics of the Document Haystack dataset, present results from prominent VLMs and discuss potential research avenues in this area.
>
---
#### [new 053] Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型在推理任务中准确性和置信度校准不一致的问题。论文提出RLCR方法，结合强化学习与校准奖励，同时优化预测准确性和置信度质量。实验表明RLCR在多个数据集上提升了校准效果，且不损失准确性。**

- **链接: [http://arxiv.org/pdf/2507.16806v1](http://arxiv.org/pdf/2507.16806v1)**

> **作者:** Mehul Damani; Isha Puri; Stewart Slocum; Idan Shenfeld; Leshem Choshen; Yoon Kim; Jacob Andreas
>
> **摘要:** When language models (LMs) are trained via reinforcement learning (RL) to generate natural language "reasoning chains", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or "hallucinate") in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score -- a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations -- outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models.
>
---
#### [new 054] Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决视觉链式推理（Visual CoT）中数据质量差和模型性能低的问题。作者构建了大规模数据集Zebra-CoT，包含18万+图文交错推理样本，并验证其在提升模型视觉推理能力上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16746v1](http://arxiv.org/pdf/2507.16746v1)**

> **作者:** Ang Li; Charles Wang; Kaiyu Yue; Zikui Cai; Ollie Liu; Deqing Fu; Peng Guo; Wang Bill Zhu; Vatsal Sharan; Robin Jia; Willie Neiswanger; Furong Huang; Tom Goldstein; Micah Goldblum
>
> **备注:** dataset link: https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT
>
> **摘要:** Humans often use visual aids, for example diagrams or sketches, when solving complex problems. Training multimodal models to do the same, known as Visual Chain of Thought (Visual CoT), is challenging due to: (1) poor off-the-shelf visual CoT performance, which hinders reinforcement learning, and (2) the lack of high-quality visual CoT training data. We introduce $\textbf{Zebra-CoT}$, a diverse large-scale dataset with 182,384 samples, containing logically coherent interleaved text-image reasoning traces. We focus on four categories of tasks where sketching or visual reasoning is especially natural, spanning scientific questions such as geometry, physics, and algorithms; 2D visual reasoning tasks like visual search and jigsaw puzzles; 3D reasoning tasks including 3D multi-hop inference, embodied and robot planning; visual logic problems and strategic games like chess. Fine-tuning the Anole-7B model on the Zebra-CoT training corpus results in an improvement of +12% in our test-set accuracy and yields up to +13% performance gain on standard VLM benchmark evaluations. Fine-tuning Bagel-7B yields a model that generates high-quality interleaved visual reasoning chains, underscoring Zebra-CoT's effectiveness for developing multimodal reasoning abilities. We open-source our dataset and models to support development and evaluation of visual CoT.
>
---
#### [new 055] SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗AI任务，旨在解决COPD诊断模型缺乏可解释性、无法理解肺功能曲线的问题。作者提出SpiroLLM，通过融合呼吸曲线形态特征与数值指标，使大语言模型能生成可解释的诊断报告，提升了诊断准确性和临床适用性。**

- **链接: [http://arxiv.org/pdf/2507.16145v1](http://arxiv.org/pdf/2507.16145v1)**

> **作者:** Shuhao Mei; Yongchao Long; Shan Cao; Xiaobo Han; Shijia Geng; Jinbo Sun; Yuxi Zhou; Shenda Hong
>
> **摘要:** Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8980 (95% CI: 0.8820-0.9132). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
>
---
#### [new 056] Characterizing Online Activities Contributing to Suicide Mortality among Youth
- **分类: cs.CY; cs.CL**

- **简介: 该论文旨在分析青少年自杀死亡相关的在线活动，属于公共健康与计算交叉任务。通过混合方法，从近3万份文本中提取12类相关在线活动主题，并构建零样本学习框架进行大规模建模，探究其与人口特征及时间变化的关系，以支持早期干预。**

- **链接: [http://arxiv.org/pdf/2507.16185v1](http://arxiv.org/pdf/2507.16185v1)**

> **作者:** Aparna Ananthasubramaniam; Elyse J. Thulin; Viktoryia Kalesnikava; Silas Falde; Jonathan Kertawidjaja; Lily Johns; Alejandro Rodríguez-Putnam; Emma Spring; Kara Zivin; Briana Mezuk
>
> **备注:** Accepted at the AAAI International Conference on Web and Social Media (ICWSM) 2026
>
> **摘要:** The recent rise in youth suicide highlights the urgent need to understand how online experiences contribute to this public health issue. Our mixed-methods approach responds to this challenge by developing a set of themes focused on risk factors for suicide mortality in online spaces among youth ages 10-24, and a framework to model these themes at scale. Using 29,124 open text summaries of death investigations between 2013-2022, we conducted a thematic analysis to identify 12 types of online activities that were considered by investigators or next of kin to be relevant in contextualizing a given suicide death. We then develop a zero-shot learning framework to model these 12 themes at scale, and analyze variation in these themes by decedent characteristics and over time. Our work uncovers several online activities related to harm to self, harm to others, interpersonal interactions, activity levels online, and life events, which correspond to different phases of suicide risk from two prominent suicide theories. We find an association between these themes and decedent characteristics like age, means of death, and interpersonal problems, and many themes became more prevalent during the 2020 COVID-19 lockdowns. While digital spaces have taken some steps to address expressions of suicidality online, our work illustrates the opportunities for developing interventions related to less explicit indicators of suicide risk by combining suicide theories with computational research.
>
---
#### [new 057] WhatsApp Tiplines and Multilingual Claims in the 2021 Indian Assembly Elections
- **分类: cs.SI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究了2021年印度选举期间WhatsApp举报热线中多语言虚假信息的处理情况。任务是分析用户提交的多语言举报内容，识别主题分类、语言差异、事实核查机构间的用户重叠及响应效率。论文通过内容分析与用户行为研究，提出优化举报热线在选举期间应用的建议。**

- **链接: [http://arxiv.org/pdf/2507.16298v1](http://arxiv.org/pdf/2507.16298v1)**

> **作者:** Gautam Kishore Shahi; Scot A. Hale
>
> **摘要:** WhatsApp tiplines, first launched in 2019 to combat misinformation, enable users to interact with fact-checkers to verify misleading content. This study analyzes 580 unique claims (tips) from 451 users, covering both high-resource languages (English, Hindi) and a low-resource language (Telugu) during the 2021 Indian assembly elections using a mixed-method approach. We categorize the claims into three categories, election, COVID-19, and others, and observe variations across languages. We compare content similarity through frequent word analysis and clustering of neural sentence embeddings. We also investigate user overlap across languages and fact-checking organizations. We measure the average time required to debunk claims and inform tipline users. Results reveal similarities in claims across languages, with some users submitting tips in multiple languages to the same fact-checkers. Fact-checkers generally require a couple of days to debunk a new claim and share the results with users. Notably, no user submits claims to multiple fact-checking organizations, indicating that each organization maintains a unique audience. We provide practical recommendations for using tiplines during elections with ethical consideration of users' information.
>
---
#### [new 058] C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文提出C2-Evo框架，用于多模态大语言模型的自提升推理。该工作属于多模态学习与自进化模型任务，旨在解决现有方法在数据增强和模型进化上的分离问题，通过联合演进数据与模型，持续生成复杂多模态问题并动态适配模型训练，从而提升数学推理性能。**

- **链接: [http://arxiv.org/pdf/2507.16518v1](http://arxiv.org/pdf/2507.16518v1)**

> **作者:** Xiuwei Chen; Wentao Hu; Hanhui Li; Jun Zhou; Zisheng Chen; Meng Cao; Yihan Zeng; Kui Zhang; Yu-Jie Yuan; Jianhua Han; Hang Xu; Xiaodan Liang
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released.
>
---
#### [new 059] AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出AlgoTune基准，评估语言模型（LM）编写高效代码解决科学计算问题的能力。任务是测试LM在算法设计和实现上的表现，对比现有开源库如SciPy等的参考实现。研究开发了基线LM代理AlgoTuner，发现其平均提速1.72倍，但模型仍缺乏算法创新，仅进行表层优化。**

- **链接: [http://arxiv.org/pdf/2507.15887v1](http://arxiv.org/pdf/2507.15887v1)**

> **作者:** Ori Press; Brandon Amos; Haoyu Zhao; Yikai Wu; Samuel K. Ainsworth; Dominik Krupke; Patrick Kidger; Touqir Sajed; Bartolomeo Stellato; Jisun Park; Nathanael Bosch; Eli Meril; Albert Steppi; Arman Zharmagambetov; Fangzhao Zhang; David Perez-Pineiro; Alberto Mercurio; Ni Zhan; Talor Abramovich; Kilian Lieret; Hanlin Zhang; Shirley Huang; Matthias Bethge; Ofir Press
>
> **摘要:** Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 155 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance.
>
---
#### [new 060] Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型微调后在分布外数据上泛化不良的问题。作者提出概念消融微调（CAFT）方法，通过解释工具识别并消除潜在不良概念方向，引导模型避免非预期泛化，无需修改训练数据。实验表明，该方法在不降低训练分布性能的情况下显著减少错误响应。**

- **链接: [http://arxiv.org/pdf/2507.16795v1](http://arxiv.org/pdf/2507.16795v1)**

> **作者:** Helena Casademunt; Caden Juang; Adam Karvonen; Samuel Marks; Senthooran Rajamanoharan; Neel Nanda
>
> **摘要:** Fine-tuning large language models (LLMs) can lead to unintended out-of-distribution generalization. Standard approaches to this problem rely on modifying training data, for example by adding data that better specify the intended generalization. However, this is not always practical. We introduce Concept Ablation Fine-Tuning (CAFT), a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning, without needing to modify the training data or otherwise use data from the target distribution. Given a set of directions in an LLM's latent space corresponding to undesired concepts, CAFT works by ablating these concepts with linear projections during fine-tuning, steering the model away from unintended generalizations. We successfully apply CAFT to three fine-tuning tasks, including emergent misalignment, a phenomenon where LLMs fine-tuned on a narrow task generalize to give egregiously misaligned responses to general questions. Without any changes to the fine-tuning data, CAFT reduces misaligned responses by 10x without degrading performance on the training distribution. Overall, CAFT represents a novel approach for steering LLM generalization without modifying training data.
>
---
#### [new 061] RDMA: Cost Effective Agent-Driven Rare Disease Discovery within Electronic Health Record Systems
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于医疗信息处理任务，旨在解决电子健康记录中罕见疾病识别困难的问题。现有方法难以处理缩写、隐式提及和隐私问题。论文提出RDMA框架，模拟医学专家识别模式，通过本地处理提升准确率与效率，减少隐私风险，支持更早诊断罕见疾病。**

- **链接: [http://arxiv.org/pdf/2507.15867v1](http://arxiv.org/pdf/2507.15867v1)**

> **作者:** John Wu; Adam Cross; Jimeng Sun
>
> **摘要:** Rare diseases affect 1 in 10 Americans, yet standard ICD coding systems fail to capture these conditions in electronic health records (EHR), leaving crucial information buried in clinical notes. Current approaches struggle with medical abbreviations, miss implicit disease mentions, raise privacy concerns with cloud processing, and lack clinical reasoning abilities. We present Rare Disease Mining Agents (RDMA), a framework that mirrors how medical experts identify rare disease patterns in EHR. RDMA connects scattered clinical observations that together suggest specific rare conditions. By handling clinical abbreviations, recognizing implicit disease patterns, and applying contextual reasoning locally on standard hardware, RDMA reduces privacy risks while improving F1 performance by upwards of 30\% and decreasing inferences costs 10-fold. This approach helps clinicians avoid the privacy risk of using cloud services while accessing key rare disease information from EHR systems, supporting earlier diagnosis for rare disease patients. Available at https://github.com/jhnwu3/RDMA.
>
---
#### [new 062] Why Braking? Scenario Extraction and Reasoning Utilizing LLM
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动驾驶场景理解任务，旨在解决从大量驾驶数据中识别危险制动场景的问题。现有方法依赖规则，局限性强。论文提出基于大语言模型（LLM）的双路径检索框架，结合分类与嵌入方法，提升复杂场景中的制动原因理解和泛化能力，并在Argoverse 2数据集上验证效果。**

- **链接: [http://arxiv.org/pdf/2507.15874v1](http://arxiv.org/pdf/2507.15874v1)**

> **作者:** Yin Wu; Daniel Slieter; Vivek Subramanian; Ahmed Abouelazm; Robin Bohn; J. Marius Zöllner
>
> **摘要:** The growing number of ADAS-equipped vehicles has led to a dramatic increase in driving data, yet most of them capture routine driving behavior. Identifying and understanding safety-critical corner cases within this vast dataset remains a significant challenge. Braking events are particularly indicative of potentially hazardous situations, motivating the central question of our research: Why does a vehicle brake? Existing approaches primarily rely on rule-based heuristics to retrieve target scenarios using predefined condition filters. While effective in simple environments such as highways, these methods lack generalization in complex urban settings. In this paper, we propose a novel framework that leverages Large Language Model (LLM) for scenario understanding and reasoning. Our method bridges the gap between low-level numerical signals and natural language descriptions, enabling LLM to interpret and classify driving scenarios. We propose a dual-path scenario retrieval that supports both category-based search for known scenarios and embedding-based retrieval for unknown Out-of-Distribution (OOD) scenarios. To facilitate evaluation, we curate scenario annotations on the Argoverse 2 Sensor Dataset. Experimental results show that our method outperforms rule-based baselines and generalizes well to OOD scenarios.
>
---
## 更新

#### [replaced 001] Continuously Updating Digital Twins using Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12091v2](http://arxiv.org/pdf/2506.12091v2)**

> **作者:** Harry Amad; Nicolás Astorga; Mihaela van der Schaar
>
> **摘要:** Digital twins are models of real-world systems that can simulate their dynamics in response to potential actions. In complex settings, the state and action variables, and available data and knowledge relevant to a system can constantly change, requiring digital twins to continuously update with these changes to remain relevant. Current approaches struggle in this regard, as they require fixed, well-defined modelling environments, and they cannot adapt to novel variables without re-designs, or incorporate new information without re-training. To address this, we frame digital twinning as an in-context learning problem using large language models, enabling seamless updates to the twin at inference time. We develop CALM-DT, a Context-Adaptive Language Model-based Digital Twin that can accurately simulate across diverse state-action spaces using in-context learning alone by utilising fine-tuned encoders for sample retrieval. We empirically demonstrate CALM-DT's competitive performance with existing digital twin approaches, and its unique ability to adapt to changes in its modelling environment without parameter updates.
>
---
#### [replaced 002] MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment
- **分类: cs.CL; cs.LG; stat.ME**

- **链接: [http://arxiv.org/pdf/2502.18699v3](http://arxiv.org/pdf/2502.18699v3)**

> **作者:** Tianze Wang; Dongnan Gui; Yifan Hu; Shuhang Lin; Linjun Zhang
>
> **备注:** ICML 2025
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has shown promise in aligning large language models (LLMs). Yet its reliance on a singular reward model often overlooks the diversity of human preferences. Recent approaches address this limitation by leveraging multi-dimensional feedback to fine-tune corresponding reward models and train LLMs using reinforcement learning. However, the process is costly and unstable, especially given the competing and heterogeneous nature of human preferences. In this paper, we propose Mixing Preference Optimization (MPO), a post-processing framework for aggregating single-objective policies as an alternative to both multi-objective RLHF (MORLHF) and MaxMin-RLHF. MPO avoids alignment from scratch. Instead, it log-linearly combines existing policies into a unified one with the weight of each policy computed via a batch stochastic mirror descent. Empirical results demonstrate that MPO achieves balanced performance across diverse preferences, outperforming or matching existing models with significantly reduced computational costs.
>
---
#### [replaced 003] Beyond English: Evaluating Automated Measurement of Moral Foundations in Non-English Discourse with a Chinese Case Study
- **分类: cs.CL; cs.SI**

- **链接: [http://arxiv.org/pdf/2502.02451v3](http://arxiv.org/pdf/2502.02451v3)**

> **作者:** Calvin Yixiang Cheng; Scott A Hale
>
> **备注:** 12 pages, 2 figures, 6 tables
>
> **摘要:** This study explores computational approaches for measuring moral foundations (MFs) in non-English corpora. Since most resources are developed primarily for English, cross-linguistic applications of moral foundation theory remain limited. Using Chinese as a case study, this paper evaluates the effectiveness of applying English resources to machine translated text, local language lexicons, multilingual language models, and large language models (LLMs) in measuring MFs in non-English texts. The results indicate that machine translation and local lexicon approaches are insufficient for complex moral assessments, frequently resulting in a substantial loss of cultural information. In contrast, multilingual models and LLMs demonstrate reliable cross-language performance with transfer learning, with LLMs excelling in terms of data efficiency. Importantly, this study also underscores the need for human-in-the-loop validation of automated MF assessment, as the most advanced models may overlook cultural nuances in cross-language measurements. The findings highlight the potential of LLMs for cross-language MF measurements and other complex multilingual deductive coding tasks.
>
---
#### [replaced 004] Alto: Orchestrating Distributed Compound AI Systems with Nested Ancestry
- **分类: cs.AI; cs.CL; cs.DC; cs.IR**

- **链接: [http://arxiv.org/pdf/2403.04311v3](http://arxiv.org/pdf/2403.04311v3)**

> **作者:** Deepti Raghavan; Keshav Santhanam; Muhammad Shahir Rahman; Nayani Modugula; Luis Gaspar Schroeder; Maximilien Cura; Houjun Liu; Pratiksha Thaker; Philip Levis; Matei Zaharia
>
> **摘要:** Compound AI applications chain together subcomponents such as generative language models, document retrievers, and embedding models. Applying traditional systems optimizations such as parallelism and pipelining in compound AI systems is difficult because each component has different constraints in terms of the granularity and type of data that it ingests. New data is often generated during intermediate computations, and text streams may be split into smaller, independent fragments (such as documents to sentences) which may then be re-aggregated at later parts of the computation. Due to this complexity, existing systems to serve compound AI queries do not fully take advantage of parallelism and pipelining opportunities. We present Alto, a framework that automatically optimizes execution of compound AI queries through streaming and parallelism. Bento introduces a new abstraction called nested ancestry, a metadata hierarchy that allows the system to correctly track partial outputs and aggregate data across the heterogeneous constraints of the components of compound AI applications. This metadata is automatically inferred from the programming model, allowing developers to express complex dataflow patterns without needing to reason manually about the details of routing and aggregation. Implementations of four applications in Alto outperform or match implementations in LangGraph, a popular existing AI programming framework. Alto implementations match or improve latency by between 10-30%.
>
---
#### [replaced 005] Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.06821v3](http://arxiv.org/pdf/2506.06821v3)**

> **作者:** Yuhan Cao; Zian Chen; Kun Quan; Ziliang Zhang; Yu Wang; Xiaoning Dong; Yeqi Feng; Guanzhong He; Jingcheng Huang; Jianhao Li; Yixuan Tan; Jiafu Tang; Yilin Tang; Junlei Wu; Qianyu Xiao; Can Zheng; Shouchen Zhou; Yuxiang Zhu; Yiming Huang; Tian Xie; Tianxing He
>
> **备注:** 37 pages, 22 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
>
---
#### [replaced 006] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.15879v3](http://arxiv.org/pdf/2503.15879v3)**

> **作者:** DongGeon Lee; Ahjeong Park; Hyeri Lee; Hyeonseo Nam; Yunho Maeng
>
> **备注:** Accepted to XLLM@ACL 2025
>
> **摘要:** Addressing non-factoid question answering (NFQA) remains challenging due to its open-ended nature, diverse user intents, and need for multi-aspect reasoning. These characteristics often reveal the limitations of conventional retrieval-augmented generation (RAG) approaches. To overcome these challenges, we propose Typed-RAG, a framework for type-aware decomposition of non-factoid questions (NFQs) within the RAG paradigm. Specifically, Typed-RAG first classifies an NFQ into a predefined type (e.g., Debate, Experience, Comparison). It then decomposes the question into focused sub-queries, each focusing on a single aspect. This decomposition enhances both retrieval relevance and answer quality. By combining the results of these sub-queries, Typed-RAG produces more informative and contextually aligned responses. Additionally, we construct Wiki-NFQA, a benchmark dataset for NFQA covering a wide range of NFQ types. Experiments show that Typed-RAG consistently outperforms existing QA approaches based on LLMs or RAG methods, validating the effectiveness of type-aware decomposition for improving both retrieval quality and answer generation in NFQA. Our code and dataset are available on https://github.com/TeamNLP/Typed-RAG.
>
---
#### [replaced 007] Lessons from the TREC Plain Language Adaptation of Biomedical Abstracts (PLABA) track
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.14096v2](http://arxiv.org/pdf/2507.14096v2)**

> **作者:** Brian Ondov; William Xia; Kush Attal; Ishita Unde; Jerry He; Dina Demner-Fushman
>
> **摘要:** Objective: Recent advances in language models have shown potential to adapt professional-facing biomedical literature to plain language, making it accessible to patients and caregivers. However, their unpredictability, combined with the high potential for harm in this domain, means rigorous evaluation is necessary. Our goals with this track were to stimulate research and to provide high-quality evaluation of the most promising systems. Methods: We hosted the Plain Language Adaptation of Biomedical Abstracts (PLABA) track at the 2023 and 2024 Text Retrieval Conferences. Tasks included complete, sentence-level, rewriting of abstracts (Task 1) as well as identifying and replacing difficult terms (Task 2). For automatic evaluation of Task 1, we developed a four-fold set of professionally-written references. Submissions for both Tasks 1 and 2 were provided extensive manual evaluation from biomedical experts. Results: Twelve teams spanning twelve countries participated in the track, with models from multilayer perceptrons to large pretrained transformers. In manual judgments of Task 1, top-performing models rivaled human levels of factual accuracy and completeness, but not simplicity or brevity. Automatic, reference-based metrics generally did not correlate well with manual judgments. In Task 2, systems struggled with identifying difficult terms and classifying how to replace them. When generating replacements, however, LLM-based systems did well in manually judged accuracy, completeness, and simplicity, though not in brevity. Conclusion: The PLABA track showed promise for using Large Language Models to adapt biomedical literature for the general public, while also highlighting their deficiencies and the need for improved automatic benchmarking tools.
>
---
#### [replaced 008] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v2](http://arxiv.org/pdf/2507.11936v2)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 009] Universal Model Routing for Efficient LLM Inference
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.08773v2](http://arxiv.org/pdf/2502.08773v2)**

> **作者:** Wittawat Jitkrittum; Harikrishna Narasimhan; Ankit Singh Rawat; Jeevesh Juneja; Congchao Wang; Zifeng Wang; Alec Go; Chen-Yu Lee; Pradeep Shenoy; Rina Panigrahy; Aditya Krishna Menon; Sanjiv Kumar
>
> **摘要:** Model routing is a simple technique for reducing the inference cost of large language models (LLMs), wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose UniRoute, a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective instantiations of UniRoute, relying on cluster-based routing and a learned cluster map respectively. We show that these are estimates of a theoretically optimal routing rule, and quantify their errors via an excess risk bound. Experiments on a range of public benchmarks show the effectiveness of UniRoute in routing amongst more than 30 unseen LLMs.
>
---
#### [replaced 010] R-Bot: An LLM-based Query Rewrite System
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.01661v2](http://arxiv.org/pdf/2412.01661v2)**

> **作者:** Zhaoyan Sun; Xuanhe Zhou; Guoliang Li; Xiang Yu; Jianhua Feng; Yong Zhang
>
> **摘要:** Query rewrite is essential for optimizing SQL queries to improve their execution efficiency without changing their results. Traditionally, this task has been tackled through heuristic and learning-based methods, each with its limitations in terms of inferior quality and low robustness. Recent advancements in LLMs offer a new paradigm by leveraging their superior natural language and code comprehension abilities. Despite their potential, directly applying LLMs like GPT-4 has faced challenges due to problems such as hallucinations, where the model might generate inaccurate or irrelevant results. To address this, we propose R-Bot, an LLM-based query rewrite system with a systematic approach. We first design a multi-source rewrite evidence preparation pipeline to generate query rewrite evidences for guiding LLMs to avoid hallucinations. We then propose a hybrid structure-semantics retrieval method that combines structural and semantic analysis to retrieve the most relevant rewrite evidences for effectively answering an online query. We next propose a step-by-step LLM rewrite method that iteratively leverages the retrieved evidences to select and arrange rewrite rules with self-reflection. We conduct comprehensive experiments on real-world datasets and widely used benchmarks, and demonstrate the superior performance of our system, R-Bot, surpassing state-of-the-art query rewrite methods. The R-Bot system has been deployed at Huawei and with real customers, and the results show that the proposed R-Bot system achieves lower query latency.
>
---
#### [replaced 011] Speech as a Multimodal Digital Phenotype for Multi-Task LLM-based Mental Health Prediction
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.23822v2](http://arxiv.org/pdf/2505.23822v2)**

> **作者:** Mai Ali; Christopher Lucasius; Tanmay P. Patel; Madison Aitken; Jacob Vorstman; Peter Szatmari; Marco Battaglia; Deepa Kundur
>
> **备注:** 6 pages, 1 figure, 3 tables. Accepted to ICSM 2025. The corresponding author is Mai Ali (maia dot ali at mail dot utoronto dot ca). Christopher Lucasius and Tanmay P. Patel contributed equally
>
> **摘要:** Speech is a noninvasive digital phenotype that can offer valuable insights into mental health conditions, but it is often treated as a single modality. In contrast, we propose the treatment of patient speech data as a trimodal multimedia data source for depression detection. This study explores the potential of large language model-based architectures for speech-based depression prediction in a multimodal regime that integrates speech-derived text, acoustic landmarks, and vocal biomarkers. Adolescent depression presents a significant challenge and is often comorbid with multiple disorders, such as suicidal ideation and sleep disturbances. This presents an additional opportunity to integrate multi-task learning (MTL) into our study by simultaneously predicting depression, suicidal ideation, and sleep disturbances using the multimodal formulation. We also propose a longitudinal analysis strategy that models temporal changes across multiple clinical interactions, allowing for a comprehensive understanding of the conditions' progression. Our proposed approach, featuring trimodal, longitudinal MTL is evaluated on the Depression Early Warning dataset. It achieves a balanced accuracy of 70.8%, which is higher than each of the unimodal, single-task, and non-longitudinal methods.
>
---
#### [replaced 012] A Method for the Architecture of a Medical Vertical Large Language Model Based on Deepseek R1
- **分类: cs.CL; cs.AI; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2505.00025v2](http://arxiv.org/pdf/2505.00025v2)**

> **作者:** Mingda Zhang; Jianglong Qin
>
> **备注:** 14 pages, 1 figures
>
> **摘要:** Despite significant advances in foundation models like DeepSeek-R1 and ChatGPT, their deployment in medical settings faces critical challenges including computational requirements and professional knowledge barriers. This paper presents an efficient lightweight medical large language model architecture that systematically addresses these challenges through three-dimensional optimization: knowledge acquisition, model compression, and computational enhancement. We design a knowledge transfer pipeline from DeepSeek-R1-Distill-70B to DeepSeek-R1-Distill-7B using Low-Rank Adaptation (LoRA) for precise medical knowledge retention. Through 4-bit quantization and mixed-precision strategies, we achieve substantial model compression while preserving medical reasoning capabilities. The inference framework incorporates Flash Attention acceleration and continuous batching, complemented by specialized prompt templates for diverse medical queries. Experimental evaluation on medical benchmarks demonstrates that our approach maintains 92.1% accuracy on USMLE examinations while reducing memory consumption by 64.7% and inference latency by 12.4% compared to baseline models. This work provides a practical solution for deploying advanced language models in resource-constrained medical environments, enabling broader accessibility of AI-assisted healthcare.
>
---
#### [replaced 013] Adaptive Graph Pruning for Multi-Agent Communication
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.02951v2](http://arxiv.org/pdf/2506.02951v2)**

> **作者:** Boyi Li; Zhonghan Zhao; Der-Horng Lee; Gaoang Wang
>
> **备注:** ECAI 2025
>
> **摘要:** Large Language Model (LLM) based multi-agent systems have shown remarkable performance in various tasks, especially when enhanced through collaborative communication. However, current methods often rely on a fixed number of agents and static communication structures, limiting their ability to adapt to varying task complexities. In this paper, we propose Adaptive Graph Pruning (AGP), a novel task-adaptive multi-agent collaboration framework that jointly optimizes agent quantity (hard-pruning) and communication topology (soft-pruning). Specifically, our method employs a two-stage training strategy: firstly, independently training soft-pruning networks for different agent quantities to determine optimal agent-quantity-specific complete graphs and positional masks across specific tasks; and then jointly optimizing hard-pruning and soft-pruning within a maximum complete graph to dynamically configure the number of agents and their communication topologies per task. Extensive experiments demonstrate that our approach is: (1) High-performing, achieving state-of-the-art results across six benchmarks and consistently generalizes across multiple mainstream LLM architectures, with a increase in performance of $2.58\%\sim 9.84\%$; (2) Task-adaptive, dynamically constructing optimized communication topologies tailored to specific tasks, with an extremely high performance in all three task categories (general reasoning, mathematical reasoning, and code generation); (3) Token-economical, having fewer training steps and token consumption at the same time, with a decrease in token consumption of $90\%+$; and (4) Training-efficient, achieving high performance with very few training steps compared with other methods. The performance will surpass the existing baselines after about ten steps of training under six benchmarks.
>
---
#### [replaced 014] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05724v2](http://arxiv.org/pdf/2507.05724v2)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model Omni-router Transformer. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [replaced 015] Multimodal Forecasting of Sparse Intraoperative Hypotension Events Powered by Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22116v3](http://arxiv.org/pdf/2505.22116v3)**

> **作者:** Jintao Zhang; Zirui Liu; Mingyue Cheng; Shilong Zhang; Tingyue Pan; Yitong zhou; Qi Liu; Yanhu Xie
>
> **摘要:** Intraoperative hypotension (IOH) frequently occurs under general anesthesia and is strongly linked to adverse outcomes such as myocardial injury and increased mortality. Despite its significance, IOH prediction is hindered by event sparsity and the challenge of integrating static and dynamic data across diverse patients. In this paper, we propose \textbf{IOHFuseLM}, a multimodal language model framework. To accurately identify and differentiate sparse hypotensive events, we leverage a two-stage training strategy. The first stage involves domain adaptive pretraining on IOH physiological time series augmented through diffusion methods, thereby enhancing the model sensitivity to patterns associated with hypotension. Subsequently, task fine-tuning is performed on the original clinical dataset to further enhance the ability to distinguish normotensive from hypotensive states. To enable multimodal fusion for each patient, we align structured clinical descriptions with the corresponding physiological time series at the token level. Such alignment enables the model to capture individualized temporal patterns alongside their corresponding clinical semantics. In addition, we convert static patient attributes into structured text to enrich personalized information. Experimental evaluations on two intraoperative datasets demonstrate that IOHFuseLM outperforms established baselines in accurately identifying IOH events, highlighting its applicability in clinical decision support scenarios. Our code is publicly available to promote reproducibility at https://github.com/zjt-gpu/IOHFuseLM.
>
---
#### [replaced 016] Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14241v2](http://arxiv.org/pdf/2507.14241v2)**

> **作者:** Rithesh Murthy; Ming Zhu; Liangwei Yang; Jielin Qiu; Juntao Tan; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** Large Language Models (LLMs) perform best with well-crafted prompts, yet prompt engineering remains manual, inconsistent, and inaccessible to non-experts. We introduce Promptomatix, an automatic prompt optimization framework that transforms natural language task descriptions into high-quality prompts without requiring manual tuning or domain expertise. Promptomatix supports both a lightweight meta-prompt-based optimizer and a DSPy-powered compiler, with modular design enabling future extension to more advanced frameworks. The system analyzes user intent, generates synthetic training data, selects prompting strategies, and refines prompts using cost-aware objectives. Evaluated across 5 task categories, Promptomatix achieves competitive or superior performance compared to existing libraries, while reducing prompt length and computational overhead making prompt optimization scalable and efficient.
>
---
#### [replaced 017] Banzhida: Advancing Large Language Models for Tibetan with Curated Data and Continual Pre-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09205v2](http://arxiv.org/pdf/2507.09205v2)**

> **作者:** Leiyu Pan; Bojian Xiong; Lei Yang; Renren Jin; Shaowei Zhang; Yue Chen; Ling Shi; Jiang Zhou; Junru Wu; Zhen Wang; Jianxiang Peng; Juesi Xiao; Tianyu Dong; Zhuowen Han; Zhuo Chen; Sangjee Dondrub; Caizang Tai; Haixing Zhao; Huaque Cairang; Suonan Cairang; Rou Te; Lengben Zhaxi; Gazang Zhaxi; Zhonglin Ye; Yuhui Zheng; Chunyan Peng; Secha Jia; Pema Tashi; Cizhen Jiacuo; Pema Dorjee; Hongkai Liu; Pema Yanggon; Tsehang Dorjee; Jiaxin Han; Qiongying Hu; Jilin Man; Huanke You; Yuqi Ren; Duo La; Deyi Xiong
>
> **备注:** paper modification
>
> **摘要:** Large language models have achieved remarkable progress across many languages. However, Tibetan, as a representative low-resource language, is particularly underrepresented in existing models due to the scarcity of high-quality training corpora. To address this gap, we curate the largest Tibetan pre-training corpus to date, aggregating data from diverse sources and applying a dedicated data cleaning and processing pipeline tailored for Tibetan. With the curated data, we continue pre/post-training a multilingual base model into Banzhida, a multilingual large language model that advances generative AI for Tibetan. To evaluate the Tibetan capabilities of the model, we create new high-quality Tibetan benchmarks, and complement them with existing public benchmarks. Experimental results demonstrate that Banzhida consistently and significantly outperforms both open-source models of similar scale and Tibetan-tailored models across a wide range of tasks.
>
---
#### [replaced 018] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16104v2](http://arxiv.org/pdf/2505.16104v2)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Gerard de Melo; Xiaoling Wang; Linlin Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** With the increasing size of Large Vision-Language Models (LVLMs), network pruning techniques aimed at compressing models for deployment in resource-constrained environments have garnered significant attention. However, we observe that pruning often leads to a degradation in safety performance. To address this issue, we present a novel and lightweight approach, termed Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the contribution of each attention head to safety, identifying the most critical ones, and then selectively restoring neurons directly within these attention heads that play a pivotal role in maintaining safety. This process hierarchically realigns the safety of pruned LVLMs, progressing from the attention head level to the neuron level. We validate HSR across various models and pruning strategies, consistently achieving notable improvements in safety performance. To our knowledge, this is the first work explicitly focused on restoring safety in LVLMs post-pruning.
>
---
#### [replaced 019] Evaluating Intermediate Reasoning of Code-Assisted Large Language Models for Mathematics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17665v2](http://arxiv.org/pdf/2504.17665v2)**

> **作者:** Zena Al-Khalili; Nick Howell; Dietrich Klakow
>
> **摘要:** Assisting LLMs with code generation improved their performance on mathematical reasoning tasks. However, the evaluation of code-assisted LLMs is generally restricted to execution correctness, lacking a rigorous evaluation of their generated programs. In this work, we bridge this gap by conducting an in-depth analysis of code-assisted LLMs generated programs in response to math reasoning tasks, with a focus on evaluating the soundness of the underlying reasoning processes. For this purpose, we assess the generations of five LLMs, on several math datasets, both manually and automatically, and propose a taxonomy of generated programs based on their logical soundness. Our findings show that the capabilities of models significantly impact the logic implemented to solve the problem. Closed-source LLMs ground their programs in mathematical concepts, whereas open-source models often resort to unsound reasoning, relying on memorized information and exhaustive searches. Furthermore, increasing the difficulty of problems decreases sound generations for all models, revealing a critical shortcoming of LLMs on complex mathematics, contrary to what accuracy metrics suggest. Our work highlights the need for more holistic evaluations of code-assisted LLMs beyond execution accuracy metrics, toward a better understanding of LLMs' limits in the math domain.
>
---
#### [replaced 020] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06261v4](http://arxiv.org/pdf/2507.06261v4)**

> **作者:** Gheorghe Comanici; Eric Bieber; Mike Schaekermann; Ice Pasupat; Noveen Sachdeva; Inderjit Dhillon; Marcel Blistein; Ori Ram; Dan Zhang; Evan Rosen; Luke Marris; Sam Petulla; Colin Gaffney; Asaf Aharoni; Nathan Lintz; Tiago Cardal Pais; Henrik Jacobsson; Idan Szpektor; Nan-Jiang Jiang; Krishna Haridasan; Ahmed Omran; Nikunj Saunshi; Dara Bahri; Gaurav Mishra; Eric Chu; Toby Boyd; Brad Hekman; Aaron Parisi; Chaoyi Zhang; Kornraphop Kawintiranon; Tania Bedrax-Weiss; Oliver Wang; Ya Xu; Ollie Purkiss; Uri Mendlovic; Ilaï Deutel; Nam Nguyen; Adam Langley; Flip Korn; Lucia Rossazza; Alexandre Ramé; Sagar Waghmare; Helen Miller; Nathan Byrd; Ashrith Sheshan; Raia Hadsell Sangnie Bhardwaj; Pawel Janus; Tero Rissa; Dan Horgan; Sharon Silver; Ayzaan Wahid; Sergey Brin; Yves Raimond; Klemen Kloboves; Cindy Wang; Nitesh Bharadwaj Gundavarapu; Ilia Shumailov; Bo Wang; Mantas Pajarskas; Joe Heyward; Martin Nikoltchev; Maciej Kula; Hao Zhou; Zachary Garrett; Sushant Kafle; Sercan Arik; Ankita Goel; Mingyao Yang; Jiho Park; Koji Kojima; Parsa Mahmoudieh; Koray Kavukcuoglu; Grace Chen; Doug Fritz; Anton Bulyenov; Sudeshna Roy; Dimitris Paparas; Hadar Shemtov; Bo-Juen Chen; Robin Strudel; David Reitter; Aurko Roy; Andrey Vlasov; Changwan Ryu; Chas Leichner; Haichuan Yang; Zelda Mariet; Denis Vnukov; Tim Sohn; Amy Stuart; Wei Liang; Minmin Chen; Praynaa Rawlani; Christy Koh; JD Co-Reyes; Guangda Lai; Praseem Banzal; Dimitrios Vytiniotis; Jieru Mei; Mu Cai; Mohammed Badawi; Corey Fry; Ale Hartman; Daniel Zheng; Eric Jia; James Keeling; Annie Louis; Ying Chen; Efren Robles; Wei-Chih Hung; Howard Zhou; Nikita Saxena; Sonam Goenka; Olivia Ma; Zach Fisher; Mor Hazan Taege; Emily Graves; David Steiner; Yujia Li; Sarah Nguyen; Rahul Sukthankar; Joe Stanton; Ali Eslami; Gloria Shen; Berkin Akin; Alexey Guseynov; Yiqian Zhou; Jean-Baptiste Alayrac; Armand Joulin; Efrat Farkash; Ashish Thapliyal; Stephen Roller; Noam Shazeer; Todor Davchev; Terry Koo; Hannah Forbes-Pollard; Kartik Audhkhasi; Greg Farquhar; Adi Mayrav Gilady; Maggie Song; John Aslanides; Piermaria Mendolicchio; Alicia Parrish; John Blitzer; Pramod Gupta; Xiaoen Ju; Xiaochen Yang; Puranjay Datta; Andrea Tacchetti; Sanket Vaibhav Mehta; Gregory Dibb; Shubham Gupta; Federico Piccinini; Raia Hadsell; Sujee Rajayogam; Jiepu Jiang; Patrick Griffin; Patrik Sundberg; Jamie Hayes; Alexey Frolov; Tian Xie; Adam Zhang; Kingshuk Dasgupta; Uday Kalra; Lior Shani; Klaus Macherey; Tzu-Kuo Huang; Liam MacDermed; Karthik Duddu; Paulo Zacchello; Zi Yang; Jessica Lo; Kai Hui; Matej Kastelic; Derek Gasaway; Qijun Tan; Summer Yue; Pablo Barrio; John Wieting; Weel Yang; Andrew Nystrom; Solomon Demmessie; Anselm Levskaya; Fabio Viola; Chetan Tekur; Greg Billock; George Necula; Mandar Joshi; Rylan Schaeffer; Swachhand Lokhande; Christina Sorokin; Pradeep Shenoy; Mia Chen; Mark Collier; Hongji Li; Taylor Bos; Nevan Wichers; Sun Jae Lee; Angéline Pouget; Santhosh Thangaraj; Kyriakos Axiotis; Phil Crone; Rachel Sterneck; Nikolai Chinaev; Victoria Krakovna; Oleksandr Ferludin; Ian Gemp; Stephanie Winkler; Dan Goldberg; Ivan Korotkov; Kefan Xiao; Malika Mehrotra; Sandeep Mariserla; Vihari Piratla; Terry Thurk; Khiem Pham; Hongxu Ma; Alexandre Senges; Ravi Kumar; Clemens Meyer; Ellie Talius; Nuo Wang Pierse; Ballie Sandhu; Horia Toma; Kuo Lin; Swaroop Nath; Tom Stone; Dorsa Sadigh; Nikita Gupta; Arthur Guez; Avi Singh; Matt Thomas; Tom Duerig; Yuan Gong; Richard Tanburn; Lydia Lihui Zhang; Phuong Dao; Mohamed Hammad; Sirui Xie; Shruti Rijhwani; Ben Murdoch; Duhyeon Kim; Will Thompson; Heng-Tze Cheng; Daniel Sohn; Pablo Sprechmann; Qiantong Xu; Srinivas Tadepalli; Peter Young; Ye Zhang; Hansa Srinivasan; Miranda Aperghis; Aditya Ayyar; Hen Fitoussi; Ryan Burnell; David Madras; Mike Dusenberry; Xi Xiong; Tayo Oguntebi; Ben Albrecht; Jörg Bornschein; Jovana Mitrović; Mason Dimarco; Bhargav Kanagal Shamanna; Premal Shah; Eren Sezener; Shyam Upadhyay; Dave Lacey; Craig Schiff; Sebastien Baur; Sanjay Ganapathy; Eva Schnider; Mateo Wirth; Connor Schenck; Andrey Simanovsky; Yi-Xuan Tan; Philipp Fränken; Dennis Duan; Bharath Mankalale; Nikhil Dhawan; Kevin Sequeira; Zichuan Wei; Shivanker Goel; Caglar Unlu; Yukun Zhu; Haitian Sun; Ananth Balashankar; Kurt Shuster; Megh Umekar; Mahmoud Alnahlawi; Aäron van den Oord; Kelly Chen; Yuexiang Zhai; Zihang Dai; Kuang-Huei Lee; Eric Doi; Lukas Zilka; Rohith Vallu; Disha Shrivastava; Jason Lee; Hisham Husain; Honglei Zhuang; Vincent Cohen-Addad; Jarred Barber; James Atwood; Adam Sadovsky; Quentin Wellens; Steven Hand; Arunkumar Rajendran; Aybuke Turker; CJ Carey; Yuanzhong Xu; Hagen Soltau; Zefei Li; Xinying Song; Conglong Li; Iurii Kemaev; Sasha Brown; Andrea Burns; Viorica Patraucean; Piotr Stanczyk; Renga Aravamudhan; Mathieu Blondel; Hila Noga; Lorenzo Blanco; Will Song; Michael Isard; Mandar Sharma; Reid Hayes; Dalia El Badawy; Avery Lamp; Itay Laish; Olga Kozlova; Kelvin Chan; Sahil Singla; Srinivas Sunkara; Mayank Upadhyay; Chang Liu; Aijun Bai; Jarek Wilkiewicz; Martin Zlocha; Jeremiah Liu; Zhuowan Li; Haiguang Li; Omer Barak; Ganna Raboshchuk; Jiho Choi; Fangyu Liu; Erik Jue; Mohit Sharma; Andreea Marzoca; Robert Busa-Fekete; Anna Korsun; Andre Elisseeff; Zhe Shen; Sara Mc Carthy; Kay Lamerigts; Anahita Hosseini; Hanzhao Lin; Charlie Chen; Fan Yang; Kushal Chauhan; Mark Omernick; Dawei Jia; Karina Zainullina; Demis Hassabis; Danny Vainstein; Ehsan Amid; Xiang Zhou; Ronny Votel; Eszter Vértes; Xinjian Li; Zongwei Zhou; Angeliki Lazaridou; Brendan McMahan; Arjun Narayanan; Hubert Soyer; Sujoy Basu; Kayi Lee; Bryan Perozzi; Qin Cao; Leonard Berrada; Rahul Arya; Ke Chen; Katrina; Xu; Matthias Lochbrunner; Alex Hofer; Sahand Sharifzadeh; Renjie Wu; Sally Goldman; Pranjal Awasthi; Xuezhi Wang; Yan Wu; Claire Sha; Biao Zhang; Maciej Mikuła; Filippo Graziano; Siobhan Mcloughlin; Irene Giannoumis; Youhei Namiki; Chase Malik; Carey Radebaugh; Jamie Hall; Ramiro Leal-Cavazos; Jianmin Chen; Vikas Sindhwani; David Kao; David Greene; Jordan Griffith; Chris Welty; Ceslee Montgomery; Toshihiro Yoshino; Liangzhe Yuan; Noah Goodman; Assaf Hurwitz Michaely; Kevin Lee; KP Sawhney; Wei Chen; Zheng Zheng; Megan Shum; Nikolay Savinov; Etienne Pot; Alex Pak; Morteza Zadimoghaddam; Sijal Bhatnagar; Yoad Lewenberg; Blair Kutzman; Ji Liu; Lesley Katzen; Jeremy Selier; Josip Djolonga; Dmitry Lepikhin; Kelvin Xu; Jacky Liang; Jiewen Tan; Benoit Schillings; Muge Ersoy; Pete Blois; Bernd Bandemer; Abhimanyu Singh; Sergei Lebedev; Pankaj Joshi; Adam R. Brown; Evan Palmer; Shreya Pathak; Komal Jalan; Fedir Zubach; Shuba Lall; Randall Parker; Alok Gunjan; Sergey Rogulenko; Sumit Sanghai; Zhaoqi Leng; Zoltan Egyed; Shixin Li; Maria Ivanova; Kostas Andriopoulos; Jin Xie; Elan Rosenfeld; Auriel Wright; Ankur Sharma; Xinyang Geng; Yicheng Wang; Sam Kwei; Renke Pan; Yujing Zhang; Gabby Wang; Xi Liu; Chak Yeung; Elizabeth Cole; Aviv Rosenberg; Zhen Yang; Phil Chen; George Polovets; Pranav Nair; Rohun Saxena; Josh Smith; Shuo-yiin Chang; Aroma Mahendru; Svetlana Grant; Anand Iyer; Irene Cai; Jed McGiffin; Jiaming Shen; Alanna Walton; Antonious Girgis; Oliver Woodman; Rosemary Ke; Mike Kwong; Louis Rouillard; Jinmeng Rao; Zhihao Li; Yuntao Xu; Flavien Prost; Chi Zou; Ziwei Ji; Alberto Magni; Tyler Liechty; Dan A. Calian; Deepak Ramachandran; Igor Krivokon; Hui Huang; Terry Chen; Anja Hauth; Anastasija Ilić; Weijuan Xi; Hyeontaek Lim; Vlad-Doru Ion; Pooya Moradi; Metin Toksoz-Exley; Kalesha Bullard; Miltos Allamanis; Xiaomeng Yang; Sophie Wang; Zhi Hong; Anita Gergely; Cheng Li; Bhavishya Mittal; Vitaly Kovalev; Victor Ungureanu; Jane Labanowski; Jan Wassenberg; Nicolas Lacasse; Geoffrey Cideron; Petar Dević; Annie Marsden; Lynn Nguyen; Michael Fink; Yin Zhong; Tatsuya Kiyono; Desi Ivanov; Sally Ma; Max Bain; Kiran Yalasangi; Jennifer She; Anastasia Petrushkina; Mayank Lunayach; Carla Bromberg; Sarah Hodkinson; Vilobh Meshram; Daniel Vlasic; Austin Kyker; Steve Xu; Jeff Stanway; Zuguang Yang; Kai Zhao; Matthew Tung; Seth Odoom; Yasuhisa Fujii; Justin Gilmer; Eunyoung Kim; Felix Halim; Quoc Le; Bernd Bohnet; Seliem El-Sayed; Behnam Neyshabur; Malcolm Reynolds; Dean Reich; Yang Xu; Erica Moreira; Anuj Sharma; Zeyu Liu; Mohammad Javad Hosseini; Naina Raisinghani; Yi Su; Ni Lao; Daniel Formoso; Marco Gelmi; Almog Gueta; Tapomay Dey; Elena Gribovskaya; Domagoj Ćevid; Sidharth Mudgal; Garrett Bingham; Jianling Wang; Anurag Kumar; Alex Cullum; Feng Han; Konstantinos Bousmalis; Diego Cedillo; Grace Chu; Vladimir Magay; Paul Michel; Ester Hlavnova; Daniele Calandriello; Setareh Ariafar; Kaisheng Yao; Vikash Sehwag; Arpi Vezer; Agustin Dal Lago; Zhenkai Zhu; Paul Kishan Rubenstein; Allen Porter; Anirudh Baddepudi; Oriana Riva; Mihai Dorin Istin; Chih-Kuan Yeh; Zhi Li; Andrew Howard; Nilpa Jha; Jeremy Chen; Raoul de Liedekerke; Zafarali Ahmed; Mikel Rodriguez; Tanuj Bhatia; Bangju Wang; Ali Elqursh; David Klinghoffer; Peter Chen; Pushmeet Kohli; Te I; Weiyang Zhang; Zack Nado; Jilin Chen; Maxwell Chen; George Zhang; Aayush Singh; Adam Hillier; Federico Lebron; Yiqing Tao; Ting Liu; Gabriel Dulac-Arnold; Jingwei Zhang; Shashi Narayan; Buhuang Liu; Orhan Firat; Abhishek Bhowmick; Bingyuan Liu; Hao Zhang; Zizhao Zhang; Georges Rotival; Nathan Howard; Anu Sinha; Alexander Grushetsky; Benjamin Beyret; Keerthana Gopalakrishnan; James Zhao; Kyle He; Szabolcs Payrits; Zaid Nabulsi; Zhaoyi Zhang; Weijie Chen; Edward Lee; Nova Fallen; Sreenivas Gollapudi; Aurick Zhou; Filip Pavetić; Thomas Köppe; Shiyu Huang; Rama Pasumarthi; Nick Fernando; Felix Fischer; Daria Ćurko; Yang Gao; James Svensson; Austin Stone; Haroon Qureshi; Abhishek Sinha; Apoorv Kulshreshtha; Martin Matysiak; Jieming Mao; Carl Saroufim; Aleksandra Faust; Qingnan Duan; Gil Fidel; Kaan Katircioglu; Raphaël Lopez Kaufman; Dhruv Shah; Weize Kong; Abhishek Bapna; Gellért Weisz; Emma Dunleavy; Praneet Dutta; Tianqi Liu; Rahma Chaabouni; Carolina Parada; Marcus Wu; Alexandra Belias; Alessandro Bissacco; Stanislav Fort; Li Xiao; Fantine Huot; Chris Knutsen; Yochai Blau; Gang Li; Jennifer Prendki; Juliette Love; Yinlam Chow; Pichi Charoenpanit; Hidetoshi Shimokawa; Vincent Coriou; Karol Gregor; Tomas Izo; Arjun Akula; Mario Pinto; Chris Hahn; Dominik Paulus; Jiaxian Guo; Neha Sharma; Cho-Jui Hsieh; Adaeze Chukwuka; Kazuma Hashimoto; Nathalie Rauschmayr; Ling Wu; Christof Angermueller; Yulong Wang; Sebastian Gerlach; Michael Pliskin; Daniil Mirylenka; Min Ma; Lexi Baugher; Bryan Gale; Shaan Bijwadia; Nemanja Rakićević; David Wood; Jane Park; Chung-Ching Chang; Babi Seal; Chris Tar; Kacper Krasowiak; Yiwen Song; Georgi Stephanov; Gary Wang; Marcello Maggioni; Stein Xudong Lin; Felix Wu; Shachi Paul; Zixuan Jiang; Shubham Agrawal; Bilal Piot; Alex Feng; Cheolmin Kim; Tulsee Doshi; Jonathan Lai; Chuqiao; Xu; Sharad Vikram; Ciprian Chelba; Sebastian Krause; Vincent Zhuang; Jack Rae; Timo Denk; Adrian Collister; Lotte Weerts; Xianghong Luo; Yifeng Lu; Håvard Garnes; Nitish Gupta; Terry Spitz; Avinatan Hassidim; Lihao Liang; Izhak Shafran; Peter Humphreys; Kenny Vassigh; Phil Wallis; Virat Shejwalkar; Nicolas Perez-Nieves; Rachel Hornung; Melissa Tan; Beka Westberg; Andy Ly; Richard Zhang; Brian Farris; Jongbin Park; Alec Kosik; Zeynep Cankara; Andrii Maksai; Yunhan Xu; Albin Cassirer; Sergi Caelles; Abbas Abdolmaleki; Mencher Chiang; Alex Fabrikant; Shravya Shetty; Luheng He; Mai Giménez; Hadi Hashemi; Sheena Panthaplackel; Yana Kulizhskaya; Salil Deshmukh; Daniele Pighin; Robin Alazard; Disha Jindal; Seb Noury; Pradeep Kumar S; Siyang Qin; Xerxes Dotiwalla; Stephen Spencer; Mohammad Babaeizadeh; Blake JianHang Chen; Vaibhav Mehta; Jennie Lees; Andrew Leach; Penporn Koanantakool; Ilia Akolzin; Ramona Comanescu; Junwhan Ahn; Alexey Svyatkovskiy; Basil Mustafa; David D'Ambrosio; Shiva Mohan Reddy Garlapati; Pascal Lamblin; Alekh Agarwal; Shuang Song; Pier Giuseppe Sessa; Pauline Coquinot; John Maggs; Hussain Masoom; Divya Pitta; Yaqing Wang; Patrick Morris-Suzuki; Billy Porter; Johnson Jia; Jeffrey Dudek; Raghavender R; Cosmin Paduraru; Alan Ansell; Tolga Bolukbasi; Tony Lu; Ramya Ganeshan; Zi Wang; Henry Griffiths; Rodrigo Benenson; Yifan He; James Swirhun; George Papamakarios; Aditya Chawla; Kuntal Sengupta; Yan Wang; Vedrana Milutinovic; Igor Mordatch; Zhipeng Jia; Jamie Smith; Will Ng; Shitij Nigam; Matt Young; Eugen Vušak; Blake Hechtman; Sheela Goenka; Avital Zipori; Kareem Ayoub; Ashok Popat; Trilok Acharya; Luo Yu; Dawn Bloxwich; Hugo Song; Paul Roit; Haiqiong Li; Aviel Boag; Nigamaa Nayakanti; Bilva Chandra; Tianli Ding; Aahil Mehta; Cath Hope; Jiageng Zhang; Idan Heimlich Shtacher; Kartikeya Badola; Ryo Nakashima; Andrei Sozanschi; Iulia Comşa; Ante Žužul; Emily Caveness; Julian Odell; Matthew Watson; Dario de Cesare; Phillip Lippe; Derek Lockhart; Siddharth Verma; Huizhong Chen; Sean Sun; Lin Zhuo; Aditya Shah; Prakhar Gupta; Alex Muzio; Ning Niu; Amir Zait; Abhinav Singh; Meenu Gaba; Fan Ye; Prajit Ramachandran; Mohammad Saleh; Raluca Ada Popa; Ayush Dubey; Frederick Liu; Sara Javanmardi; Mark Epstein; Ross Hemsley; Richard Green; Nishant Ranka; Eden Cohen; Chuyuan Kelly Fu; Sanjay Ghemawat; Jed Borovik; James Martens; Anthony Chen; Pranav Shyam; André Susano Pinto; Ming-Hsuan Yang; Alexandru Ţifrea; David Du; Boqing Gong; Ayushi Agarwal; Seungyeon Kim; Christian Frank; Saloni Shah; Xiaodan Song; Zhiwei Deng; Ales Mikhalap; Kleopatra Chatziprimou; Timothy Chung; Toni Creswell; Susan Zhang; Yennie Jun; Carl Lebsack; Will Truong; Slavica Andačić; Itay Yona; Marco Fornoni; Rong Rong; Serge Toropov; Afzal Shama Soudagar; Andrew Audibert; Salah Zaiem; Zaheer Abbas; Andrei Rusu; Sahitya Potluri; Shitao Weng; Anastasios Kementsietsidis; Anton Tsitsulin; Daiyi Peng; Natalie Ha; Sanil Jain; Tejasi Latkar; Simeon Ivanov; Cory McLean; Anirudh GP; Rajesh Venkataraman; Canoee Liu; Dilip Krishnan; Joel D'sa; Roey Yogev; Paul Collins; Benjamin Lee; Lewis Ho; Carl Doersch; Gal Yona; Shawn Gao; Felipe Tiengo Ferreira; Adnan Ozturel; Hannah Muckenhirn; Ce Zheng; Gargi Balasubramaniam; Mudit Bansal; George van den Driessche; Sivan Eiger; Salem Haykal; Vedant Misra; Abhimanyu Goyal; Danilo Martins; Gary Leung; Jonas Valfridsson; Four Flynn; Will Bishop; Chenxi Pang; Yoni Halpern; Honglin Yu; Lawrence Moore; Yuvein; Zhu; Sridhar Thiagarajan; Yoel Drori; Zhisheng Xiao; Lucio Dery; Rolf Jagerman; Jing Lu; Eric Ge; Vaibhav Aggarwal; Arjun Khare; Vinh Tran; Oded Elyada; Ferran Alet; James Rubin; Ian Chou; David Tian; Libin Bai; Lawrence Chan; Lukasz Lew; Karolis Misiunas; Taylan Bilal; Aniket Ray; Sindhu Raghuram; Alex Castro-Ros; Viral Carpenter; CJ Zheng; Michael Kilgore; Josef Broder; Emily Xue; Praveen Kallakuri; Dheeru Dua; Nancy Yuen; Steve Chien; John Schultz; Saurabh Agrawal; Reut Tsarfaty; Jingcao Hu; Ajay Kannan; Dror Marcus; Nisarg Kothari; Baochen Sun; Ben Horn; Matko Bošnjak; Ferjad Naeem; Dean Hirsch; Lewis Chiang; Boya Fang; Jie Han; Qifei Wang; Ben Hora; Antoine He; Mario Lučić; Beer Changpinyo; Anshuman Tripathi; John Youssef; Chester Kwak; Philippe Schlattner; Cat Graves; Rémi Leblond; Wenjun Zeng; Anders Andreassen; Gabriel Rasskin; Yue Song; Eddie Cao; Junhyuk Oh; Matt Hoffman; Wojtek Skut; Yichi Zhang; Jon Stritar; Xingyu Cai; Saarthak Khanna; Kathie Wang; Shriya Sharma; Christian Reisswig; Younghoon Jun; Aman Prasad; Tatiana Sholokhova; Preeti Singh; Adi Gerzi Rosenthal; Anian Ruoss; Françoise Beaufays; Sean Kirmani; Dongkai Chen; Johan Schalkwyk; Jonathan Herzig; Been Kim; Josh Jacob; Damien Vincent; Adrian N Reyes; Ivana Balazevic; Léonard Hussenot; Jon Schneider; Parker Barnes; Luis Castro; Spandana Raj Babbula; Simon Green; Serkan Cabi; Nico Duduta; Danny Driess; Rich Galt; Noam Velan; Junjie Wang; Hongyang Jiao; Matthew Mauger; Du Phan; Miteyan Patel; Vlado Galić; Jerry Chang; Eyal Marcus; Matt Harvey; Julian Salazar; Elahe Dabir; Suraj Satishkumar Sheth; Amol Mandhane; Hanie Sedghi; Jeremiah Willcock; Amir Zandieh; Shruthi Prabhakara; Aida Amini; Antoine Miech; Victor Stone; Massimo Nicosia; Paul Niemczyk; Ying Xiao; Lucy Kim; Sławek Kwasiborski; Vikas Verma; Ada Maksutaj Oflazer; Christoph Hirnschall; Peter Sung; Lu Liu; Richard Everett; Michiel Bakker; Ágoston Weisz; Yufei Wang; Vivek Sampathkumar; Uri Shaham; Bibo Xu; Yasemin Altun; Mingqiu Wang; Takaaki Saeki; Guanjie Chen; Emanuel Taropa; Shanthal Vasanth; Sophia Austin; Lu Huang; Goran Petrovic; Qingyun Dou; Daniel Golovin; Grigory Rozhdestvenskiy; Allie Culp; Will Wu; Motoki Sano; Divya Jain; Julia Proskurnia; Sébastien Cevey; Alejandro Cruzado Ruiz; Piyush Patil; Mahdi Mirzazadeh; Eric Ni; Javier Snaider; Lijie Fan; Alexandre Fréchette; AJ Pierigiovanni; Shariq Iqbal; Kenton Lee; Claudio Fantacci; Jinwei Xing; Lisa Wang; Alex Irpan; David Raposo; Yi Luan; Zhuoyuan Chen; Harish Ganapathy; Kevin Hui; Jiazhong Nie; Isabelle Guyon; Heming Ge; Roopali Vij; Hui Zheng; Dayeong Lee; Alfonso Castaño; Khuslen Baatarsukh; Gabriel Ibagon; Alexandra Chronopoulou; Nicholas FitzGerald; Shashank Viswanadha; Safeen Huda; Rivka Moroshko; Georgi Stoyanov; Prateek Kolhar; Alain Vaucher; Ishaan Watts; Adhi Kuncoro; Henryk Michalewski; Satish Kambala; Bat-Orgil Batsaikhan; Alek Andreev; Irina Jurenka; Maigo Le; Qihang Chen; Wael Al Jishi; Sarah Chakera; Zhe Chen; Aditya Kini; Vikas Yadav; Aditya Siddhant; Ilia Labzovsky; Balaji Lakshminarayanan; Carrie Grimes Bostock; Pankil Botadra; Ankesh Anand; Colton Bishop; Sam Conway-Rahman; Mohit Agarwal; Yani Donchev; Achintya Singhal; Félix de Chaumont Quitry; Natalia Ponomareva; Nishant Agrawal; Bin Ni; Kalpesh Krishna; Masha Samsikova; John Karro; Yilun Du; Tamara von Glehn; Caden Lu; Christopher A. Choquette-Choo; Zhen Qin; Tingnan Zhang; Sicheng Li; Divya Tyam; Swaroop Mishra; Wing Lowe; Colin Ji; Weiyi Wang; Manaal Faruqui; Ambrose Slone; Valentin Dalibard; Arunachalam Narayanaswamy; John Lambert; Pierre-Antoine Manzagol; Dan Karliner; Andrew Bolt; Ivan Lobov; Aditya Kusupati; Chang Ye; Xuan Yang; Heiga Zen; Nelson George; Mukul Bhutani; Olivier Lacombe; Robert Riachi; Gagan Bansal; Rachel Soh; Yue Gao; Yang Yu; Adams Yu; Emily Nottage; Tania Rojas-Esponda; James Noraky; Manish Gupta; Ragha Kotikalapudi; Jichuan Chang; Sanja Deur; Dan Graur; Alex Mossin; Erin Farnese; Ricardo Figueira; Alexandre Moufarek; Austin Huang; Patrik Zochbauer; Ben Ingram; Tongzhou Chen; Zelin Wu; Adrià Puigdomènech; Leland Rechis; Da Yu; Sri Gayatri Sundara Padmanabhan; Rui Zhu; Chu-ling Ko; Andrea Banino; Samira Daruki; Aarush Selvan; Dhruva Bhaswar; Daniel Hernandez Diaz; Chen Su; Salvatore Scellato; Jennifer Brennan; Woohyun Han; Grace Chung; Priyanka Agrawal; Urvashi Khandelwal; Khe Chai Sim; Morgane Lustman; Sam Ritter; Kelvin Guu; Jiawei Xia; Prateek Jain; Emma Wang; Tyrone Hill; Mirko Rossini; Marija Kostelac; Tautvydas Misiunas; Amit Sabne; Kyuyeun Kim; Ahmet Iscen; Congchao Wang; José Leal; Ashwin Sreevatsa; Utku Evci; Manfred Warmuth; Saket Joshi; Daniel Suo; James Lottes; Garrett Honke; Brendan Jou; Stefani Karp; Jieru Hu; Himanshu Sahni; Adrien Ali Taïga; William Kong; Samrat Ghosh; Renshen Wang; Jay Pavagadhi; Natalie Axelsson; Nikolai Grigorev; Patrick Siegler; Rebecca Lin; Guohui Wang; Emilio Parisotto; Sharath Maddineni; Krishan Subudhi; Eyal Ben-David; Elena Pochernina; Orgad Keller; Thi Avrahami; Zhe Yuan; Pulkit Mehta; Jialu Liu; Sherry Yang; Wendy Kan; Katherine Lee; Tom Funkhouser; Derek Cheng; Hongzhi Shi; Archit Sharma; Joe Kelley; Matan Eyal; Yury Malkov; Corentin Tallec; Yuval Bahat; Shen Yan; Xintian; Wu; David Lindner; Chengda Wu; Avi Caciularu; Xiyang Luo; Rodolphe Jenatton; Tim Zaman; Yingying Bi; Ilya Kornakov; Ganesh Mallya; Daisuke Ikeda; Itay Karo; Anima Singh; Colin Evans; Praneeth Netrapalli; Vincent Nallatamby; Isaac Tian; Yannis Assael; Vikas Raunak; Victor Carbune; Ioana Bica; Lior Madmoni; Dee Cattle; Snchit Grover; Krishna Somandepalli; Sid Lall; Amelio Vázquez-Reina; Riccardo Patana; Jiaqi Mu; Pranav Talluri; Maggie Tran; Rajeev Aggarwal; RJ Skerry-Ryan; Jun Xu; Mike Burrows; Xiaoyue Pan; Edouard Yvinec; Di Lu; Zhiying Zhang; Duc Dung Nguyen; Hairong Mu; Gabriel Barcik; Helen Ran; Lauren Beltrone; Krzysztof Choromanski; Dia Kharrat; Samuel Albanie; Sean Purser-haskell; David Bieber; Carrie Zhang; Jing Wang; Tom Hudson; Zhiyuan Zhang; Han Fu; Johannes Mauerer; Mohammad Hossein Bateni; AJ Maschinot; Bing Wang; Muye Zhu; Arjun Pillai; Tobias Weyand; Shuang Liu; Oscar Akerlund; Fred Bertsch; Vittal Premachandran; Alicia Jin; Vincent Roulet; Peter de Boursac; Shubham Mittal; Ndaba Ndebele; Georgi Karadzhov; Sahra Ghalebikesabi; Ricky Liang; Allen Wu; Yale Cong; Nimesh Ghelani; Sumeet Singh; Bahar Fatemi; Warren; Chen; Charles Kwong; Alexey Kolganov; Steve Li; Richard Song; Chenkai Kuang; Sobhan Miryoosefi; Dale Webster; James Wendt; Arkadiusz Socala; Guolong Su; Artur Mendonça; Abhinav Gupta; Xiaowei Li; Tomy Tsai; Qiong; Hu; Kai Kang; Angie Chen; Sertan Girgin; Yongqin Xian; Andrew Lee; Nolan Ramsden; Leslie Baker; Madeleine Clare Elish; Varvara Krayvanova; Rishabh Joshi; Jiri Simsa; Yao-Yuan Yang; Piotr Ambroszczyk; Dipankar Ghosh; Arjun Kar; Yuan Shangguan; Yumeya Yamamori; Yaroslav Akulov; Andy Brock; Haotian Tang; Siddharth Vashishtha; Rich Munoz; Andreas Steiner; Kalyan Andra; Daniel Eppens; Qixuan Feng; Hayato Kobayashi; Sasha Goldshtein; Mona El Mahdy; Xin Wang; Jilei; Wang; Richard Killam; Tom Kwiatkowski; Kavya Kopparapu; Serena Zhan; Chao Jia; Alexei Bendebury; Sheryl Luo; Adrià Recasens; Timothy Knight; Jing Chen; Mohak Patel; YaGuang Li; Ben Withbroe; Dean Weesner; Kush Bhatia; Jie Ren; Danielle Eisenbud; Ebrahim Songhori; Yanhua Sun; Travis Choma; Tasos Kementsietsidis; Lucas Manning; Brian Roark; Wael Farhan; Jie Feng; Susheel Tatineni; James Cobon-Kerr; Yunjie Li; Lisa Anne Hendricks; Isaac Noble; Chris Breaux; Nate Kushman; Liqian Peng; Fuzhao Xue; Taylor Tobin; Jamie Rogers; Josh Lipschultz; Chris Alberti; Alexey Vlaskin; Mostafa Dehghani; Roshan Sharma; Tris Warkentin; Chen-Yu Lee; Benigno Uria; Da-Cheng Juan; Angad Chandorkar; Hila Sheftel; Ruibo Liu; Elnaz Davoodi; Borja De Balle Pigem; Kedar Dhamdhere; David Ross; Jonathan Hoech; Mahdis Mahdieh; Li Liu; Qiujia Li; Liam McCafferty; Chenxi Liu; Markus Mircea; Yunting Song; Omkar Savant; Alaa Saade; Colin Cherry; Vincent Hellendoorn; Siddharth Goyal; Paul Pucciarelli; David Vilar Torres; Zohar Yahav; Hyo Lee; Lars Lowe Sjoesund; Christo Kirov; Bo Chang; Deepanway Ghoshal; Lu Li; Gilles Baechler; Sébastien Pereira; Tara Sainath; Anudhyan Boral; Dominik Grewe; Afief Halumi; Nguyet Minh Phu; Tianxiao Shen; Marco Tulio Ribeiro; Dhriti Varma; Alex Kaskasoli; Vlad Feinberg; Navneet Potti; Jarrod Kahn; Matheus Wisniewski; Shakir Mohamed; Arnar Mar Hrafnkelsson; Bobak Shahriari; Jean-Baptiste Lespiau; Lisa Patel; Legg Yeung; Tom Paine; Lantao Mei; Alex Ramirez; Rakesh Shivanna; Li Zhong; Josh Woodward; Guilherme Tubone; Samira Khan; Heng Chen; Elizabeth Nielsen; Catalin Ionescu; Utsav Prabhu; Mingcen Gao; Qingze Wang; Sean Augenstein; Neesha Subramaniam; Jason Chang; Fotis Iliopoulos; Jiaming Luo; Myriam Khan; Weicheng Kuo; Denis Teplyashin; Florence Perot; Logan Kilpatrick; Amir Globerson; Hongkun Yu; Anfal Siddiqui; Nick Sukhanov; Arun Kandoor; Umang Gupta; Marco Andreetto; Moran Ambar; Donnie Kim; Paweł Wesołowski; Sarah Perrin; Ben Limonchik; Wei Fan; Jim Stephan; Ian Stewart-Binks; Ryan Kappedal; Tong He; Sarah Cogan; Romina Datta; Tong Zhou; Jiayu Ye; Leandro Kieliger; Ana Ramalho; Kyle Kastner; Fabian Mentzer; Wei-Jen Ko; Arun Suggala; Tianhao Zhou; Shiraz Butt; Hana Strejček; Lior Belenki; Subhashini Venugopalan; Mingyang Ling; Evgenii Eltyshev; Yunxiao Deng; Geza Kovacs; Mukund Raghavachari; Hanjun Dai; Tal Schuster; Steven Schwarcz; Richard Nguyen; Arthur Nguyen; Gavin Buttimore; Shrestha Basu Mallick; Sudeep Gandhe; Seth Benjamin; Michal Jastrzebski; Le Yan; Sugato Basu; Chris Apps; Isabel Edkins; James Allingham; Immanuel Odisho; Tomas Kocisky; Jewel Zhao; Linting Xue; Apoorv Reddy; Chrysovalantis Anastasiou; Aviel Atias; Sam Redmond; Kieran Milan; Nicolas Heess; Herman Schmit; Allan Dafoe; Daniel Andor; Tynan Gangwani; Anca Dragan; Sheng Zhang; Ashyana Kachra; Gang Wu; Siyang Xue; Kevin Aydin; Siqi Liu; Yuxiang Zhou; Mahan Malihi; Austin Wu; Siddharth Gopal; Candice Schumann; Peter Stys; Alek Wang; Mirek Olšák; Dangyi Liu; Christian Schallhart; Yiran Mao; Demetra Brady; Hao Xu; Tomas Mery; Chawin Sitawarin; Siva Velusamy; Tom Cobley; Alex Zhai; Christian Walder; Nitzan Katz; Ganesh Jawahar; Chinmay Kulkarni; Antoine Yang; Adam Paszke; Yinan Wang; Bogdan Damoc; Zalán Borsos; Ray Smith; Jinning Li; Mansi Gupta; Andrei Kapishnikov; Sushant Prakash; Florian Luisier; Rishabh Agarwal; Will Grathwohl; Kuangyuan Chen; Kehang Han; Nikhil Mehta; Andrew Over; Shekoofeh Azizi; Lei Meng; Niccolò Dal Santo; Kelvin Zheng; Jane Shapiro; Igor Petrovski; Jeffrey Hui; Amin Ghafouri; Jasper Snoek; James Qin; Mandy Jordan; Caitlin Sikora; Jonathan Malmaud; Yuheng Kuang; Aga Świetlik; Ruoxin Sang; Chongyang Shi; Leon Li; Andrew Rosenberg; Shubin Zhao; Andy Crawford; Jan-Thorsten Peter; Yun Lei; Xavier Garcia; Long Le; Todd Wang; Julien Amelot; Dave Orr; Praneeth Kacham; Dana Alon; Gladys Tyen; Abhinav Arora; James Lyon; Alex Kurakin; Mimi Ly; Theo Guidroz; Zhipeng Yan; Rina Panigrahy; Pingmei Xu; Thais Kagohara; Yong Cheng; Eric Noland; Jinhyuk Lee; Jonathan Lee; Cathy Yip; Maria Wang; Efrat Nehoran; Alexander Bykovsky; Zhihao Shan; Ankit Bhagatwala; Chaochao Yan; Jie Tan; Guillermo Garrido; Dan Ethier; Nate Hurley; Grace Vesom; Xu Chen; Siyuan Qiao; Abhishek Nayyar; Julian Walker; Paramjit Sandhu; Mihaela Rosca; Danny Swisher; Mikhail Dektiarev; Josh Dillon; George-Cristian Muraru; Manuel Tragut; Artiom Myaskovsky; David Reid; Marko Velic; Owen Xiao; Jasmine George; Mark Brand; Jing Li; Wenhao Yu; Shane Gu; Xiang Deng; François-Xavier Aubet; Soheil Hassas Yeganeh; Fred Alcober; Celine Smith; Trevor Cohn; Kay McKinney; Michael Tschannen; Ramesh Sampath; Gowoon Cheon; Liangchen Luo; Luyang Liu; Jordi Orbay; Hui Peng; Gabriela Botea; Xiaofan Zhang; Charles Yoon; Cesar Magalhaes; Paweł Stradomski; Ian Mackinnon; Steven Hemingray; Kumaran Venkatesan; Rhys May; Jaeyoun Kim; Alex Druinsky; Jingchen Ye; Zheng Xu; Terry Huang; Jad Al Abdallah; Adil Dostmohamed; Rachana Fellinger; Tsendsuren Munkhdalai; Akanksha Maurya; Peter Garst; Yin Zhang; Maxim Krikun; Simon Bucher; Aditya Srikanth Veerubhotla; Yaxin Liu; Sheng Li; Nishesh Gupta; Jakub Adamek; Hanwen Chen; Bernett Orlando; Aleksandr Zaks; Joost van Amersfoort; Josh Camp; Hui Wan; HyunJeong Choe; Zhichun Wu; Kate Olszewska; Weiren Yu; Archita Vadali; Martin Scholz; Daniel De Freitas; Jason Lin; Amy Hua; Xin Liu; Frank Ding; Yichao Zhou; Boone Severson; Katerina Tsihlas; Samuel Yang; Tammo Spalink; Varun Yerram; Helena Pankov; Rory Blevins; Ben Vargas; Sarthak Jauhari; Matt Miecnikowski; Ming Zhang; Sandeep Kumar; Clement Farabet; Charline Le Lan; Sebastian Flennerhag; Yonatan Bitton; Ada Ma; Arthur Bražinskas; Eli Collins; Niharika Ahuja; Sneha Kudugunta; Anna Bortsova; Minh Giang; Wanzheng Zhu; Ed Chi; Scott Lundberg; Alexey Stern; Subha Puttagunta; Jing Xiong; Xiao Wu; Yash Pande; Amit Jhindal; Daniel Murphy; Jon Clark; Marc Brockschmidt; Maxine Deines; Kevin R. McKee; Dan Bahir; Jiajun Shen; Minh Truong; Daniel McDuff; Andrea Gesmundo; Edouard Rosseel; Bowen Liang; Ken Caluwaerts; Jessica Hamrick; Joseph Kready; Mary Cassin; Rishikesh Ingale; Li Lao; Scott Pollom; Yifan Ding; Wei He; Lizzetth Bellot; Joana Iljazi; Ramya Sree Boppana; Shan Han; Tara Thompson; Amr Khalifa; Anna Bulanova; Blagoj Mitrevski; Bo Pang; Emma Cooney; Tian Shi; Rey Coaguila; Tamar Yakar; Marc'aurelio Ranzato; Nikola Momchev; Chris Rawles; Zachary Charles; Young Maeng; Yuan Zhang; Rishabh Bansal; Xiaokai Zhao; Brian Albert; Yuan Yuan; Sudheendra Vijayanarasimhan; Roy Hirsch; Vinay Ramasesh; Kiran Vodrahalli; Xingyu Wang; Arushi Gupta; DJ Strouse; Jianmo Ni; Roma Patel; Gabe Taubman; Zhouyuan Huo; Dero Gharibian; Marianne Monteiro; Hoi Lam; Shobha Vasudevan; Aditi Chaudhary; Isabela Albuquerque; Kilol Gupta; Sebastian Riedel; Chaitra Hegde; Avraham Ruderman; András György; Marcus Wainwright; Ashwin Chaugule; Burcu Karagol Ayan; Tomer Levinboim; Sam Shleifer; Yogesh Kalley; Vahab Mirrokni; Abhishek Rao; Prabakar Radhakrishnan; Jay Hartford; Jialin Wu; Zhenhai Zhu; Francesco Bertolini; Hao Xiong; Nicolas Serrano; Hamish Tomlinson; Myle Ott; Yifan Chang; Mark Graham; Jian Li; Marco Liang; Xiangzhu Long; Sebastian Borgeaud; Yanif Ahmad; Alex Grills; Diana Mincu; Martin Izzard; Yuan Liu; Jinyu Xie; Louis O'Bryan; Sameera Ponda; Simon Tong; Michelle Liu; Dan Malkin; Khalid Salama; Yuankai Chen; Rohan Anil; Anand Rao; Rigel Swavely; Misha Bilenko; Nina Anderson; Tat Tan; Jing Xie; Xing Wu; Lijun Yu; Oriol Vinyals; Andrey Ryabtsev; Rumen Dangovski; Kate Baumli; Daniel Keysers; Christian Wright; Zoe Ashwood; Betty Chan; Artem Shtefan; Yaohui Guo; Ankur Bapna; Radu Soricut; Steven Pecht; Sabela Ramos; Rui Wang; Jiahao Cai; Trieu Trinh; Paul Barham; Linda Friso; Eli Stickgold; Xiangzhuo Ding; Siamak Shakeri; Diego Ardila; Eleftheria Briakou; Phil Culliton; Adam Raveret; Jingyu Cui; David Saxton; Subhrajit Roy; Javad Azizi; Pengcheng Yin; Lucia Loher; Andrew Bunner; Min Choi; Faruk Ahmed; Eric Li; Yin Li; Shengyang Dai; Michael Elabd; Sriram Ganapathy; Shivani Agrawal; Yiqing Hua; Paige Kunkle; Sujeevan Rajayogam; Arun Ahuja; Arthur Conmy; Alex Vasiloff; Parker Beak; Christopher Yew; Jayaram Mudigonda; Bartek Wydrowski; Jon Blanton; Zhengdong Wang; Yann Dauphin; Zhuo Xu; Martin Polacek; Xi Chen; Hexiang Hu; Pauline Sho; Markus Kunesch; Mehdi Hafezi Manshadi; Eliza Rutherford; Bo Li; Sissie Hsiao; Iain Barr; Alex Tudor; Matija Kecman; Arsha Nagrani; Vladimir Pchelin; Martin Sundermeyer; Aishwarya P S; Abhijit Karmarkar; Yi Gao; Grishma Chole; Olivier Bachem; Isabel Gao; Arturo BC; Matt Dibb; Mauro Verzetti; Felix Hernandez-Campos; Yana Lunts; Matthew Johnson; Julia Di Trapani; Raphael Koster; Idan Brusilovsky; Binbin Xiong; Megha Mohabey; Han Ke; Joe Zou; Tea Sabolić; Víctor Campos; John Palowitch; Alex Morris; Linhai Qiu; Pranavaraj Ponnuramu; Fangtao Li; Vivek Sharma; Kiranbir Sodhia; Kaan Tekelioglu; Aleksandr Chuklin; Madhavi Yenugula; Erika Gemzer; Theofilos Strinopoulos; Sam El-Husseini; Huiyu Wang; Yan Zhong; Edouard Leurent; Paul Natsev; Weijun Wang; Dre Mahaarachchi; Tao Zhu; Songyou Peng; Sami Alabed; Cheng-Chun Lee; Anthony Brohan; Arthur Szlam; GS Oh; Anton Kovsharov; Jenny Lee; Renee Wong; Megan Barnes; Gregory Thornton; Felix Gimeno; Omer Levy; Martin Sevenich; Melvin Johnson; Jonathan Mallinson; Robert Dadashi; Ziyue Wang; Qingchun Ren; Preethi Lahoti; Arka Dhar; Josh Feldman; Dan Zheng; Thatcher Ulrich; Liviu Panait; Michiel Blokzijl; Cip Baetu; Josip Matak; Jitendra Harlalka; Maulik Shah; Tal Marian; Daniel von Dincklage; Cosmo Du; Ruy Ley-Wild; Bethanie Brownfield; Max Schumacher; Yury Stuken; Shadi Noghabi; Sonal Gupta; Xiaoqi Ren; Eric Malmi; Felix Weissenberger; Blanca Huergo; Maria Bauza; Thomas Lampe; Arthur Douillard; Mojtaba Seyedhosseini; Roy Frostig; Zoubin Ghahramani; Kelvin Nguyen; Kashyap Krishnakumar; Chengxi Ye; Rahul Gupta; Alireza Nazari; Robert Geirhos; Pete Shaw; Ahmed Eleryan; Dima Damen; Jennimaria Palomaki; Ted Xiao; Qiyin Wu; Quan Yuan; Phoenix Meadowlark; Matthew Bilotti; Raymond Lin; Mukund Sridhar; Yannick Schroecker; Da-Woon Chung; Jincheng Luo; Trevor Strohman; Tianlin Liu; Anne Zheng; Jesse Emond; Wei Wang; Andrew Lampinen; Toshiyuki Fukuzawa; Folawiyo Campbell-Ajala; Monica Roy; James Lee-Thorp; Lily Wang; Iftekhar Naim; Tony; Nguy\~ên; Guy Bensky; Aditya Gupta; Dominika Rogozińska; Justin Fu; Thanumalayan Sankaranarayana Pillai; Petar Veličković; Shahar Drath; Philipp Neubeck; Vaibhav Tulsyan; Arseniy Klimovskiy; Don Metzler; Sage Stevens; Angel Yeh; Junwei Yuan; Tianhe Yu; Kelvin Zhang; Alec Go; Vincent Tsang; Ying Xu; Andy Wan; Isaac Galatzer-Levy; Sam Sobell; Abodunrinwa Toki; Elizabeth Salesky; Wenlei Zhou; Diego Antognini; Sholto Douglas; Shimu Wu; Adam Lelkes; Frank Kim; Paul Cavallaro; Ana Salazar; Yuchi Liu; James Besley; Tiziana Refice; Yiling Jia; Zhang Li; Michal Sokolik; Arvind Kannan; Jon Simon; Jo Chick; Avia Aharon; Meet Gandhi; Mayank Daswani; Keyvan Amiri; Vighnesh Birodkar; Abe Ittycheriah; Peter Grabowski; Oscar Chang; Charles Sutton; Zhixin; Lai; Umesh Telang; Susie Sargsyan; Tao Jiang; Raphael Hoffmann; Nicole Brichtova; Matteo Hessel; Jonathan Halcrow; Sammy Jerome; Geoff Brown; Alex Tomala; Elena Buchatskaya; Dian Yu; Sachit Menon; Pol Moreno; Yuguo Liao; Vicky Zayats; Luming Tang; SQ Mah; Ashish Shenoy; Alex Siegman; Majid Hadian; Okwan Kwon; Tao Tu; Nima Khajehnouri; Ryan Foley; Parisa Haghani; Zhongru Wu; Vaishakh Keshava; Khyatti Gupta; Tony Bruguier; Rui Yao; Danny Karmon; Luisa Zintgraf; Zhicheng Wang; Enrique Piqueras; Junehyuk Jung; Jenny Brennan; Diego Machado; Marissa Giustina; MH Tessler; Kamyu Lee; Qiao Zhang; Joss Moore; Kaspar Daugaard; Alexander Frömmgen; Jennifer Beattie; Fred Zhang; Daniel Kasenberg; Ty Geri; Danfeng Qin; Gaurav Singh Tomar; Tom Ouyang; Tianli Yu; Luowei Zhou; Rajiv Mathews; Andy Davis; Yaoyiran Li; Jai Gupta; Damion Yates; Linda Deng; Elizabeth Kemp; Ga-Young Joung; Sergei Vassilvitskii; Mandy Guo; Pallavi LV; Dave Dopson; Sami Lachgar; Lara McConnaughey; Himadri Choudhury; Dragos Dena; Aaron Cohen; Joshua Ainslie; Sergey Levi; Parthasarathy Gopavarapu; Polina Zablotskaia; Hugo Vallet; Sanaz Bahargam; Xiaodan Tang; Nenad Tomasev; Ethan Dyer; Daniel Balle; Hongrae Lee; William Bono; Jorge Gonzalez Mendez; Vadim Zubov; Shentao Yang; Ivor Rendulic; Yanyan Zheng; Andrew Hogue; Golan Pundak; Ralph Leith; Avishkar Bhoopchand; Michael Han; Mislav Žanić; Tom Schaul; Manolis Delakis; Tejas Iyer; Guanyu Wang; Harman Singh; Abdelrahman Abdelhamed; Tara Thomas; Siddhartha Brahma; Hilal Dib; Naveen Kumar; Wenxuan Zhou; Liang Bai; Pushkar Mishra; Jiao Sun; Valentin Anklin; Roykrong Sukkerd; Lauren Agubuzu; Anton Briukhov; Anmol Gulati; Maximilian Sieb; Fabio Pardo; Sara Nasso; Junquan Chen; Kexin Zhu; Tiberiu Sosea; Alex Goldin; Keith Rush; Spurthi Amba Hombaiah; Andreas Noever; Allan Zhou; Sam Haves; Mary Phuong; Jake Ades; Yi-ting Chen; Lin Yang; Joseph Pagadora; Stan Bileschi; Victor Cotruta; Rachel Saputro; Arijit Pramanik; Sean Ammirati; Dan Garrette; Kevin Villela; Tim Blyth; Canfer Akbulut; Neha Jha; Alban Rrustemi; Arissa Wongpanich; Chirag Nagpal; Yonghui Wu; Morgane Rivière; Sergey Kishchenko; Pranesh Srinivasan; Alice Chen; Animesh Sinha; Trang Pham; Bill Jia; Tom Hennigan; Anton Bakalov; Nithya Attaluri; Drew Garmon; Daniel Rodriguez; Dawid Wegner; Wenhao Jia; Evan Senter; Noah Fiedel; Denis Petek; Yuchuan Liu; Cassidy Hardin; Harshal Tushar Lehri; Joao Carreira; Sara Smoot; Marcel Prasetya; Nami Akazawa; Anca Stefanoiu; Chia-Hua Ho; Anelia Angelova; Kate Lin; Min Kim; Charles Chen; Marcin Sieniek; Alice Li; Tongfei Guo; Sorin Baltateanu; Pouya Tafti; Michael Wunder; Nadav Olmert; Divyansh Shukla; Jingwei Shen; Neel Kovelamudi; Balaji Venkatraman; Seth Neel; Romal Thoppilan; Jerome Connor; Frederik Benzing; Axel Stjerngren; Golnaz Ghiasi; Alex Polozov; Joshua Howland; Theophane Weber; Justin Chiu; Ganesh Poomal Girirajan; Andreas Terzis; Pidong Wang; Fangda Li; Yoav Ben Shalom; Dinesh Tewari; Matthew Denton; Roee Aharoni; Norbert Kalb; Heri Zhao; Junlin Zhang; Angelos Filos; Matthew Rahtz; Lalit Jain; Connie Fan; Vitor Rodrigues; Ruth Wang; Richard Shin; Jacob Austin; Roman Ring; Mariella Sanchez-Vargas; Mehadi Hassen; Ido Kessler; Uri Alon; Gufeng Zhang; Wenhu Chen; Yenai Ma; Xiance Si; Le Hou; Azalia Mirhoseini; Marc Wilson; Geoff Bacon; Becca Roelofs; Lei Shu; Gautam Vasudevan; Jonas Adler; Artur Dwornik; Tayfun Terzi; Matt Lawlor; Harry Askham; Mike Bernico; Xuanyi Dong; Chris Hidey; Kevin Kilgour; Gaël Liu; Surya Bhupatiraju; Luke Leonhard; Siqi Zuo; Partha Talukdar; Qing Wei; Aliaksei Severyn; Vít Listík; Jong Lee; Aditya Tripathi; SK Park; Yossi Matias; Hao Liu; Alex Ruiz; Rajesh Jayaram; Jackson Tolins; Pierre Marcenac; Yiming Wang; Bryan Seybold; Henry Prior; Deepak Sharma; Jack Weber; Mikhail Sirotenko; Yunhsuan Sung; Dayou Du; Ellie Pavlick; Stefan Zinke; Markus Freitag; Max Dylla; Montse Gonzalez Arenas; Natan Potikha; Omer Goldman; Connie Tao; Rachita Chhaparia; Maria Voitovich; Pawan Dogra; Andrija Ražnatović; Zak Tsai; Chong You; Oleaser Johnson; George Tucker; Chenjie Gu; Jae Yoo; Maryam Majzoubi; Valentin Gabeur; Bahram Raad; Rocky Rhodes; Kashyap Kolipaka; Heidi Howard; Geta Sampemane; Benny Li; Chulayuth Asawaroengchai; Duy Nguyen; Chiyuan Zhang; Timothee Cour; Xinxin Yu; Zhao Fu; Joe Jiang; Po-Sen Huang; Gabriela Surita; Iñaki Iturrate; Yael Karov; Michael Collins; Martin Baeuml; Fabian Fuchs; Shilpa Shetty; Swaroop Ramaswamy; Sayna Ebrahimi; Qiuchen Guo; Jeremy Shar; Gabe Barth-Maron; Sravanti Addepalli; Bryan Richter; Chin-Yi Cheng; Eugénie Rives; Fei Zheng; Johannes Griesser; Nishanth Dikkala; Yoel Zeldes; Ilkin Safarli; Dipanjan Das; Himanshu Srivastava; Sadh MNM Khan; Xin Li; Aditya Pandey; Larisa Markeeva; Dan Belov; Qiqi Yan; Mikołaj Rybiński; Tao Chen; Megha Nawhal; Michael Quinn; Vineetha Govindaraj; Sarah York; Reed Roberts; Roopal Garg; Namrata Godbole; Jake Abernethy; Anil Das; Lam Nguyen Thiet; Jonathan Tompson; John Nham; Neera Vats; Ben Caine; Wesley Helmholz; Francesco Pongetti; Yeongil Ko; James An; Clara Huiyi Hu; Yu-Cheng Ling; Julia Pawar; Robert Leland; Keisuke Kinoshita; Waleed Khawaja; Marco Selvi; Eugene Ie; Danila Sinopalnikov; Lev Proleev; Nilesh Tripuraneni; Michele Bevilacqua; Seungji Lee; Clayton Sanford; Dan Suh; Dustin Tran; Jeff Dean; Simon Baumgartner; Jens Heitkaemper; Sagar Gubbi; Kristina Toutanova; Yichong Xu; Chandu Thekkath; Keran Rong; Palak Jain; Annie Xie; Yan Virin; Yang Li; Lubo Litchev; Richard Powell; Tarun Bharti; Adam Kraft; Nan Hua; Marissa Ikonomidis; Ayal Hitron; Sanjiv Kumar; Loic Matthey; Sophie Bridgers; Lauren Lax; Ishaan Malhi; Ondrej Skopek; Ashish Gupta; Jiawei Cao; Mitchelle Rasquinha; Siim Põder; Wojciech Stokowiec; Nicholas Roth; Guowang Li; Michaël Sander; Joshua Kessinger; Vihan Jain; Edward Loper; Wonpyo Park; Michal Yarom; Liqun Cheng; Guru Guruganesh; Kanishka Rao; Yan Li; Catarina Barros; Mikhail Sushkov; Chun-Sung Ferng; Rohin Shah; Ophir Aharoni; Ravin Kumar; Tim McConnell; Peiran Li; Chen Wang; Fernando Pereira; Craig Swanson; Fayaz Jamil; Yan Xiong; Anitha Vijayakumar; Prakash Shroff; Kedar Soparkar; Jindong Gu; Livio Baldini Soares; Eric Wang; Kushal Majmundar; Aurora Wei; Kai Bailey; Nora Kassner; Chizu Kawamoto; Goran Žužić; Victor Gomes; Abhirut Gupta; Michael Guzman; Ishita Dasgupta; Xinyi Bai; Zhufeng Pan; Francesco Piccinno; Hadas Natalie Vogel; Octavio Ponce; Adrian Hutter; Paul Chang; Pan-Pan Jiang; Ionel Gog; Vlad Ionescu; James Manyika; Fabian Pedregosa; Harry Ragan; Zach Behrman; Ryan Mullins; Coline Devin; Aroonalok Pyne; Swapnil Gawde; Martin Chadwick; Yiming Gu; Sasan Tavakkol; Andy Twigg; Naman Goyal; Ndidi Elue; Anna Goldie; Srinivasan Venkatachary; Hongliang Fei; Ziqiang Feng; Marvin Ritter; Isabel Leal; Sudeep Dasari; Pei Sun; Alif Raditya Rochman; Brendan O'Donoghue; Yuchen Liu; Jim Sproch; Kai Chen; Natalie Clay; Slav Petrov; Sailesh Sidhwani; Ioana Mihailescu; Alex Panagopoulos; AJ Piergiovanni; Yunfei Bai; George Powell; Deep Karkhanis; Trevor Yacovone; Petr Mitrichev; Joe Kovac; Dave Uthus; Amir Yazdanbakhsh; David Amos; Steven Zheng; Bing Zhang; Jin Miao; Bhuvana Ramabhadran; Soroush Radpour; Shantanu Thakoor; Josh Newlan; Oran Lang; Orion Jankowski; Shikhar Bharadwaj; Jean-Michel Sarr; Shereen Ashraf; Sneha Mondal; Jun Yan; Ankit Singh Rawat; Sarmishta Velury; Greg Kochanski; Tom Eccles; Franz Och; Abhanshu Sharma; Ethan Mahintorabi; Alex Gurney; Carrie Muir; Vered Cohen; Saksham Thakur; Adam Bloniarz; Asier Mujika; Alexander Pritzel; Paul Caron; Altaf Rahman; Fiona Lang; Yasumasa Onoe; Petar Sirkovic; Jay Hoover; Ying Jian; Pablo Duque; Arun Narayanan; David Soergel; Alex Haig; Loren Maggiore; Shyamal Buch; Josef Dean; Ilya Figotin; Igor Karpov; Shaleen Gupta; Denny Zhou; Muhuan Huang; Ashwin Vaswani; Christopher Semturs; Kaushik Shivakumar; Yu Watanabe; Vinodh Kumar Rajendran; Eva Lu; Yanhan Hou; Wenting Ye; Shikhar Vashishth; Nana Nti; Vytenis Sakenas; Darren Ni; Doug DeCarlo; Michael Bendersky; Sumit Bagri; Nacho Cano; Elijah Peake; Simon Tokumine; Varun Godbole; Carlos Guía; Tanya Lando; Vittorio Selo; Seher Ellis; Danny Tarlow; Daniel Gillick; Alessandro Epasto; Siddhartha Reddy Jonnalagadda; Meng Wei; Meiyan Xie; Ankur Taly; Michela Paganini; Mukund Sundararajan; Daniel Toyama; Ting Yu; Dessie Petrova; Aneesh Pappu; Rohan Agrawal; Senaka Buthpitiya; Justin Frye; Thomas Buschmann; Remi Crocker; Marco Tagliasacchi; Mengchao Wang; Da Huang; Sagi Perel; Brian Wieder; Hideto Kazawa; Weiyue Wang; Jeremy Cole; Himanshu Gupta; Ben Golan; Seojin Bang; Nitish Kulkarni; Ken Franko; Casper Liu; Doug Reid; Sid Dalmia; Jay Whang; Kevin Cen; Prasha Sundaram; Johan Ferret; Berivan Isik; Lucian Ionita; Guan Sun; Anna Shekhawat; Muqthar Mohammad; Philip Pham; Ronny Huang; Karthik Raman; Xingyi Zhou; Ross Mcilroy; Austin Myers; Sheng Peng; Jacob Scott; Paul Covington; Sofia Erell; Pratik Joshi; João Gabriel Oliveira; Natasha Noy; Tajwar Nasir; Jake Walker; Vera Axelrod; Tim Dozat; Pu Han; Chun-Te Chu; Eugene Weinstein; Anand Shukla; Shreyas Chandrakaladharan; Petra Poklukar; Bonnie Li; Ye Jin; Prem Eruvbetine; Steven Hansen; Avigail Dabush; Alon Jacovi; Samrat Phatale; Chen Zhu; Steven Baker; Mo Shomrat; Yang Xiao; Jean Pouget-Abadie; Mingyang Zhang; Fanny Wei; Yang Song; Helen King; Yiling Huang; Yun Zhu; Ruoxi Sun; Juliana Vicente Franco; Chu-Cheng Lin; Sho Arora; Hui; Li; Vivian Xia; Luke Vilnis; Mariano Schain; Kaiz Alarakyia; Laurel Prince; Aaron Phillips; Caleb Habtegebriel; Luyao Xu; Huan Gui; Santiago Ontanon; Lora Aroyo; Karan Gill; Peggy Lu; Yash Katariya; Dhruv Madeka; Shankar Krishnan; Shubha Srinivas Raghvendra; James Freedman; Yi Tay; Gaurav Menghani; Peter Choy; Nishita Shetty; Dan Abolafia; Doron Kukliansky; Edward Chou; Jared Lichtarge; Ken Burke; Ben Coleman; Dee Guo; Larry Jin; Indro Bhattacharya; Victoria Langston; Yiming Li; Suyog Kotecha; Alex Yakubovich; Xinyun Chen; Petre Petrov; Tolly Powell; Yanzhang He; Corbin Quick; Kanav Garg; Dawsen Hwang; Yang Lu; Srinadh Bhojanapalli; Kristian Kjems; Ramin Mehran; Aaron Archer; Hado van Hasselt; Ashwin Balakrishna; JK Kearns; Meiqi Guo; Jason Riesa; Mikita Sazanovich; Xu Gao; Chris Sauer; Chengrun Yang; XiangHai Sheng; Thomas Jimma; Wouter Van Gansbeke; Vitaly Nikolaev; Wei Wei; Katie Millican; Ruizhe Zhao; Justin Snyder; Levent Bolelli; Maura O'Brien; Shawn Xu; Fei Xia; Wentao Yuan; Arvind Neelakantan; David Barker; Sachin Yadav; Hannah Kirkwood; Farooq Ahmad; Joel Wee; Jordan Grimstad; Boyu Wang; Matthew Wiethoff; Shane Settle; Miaosen Wang; Charles Blundell; Jingjing Chen; Chris Duvarney; Grace Hu; Olaf Ronneberger; Alex Lee; Yuanzhen Li; Abhishek Chakladar; Alena Butryna; Georgios Evangelopoulos; Guillaume Desjardins; Jonni Kanerva; Henry Wang; Averi Nowak; Nick Li; Alyssa Loo; Art Khurshudov; Laurent El Shafey; Nagabhushan Baddi; Karel Lenc; Yasaman Razeghi; Tom Lieber; Amer Sinha; Xiao Ma; Yao Su; James Huang; Asahi Ushio; Hanna Klimczak-Plucińska; Kareem Mohamed; JD Chen; Simon Osindero; Stav Ginzburg; Lampros Lamprou; Vasilisa Bashlovkina; Duc-Hieu Tran; Ali Khodaei; Ankit Anand; Yixian Di; Ramy Eskander; Manish Reddy Vuyyuru; Jasmine Liu; Aishwarya Kamath; Roman Goldenberg; Mathias Bellaiche; Juliette Pluto; Bill Rosgen; Hassan Mansoor; William Wong; Suhas Ganesh; Eric Bailey; Scott Baird; Dan Deutsch; Jinoo Baek; Xuhui Jia; Chansoo Lee; Abe Friesen; Nathaniel Braun; Kate Lee; Amayika Panda; Steven M. Hernandez; Duncan Williams; Jianqiao Liu; Ethan Liang; Arnaud Autef; Emily Pitler; Deepali Jain; Phoebe Kirk; Oskar Bunyan; Jaume Sanchez Elias; Tongxin Yin; Machel Reid; Aedan Pope; Nikita Putikhin; Bidisha Samanta; Sergio Guadarrama; Dahun Kim; Simon Rowe; Marcella Valentine; Geng Yan; Alex Salcianu; David Silver; Gan Song; Richa Singh; Shuai Ye; Hannah DeBalsi; Majd Al Merey; Eran Ofek; Albert Webson; Shibl Mourad; Ashwin Kakarla; Silvio Lattanzi; Nick Roy; Evgeny Sluzhaev; Christina Butterfield; Alessio Tonioni; Nathan Waters; Sudhindra Kopalle; Jason Chase; James Cohan; Girish Ramchandra Rao; Robert Berry; Michael Voznesensky; Shuguang Hu; Kristen Chiafullo; Sharat Chikkerur; George Scrivener; Ivy Zheng; Jeremy Wiesner; Wolfgang Macherey; Timothy Lillicrap; Fei Liu; Brian Walker; David Welling; Elinor Davies; Yangsibo Huang; Lijie Ren; Nir Shabat; Alessandro Agostini; Mariko Iinuma; Dustin Zelle; Rohit Sathyanarayana; Andrea D'olimpio; Morgan Redshaw; Matt Ginsberg; Ashwin Murthy; Mark Geller; Tatiana Matejovicova; Ayan Chakrabarti; Ryan Julian; Christine Chan; Qiong Hu; Daniel Jarrett; Manu Agarwal; Jeshwanth Challagundla; Tao Li; Sandeep Tata; Wen Ding; Maya Meng; Zhuyun Dai; Giulia Vezzani; Shefali Garg; Jannis Bulian; Mary Jasarevic; Honglong Cai; Harish Rajamani; Adam Santoro; Florian Hartmann; Chen Liang; Bartek Perz; Apoorv Jindal; Fan Bu; Sungyong Seo; Ryan Poplin; Adrian Goedeckemeyer; Badih Ghazi; Nikhil Khadke; Leon Liu; Kevin Mather; Mingda Zhang; Ali Shah; Alex Chen; Jinliang Wei; Keshav Shivam; Yuan Cao; Donghyun Cho; Angelo Scorza Scarpati; Michael Moffitt; Clara Barbu; Ivan Jurin; Ming-Wei Chang; Hongbin Liu; Hao Zheng; Shachi Dave; Christine Kaeser-Chen; Xiaobin Yu; Alvin Abdagic; Lucas Gonzalez; Yanping Huang; Peilin Zhong; Cordelia Schmid; Bryce Petrini; Alex Wertheim; Jifan Zhu; Hoang Nguyen; Kaiyang Ji; Yanqi Zhou; Tao Zhou; Fangxiaoyu Feng; Regev Cohen; David Rim; Shubham Milind Phal; Petko Georgiev; Ariel Brand; Yue Ma; Wei Li; Somit Gupta; Chao Wang; Pavel Dubov; Jean Tarbouriech; Kingshuk Majumder; Huijian Li; Norman Rink; Apurv Suman; Yang Guo; Yinghao Sun; Arun Nair; Xiaowei Xu; Mohamed Elhawaty; Rodrigo Cabrera; Guangxing Han; Julian Eisenschlos; Junwen Bai; Yuqi Li; Yamini Bansal; Thibault Sellam; Mina Khan; Hung Nguyen; Justin Mao-Jones; Nikos Parotsidis; Jake Marcus; Cindy Fan; Roland Zimmermann; Yony Kochinski; Laura Graesser; Feryal Behbahani; Alvaro Caceres; Michael Riley; Patrick Kane; Sandra Lefdal; Rob Willoughby; Paul Vicol; Lun Wang; Shujian Zhang; Ashleah Gill; Yu Liang; Gautam Prasad; Soroosh Mariooryad; Mehran Kazemi; Zifeng Wang; Kritika Muralidharan; Paul Voigtlaender; Jeffrey Zhao; Huanjie Zhou; Nina D'Souza; Aditi Mavalankar; Séb Arnold; Nick Young; Obaid Sarvana; Chace Lee; Milad Nasr; Tingting Zou; Seokhwan Kim; Lukas Haas; Kaushal Patel; Neslihan Bulut; David Parkinson; Courtney Biles; Dmitry Kalashnikov; Chi Ming To; Aviral Kumar; Jessica Austin; Alex Greve; Lei Zhang; Megha Goel; Yeqing Li; Sergey Yaroshenko; Max Chang; Abhishek Jindal; Geoff Clark; Hagai Taitelbaum; Dale Johnson; Ofir Roval; Jeongwoo Ko; Anhad Mohananey; Christian Schuler; Shenil Dodhia; Ruichao Li; Kazuki Osawa; Claire Cui; Peng Xu; Rushin Shah; Tao Huang; Ela Gruzewska; Nathan Clement; Mudit Verma; Olcan Sercinoglu; Hai Qian; Viral Shah; Masa Yamaguchi; Abhinit Modi; Takahiro Kosakai; Thomas Strohmann; Junhao Zeng; Beliz Gunel; Jun Qian; Austin Tarango; Krzysztof Jastrzębski; Robert David; Jyn Shan; Parker Schuh; Kunal Lad; Willi Gierke; Mukundan Madhavan; Xinyi Chen; Mark Kurzeja; Rebeca Santamaria-Fernandez; Dawn Chen; Alexandra Cordell; Yuri Chervonyi; Frankie Garcia; Nithish Kannen; Vincent Perot; Nan Ding; Shlomi Cohen-Ganor; Victor Lavrenko; Junru Wu; Georgie Evans; Cicero Nogueira dos Santos; Madhavi Sewak; Ashley Brown; Andrew Hard; Joan Puigcerver; Zeyu Zheng; Yizhong Liang; Evgeny Gladchenko; Reeve Ingle; Uri First; Pierre Sermanet; Charlotte Magister; Mihajlo Velimirović; Sashank Reddi; Susanna Ricco; Eirikur Agustsson; Hartwig Adam; Nir Levine; David Gaddy; Dan Holtmann-Rice; Xuanhui Wang; Ashutosh Sathe; Abhijit Guha Roy; Blaž Bratanič; Alen Carin; Harsh Mehta; Silvano Bonacina; Nicola De Cao; Mara Finkelstein; Verena Rieser; Xinyi Wu; Florent Altché; Dylan Scandinaro; Li Li; Nino Vieillard; Nikhil Sethi; Garrett Tanzer; Zhi Xing; Shibo Wang; Parul Bhatia; Gui Citovsky; Thomas Anthony; Sharon Lin; Tianze Shi; Shoshana Jakobovits; Gena Gibson; Raj Apte; Lisa Lee; Mingqing Chen; Arunkumar Byravan; Petros Maniatis; Kellie Webster; Andrew Dai; Pu-Chin Chen; Jiaqi Pan; Asya Fadeeva; Zach Gleicher; Thang Luong; Niket Kumar Bhumihar
>
> **备注:** 72 pages, 17 figures
>
> **摘要:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.
>
---
#### [replaced 021] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.05195v3](http://arxiv.org/pdf/2411.05195v3)**

> **作者:** Siting Li; Pang Wei Koh; Simon Shaolei Du
>
> **备注:** ACL 2025; 19 pages, 3 figures
>
> **摘要:** Recent research has shown that CLIP models struggle with visual reasoning tasks that require grounding compositionality, understanding spatial relationships, or capturing fine-grained details. One natural hypothesis is that the CLIP vision encoder does not embed essential information for these tasks. However, we find that this is not always the case: The encoder gathers query-relevant visual information, while CLIP fails to extract it. In particular, we show that another branch of Vision-Language Models (VLMs), Generative Multimodal Large Language Models (MLLMs), achieve significantly higher accuracy than CLIP in many of these tasks using the same vision encoder and weights, indicating that these Generative MLLMs perceive more -- as they extract and utilize visual information more effectively. We conduct a series of controlled experiments and reveal that their success is attributed to multiple key design choices, including patch tokens, position embeddings, and prompt-based weighting. On the other hand, enhancing the training data alone or applying a stronger text encoder does not suffice to solve the task, and additional text tokens offer little benefit. Interestingly, we find that fine-grained visual reasoning is not exclusive to generative models trained by an autoregressive loss: When converted into CLIP-like encoders by contrastive finetuning, these MLLMs still outperform CLIP under the same cosine similarity-based evaluation protocol. Our study highlights the importance of VLM architectural choices and suggests directions for improving the performance of CLIP-like contrastive VLMs.
>
---
#### [replaced 022] Hierarchical Budget Policy Optimization for Adaptive Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15844v2](http://arxiv.org/pdf/2507.15844v2)**

> **作者:** Shangke Lyu; Linjuan Wu; Yuchen Yan; Xingyu Wu; Hao Li; Yongliang Shen; Peisheng Jiang; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Code: https://github.com/zju-real/hbpo Project Page:https://zju-real.github.io/hbpo/
>
> **摘要:** Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet exhibit significant computational inefficiency by applying uniform reasoning strategies regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. HBPO addresses the fundamental challenge of exploration space collapse in efficiency-oriented training, where penalties on long output length systematically bias models away from necessary long reasoning paths. Through hierarchical budget exploration, our approach partitions rollout samples into multiple subgroups with distinct token budgets, aiming to enable efficient resource allocation while preventing degradation of capability. We introduce differentiated reward mechanisms that create budget-aware incentives aligned with the complexity of the problem, allowing models to discover natural correspondences between task requirements and computational effort. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Unlike existing methods that impose external constraints or rely on discrete mode selection, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity.
>
---
#### [replaced 023] Self-Correcting Code Generation Using Small Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23060v2](http://arxiv.org/pdf/2505.23060v2)**

> **作者:** Jeonghun Cho; Deokhyung Kang; Hyounghun Kim; Gary Geunbae Lee
>
> **摘要:** Self-correction has demonstrated potential in code generation by allowing language models to revise and improve their outputs through successive refinement. Recent studies have explored prompting-based strategies that incorporate verification or feedback loops using proprietary models, as well as training-based methods that leverage their strong reasoning capabilities. However, whether smaller models possess the capacity to effectively guide their outputs through self-reflection remains unexplored. Our findings reveal that smaller models struggle to exhibit reflective revision behavior across both self-correction paradigms. In response, we introduce CoCoS, an approach designed to enhance the ability of small language models for multi-turn code correction. Specifically, we propose an online reinforcement learning objective that trains the model to confidently maintain correct outputs while progressively correcting incorrect outputs as turns proceed. Our approach features an accumulated reward function that aggregates rewards across the entire trajectory and a fine-grained reward better suited to multi-turn correction scenarios. This facilitates the model in enhancing initial response quality while achieving substantial improvements through self-correction. With 1B-scale models, CoCoS achieves improvements of 35.8% on the MBPP and 27.7% on HumanEval compared to the baselines.
>
---
#### [replaced 024] Routine: A Structural Planning Framework for LLM Agent System in Enterprise
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14447v2](http://arxiv.org/pdf/2507.14447v2)**

> **作者:** Guancheng Zeng; Xueyi Chen; Jiawang Hu; Shaohua Qi; Yaxuan Mao; Zhantao Wang; Yifan Nie; Shuang Li; Qiuyang Feng; Pengxu Qiu; Yujia Wang; Wenqiang Han; Linyan Huang; Gang Li; Jingjing Mo; Haowen Hu
>
> **备注:** 26 pages, 8 figures, 5 tables
>
> **摘要:** The deployment of agent systems in an enterprise environment is often hindered by several challenges: common models lack domain-specific process knowledge, leading to disorganized plans, missing key tools, and poor execution stability. To address this, this paper introduces Routine, a multi-step agent planning framework designed with a clear structure, explicit instructions, and seamless parameter passing to guide the agent's execution module in performing multi-step tool-calling tasks with high stability. In evaluations conducted within a real-world enterprise scenario, Routine significantly increases the execution accuracy in model tool calls, increasing the performance of GPT-4o from 41.1% to 96.3%, and Qwen3-14B from 32.6% to 83.3%. We further constructed a Routine-following training dataset and fine-tuned Qwen3-14B, resulting in an accuracy increase to 88.2% on scenario-specific evaluations, indicating improved adherence to execution plans. In addition, we employed Routine-based distillation to create a scenario-specific, multi-step tool-calling dataset. Fine-tuning on this distilled dataset raised the model's accuracy to 95.5%, approaching GPT-4o's performance. These results highlight Routine's effectiveness in distilling domain-specific tool-usage patterns and enhancing model adaptability to new scenarios. Our experimental results demonstrate that Routine provides a practical and accessible approach to building stable agent workflows, accelerating the deployment and adoption of agent systems in enterprise environments, and advancing the technical vision of AI for Process.
>
---
#### [replaced 025] DCG-SQL: Enhancing In-Context Learning for Text-to-SQL with Deep Contextual Schema Link Graph
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19956v2](http://arxiv.org/pdf/2505.19956v2)**

> **作者:** Jihyung Lee; Jin-Seop Lee; Jaehoon Lee; YunSeok Choi; Jee-Hyong Lee
>
> **摘要:** Text-to-SQL, which translates a natural language question into an SQL query, has advanced with in-context learning of Large Language Models (LLMs). However, existing methods show little improvement in performance compared to randomly chosen demonstrations, and significant performance drops when smaller LLMs (e.g., Llama 3.1-8B) are used. This indicates that these methods heavily rely on the intrinsic capabilities of hyper-scaled LLMs, rather than effectively retrieving useful demonstrations. In this paper, we propose a novel approach for effectively retrieving demonstrations and generating SQL queries. We construct a Deep Contextual Schema Link Graph, which contains key information and semantic relationship between a question and its database schema items. This graph-based structure enables effective representation of Text-to-SQL samples and retrieval of useful demonstrations for in-context learning. Experimental results on the Spider benchmark demonstrate the effectiveness of our approach, showing consistent improvements in SQL generation performance and efficiency across both hyper-scaled LLMs and small LLMs. The code is available at https://github.com/jjklle/DCG-SQL}{https://github.com/jjklle/DCG-SQL.
>
---
#### [replaced 026] A Multi-granularity Concept Sparse Activation and Hierarchical Knowledge Graph Fusion Framework for Rare Disease Diagnosis
- **分类: cs.AI; cs.CL; 68T50, 92C50, 68T05; J.3; I.2.7; H.3.3; I.2.1**

- **链接: [http://arxiv.org/pdf/2507.08529v2](http://arxiv.org/pdf/2507.08529v2)**

> **作者:** Mingda Zhang; Na Zhao; Jianglong Qin; Guoyu Ye; Ruixiang Tang
>
> **备注:** 12 pages,3 figures
>
> **摘要:** Rare disease diagnosis remains challenging for medical large language models due to insufficient knowledge representation, limited concept understanding, and constrained clinical reasoning. We propose a framework combining multi-granularity sparse activation with hierarchical knowledge graphs. Our approach employs four complementary matching algorithms with diversity control and a five-level fallback strategy for precise concept activation. A three-layer knowledge graph (taxonomy, clinical features, instances) provides structured, up-to-date context. Experiments on the BioASQ rare disease dataset demonstrate significant improvements: BLEU scores increased by up to 0.13, ROUGE by up to 0.10, and diagnostic accuracy by up to 0.25, with the best model achieving 0.92 accuracy--surpassing the 0.90 clinical threshold. Expert evaluation confirms enhancements in information quality, reasoning, and professional expression. Our framework shows promise in reducing the diagnostic odyssey for rare disease patients.
>
---
#### [replaced 027] Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13618v2](http://arxiv.org/pdf/2507.13618v2)**

> **作者:** Shanbo Cheng; Yu Bao; Qian Cao; Luyang Huang; Liyan Kang; Zhicheng Liu; Yu Lu; Wenhao Zhu; Jingwen Chen; Zhichao Huang; Tao Li; Yifu Li; Huiying Lin; Sitong Liu; Ningxin Peng; Shuaijie She; Lu Xu; Nuo Xu; Sen Yang; Runsheng Yu; Yiming Yu; Liehao Zou; Hang Li; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **摘要:** Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
>
---
#### [replaced 028] Erasing Conceptual Knowledge from Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02760v3](http://arxiv.org/pdf/2410.02760v3)**

> **作者:** Rohit Gandikota; Sheridan Feucht; Samuel Marks; David Bau
>
> **备注:** Project Page: https://elm.baulab.info
>
> **摘要:** In this work, we introduce Erasure of Language Memory (ELM), a principled approach to concept-level unlearning that operates by matching distributions defined by the model's own introspective classification capabilities. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the language model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative evaluation reveals that ELM-modified models achieve near-random performance on assessments targeting erased concepts, while simultaneously preserving generation coherence, maintaining benchmark performance on unrelated tasks, and exhibiting strong robustness to adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info
>
---
#### [replaced 029] SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10706v2](http://arxiv.org/pdf/2503.10706v2)**

> **作者:** Pierre Sermanet; Anirudha Majumdar; Vikas Sindhwani
>
> **备注:** Minor improvements over previous version
>
> **摘要:** Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a state-of-the-art LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via an amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in Sci-Fi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers generated through a novel LLM-introspection process, in addition to a smaller human-labeled evaluation set.
>
---
#### [replaced 030] Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06229v4](http://arxiv.org/pdf/2507.06229v4)**

> **作者:** Xiangru Tang; Tianrui Qin; Tianhao Peng; Ziyang Zhou; Daniel Shao; Tingting Du; Xinming Wei; Peng Xia; Fang Wu; He Zhu; Ge Zhang; Jiaheng Liu; Xingyao Wang; Sirui Hong; Chenglin Wu; Hao Cheng; Chi Wang; Wangchunshu Zhou
>
> **摘要:** Current AI agents cannot effectively learn from each other's problem-solving experiences or use past successes to guide self-reflection and error correction in new tasks. We introduce Agent KB, a shared knowledge base that captures both high-level problem-solving strategies and detailed execution lessons, enabling knowledge transfer across agent frameworks. Agent KB implements a novel teacher-student dual-phase retrieval mechanism where student agents retrieve workflow-level patterns for strategic guidance while teacher agents identify execution-level patterns for refinement. This hierarchical approach enables agents to break out of limited reasoning pathways by incorporating diverse strategies from external sources. Evaluations on the GAIA benchmark demonstrate substantial performance gains, with Agent KB improving success rates by up to 6.06 percentage points overall under pass@1. For SWE-bench code repair tasks, our system significantly improved resolution rates, with o3-mini achieving an 8.67 percentage point gain (23 percent to 31.67 percent) in pass@1. Our ablation studies demonstrate that the refinement module proves most critical, with its removal causing a 3.85% drop on challenging Level 3 tasks, highlighting that effective knowledge transfer necessitates both strategic guidance and execution-level refinement.
>
---
#### [replaced 031] Hear Your Code Fail, Voice-Assisted Debugging for Python
- **分类: cs.PL; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15007v2](http://arxiv.org/pdf/2507.15007v2)**

> **作者:** Sayed Mahbub Hasan Amiri; Md. Mainul Islam; Mohammad Shakhawat Hossen; Sayed Majhab Hasan Amiri; Mohammad Shawkat Ali Mamun; Sk. Humaun Kabir; Naznin Akter
>
> **备注:** 35 pages, 20 figures
>
> **摘要:** This research introduces an innovative voice-assisted debugging plugin for Python that transforms silent runtime errors into actionable audible diagnostics. By implementing a global exception hook architecture with pyttsx3 text-to-speech conversion and Tkinter-based GUI visualization, the solution delivers multimodal error feedback through parallel auditory and visual channels. Empirical evaluation demonstrates 37% reduced cognitive load (p<0.01, n=50) compared to traditional stack-trace debugging, while enabling 78% faster error identification through vocalized exception classification and contextualization. The system achieves sub-1.2 second voice latency with under 18% CPU overhead during exception handling, vocalizing error types and consequences while displaying interactive tracebacks with documentation deep links. Criteria validate compatibility across Python 3.7+ environments on Windows, macOS, and Linux platforms. Needing only two lines of integration code, the plugin significantly boosts availability for aesthetically impaired designers and supports multitasking workflows through hands-free error medical diagnosis. Educational applications show particular promise, with pilot studies indicating 45% faster debugging skill acquisition among novice programmers. Future development will incorporate GPT-based repair suggestions and real-time multilingual translation to further advance auditory debugging paradigms. The solution represents a fundamental shift toward human-centric error diagnostics, bridging critical gaps in programming accessibility while establishing new standards for cognitive efficiency in software development workflows.
>
---
#### [replaced 032] Human Empathy as Encoder: AI-Assisted Depression Assessment in Special Education
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23631v2](http://arxiv.org/pdf/2505.23631v2)**

> **作者:** Boning Zhao
>
> **备注:** 7 pages, 6 figures. Under review
>
> **摘要:** Assessing student depression in sensitive environments like special education is challenging. Standardized questionnaires may not fully reflect students' true situations. Furthermore, automated methods often falter with rich student narratives, lacking the crucial, individualized insights stemming from teachers' empathetic connections with students. Existing methods often fail to address this ambiguity or effectively integrate educator understanding. To address these limitations by fostering a synergistic human-AI collaboration, this paper introduces Human Empathy as Encoder (HEAE), a novel, human-centered AI framework for transparent and socially responsible depression severity assessment. Our approach uniquely integrates student narrative text with a teacher-derived, 9-dimensional "Empathy Vector" (EV), its dimensions guided by the PHQ-9 framework,to explicitly translate tacit empathetic insight into a structured AI input enhancing rather than replacing human judgment. Rigorous experiments optimized the multimodal fusion, text representation, and classification architecture, achieving 82.74% accuracy for 7-level severity classification. This work demonstrates a path toward more responsible and ethical affective computing by structurally embedding human empathy
>
---
#### [replaced 033] SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07939v2](http://arxiv.org/pdf/2507.07939v2)**

> **作者:** Guoxin Zang; Xue Li; Donglin Di; Lanshun Nie; Dechen Zhan; Yang Song; Lei Fan
>
> **备注:** Accepted by ACMMM2025
>
> **摘要:** While Vision-Language Models (VLMs) have shown promising progress in general multimodal tasks, they often struggle in industrial anomaly detection and reasoning, particularly in delivering interpretable explanations and generalizing to unseen categories. This limitation stems from the inherently domain-specific nature of anomaly detection, which hinders the applicability of existing VLMs in industrial scenarios that require precise, structured, and context-aware analysis. To address these challenges, we propose SAGE, a VLM-based framework that enhances anomaly reasoning through Self-Guided Fact Enhancement (SFE) and Entropy-aware Direct Preference Optimization (E-DPO). SFE integrates domain-specific knowledge into visual reasoning via fact extraction and fusion, while E-DPO aligns model outputs with expert preferences using entropy-aware optimization. Additionally, we introduce AD-PL, a preference-optimized dataset tailored for industrial anomaly reasoning, consisting of 28,415 question-answering instances with expert-ranked responses. To evaluate anomaly reasoning models, we develop Multiscale Logical Evaluation (MLE), a quantitative framework analyzing model logic and consistency. SAGE demonstrates superior performance on industrial anomaly datasets under zero-shot and one-shot settings. The code, model and dataset are available at https://github.com/amoreZgx1n/SAGE.
>
---
#### [replaced 034] Modeling the Sacred: Considerations when Using Religious Texts in Natural Language Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.14740v3](http://arxiv.org/pdf/2404.14740v3)**

> **作者:** Ben Hutchinson
>
> **备注:** Findings of NAACL2024
>
> **摘要:** This position paper concerns the use of religious texts in Natural Language Processing (NLP), which is of special interest to the Ethics of NLP. Religious texts are expressions of culturally important values, and machine learned models have a propensity to reproduce cultural values encoded in their training data. Furthermore, translations of religious texts are frequently used by NLP researchers when language data is scarce. This repurposes the translations from their original uses and motivations, which often involve attracting new followers. This paper argues that NLP's use of such texts raises considerations that go beyond model biases, including data provenance, cultural contexts, and their use in proselytism. We argue for more consideration of researcher positionality, and of the perspectives of marginalized linguistic and religious communities.
>
---
#### [replaced 035] LLMs syntactically adapt their language use to their conversational partner
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07457v2](http://arxiv.org/pdf/2503.07457v2)**

> **作者:** Florian Kandra; Vera Demberg; Alexander Koller
>
> **备注:** 5 pages, 1 table, 3 figures, accepted at ACL (main conference) 2025
>
> **摘要:** It has been frequently observed that human speakers align their language use with each other during conversations. In this paper, we study empirically whether large language models (LLMs) exhibit the same behavior of conversational adaptation. We construct a corpus of conversations between LLMs and find that two LLM agents end up making more similar syntactic choices as conversations go on, confirming that modern LLMs adapt their language use to their conversational partners in at least a rudimentary way.
>
---
#### [replaced 036] InternAgent: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16938v3](http://arxiv.org/pdf/2505.16938v3)**

> **作者:** InternAgent Team; Bo Zhang; Shiyang Feng; Xiangchao Yan; Jiakang Yuan; Runmin Ma; Yusong Hu; Zhiyin Yu; Xiaohan He; Songtao Huang; Shaowei Hou; Zheng Nie; Zhilong Wang; Jinyao Liu; Tianshuo Peng; Peng Ye; Dongzhan Zhou; Shufei Zhang; Xiaosong Wang; Yilan Zhang; Meng Li; Zhongying Tu; Xiangyu Yue; Wangli Ouyang; Bowen Zhou; Lei Bai
>
> **备注:** Code: https://github.com/Alpha-Innovator/InternAgent, HomePage: https://alpha-innovator.github.io/InternAgent-project-page
>
> **摘要:** Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce InternAgent, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. InternAgent highlights three key advantages: 1) Scalability: InternAgent has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: InternAgent provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: InternAgent has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.65 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours.
>
---
#### [replaced 037] Atomic Calibration of LLMs in Long-Form Generations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.13246v2](http://arxiv.org/pdf/2410.13246v2)**

> **作者:** Caiqi Zhang; Ruihan Yang; Zhisong Zhang; Xinting Huang; Sen Yang; Dong Yu; Nigel Collier
>
> **备注:** ACL 2025 KnowFM Oral
>
> **摘要:** Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, which estimates the underlying uncertainty of model predictions, is essential to enhance the LLMs' trustworthiness. Existing research on LLM calibration has primarily focused on short-form tasks, providing a single confidence score at the response level (macro calibration). However, this approach is insufficient for long-form generations, where responses often contain more complex statements and may include both accurate and inaccurate information. Therefore, we introduce atomic calibration, a novel approach that evaluates factuality calibration at a fine-grained level by breaking down long responses into atomic claims. We classify confidence elicitation methods into discriminative and generative types and demonstrate that their combination can enhance calibration. Our extensive experiments on various LLMs and datasets show that atomic calibration is well-suited for long-form generation and can also improve macro calibration results. Additionally, atomic calibration reveals insightful patterns in LLM confidence throughout the generation process.
>
---
#### [replaced 038] Data Processing for the OpenGPT-X Model Family
- **分类: cs.CL; H.3.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2410.08800v3](http://arxiv.org/pdf/2410.08800v3)**

> **作者:** Nicolo' Brandizzi; Hammam Abdelwahab; Anirban Bhowmick; Lennard Helmer; Benny Jörg Stein; Pavel Denisov; Qasid Saleem; Michael Fromm; Mehdi Ali; Richard Rutmann; Farzad Naderi; Mohamad Saif Agy; Alexander Schwirjow; Fabian Küch; Luzian Hahn; Malte Ostendorff; Pedro Ortiz Suarez; Georg Rehm; Dennis Wegener; Nicolas Flores-Herr; Joachim Köhler; Johannes Leveling
>
> **摘要:** This paper presents a comprehensive overview of the data preparation pipeline developed for the OpenGPT-X project, a large-scale initiative aimed at creating open and high-performance multilingual large language models (LLMs). The project goal is to deliver models that cover all major European languages, with a particular focus on real-world applications within the European Union. We explain all data processing steps, starting with the data selection and requirement definition to the preparation of the final filtered data. We distinguish between curated data and web data, as each of these categories is handled by distinct pipelines, with curated data undergoing minimal filtering and web data requiring extensive filtering and deduplication. This distinction guided the development of specialized algorithmic solutions for both pipelines. In addition to describing the processing methodologies, we provide an in-depth analysis of the datasets, increasing transparency and alignment with European data regulations. Finally, we share key insights and challenges faced during the project, offering recommendations for future endeavors in large-scale multilingual data preparation for LLMs.
>
---
#### [replaced 039] X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14430v2](http://arxiv.org/pdf/2507.14430v2)**

> **作者:** Xiaolin Yan; Yangxing Liu; Jiazhang Zheng; Chi Liu; Mingyu Du; Caisheng Chen; Haoyang Liu; Ming Ding; Yuan Li; Qiuping Liao; Linfeng Li; Zhili Mei; Siyu Wan; Li Li; Ruyi Zhong; Jiangling Yu; Xule Liu; Huihui Hu; Jiameng Yue; Ruohui Cheng; Qi Yang; Liangqing Wu; Ke Zhu; Chi Zhang; Chufei Jing; Yifan Zhou; Yan Liang; Dongdong Li; Zhaohui Wang; Bin Zhao; Mingzhou Wu; Mingzhong Zhou; Peng Du; Zuomin Liao; Chao Dai; Pengfei Liang; Xiaoguang Zhu; Yu Zhang; Yu Gu; Kun Pan; Yuan Wu; Yanqing Guan; Shaojing Wu; Zikang Feng; Xianze Ma; Peishan Cheng; Wenjuan Jiang; Jing Ba; Huihao Yu; Zeping Hu; Yuan Xu; Zhiwei Liu; He Wang; Zhenguo Lin; Ming Liu; Yanhong Meng
>
> **备注:** Technical Report
>
> **摘要:** Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry.
>
---
#### [replaced 040] Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09279v3](http://arxiv.org/pdf/2507.09279v3)**

> **作者:** Anita Kriz; Elizabeth Laura Janes; Xing Shen; Tal Arbel
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at https://github.com/xingbpshen/prompt4trust.
>
---
#### [replaced 041] Risks of AI Scientists: Prioritizing Safeguarding Over Autonomy
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.04247v5](http://arxiv.org/pdf/2402.04247v5)**

> **作者:** Xiangru Tang; Qiao Jin; Kunlun Zhu; Tongxin Yuan; Yichi Zhang; Wangchunshu Zhou; Meng Qu; Yilun Zhao; Jian Tang; Zhuosheng Zhang; Arman Cohan; Zhiyong Lu; Mark Gerstein
>
> **摘要:** AI scientists powered by large language models have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, these agents also introduce novel vulnerabilities that require careful consideration for safety. However, there has been limited comprehensive exploration of these vulnerabilities. This perspective examines vulnerabilities in AI scientists, shedding light on potential risks associated with their misuse, and emphasizing the need for safety measures. We begin by providing an overview of the potential risks inherent to AI scientists, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we explore the underlying causes of these vulnerabilities and provide a scoping review of the limited existing works. Based on our analysis, we propose a triadic framework involving human regulation, agent alignment, and an understanding of environmental feedback (agent regulation) to mitigate these identified risks. Furthermore, we highlight the limitations and challenges associated with safeguarding AI scientists and advocate for the development of improved models, robust benchmarks, and comprehensive regulations.
>
---
#### [replaced 042] Reasoning Does Not Necessarily Improve Role-Playing Ability
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16940v2](http://arxiv.org/pdf/2502.16940v2)**

> **作者:** Xiachong Feng; Longxu Dou; Lingpeng Kong
>
> **摘要:** The application of role-playing large language models (LLMs) is rapidly expanding in both academic and commercial domains, driving an increasing demand for high-precision role-playing models. Simultaneously, the rapid advancement of reasoning techniques has continuously pushed the performance boundaries of LLMs. This intersection of practical role-playing demands and evolving reasoning capabilities raises an important research question: "Can reasoning techniques enhance the role-playing capabilities of LLMs?" To address this, we conduct a comprehensive study using 6 role-playing benchmarks, 24 LLMs, and 3 distinct role-playing strategies, comparing the effectiveness of direct zero-shot role-playing, role-playing with Chain-of-Thought (CoT), and role-playing using reasoning-optimized LLMs. Our findings reveal that CoT may reduce role-playing performance, reasoning-optimized LLMs are unsuitable for role-playing, reasoning ability disrupts the role-playing scaling law, large models still lack proficiency in advanced role-playing, and Chinese role-playing performance surpasses English role-playing performance. Furthermore, based on extensive experimental results, we propose two promising future research directions: Role-aware CoT for improving role-playing LLMs and Reinforcement Learning for role-playing LLMs, aiming to enhance the adaptability, consistency, and effectiveness of role-playing LLMs for both research and real-world applications.
>
---
#### [replaced 043] Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19951v5](http://arxiv.org/pdf/2411.19951v5)**

> **作者:** Shukang Yin; Chaoyou Fu; Sirui Zhao; Chunjiang Ge; Yan Yang; Yuhan Dai; Yongdong Luo; Tong Xu; Caifeng Shan; Enhong Chen
>
> **备注:** Project page: https://github.com/VITA-MLLM/Sparrow
>
> **摘要:** Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
>
---
#### [replaced 044] Synthetic Data Generation Using Large Language Models: Advances in Text and Code
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14023v2](http://arxiv.org/pdf/2503.14023v2)**

> **作者:** Mihai Nadas; Laura Diosan; Andreea Tomescu
>
> **备注:** 24 pages, 6 tables, 1 figure, 64 references
>
> **摘要:** This survey reviews how large language models (LLMs) are transforming synthetic training data generation in both natural language and code domains. By producing artificial but task-relevant examples, these models can significantly augment or even substitute for real-world datasets, particularly in scenarios where labeled data is scarce, expensive, or sensitive. This paper surveys recent advances in leveraging LLMs to create synthetic text and code, highlighting key techniques such as prompt-based generation, retrieval-augmented pipelines, and iterative self-refinement. We examine how these methods can enrich low-resource tasks (e.g., classification, question answering) and facilitate code-centric applications (e.g., instruction tuning, code translation, bug repair) through automated verification of functional correctness. Alongside potential benefits - cost-effectiveness, broad coverage, and controllable diversity - we discuss the accompanying challenges, including factual inaccuracies in generated text, insufficient stylistic or distributional realism, and risks of bias amplification. Proposed mitigation strategies range from filtering and weighting synthetic outputs to reinforcement learning with execution feedback in code domains. We conclude by outlining open research directions, such as automated prompt engineering, cross-modal data synthesis, and robust evaluation frameworks, underscoring the growing importance of LLM-generated synthetic data in accelerating AI development while emphasizing ethical and quality safeguards.
>
---
#### [replaced 045] Supernova: Achieving More with Less in Transformer Architectures
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15773v2](http://arxiv.org/pdf/2507.15773v2)**

> **作者:** Andrei-Valentin Tanase; Elena Pelican
>
> **摘要:** We present Supernova, a 650M-parameter decoder-only transformer that demonstrates how careful architectural design and tokenization innovation can achieve the performance of larger models while maintaining computational efficiency. Our architecture combines Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA) with a 3:1 compression ratio, RMSNorm for computational efficiency, and SwiGLU activation functions. A critical innovation is our custom 128,000-vocabulary byte-level BPE tokenizer, which achieves state-of-the-art compression performance. Through detailed analysis, we show that Supernova achieves 90% of the performance of 1B-parameter models while using 35% fewer parameters and requiring only 100B training tokens--an order of magnitude less than competing models. Our findings challenge the prevailing scaling paradigm, demonstrating that architectural efficiency and tokenization quality can compensate for reduced parameter counts.
>
---
#### [replaced 046] Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02304v2](http://arxiv.org/pdf/2505.02304v2)**

> **作者:** Siyu Liang; Yunan Li; Wentian Xin; Huizhou Chen; Xujie Liu; Kang Liu; Qiguang Miao
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Sign language recognition (SLR) faces fundamental challenges in creating accurate annotations due to the inherent complexity of simultaneous manual and non-manual signals. To the best of our knowledge, this is the first work to integrate generative large language models (LLMs) into SLR tasks. We propose a novel Generative Sign-description Prompts Multi-positive Contrastive learning (GSP-MC) method that leverages retrieval-augmented generation (RAG) with domain-specific LLMs, incorporating multi-step prompt engineering and expert-validated sign language corpora to produce precise multipart descriptions. The GSP-MC method also employs a dual-encoder architecture to bidirectionally align hierarchical skeleton features with multiple text descriptions (global, synonym, and part level) through probabilistic matching. Our approach combines global and part-level losses, optimizing KL divergence to ensure robust alignment across all relevant text-skeleton pairs while capturing both sign-level semantics and detailed part dynamics. Experiments demonstrate state-of-the-art performance against existing methods on the Chinese SLR500 (reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's cross-lingual effectiveness highlight its potential for developing inclusive communication technologies.
>
---
#### [replaced 047] SenWiCh: Sense-Annotation of Low-Resource Languages for WiC using Hybrid Methods
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23714v2](http://arxiv.org/pdf/2505.23714v2)**

> **作者:** Roksana Goworek; Harpal Karlcut; Muhammad Shezad; Nijaguna Darshana; Abhishek Mane; Syam Bondada; Raghav Sikka; Ulvi Mammadov; Rauf Allahverdiyev; Sriram Purighella; Paridhi Gupta; Muhinyia Ndegwa; Haim Dubossarsky
>
> **备注:** 8 pages, 22 figures, published at SIGTYP 2025 workshop in ACL
>
> **摘要:** This paper addresses the critical need for high-quality evaluation datasets in low-resource languages to advance cross-lingual transfer. While cross-lingual transfer offers a key strategy for leveraging multilingual pretraining to expand language technologies to understudied and typologically diverse languages, its effectiveness is dependent on quality and suitable benchmarks. We release new sense-annotated datasets of sentences containing polysemous words, spanning ten low-resource languages across diverse language families and scripts. To facilitate dataset creation, the paper presents a demonstrably beneficial semi-automatic annotation method. The utility of the datasets is demonstrated through Word-in-Context (WiC) formatted experiments that evaluate transfer on these low-resource languages. Results highlight the importance of targeted dataset creation and evaluation for effective polysemy disambiguation in low-resource settings and transfer studies. The released datasets and code aim to support further research into fair, robust, and truly multilingual NLP.
>
---
#### [replaced 048] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.15846v2](http://arxiv.org/pdf/2507.15846v2)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [replaced 049] HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14311v3](http://arxiv.org/pdf/2505.14311v3)**

> **作者:** Shamsuddeen Hassan Muhammad; Ibrahim Said Ahmad; Idris Abdulmumin; Falalu Ibrahim Lawan; Babangida Sani; Sukairaj Hafiz Imam; Yusuf Aliyu; Sani Abdullahi Sani; Ali Usman Umar; Tajuddeen Gwadabe; Kenneth Church; Vukosi Marivate
>
> **摘要:** Hausa Natural Language Processing (NLP) has gained increasing attention in recent years, yet remains understudied as a low-resource language despite having over 120 million first-language (L1) and 80 million second-language (L2) speakers worldwide. While significant advances have been made in high-resource languages, Hausa NLP faces persistent challenges, including limited open-source datasets and inadequate model representation. This paper presents an overview of the current state of Hausa NLP, systematically examining existing resources, research contributions, and gaps across fundamental NLP tasks: text classification, machine translation, named entity recognition, speech recognition, and question answering. We introduce HausaNLP (https://catalog.hausanlp.org), a curated catalog that aggregates datasets, tools, and research works to enhance accessibility and drive further development. Furthermore, we discuss challenges in integrating Hausa into large language models (LLMs), addressing issues of suboptimal tokenization and dialectal variation. Finally, we propose strategic research directions emphasizing dataset expansion, improved language modeling approaches, and strengthened community collaboration to advance Hausa NLP. Our work provides both a foundation for accelerating Hausa NLP progress and valuable insights for broader multilingual NLP research.
>
---
#### [replaced 050] Mangosteen: An Open Thai Corpus for Language Model Pretraining
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14664v2](http://arxiv.org/pdf/2507.14664v2)**

> **作者:** Wannaphong Phatthiyaphaibun; Can Udomcharoenchaikit; Pakpoom Singkorapoom; Kunat Pipatanakul; Ekapol Chuangsuwanich; Peerat Limkonchotiwat; Sarana Nutanong
>
> **备注:** Work in Progress. All artifacts in this papers: https://huggingface.co/collections/aisingapore/wangchanlion-v3-687a362d8f0ea2fe4077c6b3
>
> **摘要:** Pre-training data shapes a language model's quality, but raw web text is noisy and demands careful cleaning. Existing large-scale corpora rely on English-centric or language-agnostic pipelines whose heuristics do not capture Thai script or cultural nuances, leaving risky material such as gambling content untreated. Prior Thai-specific efforts customize pipelines or build new ones, yet seldom release their data or document design choices, hindering reproducibility and raising the question of how to construct a transparent, high-quality Thai corpus. We introduce Mangosteen: a 47 billion-token Thai corpus built through a Thai-adapted Dolma pipeline that includes custom rule-based language ID, revised C4/Gopher quality filters, and Thai-trained content filters, plus curated non-web sources such as Wikipedia, Royal Gazette texts, OCR-extracted books, and CC-licensed YouTube subtitles. Systematic ablations using GPT-2 show the pipeline trims CommonCrawl from 202M to 25M documents while raising SEA-HELM NLG from 3 to 11; an 8B-parameter SEA-LION model continually pre-trained on Mangosteen then surpasses SEA-LION-v3 and Llama-3.1 by about four points on Thai benchmarks. We release the full pipeline code, cleaning manifests, corpus snapshot, and all checkpoints, providing a fully reproducible foundation for future Thai and regional LLM research.
>
---
#### [replaced 051] Physical models realizing the transformer architecture of large language models
- **分类: cs.LG; cs.AI; cs.CL; math-ph; math.MP**

- **链接: [http://arxiv.org/pdf/2507.13354v2](http://arxiv.org/pdf/2507.13354v2)**

> **作者:** Zeqian Chen
>
> **备注:** 6 pages, minor changes, Refs [3, 13, 15] added
>
> **摘要:** The introduction of the transformer architecture in 2017 marked the most striking advancement in natural language processing. The transformer is a model architecture relying entirely on an attention mechanism to draw global dependencies between input and output. However, we believe there is a gap in our theoretical understanding of what the transformer is, and how it works physically. From a physical perspective on modern chips, such as those chips under 28nm, modern intelligent machines should be regarded as open quantum systems beyond conventional statistical systems. Thereby, in this paper, we construct physical models realizing large language models based on a transformer architecture as open quantum systems in the Fock space over the Hilbert space of tokens. Our physical models underlie the transformer architecture for large language models.
>
---

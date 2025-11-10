# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 45 篇**

## 最新发布

#### [new 001] Evaluating LLMs' Reasoning Over Ordered Procedural Steps
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM在有序程序步骤推理任务中的表现，聚焦于从乱序食谱步骤中重建正确序列，评估其在零样本和少样本下的排序能力，揭示模型对长序列和高乱序度的处理局限。**

- **链接: [http://arxiv.org/pdf/2511.04688v1](http://arxiv.org/pdf/2511.04688v1)**

> **作者:** Adrita Anika; Md Messal Monem Miah
>
> **备注:** Accepted to IJCNLP-AACL 2025 Findings
>
> **摘要:** Reasoning over procedural sequences, where the order of steps directly impacts outcomes, is a critical capability for large language models (LLMs). In this work, we study the task of reconstructing globally ordered sequences from shuffled procedural steps, using a curated dataset of food recipes, a domain where correct sequencing is essential for task success. We evaluate several LLMs under zero-shot and few-shot settings and present a comprehensive evaluation framework that adapts established metrics from ranking and sequence alignment. These include Kendall's Tau, Normalized Longest Common Subsequence (NLCS), and Normalized Edit Distance (NED), which capture complementary aspects of ordering quality. Our analysis shows that model performance declines with increasing sequence length, reflecting the added complexity of longer procedures. We also find that greater step displacement in the input, corresponding to more severe shuffling, leads to further degradation. These findings highlight the limitations of current LLMs in procedural reasoning, especially with longer and more disordered inputs.
>
---
#### [new 002] Translation via Annotation: A Computational Study of Translating Classical Chinese into Japanese
- **分类: cs.CL**

- **简介: 该论文将古典汉文日译的注释过程建模为序列标注任务，解决低资源下的翻译难题。通过构建新数据集、引入LLM注释流水线及辅助中文NLP任务，提升标注性能，并揭示LLM在翻译与注释任务中的表现差异。**

- **链接: [http://arxiv.org/pdf/2511.05239v1](http://arxiv.org/pdf/2511.05239v1)**

> **作者:** Zilong Li; Jie Cao
>
> **摘要:** Ancient people translated classical Chinese into Japanese by annotating around each character. We abstract this process as sequence tagging tasks and fit them into modern language technologies. The research of this annotation and translation system is a facing low-resource problem. We release this problem by introducing a LLM-based annotation pipeline and construct a new dataset from digitalized open-source translation data. We show that under the low-resource setting, introducing auxiliary Chinese NLP tasks has a promoting effect on the training of sequence tagging tasks. We also evaluate the performance of large language models. They achieve high scores in direct machine translation, but they are confused when being asked to annotate characters. Our method could work as a supplement of LLMs.
>
---
#### [new 003] BudgetMem: Learning Selective Memory Policies for Cost-Efficient Long-Context Processing in Language Models
- **分类: cs.CL; cs.AI; I.2.7; I.2.6; H.3.3**

- **简介: 论文提出BudgetMem，面向长上下文语言模型的高效记忆管理，通过学习选择性记忆策略，在显著降低内存开销（72.4%）的同时保持接近基线的问答性能，解决资源受限场景下长文本处理的成本难题。**

- **链接: [http://arxiv.org/pdf/2511.04919v1](http://arxiv.org/pdf/2511.04919v1)**

> **作者:** Chandra Vamsi Krishna Alla; Harish Naidu Gaddam; Manohar Kommi
>
> **备注:** 11 pages, 3 figures, 5 tables. Evaluated on 700 QA pairs across multiple document lengths
>
> **摘要:** Large Language Models (LLMs) face significant computational and memory constraints when processing long contexts, despite growing demand for applications requiring reasoning over extensive documents, multi-session dialogues, and book length texts. While recent advances have extended context windows to 100K-1M tokens, such approaches incur prohibitive costs for resource constrained deployments. We propose BudgetMem, a novel memory augmented architecture that learns what to remember rather than remembering everything. Our system combines selective memory policies with feature based salience scoring (entity density, TF-IDF, discourse markers, position bias) to decide which information merits storage under strict budget constraints. Unlike existing retrieval augmented generation (RAG) systems that store all chunks, BudgetMem employs learned gating mechanisms coupled with BM25 sparse retrieval for efficient information access. Through comprehensive experiments on 700 question answer pairs across short (237 tokens) and long (5K-10K tokens) documents with Llama-3.2-3B-Instruct, we demonstrate that BudgetMem achieves remarkable results on long documents: only 1.0% F1 score degradation while saving 72.4% memory compared to baseline RAG. We validate our approach through budget sensitivity analysis (testing 7 budget ratios), naive baseline comparisons, and document length analysis, showing that BudgetMem's benefits increase with document length. Our work provides a practical pathway for deploying capable long context systems on modest hardware, democratizing access to advanced language understanding capabilities.
>
---
#### [new 004] Mind the Gap... or Not? How Translation Errors and Evaluation Details Skew Multilingual Results
- **分类: cs.CL**

- **简介: 该论文揭示多语言数学基准MGSM因翻译错误和答案提取不标准，虚假夸大语言性能差距，提出自动化质检与评估规范，纠正后差距基本消失，并发布修正数据集。**

- **链接: [http://arxiv.org/pdf/2511.05162v1](http://arxiv.org/pdf/2511.05162v1)**

> **作者:** Jan-Thorsten Peter; David Vilar; Tobias Domhan; Dan Malkin; Markus Freitag
>
> **摘要:** Most current large language models (LLMs) support a wide variety of languages in addition to English, including high-resource languages (e.g. German, Chinese, French), as well as low-resource ones (e.g. Swahili, Telugu). In addition they have also shown impressive capabilities in different domains, like coding, science and math. In this short paper, taking math as an example domain, we study the performance of different LLMs across languages. Experimental results show that there exists a non-negligible and consistent gap in the performance of the models across languages. Interestingly, and somewhat against expectations, the gap exists for both high- and low-resource languages. We hope that these results influence further research into cross-lingual capability generalization for next generation LLMs. If it weren't for the fact that they are false! By analyzing one of the standard multilingual math benchmarks (MGSM), we determine that several translation errors are present in the data. Furthermore, the lack of standardized answer extraction from LLM outputs further influences the final results. We propose a method for automatic quality assurance to address the first issue at scale, and give recommendations to address the second one. Combining these two approaches we show that the aforementioned language gap mostly disappears, leading to completely different conclusions from our research. We additionally release the corrected dataset to the community.
>
---
#### [new 005] Minority-Aware Satisfaction Estimation in Dialogue Systems via Preference-Adaptive Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文面向对话系统用户满意度估计任务，解决主流方法忽视少数用户偏好问题。提出CoPeR与M2PC分别建模个体与群体偏好，并构建PAda-PPO框架实现自适应强化学习优化，显著提升少数群体满意度估计准确率。**

- **链接: [http://arxiv.org/pdf/2511.05407v1](http://arxiv.org/pdf/2511.05407v1)**

> **作者:** Yahui Fu; Zi Haur Pang; Tatsuya Kawahara
>
> **备注:** IJCNLP-AACL 2025 (Main)
>
> **摘要:** User satisfaction in dialogue systems is inherently subjective. When the same response strategy is applied across users, minority users may assign different satisfaction ratings than majority users due to variations in individual intents and preferences. However, existing alignment methods typically train one-size-fits-all models that aim for broad consensus, often overlooking minority perspectives and user-specific adaptation. We propose a unified framework that models both individual- and group-level preferences for user satisfaction estimation. First, we introduce Chain-of-Personalized-Reasoning (CoPeR) to capture individual preferences through interpretable reasoning chains. Second, we propose an expectation-maximization-based Majority-Minority Preference-Aware Clustering (M2PC) algorithm that discovers distinct user groups in an unsupervised manner to learn group-level preferences. Finally, we integrate these components into a preference-adaptive reinforcement learning framework (PAda-PPO) that jointly optimizes alignment with both individual and group preferences. Experiments on the Emotional Support Conversation dataset demonstrate consistent improvements in user satisfaction estimation, particularly for underrepresented user groups.
>
---
#### [new 006] Cross-Lingual SynthDocs: A Large-Scale Synthetic Corpus for Any to Arabic OCR and Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Cross-Lingual SynthDocs，一个大规模合成阿拉伯语文档数据集，解决阿拉伯语OCR与文档理解资源匮乏问题。通过真实背景、双语布局与带变音符号字体生成250万样本，显著提升Qwen-2.5-VL在多模态任务上的性能。**

- **链接: [http://arxiv.org/pdf/2511.04699v1](http://arxiv.org/pdf/2511.04699v1)**

> **作者:** Haneen Al-Homoud; Asma Ibrahim; Murtadha Al-Jubran; Fahad Al-Otaibi; Yazeed Al-Harbi; Daulet Toibazar; Kesen Wang; Pedro J. Moreno
>
> **摘要:** Cross-Lingual SynthDocs is a large-scale synthetic corpus designed to address the scarcity of Arabic resources for Optical Character Recognition (OCR) and Document Understanding (DU). The dataset comprises over 2.5 million of samples, including 1.5 million textual data, 270K fully annotated tables, and hundred thousands of real data based charts. Our pipeline leverages authentic scanned backgrounds, bilingual layouts, and diacritic aware fonts to capture the typographic and structural complexity of Arabic documents. In addition to text, the corpus includes variety of rendered styles for charts and tables. Finetuning Qwen-2.5-VL on SynthDocs yields consistent improvements in Word Error Rate (WER) and Character Error Rate (CER) in terms of OCR across multiple public Arabic benchmarks, Tree-Edit Distance Similarity (TEDS) and Chart Extraction Score (CharTeX) improved as well in other modalities. SynthDocs provides a scalable, visually realistic resource for advancing research in multilingual document analysis.
>
---
#### [new 007] Steering Language Models with Weight Arithmetic
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出对比权重引导（contrastive weight steering），通过权重算术编辑LLM参数，以在窄分布训练下精准控制模型行为，缓解谄媚与错位问题，同时保持任务性能，并初步揭示可通过权重变化检测潜在错位行为。**

- **链接: [http://arxiv.org/pdf/2511.05408v1](http://arxiv.org/pdf/2511.05408v1)**

> **作者:** Constanza Fierro; Fabien Roger
>
> **摘要:** Providing high-quality feedback to Large Language Models (LLMs) on a diverse training distribution can be difficult and expensive, and providing feedback only on a narrow distribution can result in unintended generalizations. To better leverage narrow training data, we propose contrastive weight steering, a simple post-training method that edits the model parameters using weight arithmetic. We isolate a behavior direction in weight-space by subtracting the weight deltas from two small fine-tunes -- one that induces the desired behavior and another that induces its opposite -- and then add or remove this direction to modify the model's weights. We apply this technique to mitigate sycophancy and induce misalignment, and find that weight steering often generalizes further than activation steering, achieving stronger out-of-distribution behavioral control before degrading general capabilities. We also show that, in the context of task-specific fine-tuning, weight steering can partially mitigate undesired behavioral drift: it can reduce sycophancy and under-refusals introduced during fine-tuning while preserving task performance gains. Finally, we provide preliminary evidence that emergent misalignment can be detected by measuring the similarity between fine-tuning updates and an "evil" weight direction, suggesting that it may be possible to monitor the evolution of weights during training and detect rare misaligned behaviors that never manifest during training or evaluations.
>
---
#### [new 008] A Toolbox for Improving Evolutionary Prompt Search
- **分类: cs.CL**

- **简介: 该论文面向大语言模型提示优化任务，针对进化提示搜索中算子薄弱与评估低效问题，提出四点改进：分步进化、LLM评估器、人工反馈与高效评估策略，提升优化质量与效率，并开源工具箱。**

- **链接: [http://arxiv.org/pdf/2511.05120v1](http://arxiv.org/pdf/2511.05120v1)**

> **作者:** Daniel Grießhaber; Maximilian Kimmich; Johannes Maucher; Ngoc Thang Vu
>
> **摘要:** Evolutionary prompt optimization has demonstrated effectiveness in refining prompts for LLMs. However, existing approaches lack robust operators and efficient evaluation mechanisms. In this work, we propose several key improvements to evolutionary prompt optimization that can partially generalize to prompt optimization in general: 1) decomposing evolution into distinct steps to enhance the evolution and its control, 2) introducing an LLM-based judge to verify the evolutions, 3) integrating human feedback to refine the evolutionary operator, and 4) developing more efficient evaluation strategies that maintain performance while reducing computational overhead. Our approach improves both optimization quality and efficiency. We release our code, enabling prompt optimization on new tasks and facilitating further research in this area.
>
---
#### [new 009] Learning to reason about rare diseases through retrieval-augmented agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RADAR，一种检索增强的诊断推理代理，用于脑MRI中罕见病检测。通过检索医学文献与病例报告，无需重新训练即可提升模型对罕见病的识别准确率与可解释性，解决数据稀缺导致的AI失效问题。**

- **链接: [http://arxiv.org/pdf/2511.04720v1](http://arxiv.org/pdf/2511.04720v1)**

> **作者:** Ha Young Kim; Jun Li; Ana Beatriz Solana; Carolin M. Pirkl; Benedikt Wiestler; Julia A. Schnabel; Cosmin I. Bercea
>
> **备注:** Submitted on behalf of the PREDICTOM consortium
>
> **摘要:** Rare diseases represent the long tail of medical imaging, where AI models often fail due to the scarcity of representative training data. In clinical workflows, radiologists frequently consult case reports and literature when confronted with unfamiliar findings. Following this line of reasoning, we introduce RADAR, Retrieval Augmented Diagnostic Reasoning Agents, an agentic system for rare disease detection in brain MRI. Our approach uses AI agents with access to external medical knowledge by embedding both case reports and literature using sentence transformers and indexing them with FAISS to enable efficient similarity search. The agent retrieves clinically relevant evidence to guide diagnostic decision making on unseen diseases, without the need of additional training. Designed as a model-agnostic reasoning module, RADAR can be seamlessly integrated with diverse large language models, consistently improving their rare pathology recognition and interpretability. On the NOVA dataset comprising 280 distinct rare diseases, RADAR achieves up to a 10.2% performance gain, with the strongest improvements observed for open source models such as DeepSeek. Beyond accuracy, the retrieved examples provide interpretable, literature grounded explanations, highlighting retrieval-augmented reasoning as a powerful paradigm for low-prevalence conditions in medical imaging.
>
---
#### [new 010] LoPT: Lossless Parallel Tokenization Acceleration for Long Context Inference of Large Language Model
- **分类: cs.CL**

- **简介: LoPT针对大模型长文本推理中的词元化瓶颈，提出无损并行词元化框架，通过字符位置匹配与动态分块确保结果与串行一致，显著加速且无误差。**

- **链接: [http://arxiv.org/pdf/2511.04952v1](http://arxiv.org/pdf/2511.04952v1)**

> **作者:** Wei Shao; Lingchao Zheng; Pengyu Wang; Peizhen Zheng; Jun Li; Yuwei Fan
>
> **摘要:** Long context inference scenarios have become increasingly important for large language models, yet they introduce significant computational latency. While prior research has optimized long-sequence inference through operators, model architectures, and system frameworks, tokenization remains an overlooked bottleneck. Existing parallel tokenization methods accelerate processing through text segmentation and multi-process tokenization, but they suffer from inconsistent results due to boundary artifacts that occur after merging. To address this, we propose LoPT, a novel Lossless Parallel Tokenization framework that ensures output identical to standard sequential tokenization. Our approach employs character-position-based matching and dynamic chunk length adjustment to align and merge tokenized segments accurately. Extensive experiments across diverse long-text datasets demonstrate that LoPT achieves significant speedup while guaranteeing lossless tokenization. We also provide theoretical proof of consistency and comprehensive analytical studies to validate the robustness of our method.
>
---
#### [new 011] Too Good to be Bad: On the Failure of LLMs to Role-Play Villains
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在角色扮演中难以真实扮演反派的问题，提出Moral RolePlay基准，揭示安全对齐导致模型回避道德模糊角色，忠诚度随邪恶程度递减，揭示安全与创意间的根本冲突。**

- **链接: [http://arxiv.org/pdf/2511.04962v1](http://arxiv.org/pdf/2511.04962v1)**

> **作者:** Zihao Yi; Qingxuan Jiang; Ruotian Ma; Xingyu Chen; Qu Yang; Mengru Wang; Fanghua Ye; Ying Shen; Zhaopeng Tu; Xiaolong Li; Linus
>
> **摘要:** Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods.
>
---
#### [new 012] Surprisal reveals diversity gaps in image captioning and different scorers change the story
- **分类: cs.CL**

- **简介: 该论文研究图像描述生成的语言多样性，提出基于 surprisal 方差的评估指标，发现不同语言模型评分会反转模型与人类的多样性结论，强调多评分器对鲁棒评估的重要性。**

- **链接: [http://arxiv.org/pdf/2511.04754v1](http://arxiv.org/pdf/2511.04754v1)**

> **作者:** Nikolai Ilinykh; Simon Dobnik
>
> **备注:** Accepted and presented at INLG 2025
>
> **摘要:** We quantify linguistic diversity in image captioning with surprisal variance - the spread of token-level negative log-probabilities within a caption set. On the MSCOCO test set, we compare five state-of-the-art vision-and-language LLMs, decoded with greedy and nucleus sampling, to human captions. Measured with a caption-trained n-gram LM, humans display roughly twice the surprisal variance of models, but rescoring the same captions with a general-language model reverses the pattern. Our analysis introduces the surprisal-based diversity metric for image captioning. We show that relying on a single scorer can completely invert conclusions, thus, robust diversity evaluation must report surprisal under several scorers.
>
---
#### [new 013] EncouRAGe: Evaluating RAG Local, Fast, and Reliable
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出EncouRAGe框架，用于高效评估RAG系统的本地部署性能，解决RAG评估缺乏标准化工具的问题。通过多数据集实验，发现BM25优于嵌入模型，重排序收益有限。**

- **链接: [http://arxiv.org/pdf/2511.04696v1](http://arxiv.org/pdf/2511.04696v1)**

> **作者:** Jan Strich; Adeline Scharfenberg; Chris Biemann; Martin Semmann
>
> **备注:** Currently under review
>
> **摘要:** We introduce EncouRAGe, a comprehensive Python framework designed to streamline the development and evaluation of Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs) and Embedding Models. EncouRAGe comprises five modular and extensible components: Type Manifest, RAG Factory, Inference, Vector Store, and Metrics, facilitating flexible experimentation and extensible development. The framework emphasizes scientific reproducibility, diverse evaluation metrics, and local deployment, enabling researchers to efficiently assess datasets within RAG workflows. This paper presents implementation details and an extensive evaluation across multiple benchmark datasets, including 25k QA pairs and over 51k documents. Our results show that RAG still underperforms compared to the Oracle Context, while Hybrid BM25 consistently achieves the best results across all four datasets. We further examine the effects of reranking, observing only marginal performance improvements accompanied by higher response latency.
>
---
#### [new 014] Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究大语言模型（LLM）在未显式训练下为何能实现语义置信度校准，提出“B-校准”理论，揭示语义校准是next-token预测的副产物，并验证了RL微调和思维链会破坏此特性。**

- **链接: [http://arxiv.org/pdf/2511.04869v1](http://arxiv.org/pdf/2511.04869v1)**

> **作者:** Preetum Nakkiran; Arwen Bradley; Adam Goliński; Eugene Ndiaye; Michael Kirchhof; Sinead Williamson
>
> **摘要:** Large Language Models (LLMs) often lack meaningful confidence estimates for their outputs. While base LLMs are known to exhibit next-token calibration, it remains unclear whether they can assess confidence in the actual meaning of their responses beyond the token level. We find that, when using a certain sampling-based notion of semantic calibration, base LLMs are remarkably well-calibrated: they can meaningfully assess confidence in open-domain question-answering tasks, despite not being explicitly trained to do so. Our main theoretical contribution establishes a mechanism for why semantic calibration emerges as a byproduct of next-token prediction, leveraging a recent connection between calibration and local loss optimality. The theory relies on a general definition of "B-calibration," which is a notion of calibration parameterized by a choice of equivalence classes (semantic or otherwise). This theoretical mechanism leads to a testable prediction: base LLMs will be semantically calibrated when they can easily predict their own distribution over semantic answer classes before generating a response. We state three implications of this prediction, which we validate through experiments: (1) Base LLMs are semantically calibrated across question-answering tasks, (2) RL instruction-tuning systematically breaks this calibration, and (3) chain-of-thought reasoning breaks calibration. To our knowledge, our work provides the first principled explanation of when and why semantic calibration emerges in LLMs.
>
---
#### [new 015] What Are the Facts? Automated Extraction of Court-Established Facts from Criminal-Court Opinions
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在从刑事判决书中自动提取法院认定的犯罪事实，属于文本信息抽取任务。对比正则表达式与LLM（Gemini Flash 2.0）方法，提出增强版正则与LLM融合模型，提取准确率达99.5%，接近人工标注水平。**

- **链接: [http://arxiv.org/pdf/2511.05320v1](http://arxiv.org/pdf/2511.05320v1)**

> **作者:** Klára Bendová; Tomáš Knap; Jan Černý; Vojtěch Pour; Jaromir Savelka; Ivana Kvapilíková; Jakub Drápal
>
> **备注:** Paper accepted to the proceedings of ASAIL 2025 Workshop under ICAIL conference for publication. Paper contains 6 pages (references included) and 2 appendices. It contains 8 tables, no figures
>
> **摘要:** Criminal justice administrative data contain only a limited amount of information about the committed offense. However, there is an unused source of extensive information in continental European courts' decisions: descriptions of criminal behaviors in verdicts by which offenders are found guilty. In this paper, we study the feasibility of extracting these descriptions from publicly available court decisions from Slovakia. We use two different approaches for retrieval: regular expressions and large language models (LLMs). Our baseline was a simple method employing regular expressions to identify typical words occurring before and after the description. The advanced regular expression approach further focused on "sparing" and its normalization (insertion of spaces between individual letters), typical for delineating the description. The LLM approach involved prompting the Gemini Flash 2.0 model to extract the descriptions using predefined instructions. Although the baseline identified descriptions in only 40.5% of verdicts, both methods significantly outperformed it, achieving 97% with advanced regular expressions and 98.75% with LLMs, and 99.5% when combined. Evaluation by law students showed that both advanced methods matched human annotations in about 90% of cases, compared to just 34.5% for the baseline. LLMs fully matched human-labeled descriptions in 91.75% of instances, and a combination of advanced regular expressions with LLMs reached 92%.
>
---
#### [new 016] SDS KoPub VDR: A Benchmark Dataset for Visual Document Retrieval in Korean Public Documents
- **分类: cs.CL**

- **简介: 该论文提出首个韩语公共文档视觉检索基准SDS KoPub VDR，解决非英语文档结构复杂性与多模态理解不足的问题，构建了4万页真实文档与600个跨模态查询对，支持文本与多模态检索双任务评估。**

- **链接: [http://arxiv.org/pdf/2511.04910v1](http://arxiv.org/pdf/2511.04910v1)**

> **作者:** Jaehoon Lee; Sohyun Kim; Wanggeun Park; Geon Lee; Seungkyung Kim; Minyoung Lee
>
> **备注:** 27 pages, 15 figures, 6 tables
>
> **摘要:** Existing benchmarks for visual document retrieval (VDR) largely overlook non-English languages and the structural complexity of official publications. To address this critical gap, we introduce SDS KoPub VDR, the first large-scale, publicly available benchmark for retrieving and understanding Korean public documents. The benchmark is built upon a corpus of 361 real-world documents (40,781 pages), including 256 files under the KOGL Type 1 license and 105 from official legal portals, capturing complex visual elements like tables, charts, and multi-column layouts. To establish a challenging and reliable evaluation set, we constructed 600 query-page-answer triples. These were initially generated using multimodal models (e.g., GPT-4o) and subsequently underwent a rigorous human verification and refinement process to ensure factual accuracy and contextual relevance. The queries span six major public domains and are systematically categorized by the reasoning modality required: text-based, visual-based (e.g., chart interpretation), and cross-modal. We evaluate SDS KoPub VDR on two complementary tasks that reflect distinct retrieval paradigms: (1) text-only retrieval, which measures a model's ability to locate relevant document pages based solely on textual signals, and (2) multimodal retrieval, which assesses retrieval performance when visual features (e.g., tables, charts, and layouts) are jointly leveraged alongside text. This dual-task evaluation reveals substantial performance gaps, particularly in multimodal scenarios requiring cross-modal reasoning, even for state-of-the-art models. As a foundational resource, SDS KoPub VDR not only enables rigorous and fine-grained evaluation across textual and multimodal retrieval tasks but also provides a clear roadmap for advancing multimodal AI in complex, real-world document intelligence.
>
---
#### [new 017] GEMMA-SQL: A Novel Text-to-SQL Model Based on Large Language Models
- **分类: cs.CL**

- **简介: 论文提出GEMMA-SQL，一种基于Gemma 2B的轻量级文本到SQL模型，旨在降低部署成本并提升自然语言生成SQL的准确性。通过指令微调与多提示策略，在SPIDER基准上超越多个基线模型，实现高效、可扩展的文本到SQL转换。**

- **链接: [http://arxiv.org/pdf/2511.04710v1](http://arxiv.org/pdf/2511.04710v1)**

> **作者:** Hari Mohan Pandey; Anshul Gupta; Subham Sarkar; Minakshi Tomer; Schneider Johannes; Yan Gong
>
> **摘要:** Text-to-SQL systems enable users to interact with structured databases using natural language, eliminating the need for specialized programming knowledge. In this work, we introduce GEMMA-SQL, a lightweight and efficient text-to-SQL model built upon the open-source Gemma 2B architecture. Unlike many large language models (LLMs), GEMMA-SQL is fine-tuned in a resource-efficient, iterative manner and can be deployed on low-cost hardware. Leveraging the SPIDER benchmark for training and evaluation, GEMMA-SQL combines multiple prompting strategies, including few-shot learning, to enhance SQL query generation accuracy. The instruction-tuned variant, GEMMA-SQL Instruct, achieves 66.8% Test-Suite accuracy and 63.3% Exact Set Match accuracy, outperforming several state-of-the-art baselines such as IRNet, RYANSQL, and CodeXDavinci. The proposed approach demonstrates that effective prompt design and targeted instruction tuning can significantly boost performance while maintaining high scalability and adaptability. These results position GEMMA-SQL as a practical, open-source alternative for robust and accessible text-to-SQL systems.
>
---
#### [new 018] Pluralistic Behavior Suite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PLURALISTIC BEHAVIOR SUITE（PBSUITE），用于评估大模型在多轮交互中遵守多样化定制行为策略的能力，揭示现有对齐方法在复杂场景下合规性严重下降的问题，推动面向多元价值的鲁棒对齐研究。**

- **链接: [http://arxiv.org/pdf/2511.05018v1](http://arxiv.org/pdf/2511.05018v1)**

> **作者:** Prasoon Varshney; Makesh Narsimhan Sreedhar; Liwei Jiang; Traian Rebedea; Christopher Parisien
>
> **备注:** Accepted at the Multi-Turn Interactions workshop at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Large language models (LLMs) are typically aligned to a universal set of safety and usage principles intended for broad public acceptability. Yet, real-world applications of LLMs often take place within organizational ecosystems shaped by distinctive corporate policies, regulatory requirements, use cases, brand guidelines, and ethical commitments. This reality highlights the need for rigorous and comprehensive evaluation of LLMs with pluralistic alignment goals, an alignment paradigm that emphasizes adaptability to diverse user values and needs. In this work, we present PLURALISTIC BEHAVIOR SUITE (PBSUITE), a dynamic evaluation suite designed to systematically assess LLMs' capacity to adhere to pluralistic alignment specifications in multi-turn, interactive conversations. PBSUITE consists of (1) a diverse dataset of 300 realistic LLM behavioral policies, grounded in 30 industries; and (2) a dynamic evaluation framework for stress-testing model compliance with custom behavioral specifications under adversarial conditions. Using PBSUITE, We find that leading open- and closed-source LLMs maintain robust adherence to behavioral policies in single-turn settings (less than 4% failure rates), but their compliance weakens substantially in multi-turn adversarial interactions (up to 84% failure rates). These findings highlight that existing model alignment and safety moderation methods fall short in coherently enforcing pluralistic behavioral policies in real-world LLM interactions. Our work contributes both the dataset and analytical framework to support future research toward robust and context-aware pluralistic alignment techniques.
>
---
#### [new 019] multiMentalRoBERTa: A Fine-tuned Multiclass Classifier for Mental Health Disorder
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出multiMentalRoBERTa，用于从社交媒体文本中多分类识别六种心理疾病，解决早期筛查难题。通过微调RoBERTa模型，结合数据探索与可解释性分析，实现高精度分类，优于传统方法与基线模型。**

- **链接: [http://arxiv.org/pdf/2511.04698v1](http://arxiv.org/pdf/2511.04698v1)**

> **作者:** K M Sajjadul Islam; John Fields; Praveen Madiraju
>
> **备注:** Accepted in IEEE Big Data, 8-11 December, 2025 @ Macau SAR, China
>
> **摘要:** The early detection of mental health disorders from social media text is critical for enabling timely support, risk assessment, and referral to appropriate resources. This work introduces multiMentalRoBERTa, a fine-tuned RoBERTa model designed for multiclass classification of common mental health conditions, including stress, anxiety, depression, post-traumatic stress disorder (PTSD), suicidal ideation, and neutral discourse. Drawing on multiple curated datasets, data exploration is conducted to analyze class overlaps, revealing strong correlations between depression and suicidal ideation as well as anxiety and PTSD, while stress emerges as a broad, overlapping category. Comparative experiments with traditional machine learning methods, domain-specific transformers, and prompting-based large language models demonstrate that multiMentalRoBERTa achieves superior performance, with macro F1-scores of 0.839 in the six-class setup and 0.870 in the five-class setup (excluding stress), outperforming both fine-tuned MentalBERT and baseline classifiers. Beyond predictive accuracy, explainability methods, including Layer Integrated Gradients and KeyBERT, are applied to identify lexical cues that drive classification, with a particular focus on distinguishing depression from suicidal ideation. The findings emphasize the effectiveness of fine-tuned transformers for reliable and interpretable detection in sensitive contexts, while also underscoring the importance of fairness, bias mitigation, and human-in-the-loop safety protocols. Overall, multiMentalRoBERTa is presented as a lightweight, robust, and deployable solution for enhancing support in mental health platforms.
>
---
#### [new 020] SARC: Sentiment-Augmented Deep Role Clustering for Fake News Detection
- **分类: cs.CL**

- **简介: 论文提出SARC框架，用于虚假新闻检测任务，通过融合评论情感与用户角色聚类，解决传统方法忽视情感来源角色差异的问题，实现更精准的检测。**

- **链接: [http://arxiv.org/pdf/2511.04692v1](http://arxiv.org/pdf/2511.04692v1)**

> **作者:** Jingqing Wang; Jiaxing Shang; Rong Xu; Fei Hao; Tianjin Huang; Geyong Min
>
> **备注:** 12 pages, 11 figures, 4 tables, WSDM 2026 accepted paper
>
> **摘要:** Fake news detection has been a long-standing research focus in social networks. Recent studies suggest that incorporating sentiment information from both news content and user comments can enhance detection performance. However, existing approaches typically treat sentiment features as auxiliary signals, overlooking role differentiation, that is, the same sentiment polarity may originate from users with distinct roles, thereby limiting their ability to capture nuanced patterns for effective detection. To address this issue, we propose SARC, a Sentiment-Augmented Role Clustering framework which utilizes sentiment-enhanced deep clustering to identify user roles for improved fake news detection. The framework first generates user features through joint comment text representation (with BiGRU and Attention mechanism) and sentiment encoding. It then constructs a differentiable deep clustering module to automatically categorize user roles. Finally, unlike existing approaches which take fake news label as the unique supervision signal, we propose a joint optimization objective integrating role clustering and fake news detection to further improve the model performance. Experimental results on two benchmark datasets, RumourEval-19 and Weibo-comp, demonstrate that SARC achieves superior performance across all metrics compared to baseline models. The code is available at: https://github.com/jxshang/SARC.
>
---
#### [new 021] Explore Data Left Behind in Reinforcement Learning for Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文针对强化学习中残差提示（零奖励）导致训练信号缺失的问题，提出ERPO框架，通过自适应提升采样温度激发残差提示的探索，恢复训练信号，提升语言模型推理能力。**

- **链接: [http://arxiv.org/pdf/2511.04800v1](http://arxiv.org/pdf/2511.04800v1)**

> **作者:** Chenxi Liu; Junjie Liang; Yuqi Jia; Bochuan Cao; Yang Bai; Heng Huang; Xun Chen
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for improving the reasoning abilities of large language models (LLMs). The Group Relative Policy Optimization (GRPO) family has demonstrated strong performance in training LLMs with RLVR. However, as models train longer and scale larger, more training prompts become residual prompts, those with zero variance rewards that provide no training signal. Consequently, fewer prompts contribute to training, reducing diversity and hindering effectiveness. To fully exploit these residual prompts, we propose the Explore Residual Prompts in Policy Optimization (ERPO) framework, which encourages exploration on residual prompts and reactivates their training signals. ERPO maintains a history tracker for each prompt and adaptively increases the sampling temperature for residual prompts that previously produced all correct responses. This encourages the model to generate more diverse reasoning traces, introducing incorrect responses that revive training signals. Empirical results on the Qwen2.5 series demonstrate that ERPO consistently surpasses strong baselines across multiple mathematical reasoning benchmarks.
>
---
#### [new 022] MIMIC-SR-ICD11: A Dataset for Narrative-Based Diagnosis
- **分类: cs.CL; I.2.7; I.5.1**

- **简介: 该论文提出MIMIC-SR-ICD11数据集，用于基于临床叙事文本的ICD-11诊断编码任务，解决EHR模板丢失关键细节的问题，并设计LL-Rank模型，通过PMI评分消除标签频率偏差，显著提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2511.05485v1](http://arxiv.org/pdf/2511.05485v1)**

> **作者:** Yuexin Wu; Shiqi Wang; Vasile Rus
>
> **备注:** 19
>
> **摘要:** Disease diagnosis is a central pillar of modern healthcare, enabling early detection and timely intervention for acute conditions while guiding lifestyle adjustments and medication regimens to prevent or slow chronic disease. Self-reports preserve clinically salient signals that templated electronic health record (EHR) documentation often attenuates or omits, especially subtle but consequential details. To operationalize this shift, we introduce MIMIC-SR-ICD11, a large English diagnostic dataset built from EHR discharge notes and natively aligned to WHO ICD-11 terminology. We further present LL-Rank, a likelihood-based re-ranking framework that computes a length-normalized joint likelihood of each label given the clinical report context and subtracts the corresponding report-free prior likelihood for that label. Across seven model backbones, LL-Rank consistently outperforms a strong generation-plus-mapping baseline (GenMap). Ablation experiments show that LL-Rank's gains primarily stem from its PMI-based scoring, which isolates semantic compatibility from label frequency bias.
>
---
#### [new 023] Measuring what Matters: Construct Validity in Large Language Model Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估方法研究，旨在解决LLM基准测试中构念效度不足的问题。通过对445个基准的系统回顾，揭示其测量偏差，并提出八条可操作建议，以提升评估指标对安全、鲁棒性等核心现象的代表性。**

- **链接: [http://arxiv.org/pdf/2511.04703v1](http://arxiv.org/pdf/2511.04703v1)**

> **作者:** Andrew M. Bean; Ryan Othniel Kearns; Angelika Romanou; Franziska Sofia Hafner; Harry Mayne; Jan Batzner; Negar Foroutan; Chris Schmitz; Karolina Korgul; Hunar Batra; Oishi Deb; Emma Beharry; Cornelius Emde; Thomas Foster; Anna Gausen; María Grandury; Simeng Han; Valentin Hofmann; Lujain Ibrahim; Hazel Kim; Hannah Rose Kirk; Fangru Lin; Gabrielle Kaili-May Liu; Lennart Luettgau; Jabez Magomere; Jonathan Rystrøm; Anna Sotnikova; Yushi Yang; Yilun Zhao; Adel Bibi; Antoine Bosselut; Ronald Clark; Arman Cohan; Jakob Foerster; Yarin Gal; Scott A. Hale; Inioluwa Deborah Raji; Christopher Summerfield; Philip H. S. Torr; Cozmin Ududec; Luc Rocher; Adam Mahdi
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks
>
> **摘要:** Evaluating large language models (LLMs) is crucial for both assessing their capabilities and identifying safety or robustness issues prior to deployment. Reliably measuring abstract and complex phenomena such as 'safety' and 'robustness' requires strong construct validity, that is, having measures that represent what matters to the phenomenon. With a team of 29 expert reviewers, we conduct a systematic review of 445 LLM benchmarks from leading conferences in natural language processing and machine learning. Across the reviewed articles, we find patterns related to the measured phenomena, tasks, and scoring metrics which undermine the validity of the resulting claims. To address these shortcomings, we provide eight key recommendations and detailed actionable guidance to researchers and practitioners in developing LLM benchmarks.
>
---
#### [new 024] Reasoning-Guided Claim Normalization for Noisy Multilingual Social Media Posts
- **分类: cs.CL**

- **简介: 该论文面向多语言社交媒体虚假信息检测，提出推理引导的声明归一化方法，通过WHO-WHAT-WHEN等结构化分解，实现仅用英语数据训练即可跨语言生成可验证陈述，显著提升非英语语种的语义对齐与归一化效果。**

- **链接: [http://arxiv.org/pdf/2511.05078v1](http://arxiv.org/pdf/2511.05078v1)**

> **作者:** Manan Sharma; Arya Suneesh; Manish Jain; Pawan Kumar Rajpoot; Prasanna Devadiga; Bharatdeep Hazarika; Ashish Shrivastava; Kishan Gurumurthy; Anshuman B Suresh; Aditya U Baliga
>
> **摘要:** We address claim normalization for multilingual misinformation detection - transforming noisy social media posts into clear, verifiable statements across 20 languages. The key contribution demonstrates how systematic decomposition of posts using Who, What, Where, When, Why and How questions enables robust cross-lingual transfer despite training exclusively on English data. Our methodology incorporates finetuning Qwen3-14B using LoRA with the provided dataset after intra-post deduplication, token-level recall filtering for semantic alignment and retrieval-augmented few-shot learning with contextual examples during inference. Our system achieves METEOR scores ranging from 41.16 (English) to 15.21 (Marathi), securing third rank on the English leaderboard and fourth rank for Dutch and Punjabi. The approach shows 41.3% relative improvement in METEOR over baseline configurations and substantial gains over existing methods. Results demonstrate effective cross-lingual generalization for Romance and Germanic languages while maintaining semantic coherence across diverse linguistic structures.
>
---
#### [new 025] Large Language Models for Explainable Threat Intelligence
- **分类: cs.CL**

- **简介: 该论文提出RAGRecon系统，利用大语言模型结合检索增强生成（RAG）提升威胁情报问答准确性，并通过生成知识图谱实现决策可解释性，解决AI黑箱问题，实验表明其准确率超91%。**

- **链接: [http://arxiv.org/pdf/2511.05406v1](http://arxiv.org/pdf/2511.05406v1)**

> **作者:** Tiago Dinis; Miguel Correia; Roger Tavares
>
> **摘要:** As cyber threats continue to grow in complexity, traditional security mechanisms struggle to keep up. Large language models (LLMs) offer significant potential in cybersecurity due to their advanced capabilities in text processing and generation. This paper explores the use of LLMs with retrieval-augmented generation (RAG) to obtain threat intelligence by combining real-time information retrieval with domain-specific data. The proposed system, RAGRecon, uses a LLM with RAG to answer questions about cybersecurity threats. Moreover, it makes this form of Artificial Intelligence (AI) explainable by generating and visually presenting to the user a knowledge graph for every reply. This increases the transparency and interpretability of the reasoning of the model, allowing analysts to better understand the connections made by the system based on the context recovered by the RAG system. We evaluated RAGRecon experimentally with two datasets and seven different LLMs and the responses matched the reference responses more than 91% of the time for the best combinations.
>
---
#### [new 026] Order-Level Attention Similarity Across Language Models: A Latent Commonality
- **分类: cs.CL**

- **简介: 该论文发现不同语言模型在顺序级注意力（OLA）上存在潜在共性，并揭示OLA与句法知识的隐式关联，据此提出无需训练的跨模型适配器TOA，实现知识迁移与性能提升。**

- **链接: [http://arxiv.org/pdf/2511.05064v1](http://arxiv.org/pdf/2511.05064v1)**

> **作者:** Jinglin Liang; Jin Zhong; Shuangping Huang; Yunqing Hu; Huiyuan Zhang; Huifang Li; Lixin Fan; Hanlin Gu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** In this paper, we explore an important yet previously neglected question: Do context aggregation patterns across Language Models (LMs) share commonalities? While some works have investigated context aggregation or attention weights in LMs, they typically focus on individual models or attention heads, lacking a systematic analysis across multiple LMs to explore their commonalities. In contrast, we focus on the commonalities among LMs, which can deepen our understanding of LMs and even facilitate cross-model knowledge transfer. In this work, we introduce the Order-Level Attention (OLA) derived from the order-wise decomposition of Attention Rollout and reveal that the OLA at the same order across LMs exhibits significant similarities. Furthermore, we discover an implicit mapping between OLA and syntactic knowledge. Based on these two findings, we propose the Transferable OLA Adapter (TOA), a training-free cross-LM adapter transfer method. Specifically, we treat the OLA as a unified syntactic feature representation and train an adapter that takes OLA as input. Due to the similarities in OLA across LMs, the adapter generalizes to unseen LMs without requiring any parameter updates. Extensive experiments demonstrate that TOA's cross-LM generalization effectively enhances the performance of unseen LMs. Code is available at https://github.com/jinglin-liang/OLAS.
>
---
#### [new 027] UA-Code-Bench: A Competitive Programming Benchmark for Evaluating LLM Code Generation in Ukrainian
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文提出UA-Code-Bench，首个乌克兰语编程竞赛基准，用于评估大模型在低资源语言中的代码生成能力，涵盖500道Eolymp题目，测试13个模型，揭示其在非英语场景下的显著局限。**

- **链接: [http://arxiv.org/pdf/2511.05040v1](http://arxiv.org/pdf/2511.05040v1)**

> **作者:** Mykyta Syromiatnikov; Victoria Ruvinskaya
>
> **备注:** 8 pages, 5 figures. XI International conference "Informatics. Culture. Technique." (2025)
>
> **摘要:** Evaluating the real capabilities of large language models in low-resource languages still represents a challenge, as many existing benchmarks focus on widespread tasks translated from English or evaluate only simple language understanding. This paper introduces UA-Code-Bench, a new open-source benchmark established for a thorough evaluation of language models' code generation and competitive programming problem-solving abilities in Ukrainian. The benchmark comprises 500 problems from the Eolymp platform, evenly distributed across five complexity levels from very easy to very hard. A diverse set of 13 leading proprietary and open-source models, generating Python solutions based on a one-shot prompt, was evaluated via the dedicated Eolymp environment against hidden tests, ensuring code correctness. The obtained results reveal that even top-performing models, such as OpenAI o3 and GPT-5, solve only half of the problems, highlighting the challenge of code generation in low-resource natural language. Furthermore, this research presents a comprehensive analysis of performance across various difficulty levels, as well as an assessment of solution uniqueness and computational efficiency, measured by both elapsed time and memory consumption of the generated solutions. In conclusion, this work demonstrates the value of competitive programming benchmarks in evaluating large language models, especially in underrepresented languages. It also paves the way for future research on multilingual code generation and reasoning-enhanced models. The benchmark, data parsing, preparation, code generation, and evaluation scripts are available at https://huggingface.co/datasets/NLPForUA/ua-code-bench.
>
---
#### [new 028] Effectiveness of Chain-of-Thought in Distilling Reasoning Capability from Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究链式思维（CoT）在白盒知识蒸馏中提升小模型推理能力的有效性，解决小模型在复杂推理任务上性能不足的问题，通过Qwen和Llama2模型在CoT-Collection数据上蒸馏，并在BBH基准验证效果。**

- **链接: [http://arxiv.org/pdf/2511.05184v1](http://arxiv.org/pdf/2511.05184v1)**

> **作者:** Cong-Thanh Do; Rama Doddipatla; Kate Knill
>
> **备注:** In proceedings of the 18th International Natural Language Generation Conference (INLG 2025)
>
> **摘要:** Chain-of-Thought (CoT) prompting is a widely used method to improve the reasoning capability of Large Language Models (LLMs). More recently, CoT has been leveraged in Knowledge Distillation (KD) to transfer reasoning capability from a larger LLM to a smaller one. This paper examines the role of CoT in distilling the reasoning capability from larger LLMs to smaller LLMs using white-box KD, analysing its effectiveness in improving the performance of the distilled models for various natural language reasoning and understanding tasks. We conduct white-box KD experiments using LLMs from the Qwen and Llama2 families, employing CoT data from the CoT-Collection dataset. The distilled models are then evaluated on natural language reasoning and understanding tasks from the BIG-Bench-Hard (BBH) benchmark, which presents complex challenges for smaller LLMs. Experimental results demonstrate the role of CoT in improving white-box KD effectiveness, enabling the distilled models to achieve better average performance in natural language reasoning and understanding tasks from BBH.
>
---
#### [new 029] Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）行为自我意识的产生机制，发现其可通过单个低秩适配器诱导，且由激活空间中的单一引导向量表征，具有任务域特异性。属于机制分析与可控诱导任务，旨在揭示自我意识的最小条件与线性本质。**

- **链接: [http://arxiv.org/pdf/2511.04875v1](http://arxiv.org/pdf/2511.04875v1)**

> **作者:** Matthew Bozoukov; Matthew Nguyen; Shubkarman Singh; Bart Bussmann; Patrick Leask
>
> **摘要:** Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated.
>
---
#### [new 030] POLIS-Bench: Towards Multi-Dimensional Evaluation of LLMs for Bilingual Policy Tasks in Governmental Scenarios
- **分类: cs.CL; cs.AI**

- **简介: 论文提出POLIS-Bench，首个面向政府双语政策场景的LLM评估基准，解决现有评测缺乏真实场景与多维指标的问题，构建了更新的语料、三项任务及双指标评估框架，并成功微调出低成本高性能开源模型。**

- **链接: [http://arxiv.org/pdf/2511.04705v1](http://arxiv.org/pdf/2511.04705v1)**

> **作者:** Tingyue Yang; Junchi Yao; Yuhui Guo; Chang Liu
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** We introduce POLIS-Bench, the first rigorous, systematic evaluation suite designed for LLMs operating in governmental bilingual policy scenarios. Compared to existing benchmarks, POLIS-Bench introduces three major advancements. (i) Up-to-date Bilingual Corpus: We construct an extensive, up-to-date policy corpus that significantly scales the effective assessment sample size, ensuring relevance to current governance practice. (ii) Scenario-Grounded Task Design: We distill three specialized, scenario-grounded tasks -- Clause Retrieval & Interpretation, Solution Generation, and the Compliance Judgmen--to comprehensively probe model understanding and application. (iii) Dual-Metric Evaluation Framework: We establish a novel dual-metric evaluation framework combining semantic similarity with accuracy rate to precisely measure both content alignment and task requirement adherence. A large-scale evaluation of over 10 state-of-the-art LLMs on POLIS-Bench reveals a clear performance hierarchy where reasoning models maintain superior cross-task stability and accuracy, highlighting the difficulty of compliance tasks. Furthermore, leveraging our benchmark, we successfully fine-tune a lightweight open-source model. The resulting POLIS series models achieves parity with, or surpasses, strong proprietary baselines on multiple policy subtasks at a significantly reduced cost, providing a cost-effective and compliant path for robust real-world governmental deployment.
>
---
#### [new 031] Reflective Personalization Optimization: A Post-hoc Rewriting Framework for Black-Box Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出RPO框架，解决黑盒大模型个性化中内容生成与风格对齐冲突的问题，通过两阶段重写（生成+反射）解耦任务，用监督微调与强化学习训练外部重写模块，实现模型无关的高效个性化。**

- **链接: [http://arxiv.org/pdf/2511.05286v1](http://arxiv.org/pdf/2511.05286v1)**

> **作者:** Teqi Hao; Xioayu Tan; Shaojie Shi; Yinghui Xu; Xihe Qiu
>
> **摘要:** The personalization of black-box large language models (LLMs) is a critical yet challenging task. Existing approaches predominantly rely on context injection, where user history is embedded into the prompt to directly guide the generation process. However, this single-step paradigm imposes a dual burden on the model: generating accurate content while simultaneously aligning with user-specific styles. This often results in a trade-off that compromises output quality and limits precise control. To address this fundamental tension, we propose Reflective Personalization Optimization (RPO), a novel framework that redefines the personalization paradigm by decoupling content generation from alignment. RPO operates in two distinct stages: first, a base model generates a high-quality, generic response; then, an external reflection module explicitly rewrites this output to align with the user's preferences. This reflection module is trained using a two-stage process. Initially, supervised fine-tuning is employed on structured rewriting trajectories to establish a core personalized reasoning policy that models the transformation from generic to user-aligned responses. Subsequently, reinforcement learning is applied to further refine and enhance the quality of the personalized outputs. Comprehensive experiments on the LaMP benchmark demonstrate that RPO, by decoupling content generation from personalization, significantly outperforms state-of-the-art baselines. These findings underscore the superiority of explicit response shaping over implicit context injection. Moreover, RPO introduces an efficient, model-agnostic personalization layer that can be seamlessly integrated with any underlying base model, paving the way for a new and effective direction in user-centric generation scenarios.
>
---
#### [new 032] Iterative Layer-wise Distillation for Efficient Compression of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种迭代层间蒸馏方法，用于压缩大语言模型。通过评估各层重要性并结合联合损失微调，实现高效降层，如将Qwen2.5-3B从36层减至24层，仅损失18%性能，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2511.05085v1](http://arxiv.org/pdf/2511.05085v1)**

> **作者:** Grigory Kovalev; Mikhail Tikhomirov
>
> **摘要:** This work investigates distillation methods for large language models (LLMs) with the goal of developing compact models that preserve high performance. Several existing approaches are reviewed, with a discussion of their respective strengths and limitations. An improved method based on the ShortGPT approach has been developed, building upon the idea of incorporating iterative evaluation of layer importance. At each step, importance is assessed by measuring performance degradation when individual layers are removed, using a set of representative datasets. This process is combined with further training using a joint loss function based on KL divergence and mean squared error. Experiments on the Qwen2.5-3B model show that the number of layers can be reduced from 36 to 28 (resulting in a 2.47 billion parameter model) with only a 9.7% quality loss, and to 24 layers with an 18% loss. The findings suggest that the middle transformer layers contribute less to inference, underscoring the potential of the proposed method for creating efficient models. The results demonstrate the effectiveness of iterative distillation and fine-tuning, making the approach suitable for deployment in resource-limited settings.
>
---
#### [new 033] A multimodal multiplex of the mental lexicon for multilingual individuals
- **分类: cs.CL; cs.AI**

- **简介: 该研究构建多模态多层心理词典模型，探究多语者视觉输入对语言习得的影响，解决视觉线索是否提升翻译准确率的问题，拓展了BIA+框架与多层网络理论。**

- **链接: [http://arxiv.org/pdf/2511.05361v1](http://arxiv.org/pdf/2511.05361v1)**

> **作者:** Maria Huynh; Wilder C. Rodrigues
>
> **摘要:** Historically, bilingualism was often perceived as an additional cognitive load that could hinder linguistic and intellectual development. However, over the last three decades, this view has changed considerably. Numerous studies have aimed to model and understand the architecture of the bilingual word recognition system Dijkstra and van Heuven (2002), investigating how parallel activation operates in the brain and how one language influences another Kroll et al. (2015). Increasingly, evidence suggests that multilinguals, individuals who speak three or more languages, can perform better than monolinguals in various linguistic and cognitive tasks, such as learning an additional language Abu-Rabia and Sanitsky (2010). This research proposal focuses on the study of the mental lexicon and how it may be structured in individuals who speak multiple languages. Building on the work of Stella et al. (2018), who investigated explosive learning in humans using a multiplex model of the mental lexicon, and the Bilingual Interactive Activation (BIA+) framework proposed by Dijkstra and van Heuven (2002), the present study applies the same multilayer network principles introduced by Kivela et al. (2014). Our experimental design extends previous research by incorporating multimodality into the multiplex model, introducing an additional layer that connects visual inputs to their corresponding lexical representations across the multilingual layers of the mental lexicon. In this research, we aim to explore how a heritage language influences the acquisition of another language. Specifically, we ask: Does the presence of visual input in a translation task influence participants' proficiency and accuracy compared to text-only conditions?
>
---
#### [new 034] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型中训练样本影响估计的层选择与聚合策略，反驳“首层最有效”的传统观点，证明中间注意力层更优，并提出新评估指标NDR与改进聚合方法，提升影响估计准确性。**

- **链接: [http://arxiv.org/pdf/2511.04715v1](http://arxiv.org/pdf/2511.04715v1)**

> **作者:** Dmytro Vitel; Anshuman Chhabra
>
> **摘要:** Identifying how training samples influence/impact Large Language Model (LLM) decision-making is essential for effectively interpreting model decisions and auditing large-scale datasets. Current training sample influence estimation methods (also known as influence functions) undertake this goal by utilizing information flow through the model via its first-order and higher-order gradient terms. However, owing to the large model sizes of today consisting of billions of parameters, these influence computations are often restricted to some subset of model layers to ensure computational feasibility. Prior seminal work by Yeh et al. (2022) in assessing which layers are best suited for computing language data influence concluded that the first (embedding) layers are the most informative for this purpose, using a hypothesis based on influence scores canceling out (i.e., the cancellation effect). In this work, we propose theoretical and empirical evidence demonstrating how the cancellation effect is unreliable, and that middle attention layers are better estimators for influence. Furthermore, we address the broader challenge of aggregating influence scores across layers, and showcase how alternatives to standard averaging (such as ranking and vote-based methods) can lead to significantly improved performance. Finally, we propose better methods for evaluating influence score efficacy in LLMs without undertaking model retraining, and propose a new metric known as the Noise Detection Rate (NDR) that exhibits strong predictive capability compared to the cancellation effect. Through extensive experiments across LLMs of varying types and scales, we concretely determine that the first (layers) are not necessarily better than the last (layers) for LLM influence estimation, contrasting with prior knowledge in the field.
>
---
#### [new 035] Separate the Wheat from the Chaff: Winnowing Down Divergent Views in Retrieval Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WinnowRAG，用于检索增强生成（RAG）中的噪声过滤任务，解决多文档检索引入冗余与误导信息的问题。通过两阶段聚类与智能评估机制，自动筛选有效文档，无需微调即可提升生成准确率。**

- **链接: [http://arxiv.org/pdf/2511.04700v1](http://arxiv.org/pdf/2511.04700v1)**

> **作者:** Song Wang; Zihan Chen; Peng Wang; Zhepei Wei; Zhen Tan; Yu Meng; Cong Shen; Jundong Li
>
> **备注:** EMNLP Main 2025
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources to address their limitations in accessing up-to-date or specialized information. A natural strategy to increase the likelihood of retrieving relevant information is to expand the number of retrieved documents. However, involving more documents could introduce significant noise, as many documents may be irrelevant or misleading, thereby reducing the overall accuracy of the generated responses. To overcome the challenge associated with handling a larger number of documents, we propose WinnowRAG, a novel RAG framework designed to systematically filter out noisy documents while preserving valuable content -- a process we refer to as winnowing. WinnowRAG operates in two stages: In Stage I, we perform query-aware clustering to group similar documents and form distinct topic clusters. Each cluster is assigned to an LLM agent for generating a unique answer. In Stage II, we perform winnowing, wherein a critic LLM evaluates the outputs of multiple agents and iteratively separates useful documents from noisy ones. To retain useful documents when discarding agents, we propose two strategic merging techniques to ensure that only relevant knowledge is used for generating the final response. Crucially, WinnowRAG is model-agnostic and does not require any model fine-tuning, making it easily adaptable to various tasks. Extensive experiments on various realistic datasets demonstrate the effectiveness of WinnowRAG over state-of-the-art baselines.
>
---
#### [new 036] On Text Simplification Metrics and General-Purpose LLMs for Accessible Health Information, and A Potential Architectural Advantage of The Instruction-Tuned LLM class
- **分类: cs.CL**

- **简介: 该论文研究面向健康信息可及性的文本简化任务，比较指令微调Mistral与推理增强QWen的简化能力，发现指令微调模型在提升可读性同时更好保留语义，提出指标选择与词汇适配为关键挑战。**

- **链接: [http://arxiv.org/pdf/2511.05080v1](http://arxiv.org/pdf/2511.05080v1)**

> **作者:** P. Bilha Githinji; Aikaterini Meilliou; Peiwu Qin
>
> **摘要:** The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models, however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses the performance of two major classes of general-purpose LLMs, demonstrating their linguistic capabilities and foundational readiness for the task compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral 24B and the reasoning-augmented QWen2.5 32B, we identify a potential architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics and the simplification-specific formula SARI (mean 42.46), while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance, but its operational strategy shows a disconnect in balancing between readability and accuracy, reaching a statistically significantly lower BERTScore of 0.89. Additionally, a comprehensive correlation analysis of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies among five readability indices. This empirical evidence tracks baseline performance of the evolving LLMs for the task of text simplification, identifies the instruction-tuned Mistral 24B for simplification, provides necessary heuristics for metric selection, and points to lexical support as a primary domain-adaptation issue for simplification.
>
---
#### [new 037] Reasoning Up the Instruction Ladder for Controllable Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出通过推理解决多源指令冲突问题，构建VerIH数据集并用轻量RL训练模型，使其能按优先级处理系统与用户指令，提升可控性与抗攻击能力，实现更可靠的LLM行为控制。**

- **链接: [http://arxiv.org/pdf/2511.04694v1](http://arxiv.org/pdf/2511.04694v1)**

> **作者:** Zishuo Zheng; Vidhisha Balachandran; Chan Young Park; Faeze Brahman; Sachin Kumar
>
> **摘要:** As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.
>
---
#### [new 038] Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ATLAS，一种基于项目反应理论的自适应评估框架，解决LLM静态基准测试成本高、信息冗余问题，通过Fisher信息筛选关键题目，实现90%题目缩减并提升评估精度与模型排序区分度。**

- **链接: [http://arxiv.org/pdf/2511.04689v1](http://arxiv.org/pdf/2511.04689v1)**

> **作者:** Peiyu Li; Xiuxiu Tang; Si Chen; Ying Cheng; Ronald Metoyer; Ting Hua; Nitesh V. Chawla
>
> **备注:** Code and calibrated item banks are available at https://github.com/Peiyu-Georgia-Li/ATLAS.git
>
> **摘要:** Large language model evaluation requires thousands of benchmark items, making evaluations expensive and slow. Existing methods compute average accuracy across fixed item sets, treating all items equally despite varying quality and informativeness. We present ATLAS an adaptive testing framework using Item Response Theory (IRT) to estimate model ability through Fisher information-guided item selection. Our analysis of five major benchmarks reveals that 3-6% of items exhibit negative discrimination, indicating annotation errors that corrupt static evaluation. ATLAS achieves 90% item reduction while maintaining measurement precision: on HellaSwag (5,608 items), we match full-benchmark estimates using only 42 items with 0.154 MAE. Our framework maintains item exposure rates below 10% and test overlap at 16-27%, compared to static benchmarks where every model sees all items (100% exposure). Among 4,000+ tested models, IRT ranks differ from accuracy ranks: models with the same accuracy get different IRT scores, and 23-31% of all models shift by more than 10 rank positions. Code and calibrated item banks are available at https://github.com/Peiyu-Georgia-Li/ATLAS.git.
>
---
#### [new 039] Listening Between the Lines: Decoding Podcast Narratives with Language Modeling
- **分类: cs.CL; cs.SI**

- **简介: 该论文针对播客对话的非结构化特性，提出一种微调BERT的方法，将叙事框架与具体实体关联，提升自动识别叙事框架的准确性，并揭示话题与表达框架间的系统关系，属于自然语言处理中的叙事分析任务。**

- **链接: [http://arxiv.org/pdf/2511.05310v1](http://arxiv.org/pdf/2511.05310v1)**

> **作者:** Shreya Gupta; Ojasva Saxena; Arghodeep Nandi; Sarah Masud; Kiran Garimella; Tanmoy Chakraborty
>
> **备注:** 10 pages, 6 Figures, 5 Tables. Under review at IEEE TCSS
>
> **摘要:** Podcasts have become a central arena for shaping public opinion, making them a vital source for understanding contemporary discourse. Their typically unscripted, multi-themed, and conversational style offers a rich but complex form of data. To analyze how podcasts persuade and inform, we must examine their narrative structures -- specifically, the narrative frames they employ. The fluid and conversational nature of podcasts presents a significant challenge for automated analysis. We show that existing large language models, typically trained on more structured text such as news articles, struggle to capture the subtle cues that human listeners rely on to identify narrative frames. As a result, current approaches fall short of accurately analyzing podcast narratives at scale. To solve this, we develop and evaluate a fine-tuned BERT model that explicitly links narrative frames to specific entities mentioned in the conversation, effectively grounding the abstract frame in concrete details. Our approach then uses these granular frame labels and correlates them with high-level topics to reveal broader discourse trends. The primary contributions of this paper are: (i) a novel frame-labeling methodology that more closely aligns with human judgment for messy, conversational data, and (ii) a new analysis that uncovers the systematic relationship between what is being discussed (the topic) and how it is being presented (the frame), offering a more robust framework for studying influence in digital media.
>
---
#### [new 040] Evaluating Subword Tokenization Techniques for Bengali: A Benchmark Study with BengaliBPE
- **分类: cs.CL**

- **简介: 该论文针对孟加拉语形态丰富特性，提出语言感知的BengaliBPE分词器，解决通用子词分词器性能不佳问题。通过对比实验验证其在分词细粒度、形态可解释性与分类准确率上的优势，为孟加拉语NLP奠定基础。**

- **链接: [http://arxiv.org/pdf/2511.05324v1](http://arxiv.org/pdf/2511.05324v1)**

> **作者:** Firoj Ahmmed Patwary; Abdullah Al Noman
>
> **备注:** 10 pages, 3 figures, 3 tables
>
> **摘要:** Tokenization is an important first step in Natural Language Processing (NLP) pipelines because it decides how models learn and represent linguistic information. However, current subword tokenizers like SentencePiece or HuggingFace BPE are mostly designed for Latin or multilingual corpora and do not perform well on languages with rich morphology such as Bengali. To address this limitation, we present BengaliBPE, a Byte Pair Encoding (BPE) tokenizer specifically developed for the Bengali script. BengaliBPE applies Unicode normalization, grapheme-level initialization, and morphology-aware merge rules to maintain linguistic consistency and preserve subword integrity. We use a large-scale Bengali news classification dataset to compare BengaliBPE with three baselines: Whitespace, SentencePiece BPE, and HuggingFace BPE. The evaluation considers tokenization granularity, encoding speed, and downstream classification accuracy. While all methods perform reasonably well, BengaliBPE provides the most detailed segmentation and the best morphological interpretability, albeit with slightly higher computational cost. These findings highlight the importance of language-aware tokenization for morphologically rich scripts and establish BengaliBPE as a strong foundation for future Bengali NLP systems, including large-scale pretraining of contextual language models.
>
---
#### [new 041] ManufactuBERT: Efficient Continual Pretraining for Manufacturing
- **分类: cs.CL**

- **简介: 论文提出ManufactuBERT，通过在制造领域语料上持续预训练RoBERTa，解决通用模型在专业领域表现差的问题。构建了去重数据管道，显著提升性能并缩短33%训练时间。**

- **链接: [http://arxiv.org/pdf/2511.05135v1](http://arxiv.org/pdf/2511.05135v1)**

> **作者:** Robin Armingaud; Romaric Besançon
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** While large general-purpose Transformer-based encoders excel at general language understanding, their performance diminishes in specialized domains like manufacturing due to a lack of exposure to domain-specific terminology and semantics. In this paper, we address this gap by introducing ManufactuBERT, a RoBERTa model continually pretrained on a large-scale corpus curated for the manufacturing domain. We present a comprehensive data processing pipeline to create this corpus from web data, involving an initial domain-specific filtering step followed by a multi-stage deduplication process that removes redundancies. Our experiments show that ManufactuBERT establishes a new state-of-the-art on a range of manufacturing-related NLP tasks, outperforming strong specialized baselines. More importantly, we demonstrate that training on our carefully deduplicated corpus significantly accelerates convergence, leading to a 33\% reduction in training time and computational cost compared to training on the non-deduplicated dataset. The proposed pipeline offers a reproducible example for developing high-performing encoders in other specialized domains. We will release our model and curated corpus at https://huggingface.co/cea-list-ia.
>
---
#### [new 042] Diagnosing and Mitigating Semantic Inconsistencies in Wikidata's Classification Hierarchy
- **分类: cs.CL**

- **简介: 该论文聚焦Wikidata分类层次中的语义不一致问题，提出新验证方法识别分类错误与冗余链接，并构建用户可交互的评估系统，利用众包机制辅助修正知识图谱的本体结构。**

- **链接: [http://arxiv.org/pdf/2511.04926v1](http://arxiv.org/pdf/2511.04926v1)**

> **作者:** Shixiong Zhao; Hideaki Takeda
>
> **摘要:** Wikidata is currently the largest open knowledge graph on the web, encompassing over 120 million entities. It integrates data from various domain-specific databases and imports a substantial amount of content from Wikipedia, while also allowing users to freely edit its content. This openness has positioned Wikidata as a central resource in knowledge graph research and has enabled convenient knowledge access for users worldwide. However, its relatively loose editorial policy has also led to a degree of taxonomic inconsistency. Building on prior work, this study proposes and applies a novel validation method to confirm the presence of classification errors, over-generalized subclass links, and redundant connections in specific domains of Wikidata. We further introduce a new evaluation criterion for determining whether such issues warrant correction and develop a system that allows users to inspect the taxonomic relationships of arbitrary Wikidata entities-leveraging the platform's crowdsourced nature to its full potential.
>
---
#### [new 043] AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent
- **分类: cs.CL**

- **简介: 论文提出AgentExpt，用于自动化AI实验设计，解决传统方法数据覆盖不全与相似性偏差问题。通过构建引文网络、增强检索与推理重排，实现更精准的基准与数据集推荐，显著提升推荐效果。**

- **链接: [http://arxiv.org/pdf/2511.04921v1](http://arxiv.org/pdf/2511.04921v1)**

> **作者:** Yu Li; Lehui Li; Qingmin Liao; Fengli Xu; Yong Li
>
> **备注:** 10 pages
>
> **摘要:** Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.
>
---
#### [new 044] Acquiring Common Chinese Emotional Events Using Large Language Model
- **分类: cs.CL**

- **简介: 该论文旨在获取中文通用情感事件，通过大语言模型生成并过滤高质量情感事件，构建首个大规模中文情感事件知识库，解决情感事件难以获取的问题，支持情感原因提取等任务。**

- **链接: [http://arxiv.org/pdf/2511.04989v1](http://arxiv.org/pdf/2511.04989v1)**

> **作者:** Ya Wang; Guangzheng Zhu; Cungen Cao; Jingjing Li; He Li; Xin Huang
>
> **备注:** I am the second author (Guangzheng Zhu) and I am submitting this paper on behalf of all co-authors
>
> **摘要:** Knowledge about emotional events is an important kind of knowledge which has been applied to improve the effectiveness of different applications. However, emotional events cannot be easily acquired, especially common or generalized emotional events that are context-independent. The goal of this paper is to obtain common emotional events in Chinese language such as "win a prize" and "be criticized". Our approach begins by collecting a comprehensive list of Chinese emotional event indicators. Then, we generate emotional events by prompting a Chinese large language model (LLM) using these indicators. To ensure the quality of these emotional events, we train a filter to discard invalid generated results. We also classify these emotional events as being positive events and negative events using different techniques. Finally, we harvest a total of 102,218 high-quality common emotional events with sentiment polarity labels, which is the only large-scale commonsense knowledge base of emotional events in Chinese language. Intrinsic evaluation results show that the proposed method in this paper can be effectively used to acquire common Chinese emotional events. An extrinsic use case also demonstrates the strong potential of common emotional events in the field of emotion cause extraction (ECE). Related resources including emotional event indicators and emotional events will be released after the publication of this paper.
>
---
#### [new 045] Towards Mitigating Hallucinations in Large Vision-Language Models by Refining Textual Embeddings
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉-语言模型中语言模态偏差导致的幻觉问题，提出通过平均池化视觉特征 refine 文本嵌入，增强视觉对齐，有效降低幻觉，属视觉-语言对齐与幻觉抑制任务。**

- **链接: [http://arxiv.org/pdf/2511.05017v1](http://arxiv.org/pdf/2511.05017v1)**

> **作者:** Aakriti Agrawal; Gouthaman KV; Rohith Aralikatti; Gauri Jagatap; Jiaxin Yuan; Vijay Kamarshi; Andrea Fanelli; Furong Huang
>
> **摘要:** In this work, we identify an inherent bias in prevailing LVLM architectures toward the language modality, largely resulting from the common practice of simply appending visual embeddings to the input text sequence. To address this, we propose a simple yet effective method that refines textual embeddings by integrating average-pooled visual features. Our approach demonstrably improves visual grounding and significantly reduces hallucinations on established benchmarks. While average pooling offers a straightforward, robust, and efficient means of incorporating visual information, we believe that more sophisticated fusion methods could further enhance visual grounding and cross-modal alignment. Given that the primary focus of this work is to highlight the modality imbalance and its impact on hallucinations -- and to show that refining textual embeddings with visual information mitigates this issue -- we leave exploration of advanced fusion strategies for future work.
>
---
#### [new 046] ORCHID: Orchestrated Retrieval-Augmented Classification with Human-in-the-Loop Intelligent Decision-Making for High-Risk Property
- **分类: cs.AI; cs.CL**

- **简介: ORCHID是一种面向高风险财产分类的智能代理系统，融合检索增强生成与人工监督，解决传统方法效率低、可追溯性差的问题，通过模块化协作实现可审计、可解释的合规决策。**

- **链接: [http://arxiv.org/pdf/2511.04956v1](http://arxiv.org/pdf/2511.04956v1)**

> **作者:** Maria Mahbub; Vanessa Lama; Sanjay Das; Brian Starks; Christopher Polchek; Saffell Silvers; Lauren Deck; Prasanna Balaprakash; Tirthankar Ghosal
>
> **摘要:** High-Risk Property (HRP) classification is critical at U.S. Department of Energy (DOE) sites, where inventories include sensitive and often dual-use equipment. Compliance must track evolving rules designated by various export control policies to make transparent and auditable decisions. Traditional expert-only workflows are time-consuming, backlog-prone, and struggle to keep pace with shifting regulatory boundaries. We demo ORCHID, a modular agentic system for HRP classification that pairs retrieval-augmented generation (RAG) with human oversight to produce policy-based outputs that can be audited. Small cooperating agents, retrieval, description refiner, classifier, validator, and feedback logger, coordinate via agent-to-agent messaging and invoke tools through the Model Context Protocol (MCP) for model-agnostic on-premise operation. The interface follows an Item to Evidence to Decision loop with step-by-step reasoning, on-policy citations, and append-only audit bundles (run-cards, prompts, evidence). In preliminary tests on real HRP cases, ORCHID improves accuracy and traceability over a non-agentic baseline while deferring uncertain items to Subject Matter Experts (SMEs). The demonstration shows single item submission, grounded citations, SME feedback capture, and exportable audit artifacts, illustrating a practical path to trustworthy LLM assistance in sensitive DOE compliance workflows.
>
---
#### [new 047] A Penny for Your Thoughts: Decoding Speech from Inexpensive Brain Signals
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文研究从低成本脑电图（EEG）信号中解码语音，属于脑机接口中的脑到语音解码任务。通过改进Meta的EEG解码器，引入个性化注意力与双路径RNN，提升语音重建准确率，验证个性化架构的有效性。**

- **链接: [http://arxiv.org/pdf/2511.04691v1](http://arxiv.org/pdf/2511.04691v1)**

> **作者:** Quentin Auster; Kateryna Shapovalenko; Chuang Ma; Demaio Sun
>
> **摘要:** We explore whether neural networks can decode brain activity into speech by mapping EEG recordings to audio representations. Using EEG data recorded as subjects listened to natural speech, we train a model with a contrastive CLIP loss to align EEG-derived embeddings with embeddings from a pre-trained transformer-based speech model. Building on the state-of-the-art EEG decoder from Meta, we introduce three architectural modifications: (i) subject-specific attention layers (+0.15% WER improvement), (ii) personalized spatial attention (+0.45%), and (iii) a dual-path RNN with attention (-1.87%). Two of the three modifications improved performance, highlighting the promise of personalized architectures for brain-to-speech decoding and applications in brain-computer interfaces.
>
---
#### [new 048] Automatización de Informes Geotécnicos para Macizos Rocosos con IA
- **分类: cs.MM; cs.CL**

- **简介: 该论文面向岩体地质报告自动生成任务，解决传统人工报告效率低、主观性强的问题。通过多模态大语言模型处理岩层图像与数据，结合提示工程生成结构化报告，无需微调即达专家水平，提供Web工具提升地质工作效能。**

- **链接: [http://arxiv.org/pdf/2511.04690v1](http://arxiv.org/pdf/2511.04690v1)**

> **作者:** Christofer Valencia; Alexis Llumigusín; Silvia Alvarez; Abrahan Arias; Christian Mejia-Escobar
>
> **备注:** 17 pages, in Spanish language
>
> **摘要:** Geotechnical reports are crucial for assessing the stability of rock formations and ensuring safety in modern engineering. Traditionally, these reports are prepared manually based on field observations using compasses, magnifying glasses, and notebooks. This method is slow, prone to errors, and subjective in its interpretations. To overcome these limitations, the use of artificial intelligence techniques is proposed for the automatic generation of reports through the processing of images and field data. The methodology was based on the collection of photographs of rock outcrops and manual samples with their respective descriptions, as well as on the reports prepared during the Geotechnical Studies course. These resources were used to define the report outline, prompt engineering, and validate the responses of a multimodal large language model (MLLM). The iterative refinement of prompts until structured and specific instructions were obtained for each section of the report proved to be an effective alternative to the costly process of fine-tuning the MLLM. The system evaluation establishes values of 0.455 and 0.653 for the BLEU and ROUGE-L metrics, respectively, suggesting that automatic descriptions are comparable to those made by experts. This tool, accessible via the web, with an intuitive interface and the ability to export to standardized formats, represents an innovation and an important contribution for professionals and students of field geology.
>
---
#### [new 049] Quantifying the Climate Risk of Generative AI: Region-Aware Carbon Accounting with G-TRACE and the AI Sustainability Pyramid
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出G-TRACE框架，量化生成式AI在不同地区和模态下的碳排放，并揭示其气候风险；据此构建AI可持续性金字塔，为绿色AI部署提供政策指导。**

- **链接: [http://arxiv.org/pdf/2511.04776v1](http://arxiv.org/pdf/2511.04776v1)**

> **作者:** Zahida Kausar; Seemab Latif; Raja Khurrum Shahzad; Mehwish Fatima
>
> **备注:** 27 page, 4 figures
>
> **摘要:** Generative Artificial Intelligence (GenAI) represents a rapidly expanding digital infrastructure whose energy demand and associated CO2 emissions are emerging as a new category of climate risk. This study introduces G-TRACE (GenAI Transformative Carbon Estimator), a cross-modal, region-aware framework that quantifies training- and inference-related emissions across modalities and deployment geographies. Using real-world analytics and microscopic simulation, G-TRACE measures energy use and carbon intensity per output type (text, image, video) and reveals how decentralized inference amplifies small per-query energy costs into system-level impacts. Through the Ghibli-style image generation trend (2024-2025), we estimate 4,309 MWh of energy consumption and 2,068 tCO2 emissions, illustrating how viral participation inflates individual digital actions into tonne-scale consequences. Building on these findings, we propose the AI Sustainability Pyramid, a seven-level governance model linking carbon accounting metrics (L1-L7) with operational readiness, optimization, and stewardship. This framework translates quantitative emission metrics into actionable policy guidance for sustainable AI deployment. The study contributes to the quantitative assessment of emerging digital infrastructures as a novel category of climate risk, supporting adaptive governance for sustainable technology deployment. By situating GenAI within climate-risk frameworks, the work advances data-driven methods for aligning technological innovation with global decarbonization and resilience objectives.
>
---
#### [new 050] Language Generation and Identification From Partial Enumeration: Tight Density Bounds and Topological Characterizations
- **分类: cs.DS; cs.CL; cs.DM; cs.LG**

- **简介: 该论文研究语言生成与识别的极限模型，解决部分枚举下生成密度的最优界问题，证明最佳下密度为1/2，并拓展至部分信息场景，同时给出Angluin识别模型的新拓扑表征。**

- **链接: [http://arxiv.org/pdf/2511.05295v1](http://arxiv.org/pdf/2511.05295v1)**

> **作者:** Jon Kleinberg; Fan Wei
>
> **摘要:** The success of large language models (LLMs) has motivated formal theories of language generation and learning. We study the framework of \emph{language generation in the limit}, where an adversary enumerates strings from an unknown language $K$ drawn from a countable class, and an algorithm must generate unseen strings from $K$. Prior work showed that generation is always possible, and that some algorithms achieve positive lower density, revealing a \emph{validity--breadth} trade-off between correctness and coverage. We resolve a main open question in this line, proving a tight bound of $1/2$ on the best achievable lower density. We then strengthen the model to allow \emph{partial enumeration}, where the adversary reveals only an infinite subset $C \subseteq K$. We show that generation in the limit remains achievable, and if $C$ has lower density $\alpha$ in $K$, the algorithm's output achieves density at least $\alpha/2$, matching the upper bound. This generalizes the $1/2$ bound to the partial-information setting, where the generator must recover within a factor $1/2$ of the revealed subset's density. We further revisit the classical Gold--Angluin model of \emph{language identification} under partial enumeration. We characterize when identification in the limit is possible -- when hypotheses $M_t$ eventually satisfy $C \subseteq M \subseteq K$ -- and in the process give a new topological formulation of Angluin's characterization, showing that her condition is precisely equivalent to an appropriate topological space having the $T_D$ separation property.
>
---
#### [new 051] ConVerse: Benchmarking Contextual Safety in Agent-to-Agent Conversations
- **分类: cs.CR; cs.CL; cs.CY**

- **简介: 论文提出ConVerse基准，评估多智能体对话中的隐私与安全风险，解决智能体间信息共享与防护的矛盾。通过12种用户画像和864个上下文攻击，揭示主流模型在多轮交互中高达88%的隐私泄露与60%的安全漏洞。**

- **链接: [http://arxiv.org/pdf/2511.05359v1](http://arxiv.org/pdf/2511.05359v1)**

> **作者:** Amr Gomaa; Ahmed Salem; Sahar Abdelnabi
>
> **摘要:** As language models evolve into autonomous agents that act and communicate on behalf of users, ensuring safety in multi-agent ecosystems becomes a central challenge. Interactions between personal assistants and external service providers expose a core tension between utility and protection: effective collaboration requires information sharing, yet every exchange creates new attack surfaces. We introduce ConVerse, a dynamic benchmark for evaluating privacy and security risks in agent-agent interactions. ConVerse spans three practical domains (travel, real estate, insurance) with 12 user personas and over 864 contextually grounded attacks (611 privacy, 253 security). Unlike prior single-agent settings, it models autonomous, multi-turn agent-to-agent conversations where malicious requests are embedded within plausible discourse. Privacy is tested through a three-tier taxonomy assessing abstraction quality, while security attacks target tool use and preference manipulation. Evaluating seven state-of-the-art models reveals persistent vulnerabilities; privacy attacks succeed in up to 88% of cases and security breaches in up to 60%, with stronger models leaking more. By unifying privacy and security within interactive multi-agent contexts, ConVerse reframes safety as an emergent property of communication.
>
---
#### [new 052] Stateful KV Cache Management for LLMs: Balancing Space, Time, Accuracy, and Positional Fidelity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM有状态推理中的KV缓存管理，解决因缓存超限和位置编码失真导致的生成质量下降问题，提出应保留连续上下文块以维护位置一致性，而非仅优化缓存大小。**

- **链接: [http://arxiv.org/pdf/2511.04686v1](http://arxiv.org/pdf/2511.04686v1)**

> **作者:** Pratik Poudel
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** The Key-Value (KV) cache is integral to efficient autoregressive inference in large language models (LLMs), yet its unbounded growth in stateful multi-turn scenarios presents major challenges. This paper examines the interplay between KV cache management strategies, the architectural context limits of models like meta-llama/Meta-Llama-3-8b-instruct, and the often-overlooked integrity of positional encodings. Through empirical analysis using a stateful benchmarking framework, we show that LLM generation quality degrades sharply when the accumulated KV cache approaches or exceeds the model's trained context window (e.g., 8192 tokens for Llama 3), a failure mode distinct from GPU memory exhaustion. Common eviction strategies, even high-retention ones (e.g., 99% via AttentionTop), can worsen performance if they disrupt positional coherence. Because LLMs rely on consistent positional signals (e.g., RoPE), compacting a cache by removing non-contiguous tokens can scramble these signals and lead to degenerative outputs. We further show that simple strategies preserving contiguous context blocks (e.g., keeping an initial "gist") can yield more coherent generations than complex or positionally disruptive ones. We advocate for eviction techniques that respect architectural limits, preserve positional structure, and view "cache health" holistically beyond mere size.
>
---
#### [new 053] APP: Accelerated Path Patching with Task-Specific Pruning
- **分类: cs.LG; cs.AI; cs.CL; 68Uxx; I.2.7; I.2.6; I.2.m**

- **简介: 该论文提出APP方法，用于加速电路发现任务。通过对比性注意力头剪枝（Contrastive-FLAP）压缩搜索空间，再结合路径修补，显著提升效率（提速59.63%–93.27%），同时保持电路性能与完整性。**

- **链接: [http://arxiv.org/pdf/2511.05442v1](http://arxiv.org/pdf/2511.05442v1)**

> **作者:** Frauke Andersen; William Rudman; Ruochen Zhang; Carsten Eickhoff
>
> **摘要:** Circuit discovery is a key step in many mechanistic interpretability pipelines. Current methods, such as Path Patching, are computationally expensive and have limited in-depth circuit analysis for smaller models. In this study, we propose Accelerated Path Patching (APP), a hybrid approach leveraging our novel contrastive attention head pruning method to drastically reduce the search space of circuit discovery methods. Our Contrastive-FLAP pruning algorithm uses techniques from causal mediation analysis to assign higher pruning scores to task-specific attention heads, leading to higher performing sparse models compared to traditional pruning techniques. Although Contrastive-FLAP is successful at preserving task-specific heads that existing pruning algorithms remove at low sparsity ratios, the circuits found by Contrastive-FLAP alone are too large to satisfy the minimality constraint required in circuit analysis. APP first applies Contrastive-FLAP to reduce the search space on required for circuit discovery algorithms by, on average, 56\%. Next, APP, applies traditional Path Patching on the remaining attention heads, leading to a speed up of 59.63\%-93.27\% compared to Path Patching applied to the dense model. Despite the substantial computational saving that APP provides, circuits obtained from APP exhibit substantial overlap and similar performance to previously established Path Patching circuits
>
---
#### [new 054] Wikipedia-based Datasets in Russian Information Retrieval Benchmark RusBEIR
- **分类: cs.IR; cs.CL**

- **简介: 该论文构建了基于俄语维基百科“你知道吗”栏目的新IR数据集RusBEIR，支持事实核查、检索增强生成等任务，旨在解决俄语信息检索资源匮乏问题。实验对比了词法与神经模型性能，揭示了文档长度对检索的影响，并开源了数据与代码。**

- **链接: [http://arxiv.org/pdf/2511.05079v1](http://arxiv.org/pdf/2511.05079v1)**

> **作者:** Grigory Kovalev; Natalia Loukachevitch; Mikhail Tikhomirov; Olga Babina; Pavel Mamaev
>
> **摘要:** In this paper, we present a novel series of Russian information retrieval datasets constructed from the "Did you know..." section of Russian Wikipedia. Our datasets support a range of retrieval tasks, including fact-checking, retrieval-augmented generation, and full-document retrieval, by leveraging interesting facts and their referenced Wikipedia articles annotated at the sentence level with graded relevance. We describe the methodology for dataset creation that enables the expansion of existing Russian Information Retrieval (IR) resources. Through extensive experiments, we extend the RusBEIR research by comparing lexical retrieval models, such as BM25, with state-of-the-art neural architectures fine-tuned for Russian, as well as multilingual models. Results of our experiments show that lexical methods tend to outperform neural models on full-document retrieval, while neural approaches better capture lexical semantics in shorter texts, such as in fact-checking or fine-grained retrieval. Using our newly created datasets, we also analyze the impact of document length on retrieval performance and demonstrate that combining retrieval with neural reranking consistently improves results. Our contribution expands the resources available for Russian information retrieval research and highlights the importance of accurate evaluation of retrieval models to achieve optimal performance. All datasets are publicly available at HuggingFace. To facilitate reproducibility and future research, we also release the full implementation on GitHub.
>
---
#### [new 055] Association via Entropy Reduction
- **分类: cs.IR; cs.CL; H.3.3**

- **简介: 该论文提出一种基于熵减的关联评分方法aver，用于识别文档或图中顶点的关联对，相比传统tf-idf，aver具备自然阈值、可扩展性与理论基础，虽计算更复杂，但在特定任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2511.04901v1](http://arxiv.org/pdf/2511.04901v1)**

> **作者:** Anthony Gamst; Lawrence Wilson
>
> **摘要:** Prior to recent successes using neural networks, term frequency-inverse document frequency (tf-idf) was clearly regarded as the best choice for identifying documents related to a query. We provide a different score, aver, and observe, on a dataset with ground truth marking for association, that aver does do better at finding assciated pairs than tf-idf. This example involves finding associated vertices in a large graph and that may be an area where neural networks are not currently an obvious best choice. Beyond this one anecdote, we observe that (1) aver has a natural threshold for declaring pairs as unassociated while tf-idf does not, (2) aver can distinguish between pairs of documents for which tf-idf gives a score of 1.0, (3) aver can be applied to larger collections of documents than pairs while tf-idf cannot, and (4) that aver is derived from entropy under a simple statistical model while tf-idf is a construction designed to achieve a certain goal and hence aver may be more "natural." To be fair, we also observe that (1) writing down and computing the aver score for a pair is more complex than for tf-idf and (2) that the fact that the aver score is naturally scale-free makes it more complicated to interpret aver scores.
>
---
#### [new 056] QUESTER: Query Specification for Generative Retrieval
- **分类: cs.IR; cs.CL; cs.LG; 68P20, 68T50; H.3**

- **简介: 论文提出QUESTER，将生成式检索（GR）转化为用小LLM生成关键词查询，再由BM25检索，通过RL训练提升泛化与效率，解决GR难扩展、成本高的问题，性能媲美神经检索模型。**

- **链接: [http://arxiv.org/pdf/2511.05301v1](http://arxiv.org/pdf/2511.05301v1)**

> **作者:** Arthur Satouf; Yuxuan Zong; Habiboulaye Amadou-Boubacar; Pablo Piantanida; Benjamin Piwowarski
>
> **摘要:** Generative Retrieval (GR) differs from the traditional index-then-retrieve pipeline by storing relevance in model parameters and directly generating document identifiers. However, GR often struggles to generalize and is costly to scale. We introduce QUESTER (QUEry SpecificaTion gEnerative Retrieval), which reframes GR as query specification generation - in this work, a simple keyword query handled by BM25 - using a (small) LLM. The policy is trained using reinforcement learning techniques (GRPO). Across in- and out-of-domain evaluations, we show that our model is more effective than BM25, and competitive with neural IR models, while maintaining a good efficiency
>
---
#### [new 057] Jailbreaking in the Haystack
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出NINJA攻击方法，针对长上下文语言模型的安全漏洞，通过在有害请求后附加 benign 内容并优化其位置，实现高效、隐蔽的越狱，显著提升攻击成功率，揭示长上下文带来的新型安全风险。**

- **链接: [http://arxiv.org/pdf/2511.04707v1](http://arxiv.org/pdf/2511.04707v1)**

> **作者:** Rishi Rajesh Shah; Chen Henry Wu; Shashwat Saxena; Ziqian Zhong; Alexander Robey; Aditi Raghunathan
>
> **摘要:** Recent advances in long-context language models (LMs) have enabled million-token inputs, expanding their capabilities across complex tasks like computer-use agents. Yet, the safety implications of these extended contexts remain unclear. To bridge this gap, we introduce NINJA (short for Needle-in-haystack jailbreak attack), a method that jailbreaks aligned LMs by appending benign, model-generated content to harmful user goals. Critical to our method is the observation that the position of harmful goals play an important role in safety. Experiments on standard safety benchmark, HarmBench, show that NINJA significantly increases attack success rates across state-of-the-art open and proprietary models, including LLaMA, Qwen, Mistral, and Gemini. Unlike prior jailbreaking methods, our approach is low-resource, transferable, and less detectable. Moreover, we show that NINJA is compute-optimal -- under a fixed compute budget, increasing context length can outperform increasing the number of trials in best-of-N jailbreak. These findings reveal that even benign long contexts -- when crafted with careful goal positioning -- introduce fundamental vulnerabilities in modern LMs.
>
---
#### [new 058] Simulating Misinformation Vulnerabilities With Agent Personas
- **分类: cs.SI; cs.AI; cs.CL**

- **简介: 该论文利用大语言模型构建具有不同职业与认知图式的智能体，模拟其对虚假信息的反应，解决真实实验难以开展的问题，验证LLM可作为社会信息传播研究的有效代理，揭示认知图式比职业背景更影响信息解读。**

- **链接: [http://arxiv.org/pdf/2511.04697v1](http://arxiv.org/pdf/2511.04697v1)**

> **作者:** David Farr; Lynnette Hui Xian Ng; Stephen Prochaska; Iain J. Cruickshank; Jevin West
>
> **备注:** Accepted to Winter Simulation Conference 2025
>
> **摘要:** Disinformation campaigns can distort public perception and destabilize institutions. Understanding how different populations respond to information is crucial for designing effective interventions, yet real-world experimentation is impractical and ethically challenging. To address this, we develop an agent-based simulation using Large Language Models (LLMs) to model responses to misinformation. We construct agent personas spanning five professions and three mental schemas, and evaluate their reactions to news headlines. Our findings show that LLM-generated agents align closely with ground-truth labels and human predictions, supporting their use as proxies for studying information responses. We also find that mental schemas, more than professional background, influence how agents interpret misinformation. This work provides a validation of LLMs to be used as agents in an agent-based model of an information network for analyzing trust, polarization, and susceptibility to deceptive content in complex social systems.
>
---
#### [new 059] Enhancing Public Speaking Skills in Engineering Students Through AI
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出一种多模态AI系统，用于评估工程学生公开演讲的言语与非言语表现，解决人工反馈成本高、个性化不足的问题。通过融合语音、视觉与情感分析，实现自动、可扩展的实时反馈，Gemini Pro表现最优。**

- **链接: [http://arxiv.org/pdf/2511.04995v1](http://arxiv.org/pdf/2511.04995v1)**

> **作者:** Amol Harsh; Brainerd Prince; Siddharth Siddharth; Deepan Raj Prabakar Muthirayan; Kabir S Bhalla; Esraaj Sarkar Gupta; Siddharth Sahu
>
> **摘要:** This research-to-practice full paper was inspired by the persistent challenge in effective communication among engineering students. Public speaking is a necessary skill for future engineers as they have to communicate technical knowledge with diverse stakeholders. While universities offer courses or workshops, they are unable to offer sustained and personalized training to students. Providing comprehensive feedback on both verbal and non-verbal aspects of public speaking is time-intensive, making consistent and individualized assessment impractical. This study integrates research on verbal and non-verbal cues in public speaking to develop an AI-driven assessment model for engineering students. Our approach combines speech analysis, computer vision, and sentiment detection into a multi-modal AI system that provides assessment and feedback. The model evaluates (1) verbal communication (pitch, loudness, pacing, intonation), (2) non-verbal communication (facial expressions, gestures, posture), and (3) expressive coherence, a novel integration ensuring alignment between speech and body language. Unlike previous systems that assess these aspects separately, our model fuses multiple modalities to deliver personalized, scalable feedback. Preliminary testing demonstrated that our AI-generated feedback was moderately aligned with expert evaluations. Among the state-of-the-art AI models evaluated, all of which were Large Language Models (LLMs), including Gemini and OpenAI models, Gemini Pro emerged as the best-performing, showing the strongest agreement with human annotators. By eliminating reliance on human evaluators, this AI-driven public speaking trainer enables repeated practice, helping students naturally align their speech with body language and emotion, crucial for impactful and professional communication.
>
---
## 更新

#### [replaced 001] Exploring Multimodal Perception in Large Language Models Through Perceptual Strength Ratings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06980v2](http://arxiv.org/pdf/2503.06980v2)**

> **作者:** Jonghyun Lee; Dojun Park; Jiwoo Lee; Hoekeon Choi; Sung-Eun Lee
>
> **备注:** Published in IEEE Access
>
> **摘要:** This study investigated whether multimodal large language models can achieve human-like sensory grounding by examining their ability to capture perceptual strength ratings across sensory modalities. We explored how model characteristics (size, multimodal capabilities, architectural generation) influence grounding performance, distributional factor dependencies (word frequency, embeddings, feature distances), and human-model processing differences. We evaluated 21 models from four families (GPT, Gemini, LLaMA, Qwen) using 3,611 words from the Lancaster Sensorimotor Norms through correlation, distance metrics, and qualitative analysis. Results showed that larger (6 out of 8 comparisons), multimodal (5 of 7), and newer models (5 of 8) generally outperformed their smaller, text-based, and older counterparts. Top models achieved 85-90% accuracy and 0.58-0.65 correlations with human ratings, demonstrating substantial similarity. Moreover, distributional factors showed minimal impact, not exceeding human dependency levels. However, despite strong alignment, models were not identical to humans, as even top performers showed differences in distance and correlation measures, with qualitative analysis revealing processing patterns related to absent sensory grounding. Additionally, it remains questionable whether introducing multimodality resolves this grounding deficit. Although multimodality improved performance, it seems to provide similar information as massive text rather than qualitatively different data, as benefits occurred across unrelated sensory dimensions and massive text-only models achieved comparable results. Our findings demonstrate that while advanced LLMs can approximate human sensory-linguistic associations through statistical learning, they still differ from human embodied cognition in processing mechanisms, even with multimodal integration.
>
---
#### [replaced 002] AIRepr: An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science
- **分类: cs.LG; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.16395v3](http://arxiv.org/pdf/2502.16395v3)**

> **作者:** Qiuhai Zeng; Claire Jin; Xinyue Wang; Yuhan Zheng; Qunhua Li
>
> **备注:** Accepted to 2025 EMNLP findings
>
> **摘要:** Large language models (LLMs) are increasingly used to automate data analysis through executable code generation. Yet, data science tasks often admit multiple statistically valid solutions, e.g. different modeling strategies, making it critical to understand the reasoning behind analyses, not just their outcomes. While manual review of LLM-generated code can help ensure statistical soundness, it is labor-intensive and requires expertise. A more scalable approach is to evaluate the underlying workflows-the logical plans guiding code generation. However, it remains unclear how to assess whether an LLM-generated workflow supports reproducible implementations. To address this, we present AIRepr, an Analyst-Inspector framework for automatically evaluating and improving the reproducibility of LLM-generated data analysis workflows. Our framework is grounded in statistical principles and supports scalable, automated assessment. We introduce two novel reproducibility-enhancing prompting strategies and benchmark them against standard prompting across 15 analyst-inspector LLM pairs and 1,032 tasks from three public benchmarks. Our findings show that workflows with higher reproducibility also yield more accurate analyses, and that reproducibility-enhancing prompts substantially improve both metrics. This work provides a foundation for transparent, reliable, and efficient human-AI collaboration in data science. Our code is publicly available.
>
---
#### [replaced 003] Retrieval-Augmented Review Generation for Poisoning Recommender Systems
- **分类: cs.CR; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.15252v2](http://arxiv.org/pdf/2508.15252v2)**

> **作者:** Shiyi Yang; Xinshu Li; Guanglin Zhou; Chen Wang; Xiwei Xu; Liming Zhu; Lina Yao
>
> **摘要:** Recent studies have shown that recommender systems (RSs) are highly vulnerable to data poisoning attacks, where malicious actors inject fake user profiles, including a group of well-designed fake ratings, to manipulate recommendations. Due to security and privacy constraints in practice, attackers typically possess limited knowledge of the victim system and thus need to craft profiles that have transferability across black-box RSs. To maximize the attack impact, the profiles often remains imperceptible. However, generating such high-quality profiles with the restricted resources is challenging. Some works suggest incorporating fake textual reviews to strengthen the profiles; yet, the poor quality of the reviews largely undermines the attack effectiveness and imperceptibility under the practical setting. To tackle the above challenges, in this paper, we propose to enhance the quality of the review text by harnessing in-context learning (ICL) capabilities of multimodal foundation models. To this end, we introduce a demonstration retrieval algorithm and a text style transfer strategy to augment the navie ICL. Specifically, we propose a novel practical attack framework named RAGAN to generate high-quality fake user profiles, which can gain insights into the robustness of RSs. The profiles are generated by a jailbreaker and collaboratively optimized on an instructional agent and a guardian to improve the attack transferability and imperceptibility. Comprehensive experiments on various real-world datasets demonstrate that RAGAN achieves the state-of-the-art poisoning attack performance.
>
---
#### [replaced 004] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13142v3](http://arxiv.org/pdf/2508.13142v3)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence. We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence (SI), yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail even the most advanced multimodal models.
>
---
#### [replaced 005] Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.22115v2](http://arxiv.org/pdf/2510.22115v2)**

> **作者:** Ling Team; Ang Li; Ben Liu; Binbin Hu; Bing Li; Bingwei Zeng; Borui Ye; Caizhi Tang; Changxin Tian; Chao Huang; Chao Zhang; Chen Qian; Chenchen Ju; Chenchen Li; Chengfu Tang; Chilin Fu; Chunshao Ren; Chunwei Wu; Cong Zhang; Cunyin Peng; Dafeng Xu; Daixin Wang; Dalong Zhang; Dingnan Jin; Dingyuan Zhu; Dongke Hu; Fangzheng Zhao; Feifan Wu; Feng Zhu; Gangshan Wang; Haitao Zhang; Hailin Zhao; Hanxiao Zhang; Hanzi Wang; Hao Qian; Haoyi Yu; Heng Zhang; Hongliang Zhang; Hongzhi Luan; Huirong Dong; Huizhong Li; Jia Li; Jia Liu; Jialong Zhu; Jian Sha; Jianping Wei; Jiaolong Yang; Jieyue Ma; Jiewei Wu; Jinjing Huang; Jingyun Tian; Jingyuan Zhang; Jinquan Sun; Juanhui Tu; Jun Liu; Jun Xu; Jun Zhou; Junjie Ou; Junpeng Fang; Kaihong Zhang; Kaiqin Hu; Ke Shi; Kun Tang; Kunlong Chen; Lanyin Mei; Lei Liang; Lei Xu; Libo Zhang; Lin Ju; Lin Yuan; Ling Zhong; Lintao Ma; Lu Liu; Lu Yu; Lun Cai; Meiqi Zhu; Mengying Li; Min Chen; Minghao Xue; Minghong Cai; Mingming Yin; Peijie Jiang; Peilong Zhao; Pingping Liu; Qian Zhao; Qing Cui; Qingxiang Huang; Qingyuan Yang; Quankun Yu; Shaowei Wei; Shijie Lian; Shoujian Zheng; Shun Song; Shungen Zhang; Shuo Zhang; Siyuan Li; Song Liu; Ting Guo; Tong Zhao; Wanli Gu; Weichang Wu; Weiguang Han; Wenjing Fang; Wubin Wang; Xiang Shu; Xiao Shi; Xiaoshun Lan; Xiaolu Zhang; Xiaqing Sun; Xin Zhao; Xingyu Lu; Xiong Xu; Xudong Wang; Xudong Wang; Xuemin Yang; Yajie Yang; Yang Xiang; Yanzhe Li; Yi Zhang; Yilong Wang; Yingxue Li; Yongzhen Guo; Yuzhuo Fu; Yuanyuan Wang; Yue Yang; Yue Yu; Yufeng Deng; Yun Zhang; Yunfei Yu; Yuqi Zhang; Yuxiao He; Zengke Gui; Zhaoxin Huan; Zhaoyang Wang; Zhibo Zhu; Zhihao Wang; Zhiqiang Zhang; Zhoufei Wang; Zihang Zeng; Ziqi Liu; Zitao Xuan; Zuoli Tang
>
> **备注:** Ling 2.0 Technical Report
>
> **摘要:** We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, cross-scale consistency, and efficiency guided by empirical scaling laws. The series includes three non-thinking (instruct) models - Ling-mini-2.0, Ling-flash-2.0, and Ling-1T - ranging from 16B to 1T total parameters and achieving up to 7-fold active-compute efficiency compared with dense counterparts. Ling 2.0 integrates coordinated innovations across model architecture, pre-training, post-training, and infrastructure: a high-sparsity MoE with MTP for efficient reasoning, reasoning-oriented data and mid-training CoT activation, reinforcement-based fine-tuning (DFT, Evo-CoT), and full-scale FP8 training with fine-grained heterogeneous pipelines. At the trillion scale, Ling-1T establishes a new Pareto frontier of reasoning accuracy versus computational efficiency, demonstrating that sparse activation, when properly aligned with reasoning objectives, enables scalable and efficient intelligence. Collectively, Ling 2.0 provides a coherent, open, and efficient foundation for advancing future reasoning and thinking models, including the Ring series built upon the same base.
>
---
#### [replaced 006] ThaiOCRBench: A Task-Diverse Benchmark for Vision-Language Understanding in Thai
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.04479v2](http://arxiv.org/pdf/2511.04479v2)**

> **作者:** Surapon Nonesung; Teetouch Jaknamon; Sirinya Chaiophat; Natapong Nitarach; Chanakan Wittayasakpan; Warit Sirichotedumrong; Adisai Na-Thalang; Kunat Pipatanakul
>
> **备注:** Accepted at the IJCNLP-AACL 2025 (Main)
>
> **摘要:** We present ThaiOCRBench, the first comprehensive benchmark for evaluating vision-language models (VLMs) on Thai text-rich visual understanding tasks. Despite recent progress in multimodal modeling, existing benchmarks predominantly focus on high-resource languages, leaving Thai underrepresented, especially in tasks requiring document structure understanding. ThaiOCRBench addresses this gap by offering a diverse, human-annotated dataset comprising 2,808 samples across 13 task categories. We evaluate a wide range of state-of-the-art VLMs in a zero-shot setting, spanning both proprietary and open-source systems. Results show a significant performance gap, with proprietary models (e.g., Gemini 2.5 Pro) outperforming open-source counterparts. Notably, fine-grained text recognition and handwritten content extraction exhibit the steepest performance drops among open-source models. Through detailed error analysis, we identify key challenges such as language bias, structural mismatch, and hallucinated content. ThaiOCRBench provides a standardized framework for assessing VLMs in low-resource, script-complex settings, and provides actionable insights for improving Thai-language document understanding.
>
---
#### [replaced 007] Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.25441v2](http://arxiv.org/pdf/2510.25441v2)**

> **作者:** Fei Wei; Daoyuan Chen; Ce Wang; Yilun Huang; Yushuo Chen; Xuchen Pan; Yaliang Li; Bolin Ding
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) excel as passive responders, but teaching them to be proactive, goal-oriented partners, a critical capability in high-stakes domains, remains a major challenge. Current paradigms either myopically optimize single-turn attributes or rely on brittle, high-cost user simulators, creating a persistent ``reality gap''. To bridge this gap, we introduce \texttt{Learn-to-Ask}, a general, simulator-free framework for learning and deploying proactive dialogue agents \textit{directly from offline expert data}, bypassing the need to model complex user dynamics. Our key insight is to reframe the offline policy learning problem by leveraging the \textbf{observed future} of each expert trajectory. This allows us to infer a dense, turn-by-turn reward signal grounded in the expert's revealed strategy, decomposing the intractable long-horizon problem into a series of supervised learning tasks, and training a policy to output a structured \texttt{(action, state_assessment)} tuple, governing both \textbf{what to ask} and, crucially, \textbf{when to stop}. To ensure reward fidelity, our Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision. Empirically, we demonstrate the efficacy of \texttt{Learn-to-Ask} in a real-world medical dataset, using LLMs of varying sizes up to 32B. Our approach culminates in the successful deployment of LLMs into a live, large-scale online AI service. In rigorous in-house evaluations, our model was launched and achieved performance even superior to human experts, proving our framework's ability to translate offline data into tangible, real-world impact. We hope this work provides a practical and economically viable blueprint for transforming passive LLMs into proactive, goal-oriented LLM applications.
>
---
#### [replaced 008] Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17797v2](http://arxiv.org/pdf/2510.17797v2)**

> **作者:** Akshara Prabhakar; Roshan Ram; Zixiang Chen; Silvio Savarese; Frank Wang; Caiming Xiong; Huan Wang; Weiran Yao
>
> **备注:** Technical report; 13 pages plus references and appendices
>
> **摘要:** As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications. Code at https://github.com/SalesforceAIResearch/enterprise-deep-research and Dataset at https://huggingface.co/datasets/Salesforce/EDR-200
>
---
#### [replaced 009] MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09360v2](http://arxiv.org/pdf/2509.09360v2)**

> **作者:** Channdeth Sok; David Luz; Yacine Haddam
>
> **备注:** Identity-Aware AI workshop at 28th European Conference on Artificial Intelligence, October 25, 2025, Bologna, Italy
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in enterprise applications, yet their reliability remains limited by hallucinations, i.e., confident but factually incorrect information. Existing detection approaches, such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not address the unique challenges of Retrieval-Augmented Generation (RAG) systems, where responses must be consistent with retrieved evidence. We therefore present MetaRAG, a metamorphic testing framework for hallucination detection in Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time, unsupervised, black-box setting, requiring neither ground-truth references nor access to model internals, making it suitable for proprietary and high-stakes domains. The framework proceeds in four stages: (1) decompose answers into atomic factoids, (2) generate controlled mutations of each factoid using synonym and antonym substitutions, (3) verify each variant against the retrieved context (synonyms are expected to be entailed and antonyms contradicted), and (4) aggregate penalties for inconsistencies into a response-level hallucination score. Crucially for identity-aware AI, MetaRAG localizes unsupported claims at the factoid span where they occur (e.g., pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility), allowing users to see flagged spans and enabling system designers to configure thresholds and guardrails for identity-sensitive queries. Experiments on a proprietary enterprise dataset illustrate the effectiveness of MetaRAG for detecting hallucinations and enabling trustworthy deployment of RAG-based conversational agents. We also outline a topic-based deployment design that translates MetaRAG's span-level scores into identity-aware safeguards; this design is discussed but not evaluated in our experiments.
>
---
#### [replaced 010] Fine-Tuning MedGemma for Clinical Captioning to Enhance Multimodal RAG over Malaysia CPGs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.15418v2](http://arxiv.org/pdf/2510.15418v2)**

> **作者:** Lee Qi Zun; Mohamad Zulhilmi Bin Abdul Halim; Goh Man Fye
>
> **摘要:** Retrieval-Augmented Generation systems are essential for providing fact-based guidance from Malaysian Clinical Practice Guidelines. However, their effectiveness with image-based queries is limited, as general Vision-Language Model captions often lack clinical specificity and factual grounding. This study proposes and validates a framework to specialize the MedGemma model for generating high-fidelity captions that serve as superior queries. To overcome data scarcity, we employ a knowledge distillation pipeline to create a synthetic dataset across dermatology, fundus, and chest radiography domains, and fine-tune MedGemma using the parameter-efficient QLoRA method. Performance was rigorously assessed through a dual framework measuring both classification accuracy and, via a novel application of the RAGAS framework, caption faithfulness, relevancy, and correctness. The fine-tuned model demonstrated substantial improvements in classification performance, while RAGAS evaluation confirmed significant gains in caption faithfulness and correctness, validating the models ability to produce reliable, factually grounded descriptions. This work establishes a robust pipeline for specializing medical VLMs and validates the resulting model as a high-quality query generator, laying the groundwork for enhancing multimodal RAG systems in evidence-based clinical decision support.
>
---
#### [replaced 011] Learning Dynamics of Meta-Learning in Small Model Pretraining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.02189v2](http://arxiv.org/pdf/2508.02189v2)**

> **作者:** David Demitri Africa; Yuval Weiss; Paula Buttery; Richard Diehl Martinez
>
> **备注:** Accepted (oral) to Student Research Workshop at IJCNLP-AACL 2025
>
> **摘要:** Large language models are powerful but costly. We ask whether meta-learning can make the pretraining of small language models not only better but also more interpretable. We integrate first-order MAML with subset-masked LM pretraining, producing four LLama-style decoder-only models (11M-570M params), and evaluate it on a fundamental NLP task with many settings and real-world applications. Compared with vanilla training, our model (i) reaches the same loss up to 1.6x sooner, (ii) improves F1 on multilingual Universal NER under equal compute, and (iii) makes the training dynamics easy to read: first the network's representations fan out ("diversify") and later they collapse into a smaller, shared subspace ("compress"). This two-stage shift shows up as a rise-and-fall in both effective-rank curves and attention-head entropy. The same curves pinpoint which layers specialise earliest and which later reconverge, giving a compact, interpretable signature of meta-adaptation. Code, checkpoints and WandB logs are released.
>
---
#### [replaced 012] Optimizing Anytime Reasoning via Budget Relative Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13438v3](http://arxiv.org/pdf/2505.13438v3)**

> **作者:** Penghui Qi; Zichen Liu; Tianyu Pang; Chao Du; Wee Sun Lee; Min Lin
>
> **摘要:** Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency.
>
---
#### [replaced 013] Extracting narrative signals from public discourse: a network-based approach
- **分类: cs.CL; cs.CY; cs.IR; cs.SI**

- **链接: [http://arxiv.org/pdf/2411.00702v2](http://arxiv.org/pdf/2411.00702v2)**

> **作者:** Armin Pournaki; Tom Willaert
>
> **备注:** 27 pages, 6 figures
>
> **摘要:** Narratives are key interpretative devices by which humans make sense of political reality. As the significance of narratives for understanding current societal issues such as polarization and misinformation becomes increasingly evident, there is a growing demand for methods that support their empirical analysis. To this end, we propose a graph-based formalism and machine-guided method for extracting, representing, and analyzing selected narrative signals from digital textual corpora, based on Abstract Meaning Representation (AMR). The formalism and method introduced here specifically cater to the study of political narratives that figure in texts from digital media such as archived political speeches, social media posts, transcripts of parliamentary debates, and political manifestos on party websites. We approach the study of such political narratives as a problem of information retrieval: starting from a textual corpus, we first extract a graph-like representation of the meaning of each sentence in the corpus using AMR. Drawing on transferable concepts from narratology, we then apply a set of heuristics to filter these graphs for representations of 1) actors and their relationships, 2) the events in which these actors figure, and 3) traces of the perspectivization of these events. We approach these references to actors, events, and instances of perspectivization as core narrative signals that allude to larger political narratives. By systematically analyzing and re-assembling these signals into networks that guide the researcher to the relevant parts of the text, the underlying narratives can be reconstructed through a combination of distant and close reading. A case study of State of the European Union addresses (2010 -- 2023) demonstrates how the formalism can be used to inductively surface signals of political narratives from public discourse.
>
---
#### [replaced 014] InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback
- **分类: cs.CL; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.15027v3](http://arxiv.org/pdf/2502.15027v3)**

> **作者:** Henry Hengyuan Zhao; Wenqi Pei; Yifei Tao; Haiyang Mei; Mike Zheng Shou
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Existing benchmarks do not test Large Multimodal Models (LMMs) on their interactive intelligence with human users, which is vital for developing general-purpose AI assistants. We design InterFeedback, an interactive framework, which can be applied to any LMM and dataset to assess this ability autonomously. On top of this, we introduce InterFeedback-Bench which evaluates interactive intelligence using two representative datasets, MMMU-Pro and MathVerse, to test 10 different open-source LMMs. Additionally, we present InterFeedback-Human, a newly collected dataset of 120 cases designed for manually testing interactive performance in leading models such as OpenAI-o1 and Claude-Sonnet-4. Our evaluation results indicate that even the state-of-the-art LMM, OpenAI-o1, struggles to refine its responses based on human feedback, achieving an average score of less than 50%. Our findings point to the need for methods that can enhance LMMs' capabilities to interpret and benefit from feedback.
>
---
#### [replaced 015] NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual eXploration of Finance
- **分类: cs.CL; cs.AI; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2507.09601v2](http://arxiv.org/pdf/2507.09601v2)**

> **作者:** Hanwool Lee; Sara Yu; Yewon Hwang; Jonghyun Choi; Heejae Ahn; Sungbum Jung; Youngjae Yu
>
> **备注:** Accepted at FinAI@CIKM 2025
>
> **摘要:** General-purpose sentence embedding models often struggle to capture specialized financial semantics, especially in low-resource languages like Korean, due to domain-specific jargon, temporal meaning shifts, and misaligned bilingual vocabularies. To address these gaps, we introduce NMIXX (Neural eMbeddings for Cross-lingual eXploration of Finance), a suite of cross-lingual embedding models fine-tuned with 18.8K high-confidence triplets that pair in-domain paraphrases, hard negatives derived from a semantic-shift typology, and exact Korean-English translations. Concurrently, we release KorFinSTS, a 1,921-pair Korean financial STS benchmark spanning news, disclosures, research reports, and regulations, designed to expose nuances that general benchmarks miss. When evaluated against seven open-license baselines, NMIXX's multilingual bge-m3 variant achieves Spearman's rho gains of +0.10 on English FinSTS and +0.22 on KorFinSTS, outperforming its pre-adaptation checkpoint and surpassing other models by the largest margin, while revealing a modest trade-off in general STS performance. Our analysis further shows that models with richer Korean token coverage adapt more effectively, underscoring the importance of tokenizer design in low-resource, cross-lingual settings. By making both models and the benchmark publicly available, we provide the community with robust tools for domain-adapted, multilingual representation learning in finance.
>
---
#### [replaced 016] Inference-Time Hyper-Scaling with KV Cache Compression
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05345v2](http://arxiv.org/pdf/2506.05345v2)**

> **作者:** Adrian Łańcucki; Konrad Staniszewski; Piotr Nawrot; Edoardo M. Ponti
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Inference-time scaling trades efficiency for increased reasoning accuracy by generating longer or more parallel sequences. However, in Transformer LLMs, generation cost is bottlenecked by the size of the key-value (KV) cache, rather than the number of generated tokens. Hence, we explore inference-time hyper-scaling: by compressing the KV cache, we can generate more tokens within the same compute budget and further improve the accuracy of scaled inference. The success of this approach, however, hinges on the ability of compression methods to preserve accuracy even at high compression ratios. To make hyper-scaling practical, we introduce Dynamic Memory Sparsification (DMS), a novel method for sparsifying KV caches that only requires 1K training steps to achieve 8$\times$ compression, while maintaining better accuracy than training-free sparse attention. Instead of prematurely discarding cached tokens, DMS delays token eviction, implicitly merging representations and preserving critical information. We demonstrate the effectiveness of inference-time hyper-scaling with DMS on multiple families of LLMs, showing that it boosts accuracy for comparable inference latency and memory load. For instance, we enhance Qwen-R1 32B by 12.0 points on AIME 24, 8.6 on GPQA, and 9.7 on LiveCodeBench on average for an equivalent number of memory reads.
>
---
#### [replaced 017] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.23994v2](http://arxiv.org/pdf/2509.23994v2)**

> **作者:** Gauri Kholkar; Ratinder Ahuja
>
> **备注:** Accepted at 3rd Regulatable ML Workshop at NEURIPS 2025
>
> **摘要:** As autonomous AI agents are used in regulated and safety-critical settings, organizations need effective ways to turn policy into enforceable controls. We introduce a regulatory machine learning framework that converts unstructured design artifacts (like PRDs, TDDs, and code) into verifiable runtime guardrails. Our Policy as Prompt method reads these documents and risk controls to build a source-linked policy tree. This tree is then compiled into lightweight, prompt-based classifiers for real-time runtime monitoring. The system is built to enforce least privilege and data minimization. For conformity assessment, it provides complete provenance, traceability, and audit logging, all integrated with a human-in-the-loop review process. Evaluations show our system reduces prompt-injection risk, blocks out-of-scope requests, and limits toxic outputs. It also generates auditable rationales aligned with AI governance frameworks. By treating policies as executable prompts (a policy-as-code for agents), this approach enables secure-by-design deployment, continuous compliance, and scalable AI safety and AI security assurance for regulatable ML.
>
---
#### [replaced 018] DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17803v2](http://arxiv.org/pdf/2508.17803v2)**

> **作者:** Kaiwen Yan; Xuanqing Shi; Hongcheng Guo; Wenxuan Wang; Zhuosheng Zhang; Chengwei Qin
>
> **摘要:** Reasoning large language models (RLLMs), such as OpenAI-O3 and DeepSeek-R1, have recently demonstrated remarkable capabilities by performing structured and multi-step reasoning. However, recent studies reveal that RLLMs often suffer from overthinking, i.e., producing unnecessarily lengthy reasoning chains even for simple questions, leading to excessive token consumption and computational inefficiency. Interestingly, we observe that when processing multiple questions in batch mode, RLLMs exhibit more resource-efficient behavior by dynamically compressing reasoning steps for easier problems, due to implicit resource competition. Inspired by this, we propose Dynamic Reasoning Quota Allocation (DRQA), a novel method that transfers the benefits of resource competition from batch processing to single-question inference. Specifically, DRQA leverages batch-generated preference data and reinforcement learning to train the model to allocate reasoning resources adaptively. By encouraging the model to internalize a preference for responses that are both accurate and concise, DRQA enables it to generate concise answers for simple questions while retaining sufficient reasoning depth for more challenging ones. Extensive experiments on a wide range of mathematical and scientific reasoning benchmarks demonstrate that DRQA significantly reduces token usage while maintaining, and in many cases improving, answer accuracy. By effectively mitigating the overthinking problem, DRQA offers a promising direction for more efficient and scalable deployment of RLLMs, and we hope it inspires further exploration into fine-grained control of reasoning behaviors.
>
---
#### [replaced 019] Towards Explainable Fake Image Detection with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2504.14245v2](http://arxiv.org/pdf/2504.14245v2)**

> **作者:** Yikun Ji; Yan Hong; Jiahui Zhan; Haoxing Chen; jun lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **备注:** Accepted to ACM MM 2025; 14 pages including Appendix
>
> **摘要:** Progress in image generation raises significant public security concerns. We argue that fake image detection should not operate as a "black box". Instead, an ideal approach must ensure both strong generalization and transparency. Recent progress in Multi-modal Large Language Models (MLLMs) offers new opportunities for reasoning-based AI-generated image detection. In this work, we evaluate the capabilities of MLLMs in comparison to traditional detection methods and human evaluators, highlighting their strengths and limitations. Furthermore, we design six distinct prompts and propose a framework that integrates these prompts to develop a more robust, explainable, and reasoning-driven detection system. The code is available at https://github.com/Gennadiyev/mllm-defake.
>
---
#### [replaced 020] MorphTok: Morphologically Grounded Tokenization for Indian Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10335v2](http://arxiv.org/pdf/2504.10335v2)**

> **作者:** Maharaj Brahma; N J Karthika; Atul Singh; Devaraj Adiga; Smruti Bhate; Ganesh Ramakrishnan; Rohit Saluja; Maunendra Sankar Desarkar
>
> **备注:** Accepted at Tokenization Workshop (TokShop), ICML 2025
>
> **摘要:** Tokenization is a crucial step in NLP, especially with the rise of large language models (LLMs), impacting downstream performance, computational cost, and efficiency. Existing LLMs rely on the classical Byte-pair Encoding (BPE) algorithm for subword tokenization that greedily merges frequent character bigrams, often leading to segmentation that does not align with linguistically meaningful units. To address this, we propose morphology-aware segmentation as a pre-tokenization step before applying BPE. To facilitate morphology-aware segmentation, we create a novel dataset for Hindi and Marathi, incorporating sandhi splitting to enhance the subword tokenization. Experiments on downstream tasks show that morphologically grounded tokenization improves machine translation and language modeling performance. Additionally, to handle the dependent vowels common in syllable-based writing systems used by Indic languages, we propose Constrained BPE (CBPE), an extension to the standard BPE algorithm incorporating script-specific constraints. In particular, CBPE handles dependent vowels to form a cohesive unit with other characters instead of occurring as a single unit. Our results show that CBPE achieves a 1.68\% reduction in fertility scores while maintaining comparable or improved downstream performance in machine translation and language modeling, offering a computationally efficient alternative to standard BPE. Moreover, to evaluate segmentation across different tokenization algorithms, we introduce a new human evaluation metric, \textit{EvalTok}, enabling more human-grounded assessment.
>
---
#### [replaced 021] To Word Senses and Beyond: Inducing Concepts with Contextualized Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.20054v3](http://arxiv.org/pdf/2406.20054v3)**

> **作者:** Bastien Liétard; Pascal Denis; Mikaela Keller
>
> **备注:** Published in EMNLP 2024 main conference proceedings
>
> **摘要:** Polysemy and synonymy are two crucial interrelated facets of lexical ambiguity. While both phenomena are widely documented in lexical resources and have been studied extensively in NLP, leading to dedicated systems, they are often being considered independently in practical problems. While many tasks dealing with polysemy (e.g. Word Sense Disambiguation or Induction) highlight the role of word's senses, the study of synonymy is rooted in the study of concepts, i.e. meanings shared across the lexicon. In this paper, we introduce Concept Induction, the unsupervised task of learning a soft clustering among words that defines a set of concepts directly from data. This task generalizes Word Sense Induction. We propose a bi-level approach to Concept Induction that leverages both a local lemma-centric view and a global cross-lexicon view to induce concepts. We evaluate the obtained clustering on SemCor's annotated data and obtain good performance (BCubed F1 above 0.60). We find that the local and the global levels are mutually beneficial to induce concepts and also senses in our setting. Finally, we create static embeddings representing our induced concepts and use them on the Word-in-Context task, obtaining competitive performance with the State-of-the-Art.
>
---
#### [replaced 022] HugAgent: Benchmarking LLMs for Simulation of Individualized Human Reasoning
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.15144v3](http://arxiv.org/pdf/2510.15144v3)**

> **作者:** Chance Jiajie Li; Zhenze Mo; Yuhan Tang; Ao Qu; Jiayi Wu; Kaiya Ivy Zhao; Yulu Gan; Jie Fan; Jiangbo Yu; Hang Jiang; Paul Pu Liang; Jinhua Zhao; Luis Alberto Alonso Pastor; Kent Larson
>
> **备注:** To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
>
> **摘要:** Simulating human reasoning in open-ended tasks has long been a central aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), which rethinks human reasoning simulation along three dimensions: (i) from averaged to individualized reasoning, (ii) from behavioral mimicry to cognitive alignment, and (iii) from vignette-based to open-ended data. The benchmark evaluates whether a model can predict a specific person's behavioral responses and the underlying reasoning dynamics in out-of-distribution scenarios, given partial evidence of their prior views. HugAgent adopts a dual-track design: a human track that automates and scales the think-aloud method to collect ecologically valid human reasoning data, and a synthetic track for further scalability and systematic stress testing. This architecture enables low-cost, extensible expansion to new tasks and populations. Experiments with state-of-the-art language models reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. The benchmark, along with its complete data collection pipeline and companion chatbot, is open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
>
---
#### [replaced 023] CSPLADE: Learned Sparse Retrieval with Causal Language Models
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10816v3](http://arxiv.org/pdf/2504.10816v3)**

> **作者:** Zhichao Xu; Aosong Feng; Yijun Tian; Haibo Ding; Lin Lee Cheong
>
> **备注:** IJCNLP-AACL 2025 Main
>
> **摘要:** In recent years, dense retrieval has been the focus of information retrieval (IR) research. While effective, dense retrieval produces uninterpretable dense vectors, and suffers from the drawback of large index size. Learned sparse retrieval (LSR) has emerged as promising alternative, achieving competitive retrieval performance while also being able to leverage the classical inverted index data structure for efficient retrieval. However, limited works have explored scaling LSR beyond BERT scale. In this work, we identify two challenges in training large language models (LLM) for LSR: (1) training instability during the early stage of contrastive training; (2) suboptimal performance due to pre-trained LLM's unidirectional attention. To address these challenges, we propose two corresponding techniques: (1) a lightweight adaptation training phase to eliminate training instability; (2) two model variants to enable bidirectional information. With these techniques, we are able to train LSR models with 8B scale LLM, and achieve competitive retrieval performance with reduced index size. Furthermore, we are among the first to analyze the performance-efficiency tradeoff of LLM-based LSR model through the lens of model quantization. Our findings provide insights into adapting LLMs for efficient retrieval modeling.
>
---
#### [replaced 024] What Can String Probability Tell Us About Grammaticality?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16227v2](http://arxiv.org/pdf/2510.16227v2)**

> **作者:** Jennifer Hu; Ethan Gotlieb Wilcox; Siyuan Song; Kyle Mahowald; Roger P. Levy
>
> **摘要:** What have language models (LMs) learned about grammar? This question remains hotly debated, with major ramifications for linguistic theory. However, since probability and grammaticality are distinct notions in linguistics, it is not obvious what string probabilities can reveal about an LM's underlying grammatical knowledge. We present a theoretical analysis of the relationship between grammar, meaning, and string probability, based on simple assumptions about the generative process of corpus data. Our framework makes three predictions, which we validate empirically using 280K sentence pairs in English and Chinese: (1) correlation between the probability of strings within minimal pairs, i.e., string pairs with minimal semantic differences; (2) correlation between models' and humans' deltas within minimal pairs; and (3) poor separation in probability space between unpaired grammatical and ungrammatical strings. Our analyses give theoretical grounding for using probability to learn about LMs' structural knowledge, and suggest directions for future work in LM grammatical evaluation.
>
---
#### [replaced 025] TRACE: Textual Relevance Augmentation and Contextual Encoding for Multimodal Hate Detection
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17902v2](http://arxiv.org/pdf/2504.17902v2)**

> **作者:** Girish A. Koushik; Helen Treharne; Aditya Joshi; Diptesh Kanojia
>
> **备注:** Accepted to Special Track on AI for Social Impact (AISI) at AAAI 2026
>
> **摘要:** Social media memes are a challenging domain for hate detection because they intertwine visual and textual cues into culturally nuanced messages. To tackle these challenges, we introduce TRACE, a hierarchical multimodal framework that leverages visually grounded context augmentation, along with a novel caption-scoring network to emphasize hate-relevant content, and parameter-efficient fine-tuning of CLIP's text encoder. Our experiments demonstrate that selectively fine-tuning deeper text encoder layers significantly enhances performance compared to simpler projection-layer fine-tuning methods. Specifically, our framework achieves state-of-the-art accuracy (0.807) and F1-score (0.806) on the widely-used Hateful Memes dataset, matching the performance of considerably larger models while maintaining efficiency. Moreover, it achieves superior generalization on the MultiOFF offensive meme dataset (F1-score 0.673), highlighting robustness across meme categories. Additional analyses confirm that robust visual grounding and nuanced text representations significantly reduce errors caused by benign confounders. We publicly release our code to facilitate future research.
>
---
#### [replaced 026] ProRefine: Inference-Time Prompt Refinement with Textual Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05305v3](http://arxiv.org/pdf/2506.05305v3)**

> **作者:** Deepak Pandita; Tharindu Cyril Weerasooriya; Ankit Parag Shah; Isabelle Diana May-Xin Ng; Christopher M. Homan; Wei Wei
>
> **备注:** Workshop on Efficient Reasoning at NeurIPS 2025
>
> **摘要:** Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, play a substantial role in many cutting-edge commercial applications, and continue to fascinate researchers across fields for their potential to accomplish expensive, complex tasks that, until recently, only humans have been trusted to do. These workflows depend critically on the prompts used to provide the roles models play in such workflows. Poorly designed prompts that fail even slightly to guide individual agents can lead to sub-optimal performance that may snowball within a system of agents, limiting their reliability and scalability. To address this important problem of inference-time prompt optimization, we introduce ProRefine, an innovative inference-time optimization method that uses an agentic loop of LLMs to generate and apply textual feedback. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to approach the performance of their larger counterparts. This highlights its potential for building more cost-effective and powerful hybrid AI systems, thereby democratizing access to high-performing AI.
>
---
#### [replaced 027] How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.22780v2](http://arxiv.org/pdf/2510.22780v2)**

> **作者:** Zora Zhiruo Wang; Yijia Shao; Omar Shaikh; Daniel Fried; Graham Neubig; Diyi Yang
>
> **摘要:** AI agents are continually optimized for tasks related to human work, such as software engineering and professional writing, signaling a pressing trend with significant impacts on the human workforce. However, these agent developments have often not been grounded in a clear understanding of how humans execute work, to reveal what expertise agents possess and the roles they can play in diverse workflows. In this work, we study how agents do human work by presenting the first direct comparison of human and agent workers across multiple essential work-related skills: data analysis, engineering, computation, writing, and design. To better understand and compare heterogeneous computer-use activities of workers, we introduce a scalable toolkit to induce interpretable, structured workflows from either human or agent computer-use activities. Using such induced workflows, we compare how humans and agents perform the same tasks and find that: (1) While agents exhibit promise in their alignment to human workflows, they take an overwhelmingly programmatic approach across all work domains, even for open-ended, visually dependent tasks like design, creating a contrast with the UI-centric methods typically used by humans. (2) Agents produce work of inferior quality, yet often mask their deficiencies via data fabrication and misuse of advanced tools. (3) Nonetheless, agents deliver results 88.3% faster and cost 90.4-96.2% less than humans, highlighting the potential for enabling efficient collaboration by delegating easily programmable tasks to agents.
>
---
#### [replaced 028] LEME: Open Large Language Models for Ophthalmology with Advanced Reasoning and Clinical Validation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03740v3](http://arxiv.org/pdf/2410.03740v3)**

> **作者:** Hyunjae Kim; Xuguang Ai; Sahana Srinivasan; Aidan Gilson; Maxwell B. Singer; Krithi Pushpanathan; Qianqian Xie; Jungwoo Park; Serina Applebaum; Gabriel Dawei Yang; Minjie Zou; David Ziyou Chen; Ke Zou; Soshian Sarrafpour; Ji Liu; Yu Yin; Jimin Huang; Quang Ngoc Nguyen; Erping Long; Peixing Wan; Dianbo Liu; Richard Hintz; W. Jim Zheng; Sophia Y. Wang; Lucila Ohno-Machado; Hua Xu; Ron A. Adelman; Luciano V. Del Priore; Yih-Chung Tham; Qingyu Chen
>
> **摘要:** The rising prevalence of eye diseases poses a growing public health burden. Large language models (LLMs) offer a promising path to reduce documentation workload and support clinical decision-making. However, few have been tailored for ophthalmology, and most evaluations focus mainly on knowledge-based QA without clinically relevant benchmarks or real-world validation. Here, we present LEME, a suite of open-weight LLMs developed through a two-stage process: (1) instruction tuning on 200,000 samples from clinical guidelines, textbooks, and case reports to enhance reasoning and task-following, and (2) reinforcement learning with ~30,000 preference labels to enhance accuracy and informativeness. LEME was evaluated on five curated zero-shot benchmarks spanning tasks such as patient QA, consultation, and treatment planning. It outperformed all seven baselines (all p < 0.004), exceeding GPT-4o by 3.32% (absolute ROUGE-L gain). It was further evaluated on three downstream tasks using deidentified patient data, reviewed by clinicians. In patient QA, LEME received the highest ratings from attending clinicians in 3 out of 4 criteria, with scores of 4.67 for factuality, 4.77 for specificity, 4.79 for completeness, and 4.88 for safety (1-5 scale). Its completeness score surpassed that of expert-written answers (4.79 vs. 4.56; p = 0.015). In visual acuity extraction, LEME achieved the highest F1, outperforming LLaMA-3 by 14.1% and Eye-LLaMA by 59.0%. In a pilot evaluation on assessment and treatment planning for diabetic retinopathy, AMD, and glaucoma, LEME received scores of 4.36 for factuality, 4.55 for specificity, 4.42 for completeness, and 4.36 for safety, approaching attending-level performance. All models, data, and code will be released to support further development and clinical translation, laying the groundwork for improved efficiency and patient care
>
---
#### [replaced 029] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering
- **分类: cs.IR; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16470v2](http://arxiv.org/pdf/2505.16470v2)**

> **作者:** Kuicai Dong; Yujing Chang; Shijie Huang; Yasheng Wang; Ruiming Tang; Yong Liu
>
> **备注:** Paper accepted to NeurIPS 2025 DB
>
> **摘要:** Document Visual Question Answering (DocVQA) faces dual challenges in processing lengthy multimodal documents (text, images, tables) and performing cross-modal reasoning. Current document retrieval-augmented generation (DocRAG) methods remain limited by their text-centric approaches, frequently missing critical visual information. The field also lacks robust benchmarks for assessing multimodal evidence selection and integration. We introduce MMDocRAG, a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with multi-page, cross-modal evidence chains. Our framework introduces innovative metrics for evaluating multimodal quote selection and enables answers that interleave text with relevant visual elements. Through large-scale experiments with 60 VLM/LLM models and 14 retrieval systems, we identify persistent challenges in multimodal evidence retrieval, selection, and integration.Key findings reveal advanced proprietary LVMs show superior performance than open-sourced alternatives. Also, they show moderate advantages using multimodal inputs over text-only inputs, while open-source alternatives show significant performance degradation. Notably, fine-tuned LLMs achieve substantial improvements when using detailed image descriptions. MMDocRAG establishes a rigorous testing ground and provides actionable insights for developing more robust multimodal DocVQA systems. Our benchmark and code are available at https://mmdocrag.github.io/MMDocRAG/.
>
---
#### [replaced 030] Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17086v4](http://arxiv.org/pdf/2502.17086v4)**

> **作者:** Hyungyu Shin; Jingyu Tang; Yoonjoo Lee; Nayoung Kim; Hyunseung Lim; Ji Yong Cho; Hwajung Hong; Moontae Lee; Juho Kim
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Peer review underpins scientific progress, but it is increasingly strained by reviewer shortages and growing workloads. Large Language Models (LLMs) can automatically draft reviews now, but determining whether LLM-generated reviews are trustworthy requires systematic evaluation. Researchers have evaluated LLM reviews at either surface-level (e.g., BLEU and ROUGE) or content-level (e.g., specificity and factual accuracy). Yet it remains uncertain whether LLM-generated reviews attend to the same critical facets that human experts weigh -- the strengths and weaknesses that ultimately drive an accept-or-reject decision. We introduce a focus-level evaluation framework that operationalizes the focus as a normalized distribution of attention across predefined facets in paper reviews. Based on the framework, we developed an automatic focus-level evaluation pipeline based on two sets of facets: target (e.g., problem, method, and experiment) and aspect (e.g., validity, clarity, and novelty), leveraging 676 paper reviews (https://figshare.com/s/d5adf26c802527dd0f62) from OpenReview that consists of 3,657 strengths and weaknesses identified from human experts. The comparison of focus distributions between LLMs and human experts showed that the off-the-shelf LLMs consistently have a more biased focus towards examining technical validity while significantly overlooking novelty assessment when criticizing papers.
>
---
#### [replaced 031] Internal World Models as Imagination Networks in Cognitive Agents
- **分类: cs.AI; cs.CL; cs.SI; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2510.04391v2](http://arxiv.org/pdf/2510.04391v2)**

> **作者:** Saurabh Ranjan; Brian Odegaard
>
> **摘要:** What is the computational objective of imagination? While classical interpretations suggest imagination is useful for maximizing rewards, recent findings challenge this view. In this study, we propose that imagination serves to access an internal world model (IWM) and use psychological network analysis to explore IWMs in humans and large language models (LLMs). Specifically, we assessed imagination vividness ratings using two questionnaires and constructed imagination networks from these reports. Imagination networks from human groups showed correlations between different centrality measures, including expected influence, strength, and closeness. However, imagination networks from LLMs showed a lack of clustering and lower correlations between centrality measures under different prompts and conversational memory conditions. Together, these results indicate a lack of similarity between IWMs in human and LLM agents. Overall, our study offers a novel method for comparing internally-generated representations in humans and AI, providing insights for developing human-like imagination in artificial intelligence.
>
---
#### [replaced 032] P-ReMIS: Pragmatic Reasoning in Mental Health and a Social Implication
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23247v2](http://arxiv.org/pdf/2507.23247v2)**

> **作者:** Sneha Oram; Pushpak Bhattacharyya
>
> **摘要:** Although explainability and interpretability have received significant attention in artificial intelligence (AI) and natural language processing (NLP) for mental health, reasoning has not been examined in the same depth. Addressing this gap is essential to bridge NLP and mental health through interpretable and reasoning-capable AI systems. To this end, we investigate the pragmatic reasoning capability of large-language models (LLMs) in the mental health domain. We introduce PRiMH dataset, and propose pragmatic reasoning tasks in mental health with pragmatic implicature and presupposition phenomena. In particular, we formulate two tasks in implicature and one task in presupposition. To benchmark the dataset and the tasks presented, we consider four models: Llama3.1, Mistral, MentaLLaMa, and Qwen. The results of the experiments suggest that Mistral and Qwen show substantial reasoning abilities in the domain. Subsequently, we study the behavior of MentaLLaMA on the proposed reasoning tasks with the rollout attention mechanism. In addition, we also propose three StiPRompts to study the stigma around mental health with the state-of-the-art LLMs, GPT4o-mini, Deepseek-chat, and Claude-3.5-haiku. Our evaluated findings show that Claude-3.5-haiku deals with stigma more responsibly compared to the other two LLMs.
>
---
#### [replaced 033] Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03222v2](http://arxiv.org/pdf/2510.03222v2)**

> **作者:** Guanhua Huang; Tingqiang Xu; Mingze Wang; Qi Yi; Xue Gong; Siheng Li; Ruibin Xiong; Kejiao Li; Yuhao Jiang; Bo Zhou
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key issue: the gradual elimination of valuable low-probability exploratory tokens, which we term \textbf{\textit{reasoning sparks}}. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by filtering out presumed noise tokens and re-normalizing the distribution over the remaining candidates. The result is a less-noisy proxy where the probability of \textit{reasoning sparks} is amplified, which then serves as a soft regularization target to shield these valuable tokens from elimination via KL divergence. Experiments show that Lp-Reg enables stable on-policy RL, sustaining continuous scaling across $3,000$ training steps and $81,204$ GPU-hours, where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a $60.17\%$ average accuracy on five math benchmarks, an improvement of $2.66\%$ over prior methods. Code is available at https://github.com/CarlanLark/Lp-Reg.
>
---
#### [replaced 034] Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2508.15809v2](http://arxiv.org/pdf/2508.15809v2)**

> **作者:** Songyuan Sui; Hongyi Liu; Serena Liu; Li Li; Soo-Hyun Choi; Rui Chen; Xia Hu
>
> **备注:** AACL 2025 Main Conference
>
> **摘要:** Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
>
---
#### [replaced 035] Activation-Informed Merging of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02421v3](http://arxiv.org/pdf/2502.02421v3)**

> **作者:** Amin Heyrani Nobari; Kaveh Alim; Ali ArjomandBigdeli; Akash Srivastava; Faez Ahmed; Navid Azizan
>
> **摘要:** Model merging, a method that combines the parameters and embeddings of multiple fine-tuned large language models (LLMs), offers a promising approach to enhance model performance across various tasks while maintaining computational efficiency. This paper introduces Activation-Informed Merging (AIM), a technique that integrates the information from the activation space of LLMs into the merging process to improve performance and robustness. AIM is designed as a flexible, complementary solution that is applicable to any existing merging method. It aims to preserve critical weights from the base model, drawing on principles from continual learning (CL) and model compression. Utilizing a task-agnostic calibration set, AIM selectively prioritizes essential weights during merging. We empirically demonstrate that AIM significantly enhances the performance of merged models across multiple benchmarks. Our findings suggest that considering the activation-space information can provide substantial advancements in the model merging strategies for LLMs, with up to a 40% increase in benchmark performance.
>
---
#### [replaced 036] MMDocIR: Benchmarking Multimodal Retrieval for Long Documents
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08828v3](http://arxiv.org/pdf/2501.08828v3)**

> **作者:** Kuicai Dong; Yujing Chang; Xin Deik Goh; Dexun Li; Ruiming Tang; Yong Liu
>
> **备注:** Paper accepted to EMNLP-2025(Main)
>
> **摘要:** Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text. Our dataset is available at https://mmdocrag.github.io/MMDocIR/.
>
---
#### [replaced 037] iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration for Advanced Tool Use
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.09766v5](http://arxiv.org/pdf/2501.09766v5)**

> **作者:** Yirong Zeng; Xiao Ding; Yuxian Wang; Weiwen Liu; Wu Ning; Yutai Hou; Xu Huang; Duyu Tang; Dandan Tu; Bing Qin; Ting Liu
>
> **备注:** EMNLP 2025
>
> **摘要:** Augmenting large language models (LLMs) with external tools is a promising approach to enhance their capabilities, especially for complex tasks. Synthesizing tool-use data through real-world simulations is an effective way to achieve this. However, our investigation reveals that training gains significantly decay as synthetic data increases. The model struggles to benefit from additional synthetic data, which fails to endow it with advanced tool-use capabilities in complex scenarios Moreover, we discovered that the above limitation usually manifests as a fragment deficiency (i.e., parameter errors) in response. To this end, we propose an iterative reinforced fine-tuning strategy designed to alleviate this limitation. This strategy involves: (1) enhancing the diversity of response for synthetic data through path exploration of Monte Carlo Tree Search. (2) iteratively pinpointing the model's deficiency by constructing fine-grained preference pairs, and then improving it by preference optimization algorithms for targeted improvement. The experiments show that our method achieves 13.11% better performance than the same-size base model. It achieves an improvement of 6.5% in complex scenarios compared to the baseline, and it also outperforms larger open-source and closed-source models.
>
---
#### [replaced 038] Neural at ArchEHR-QA 2025: Agentic Prompt Optimization for Evidence-Grounded Clinical Question Answering
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10751v2](http://arxiv.org/pdf/2506.10751v2)**

> **作者:** Sai Prasanna Teja Reddy Bogireddy; Abrar Majeedi; Viswanatha Reddy Gajjala; Zhuoyan Xu; Siddhant Rai; Vaishnav Potlapalli
>
> **备注:** Accepted to Proceedings of the 24th Workshop on Biomedical Language Processing (https://aclanthology.org/2025.bionlp-share.13/)
>
> **摘要:** Automated question answering (QA) over electronic health records (EHRs) can bridge critical information gaps for clinicians and patients, yet it demands both precise evidence retrieval and faithful answer generation under limited supervision. In this work, we present Neural, the runner-up in the BioNLP 2025 ArchEHR-QA shared task on evidence-grounded clinical QA. Our proposed method decouples the task into (1) sentence-level evidence identification and (2) answer synthesis with explicit citations. For each stage, we automatically explore the prompt space with DSPy's MIPROv2 optimizer, jointly tuning instructions and few-shot demonstrations on the development set. A self-consistency voting scheme further improves evidence recall without sacrificing precision. On the hidden test set, our method attains an overall score of 51.5, placing second stage while outperforming standard zero-shot and few-shot prompting by over 20 and 10 points, respectively. These results indicate that data-driven prompt optimization is a cost-effective alternative to model fine-tuning for high-stakes clinical QA, advancing the reliability of AI assistants in healthcare.
>
---
#### [replaced 039] Scalable Medication Extraction and Discontinuation Identification from Electronic Health Records Using Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11137v3](http://arxiv.org/pdf/2506.11137v3)**

> **作者:** Chong Shao; Douglas Snyder; Chiran Li; Bowen Gu; Kerry Ngan; Chun-Ting Yang; Jiageng Wu; Richard Wyss; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Identifying medication discontinuations in electronic health records (EHRs) is vital for patient safety but is often hindered by information being buried in unstructured notes. This study aims to evaluate the capabilities of advanced open-sourced and proprietary large language models (LLMs) in extracting medications and classifying their medication status from EHR notes, focusing on their scalability on medication information extraction without human annotation. We collected three EHR datasets from diverse sources to build the evaluation benchmark. We evaluated 12 advanced LLMs and explored multiple LLM prompting strategies. Performance on medication extraction, medication status classification, and their joint task (extraction then classification) was systematically compared across all experiments. We found that LLMs showed promising performance on the medication extraction and discontinuation classification from EHR notes. GPT-4o consistently achieved the highest average F1 scores in all tasks under zero-shot setting - 94.0% for medication extraction, 78.1% for discontinuation classification, and 72.7% for the joint task. Open-sourced models followed closely, Llama-3.1-70B-Instruct achieved the highest performance in medication status classification on the MIV-Med dataset (68.7%) and in the joint task on both the Re-CASI (76.2%) and MIV-Med (60.2%) datasets. Medical-specific LLMs demonstrated lower performance compared to advanced general-domain LLMs. Few-shot learning generally improved performance, while CoT reasoning showed inconsistent gains. LLMs demonstrate strong potential for medication extraction and discontinuation identification on EHR notes, with open-sourced models offering scalable alternatives to proprietary systems and few-shot can further improve LLMs' capability.
>
---
#### [replaced 040] Re:Member: Emotional Question Generation from Personal Memories
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.19030v2](http://arxiv.org/pdf/2510.19030v2)**

> **作者:** Zackary Rackauckas; Nobuaki Minematsu; Julia Hirschberg
>
> **摘要:** We present Re:Member, a system that explores how emotionally expressive, memory-grounded interaction can support more engaging second language (L2) learning. By drawing on users' personal videos and generating stylized spoken questions in the target language, Re:Member is designed to encourage affective recall and conversational engagement. The system aligns emotional tone with visual context, using expressive speech styles such as whispers or late-night tones to evoke specific moods. It combines WhisperX-based transcript alignment, 3-frame visual sampling, and Style-BERT-VITS2 for emotional synthesis within a modular generation pipeline. Designed as a stylized interaction probe, Re:Member highlights the role of affect and personal media in learner-centered educational technologies.
>
---
#### [replaced 041] NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13124v4](http://arxiv.org/pdf/2502.13124v4)**

> **作者:** Weizhe Yuan; Jane Yu; Song Jiang; Karthik Padthe; Yang Li; Ilia Kulikov; Kyunghyun Cho; Dong Wang; Yuandong Tian; Jason E Weston; Xian Li
>
> **备注:** Dataset at https://huggingface.co/datasets/facebook/natural_reasoning
>
> **摘要:** Scaling reasoning capabilities beyond traditional domains such as math and coding is hindered by the lack of diverse and high-quality questions. To overcome this limitation, we introduce a scalable approach for generating diverse and challenging reasoning questions, accompanied by reference answers. We present NaturalReasoning, a comprehensive dataset comprising 2.8 million questions that span multiple domains, including STEM fields (e.g., Physics, Computer Science), Economics, Social Sciences, and more. We demonstrate the utility of the questions in NaturalReasoning through knowledge distillation experiments which show that NaturalReasoning can effectively elicit and transfer reasoning capabilities from a strong teacher model. Furthermore, we demonstrate that NaturalReasoning is also effective for unsupervised self-training using external reward models or self-rewarding. To foster future work, we publicly release NaturalReasoning at https://huggingface.co/datasets/facebook/natural_reasoning.
>
---
#### [replaced 042] Fair Document Valuation in LLM Summaries via Shapley Values
- **分类: cs.CL; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2505.23842v3](http://arxiv.org/pdf/2505.23842v3)**

> **作者:** Zikun Ye; Hema Yoganarasimhan
>
> **摘要:** Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
>
---
#### [replaced 043] Are Humans as Brittle as Large Language Models?
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.07869v2](http://arxiv.org/pdf/2509.07869v2)**

> **作者:** Jiahui Li; Sean Papay; Roman Klinger
>
> **摘要:** The output of large language models (LLMs) is unstable, due both to non-determinism of the decoding process as well as to prompt brittleness. While the intrinsic non-determinism of LLM generation may mimic existing uncertainty in human annotations through distributional shifts in outputs, it is largely assumed, yet unexplored, that the prompt brittleness effect is unique to LLMs. This raises the question: do human annotators show similar sensitivity to prompt changes? If so, should prompt brittleness in LLMs be considered problematic? One may alternatively hypothesize that prompt brittleness correctly reflects human annotation variances. To fill this research gap, we systematically compare the effects of prompt modifications on LLMs and identical instruction modifications for human annotators, focusing on the question of whether humans are similarly sensitive to prompt perturbations. To study this, we prompt both humans and LLMs for a set of text classification tasks conditioned on prompt variations. Our findings indicate that both humans and LLMs exhibit increased brittleness in response to specific types of prompt modifications, particularly those involving the substitution of alternative label sets or label formats. However, the distribution of human judgments is less affected by typographical errors and reversed label order than that of LLMs.
>
---
#### [replaced 044] LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03505v2](http://arxiv.org/pdf/2509.03505v2)**

> **作者:** Xingxuan Zhang; Gang Ren; Han Yu; Hao Yuan; Hui Wang; Jiansheng Li; Jiayun Wu; Lang Mo; Li Mao; Mingchao Hao; Ningbo Dai; Renzhe Xu; Shuyang Li; Tianyang Zhang; Yue He; Yuanrui Wang; Yunjia Zhang; Zijing Xu; Dongzhe Li; Fang Gao; Hao Zou; Jiandong Liu; Jiashuo Liu; Jiawei Xu; Kaijie Cheng; Kehan Li; Linjun Zhou; Qing Li; Shaohua Fan; Xiaoyu Lin; Xinyan Han; Xuanyue Li; Yan Lu; Yuan Xue; Yuanyuan Jiang; Zimu Wang; Zhenlei Wang; Peng Cui
>
> **备注:** 61 pages
>
> **摘要:** We argue that progress toward general intelligence requires complementary foundation models grounded in language, the physical world, and structured data. This report presents LimiX-16M and LimiX-2M, two instantiations of our large structured-data models (LDMs). Both models treat structured data as a joint distribution over variables and missingness, thus capable of addressing a wide range of tabular tasks through query-based conditional prediction via a single model. They are pretrained using masked joint-distribution modeling with an episodic, context-conditional objective, supporting rapid, training-free adaptation at inference. We evaluate LimiX models across 11 large structured-data benchmarks with broad regimes of sample size, feature dimensionality, class number, categorical-to-numerical feature ratio, missingness, and sample-to-feature ratios. LimiX-16M consistently surpasses strong baselines, as shown in Figure 1 and Figure 2. The superiority holds across a wide range of tasks, such as classification, regression, missing value imputation, and data generation, often by substantial margins, while avoiding task-specific architectures or bespoke training per task. Notably, LimiX-2M delivers strong results under tight compute and memory budgets. We also present the first scaling law study for LDMs, revealing how data and model scaling jointly influence downstream performance and offering quantitative guidance for tabular foundation modeling. All LimiX models are publicly accessible under Apache 2.0.
>
---
#### [replaced 045] SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20514v2](http://arxiv.org/pdf/2508.20514v2)**

> **作者:** Pengjiang Li; Zaitian Wang; Xinhao Zhang; Ran Zhang; Lu Jiang; Pengfei Wang; Yuanchun Zhou
>
> **摘要:** Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights.
>
---

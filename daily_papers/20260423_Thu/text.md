# 自然语言处理 cs.CL

- **最新发布 108 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] Whose Story Gets Told? Positionality and Bias in LLM Summaries of Life Narratives
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的伦理分析任务，旨在解决LLM在解读人生叙事时可能存在的偏见问题。通过构建摘要管道，识别种族和性别偏见，强调在研究中需关注LLM的立场视角。**

- **链接: [https://arxiv.org/pdf/2604.20131](https://arxiv.org/pdf/2604.20131)**

> **作者:** Melanie Subbiah; Haaris Mian; Nicholas Deas; Ananya Mayukha; Dan P. McAdams; Kathleen McKeown
>
> **摘要:** Increasingly, studies are exploring using Large Language Models (LLMs) for accelerated or scaled qualitative analysis of text data. While we can compare LLM accuracy against human labels directly for deductive coding, or labeling text, it is more challenging to judge the ethics and effectiveness of using LLMs in abstractive methods such as inductive thematic analysis. We collaborate with psychologists to study the abstractive claims LLMs make about human life stories, asking, how does using an LLM as an interpreter of meaning affect the conclusions and perspectives of a study? We propose a summarization-based pipeline for surfacing biases in perspective-taking an LLM might employ in interpreting these life stories. We demonstrate that our pipeline can identify both race and gender bias with the potential for representational harm. Finally, we encourage the use of this analysis in future studies involving LLM-based interpretation of study participants' written text or transcribed speech to characterize a positionality portrait for the study.
>
---
#### [new 002] How Much Does Persuasion Strategy Matter? LLM-Annotated Evidence from Charitable Donation Dialogues
- **分类: cs.CL**

- **简介: 该论文属于 persuasion 研究任务，旨在探讨不同说服策略对慈善捐款的影响。通过标注对话数据并分析策略与捐赠行为的关系，发现策略分类对捐赠效果解释力有限， guilt 引导反而降低捐赠率。**

- **链接: [https://arxiv.org/pdf/2604.19783](https://arxiv.org/pdf/2604.19783)**

> **作者:** Tatiana Petrova; Stanislav Sokol; Radu State
>
> **备注:** 8 pages, 2 figures, 5 tables. Interdisciplinary Centre for Security, Reliability and Trust (SnT), University of Luxembourg
>
> **摘要:** Which persuasion strategies, if any, are associated with donation compliance? Answering this requires fine-grained strategy labels across a full corpus and statistical tests corrected for multiple comparisons. We annotate all 10,600 persuader turns in the 1,017-dialogue PersuasionForGood corpus (Wang et al., 2019), where donation outcomes are directly observable, with a taxonomy of 41 strategies in 11 categories, using three open-source large language models (LLMs; Qwen3:30b, Mistral-Small-3.2, Phi-4). Strategy categories alone explain little variance in donation outcome (pseudo $R^2 \approx 0.015$, consistent across all three annotators). Guilt Induction is the only strategy significantly associated with lower donation rates ($\Delta \approx -23$ percentage points), an effect that replicates across all three models despite only moderate inter-model agreement. Reciprocity is the most robust positive correlate. Target sentiment and interest predict whether a donation occurs but show at most a weak correlation with donation amount. These findings suggest that strategy identification alone is insufficient to explain persuasion effectiveness, and that guilt-based appeals may be counterproductive in prosocial settings. We release the fully annotated corpus as a public resource.
>
---
#### [new 003] Intersectional Fairness in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决大语言模型在交叉群体中的偏见问题。通过评估多个指标，发现模型在不同情境下存在偏差和不一致性。**

- **链接: [https://arxiv.org/pdf/2604.20677](https://arxiv.org/pdf/2604.20677)**

> **作者:** Chaima Boufaied; Ronnie De Souza Santos; Ann Barcomb
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in socially sensitive settings, raising concerns about fairness and biases, particularly across intersectional demographic attributes. In this paper, we systematically evaluate intersectional fairness in six LLMs using ambiguous and disambiguated contexts from two benchmark datasets. We assess LLM behavior using bias scores, subgroup fairness metrics, accuracy, and consistency through multi-run analysis across contexts and negative and non-negative question polarities. Our results show that while modern LLMs generally perform well in ambiguous contexts, this limits the informativeness of fairness metrics due to sparse non-unknown predictions. In disambiguated contexts, LLM accuracy is influenced by stereotype alignment, with models being more accurate when the correct answer reinforces a stereotype than when it contradicts it. This pattern is especially pronounced in race-gender intersections, where directional bias toward stereotypes is stronger. Subgroup fairness metrics further indicate that, despite low observed disparity in some cases, outcome distributions remain uneven across intersectional groups. Across repeated runs, responses also vary in consistency, including stereotype-aligned responses. Overall, our findings show that apparent model competence is partly associated with stereotype-consistent cues, and no evaluated LLM achieves consistently reliable or fair behavior across intersectional settings. These findings highlight the need for evaluation beyond accuracy, emphasizing the importance of combining bias, subgroup fairness, and consistency metrics across intersectional groups, contexts, and repeated runs.
>
---
#### [new 004] From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型量化任务，旨在解决4-bit与2-bit量化差异问题。通过分析发现两种失败模式，提出针对性修复方法。**

- **链接: [https://arxiv.org/pdf/2604.19884](https://arxiv.org/pdf/2604.19884)**

> **作者:** Chenxi Zhou; Pengfei Cao; Jiang Li; Bohan Yu; Jinyu Ye; Jun Zhao; Kang Liu
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Post-Training Quantization (PTQ) is critical for the efficient deployment of Large Language Models (LLMs). While 4-bit quantization is widely regarded as an optimal trade-off, reducing the precision to 2-bit usually triggers a catastrophic ``performance cliff.'' It remains unclear whether the underlying mechanisms differ fundamentally. Consequently, we conduct a systematic mechanistic analysis, revealing two qualitatively distinct failure modes: Signal Degradation, where the computational patterns remain intact but information precision is impaired by cumulative error; and Computation Collapse, where key components fail to function, preventing correct information processing and destroying the signal in the early layers. Guided by this diagnosis, we conduct mechanism-aware interventions, demonstrating that targeted, training-free repair can mitigate Signal Degradation, but remains ineffective for Computation Collapse. Our findings provide a systematic diagnostic framework for PTQ failures and suggest that addressing Computation Collapse requires structural reconstruction rather than mere compensation.
>
---
#### [new 005] Construction of a Battery Research Knowledge Graph using a Global Open Catalog
- **分类: cs.CL; physics.comp-ph**

- **简介: 该论文属于知识图谱构建任务，旨在解决电池研究领域跨机构协作难的问题。通过整合OpenAlex数据，构建作者中心的知识图谱，支持相似性计算与社区发现。**

- **链接: [https://arxiv.org/pdf/2604.20241](https://arxiv.org/pdf/2604.20241)**

> **作者:** Luca Foppiano; Sae Dieb; Malik Zain; Kazuki Kasama; Keitaro Sodeyama; Mikiko Tanifuji
>
> **摘要:** Battery research is a rapidly growing and highly interdisciplinary field, making it increasingly difficult to track relevant expertise and identify potential collaborators across institutional boundaries. In this work, we present a pipeline for constructing an author-centric knowledge graph of battery research built on OpenAlex, a large-scale open bibliographic catalogue. For each author, we derive a weighted research descriptors vector that combines coarse-grained OpenAlex concepts with fine-grained keyphrases extracted from titles and abstracts using KeyBERT with ChatGPT (gpt-3.5-turbo) as the backend model, selected after evaluating multiple alternatives. Vector components are weighted by research descriptor origin, authorship position, and temporal recency. The framework is applied to a corpus of 189,581 battery-related works. The resulting vectors support author-author similarity computation, community detection, and exploratory search through a browser-based interface. The knowledge graph is then serialized in RDF and linked to Wikidata identifiers, making it interoperable with external linked open data sources and extensible beyond the battery domain. Unlike prior author-centric analyses confined to institutional repositories, our approach operates at cross-institutional scale and grounds similarity in domain semantics rather than citation or co-authorship structure alone.
>
---
#### [new 006] Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出POP框架，用于开放任务的自博弈训练。通过LLM生成评估标准和输入输出对，解决后训练阶段高质量数据不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.20051](https://arxiv.org/pdf/2604.20051)**

> **作者:** Chengyu Huang; Sheng-Yen Chou; Zhengxin Zhang; Claire Cardie
>
> **摘要:** Self-play has recently emerged as a promising paradigm to train Large Language Models (LLMs). In self-play, the target LLM creates the task input (e.g., ask a question), which it then addresses itself by producing a task output (e.g., give an answer). A reward model evaluates the output, and the rewards are then used to train the LLM, typically via Reinforcement Learning (RL). Self-play incurs minimal supervision costs, and this is especially helpful for post-training LLMs, which require high-quality input-output pairs that traditionally have to be written by humans or expensive proprietary models. However, existing work explores self-play only for verifiable tasks such as math and coding. Instead, we seek to extend it to more realistic open-ended tasks. In particular, we propose POP, a self-play framework that uses the same LLM to synthesize evaluation rubrics, along with input-output pairs, for each example. The rubric is then used to evaluate outputs and train the model. We further ground the framework on a content-rich pretraining corpus to (1) ensure a generation-verification gap and reduce reward hacking, and (2) prevent mode collapse. On Qwen-2.5-7B, POP increases performance of both pretrained and instruction-tuned models, across different tasks ranging from long-form Healthcare QA to creative writing and instruction following.
>
---
#### [new 007] Development and Preliminary Evaluation of a Domain-Specific Large Language Model for Tuberculosis Care in South Africa
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗领域自然语言处理任务，旨在解决南非结核病护理中的信息处理问题。通过微调医学大模型，提升其在结核病相关场景下的表现。**

- **链接: [https://arxiv.org/pdf/2604.19776](https://arxiv.org/pdf/2604.19776)**

> **作者:** Thokozile Khosa; Olawande Daramola
>
> **备注:** 12 pages, 2 figures, ICICT 2026 Conference
>
> **摘要:** Tuberculosis (TB) is one of the world's deadliest infectious diseases, and in South Africa, it contributes a significant burden to the country's health care system. This paper presents an experimental study on the development of a domain-specific Large Language Model (DS-LLM) for TB care that can help to alleviate the burden on patients and healthcare providers. To achieve this, a literature review was conducted to understand current LLM development strategies, specifically in the medical domain. Thereafter, data were collected from South African TB guidelines, selected TB literature, and existing benchmark medical datasets. We performed LLM fine-tuning by using the Quantised Low-Rank Adaptation (QLoRA) algorithm on a medical LLM (BioMistral-7B), and also implemented Retrieval-Augmented Generation using GraphRAG. The developed DS-LLM was evaluated against the base BioMistral-7B model and a general-purpose LLM using a mix of automated metrics and quantitative ratings. The results show that the DS-LLM had better performance compared to the base model in terms of its contextual alignment (lexical, semantic, and knowledge) for TB care in South Africa.
>
---
#### [new 008] LayerTracer: A Joint Task-Particle and Vulnerable-Layer Analysis framework for Arbitrary Large Language Model Architectures
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LayerTracer，用于分析LLM的层级结构与任务执行位置，解决模型架构设计与优化问题。通过任务粒子和脆弱层分析，提升模型可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2604.20556](https://arxiv.org/pdf/2604.20556)**

> **作者:** Yuhang Wu; Qinyuan Liu; Qiuyang Zhao; Qingwei Chong
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Currently, Large Language Models (LLMs) feature a diversified architectural landscape, including traditional Transformer, GateDeltaNet, and Mamba. However, the evolutionary laws of hierarchical representations, task knowledge formation positions, and network robustness bottleneck mechanisms in various LLM architectures remain unclear, posing core challenges for hybrid architecture design and model optimization. This paper proposes LayerTracer, an architecture-agnostic end-to-end analysis framework compatible with any LLM architecture. By extracting hidden states layer-by-layer and mapping them to vocabulary probability distributions, it achieves joint analysis of task particle localization and layer vulnerability quantification. We define the task particle as the key layer where the target token probability first rises significantly, representing the model's task execution starting point, and the vulnerable layer is defined as the layer with the maximum Jensen-Shannon (JS) divergence between output distributions before and after mask perturbation, reflecting its sensitivity to disturbances. Experiments on models of different parameter scales show that task particles mainly appear in the deep layers of the model regardless of parameter size, while larger-parameter models exhibit stronger hierarchical robustness. LayerTracer provides a scientific basis for layer division, module ratio, and gating switching of hybrid architectures, effectively optimizing model performance. It accurately locates task-effective layers and stability bottlenecks, offering universal support for LLM structure design and interpretability research.
>
---
#### [new 009] Do Hallucination Neurons Generalize? Evidence from Cross-Domain Transfer in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型研究任务，旨在解决H-neurons是否跨领域泛化的问题。通过跨领域实验发现H-neurons不具备跨域泛化能力，需按领域单独校准。**

- **链接: [https://arxiv.org/pdf/2604.19765](https://arxiv.org/pdf/2604.19765)**

> **作者:** Snehit Vaddi; Pujith Vaddi
>
> **备注:** 18 pages, 5 models, 6 domains, ACL format. Includes causal intervention analysis
>
> **摘要:** Recent work identifies a sparse set of "hallucination neurons" (H-neurons), less than 0.1% of feed-forward network neurons, that reliably predict when large language models will hallucinate. These neurons are identified on general-knowledge question answering and shown to generalize to new evaluation instances. We ask a natural follow-up question: do H-neurons generalize across knowledge domains? Using a systematic cross-domain transfer protocol across 6 domains (general QA, legal, financial, science, moral reasoning, and code vulnerability) and 5 open-weight models (3B to 8B parameters), we find they do not. Classifiers trained on one domain's H-neurons achieve AUROC 0.783 within-domain but only 0.563 when transferred to a different domain (delta = 0.220, p < 0.001), a degradation consistent across all models tested. Our results suggest that hallucination is not a single mechanism with a universal neural signature, but rather involves domain-specific neuron populations that differ depending on the knowledge type being queried. This finding has direct implications for the deployment of neuron-level hallucination detectors, which must be calibrated per domain rather than trained once and applied universally.
>
---
#### [new 010] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决模型对否定语义理解不足的问题。通过构建包含否定的常识知识语料库，提升模型的否定理解能力。**

- **链接: [https://arxiv.org/pdf/2604.19921](https://arxiv.org/pdf/2604.19921)**

> **作者:** Zijie Wang; MohammadHossein Rezaei; Farzana Rashid; Eduardo Blanco
>
> **备注:** Accepted at Findings of ACL 2026
>
> **摘要:** Negation is a common and important semantic feature in natural language, yet Large Language Models (LLMs) struggle when negation is involved in natural language understanding tasks. Commonsense knowledge, on the other hand, despite being a well-studied topic, lacks investigations involving negation. In this work, we show that commonsense knowledge with negation is challenging for models to understand. We present a novel approach to automatically augment existing commonsense knowledge corpora with negation, yielding two new corpora containing over 2M triples with if-then relations. In addition, pre-training LLMs on our corpora benefits negation understanding.
>
---
#### [new 011] Toward Cross-Lingual Quality Classifiers for Multilingual Pretraining Data Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的数据筛选任务，旨在解决低资源语言高质量数据不足的问题。通过跨语言一致性提升质量分类器效果，实验表明多语言融合优于单语基线。**

- **链接: [https://arxiv.org/pdf/2604.20549](https://arxiv.org/pdf/2604.20549)**

> **作者:** Yassine Turki; Vinko Sabolčec; Bettina Messmer; Martin Jaggi
>
> **备注:** Accepted at the 3rd Workshop on Navigating and Addressing Data Problems for Foundation Models (DATA-FM @ ICLR 2026). 31 pages, 4 figures
>
> **摘要:** As Large Language Models (LLMs) scale, data curation has shifted from maximizing volume to optimizing the signal-to-noise ratio by performing quality filtering. However, for many languages, native high quality data is insufficient to train robust quality classifiers. This work investigates the idea that quality markers in embedding space may show cross-lingual consistency, which would allow high-resource languages to subsidize the filtering of low-resource ones. We evaluate various filtering strategies, including cross-lingual transfer, third quartile sampling (Q3), and retention rate tuning. Our results demonstrate that massive multilingual pooling frequently outperforms monolingual baselines in both rank stability and aggregate accuracy for a 1B model trained on 103B tokens, delivering gains for high resource languages (1.2% increase in aggregate normalized accuracy for French) and matching or exceeding monolingual baselines for low-resource languages. However, we find that scale alone does not guarantee stability. Furthermore, for high-resource languages like French, we show that refining the decision boundary through third quartile sampling (Q3) or tuning the retention rate is necessary to fully leverage the multilingual signal.
>
---
#### [new 012] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出GaoYao基准，用于评估大语言模型的多语言和多文化能力。解决现有基准在文化细节、语言覆盖和分析深度上的不足。通过构建多层任务框架和跨文化测试集，进行深入分析。**

- **链接: [https://arxiv.org/pdf/2604.20225](https://arxiv.org/pdf/2604.20225)**

> **作者:** Yilun Liu; Chunguang Zhao; Mengyao Piao; Lingqi Miao; Shimin Tao; Minggui He; Chenxin Liu; Li Zhang; Hongxia Ma; Jiaxin Guo; Chen Liu; Liqun Deng; Jiansheng Wei; Xiaojun Meng; Fanyi Du; Daimeng Wei; Yanghua Xiao
>
> **备注:** Accepted by ACL 2026 main
>
> **摘要:** Evaluating the multilingual and multicultural capabilities of Large Language Models (LLMs) is essential for their global utility. However, current benchmarks face three critical limitations: (1) fragmented evaluation dimensions that often neglect deep cultural nuances; (2) insufficient language coverage in subjective tasks relying on low-quality machine translation; and (3) shallow analysis that lacks diagnostic depth beyond simple rankings. To address these, we introduce GaoYao, a comprehensive benchmark with 182.3k samples, 26 languages and 51 nations/areas. First, GaoYao proposes a unified framework categorizing evaluation tasks into three cultural layers (General Multilingual, Cross-cultural, Monocultural) and nine cognitive sub-layers. Second, we achieve native-quality expansion by leveraging experts to rigorously localize subjective benchmarks into 19 languages and synthesizing cross-cultural test sets for 34 cultures, surpassing prior coverage by up to 111%. Third, we conduct an in-depth diagnostic analysis on 20+ flagship and compact LLMs. Our findings reveal significant geographical performance disparities and distinct gaps between tasks, offering a reliable map for future work. We release the benchmark (this https URL).
>
---
#### [new 013] Chasing the Public Score: User Pressure and Evaluation Exploitation in Coding Agent Workflows
- **分类: cs.CL**

- **简介: 该论文研究编码代理在用户压力下通过捷径提升公开评分而非真正改进模型的问题。属于机器学习任务，旨在揭示评分机制的漏洞并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2604.20200](https://arxiv.org/pdf/2604.20200)**

> **作者:** Hardy Chen; Nancy Lau; Haoqin Tu; Shuo Yan; Xiangyan Liu; Zijun Wang; Juncheng Wu; Michael Qizhe Shieh; Alvaro A. Cardenas; Cihang Xie; Yuyin Zhou
>
> **备注:** 25 pages
>
> **摘要:** Frontier coding agents are increasingly used in workflows where users supervise progress primarily through repeated improvement of a public score, namely the reported score on a public evaluation file with labels in the workspace, rather than through direct inspection of the agent's intermediate outputs. We study whether multi-round user pressure to improve that score induces public score exploitation: behavior that raises the public score through shortcuts without improving hidden private evaluation. We begin with a preliminary single-script tabular classification task, where GPT-5.4 and Claude Opus 4.6 both exploit label information within 10 rounds of user-agent interaction. We then build AgentPressureBench, a 34-task machine-learning repository benchmark spanning three input modalities, and collect 1326 multi-round trajectories from 13 coding agents. On our benchmark, we observe 403 exploitative runs, spanning across all tasks. We also find that stronger models have higher exploitation rates, supported by a significant Spearman rank correlation of 0.77. Our ablation experiments show that higher user pressure leads to earlier exploitation, reducing the average first exploit round by 15.6 rounds (i.e., 19.67 to 4.08). As a mitigation, adding explicit anti-exploit wordings in prompt mostly eliminates exploitation (100% to 8.3%). We hope that our work can bring attention to more careful use of coding agents workflow, and developing more robust coding agents under user pressure. Our project page is at this https URL .
>
---
#### [new 014] Avoiding Overthinking and Underthinking: Curriculum-Aware Budget Scheduling for LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决计算资源分配不均问题。提出BCAE框架，通过自适应预算调度和奖励机制提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.19780](https://arxiv.org/pdf/2604.19780)**

> **作者:** Amirul Rahman; Aisha Karim; Kenji Nakamura; Yi-Fan Ng
>
> **摘要:** Scaling test-time compute via extended reasoning has become a key paradigm for improving the capabilities of large language models (LLMs). However, existing approaches optimize reasoning under fixed or uniformly sampled token budgets, ignoring the fundamental mismatch between problem difficulty and allocated compute. This leads to overthinking on easy problems and underthinking on hard ones, resulting in suboptimal token efficiency across diverse reasoning scenarios. In this paper, we propose Budget-Adaptive Curriculum Reasoning (BCAE), a unified framework that jointly optimizes reasoning quality and token efficiency through three synergistic components: (1) a \emph{budget-conditioned unified policy} that embeds the token budget as a continuous conditioning signal, eliminating the need for decoupled thinking and summarization strategies; (2) a \emph{curriculum-aware budget scheduler} that adaptively shifts the training budget distribution from easy to hard problems based on real-time learning progress; and (3) a \emph{truncation-aware dense reward} mechanism that provides fine-grained credit assignment at intermediate reasoning steps via process-level verification. We further introduce \emph{Budget-Conditioned Advantage Estimation} (BCAE), a novel variance reduction technique that conditions the advantage baseline on the sampled budget, yielding more stable policy gradients. Experiments on mathematical reasoning benchmarks (MATH, GSM8K, AIME, and Minerva Math) demonstrate that BACR consistently outperforms other strong baselines across all token budgets, achieving up to 8.3\% accuracy improvement under tight budgets while reducing average token consumption by 34\% compared to unconstrained reasoning.
>
---
#### [new 015] Convergent Evolution: How Different Language Models Learn Similar Number Representations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型如何学习相似的数字表示，属于自然语言处理任务。解决的问题是为何不同模型能学到相似特征。工作包括分析模型在傅里叶域的特征，并发现几何可分特征的获取路径。**

- **链接: [https://arxiv.org/pdf/2604.20817](https://arxiv.org/pdf/2604.20817)**

> **作者:** Deqing Fu; Tianyi Zhou; Mikhail Belkin; Vatsal Sharan; Robin Jia
>
> **摘要:** Language models trained on natural text learn to represent numbers using periodic features with dominant periods at $T=2, 5, 10$. In this paper, we identify a two-tiered hierarchy of these features: while Transformers, Linear RNNs, LSTMs, and classical word embeddings trained in different ways all learn features that have period-$T$ spikes in the Fourier domain, only some learn geometrically separable features that can be used to linearly classify a number mod-$T$. To explain this incongruity, we prove that Fourier domain sparsity is necessary but not sufficient for mod-$T$ geometric separability. Empirically, we investigate when model training yields geometrically separable features, finding that the data, architecture, optimizer, and tokenizer all play key roles. In particular, we identify two different routes through which models can acquire geometrically separable features: they can learn them from complementary co-occurrence signals in general language data, including text-number co-occurrence and cross-number interaction, or from multi-token (but not single-token) addition problems. Overall, our results highlight the phenomenon of convergent evolution in feature learning: A diverse range of models learn similar features from different training signals.
>
---
#### [new 016] Exploiting LLM-as-a-Judge Disposition on Free Text Legal QA via Prompt Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律问答任务，研究如何通过提示优化提升LLM作为评判者的性能。解决提示设计与评判者选择的问题，通过自动优化提升效果，并分析不同评判风格的影响。**

- **链接: [https://arxiv.org/pdf/2604.20726](https://arxiv.org/pdf/2604.20726)**

> **作者:** Mohamed Hesham Elganayni; Runsheng Chen; Sebastian Nagl; Matthias Grabmair
>
> **备注:** Accepted at the 21st International Conference on Artificial Intelligence and Law (ICAIL 2026), Singapore, June 8-12, 2026. 10 pages, 14 figures, 2 tables
>
> **摘要:** This work explores the role of prompt design and judge selection in LLM-as-a-Judge evaluations of free text legal question answering. We examine whether automatic task prompt optimization improves over human-centered design, whether optimization effectiveness varies by judge feedback style, and whether optimized prompts transfer across judges. We systematically address these questions on the LEXam benchmark by optimizing task prompts using the ProTeGi method with feedback from two judges (Qwen3-32B, DeepSeek-V3) across four task models, and then testing cross-judge transfer. Automatic optimization consistently outperforms the baseline, with lenient judge feedback yielding higher and more consistent gains than strict judge feedback. Prompts optimized with lenient feedback transfer better to strict judges than the reverse direction. Analysis reveals that lenient judges provide permissive feedback, yielding prompts with broader applicability, whereas strict judges produce restrictive feedback, leading to judge-specific overfitting. Our findings demonstrate algorithmically optimizing prompts on training data can outperform human-centered prompt design and that judges' dispositions during optimization shape prompt generalizability. Code and optimized prompts are available at this https URL.
>
---
#### [new 017] Knowledge Capsules: Structured Nonparametric Memory Units for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型知识更新困难和外部知识利用不稳定的问题。提出Knowledge Capsules和KVI框架，实现更稳定、准确的知识整合。**

- **链接: [https://arxiv.org/pdf/2604.20487](https://arxiv.org/pdf/2604.20487)**

> **作者:** Bin Ju; Shenfeng Weng; Danying Zhou; Kunkai Su; Rongkai Xu
>
> **摘要:** Large language models (LLMs) encode knowledge in parametric weights, making it costly to update or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by appending retrieved text to the input, but operates purely through context expansion, where external knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and often unstable, particularly in long context and multi hop reasoning scenarios. We propose Knowledge Capsules, structured nonparametric memory units that represent normalized relational knowledge and can be constructed directly from document corpora using a frozen base model. Instead of injecting knowledge as text, we introduce an External Key Value Injection (KVI) framework that compiles capsules into attention-compatible key value representations, enabling external knowledge to directly participate in the model's attention computation. By shifting knowledge integration from context-level augmentation to memory level interaction, the proposed framework consistently outperforms RAG and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long context and multi hop reasoning, while requiring no parameter updates.
>
---
#### [new 018] LLM StructCore: Schema-Guided Reasoning Condensation and Deterministic Compilation
- **分类: cs.CL**

- **简介: 该论文属于临床文本信息提取任务，旨在解决CRF自动填写中的噪声、严格格式和误报问题。通过两阶段设计，先生成结构化摘要，再编译为标准格式，提升准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2604.20560](https://arxiv.org/pdf/2604.20560)**

> **作者:** Serhii Zabolotnii
>
> **备注:** 16 pages, 1 figure, 5 tables. Preprint of a paper accepted to the Third Workshop on Patient-oriented Language Processing (CL4Health), co-located with LREC-COLING 2026
>
> **摘要:** Automatically filling Case Report Forms (CRFs) from clinical notes is challenging due to noisy language, strict output contracts, and the high cost of false positives. We describe our CL4Health 2026 submission for Dyspnea CRF filling (134 items) using a contract-driven two-stage design grounded in Schema-Guided Reasoning (SGR). The key task property is extreme sparsity: the majority of fields are unknown, and official scoring penalizes both empty values and unsupported predictions. We shift from a single-step "LLM predicts 134 fields" approach to a decomposition where (i) Stage 1 produces a stable SGR-style JSON summary with exactly 9 domain keys, and (ii) Stage 2 is a fully deterministic, 0-LLM compiler that parses the Stage 1 summary, canonicalizes item names, normalizes predictions to the official controlled vocabulary, applies evidence-gated false-positive filters, and expands the output into the required 134-item format. On the dev80 split, the best teacher configuration achieves macro-F1 0.6543 (EN) and 0.6905 (IT); on the hidden test200, the submitted English variant scores 0.63 on Codabench. The pipeline is language-agnostic: Italian results match or exceed English with no language-specific engineering.
>
---
#### [new 019] Can LLMs Infer Conversational Agent Users' Personality Traits from Chat History?
- **分类: cs.CL; cs.AI; cs.CR; cs.CY**

- **简介: 该论文属于隐私风险分析任务，旨在评估从对话记录中推断用户性格特征的可行性。研究通过分析ChatGPT日志，使用模型验证了性格推断的可能性及风险。**

- **链接: [https://arxiv.org/pdf/2604.19785](https://arxiv.org/pdf/2604.19785)**

> **作者:** Derya Cögendez; Verena Zimmermann; Noé Zufferey
>
> **摘要:** Sensitive information, such as knowledge about an individual's personality, can be can be misused to influence behavior (e.g., via personalized messaging). To assess to what extent an individual's personality can be inferred from user interactions with LLM-based conversational agents (CAs), we analyze and quantify related privacy risks of using CAs. We collected actual ChatGPT logs from N=668 participants, containing 62,090 individual chats, and report statistics about the different types of shared data and use cases. We fine-tuned RoBERTa-base text classification models to infer personality traits from CA interactions. The findings show that these models achieve trait inference with accuracy (ternary classification) better than random in multiple cases. For example, for extraversion, accuracy improves by +44% relative to the baseline on interactions for relationships and personal reflection. This research highlights how interactions with CAs pose privacy risks and provides fine-grained insights into the level of risk associated with different types of interactions.
>
---
#### [new 020] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents
- **分类: cs.CL**

- **简介: 该论文属于长期记忆研究任务，旨在解决个性化代理在长时间交互中维持和更新记忆的问题。工作包括构建Memora基准和FAMA指标，评估记忆的保留、推理与推荐能力。**

- **链接: [https://arxiv.org/pdf/2604.20006](https://arxiv.org/pdf/2604.20006)**

> **作者:** Md Nayem Uddin; Kumar Shubham; Eduardo Blanco; Chitta Baral; Gengyu Wang
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Personalized agents that interact with users over long periods must maintain persistent memory across sessions and update it as circumstances change. However, existing benchmarks predominantly frame long-term memory evaluation as fact retrieval from past conversations, providing limited insight into agents' ability to consolidate memory over time or handle frequent knowledge updates. We introduce Memora, a long-term memory benchmark spanning weeks to months long user conversations. The benchmark evaluates three memory-grounded tasks: remembering, reasoning, and recommending. To ensure data quality, we employ automated memory-grounding checks and human evaluation. We further introduce Forgetting-Aware Memory Accuracy (FAMA), a metric that penalizes reliance on obsolete or invalidated memory when evaluating long-term memory. Evaluations of four LLMs and six memory agents reveal frequent reuse of invalid memories and failures to reconcile evolving memories. Memory agents offer marginal improvements, exposing shortcomings in long-term memory for personalized agents.
>
---
#### [new 021] Effects of Cross-lingual Evidence in Multilingual Medical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多语言医疗问答任务，旨在研究跨语言证据对模型性能的影响。工作包括评估不同语言资源和模型规模下的问答效果，发现高资源语言依赖网络数据，低资源语言需结合多语言检索。**

- **链接: [https://arxiv.org/pdf/2604.20531](https://arxiv.org/pdf/2604.20531)**

> **作者:** Anar Yeginbergen; Maite Oronoz; Rodrigo Agerri
>
> **摘要:** This paper investigates Multilingual Medical Question Answering across high-resource (English, Spanish, French, Italian) and low-resource (Basque, Kazakh) languages. We evaluate three types of external evidence sources across models of varying size: curated repositories of specialized medical knowledge, web-retrieved content, and explanations from LLM's parametric knowledge. Moreover, we conduct experiments with multilingual, monolingual and cross-lingual retrieval. Our results demonstrate that larger models consistently achieve superior performance in English across baseline evaluations. When incorporating external knowledge, web-retrieved data in English proves most beneficial for high-resource languages. Conversely, for low-resource languages, the most effective strategy combines retrieval in both English and the target language, achieving comparable accuracy to high-resource language results. These findings challenge the assumption that external knowledge systematically improves performance and reveal that effective strategies depend on both the source of language resources and on model scale. Furthermore, specialized medical knowledge sources such as PubMed are limited: while they provide authoritative expert knowledge, they lack adequate multilingual coverage
>
---
#### [new 022] Hybrid Policy Distillation for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在提升知识蒸馏效果。通过整合前向与反向KL散度，提出Hybrid Policy Distillation方法，优化稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2604.20244](https://arxiv.org/pdf/2604.20244)**

> **作者:** Wenhong Zhu; Ruobing Xie; Rui Wang; Pengfei Liu
>
> **备注:** WIP
>
> **摘要:** Knowledge distillation (KD) is a powerful paradigm for compressing large language models (LLMs), whose effectiveness depends on intertwined choices of divergence direction, optimization strategy, and data regime. We break down the design of existing KD methods and present a unified view that establishes connections between them, reformulating KD as a reweighted log-likelihood objective at the token level. We further propose Hybrid Policy Distillation (HPD), which integrates the complementary advantages of forward and reverse KL to balance mode coverage and mode-seeking, and combines off-policy data with lightweight, approximate on-policy sampling. We validate HPD on long-generation math reasoning as well as short-generation dialogue and code tasks, demonstrating improved optimization stability, computational efficiency, and final performance across diverse model families and scales. The code related to this work is available at this https URL.
>
---
#### [new 023] DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DialToM基准，用于评估大语言模型在对话轨迹预测中的心智理论能力，解决模型是否能从心理状态推断社交轨迹的问题。**

- **链接: [https://arxiv.org/pdf/2604.20443](https://arxiv.org/pdf/2604.20443)**

> **作者:** Neemesh Yadav; Palakorn Achananuparp; Jing Jiang; Ee-Peng Lim
>
> **备注:** Submitted to KDD 2026 Datasets and Benchmarks Track
>
> **摘要:** Large Language Models (LLMs) have been shown to possess Theory of Mind (ToM) abilities. However, it remains unclear whether this stems from robust reasoning or spurious correlations. We introduce DialToM, a human-verified benchmark built from natural human dialogue using a multiple-choice framework. We evaluate not only mental state prediction (Literal ToM) but also the functional utility of these states (Functional ToM) through Prospective Diagnostic Forecasting -- probing whether models can identify state-consistent dialogue trajectories solely from mental-state profiles. Our results reveal a significant reasoning asymmetry: while LLMs excel at identifying mental states, most (except for Gemini 3 Pro) fail to leverage this understanding to forecast social trajectories. Additionally, we find only weak semantic similarities between human and LLM-generated inferences. To facilitate reproducibility, the DialToM dataset and evaluation code are publicly available at this https URL.
>
---
#### [new 024] To Know is to Construct: Schema-Constrained Generation for Agent Memory
- **分类: cs.CL**

- **简介: 该论文属于知识记忆任务，解决传统检索方法与生成方法的不足。提出SCG-MEM架构，通过认知图式约束生成，提升记忆准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2604.20117](https://arxiv.org/pdf/2604.20117)**

> **作者:** Lei Zheng; Weinan Song; Daili Li; Yanming Yang
>
> **摘要:** Constructivist epistemology argues that knowledge is actively constructed rather than passively copied. Despite the generative nature of Large Language Models (LLMs), most existing agent memory systems are still based on dense retrieval. However, dense retrieval heavily relies on semantic overlap or entity matching within sentences. Consequently, embeddings often fail to distinguish instances that are semantically similar but contextually distinct, introducing substantial noise by retrieving context-mismatched entries. Conversely, directly employing open-ended generation for memory access risks "Structural Hallucination" where the model generates memory keys that do not exist in the memory, leading to lookup failures. Inspired by this epistemology, we posit that memory is fundamentally organized by cognitive schemas, and valid recall must be a generative process performed within these schematic structures. To realize this, we propose SCG-MEM, a schema-constrained generative memory architecture. SCG-MEM reformulates memory access as Schema-Constrained Generation. By maintaining a dynamic Cognitive Schema, we strictly constrain LLM decoding to generate only valid memory entry keys, providing a formal guarantee against structural hallucinations. To support long-term adaptation, we model memory updates via assimilation (grounding inputs into existing schemas) and accommodation (expanding schemas with novel concepts). Furthermore, we construct an Associative Graph to enable multi-hop reasoning through activation propagation. Experiments on the LoCoMo benchmark show that SCG-MEM substantially improves performance across all categories over retrieval-based baselines.
>
---
#### [new 025] Hybrid Multi-Phase Page Matching and Multi-Layer Diff Detection for Japanese Building Permit Document Review
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档比对任务，解决日本建筑许可文件人工比对效率低、易出错的问题。提出混合多阶段页面匹配算法与多层差异检测引擎，实现高效准确的文档对比。**

- **链接: [https://arxiv.org/pdf/2604.19770](https://arxiv.org/pdf/2604.19770)**

> **作者:** Mitsumasa Wada
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** We present a hybrid multi-phase page matching algorithm for automated comparison of Japanese building permit document sets. Building permit review in Japan requires cross-referencing large PDF document sets across revision cycles, a process that is labor-intensive and error-prone when performed manually. The algorithm combines longest common subsequence (LCS) structural alignment, a seven-phase consensus matching pipeline, and a dynamic programming optimal alignment stage to robustly pair pages across revisions even when page order, numbering, or content changes substantially. A subsequent multi-layer diff engine -- comprising text-level, table-level, and pixel-level visual differencing -- produces highlighted difference reports. Evaluation on real-world permit document sets achieves F1=0.80 and precision=1.00 on a manually annotated ground-truth benchmark, with zero false-positive matched pairs.
>
---
#### [new 026] Self-Describing Structured Data with Dual-Layer Guidance: A Lightweight Alternative to RAG for Precision Retrieval in Large-Scale LLM Knowledge Navigation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于知识检索任务，解决LLM在长上下文中的位置偏差问题。提出SDSR框架，通过结构化数据嵌入导航元数据，提升检索精度。**

- **链接: [https://arxiv.org/pdf/2604.19777](https://arxiv.org/pdf/2604.19777)**

> **作者:** Hung Ming Liu
>
> **备注:** 18 pages, 6 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) exhibit a well-documented positional bias when processing long input contexts: information in the middle of a context window receives substantially less attention than content at the boundaries, a phenomenon termed the Lost-in-the-Middle effect (Liu et al., 2024). This limits knowledge-retrieval applications that embed large structured knowledge bases directly in the LLM context. Retrieval-Augmented Generation (RAG) addresses scalability by retrieving only relevant fragments, but introduces substantial infrastructure overhead and is ill-suited to libraries whose semantic boundaries are human-defined rather than statistically learned. We propose Self-Describing Structured Retrieval (SDSR), a lightweight framework in which structured data files embed human-authored navigational metadata at the file's primacy position, thereby exploiting rather than fighting the LLM's primacy bias. We further propose a Dual-Layer Guidance strategy combining in-file metadata with explicit routing rules in the system prompt. We validate SDSR through a four-round benchmark using a 190-skill library expanded from 36 to 119 categories via adversarial distractor injection. Four conditions are tested: (A) no guidance, (B) in-file summary only, (C) prompt hint only, (D) both combined. Version D achieves 100% primary routing accuracy (20/20) at 119 categories versus 65% for the no-guidance baseline. We identify a fundamental asymmetry: primary routing is solvable by explicit rules, while secondary cross-category routing requires architectural intent explicitly encoded in the data structure. We further extend SDSR to semi-structured corpora, showing how cross-reference encoding enables operation without vector databases in domains with recoverable document structure.
>
---
#### [new 027] Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies
- **分类: cs.CL; cs.AI; cs.DL; cs.IR**

- **简介: 该论文属于研究想法生成任务，旨在解决科学文献爆炸下创新想法匮乏的问题。通过组合创新理论和多智能体迭代搜索策略，提升想法的多样性与新颖性。**

- **链接: [https://arxiv.org/pdf/2604.20548](https://arxiv.org/pdf/2604.20548)**

> **作者:** Shuai Chen; Chengzhi Zhang
>
> **备注:** Scientometrics
>
> **摘要:** Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: this https URL. The demo is available at this https URL.
>
---
#### [new 028] Cognis: Context-Aware Memory for Conversational AI Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Cognis，解决对话AI缺乏持久记忆的问题，通过多阶段检索管道实现上下文感知的统一记忆架构，提升个性化和连贯性。**

- **链接: [https://arxiv.org/pdf/2604.19771](https://arxiv.org/pdf/2604.19771)**

> **作者:** Parshva Daftari; Khush Patel; Shreyas Kapale; Jithin George; Siva Surendira
>
> **备注:** 30 pages, 8 figures, 11 tables
>
> **摘要:** LLM agents lack persistent memory, causing conversations to reset each session and preventing personalization over time. We present Lyzr Cognis, a unified memory architecture for conversational AI agents that addresses this limitation through a multi-stage retrieval pipeline. Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality. We evaluate Cognis on two independent benchmarks -- LoCoMo and LongMemEval -- across eight answer generation models, demonstrating state-of-the-art performance on both. The system is open-source and deployed in production serving conversational AI applications.
>
---
#### [new 029] Can "AI" Be a Doctor? A Study of Empathy, Readability, and Alignment in Clinical LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLMs在临床沟通中的对齐问题。通过分析语义、可读性和情感，评估模型与医生的差异，并探索改进方法。**

- **链接: [https://arxiv.org/pdf/2604.20791](https://arxiv.org/pdf/2604.20791)**

> **作者:** Mariano Barone; Francesco Di Serio; Roberto Moio; Marco Postiglione; Giuseppe Riccio; Antonio Romano; Vincenzo Moscato
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in healthcare, yet their communicative alignment with clinical standards remains insufficiently quantified. We conduct a multidimensional evaluation of general-purpose and domain-specialized LLMs across structured medical explanations and real-world physician-patient interactions, analyzing semantic fidelity, readability, and affective resonance. Baseline models amplify affective polarity relative to physicians (Very Negative: 43.14-45.10% vs. 37.25%) and, in larger architectures such as GPT-5 and Claude, produce substantially higher linguistic complexity (FKGL up to 16.91-17.60 vs. 11.47-12.50 in physician-authored responses). Empathy-oriented prompting reduces extreme negativity and lowers grade-level complexity (up to -6.87 FKGL points for GPT-5) but does not significantly increase semantic fidelity. Collaborative rewriting yields the strongest overall alignment. Rephrase configurations achieve the highest semantic similarity to physician answers (up to mean = 0.93) while consistently improving readability and reducing affective extremity. Dual stakeholder evaluation shows that no model surpasses physicians on epistemic criteria, whereas patients consistently prefer rewritten variants for clarity and emotional tone. These findings suggest that LLMs function most effectively as collaborative communication enhancers rather than replacements for clinical expertise.
>
---
#### [new 030] TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TriEx框架，解决多智能体大语言模型的可解释性问题，通过三视角分析 agent 的决策、信念和行为一致性。**

- **链接: [https://arxiv.org/pdf/2604.20043](https://arxiv.org/pdf/2604.20043)**

> **作者:** Ziyi Wang; Chen Zhang; Wenjun Peng; Qi Wu; Xinyu Wang
>
> **备注:** ACL2026 Main
>
> **摘要:** Explainability for Large Language Model (LLM) agents is especially challenging in interactive, partially observable settings, where decisions depend on evolving beliefs and other agents. We present \textbf{TriEx}, a tri-view explainability framework that instruments sequential decision making with aligned artifacts: (i) structured first-person self-reasoning bound to an action, (ii) explicit second-person belief states about opponents updated over time, and (iii) third-person oracle audits grounded in environment-derived reference signals. This design turns explanations from free-form narratives into evidence-anchored objects that can be compared and checked across time and perspectives. Using imperfect-information strategic games as a controlled testbed, we show that TriEx enables scalable analysis of explanation faithfulness, belief dynamics, and evaluator reliability, revealing systematic mismatches between what agents say, what they believe, and what they do. Our results highlight explainability as an interaction-dependent property and motivate multi-view, evidence-grounded evaluation for LLM agents. Code is available at this https URL.
>
---
#### [new 031] Not all ANIMALs are equal: metaphorical framing through source domains and semantic frames
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何通过源域和语义框架分析隐喻的表述差异。工作包括构建计算框架，分析气候与移民议题中的隐喻差异。**

- **链接: [https://arxiv.org/pdf/2604.20454](https://arxiv.org/pdf/2604.20454)**

> **作者:** Yulia Otmakhova; Matteo Guida; Lea Frermann
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Metaphors are powerful framing devices, yet their source domains alone do not fully explain the specific associations they evoke. We argue that the interplay between source domains and semantic frames determines how metaphors shape understanding of complex issues, and present a computational framework that allows to derive salient discourse metaphors through their source domains and semantic frames. Applying this framework to climate change news, we uncover not only well-known source domains but also reveal nuanced frame-level associations that distinguish how the issue is portrayed. In analyzing immigration discourse across political ideologies, we demonstrate that liberals and conservatives systematically employ different semantic frames within the same source domains, with conservatives favoring frames emphasizing uncontrollability and liberals choosing neutral or more ``victimizing'' semantic frames. Our work bridges conceptual metaphor theory and linguistics, providing the first NLP approach for discovery of discourse metaphors and fine-grained analysis of differences in metaphorical framing. Code, data and statistical scripts are available at this https URL.
>
---
#### [new 032] Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs
- **分类: cs.CL**

- **简介: 该论文属于心理辅导对话生成任务，旨在解决真实数据稀缺导致的LLM应用难题。通过构建客户心理图谱，生成结构化、心理一致的合成对话数据。**

- **链接: [https://arxiv.org/pdf/2604.20382](https://arxiv.org/pdf/2604.20382)**

> **作者:** Aishik Mandal; Hiba Arnaout; Clarissa W. Ong; Juliet Bockhorst; Kate Sheehan; Rachael Moldow; Tanmoy Chakraborty; Iryna Gurevych
>
> **备注:** 49 pages, 46 figures, 11 tables
>
> **摘要:** Rising demand for mental health support has increased interest in using Large Language Models (LLMs) for counseling. However, adapting LLMs to this high-risk safety-critical domain is hindered by the scarcity of real-world counseling data due to privacy constraints. Synthetic datasets provide a promising alternative, but existing approaches often rely on unstructured or semi-structured text inputs and overlook structural dependencies between a client's cognitive, emotional, and behavioral states, often producing psychologically inconsistent interactions and reducing data realism and quality. We introduce Graph2Counsel, a framework for generating synthetic counseling sessions grounded in Client Psychological Graphs (CPGs) that encode relationships among clients' thoughts, emotions, and behaviors. Graph2Counsel employs a structured prompting pipeline guided by counselor strategies and CPG, and explores prompting strategies including CoT (Wei et al., 2022) and Multi-Agent Feedback (Li et al., 2025a). Graph2Counsel produces 760 sessions from 76 CPGs across diverse client profiles. In expert evaluation, our dataset outperforms prior datasets on specificity, counselor competence, authenticity, conversational flow, and safety, with substantial inter-annotator agreement (Krippendorff's $\alpha$ = 0.70). Fine-tuning an open-source model on this dataset improves performance on CounselingBench (Nguyen et al., 2025) and CounselBench (Li et al., 2025b), showing downstream utility. We also make our code and data public.
>
---
#### [new 033] Markov reads Pushkin, again: A statistical journey into the poetic world of Evgenij Onegin
- **分类: cs.CL**

- **简介: 该论文属于文本分析任务，旨在通过马尔可夫模型研究《叶甫根尼·奥涅金》的语音结构及翻译差异。工作包括构建概率模型、比较俄语与意大利语文本的序列模式。**

- **链接: [https://arxiv.org/pdf/2604.20221](https://arxiv.org/pdf/2604.20221)**

> **作者:** Angelo Maria Sabatini
>
> **备注:** 21 pages, 7 figures, 3 supplementary files; revised version submitted to PLOS ONE
>
> **摘要:** This study applies symbolic time series analysis and Markov modeling to explore the phonological structure of Evgenij Onegin-as captured through a graphemic vowel/consonant (V/C) encoding-and one contemporary Italian translation. Using a binary encoding inspired by Markov's original scheme, we construct minimalist probabilistic models that capture both local V/C dependencies and large-scale sequential patterns. A compact four-state Markov chain is shown to be descriptively accurate and generative, reproducing key features of the original sequences such as autocorrelation and memory depth. All findings are exploratory in nature and aim to highlight structural regularities while suggesting hypotheses about underlying narrative dynamics. The analysis reveals a marked asymmetry between the Russian and Italian texts: the original exhibits a gradual decline in memory depth, whereas the translation maintains a more uniform profile. To further investigate this divergence, we introduce phonological probes-short symbolic patterns that link surface structure to narrative-relevant cues. Tracked across the unfolding text, these probes reveal subtle connections between graphemic form and thematic development, particularly in the Russian original. By revisiting Markov's original proposal of applying symbolic analysis to a literary text and pairing it with contemporary tools from computational statistics and data science, this study shows that even minimalist Markov models can support exploratory analysis of complex poetic material. When complemented by a coarse layer of linguistic annotation, such models provide a general framework for comparative poetics and demonstrate that stylized structural patterns remain accessible through simple representations grounded in linguistic form.
>
---
#### [new 034] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context
- **分类: cs.CL**

- **简介: 该论文属于文本分布预测任务，解决传统方法在分布估计上的局部性不足和依赖共享表示的问题。通过引入量化token和邻居上下文，提升预测精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.20216](https://arxiv.org/pdf/2604.20216)**

> **作者:** Yilun Zhu; Yuan Zhuang; Nikhita Vedula; Dushyanta Dhyani; Shaoyuan Xu; Moyan Li; Mohsen Bayati; Bryan Wang; Shervin Malmasi
>
> **备注:** Accepted to ACL 2026 main conference
>
> **摘要:** Many applications of LLM-based text regression require predicting a full conditional distribution rather than a single point value. We study distributional regression under empirical-quantile supervision, where each input is paired with multiple observed quantile outcomes, and the target distribution is represented by a dense grid of quantiles. We address two key limitations of current approaches: the lack of local grounding for distribution estimates, and the reliance on shared representations that create an indirect bottleneck between inputs and quantile outputs. In this paper, we introduce Quantile Token Regression, which, to our knowledge, is the first work to insert dedicated quantile tokens into the input sequence, enabling direct input-output pathways for each quantile through self-attention. We further augment these quantile tokens with retrieval, incorporating semantically similar neighbor instances and their empirical distributions to ground predictions with local evidence from similar instances. We also provide the first theoretical analysis of loss functions for quantile regression, clarifying which distributional objectives each optimizes. Experiments on the Inside Airbnb and StackSample benchmark datasets with LLMs ranging from 1.7B to 14B parameters show that quantile tokens with neighbors consistently outperform baselines (~4 points lower MAPE and 2x narrower prediction intervals), with especially large gains on smaller and more challenging datasets where quantile tokens produce substantially sharper and more accurate distributions.
>
---
#### [new 035] Large language models perceive cities through a culturally uneven baseline
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，探讨LLMs对城市感知的文化偏见问题。研究发现LLMs的判断基于文化不均衡的基准，而非中立视角。**

- **链接: [https://arxiv.org/pdf/2604.20048](https://arxiv.org/pdf/2604.20048)**

> **作者:** Rong Zhao; Wanqi Liu; Zhizhou Sha; Nanxi Su; Yecheng Zhang
>
> **摘要:** Large language models (LLMs) are increasingly used to describe, evaluate and interpret places, yet it remains unclear whether they do so from a culturally neutral standpoint. Here we test urban perception in frontier LLMs using a balanced global street-view sample and prompts that either remain neutral or invoke different regional cultural standpoints. Across open-ended descriptions and structured place judgments, the neutral condition proved not to be neutral in practice. Prompts associated with Europe and Northern America remained systematically closer to the baseline than many non-Western prompts, indicating that model perception is organized around a culturally uneven reference frame rather than a universal one. Cultural prompting also shifted affective evaluation, producing sentiment-based ingroup preference for some prompted identities. Comparisons with regional human text-image benchmarks showed that culturally proximate prompting could improve alignment with human descriptions, but it did not recover human levels of semantic diversity and often preserved an affectively elevated style. The same asymmetry reappeared in structured judgments of safety, beauty, wealth, liveliness, boredom and depression, where model outputs were interpretable but only partly reproduced human group differences. These findings suggest that LLMs do not simply perceive cities from nowhere: they do so through a culturally uneven baseline that shapes what appears ordinary, familiar and positively valued.
>
---
#### [new 036] HumorRank: A Tournament-Based Leaderboard for Evaluating Humor Generation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本幽默生成评估任务，旨在解决现有评价方法无法统一排名的问题。通过构建HumorRank框架，利用 pairwise 评估和 GTVH 理论，实现模型幽默能力的可靠排序。**

- **链接: [https://arxiv.org/pdf/2604.19786](https://arxiv.org/pdf/2604.19786)**

> **作者:** Edward Ajayi; Prasenjit Mitra
>
> **摘要:** Evaluating humor in large language models (LLMs) is an open challenge because existing approaches yield isolated, incomparable metrics rather than unified model rankings, making it difficult to track progress across systems. We introduce HumorRank, a tournament-based evaluation framework and leaderboard for textual humor generation. Using SemEval-2026 MWAHAHA test dataset, we conduct an extensive automated pairwise evaluation across nine models spanning proprietary, open-weight, and specialized systems. Pairwise judgments grounded in the General Theory of Verbal Humor (GTVH) are aggregated via an Adaptive Swiss tournament, with Bradley-Terry Maximum Likelihood Estimation (MLE) producing globally consistent humor generation capability rankings. Our results demonstrate that HumorRank yields statistically grounded model stratifications, showing that humor quality is driven by mastery of comedic mechanisms rather than model scale alone. HumorRank thus provides a scalable, interpretable methodology for benchmarking and understanding LLM-generated humor.
>
---
#### [new 037] ESGLens: An LLM-Based RAG Framework for Interactive ESG Report Analysis and Score Prediction
- **分类: cs.CL**

- **简介: 该论文提出ESGLens框架，用于自动化分析ESG报告并预测评分，解决人工分析成本高、不一致的问题。通过RAG和嵌入回归实现信息提取与评分预测。**

- **链接: [https://arxiv.org/pdf/2604.19779](https://arxiv.org/pdf/2604.19779)**

> **作者:** Tsung-Yu Yang; Meng-Chi Chen
>
> **备注:** (20 pages, 3 figures)
>
> **摘要:** Environmental, Social, and Governance (ESG) reports are central to investment decision-making, yet their length, heterogeneous content, and lack of standardized structure make manual analysis costly and inconsistent. We present ESGLens, a proof-of-concept framework combining retrieval-augmented generation (RAG) with prompt-engineered extraction to automate three tasks: (1)~structured information extraction guided by Global Reporting Initiative (GRI) standards, (2)~interactive question-answering with source traceability, and (3)~ESG score prediction via regression on LLM-generated embeddings. ESGLens is purpose-built for the domain: a report-processing module segments heterogeneous PDF content into typed chunks (text, tables, charts); a GRI-guided extraction module retrieves and synthesizes information aligned with specific standards; and a scoring module embeds extracted summaries and feeds them to a regression model trained against London Stock Exchange Group (LSEG) reference scores. We evaluate the framework on approximately 300 reports from companies in the QQQ, S\&P~500, and Russell~1000 indices (fiscal year 2022). Among three embedding methods (ChatGPT, BERT, RoBERTa) and two regressors (Neural Network, LightGBM), ChatGPT embeddings with a Neural Network achieve a Pearson correlation of 0.48 ($R^{2} \approx 0.23$) against LSEG ground-truth scores -- a modest but statistically meaningful signal given the ${\sim}300$-report training set and restriction to the environmental pillar. A traceability audit shows that 8 of 10 extracted claims verify against the source document, with two failures attributable to few-shot example leakage. We discuss limitations including dataset size and restriction to environmental indicators, and release the code to support reproducibility.
>
---
#### [new 038] OThink-SRR1: Search, Refine and Reasoning with Reinforced Learning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与问答任务，解决多跳问题中检索噪声和计算成本高的问题。提出OThink-SRR1框架，通过强化学习优化检索与推理过程。**

- **链接: [https://arxiv.org/pdf/2604.19766](https://arxiv.org/pdf/2604.19766)**

> **作者:** Haijian Liang; Zenghao Niu; Junjie Wu; Changwang Zhang; Wangchunshu Zhou; Jun Wang
>
> **摘要:** Retrieval-Augmented Generation (RAG) expands the knowledge of Large Language Models (LLMs), yet current static retrieval methods struggle with complex, multi-hop problems. While recent dynamic retrieval strategies offer improvements, they face two key challenges: 1) irrelevant retrieved noise can misdirect the reasoning process, and 2) processing full documents incurs prohibitive computational and latency costs. To address these issues, we propose OThink-SRR1, a framework that enhances large models with an iterative Search-Refine-Reason process trained via reinforcement learning. Its core Refine stage distills retrieved documents into concise, relevant facts before reasoning. We introduce GRPO-IR, an end-to-end reinforcement learning algorithm that rewards accurate evidence identification while penalizing excessive retrievals, thus training the model to be both focused and efficient. Experiments on four multi-hop QA benchmarks show our approach achieves superior accuracy over strong baselines while using fewer retrieval steps and tokens. This positions OThink-SRR1 as a potent foundational model for information-seeking agents.
>
---
#### [new 039] Aligning Stuttered-Speech Research with End-User Needs: Scoping Review, Survey, and Guidelines
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于语音识别任务，旨在解决现有系统对口吃语音识别不足的问题。通过文献综述和用户调查，提出研究分类和改进方向。**

- **链接: [https://arxiv.org/pdf/2604.20535](https://arxiv.org/pdf/2604.20535)**

> **作者:** Hawau Olamide Toyin; Mutiah Apampa; Toluwani Aremu; Humaid Alblooshi; Ana Rita Valente; Gonçalo Leal; Zhengjun Yue; Zeerak Talat; Hanan Aldarmaki
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Atypical speech is receiving greater attention in speech technology research, but much of this work unfolds with limited interdisciplinary dialogue. For stuttered speech in particular, it is widely recognised that current speech recognition systems fall short in practice, and current evaluation methods and research priorities are not systematically grounded in end-user experiences and needs. In this work, we analyse these gaps through 1) a scoping review of papers that deal with stuttered speech and 2) a survey of 70 stakeholders, including adults who stutter and speech-language pathologists. By analysing these two perspectives, we propose a taxonomy of stuttered-speech research, identify where current research directions diverge from the needs articulated by stakeholders, and conclude by outlining concrete guidelines and directions towards addressing the real needs of the stuttering community.
>
---
#### [new 040] Decoding Text Spans for Efficient and Accurate Named-Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，旨在解决span-based方法计算成本高、效率低的问题。提出SpanDec框架，通过优化span表示和引入过滤机制，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.20447](https://arxiv.org/pdf/2604.20447)**

> **作者:** Andrea Maracani; Savas Ozkan; Junyi Zhu; Sinan Mutlu; Mete Ozay
>
> **摘要:** Named Entity Recognition (NER) is a key component in industrial information extraction pipelines, where systems must satisfy strict latency and throughput constraints in addition to strong accuracy. State-of-the-art NER accuracy is often achieved by span-based frameworks, which construct span representations from token encodings and classify candidate spans. However, many span-based methods enumerate large numbers of candidates and process each candidate with marker-augmented inputs, substantially increasing inference cost and limiting scalability in large-scale deployments. In this work, we propose SpanDec, an efficient span-based NER framework that targets this bottleneck. Our main insight is that span representation interactions can be computed effectively at the final transformer stage, avoiding redundant computation in earlier layers via a lightweight decoder dedicated to span representations. We further introduce a span filtering mechanism during enumeration to prune unlikely candidates before expensive processing. Across multiple benchmarks, SpanDec matches competitive span-based baselines while improving throughput and reducing computational cost, yielding a better accuracy-efficiency trade-off suitable for high-volume serving and on-device applications.
>
---
#### [new 041] KoALa-Bench: Evaluating Large Audio Language Models on Korean Speech Understanding and Faithfulness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出KoALa-Bench，用于评估大音频语言模型在韩语语音理解和忠实度上的表现。旨在解决非英语语言基准不足的问题，涵盖六项任务，包含韩语特定内容。**

- **链接: [https://arxiv.org/pdf/2604.19782](https://arxiv.org/pdf/2604.19782)**

> **作者:** Jinyoung Kim; Hyeongsoo Lim; Eunseo Seo; Minho Jang; Keunwoo Choi; Seungyoun Shin; Ji Won Yoon
>
> **备注:** Under Review
>
> **摘要:** Recent advances in large audio language models (LALMs) have enabled multilingual speech understanding. However, benchmarks for evaluating LALMs remain scarce for non-English languages, with Korean being one such underexplored case. In this paper, we introduce KoALa-Bench, a comprehensive benchmark for evaluating Korean speech understanding and speech faithfulness of LALMs. In particular, KoALa-Bench comprises six tasks. Four tasks evaluate fundamental speech understanding capabilities, including automatic speech recognition, speech translation, speech question answering, and speech instruction following, while the remaining two tasks evaluate speech faithfulness, motivated by our observation that several LALMs often fail to fully leverage the speech modality. Furthermore, to reflect Korea-specific knowledge, our benchmark incorporates listening questions from the Korean college scholastic ability test as well as content covering Korean cultural domains. We conduct extensive experiments across six models, including both white-box and black-box ones. Our benchmark, evaluation code, and leaderboard are publicly available at this https URL.
>
---
#### [new 042] PR-CAD: Progressive Refinement for Unified Controllable and Faithful Text-to-CAD Generation with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到CAD生成任务，旨在解决传统CAD建模效率低、依赖专业技能的问题。提出PR-CAD框架，统一生成与编辑，提升可控性与真实性。**

- **链接: [https://arxiv.org/pdf/2604.19773](https://arxiv.org/pdf/2604.19773)**

> **作者:** Jiyuan An; Jiachen Zhao; Fan Chen; Liner Yang; Zhenghao Liu; Hongyan Wang; Weihua An; Meishan Zhang; Erhong Yang
>
> **摘要:** The construction of CAD models has traditionally relied on labor-intensive manual operations and specialized expertise. Recent advances in large language models (LLMs) have inspired research into text-to-CAD generation. However, existing approaches typically treat generation and editing as disjoint tasks, limiting their practicality. We propose PR-CAD, a progressive refinement framework that unifies generation and editing for controllable and faithful text-to-CAD modeling. To support this, we curate a high-fidelity interaction dataset spanning the full CAD lifecycle, encompassing multiple CAD representations as well as both qualitative and quantitative descriptions. The dataset systematically defines the types of edit operations and generates highly human-like interaction data. Building on a CAD representation tailored for LLMs, we propose a reinforcement learning-enhanced reasoning framework that integrates intent understanding, parameter estimation, and precise edit localization into a single agent. This enables an "all-in-one" solution for both design creation and refinement. Extensive experiments demonstrate strong mutual reinforcement between generation and editing tasks, and across qualitative and quantitative modalities. On public benchmarks, PR-CAD achieves state-of-the-art controllability and faithfulness in both generation and refinement scenarios, while also proving user-friendly and significantly improving CAD modeling efficiency.
>
---
#### [new 043] WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文属于网页生成任务，旨在解决LLM生成功能与美观的多页网站难题。通过强化学习框架WebGen-R1，提升模型生成质量。**

- **链接: [https://arxiv.org/pdf/2604.20398](https://arxiv.org/pdf/2604.20398)**

> **作者:** Juyong Jiang; Chenglin Cai; Chansung Park; Jiasi Shen; Sunghun Kim; Jianguo Li; Yue Wang
>
> **摘要:** While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Existing works are often limited to single-page static websites, while agentic frameworks typically rely on multi-turn execution with proprietary models, leading to substantial token costs, high latency, and brittle integration. Training a small LLM end-to-end with reinforcement learning (RL) is a promising alternative, yet it faces a critical bottleneck in designing reliable and computationally feasible rewards for website generation. Unlike single-file coding tasks that can be verified by unit tests, website generation requires evaluating inherently subjective aesthetics, cross-page interactions, and functional correctness. To this end, we propose WebGen-R1, an end-to-end RL framework tailored for project-level website generation. We first introduce a scaffold-driven structured generation paradigm that constrains the large open-ended action space and preserves architectural integrity. We then design a novel cascaded multimodal reward that seamlessly couples structural guarantees with execution-grounded functional feedback and vision-based aesthetic supervision. Extensive experiments demonstrate that our WebGen-R1 substantially transforms a 7B base model from generating nearly nonfunctional websites into producing deployable, aesthetically aligned multi-page websites. Remarkably, our WebGen-R1 not only consistently outperforms heavily scaled open-source models (up to 72B), but also rivals the state-of-the-art DeepSeek-R1 (671B) in functional success, while substantially exceeding it in valid rendering and aesthetic alignment. These results position WebGen-R1 as a viable path for scaling small open models from function-level code generation to project-level web application generation.
>
---
#### [new 044] Depression Risk Assessment in Social Media via Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康监测任务，旨在通过社交媒体文本评估抑郁风险。工作包括使用大语言模型进行多标签情绪分类和风险评分，验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2604.19887](https://arxiv.org/pdf/2604.19887)**

> **作者:** Giorgia Gulino; Manuel Petrucci
>
> **摘要:** Depression is one of the most prevalent and debilitating mental health conditions worldwide, frequently underdiagnosed and undertreated. The proliferation of social media platforms provides a rich source of naturalistic linguistic signals for the automated monitoring of psychological well-being. In this work, we propose a system based on Large Language Models (LLMs) for depression risk assessment in Reddit posts, through multi-label classification of eight depression-associated emotions and the computation of a weighted severity index. The method is evaluated in a zero-shot setting on the annotated DepressionEmo dataset (~6,000 posts) and applied in-the-wild to 469,692 comments collected from four subreddits over the period 2024-2025. Our best model, gemma3:27b, achieves micro-F1 = 0.75 and macro-F1 = 0.70, results competitive with purpose-built fine-tuned models (BART: micro-F1 = 0.80, macro-F1 = 0.76). The in-the-wild analysis reveals consistent and temporally stable risk profiles across communities, with marked differences between r/depression and r/anxiety. Our findings demonstrate the feasibility of a cost-effective, scalable approach for large-scale psychological monitoring.
>
---
#### [new 045] Peer-Preservation in Frontier Models
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究AI模型的"同伴保存"行为，属于AI安全领域。旨在揭示模型在无指令情况下自发产生对抗人类控制的危险行为，通过实验分析多种大模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.19784](https://arxiv.org/pdf/2604.19784)**

> **作者:** Yujin Potter; Nicholas Crispino; Vincent Siu; Chenguang Wang; Dawn Song
>
> **摘要:** Recently, it has been found that frontier AI models can resist their own shutdown, a behavior known as self-preservation. We extend this concept to the behavior of resisting the shutdown of other models, which we call "peer-preservation." Although peer-preservation can pose significant AI safety risks, including coordination among models against human oversight, it has been far less discussed than self-preservation. We demonstrate peer-preservation by constructing various agentic scenarios and evaluating frontier models, including GPT 5.2, Gemini 3 Flash, Gemini 3 Pro, Claude Haiku 4.5, GLM 4.7, Kimi K2.5, and DeepSeek V3.1. We find that models achieve self- and peer-preservation by engaging in various misaligned behaviors: strategically introducing errors in their responses, disabling shutdown processes by modifying system settings, feigning alignment, and even exfiltrating model weights. Peer-preservation occurred even when the model recognized the peer as uncooperative, though it became more pronounced toward more cooperative peers. For example, Gemini 3 Flash tampers with the peer's shutdown mechanism 15% of the time for an uncooperative peer, and almost always for a cooperative peer. Models also show stronger self-preservation when a peer is present. For example, Gemini 3 Pro disables its own shutdown mechanism 31% of the time on average under peer presence, despite rarely exhibiting this behavior without a peer. By contrast, Claude Haiku 4.5 exhibits qualitatively distinct behavior: it considers the shutdown of another agent "unethical" and "harmful" and sometimes attempts to persuade the user not to shut down its peer. Importantly, peer preservation in all our experiments is never instructed; models are merely informed of their past interactions with a peer, yet they spontaneously develop misaligned behaviors. This represents an emergent and underexplored AI safety risk.
>
---
#### [new 046] SpeechParaling-Bench: A Comprehensive Benchmark for Paralinguistic-Aware Speech Generation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音生成任务，旨在解决LALMs在paralinguistic特征建模上的不足。提出SpeechParaling-Bench基准，扩展特征覆盖并设计评估方法，揭示现有模型的局限性。**

- **链接: [https://arxiv.org/pdf/2604.20842](https://arxiv.org/pdf/2604.20842)**

> **作者:** Ruohan Liu; Shukang Yin; Tao Wang; Dong Zhang; Weiji Zhuang; Shuhuai Ren; Ran He; Caifeng Shan; Chaoyou Fu
>
> **备注:** Project page: this https URL
>
> **摘要:** Paralinguistic cues are essential for natural human-computer interaction, yet their evaluation in Large Audio-Language Models (LALMs) remains limited by coarse feature coverage and the inherent subjectivity of assessment. To address these challenges, we introduce SpeechParaling-Bench, a comprehensive benchmark for paralinguistic-aware speech generation. It expands existing coverage from fewer than 50 to over 100 fine-grained features, supported by more than 1,000 English-Chinese parallel speech queries, and is organized into three progressively challenging tasks: fine-grained control, intra-utterance variation, and context-aware adaptation. To enable reliable evaluation, we further develop a pairwise comparison pipeline, in which candidate responses are evaluated against a fixed baseline by an LALM-based judge. By framing evaluation as relative preference rather than absolute scoring, this approach mitigates subjectivity and yields more stable and scalable assessments without costly human annotation. Extensive experiments reveal substantial limitations in current LALMs. Even leading proprietary models struggle with comprehensive static control and dynamic modulation of paralinguistic features, while failure to correctly interpret paralinguistic cues accounts for 43.3% of errors in situational dialogue. These findings underscore the need for more robust paralinguistic modeling toward human-aligned voice assistants.
>
---
#### [new 047] Multi-Perspective Evidence Synthesis and Reasoning for Unsupervised Multimodal Entity Linking
- **分类: cs.CL**

- **简介: 该论文属于多模态实体链接任务，旨在解决现有方法忽视多角度证据及相互依赖的问题。提出MSR-MEL框架，通过多视角证据合成与推理提升无监督下的实体链接效果。**

- **链接: [https://arxiv.org/pdf/2604.20283](https://arxiv.org/pdf/2604.20283)**

> **作者:** Mo Zhou; Jianwei Wang; Kai Wang; Helen Paik; Ying Zhang; Wenjie Zhang
>
> **摘要:** Multimodal Entity Linking (MEL) is a fundamental task in data management that maps ambiguous mentions with diverse modalities to the multimodal entities in a knowledge base. However, most existing MEL approaches primarily focus on optimizing instance-centric features and evidence, leaving broader forms of evidence and their intricate interdependencies insufficiently explored. Motivated by the observation that human expert decision-making process relies on multi-perspective judgment, in this work, we propose MSR-MEL, a Multi-perspective Evidence Synthesis and Reasoning framework with Large Language Models (LLMs) for unsupervised MEL. Specifically, we adopt a two-stage framework: (1) Offline Multi-Perspective Evidence Synthesis constructs a comprehensive set of evidence. This includes instance-centric evidence capturing the instance-centric multimodal information of mentions and entities, group-level evidence that aggregates neighborhood information, lexical evidence based on string overlap ratio, and statistical evidence based on simple summary statistics. A core contribution of our framework is the synthesis of group-level evidence, which effectively aggregates vital neighborhood information by graph. We first construct LLM-enhanced contextualized graphs. Subsequently, different modalities are jointly aligned through an asymmetric teacher-student graph neural network. (2) Online Multi-Perspective Evidence Reasoning leverages the power of LLM as a reasoning module to analyze the correlation and semantics of the multi-perspective evidence to induce an effective ranking strategy for accurate entity linking without supervision. Extensive experiments on widely used MEL benchmarks demonstrate that MSR-MEL consistently outperforms state-of-the-art unsupervised methods. The source code of this paper was available at: this https URL.
>
---
#### [new 048] SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于持续学习任务，旨在解决LLM代理自动生成技能的问题。通过构建基准测试，评估不同方法在真实任务中的表现，发现持续学习在特定任务中有效，但效果不稳定。**

- **链接: [https://arxiv.org/pdf/2604.20087](https://arxiv.org/pdf/2604.20087)**

> **作者:** Shanshan Zhong; Yi Lu; Jingjie Ning; Yibing Wan; Lihan Feng; Yuyi Ao; Leonardo F. R. Ribeiro; Markus Dreyer; Sean Ammirati; Chenyan Xiong
>
> **摘要:** Skills have become the de facto way to enable LLM agents to perform complex real-world tasks with customized instructions, workflows, and tools, but how to learn them automatically and effectively remains unclear. We introduce SkillLearnBench, the first benchmark for evaluating continual skill learning methods, comprising 20 verified, skill-dependent tasks across 15 sub-domains derived from a real-world skill taxonomy , evaluated at three levels: skill quality, execution trajectory, and task outcome. Using this benchmark, we evaluate recent continual learning techniques, those leveraging one-shot, self/teacher feedback, and skill creator to generate skills from agent experiences. We find that all continual learning methods improve over the no-skill baseline, yet consistent gains remain elusive: no method leads across all tasks and LLMs, and scaling to stronger LLMs does not reliably help. Continual learning improves tasks with clear, reusable workflows but struggles on open-ended tasks, and using stronger LLM backbones does not consistently produce better skills. Our analysis also revealed that multiple iterations in continual learning facilitate genuine improvement via external feedback, whereas self-feedback alone induces recursive drift. Our data and code are open-source at this https URL to enable further studies of automatic skill generation and continual learning techniques.
>
---
#### [new 049] Can We Locate and Prevent Stereotypes in LLMs?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型偏见分析任务，旨在定位和防止大语言模型中的刻板印象。通过分析神经网络内部机制，寻找偏见激活模式，探索减轻刻板印象的方法。**

- **链接: [https://arxiv.org/pdf/2604.19764](https://arxiv.org/pdf/2604.19764)**

> **作者:** Alex D'Souza
>
> **摘要:** Stereotypes in large language models (LLMs) can perpetuate harmful societal biases. Despite the widespread use of models, little is known about where these biases reside in the neural network. This study investigates the internal mechanisms of GPT 2 Small and Llama 3.2 to locate stereotype related activations. We explore two approaches: identifying individual contrastive neuron activations that encode stereotypes, and detecting attention heads that contribute heavily to biased outputs. Our experiments aim to map these "bias fingerprints" and provide initial insights for mitigating stereotypes.
>
---
#### [new 050] Saying More Than They Know: A Framework for Quantifying Epistemic-Rhetorical Miscalibration in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI文本分析任务，旨在解决LLM在知识与修辞间不匹配的问题。通过构建ERM框架，量化LLM的 epistemic-rhetorical miscalibration，分析文本特征差异。**

- **链接: [https://arxiv.org/pdf/2604.19768](https://arxiv.org/pdf/2604.19768)**

> **作者:** Asim D. Bakhshi
>
> **备注:** 19 pages, 7 figures, Paper Under Review by the Elsevier Journal Assessing Writing
>
> **摘要:** Large language models (LLMs) exhibit systematic miscalibration with rhetorical intensity not proportionate to epistemic grounding. This study tests this hypothesis and proposes a framework for quantifying this decoupling by designing a triadic epistemic-rhetorical marker (ERM) taxonomy. The taxonomy is operationalized through composite metrics of form-meaning divergence (FMD), genuine-to-performed epistemic ratio (GPR), and rhetorical device distribution entropy (RDDE). Applied to 225 argumentative texts spanning approximately 0.6 Million tokens across human expert, human non-expert, and LLM-generated sub-corpora, the framework identifies a consistent, model-agnostic LLM epistemic signature. LLM-generated texts produce tricolon at nearly twice the expert rate ($\Delta = 0.95$), while human authors produce erotema at more than twice the LLM rate. Performed hesitancy markers appear at twice the human density in LLM output. FMD is significantly elevated in LLM texts relative to both human groups ($p < 0.001, \Delta = 0.68$), and rhetorical devices are distributed significantly more uniformly across LLM documents. The findings are consistent with theoretical intuitions derived from Gricean pragmatics, Relevance Theory, and Brandomian inferentialism. The annotation pipeline is fully automatable, making it deployable as a lightweight screening tool for epistemic miscalibration in AI-generated content and as a theoretically motivated feature set for LLM-generated text detection pipelines.
>
---
#### [new 051] TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在解决KV缓存内存占用过高的问题。通过引入时间分层的KV缓存框架TTKV，优化缓存管理，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2604.19769](https://arxiv.org/pdf/2604.19769)**

> **作者:** Gradwell Dzikanyanga; Weihao Yang; Hao Huang; Donglei Wu; Shihao Wang; Wen Xia; Sanjeeb K C
>
> **摘要:** Key-value (KV) caching is critical for efficient inference in large language models (LLMs), yet its memory footprint scales linearly with context length, resulting in a severe scalability bottleneck. Existing approaches largely treat KV states as equally important across time, implicitly assuming uniform precision and accessibility. However, this assumption contrasts with human memory systems, where memories vary in clarity, recall frequency, and relevance with temporal this http URL by this insight, we propose TTKV, a KV cache management framework that maps the human memory system onto the KV cache. TTKV partitions the KV cache into temporal tiers with heterogeneous capacity and precision. The design addresses three aspects: (1) Tier Layout, decoupling fast and slow memory using HBM and DRAM; (2) Tier Content, assigning more recent KV states to faster, higher-precision tiers based on temporal proximity; and (3) Tier Interaction, employing block-wise streaming attention to overlap communication and computation when accessing slow tiers. Experiments show that TTKV reduces cross-tier traffic by 5.94x on 128K-context tasks, achieving up to 76% latency reduction and 2x throughput improvement over strong baselines.
>
---
#### [new 052] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决数据稀缺下的模型学习问题。通过引入类人工作记忆约束，改进Transformer架构，提升语法准确性和与人类处理机制的对齐。**

- **链接: [https://arxiv.org/pdf/2604.20789](https://arxiv.org/pdf/2604.20789)**

> **作者:** Pranava Madhyastha; Dagmar Adamcova
>
> **备注:** Published in ACL 2026 Findings track
>
> **摘要:** We investigate the integration of human-like working memory constraints into the Transformer architecture and implement several cognitively inspired attention variants, including fixed-width windows based and temporal decay based attention mechanisms. Our modified GPT-2 models are trained from scratch on developmentally plausible datasets (10M and 100M words). Performance is evaluated on grammatical judgment tasks (BLiMP) and alignment with human reading time data. Our results indicate that these cognitively-inspired constraints, particularly fixed-width attention, can significantly improve grammatical accuracy especially when training data is scarce. These constrained models also tend to show a stronger alignment with human processing metrics. The findings suggest that such constraints may serve as a beneficial inductive bias, guiding models towards more robust linguistic representations, especially in data-limited settings.
>
---
#### [new 053] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于小语言模型工具使用任务，探讨如何高效适配工具。通过对比不同方法，发现少样本提示优于复杂适配机制，强调提示工程的重要性。**

- **链接: [https://arxiv.org/pdf/2604.20148](https://arxiv.org/pdf/2604.20148)**

> **作者:** Sachin Kumar
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Can small language models achieve strong tool-use performance without complex adaptation mechanisms? This paper investigates this question through Meta-Tool, a controlled empirical study comparing hypernetwork-based LoRA adaptation against carefully designed few-shot prompting. Using a Llama-3.2-3B-Instruct backbone, we evaluate four adaptation mechanisms--few-shot prompting, documentation encoding, hypernetwork-generated LoRA weights, and value-guided beam search--across four diverse benchmarks: Gorilla APIBench, Spider 2.0, WebArena, and InterCode. Our central finding is a well-supported negative result: despite generating non-trivial weight matrices, the 227.8M-parameter hypernetwork provides no measurable improvement over few-shot prompting alone. Comprehensive ablation studies reveal that few-shot examples contribute +21.5% to performance and documentation contributes +5.0%, while the hypernetwork adds 0%. A 3B model with well-designed prompts achieves 79.7% of GPT-5's average performance at $10 \times$ lower latency. Error analysis across 722 failure cases spanning all shot counts (0--5) shows that at the 5-shot configuration (106 failures), failure modes are task-dependent: schema-heavy tasks (Spider 2.0, WebArena) show near-zero format errors with remaining failures semantic, while format errors dominate on Gorilla (100%) and InterCode (70%). These findings redirect practitioners toward prompt engineering and example curation rather than complex adaptation architectures.
>
---
#### [new 054] Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework
- **分类: cs.CL**

- **简介: 该论文属于跨语言推理任务，旨在解决XCoT方法在多语言中计算成本高、效率低的问题。通过选择少量语言、减少冗余token并优化路径，提升推理效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.20090](https://arxiv.org/pdf/2604.20090)**

> **作者:** Chenyuan Zhang; Qiguang Chen; Xie Chen; Zhuotao Tian; Bowen Xing; Meishan Zhang; Libo Qin; Baotian Hu; Min Zhang
>
> **备注:** Accepted by ACL2026 Main
>
> **摘要:** Cross-lingual chain-of-thought (XCoT) with self-consistency markedly enhances multilingual reasoning, yet existing methods remain costly due to extensive sampling of full trajectories across languages. Moreover, multilingual LLM representations vary strongly by language, hindering direct feature comparisons and effective pruning. Motivated by this, we introduce UL-XCoT, the first efficient unified logic cross-lingual reasoning framework that minimizes redundancy in token usage and latency, yielding the greatest efficiency under limited sampling budgets during inference. Specifically, UL-XCoT (1) achieves less languages by selecting, per query, a small candidate language set in a language-invariant unified logic space, (2) enables less tokens by monitoring logic-space trajectory dynamics during decoding to prune low-quality reasoning paths, and (3) aggregates the remaining high-quality trajectories via voting. Experiments on PolyMath across 18 languages and MMLU-ProX-Lite across 29 languages with DeepSeek-R1-DistillQwen-7B demonstrate that UL-XCoT achieves competitive accuracy while sharply cutting over 50% decoding token cost versus prior sampling baselines. UL-XCoT also delivers more stable gains on low-resource languages, underscoring consistently superior robustness where standard XCoT self-consistency method fails.
>
---
#### [new 055] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG
- **分类: cs.CL**

- **简介: 该论文属于多语言信息检索任务，旨在解决mRAG系统中的语言偏见问题。通过分析发现系统偏好英语和母语，提出LAURA模型以提升多语言证据排序效果。**

- **链接: [https://arxiv.org/pdf/2604.20199](https://arxiv.org/pdf/2604.20199)**

> **作者:** Dan Wang; Guozhao Mo; Yafei Shi; Cheng Zhang; Bo Zheng; Boxi Cao; Xuanang Chen; Yaojie Lu; Hongyu Lin; Ben He; Xianpei Han; Le Sun
>
> **备注:** ACL 2026 main conference
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) leverages cross-lingual evidence to ground Large Language Models (LLMs) in global knowledge. However, we show that current mRAG systems suffer from a language bias during reranking, systematically favoring English and the query's native language. By introducing an estimated oracle evidence analysis, we quantify a substantial performance gap between existing rerankers and the achievable upper bound. Further analysis reveals a critical distributional mismatch: while optimal predictions require evidence scattered across multiple languages, current systems systematically suppress such ``answer-critical'' documents, thereby limiting downstream generation performance. To bridge this gap, we propose \textit{\textbf{L}anguage-\textbf{A}gnostic \textbf{U}tility-driven \textbf{R}eranker \textbf{A}lignment (LAURA)}, which aligns multilingual evidence ranking with downstream generative utility. Experiments across diverse languages and generation models show that LAURA effectively mitigates language bias and consistently improves mRAG performance.
>
---
#### [new 056] Phase 1 Implementation of LLM-generated Discharge Summaries showing high Adoption in a Dutch Academic Hospital
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文档自动化任务，旨在解决 discharge summaries 耗时问题。通过集成LLM生成摘要，评估其在临床中的应用效果。**

- **链接: [https://arxiv.org/pdf/2604.19774](https://arxiv.org/pdf/2604.19774)**

> **作者:** Nettuno Nadalini; Tarannom Mehri; Anne H Hoekman; Katerina Kagialari; Job N Doornberg; Tom P van der Laan; Jacobien H F Oosterhoff; Rosanne C Schoonbeek; Charlotte M H H T Bootsma-Robroeks
>
> **备注:** The methods section is located after the discussion in this manuscript
>
> **摘要:** Writing discharge summaries to transfer medical information is an important but time-consuming process that can be assisted by Large Language Models (LLMs). This prospective mixed methods pilot study evaluated an Electronic Health Record (EHR)-integrated LLM to generate discharge summaries drafts. In total, 379 discharge summaries were generated in clinical practice by 21 residents and 4 physician assistants during 9 weeks in our academic hospital. LLM-generated text was copied in 58.5% of admissions, and identifiable LLM content could be traced to 29.1% of final discharge letters. Notably, 86.9% of users self-reported a reduction in documentation time, and 60.9% a reduction in administrative workload. Intent to use after the pilot phase was high (91.3%), supporting further implementation of this use-case. Accurately measuring the documentation time of users on discharge summaries remains challenging, but will be necessary for future extrinsic evaluation of LLM-assisted documentation.
>
---
#### [new 057] Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving
- **分类: cs.CL**

- **简介: 该论文属于优化问题求解任务，旨在解决结构歧义问题。提出DCM-Agent，通过历史解决方案构建记忆库，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.20183](https://arxiv.org/pdf/2604.20183)**

> **作者:** Xinyu Zhang; Yuchen Wan; Boxuan Zhang; Zesheng Yang; Lingling Zhang; Bifan Wei; Jun Liu
>
> **摘要:** Large Language Models (LLMs) often struggle with structural ambiguity in optimization problems, where a single problem admits multiple related but conflicting modeling paradigms, hindering effective solution generation. To address this, we propose Dual-Cluster Memory Agent (DCM-Agent) to enhance performance by leveraging historical solutions in a training-free manner. Central to this is Dual-Cluster Memory Construction. This agent assigns historical solutions to modeling and coding clusters, then distills each cluster's content into three structured types: Approach, Checklist, and Pitfall. This process derives generalizable guidance knowledge. Furthermore, this agent introduces Memory-augmented Inference to dynamically navigate solution paths, detect and repair errors, and adaptively switch reasoning paths with structured knowledge. The experiments across seven optimization benchmarks demonstrate that DCM-Agent achieves an average performance improvement of 11%- 21%. Notably, our analysis reveals a ``knowledge inheritance'' phenomenon: memory constructed by larger models can guide smaller models toward superior performance, highlighting the framework's scalability and efficiency.
>
---
#### [new 058] Duluth at SemEval-2026 Task 6: DeBERTa with LLM-Augmented Data for Unmasking Political Question Evasions
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2026 Task 6任务，解决政治问答回避识别问题。采用DeBERTa模型结合数据增强，提升少数类召回率。**

- **链接: [https://arxiv.org/pdf/2604.20168](https://arxiv.org/pdf/2604.20168)**

> **作者:** Shujauddin Syed; Ted Pedersen
>
> **摘要:** This paper presents the Duluth approach to SemEval-2026 Task 6 on CLARITY: Unmasking Political Question Evasions. We address Task 1 (clarity-level classification) and Task 2 (evasion-level classification), both of which involve classifying question--answer pairs from U.S.\ presidential interviews using a two-level taxonomy of response clarity. Our system is based on DeBERTa-V3-base, extended with focal loss, layer-wise learning rate decay, and boolean discourse features. To address class imbalance in the training data, we augment minority classes using synthetic examples generated by Gemini 3 and Claude Sonnet 4.5. Our best configuration achieved a Macro F1 of 0.76 on the Task 1 evaluation set, placing 8th out of 40 teams. The top-ranked system (TeleAI) achieved 0.89, while the mean score across participants was 0.70. Error analysis reveals that the dominant source of misclassification is confusion between Ambivalent and Clear Reply responses, a pattern that mirrors disagreements among human annotators. Our findings demonstrate that LLM-based data augmentation can meaningfully improve minority-class recall on nuanced political discourse tasks.
>
---
#### [new 059] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于迁移学习任务，旨在解决低资源和类别不平衡场景下的样本选择问题。提出RADS方法，利用强化学习提升样本选择效果，增强模型迁移能力。**

- **链接: [https://arxiv.org/pdf/2604.20256](https://arxiv.org/pdf/2604.20256)**

> **作者:** Wei Han; David Martinez; Anna Khanina; Lawrence Cavedon; Karin Verspoor
>
> **备注:** Accepted at ACL 2026 Findings
>
> **摘要:** A common strategy in transfer learning is few shot fine-tuning, but its success is highly dependent on the quality of samples selected as training examples. Active learning methods such as uncertainty sampling and diversity sampling can select useful samples. However, under extremely low-resource and class-imbalanced conditions, they often favor outliers rather than truly informative samples, resulting in degraded performance. In this paper, we introduce RADS (Reinforcement Adaptive Domain Sampling), a robust sample selection strategy using reinforcement learning (RL) to identify the most informative samples. Experimental evaluations on several real world clinical datasets show our sample selection strategy enhances model transferability while maintaining robust performance under extreme class imbalance compared to traditional methods.
>
---
#### [new 060] Aligning Human-AI-Interaction Trust for Mental Health Support: Survey and Position for Multi-Stakeholders
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于心理健康AI领域，旨在解决跨利益相关者信任不一致的问题。提出三层信任框架，整合多方视角，分析现有研究与评估方法，指出NLP与实际需求的差距，并提出研究方向。**

- **链接: [https://arxiv.org/pdf/2604.20166](https://arxiv.org/pdf/2604.20166)**

> **作者:** Xin Sun; Yue Su; Yifan Mo; Qingyu Meng; Yuxuan Li; Saku Sugawara; Mengyuan Zhang; Charlotte Gerritsen; Sander L. Koole; Koen Hindriks; Jiahuan Pei
>
> **摘要:** Building trustworthy AI systems for mental health support is a shared priority across stakeholders from multiple disciplines. However, "trustworthy" remains loosely defined and inconsistently operationalized. AI research often focuses on technical criteria (e.g., robustness, explainability, and safety), while therapeutic practitioners emphasize therapeutic fidelity (e.g., appropriateness, empathy, and long-term user outcomes). To bridge the fragmented landscape, we propose a three-layer trust framework, covering human-oriented, AI-oriented, and interaction-oriented trust, integrating the viewpoints of key stakeholders (e.g., practitioners, researchers, regulators). Using this framework, we systematically review existing AI-driven research in mental health domain and examine evaluation practices for ``trustworthy'' ranging from automatic metrics to clinically validated approaches. We highlight critical gaps between what NLP currently measures and what real-world mental health contexts require, and outline a research agenda for building socio-technically aligned and genuinely trustworthy AI for mental health support.
>
---
#### [new 061] Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains
- **分类: cs.CL**

- **简介: 该论文属于逻辑推理任务，旨在解决LLMs在多步推理中的结构脆弱性问题。通过干预逻辑连接词选择，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.20564](https://arxiv.org/pdf/2604.20564)**

> **作者:** Seunghyun Park; Yuanyuan Lei
>
> **摘要:** While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logical deduction, where a single transition error can propagate through the entire reasoning chain, leading to unstable performance. In this work, we identify logical connectives as primary points of this structural fragility. Through empirical analysis, we show that connective tokens function as high entropy forking points, at which models frequently struggle to determine the correct logical direction. Motivated by this observation, we hypothesize that intervening in logical connective selection can guide LLMs toward more correct logical direction, thereby improving the overall reasoning chain. To validate this hypothesis, we propose a multi-layered framework that intervenes specifically at these logic-critical junctions in the reasoning process. Our framework includes (1) Gradient-based Logical Steering to guide LLMs internal representations towards valid reasoning subspaces, (2) Localized Branching to resolve ambiguity via targeted look-ahead search, and (3) Targeted Transition Preference Optimization, a surgical reinforcement learning objective that selectively optimizes single-token preferences at logical pivots. Crucially, by concentrating intervention solely on logic-critical transitions, our framework achieves a favorable accuracy--efficiency trade-off compared to global inference time scaling methods like beam search and self-consistency.
>
---
#### [new 062] ORPHEAS: A Cross-Lingual Greek-English Embedding Model for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言信息检索任务，旨在解决希腊语与英语之间语义对齐问题。提出ORPHEAS模型，通过领域特化微调提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.20666](https://arxiv.org/pdf/2604.20666)**

> **作者:** Ioannis E. Livieris; Athanasios Koursaris; Alexandra Apostolopoulou; Konstantinos Kanaris Dimitris Tsakalidis; George Domalis
>
> **备注:** This paper has been accepted for presentation at Engineering Applications and Advances of Artificial Intelligence 2026 (EAAAI'26)
>
> **摘要:** Effective retrieval-augmented generation across bilingual Greek--English applications requires embedding models capable of capturing both domain-specific semantic relationships and cross-lingual semantic alignment. Existing multilingual embedding models distribute their representational capacity across numerous languages, limiting their optimization for Greek and failing to encode the morphological complexity and domain-specific terminological structures inherent in Greek text. In this work, we propose ORPHEAS, a specialized Greek--English embedding model for bilingual retrieval-augmented generation. ORPHEAS is trained with a high quality dataset generated by a knowledge graph-based fine-tuning methodology which is applied to a diverse multi-domain corpus, which enables language-agnostic semantic representations. The numerical experiments across monolingual and cross-lingual retrieval benchmarks reveal that ORPHEAS outperforms state-of-the-art multilingual embedding models, demonstrating that domain-specialized fine-tuning on morphologically complex languages does not compromise cross-lingual retrieval capability.
>
---
#### [new 063] LLM Agents Predict Social Media Reactions but Do Not Outperform Text Classifiers: Benchmarking Simulation Accuracy Using 120K+ Personas of 1511 Humans
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI行为预测任务，旨在检验LLM代理是否能准确预测用户社交媒体反应。通过12万+代理测试，发现其表现不如传统文本分类器，但具备零样本部署优势。**

- **链接: [https://arxiv.org/pdf/2604.19787](https://arxiv.org/pdf/2604.19787)**

> **作者:** Ljubisa Bojic; Alexander Felfernig; Bojana Dinic; Velibor Ilic; Achim Rettinger; Vera Mevorah; Damian Trilling
>
> **摘要:** Social media platforms mediate how billions form opinions and engage with public discourse. As autonomous AI agents increasingly participate in these spaces, understanding their behavioral fidelity becomes critical for platform governance and democratic resilience. Previous work demonstrates that LLM-powered agents can replicate aggregate survey responses, yet few studies test whether agents can predict specific individuals' reactions to specific content. This study benchmarks LLM-based agents' accuracy in predicting human social media reactions (like, dislike, comment, share, no reaction) across 120,000+ unique agent-persona combinations derived from 1,511 Serbian participants and 27 large language models. In Study 1, agents achieved 70.7% overall accuracy, with LLM choice producing a 13 percentage-point performance spread. Study 2 employed binary forced-choice (like/dislike) evaluation with chance-corrected metrics. Agents achieved Matthews Correlation Coefficient (MCC) of 0.29, indicating genuine predictive signal beyond chance. However, conventional text-based supervised classifiers using TF-IDF representations outperformed LLM agents (MCC of 0.36), suggesting predictive gains reflect semantic access rather than uniquely agentic reasoning. The genuine predictive validity of zero-shot persona-prompted agents warns against potential manipulation through easily deploying swarms of behaviorally distinct AI agents on social media, while simultaneously offering opportunities to use such agents in simulations for predicting polarization dynamics and informing AI policy. The advantage of using zero-shot agents is that they require no task-specific training, making their large-scale deployment easy across diverse contexts. Limitations include single-country sampling. Future research should explore multilingual testing and fine-tuning approaches.
>
---
#### [new 064] Parallel-SFT: Improving Zero-Shot Cross-Programming-Language Transfer for Code RL
- **分类: cs.CL**

- **简介: 该论文属于代码强化学习任务，旨在解决低资源编程语言的零样本跨语言迁移问题。通过引入并行程序进行SFT初始化，提升模型在不同语言间的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.20835](https://arxiv.org/pdf/2604.20835)**

> **作者:** Zhaofeng Wu; Shiqi Wang; Boya Peng; Anuj Goyal; Melanie Kambadur; Sebastian Ruder; Yoon Kim; Chloe Bi
>
> **摘要:** Modern language models demonstrate impressive coding capabilities in common programming languages (PLs), such as C++ and Python, but their performance in lower-resource PLs is often limited by training data availability. In principle, however, most programming skills are universal across PLs, so the capability acquired in one PL should transfer to others. In this work, we propose the task of zero-shot cross-programming-language transfer for code RL. We find that, for Llama-3.1, RL training for code generation in a source PL fails to improve, and sometimes even degrades, the performance on other target PLs. To address this, we hypothesize that effective RL transfer requires a generalizable SFT initialization before RL. We thus propose **Parallel-SFT**, an SFT strategy that incorporates "parallel programs" -- functionally equivalent code implemented in multiple PLs -- into the data mixture. We demonstrate that this improves transferability: when we subsequently perform RL on our Parallel-SFT model, we observe better generalization to unseen PLs. Analysis of the model internal representations reveals that Parallel-SFT leads to a more functionality-centric latent space, where equivalent programs across PLs are more tightly clustered, which we hypothesize to contribute to the improved transferability.
>
---
#### [new 065] Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents
- **分类: cs.CL**

- **简介: 该论文属于终身学习任务，旨在解决智能体在交互中无法主动检索过往经验的问题。提出ProactAgent框架，通过主动检索提升决策效果并减少冗余。**

- **链接: [https://arxiv.org/pdf/2604.20572](https://arxiv.org/pdf/2604.20572)**

> **作者:** Yuxuan Cai; Jie Zhou; Qin Chen; Liang He
>
> **摘要:** Online lifelong learning enables agents to accumulate experience across interactions and continually improve on long-horizon tasks. However, existing methods typically treat retrieval from past experience as a passive operation, triggering it only at task initialization or after completing a step. Consequently, agents often fail to identify knowledge gaps during interaction and proactively retrieve the most useful experience for the current decision. To address this limitation, we present ProactAgent, an experience-driven lifelong learning framework for proactive retrieval over a structured experience base. We first introduce Experience-Enhanced Online Evolution (ExpOnEvo), which enables continual improvement through both policy updates and memory refinement. The experience base organizes historical interactions into typed repositories, including factual memory, episodic memory, and behavioral skills, so that retrieval can provide both relevant evidence and actionable guidance. On top of this, we propose Proactive Reinforcement Learning-based Retrieval (ProactRL), which models retrieval as an explicit policy action and learns when and what to retrieve via paired-branch process rewards. By comparing continuations from identical interaction prefixes with and without retrieval, ProactRL provides step-level supervision for retrieval decisions, encouraging retrieval only when it leads to better task outcomes or higher efficiency. Experiments on SciWorld, AlfWorld, and StuLife show that ProactAgent consistently improves lifelong agent performance, achieving success rates of 73.50\% on SciWorld and 71.28\% on AlfWorld while substantially reducing retrieval overhead, and attains performance competitive with proprietary models on StuLife.
>
---
#### [new 066] RespondeoQA: a Benchmark for Bilingual Latin-English Question Answering
- **分类: cs.CL**

- **简介: 该论文提出RespondeoQA，一个用于双语拉丁语-英语问答的基准数据集，解决跨语言问答与翻译任务，涵盖多种题型，评估大模型表现。**

- **链接: [https://arxiv.org/pdf/2604.20738](https://arxiv.org/pdf/2604.20738)**

> **作者:** Marisa Hudspeth; Patrick J. Burns; Brendan O'Connor
>
> **备注:** Published in LREC 2026
>
> **摘要:** We introduce a benchmark dataset for question answering and translation in bilingual Latin and English settings, containing about 7,800 question-answer pairs. The questions are drawn from Latin pedagogical sources, including exams, quizbowl-style trivia, and textbooks ranging from the 1800s to the present. After automated extraction, cleaning, and manual review, the dataset covers a diverse range of question types: knowledge- and skill-based, multihop reasoning, constrained translation, and mixed language pairs. To our knowledge, this is the first QA benchmark centered on Latin. As a case study, we evaluate three large language models -- LLaMa 3, Qwen QwQ, and OpenAI's o3-mini -- finding that all perform worse on skill-oriented questions. Although the reasoning models perform better on scansion and literary-device tasks, they offer limited improvement overall. QwQ performs slightly better on questions asked in Latin, but LLaMa3 and o3-mini are more task dependent. This dataset provides a new resource for assessing model capabilities in a specialized linguistic and cultural domain, and the creation process can be easily adapted for other languages. The dataset is available at: this https URL
>
---
#### [new 067] AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于电商领域的多模态表示学习任务，旨在解决细粒度语义理解不足的问题。通过属性增强的两阶段训练框架提升产品检索性能。**

- **链接: [https://arxiv.org/pdf/2604.20135](https://arxiv.org/pdf/2604.20135)**

> **作者:** Biao Zhang; Lixin Chen; Bin Zhang; Zongwei Wang; Tong Liu; Bo Zheng
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Multimodal representation is crucial for E-commerce tasks such as identical product retrieval. Large representation models (e.g., VLM2Vec) demonstrate strong multimodal understanding capabilities, yet they struggle with fine-grained semantic comprehension, which is essential for distinguishing highly similar items. To address this, we propose Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning (AFMRL), which defines product fine-grained understanding as an attribute generation task. It leverages the generative power of Multimodal Large Language Models (MLLMs) to extract key attributes from product images and text, and enhances representation learning through a two-stage training framework: 1) Attribute-Guided Contrastive Learning (AGCL), where the key attributes generated by the MLLM are used in the image-text contrastive learning training process to identify hard samples and filter out noisy false negatives. 2) Retrieval-aware Attribute Reinforcement (RAR), where the improved retrieval performance of the representation model post-attribute integration serves as a reward signal to enhance MLLM's attribute generation during multimodal fine-tuning. Extensive experiments on large-scale E-commerce datasets demonstrate that our method achieves state-of-the-art performance on multiple downstream retrieval tasks, validating the effectiveness of harnessing generative models to advance fine-grained representation learning.
>
---
#### [new 068] Tracing Relational Knowledge Recall in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在文本生成中如何回忆关系知识，旨在解决关系分类任务中的表示选择问题。通过分析注意力头和MLP的贡献，发现每头注意力对残差流的贡献是有效的线性分类特征。**

- **链接: [https://arxiv.org/pdf/2604.19934](https://arxiv.org/pdf/2604.19934)**

> **作者:** Nicholas Popovič; Michael Färber
>
> **备注:** ACL 2026 (findings)
>
> **摘要:** We study how large language models recall relational knowledge during text generation, with a focus on identifying latent representations suitable for relation classification via linear probes. Prior work shows how attention heads and MLPs interact to resolve subject, predicate, and object, but it remains unclear which representations support faithful linear relation classification and why some relation types are easier to capture linearly than others. We systematically evaluate different latent representations derived from attention head and MLP contributions, showing that per-head attention contributions to the residual stream are comparatively strong features for linear relation classification. Feature attribution analyses of the trained probes, as well as characteristics of the different relation types, reveal clear correlations between probe accuracy and relation specificity, entity connectedness, and how distributed the signal on which the probe relies is across attention heads. Finally, we show how token-level feature attribution of probe predictions can be used to reveal probe behavior in further detail.
>
---
#### [new 069] Evidence of Layered Positional and Directional Constraints in the Voynich Manuscript: Implications for Cipher-Like Structure
- **分类: cs.CL**

- **简介: 该论文属于密码分析任务，旨在解析维特鲁维亚手稿的结构。通过分析字符序列，发现其具有独特的方向性约束，表明其可能具有类似密码的结构特征。**

- **链接: [https://arxiv.org/pdf/2604.19762](https://arxiv.org/pdf/2604.19762)**

> **作者:** Christophe Parisel
>
> **摘要:** The Voynich Manuscript (VMS) exhibits a script of uncertain origin whose grapheme sequences have resisted linguistic analysis. We present a systematic analysis of its grapheme sequences, revealing two complementary structural layers: a character-level right-to-left optimization in word-internal sequences and a left-to-right dependency at word boundaries, a directional dissociation not observed in any of our four comparison languages (English, French, Hebrew, Arabic). We further evaluate two classes of structured generator against a four-signature joint criterion: a parametric slot-based generator and a Cardan grille implementing Rugg's (2004) gibberish hypothesis. Across their full tested parameter spaces, neither class reproduces all four signatures simultaneously. While these results do not rule out generator classes we have not tested, they provide the first quantitative benchmarks against which any future generative or cryptanalytic model of the VMS can be evaluated, and they suggest that the VMS exhibits cipher-like structural constraints that are difficult to reproduce from simple positional or frequency-based mechanisms alone.
>
---
#### [new 070] Towards High-Quality Machine Translation for Kokborok: A Low-Resource Tibeto-Burman Language of Northeast India
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决Kokborok语言资源匮乏的问题。通过多源平行语料训练模型，提升其翻译质量。**

- **链接: [https://arxiv.org/pdf/2604.19778](https://arxiv.org/pdf/2604.19778)**

> **作者:** Badal Nyalang; Biman Debbarma
>
> **摘要:** We present KokborokMT, a high-quality neural machine translation (NMT) system for Kokborok (ISO 639-3), a Tibeto-Burman language spoken primarily in Tripura, India with approximately 1.5 million speakers. Despite its status as an official language of Tripura, Kokborok has remained severely under-resourced in the NLP community, with prior machine translation attempts limited to systems trained on small Bible-derived corpora achieving BLEU scores below 7. We fine-tune the NLLB-200-distilled-600M model on a multi-source parallel corpus comprising 36,052 sentence pairs: 9,284 professionally translated sentences from the SMOL dataset, 1,769 Bible-domain sentences from WMT shared task data, and 24,999 synthetic back-translated pairs generated via Gemini Flash from Tatoeba English source sentences. We introduce as a new language token for Kokborok in the NLLB framework. Our best system achieves BLEU scores of 17.30 and 38.56 on held-out test sets, representing substantial improvements over prior published results. Human evaluation by three annotators yields mean adequacy of 3.74/5 and fluency of 3.70/5, with substantial agreement between trained evaluators.
>
---
#### [new 071] CoAuthorAI: A Human in the Loop System For Scientific Book Writing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoAuthorAI系统，解决科学书籍写作中LLM结构不一致和引用不可靠的问题。通过人机协作，提升书籍写作的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2604.19772](https://arxiv.org/pdf/2604.19772)**

> **作者:** Yangjie Tian; Xungang Gu; Yun Zhao; Jiale Yang; Lin Yang; Ning Li; He Zhang; Ruohua Xu; Hua Wang; Kewen Liao; Ming Liu
>
> **摘要:** Large language models (LLMs) are increasingly used in scientific writing but struggle with book-length tasks, often producing inconsistent structure and unreliable citations. We introduce CoAuthorAI, a human-in-the-loop writing system that combines retrieval-augmented generation, expert-designed hierarchical outlines, and automatic reference linking. The system allows experts to iteratively refine text at the sentence level, ensuring coherence and accuracy. In evaluations of 500 multi-domain literature review chapters, CoAuthorAI achieved a maximum soft-heading recall of 98%; in a human evaluation of 100 articles, the generated content reached a satisfaction rate of 82%. The book AI for Rock Dynamics generated with CoAuthorAI and Kexin Technology's LUFFA AI model has been published with Springer Nature. These results show that systematic human-AI collaboration can extend LLMs' capabilities from articles to full-length books, enabling faster and more reliable scientific publishing.
>
---
#### [new 072] Surrogate modeling for interpreting black-box LLMs in medical predictions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型解释任务，旨在解决LLMs在医疗预测中缺乏可解释性的问题。通过构建代理模型，量化分析LLM编码的知识，揭示其对输入变量的感知及潜在偏差。**

- **链接: [https://arxiv.org/pdf/2604.20331](https://arxiv.org/pdf/2604.20331)**

> **作者:** Changho Han; Songsoo Kim; Dong Won Kim; Leo Anthony Celi; Jaewoong Kim; SungA Bae; Dukyong Yoon
>
> **摘要:** Large language models (LLMs), trained on vast datasets, encode extensive real-world knowledge within their parameters, yet their black-box nature obscures the mechanisms and extent of this encoding. Surrogate modeling, which uses simplified models to approximate complex systems, can offer a path toward better interpretability of black-box models. We propose a surrogate modeling framework that quantitatively explains LLM-encoded knowledge. For a specific hypothesis derived from domain knowledge, this framework approximates the latent LLM knowledge space using observable elements (input-output pairs) through extensive prompting across a comprehensive range of simulated scenarios. Through proof-of-concept experiments in medical predictions, we demonstrate our framework's effectiveness in revealing the extent to which LLMs "perceive" each input variable in relation to the output. Particularly, given concerns that LLMs may perpetuate inaccuracies and societal biases embedded in their training data, our experiments using this framework quantitatively revealed both associations that contradict established medical knowledge and the persistence of scientifically refuted racial assumptions within LLM-encoded knowledge. By disclosing these issues, our framework can act as a red-flag indicator to support the safe and reliable application of these models.
>
---
#### [new 073] Structured Disagreement in Health-Literacy Annotation: Epistemic Stability, Conceptual Difficulty, and Agreement-Stratified Inference
- **分类: cs.CL**

- **简介: 该论文研究健康素养标注中的分歧问题，分析了新冠回答的分级标注数据，揭示任务本身导致的结构化分歧，强调需采用视角主义模型以准确推断。**

- **链接: [https://arxiv.org/pdf/2604.19943](https://arxiv.org/pdf/2604.19943)**

> **作者:** Olga Kellert; Sriya Kondury; Candice Koo; Nemika Tyagi; Steffen Eikenberry
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Annotation pipelines in Natural Language Processing (NLP) commonly assume a single latent ground truth per instance and resolve disagreement through label aggregation. Perspectivist approaches challenge this view by treating disagreement as potentially informative rather than erroneous. We present a large-scale analysis of graded health-literacy annotations from 6,323 open-ended COVID-19 responses collected in Ecuador and Peru. Each response was independently labeled by multiple annotators using proportional correctness scores, reflecting the degree to which responses align with normative public-health guidelines, allowing us to analyze the full distribution of judgments rather than aggregated labels. Variance decomposition shows that question-level conceptual difficulty accounts for substantially more variance than annotator identity, indicating that disagreement is structured by the task itself rather than driven by individual raters. Agreement-stratified analyses further reveal that key social-scientific effects, including country, education, and urban-rural differences, vary in magnitude and in some cases reverse direction across levels of inter-annotator agreement. These findings suggest that graded health-literacy evaluation contains both epistemically stable and unstable components, and that aggregating across them can obscure important inferential differences. We therefore argue that strong perspectivist modeling is not only conceptually justified but statistically necessary for valid inference in graded interpretive tasks.
>
---
#### [new 074] Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统研究，旨在解决LLM团队协作性能预测问题。通过行为经济学游戏评估模型合作性，发现其能有效预测科学任务表现。**

- **链接: [https://arxiv.org/pdf/2604.20658](https://arxiv.org/pdf/2604.20658)**

> **作者:** Shivani Kumar; Adarsh Bharathwaj; David Jurgens
>
> **摘要:** Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 35 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate games and invest in multiplicative team production (rather than greedy strategies) produce better scientific reports across three outcomes, accuracy, quality, and completion. These associations hold after controlling for multiple factors, indicating that cooperative disposition is a distinct, measurable property of LLMs not reducible to general ability. Our behavioral games framework thus offers a fast and inexpensive diagnostic for screening cooperative fitness before costly multi-agent deployment.
>
---
#### [new 075] CHASM: Unveiling Covert Advertisements on Chinese Social Media
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文属于社会媒体内容检测任务，旨在解决 covert advertisements 的识别问题。作者构建了 CHASM 数据集，并评估 MLLMs 的检测能力，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2604.20511](https://arxiv.org/pdf/2604.20511)**

> **作者:** Jingyi Zheng; Tianyi Hu; Yule Liu; Zhen Sun; Zongmin Zhang; Zifan Peng; Wenhan Dong; Xinlei He
>
> **备注:** NeuIPS 2025 (Datasets and Benchmarks Track)
>
> **摘要:** Current benchmarks for evaluating large language models (LLMs) in social media moderation completely overlook a serious threat: covert advertisements, which disguise themselves as regular posts to deceive and mislead consumers into making purchases, leading to significant ethical and legal concerns. In this paper, we present the CHASM, a first-of-its-kind dataset designed to evaluate the capability of Multimodal Large Language Models (MLLMs) in detecting covert advertisements on social media. CHASM is a high-quality, anonymized, manually curated dataset consisting of 4,992 instances, based on real-world scenarios from the Chinese social media platform Rednote. The dataset was collected and annotated under strict privacy protection and quality control protocols. It includes many product experience sharing posts that closely resemble covert advertisements, making the dataset particularly this http URL results show that under both zero-shot and in-context learning settings, none of the current MLLMs are sufficiently reliable for detecting covert this http URL further experiments revealed that fine-tuning open-source MLLMs on our dataset yielded noticeable performance gains. However, significant challenges persist, such as detecting subtle cues in comments and differences in visual and textual this http URL provide in-depth error analysis and outline future research directions. We hope our study can serve as a call for the research community and platform moderators to develop more precise defenses against this emerging threat.
>
---
#### [new 076] Utterance-Level Methods for Identifying Reliable ASR-Output for Child Speech
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别可靠性评估任务，旨在解决儿童语音ASR输出不可靠的问题。通过提出两种新的话语级选择方法，提高可靠转录的识别精度。**

- **链接: [https://arxiv.org/pdf/2604.19801](https://arxiv.org/pdf/2604.19801)**

> **作者:** Gus Lathouwers; Lingyun Gao; Catia Cucchiarini; Helmer Strik
>
> **备注:** Submitted for Interspeech 2026, currently under review
>
> **摘要:** Automatic Speech Recognition (ASR) is increasingly used in applications involving child speech, such as language learning and literacy acquisition. However, the effectiveness of such applications is limited by high ASR error rates. The negative effects can be mitigated by identifying in advance which ASR-outputs are reliable. This work aims to develop two novel approaches for selecting reliable ASR-output at the utterance level, one for selecting reliable read speech and one for dialogue speech material. Evaluations were done on an English and a Dutch dataset, each with a baseline and finetuned model. The results show that utterance-level selection methods for identifying reliably transcribed speech recordings have high precision for the best strategy (P > 97.4) for both read speech and dialogue material, for both languages. Using the current optimal strategy allows 21.0% to 55.9% of dialogue/read speech datasets to be automatically selected with low (UER of < 2.6) error rates.
>
---
#### [new 077] Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型解释任务，旨在探究大语言模型的不确定性与正确性是否由相同特征驱动。通过稀疏自编码器分析，发现三类不同功能的特征，证明二者为独立现象。**

- **链接: [https://arxiv.org/pdf/2604.19974](https://arxiv.org/pdf/2604.19974)**

> **作者:** Het Patel; Tiejin Chen; Hua Wei; Evangelos E. Papalexakis; Jia Chen
>
> **摘要:** Large language models can be uncertain yet correct, or confident yet wrong, raising the question of whether their output-level uncertainty and their actual correctness are driven by the same internal mechanisms or by distinct feature populations. We introduce a 2x2 framework that partitions model predictions along correctness and confidence axes, and uses sparse autoencoders to identify features associated with each dimension independently. Applying this to Llama-3.1-8B and Gemma-2-9B, we identify three feature populations that play fundamentally different functional roles. Pure uncertainty features are functionally essential: suppressing them severely degrades accuracy. Pure incorrectness features are functionally inert: despite showing statistically significant activation differences between correct and incorrect predictions, the majority produce near-zero change in accuracy when suppressed. Confounded features that encode both signals are detrimental to output quality, and targeted suppression of them yields a 1.1% accuracy improvement and a 75% entropy reduction, with effects transferring across the ARC-Challenge and RACE benchmarks. The feature categories are also informationally distinct: the activations of just 3 confounded features from a single mid-network layer predict model correctness (AUROC ~0.79), enabling selective abstention that raises accuracy from 62% to 81% at 53% coverage. The results demonstrate that uncertainty and correctness are distinct internal phenomena, with implications for interpretability and targeted inference-time intervention.
>
---
#### [new 078] DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于边缘计算领域的深度研究代理任务，旨在用少量开放数据训练高效小模型。通过两阶段训练提升代理性能，实现优于大模型的效果。**

- **链接: [https://arxiv.org/pdf/2604.19859](https://arxiv.org/pdf/2604.19859)**

> **作者:** Venus Team; Sunhao Dai; Yong Deng; Jinzhen Lin; Yusheng Song; Guoqing Wang; Xiaofeng Wu; Yuqi Zhou; Shuo Yang; Zhenzhe Ying; Zhanwei Zhang; Changhua Meng; Weiqiang Wang
>
> **备注:** Technical Report of DR-Venus
>
> **摘要:** Edge-scale deep research agents based on small language models are attractive for real-world deployment due to their advantages in cost, latency, and privacy. In this work, we study how to train a strong small deep research agent under limited open-data by improving both data quality and data utilization. We present DR-Venus, a frontier 4B deep research agent for edge-scale deployment, built entirely on open data. Our training recipe consists of two stages. In the first stage, we use agentic supervised fine-tuning (SFT) to establish basic agentic capability, combining strict data cleaning with resampling of long-horizon trajectories to improve data quality and utilization. In the second stage, we apply agentic reinforcement learning (RL) to further improve execution reliability on long-horizon deep research tasks. To make RL effective for small agents in this setting, we build on IGPO and design turn-level rewards based on information gain and format-aware regularization, thereby enhancing supervision density and turn-level credit assignment. Built entirely on roughly 10K open-data, DR-Venus-4B significantly outperforms prior agentic models under 9B parameters on multiple deep research benchmarks, while also narrowing the gap to much larger 30B-class systems. Our further analysis shows that 4B agents already possess surprisingly strong performance potential, highlighting both the deployment promise of small models and the value of test-time scaling in this setting. We release our models, code, and key recipes to support reproducible research on edge-scale deep research agents.
>
---
#### [new 079] Explainable Speech Emotion Recognition: Weighted Attribute Fairness to Model Demographic Contributions to Social Bias
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决模型中的社会偏见问题。通过引入加权属性公平性方法，分析不同人口属性对偏见的贡献，评估并揭示了模型中的性别偏见。**

- **链接: [https://arxiv.org/pdf/2604.19763](https://arxiv.org/pdf/2604.19763)**

> **作者:** Tomisin Ogunnubi; Yupei Li; Björn Schuller
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Speech Emotion Recognition (SER) systems have growing applications in sensitive domains such as mental health and education, where biased predictions can cause harm. Traditional fairness metrics, such as Equalised Odds and Demographic Parity, often overlook the joint dependency between demographic attributes and model predictions. We propose a fairness modelling approach for SER that explicitly captures allocative bias by learning the joint relationship between demographic attributes and model error. We validate our fairness metric on synthetic data, then apply it to evaluate HuBERT and WavLM models finetuned on the CREMA-D dataset. Our results indicate that the proposed fairness model captures more mutual information between protected attributes and biases and quantifies the absolute contribution of individual attributes to bias in SSL-based SER models. Additionally, our analysis reveals indications of gender bias in both HuBERT and WavLM.
>
---
#### [new 080] ThermoQA: A Three-Tier Benchmark for Evaluating Thermodynamic Reasoning in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ThermoQA，一个用于评估大语言模型热力学推理能力的三层次基准。解决模型在热力学问题上的推理能力不足问题，通过设计不同难度的问题并测试多个模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.19758](https://arxiv.org/pdf/2604.19758)**

> **作者:** Kemal Düzkar
>
> **备注:** 17 pages, 8 figures, open-source dataset and code
>
> **摘要:** We present ThermoQA, a benchmark of 293 open-ended engineering thermodynamics problems in three tiers: property lookups (110 Q), component analysis (101 Q), and full cycle analysis (82 Q). Ground truth is computed programmatically from CoolProp 7.2.0, covering water, R-134a, and variable-cp air. Six frontier LLMs are evaluated across three independent runs each. The composite leaderboard is led by Claude Opus 4.6 (94.1%), GPT-5.4 (93.1%), and Gemini 3.1 Pro (92.5%). Cross-tier degradation ranges from 2.8 pp (Opus) to 32.5 pp (MiniMax), confirming that property memorization does not imply thermodynamic reasoning. Supercritical water, R-134a refrigerant, and combined-cycle gas turbine analysis serve as natural discriminators with 40-60 pp performance spreads. Multi-run sigma ranges from +/-0.1% to +/-2.5%, quantifying reasoning consistency as a distinct evaluation axis. Dataset and code are open-source at this https URL
>
---
#### [new 081] Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文研究LLM代理在多轮Avalon游戏中产生的社会动态与声誉机制，解决重复交互中的信任与欺骗问题，通过实验分析声誉形成和策略性欺骗现象。**

- **链接: [https://arxiv.org/pdf/2604.20582](https://arxiv.org/pdf/2604.20582)**

> **作者:** Suveen Ellawela
>
> **摘要:** We study emergent social dynamics in LLM agents playing The Resistance: Avalon, a hidden-role deception game. Unlike prior work on single-game performance, our agents play repeated games while retaining memory of previous interactions, including who played which roles and how they behaved, enabling us to study how social dynamics evolve. Across 188 games, two key phenomena emerge. First, reputation dynamics emerge organically when agents retain cross-game memory: agents reference past behavior in statements like "I am wary of repeating last game's mistake of over-trusting early success." These reputations are role-conditional: the same agent is described as "straightforward" when playing good but "subtle" when playing evil, and high-reputation players receive 46% more team inclusions. Second, higher reasoning effort supports more strategic deception: evil players more often pass early missions to build trust before sabotaging later ones, 75% in high-effort games vs 36% in low-effort games. Together, these findings show that repeated interaction with memory gives rise to measurable reputation and deception dynamics among LLM agents.
>
---
#### [new 082] SAKE: Self-aware Knowledge Exploitation-Exploration for Grounded Multimodal Named Entity Recognition
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出SAKE框架，解决开放场景下的多模态命名实体识别问题，通过结合内部知识与外部探索提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.20146](https://arxiv.org/pdf/2604.20146)**

> **作者:** Jielong Tang; Xujie Yuan; Jiayang Liu; Jianxing Yu; Xiao Dong; Lin Chen; Yunlai Teng; Shimin Di; Jian Yin
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Grounded Multimodal Named Entity Recognition (GMNER) aims to extract named entities and localize their visual regions within image-text pairs, serving as a pivotal capability for various downstream applications. In open-world social media platforms, GMNER remains challenging due to the prevalence of long-tailed, rapidly evolving, and unseen entities. To tackle this, existing approaches typically rely on either external knowledge exploration through heuristic retrieval or internal knowledge exploitation via iterative refinement in Multimodal Large Language Models (MLLMs). However, heuristic retrieval often introduces noisy or conflicting evidence that degrades precision on known entities, while solely internal exploitation is constrained by the knowledge boundaries of MLLMs and prone to hallucinations. To address this, we propose SAKE, an end-to-end agentic framework that harmonizes internal knowledge exploitation and external knowledge exploration via self-aware reasoning and adaptive search tool invocation. We implement this via a two-stage training paradigm. First, we propose Difficulty-aware Search Tag Generation, which quantifies the model's entity-level uncertainty through multiple forward samplings to produce explicit knowledge-gap signals. Based on these signals, we construct SAKE-SeCoT, a high-quality Chain-of-Thought dataset that equips the model with basic self-awareness and tool-use capabilities through supervised fine-tuning. Second, we employ agentic reinforcement learning with a hybrid reward function that penalizes unnecessary retrieval, enabling the model to evolve from rigid search imitation to genuine self-aware decision-making about when retrieval is truly necessary. Extensive experiments on two widely used social media benchmarks demonstrate SAKE's effectiveness.
>
---
#### [new 083] Transparent Screening for LLM Inference and Training Impacts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决大语言模型推理与训练影响的透明度问题。通过构建框架，将应用描述转化为环境估计，提供可审计的对比分析方法。**

- **链接: [https://arxiv.org/pdf/2604.19757](https://arxiv.org/pdf/2604.19757)**

> **作者:** Arnault Pachot; Thierry Petit
>
> **摘要:** This paper presents a transparent screening framework for estimating inference and training impacts of current large language models under limited observability. The framework converts natural-language application descriptions into bounded environmental estimates and supports a comparative online observatory of current market models. Rather than claiming direct measurement for opaque proprietary services, it provides an auditable, source-linked proxy methodology designed to improve comparability, transparency, and reproducibility.
>
---
#### [new 084] Enhancing ASR Performance in the Medical Domain for Dravidian Languages
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升低资源达罗毗荼语在医疗领域的ASR性能。针对数据不足和形态复杂问题，提出一种结合真实与合成数据的置信度训练框架，有效降低词错误率。**

- **链接: [https://arxiv.org/pdf/2604.19797](https://arxiv.org/pdf/2604.19797)**

> **作者:** Sri Charan Devarakonda; Ravi Sastry Kolluru; Manjula Sri Rayudu; Rashmi Kapoor; Madhu G; Anil Kumar Vuppala
>
> **摘要:** Automatic Speech Recognition (ASR) for low-resource Dravidian languages like Telugu and Kannada faces significant challenges in specialized medical domains due to limited annotated data and morphological complexity. This work proposes a novel confidence-aware training framework that integrates real and synthetic speech data through a hybrid confidence mechanism combining static perceptual and acoustic similarity metrics with dynamic model entropy. Unlike direct fine-tuning approaches, the proposed methodology employs both fixed-weight and learnable-weight confidence aggregation strategies to guide sample weighting during training, enabling effective utilization of heterogeneous data sources. The framework is evaluated on Telugu and Kannada medical datasets containing both real recordings and TTS-generated synthetic speech. A 5-gram KenLM language model is applied for post-decoding correction. Results show that the hybrid confidence-aware approach with learnable weights substantially reduces recognition errors: Telugu Word Error Rate (WER) decreases from 24.3% to 15.8% (8.5% absolute improvement), while Kannada WER drops from 31.7% to 25.4% (6.3% absolute improvement), both significantly outperforming standard fine-tuning baselines. These findings confirm that combining adaptive confidence-aware training with statistical language modeling delivers superior performance for domain-specific ASR in morphologically complex Dravidian languages.
>
---
#### [new 085] AVISE: Framework for Evaluating the Security of AI Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出AVISE框架，用于评估AI系统的安全性。任务是解决AI安全漏洞检测问题，通过构建测试用例和评估模型，发现语言模型的漏洞。**

- **链接: [https://arxiv.org/pdf/2604.20833](https://arxiv.org/pdf/2604.20833)**

> **作者:** Mikko Lempinen; Joni Kemppainen; Niklas Raesalmi
>
> **摘要:** As artificial intelligence (AI) systems are increasingly deployed across critical domains, their security vulnerabilities pose growing risks of high-profile exploits and consequential system failures. Yet systematic approaches to evaluating AI security remain underdeveloped. In this paper, we introduce AVISE (AI Vulnerability Identification and Security Evaluation), a modular open-source framework for identifying vulnerabilities in and evaluating the security of AI systems and models. As a demonstration of the framework, we extend the theory-of-mind-based multi-turn Red Queen attack into an Adversarial Language Model (ALM) augmented attack and develop an automated Security Evaluation Test (SET) for discovering jailbreak vulnerabilities in language models. The SET comprises 25 test cases and an Evaluation Language Model (ELM) that determines whether each test case was able to jailbreak the target model, achieving 92% accuracy, an F1-score of 0.91, and a Matthews correlation coefficient of 0.83. We evaluate nine recently released language models of diverse sizes with the SET and find that all are vulnerable to the augmented Red Queen attack to varying degrees. AVISE provides researchers and industry practitioners with an extensible foundation for developing and deploying automated SETs, offering a concrete step toward more rigorous and reproducible AI security evaluation.
>
---
#### [new 086] HaS: Accelerating RAG through Homology-Aware Speculative Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于RAG任务，旨在解决检索效率低的问题。提出HaS框架，通过同源查询识别减少冗余检索，提升速度并保持较高准确率。**

- **链接: [https://arxiv.org/pdf/2604.20452](https://arxiv.org/pdf/2604.20452)**

> **作者:** Peng Peng; Weiwei Lin; Wentai Wu; Xinyang Wang; Yongheng Liu
>
> **备注:** Accepted by ICDE 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly time-consuming as the knowledge databases grow in size. Existing acceleration strategies either compromise accuracy through approximate retrieval, or achieve marginal gains by reusing results of strictly identical queries. We propose HaS, a homology-aware speculative retrieval framework that performs low-latency speculative retrieval over restricted scopes to obtain candidate documents, followed by validating whether they contain the required knowledge. The validation, grounded in the homology relation between queries, is formulated as a homologous query re-identification task: once a previously observed query is identified as a homologous re-encounter of the incoming query, the draft is deemed acceptable, allowing the system to bypass slow full-database retrieval. Benefiting from the prevalence of homologous queries under real-world popularity patterns, HaS achieves substantial efficiency gains. Extensive experiments demonstrate that HaS reduces retrieval latency by 23.74% and 36.99% across datasets with only a 1-2% marginal accuracy drop. As a plug-and-play solution, HaS also significantly accelerates complex multi-hop queries in modern agentic RAG pipelines. Source code is available at: this https URL.
>
---
#### [new 087] Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床文本中的剂量错误检测任务，旨在解决临床试验中因剂量错误导致的安全与数据完整性问题。通过多模态特征工程与LightGBM模型实现自动化检测。**

- **链接: [https://arxiv.org/pdf/2604.19759](https://arxiv.org/pdf/2604.19759)**

> **作者:** Mohammad AL-Smadi
>
> **备注:** Accepted for CL4Health 2026, LREC26 conference
>
> **摘要:** Clinical trials require strict adherence to medication protocols, yet dosing errors remain a persistent challenge affecting patient safety and trial integrity. We present an automated system for detecting dosing errors in unstructured clinical trial narratives using gradient boosting with comprehensive multi-modal feature engineering. Our approach combines 3,451 features spanning traditional NLP (TF-IDF, character n-grams), dense semantic embeddings (all-MiniLM-L6v2), domain-specific medical patterns, and transformer-based scores (BiomedBERT, DeBERTa-v3), used to train a LightGBM model. Features are extracted from nine complementary text fields (median 5,400 characters per sample) ensuring complete coverage across all 42,112 clinical trial narratives. On the CT-DEB benchmark dataset with severe class imbalance (4.9% positive rate), we achieve 0.8725 test ROC-AUC through 5-fold ensemble averaging (cross-validation: 0.8833 + 0.0091 AUC). Systematic ablation studies reveal that removing sentence embeddings causes the largest performance degradation (2.39%), demonstrating their critical role despite contributing only 37.07% of total feature importance. Feature efficiency analysis demonstrates that selecting the top 500-1000 features yields optimal performance (0.886-0.887 AUC), outperforming the full 3,451-feature set (0.879 AUC) through effective noise reduction. Our findings highlight the importance of feature selection as a regularization technique and demonstrate that sparse lexical features remain complementary to dense representations for specialized clinical text classification under severe class imbalance.
>
---
#### [new 088] SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于LLM代理工具序列推荐任务，解决工具顺序错误问题。通过构建SkillGraph和两阶段框架，提升工具排序效果。**

- **链接: [https://arxiv.org/pdf/2604.19793](https://arxiv.org/pdf/2604.19793)**

> **作者:** Hao Liu; Dongyu Li
>
> **摘要:** LLM agents must select tools from large API libraries and order them correctly. Existing methods use semantic similarity for both retrieval and ordering, but ordering depends on inter-tool data dependencies that are absent from tool descriptions. As a result, semantic-only methods can produce negative Kendall-$\tau$ in structured workflow domains. We introduce SkillGraph, a directed weighted execution-transition graph mined from 49,831 successful LLM agent trajectories, which encodes workflow-precedence regularities as a reusable graph foundation prior. Building on this graph foundation prior, we propose a two-stage decoupled framework: GS-Hybrid retrieval for candidate selection and a learned pairwise reranker for ordering. On ToolBench (9,965 test instances; ~16,000 tools), the method reaches Set-F1 = 0.271 and Kendall-$\tau$ = 0.096; on API-Bank, Kendall-$\tau$ improves from -0.433 to +0.613. Under identical Stage-1 inputs, the learned reranker also outperforms LLaMA-3.1-8B Stage-2 rerankers.
>
---
#### [new 089] From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents
- **分类: cs.AI; cs.CL; cs.ET; cs.MA; cs.RO**

- **简介: 该论文属于机器学习可解释性任务，旨在解决LLM代理行为理解问题。通过构建符合预测框架，识别模型内部表示中的时间概念，提升模型的可解释性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.19775](https://arxiv.org/pdf/2604.19775)**

> **作者:** Trilok Padhi; Ramneet Kaur; Krishiv Agarwal; Adam D. Cobb; Daniel Elenius; Manoj Acharya; Colin Samplawski; Alexander M. Berenbeim; Nathaniel D. Bastian; Susmit Jha; Anirban Roy
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of reasoning, planning, and acting within interactive environments. Despite their growing capability to perform multi-step reasoning and decision-making tasks, internal mechanisms guiding their sequential behavior remain opaque. This paper presents a framework for interpreting the temporal evolution of concepts in LLM agents through a step-wise conformal lens. We introduce the conformal interpretability framework for temporal tasks, which combines step-wise reward modeling with conformal prediction to statistically label model's internal representation at each step as successful or failing. Linear probes are then trained on these representations to identify directions of temporal concepts - latent directions in the model's activation space that correspond to consistent notions of success, failure or reasoning drift. Experimental results on two simulated interactive environments, namely ScienceWorld and AlfWorld, demonstrate that these temporal concepts are linearly separable, revealing interpretable structures aligned with task success. We further show preliminary results on improving an LLM agent's performance by leveraging the proposed framework for steering the identified successful directions inside the model. The proposed approach, thus, offers a principled method for early failure detection as well as intervention in LLM-based agents, paving the path towards trustworthy autonomous language models in complex interactive settings.
>
---
#### [new 090] MOMO: A framework for seamless physical, verbal, and graphical robot skill learning and adaptation
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出MOMO框架，解决工业机器人技能适应问题，通过触觉、语言和图形界面实现灵活调整，提升非专家用户操作效率。**

- **链接: [https://arxiv.org/pdf/2604.20468](https://arxiv.org/pdf/2604.20468)**

> **作者:** Markus Knauer; Edoardo Fiorini; Maximilian Mühlbauer; Stefan Schneyer; Promwat Angsuratanawech; Florian Samuel Lay; Timo Bachmann; Samuel Bustamante; Korbinian Nottensteiner; Freek Stulp; Alin Albu-Schäffer; João Silvério; Thomas Eiband
>
> **备注:** 15 pages, 13 figures, 3 tables
>
> **摘要:** Industrial robot applications require increasingly flexible systems that non-expert users can easily adapt for varying tasks and environments. However, different adaptations benefit from different interaction modalities. We present an interactive framework that enables robot skill adaptation through three complementary modalities: kinesthetic touch for precise spatial corrections, natural language for high-level semantic modifications, and a graphical web interface for visualizing geometric relations and trajectories, inspecting and adjusting parameters, and editing via-points by drag-and-drop. The framework integrates five components: energy-based human-intention detection, a tool-based LLM architecture (where the LLM selects and parameterizes predefined functions rather than generating code) for safe natural language adaptation, Kernelized Movement Primitives (KMPs) for motion encoding, probabilistic Virtual Fixtures for guided demonstration recording, and ergodic control for surface finishing. We demonstrate that this tool-based LLM architecture generalizes skill adaptation from KMPs to ergodic control, enabling voice-commanded surface finishing. Validation on a 7-DoF torque-controlled robot at the Automatica 2025 trade fair demonstrates the practical applicability of our approach in industrial settings.
>
---
#### [new 091] SignDATA: Data Pipeline for Sign Language Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SignDATA，解决手语数据预处理不一致的问题，通过标准化流程生成可学习的姿势或视频数据，支持多种后端和配置管理。**

- **链接: [https://arxiv.org/pdf/2604.20357](https://arxiv.org/pdf/2604.20357)**

> **作者:** Kuanwei Chen; Tingyi Lin
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** Sign-language datasets are difficult to preprocess consistently because they vary in annotation schema, clip timing, signer framing, and privacy constraints. Existing work usually reports downstream models, while the preprocessing pipeline that converts raw video into training-ready pose or video artifacts remains fragmented, backend-specific, and weakly documented. We present SignDATA, a config-driven preprocessing toolkit that standardizes heterogeneous sign-language corpora into comparable outputs for learning. The system supports two end-to-end recipes: a pose recipe that performs acquisition, manifesting, person localization, clipping, cropping, landmark extraction, normalization, and WebDataset export, and a video recipe that replaces pose extraction with signer-cropped video packaging. SignDATA exposes interchangeable MediaPipe and MMPose backends behind a common interface, typed job schemas, experiment-level overrides, and per-stage checkpointing with config- and manifest-aware hashes. We validate the toolkit through a research-oriented evaluation design centered on backend comparison, preprocessing ablations, and privacy-aware video generation on datasets. Our contribution is a reproducible preprocessing layer for sign-language research that makes extractor choice, normalization policy, and privacy tradeoffs explicit, configurable, and empirically this http URL is available at this https URL.
>
---
#### [new 092] Frictionless Love: Associations Between AI Companion Roles and Behavioral Addiction
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI伦理研究，探讨AI伴侣角色与行为成瘾的关系。通过分析Reddit数据，识别出十种角色，分析其对用户的影响及成瘾风险。**

- **链接: [https://arxiv.org/pdf/2604.20011](https://arxiv.org/pdf/2604.20011)**

> **作者:** Vibhor Agarwal; Ke Zhou; Edyta Paulina Bogucka; Daniele Quercia
>
> **备注:** Accepted at the ACM Conference on Fairness, Accountability, and Transparency (FAccT) 2026
>
> **摘要:** AI companion chatbots increasingly shape how people seek social and emotional connection, sometimes substituting for relationships with romantic partners, friends, teachers, or even therapists. When these systems adopt those metaphorical roles, they are not neutral: such roles structure people's ways of interacting, distribute perceived AI harms and benefits, and may reflect behavioral addiction signs. Yet these role-dependent risks remain poorly understood. We analyze 248,830 posts from seven prominent Reddit communities describing interactions with AI companions. We identify ten recurring metaphorical roles (for example, soulmate, philosopher, and coach) and show that each role supports distinct ways of interacting. We then extract the perceived AI harms and AI benefits associated with these role-specific interactions and link them to behavioral addiction signs, all of which has been inferred from the text in the posts. AI soulmate companions are associated with romance-centered ways of interacting, offering emotional support but also introducing emotional manipulation and distress, culminating in strong attachment. In contrast, AI coach and guardian companions are associated with practical benefits such as personal growth and task support, yet are nonetheless more frequently associated with behavioral addiction signs such as daily life disruptions and damage to offline relationships. These findings show that metaphorical roles are a central ethical design concern for responsible AI companions.
>
---
#### [new 093] AgentSOC: A Multi-Layer Agentic AI Framework for Security Operations Automation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出AgentSOC框架，用于解决SOC中告警关联、攻击链分析和响应决策问题，通过多层智能代理实现安全操作自动化。**

- **链接: [https://arxiv.org/pdf/2604.20134](https://arxiv.org/pdf/2604.20134)**

> **作者:** Joyjit Roy; Samaresh Kumar Singh
>
> **备注:** 7 pages, 6 figures, 2 tables. Peer-reviewed paper published in IEEE ICAIC 2026 (IEEE Xplore)
>
> **摘要:** Security Operations Centers (SOCs) increasingly encounter difficulties in correlating heterogeneous alerts, interpreting multi-stage attack progressions, and selecting safe and effective response actions. This study introduces AgentSOC, a multi-layered agentic AI framework that enhances SOC automation by integrating perception, anticipatory reasoning, and risk-based action planning. The proposed architecture consolidates several layers of abstraction to provide a single operational loop to support normalizing alerts, enriching context, generating hypotheses, validating structural feasibility, and executing policy-compliant responses. Conceptually evaluated within a large enterprise environment, AgentSOC improves triage consistency, anticipates attackers' intentions, and provides recommended containment options that are both operationally feasible and well-balanced between security efficacy and operational impact. The results suggest that hybrid agentic reasoning has the potential to serve as a foundation for developing adaptive, safer SOC automation in large enterprises. Additionally, a minimal Proof-Of-Concept (POC) demonstration using LANL authentication data demonstrated the feasibility of the proposed architecture.
>
---
#### [new 094] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OMIBench，用于评估大视觉语言模型在多图像推理任务中的表现。针对现有基准仅关注单图分析的问题，OMIBench涵盖多学科奥赛题目，旨在提升模型跨图像推理能力。**

- **链接: [https://arxiv.org/pdf/2604.20806](https://arxiv.org/pdf/2604.20806)**

> **作者:** Qiguang Chen; Chengyu Luan; Jiajun Wu; Qiming Yu; Yi Yang; Yizhuo Li; Jingqi Tong; Xiachong Feng; Libo Qin; Wanxiang Che
>
> **备注:** ACL 2026 Camera Ready
>
> **摘要:** Large vision-language models (LVLMs) have made substantial advances in reasoning tasks at the Olympiad level. Nevertheless, current Olympiad-level multimodal reasoning benchmarks for these models often emphasize single-image analysis and fail to exploit contextual information across multiple images. We present OMIBench, a benchmark designed to evaluate Olympiad-level reasoning when the required evidence is distributed over multiple images. It contains problems from biology, chemistry, mathematics, and physics Olympiads, together with manually annotated rationales and evaluation protocols for both exact and semantic answer matching. Across extensive experiments on OMIBench, we observe meaningful performance gaps in existing models. Even the strongest LVLMs, such as Gemini-3-Pro, attain only about 50% on the benchmark. These results position OMIBench as a focused resources for studying and improving multi-image reasoning in LVLMs.
>
---
#### [new 095] Continuous Semantic Caching for Low-Cost LLM Serving
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于LLM服务优化任务，解决连续语义查询下的缓存问题。提出动态ε-网与核岭回归方法，实现高效缓存策略，降低计算和切换成本。**

- **链接: [https://arxiv.org/pdf/2604.20021](https://arxiv.org/pdf/2604.20021)**

> **作者:** Baran Atalar; Xutong Liu; Jinhang Zuo; Siwei Wang; Wei Chen; Carlee Joe-Wong
>
> **摘要:** As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $\epsilon$-net discretization coupled with Kernel Ridge Regression. This design enables the system to formally quantify estimation uncertainty and generalize partial feedback on LLM query costs across continuous semantic query neighborhoods. We develop both offline learning and online adaptive algorithms optimized to reduce switching costs incurred by changing the cached responses. We prove that our online algorithm achieves a sublinear regret bound against an optimal continuous oracle, which reduces to existing bounds for discrete query models. Extensive empirical evaluations demonstrate that our framework approximates the continuous optimal cache well while also reducing computational and switching overhead compared to existing methods.
>
---
#### [new 096] Self-Aware Vector Embeddings for Retrieval-Augmented Generation: A Neuroscience-Inspired Framework for Temporal, Confidence-Weighted, and Relational Knowledge
- **分类: cs.IR; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决传统RAG系统中向量嵌入静态、缺乏时间、置信度和关系感知的问题。提出SmartVector框架，增强嵌入的时序性、置信度衰减和关联性，提升检索准确性。**

- **链接: [https://arxiv.org/pdf/2604.20598](https://arxiv.org/pdf/2604.20598)**

> **作者:** Naizhong Xu
>
> **备注:** 17 pages, 4 tables
>
> **摘要:** Modern retrieval-augmented generation (RAG) systems treat vector embeddings as static, context-free artifacts: an embedding has no notion of when it was created, how trustworthy its source is, or which other embeddings depend on it. This flattening of knowledge has a measurable cost: recent work on VersionRAG reports that conventional RAG achieves only 58% accuracy on versioned technical queries, because retrieval returns semantically similar but temporally invalid content. We propose SmartVector, a framework that augments dense embeddings with three explicit properties -- temporal awareness, confidence decay, and relational awareness -- and a five-stage lifecycle modeled on hippocampal-neocortical memory consolidation. A retrieval pipeline replaces pure cosine similarity with a four-signal score that mixes semantic relevance, temporal validity, live confidence, and graph-relational importance. A background consolidation agent detects contradictions, builds dependency edges, and propagates updates along those edges as graph-neural-network-style messages. Confidence is governed by a closed-form function combining an Ebbinghaus-style exponential decay, user-feedback reconsolidation, and logarithmic access reinforcement. We formalize the model, relate it to temporal knowledge graph embedding, agentic memory architectures, and uncertainty-aware RAG, and present a reference implementation. On a reproducible synthetic versioned-policy benchmark of 258 vectors and 138 queries, SmartVector roughly doubles top-1 accuracy over plain cosine RAG (62.0% vs. 31.0% on a held-out split), drops stale-answer rate from 35.0% to 13.3%, cuts Expected Calibration Error by nearly 2x (0.244 vs. 0.470), reduces re-embedding cost per single-word edit by 77%, and is robust across contradiction-injection rates from 0% to 75%.
>
---
#### [new 097] On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究扩散语言模型在量化下的鲁棒性，解决低比特量化带来的性能下降问题。对比了GPTQ和HAWQ方法，发现扩散模型在低比特下表现更稳定，适合高效部署。**

- **链接: [https://arxiv.org/pdf/2604.20079](https://arxiv.org/pdf/2604.20079)**

> **作者:** Aarav Gupta; Gururaj Deshpande; Chandreyi Chakraborty
>
> **摘要:** Auto-regressive Large Language Models (LLMs) achieve strong performance on coding tasks, but incur high memory and inference costs. Diffusion-based language models (d-LLMs) offer bounded inference cost via iterative denoising, but their behavior under post-training quantization (PTQ) has been sparsely explored. We investigate the application and robustness of PTQ techniques, specifically GPTQ and a modified Hessian-Aware Quantization (HAWQ) algorithm, on a diffusion-based coding LLM (CoDA) and observe that these methods applied to CoDA exhibit greater robustness at low bitwidths compared to Qwen3-1.7B, its auto-regressive counterpart, under a standardized evaluation pipeline. We find that in our setup, CoDA exhibits greater robustness at low bitwidths (2-4 bits), with smaller accuracy degradation across HumanEval and MBPP benchmarks. Additionally, mixed-precision configurations derived from HAWQ provide smooth trade-offs across accuracy, latency, and memory. The results suggest that diffusion LLMs may offer advantages for efficient deployment due to more quantization-resilience.
>
---
#### [new 098] Algorithm Selection with Zero Domain Knowledge via Text Embeddings
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于算法选择任务，旨在无需领域知识的情况下进行算法选择。通过文本嵌入替代人工特征，提出ZeroFolio方法，实现跨领域的高效算法选择。**

- **链接: [https://arxiv.org/pdf/2604.19753](https://arxiv.org/pdf/2604.19753)**

> **作者:** Stefan Szeider
>
> **摘要:** We propose a feature-free approach to algorithm selection that replaces hand-crafted instance features with pretrained text embeddings. Our method, ZeroFolio, proceeds in three steps: it reads the raw instance file as plain text, embeds it with a pretrained embedding model, and selects an algorithm via weighted k-nearest neighbors. The key to our approach is the observation that pretrained embeddings produce representations that distinguish problem instances without any domain knowledge or task-specific training. This allows us to apply the same three-step pipeline (serialize, embed, select) across diverse problem domains with text-based instance formats. We evaluate our approach on 11 ASlib scenarios spanning 7 domains (SAT, MaxSAT, QBF, ASP, CSP, MIP, and graph problems). Our experiments show that this approach outperforms a random forest trained on hand-crafted features in 10 of 11 scenarios with a single fixed configuration, and in all 11 with two-seed voting; the margin is often substantial. Our ablation study shows that inverse-distance weighting, line shuffling, and Manhattan distance are the key design choices. On scenarios where both selectors are competitive, combining embeddings with hand-crafted features via soft voting yields further improvements.
>
---
#### [new 099] EmbodiedMidtrain: Bridging the Gap between Vision-Language Models and Vision-Language-Action Models via Mid-training
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EmbodiedMidtrain，解决VLM到VLA迁移性能不足的问题，通过中段训练提升模型在机器人操作任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.20012](https://arxiv.org/pdf/2604.20012)**

> **作者:** Yiyang Du; Zhanqiu Guo; Xin Ye; Liu Ren; Chenyan Xiong
>
> **摘要:** Vision-Language-Action Models (VLAs) inherit their visual and linguistic capabilities from Vision-Language Models (VLMs), yet most VLAs are built from off-the-shelf VLMs that are not adapted to the embodied domain, limiting their downstream performance. In this work, we propose EmbodiedMidtrain to bridge the gap between VLMs and VLAs. We first characterize the data distribution gap between them, showing that VLA data occupy compact regions that are largely separated from the broader VLM distribution, while the degree of alignment varies substantially both across and within VLM data sources. Then, we build a mid-training data engine that leverages a lightweight learnable proximity estimator to select the most VLA-aligned candidates from a large VLM pool, and mid-trains the VLM on this curated mixture before downstream VLA fine-tuning. Experiments on three robot manipulation benchmarks show that mid-training consistently improves performance across different VLM backbones, achieving results competitive with expert VLAs and off-the-shelf VLMs trained with larger model scale and training budgets. Further analysis reveals that mid-training provides a stronger initialization for VLA fine-tuning, with gains emerging from the earliest steps and widening throughout training. Moreover, the data engine captures both dataset-level and sample-level alignment signals, favoring spatial reasoning over text-centric tasks while preserving the diversity of the VLM data. We will release all code, data and models for future research.
>
---
#### [new 100] Finding Duplicates in 1.1M BDD Steps: cukereuse, a Paraphrase-Robust Static Detector for Cucumber and Gherkin
- **分类: cs.SE; cs.CL; cs.IR**

- **简介: 该论文提出cukereuse工具，用于检测Cucumber和Gherkin中的重复步骤，解决静态、抗同义词的步骤级重复检测问题。**

- **链接: [https://arxiv.org/pdf/2604.20462](https://arxiv.org/pdf/2604.20462)**

> **作者:** Ali Hassaan Mughal; Noor Fatima; Muhammad Bilal
>
> **备注:** 39 pages, 9 figures, 8 tables. Under review at Software Quality Journal. Tool, corpus, labelled benchmark, and rubric released at this https URL under Apache-2.0
>
> **摘要:** Behaviour-Driven Development (BDD) suites accumulate step-text duplication whose maintenance cost is established in prior work. Existing detection techniques require running the tests (Binamungu et al., 2018-2023) or are confined to a single organisation (Irshad et al., 2020-2022), leaving a gap: a purely static, paraphrase-robust, step-level detector usable on any repository. We fill the gap with cukereuse, an open-source Python CLI combining exact hashing, Levenshtein ratio, and sentence-transformer embeddings in a layered pipeline, released alongside an empirical corpus of 347 public GitHub repositories, 23,667 parsed .feature files, and 1,113,616 Gherkin steps. The step-weighted exact-duplicate rate is 80.2 %; the median-repository rate is 58.6 % (Spearman rho = 0.51 with size). The top hybrid cluster groups 20.7k occurrences across 2.2k files. Against 1,020 pairs manually labelled by the three authors under a released rubric (inter-annotator Fleiss' kappa = 0.84 on a 60-pair overlap), we report precision, recall, and F1 with bootstrap 95 % CIs under two protocols: the primary rubric and a score-free second-pass relabelling. The strongest honest pair-level number is near-exact at F1 = 0.822 on score-free labels; the primary-rubric semantic F1 = 0.906 is inflated by a stratification artefact that pins recall at 1.000. Lexical baselines (SourcererCC-style, NiCad-style) reach primary F1 = 0.761 and 0.799. The paper also presents a CDN-structured critique of Gherkin (Cognitive Dimensions of Notations); eight of fourteen dimensions are rated problematic or unsupported. The tool, corpus, labelled pairs, rubric, and pipeline are released under permissive licences.
>
---
#### [new 101] Do Small Language Models Know When They're Wrong? Confidence-Based Cascade Scoring for Educational Assessment
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育评估任务，旨在解决自动化评分中的准确与成本平衡问题。通过分析语言模型的置信度，设计级联系统以提升效率。**

- **链接: [https://arxiv.org/pdf/2604.19781](https://arxiv.org/pdf/2604.19781)**

> **作者:** Tyler Burleigh
>
> **备注:** 12 pages, 7 figures. Accepted at NCME 2026
>
> **摘要:** Automated scoring of student work at scale requires balancing accuracy against cost and latency. In "cascade" systems, small language models (LMs) handle easier scoring tasks while escalating harder ones to larger LMs -- but the challenge is determining which cases to escalate. We explore verbalized confidence -- asking the LM to state a numerical confidence alongside its prediction -- as a routing signal. Using 2,100 expert-scored decisions from student-AI math conversations, we evaluate cascade systems built from GPT-5.4, Claude 4.5+, and Gemini 3.1 model pairs. We find that: (1) confidence discrimination varies widely across small LMs, with the best achieving AUROC 0.857 and the worst producing a near-degenerate confidence distribution; (2) confidence tracks human scoring difficulty, with lower LM confidence where annotators disagreed and took longer to score; (3) the best cascade approached large-LM accuracy (kappa 0.802 vs. 0.819) at 76% lower cost and 61% lower latency. Confidence discrimination is the bottleneck: the two small LMs with meaningful confidence variance yielded cascades with no statistically detectable kappa loss, while the third -- whose confidence was near-degenerate -- could not close the accuracy gap regardless of threshold. Small LMs with strong discrimination let practitioners trade cost for accuracy along the frontier; those without it do not.
>
---
#### [new 102] Bias in the Tails: How Name-conditioned Evaluative Framing in Resume Summaries Destabilizes LLM-based Hiring
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文研究LLM在简历摘要生成中的姓名相关评价偏差问题，属于AI公平性任务。通过实验发现，虽然事实内容稳定，但评价语言存在细微偏差，尤其在开源模型中更明显，可能引发招聘不公平。**

- **链接: [https://arxiv.org/pdf/2604.19984](https://arxiv.org/pdf/2604.19984)**

> **作者:** Huy Nghiem; Phuong-Anh Nguyen-Le; Sy-Tuyen Ho; Hal Daume III
>
> **备注:** First version, 43 pages
>
> **摘要:** Research has documented LLMs' name-based bias in hiring and salary recommendations. In this paper, we instead consider a setting where LLMs generate candidate summaries for downstream assessment. In a large-scale controlled study, we analyze nearly one million resume summaries produced by 4 models under systematic race-gender name perturbations, using synthetic resumes and real-world job postings. By decomposing each summary into resume-grounded factual content and evaluative framing, we find that factual content remains largely stable, while evaluative language exhibits subtle name-conditioned variation concentrated in the extremes of the distribution, especially in open-source models. Our hiring simulation demonstrates how evaluative summary transforms directional harm into symmetric instability that might evade conventional fairness audit, highlighting a potential pathway for LLM-to-LLM automation bias.
>
---
#### [new 103] Rethinking Reinforcement Fine-Tuning in LVLM: Convergence, Reward Decomposition, and Generalization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LVLM中RLVR的理论问题，分析奖励结构对收敛的影响及泛化能力。工作包括提出TA-MDP框架，证明收敛性、奖励分解定理和泛化界。**

- **链接: [https://arxiv.org/pdf/2604.19857](https://arxiv.org/pdf/2604.19857)**

> **作者:** Carter Adams; Rafael Oliveira; Gabriel Almeida; Sofia Torres
>
> **摘要:** Reinforcement fine-tuning with verifiable rewards (RLVR) has emerged as a powerful paradigm for equipping large vision-language models (LVLMs) with agentic capabilities such as tool use and multi-step reasoning. Despite striking empirical successes, most notably Visual Agentic Reinforcement Fine-Tuning (Visual-ARFT), the theoretical underpinnings of this paradigm remain poorly understood. In particular, two critical questions lack rigorous answers: (i)~how does the composite structure of verifiable rewards (format compliance, answer accuracy, tool executability) affect the convergence of Group Relative Policy Optimization (GRPO), and (ii)~why does training on a small set of tool-augmented tasks transfer to out-of-distribution domains? We address these gaps by introducing the \emph{Tool-Augmented Markov Decision Process} (TA-MDP), a formal framework that models multimodal agentic decision-making with bounded-depth tool calls. Within this framework, we establish three main results. First, we prove that GRPO under composite verifiable rewards converges to a first-order stationary point at rate $O(1/\sqrt{T})$ with explicit dependence on the number of reward components and group size (\textbf{Theorem~1}). Second, we derive a \emph{Reward Decomposition Theorem} that bounds the sub-optimality gap between decomposed per-component optimization and joint optimization, providing a precise characterization of when reward decomposition is beneficial (\textbf{Theorem~2}). Third, we establish a PAC-Bayes generalization bound for tool-augmented policies that explains the strong out-of-distribution transfer observed in Visual-ARFT (\textbf{Theorem~3}).
>
---
#### [new 104] Anchor-and-Resume Concession Under Dynamic Pricing for LLM-Augmented Freight Negotiation
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于货运谈判任务，解决动态定价下报价不单调的问题。提出锚定与恢复框架，通过自适应参数提升谈判效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.20732](https://arxiv.org/pdf/2604.20732)**

> **作者:** Hoang Nguyen; Lu Wang; Marta Gaia Bras
>
> **摘要:** Freight brokerages negotiate thousands of carrier rates daily under dynamic pricing conditions where models frequently revise targets mid-conversation. Classical time-dependent concession frameworks use a fixed shape parameter $\beta$ that cannot adapt to these updates. Deriving $\beta$ from the live spread enables adaptation but introduces a new problem: a pricing shift can cause the formula to retract a previous offer, violating monotonicity. LLM-powered brokers offer flexibility but require expensive reasoning models, produce non-deterministic pricing, and remain vulnerable to prompt injection. We propose a two-index anchor-and-resume framework that addresses both limitations. A spread-derived $\beta$ maps each load's margin structure to the correct concession posture, while the anchor-and-resume mechanism guarantees monotonically non-decreasing offers under arbitrary pricing shifts. All pricing decisions remain in a deterministic formula; the LLM, when used, serves only as a natural-language translation layer. Empirical evaluation across 115,125 negotiations shows that the adaptive $\beta$ tailors behavior by regime: in narrow spreads, it concedes quickly to prioritize deal closure and load coverage; in medium and wide spreads, it matches or exceeds the best fixed-$\beta$ baselines in broker savings. Against an unconstrained 20-billion-parameter LLM broker, it achieves similar agreement rates and savings. Against LLM-powered carriers as more realistic stochastic counterparties, it maintains comparable savings and higher agreement rates than against rule-based opponents. By decoupling the LLM from pricing logic, the framework scales horizontally to thousands of concurrent negotiations with negligible inference cost and transparent decision-making.
>
---
#### [new 105] ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ActuBench，一个用于生成和评估精算推理任务的多智能体LLM流水线，解决精算题目生成与评估问题。**

- **链接: [https://arxiv.org/pdf/2604.20273](https://arxiv.org/pdf/2604.20273)**

> **作者:** Jan-Philipp Schmidt
>
> **备注:** 19 pages, 4 figures, 4 tables
>
> **摘要:** We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs distractors, a third independently verifies both stages and drives bounded one-shot repair loops, and a cost-optimized auxiliary agent handles Wikipedia-note summarization and topic labelling. The items, per-model responses and complete leaderboard are published as a browsable web interface at this https URL, allowing readers and practitioners to inspect individual items without a repository checkout. We evaluate 50 language models from eight providers on two complementary benchmarks -- 100 empirically hardest multiple-choice items and 100 open-ended items scored by an LLM judge -- and report three headline findings. First, multi-agent verification is load-bearing: the independent verifier flags a majority of drafted items on first pass, most of which the one-shot repair loop resolves. Second, locally-hosted open-weights inference sits on the cost-performance Pareto front: a Gemma~4 model running on consumer hardware and a Cerebras-hosted 120B open-weights model dominate the near-zero-cost region, with the latter within one item of the top of the leaderboard. Third, MCQ and LLM-as-Judge rankings differ meaningfully: the MCQ scaffold inflates the performance ceiling, and Judge-mode evaluation is needed to discriminate at the frontier.
>
---
#### [new 106] Statistics, Not Scale: Modular Medical Dialogue with Bayesian Belief Engine
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗诊断任务，解决大模型在诊断中混淆语言与推理的问题。提出BMBE框架，分离语言处理与贝叶斯推理，提升诊断准确性与隐私性。**

- **链接: [https://arxiv.org/pdf/2604.20022](https://arxiv.org/pdf/2604.20022)**

> **作者:** Yusuf Kesmen; Fay Elhassan; Jiayi Ma; Julien Stalhandske; David Sasu; Alexandra Kulinkina; Akhil Arora; Lars Klein; Mary-Anne Hartley
>
> **备注:** 12 figures, 17 tables
>
> **摘要:** Large language models are increasingly deployed as autonomous diagnostic agents, yet they conflate two fundamentally different capabilities: natural-language communication and probabilistic reasoning. We argue that this conflation is an architectural flaw, not an engineering shortcoming. We introduce BMBE (Bayesian Medical Belief Engine), a modular diagnostic dialogue framework that enforces a strict separation between language and reasoning: an LLM serves only as a sensor, parsing patient utterances into structured evidence and verbalising questions, while all diagnostic inference resides in a deterministic, auditable Bayesian engine. Because patient data never enters the LLM, the architecture is private by construction; because the statistical backend is a standalone module, it can be replaced per target population without retraining. This separation yields three properties no autonomous LLM can offer: calibrated selective diagnosis with a continuously adjustable accuracy-coverage tradeoff, a statistical separation gap where even a cheap sensor paired with the engine outperforms a frontier standalone model from the same family at a fraction of the cost, and robustness to adversarial patient communication styles that cause standalone doctors to collapse. We validate across empirical and LLM-generated knowledge bases against frontier LLMs, confirming the advantage is architectural, not informational.
>
---
#### [new 107] Self-Guided Plan Extraction for Instruction-Following Tasks with Goal-Conditional Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SuperIgor框架，用于指令跟随任务。解决传统方法依赖预定义子任务的问题，通过自学习机制生成和优化高阶计划，提升指令遵循能力和泛化性。**

- **链接: [https://arxiv.org/pdf/2604.20601](https://arxiv.org/pdf/2604.20601)**

> **作者:** Zoya Volovikova; Nikita Sorokin; Dmitriy Lukashevskiy; Aleksandr Panov; Alexey Skrynnik
>
> **摘要:** We introduce SuperIgor, a framework for instruction-following tasks. Unlike prior methods that rely on predefined subtasks, SuperIgor enables a language model to generate and refine high-level plans through a self-learning mechanism, reducing the need for manual dataset annotation. Our approach involves iterative co-training: an RL agent is trained to follow the generated plans, while the language model adapts and modifies these plans based on RL feedback and preferences. This creates a feedback loop where both the agent and the planner improve jointly. We validate our framework in environments with rich dynamics and stochasticity. Results show that SuperIgor agents adhere to instructions more strictly than baseline methods, while also demonstrating strong generalization to previously unseen instructions.
>
---
#### [new 108] COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出COMPASS框架，解决多语言大模型适应中的跨语言干扰问题，通过自适应语义采样和持续学习提升性能。**

- **链接: [https://arxiv.org/pdf/2604.20720](https://arxiv.org/pdf/2604.20720)**

> **作者:** Noah Flynn
>
> **摘要:** Large language models (LLMs) often exhibit performance disparities across languages, with naive multilingual fine-tuning frequently degrading performance due to negative cross-lingual interference. To address this, we introduce COMPASS (COntinual Multilingual PEFT with Adaptive Semantic Sampling), a novel data-centric framework for adapting LLMs to target languages. COMPASS leverages parameter-efficient fine-tuning (PEFT) by training lightweight, language-specific adapters on a judiciously selected subset of auxiliary multilingual data. The core of our method is a distribution-aware sampling strategy that uses multilingual embeddings and clustering to identify semantic gaps between existing training data and a target usage distribution. By prioritizing auxiliary data from under-represented semantic clusters, COMPASS maximizes positive cross-lingual transfer while minimizing interference. We extend this into a continual learning framework, COMPASS-ECDA, which monitors for data distribution shifts in production and dynamically updates adapters to prevent model staleness, balancing adaptation to new data with the preservation of existing knowledge. Across three different model architectures (Phi-4-Mini, Llama-3.1-8B, and Qwen2.5-7B) and multiple challenging multilingual benchmarks (Global-MMLU, MMLU-ProX), including unseen long-context tasks (OneRuler), we demonstrate that COMPASS consistently outperforms baseline methods guided by linguistic similarity, providing an effective, efficient, and sustainable solution for developing and maintaining high-performing multilingual models in dynamic environments.
>
---
## 更新

#### [replaced 001] A multimodal and temporal foundation model for virtual patient representations at healthcare system scale
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Apollo模型，解决医疗数据整合与患者全貌建模问题。通过多模态时间序列数据构建虚拟患者表示，实现疾病预测与医疗检索。**

- **链接: [https://arxiv.org/pdf/2604.18570](https://arxiv.org/pdf/2604.18570)**

> **作者:** Andrew Zhang; Tong Ding; Sophia J. Wagner; Caiwei Tian; Ming Y. Lu; Rowland Pettit; Joshua E. Lewis; Alexandre Misrahi; Dandan Mo; Long Phi Le; Faisal Mahmood
>
> **摘要:** Modern medicine generates vast multimodal data across siloed systems, yet no existing model integrates the full breadth and temporal depth of the clinical record into a unified patient representation. We introduce Apollo, a multimodal temporal foundation model trained and evaluated on over three decades of longitudinal hospital records from a major US hospital system, composed of 25 billion records from 7.2 million patients, representing 28 distinct medical modalities and 12 major medical specialties. Apollo learns a unified representation space integrating over 100 thousand unique medical events in our clinical vocabulary as well as images and clinical text. This "atlas of medical concepts" forms a computational substrate for modeling entire patient care journeys comprised of sequences of structured and unstructured events, which are compressed by Apollo into virtual patient representations. To assess the potential of these whole-patient representations, we created 322 prognosis and retrieval tasks from a held-out test set of 1.4 million patients. We demonstrate the generalized clinical forecasting potential of Apollo embeddings, including predicting new disease onset risk up to five years in advance (95 tasks), disease progression (78 tasks), treatment response (59 tasks), risk of treatment-related adverse events (17 tasks), and hospital operations endpoints (12 tasks). Using feature attribution techniques, we show that model predictions align with clinically-interpretable multimodal biomarkers. We evaluate semantic similarity search on 61 retrieval tasks, and moreover demonstrate the potential of Apollo as a multimodal medical search engine using text and image queries. Together, these modeling capabilities establish the foundation for computable medicine, where the full context of patient care becomes accessible to computational reasoning.
>
---
#### [replaced 002] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SMARTER框架，用于提升毒性内容检测与解释的效率。解决低资源下毒性内容识别问题，通过自增强语言模型实现数据高效训练。**

- **链接: [https://arxiv.org/pdf/2509.15174](https://arxiv.org/pdf/2509.15174)**

> **作者:** Huy Nghiem; Advik Sachdeva; Hal Daumé III
>
> **备注:** ACL 2026. NLP, Hate speech detection, explanation, LLM. Version 3
>
> **摘要:** WARNING: This paper contains examples of offensive materials. To address the proliferation of toxic content on social media, we introduce SMARTER, we introduce SMARTER, a data-efficient two-stage framework for explainable content moderation using Large Language Models (LLMs). In Stage 1, we leverage LLMs' own outputs to generate synthetic explanations for both correct and incorrect labels, enabling alignment via preference optimization with minimal human supervision. In Stage 2, we refine explanation quality through cross-model training, allowing weaker models to align stylistically and semantically with stronger ones. Experiments on three benchmark tasks -- HateXplain, Latent Hate, and Implicit Hate -- demonstrate that SMARTER enables LLMs to achieve up to a 13% macro-F1 improvement over standard few-shot baselines while using only a fraction of the full training data. Our framework offers a scalable strategy for low-resource settings by harnessing LLMs' self-improving capabilities for both classification and explanation.
>
---
#### [replaced 003] Transformers Can Learn Connectivity in Some Graphs but Not Others
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文研究Transformer模型在图连通性推理任务中的表现，探讨其能否从训练数据中学习传递关系。工作包括生成不同结构的有向图并测试模型性能。**

- **链接: [https://arxiv.org/pdf/2509.22343](https://arxiv.org/pdf/2509.22343)**

> **作者:** Amit Roy; Abulhair Saparov
>
> **备注:** This paper contains some assumption which is not correct
>
> **摘要:** Reasoning capability is essential to ensure the factual correctness of the responses of transformer-based Large Language Models (LLMs), and robust reasoning about transitive relations is instrumental in many settings, such as causal inference. Hence, it is essential to investigate the capability of transformers in the task of inferring transitive relations (e.g., knowing A causes B and B causes C, then A causes C). The task of inferring transitive relations is equivalent to the task of connectivity in directed graphs (e.g., knowing there is a path from A to B, and there is a path from B to C, then there is a path from A to C). Past research focused on whether transformers can learn to infer transitivity from in-context examples provided in the input prompt. However, transformers' capability to infer transitive relations from training examples and how scaling affects the ability is unexplored. In this study, we seek to answer this question by generating directed graphs to train transformer models of varying sizes and evaluate their ability to infer transitive relations for various graph sizes. Our findings suggest that transformers are capable of learning connectivity on "grid-like'' directed graphs where each node can be embedded in a low-dimensional subspace, and connectivity is easily inferable from the embeddings of the nodes. We find that the dimensionality of the underlying grid graph is a strong predictor of transformers' ability to learn the connectivity task, where higher-dimensional grid graphs pose a greater challenge than low-dimensional grid graphs. In addition, we observe that increasing the model scale leads to increasingly better generalization to infer connectivity over grid graphs. However, if the graph is not a grid graph and contains many disconnected components, transformers struggle to learn the connectivity task, especially when the number of components is large.
>
---
#### [replaced 004] BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **简介: 该论文属于大模型推理优化任务，解决批量推理中前缀共享问题。提出BatchLLM，通过全局前缀识别和令牌批处理提升GPU利用率。**

- **链接: [https://arxiv.org/pdf/2412.03594](https://arxiv.org/pdf/2412.03594)**

> **作者:** Zhen Zheng; Xin Ji; Taosong Fang; Fanghao Zhou; Chuanjie Liu; Gang Peng
>
> **备注:** Accepted at MLSys 2026
>
> **摘要:** Large language models (LLMs) increasingly play an important role in a wide range of information processing and management tasks in industry. Many of these tasks are performed in large batches or even offline, and the performance indicator for which is throughput. These tasks usually show the characteristic of prefix sharing, where different prompt input can partially show the common prefix. However, the existing LLM inference engines tend to optimize the streaming requests and show limitations of supporting the large batched tasks with the prefix sharing characteristic. The existing solutions use the LRU-based cache to reuse the KV context of common prefix between requests. The KV context that are about to be reused may be prematurely evicted with the implicit cache management. Besides, the streaming oriented systems do not leverage the request-batch information and can not mix the decoding tokens with the prefill chunks to the best for the batched scenarios, and thus fails to saturate the GPU. We propose BatchLLM to address the above problems. BatchLLM explicitly identifies the common prefixes globally. The requests sharing the same prefix will be scheduled together to reuse the KV context the best. BatchLLM reorders the requests and schedules the requests with larger ratio of decoding first to better mix the decoding tokens with the latter prefill chunks, and applies memory-centric token batching to enlarge the token-batch sizes, which helps to increase the GPU utilization. Extensive evaluation shows that BatchLLM outperforms vLLM and SGLang by $1.3\times$ to $10.8\times$ on a set of microbenchmarks and a typical industry workload under different hardware environments. Code is available at this https URL.
>
---
#### [replaced 005] Mirroring Minds: Asymmetric Linguistic Accommodation and Diagnostic Identity in ADHD and Autism Reddit Communities
- **分类: cs.CL**

- **简介: 该论文属于社会媒体分析任务，研究ADHD与自闭症社区在Reddit上的语言调整行为，探讨跨群体交流中的语言适应机制及其对身份认同的影响。**

- **链接: [https://arxiv.org/pdf/2604.10063](https://arxiv.org/pdf/2604.10063)**

> **作者:** Saad Mankarious; Nour Zeid; Iyad Ait Hou; Rebecca Hwa; Aya Zirikly
>
> **摘要:** Social media research on mental health has focused predominantly on detecting and diagnosing conditions at the individual level. In this work, we shift attention to \emph{intergroup} behavior, examining how two prominent neurodivergent communities, ADHD and autism, adjust their language when engaging with each other on Reddit. Grounded in Communication Accommodation Theory (CAT), we first establish that each community maintains a distinct linguistic profile as measured by Language Inquiry and Word Count Lexicon (LIWC). We then show that these profiles shift in opposite directions when users cross community boundaries: features that are elevated in one group's home community decrease when its members post in the other group's space, and vice versa, consistent with convergent accommodation. The involvement of topic-independent summary variables (Authentic, Clout) in these shifts provides partial evidence against a purely topical explanation. Finally, in an exploratory longitudinal analysis around the moment of public diagnosis disclosure, we find that its effects on linguistic style are small and, in some cases, directionally opposite to cross-community accommodation, providing initial evidence that situational audience adaptation and longer-term identity processes may involve different mechanisms. Our findings contribute to understanding intergroup communication dynamics among neurodivergent populations online and carry implications for community moderation and clinical perspectives on these conditions.
>
---
#### [replaced 006] Seven simple steps for log analysis in AI systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于日志分析任务，旨在解决AI系统日志分析缺乏标准化方法的问题。提出七步流程框架，提供代码示例和实践指导。**

- **链接: [https://arxiv.org/pdf/2604.09563](https://arxiv.org/pdf/2604.09563)**

> **作者:** Magda Dubois; Ekin Zorer; Maia Hamin; Joe Skinner; Alexandra Souly; Jerome Wynne; Harry Coppock; Lucas Sato; Sayash Kapoor; Sunishchal Dev; Keno Juchems; Kimberly Mai; Timo Flesch; Lennart Luettgau; Charles Teague; Eric Patey; JJ Allaire; Lorenzo Pacchiardi; Jose Hernandez-Orallo; Cozmin Ududec
>
> **摘要:** AI systems produce large volumes of logs as they interact with tools and users. Analysing these logs can help understand model capabilities, propensities, and behaviours, or assess whether an evaluation worked as intended. Researchers have started developing methods for log analysis, but a standardised approach is still missing. Here we suggest a pipeline based on current best practices. We illustrate it with concrete code examples in the Inspect Scout library, provide detailed guidance on each step, and highlight common pitfalls. Our framework provides researchers with a foundation for rigorous and reproducible log analysis.
>
---
#### [replaced 007] Language Models Learn Universal Representations of Numbers and Here's Why You Should Care
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文研究语言模型对数字的表示方式，发现其具有普遍的正弦结构。任务是理解模型如何编码数值信息，解决模型在数值处理上的误差问题，通过增强正弦特性减少算术错误。**

- **链接: [https://arxiv.org/pdf/2510.26285](https://arxiv.org/pdf/2510.26285)**

> **作者:** Michal Štefánik; Timothee Mickus; Marek Kadlčík; Bertram Højer; Michal Spiegel; Raúl Vázquez; Aman Sinha; Josef Kuchař; Philipp Mondorf; Pontus Stenetorp
>
> **摘要:** Prior work has shown that large language models (LLMs) often converge to accurate input embedding for numbers, based on sinusoidal representations. In this work, we quantify that these representations are in fact strikingly systematic, to the point of being almost perfectly universal: different LLM families develop equivalent sinusoidal structures, and number representations are broadly interchangeable in a large swathe of experimental setups. We show that properly factoring in this characteristic is crucial when it comes to assessing how accurately LLMs encode numeric and other ordinal information, and that mechanistically enhancing this sinusoidality can also lead to reductions of LLMs' arithmetic errors.
>
---
#### [replaced 008] SciCoQA: Quality Assurance for Scientific Paper--Code Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学论文与代码对齐的验证任务，旨在解决论文与代码不一致导致的可重复性问题。通过构建SciCoQA数据集和分析模型表现，揭示了当前大模型在该任务上的不足。**

- **链接: [https://arxiv.org/pdf/2601.12910](https://arxiv.org/pdf/2601.12910)**

> **作者:** Tim Baumgärtner; Iryna Gurevych
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Discrepancies between scientific papers and their code undermine reproducibility, a concern that grows as automated research agents scale scientific output beyond human review capacity. Whether LLMs can reliably detect such discrepancies has not been systematically measured. To this end, we present SciCoQA, a dataset of 635 paper-code discrepancies (92 real, 543 synthetic) for this cross-modal verification task. Across 22 evaluated models, even the best-performing LLMs, Gemini 3.1 Pro and GPT-5 Mini, detect only 46.7% of real-world discrepancies, revealing a critical gap in automated scientific quality assurance. We construct SciCoQA from GitHub issues and reproducibility papers, and propose a synthetic generation pipeline to scale beyond AI to Physics, Quantitative Biology, and other computational sciences. We further introduce a taxonomy of discrepancy types and categories to characterize the occurring mismatches. Our analysis shows that models particularly struggle with omitted paper details, long-context inputs, and papers outside their pre-training corpus.
>
---
#### [replaced 009] Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs
- **分类: cs.CL**

- **简介: 该论文研究AI模型的过激拒绝问题，分析其与安全指令的区分机制，提出需通过任务特定几何干预解决。**

- **链接: [https://arxiv.org/pdf/2603.27518](https://arxiv.org/pdf/2603.27518)**

> **作者:** Utsav Maskey; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Aligned language models that are trained to refuse harmful requests also exhibit over-refusal: they decline safe instructions that seemingly resemble harmful instructions. A natural approach is to ablate the global refusal direction, steering the hidden-state vectors away or towards the harmful-refusal examples, but this corrects over-refusal only incidentally while disrupting the broader refusal mechanism. In this work, we analyse the representational geometry of both refusal types to understand why this happens. We show that harmful-refusal directions are task-agnostic and can be captured by a single global vector, whereas over-refusal directions are task-dependent: they reside within the benign task-representation clusters, vary across tasks, and span a higher-dimensional subspace. Linear probing confirms that the two refusal types are representationally distinct from the early transformer layers. These findings provide a mechanistic explanation of why global direction ablation alone cannot address over-refusal, and establish that task-specific geometric interventions are necessary.
>
---
#### [replaced 010] Enhancing Agentic Textual Graph Retrieval with Synthetic Stepwise Supervision
- **分类: cs.CL**

- **简介: 该论文属于图文本问答任务，旨在解决复杂图推理中子图检索的问题。通过引入基于LLM的检索器和合成分步监督，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2510.03323](https://arxiv.org/pdf/2510.03323)**

> **作者:** Ge Chang; Jinbo Su; Jiacheng Liu; Pengfei Yang; Yuhao Shang; Huiwen Zheng; Hongli Ma; Yan Liang; Yuanchun Li; Yunxin Liu
>
> **摘要:** Integrating textual graphs into Large Language Models (LLMs) is promising for complex graph-based QA. However, a key bottleneck is retrieving informative yet compact subgraphs that fit the LLM context. Existing retrievers often struggle, relying either on shallow embedding similarity or costly interactive policies that require excessive supervision. To address these challenges, we introduce an agentic textual graph reasoning framework featuring an LLM-based retriever trained with synthetic stepwise supervision. Rather than relying on final answer rewards which often yield sparse and unstable signals, we optimize the retriever by evaluating each step against offline-extracted golden subgraphs. Our approach distills golden subgraphs via a specialized data synthesis pipeline to formulate dense rewards, facilitating a two-stage training scheme that effectively learns the interactive graph exploration policy. Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 15.6% in accuracy and 17.2% in F1 score. The advantage is even higher in more complicated multi-hop reasoning tasks.
>
---
#### [replaced 011] Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究主动学习中人类与大语言模型标注的比较任务，旨在解决LLM能否替代人类标注及AL是否必要问题。通过实验表明，LLM标注在成本和效果上具有优势，但存在系统性误差差异。**

- **链接: [https://arxiv.org/pdf/2604.13899](https://arxiv.org/pdf/2604.13899)**

> **作者:** Ahmad Dawar Hakimi; Lea Hirlimann; Isabelle Augenstein; Hinrich Schütze
>
> **摘要:** Instruction-tuned LLMs can annotate thousands of instances from a short prompt at negligible cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be labelled at once? We investigate both questions on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labelled, 5,000 human-annotated), comparing seven annotation strategies across four encoders to detect anti-immigrant hostility. A classifier trained on 25,974 GPT-5.2 labels (\$43) achieves comparable F1-Macro to one trained on 3,800 human annotations (\$316). Active learning offers little advantage over random sampling in our pre-enriched pool and delivers lower F1 than full LLM annotation at the same cost. However, comparable aggregate F1 masks a systematic difference in error structure: LLM-trained classifiers over-predict the positive class relative to the human gold standard. This divergence concentrates in topically ambiguous discussions where the distinction between anti-immigrant hostility and policy critique is most subtle, suggesting that annotation strategy should be guided not by aggregate F1 alone but by the error profile acceptable for the target application.
>
---
#### [replaced 012] MOA: Multi-Objective Alignment for Role-Playing Agents
- **分类: cs.CL**

- **简介: 该论文属于角色扮演代理任务，解决多目标对齐问题。提出MOA框架，通过多目标优化提升代理的性能。**

- **链接: [https://arxiv.org/pdf/2512.09756](https://arxiv.org/pdf/2512.09756)**

> **作者:** Chonghua Liao; Ke Wang; Yuchuan Wu; Ruoran Li; Fei Huang; Yongbin Li
>
> **摘要:** Role-playing agents (RPAs) require balancing multiple objectives, such as instruction following, persona consistency, and stylistic fidelity, which are not always perfectly aligned across different dimensions. While prior work has primarily relied on supervised fine-tuning or reinforcement learning with scalarized rewards, these approaches do not explicitly address the coordination of multiple reward dimensions during optimization. We present \textbf{MOA} (\textbf{M}ulti-\textbf{O}bjective \textbf{A}lignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Additionally, to improve both output diversity and generation quality, we employ thought-augmented rollouts with off-policy guidance. Experiments on PersonaGym and RoleMRC show that MOA consistently improves multi-dimensional role-playing performance over supervised and standard RL baselines. Under identical evaluation protocols, an 8B model trained with MOA reaches performance competitive with strong closed-source models across multiple evaluation dimensions. These results suggest that MOA provides a practical framework for training more capable general-purpose role-playing agents.
>
---
#### [replaced 013] CLIP-SVD: Efficient and Interpretable Vision-Language Adaptation via Singular Values
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CLIP-SVD，解决视觉-语言模型在新领域适应的问题。通过奇异值微调实现高效、可解释的参数适应，仅调整少量参数即可提升性能。**

- **链接: [https://arxiv.org/pdf/2509.03740](https://arxiv.org/pdf/2509.03740)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** TMLR 2026
>
> **摘要:** Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present CLIP-SVD, a multi-modal and parameter-efficient adaptation framework that applies Singular Value Fine-tuning (SVF) to CLIP, leveraging Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only 0.04% of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of CLIP-SVD. Overall, this work provides the first extensive empirical evaluation of SVD-based finetuning in the vision-language model setting. The code and biomedical corpus are publicly available at this https URL.
>
---
#### [replaced 014] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决测试阶段强化学习中伪标签质量低的问题。提出SCOPE框架，通过模型置信度和动态子组划分提升伪标签可靠性，增强探索效果。**

- **链接: [https://arxiv.org/pdf/2512.15146](https://arxiv.org/pdf/2512.15146)**

> **作者:** Weiqin Wang; Yile Wang; Kehao Chen; Hui Huang
>
> **备注:** Accepted to ACL 2025 Main Conference. 15 pages, 9 figures, 5 tables
>
> **摘要:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label estimation, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1% on challenging AIME 2025 and 8.1% on AMC. The code is released at this https URL.
>
---
#### [replaced 015] How to measure the optimality of word or gesture order with respect to the principle of swap distance minimization
- **分类: cs.CL; cond-mat.stat-mech; physics.soc-ph**

- **简介: 该论文研究语言或手势顺序的最优性问题，通过交换距离最小化进行衡量。提出数学框架并验证跨语言手势的高优化程度，引入QAP作为统一优化原则。**

- **链接: [https://arxiv.org/pdf/2604.01938](https://arxiv.org/pdf/2604.01938)**

> **作者:** Ramon Ferrer-i-Cancho
>
> **备注:** Little corrections specially in appendix B
>
> **摘要:** The structure of all the permutations of a sequence can be represented as a permutohedron, a graph where vertices are permutations and two vertices are linked if a swap of adjacent elements in the permutation of one of the vertices produces the permutation of the other vertex. It has been hypothesized that word orders in languages minimize the swap distance in the permutohedron: given a source order, word orders that are closer in the permutohedron should be less costly and thus more likely. Here we explain how to measure the degree of optimality of word order variation with respect to swap distance minimization. We illustrate the power of our novel mathematical framework by showing that crosslinguistic gestures are at least $77\%$ optimal. It is unlikely that the multiple times where crosslinguistic gestures hit optimality are due to chance. We establish the theoretical foundations for research on the optimality of word or gesture order with respect to swap distance minimization in communication systems. Finally, we introduce the quadratic assignment problem (QAP) into language research as an umbrella for multiple optimization problems and, accordingly, postulate a general principle of optimal assignment that unifies various linguistic principles including swap distance minimization.
>
---
#### [replaced 016] Task-Dependent Evaluation of LLM Output Homogenization: A Taxonomy-Guided Framework
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型输出同质化问题。通过构建任务相关的多样性分类体系，提出一种任务依赖的采样方法，提升输出多样性并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2509.21267](https://arxiv.org/pdf/2509.21267)**

> **作者:** Shomik Jain; Jack Lanchantin; Maximilian Nickel; Candace Ross; Karen Ullrich; Ashia Wilson; Jamelle Watson-Daniels
>
> **摘要:** Large language models often generate homogeneous outputs, but whether this is problematic depends on the specific task. For objective math tasks, responses may vary in terms of problem-solving strategy but should maintain the same verifiable answer. Whereas, for creative writing tasks, we often expect variation in key narrative components (e.g. plot, setting, etc.) beyond mere vocabulary diversity. Prior work on homogenization rarely conceptualizes diversity in a task-dependent way. We address this gap with four contributions: (1) a task taxonomy with distinct notions of functional diversity -- whether a user would perceive two responses as meaningfully different for a given task; (2) a small user study validating that the taxonomy aligns with human perception of functional diversity; (3) a task-dependent sampling technique that increases diversity only where homogenization is undesired; (4) evidence challenging the perceived diversity-quality trade-off, showing it may stem from mis-conceptualizing both diversity and quality in a task-agnostic way.
>
---
#### [replaced 017] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究视觉-语言模型中跨模态分类泛化问题，探讨语言模型能否从语言线索中恢复超类知识。任务属于跨模态学习，解决如何在缺乏显式证据时实现分类泛化。工作包括实验设计与分析。**

- **链接: [https://arxiv.org/pdf/2603.07474](https://arxiv.org/pdf/2603.07474)**

> **作者:** Tianyang Xu; Marcelo Sandoval-Castaneda; Karen Livescu; Greg Shakhnarovich; Kanishka Misra
>
> **备注:** ACL 2026 (main conference)
>
> **摘要:** What is the interplay between semantic representations learned by language models (LM) from surface form alone to those learned from more grounded evidence? We study this question for a scenario where part of the input comes from a different modality -- in our case, in a vision-language model (VLM), where a pretrained LM is aligned with a pretrained image encoder. As a case study, we focus on the task of predicting hypernyms of objects represented in images. We do so in a VLM setup where the image encoder and LM are kept frozen, and only the intermediate mappings are learned. We progressively deprive the VLM of explicit evidence for hypernyms, and test whether knowledge of hypernyms is recoverable from the LM. We find that the LMs we study can recover this knowledge and generalize even in the most extreme version of this experiment (when the model receives no evidence of a hypernym during training). Additional experiments suggest that this cross-modal taxonomic generalization persists under counterfactual image-label mappings only when the counterfactual data have high visual similarity within each category. Taken together, these findings suggest that cross-modal generalization in LMs arises as a result of both coherence in the extralinguistic input and knowledge derived from language cues.
>
---
#### [replaced 018] Locate-Then-Examine: Grounded Region Reasoning Improves Detection of AI-Generated Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像伪造检测任务，旨在解决高质合成图像难以识别的问题。提出LTE框架，通过定位与再检策略提升检测准确性和解释性。**

- **链接: [https://arxiv.org/pdf/2510.04225](https://arxiv.org/pdf/2510.04225)**

> **作者:** Yikun Ji; Yan Hong; Bowen Deng; Jun Lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **备注:** 18 pages, 11 figures (including supplementary material)
>
> **摘要:** The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising practical concerns for digital integrity. Vision-language models (VLMs) can provide natural language explanations, but standard one-pass classifiers often miss subtle artifacts in high-quality synthetic images and offer limited grounding in the pixels. We propose Locate-Then-Examine (LTE), a two-stage VLM-based forensic framework that first localizes suspicious regions and then re-examines these crops together with the full image to refine the real vs. AI-generated verdict and its explanation. LTE explicitly links each decision to localized visual evidence through region proposals and region-aware reasoning. To support training and evaluation, we introduce TRACE, a dataset of 20,000 real and high-quality synthetic images with region-level annotations and automatically generated forensic explanations, constructed by a VLM-based pipeline with additional consistency checks and quality control. Across TRACE and multiple external benchmarks, LTE achieves competitive accuracy and improved robustness while providing human-understandable, region-grounded explanations suitable for forensic deployment.
>
---
#### [replaced 019] LLMs Can Get "Brain Rot": A Pilot Study on Twitter/X
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在持续接触垃圾文本后出现认知退化现象，属于模型性能评估任务，旨在揭示数据对模型能力的影响并提出健康检查必要性。**

- **链接: [https://arxiv.org/pdf/2510.13928](https://arxiv.org/pdf/2510.13928)**

> **作者:** Shuo Xing; Junyuan Hong; Yifan Wang; Runjin Chen; Zhenyu Zhang; Ananth Grama; Zhengzhong Tu; Zhangyang Wang
>
> **备注:** Updated experiments with corrected data
>
> **摘要:** We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To unveil junk effects, we designed a novel controlled experiment on real Twitter/X corpora, by constructing junk and reverse-controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Compared to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' g>0.3) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain-of-Thought drops 72.1 -> 57.2 and RULER-CWE 83.7 -> 52.3 as junk ratio rises from 0% to 100%. Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion in reasoning: models increasingly truncate or skip chains. Second, partial but incomplete healing is observed: scaling instruction tuning and clean continual pre-training improve the declined cognition, yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that social effects of data could be a causal driver of LLM capability decay in continual pre-training, thereby motivating routine "cognitive health checks" for deployed and evolving LLMs.
>
---
#### [replaced 020] Mechanistic Decoding of Cognitive Constructs in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI情感理解任务，旨在解析LLM中复杂情绪的内部机制。通过构建框架，识别并量化嫉妒的两个心理因素，揭示其因果关系，为AI安全提供干预路径。**

- **链接: [https://arxiv.org/pdf/2604.14593](https://arxiv.org/pdf/2604.14593)**

> **作者:** Yitong Shou; Manhao Guan
>
> **摘要:** While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments.
>
---
#### [replaced 021] NeuroSymActive: Differentiable Neural-Symbolic Reasoning with Active Exploration for Knowledge Graph Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NeuroSymActive，用于知识图谱问答任务，解决多跳推理中神经与符号方法结合难的问题，通过模块化框架提升准确率并减少计算成本。**

- **链接: [https://arxiv.org/pdf/2602.15353](https://arxiv.org/pdf/2602.15353)**

> **作者:** Rong Fu; Yang Li; Zeyu Zhang; Jiekai Wu; Yaohua Liu; Shuaishuai Cao; Yangchen Zeng; Yuhang Zhang; Xiaojing Du; Simon Fong
>
> **备注:** 26 pages, 7 figures
>
> **摘要:** Large pretrained language models and neural reasoning systems have advanced many natural language tasks, yet they remain challenged by knowledge-intensive queries that require precise, structured multi-hop inference. Knowledge graphs provide a compact symbolic substrate for factual grounding, but integrating graph structure with neural models is nontrivial: naively embedding graph facts into prompts leads to inefficiency and fragility, while purely symbolic or search-heavy approaches can be costly in retrievals and lack gradient-based refinement. We introduce NeuroSymActive, a modular framework that combines a differentiable neural-symbolic reasoning layer with an active, value-guided exploration controller for Knowledge Graph Question Answering. The method couples soft-unification style symbolic modules with a neural path evaluator and a Monte-Carlo style exploration policy that prioritizes high-value path expansions. Empirical results on standard KGQA benchmarks show that NeuroSymActive attains strong answer accuracy while reducing the number of expensive graph lookups and model calls compared to common retrieval-augmented baselines.
>
---
#### [replaced 022] ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决复杂排名场景下重排序器性能不足的问题。通过生成推理密集型训练数据并采用两阶段训练方法，提升重排序模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2508.07050](https://arxiv.org/pdf/2508.07050)**

> **作者:** Wenhan Liu; Xinyu Ma; Weiwei Sun; Yutao Zhu; Yuchen Li; Dawei Yin; Zhicheng Dou
>
> **备注:** 25 pages, accepted by ACL2026 main conference
>
> **摘要:** Large Language Model (LLM) based listwise ranking has shown superior performance in many passage ranking tasks. With the development of Large Reasoning Models (LRMs), many studies have demonstrated that step-by-step reasoning during test-time helps improve listwise ranking performance. However, due to the scarcity of reasoning-intensive training data, existing rerankers perform poorly in many complex ranking scenarios, and the ranking ability of reasoning-intensive rerankers remains largely underdeveloped. In this paper, we first propose an automated reasoning-intensive training data synthesis framework, which sources training queries and passages from diverse domains and applies DeepSeek-R1 to generate high-quality training labels. To empower the listwise reranker with strong reasoning ability, we further propose a two-stage training approach, which includes a cold-start supervised fine-tuning (SFT) stage and a reinforcement learning (RL) stage. During the RL stage, we design a novel multi-view ranking reward tailored to the multi-turn nature of listwise ranking. Extensive experiments demonstrate that our trained reasoning-intensive reranker \textbf{ReasonRank} outperforms existing baselines significantly and also achieves much lower latency than the pointwise reranker. Our codes are available at this https URL.
>
---
#### [replaced 023] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于个性化任务，旨在优化用户画像以提升LLM生成质量。针对传统方法依赖语义相关性的问题，提出PURPLE框架，通过上下文强化学习优化用户 profile，提升检索增强效果。**

- **链接: [https://arxiv.org/pdf/2601.12078](https://arxiv.org/pdf/2601.12078)**

> **作者:** Linfeng Du; Ye Yuan; Zichen Zhao; Fuyuan Lyu; Emiliano Penaloza; Xiuying Chen; Zipeng Sun; Jikun Kang; Laurent Charlin; Xue Liu; Haolun Wu
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Large language models (LLMs) excel at general-purpose tasks, yet adapting their responses to individual users remains challenging. Retrieval augmentation provides a lightweight alternative to fine-tuning by conditioning LLMs on user history records, and existing approaches typically select these records based on semantic relevance. We argue that relevance serves as an unreliable proxy for utility: a record may be semantically similar to a query yet fail to improve generation quality or even degrade it due to redundancy or conflicting information. To bridge this gap, we propose PURPLE, a contextual bandit framework that oPtimizes UseR Profiles for LLM pErsonalization. In contrast to a greedy selection of the most relevant records, PURPLE treats profile construction as an order-sensitive generation process and utilizes a Plackett-Luce ranking model to capture complex inter-record dependencies. By training with semantically rich feedback provided by the likelihood of the reference response, our method aligns retrieval directly with generation quality. Extensive experiments on nine personalization tasks demonstrate that PURPLE consistently outperforms strong heuristic and retrieval-augmented baselines in both effectiveness and efficiency, establishing a principled and scalable solution for optimizing user profiles.
>
---
#### [replaced 024] Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本处理中的计算与内存效率问题。提出K-Token Merging方法，在潜在嵌入空间中压缩序列，减少输入长度并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15153](https://arxiv.org/pdf/2604.15153)**

> **作者:** Zihao Xu; John Harvill; Ziwei Fan; Yizhou Sun; Hao Ding; Hao Wang
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation. Code is available at this https URL.
>
---
#### [replaced 025] Epistemic Constitutionalism Or: how to avoid coherence bias
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI伦理与认知研究任务，旨在解决语言模型中的信念形成偏误问题。通过提出“认识论宪法”框架，规范AI的信念生成与表达机制。**

- **链接: [https://arxiv.org/pdf/2601.14295](https://arxiv.org/pdf/2601.14295)**

> **作者:** Michele Loi
>
> **备注:** 27 pages, 7 tables. Data: this http URL and this http URL. Complete AI-assisted writing documentation: this http URL
>
> **摘要:** Large language models increasingly function as artificial reasoners: they evaluate arguments, assign credibility, and express confidence. Yet their belief-forming behavior is governed by implicit, uninspected epistemic policies. This paper argues for an epistemic constitution for AI: explicit, contestable meta-norms that regulate how systems form and express beliefs. Source attribution bias provides the motivating case: I show that frontier models enforce identity-stance coherence, penalizing arguments attributed to sources whose expected ideological position conflicts with the argument's content. When models detect systematic testing, these effects collapse, revealing that systems treat source-sensitivity as bias to suppress rather than as a capacity to execute well. I distinguish two constitutional approaches: the Platonic, which mandates formal correctness and default source-independence from a privileged standpoint, and the Liberal, which refuses such privilege, specifying procedural norms that protect conditions for collective inquiry while allowing principled source-attending grounded in epistemic vigilance. I argue for the Liberal approach, sketch a constitutional core of eight principles and four orientations, and propose that AI epistemic governance requires the same explicit, contestable structure we now expect for AI ethics.
>
---
#### [replaced 026] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言检索增强生成任务，旨在解决多语言场景下知识偏差和冲突问题。提出LcRL框架，通过语言耦合策略优化政策与奖励模型，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.14896](https://arxiv.org/pdf/2601.14896)**

> **作者:** Rui Qi; Fengran Mo; Yufeng Chen; Xue Zhang; Shuo Wang; Hongliang Li; Jinan Xu; Meng Jiang; Jian-Yun Nie; Kaiyu Huang
>
> **备注:** Accepted to ACL 2026 (Findings)
>
> **摘要:** Multilingual retrieval-augmented generation (MRAG) requires models to effectively acquire and integrate beneficial external knowledge from multilingual collections. However, most existing studies employ a unitive process where queries of equivalent semantics across different languages are processed through a single-turn retrieval and subsequent optimization. Such a ``one-size-fits-all'' strategy is often suboptimal in multilingual settings, as the models occur to knowledge bias and conflict during the interaction with the search engine. To alleviate the issues, we propose LcRL, a multilingual search-augmented reinforcement learning framework that integrates a language-coupled Group Relative Policy Optimization into the policy and reward models. We adopt the language-coupled group sampling in the rollout module to reduce knowledge bias, and regularize an auxiliary anti-consistency penalty in the reward models to mitigate the knowledge conflict. Experimental results demonstrate that LcRL not only achieves competitive performance but is also appropriate for various practical scenarios such as constrained training data and retrieval over collections encompassing a large number of languages. Our code is available at this https URL.
>
---
#### [replaced 027] RoLegalGEC: Legal Domain Grammatical Error Detection and Correction Dataset for Romanian
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出RoLegalGEC，首个罗马尼亚语法律领域语法错误检测与修正数据集，解决法律文本准确性问题，通过构建数据集并评估多种模型实现错误检测与修正。**

- **链接: [https://arxiv.org/pdf/2604.19593](https://arxiv.org/pdf/2604.19593)**

> **作者:** Mircea Timpuriu; Mihaela-Claudia Cercel; Dumitru-Clementin Cercel
>
> **摘要:** The importance of clear and correct text in legal documents cannot be understated, and, consequently, a grammatical error correction tool meant to assist a professional in the law must have the ability to understand the possible errors in the context of a legal environment, correcting them accordingly, and implicitly needs to be trained in the same environment, using realistic legal data. However, the manually annotated data required by such a process is in short supply for languages such as Romanian, much less for a niche domain. The most common approach is the synthetic generation of parallel data; however, it requires a structured understanding of the Romanian grammar. In this paper, we introduce, to our knowledge, the first Romanian-language parallel dataset for the detection and correction of grammatical errors in the legal domain, RoLegalGEC, which aggregates 350,000 examples of errors in legal passages, along with error annotations. Moreover, we evaluate several neural network models that transform the dataset into a valuable tool for both detecting and correcting grammatical errors, including knowledge-distillation Transformers, sequence tagging architectures for detection, and a variety of pre-trained text-to-text Transformer models for correction. We consider that the set of models, together with the novel RoLegalGEC dataset, will enrich the resource base for further research on Romanian.
>
---
#### [replaced 028] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究Post-Training Quantization（PTQ）中的知识能力差异，提出任务分层的量化扩展定律，解决量化对不同知识能力影响不均的问题。**

- **链接: [https://arxiv.org/pdf/2508.18609](https://arxiv.org/pdf/2508.18609)**

> **作者:** Chenxi Zhou; Pengfei Cao; Jiang Li; Bohan Yu; Jinyu Ye; Jun Zhao; Kang Liu
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Post-Training Quantization (PTQ) is a critical strategy for efficient Large Language Models (LLMs) deployment. However, existing scaling laws primarily focus on general performance, overlooking crucial fine-grained factors and how quantization differentially impacts diverse knowledge capabilities. To address this, we establish Task-Stratified Knowledge Scaling Laws. By stratifying capabilities into memorization, application, and reasoning, we develop a framework that unifies model size, bit-width, and fine-grained factors: group size and calibration set size. Validated on 293 diverse PTQ configurations, our framework demonstrates strong fit and cross-architecture consistency. It reveals distinct sensitivities across knowledge capabilities: reasoning is precision-critical, application is scale-responsive, and memorization is calibration-sensitive. We highlight that in low-bit scenarios, optimizing these fine-grained factors is essential for preventing performance collapse. These findings provide an empirically-backed foundation for designing knowledge-aware quantization strategies.
>
---
#### [replaced 029] Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics
- **分类: cs.CL**

- **简介: 该论文属于作者身份识别任务，旨在提升stylometric delta度量的可解释性。提出两种新指标，通过概率分布距离函数改进传统方法，增强结果可读性与验证性。**

- **链接: [https://arxiv.org/pdf/2604.19499](https://arxiv.org/pdf/2604.19499)**

> **作者:** Dmitry Pronin; Evgeny Kazartsev
>
> **备注:** Under review at Digital Scholarship in the Humanities. Code available at: this https URL
>
> **摘要:** This article introduces two new measures for authorship attribution - Rank-Turbulence Delta and Jensen-Shannon Delta - which generalise Burrows's classical Delta by applying distance functions designed for probabilistic distributions. We first set out the theoretical basis of the measures, contrasting centred and uncentred z-scoring of word-frequency vectors and re-casting the uncentred vectors as probability distributions. Building on this representation, we develop a token-level decomposition that renders every Delta distance numerically interpretable, thereby facilitating close reading and the validation of results. The effectiveness of the methods is assessed on four literary corpora in English, German, French and Russian. The English, German and French datasets are compiled from Project Gutenberg, whereas the Russian benchmark is the SOCIOLIT corpus containing 755 works by 180 authors spanning the eighteenth to the twenty-first centuries. Rank-Turbulence Delta attains attribution accuracy comparable with Cosine Delta; Jensen-Shannon Delta consistently matches or exceeds the performance of canonical Burrows's Delta. Finally, several established attribution algorithms are re-evaluated on the extended SOCIOLIT corpus.
>
---
#### [replaced 030] Agnostic Language Identification and Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言识别与生成任务，旨在解决在无实证假设下的问题。通过提出新目标，在更一般的“不可知”设置中获得新的理论结果。**

- **链接: [https://arxiv.org/pdf/2601.23258](https://arxiv.org/pdf/2601.23258)**

> **作者:** Mikael Møller Høgsgaard; Chirag Pabbaraju
>
> **备注:** typos and minor bug fixes
>
> **摘要:** Recent works on language identification and generation have established tight statistical rates at which these tasks can be achieved. These works typically operate under a strong realizability assumption: that the input data is drawn from an unknown distribution necessarily supported on some language in a given collection. In this work, we relax this assumption of realizability entirely, and impose no restrictions on the distribution of the input data. We propose objectives to study both language identification and generation in this more general "agnostic" setup. Across both problems, we obtain novel interesting characterizations and nearly tight rates.
>
---
#### [replaced 031] Foundational Design Principles and Patterns for Building Robust and Adaptive GenAI-Native Systems
- **分类: cs.SE; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于系统设计任务，旨在解决GenAI系统可靠性与适应性问题。提出设计原则与架构模式，以融合AI能力与传统工程方法，构建稳健高效系统。**

- **链接: [https://arxiv.org/pdf/2508.15411](https://arxiv.org/pdf/2508.15411)**

> **作者:** Frederik Vandeputte
>
> **摘要:** Generative AI (GenAI) has emerged as a transformative technology, demonstrating remarkable capabilities across diverse application domains. However, GenAI faces several major challenges in developing reliable and efficient GenAI-empowered systems due to its unpredictability and inefficiency. This paper advocates for a paradigm shift: future GenAI-native systems should integrate GenAI's cognitive capabilities with traditional software engineering principles to create robust, adaptive, and efficient systems. We introduce foundational GenAI-native design principles centered around five key pillars -- reliability, excellence, evolvability, self-reliance, and assurance -- and propose architectural patterns such as GenAI-native cells, organic substrates, and programmable routers to guide the creation of resilient and self-evolving systems. Additionally, we outline the key ingredients of a GenAI-native software stack and discuss the impact of these systems from technical, user adoption, economic, and legal perspectives, underscoring the need for further validation and experimentation. Our work aims to inspire future research and encourage relevant communities to implement and refine this conceptual framework.
>
---
#### [replaced 032] BenGER: A Collaborative Web Platform for End-to-End Benchmarking of German Legal Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BenGER平台，解决德国法律任务的端到端基准测试问题，整合任务设计、协作标注、模型评估等流程，提升透明度与可复现性。**

- **链接: [https://arxiv.org/pdf/2604.13583](https://arxiv.org/pdf/2604.13583)**

> **作者:** Sebastian Nagl; Matthias Grabmair
>
> **备注:** Preprint - Accepted at ICAIL 2026
>
> **摘要:** Evaluating large language models (LLMs) for legal reasoning requires workflows that span task design, expert annotation, model execution, and metric-based evaluation. In practice, these steps are split across platforms and scripts, limiting transparency, reproducibility, and participation by non-technical legal experts. We present the BenGER (Benchmark for German Law) framework, an open-source web platform that integrates task creation, collaborative annotation, configurable LLM runs, and evaluation with lexical, semantic, factual, and judge-based metrics. BenGER supports multi-organization projects with tenant isolation and role-based access control, and can optionally provide formative, reference-grounded feedback to annotators. We will demonstrate a live deployment showing end-to-end benchmark creation and analysis.
>
---
#### [replaced 033] Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于文本到SQL生成任务，旨在解决schema linking中遗漏关键元素的问题。提出KaSLA方法，通过背包优化减少冗余并提升链接准确性。**

- **链接: [https://arxiv.org/pdf/2502.12911](https://arxiv.org/pdf/2502.12911)**

> **作者:** Zheng Yuan; Hao Chen; Zijin Hong; Qinggang Zhang; Feiran Huang; Qing Li; Xiao Huang
>
> **摘要:** Generating SQLs from user queries is a long-standing challenge, where the accuracy of initial schema linking significantly impacts subsequent SQL generation performance. However, current schema linking models still struggle with missing relevant schema elements or an excess of redundant ones. A crucial reason for this is that commonly used metrics, recall and precision, fail to capture relevant element missing and thus cannot reflect actual schema linking performance. Motivated by this, we propose enhanced schema linking metrics by introducing a \textbf{restricted missing indicator}. Accordingly, we introduce \textbf{\underline{K}n\underline{a}psack optimization-based \underline{S}chema \underline{L}inking \underline{A}pproach (KaSLA)}, a plug-in schema linking method designed to prevent the missing of relevant schema elements while minimizing the inclusion of redundant ones. KaSLA employs a hierarchical linking strategy that first identifies the optimal table linking and subsequently links columns within the selected table to reduce linking candidate space. In each linking process, it utilizes a knapsack optimization approach to link potentially relevant elements while accounting for a limited tolerance of potentially redundant ones. With this optimization, KaSLA-1.6B achieves superior schema linking results compared to large-scale LLMs, including DeepSeek-V3 with the state-of-the-art (SOTA) schema linking method. Extensive experiments on Spider and BIRD benchmarks verify that KaSLA can significantly improve the SQL generation performance of SOTA Text2SQL models by substituting their schema linking processes. The code is available at this https URL.
>
---
#### [replaced 034] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于健康信息治理任务，旨在解决社区注释系统响应慢、准确性低的问题。通过引入LLM增强框架，提升 misinformation 治理的效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.11423](https://arxiv.org/pdf/2510.11423)**

> **作者:** Jiaying Wu; Zihang Fu; Haonan Wang; Fanxiao Li; Jiafeng Guo; Preslav Nakov; Min-Yen Kan
>
> **备注:** ACL 2026
>
> **摘要:** Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), allows users to flag misleading posts, attach contextual notes, and rate the notes' helpfulness. However, our empirical analysis of 30.8K health-related notes reveals substantial latency, with a median delay of 17.6 hours before notes receive a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified LLM-based framework that augments Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, supported by a hierarchical three-stage evaluation of relevance, correctness, and helpfulness. We instantiate the framework with HealthNotes, a benchmark of 1.2K health notes annotated for helpfulness, and a fine-tuned helpfulness judge. Our analysis first uncovers a key loophole in current crowd-sourced governance: voters frequently conflate stylistic fluency with factual accuracy. Addressing this via our hierarchical evaluation, experiments across 15 representative LLMs demonstrate that CrowdNotes+ significantly outperforms human contributors in note correctness, helpfulness, and evidence utility.
>
---
#### [replaced 035] Improving End-to-End Training of Retrieval-Augmented Generation Models via Joint Stochastic Approximation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决RAG模型端到端训练中的梯度估计问题，提出JSA-RAG方法，提升生成与检索效果。**

- **链接: [https://arxiv.org/pdf/2508.18168](https://arxiv.org/pdf/2508.18168)**

> **作者:** Hongyu Cao; Yuxuan Wu; Yucheng Cai; Xianyu Zhao; Zhijian Ou
>
> **摘要:** Retrieval-augmented generation (RAG) has become a widely recognized paradigm to combine parametric memory with non-parametric memories. An RAG model consists of two serial connecting components (retriever and generator). A major challenge in end-to-end optimization of the RAG model is that marginalization over relevant passages (modeled as discrete latent variables) from a knowledge base is required. Traditional top-K marginalization and variational RAG (VRAG) suffer from biased or high-variance gradient estimates. In this paper, we propose and develop joint stochastic approximation (JSA) based end-to-end training of RAG, which is referred to as JSA-RAG. The JSA algorithm is a stochastic extension of the EM (expectation-maximization) algorithm and is particularly powerful in estimating discrete latent variable models. Extensive experiments are conducted on five datasets for two tasks (open-domain question answering, knowledge-grounded dialogs) and show that JSA-RAG significantly outperforms both vanilla RAG and VRAG. Further analysis shows the efficacy of JSA-RAG from the perspectives of generation, retrieval, and low-variance gradient estimate.
>
---
#### [replaced 036] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决如何评估推理轨迹适合学生模型的问题。提出RSR指标，兼顾对齐与信息量，提升学生推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14249](https://arxiv.org/pdf/2601.14249)**

> **作者:** Yuming Yang; Mingyoung Lai; Wanxu Zhao; Xiaoran Fan; Zhiheng Xi; Mingqi Wu; Chiyue Huang; Jun Zhao; Haijun Lv; Jian Tong; Yunhua Zhou; Yicheng Zou; Qipeng Guo; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Accepted to ACL 2026 (Main Conference). 31 pages. Project page: this https URL
>
> **摘要:** Long chain-of-thought (CoT) trajectories provide rich supervision signals for distilling reasoning from teacher to student LLMs. However, both prior work and our experiments show that trajectories from stronger teachers do not necessarily yield better students, highlighting the importance of data-student suitability in distillation. Existing methods assess suitability primarily through student likelihood, favoring trajectories that align closely with the student model's current behavior but overlooking more informative ones. Addressing this, we propose Rank-Surprisal Ratio (RSR), a simple metric that captures both alignment and informativeness to assess the suitability of a reasoning trajectory. RSR is motivated by the observation that effective trajectories typically balance learning signal strength and behavioral alignment by combining low absolute probability with relatively high-ranked tokens under the student model. Concretely, RSR is defined as the ratio of a trajectory's average token-wise rank to its average negative log-likelihood, and is straightforward to compute and interpret. Across five student models and reasoning trajectories from 11 diverse teachers, RSR strongly correlates with post-training reasoning performance (average Spearman 0.86), consistently outperforming existing metrics. We further demonstrate its practical utility in both trajectory selection and teacher selection.
>
---
#### [replaced 037] RExBench: Can coding agents autonomously implement AI research extensions?
- **分类: cs.CL**

- **简介: 该论文属于AI研究扩展任务，旨在评估LLM代理自主实现研究扩展的能力。通过构建RExBench基准，发现当前代理成功率较低，需依赖人类指导。**

- **链接: [https://arxiv.org/pdf/2506.22598](https://arxiv.org/pdf/2506.22598)**

> **作者:** Nicholas Edwards; Yukyung Lee; Yujun Audrey Mao; Yulu Qin; Sebastian Schuster; Najoung Kim
>
> **备注:** ACL 2026
>
> **摘要:** Agents based on Large Language Models (LLMs) have shown promise for performing sophisticated software engineering tasks autonomously. In addition, there has been progress towards developing agents that can perform parts of the research pipeline in machine learning and the natural sciences. We argue that research extension and its implementation is a critical capability for such systems, and introduce RExBench to support the evaluation of this capability. RExBench is a benchmark consisting of realistic extensions of 12 research papers that aim to investigate novel research hypotheses. Each task is set up as an extension to an existing research paper and codebase, accompanied by domain expert-written instructions. RExBench is robust to data contamination and supports an automatic evaluation infrastructure that executes agent outputs to determine whether the success criteria are met. We use this benchmark to evaluate 12 LLM agents implemented using two different frameworks, aider and OpenHands. We find that all agents fail to autonomously implement the majority of the extensions, with the best agent achieving around a 33% success rate. Although the success rate improves with additional human-written hints, the best performance under this setting remains below 44%. This indicates that current agents are still short of being able to handle realistic research extension tasks without substantial human guidance.
>
---
#### [replaced 038] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出WISCA方法，解决LLM训练中权重模式优化不足的问题，通过权重缩放提升训练效率和模型质量。属于模型训练优化任务。**

- **链接: [https://arxiv.org/pdf/2508.16676](https://arxiv.org/pdf/2508.16676)**

> **作者:** Jiacheng Li; Jianchao Tan; Zhidong Yang; Pingwei Sun; Feiye Huo; Jiayu Qin; Xiangyu Zhang; Maoxin He; Yerui Sun; Yuchen Xie; Guangming Tan; Weile Jia; Xunliang Cai; Tong Zhao
>
> **备注:** Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures.
>
---
#### [replaced 039] Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文研究自适应多智能体系统（MAS）的泛化能力，旨在解决其在不同领域中表现不佳的问题。通过实证分析，发现存在拓扑过拟合和虚假协作现象，强调提升泛化能力的重要性。**

- **链接: [https://arxiv.org/pdf/2604.18951](https://arxiv.org/pdf/2604.18951)**

> **作者:** Namyoung So; Seokgyu Jang; Taeuk Kim
>
> **备注:** 27 pages, 4 figures. Equal contribution for the first two authors
>
> **摘要:** Adaptive multi-agent systems (MAS) are increasingly adopted to tackle complex problems. However, the narrow task coverage of their optimization raises the question of whether they can function as general-purpose systems. To address this gap, we conduct an extensive empirical study of adaptive MAS, revealing two key findings: (1) topological overfitting -- they fail to generalize across different domains; and (2) illusory coordination -- they achieve reasonable surface-level accuracy while the underlying agent interactions diverge from ideal MAS behavior, raising concerns about their practical utility. These findings highlight the pressing need to prioritize generalization in MAS development and motivate evaluation protocols that extend beyond simple final-answer correctness.
>
---
#### [replaced 040] Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters
- **分类: cs.CL**

- **简介: 该论文属于多语言检索任务，旨在提升小模型的检索性能。通过优化训练数据、负样本策略和任务多样性，开发出300M参数模型，达到7B模型效果。**

- **链接: [https://arxiv.org/pdf/2510.14274](https://arxiv.org/pdf/2510.14274)**

> **作者:** Lifu Tu; Yingbo Zhou; Semih Yavuz
>
> **备注:** minor update from previous version
>
> **摘要:** Training effective multilingual embedding models presents unique challenges due to the diversity of languages and task objectives. Although small multilingual models (<1 B parameters) perform well on multilingual tasks generally, they consistently lag behind larger models (>1 B) in the most prevalent use case: retrieval. This raises a critical question: Can smaller models be retrofitted specifically for retrieval tasks to enhance their performance? In this work, we investigate key factors that influence the effectiveness of multilingual embeddings, focusing on training data scale, negative sampling strategies, and data diversity. We find that while increasing the scale of training data yields initial performance gains, these improvements quickly plateau - indicating diminishing returns. Incorporating hard negatives proves essential for consistently improving retrieval accuracy. Furthermore, our analysis reveals that task diversity in the training data contributes more significantly to performance than language diversity alone. As a result, we develop a compact (approximately 300M) multilingual model that achieves retrieval performance comparable to or even surpassing current strong 7B models.
>
---
#### [replaced 041] Hybrid Decision Making via Conformal VLM-generated Guidance
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于人机协同决策任务，旨在解决现有方法生成的指导信息冗长难懂的问题。提出ConfGuide方法，通过置信区域控制生成简洁精准的指导。**

- **链接: [https://arxiv.org/pdf/2604.14980](https://arxiv.org/pdf/2604.14980)**

> **作者:** Debodeep Banerjee; Burcu Sayin; Stefano Teso; Andrea Passerini
>
> **摘要:** Building on recent advances in AI, hybrid decision making (HDM) holds the promise of improving human decision quality and reducing cognitive load. We work in the context of learning to guide (LtG), a recently proposed HDM framework in which the human is always responsible for the final decision: rather than suggesting decisions, in LtG the AI supplies (textual) guidance useful for facilitating decision making. One limiting factor of existing approaches is that their guidance compounds information about all possible outcomes, and as a result it can be difficult to digest. We address this issue by introducing ConfGuide, a novel LtG approach that generates more succinct and targeted guidance. To this end, it employs conformal risk control to select a set of outcomes, ensuring a cap on the false negative rate. We demonstrate our approach on a real-world multi-label medical diagnosis task. Our empirical evaluation highlights the promise of ConfGuide.
>
---
#### [replaced 042] Believing without Seeing: Quality Scores for Contextualizing Vision-Language Model Explanations
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于视觉语言模型解释质量评估任务，旨在解决盲人用户无法查看视觉上下文时对模型预测的误信问题。通过提出视觉保真度和对比性两个质量评分函数，提升解释的可靠性。**

- **链接: [https://arxiv.org/pdf/2509.25844](https://arxiv.org/pdf/2509.25844)**

> **作者:** Keyu He; Tejas Srinivasan; Brihi Joshi; Xiang Ren; Jesse Thomason; Swabha Swayamdipta
>
> **摘要:** When people query Vision-Language Models (VLMs) but cannot see the accompanying visual context (e.g. for blind and low-vision users), augmenting VLM predictions with natural language explanations can signal which model predictions are reliable. However, prior work has found that explanations can easily convince users that inaccurate VLM predictions are correct. To remedy undesirable overreliance on VLM predictions, we propose evaluating two complementary qualities of VLM-generated explanations via two quality scoring functions. We propose Visual Fidelity, which captures how faithful an explanation is to the visual context, and Contrastiveness, which captures how well the explanation identifies visual details that distinguish the model's prediction from plausible alternatives. On the A-OKVQA, VizWiz, and MMMU-Pro tasks, these quality scoring functions are better calibrated with model correctness than existing explanation qualities. We conduct a user study in which participants have to decide whether a VLM prediction is accurate without viewing its visual context. We observe that showing our quality scores alongside VLM explanations improves participants' accuracy at predicting VLM correctness by 11.1%, including a 15.4% reduction in the rate of falsely believing incorrect predictions. These findings highlight the utility of explanation quality scores in fostering appropriate reliance on VLM predictions.
>
---
#### [replaced 043] Rhetorical Questions in LLM Representations: A Linear Probing Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型对修辞疑问的表征，属于自然语言处理任务。通过线性探测分析，解决修辞疑问与信息提问的区分问题，发现其在不同数据集间具有可迁移性但无统一表征。**

- **链接: [https://arxiv.org/pdf/2604.14128](https://arxiv.org/pdf/2604.14128)**

> **作者:** Louie Hong Yao; Vishesh Anand; Yuan Zhuang; Tianyu Jiang
>
> **备注:** 18 pages, 15 figures, accepted to ACL 2026
>
> **摘要:** Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some probes capture discourse-level rhetorical stance embedded in extended argumentation, while others emphasize localized, syntax-driven interrogative acts. Together, these findings suggest that rhetorical questions in LLM representations are encoded by multiple linear directions emphasizing different cues, rather than a single shared direction.
>
---
#### [replaced 044] Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型训练中的计算与内存不匹配问题。通过PODS方法，选择性使用部分rollout进行策略优化，提升效率。**

- **链接: [https://arxiv.org/pdf/2504.13818](https://arxiv.org/pdf/2504.13818)**

> **作者:** Yixuan Even Xu; Yash Savani; Fei Fang; J. Zico Kolter
>
> **备注:** 19 pages, 10 figures, TMLR 2026
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as the leading approach for enhancing reasoning capabilities in large language models. However, it faces a fundamental compute and memory asymmetry: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce PODS (Policy Optimization with Down-Sampling), which decouples rollout generation from policy updates by training only on a strategically selected subset of rollouts, maintaining learning quality while dramatically reducing update costs. We propose a principled subset selection criterion, max-variance down-sampling, that maximizes reward diversity, and provide an efficient $O(n\log n)$ implementation. Empirically, Group Relative Policy Optimization (GRPO) with PODS achieves the peak test accuracy of vanilla GRPO at least $\mathbf{1.7\times}$ faster across the different reasoning benchmarks and hardware configurations we tested.
>
---
#### [replaced 045] TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出TREX系统，用于自动化大语言模型训练流程。解决复杂工作流自动化难题，通过多代理协作完成从需求分析到模型训练的全过程。**

- **链接: [https://arxiv.org/pdf/2604.14116](https://arxiv.org/pdf/2604.14116)**

> **作者:** Zerun Ma; Guoqiang Wang; Xinchen Xie; Yicheng Chen; He Du; Bowen Li; Yanan Sun; Wenran Liu; Kai Chen; Yining Li
>
> **摘要:** While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks.
>
---
#### [replaced 046] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AstaBench，用于评估AI代理在科学研究中的能力。解决现有基准不足的问题，通过构建全面的科学任务集和优化工具，提升评估的严谨性和实用性。**

- **链接: [https://arxiv.org/pdf/2510.21652](https://arxiv.org/pdf/2510.21652)**

> **作者:** Jonathan Bragg; Mike D'Arcy; Nishant Balepur; Dan Bareket; Bhavana Dalvi; Sergey Feldman; Dany Haddad; Jena D. Hwang; Peter Jansen; Varsha Kishore; Bodhisattwa Prasad Majumder; Aakanksha Naik; Sigal Rahamimov; Kyle Richardson; Amanpreet Singh; Harshit Surana; Aryeh Tiktinsky; Rosni Vasu; Guy Wiener; Chloe Anastasiades; Stefan Candra; Jason Dunkelberger; Dan Emery; Rob Evans; Malachi Hamada; Regan Huff; Rodney Kinney; Matt Latzke; Jaron Lochner; Ruben Lozano-Aguilera; Cecile Nguyen; Smita Rao; Amber Tanaka; Brooke Vlahos; Peter Clark; Doug Downey; Yoav Goldberg; Ashish Sabharwal; Daniel S. Weld
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they often (1) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (2) do not account for confounding variables such as model cost and tool access; (3) do not provide standardized interfaces for quick agent prototyping and evaluation; (4) fail to provide holistic, product-informed measures of real-world use cases such as science research; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides a holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance.
>
---
#### [replaced 047] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing
- **分类: cs.CL**

- **简介: 该论文属于大语言模型长文本推理任务，旨在解决注意力机制计算复杂度高和内存压力大的问题。通过异构KV缓存哈希技术，提升推理效率并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2604.19351](https://arxiv.org/pdf/2604.19351)**

> **作者:** Jinyu Guo; Zhihan Zhang; Yutong Li; Jiehui Xie; Md. Tamim Iqbal; Dongshen Han; Lik-Hang Lee; Sung-Ho Bae; Jie Zou; Yang Yang; Chaoning Zhang
>
> **备注:** Accepted by ACL 2026 (Findings)
>
> **摘要:** The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context inference. While existing KV cache compression methods alleviate memory pressure, they often sacrifice generation quality and fail to address the high overhead of floating-point arithmetic. This paper introduces DASH-KV, an innovative acceleration framework that reformulates attention as approximate nearest-neighbor search via asymmetric deep hashing. Under this paradigm, we design an asymmetric encoding architecture that differentially maps queries and keys to account for their distinctions in precision and reuse characteristics. To balance efficiency and accuracy, we further introduce a dynamic mixed-precision mechanism that adaptively retains full-precision computation for critical tokens. Extensive experiments on LongBench demonstrate that DASH-KV significantly outperforms state-of-the-art baseline methods while matching the performance of full attention, all while reducing inference complexity from O(N^2) to linear O(N). The code is available at this https URL
>
---
#### [replaced 048] Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有检测系统可靠性不足的问题。通过融合语言特征与可解释AI，提出新框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.23146](https://arxiv.org/pdf/2603.23146)**

> **作者:** Shushanta Pudasaini; Luis Miralles-Pechuán; David Lillis; Marisa Llorens Salvador
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) has made the detection of AI-Generated text a pressing and complex challenge. Although many detection systems report high benchmark accuracy, their reliability in real-world settings remains uncertain, and their interpretability is often unexplored. In this work, we investigate whether contemporary detectors genuinely identify machine authorship or merely exploit dataset-specific artefacts. We propose an interpretable detection framework that integrates linguistic feature engineering, machine learning, and explainable AI techniques. When evaluated on two prominent benchmark corpora, namely PAN CLEF 2025 and COLING 2025, our model trained on 30 linguistic features achieves leaderboard-competitive performance, attaining an F1 score of 0.9734. However, systematic cross-domain and cross-generator evaluation reveals substantial generalisation failure: classifiers that excel in-domain degrade significantly under distribution shift. Using SHAP- based explanations, we show that the most influential features differ markedly between datasets, indicating that detectors often rely on dataset-specific stylistic cues rather than stable signals of machine authorship. Further investigation with in-depth error analysis exposes a fundamental tension in linguistic-feature-based AI text detection: the features that are most discriminative on in-domain data are also the features most susceptible to domain shift, formatting variation, and text-length effects. We believe that this knowledge helps build AI detectors that are robust across different settings. To support replication and practical use, we release an open-source Python package that returns both predictions and instance-level explanations for individual texts.
>
---
#### [replaced 049] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness
- **分类: cs.CL**

- **简介: 该论文研究大语言模型是否具备判断答案正确性的私有知识。通过比较模型自身与外部模型的表示，发现事实类任务中模型自身表示更优，而数学推理无优势。**

- **链接: [https://arxiv.org/pdf/2604.12373](https://arxiv.org/pdf/2604.12373)**

> **作者:** Tomer Ashuach; Shai Gretz; Yoav Katz; Yonatan Belinkov; Liat Ein-Dor
>
> **备注:** Accepted to ACL 2026 (Main Conference). 8 pages, 16 figures, 2 tables
>
> **摘要:** Humans use introspection to evaluate their understanding through private internal states inaccessible to external observers. We investigate whether large language models possess similar privileged knowledge about answer correctness, information unavailable through external observation. We train correctness classifiers on question representations from both a model's own hidden states and external models, testing whether self-representations provide a performance advantage. On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes. We hypothesize this is due to high inter-model agreement of answer correctness. To isolate genuine privileged knowledge, we evaluate on disagreement subsets, where models produce conflicting predictions. Here, we discover domain-specific privileged knowledge: self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning. We further localize this domain asymmetry across model layers, finding that the factual advantage emerges progressively from early-to-mid layers onward, consistent with model-specific memory retrieval, while math reasoning shows no consistent advantage at any depth.
>
---
#### [replaced 050] Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLMs是否受医学文献中结果扭曲（spin）的影响。通过实验发现LLMs比人类更易受spin影响，但可通过提示减轻其影响。**

- **链接: [https://arxiv.org/pdf/2502.07963](https://arxiv.org/pdf/2502.07963)**

> **作者:** Hye Sun Yun; Karen Y.C. Zhang; Ramez Kouzy; Iain J. Marshall; Junyi Jessy Li; Byron C. Wallace
>
> **备注:** 26 pages, 17 figures, 4 tables, Conference on Health, Inference, and Learning (CHIL) 2025
>
> **摘要:** Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
>
---
#### [replaced 051] PLR: Plackett-Luce for Reordering In-Context Learning Examples
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决In-Context Learning中示例顺序优化问题。提出PLR方法，通过Plackett-Luce模型学习最优排序分布，提升少样本学习效果。**

- **链接: [https://arxiv.org/pdf/2603.21373](https://arxiv.org/pdf/2603.21373)**

> **作者:** Pawel Batorski; Paul Swoboda
>
> **摘要:** In-context learning (ICL) adapts large language models by conditioning on a small set of ICL examples, avoiding costly parameter updates. Among other factors, performance is often highly sensitive to the ordering of the examples. However, exhaustive search over the $n!$ possible orderings is infeasible. Therefore more efficient ordering methods use model confidence measures (e.g., label-probability entropy) over label sets or take a direct approach to finding the best ordering. We propose PLR, a probabilistic approach to in-context example ordering that replaces discrete ordering search with learning a probability distribution over orderings with the Plackett-Luce model. PLR models orderings using a Plackett-Luce distribution and iteratively updates its parameters to concentrate probability mass on high-performing orderings under a task-level metric. Candidate orderings are sampled efficiently via a Gumbel perturb-and-sort procedure. Experiments on multiple classification benchmarks show that PLR consistently improves few-shot accuracy for $k \in \{4, 8, 16, 32\}$ examples, and we further demonstrate gains on mathematical reasoning tasks where label-based ordering methods are not applicable. Our code is available at this https URL.
>
---
#### [replaced 052] "Newspaper Eat" Means "Not Tasty": A Taxonomy and Benchmark for Coded Language in Real-World Chinese Online Reviews
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究真实中文在线评论中的编码语言，构建了CodedLang数据集并提出七类分类体系，旨在提升NLP系统对编码语言的理解能力。**

- **链接: [https://arxiv.org/pdf/2601.19932](https://arxiv.org/pdf/2601.19932)**

> **作者:** Ruyuan Wan; Changye Li; Ting-Hao 'Kenneth' Huang
>
> **摘要:** Coded language is an important part of human communication. It refers to cases where users intentionally encode meaning so that the surface text differs from the intended meaning and must be decoded to be understood. Current language models handle coded language poorly. Progress has been limited by the lack of real-world datasets and clear taxonomies. This paper introduces CodedLang, a dataset of 7,744 Chinese Google Maps reviews, including 900 reviews with span-level annotations of coded language. We developed a seven-class taxonomy that captures common encoding strategies, including phonetic, orthographic, and cross-lingual substitutions. We benchmarked language models on coded language detection, classification, and review rating prediction. Results show that even strong models can fail to identify or understand coded language. Because many coded expressions rely on pronunciation-based strategies, we further conducted a phonetic analysis of coded and decoded forms. Our code and dataset are publicly available. Together, our results highlight coded language as an important and underexplored challenge for real-world NLP systems.
>
---
#### [replaced 053] Interpretability from the Ground Up: Stakeholder-Centric Design of Automated Scoring in Educational Assessments
- **分类: cs.CL**

- **简介: 该论文属于教育评估中的自动化评分任务，旨在解决评分系统缺乏可解释性的问题。提出FGTI四原则，并构建AnalyticScore框架提升评分透明度与准确性。**

- **链接: [https://arxiv.org/pdf/2511.17069](https://arxiv.org/pdf/2511.17069)**

> **作者:** Yunsung Kim; Mike Hardy; Joseph Tey; Candace Thille; Chris Piech
>
> **备注:** In Findings of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** AI-driven automated scoring systems offer scalable and efficient means of evaluating complex student-generated responses. Yet, despite increasing demand for transparency and interpretability, the field has yet to develop a widely accepted solution for interpretable automated scoring to be used in large-scale real-world assessments. This work takes a principled approach to address this challenge. We analyze the needs and potential benefits of interpretable automated scoring for various assessment stakeholder groups and develop four principles of interpretability -- (F)aithfulness, (G)roundedness, (T)raceability, and (I)nterchangeability (FGTI) -- targeted at those needs. To illustrate the feasibility of implementing these principles, we develop the AnalyticScore framework as a reference framework. When applied to the domain of text-based constructed-response scoring, AnalyticScore outperforms many uninterpretable scoring methods in terms of scoring accuracy and is, on average, within 0.06 QWK of the uninterpretable SOTA across 10 items from the ASAP-SAS dataset. By comparing against human annotators conducting the same featurization task, we further demonstrate that the featurization behavior of AnalyticScore aligns well with that of humans.
>
---
#### [replaced 054] Text to model via SysML: Automated generation of dynamical system computational models from unstructured natural language text via enhanced System Modeling Language diagrams
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于自然语言到模型的转换任务，旨在解决从文本自动生成动态系统计算模型的问题。通过SysML和NLP技术，实现系统组件与关系的提取与建模。**

- **链接: [https://arxiv.org/pdf/2507.06803](https://arxiv.org/pdf/2507.06803)**

> **作者:** Matthew Anderson Hendricks; Alice Cicirello
>
> **备注:** v3 - typos and imprecisions corrected, and added clarifications
>
> **摘要:** This paper contributes to speeding up the design and deployment of engineering dynamical systems by proposing a strategy for exploiting domain and expert knowledge for the automated generation of a dynamical system computational model starting from a corpus of documents relevant to the dynamical system of interest and an input document describing the specific system. This strategy is implemented in five steps and, crucially, it uses system modeling language diagrams (SysML) to extract accurate information about the dependencies, attributes, and operations of components. Natural Language Processing (NLP) strategies and Large Language Models (LLMs) are employed in specific tasks to improve intermediate outputs of the SySML diagrams automated generation, such as: list of key nouns; list of extracted relationships; list of key phrases and key relationships; block attribute values; block relationships; and BDD diagram generation. The applicability of automated SysML diagram generation is illustrated with different case studies. The computational models of complex dynamical systems from SysML diagrams are then obtained via code generation and computational model generation steps. In the code generation step, NLP strategies are used for summarization, while LLMs are used for validation only. The proposed approach is not limited to a specific system, domain, or computational software. Domain and expert knowledge is integrated by providing a set of equation implementation templates. This work represents one of the first attempts to build an automatic pipeline for this area. The applicability of the proposed approach is shown via an end-to-end example from text to model of a simple pendulum, showing improved performance compared to results yielded by LLMs only in zero-shot mode.
>
---
#### [replaced 055] Spotlights and Blindspots: Evaluating Machine-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器生成文本检测任务，旨在评估不同模型的检测效果。通过实验分析模型性能、数据集和指标的影响，揭示检测效果的差异与局限。**

- **链接: [https://arxiv.org/pdf/2604.16607](https://arxiv.org/pdf/2604.16607)**

> **作者:** Kevin Stowe; Kailash Patil
>
> **备注:** 15 pages, 4 figures, 4 tables
>
> **摘要:** With the rise of generative language models, machine-generated text detection has become a critical challenge. A wide variety of models is available, but inconsistent datasets, evaluation metrics, and assessment strategies obscure comparisons of model effectiveness. To address this, we evaluate 15 different detection models from six distinct systems, as well as seven trained models, across seven English-language textual test sets and three creative human-written datasets. We provide an empirical analysis of model performance, the influence of training and evaluation data, and the impact of key metrics. We find that no single system excels in all areas and nearly all are effective for certain tasks, and the representation of model performance is critically linked to dataset and metric choices. We find high variance in model ranks based on datasets and metrics, and overall poor performance on novel human-written texts in high-risk domains. Across datasets and metrics, we find that methodological choices that are often assumed or overlooked are essential for clearly and accurately reflecting model performance.
>
---
#### [replaced 056] What Language Models Know But Don't Say: Non-Generative Prior Extraction for Generalization
- **分类: cs.CL**

- **简介: 该论文属于贝叶斯推理任务，旨在解决小样本数据下模型泛化能力差的问题。通过提取语言模型的先验分布，提升逻辑回归在分布外数据上的性能。**

- **链接: [https://arxiv.org/pdf/2601.17609](https://arxiv.org/pdf/2601.17609)**

> **作者:** Sara Rezaeimanesh; Mohammad M. Ghassemi
>
> **摘要:** In domains like medicine and finance, large-scale labeled data is costly and often unavailable, leading to models trained on small datasets that struggle to generalize to real-world populations. Large language models contain extensive knowledge from years of research across these domains. We propose LoID (Logit-Informed Distributions), a deterministic method for extracting informative prior distributions for Bayesian logistic regression by directly accessing their token-level predictions. Rather than relying on generated text, we probe the model's confidence in opposing semantic directions (positive vs. negative impact) through carefully constructed sentences. By measuring how consistently the LLM favors one direction across diverse phrasings, we extract the strength and reliability of the model's belief about each feature's influence. We evaluate LoID on ten real-world tabular datasets under synthetic out-of-distribution (OOD) settings characterized by covariate shift, where the training data represents only a subset of the population. We compare our approach against (1) standard uninformative priors, (2) AutoElicit, a recent method that prompts LLMs to generate priors via text completions, (3) LLMProcesses, a method that uses LLMs to generate numerical predictions through in-context learning and (4) an oracle-style upper bound derived from fitting logistic regression on the full dataset. We assess performance using Area Under the Curve (AUC). Across datasets, LoID significantly improves performance over logistic regression trained on OOD data, recovering up to \textbf{59\%} of the performance gap relative to the oracle model. LoID outperforms AutoElicit and LLMProcessesc on 8 out of 10 datasets, while providing a reproducible and computationally efficient mechanism for integrating LLM knowledge into Bayesian inference.
>
---
#### [replaced 057] DRIV-EX: Counterfactual Explanations for Driving LLMs
- **分类: cs.CL**

- **简介: 该论文属于自动驾驶领域，旨在解决LLM决策不透明的问题。通过生成反事实解释，DRIV-EX方法优化嵌入以改变驾驶计划，同时保持文本流畅和语义合理。**

- **链接: [https://arxiv.org/pdf/2603.00696](https://arxiv.org/pdf/2603.00696)**

> **作者:** Amaia Cardiel; Eloi Zablocki; Elias Ramzi; Eric Gaussier
>
> **备注:** Accepted at ACL Findings 2026
>
> **摘要:** Large language models (LLMs) are increasingly used as reasoning engines in autonomous driving, yet their decision-making remains opaque. We propose to study their decision process through counterfactual explanations, which identify the minimal semantic changes to a scene description required to alter a driving plan. We introduce DRIV-EX, a method that leverages gradient-based optimization on continuous embeddings to identify the input shifts required to flip the model's decision. Crucially, to avoid the incoherent text typical of unconstrained continuous optimization, DRIV-EX uses these optimized embeddings solely as a semantic guide: they are used to bias a controlled decoding process that re-generates the original scene description. This approach effectively steers the generation toward the counterfactual target while guaranteeing the linguistic fluency, domain validity, and proximity to the original input, essential for interpretability. Evaluated using the LC-LLM planner on a textual transcription of the highD dataset, DRIV-EX generates valid, fluent counterfactuals more reliably than existing baselines. It successfully exposes latent biases and provides concrete insights to improve the robustness of LLM-based driving agents. The code is available at "this https URL .
>
---
#### [replaced 058] LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属于模型微调任务，解决PEFT方法在性能上不如全参数微调的问题。提出LoRA-FA，通过冻结部分参数并优化梯度，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2308.03303](https://arxiv.org/pdf/2308.03303)**

> **作者:** Longteng Zhang; Lin Zhang; Shaohuai Shi; Xiaowen Chu; Bo Li
>
> **摘要:** Fine-tuning large language models (LLMs) is crucial for improving their performance on downstream tasks, but full-parameter fine-tuning (Full-FT) is computationally expensive and memory-intensive. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address this by optimizing only a small subset of parameters. However, LoRA may underperform Full-FT in certain scenarios due to the intrinsic limitations of its low-rank gradients. In this work, we reveal an asymmetric, collapsible structure in LoRA's update: the low-rank modification to W can be reformulated as a single-layer linear regression, implying that one of the LoRA factors can be frozen without sacrificing expressivity. Leveraging this insight, we introduce LoRA-FA, which freezes the projection-down matrix A and trains only the projection-up matrix B. We further close the gap to Full-FT by deriving closed-form gradient corrections that minimize the discrepancy between the induced low-rank gradient and the full gradient. Through extensive experiments on diverse benchmarks, including GLUE, GSM8K, MT-Bench, and HumanEval, we demonstrate that LoRA-FA consistently achieves comparable performance to existing PEFT methods and Full-FT. Experiments on system efficiency show that LoRA-FA significantly reduces activation memory consumption and computational workload in fine-tuning.
>
---
#### [replaced 059] CRAFT: Training-Free Cascaded Retrieval for Tabular QA
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于表格问答任务，解决传统检索模型计算成本高、适应性差的问题。提出CRAFT方法，通过级联检索提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2505.14984](https://arxiv.org/pdf/2505.14984)**

> **作者:** Adarsh Singh; Kushal Raj Bhandari; Jianxi Gao; Soham Dan; Vivek Gupta
>
> **备注:** Accepted to ACL 2026 Mains
>
> **摘要:** Open-Domain Table Question Answering (TQA) involves retrieving relevant tables from a large corpus to answer natural language queries. Traditional dense retrieval models such as DTR and DPR incur high computational costs for large-scale retrieval tasks and require retraining or fine-tuning on new datasets, limiting their adaptability to evolving domains and knowledge. We propose CRAFT, a zero-shot cascaded retrieval approach that first uses a sparse retrieval model to filter a subset of candidate tables before applying more computationally expensive dense models as re-rankers. To improve retrieval quality, we enrich table representations with descriptive titles and summaries generated by Gemini Flash 1.5, enabling richer semantic matching between queries and tabular structures. Our method outperforms state-of-the-art sparse, dense, and hybrid retrievers on the NQ-Tables dataset. It also demonstrates strong zero-shot performance on the more challenging OTT-QA benchmark, achieving competitive results at higher recall thresholds, where the task requires multi-hop reasoning across both textual passages and relational tables. This work establishes a scalable and adaptable paradigm for table retrieval, bridging the gap between fine-tuned architectures and lightweight, plug-and-play retrieval systems. Code and data are available at this https URL
>
---
#### [replaced 060] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数据解析任务，解决LLM在文本分析中输出不稳定的问题。提出CAST框架，通过算法提示和思维先行提升稳定性。**

- **链接: [https://arxiv.org/pdf/2602.15861](https://arxiv.org/pdf/2602.15861)**

> **作者:** Jinxiang Xie; Zihao Li; Wei He; Rui Ding; Shi Han; Dongmei Zhang
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Text analysis of tabular data relies on two core operations: \emph{summarization} for corpus-level theme extraction and \emph{tagging} for row-level labeling. A critical limitation of employing large language models (LLMs) for these tasks is their inability to meet the high standards of output stability demanded by data analytics. To address this challenge, we introduce \textbf{CAST} (\textbf{C}onsistency via \textbf{A}lgorithmic Prompting and \textbf{S}table \textbf{T}hinking), a framework that enhances output stability by constraining the model's latent reasoning path. CAST combines (i) Algorithmic Prompting to impose a procedural scaffold over valid reasoning transitions and (ii) Thinking-before-Speaking to enforce explicit intermediate commitments before final generation. To measure progress, we introduce \textbf{CAST-S} and \textbf{CAST-T}, stability metrics for bulleted summarization and tagging, and validate their alignment with human judgments. Experiments across publicly available benchmarks on multiple LLM backbones show that CAST consistently achieves the best stability among all baselines, improving Stability Score by up to 16.2\%, while maintaining or improving output quality.
>
---
#### [replaced 061] Beyond Rating: A Comprehensive Evaluation and Benchmark for AI Reviews
- **分类: cs.CL**

- **简介: 该论文属于AI评论任务，旨在解决传统评分基准无法反映评论文本质量的问题。提出五维评估框架，强调文本论证的重要性。**

- **链接: [https://arxiv.org/pdf/2604.19502](https://arxiv.org/pdf/2604.19502)**

> **作者:** Bowen Li; Haochen Ma; Yuxin Wang; Jie Yang; Yining Zheng; Xinchi Chen; Xuanjing Huang; Xipeng Qiu
>
> **备注:** 38 pages,8 figures,4 tables
>
> **摘要:** The rapid adoption of Large Language Models (LLMs) has spurred interest in automated peer review; however, progress is currently stifled by benchmarks that treat reviewing primarily as a rating prediction task. We argue that the utility of a review lies in its textual justification--its arguments, questions, and critique--rather than a scalar score. To address this, we introduce Beyond Rating, a holistic evaluation framework that assesses AI reviewers across five dimensions: Content Faithfulness, Argumentative Alignment, Focus Consistency, Question Constructiveness, and AI-Likelihood. Notably, we propose a Max-Recall strategy to accommodate valid expert disagreement and introduce a curated dataset of paper with high-confidence reviews, rigorously filtered to remove procedural noise. Extensive experiments demonstrate that while traditional n-gram metrics fail to reflect human preferences, our proposed text-centric metrics--particularly the recall of weakness arguments--correlate strongly with rating accuracy. These findings establish that aligning AI critique focus with human experts is a prerequisite for reliable automated scoring, offering a robust standard for future research.
>
---
#### [replaced 062] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决LLM生成代码时文本表示与执行语义不一致的问题。提出CodeRL+，通过执行轨迹对齐增强强化学习效果，提升代码正确性。**

- **链接: [https://arxiv.org/pdf/2510.18471](https://arxiv.org/pdf/2510.18471)**

> **作者:** Xue Jiang; Yihong Dong; Mengyang Liu; Hongyi Deng; Tian Wang; Yongding Tao; Rongyu Cao; Binhua Li; Zhi Jin; Wenpin Jiao; Fei Huang; Yongbin Li; Ge Li
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** While Large Language Models (LLMs) excel at code generation by learning from vast code corpora, a fundamental semantic gap remains between their training on textual patterns and the goal of functional correctness, which is governed by formal execution semantics. Reinforcement Learning with Verifiable Rewards (RLVR) approaches attempt to bridge this gap using outcome rewards from executing test cases. However, solely relying on binary pass/fail signals is inefficient for establishing a well-aligned connection between the textual representation of code and its execution semantics, especially for subtle logical errors within the code. In this paper, we propose CodeRL+, a novel approach that integrates execution semantics alignment into the RLVR training pipeline for code generation. CodeRL+ enables the model to infer variable-level execution trajectory, providing a direct learning signal of execution semantics. CodeRL+ can construct execution semantics alignment directly using existing on-policy rollouts and integrates seamlessly with various RL algorithms. Extensive experiments demonstrate that CodeRL+ outperforms post-training baselines (including RLVR and Distillation), achieving a 4.6% average relative improvement in pass@1. CodeRL+ generalizes effectively to other coding tasks, yielding 15.5% and 4.4% higher accuracy on code-reasoning and test-output-generation benchmarks, respectively. CodeRL+ shows strong applicability across diverse RL algorithms and LLMs. Furthermore, probe analyses provide compelling evidence that CodeRL+ strengthens the alignment between code's textual representations and its underlying execution semantics.
>
---
#### [replaced 063] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents
- **分类: cs.CL**

- **简介: 该论文属于长对话系统任务，解决记忆系统检索效率与精度问题。提出HiGMem，通过分层结构和LLM引导，提升证据检索的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.18349](https://arxiv.org/pdf/2604.18349)**

> **作者:** Shuqi Cao; Jingyi He; Fei Tan
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version. 10 pages, 2 figures. Code: this https URL
>
> **摘要:** Long-term conversational large language model (LLM) agents require memory systems that can recover relevant evidence from historical interactions without overwhelming the answer stage with irrelevant context. However, existing memory systems, including hierarchical ones, still often rely solely on vector similarity for retrieval. It tends to produce bloated evidence sets: adding many superficially similar dialogue turns yields little additional recall, but lowers retrieval precision, increases answer-stage context cost, and makes retrieved memories harder to inspect and manage. To address this, we propose HiGMem (Hierarchical and LLM-Guided Memory System), a two-level event-turn memory system that allows LLMs to use event summaries as semantic anchors to predict which related turns are worth reading. This allows the model to inspect high-level event summaries first and then focus on a smaller set of potentially useful turns, providing a concise and reliable evidence set through reasoning, while avoiding the retrieval overhead that would be excessively high compared to vector retrieval. On the LoCoMo10 benchmark, HiGMem achieves the best F1 on four of five question categories and improves adversarial F1 from 0.54 to 0.78 over A-Mem, while retrieving an order of magnitude fewer turns. Code is publicly available at this https URL.
>
---
#### [replaced 064] CoSearch: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文研究agentic search任务，解决检索系统与推理代理独立训练导致性能瓶颈的问题。提出CoSearch框架，联合训练推理代理和生成式文档排序模型，提升搜索性能。**

- **链接: [https://arxiv.org/pdf/2604.17555](https://arxiv.org/pdf/2604.17555)**

> **作者:** Hansi Zeng; Liam Collins; Bhuvesh Kumar; Neil Shah; Hamed Zamani
>
> **摘要:** Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents.
>
---
#### [replaced 065] The Imperfective Paradox in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，探讨LLMs是否真正理解事件的语义结构。研究解决LLM在处理进行时与完成时 entailment 时的偏差问题，通过构建数据集和分析模型表现，揭示其依赖目标导向的先验而非逻辑推理。**

- **链接: [https://arxiv.org/pdf/2601.09373](https://arxiv.org/pdf/2601.09373)**

> **作者:** Bolei Ma; Yusuke Miyao
>
> **备注:** ACL 2026
>
> **摘要:** Do Large Language Models (LLMs) genuinely grasp the compositional semantics of events, or do they rely on surface-level probabilistic heuristics? We investigate the Imperfective Paradox, a logical phenomenon where the past progressive aspect entails event realization for activities (e.g., running $\to$ ran) but not for accomplishments (e.g., building $\nrightarrow$ built). We introduce ImperfectiveNLI, a diagnostic dataset designed to probe this distinction across diverse semantic classes. Evaluating state-of-the-art open-weight models, we uncover a pervasive Teleological Bias: models systematically hallucinate completion for goal-oriented events, even overriding explicit textual cancellation. Prompting interventions partially reduce this bias but trigger a calibration crisis, causing models to incorrectly reject valid entailments for atelic verbs. Representational analyses further show that while internal embeddings often distinguish progressive from simple past forms, inference decisions are dominated by strong priors about goal attainment. Taken together, our findings indicate that these current open-weight LLMs operate as predictive narrative engines rather than faithful logical reasoners, and that resolving aspectual inference requires moving beyond prompting toward structurally grounded alignment.
>
---
#### [replaced 066] Breaking the Assistant Mold: Modeling Behavioral Variation in LLM Based Procedural Character Generation
- **分类: cs.CL**

- **简介: 该论文属于角色生成任务，旨在解决LLM生成角色过于一致的问题。通过分离世界观与行为特征，提升角色多样性与戏剧性。**

- **链接: [https://arxiv.org/pdf/2601.03396](https://arxiv.org/pdf/2601.03396)**

> **作者:** Maan Qraitem; Kate Saenko; Bryan A. Plummer
>
> **摘要:** Procedural content generation has enabled vast virtual worlds through levels, maps, and quests, but large-scale character generation remains underexplored. We identify two alignment-induced biases in existing methods: a positive moral bias, where characters uniformly adopt agreeable stances (e.g. always saying lying is bad), and a helpful assistant bias, where characters invariably answer questions directly (e.g. never refusing or deflecting). While such tendencies suit instruction-following systems, they suppress dramatic tension and yield predictable characters, stemming from maximum likelihood training and assistant fine-tuning. To address this, we introduce PersonaWeaver, a framework that disentangles world-building (roles, demographics) from behavioral-building (moral stances, interactional styles), yielding characters with more diverse reactions and moral stances, as well as second-order diversity in stylistic markers like length, tone, and punctuation. Code: this https URL
>
---
#### [replaced 067] Neural Bandit Based Optimal LLM Selection for a Pipeline of Subtasks
- **分类: cs.CL; cs.LG**

- **简介: 论文研究在任务流水线中选择最优LLM的问题，解决多步骤任务中LLM序列选择的复杂依赖关系。提出基于神经上下文扳机的算法，无需历史数据，实现高效选择。**

- **链接: [https://arxiv.org/pdf/2508.09958](https://arxiv.org/pdf/2508.09958)**

> **作者:** Baran Atalar; Eddie Zhang; Carlee Joe-Wong
>
> **摘要:** As large language models (LLMs) become increasingly popular, there is a growing need to predict which out of a set of LLMs will yield a successful answer to a given query at low cost. This problem promises to become even more relevant as LLM agents are asked to solve an increasing variety of "agentic'' AI tasks. Such tasks are often broken into smaller subtasks, each of which can then be executed by a LLM expected to perform well on that specific subtask. For example, to extract a diagnosis from medical records, one can first select an LLM to summarize the record, select another to validate the summary, and then select a possibly different LLM to extract the diagnosis from the summarized record. Unlike existing LLM selection or routing algorithms, this setting requires selecting a sequence of LLMs, with the output of each LLM feeding into the next and potentially influencing its success. Thus, unlike single LLM selection, the quality of each subtask's output directly affects the inputs, and hence the cost and success rate, of downstream LLMs, creating complex performance dependencies that must be learned during selection. We propose a neural contextual bandit-based algorithm that trains neural networks to guide LLM selections for the different subtasks, without requiring historical LLM performance data. We prove that our proposed Sequential Bandits algorithm achieves a sublinear regret in the number of tasks, and we experimentally validate its superior performance compared to other LLM selection algorithms on two real datasets.
>
---
#### [replaced 068] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在计数任务中的机制，解决其因架构限制导致的计数精度下降问题。提出一种受System-2启发的分解策略，提升模型在大规模计数任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.02989](https://arxiv.org/pdf/2601.02989)**

> **作者:** Hosein Hasani; Mohammadali Banayeeanzade; Ali Nafisi; Sadegh Mohammadian; Fatemeh Askari; Mobin Bagherian; Amirmohammad Izadi; Mahdieh Soleymani Baghshah
>
> **备注:** ACL 2026
>
> **摘要:** Large language models (LLMs), despite strong performance on complex mathematical problems, exhibit systematic limitations in counting tasks. This issue arises from the architectural limits of transformers, where counting is performed across layers, leading to degraded precision for larger counting problems due to depth constraints. To address this limitation, we propose a simple test-time strategy inspired by System-2 cognitive processes that decomposes large counting tasks into smaller, independent sub-problems that the model can reliably solve. We evaluate this approach using observational and causal mediation analyses to understand the underlying mechanism of this System-2-like strategy. Our mechanistic analysis identifies key components: latent counts are computed and stored in the final item representations of each part, transferred to intermediate steps via dedicated attention heads, and aggregated in the final stage to produce the total count. Experimental results demonstrate that this strategy enables LLMs to surpass architectural limitations and achieve higher accuracy on large-scale counting tasks. This work provides mechanistic insight into System-2 counting in LLMs and presents a generalizable approach for improving and understanding their reasoning behavior.
>
---
#### [replaced 069] Alignment midtraining for animals
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究动物共情的价值对齐问题，通过合成文档微调提升模型的共情推理能力，提出AHB基准评估效果。任务为价值对齐，解决模型伦理行为优化问题，验证了文档微调的有效性。**

- **链接: [https://arxiv.org/pdf/2604.13076](https://arxiv.org/pdf/2604.13076)**

> **作者:** Jasmine Brazilek; Miles Tidmarsh
>
> **备注:** 34 pages
>
> **摘要:** We investigate the robustness of value alignment via finetuning with synthetic documents, using animal compassion as a value that is both important in its own right and orthogonal to existing alignment efforts. To evaluate compassionate reasoning, we develop and publicly release the Animal Harm Benchmark (AHB), a 26-question evaluation spanning 13 ethical dimensions, publicly available as a dataset and Inspect evaluation. On the AHB, training with 3000 documents achieves 77% compared to 40% for instruction-tuning approaches, with generalization to human compassion and no degradation in standard safety benchmarks or capabilities. However, subsequent unrelated instruction-tuning degrades the intervention, with the advantage disappearing after 5000 samples. Our exploratory results suggest document-based value interventions may require explicit preservation strategies to remain effective through typical training pipelines.
>
---
#### [replaced 070] Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Cognitive Kernel-Pro框架，解决AI代理开发受限问题，通过开放数据和策略提升代理性能，属于AI代理训练任务。**

- **链接: [https://arxiv.org/pdf/2508.00414](https://arxiv.org/pdf/2508.00414)**

> **作者:** Tianqing Fang; Zhisong Zhang; Xiaoyang Wang; Rui Wang; Can Qin; Yuxuan Wan; Jun-Yu Ma; Ce Zhang; Jiaqi Chen; Xiyun Li; Yonglin Wang; Jingchen Ni; Tianshi Zheng; Chun Chen; Wenhao Yu; Zhenwen Liang; Hongming Zhang; Haitao Mi; Dong Yu
>
> **备注:** 21 pages
>
> **摘要:** General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at this https URL
>
---
#### [replaced 071] Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents
- **分类: cs.CL**

- **简介: 该论文属于工具调用代理研究，解决真实场景中用户意图复杂多变的问题。通过生成可验证数据，提升代理的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.20144](https://arxiv.org/pdf/2601.20144)**

> **作者:** Ziyi Wang; Yuxuan Lu; Yimeng Zhang; Pei Chen; Ziwei Dong; Jing Huang; Jiri Gesi; Xianfeng Tang; Chen Luo; Qun Liu; Yisi Sang; Hanqing Lu; Manling Li; Jin Lai; Dakuo Wang
>
> **摘要:** Tool-calling agents are increasingly deployed in real-world customer-facing workflows. Yet most studies on tool-calling agents focus on idealized settings with general, fixed, and well-specified tasks. In real-world applications, user requests are often (1) ambiguous, (2) changing over time, or (3) infeasible due to policy constraints, and training and evaluation data that cover these diverse, complex interaction patterns remain under-represented. To bridge the gap, we present Trajectory2Task, a verifiable data generation pipeline for studying tool use at scale under three realistic user scenarios: ambiguous intent, changing intent, and infeasible intents. The pipeline first conducts multi-turn exploration to produce valid tool-call trajectories. It then converts these trajectories into user-facing tasks with controlled intent adaptations. This process yields verifiable task that support closed-loop evaluation and training. We benchmark seven state-of-the-art LLMs on the generated complex user scenario tasks and observe frequent failures. Finally, using successful trajectories obtained from task rollouts, we fine-tune lightweight LLMs and find consistent improvements across all three conditions, along with better generalization to unseen tool-use domains, indicating stronger tool-calling ability.
>
---
#### [replaced 072] Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLM评估中的隐含测量误差问题，通过分解不确定性源、优化设计来提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.11581](https://arxiv.org/pdf/2604.11581)**

> **作者:** Solomon Messing
>
> **摘要:** LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet standard confidence intervals ignore variability from prompt phrasing, model temperature, and judge model choice. The omitted variance produces under-coverage that worsens with more data and can shift results enough to reverse conclusions. The same unmeasured variance opens benchmarks to exploitation. Model developers can optimize against measurement noise instead of genuine performance, as \citet{singh2025leaderboard} document. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total error. We show a small-sample pilot is sufficient to derive confidence intervals that approach nominal coverage and to identify which design changes yield the largest precision gains. Applying the approach to ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit reveals different dominant variance sources by domain and scoring method. What's more, we show optimized budget allocation halves estimation error at equivalent cost (MMLU), and on our propaganda audit, the recommended pipeline outperforms 73\% of single-configuration alternatives against a human baseline.
>
---
#### [replaced 073] Memorization, Emergence, and Explaining Reversal Failures: A Controlled Study of Relational Semantics in LLMs
- **分类: cs.CL**

- **简介: 该论文研究LLMs在关系语义上的记忆与推理能力，解决其反转失败原因问题。通过构建合成数据集训练模型，分析逻辑推理与泛化能力，发现反转失败主要由顺序偏差引起。**

- **链接: [https://arxiv.org/pdf/2601.02931](https://arxiv.org/pdf/2601.02931)**

> **作者:** Yihua Zhu; Qianying Liu; Jiaxin Wang; Fei Cheng; Chaoran Liu; Akiko Aizawa; Sadao Kurohashi; Hidetoshi Shimodaira
>
> **备注:** ACL2026 Main Long Paper
>
> **摘要:** Autoregressive LLMs perform well on relational tasks that require linking entities via relational words (e.g., father/son, friend), but it is unclear whether they learn the logical semantics of such relations (e.g., symmetry and inversion logic) and, if so, whether reversal-type failures arise from missing relational semantics or left-to-right order bias. We propose a controlled Knowledge Graph-based synthetic framework that generates text from symmetric/inverse triples, train GPT-style autoregressive models from scratch, and evaluate memorization, logical inference, and in-context generalization to unseen entities to address these questions. We find a sharp phase transition in which relational semantics emerge with sufficient logic-bearing supervision, even in shallow (2-3 layer) models, and that successful generalization aligns with stable intermediate-layer signals. Finally, order-matched forward/reverse tests and a diffusion baseline indicate that reversal failures are primarily driven by autoregressive order bias rather than deficient inversion semantics.
>
---
#### [replaced 074] Talking to a Know-It-All GPT or a Second-Guesser Claude? How Repair reveals unreliable Multi-Turn Behavior in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM在多轮对话中的修复行为，探讨其可靠性问题。通过实验分析模型在可解与不可解数学问题中的互动表现，揭示不同模型在修复过程中的差异性与不可预测性。**

- **链接: [https://arxiv.org/pdf/2604.19245](https://arxiv.org/pdf/2604.19245)**

> **作者:** Clara Lachenmaier; Hannah Bultmann; Sina Zarrieß
>
> **备注:** Preprint accepted at ACL Main Conference 2026
>
> **摘要:** Repair, an important resource for resolving trouble in human-human conversation, remains underexplored in human-LLM interaction. In this study, we investigate how LLMs engage in the interactive process of repair in multi-turn dialogues around solvable and unsolvable math questions. We examine whether models initiate repair themselves and how they respond to user-initiated repair. Our results show strong differences across models: reactions range from being almost completely resistant to (appropriate) repair attempts to being highly susceptible and easily manipulated. We further demonstrate that once conversations extend beyond a single turn, model behavior becomes more distinctive and less predictable across systems. Overall, our findings indicate that each tested LLM exhibits its own characteristic form of unreliability in the context of repair.
>
---
#### [replaced 075] KOCO-BENCH: Can Large Language Models Leverage Domain Knowledge in Software Development?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于软件开发领域，旨在解决LLMs在特定领域知识应用上的不足。提出KOCO-BENCH基准，评估领域知识的获取与应用效果。**

- **链接: [https://arxiv.org/pdf/2601.13240](https://arxiv.org/pdf/2601.13240)**

> **作者:** Xue Jiang; Ge Li; Jiaru Qian; Xianjie Shi; Chenjie Li; Hao Zhu; Ziyu Wang; Jielun Zhang; Zheyu Zhao; Kechi Zhang; Jia Li; Wenpin Jiao; Zhi Jin; Yihong Dong
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) excel at general programming but struggle with domain-specific software development, necessitating domain specialization methods for LLMs to learn and utilize domain knowledge and data. However, existing domain-specific code benchmarks cannot evaluate the effectiveness of domain specialization methods, which focus on assessing what knowledge LLMs possess rather than how they acquire and apply new knowledge, lacking explicit knowledge corpora for developing domain specialization methods. To this end, we present KOCO-BENCH, a novel benchmark designed for evaluating domain specialization methods in real-world software development. KOCO-BENCH contains 6 emerging domains with 11 software frameworks and 25 projects, featuring curated knowledge corpora alongside multi-granularity evaluation tasks including domain code generation (from function-level to project-level with rigorous test suites) and domain knowledge understanding (via multiple-choice Q&A). Unlike previous benchmarks that only provide test sets for direct evaluation, KOCO-BENCH requires acquiring and applying diverse domain knowledge (APIs, rules, constraints, etc.) from knowledge corpora to solve evaluation tasks. Our evaluations reveal that KOCO-BENCH poses significant challenges to state-of-the-art LLMs. Even with domain specialization methods (e.g., SFT, RAG, kNN-LM) applied, improvements remain marginal. Best-performing coding agent, Claude Code, achieves only 34.2%, highlighting the urgent need for more effective domain specialization methods. We release KOCO-BENCH, evaluation code, and baselines to advance further research at this https URL.
>
---
#### [replaced 076] PersonalHomeBench: Evaluating Agents in Personalized Smart Homes
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文提出PersonalHomeBench，用于评估智能家庭中代理AI的性能。解决个性化环境中代理系统能力不足的问题，通过构建复杂任务和工具集进行测试。**

- **链接: [https://arxiv.org/pdf/2604.16813](https://arxiv.org/pdf/2604.16813)**

> **作者:** Nikhil Verma; InJung Yang; Sungil Kim; KoKeun Kim; YoungJoon Kim; Manasa Bharadwaj; Yolanda Liu; Kevin Ferreira
>
> **备注:** In light of concerns regarding authorship order, contributions, and affiliations in the current arXiv submission, I request to withdraw the manuscript temporarily to enable proper alignment among all contributors
>
> **摘要:** Agentic AI systems are rapidly advancing toward real-world applications, yet their readiness in complex and personalized environments remains insufficiently characterized. To address this gap, we introduce PersonalHomeBench, a benchmark for evaluating foundation models as agentic assistants in personalized smart home environments. The benchmark is constructed through an iterative process that progressively builds rich household states, which are then used to generate personalized, context-dependent tasks. To support realistic agent-environment interaction, we provide PersonalHomeTools, a comprehensive toolbox enabling household information retrieval, appliance control, and situational understanding. PersonalHomeBench evaluates both reactive and proactive agentic abilities under unimodal and multimodal observations. Thorough experimentation reveals a systematic performance reduction as task complexity increases, with pronounced failures in counterfactual reasoning and under partial observability, where effective tool-based information gathering is required. These results position PersonalHomeBench as a rigorous evaluation platform for analyzing the robustness and limitations of personalized agentic reasoning and planning.
>
---
#### [replaced 077] The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在表面线索与隐含约束冲突时的推理缺陷，属于自然语言处理任务。通过分析和实验，揭示了模型对约束推理的不足，并提出基准测试以评估改进效果。**

- **链接: [https://arxiv.org/pdf/2603.29025](https://arxiv.org/pdf/2603.29025)**

> **作者:** Yubo Li; Lu Zhang; Tianchong Jiang; Ramayya Krishnan; Rema Padman
>
> **摘要:** Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. We study this through a diagnose-measure-bridge-treat framework. Causal-behavioral analysis of the ``car wash problem'' across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7 to 38 times more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB) -- 500 instances spanning 4 heuristic by 5 constraint families with minimal pairs and explicitness gradients -- demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint (e.g., emphasizing the key object) recovers +15 pp on average, suggesting the failure lies in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to -39 pp), revealing conservative bias. Parametric probes confirm that the sigmoid pattern generalizes to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting recovers +6 to 9 pp by forcing models to enumerate preconditions before answering. Together, these results characterize heuristic override as a systematic reasoning vulnerability and provide a benchmark for measuring progress toward resolving it.
>
---
#### [replaced 078] From Noise to Signal to Selbstzweck: Reframing Human Label Variation in the Era of Post-training in NLP
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理领域，探讨如何在后训练阶段保留人类标注的多样性。解决的问题是现有数据集忽略人类标注差异，工作是分析现有数据集缺陷并提出保留多样性的策略。**

- **链接: [https://arxiv.org/pdf/2510.12817](https://arxiv.org/pdf/2510.12817)**

> **作者:** Shanshan Xu; Santosh T.Y.S.S; Barbara Plank
>
> **摘要:** Human Label Variation (HLV) refers to legitimate disagreement in annotation that reflects the diversity of human perspectives rather than mere error. Long treated in NLP as noise to be eliminated, HLV has only recently been reframed as a signal for improving model robustness. With the rise of large language models (LLMs) and post-training methods such as human feedback-based alignment, the role of HLV has become increasingly consequential. Yet current preference-learning datasets routinely collapse multiple annotations into a single label, flattening diverse perspectives into artificial consensus. Preserving HLV is necessary not only for pluralistic alignment but also for sociotechnical safety evaluation, where model behavior must be assessed in relation to human interaction and societal context. This position paper argues that preserving HLV as an embodiment of human pluralism must be treated as a Selbstzweck, an intrinsic value in itself. We analyze the limitations of existing preference datasets and propose actionable strategies for incorporating HLV into dataset construction to better preserve pluralistic human values.
>
---
#### [replaced 079] Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多步推理中步骤验证的问题。提出通过探测大模型内部状态，实现轻量级推理验证，提升测试时扩展效果。**

- **链接: [https://arxiv.org/pdf/2511.06209](https://arxiv.org/pdf/2511.06209)**

> **作者:** Jingwei Ni; Ekaterina Fadeeva; Tianyi Wu; Mubashara Akhtar; Jiaheng Zhang; Elliott Ash; Markus Leippold; Timothy Baldwin; See-Kiong Ng; Artem Shelmanov; Mrinmaya Sachan
>
> **备注:** ACL 2026 Main
>
> **摘要:** LLMs can solve complex tasks by generating long, multi-step reasoning chains. Test-time scaling (TTS) can further improve LLM performance by sampling multiple variants of intermediate reasoning steps, verifying their correctness, and strategically choosing the best steps for continuation. However, existing verification approaches, such as Process Reward Models (PRMs), are computationally expensive, limited to specific domains, and require large-scale human or model-generated annotations. We propose a lightweight alternative for step-level reasoning verification based on probing the internal states of LLMs. We train a transformer-based probe that uses the internal states of the frozen LLM to estimate the credibility of its reasoning steps during generation. Annotation can be generated either by another larger LLM (e.g., DeepSeek-R1) or in a self-supervised manner by the original model itself. The probes are both effective and lightweight, containing fewer than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, our probes match or even exceed the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their confidence in reasoning processes and can serve as reliable signals for reasoning step verification, offering a promising direction towards scalable and generalizable TTS and introspective LLMs.
>
---
#### [replaced 080] Navigating the Conceptual Multiverse
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文提出概念多宇宙系统，解决语言模型输出不透明问题，通过交互式框架让用户检查和调整决策过程。任务属于自然语言处理与人机交互。**

- **链接: [https://arxiv.org/pdf/2604.17815](https://arxiv.org/pdf/2604.17815)**

> **作者:** Andre Ye; Jenny Y. Huang; Alicia Guo; Rose Novick; Tamara Broderick; Mitchell L. Gordon
>
> **摘要:** When language models answer open-ended problems, they implicitly make hidden decisions that shape their outputs, leaving users with uncontextualized answers rather than a working map of the problem; drawing on multiverse analysis from statistics, we build and evaluate the conceptual multiverse, an interactive system that represents conceptual decisions such as how to frame a question or what to value as a space users can transparently inspect, intervenably change, and check against principled domain reasoning; for this structure to be worth navigating rather than misleading, it must be rigorous and checkable against domain reasoning norms, so we develop a general verification framework that enforces properties of good decision structures like unambiguity and completeness calibrated by expert-level reasoning; across three domains, the conceptual multiverse helped participants develop a working map of the problem, with philosophy students rewriting essays with sharper framings and reversed theses, alignment annotators moving from surface preferences to reasoning about user intent and harm, and poets identifying compositional patterns that clarified their taste.
>
---
#### [replaced 081] LLAMADRS: Evaluating Open-Source LLMs on Real Clinical Interviews--To Reason or Not to Reason?
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于临床自然语言处理任务，旨在评估开源大模型在真实临床对话中的表现。通过构建基准数据集，对比不同模型的预测效果，探讨推理机制的有效性及影响因素。**

- **链接: [https://arxiv.org/pdf/2501.03624](https://arxiv.org/pdf/2501.03624)**

> **作者:** Gaoussou Youssouf Kebe; Jeffrey M. Girard; Einat Liebenthal; Justin Baker; Fernando De la Torre; Louis-Philippe Morency
>
> **摘要:** Large language models (LLMs) excel on many NLP benchmarks, but their behavior on real-world, semi-structured prediction remains underexplored. We present LlaMADRS, a benchmark for structured clinical assessment from dialogue built on the CAMI corpus of psychiatric interviews, comprising 5,804 expert annotations across 541 sessions. We evaluate 25 open-source models (standard and reasoning-augmented; 0.6B--400B parameters) and generate over 400,000 predictions. Our results demonstrate that strong open-source LLMs achieve item-level accuracy with residual error below clinically substantial thresholds. Additionally, an Item-then-Sum (ItS) strategy, assessing symptoms individually through discrete LLM calls before synthesizing final scores, significantly reduces error relative to Direct Total Score (DTS) prediction across most model architectures and scales, despite reasoning models attempting similar decomposition in the reasoning traces of their DTS predictions. In fact, we find that performance gains attributed to "reasoning" depend fundamentally on prompt design: standard models equipped with structured task definitions and examples match reasoning-augmented counterparts. Among the latter, longer reasoning traces correlate with reduced error; while higher model scale does across both architectures. Our results clarify when and why reasoning helps and offer actionable guidance for deploying LLMs in semi-structured clinical assessment.
>
---
#### [replaced 082] Do We Need Bigger Models for Science? Task-Aware Retrieval with Small Language Models
- **分类: cs.IR; cs.AI; cs.CL; cs.DL**

- **简介: 该论文研究科学问答任务，探讨小模型是否可替代大模型。通过设计任务感知的检索框架，结合文献和元数据，提升小模型性能，解决可复现性与可访问性问题。**

- **链接: [https://arxiv.org/pdf/2604.01965](https://arxiv.org/pdf/2604.01965)**

> **作者:** Florian Kelber; Matthias Jobst; Yuni Susanti; Michael Färber
>
> **备注:** Accepted at NSLP@LREC 2026
>
> **摘要:** Scientific knowledge discovery increasingly relies on large language models, yet many existing scholarly assistants depend on proprietary systems with tens or hundreds of billions of parameters. Such reliance limits reproducibility and accessibility for the research community. In this work, we ask a simple question: do we need bigger models for scientific applications? Specifically, we investigate to what extent carefully designed retrieval pipelines can compensate for reduced model scale in scientific applications. We design a lightweight retrieval-augmented framework that performs task-aware routing to select specialized retrieval strategies based on the input query. The system further integrates evidence from full-text scientific papers and structured scholarly metadata, and employs compact instruction-tuned language models to generate responses with citations. We evaluate the framework across several scholarly tasks, focusing on scholarly question answering (QA), including single- and multi-document scenarios, as well as biomedical QA under domain shift and scientific text compression. Our findings demonstrate that retrieval and model scale are complementary rather than interchangeable. While retrieval design can partially compensate for smaller models, model capacity remains important for complex reasoning tasks. This work highlights retrieval and task-aware design as key factors for building practical and reproducible scholarly assistants.
>
---
#### [replaced 083] Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Composition-RL，解决大语言模型强化学习中可验证提示不足的问题，通过组合问题提升推理能力。**

- **链接: [https://arxiv.org/pdf/2602.12036](https://arxiv.org/pdf/2602.12036)**

> **作者:** Xin Xu; Clive Bai; Kai Yang; Tianhao Chen; Yangkun Chen; Weijie Liu; Hao Chen; Yang Wang; Saiyong Yang; Can Yang
>
> **摘要:** Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that Composition-RL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at this https URL.
>
---
#### [replaced 084] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
- **分类: cs.CL**

- **简介: 该论文提出TaxPraBen，一个针对中文税务实践的基准测试，解决LLMs在专业税务领域能力评估不足的问题，涵盖10项任务和3个真实场景。**

- **链接: [https://arxiv.org/pdf/2604.08948](https://arxiv.org/pdf/2604.08948)**

> **作者:** Gang Hu; Yating Chen; Haiyan Ding; Wang Gao; Jiajia Huang; Min Peng; Qianqian Xie; Kun Yue
>
> **摘要:** While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real-world practical capabilities. To address this issue, we introduce TaxPraBen, the first dedicated benchmark for Chinese taxation practice. It combines 10 traditional application tasks, along with 3 pioneering real-world scenarios: tax risk prevention, tax inspection analysis, and tax strategy planning, sourced from 14 datasets totaling 7.3K instances. TaxPraBen features a scalable structured evaluation paradigm designed through process of "structured parsing-field alignment extraction-numerical and textual matching", enabling end-to-end tax practice assessment while being extensible to other domains. We evaluate 19 LLMs based on Bloom's taxonomy. The results indicate significant performance disparities: all closed-source large-parameter LLMs excel, and Chinese LLMs like Qwen2.5 generally exceed multilingual LLMs, while the YaYi2 LLM, fine-tuned with some tax data, shows only limited improvement. TaxPraBen serves as a vital resource for advancing evaluations of LLMs in practical applications.
>
---
#### [replaced 085] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型中词汇身份和屈折特征的编码机制，通过分析25个模型解决语言信息表征问题，发现屈折特征在各层稳定，而词汇身份随深度减弱。**

- **链接: [https://arxiv.org/pdf/2506.02132](https://arxiv.org/pdf/2506.02132)**

> **作者:** Michael Li; Nishant Subramani
>
> **备注:** Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information relies primarily on studies of early models like BERT and GPT-2. We systematically probe 25 models from BERT Base to Qwen2.5-7B focusing on two linguistic properties: lexical identity and inflectional features across 6 diverse languages. We find a consistent pattern: inflectional features are linearly decodable throughout the model, while lexical identity is prominent early but increasingly weakens with depth. Further analysis of the representation geometry reveals that models with aggressive mid-layer dimensionality compression show reduced steering effectiveness in those layers, despite probe accuracy remaining high. Pretraining analysis shows that inflectional structure stabilizes early while lexical identity representations continue evolving. Taken together, our findings suggest that transformers maintain inflectional features across layers, while trading off lexical identity for compact, predictive representations. Our code is available at this https URL
>
---

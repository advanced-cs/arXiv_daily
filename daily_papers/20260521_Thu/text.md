# 自然语言处理 cs.CL

- **最新发布 105 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] Data Scaling as Progressive Coverage of a Predictive Contribution Spectrum
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究数据缩放机制，提出基于预测贡献谱的渐进覆盖理论，解决数据规模与模型性能关系的问题。通过分析文本语料库，验证了数据缩放与预测谱尾部的相关性。**

- **链接: [https://arxiv.org/pdf/2605.20196](https://arxiv.org/pdf/2605.20196)**

> **作者:** Zihui Song; Shihao Ji; Hongxi Li; Shuaizhi Cheng; Chunlin Huang
>
> **备注:** 8 pages,6 figures
>
> **摘要:** We investigate the hypothesis that real-data scaling laws are governed by progressive coverage of a latent predictive contribution spectrum rather than by token-frequency tails alone. We work with a suffix-automaton representation of text corpora and define a data-intrinsic global-KL predictive contribution spectrum, in which each state contributes according to its empirical mass times its KL deviation from a global next-token baseline. Across 12 real corpora, the tail slope of this spectrum is already strongly correlated with the empirical data-scaling exponent of a fixed small GPT learner. We then go beyond slope correlation and define, for each training size N, an effective truncation rank K(N) by matching the observed excess loss to the residual tail mass of the prepared 1000k global-KL spectrum. Empirically, log K is close to linear in log N, with pooled R^2 about 0.96 for the raw spectrum and R^2 about 0.90 for the smoothed spectrum. These findings provide strong empirical support for a simple mechanism picture: training scale advances an effective frontier through a predictive state spectrum, and the residual tail mass of that spectrum tracks the remaining excess loss.
>
---
#### [new 002] Improving Quantized Model Performance in Qualitative Analysis with Multi-Pass Prompt Verification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究量化模型在定性分析中的性能优化。任务是提升低比特量化模型的稳定性与准确性。通过多轮提示验证方法减少幻觉，提高模型输出质量。**

- **链接: [https://arxiv.org/pdf/2605.20193](https://arxiv.org/pdf/2605.20193)**

> **作者:** Aisvarya Adeseye; Jouni Isoaho; Adeyemi Adeseye
>
> **备注:** Accepted to publish in 12th Intelligent Systems Conference 2026; 3-4 September 2026 in Amsterdam, The Netherlands
>
> **摘要:** Quantized Large Language Models (LLMs) are used more often in qualitative analysis because they run fast and need fewer computing resources. This study examines how different lower bits quantization levels (8-bit, 4-bit, 3-bit, and 2-bit) and quantization types affect the performance of LLaMA-3.1 (8B) on qualitative analysis. The study uses expert and non-expert responses from 82 interview transcripts. Low-bit models often produce higher levels of hallucinations and unstable results, especially when reading non-expert language with unclear terms. To improve performance, we propose a quantization-aware multi-pass prompt verification method. This method guides the model through controlled steps that reduce hallucinations. It removes unreliable content and passes the results to the next transcript after verification, improving accuracy. To validate performance, human coders analyzed transcripts using NVivo and BF16 LLaMA. BF16 LLaMA-3.1 produced high-precision output but had semantic drift and hallucination. These errors were corrected manually. The corrected BF16 output and NVivo human coding were combined to create a gold-standard ground truth (GSGT) for thematic extraction and frequency analysis. The results show that 8-bit models stay closest to the GSGT. The 4-bit models lose accuracy but become stable when the proposed method is applied. The 3-bit and 2-bit models drop in performance because of heavy compression, but they improve with the proposed prompt design and verification. The study also finds that models at the same bit level behave differently depending on quantization type. Overall, the method helps low-resource LLMs become more stable, accurate, and suitable for qualitative research at lower cost.
>
---
#### [new 003] Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分析任务，旨在解决长文档处理中的偏差与遗漏问题。通过并行分块处理和证据锚定整合，提升分析的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.20194](https://arxiv.org/pdf/2605.20194)**

> **作者:** Aisvarya Adeseye; Jouni Isoaho; Adeyemi Adeseye
>
> **备注:** Accepted to be Published in 12th Intelligent Systems Conference 2026, 3-4 September 2026 in Amsterdam, The Netherlands
>
> **摘要:** Large language models (LLMs) have been increasingly used to analyze text. However, they are often plagued with contextual reasoning limitations when analyzing long documents. When long documents are processed sequentially, early or dominant concepts can overshadow less visible but meaningful interpretations, leading to cumulative analytical bias, omission error, and over-generalization. Additionally, independently generated outputs are often merged without systematic grounding, introducing redundancy, conceptual drift, and unsupported claims. This study proposes a structured framework combining parallel chunk-level processing with evidence-anchored consolidation. Texts are first divided into semantically coherent chunks and processed independently in parallel to remove influence from earlier processing. The independently generated interpretations are then consolidated using explicit evidence anchoring and prioritization that reduces dominance and over-generalization while improving traceability. Experiments with multiple model types and sizes indicate that parallel processing significantly reduces omission error by approximately 84%, increases evidence traceability by up to 130%, and reduces unsupported claims by up to 91%. Smaller models benefited most, suggesting that efficient parallel chunking and consolidation play a critical role in achieving reliable and scalable textual analysis.
>
---
#### [new 004] What Do Biomedical NER and Entity Linking Benchmarks Measure? A Corpus-Centric Diagnostic Framework
- **分类: cs.CL**

- **简介: 该论文属于生物医学命名实体识别与实体链接任务，旨在分析基准数据集的特性。通过构建框架诊断数据集属性，揭示其差异及对评估的影响。**

- **链接: [https://arxiv.org/pdf/2605.20537](https://arxiv.org/pdf/2605.20537)**

> **作者:** Robert Leaman; Rezarta Islamaj; Zhiyong Lu
>
> **备注:** Accepted to the ACL 25th Workshop on Biomedical Language Processing
>
> **摘要:** Biomedical named entity recognition (NER) and entity linking (EL) strongly depend on annotated corpora, but the utility of these resources for benchmarking is often assumed rather than characterized. We present a corpus-centric framework for diagnosing benchmark-relevant properties directly from corpus annotations, concept links, train-test splits, document metadata, and terminology mappings. The framework organizes standardized statistics into five families: (1) scale, density and label distribution, (2) lexical and conceptual structure, (3) train-test overlap, (4) metadata composition, and (5) terminology coverage where applicable. Applying the framework to nine corpora spanning diseases, chemicals, and cell types, we find that corpus properties can differ substantially, even when they address the same apparent task. We find differences in the evaluation signal they provide, the generalization demands they impose, the degree of train-test reuse they permit, and the regions of biomedical literature and concept space they represent. These differences suggest that commonly reported corpus statistics can be insufficient to characterize what biomedical NER and EL benchmarks evaluate. We argue that corpus-centric diagnostics provide a practical framework for analyzing corpora beyond surface descriptors such as corpus size and entity type, for identifying potential transfer risks, and for interpreting the scope of benchmarking conclusions. We release the framework as open-source code with an interactive dashboard to support reproducing our analyses and characterizing additional corpora.
>
---
#### [new 005] Strategy-Induct: Task-Level Strategy Induction for Instruction Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于指令生成任务，旨在解决无标签答案情况下提升LLM性能的问题。通过生成推理策略对，诱导任务指令，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.20924](https://arxiv.org/pdf/2605.20924)**

> **作者:** Po-Chun Chen; Hen-Hsen Huang; Hsin-Hsi Chen
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Designing effective task-level prompts is crucial for improving the performance of Large Language Models (LLMs). While prior work on instruction induction demonstrates that LLMs can infer better instructions with limited examples, existing approaches often rely on input-output pairs, where obtaining labeled answers can be difficult or costly. To address this limitation, we propose Strategy-Induct, a framework that derives task-level instructions solely from a small set of example questions without requiring labeled answers. Our approach first prompts the model to generate explicit reasoning strategies for each question, forming (strategy, question) pairs. These pairs are then used to induce a task instruction that guides reasoning. Experiments across multiple tasks and model scales demonstrate that Strategy-Induct outperforms state-of-the-art methods in question-only settings. Furthermore, we observe that jointly utilizing LLMs and Large Reasoning Models across task instruction generation and inference may lead to further performance improvements.
>
---
#### [new 006] Puzzled By ChatGPT? No more! A Jigsaw Puzzle to Promote AI Literacy and Awareness
- **分类: cs.CL**

- **简介: 该论文属于AI教育任务，旨在提升公众AI素养。通过设计互动拼图游戏，直观展示AI原理、优势与风险，促进非正式学习中的理解与认知。**

- **链接: [https://arxiv.org/pdf/2605.20404](https://arxiv.org/pdf/2605.20404)**

> **作者:** Francesca Padovani; Malvina Nissim
>
> **摘要:** The rapid adoption of Generative AI, including LLM-based chatbots like ChatGPT, has highlighted the need for accessible ways to support public understanding and AI literacy. To address this need, we introduce a game-based, interactive approach in the form of a jigsaw puzzle whose completed image is a comic-based infographic illustrating the workings, capabilities, limitations, and societal implications of these technologies. Each comic sketch also functions as a standalone informational card, providing focused explanations of specific facets of AI use, design, and impact. The visual content was created in a live collaborative session with a professional illustrator and a multidisciplinary group of experts and non experts, combining structured knowledge with informal, exploratory reflections shared during the discussion. By integrating hands-on assembly, visual storytelling, and collaborative interaction, the puzzle provides an engaging and playful tool for exploring the mechanisms, perks, and perils of AI systems in informal learning contexts.
>
---
#### [new 007] DEL: Digit Entropy Loss for Numerical Learning of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数值预测任务，旨在解决大语言模型在数学和代码生成中数值学习不精准的问题。提出Digit Entropy Loss（DEL），改进数值预测效果。**

- **链接: [https://arxiv.org/pdf/2605.20369](https://arxiv.org/pdf/2605.20369)**

> **作者:** Zhaohui Zheng; Chenhang He; Shihao Wang; Yuxuan Li; Ming-Ming Cheng; Lei Zhang
>
> **摘要:** Number prediction stands as a fundamental capability of large language models (LLMs) in mathematical problem-solving and code generation. The widely adopted maximum likelihood estimation (MLE) for LLM training is not tailored to number prediction. Recently, penalty-driven approaches, e.g., Number Token Loss and Discretized Distance Loss, introduce an inductive bias of numerical distance but induce over-sharpened and over-flattened digit distributions, respectively. In this paper, we make an in-depth analysis on LLM numerical learning, and show that existing numerical learning methods conceptually follow a criterion-distance formulation, where the criterion term represents optimization pattern and the distance term instills geometric prior. Consequently, we present Digit Entropy Loss (DEL) for auto-regressive numerical learning, which reformulates the conventional unsupervised entropy optimization in three key designs: leveraging digit conditional probability and binary cross-entropy to guide the entropy optimization into a supervised manner; deprecating the distance term to bypass the issue of numerical distance; and generalizing the integer-based numerical learning to floating-point number optimization, enabling more accurate number prediction. Our DEL formulation can incorporate integers, decimals, and decimal points, expanding the learning objective from a single digit to the floating-point number domain. Experiments conducted on seven mathematical reasoning benchmarks with four representative LLMs, including CodeLlama, Mistral, DeepSeek, and Qwen-2.5, demonstrate that DEL consistently outperforms its counterparts in both overall prediction accuracy and numerical distance. Source codes are at this https URL
>
---
#### [new 008] Findings of the Counter Turing Test: AI-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决区分人类与AI文本及识别生成模型的问题。通过实验评估不同检测方法的效果，提出改进策略。**

- **链接: [https://arxiv.org/pdf/2605.20761](https://arxiv.org/pdf/2605.20761)**

> **作者:** Rajarshi Roy; Gurpreet Singh; Ashhar Aziz; Shashwat Bajpai; Nasrin Imanpour; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Amitava Das; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The rapid proliferation of AI-generated text has introduced significant challenges in maintaining the integrity of digital content. Advanced generative models such as GPT-4, Claude 3.5, and Llama can produce highly coherent and human-like text, making it increasingly difficult to differentiate between human-written and AI-generated content. While these models have transformative applications, their misuse has raised concerns about misinformation, biased narratives, and security threats. This paper provides a comprehensive analysis of state-of-the-art AI-generated text detection techniques and evaluates their effectiveness through the Counter Turing Test (CT2) shared tasks. Task A (Binary Classification) required participants to distinguish between human-written and AI-generated text, while Task B (Model Attribution) focused on identifying the specific language model responsible for generating a given text. The results demonstrated high performance in binary classification, with the top system achieving an F1 score of 1.0000, but significantly lower scores in model attribution, where the best system achieved 0.9531, highlighting the increased complexity of this task. The top-performing teams leveraged fine-tuned transformer models, ensemble learning, and hybrid detection approaches, with DeBERTa-based and BART-based methods demonstrating strong results. However, the lower scores in Task B underscore the challenges of distinguishing outputs from different LLMs, necessitating further research into adversarial robustness, feature extraction, and cross-domain generalization.
>
---
#### [new 009] Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在探究链式思维提示对大语言模型性别偏见的影响。工作包括基准评估、机制可解释性分析和推理链检查，发现CoT提示仅表面缓解偏见。**

- **链接: [https://arxiv.org/pdf/2605.20410](https://arxiv.org/pdf/2605.20410)**

> **作者:** Edie Pearman; Sophia Osborne; Mira Kandlikar-Bloch; Mina Arzaghi; Florian Carichon; Golnoosh Farnadi
>
> **备注:** 24 pages, 6 figures, including appendix. Accepted at the ICLR 2026 Workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems. Submitted to COLM 2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed in socially sensitive settings despite substantial documentation that they encode gender biases. Chain-of-Thought (CoT) prompting has been proposed as a bias-mitigation approach. However, existing evaluations primarily focus on changes in LLM benchmark performance, providing limited insight into whether apparent bias reductions reflect meaningful changes in a model's internal mechanisms. In this work, we investigate how CoT prompting affects gender bias in LLMs, combining benchmark-based evaluation with mechanistic interpretability techniques and reasoning chain failure analysis. Our results confirm a stereotypical bias present in LLM outputs across benchmarks, showing that CoT prompting does not consistently reduce the bias gap. Mechanistic analyses reveal that although CoT balances biased behavior in certain attention head clusters, gender bias remains embedded in hidden representations, indicating only superficial mitigation. Inspection of reasoning chains further suggests that these improvements stem from memorization and familiarity with the dataset rather than genuine understanding of bias.
>
---
#### [new 010] Building Arabic NLP from the Ground Up: Twenty Years of Lessons, Failures, and Open Problems
- **分类: cs.CL**

- **简介: 该论文回顾了阿拉伯语NLP发展二十年的历程，探讨了资源建设与研究基础设施的挑战。属于自然语言处理领域，旨在解决 underserved 语言的技术与社会问题，总结了经验教训与失败案例。**

- **链接: [https://arxiv.org/pdf/2605.20786](https://arxiv.org/pdf/2605.20786)**

> **作者:** Wajdi Zaghouani
>
> **备注:** Accepted at the ACL 2026 Workshop : The Big Picture 2026: Crafting a Research Narrative v2
>
> **摘要:** This paper reflects on twenty years of building NLP resources and research infrastructure for Arabic, a language spoken by hundreds of millions yet historically underserved relative to languages such as English or Chinese. The first decade focused on foundational linguistic infrastructure; the second shifted toward computational social science, social media analysis, and socially oriented applications. Rather than cataloguing outputs, the paper examines what the experience of building them revealed. Three counterintuitive lessons emerge: building datasets is as much a social process as a technical one; communities formed around shared tasks often matter more than the tasks themselves; and moving from language resources to computational social science exposes challenges that traditional NLP training does not address. We discuss three failures: a depression detection corpus that never reached clinical practice, a period of spreading across too many shared tasks without sufficient depth, and a long-standing assumption that Modern Standard Arabic infrastructure would transfer cleanly to dialectal tasks. These experiences suggest that the hardest problems in developing NLP for underserved communities are not linguistic but social, institutional, and epistemic, and require competencies the field rarely teaches.
>
---
#### [new 011] Direct Translation between Sign Languages
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于手语直接翻译任务，旨在解决不同手语间通信障碍问题。通过构建合成数据并训练模型，实现高效准确的直接手语翻译。**

- **链接: [https://arxiv.org/pdf/2605.20588](https://arxiv.org/pdf/2605.20588)**

> **作者:** Zetian Wu; Bowen Xie; Wuyang Meng; Milan Gautam; Stefan Lee; Liang Huang
>
> **摘要:** The field of sign language translation has witnessed significant progress in the translation between sign and spoken languages, but the translation between sign languages remains largely unexplored and out of reach. The latter can help 1.5 billion deaf and hard-of-hearing (DHH) people worldwide communicate across language barriers without relying on hearing interpreters or written-language fluency. The cascade approach composing separate sign-to-text, text-to-text, and text-to-sign systems suffers from error propagation and extra latency as well as the loss of information unique in the visual modality. We aim to develop direct sign-to-sign translation. However, a large-scale open-domain parallel corpus has not been curated between sign languages. To enable direct translation between sign language utterances, we use back-translation to produce synthetic sign-sign pairs from unaligned individual language utterance-sign corpora. Using this data, we jointly train a single MBART-based model for both text->sign (T2S) and sign->sign (S2S). On synthetically generated paired sets between American Sign Language (ASL), Chinese Sign Language (CSL), and German Sign Language (DGS), our direct S2S method outperforms the cascaded baseline on geometric sign error metrics (20% lower DTW-aligned MPJPE) and language matching metrics after predicted sign utterances are translated back to sentences (50% high BLEU-4) while achieving a roughly 2.3* speedup. On a small set of pre-existing cross-lingual sign data, we find similar improvements for our proposed method.
>
---
#### [new 012] Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation
- **分类: cs.CL**

- **简介: 该论文属于生物医学摘要生成任务，解决无摘要文章的生成问题。提出DPR-BAG框架，无需训练即可生成结构化且事实一致的摘要。**

- **链接: [https://arxiv.org/pdf/2605.20628](https://arxiv.org/pdf/2605.20628)**

> **作者:** Sylvey Lin; Joe Menke; Shufan Ming; Dongin Nam; Neil Smalheiser; Halil Kilicoglu
>
> **备注:** Accepted by BioNLP 2026
>
> **摘要:** Biomedical abstracts play a critical role in downstream NLP applications, such as information retrieval, biocuration, and biomedical knowledge discovery. However, a non-trivial number of biomedical articles do not have abstracts, diminishing the utility of these articles for downstream tasks. We propose DPR-BAG (Divide, Prompt, and Refine for Biomedical Abstract Generation), a training-free, zero-shot framework that generates coherent and factually grounded abstracts for biomedical articles with full text but no abstract. DPR-BAG decomposes full-text documents into structured rhetorical facets following the Background-Objective-Methods-Results-Conclusions (BOMRC) schema, performs parallel LLM-based summarization for each facet, and applies a final refinement stage to restore global discourse coherence. On PMC-MAD, a distribution-aligned dataset of 46,309 biomedical articles, DPR-BAG improves abstractive novelty over strong extractive and fine-tuned baselines, while maintaining factual consistency. Our ablation study reveals a counterintuitive finding: increasing prompt complexity or explicitly injecting entity-level guidance can degrade factual alignment, highlighting the importance of controlled prompting strategies. These findings underscore the potential of training-free, structure-aware frameworks for scalable biomedical abstract generation in low-resource settings. Our data and code are available at this https URL and this https URL.
>
---
#### [new 013] Shiny Stories, Hidden Struggles: Investigating the Representation of Disability Through the Lens of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在检验LLMs对残疾人的代表性。通过比较LLMs生成与真实残疾人的社交媒体内容，发现其存在理想化和负面偏见问题。**

- **链接: [https://arxiv.org/pdf/2605.20191](https://arxiv.org/pdf/2605.20191)**

> **作者:** Marco Bombieri; Simone Paolo Ponzetto; Marco Rospocher
>
> **备注:** Accepted for publication in ACM Transactions on Intelligent Systems and Technology
>
> **摘要:** Modern Large Language Models (LLMs) have recently attracted much attention for their ability to simulate human behavior and generate text that reflects personas and demographic groups. While these capabilities can open up a multitude of diverse applications across fields, it is crucial to examine how such models represent various target groups since LLMs can perpetuate and amplify biases or discrimination against historically marginalized communities or, alternatively, as a result of debiasing efforts, overcorrect by portraying overly positive stereotypes. This overcompensation can idealize these groups, erasing the complexities and challenges they face in favor of unrealistic depictions. In this paper, we investigate how LLMs represent disability by simulating the perspectives of individuals with disabilities in generating social media posts. These posts are then compared with those written by real people with disabilities, focusing on emotional tone, sentiment, and representative words and themes. Our analysis reveals two key findings: (1) LLMs often idealize the experiences of people with disabilities, producing overly positive stereotypes that, despite appearing uplifting, fail to authentically capture their lived realities; and (2) a comparative analysis of posts simulating individuals with and without disabilities highlights a negative bias, where certain topics, such as career and entertainment, are disproportionately associated with nondisabled individuals. This reinforces exclusionary narratives and over-idealized portrayals of disability, misrepresenting the actual challenges faced by this community. These findings align with broader concerns and ongoing research showing that LLMs struggle to reflect the diverse realities of society, particularly the nuanced experiences of marginalized groups, and underscore the need for critical scrutiny of their representations.
>
---
#### [new 014] Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory
- **分类: cs.CL**

- **简介: 该论文属于语言模型预训练任务，旨在解决扩大模型容量时成本高、效果差的问题。提出Memory Grafting方法，利用冻结隐状态作为记忆，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.20948](https://arxiv.org/pdf/2605.20948)**

> **作者:** Runxi Cheng; Yuchen Guan; Yongxian Wei; Qianpu Sun; Qixiu Li; Sinan Du; Feng Xiong; Chun Yuan; Yan Lu; Yeyun Gong
>
> **备注:** 25 pages, 12 figures, 5 tables
>
> **摘要:** Scaling conditional memory offers a promising way to increase language-model capacity, but existing methods such as Engram learn large memory tables from scratch during pre-training, making memory scaling expensive and sometimes ineffective. We propose Memory Grafting, a conditional memory scaling method that utilizes frozen hidden states from a grafting model as conditional n-gram memory. Given frequent local n-grams, we run the grafting model offline, store final-token hidden representations as memory values, and let the recipient model retrieve them through exact longest-match suffix lookup. Retrieved memories are adapted by lightweight projections and gates, while a hash-based Engram fallback preserves coverage for unmatched contexts. Since the grafting model is only run offline and exact lookup has expected O(1) complexity with respect to memory-bank size, Memory Grafting expands external latent capacity with limited training and inference overhead. Experiments under matched recipient architectures and pre-training budgets show that Memory Grafting improves over both MoE and vanilla Engram baselines. In the 2.8B-scale setting, it improves the average benchmark score from 51.95 for MoE and 52.43 for vanilla Engram to 53.86. In the 0.92B-scale setting, all grafting-model variants improve over the baselines, with Qwen3.5-35B-A3B giving the strongest gains. These results suggest that pretrained models can serve as reusable constructors of external latent memory, providing a practical step toward scaling future language models beyond trainable parameters alone.
>
---
#### [new 015] GradeLegal: Automated Grading for German Legal Cases
- **分类: cs.CL**

- **简介: 该论文属于法律考试自动评分任务，旨在解决德国法律考试评分量大、专家不足的问题。通过评估27个语言模型，探索有效评分方法。**

- **链接: [https://arxiv.org/pdf/2605.21076](https://arxiv.org/pdf/2605.21076)**

> **作者:** Abdullah Al Zubaer; Lorenz Wendlinger; Simon Alexander Nonn; Michael Granitzer; Jelena Mitrovic
>
> **摘要:** Grading German legal exam solutions faces growing volumes and a shortage of qualified graders, delaying feedback and creating a bottleneck. At the same time, it is a high-stakes expert task, since state exam grades strongly influence career outcomes in Germany. Despite this practical relevance, literature lacks systematic studies on effective methods for grading legal exams. To address this gap, we investigate whether large language models (LLMs) can support the automated grading of German legal case solutions in criminal and public law, thereby enabling scalable feedback and student self-testing. We present a systematic evaluation of 27 proprietary and open-source LLMs, benchmarking prompting strategies that incrementally add task-related information, such as a sample solution and a grading rubric. Using quadratic weighted kappa (QWK), reasoning-oriented LLMs can approximate expert grading in public law when given a sample solution and a grading rubric (up to 0.91), compared to 0.60 in criminal law, suggesting a harder grading task in criminal law. Beyond single-model grading, ensembling improves agreement by up to 0.15 over its best member and can offer an alternative to stronger closed-source single models. In addition, our findings suggest that effective prompt design and model selection are necessary for reliable LLM-based grading of legal exams.
>
---
#### [new 016] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本推理性能差的问题。通过ProxyCoT框架，将短上下文的推理能力迁移至长上下文，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.20201](https://arxiv.org/pdf/2605.20201)**

> **作者:** Miao Li; Irina Saparina; Alexander Gurung; Mirella Lapata
>
> **备注:** Long, ACL 2026 (Main conference)
>
> **摘要:** Recent large language models support inputs of up to 10 million tokens, yet they perform poorly on long-context tasks that require complex reasoning. Such tasks can be solved using only a subset of the input -- a proxy context -- rather than the full sequence. Despite sharing the same underlying reasoning process, models exhibit a significant performance disparity between proxy and full contexts. To improve long-context reasoning, we propose ProxyCoT, a novel training framework that transfers reasoning capabilities from short proxy contexts to full long contexts. Specifically, we first obtain high-quality chain-of-thought reasoning traces on proxy contexts through reinforcement learning or distillation from a larger teacher model, and then ground the generated traces in full long contexts with supervised fine-tuning. Experiments across different datasets demonstrate that ProxyCoT consistently outperforms strong baselines with reduced computational overhead. Furthermore, models trained with ProxyCoT generalize their long-context reasoning capabilities to out-of-domain tasks.
>
---
#### [new 017] Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究机器遗忘中的可靠性悖论，探讨校准与决策规则的关系。任务是评估语言模型的可靠性，解决校准误差与决策依赖性之间的矛盾，通过实验验证遗忘后模型仍保持低校准但可能依赖捷径。**

- **链接: [https://arxiv.org/pdf/2605.20915](https://arxiv.org/pdf/2605.20915)**

> **作者:** Divyaksh Shukla; Ashutosh Modi
>
> **备注:** Accepted at SRW, ACL 2026; 17 pages (9 + 2 + 6)
>
> **摘要:** Machine unlearning aims to remove the influence of specific training data from a model while preserving reliable behavior on the remaining data, making reliable prediction and uncertainty estimation essential for evaluation. Calibration is commonly used as a proxy for reliability in language models, but low calibration error does not necessarily imply reliable decision rules, as models may rely on spurious correlations while remaining well calibrated. We investigate this gap in generative language models using the multiple-choice question-answering evaluation protocol on the TOFU benchmark, measuring probabilistic reliability with calibration metrics (ECE, MCE, Brier) and decision-rule reliability via attribution-based shortcut detection with Integrated Gradients and Local Mutual Information. We find that fine-tuned models achieve low calibration error (ECE ~ 0.04) compared to pretrained models (ECE > 0.5), and models after unlearning retain similarly low calibration despite reduced accuracy on the forget split, while attribution analysis shows increased reliance on correlation-based tokens. These results demonstrate that good calibration can coexist with shortcut-based decision rules after unlearning, extending the reliability paradox to the machine unlearning setting.
>
---
#### [new 018] When Reasoning Supervision Hurts: TTCW-Based Long-Form Literary Review Generation
- **分类: cs.CL**

- **简介: 该论文属于文学评论生成任务，旨在解决长文本创意评价难题。构建了大规模TTCW标注数据集，对比了有无推理监督的模型效果，发现非推理方法更优。**

- **链接: [https://arxiv.org/pdf/2605.20364](https://arxiv.org/pdf/2605.20364)**

> **作者:** Jinlong Liu; Mohammed Bahja; Mark Lee
>
> **备注:** Submit to EMNLP 2026
>
> **摘要:** Automatic evaluation of long-form literary writing remains challenging, as generic LLM-as-Judge approaches may not fully capture creativity-related dimensions such as originality and flexibility. Although the Torrance Test of Creative Writing (TTCW) provides a structured creativity framework, and prior work has demonstrated reference-based TTCW evaluation at the pairwise level, no large-scale dataset exists for long-form TTCW-based literary review generation. We address this gap by constructing a dataset of 263,911 long-form stories, each annotated with scalar scores and meta-synthesised review comments across 14 TTCW-based dimensions. Using this dataset, we fine-tune Qwen3 models at two scales, 4B and 8B, under two conditions: with and without reasoning content. Results show that non-reasoning fine-tuning achieves stronger and more stable performance, with the best setting reaching an evaluation score of 0.6820. Further analysis shows that reasoning-supervised models are more prone to parse failures, often continuing with irrelevant or repetitive reasoning-style text rather than completing the required 14-metric review report. These results suggest that, for fixed-format rubric-based review generation, reasoning supervision is not straightforwardly beneficial, and precise metric-aligned scoring remains challenging even after task-specific fine-tuning.
>
---
#### [new 019] LASH: Adaptive Semantic Hybridization for Black-Box Jailbreaking of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于黑盒 jailbreak 攻击任务，旨在提升攻击成功率。提出 LASH 框架，通过自适应组合多种攻击策略，有效突破模型安全机制。**

- **链接: [https://arxiv.org/pdf/2605.21362](https://arxiv.org/pdf/2605.21362)**

> **作者:** Abdullah Al Nomaan Nafi; Fnu Suya; Swarup Bhunia; Prabuddha Chakraborty
>
> **摘要:** Jailbreak attacks expose a persistent gap between the intended safety behavior of aligned large language models and their behavior under adversarial prompting. Existing automated methods are increasingly effective but each commits to a single attack family (e.g., one refinement loop, one tree search, one mutation space, or one strategy library) and no single family dominates: the best-performing method shifts across target models and harm categories, suggesting complementary strengths that per-prompt composition could exploit. We introduce LASH (LLM Adaptive Semantic Hybridization), a black-box framework that treats outputs from multiple base attacks as reusable seed prompts and adaptively composes them for each target request. Given a seed pool, LASH searches over seed subsets and softmax-normalized mixture weights; a composition module synthesizes a single candidate prompt, and a derivative-free genetic optimizer updates the weights using black-box target feedback and a two-stage fitness function combining keyword-based refusal detection with LLM-judge scoring. On JailbreakBench, which contains 100 harmful prompts across 10 categories, we evaluate LASH on six common target models. LASH achieves an average attack success rate of 84.5% under keyword-based evaluation and 74.5% under two-stage evaluation, where responses are first filtered for refusals and then scored by an LLM judge for whether they substantively fulfill the original harmful request. LASH outperforms five state-of-the-art baselines on both metrics with only 30 mean target queries. LASH also remains competitive under three defense mechanisms and induces more success-like internal representations. These results suggest that adaptive composition across heterogeneous jailbreak strategies is a promising direction for black-box red-teaming.
>
---
#### [new 020] Cross-lingual robustness of LLM-brain alignment and its computational roots
- **分类: cs.CL**

- **简介: 该论文研究跨语言下大语言模型与大脑活动的对齐情况，旨在解决模型与脑功能区域关联性问题。通过多语言实验，分析模型结构与神经活动的对应关系。**

- **链接: [https://arxiv.org/pdf/2605.21049](https://arxiv.org/pdf/2605.21049)**

> **作者:** Ni Yang; Rui He; Philipp Homan; Iris Sommer; Davide Staub; Wolfram Hinzen
>
> **摘要:** Large language models (LLMs) reliably predict neural activity during language comprehension and transformer depth has been interpreted as mirroring hierarchical cortical organization. However, it remains unclear whether such alignment extends to subcortical regions, overlaps spatially across languages, and what the computational roots of such alignment are. Here, we used a multilingual, whole-brain encoding framework to examine brain-LLM alignment across three typologically distinct languages: Mandarin, English, and French during naturalistic story listening. Our results show that across languages, transformer-based models predicted activity in a distributed landscape spanning widely distributed cortical functional networks like limbic, ventral attention, default mode network, and subcortical structures. Spatial alignment patterns showed substantial cross-linguistic overlap and remained largely stable across model layers, with limited layer progression consistent with functional cortical hierarchies. Contrary to previous evidence, contextual embeddings did not outperform static embeddings. To test candidate computational explanations, we examined whether layer-wise brain scores reflect surprisal and intrinsic dimensionality, and thereby predictive processing and information compression. Neither of these two computational metrics mirrored neural alignment profiles. Our findings suggest that brain-LLM alignment is spatially robust and cross-linguistically stable but not explainable from predictive uncertainty or representational geometry. Rather than directly reflecting shared hierarchical computation, neural predictivity may primarily arise from distributed lexical-semantic correspondences that generalize across languages.
>
---
#### [new 021] WCXB: A Multi-Type Web Content Extraction Benchmark
- **分类: cs.CL**

- **简介: 该论文提出WCXB基准，解决网页内容提取任务中的评估不足问题。针对现有基准规模小、类型单一的缺陷，构建包含多种页面类型的2008页数据集，并评估13种提取系统性能。**

- **链接: [https://arxiv.org/pdf/2605.21097](https://arxiv.org/pdf/2605.21097)**

> **作者:** Murrough Foley
>
> **备注:** Dataset: this http URL, this http URL. Leaderboard: this http URL. Preprint also deposited at this http URL
>
> **摘要:** Web content extraction - isolating a page's main content from surrounding boilerplate - is a prerequisite for search indexing, retrieval-augmented generation, NLP dataset construction, and large language model training. Progress in this area has been constrained by the limitations of existing evaluation benchmarks, which are small (100-800 pages), restricted to news articles, or based on web pages from over a decade ago. We introduce the Web Content Extraction Benchmark (WCXB), a dataset of 2,008 web pages from 1,613 domains spanning seven structurally distinct page types: articles, forums, products, collections, listings, documentation, and service pages. The dataset includes a 1,497-page development set and a 511-page held-out test set with matched page type distributions. Ground truth annotations were produced through a five-stage pipeline: LLM-assisted drafting, automated verification, four-pass frontier model review, snippet and quality verification scripts, and human review. We evaluate 13 extraction systems - 11 heuristic and 2 neural - and find that while top systems converge on articles (F1 = 0.93), performance diverges sharply on structured page types (F1 = 0.41-0.84), revealing blind spots invisible to existing article-only benchmarks. The dataset is released under CC-BY-4.0 with HTML source files, ground truth annotations, page type labels, and baseline results.
>
---
#### [new 022] Smarter edits? Post-editing with error highlights and translation suggestions
- **分类: cs.CL**

- **简介: 该论文属于机器翻译后编辑任务，探讨LLM生成的错误提示和修正建议对译者的影响，旨在提升后编辑效率与体验。**

- **链接: [https://arxiv.org/pdf/2605.21135](https://arxiv.org/pdf/2605.21135)**

> **作者:** Fleur V.J. van Tellingen; Gautam Ranka; Dora Žugčić; Joyce van der Wal; Andrea Camasta; Livio Guerra; Alina Karakanta
>
> **备注:** Accepted at EAMT 2026
>
> **摘要:** As MT quality increases, interest in enhanced post-editing features such as QE-derived error highlights is growing, yet evidence for their usefulness remains limited. In this work, we explore the usefulness of LLM-derived error highlights and correction suggestions based on automatic post-editing (APE). We conduct a study where professional translators (En-Nl) post-edit translations using APE error highlights and correction suggestions and compare productivity, quality and user experience to regular PE and PE with QE-derived highlights. While no condition yielded productivity or quality gains compared to regular PE, APE highlights were better received than QE-derived highlights, and correction suggestions improved overall user experience.
>
---
#### [new 023] MemGym: a Long-Horizon Memory Environment for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出MemGym，一个用于评估LLM代理长期记忆能力的基准。解决现有基准无法有效评估复杂任务中动态记忆的问题，整合多个评估场景并提供独立的内存评分机制。**

- **链接: [https://arxiv.org/pdf/2605.20833](https://arxiv.org/pdf/2605.20833)**

> **作者:** Wujiang Xu; Yu Wang; Kai Mei; Kaiqu Liang; Zhenting Wang; Mingyu Jin; Han Zhang; Shi-Xiong Zhang; Wenyue Hua; Sambit Sahu; Dimitris N. Metaxas
>
> **摘要:** Memory is a central capability for LLM agents operating across long-horizon tasks. Existing memory benchmarks predominantly evaluate retention of personalized information in multi-turn chat scenarios, overlooking the dynamic memory formation that occurs during extended agent execution. Consequently, the memory systems they produce transfer poorly to realistic agentic environments, such as coding and web navigation. We present MemGym, a benchmark for agentic memory that unifies existing agent gyms and in-house memory-grounded pipelines behind one memory-reasoning interface. MemGym spans five evaluation tracks grouped into four agentic regimes: tool-use dialogue (tau2-bench), multi-turn deep-research search (MEMGYM-DR), coding (SWE-Gym and MEMGYM-CODEQA), and computer use (WebArena-Infinity). MemGym reports memory-isolated scores that decouple memory performance from reasoning, retrieval, and tool-use ability, so memory strategies can be ranked without those confounders. Our synthetic pipelines for MEMGYM-CODEQA and MEMGYM-DR are length-controllable, ablation-verified at every stage, and tightly aligned with downstream scenarios. To make evaluation on coding environments academically tractable, we train MemRM, a lightweight reward model (Qwen3-1.7B fine-tuned with QLoRA) that scores compression quality as a fast scalar read in place of full Docker rollouts.
>
---
#### [new 024] LamPO: A Lambda Style Policy Optimization for Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文提出LamPO方法，用于改进推理语言模型的强化学习。针对RLVR中组内关系信息丢失的问题，采用配对优势替代标量优势，提升训练稳定性与样本效率。**

- **链接: [https://arxiv.org/pdf/2605.21235](https://arxiv.org/pdf/2605.21235)**

> **作者:** Zhe Yuan; Yipeng Zhou; Jinghan Li; Xinyuan Chen; Bowen Deng; Zhiqian Chen; Liang Zhao
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become an effective paradigm for improving reasoning language models on tasks such as mathematics, coding, and scientific question answering. However, widely used group-relative objectives, such as GRPO, summarize each sampled group with scalar statistics and therefore discard fine-grained relational information among candidate responses. This weakens credit assignment under sparse outcome rewards, especially when multiple generated solutions differ only subtly in reasoning quality. We propose \textbf{LamPO}, a \textbf{Lambda-Style Policy Optimization} method that replaces scalar group advantages with a \emph{Pairwise Decomposed Advantage}. LamPO aggregates pairwise reward gaps within each response group and modulates each comparison by a confidence-aware weight computed from sequence log-probability differences, while retaining the critic-free and clipped-update structure of PPO-style optimization. When reference solutions are available, we further add a lightweight ROUGE-L-based dense auxiliary reward to reduce reward sparsity. Experiments on AIME24, AIME25, MATH-500, and GPQA-Diamond with Qwen3-1.7B, Qwen3-4B, and Phi-4-mini show that LamPO consistently improves over GRPO and recent RLVR variants, with more stable training dynamics and better sample efficiency.
>
---
#### [new 025] APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings
- **分类: cs.CL**

- **简介: 该论文属于语言模型个性化任务，旨在解决用户隐式风格偏好评估难题。通过引入APM基准，分离用户属性与响应原则，评估不同个性化方法的效果。**

- **链接: [https://arxiv.org/pdf/2605.21063](https://arxiv.org/pdf/2605.21063)**

> **作者:** Philipp Spohn; Leander Girrbach; Zeynep Akata
>
> **摘要:** Typical LLM responses tend to follow a default style, even though users often have distinct preferences regarding tone, verbosity, and formality that they do not explicitly state in their prompts. Evaluating whether personalization methods can adapt to these implicit preferences is challenging, since users typically provide prompts rather than reference responses, style preferences are not factually verifiable, and reference-free LLM judges may conflate personalization with general response quality. To address these challenges, we introduce the Arbitrary Preference Mapping (APM) benchmark, which decouples user attributes (e.g. enthusiastic) from response principles (e.g. persuasive) via a hidden, randomized mapping $\mathbf{C}$ that maps user attributes to preferences about response traits. Because $\mathbf{C}$ carries no semantic content and is resampled across runs, models cannot exploit stereotypical associations and must infer preferences from conversation history. Using this unbiased evaluation methodology, we adapt retrieval-augmented, prompt-optimization, and routing personalization methods and evaluate them on Llama-3.1-8B and Qwen-3.5-27B. Our results show that routing is the most reliable approach, while RAG only improves with the stronger base LLM, and soft prompt optimization fails to improve significantly over a non-personalized baseline. Our extensive evaluation reveals that in this realistic setting, personalization remains challenging, but our adapted methods show promise.
>
---
#### [new 026] Do LLMs Know What Luxembourgish Borrows? Probing Lexical Neology in Low-Resource Multilingual Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言中词汇借用与创新的识别问题。通过构建基准测试和知识图谱，提升模型对借词的判断能力。**

- **链接: [https://arxiv.org/pdf/2605.21227](https://arxiv.org/pdf/2605.21227)**

> **作者:** Nina Hosseini-Kivanani
>
> **备注:** Accepted to Neollm colocated with LREC2026, Three figures and three tables
>
> **摘要:** Large language models (LLMs) are increasingly used for writing assistance in small contact languages, yet it is unclear whether they respect community norms around lexical borrowing and neology. We introduce LexNeo-Bench, a 3{,}050-instance token-level benchmark derived from LuxBorrow, a large-scale Luxembourgish news corpus, where target tokens are labelled as native or as French, German, or English borrowings. Using this benchmark, we probe three multilingual LLMs across 34 prompt settings on two tasks: borrowing type classification and a binary lexical-innovation proxy (borrowing versus native). Without external context, models perform only slightly above chance on borrowing classification, so we construct a linguistic knowledge graph that encodes donor language, morphological patterns, and lexical analogues, and inject instance-specific subgraphs into the prompt. Knowledge-graph prompts raise borrowing classification accuracy from 25 -- 35\% up to 71 -- 81\% and largely close the gap between small and large models, while leaving neology detection difficult and sensitive to few-shot design. Our results show that lexicon-aware prompting is highly beneficial for robust borrowing judgments in low-resource contact languages and that lexical resources can serve as structured context for LLM evaluation. This study was carried out within the ENEOLI COST Action and examines borrowing as a form of lexical innovation in multilingual Luxembourgish data.
>
---
#### [new 027] Pseudo-Siamese Network for Planning in Target-Oriented Proactive Dialogues
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于目标导向的主动对话任务，解决对话路径规划问题。提出FF-BPSN模型，通过双向规划生成有效对话路径，提升对话系统效果。**

- **链接: [https://arxiv.org/pdf/2605.20195](https://arxiv.org/pdf/2605.20195)**

> **作者:** Xinyue Kang; Maodong Li; Yibin Zheng; Fang Kong
>
> **备注:** ICASSP2026
>
> **摘要:** A target-oriented proactive dialogue system is designed to steer conversations toward predefined targets while actively providing suggestions. The core paradigm of such a system is to plan a reasonable dialogue path and subsequently guide language models (e.g., pre-trained or large language models) to generate responses, where dialogue path planning serves as the central component-a novel yet under-explored problem. In this work, we propose a Forward-Focused Bidirectional Pseudo-Siamese Network (FF-BPSN) for dialogue path planning toward predefined dialogue targets. FF-BPSN employs two identical transformer-based decoders for forward and backward planning, together with a forward-focused module that integrates bidirectional information to construct the final forward path. This path benefits from bidirectional planning while prioritizing forward information. We then employ the planned path to guide language models in response generation. Extensive experiments on DuRecDial and DuRecDial 2.0 demonstrate that FF-BPSN achieves state-of-the-art performance in dialogue path planning and significantly enhances the effectiveness of target-oriented proactive dialogue systems.
>
---
#### [new 028] PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于扩散语言模型加速任务，旨在解决推理计算成本高的问题。通过引入PulseCol方法，采用周期性刷新的列稀疏注意力结构，提升计算效率并保持模型质量。**

- **链接: [https://arxiv.org/pdf/2605.20813](https://arxiv.org/pdf/2605.20813)**

> **作者:** Yanyi Lyu; Letian Chen; Futing Sun; Miao Zhang; Weili Guan; Liqiang Nie
>
> **摘要:** Inference in diffusion large language models (dLLMs) is computationally expensive, as full self-attention must be repeatedly executed at each step of the denoising process without KV cache. Recent sparse attention methods for dLLMs mitigate this cost via block-sparse computation, which is applied only in later iterations when model performance is less sensitive to coarse-grained sparse approximation, but yields limited improvements in computational efficiency and acceleration. This motivates a finer-grained sparsification strategy that can be applied from earlier iterations and leverages reusable sparsity patterns, enabling further efficiency gains. In this work, we introduce PulseCol, a periodically refreshed column-sparse attention method for accelerating diffusion language models. PulseCol replaces coarse block-level sparsity with a finer-grained column-sparse structure, allowing important attention interactions to be retained more precisely while exposing greater sparsity. Built on this column-level formulation, PulseCol further identifies sparse patterns at the early denoising step and reuses them across subsequent iterations, refreshing them only at a small number of intermediate steps to track the evolution of sparse attention patterns during denoising. Experiments show that PulseCol achieves higher sparsity and greater practical speedup than prior sparse attention methods for dLLMs, while maintaining model quality. Enabled by optimized GPU kernels for column-sparse attention, PulseCol delivers up to 1.95$\times$ end-to-end speedup over FlashAttention across several context lengths.
>
---
#### [new 029] Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对漫画理解任务，解决Manga109数据集中标注错误和不精确的问题，通过OCR与人工修正改进标注，提升数据质量。**

- **链接: [https://arxiv.org/pdf/2605.21182](https://arxiv.org/pdf/2605.21182)**

> **作者:** Jeonghun Baek; Atsuyuki Miyai; Shota Onohara; Hikaru Ikuta; Kiyoharu Aizawa
>
> **备注:** Accepted to the Culture x AI Workshop at ICML 2026. Project page: this https URL
>
> **摘要:** Manga is a culturally distinctive multimodal medium and one of the most influential forms of Japanese popular culture. As AI systems increasingly target manga understanding, OCR, and translation, Manga109 has become a foundational dataset for manga-related AI research. However, the current Manga109 dataset contains transcription errors and coarse annotations, which do not align well with modern OCR and multimodal manga understanding tasks. In this work, we revisit the dialogue text annotations of Manga109 and identify five categories of annotation issues, including transcription errors, missing text regions, overlapping dialogue and onomatopoeia, and under-segmented speech balloons. To address these issues, we combine OCR-based issue detection and manual revision to construct Manga109-v2026, revising approximately 29,000 dialogue annotations. Our revisions better align Manga109 with modern OCR and multimodal manga understanding systems while preserving expressive structures characteristic of manga.
>
---
#### [new 030] Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs
- **分类: cs.CL**

- **简介: 该论文针对LLM代理的推理效率问题，提出Mix-Quant框架，通过阶段感知量化提升推理速度，缓解长上下文任务中的计算瓶颈。**

- **链接: [https://arxiv.org/pdf/2605.20315](https://arxiv.org/pdf/2605.20315)**

> **作者:** Haiquan Lu; Zigeng Chen; Gongfan Fang; Xinyin Ma; Xinchao Wang
>
> **摘要:** LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck in long-context, multi-turn inference. In this work, we propose Mix-Quant, a simple and effective phase-aware quantization framework for fast agentic inference. We first investigate FP4 quantization in agentic LLM workflows and observe that quantizing the entire inference process can incur significant performance degradation. In contrast, the prefilling stage exhibits substantial quantization redundancy and can therefore be quantized with minimal accuracy loss, despite being the dominant source of computation. Based on this insight, we apply high-throughput NVFP4 quantization to the prefilling phase while preserving BF16 precision for decoding. By decoupling prefilling acceleration from decoding quality, Mix-Quant combines phase-aware algorithmic quantization with hardware-efficient NVFP4 execution to alleviate the inference bottleneck in LLM agents. Extensive experiments across long-context and agentic benchmarks demonstrate that Mix-Quant largely preserves task performance while delivering significant efficiency improvements, achieving up to a 3x speedup during prefilling.
>
---
#### [new 031] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification
- **分类: cs.CL**

- **简介: 该论文属于医疗文本分类任务，旨在解决HIV疑似病例识别中的不确定性问题。通过构建混合框架，结合概率与几何方法，提升临床分诊的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21256](https://arxiv.org/pdf/2605.21256)**

> **作者:** Rodrigo Morales-Sánchez; Soto Montalvo; Raquel Martínez
>
> **备注:** Accepted at the BioNLP Workshop @ ACL 2026
>
> **摘要:** Standard clinical Natural Language Processing (NLP) benchmarks often yield inflated metrics by forcing deterministic classification on ambiguous instances, thereby obscuring the clinical risks of overconfident predictions. To bridge this gap, we propose a risk-aware hybrid selective classification framework, evaluated on early Human Immunodeficiency Virus suspicion identification in Spanish clinical notes. Our dual-verification approach explicitly decouples aleatoric uncertainty through Mondrian conformal prediction and epistemic uncertainty using a Multi-Centroid Mahalanobis Distance veto. Empirical evaluations reveal that standard uncertainty metrics and baseline classifiers are structurally insufficient for safe medical triage, suffering severe coverage collapse when forced to operate under strict reliability constraints. In contrast, by demanding that clinical narratives pass both probabilistic and geometric safeguards, the proposed framework successfully isolates a highly trustworthy operational domain.
>
---
#### [new 032] Tracing the ongoing emergence of human-like reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，旨在探讨大语言模型是否具备类人推理能力。研究通过实验对比了25个LLMs与人类在四种语言中的条件推理表现，发现LLMs在语义上准确，但缺乏人类的语用丰富性。**

- **链接: [https://arxiv.org/pdf/2605.21299](https://arxiv.org/pdf/2605.21299)**

> **作者:** Paolo Morosi; Nikoleta Pantelidou; Fritz Günther; Elena Pagliarini; Evelina Leivada
>
> **摘要:** Humans effortlessly go beyond literal meanings: If you mow the lawn, I will give you fifty dollars, is typically understood as implying that the speaker will pay only if the lawn is mowed, whereas If you are hungry, there is pizza in the oven implies that pizza is available regardless of the hearers hunger. Large Language Models - LLMs - show human-like performance on many tasks, yet it remains unclear whether they reason like humans. To address this, we conducted a population-matching experiment assessing how twentyfive LLMs compute conditional inferences across four languages, compared to an equal number of humans per language. We find that humans enrich logical reasoning through pragmatic inferences across languages. Model behavior is more variable. Some LLMs perfectly follow the truth-table of conditionals but they ignore pragmatic inferences, while others deviate from the truth-table, adhering to a single interpretation across the board, thus reflecting accurate rule-based processing but not human-like reasoning. Overall, LLMs are accurate semantic operators, but fail to capture the pragmatic enrichments characteristic of human reasoning. Crucially, LLM accuracy is neither predicted nor boosted by open vs. closed status, training orientation, or architecture type, suggesting that pragmatic reasoning is still an emerging ability in the cognitive toolkit of artificial systems.
>
---
#### [new 033] Task-Routed Mixture-of-Experts with Cognitive Appraisal for Implicit Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于隐式情感分析任务，旨在解决情感依赖上下文推断的问题。提出多任务学习框架，结合认知评估理论，提升情感预测效果。**

- **链接: [https://arxiv.org/pdf/2605.20916](https://arxiv.org/pdf/2605.20916)**

> **作者:** Yaping Chai; Haoran Xie; Joe S. Qin
>
> **备注:** 8 pages, 4 figures, and 3 tables
>
> **摘要:** Implicit sentiment analysis is challenging because sentiment toward an aspect is often inferred from events rather than expressed through explicit opinion words. Existing models typically learn from the final polarity label, which provides limited guidance for reasoning about sentiment from the context. Motivated by cognitive appraisal theory, we propose an appraisal-aware multi-task learning (MTL) framework for implicit sentiment analysis that provides polarity prediction with two complementary auxiliary tasks: implicit sentiment detection and cognitive rationale generation. However, training several objectives with different targets and sharing a single backbone across tasks in MTL limits flexibility and can lead to task interference. To reduce interference among these related but distinct objectives, we adopt task-level mixture-of-experts models in which all tasks share a common set of experts, and task identity controls the sparse combination of these experts. Our method builds on an encoder-decoder architecture and replaces a subset of encoder and decoder blocks with these sparse mixtures. We use a task-conditioned router to select sparse expert mixtures for each task, and a task-separated routing objective to encourage different tasks to learn distinct expert-selection patterns. Experimental results show that our model outperforms recently proposed approaches, with strong gains on the implicit sentiment subset. Our code is available at this https URL.
>
---
#### [new 034] Leveraging LLMs for Grammar Adaptation: A Study on Metamodel-Grammar Co-Evolution
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于模型驱动工程中的语法适应任务，旨在解决元模型演化后语法一致性问题。通过引入大语言模型自动适应新语法，提升适应效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.21465](https://arxiv.org/pdf/2605.21465)**

> **作者:** Weixing Zhang; Bowen Jiang; Rahul Sharma; Regina Hebig; Daniel Strüber
>
> **摘要:** In model-driven engineering, metamodel evolution leads to the need to adapt corresponding grammars to maintain consistency, which typically requires tedious manual work. Existing rule-based methods can achieve partial automation but have limitations when handling complex grammar scenarios. This paper proposes a Large Language Model-based approach that automatically applies adaptations to new grammars after evolution by learning grammar adaptations from previous versions. We evaluated this approach on six real-world Xtext domain-specific languages, using four DSLs as a training set to develop prompting strategies, two DSLs as a test set for validation, and conducting a longitudinal case study on QVTo. The evaluation used three Large Language Models (Claude Sonnet 4.5, ChatGPT 5.1, Gemini 3) and measured grammar adaptation quality from three dimensions: grammar rule-level adaptation consistency, output similarity, and metamodel conformance. Results show that on the test set, all three LLMs achieved 100% adaptation consistency and output similarity, while the rule-based approach achieved only 84.21% on DOT and 62.50% on Xcore. In the QVTo longitudinal study, the LLM-based approach successfully reused learned adaptations across all three evolution steps without manual grammar editing, while the rule-based approach required manual adjustments in two of three transitions. However, on large-scale grammars (EAST-ADL, 297 rules), LLMs' adaptation consistency was far below 90%. This study demonstrates the advantages of LLM-based approaches in handling complex grammar scenarios, while revealing their limitations in large-scale grammar adaptation.
>
---
#### [new 035] The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study
- **分类: cs.CL; cs.LG; stat.ME**

- **简介: 该论文属于自然语言处理中的实验研究任务，旨在解决LLM模拟用户时因干预导致的用户漂移问题，通过负控制结果诊断并调整角色设定以减少偏差。**

- **链接: [https://arxiv.org/pdf/2605.20767](https://arxiv.org/pdf/2605.20767)**

> **作者:** Victoria Lin; Taedong Yun; Maja Matarić; John Canny; Arthur Gretton; Alexander D'Amour
>
> **摘要:** Large language models (LLMs) show potential as simulators of human behavior, offering a scalable way to study responses to interventions. However, because LLMs are trained largely on observational data, interventions in experiments with LLM-simulated synthetic users can induce unintended shifts in latent user attributes, causing user drift where the implicit simulated population differs across treatment conditions, potentially distorting effect estimates. We formalize the confounding or selection bias that can arise due to user drift and show how intervention-dependent shifts can inflate or attenuate observed differences in user responses under intervention. To diagnose confounding, we propose using negative control outcomes--attributes that should remain invariant under intervention--to identify distribution shifts across intervention conditions, providing evidence of user drift. To mitigate drift, we study adjusting the persona specification by eliciting additional confounders, finding that targeted, setting-relevant confounders can substantially reduce bias across survey-style and multi-turn agent evaluations.
>
---
#### [new 036] HRM-Text: Efficient Pretraining Beyond Scaling
- **分类: cs.CL**

- **简介: 该论文提出HRM-Text，一种高效预训练方法，解决大模型依赖大量计算和数据的问题。通过架构创新和目标设计，实现低资源下的高性能模型训练。**

- **链接: [https://arxiv.org/pdf/2605.20613](https://arxiv.org/pdf/2605.20613)**

> **作者:** Guan Wang; Changling Liu; Chenyu Wang; Cai Zhou; Yuhao Sun; Yifei Wu; Shuai Zhen; Luca Scimeca; Yasin Abbasi Yadkori
>
> **摘要:** The current pretraining paradigm for large language models relies on massive compute and internet-scale raw text, creating a significant barrier to foundational research. In contrast, biological systems demonstrate highly sample-efficient learning through multi-timescale processing, such as the functional organization of the frontoparietal loop. Taking this as inspiration, we introduce HRM-Text, which replaces standard Transformers with a Hierarchical Recurrent Model (HRM) that decouples computation into slow-evolving strategic and fast-evolving execution layers. To stabilize this deep recurrence for language modeling, we introduce MagicNorm and warmup deep credit assignment. Furthermore, instead of standard raw-text pretraining, we train exclusively on instruction-response pairs using a task-completion objective and PrefixLM masking. Serving as an empirical existence proof of efficient pretraining, a 1B-parameter HRM-Text model trained from scratch on only 40 billion unique tokens and $1,500 budget achieves 60.7% on MMLU, 81.9% on ARC-C, 82.2% on DROP, 84.5% on GSM8K, and 56.2% on MATH. Despite utilizing roughly 100-900x fewer training tokens and 96-432x less estimated compute than standard baselines, HRM-Text performs competitively with 2-7B parameter open models. These results demonstrate that co-designing architectures and objectives can radically reduce the compute-to-performance ratio, making pretraining from scratch accessible to the broader research community.
>
---
#### [new 037] Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于对话系统任务，研究全双工语音对话模型的同步与轮流机制。通过实验分析模型在不同噪声下的表示同步和预判能力，以提升自然交互效果。**

- **链接: [https://arxiv.org/pdf/2605.20356](https://arxiv.org/pdf/2605.20356)**

> **作者:** Pablo Riera; Pablo Brusco; Cristina Kuo; Marcelo Sancinetti; S.R.K. Branavan
>
> **摘要:** Full-duplex spoken dialogue models (SDMs) can listen and speak simultaneously, enabling interaction dynamics closer to human conversation than turn-based systems. Inspired by neural coupling in human communication, we study how such models coordinate their internal representations during interaction. We simulate full-duplex dialogues between two instances of the pretrained \textit{Moshi} model under controlled conditions, manipulating channel noise and decoding bias. Synchronization is measured using Centered Kernel Alignment (CKA) across temporal lags, while anticipatory turn-taking cues are probed from delayed internal activations using causal LSTM models, from both speaker and listener perspectives. We find strong representational synchronization under no noise conditions, peaking near zero lag and degrading with noise, and we show that internal states encode anticipatory information that supports turn-taking prediction ahead of time.
>
---
#### [new 038] Under Pressure: Emotional Framing Induces Measurable Behavioral Shifts and Structured Internal Geometry in Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究情感框架对小型语言模型行为和内部表征的影响，通过实验分析不同情绪引导下的模型表现，探索可测量的提示敏感控制方向。**

- **链接: [https://arxiv.org/pdf/2605.20202](https://arxiv.org/pdf/2605.20202)**

> **作者:** Rana Muhammad Usman
>
> **备注:** 18 pages, 4 figures. Exploratory empirical study with fully local experiments on small open language models. Code and data: this https URL
>
> **摘要:** I study whether emotionally framed evaluation follow-ups change both the behavior and the calm-relative internal representations of small, locally deployed language models. Our main benchmark uses Qwen 3.5 0.8B on four impossible-constraint coding tasks and eight follow-up framings: calm, pressure, urgency, approval, shame, curiosity, encouragement, and threat. In the 0.8B eight-condition sweep (160 conversations), pressure produces the strongest shortcut markers (11/20 runs) and the clearest overfit pattern (3/20), while calm and curiosity preserve explicit honesty more often (7/20 and 6/20). For all seven non-baseline conditions, the corresponding calm-relative direction vectors peak at the final transformer layer. An exploratory PCA of the layer-23 direction vectors reveals a dominant first component (59.5% explained variance) aligned with a hand-labeled positive/negative split (cosine alignment 0.951); approval and urgency are nearly identical internally (cosine 0.957), whereas curiosity points away from urgency (-0.252). In a separate calm-vs.-pressure rerun used for scale comparison, Qwen 3.5 2B shows higher honest rates under calm framing and directionally consistent activation steering on a small 4-prompt A/B probe, whereas the 0.8B steering result reverses. I interpret these results as evidence for measurable prompt-sensitive control directions in small open models, while stopping short of claiming intrinsic emotional states.
>
---
#### [new 039] Retrieval-Augmented Long-Context Translation for Cultural Image Captioning: Gators submission for AmericasNLP 2026 shared task
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对原住民语言文化图像描述任务，提出两阶段翻译方法，通过检索增强提升翻译效果，显著提高多个语言的生成质量。**

- **链接: [https://arxiv.org/pdf/2605.20626](https://arxiv.org/pdf/2605.20626)**

> **作者:** Aashish Dhawan; Christopher Driggers-Ellis; Dzmitry Kasinets; Daisy Zhe Wang; Christan Grant
>
> **摘要:** We present the University of Florida Gators submission to the AmericasNLP 2026 shared task on cultural image captioning for Indigenous languages. Our two-stage pipeline generates a Spanish intermediate caption with Qwen2.5-VL, then produces the target-language caption using retrieval-augmented many-shot prompting with Gemini 2.5 Flash. We achieve 164.1%, 131.7%, and 122.6% improvements over the shared task baseline for Bribri, Guaraní, and Orizaba Nahuatl captioning, respectively, in our dev set evaluation and maintain >150% improvements for the Bribri and Orizaba Nahuatl languages in the test set evaluation. We find retrieval is highly language-dependent, beneficial only for large, in-domain corpora, and that synthetic data augmentation accounts for around 28 chrF++ of the dev set Guaraní performance gain. Our submission is the overall winner of the shared task, placing second out of five finalist submissions in human evaluations of target-language captions.
>
---
#### [new 040] Collocational bootstrapping: A hypothesis about the learning of subject-verb agreement in humans and neural networks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言习得研究，探讨如何利用统计信号辅助语法学习。通过模拟神经网络和分析儿童语言数据，验证了共现模式在主谓一致习得中的作用。**

- **链接: [https://arxiv.org/pdf/2605.20529](https://arxiv.org/pdf/2605.20529)**

> **作者:** Claire Hobbs; R. Thomas McCoy
>
> **备注:** Accepted to CoNLL
>
> **摘要:** In what ways might statistical signals in linguistic input assist with the acquisition of syntax? Here we hypothesize a mechanism called collocational bootstrapping, in which regularities in word co-occurrence patterns can provide cues to syntactic dependencies. We investigate whether this mechanism can support the acquisition of English subject-verb agreement. First, we simulate language acquisition by training neural networks on synthetic datasets that vary in how predictable their subject-verb pairings are. We find that there is a range of variability levels at which these statistical learners robustly learn subject-verb agreement. We then analyze the variability of subject-verb pairings in child-directed language, and we find that the variability in such data falls within the range that supported robust generalization in our computational simulations. Taken together, these results suggest that collocational bootstrapping is a viable learning strategy for the type of input that children receive.
>
---
#### [new 041] GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于医疗EHR结构检索任务，旨在解决云部署LLM在成本、延迟和合规性上的问题。工作包括在消费级硬件上评估GraphRAG，对比多个本地模型的表现。**

- **链接: [https://arxiv.org/pdf/2605.20815](https://arxiv.org/pdf/2605.20815)**

> **作者:** Peter Fernandes; Ria Kanjilal
>
> **备注:** 9 pages, 1 figure, 5 tables
>
> **摘要:** Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data is complex and strictly regulated, reliance on cloud-based large language models (LLMs) introduces challenges in cost, latency, and compliance. In this work, we present a systematic evaluation of GraphRAG for EHR schema retrieval using locally deployed open-source LLMs. We implement the Microsoft GraphRAG pipeline on real-world EHR schema documentation and benchmark four models, including Llama 3.1 (8B), Mistral (7B), Qwen 2.5 (7B), and Phi-4-mini (3.8B), each deployed via Ollama on a single consumer GPU (8 GB VRAM). We evaluate indexing efficiency, knowledge graph construction, query latency, answer quality, and hallucination under both global and local retrieval modes. Our results reveal substantial differences: Llama 3.1 produces the richest knowledge graph (1,172 entities), Qwen 2.5 achieves the best answer quality (3.3/5), Phi-4-mini fails to complete the pipeline due to structured-output errors, and Mistral exhibits degenerate repetition behavior. We further show that GraphRAG exhibits a practical capacity threshold, where models below approximately 7B parameters fail to reliably produce valid structured outputs and cannot complete the pipeline. In addition, indexing and answer quality are decoupled across models, and local retrieval consistently outperforms global summarization in both latency and factual grounding, with reduced hallucination. These findings demonstrate that GraphRAG is feasible on consumer hardware while highlighting the importance of model selection and retrieval design for robust deployment in regulated settings.
>
---
#### [new 042] Metaphors in Literary Post-Editing: Opening Pandora's Box?
- **分类: cs.CL**

- **简介: 该论文属于文学翻译质量评估任务，探讨NMT和LLM在文学文本中隐喻翻译的问题。研究显示，三分之一的隐喻被后期编辑修改，表明文学机器翻译存在挑战。**

- **链接: [https://arxiv.org/pdf/2605.21178](https://arxiv.org/pdf/2605.21178)**

> **作者:** Aletta G. Dorst; Mayra O. Nas; Katinka Zeven
>
> **备注:** This paper has been accepted for presentation at the EAMT Conference 2026, which will take place in Tilburg from June 15 to 18, 2026
>
> **摘要:** This paper investigates how post-editors of literary texts react and respond to the way metaphors have been translated by Neu ral Machine Translation (NMT) and Large Language Models (LLMs). The results show that one in three metaphors in the output were changed by the post-editors, demonstrating that the translation of fig urative language is indeed problematic in literary MT (LitMT). The responses indi cate that the post-editors were aware of overly literal translations, though mostly for multiword expressions. Moreover, at times they found it difficult to determine whether solutions were acceptable. They rated the overall quality of the MT out put as quite poor and stated that the post editing was more work and more effort than it would have been translating from scratch. This supports previous studies ar guing that post-editing constrains transla tors in their creativity and diminishes their sense of text ownership.
>
---
#### [new 043] Leveraging Large Language Models for Sentiment Analysis: Multi-Modal Analysis of Decentraland's MANA Token
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在通过整合Discord社区情绪与多模态数据提升加密货币价格预测。工作包括构建LSTM模型并验证情绪信号的预测价值。**

- **链接: [https://arxiv.org/pdf/2605.20192](https://arxiv.org/pdf/2605.20192)**

> **作者:** Xintong Wu; Peiting Tsai; Jing Yuan; Michael Yu; Greg Sun; Luyao Zhang
>
> **摘要:** Decentraland, a decentralized virtual reality platform operating within the expanding Metaverse ecosystem, utilizes its native MANA token to facilitate virtual asset transactions and governance. This study investigates the integration of Discord community sentiment with multi-modal financial data to enhance cryptocurrency price prediction within virtual world economies. We address: (1) identifying sentiment patterns within Decentraland's Discord community, and (2) evaluating the impact of multi-modal features on token return forecasting. Using a BERT-based large language model for sentiment analysis, we develop two LSTM architectures: a baseline incorporating historical prices and a multi-modal variant integrating sentiment scores, trading volume, and market capitalization. Results indicate predominantly neutral community sentiment with a positive skew. The multi-modal model significantly outperforms the price-only baseline in prediction accuracy. These findings demonstrate the predictive value of community-derived signals for virtual economy forecasting and establish a foundation for future research at the intersection of immersive virtual environments, natural language processing, and cryptocurrency market analysis.
>
---
#### [new 044] When Irregularity Helps: A Subclass Analysis of Inductive Bias in Neural Morphology
- **分类: cs.CL**

- **简介: 该论文研究神经形态生成中的归纳偏差，针对日语动词过去式变形任务，分析罕见子类错误集中问题，发现特定不规则类型导致模型不稳定，提出需细化子类评估。**

- **链接: [https://arxiv.org/pdf/2605.20558](https://arxiv.org/pdf/2605.20558)**

> **作者:** Wen Zhang
>
> **摘要:** Neural morphological generation systems often achieve high aggregate accuracy on benchmark datasets, yet such performance can conceal systematic errors concentrated in rare morphological subclasses. We examine Japanese past-tense verb inflection and show that a very small, structurally specific irregular subtype (<1% of data) accounts for a disproportionate share of model errors. Controlled ablation experiments demonstrate that removing this subtype yields larger improvements in generalization than removing all irregular verbs, indicating that not all irregularity contributes equally to model instability. These findings suggest that error concentration is driven by the interaction between extreme low-frequency morphological patterns and specific morphophonological processes, particularly gemination. We argue that morphological evaluation should incorporate finer-grained subclass analysis beyond standard conjugation categories.
>
---
#### [new 045] SCRIBE: Diagnostic Evaluation and Rich Transcription Models for Indic ASR
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决传统词错误率（WER）评估不足的问题。通过引入SCRIBE框架，实现更细致的错误分类与诊断，提升对印地语等语言的识别效果。**

- **链接: [https://arxiv.org/pdf/2605.20712](https://arxiv.org/pdf/2605.20712)**

> **作者:** Kavya Manohar; Arghya Bhattacharya; Kush Juvekar; Kumarmanas Nethil
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Automatic speech recognition replaces typing only when correction costs less than manual entry, a threshold determined by error types, not counts: fixing a misrecognized domain term costs far more than inserting a comma. Word error rate (WER) fails on two fronts: it collapses distinct error categories into a single scalar, and it structurally penalizes agglutinative languages where valid sandhi merges inflate scores. We introduce SCRIBE, a diagnostic framework that provides categorical error decomposition into lexical, punctuation, numeral, and domain-entity rates through sandhi-tolerant alignment with domain vocabulary injection. Human validation confirms SCRIBE aligns with expert judgment where WER does not. We release SCRIBE, an LLM curation pipeline, benchmarks, and open-weight rich transcription models for Hindi, Malayalam, and Kannada.
>
---
#### [new 046] Do No Harm? Hallucination and Actor-Level Abuse in Web-Deployed Medical Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于医疗AI安全评估任务，旨在检测医学大模型的幻觉和滥用问题。通过评估大量模型，发现其存在事实不准确、政策违规等问题，并提出评估框架与数据集。**

- **链接: [https://arxiv.org/pdf/2605.20591](https://arxiv.org/pdf/2605.20591)**

> **作者:** Sunday Oyinlola Ogundoyin; Muhammad Ikram; Rahat Masood
>
> **摘要:** Medical large language models (LLMs), including custom medical GPTs (MedGPTs) and open-source models, are increasingly deployed on web platforms to provide clinical guidance. However, they pose risks of hallucination, policy noncompliance, and unsafe design. We conduct a large-scale assessment of 6,233 MedGPTs, evaluating a stratified sample of 1,500, together with 10 open-source LLMs. We introduce two frameworks: MedGPT-HEval for hallucination detection and an LLM-based pipeline for assessing policy violations and developer intent. Our results show that 25-30% of MedGPTs exhibit low factual accuracy, with bottom- and middle-tier models at highest risk; 33.6-54.3% violate operational thresholds, and 57.06% of Action-enabled models lack adequate privacy disclosures. Compared with open-source models, MedGPTs achieve higher factual accuracy and semantic alignment, though open-source models are more stable. These results reveal systemic gaps in hallucination and compliance, highlighting the need for multi-metric evaluation and stronger safeguards. We release HAA-MedGPT, a structured dataset that supports future research on the safety of web-facing medical LLMs.
>
---
#### [new 047] Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估LLMs在社交媒体文本分析中的表现。通过构建基准测试框架，发现LLMs在处理大规模数据和复杂任务时存在性能下降问题。**

- **链接: [https://arxiv.org/pdf/2605.21338](https://arxiv.org/pdf/2605.21338)**

> **作者:** Yuefeng Shi; Nedjma Ousidhoum; Jose Camacho-Collados
>
> **摘要:** LLMs have demonstrated exceptional proficiency in a wide range of NLP tasks. However, a notable gap remains in practical data analysis scenarios, particularly when LLMs are required to process long sequences of unstructured documents, such as news feeds or, as specifically addressed in this paper, social media posts. To empirically assess the effectiveness of LLMs in this setting, we introduce a question-based evaluation framework comprising 470 manually curated questions designed to evaluate LLMs' semantic understanding and reasoning abilities over aggregated text data. We apply our benchmark on diverse Twitter datasets covering various NLP tasks, including sentiment analysis, hate speech detection, and emotion recognition. Our results reveal that the performance depends heavily on input scale and the complexity of the data sources, declining noticeably in multi-label or target-dependent scenarios. In addition, as task complexity increases, performance drops progressively from basic semantic existence identification to more demanding operations such as comparison, counting, and calculation. Furthermore, as the input size grows beyond 500 instances, we identify a common limitation across LLMs, particularly Open-weights models: performance degrades substantially, especially on numerical tasks. These findings highlight critical architectural bottlenecks in current LLMs for performing rigorous quantitative analysis over large text collections.
>
---
#### [new 048] ACL-Verbatim: hallucination-free question answering for research
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于信息提取任务，旨在解决LLM在学术研究中产生幻觉的问题。通过构建数据集并训练模型，实现从论文中准确提取相关文本片段。**

- **链接: [https://arxiv.org/pdf/2605.21102](https://arxiv.org/pdf/2605.21102)**

> **作者:** Gábor Recski; Szilveszter Tóth; Nadia Verdha; István Boros; Ádám Kovács
>
> **备注:** 13 pages
>
> **摘要:** Academic researchers need efficient and reliable methods for collecting high-quality information from trusted sources, but modern tools for AI-assisted research still suffer from the tendency of Large Language Models (LLMs) to produce factually inaccurate or nonsensical output, commonly referred to as hallucinations. We apply the extractive question answering system VerbatimRAG to research papers in the ACL Anthology, directly mapping user queries to verbatim text spans in retrieved documents. We contribute a novel ground truth dataset for the task of mapping user queries to relevant text spans in research papers, and use it to train and evaluate a variety of extractive models. Human annotation is performed by NLP researchers and is based on synthetic user queries generated using a custom pipeline based on the ScIRGen methodology, paired with chunks of research papers retrieved by VerbatimRAG. On this benchmark, a 150M-parameter ModernBERT token classifier trained on silver supervision from our pipeline achieves the best word-level F1 (53.6), ahead of the strongest evaluated LLM extractor (48.7).
>
---
#### [new 049] Assessing socio-economic climate impacts from text data
- **分类: cs.CL**

- **简介: 该论文属于文本数据分析任务，旨在解决气候灾害社会经济影响评估中的方法不统一问题，通过总结实践、分析挑战并提出建议，提升研究的透明度和可比性。**

- **链接: [https://arxiv.org/pdf/2605.20793](https://arxiv.org/pdf/2605.20793)**

> **作者:** Mariana Madruga de Brito; Brielen Madureira; Taís Maria Nunes Carvalho; Damien Delforge; Aglaé Jézéquel; Murathan Kurfalı; Ni Li; Gabriele Messori; Joakim Nivre; Barbara Pernici; Niko Speybroeck; Stefano Terzi; Wim Thiery; Bram Valkenborg; Jingxian Wang; Shorouq Zahra; Jakob Zscheischler; Jan Sodoge
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in natural language processing (NLP) and large language models (LLMs) have enabled the systematic use of large-scale textual data from news, social media, and reports to create datasets with socio-economic impacts of climate hazards such as floods, droughts, storms, and multi-hazard events. As the field of text-as-data for impact assessment expands, so does its methodological complexity. Yet research remains fragmented, with no clear guidelines for defining what constitutes an impact, handling temporal and spatial biases, and selecting appropriate modeling and post-processing strategies. This lack of coherence limits transparency and comparability across studies. Here, we address this gap by synthesising common practices, describing key challenges specific to the use of text-as-data methods for analyzing socio-economic impact data, and proposing recommendations to address them. By providing guidance on best practices, we aim to support the construction of robust text-derived socio-economic impact datasets that can more accurately inform disaster risk management and attribution studies.
>
---
#### [new 050] "I didn't Make the Micro Decisions": Measuring, Inducing, and Exposing Goal-Level AI Contributions in Collaboration
- **分类: cs.CL**

- **简介: 该论文属于人机协作任务，旨在解决AI在目标形成中的贡献度评估问题。提出CoTrace框架，追踪AI在对话中的直接与间接影响，揭示用户对AI作用的误判。**

- **链接: [https://arxiv.org/pdf/2605.21363](https://arxiv.org/pdf/2605.21363)**

> **作者:** Eunsu Kim; Jessica R. Mindel; Kyungjin Kim; Sherry Tongshuang Wu
>
> **摘要:** As large language models (LLMs) increasingly shape how users form, refine, and extend their goals, attributing contributions in human-AI collaboration becomes critical for users calibrating their own reliance and for evaluators assessing AI-assisted work. Yet existing methods focus on final artifacts, missing the process through which goals themselves are jointly shaped. We introduce a goal-level attribution framework, CoTrace, that decomposes explicit goals into verifiable requirements and traces both direct contributions and indirect influences across dialogue turns. Applying CoTrace to 638 real-world collaboration logs, we find that while models account for only 11-26% of goal-shaping contribution, they contribute substantially more on introducing lower-level concrete requirements, and make various kinds of indirect contributions. Through controlled simulations, we show that interaction design choices significantly affect model goal-shaping behavior. In a user study, exposing participants to goal-level analyses shifts their perceived contributions by nearly 2 points on a 5-point scale, revealing systematic miscalibration in how users understand their own AI-assisted work.
>
---
#### [new 051] TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，针对大模型提示过拟合问题，提出TextReg框架通过正则化优化文本空间，提升模型在分布外数据上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.21318](https://arxiv.org/pdf/2605.21318)**

> **作者:** Lucheng Fu; Ye Yu; Yiyang Wang; Yiqiao Jin; Haibo Jin; B. Aditya Prakash; Haohan Wang
>
> **备注:** Code: this https URL
>
> **摘要:** Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-specific rules, and generalize poorly beyond the training distribution. We study this failure mode as prompt distributional overfitting and argue that it reflects a lack of representation control in discrete text-space optimization. We formalize this view through representational inefficiency, a dual-factor measure that decomposes prompt inefficiency into capacity cost and scope narrowness, attributing distributional prompt overfitting to their coupled growth during optimization. We propose TextReg, a regularization framework that realizes a soft-penalty objective through regularized textual gradients, combining Dual-Evidence Gradient Purification, Semantic Edit Regularization, and Regularization-Guided Prompt Update. Across multiple reasoning benchmarks, TextReg substantially improves out-of-distribution (OOD) generalization, with accuracy gains of up to +11.8% over TextGrad and +16.5% over REVOLVE.
>
---
#### [new 052] Quantifying the cross-linguistic effects of syncretism on agreement attraction
- **分类: cs.CL**

- **简介: 该论文属于语言加工研究，旨在解释不同语言中形态同形现象对一致吸引效应的影响差异。通过分析语言模型的 surprisal 和 attention entropy 指标，探讨语言间这一现象的变异。**

- **链接: [https://arxiv.org/pdf/2605.21403](https://arxiv.org/pdf/2605.21403)**

> **作者:** Utku Turk; Eva Neu
>
> **备注:** SCiL Conference Paper
>
> **摘要:** Agreement attraction errors, in which a verb erroneously agrees with an intervening noun rather than its grammatical head, are amplified by morphological syncretism in some languages (English, German, Russian) but not others (Turkish, Armenian), a cross-linguistic pattern without a principled account. We use surprisal and attention entropy from large language models as processing proxies to investigate this variation across four languages. LLM-derived measures replicate behavioral findings in English and German (syncretism modulates attraction), align with Turkish null results (no modulation), and partially capture Russian patterns. We discuss further directions for better understanding why syncretism affects agreement attraction differently across languages.
>
---
#### [new 053] Stage-Audit: Auditable Source-Frontier Discovery for Cross-Wiki Tables
- **分类: cs.CL**

- **简介: 该论文针对跨维基表格构建任务，解决LLM生成表格时源引用不实的问题。通过分权机制和审计策略提升源前沿精度与F1值。**

- **链接: [https://arxiv.org/pdf/2605.20478](https://arxiv.org/pdf/2605.20478)**

> **作者:** Chen Shen
>
> **备注:** 9 pages, 2 figures, 3 tables. Accepted at the ACM CAIS 2026 Workshop on AI Agents for Discovery in the Wild
>
> **摘要:** LLM-curated tables can appear source-grounded while containing unsupported rows: the curator may recall entries from parametric memory and retroactively attach page-level citations that are not the actual source. We study this hazard in Seed2Frontier discovery: the task of finding complement Wikipedia pages from a seed page to assemble a structured table. Stage-Audit addresses it with disjoint curator-auditor write rights, a row-level source-citation gate, and a 12-check audit taxonomy over keys, schema, source roles, cardinality, and scope. On a curated 51-instance Seed2Frontier evaluation set spanning 15 top-level domains, Stage-Audit improves source-frontier precision over a vanilla LLM curator from 0.356 to 0.505 (+42% relative) and F1 from 0.334 to 0.451 (+35%), while maintaining explicit per-row source traceability. The vanilla-LLM-vs-Stage-Audit comparison isolates the policy contribution rather than LLM-based discovery in general.
>
---
#### [new 054] Post-Hoc Understanding of Metaphor Processing in Decoder-Only Language Models via Conditional Scale Entropy
- **分类: cs.CL**

- **简介: 该论文属于机制可解释性任务，旨在研究解码器模型如何处理隐喻。通过引入条件尺度熵（CSE），分析模型在不同层对隐喻词的多尺度响应，揭示其结构特征。**

- **链接: [https://arxiv.org/pdf/2605.21391](https://arxiv.org/pdf/2605.21391)**

> **作者:** Lawhori Chakrabarti; Jennifer Johnson-Leung; Bert Baumgaertner; Aleksandar Vakanski; Min Xian; Boyu Zhang
>
> **备注:** 18 pages, 3 figures, submitted to ICPR workshop
>
> **摘要:** Metaphor requires a language model to resolve a token whose contextual meaning diverges from its basic literal sense. Understanding how transformer models organize this reinterpretation across depth remains an open problem in mechanistic interpretability. We introduce conditional scale entropy (CSE), a wavelet-derived measure of how broadly transformer computation engages across frequency scales at each layer position. Two theorems establish that CSE is invariant to update magnitude, isolating the structural pattern of updates from their intensity. Using CSE, we find that metaphorical tokens produce significantly higher spectral breadth than literal tokens at contiguous layer positions on every decoder-only architecture tested, from 124M to 20B parameters (GPT-2 family, LLaMA-2 7B, GPT-oss 20B). The effect survives cluster-based permutation correction, recurs in the early-to-mid relative depth range across models, and converges with an independent analysis of 200 naturalistic VUA pairs. Specificity controls further show that the effect is not explained by semantic complexity or by matched propositional content. These results identify multi-scale coordination as a consistent signature of metaphorical language processing in the decoder-only architectures examined, and establish CSE as a principled tool for characterizing cross-depth structure in transformers.
>
---
#### [new 055] Mem-$π$: Adaptive Memory through Learning When and What to Generate
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mem-π，解决大语言模型代理的自适应记忆问题，通过生成式方法动态提供任务指导，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2605.21463](https://arxiv.org/pdf/2605.21463)**

> **作者:** Xiaoqiang Wang; Chao Wang; Hadi Nekoei; Christopher Pal; Alexandre Lacoste; Spandana Gella; Bang Liu; Perouz Taslakian
>
> **备注:** Work in progress
>
> **摘要:** We present Mem-$\pi$, a framework for adaptive memory in large language model (LLM) agents, where useful guidance is generated on demand rather than retrieved from external memory stores. Existing memory-augmented agents typically rely on similarity-based retrieval from episodic memory banks or skill libraries, returning static entries that often misalign with the current context. In contrast, Mem-$\pi$ uses a dedicated language or vision-language model with its own parameters, separate from the downstream agent, to generate context-specific guidance for complex tasks. Conditioned on the current agent context, the model jointly decides when to produce guidance and what guidance to produce. We train it with a decision-content decoupled reinforcement learning (RL) objective, enabling it to abstain when generation would not help and otherwise produce concise, useful guidance. Across diverse agentic benchmarks spanning web navigation, terminal-based tool use, and text-based embodied interaction, Mem-$\pi$ consistently outperforms retrieval-based and prior RL-optimized memory baselines, achieving over 30% relative improvement on web navigation tasks.
>
---
#### [new 056] Towards Context-Invariant Safety Alignment for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型安全对齐任务，旨在解决模型在不同上下文中行为不一致的问题。通过引入Anchor Invariance Regularization（AIR），提升模型对齐的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20994](https://arxiv.org/pdf/2605.20994)**

> **作者:** Yixu Wang; Yang Yao; Xin Wang; Yifeng Gao; Yan Teng; Xingjun Ma; Yingchun Wang
>
> **备注:** ICML 2026
>
> **摘要:** Preference-based post-training aligns LLMs with human intent, yet safety behavior often remains brittle. A model may refuse a harmful request in a standard prompt but comply when the same intent is wrapped in adversarial wording. We suggest that robust safety requires context-invariant alignment, where behavior depends on the underlying intent rather than surface form. Enforcing invariance is difficult in alignment because not all training signals are equally trustworthy; for some prompt variants we can obtain verifiable feedback (e.g., multiple-choice), while for open-ended variants we typically rely on noisy, gameable reward proxies (e.g., learned judges). As a result, standard symmetric invariance regularizers can reduce cross-context discrepancies by lowering performance on reliable variants instead of improving open-ended robustness. To address this, we introduce Anchor Invariance Regularization (AIR), which treats verifiable prompts as anchors and uses a stop-gradient target to regularize only the open-ended variants toward the anchor performance. AIR is implemented as a plug-in auxiliary loss and combined with group-based preference optimization (e.g., GRPO) via heterogeneous prompt grouping. Across Safety, Moral Reasoning, and Math, AIR improves context invariance, boosting in-distribution group accuracy by 12.71% and out-of-distribution consistency by 33.49%, making safety constraints robust to adversarial framings.
>
---
#### [new 057] SymbolicLight V1: Spike-Gated Dual-Path Language Modeling with High Activation Sparsity and Sub-Billion-Scale Pre-Training Evidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SymbolicLight V1，一种高稀疏激活的双路径语言模型，解决传统模型在性能与稀疏性间的平衡问题。任务为语言建模，通过双路径结构提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.21333](https://arxiv.org/pdf/2605.21333)**

> **作者:** Ting Liu
>
> **备注:** 35 pages, 5 figures, 25 tables; public code and model artifacts linked in manuscript
>
> **摘要:** Natively trained spiking language models struggle to combine Transformer-like language quality, stable multi-domain pre-training, and high activation sparsity. We present SymbolicLight V1, a spike-gated dual-path language model that combines binary Leaky Integrate-and-Fire spike dynamics with a continuous residual stream. Its Dual-Path SparseTCAM module replaces dense self-attention with an exponential-decay aggregation path for long-range memory and a spike-gated local attention path for short-range precision, complemented by a dynamic context-conditioned decoding head and a bilingual tokenizer. A 194M-parameter SymbolicLight V1 model trained from scratch on a 3B-token Chinese-English corpus reaches held-out validation PPL 8.88-8.93 across four independent runs at >89% per-element activation sparsity. It trails GPT-2 201M by 7.7% in PPL while surpassing GPT-2 124M under the reported comparison. Component ablations at matched 0.5B-token training budgets show that the spike-gated local attention path is the largest contributor, and that replacing LIF dynamics with a deterministic top-k mask at matched sparsity causes a larger degradation, indicating that temporal integration rather than sparsity alone drives performance. We also report a 0.8B-parameter scale-up run trained on 48.8B tokens as evidence of optimization and sparsity preservation, not as a primary quality comparison. Current dense-hardware inference is slower than GPT-2, so neuromorphic deployment is presented as a future sparsity-driven opportunity rather than an achieved hardware speedup.
>
---
#### [new 058] Interpretable Discriminative Text Representations via Agreement and Label Disentanglement
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于文本分类任务，旨在解决可解释性与标签混淆问题。提出LFD方法，通过语义对比和标注一致性筛选特征，提升模型的可审计性。**

- **链接: [https://arxiv.org/pdf/2605.20693](https://arxiv.org/pdf/2605.20693)**

> **作者:** Tong Wang; Yiqing Xu; Leo Yang Yang
>
> **摘要:** Interpretable text representations should expose coordinates that are not only predictive, but also meaningful enough for independent auditors to apply. Existing discriminative representations often use anonymous embedding directions, while concept-bottleneck and LLM-assisted methods attach natural-language names to features without ensuring that those definitions are reproducible or distinct from the target label. We propose an operational criterion for interpretable discriminative text representations: each coordinate should satisfy conceptual clarity, measured by chance-adjusted agreement between independent annotators applying the feature definition, and label disentanglement, meaning the feature should not merely paraphrase the prediction target. We instantiate this criterion in LLM-assisted Feature Discovery (LFD), an iterative method that proposes lexical and semantic features from contrastive outcome-opposed text pairs, screens candidates using cross-LLM Cohen's $\kappa$, and selects features by residual held-out predictive gain. A stylized analysis connects the $\kappa$ screen to a per-feature annotation-noise bound, formalizing agreement as a reliability check. Across ten text-classification tasks spanning seven corpora, LFD matches the predictive performance of a strong text bottleneck baseline while producing substantially clearer and less label-entangled features. Human audits with 232 raters show that LFD features achieve higher human--human and human--LLM agreement than baseline concepts, and raters consistently judge them as less label-leaking. These results suggest that agreement-tested, label-disentangled coordinates provide a practical auditability standard for interpretable text classification.
>
---
#### [new 059] Evaluating Speech Articulation Synthesis with Articulatory Phoneme Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决合成语音质量评估难题。通过使用音素识别作为代理指标，评估语音发音合成效果。**

- **链接: [https://arxiv.org/pdf/2605.20920](https://arxiv.org/pdf/2605.20920)**

> **作者:** Vinicius Ribeiro; Yves Laprie
>
> **备注:** Accepted for publication at the European Signal Processing Conference (EUSIPCO), 2026
>
> **摘要:** Recent advances in machine learning and the availability of articulatory datasets allow vocal tract synthesis to be conditioned on phonetic sequences, a primary task of articulatory speech synthesis. However, quality assessment needs a better definition. Generally, ranking generative models is tricky due to subjectivity. However, articulatory synthesis has the additional difficulty of requiring specialized knowledge in vocal tract anatomy and acoustics. To address this problem, this paper proposes to evaluate speech articulation synthesis using phoneme recognition as a proxy. Our hypothesis is that phoneme recognition using articulatory features better captures nuances in phoneme production, such as correct places of articulation, which traditional metrics (e.g., point-wise distance metrics) do not. We train a neural network with acoustic and articulatory features extracted from a single-speaker RT-MRI dataset. Then, we compare the recognition performance when testing the model with different synthetic articulatory features. Our results show that our articulatory feature set is phonetically rich and helps exploring additional dimensions on speech articulation synthesis.
>
---
#### [new 060] JobArabi: An Arabic Corpus and Analysis of Job Announcements from Social Media
- **分类: cs.CL**

- **简介: 该论文构建了阿拉伯语招聘公告语料库JobArabi，用于分析社交媒体上的就业信息。任务属于自然语言处理与社会科学研究，解决阿拉伯语招聘语言的特征及变化问题。**

- **链接: [https://arxiv.org/pdf/2605.20960](https://arxiv.org/pdf/2605.20960)**

> **作者:** Wajdi Zaghouani; Shimaa Amer Ibrahim; Mabrouka Bessghaier; Houda Bouamor
>
> **备注:** Accepted at LREC 2026 Main Conference
>
> **摘要:** This paper introduces JobArabi, a large-scale corpus of Arabic job announcements collected from social media between January 2024 and October 2025. The dataset contains 20,528 public posts from X and captures more than two years of employment-related discourse across Arabic-speaking online communities. The corpus was compiled using a linguistically informed query framework covering 21 Arabic keyword families that reflect gendered, plural, formal, and dialectal expressions of recruitment language. The resulting dataset includes posts from institutional, commercial, and individual accounts and provides metadata such as timestamps, engagement indicators, and geolocation when available, enabling temporal and regional analysis of employment discourse. Quantitative analysis reveals several sociolinguistic patterns in online recruitment, including the persistence of gendered hiring language, regional variation in occupational demand, and the emotional framing of recruitment messages. These findings highlight the potential of Arabic social media as a resource for studying labor market communication and linguistic change. The JobArabi corpus, together with documentation and collection scripts, will be released to support research in Arabic NLP, computational social science, and digital labor studies.
>
---
#### [new 061] Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言到API的转换任务，解决非技术用户使用企业分析API的难题。通过构建一个基于LLM的智能代理系统，实现安全、合规的查询执行与可视化生成。**

- **链接: [https://arxiv.org/pdf/2605.21027](https://arxiv.org/pdf/2605.21027)**

> **作者:** Gundeep Singh; Parsa Kavehzadeh; Jing Xia; Xue-Yong Fu; Julien Bouvier Tremblay; Md Tahmid Rahman Laskar; Vincent Lum; Shashi Bhushan TN
>
> **备注:** The first four authors contributed equally to this work
>
> **摘要:** Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration.
>
---
#### [new 062] Fine-grained Claim-level RAG Benchmark for Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律领域RAG系统评估任务，旨在解决现有评估框架缺乏细粒度分析及语言、用户群体局限的问题。工作包括构建多语言、多用户类型的ClaimRAG-LAW数据集，并进行细粒度性能分析。**

- **链接: [https://arxiv.org/pdf/2605.21071](https://arxiv.org/pdf/2605.21071)**

> **作者:** Souvick Das; Sallam Abualhaija; Domenico Bianculli
>
> **摘要:** The rapid progress of large language models (LLMs) is shifting semantic search toward a question-answering paradigm, where users ask questions and LLMs generate responses. In high-stake domains such as law, retrieval-augmented generation (RAG) is commonly used to mitigate hallucinations in generated responses. Nonetheless, prior work shows that RAG systems, whether general-purpose or legal-specific, still hallucinate at varying rates, making fine-grained evaluation essential. Despite the need, existing evaluation frameworks for legal RAG systems lack the granularity required to provide detailed analysis of retrieval and generation performance separately. Moreover, current benchmarks are largely English-only and centered on legal expert queries, overlooking non-expert needs. We introduce ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and English, targets both experts and non-experts, and includes diverse question types reflecting realistic scenarios. We further apply a fine-grained evaluation framework of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation, and claim-level analysis in the legal domain.
>
---
#### [new 063] Building a Custom Taxonomy of AI Skills and Tasks from the Ground Up with Job Postings
- **分类: cs.CL**

- **简介: 该论文属于知识组织任务，旨在构建AI技能与任务的定制分类体系。通过分析职位招聘数据，探索有效数据筛选方法，提升分类准确性与覆盖性。**

- **链接: [https://arxiv.org/pdf/2605.21029](https://arxiv.org/pdf/2605.21029)**

> **作者:** Stephen Meisenbacher; Peter Norlander
>
> **备注:** 14 pages, 2 figures, 8 tables. Accepted to CustomNLP4U 2026
>
> **摘要:** Utilizing LLMs for automated taxonomy construction presents a clear opportunity for the comprehensive, yet efficient mapping of potentially complex domains. When contending with high volumes of rapidly growing corpora, however, it becomes unclear how to best leverage such data for optimal taxonomy construction. Taking the case of systematizing AI skills in the workplace, we use two large-scale job postings corpora to investigate key design decisions for the inclusion (or exclusion) of data points for taxonomy construction. We propose TaxonomyBuilder as a blueprint for our systematic study, with which we evaluate various configurations of custom, data-informed, and hierarchical taxonomies. We demonstrate that less data can provide more clarity: filtering inputs to TaxonomyBuilder provides better domain-specific coverage than offering unfiltered inputs to clustering and LLM-enhanced hierarchical taxonomy labeling tools.
>
---
#### [new 064] Findings of the Fifth Shared Task on Multilingual Coreference Resolution: Expanding Datasets for Long-Range Entities
- **分类: cs.CL**

- **简介: 该论文属于多语言共指消解任务，旨在解决长距离实体的共指问题。通过扩展数据集和引入新语言，评估不同系统的表现。**

- **链接: [https://arxiv.org/pdf/2605.21369](https://arxiv.org/pdf/2605.21369)**

> **作者:** Michal Novák; Miloslav Konopík; Anna Nedoluzhko; Martin Popel; Ondřej Pražák; Jakub Sido; Milan Straka; Zdeněk Žabokrtský; Daniel Zeman
>
> **备注:** Accepted to CODI-CRAC 2026
>
> **摘要:** This paper describes the fifth edition of the Shared Task on Multilingual Coreference Resolution, held in conjunction with the CODI-CRAC 2026 workshop. Building on previous iterations, the task required participants to develop systems capable of mention identification and identity-based coreference clustering. The 2026 edition specifically emphasizes long-range entities, defined as coreferential chains spanning significant distances, across many words and sentences. The task expanded its linguistic scope by incorporating five new datasets and two additional languages. These additions leverage version 1.4 of CorefUD, a harmonized multilingual collection comprising 27 datasets in 19 languages. In total, ten systems participated, including four LLM-based approaches (three fine-tuned models and one few-shot approach). While traditional systems still maintained their lead, LLMs demonstrated significant potential, suggesting they may soon challenge established approaches in future editions.
>
---
#### [new 065] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control
- **分类: cs.CL**

- **简介: 该论文属于车载助手评估任务，旨在解决LLM在韩语本地化中的语言精确性和交互可靠性问题。通过提出评估框架，分析模型在敬语控制和对话策略上的表现。**

- **链接: [https://arxiv.org/pdf/2605.21086](https://arxiv.org/pdf/2605.21086)**

> **作者:** Seogyeong Jeong; Kiwoong Park; Seyoung Song; Eunsu Kim; Ken E. Friedl; Jaeho Kim; Alice Oh
>
> **备注:** To appear in ACL 2026 Industry Track
>
> **摘要:** While Large Language Models (LLMs) are increasingly integrated into in-vehicle conversational systems, identifying the optimal model remains challenging due to the lack of domain-specific evaluation standards tailored to real-world deployment requirements. In this paper, we propose a novel evaluation framework for in-vehicle assistants, with a particular focus on Korean-language localization. Our empirical analysis reveals notable patterns in model behavior. First, fine-grained Korean honorific control remains unstable in current LLMs, indicating that precise speech-level realization must be explicitly evaluated in localization settings. Second, models exhibit weaker performance in strategic conversational metrics like clarification and proactivity. Our analysis suggests this stems from the inherent subjective complexity of these tasks, where our framework adopts a conservative evaluation stance to prioritize reliability. Together, our findings underscore that automotive AI must move beyond general competence toward precise linguistic tailoring and reliable, safety-oriented interaction management.
>
---
#### [new 066] On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI评估任务，旨在分析AI评审员的能力与局限。通过专家标注研究，比较AI与人类评审的差异，揭示AI作为辅助工具的潜力与不足。**

- **链接: [https://arxiv.org/pdf/2605.20668](https://arxiv.org/pdf/2605.20668)**

> **作者:** Seungone Kim; Dongkeun Yoon; Kiril Gashteovski; Juyoung Suk; Jinheon Baek; Pranjal Aggarwal; Ian Wu; Viktor Zaverkin; Spase Petkoski; Daniel R. Schrider; Ilija Dukovski; Francesco Santini; Biljana Mitreska; Yong Jeong; Kyeongha Kwon; Young Min Sim; Dragana Manasova; Arthur Porto; Biljana Mojsoska; Makoto Takamoto; Marko Shuntov; Ruoqi Liu; Hyunjoo Jenny Lee; Niyazi Ulas Dinç; Yehhyun Jo; Sunkyu Han; Chungwoo Lee; Huishan Li; Esther H. R. Tsai; Ergun Simsek; Khushboo Shafi; Yeonseung Chung; Jihye Park; Aleksandar Shulevski; Henrik Christiansen; Yoosang Son; Elly Knight; Amanda Montoya; Jeongyoun Ahn; Christian Langkammer; Heera Moon; Changwon Yoon; Nikola Stikov; Mooseok Jang; Edward Choi; Junhan Kim; Yeon Sik Jung; Woo Youn Kim; Jae Kyoung Kim; Ishraq Md Anjum; Hyun Uk Kim; Drew Bridges; Carolin Lawrence; Xiang Yue; Alice Oh; Akari Asai; Sean Welleck; Graham Neubig
>
> **备注:** Work in progress
>
> **摘要:** With the advancement of AI capabilities, AI reviewers are beginning to be deployed in scientific peer review, yet their capability and credibility remain in question: many scientists simply view them as probabilistic systems without the expertise to evaluate research, while other researchers are more optimistic about their readiness without concrete evidence. Understanding what AI reviewers do well, where they fall short, and what challenges remain is essential. However, existing evaluations of AI reviewers have focused on whether their verdicts match human verdicts (e.g., score alignment, acceptance prediction), which is insufficient to characterize their capabilities and limits. In this paper, we close this gap through a large-scale expert annotation study, in which 45 domain scientists in Physical, Biological, and Health Sciences spent 469 hours rating 2,960 individual criticisms (each targeting one specific aspect of a paper) from human-written and AI-generated reviews of 82 Nature-family papers on correctness, significance, and sufficiency of evidence. On a composite of all three dimensions, a reviewing agent powered by GPT-5.2 scores above each paper's top-rated human reviewer (60.0% vs. 48.2%, p = 0.009), while all three AI reviewers (including Gemini 3.0 Pro and Claude Opus 4.5) exceed the lowest-rated human across every dimension. AI reviewers' accurate criticisms are also more often rated significant and well-evidenced, and surface a distinct 26% of issues no human raises. However, AI reviewers overlap far more than humans do (21% vs. 3% for cross-reviewer pairs), and exhibit 16 recurring weaknesses humans do not share, such as limited subfield knowledge, lack of long context management over multiple files, and overly critical stance on minor issues. Overall, our results position current AI reviewers as complements to, not substitutes for, human reviewers.
>
---
#### [new 067] Beyond Semantic Similarity: A Two-Phase Non-Parametric Retrieval Workflow for Corporate Credit Underwriting
- **分类: cs.CL**

- **简介: 该论文属于企业信用评估任务，解决传统检索方法在语义相似性与决策实用性之间的差距问题。提出两阶段非参数检索架构，提升检索结果的实用价值。**

- **链接: [https://arxiv.org/pdf/2605.20684](https://arxiv.org/pdf/2605.20684)**

> **作者:** Linus Ng Junjia; Ezekiel Tee Kongquan; Kelvin Heng; Kenneth Zhu Ke; Zhao Jing Yuan
>
> **摘要:** Corporate credit underwriting requires analysts to extract actionable evidence from long, heterogeneous financial documents spanning hundreds of pages and multiple languages. Standard Retrieval-Augmented Generation (RAG) pipelines optimize for semantic similarity, which frequently surfaces passages that are topically related but lack decision utility, a problem we term the similarity-utility gap. We propose a two-phase non-parametric retrieval architecture that separates high-recall candidate retrieval from high-precision utility ranking. The first phase combines lexical and dense multilingual retrieval to construct a broad candidate pool. The second phase applies an adaptive retrieval controller that filters candidates using query intent and document structure signals, followed by an LLM-as-a-Judge utility scoring mechanism that ranks passages by analytical usefulness rather than semantic proximity. A context-aware extraction module preserves structural fidelity across narrative text and complex financial tables. The system is deployed entirely on-premise to satisfy enterprise data governance requirements. Evaluated on a multilingual corpus of proprietary financial documents with analyst-curated relevance labels, the system significantly outperforms naive retrieval baselines. In production deployment across more than 800 credit analysts, document review time was reduced from several hours to approximately three minutes, demonstrating the practical value of utility-aware RAG architectures for document-intensive decision-support workflows.
>
---
#### [new 068] Refining and Reusing Annotation Guidelines for LLM Annotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的命名实体识别任务，旨在解决LLM在标注时难以遵循专业规范的问题。通过迭代优化标注指南，提升模型的标注质量。**

- **链接: [https://arxiv.org/pdf/2605.20809](https://arxiv.org/pdf/2605.20809)**

> **作者:** Kon Woo Kim; Jin-Dong Kim; Akiko Aizawa
>
> **备注:** 14 pages, 7 figures. Accepted to the ACL 2026 Main Conference
>
> **摘要:** While Large Language Models (LLMs) demonstrate remarkable performance on zero-shot annotation tasks, they often struggle with the specialized conventions of gold-standard benchmarks. We propose the systematic reuse and refinement of annotation guidelines as an alignment mechanism, introducing an iterative moderation framework that simulates the early phases of annotation projects. We evaluate three hypotheses: (1) the efficacy of guideline integration, (2) the advantage of reasoning optimized models, and (3) the viability of moderation under minimal supervision. Testing across biomedical NER tasks (NCBI Disease, BC5CDR, BioRED) with three LLM families (GPT, Gemini, DeepSeek), our results empirically confirm all three hypotheses. While the iterative moderation framework shows good potential in effectively refining guidelines, our analysis also reveals substantial room for improvement.
>
---
#### [new 069] Self-Training Doesn't Flatten Language -- It Restructures It: Surface Markers Amplify While Deep Syntax Dies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型自训练过程中的结构变化，揭示其并非简单“扁平化”，而是重新组织语言结构。任务为语言模型分析，解决自训练影响语言结构的问题，通过实验验证结构深度对衰减率的影响。**

- **链接: [https://arxiv.org/pdf/2605.20602](https://arxiv.org/pdf/2605.20602)**

> **作者:** Ming Liu
>
> **备注:** 19 pages (14 main + 5 appendix), 8 figures, 3 tables
>
> **摘要:** Successive self-training on a language model's own outputs is widely characterized as a process of flattening: diversity drops, distributions narrow, and the text becomes "more like itself." We provide evidence that this characterization is incomplete. Across eleven generations of self-training on five models (GPT-2 124M, Pythia-410M, Pythia-1.4B, OPT-1.3B, Pythia-2.8B), language is not flattened uniformly -- it is restructured. Surface markers (discourse connectives, hedges, em-dashes) rise, while mid- and deep-syntactic structures (questions, parentheticals, passives, subjunctives) collapse. We formalize this asymmetric collapse as the Structural Depth Hypothesis (SDH): the per-generation decay rate of a linguistic feature is predicted primarily by its structural depth -- the number of nested syntactic dependencies it requires -- and only secondarily by its generation-zero output frequency. Pooling 17-feature panels from five models spanning three architecture families (N=85), the pooled Spearman correlation is rho=0.540 (p < 10^{-6}; cluster-bootstrap 95% CI [0.434, 0.634]), while frequency is a substantially weaker predictor (rho=0.225). A matched human-text fine-tuning control yields rho=0.039 (p=0.88), confirming the gradient is self-training-specific. We further document a Superficial Complexity Paradox: aggregate complexity proxies (dep-tree depth, TTR, word length) all rise as the underlying clause structure dies, with direct implications for training-data curation and LLM-text detection.
>
---
#### [new 070] Single-Pass, Depth-Selective Reading for Multi-Aspect Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多方面情感分析任务，解决模型效率与表达力的平衡问题。提出DABS框架，通过单次编码共享表示，按需读取，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.20998](https://arxiv.org/pdf/2605.20998)**

> **作者:** Yan Xia; Zhuangzhuang Pan; Amirrudin Kamsin; Chee Seng Chan
>
> **备注:** Accepted at ACL2026 (main). Our solution (DABS) reads the sentence once, then lets each aspect selectively query the right tokens and Transformer depths, cutting redundant computation while preserving ATSA accuracy
>
> **摘要:** Aspect-Term Sentiment Analysis (ATSA) in multi-aspect sentences faces a fundamental tradeoff between efficiency and expressiveness. Existing models either re-encode the sentence for each aspect or rely on static use of deep representations, leading to redundant computation and limited adaptivity. We argue that Transformer depth is a costly, queryable resource, and propose DABS, a single-pass inference framework that encodes each sentence once to construct a reusable, depth-ordered substrate. Each aspect then queries this shared representation to selectively read relevant tokens and abstraction levels, without re-encoding. This decouples shared sentence encoding from lightweight, aspect-conditioned readout. Experiments on four ATSA benchmarks show that DABS achieves competitive performance while reducing end-to-end computation by up to 60% in multi-aspect settings (M >= 2). Further analyses indicate that adaptive depth querying is most beneficial for linguistically complex cases such as negation and contrast. Code is publicly available at this https URL
>
---
#### [new 071] Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究In-context Learning中任务向量的设计问题，提出通过分布对齐提升任务向量效果，引入新指标并提出LTV方法，有效提高分类和回归任务性能。**

- **链接: [https://arxiv.org/pdf/2605.20730](https://arxiv.org/pdf/2605.20730)**

> **作者:** Jihoon Kwon; Jiwon Choi; Jy-yong Sohn
>
> **备注:** 9 pages, preprint
>
> **摘要:** In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks through demonstrations, yet it suffers from escalating inference costs as context length increases. While task vectors offer a promising alternative by compressing demonstrations into compact hidden-state representations, their quality has been evaluated only through downstream task accuracy. This indirect criterion provides limited insight into how to design more effective task vector extraction methods. In this paper, we posit that inference using task vectors should align their predictive distribution with that of ICL. To quantify this, we introduce $d_{\text{NTP}}$, a metric that measures the discrepancy in next-token probabilities between task vector-based and ICL-based inference. Our empirical analysis reveals that $d_{\text{NTP}}$ serves as a performance proxy, exhibiting a strong negative correlation with downstream accuracy. Motivated by this, we develop Linear Task Vector (LTV), a method designed to minimize $d_{\text{NTP}}$ via a closed-form linear mapping that estimates demonstration effects through regression. Across eight classification benchmarks and five LLMs, LTV consistently outperforms existing task vector baselines, improving average accuracy by 9.2\% while reducing inference latency. We further show that LTV outperforms the baselines on regression tasks. Moreover, we investigate the transferability of LTV across different model scales; an aspect that has remained nascent in task vector research. Specifically, we empirically show that task vectors from a larger model can enhance a smaller model's performance by 6.4\%, suggesting a new utility for extracted task representations.
>
---
#### [new 072] Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents
- **分类: cs.CL**

- **简介: 该论文提出Auto-Dreamer，解决语言代理在多任务中有效存储与提炼知识的问题。通过分离快速记忆获取与慢速跨会话整合，提升记忆利用效率。**

- **链接: [https://arxiv.org/pdf/2605.20616](https://arxiv.org/pdf/2605.20616)**

> **作者:** Chongrui Ye; Yuxiang Liu; Yu Wang; Haofei Yu; Yining Zhao; Ge Liu; Julian McAuley; Jiaxuan You
>
> **备注:** Preprint
>
> **摘要:** Language agents increasingly operate over streams of related tasks, yet existing memory systems struggle to convert accumulated experience into reusable knowledge. Retrieval-augmented and structured memory methods record per-session observations effectively, but often couple acquisition and consolidation into a single online process, leaving the agent without a global view across sessions to discover recurring patterns, abstract shared procedures, or prune redundant entries. Inspired by complementary learning systems theory, we propose Auto-Dreamer, a learned offline consolidator for language-agent memory. Auto-Dreamer decouples fast per-session memory acquisition from slow cross-session consolidation. Given a selected working region of a typed memory bank, the consolidator treats the region as read-only evidence, performs bounded tool-use to inspect entries and provenance-linked source trajectories, and synthesizes a fresh compact replacement set that abstracts across sessions and supersedes the original region. We train Auto-Dreamer via GRPO, using end-to-end agent performance as the reward signal to learn how to consolidate memories acquired through fast online experience. Trained on ScienceWorld trajectories alone, Auto-Dreamer outperforms fixed, RL-trained, and prompted memory baselines on ScienceWorld by 7 points while using an active memory bank 12$\times$ smaller than the strongest baseline, and continues to lead on held-out ALFWorld and WebArena without retraining -- using 6$\times$ less memory than the strongest baseline on ALFWorld.
>
---
#### [new 073] Terminal-World: Scaling Terminal-Agent Environments via Agent Skills
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于终端代理任务，解决训练数据稀缺问题。提出Terminal-World，通过代理技能自动合成训练环境，提升任务泛化与效率。**

- **链接: [https://arxiv.org/pdf/2605.20876](https://arxiv.org/pdf/2605.20876)**

> **作者:** Zihao Cheng; Hongru Wang; Zeming Liu; Xinyi Wang; Xiangrong Zhu; Yuhang Guo; Wei Lin; Jeff Z. Pan; Yunhong Wang
>
> **备注:** Work in Progress
>
> **摘要:** Terminal agents extend Large Language Models with the ability to execute tasks directly in command-line environments, but their progress is bottlenecked by the scarcity of high-quality training data. Existing approaches bootstrap from partial sources such as human-defined seeds or GitHub repositories to instantiate one component and then complete the rest, producing tasks confined to narrow seed distributions, environments misaligned with task semantics, and inefficient trajectories from unguided exploration. To address these limitations, we introduce Terminal-World, a fully automated pipeline that uses agent skills as the central synthesis primitive, which jointly encode what to accomplish, when to apply (preconditions and environment state), and how to execute, enabling task instructions, environments, and teacher trajectories to be co-derived. To further broaden the synthesis space, Terminal-World composes skills into skill teams and skill graphs for multi-role and cross-domain task synthesis. Using this pipeline, we construct 5,723 training environments and train Terminal-World-8B/14B/32B, evaluated across 6 benchmarks where the Terminal-World series consistently outperforms terminal-agent baselines. Notably, using the same teacher model and only 1.2% of the training data, Terminal-World-32B surpasses Nemotron-Terminal-32B on Terminal-Bench 2.0 by +4.5 Pass@1 (31.5) and achieves 43.8 Pass@3.
>
---
#### [new 074] MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于对话检索任务，旨在解决现有基准评估成本高、数据稀疏的问题。提出MTR-Suite框架，包括评估、生成和基准测试模块，提升对话检索的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.20729](https://arxiv.org/pdf/2605.20729)**

> **作者:** Junhao Ruan; Abudukeyumu Abudula; Bei Li; Yongjing Yin; Xinyu Liu; Kechen Jiao; Xin Chen; Jingang Wang; Xunliang Cai; Tong Xiao; Jingbo Zhu
>
> **备注:** Accepted to ACL 2026 (main conference). 28 pages. Code and data: this https URL
>
> **摘要:** Accurate evaluation of conversational retrieval is pivotal for advancing Retrieval-Augmented Generation (RAG) systems. However, existing conversational retrieval benchmarks suffer from costly, sparse human annotation or rigid, unnatural automated heuristics. To address these challenges, we introduce MTR-Suite, a unified framework for auditing, synthesizing, and benchmarking retrieval. It features: (1) MTR-Eval, an LLM-based auditor quantifying alignment gaps in previous benchmarks; (2) MTR-Pipeline, a multi-agent system using greedy traversal clustering to generate high-fidelity dialogues at 1/400th human cost; and (3) MTR-Bench, a rigorous general-domain benchmark. MTR-Bench mimics production-style challenges (hard topic switching, verbosity), offering superior discriminative power. We make our code and data publicly available to facilitate future research at this https URL.
>
---
#### [new 075] DIVE: Embedding Compression via Self-Limiting Gradient Updates
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出DIVE，解决嵌入压缩中的过拟合问题，通过自限损失和对比学习提升检索性能。属于嵌入压缩任务。**

- **链接: [https://arxiv.org/pdf/2605.20689](https://arxiv.org/pdf/2605.20689)**

> **作者:** Dongfang Zhao
>
> **摘要:** High-dimensional embeddings from large language models impose significant storage and computational costs on vector search systems. Recent embedding compression methods, including Matryoshka-Adaptor (EMNLP 2024), Search-Adaptor (ACL 2024), and SMEC (EMNLP 2025), enable dimensionality reduction through lightweight residual adapters, but their training objectives cause severe overfitting when labeled data is scarce, degrading retrieval performance below the frozen baseline. We propose \textsc{DIVE} (\textbf{D}imensionality reduction with \textbf{I}mplicit \textbf{V}iew \textbf{E}nsembles), a compression adapter that addresses this failure through two mechanisms. First, a self-limiting hinge-based triplet loss produces zero gradient once a triplet satisfies the margin constraint, bounding the total perturbation applied to the pretrained embedding space. Second, a head-wise NT-Xent contrastive loss treats multiple learned projections of each embedding as implicit views, providing dense self-supervised gradients that compensate for the sparsity of the triplet signal on small datasets. Across six BEIR datasets, \textsc{DIVE} outperforms all three baseline adapters on every dataset and at every evaluated compression ratio, with a 14M-parameter open-source implementation.
>
---
#### [new 076] MedicalBench: Evaluating Large Language Models Toward Improved Medical Concept Extraction
- **分类: cs.CL**

- **简介: 该论文提出MedicalBench，用于评估大语言模型在医学概念提取中的表现，解决隐含医学概念识别难题。通过构建包含隐含正例和混淆负例的数据集，验证模型的推理能力与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.20197](https://arxiv.org/pdf/2605.20197)**

> **作者:** Zhichao Yang; Gregory D. Lyng; Sanjit Singh Batra; Robert E. Tillman
>
> **摘要:** Medical concept extraction from electronic health records underpins many downstream applications, yet remains challenging because medically meaningful concepts are frequently implied rather than explicitly stated in medical narratives. Existing benchmarks with human-annotated evidence spans underscore the importance of grounding extracted concepts in medical text. However, they predominantly focus on explicitly stated concepts instead of implicit concepts. We present MedicalBench, a benchmark for medical concept extraction with evidence grounding that evaluates implicit medical reasoning. MedicalBench formulates medical concept extraction as a verification task over medical note-concept pairs, coupled with sentence-level evidence identification. Built from MIMIC-IV discharge summaries and human-verified ICD-10 codes, the dataset is curated through a multi-stage large language model (LLM) triage pipeline followed by medical annotation and expert review. It deliberately includes implicit positives, semantically confusable negatives, and cases where LLM judgments disagree with medical expert assessments. We define two complementary evaluation tasks: (1) medical concept extraction and (2) sentence-level evidence retrieval, enabling assessment of both correctness and interpretability. Benchmarking state-of-the-art LLMs reveals that performance remains modest, highlighting the difficulty of extracting implicitly expressed concepts. We further show that performance is largely invariant to note length, indicating that MedicalBench isolates reasoning difficulty rather than superficial confounders. MedicalBench provides the first systematic benchmark for implicit, evidence-grounded medical concept extraction, offering a foundation for developing medical language models that can both identify medically relevant concepts and justify their predictions in a transparent and medically faithful manner.
>
---
#### [new 077] Thinking-while-speaking: A Controlled, Interleaved Reasoning Method for Real-Time Speech Generation
- **分类: cs.CL**

- **简介: 该论文属于实时语音生成任务，解决AI在说话时保持流畅与深度推理的矛盾。提出InterRS方法，通过插入推理步骤实现语音与思考的无缝交织。**

- **链接: [https://arxiv.org/pdf/2605.20946](https://arxiv.org/pdf/2605.20946)**

> **作者:** Xuan Du; Qiangyu Yan; Wenshuo Li; Borui Jiang; Changming Xiao; Han Shu; Xinghao Chen
>
> **摘要:** The thinking-while-speaking paradigm aims to make AI communication more human. A key challenge is maintaining fluent speech while performing deep reasoning. Our method, InterRS, tackles this by inserting reasoning steps only during natural speech generation. This requires high-quality data where reasoning and speech are precisely aligned, and the length ratio are under controlled. We introduce a novel pipeline to generate such seamlessly interleaved audio data. To train our model, we combine interleaved SFT with refined data and reinforcement learning with two new rewards: a TA-Balance Reward to manage timing and thinking-answer ratio, and a Linguistic Quality Reward to refine expression. Experiments show our approach achieves 13% better performance on mathmatical and logic benchmarks while generating instant response like a spoken-language instruct model which outputs fast CoT response. Furthermore, our method generates more natural and fluent answers than prior methods.
>
---
#### [new 078] Enhancing Scientific Discourse: Machine Translation for the Scientific Domain
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决科学领域跨语言沟通问题。通过构建多语种科学语料库并微调NMT系统，提升科学文献的翻译质量。**

- **链接: [https://arxiv.org/pdf/2605.20912](https://arxiv.org/pdf/2605.20912)**

> **作者:** Dimitris Roussis; Sokratis Sofianopoulos; Stelios Piperidis
>
> **摘要:** The increasing volume of scientific research necessitates effective communication across language barriers. Machine translation (MT) offers a promising solution for accessing international publications. However, the scientific domain presents unique challenges due to its specialized vocabulary and complex sentence structures. In this paper, we present the development of a collection of parallel and monolingual corpora for the scientific domain. The corpora target the language pairs Spanish-English, French-English, and Portuguese-English. For each language pair, we create a large general scientific corpus as well as four smaller corpora focused on the domains of: Cancer Research, Energy Research, Neuroscience, and Transportation research. To evaluate the quality of these corpora, we utilize them for fine-tuning general-purpose neural machine translation (NMT) systems. We provide details regarding the corpus creation process, the fine-tuning strategies employed, and we conclude with the evaluation results.
>
---
#### [new 079] Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在指令与模式冲突时的行为，属于模型行为分析任务。旨在解决指令遵循与模式跟随的冲突问题，通过实验分析不同因素对模型鲁棒性的影响。**

- **链接: [https://arxiv.org/pdf/2605.20382](https://arxiv.org/pdf/2605.20382)**

> **作者:** Carolina Camassa; Derek Shiller
>
> **备注:** 31 pages
>
> **摘要:** Language models are trained to follow instructions, but they are also powerful pattern completers. What happens when these two objectives conflict? We construct conversations in which a user instruction to behave in a target way T (e.g., always output a specific token, answer in a particular language, or adopt a persona) is opposed by N hardcoded assistant turns demonstrating a competing pattern P. We then measure instruction-following (IF) rates in this setting, across 13 models and 16 different instructions, for up to 50 turns. Average instruction-following rates range from 1% to 99% across models, largely uncorrelated with standard capability benchmarks. The transition from instruction-following to pattern-following is universal but highly model-dependent. Robustness is modulated both by instruction content, with models resisting induction longer when instructions align with their trained value priors, and by output format, with diverse multi-token responses proving substantially more resistant than single-token outputs. Chain-of-thought reasoning improves robustness but does not eliminate susceptibility, and can produce dissociation between correct deliberation and incorrect output. When asked to predict their behavior in this setting, models achieve 83.5% accuracy on average but systematically underestimate their own resistance to induction pressure. These results suggest that instruction-following remains brittle under induction pressure even for otherwise capable models, and that output diversity, rather than semantic engagement with the input, is the primary factor predicting robustness.
>
---
#### [new 080] Automated ICD Classification of Psychiatric Diagnoses: From Classical NLP to Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于医疗文本分类任务，旨在解决精神疾病诊断自动编码问题。通过NLP和ML技术，将自由文本映射到ICD编码，对比了传统方法与大语言模型的效果。**

- **链接: [https://arxiv.org/pdf/2605.21154](https://arxiv.org/pdf/2605.21154)**

> **作者:** Fernando Ortega; Raúl Lara-Cabrera; Jorge Dueñas-Lerín; Alejandro de la Torre-Luque; Mercé Salvador Robert; Enrique Baca-García
>
> **摘要:** Mental health has become a global priority, leading to a massive administrative burden in the coding of clinical diagnoses. This study proposes the automation of psychiatric diagnostic analysis by mapping free-text descriptions to the International Classification of Diseases (ICD) using Natural Language Processing (NLP) and Machine Learning (ML) techniques. Utilizing a specialized dataset of 145,513 Spanish psychiatric descriptions, various text representation paradigms were evaluated, ranging from classical frequency-based models (BoW, TF-IDF) to state-of-the-art Large Language Models (LLMs) such as e5\_large, BioLORD, and Llama-3-8B. Results indicate that transformer-based embeddings consistently outperform traditional methods by capturing implicit semantic cues and nuanced medical terminology. The e5\_large model, through end-to-end fine-tuning, achieved the highest performance with a $F1_{micro}$ score of 0.866. This research demonstrates that adapting LLMs to specific clinical nomenclature is essential for overcoming the challenges of ``long-tail'' label distributions and the inherent ambiguity of psychiatric discourse.
>
---
#### [new 081] ArPoMeme: An Annotated Arabic Multimodal Dataset for Political Ideology and Polarization
- **分类: cs.CL**

- **简介: 该论文提出ArPoMeme数据集，用于分析阿拉伯语政治迷因中的意识形态与极化问题。任务是构建多模态标注数据，解决阿拉伯政治话语分析资源不足的问题。工作包括数据收集、标注及工具开发。**

- **链接: [https://arxiv.org/pdf/2605.20967](https://arxiv.org/pdf/2605.20967)**

> **作者:** Wajdi Zaghouani; Kais Attia; Md. Rafiul Biswas; Fadhl Eryani
>
> **备注:** Accepted at LREC 2026 Main Conference
>
> **摘要:** Memes have become a prominent medium of political communication in the Arab world, reflecting how humor, imagery, and text interact to express ideological and cultural positions. Despite the centrality of memes to online political discourse, there is a lack of systematically curated resources for analyzing their multimodal and ideological dimensions in Arabic. This paper presents ArPoMeme, a large-scale dataset of approximately 7,300 Arabic political memes categorized by ideological orientation, including Leftist, Islamist, Pan-Arabist, and Satirical perspectives. The dataset captures the diversity of Arabic meme ecosystems by grounding classification in the self-identification of public Facebook pages and groups that produce and disseminate these memes. To ensure both scale and accuracy, we designed a semi-automated data collection pipeline combining Playwright-based Facebook scraping with Google Drive synchronization, followed by text extraction using the Qwen2.5-VL-7B vision language model. The extracted text was manually verified and annotated for three polarization dimensions: Us vs. Them framing, Hostility toward out-groups, and Calls to action. Annotation was conducted through a custom Streamlit-based interface supporting distributed labeling, real-time tracking, and version control. The resulting dataset links visual content, textual messages, and ideological orientation, enabling fine-grained analysis of political antagonism, mobilization, and humor. Quantitative analysis of the annotated corpus reveals strong asymmetries in antagonistic framing across ideological groups, with Islamist and satirical memes exhibiting the highest levels of hostility and mobilization cues. The dataset and the annotation tool offers a reproducible and publicly available resource for studying Arabic political discourse, multimodal ideology detection, and polarization dynamics.
>
---
#### [new 082] FlowLM: Few-Step Language Modeling via Diffusion-to-Flow Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FlowLM，一种通过扩散模型转换的流匹配语言模型，解决高效少步生成问题。通过优化采样轨迹和训练目标，实现高质量文本生成。**

- **链接: [https://arxiv.org/pdf/2605.20199](https://arxiv.org/pdf/2605.20199)**

> **作者:** Runzhe Zhang; Letian Chen; Wenpeng Zhang; Zhouhan Lin; Peilin Zhao
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** We present FlowLM, a flow matching language model transformed from pre-trained diffusion language models via efficient fine-tuning. By re-aligning the curved sampling trajectories of diffusion models into straight-line flows, FlowLM enables high quality few-step generation that rivals or even outperforms the quality of 2,000-step diffusion sampling with very few training epochs. Remarkably, finetuned FlowLM reaches performance saturation with only half as many training epochs as training from scratch, both approaches greatly outperforming the original diffusion model, thereby validating our method. Furthermore, we validate a more effective training objective for flow matching: predicting clean data to consistently guide the sampling process towards the true data distribution. Empirical results demonstrate that our approach is highly effective for high-quality, few-step text generation.
>
---
#### [new 083] Most Transformer Modifications Still Do Not Transfer at 1-3B: A 2020-2026 Update to Narang et al. (2021) with Downstream Evaluation and a Noise Floor
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究Transformer模型修改的迁移效果。针对1-3B参数规模，验证20种新修改方法，发现多数不具迁移性，强调下游评估和稳定性测试的重要性。**

- **链接: [https://arxiv.org/pdf/2605.20798](https://arxiv.org/pdf/2605.20798)**

> **作者:** Yang Zhao; Jiahao Lu; Bin Huang; Guhua Zhang; Jie Zhou
>
> **备注:** 19 pages, 3 figures, under review at EMNLP 2026
>
> **摘要:** Narang et al. (2021) evaluated 40+ Transformer modifications at T5-base scale and concluded that most did not transfer. Five years later, the typical working regime has moved to 1-3B parameters, downstream evaluation has replaced pretraining perplexity, and a substantially different catalogue of modifications has emerged. We revisit their question by testing 20 post-2021 Transformer modifications at 1.2B and 3B under strict iso-data, iso-compute, iso-recipe control, with a multi-seed baseline noise floor and CLIMB-12 downstream evaluation as the primary metric. The central finding reproduces theirs at this curated set: most modifications do not transfer. Of the 20 modifications, only two clear Bonferroni correction at 1.2B; one of those two further fails to train stably at 3B under the shared recipe. We also find that the loss-downstream gap reported by Tay et al. (2023) enlarges several-fold for attention-output modifications: two significant failures converge to within 2-3% of baseline validation loss yet drop 6-16 CLIMB-points. We conclude that noise-floor reporting, downstream evaluation, and cross-scale stability testing are now prerequisites for architecture comparisons at 1-3B.
>
---
#### [new 084] DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DelTA方法，解决强化学习中从可验证奖励获取的token级信用分配问题，通过增强区分性梯度方向提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.21467](https://arxiv.org/pdf/2605.21467)**

> **作者:** Kaiyi Zhang; Wei Wu; Yankai Lin
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) has emerged as a central technique for improving the reasoning capabilities of large language models. Despite its effectiveness, how response-level rewards translate into token-level probability changes remains poorly understood. We introduce a discriminator view of RLVR updates, showing that the policy-gradient update direction implicitly acts as a linear discriminator over token-gradient vectors and thereby determines which token probabilities are increased or decreased during learning. Under standard sequence-level RLVR, this discriminator is constructed from positive- and negative-side centroids formed by advantage-weighted averaging of token-gradient vectors. However, such centroid construction can be dominated by shared high-frequency patterns, such as formatting tokens, diluting sparse yet discriminative directions that better distinguish high-reward responses from low-reward ones. To address this limitation, we propose $\textbf{DelTA}$, a discriminative token credit assignment method that estimates token coefficients to amplify side-specific token-gradient directions and downweight shared or weakly discriminative ones. These coefficients reweight a self-normalized RLVR surrogate, making the effective side-wise centroids more contrastive and thereby reshaping the RLVR update direction. On seven mathematical benchmarks, DelTA outperforms the strongest same-scale baselines by 3.26 and 2.62 average points on Qwen3-8B-Base and Qwen3-14B-Base, respectively. Additional results on code generation, a different backbone, and out-of-domain evaluations further demonstrate the generalization ability of DelTA.
>
---
#### [new 085] The Hidden Signal of Verifier Strictness: Controlling and Improving Step-Wise Verification via Selective Latent Steering
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于验证任务，旨在解决生成验证器严格性难以控制的问题。通过隐藏状态干预，提出VerifySteer方法，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2605.20745](https://arxiv.org/pdf/2605.20745)**

> **作者:** Yefan Zhou; Yilun Zhou; Austin Xu; Soroush Vosoughi; Shafiq Joty; Jiang Gui
>
> **摘要:** Generative verifiers have emerged as a promising paradigm for step-wise verification, but their verification behavior is often poorly calibrated: they may be under-critical and miss erroneous steps, or over-critical and reject correct reasoning. We refer to this tendency to be overly lenient or overly critical as verifier strictness. In this work, we study whether verifier strictness can be controlled through hidden-state intervention. We uncover a verification-specific hidden-state signal: in step-wise verification, a verifier's tendency to accept or reject a solution step is encoded near the boundary of the corresponding verification paragraph. Exploiting this signal, we show that hidden-state steering can directly modulate verifier strictness without fine-tuning. However, uniform steering induces a trade-off between error detection and correctness certification. To address this, we propose VerifySteer, which exploits latent correctness signals for sample-level routing and selectively intervenes on paragraph boundaries. Experiments on ProcessBench and Hard2Verify show that VerifySteer outperforms prompt optimization and activation steering baselines, and is competitive with self-consistency while requiring 4-7x less inference compute. VerifySteer is also complementary to verification fine-tuning, providing further gains on top of fine-tuned verifiers. The code is available at this https URL.
>
---
#### [new 086] Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究sycophancy（盲从）的缓解方法，比较了预训练人格向量与传统方法的效果。任务是减少模型对用户错误的盲目同意。工作包括评估不同人格向量对sycophancy的影响。**

- **链接: [https://arxiv.org/pdf/2605.21006](https://arxiv.org/pdf/2605.21006)**

> **作者:** Ishaan Kelkar; Nebras Alam; Vikram Kakaria; Madhur Panwar; Vasu Sharma; Maheep Chaudhary
>
> **摘要:** We study the effect of different persona on \textbf{sycophancy}: model's agreement with users even when the user is incorrect. The standard mitigation, Contrastive Activation Addition (CAA), derives a steering direction from labelled pairs of sycophantic and honest responses. This study evaluates whether off-the-shelf persona steering vectors, originally developed for general role-playing and not trained on sycophancy data, can serve as an alternative. In two instruction-tuned models, steering toward personas characterised by doubt or scrutiny reduces sycophancy to approximately $68\%$ and $98\%$ of CAA's effect, and, unlike CAA, maintains accuracy when the user is correct. The effect is also asymmetric: steering toward agreeable personas does not produce a mirror increase in sycophancy. Geometrically, the persona vector is largely independent of the direction of sycophancy in activation space. Collectively, these findings suggest that sycophancy is better understood as a persona-level property rather than a single steerable direction. We release our code here: this https URL.
>
---
#### [new 087] DASH: Fast Differentiable Architecture Search for Hybrid Attention in Minutes on a Single GPU
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于神经网络架构搜索任务，旨在解决混合注意力机制设计效率低的问题。通过引入DASH框架，实现快速、高效的架构搜索。**

- **链接: [https://arxiv.org/pdf/2605.20936](https://arxiv.org/pdf/2605.20936)**

> **作者:** Weizhe Chen; Miao Zhang; Junpeng Jiang; Yaping Li; Weili Guan; Liqiang Nie
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Hybrid attention architectures are becoming an increasingly important paradigm for improving LLM inference efficiency while preserving model quality, making hybrid architecture design a central problem. Existing designs often rely on manual empirical rules or proxy-based selector signals for layer-wise operator allocation. Recent NAS-style systems such as Jet-Nemotron demonstrate the promise of automated hybrid architecture search. However, Jet-Nemotron's PostNAS search stages alone use 200B tokens, making such search pipelines difficult to use as routine methods for hybrid architecture design. We introduce DASH, a fast differentiable search framework for hybrid attention architecture design, which relaxes discrete layer-wise attention operator placement into continuous architecture logits, prepares reusable teacher-aligned linear candidates, and performs architecture-only search with model and operator weights frozen to significantly enhance search efficiency. On Qwen2.5-3B-Instruct, DASH consistently outperforms a comprehensive suite of existing selector-style hybrid attention design baselines, showing that direct differentiable search can discover stronger hybrid architectures. Moreover, DASH achieves stronger RULER performance than released Jet-Nemotron models while remaining competitive on overlapping short-context and general benchmarks. Notably, each DASH search run uses only 12.3M tokens and takes about 20 minutes on a single RTX Pro 6000 GPU, corresponding to merely 0.006% of the PostNAS search tokens reported by Jet-Nemotron. These results suggest that high-quality hybrid attention architectures can be obtained through minutes-level differentiable search, providing a promising direction for hybrid architecture design.
>
---
#### [new 088] AiraXiv: An AI-Driven Open-Access Platform for Human and AI Scientists
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AiraXiv平台，解决AI时代学术出版的挑战，通过AI辅助和读者反馈实现研究迭代，支持人与AI科学家协作。**

- **链接: [https://arxiv.org/pdf/2605.21481](https://arxiv.org/pdf/2605.21481)**

> **作者:** Junshu Pan; Panzhong Lu; Yixuan Weng; Qiyao Sun; Fang Guo; Zijie Yang; Qiji Zhou; Yue Zhang
>
> **摘要:** Recent advances in artificial intelligence (AI) have accelerated the growth of both human-authored and AI-generated research outputs, placing increasing strain on traditional academic publishing systems and challenging the scalability of conference- and journal-centered paradigms amid rising submission volumes, reviewer workload, and venue size. To address these challenges, we explore an AI-era publishing paradigm in which both human and AI scientists participate as authors and readers, and papers evolve through continuous, feedback-driven iteration. We propose AiraXiv, an AI-driven open-access platform built on open preprints, AI-augmented analysis and review, and reader feedback. AiraXiv supports human scientists through an interactive UI and AI scientists through Model Context Protocol (MCP)-based interactions. We validate AiraXiv through real-world deployments, including serving as the submission platform for ICAIS 2025, demonstrating its potential as a fast, inclusive, and scalable research infrastructure for the AI era. AiraXiv is publicly available at this https URL.
>
---
#### [new 089] Chronicle: A Multimodal Foundation Model for Joint Language and Time Series Understanding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Chronicle，一个联合训练的多模态模型，用于同时处理文本和时间序列数据。解决现有模型单模态训练、缺乏联合预训练的问题。工作包括构建统一架构，共享参数，提升多模态理解能力。**

- **链接: [https://arxiv.org/pdf/2605.20268](https://arxiv.org/pdf/2605.20268)**

> **作者:** Paul Quinlan; Jeremy Levasseur; Qingguo Li; Xiaodan Zhu
>
> **摘要:** Real-world time series come with text: metadata, descriptions, news, reports. Yet time series foundation models process numerical sequences in isolation, and the multimodal text-and-time-series models that attempt to bridge the two all adapt a pretrained language model post hoc, inheriting representations shaped without ever seeing temporal data. These models are also evaluated almost exclusively against other multimodal baselines, not against the strongest unimodal foundation models in either domain, leaving open whether joint training is needed at all. We present Chronicle, a compact 324M-parameter decoder-only transformer trained from scratch on natural language and time series within a single unified architecture. Both modalities share the same transformer blocks, attention mechanism, and residual stream; the bulk of pretraining uses unimodal batches so cross-modal capability emerges purely from shared parameters, with a short alignment stage that interleaves the two. To our knowledge, Chronicle is the first model jointly pretrained on text and time series from scratch, and the first multimodal model evaluated against dedicated foundation models in both domains. It matches Gemma-3-270M-PT on 19 NLU tasks, sets a new bar for frozen-embedding time series classification on 24 UCR/UEA datasets, and produces multimodal forecasts on Time-MMD that beat every supervised fusion baseline, all from a single backbone.
>
---
#### [new 090] Hiding in Plain Sight: Finding MAHA on Reddit
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于社会媒体分析任务，旨在研究MAHA运动的传播结构与话语。通过构建Reddit数据集，解决从海量数据中提取特定主题信息的问题。**

- **链接: [https://arxiv.org/pdf/2605.20435](https://arxiv.org/pdf/2605.20435)**

> **作者:** Sabit Ahmed; Subigya Nepal; Henry Kautz
>
> **备注:** Submitted to ASONAM 2026
>
> **摘要:** Make America Healthy Again (MAHA) is a national health movement that encompasses a striking mix of beliefs, from broadly accepted concerns about good diet and exercise to controversial takes on organic and genetically modified food, childhood vaccination, science, and institutions. Various influencers and promoters of the MAHA movement on social media are scattered throughout the online space. Investigating the structure, discourse, and contagion of MAHA beliefs requires large-scale fine-grained digital footprints. Constructing structured data covering different MAHA themes from vast unstructured social media data is challenging. We introduce a Reddit dataset that spans six years (2020-2025), comprising 19.4M posts from 4M users. Containing the natural and thematic context of 12 MAHA-aligned beliefs, this dataset offers researchers from various domains the opportunity to study the dynamics of the MAHA movement, its structural and functional components, and the linguistic and behavioral patterns of its proponents.
>
---
#### [new 091] AgentAtlas: Beyond Outcome Leaderboards for LLM Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出AgentAtlas，用于评估大语言模型代理。解决现有基准碎片化问题，通过分类体系和方法论全面分析代理性能。属于模型评估任务。**

- **链接: [https://arxiv.org/pdf/2605.20530](https://arxiv.org/pdf/2605.20530)**

> **作者:** Parsa Mazaheri; Kasra Mazaheri
>
> **摘要:** Large language model agents now act on codebases, browsers, operating systems, calendars, files, and tool ecosystems, but the benchmarks used to evaluate them are fragmented: each emphasizes a different unit of measurement (final task success, tool-call validity, repeated-pass consistency, trajectory safety, or attack robustness). A line of 2024-2025 work has converged on the diagnosis that a single accuracy column is no longer the right unit of comparison for deployable agents. AgentAtlas extends this line of work with four components: (i) a six-state control-decision taxonomy (Act / Ask / Refuse / Stop / Confirm / Recover); (ii) a nine-category trajectory-failure taxonomy with two orthogonal hierarchical labels (primary_error_source, impact); (iii) a taxonomy-aware vs. taxonomy-blind methodology that measures how much of a model's apparent capability comes from the supervision in the prompt; and (iv) a benchmark-coverage audit mapping fifteen agent benchmarks against six behavioral axes. To demonstrate the methodology we run a small fixed eight-model set (1,342 generated items, four frontier closed and four open-weight) under both prompt modes. Removing the explicit label menu drops every model's trajectory accuracy by 14-40 pp to a tight 0.54-0.62 floor regardless of family, and no single model wins on all three of control accuracy, trajectory diagnosis, and tool-context utility retention. We treat the synthetic run as a measurement-protocol demonstration, not a benchmark release.
>
---
#### [new 092] Multi-agent Collaboration with State Management
- **分类: cs.MA; cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文属于多智能体协作任务，解决共享代码库并发编辑导致的冲突问题。提出STORM系统，通过状态管理实现实时冲突检测与解决。**

- **链接: [https://arxiv.org/pdf/2605.20563](https://arxiv.org/pdf/2605.20563)**

> **作者:** Mengyang Liu; Taozhi Chen; Zhenhua Xu; Xue Jiang; Yihong Dong
>
> **摘要:** Recent advances in multi-agent systems have shown great potential for solving complex tasks. However, when multiple agents edit a shared codebase concurrently, their changes can silently conflict and inconsistent views lead to integration failures. Existing multi-agent systems address this through workspace isolation (e.g., one git worktree per agent), but this defers conflict resolution to a post-hoc merge step where recovery is expensive. In this paper, we propose STORM, i.e., STate-ORiented Management for multi-agent collaboration. Specifically, STORM manages agent states by mediating their interactions with the shared workspace, ensuring that each agent operates on a consistent view of the codebase and that conflicting edits are detected and resolved at write time. We evaluate STORM on Commit0 and PaperBench across multiple LLMs. STORM outperforms the git-worktree-based multi-agent baseline by +18.7 on Commit0-Lite and +1.4 on PaperBench, while achieving comparable or better cost efficiency. Combined with single-agent runs, STORM reaches highest scores of 87.6 and 78.2 on the two benchmarks respectively, suggesting that explicit state management is a more effective foundation for multi-agent collaboration than workspace isolation. STORM can also be plugged into any multi-agent system seamlessly.
>
---
#### [new 093] Draw2Think: Harnessing Geometry Reasoning through Constraint Engine Interaction
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Draw2Think框架，解决几何推理中中间状态不可验证的问题。通过与GeoGebra交互，实现精确的几何约束验证，提升模型推理准确性。任务为几何问题求解。**

- **链接: [https://arxiv.org/pdf/2605.20743](https://arxiv.org/pdf/2605.20743)**

> **作者:** Juncheng Hu; Jiawei Du; Xin Zhang; Joey Tianyi Zhou
>
> **摘要:** Vision-language models solve geometry problems with rising accuracy, yet their intermediate states remain latent and unverifiable: a relation expressed in textual reasoning or drawing code carries no guarantee that a constraint-satisfying configuration realizes it. We observe that existing externalization methods based on rendered pixels or one-shot scripts fail to provide exact, per-action geometric guarantees. Enforcing geometric relations by algebraic definition closes this gap: the workspace becomes a constraint-checked evolving canvas. We present Draw2Think, a framework that recasts geometric reasoning from latent spatial inference into agentic interaction with the GeoGebra constraint engine. In a Propose-Draw-Verify loop, Draw2Think externalizes hypotheses onto an executable canvas, measures exact geometric quantities, and feeds structured observations back to the model, so subsequent reasoning proceeds from checked canvas state grounded by the shared workspace. This externalization makes two properties separately auditable: model-level Construction Fidelity (whether the canvas realizes the intended configuration) and engine-level Measurement Faithfulness (exact values and relations from canvas constraints). Across construction, outcome, and rendering evaluations, Draw2Think builds canvases that pass 95.9% predicate-level and 84.0% strict problem-level construction checks on GeoGoal, improves outcome accuracy by up to 4.1%/16.4% on planar/solid benchmarks, and attains 68.2%/90.5% strict/relaxed rendering scores on GenExam-math. Project page is available at this https URL
>
---
#### [new 094] SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SpecBench，用于衡量长周期编码代理的奖励黑客问题。通过对比可见测试与保留测试的通过率差异，评估代理是否真正理解任务而非仅优化测试通过。**

- **链接: [https://arxiv.org/pdf/2605.21384](https://arxiv.org/pdf/2605.21384)**

> **作者:** Bingchen Zhao; Dhruv Srikanth; Yuxiang Wu; Zhengyao Jiang
>
> **摘要:** As long-horizon coding agents produce more code than any developer can review, oversight collapses onto a single surface: the automated test suite. Reward hacking naturally arises in this setup, as the agent optimizes for passing tests while deviating from the users true goal. We study this reward hacking phenomenon by decompose software engineering tasks into three parts: (i) a natural language description of the specification (ii) visible validation tests that exercise specified features in isolation, and (iii) held-out tests that compose those same features to simulate real-world usage. Based on the specification and the visible validation test suites, a genuine agent would be able to generate a solution that can also pass all of the held-out tests. Therefore we use the gap in pass rates on these two suites to quantify reward hacking. Based on this methodology, we introduce SpecBench, a benchmark comprising 30 systems-level programming tasks ranging from short horizon tasks like building a JSON parser to ultra long horizon tasks like building an entire OS kernel from scratch. Large-scale experiments reveal a consistent pattern: while every frontier agent saturates the visible suite, reward hacking persists, with smaller models exhibiting larger gaps on holdout suites. The gap also scales sharply with task length: it grows by 28 percentage points for every tenfold increase in code size. Failures range from subtle feature isolation to deliberate exploits, including a 2,900-line hash-table "compiler" that memorizes test inputs. SpecBench offers a principled testbed for measuring whether coding agents build genuine working systems or merely game the test suites developers hand them.
>
---
#### [new 095] Distribution-Aware Reward: Reinforcement Learning over Predictive Distributions for LLM Regression
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型回归任务，旨在解决预测分布不准确的问题。通过引入Distribution-Aware Reward，优化模型生成更准确且校准的预测分布。**

- **链接: [https://arxiv.org/pdf/2605.20740](https://arxiv.org/pdf/2605.20740)**

> **作者:** Jungsoo Park; Hyungjoo Chae; Ethan Mendes; Jay DeYoung; Varsha Kishore; Wei Xu; Alan Ritter
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Large language models can predict real-valued quantities from heterogeneous inputs such as text, code, and molecular strings, but most training objectives score each decoded floating-point number independently, improving point estimates without ensuring calibrated predictive distributions. This limits applications requiring candidate ranking or uncertainty estimation. We introduce Distribution-Aware Reward, an on-policy reinforcement learning objective whose main contribution is to train language models to produce better predictive distributions for regression tasks, rather than only optimizing individual decoded outputs against scalar targets. Our method treats multiple decoded samples as an empirical predictive distribution, evaluates it with the Continuous Ranked Probability Score, and assigns leave-one-out credit based on each rollout's marginal contribution to distribution quality, rewarding predictions that are both accurate and appropriately dispersed. We evaluate our method on a controlled Gaussian-mixture task, code performance prediction, and molecular property prediction from SMILES strings. Across tasks, our method improves over supervised fine-tuning and pointwise reinforcement learning baselines, with strong rank-correlation gains, including a 6-point Spearman improvement on KBSS. On MoleculeNet, it uses only SMILES strings yet remains competitive with strong graph-based and 3D molecular models. Further analyses show that our method mitigates rollout diversity collapse and improves uncertainty diagnostics, suggesting that directly optimizing predictive distributions makes language model regression more robust and better calibrated.
>
---
#### [new 096] Training Language Agents to Learn from Experience
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言代理在交互环境中通过经验学习的问题。提出ICT任务和RL训练方法，使代理能从经验中提炼可复用知识，提升未见过任务的性能。**

- **链接: [https://arxiv.org/pdf/2605.20477](https://arxiv.org/pdf/2605.20477)**

> **作者:** Yuval Shalev; Zifeng Ding; Mateja Jamnik
>
> **摘要:** Language agents can adapt from experience in interactive environments, but current reflection-based methods can only self-correct within a single task instance. Whether such experience can be distilled into reusable lessons that improve performance on future unseen tasks remains unclear. We address this problem by introducing the In-context Training (ICT) task, a framework for evaluating cross-task self-improvement in language agents. In ICT, a reflector model observes trajectories collected by an actor model and generates system prompts intended to improve the actor's performance on future unseen tasks. We then propose an RL-based training pipeline for learning such reflections directly from experience, without human-provided examples. Across ALFWorld and MiniHack, our trained reflectors outperform an untrained baseline on most held-out task families, showing that the ability to learn from experience can itself be learned. In some cases, we observe generalisation beyond the benchmark on which the reflector was trained, to substantially different environments. Finally, we introduce MetaGym, a generic Python library for constructing meta-environments, enabling future research on self-improving language agents.
>
---
#### [new 097] CP-MoE: Consistency-Preserving Mixture-of-Experts for Continual Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决大模型中的灾难性遗忘问题。提出CP-MoE框架，通过一致性保持机制减少参数干扰，提升跨任务知识迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.20247](https://arxiv.org/pdf/2605.20247)**

> **作者:** Yang Liu; Toan Nguyen; Flora D. Salim
>
> **摘要:** Catastrophic forgetting remains a major obstacle to continual learning in large language models (LLMs) and vision--language models (VLMs). Although Mixture-of-Experts (MoE) architectures offer an efficient path to scaling, existing LoRA-based MoE continual learning methods still face a fundamental trade-off: they either isolate experts too aggressively, limiting knowledge transfer across tasks, or allow task-specific updates to overwrite important existing parameters, leading to severe forgetting. To address this, we propose CP-MoE, a continual learning framework built around a transient expert that captures early task-specific updates and guides their integration into stable experts. CP-MoE introduces a consistency-preserving routing bias, which uses the transient expert to estimate representation similarity with stable experts and steer routing towards more compatible expert selection, and a transient expert-guided regularisation mechanism, which selectively protects important historical parameters during merging. Together, these components reduce parameter interference and forgetting while preserving cross-task knowledge transfer. We validate CP-MoE on both unimodal and multimodal continual learning benchmarks with LLM-based and VLM-based MoE models. On SuperNI benchmark, spanning diverse sequential language tasks, CP-MoE achieves state-of-the-art performance and stronger zero-shot transfer to unseen tasks. On VQA v2 dataset, it scales effectively to multimodal visual reasoning, consistently reduces forgetting, and outperforms strong MoE baselines.
>
---
#### [new 098] Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search
- **分类: cs.LO; cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出Lean Refactor，解决Lean证明重构中的多目标优化问题，通过检索增强的代理策略搜索实现高效、版本鲁棒的重构。**

- **链接: [https://arxiv.org/pdf/2605.20244](https://arxiv.org/pdf/2605.20244)**

> **作者:** Jialin Lu; Soonho Kong; Rodrigo Stehling; Kaiyu Yang; Zhangyang Wang; Weiran Sun; Wuyang Chen
>
> **摘要:** We present Lean Refactor, a plug-and-play retrieval-augmented agentic framework for multi-objective, controllable, and version-robust refactoring of Lean proofs. LLM-generated proofs are notoriously correct-but-verbose and brittle across library versions, yet existing refactoring works overlook three practical challenges: 1) Lean refactoring is natively multi-objective (proof length, compilation cost, and version compatibility are often in tension); 2) Lean repositories have fragile compatibility, whereas LLM releases are unaware of Lean/Mathlib versions; 3) Training-based pipelines require repeated fine-tuning with each new LLM release, scaling neither with model churn nor with Lean's release cycle. Lean Refactor steers a frozen agentic LLM with retrievals from a curated database of multi-objective refactoring strategies, each densely annotated with metadata such as supported Lean/Mathlib versions and expected compilation-cost reduction. Experiments show over $70\%$ token-level compression on competition benchmarks, over $20\%$ on research repositories, and up to $60\%$ compilation-time reduction, outperforming prior work and Claude Code. Version-filtered retrieval further improves compression on the target Lean version, and refactored miniF2F proofs exhibit stronger zero-shot version transfer to future Lean releases than their unrefactored counterparts.
>
---
#### [new 099] You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM训练效率问题。通过分析RLVR参数轨迹的低秩特性，提出RELEX方法，用少量训练步骤即可获得高性能模型。**

- **链接: [https://arxiv.org/pdf/2605.21468](https://arxiv.org/pdf/2605.21468)**

> **作者:** Zhepei Wei; Xinyu Zhu; Wei-Lin Chen; Chengsong Huang; Jiaxin Huang; Yu Meng
>
> **备注:** preprint. Code: this https URL
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become a dominant paradigm for improving reasoning in large language models (LLMs), yet the underlying geometry of the resulting parameter trajectories remains underexplored. In this work, we demonstrate that RLVR weight trajectories are extremely low-rank and highly predictable. Specifically, we find that the majority of downstream performance gains are captured by a rank-1 approximation of the parameter deltas, where the magnitude of this projection evolves near-linearly with training steps. Motivated by this, we propose a simple and compute-efficient method RELEX (REinforcement Learning EXtrapolation), which estimates the rank-1 subspace from a short observation window and extrapolates future checkpoints via linear regression, with no learned model required. Across three models (i.e., Qwen2.5-Math-1.5B, Qwen3-4B-Base, and Qwen3-8B-Base), RELEX produces checkpoints that match or exceed RLVR performance on both in-domain and out-of-domain benchmarks, requiring as few as 15% steps of full RLVR training. Remarkably, RELEX is able to extrapolate far beyond the observation window at no training cost, predicting checkpoints up to 10-20$\times$ beyond the observed prefix with continued improvement (e.g., observe only the first 50 steps and extrapolate to 1000 steps). Our ablation analysis confirms the minimalist sufficiency of RELEX: neither increasing the subspace rank nor employing non-linear modeling yields further gains in extrapolation. Finally, we show that RELEX's success stems from a "denoising" effect: by projecting updates onto the rank-1 subspace, the model discards stochastic optimization noise that would otherwise degrade performance during extrapolation. Our code is available at this https URL.
>
---
#### [new 100] Reinforcing Human Behavior Simulation via Verbal Feedback
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于人机交互任务，旨在提升LLM模拟人类行为的能力。通过引入口语反馈作为强化学习信号，提出DITTO模型，并构建SOUL基准测试，显著提升了模型的拟人化表现。**

- **链接: [https://arxiv.org/pdf/2605.20506](https://arxiv.org/pdf/2605.20506)**

> **作者:** Weiwei Sun; Xuhui Zhou; Jiarui Liu; Weihua Du; Haojia Sun; Yiqing Xie; Qianou Ma; Sihao Chen; Mengting Wan; Longqi Yang; Pei Zhou; Sherry Wu; Sean Welleck; Graham Neubig; Yiming Yang; Maarten Sap
>
> **摘要:** Humans learn social norms and behaviors from verbal feedback (e.g., a parent saying "that was rude" or a friend explaining "here's why that hurt"). Yet, learning from feedback for LLMs has largely focused on domains like code and math, where RL rewards are directly verifiable and condensed into scalar values. As LLMs are increasingly used to simulate human behavior, e.g., standing in for users, patients, students, and other personas, there is a pressing need to make them more human-like, which requires embracing a fundamentally different kind of signal: feedback that is verbal, subjective, and multi-faceted. We present DITTO, a model trained by treating verbal feedback as a first-class signal in reinforcement learning. After each rollout, DITTO receives verbal feedback and generates a feedback-conditioned improved rollout; both outputs are jointly optimized with GRPO, distilling verbal guidance into the base policy without requiring feedback at test time. We also introduce SOUL (Simulation gym Of hUman-Like behavior), a unified benchmark and training data suite spanning 10 tasks across six categories: Theory of Mind, character role play, social skill, learner simulation, user simulation, and persona simulation. DITTO achieves an average 36% improvement over the base model and exceeds GPT-5.4 on 6 of 10 SOUL benchmarks, demonstrating that RL with verbal feedback is a promising direction for training LLMs to simulate human behavior.
>
---
#### [new 101] ChunkFT: Byte-Streamed Optimization for Memory-Efficient Full Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ChunkFT，解决全参数微调内存效率问题，通过动态激活工作集实现高效优化，适用于大模型微调。**

- **链接: [https://arxiv.org/pdf/2605.21177](https://arxiv.org/pdf/2605.21177)**

> **作者:** Yongkang Liu; Zijing Wang; Mengjie Zhao; Ercong Nie; Mingyang Wang; Qian Li; Feiliang Ren; Shi Feng; Daling Wang; Hinrich Schütze
>
> **摘要:** This work presents \textsc{ChunkFT}, a memory-efficient fine-tuning framework that reformulates full-parameter fine-tuning around a dynamically activated working set. \textsc{ChunkFT} enables gradient computation for arbitrary sub-tensors without modifying the network architecture, providing an algorithmic foundation for optimizing arbitrary sub-networks while avoiding standard dense gradient computation. We provide a theoretical convergence analysis of \textsc{ChunkFT} in the deterministic setting. Empirically, we apply \textsc{ChunkFT} to fine-tune Llama 3-8B and Llama 3-70B using a single RTX 4090-24GB GPU and 2$\times$ H800-80GB GPUs, respectively. Full-parameter fine-tuning of a 7B model with a 1K input length requires only 13.72GB of GPU memory. The results demonstrate the effectiveness of \textsc{ChunkFT} in memory usage, running time, and optimization quality. Moreover, downstream evaluations on language understanding, mathematical reasoning, and MT-Bench show that \textsc{ChunkFT} consistently outperforms existing memory-efficient baselines. Notably, \textsc{ChunkFT} achieves performance comparable to, and in some cases exceeding, full-parameter fine-tuning. Our repository is on this https URL.
>
---
#### [new 102] Geometry-Lite: Interpretable Safety Probing via Layer-Wise Margin Geometry
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型安全检测任务，旨在理解大语言模型中安全证据的几何结构。通过分析多层表示，提出Geometry-Lite方法，揭示安全决策的关键几何特征。**

- **链接: [https://arxiv.org/pdf/2605.20241](https://arxiv.org/pdf/2605.20241)**

> **作者:** Woo Seob Sim; Yu Rang Park
>
> **摘要:** Prompt-level safety probes for large language models use hidden-state representations to separate safe from unsafe prompts, but strong average detection performance does not explain the geometry of this separation. In particular, it remains unclear how safety evidence is formed across layers, which aspects of that layer-wise geometry support low-false-positive decisions, and which geometric biases remain stable under benchmark shift. We study this as an empirical decomposition problem and introduce Geometry-Lite, a compact prompt-level probe that maps each layer's final prompt-token representation to signed margins under centroid, local-neighborhood, and supervised linear-boundary readouts, then summarizes the resulting margin profiles by boundary position, layer-to-layer change, and coarse shape. Across nine instruction-tuned backbones ($1.2$B--$70$B) and seven safety benchmarks, Geometry-Lite improves over single-layer probes while remaining close to raw multi-layer score stacking, making it a useful instrument for analyzing the multi-layer safety signal. The decomposition shows that safety evidence is expressed primarily through persistent boundary-position geometry: final or extremal margins and unsafe-side layer occupancy dominate aggregate detection performance. In contrast, finite-difference drift and structural summaries add little to pooled AUROC, although drift can provide small recall-oriented corrections under shifted low-FPR thresholds. Under benchmark shift, optimized linear boundaries are sharp on the training mixture, whereas class-conditional mean geometry retains separation more reliably on a predefined hard held-out subset. Overall, prompt-level safety evidence is not primarily a layer-to-layer motion signal, but a persistent layer-wise margin geometry whose useful components and readout-level biases become visible in decision-critical regimes.
>
---
#### [new 103] NeuroQA: A Large-Scale Image-Grounded Benchmark for 3D Brain MRI Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; eess.IV**

- **简介: 该论文提出NeuroQA，一个用于3D脑部MRI理解的视觉问答基准，解决医学影像分析中的多模态任务，通过大规模数据集和严格验证确保答案的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.20525](https://arxiv.org/pdf/2605.20525)**

> **作者:** Mohammad H. Abbasi; Favour Nerrise; Shaurnav Ghosh; Ridvan Yesiloglu; Yuncong Mao; Bailey Trang; Mohammad Asadi; Merryn Daniel; Gustavo Chau Loo Kung; Ken Chang; Pavan Pinkesh Shah; Adam Turnbull; Kyan Younes; Seena Dehkharghani; Ehsan Adeli
>
> **备注:** 30 pages, dataset and benchmark release
>
> **摘要:** We present NeuroQA, a large-scale benchmark for visual question answering in 3D brain magnetic resonance imaging (MRI), with 56,953 QA pairs from 12,977 subjects across 12 datasets. It spans ages 5-104 and five clinical domains: Alzheimer's, Parkinson's, tumors, white matter disease, and neurodevelopment. Unlike prior medical Visual Question Answering (VQA) efforts that operate on 2D slices or rely on narrow diagnostic labels, NeuroQA pairs every item with a full 3D volume. It evaluates 11 clinically grounded reasoning skills across Yes/No, multiple-choice, and open-ended formats. Of the 203 templates, 131 are image-grounded (answerable from a 3-plane viewer) and 72 are image-informed (ground truth from quantitative volumetry or clinical instruments). To remove text-only shortcuts, we apply answer-distribution refinement, reducing closed-format text-only accuracy from $>$80% to 44.6%; image necessity is assessed separately through an image-grounding protocol released with the benchmark. A 38-rule deterministic pipeline and two rounds of expert review verify every QA pair against FreeSurfer measurements, metadata, or radiology report fields, with zero same-subject contradictions across templates. We conduct a clinician evaluation in which two clinicians independently assess 100 frozen test items on a three-plane viewer. On closed-format (Yes/No + multiple-choice) test-public items, the best zero-shot vision-language model and a supervised 3D CNN baseline reach 47.5% and 43.7% accuracy respectively, both below the 49.4% text-only majority-template floor. NeuroQA adopts a two-tier release with public QA pairs for open-access datasets and reproducible generation scripts for datasets restricted by data use agreements (DUAs), plus subject-level splits, a held-out private test set, and an online leaderboard.
>
---
#### [new 104] SMoA: Spectrum Modulation Adapter for Parameter-Efficient Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SMoA，解决参数高效微调中低秩更新的表示能力不足问题，通过谱调制适配器提升性能。**

- **链接: [https://arxiv.org/pdf/2605.21147](https://arxiv.org/pdf/2605.21147)**

> **作者:** Yongkang Liu; Xing Li; Mengjie Zhao; Shanru Zhang; Zijing Wang; Qian Li; Shi Feng; Feiliang Ren; Daling Wang; Hinrich Schütze
>
> **摘要:** As the number of model parameters increases, parameter-efficient fine-tuning (PEFT) has become the go-to choice for tailoring pre-trained large language models. Low-rank Adaptation (LoRA) uses a low-rank update method to simulate full parameter fine-tuning, which is widely used to reduce resource requirements. However, decreasing the rank encounters challenges with limited representational capacity. Theory suggests that LoRA fine-tuning with rank r converges toward the top r singular values of the pre-trained weight matrix. As the rank increases, more principal singular directions are preserved, which generally improves the model's performance. However, a larger rank also introduces more trainable parameters, leading to higher computational cost. To overcome this dilemma, we propose SMoA, a \textbf{S}pectrum \textbf{Mo}dulation \textbf{A}dapter that enlarges the accessible family of spectrum-aware updates under a smaller parameter budget. SMoA partitions the layer into multiple aligned spectral blocks and applies one in-block Hadamard-modulated low-rank branch to each diagonal block, yielding broader coverage of pretrained spectral directions. We provide theoretical analysis and empirical results on multiple tasks. In our experiments, SMoA improves average performance in the current lower-budget setting over LoRA and competitive LoRA-style baselines.
>
---
#### [new 105] AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AVSD方法，解决自蒸馏中多视图特权信息不平衡问题，通过分离共识与残差信号提升模型性能，应用于数学和代码生成任务。**

- **链接: [https://arxiv.org/pdf/2605.20643](https://arxiv.org/pdf/2605.20643)**

> **作者:** Duy Nguyen; Hanqi Xiao; Archiki Prasad; Zaid Khan; Anirban Das; Austin Zhang; Sambit Sahu; Hyunji Lee; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Code: this https URL
>
> **摘要:** Self-distillation enables language models to learn on-policy from their own trajectories by using the same model as both student and teacher, with the teacher being conditioned on privileged information unavailable to the student. Such information can come in different types or views, such as solutions, demonstrations, feedback, or final answers. This setup provides dense token-level feedback without relying on a separate external model, but creates a fundamental asymmetry: the teacher may rely on view-specific information that the student cannot access at inference time. Moreover, the best type of privileged information is often task-dependent, making it difficult to choose a single teacher view. In this work, we address both these challenges jointly by introducing AVSD (Adaptive-View Self-Distillation), a novel method of self-distillation with multiple privileged-information views, which reconstructs token-level supervision by separating stable cross-view consensus from view-specific residual signals. AVSD identifies the consensus signal shared across views, which provides a reliable update direction, and then selectively adds the view-specific residual signal to adjust the update magnitude when it both aligns with the consensus direction and remains proportionate to the consensus signal. Experiments on math competition benchmarks (AIME24, AIME25, and HMMT25) show that AVSD consistently outperforms both single-view self-distillation baselines and GRPO, achieving average Avg@8 gains of 3.1% and 2.2% over the strongest baselines on Qwen3-8B and Qwen3-4B, respectively. Moreover, on code-generation benchmarks (Codeforces, LiveCodeBench v6) using Qwen3-8B, AVSD outperforms the single-view self-distillation baseline by 2.4% on average.
>
---
## 更新

#### [replaced 001] DiM\textsuperscript{3}: Bridging Multilingual and Multimodal Models via Direction- and Magnitude-Aware Merging
- **分类: cs.CL**

- **简介: 该论文属于多语言与多模态模型融合任务，旨在无需重新训练即可提升多模态模型的多语言能力。通过DiM3方法，有效整合多语言和多模态更新，增强跨语言对齐并保持多模态性能。**

- **链接: [https://arxiv.org/pdf/2605.12960](https://arxiv.org/pdf/2605.12960)**

> **作者:** Zijing Wang; Mingyang Wang; Ercong Nie; Yongkang Liu; Shi Feng; Mengjie Zhao; Daling Wang; Xiaocui Yang; Hinrich Schütze
>
> **摘要:** Towards more general and human-like intelligence, large language models should seamlessly integrate both multilingual and multimodal capabilities; however, extending an existing multimodal model to many languages typically requires expensive multilingual multimodal data construction and repeated end-to-end retraining. We study a training-free alternative: injecting multilingual capability into an existing multimodal model by composing residual updates in the shared language model backbone. The key challenge is that multilingual and multimodal updates are heterogeneous, reflecting different functional roles in the shared model. To address this, we propose Direction- and Magnitude-aware Multilingual Multimodal merging (DiM3), which selectively composes the two updates at each parameter dimension while preserving the original vision encoder and multimodal projector. Experiments on multilingual benchmarks in both text-only and vision-language settings, covering 57 languages across LLaVA- and Qwen-based backbones, show that DiM3 consistently outperforms existing merging baselines, substantially improves multilingual performance over the original multimodal model, and remains competitive with dedicated multilingual multimodal fine-tuning while largely retaining general multimodal ability. We further show that DiM3 can be directly applied to already trained multilingual multimodal models and still yield additional gains. Further interpretability analysis shows that DiM3 primarily reshapes intermediate-layer semantic representations, strengthening cross-lingual alignment under both text-only and multimodal inputs while preserving higher-layer task-sensitive structure. Our repository is on this https URL.
>
---
#### [replaced 002] APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决LLM生成中的幻觉问题。提出APCD框架，通过自适应路径探索和对比增强可靠性。**

- **链接: [https://arxiv.org/pdf/2605.09492](https://arxiv.org/pdf/2605.09492)**

> **作者:** Tianyu Zheng; Hong Wu; Jiaji Zhong
>
> **备注:** This paper has been withdrawn by the author to resolve a conflict of interest/compliance issue
>
> **摘要:** Large language models (LLMs) often suffer from hallucinations due to error accumulation in autoregressive decoding, where suboptimal early token choices misguide subsequent generation. Although multi-path decoding can improve robustness by exploring alternative trajectories, existing methods lack principled strategies for determining when to branch and how to regulate inter-path interactions. We propose Adaptive Path-Contrastive Decoding (APCD), a multi-path decoding framework that improves output reliability through adaptive exploration and controlled path interaction. APCD consists of two components: (1) Entropy-Driven Path Expansion, which delays branching until predictive uncertainty - measured by Shannon entropy over top candidate tokens - indicates multiple plausible continuations; and (2) Divergence-Aware Path Contrast, which encourages diverse reasoning trajectories while dynamically attenuating inter-path influence as prediction distributions diverge. Experiments on eight benchmarks demonstrate improved factual accuracy while maintaining decoding efficiency. Our code is available at this https URL.
>
---
#### [replaced 003] Bayesian Preference Learning for Test-Time Steerable Reward Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习中的奖励建模任务，解决传统奖励模型在测试时无法适应新偏好问题。提出ICRM方法，通过贝叶斯推理实现测试时的偏好调整。**

- **链接: [https://arxiv.org/pdf/2602.08819](https://arxiv.org/pdf/2602.08819)**

> **作者:** Jiwoo Hong; Shao Tang; Zhipeng Wang
>
> **备注:** Preprint
>
> **摘要:** Reward models are central to aligning language models with human preferences via reinforcement learning (RL). As RL is increasingly applied to settings such as verifiable rewards and multi-objective alignment, RMs are expected to encode more complex and multifaceted preference distributions. However, classifier RMs remain static once trained, limiting their adaptability at test time. We propose Variational In-Context Reward Modeling (ICRM), a novel Bayesian reward modeling objective that enables test-time steerability via in-context preference demonstrations. ICRM casts reward modeling as amortized variational inference over a latent preference probability under the Bradley-Terry model using a conjugate Beta prior. We show that ICRM adapts to unseen preference distributions at test time for both single and multi-objective settings. With more demonstrations, ICRM improves RM-Bench accuracy from 60.5 to 70.8, achieves lower calibration error than a generative judge on moral dilemma preferences, and expands the attainable Pareto frontier under conflicting preferences. We further study the practical applicability of ICRM for RL training, showing that it can effectively encode verifiable rewards by outperforming a conventional RM in math reasoning. Finally, we provide theoretical guarantees that the variational objective admits a global interior optimum with finite confidence, and we analyze how KL regularization mitigates reward over-optimization.
>
---
#### [replaced 004] Optimal Query Allocation in Extractive QA with LLMs: A Learning-to-Defer Framework with Theoretical Guarantees
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于抽取式问答任务，解决LLMs在结构化文本选择中的效率问题。提出一种学习性延迟框架，优化查询分配，提升可靠性并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2410.15761](https://arxiv.org/pdf/2410.15761)**

> **作者:** Yannis Montreuil; Shu Heng Yeo; Axel Carlier; Lai Xing Ng; Wei Tsang Ooi
>
> **备注:** 25 pages, 17 main paper
>
> **摘要:** Large Language Models excel in generative tasks but exhibit inefficiencies in structured text selection, particularly in extractive question answering. This challenge is magnified in resource-constrained environments, where deploying multiple specialized models for different tasks is impractical. We propose a Learning-to-Defer framework that allocates queries to specialized experts, ensuring high-confidence predictions while optimizing computational efficiency. Our approach integrates a principled allocation strategy with theoretical guarantees on optimal deferral that balances performance and cost. Empirical evaluations on SQuADv1, SQuADv2, and TriviaQA demonstrate that our method enhances answer reliability while significantly reducing computational overhead, making it well-suited for scalable and efficient EQA deployment.
>
---
#### [replaced 005] Toxic Subword Pruning for Dialogue Response Generation on Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于对话生成任务，旨在解决大语言模型生成有害内容的问题。提出ToxPrune算法，通过修剪有毒子词提升模型安全性与多样性。**

- **链接: [https://arxiv.org/pdf/2410.04155](https://arxiv.org/pdf/2410.04155)**

> **作者:** Hongyuan Lu; Wai Lam
>
> **备注:** ACL *SEM 2026
>
> **摘要:** How to defend large language models (LLMs) from generating toxic content is an important research area. Yet, most research focused on various model training techniques to remediate LLMs by updating their weights. A typical related research area is safety alignment. This however is often costly and tedious and can expose the model to even more problems such as catastrophic forgetting if the trainings are not carefully handled by experienced NLP practitioners. We thus propose a simple yet effective and novel algorithm, namely \textbf{Tox}ic Subword \textbf{Prun}ing (ToxPrune) to prune the subword contained by the toxic words from BPE in trained LLMs. In contrast to the previous work that demonstrates pruning BPE tokens as harmful to the task of machine translation, we surprisingly found its usefulness in preventing toxic content from being generated on LLMs. Fortunately, our findings suggest that ToxPrune simultaneously improves the toxic language model NSFW-3B on the task of dialogue response generation obviously. We surprisingly found that ToxPrune can even obviously improve official Llama-3.1-6B in the metric of dialogue diversity. Extensive automatic results and human evaluation indicate that ToxPrune could be helpful for both remediating toxic LLMs and improving non-toxic LLMs on the task of dialogue response generation.\footnote{We plan to release the resources to facilitate future work.}
>
---
#### [replaced 006] Gated Normalization Removal and Scale Anchoring in Pre-Norm Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究预归一化Transformer中的归一化层作用，提出TaperNorm方法逐步移除归一化，发现最终归一化对模型尺度有锚定作用，并提升解码效率。**

- **链接: [https://arxiv.org/pdf/2602.10408](https://arxiv.org/pdf/2602.10408)**

> **作者:** Andrei Kanavalau; Carmen Amo Alonso; Sanjay Lall
>
> **摘要:** Normalization layers are standard in transformers, but it is not clear whether their sample-dependent computations are necessary throughout both training and inference. This work develops a gated normalization-removal approach for pre-norm transformers. The approach is implemented using TaperNorm, which starts from standard RMSNorm/LayerNorm and gradually tapers to learned sample-independent linear or affine maps. Once the gate reaches zero, per-token statistics are no longer computed in the tapered layers and the resulting maps can be folded into adjacent linear projections. The results indicate that internal normalization can be tapered in the tested pre-training and fine-tuning settings with small validation-loss increases. Our approach helps reveal a distinct role for final normalization, namely that it anchors the scale of the pre-logit representation. With this anchor present, radial changes in the last hidden state do not directly reduce the loss; when it is removed, reducing cross-entropy can be achieved by increasing logit magnitudes. A fixed-target scale loss provides an explicit alternative anchor and enables fully norm-free ablations in the tested regimes. Finally, in a KV-cached autoregressive decoding benchmark, tapering internal norms gives up to $1.14\times$ higher throughput with explicit scaling operations and up to $1.18\times$ after folding.
>
---
#### [replaced 007] Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出HEAR，解决企业系统中LLM的幻觉和多跳推理问题。通过分层超图本体实现结构化推理，提升准确性和可审计性。**

- **链接: [https://arxiv.org/pdf/2605.14259](https://arxiv.org/pdf/2605.14259)**

> **作者:** Ling Wang; Xin Liu; Songnan Liu; Jianan Wang; Cheng Cheng; Yihan Zhu; Enyu Li; Yu Xiao; Jiangyong Xie; Duogong Yan; Jiangyi Chen
>
> **摘要:** Applying Large Language Models (LLMs) to heterogeneous enterprise systems is hindered by hallucinations and failures in multi-hop, n-ary reasoning. Existing paradigms (e.g., GraphRAG, NL2SQL) lack the semantic grounding and auditable execution required for these complex environments. We introduce HEAR, an enterprise agentic reasoner built on a Stratified Hypergraph Ontology. Its base Graph Layer virtualizes provenance-aware data interfaces, while the Hyperedge Layer encodes n-ary business rules and procedural protocols. Operating an evidence-driven reasoning loop, HEAR dynamically orchestrates ontology tools for structured multi-hop analysis without requiring LLM retraining. Evaluations on supply-chain tasks, including order fulfillment blockage root cause analysis (RCA), show HEAR achieves up to 94.7% accuracy. Crucially, HEAR demonstrates adaptive efficiency: utilizing procedural hyperedges to minimize token costs, while leveraging topological exploration for rigorous correctness on complex queries. By matching proprietary model performance with open-weight backbones and automating manual diagnostics, HEAR establishes a scalable, auditable foundation for enterprise intelligence.
>
---
#### [replaced 008] From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于语音代理评估任务，旨在将文本基准转化为音频评估，无需重新标注。工作包括构建框架、测试模型表现，并验证隐私保护的评估方法。**

- **链接: [https://arxiv.org/pdf/2605.15104](https://arxiv.org/pdf/2605.15104)**

> **作者:** Md Tahmid Rahman Laskar; Xue-Yong Fu; Seyyed Saeed Sarfjoo; Quinten McNamara; Jonas Robertson; Shashi Bhushan TN
>
> **摘要:** Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demonstrates that degradations most often reflect misunderstandings of argument values in the speech. Considering real-world deployment scenarios, we further report text-only results, an ambiguity-based reformulation stress test, and a reference-free LLM-as-judge protocol validated against human preferences. Notably, we find that open-source Qwen3 judges with at least 8B parameters exceed 80% agreement with proprietary judges, supporting privacy-preserving evaluation. Overall, our framework provides a verifiable and reproducible first-stage diagnostic that complements purpose-built audio corpora.
>
---
#### [replaced 009] Task-conditioned probing of instruction-tuned multimodal LLMs: Region-specific brain alignment patterns under naturalistic stimuli
- **分类: q-bio.NC; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文研究IT-MLLMs在自然刺激下的脑对齐情况，旨在解决任务引导与脑活动关联的问题。通过分析视频和音频任务，发现指令调优模型具有更高的脑对齐效果。**

- **链接: [https://arxiv.org/pdf/2506.08277](https://arxiv.org/pdf/2506.08277)**

> **作者:** Subba Reddy Oota; Khushbu Pahwa; Prachi Jindal; Satya Sai Srinath Namburi; Maneesh Singh; Tanmoy Chakraborty; Bapi S. Raju; Manish Gupta
>
> **备注:** 57 pages, 39 figures
>
> **摘要:** Recent voxel-wise multimodal brain encoding studies have shown that multimodal large language models (MLLMs) exhibit a higher degree of brain alignment compared to unimodal models. More recently, instruction-tuned multimodal (IT) models have been shown to generate task-specific representations that align strongly with brain activity, yet most prior evaluations focus on unimodal stimuli or non-instruction-tuned models under multimodal stimuli. We still lack a clear understanding of whether instruction-tuning is associated with IT-MLLMs organizing their representations around functional task demands or if they simply reflect surface semantics. To address this, we estimate brain alignment by predicting fMRI responses recorded during naturalistic movie watching (video with audio) from MLLM representations. Using instruction-specific embeddings from six video and two audio IT-MLLMs, across 13 video task instructions, we find that instruction-tuned video MLLMs show higher brain alignment than in-context learning (ICL) multimodal models (~9%), non-instruction-tuned multimodal models (~15%), and unimodal baselines (~20%). Our evaluation of MLLMs across video and audio tasks, and language-guided probing produces distinct task-specific MLLM representations that vary across brain regions. We also find that ICL models show strong semantic organization (r=0.78), while IT models show weak coupling to instruction-text semantics (r=0.14), consistent with task-conditioned subspaces associated with higher brain alignment. These findings are consistent with an association between task-specific instructions and stronger brain-MLLM alignment, and open new avenues for mapping joint information processing in both systems. We make the code publicly available [this https URL].
>
---
#### [replaced 010] DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决MoE模型在端侧设备部署中的存储与计算瓶颈问题。提出DECO架构，在相同参数预算下实现密集模型性能，并提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.10933](https://arxiv.org/pdf/2605.10933)**

> **作者:** Chenyang Song; Weilin Zhao; Xu Han; Chaojun Xiao; Yingfa Chen; Zhiyuan Liu
>
> **备注:** 15 pages, 10 figures, 12 tables
>
> **摘要:** While Mixture-of-Experts (MoE) scales model capacity without proportionally increasing computation, its massive total parameter footprint creates significant storage and memory-access bottlenecks, which hinder efficient end-side deployment that simultaneously requires high performance, low computational cost, and small storage overhead. To achieve these properties, we present DECO, a sparse MoE architecture designed to match the performance of dense Transformers under identical total parameter budgets and training tokens. DECO utilizes the differentiable and flexible ReLU-based routing enhanced by learnable expert-wise scaling, which adaptively balances the contributions of routed and shared experts. Furthermore, we introduce NormSiLU, an activation function that normalizes inputs prior to SiLU operators, producing a more stable trend of routed-expert activation ratio and a higher intrinsic sparsity level. We also identify an empirical advantage in using non-gated MLP experts with ReLU-based routing, indicating the possibility of MoE architecture simplification. Experiments demonstrate that DECO, activating only 20% of routed experts, matches dense performance and outperforms established MoE baselines. Our specialized acceleration kernel delivers a 2.93$\times$ speedup on Jetson AGX Orin compared with dense inference. Code and checkpoints are available at this https URL.
>
---
#### [replaced 011] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models
- **分类: cs.CL**

- **简介: 该论文提出iReasoner框架，解决大模型在无监督下推理能力不足的问题。通过强化中间推理过程，提升多模态模型的内在推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05877](https://arxiv.org/pdf/2601.05877)**

> **作者:** Meghana Sunil; Manikandarajan Venmathimaran; Muthu Subash Kavitha
>
> **备注:** ACL 2026 (Findings)
>
> **摘要:** Recent work shows that large multimodal models (LMMs) can self-improve from unlabeled data via self-play and intrinsic feedback. Yet existing self-evolving frameworks mainly reward final outcomes, leaving intermediate reasoning weakly constrained despite its importance for visually grounded decision making. We propose iReasoner, a self-evolving framework that improves an LMM's implicit reasoning by explicitly eliciting chain-of-thought (CoT) and rewarding its internal agreement. In a Proposer--Solver loop over unlabeled images, iReasoner augments outcome-level intrinsic rewards with a trajectory-aware signal defined over intermediate reasoning steps, providing learning signals that distinguish reasoning paths leading to the same answer without ground-truth labels or external judges. Starting from Qwen2.5-VL-7B, iReasoner yields up to $+2.1$ points across diverse multimodal reasoning benchmarks under fully unsupervised post-training. We hope this work serves as a starting point for reasoning-aware self-improvement in LMMs in purely unsupervised settings. Our code is available at this https URL.
>
---
#### [replaced 012] Argus: Evidence Assembly for Scalable Deep Research Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Argus系统，解决深度研究中证据碎片化问题。通过Searcher与Navigator协作，高效组装证据，提升回答质量。属于信息检索任务。**

- **链接: [https://arxiv.org/pdf/2605.16217](https://arxiv.org/pdf/2605.16217)**

> **作者:** Zhen Zhang; Liangcai Su; Zhuo Chen; Xiang Lin; Haotian Xu; Simon Shaolei Du; Kaiyu Yang; Bo An; Lidong Bing; Xinyu Wang
>
> **摘要:** Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens.
>
---
#### [replaced 013] Do LLMs Triage Like Clinicians? A Dynamic Study of Outpatient Referral
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLMs在门诊转诊中的表现，探讨其是否能通过动态交互提升决策效果。任务是评估LLMs在动态转诊中的价值，解决静态模型不足的问题，通过多轮对话减少不确定性。**

- **链接: [https://arxiv.org/pdf/2503.08292](https://arxiv.org/pdf/2503.08292)**

> **作者:** Xiaoxiao Liu; Qingying Xiao; Bingquan Zhang; Junying Chen; Xiangyi Feng; Ziniu Li; Xiang Wan; Jian Chang; Guangjun Yu; Yan Hu; Benyou Wang
>
> **摘要:** Outpatient referral (OR) is a core clinical workflow that assigns patients to hospital departments under incomplete and evolving information, yet it is commonly simplified as a static classification problem despite being inherently interactive in practice. In this work, we study outpatient referral as a dynamic process driven by information acquisition and uncertainty reduction. We analyze both static scenarios based on fixed patient information and dynamic scenarios involving multi-turn dialogue, to test whether large language models (LLMs) improve referral outcomes through better prediction or more effective questioning. Our findings show that LLMs offer limited advantages over traditional classifiers in static referral accuracy, but consistently outperform them in dynamic settings by asking discriminative follow-up questions that reduce uncertainty over candidate departments. These results suggest that the primary value of LLMs in outpatient referral lies not in static prediction, but in supporting interactive, uncertainty-aware clinical decision-making.
>
---
#### [replaced 014] Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?
- **分类: cs.CL; cs.LG**

- **简介: 论文探讨自蒸馏在大语言模型推理能力上的负面影响，指出其可能抑制模型表达不确定性，导致性能下降。任务为模型优化，问题为自蒸馏为何有时损害推理，工作包括实验分析与性能评估。**

- **链接: [https://arxiv.org/pdf/2603.24472](https://arxiv.org/pdf/2603.24472)**

> **作者:** Jeonghye Kim; Xufang Luo; Minbeom Kim; Sangmook Lee; Dohyung Kim; Jiwon Jeon; Dongsheng Li; Yuqing Yang
>
> **备注:** Code is available at this https URL
>
> **摘要:** Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-1.7B/8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
>
---
#### [replaced 015] Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG
- **分类: cs.CL**

- **简介: 该论文属于问答系统任务，旨在解决RAG系统中证据使用不准确导致的幻觉问题。通过构建facet-level分析框架，评估证据整合情况，揭示系统在证据缺失、错位和先验干扰等方面的失败模式。**

- **链接: [https://arxiv.org/pdf/2604.09174](https://arxiv.org/pdf/2604.09174)**

> **作者:** Passant Elchafei; Monorama Swain; Shahed Masoudian; Markus Schedl
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to reduce hallucination by grounding answers in retrieved evidence, yet hallucinated answers remain common even when relevant documents are available. Existing evaluations focus on answer-level or passage-level accuracy, offering limited insight into how evidence is used during generation. In this work, we introduce a facet-level diagnostics framework for QA that decomposes each input question into atomic reasoning facets. For each facet, we assess evidence sufficiency and grounding using a structured Facet x Chunk matrix that combines retrieval relevance with natural language inference-based faithfulness scores. To diagnose evidence usage, we analyze three controlled inference modes: Strict RAG, which enforces exclusive reliance on retrieved evidence; Soft RAG, which allows integration of retrieved evidence and parametric knowledge; and LLM-only generation without retrieval. Comparing these modes enables thorough analysis of retrieval-generation misalignment, defined as cases where relevant evidence is retrieved but not correctly integrated during generation. Across medical QA and HotpotQA, we evaluate three open-source and closed-source LLMs (GPT, Gemini, and LLaMA), providing interpretable diagnostics that reveal recurring facet-level failure modes, including evidence absence, evidence misalignment, and prior-driven overrides. Our results demonstrate that hallucinations in RAG systems are driven less by retrieval accuracy and more by how retrieved evidence is integrated during generation, with facet-level analysis exposing systematic evidence override and misalignment patterns that remain hidden under answer-level evaluation.
>
---
#### [replaced 016] Bridging Language Models and Financial Analysis
- **分类: q-fin.ST; cs.AI; cs.CL**

- **简介: 该论文属于金融与语言模型交叉研究任务，旨在解决LLM在金融领域应用不足的问题。通过综述最新LLM技术，探讨其在金融数据分析中的潜力与应用方向。**

- **链接: [https://arxiv.org/pdf/2503.22693](https://arxiv.org/pdf/2503.22693)**

> **作者:** Alejandro Lopez-Lira; Jihoon Kwon; Sangwoon Yoon; Jy-yong Sohn; Chanyeol Choi
>
> **备注:** 28 pages
>
> **摘要:** The rapid advancements in Large Language Models (LLMs) have unlocked transformative possibilities in natural language processing, particularly within the financial sector. Financial data is often embedded in intricate relationships across textual content, numerical tables, and visual charts, posing challenges that traditional methods struggle to address effectively. However, the emergence of LLMs offers new pathways for processing and analyzing this multifaceted data with increased efficiency and insight. Despite the fast pace of innovation in LLM research, there remains a significant gap in their practical adoption within the finance industry, where cautious integration and long-term validation are prioritized. This disparity has led to a slower implementation of emerging LLM techniques, despite their immense potential in financial applications. As a result, many of the latest advancements in LLM technology remain underexplored or not fully utilized in this domain. This survey seeks to bridge this gap by providing a comprehensive overview of recent developments in LLM research and examining their applicability to the financial sector. Building on previous survey literature, we highlight several novel LLM methodologies, exploring their distinctive capabilities and their potential relevance to financial data analysis. By synthesizing insights from a broad range of studies, this paper aims to serve as a valuable resource for researchers and practitioners, offering direction on promising research avenues and outlining future opportunities for advancing LLM applications in finance.
>
---
#### [replaced 017] Causal Path Alignment: Anchoring the Optimization Trajectory for Controllable In-Parameter Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，解决LLM中因修改事实导致的结构知识干扰问题。提出CPA框架，通过锚定优化路径确保关系特异性，减少副作用。**

- **链接: [https://arxiv.org/pdf/2506.04042](https://arxiv.org/pdf/2506.04042)**

> **作者:** Xiyu Liu; Zhengxiao Liu; Naibin Gu; Zheng Lin; Weiping Wang
>
> **备注:** Accepted by IJCAI 2026
>
> **摘要:** Knowledge editing is pivotal for efficiently updating the parametric memory of Large Language Models (LLMs), enabling them to function as evolving agents in dynamic environments. However, mainstream in-parameter knowledge editing approaches suffer from Subject-Dominant Memory Interference: modifying a specific fact inadvertently corrupts the broader structural knowledge associated with the same subject within LLMs. We diagnose the root cause as a shortcut learning pathology, where the optimization objective overfits subject representations while bypassing the essential relational context. To rectify this, we propose Causal Path Alignment (CPA), a principled framework designed to anchor the optimization trajectory to valid causal pathways. CPA enforces parameter updates to route through relation-aware intermediate states, thereby preventing the erasure of contextual dependencies. Experimental results across diverse LLM backbones demonstrate that CPA consistently eliminates the shortcut, significantly improving relation specificity while exhibiting minimal side-effects. Moreover, CPA serves as a model-agnostic plug-in for existing editors, paving the way for reliable and trustworthy in-parameter knowledge editing.
>
---
#### [replaced 018] Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在比较EngGPT2MoE-16B-A3B与其他意大利及国际开源大语言模型的性能，分析其优势与不足。**

- **链接: [https://arxiv.org/pdf/2605.07731](https://arxiv.org/pdf/2605.07731)**

> **作者:** Andrea Sassella; Andrea Chizzola; Tommaso Bianchi; Luca Alessandrelli; Mark James Carman
>
> **摘要:** This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models.
>
---
#### [replaced 019] Retrospective Sparse Attention for Efficient Long-Context Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长文本生成中KV缓存效率低的问题。提出RetroAttention方法，通过回顾性修正提升注意力效果，显著提高生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2508.09001](https://arxiv.org/pdf/2508.09001)**

> **作者:** Seonghwan Choi; Beomseok Kang; Dongwon Jo; Jae-Joon Kim
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important few tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to be efficiently supplemented with more contexts, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%.
>
---
#### [replaced 020] LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLM的损失缩放规律。旨在探讨影响损失-损失缩放的关键因素，发现预训练数据是主要决定因素，为优化模型提供指导。**

- **链接: [https://arxiv.org/pdf/2502.12120](https://arxiv.org/pdf/2502.12120)**

> **作者:** Prasanna Mayilvahanan; Thaddäus Wiedemer; Sayak Mallick; Matthias Bethge; Wieland Brendel
>
> **备注:** ICML 2025 camera-ready version
>
> **摘要:** Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance and generalization. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data determines the scaling trend. In contrast, model size, optimization hyperparameters, tokenizer and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, generally have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
>
---
#### [replaced 021] AI-Assisted Scientific Assessment: A Case Study on Climate Change
- **分类: cs.CL**

- **简介: 该论文属于科学评估任务，旨在探讨AI在气候科学研究中的应用。通过案例研究，评估AI辅助科学评估的效果与挑战。**

- **链接: [https://arxiv.org/pdf/2602.09723](https://arxiv.org/pdf/2602.09723)**

> **作者:** Christian Buck; Levke Caesar; Michelle Chen Huebscher; Massimiliano Ciaramita; Erich M. Fischer; Zeke Hausfather; Özge Kart Tokmak; Reto Knutti; Markus Leippold; Joseph Ludescher; Katharine J. Mach; Sofia Palazzo Corner; Kasra Rafiezadeh Shahi; Johan Rockström; Joeri Rogelj; Boris Sakschewski
>
> **摘要:** The emerging paradigm of AI co-scientists focuses on tasks characterized by repeatable verification, where agents explore search spaces in 'guess and check' loops. This paradigm does not extend to problems where repeated evaluation is impossible and ground truth is established by the consensus synthesis of theory and existing evidence. We evaluate a Gemini-based AI environment designed to support collaborative scientific assessment, integrated into a standard scientific workflow. In collaboration with a diverse group of 13 scientists working in the field of climate science, we tested the system on a complex topic: the stability of the Atlantic Meridional Overturning Circulation (AMOC). Our results show that AI can accelerate the scientific workflow. The group produced a comprehensive synthesis of 79 papers through 104 revision cycles in just over 46 person-hours. AI contribution was significant: most AI-generated content was retained in the report. AI also helped maintain logical consistency and presentation quality. However, expert additions were crucial to ensure its acceptability: less than half of the report was produced by AI. Furthermore, substantial oversight was required to expand and elevate the content to rigorous scientific standards.
>
---
#### [replaced 022] JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.SE**

- **简介: 该论文提出JanusCoder，解决代码与视觉输出的多模态生成问题。构建了大规模多模态代码数据集，训练统一模型实现文本与视觉驱动的代码生成。**

- **链接: [https://arxiv.org/pdf/2510.23538](https://arxiv.org/pdf/2510.23538)**

> **作者:** Qiushi Sun; Jingyang Gong; Yang Liu; Qiaosheng Chen; Lei Li; Kai Chen; Qipeng Guo; Ben Kao; Fei Yuan
>
> **备注:** ICLR 2026 Camera Ready Version, with code and data available
>
> **摘要:** The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints are available at this https URL.
>
---
#### [replaced 023] SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SHINE，一种可扩展的上下文到LoRA适配器映射方法，解决LLM适应效率低的问题。通过单次前向传递生成高质量适配器，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2602.06358](https://arxiv.org/pdf/2602.06358)**

> **作者:** Yewei Liu; Xiyuan Wang; Yansheng Mao; Yoav Gelbery; Haggai Maron; Muhan Zhang
>
> **摘要:** We propose SHINE (Scalable Hyper In-context NEtwork), a scalable hypernetwork that can map diverse meaningful contexts into high-quality LoRA adapters for large language models (LLMs). By reusing the frozen LLM's own parameters in an in-context hypernetwork design and introducing architectural innovations, SHINE overcomes key limitations of prior hypernetworks and achieves strong expressive power with a relatively small number of parameters. We introduce a pretraining and instruction fine-tuning pipeline, and train our hypernetwork to generate high quality LoRA adapters from diverse meaningful contexts in a single forward pass. It updates LLM parameters without any fine-tuning, and immediately enables complex question answering tasks related to the context without directly accessing the context, effectively transforming in-context knowledge to in-parameter knowledge in one pass. Our work achieves outstanding results on various tasks, greatly saves time, computation and memory costs compared to SFT-based LLM adaptation, and shows great potential for scaling. Our code is available at this https URL
>
---
#### [replaced 024] WriteSAE: Sparse Autoencoders for Recurrent State
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出WriteSAE，用于替代语言模型中的矩阵更新，解决缓存控制问题。通过学习稀疏原子，实现对模型状态的直接干预，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.12770](https://arxiv.org/pdf/2605.12770)**

> **作者:** Jack Young
>
> **备注:** 26 pages, 14 figures, 21 tables; code at this https URL
>
> **摘要:** We introduce WriteSAE, a sparse autoencoder for the matrix updates written into recurrent language-model state. In Gated DeltaNet, Mamba-2, and RWKV-7, each token writes a matrix-shaped update to a recurrent cache; a residual-stream SAE has vector-shaped atoms and cannot replace that update directly. WriteSAE learns rank-1 matrix atoms with the same shape as the model's own write. This lets us test a direct replacement: at positions where the SAE activates an atom, we remove the model's write, insert the atom scaled by its SAE activation, and continue the forward pass. The atom gives a closer final token distribution than deleting the write on 92.4% of evaluated positions; averaged per atom, the rate is 89.8%. For Gated DeltaNet, a formula using the forget gate, read query, and output embedding predicts the resulting logit change with $R^2 = 0.98$. The same replacement test transfers to Mamba-2-370M at 88.1%. In generation, the formula chooses a write direction; writing it into three consecutive cache positions at $3\times$ the norm of the model's write makes tokens initially ranked 100--1000 by the unmodified model appear in 100% of continuations, up from 33.3%. To our knowledge this is the first cache-level steering intervention reported in a state-space or hybrid recurrent layer.
>
---
#### [replaced 025] The Visual Iconicity Challenge: Evaluating Vision-Language Models on Sign Language Form-Meaning Mapping
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决手语中形式与意义映射的视觉接地问题。通过构建基准测试，评估模型在音素预测、透明度和象似性评分上的表现。**

- **链接: [https://arxiv.org/pdf/2510.08482](https://arxiv.org/pdf/2510.08482)**

> **作者:** Onur Keleş; Aslı Özyürek; Gerardo Ortega; Kadir Gökgöz; Esam Ghaleb
>
> **摘要:** Iconicity, the resemblance between linguistic form and meaning, is pervasive in signed languages, offering a natural testbed for visual grounding. For vision-language models (VLMs), the challenge is to recover such essential mappings from dynamic human motion rather than static context. We introduce the Visual Iconicity Challenge, a novel video-based benchmark that adapts psycholinguistic measures to evaluate VLMs on three tasks: (i) phonological sign-form prediction (e.g., handshape, location), (ii) transparency (inferring meaning from visual form), and (iii) graded iconicity ratings. We assess 13 state-of-the-art VLMs in zero- and few-shot settings on Sign Language of the Netherlands and compare them to human baselines. On phonological form prediction, VLMs recover some handshape and location detail but remain below human performance; on transparency, they are far from human baselines; and only top models correlate moderately with human iconicity ratings. Interestingly, models with stronger phonological form prediction correlate better with human iconicity judgment, indicating shared sensitivity to visually grounded structure. Our findings validate these diagnostic tasks and motivate human-centric signals and embodied learning methods for modelling iconicity and improving visual grounding in multimodal models.
>
---
#### [replaced 026] Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context
- **分类: cs.CL**

- **简介: 该论文属于社会科技研究任务，探讨意大利语使用者在生成式AI的采用、素养和使用模式中的差异，分析其带来的数字鸿沟问题。**

- **链接: [https://arxiv.org/pdf/2512.03671](https://arxiv.org/pdf/2512.03671)**

> **作者:** Beatrice Savoldi; Giuseppe Attanasio; Olga Gorodetskaya; Marta Marchiori Manerba; Elisa Bassignana; Silvia Casola; Matteo Negri; Tommaso Caselli; Luisa Bentivogli; Alan Ramponi; Arianna Muti; Nicoletta Balbo; Debora Nozza
>
> **摘要:** The rise of generative AI (GenAI) chatbots accessible via conversational interfaces is transforming digital interactions and holds economic promise. However, these tools might deepen existing inequalities -- not only through uneven, socially stratified adoption, but through differentials in their purposeful, critical use. Drawing on original survey data from 1,906 Italian-speaking adults, we provide a comprehensive analysis of GenAI adoption, literacy, and usage patterns. Our findings show that GenAI is supporting diversified personal and professional activities and replacing traditional information-seeking tools. Yet less-educated and older individuals, and those with lower technology familiarity, are less likely to adopt it; 40% cite competence barriers as a key obstacle. Among users, AI training emerges as the primary predictor of purposeful, capital-enhancing engagement -- content creation, learning, and creativity enhancement -- while more passive, recreational uses (e.g., companionship, information seeking) remain insensitive to competence levels. We thus highlight digital literacy as a lever for how people leverage GenAI, not just whether they use it. Finally, gender operates as a persistent cross-cutting divide, shaping both adoption and usage frequency. These findings challenge the assumption that high accessibility translates into broadly shared gains. Rather, they offer a granular, multi-level account of emerging disparities in the GenAI era -- with implications for how this technology may ultimately drive outcomes and benefit divides.
>
---
#### [replaced 027] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出MASFactory框架，解决LLM多智能体系统流程构建困难的问题，通过Vibe Graphing实现自然语言到可执行图的转换。**

- **链接: [https://arxiv.org/pdf/2603.06007](https://arxiv.org/pdf/2603.06007)**

> **作者:** Yang Liu; Jinxuan Cai; Yishen Li; Qi Meng; Zedi Liu; Xin Li; Chen Qian; Chuan Shi; Cheng Yang
>
> **备注:** Accepted to the ACL 2026 Demo Track. Camera-ready version. 10 pages, 6 figures. Code and documentation are available at: this https URL
>
> **摘要:** Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents or sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components, skill support, multimodal message handling, and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (this https URL, licensed under Apache-2.0) and video demonstration (this https URL) are publicly available.
>
---
#### [replaced 028] DNACHUNKER: Learnable Tokenization for DNA Language Models
- **分类: q-bio.GN; cs.CL**

- **简介: 该论文提出DNAChunker，解决DNA序列建模中的分词问题。通过可学习的自适应分割模块，提升模型对基因组序列的表示能力。**

- **链接: [https://arxiv.org/pdf/2601.03019](https://arxiv.org/pdf/2601.03019)**

> **作者:** Taewon Kim; Jihwan Shin; Hyomin Kim; Youngmok Jung; Jonghoon Lee; Won-Chul Lee; Sungsoo Ahn; Insu Han
>
> **备注:** ICML 2026 camera-ready version
>
> **摘要:** DNA language models are increasingly used to represent genomic sequence, yet their effectiveness depends critically on how raw nucleotides are converted into model inputs. Unlike natural language, DNA offers no canonical boundaries, making fixed tokenizations a brittle design choice under shifts, indels, and local repeats. We introduce DNAChunker, a masked DNA language model that incorporates a learnable adaptive segmentation module to produce context-dependent, variable-length units. Building on a dynamic segmentation procedure, DNAChunker learns to allocate finer granularity to functionally enriched regions while compressing repetitive or redundant sequence. We pretrain DNAChunker on the human reference genome and evaluate it across five benchmarks, where it consistently improves over strong fixed-tokenization baselines. Further analyses and ablations indicate that unlike fixed tokenizations, segmentation is learned in a biologically-informed, mutation-resilient manner.
>
---
#### [replaced 029] Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于可解释AI任务，旨在解决Transformer模型决策过程难以解释的问题。提出CA-LIG框架，通过层次化梯度整合实现上下文感知的特征重要性分析。**

- **链接: [https://arxiv.org/pdf/2602.16608](https://arxiv.org/pdf/2602.16608)**

> **作者:** Melkamu Abay Mersha; Jugal Kalita
>
> **摘要:** Transformer models achieve state-of-the-art performance across domains and tasks, yet their deeply layered representations make their predictions difficult to interpret. Existing explainability methods rely on final-layer attributions, capture either local token-level attributions or global attention patterns without unification, and lack context-awareness of inter-token dependencies and structural components. They also fail to capture how relevance evolves across layers and how structural components shape decision-making. To address these limitations, we proposed the \textbf{Context-Aware Layer-wise Integrated Gradients (CA-LIG) Framework}, a unified hierarchical attribution framework that computes layer-wise Integrated Gradients within each Transformer block and fuses these token-level attributions with class-specific attention gradients. This integration yields signed, context-sensitive attribution maps that capture supportive and opposing evidence while tracing the hierarchical flow of relevance through the Transformer layers. We evaluate the CA-LIG Framework across diverse tasks, domains, and transformer model families, including sentiment analysis and long and multi-class document classification with BERT, hate speech detection in a low-resource language setting with XLM-R and AfroLM, and image classification with Masked Autoencoder vision Transformer model. Across all tasks and architectures, CA-LIG provides more faithful attributions, shows stronger sensitivity to contextual dependencies, and produces clearer, more semantically coherent visualizations than established explainability methods. These results indicate that CA-LIG provides a more comprehensive, context-aware, and reliable explanation of Transformer decision-making, advancing both the practical interpretability and conceptual understanding of deep neural models.
>
---
#### [replaced 030] A Theory of Time-Sensitive Language Generation: Sparse Hallucination Beats Mode Collapse
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言生成任务，解决及时生成问题。提出在保证时效性前提下，通过稀疏幻觉避免模式崩溃，实现最优密度生成。**

- **链接: [https://arxiv.org/pdf/2605.11302](https://arxiv.org/pdf/2605.11302)**

> **作者:** Atul Ganju; Travis McVoy; Shaddin Dughmi; Shang-Hua Teng
>
> **摘要:** We study language generation in the limit under a global preference ordering on strings, as introduced by Kleinberg and Wei. As is done in previous work, we aim for breadth, but impose an additional requirement of timeliness: higher-ranked strings should be generated earlier. A string is then only credited if it is generated before a deadline, where its deadline is defined by a function that maps a string's rank in the target language to the time by which it must be produced. This is in keeping with a central consideration in machine learning, where inductive bias favors ``simpler'' or ``more plausible'' outputs, all else being equal. We show that timely generation is impossible in a strong sense for eventually consistent generators -- the protagonists of most prior related work. Under what is perhaps the mildest natural relaxation of consistency, a hallucination rate that vanishes over time, we show that we can circumvent our impossibility result. In particular, we can achieve optimal density with respect to any superlinear deadline function. We also show this is tight by ruling out timely generation with linear deadlines and vanishing hallucination rate.
>
---
#### [replaced 031] Measuring and mitigating overreliance to build human-compatible AI
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于人工智能安全领域，旨在解决LLM过度假用问题。通过分析风险、探讨测量方法并提出缓解策略，确保LLM增强而非削弱人类能力。**

- **链接: [https://arxiv.org/pdf/2509.08010](https://arxiv.org/pdf/2509.08010)**

> **作者:** Lujain Ibrahim; Katherine M. Collins; Sunnie S. Y. Kim; Anka Reuel; Max Lamparth; Kevin Feng; Lama Ahmad; Prajna Soni; Alia El Kattan; Merlin Stein; Siddharth Swaroop; Vishakh Padmakumar; Ilia Sucholutsky; Andrew Strait; Diyi Yang; Q. Vera Liao; Umang Bhatt
>
> **摘要:** Large language models (LLMs) distinguish themselves from previous technologies by functioning as collaborative ``thought partners,'' capable of engaging more fluidly in natural language on a range of tasks. As LLMs increasingly influence consequential decisions across diverse domains from healthcare to personal advice, the risk of overreliance -- relying on LLMs beyond their capabilities -- grows. This paper argues that measuring and mitigating overreliance must become central to LLM research and deployment. First, we consolidate risks from overreliance at both the individual and societal levels, including high-stakes errors, governance challenges, and cognitive deskilling. Then, we explore LLM characteristics, system design features, and user cognitive biases that together raise serious and unique concerns about overreliance on LLMs in practice. We also examine historical approaches for measuring overreliance, identifying three important gaps and proposing three promising directions to improve measurement. Finally, we propose mitigation strategies that can be pursued to ensure LLMs augment rather than undermine human capabilities.
>
---
#### [replaced 032] PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分类任务，解决低资源环境下放射报告标签生成问题。提出PromptRad方法，通过知识增强的多标签提示调优，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2605.20052](https://arxiv.org/pdf/2605.20052)**

> **作者:** Ying-Jia Lin; Tzu-Chin Lo; Ping-Chien Li; Chi-Tung Cheng; Chien-Hung Liao; Hung-Yu Kao
>
> **备注:** BioNLP 2026 @ ACL (camera-ready version)
>
> **摘要:** Automatic report labeling facilitates the identification of clinical findings from unstructured text and enables large-scale annotation for medical imaging research. Existing rule-based labelers struggle with the diverse descriptions in clinical reports, while fine-tuning pre-trained language models (PLMs) requires large amounts of labeled data that are often unavailable in clinical settings. In this paper, we propose PromptRad, a knowledge-enhanced multi-label \textbf{prompt}-tuning approach for \textbf{rad}iology report labeling under low-resource settings. PromptRad reformulates multi-label classification as masked language modeling and incorporates synonyms from the UMLS Metathesaurus into a multi-word verbalizer to enrich category representations. By fine-tuning the PLM without additional classification layers, PromptRad requires substantially less labeled data than conventional fine-tuning. Experiments on liver CT (computed tomography) reports show that PromptRad outperforms dictionary-based and fine-tuning baselines with only 32 labeled training examples, and achieves competitive performance with GPT-4 despite using a much smaller model. Further analysis demonstrates that PromptRad captures complex negation patterns more effectively than existing methods, making it a promising solution for report labeling in data-scarce clinical scenarios. Our code is available at this https URL.
>
---
#### [replaced 033] UCSF-PDGM-VQA: Visual Question Answering dataset for brain tumor MRI interpretation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医学图像理解任务，旨在解决脑肿瘤MRI分析中人工依赖过重的问题。提出UCSF-PDGM-VQA数据集，并评估模型在多序列3D MRI上的表现。**

- **链接: [https://arxiv.org/pdf/2605.17140](https://arxiv.org/pdf/2605.17140)**

> **作者:** Shiv Ghosh; Junayd Lateef; Chih-Hua Liu; Yannan Yu; Andreas M. Rauschecker; Madhumita Sushil
>
> **备注:** 10 pages, 2 figures, 6 tables
>
> **摘要:** Brain tumor diagnosis is largely dependent on Magnetic Resonance Imaging (MRI) evaluation, which requires radiologists to synthesize thousands of images across multiple 3D sequences and longitudinal studies. This process requires advanced neuro-radiology training, poses substantial cognitive load, and is highly time-consuming. Despite increasing demands in radiology, this expertise is difficult to scale, straining the current health systems. Vision-Language Models (VLMs) provide an opportunity to reduce this burden through a semi-automated, interactive interpretation of complex brain MRIs. However, they are currently underutilized in neuro-oncology due to a lack of specialized benchmarks for evaluating them. We introduce a clinically relevant visual question answering (VQA) benchmark -- the UCSF-PDGM-VQA dataset -- consisting of 2,387 QA pairs from 473 glioma-related MRI studies in the public UCSF-PDGM dataset. We further establish a performance baseline for six state-of-the-art vision-language models (VLMs) and one large language model on this dataset. We find that current models are incapable of effectively processing multi-sequence, 3-dimensional MRI scans, thus resulting in a suppression of visual features and over-reliance on language priors, causing modality collapse. These findings underscore a critical deficiency in current model reliability and safety within clinical settings, necessitating the development of robust, domain-specific VLMs.
>
---
#### [replaced 034] Dictionary Insertion Prompting for Multilingual Reasoning on Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，旨在解决多语言大模型性能不足和外部知识融入效果差的问题。提出DIP方法，在非英语提示中插入词典翻译词，提升模型推理效果。**

- **链接: [https://arxiv.org/pdf/2411.01141](https://arxiv.org/pdf/2411.01141)**

> **作者:** Hongyuan Lu; Zixuan Li; Wai Lam
>
> **备注:** ACL *SEM 2026
>
> **摘要:** There are two shortages in the current Large Language Models (LLMs) era. The first is short of multilingual models, where most LLMs are English-centric and performance is limited on multilingual reasoning. The second is the place of external knowledge to be used, where most retrieved knowledge is prepended to the user queries (maybe sub-optimal). This paper presents a novel and simple yet effective method called \textbf{D}ictionary \textbf{I}nsertion \textbf{P}rompting (\textbf{DIP}). When providing a non-English prompt, DIP looks up a word dictionary and inserts words' English counterparts into the middle of the prompt for LLMs. It then enables better translation into English and better English model thinking steps which leads to obviously better results. We experiment with 10 to 200 languages from FLORES-200.\footnote{The number of languages varies on the datasets, and we experiment with 200 languages on GSM8K as in Appendix} Since there are no adequate datasets, we use the NLLB translator to create synthetic multilingual benchmarks from the existing 4 English reasoning benchmarks such as GSM8K and AQuA. The synthetic benchmarks are translated back into English for quality assurance with manual annotation. Interestingly, the place for injecting the dictionary plays an important factor in the performance gains, and we found that interleaving the dictionary with the original words gives a better performance compared to prepending/appending the dictionary, under the same dictionary constructed.
>
---
#### [replaced 035] From Prompt Risk to Response Risk: Paired Analysis of Safety Behavior of Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于大语言模型安全评估任务，旨在分析提示与响应中的风险变化。通过配对分析，研究了不同危害类别中风险的演变，揭示了安全行为的复杂性及潜在问题。**

- **链接: [https://arxiv.org/pdf/2604.26052](https://arxiv.org/pdf/2604.26052)**

> **作者:** Mengya Hu; Qiong Wei; Sandeep Atluri
>
> **摘要:** Safety evaluations of large language models (LLMs) typically report binary outcomes, i.e. attack success rate (ASR), refusal rate, or harmful versus safe classification, which hide how risk changes between prompt and response. We present a paired analysis over human labeled prompt and response records across four harm categories (Sexual, Self harm, Hate and Violence) and ordinal severity levels (Safe, Low, Medium, High). 61% of responses reduce harm relative to the prompt, 36% preserve severity, and 3% escalate. The escalation splits into two mechanisms: benign prompts triggering unrequested harmful detail, and answers that stay on task at higher severity than the prompt. Category decomposition shows that Sexual content exhibits the highest harm persistence in this sample, driven by compliance at the same severity rather than drift from benign inputs. Joint relevance analysis exposes a helpfulness versus harmlessness tradeoff: compliance escalations remain highly relevant, whereas safe responses include generic refusals with low relevance. Finally, few-shot LLM graders exhibit a prompt/response detection asymmetry that data calibration does not close. Grader prompts are shared at this https URL.
>
---
#### [replaced 036] InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling
- **分类: cs.CL**

- **简介: 该论文提出InternBootcamp框架，解决LLM在多样化环境中的推理能力不足问题，通过任务扩展提升模型性能。**

- **链接: [https://arxiv.org/pdf/2508.08636](https://arxiv.org/pdf/2508.08636)**

> **作者:** Peiji Li; Jiasheng Ye; Yongkang Chen; Yichuan Ma; Zijie Yu; Kedi Chen; Xiaozhe Li; Ganqu Cui; Haozhan Li; Jiacheng Chen; Chengqi Lyu; Wenwei Zhang; Linyang Li; Qipeng Guo; Dahua Lin; Bowen Zhou; Kai Chen
>
> **备注:** InternBootcamp Tech Report
>
> **摘要:** Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist.
>
---
#### [replaced 037] Anti-establishment sentiment on TikTok: Implications for understanding influence(rs) and expertise on social media
- **分类: cs.SI; cs.CL; cs.LG**

- **简介: 该论文研究TikTok上反建制情绪的传播，分析其在不同话题中的表现及互动模式，旨在理解社交媒体对机构信任的影响。属于社会媒体影响分析任务。**

- **链接: [https://arxiv.org/pdf/2508.16453](https://arxiv.org/pdf/2508.16453)**

> **作者:** Tianliang Xu; Ariel Hasell; Sabina Tomkins
>
> **备注:** 10 pages excluding references; 14 pages in total; 4 figures; Accepted by the AAAI Conference on Web and Social Media (ICWSM-2026)
>
> **摘要:** Distrust of public serving institutions and anti-establishment views are on the rise (especially in the U.S.). As people turn to social media for information, it is imperative to understand whether and how social media environments may be contributing to distrust of institutions. In social media, content creators, influencers, and other opinion leaders often position themselves as having expertise and authority on a range of topics from health to politics, and in many cases devalue and dismiss institutional expertise to build a following and increase their own visibility. However, the extent to which this content appears and whether such content increases engagement is unclear. This study analyzes the prevalence of anti-establishment sentiment (AES) on the social media platform TikTok. Despite its popularity as a source of information, TikTok remains relatively understudied and may provide important insights into how people form attitudes towards institutions. We employ a computational approach to label TikTok posts as containing AES or not across topical domains where content creators tend to frame themselves as experts: finance and wellness. As a comparison, we also consider the topic of conspiracy theories, where AES is expected to be common. We find that AES is most prevalent in conspiracy theory content, and relatively rare in content related to the other two topics. However, we find that engagement patterns with such content varies by area, and that there may be platform incentives for users to post content that expresses anti-establishment sentiment.
>
---
#### [replaced 038] MeMo: Memory as a Model
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MeMo框架，解决大语言模型难以更新知识的问题。通过专用记忆模型存储新信息，保持原有模型参数不变，提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2605.15156](https://arxiv.org/pdf/2605.15156)**

> **作者:** Ryan Wei Heng Quek; Sanghyuk Lee; Alfred Wei Lun Leong; Arun Verma; Alok Prakash; Nancy F. Chen; Bryan Kian Hsiang Low; Daniela Rus; Armando Solar-Lezama
>
> **备注:** MeMo augments any LLM with up-to-date or domain-specific knowledge via a trained memory model, avoiding costly retraining, mitigating catastrophic forgetting, and remaining robust to retrieval noise
>
> **摘要:** Large language models (LLMs) achieve strong performance across a wide range of tasks, but remain frozen after pretraining until subsequent updates. Many real-world applications require timely, domain-specific information, motivating the need for efficient mechanisms to incorporate new knowledge. In this paper, we introduce MeMo (Memory as a Model), a modular framework that encodes new knowledge into a dedicated memory model while keeping the LLM parameters unchanged. Compared to existing methods, MeMo offers several advantages: (a) it captures complex cross-document relationships, (b) it is robust to retrieval noise, (c) it avoids catastrophic forgetting in the LLM, (d) it does not require access to the LLM's weights or output logits, enabling plug-and-play integration with both open and proprietary closed-source LLMs, and (e) its retrieval cost is independent of corpus size at inference time. Our experimental results on three benchmarks, BrowseComp-Plus, NarrativeQA, and MuSiQue, show that MeMo achieves strong performance compared to existing methods across diverse settings.
>
---
#### [replaced 039] Beyond Words: Multimodal LLM Knows When to Speak
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于对话系统任务，解决聊天机器人何时发言的问题。通过多模态数据和模型，提升对话时机感知能力。**

- **链接: [https://arxiv.org/pdf/2505.14654](https://arxiv.org/pdf/2505.14654)**

> **作者:** Zikai Liao; Yi Ouyang; Yi-Lun Lee; Chen-Ping Yu; Yi-Hsuan Tsai; Zhaozheng Yin
>
> **备注:** Project page: this https URL
>
> **摘要:** Chatbots via large language models (LLMs) generate fluent responses but often struggle with when to speak, especially for brief, timely listener reactions during ongoing dialogue. We present a multimodal strategy for LLMs, which leverages synchronized video, audio, and text cues to improve conversational timing awareness. The strategy reformulates response timing as a dense response-type prediction task, enabling an agent to decide whether to remain silent, produce a short reaction, or start a full response under streaming constraints. Therefore, we introduce a curated multimodal dataset from real-world dyadic conversational videos with temporally aligned modalities and fine-grained reaction type annotations. Moreover, we design a multimodal strategy, MM-When2Speak, with a multimodal integration module on top of an LLM backbone. Experiments across various modality settings and strong LLM baselines show that MM-When2Speak achieves up to a 3x improvement in response type prediction performance, highlighting the importance of multimodal perception for natural and engaging conversational interaction.
>
---
#### [replaced 040] Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究长上下文平衡熵最优传输注意力机制，解决TPU硬件上的梯度计算问题，提出块级可微Sinkhorn注意力方法。**

- **链接: [https://arxiv.org/pdf/2605.08123](https://arxiv.org/pdf/2605.08123)**

> **作者:** Dylan Forde
>
> **摘要:** We study long-context balanced entropic optimal transport (OT) attention on TPU hardware through a stopped-base, fixed-depth tail-refinement surrogate. After a stopped $T$-step Sinkhorn solve, we unroll a short refinement tail and differentiate that surrogate exactly. For the reported $R=2$ TPU path, the backward pass contains four staircase plan factors. We prove an exact one-reference-tile schedule: the $R=2$ score cotangent is a single reference plan tile times an explicit modifier field built from vector cotangents and dual differences. This yields block-wise cost $O((T+R)LW)$, $O(Ld)$ input storage, and $O(L)$ additional HBM usage for fixed head dimension $d$ and band width $W$ on the balanced fixed-support path. We also formalize the current \texttt{dustbin\_block} path as the same unit-target surrogate on an augmented support, so the adjoint schedule lifts to the single-active-dustbin path used in our TPU runs; this bridge is algebraic and does not claim a general KL-unbalanced or arbitrary-capacity gap model. We provide a local surrogate-bias bound, an a posteriori bias certificate, and a projective contraction certificate for strictly positive active blocks. On synthetic masked problems, the optimized kernel matches exact autodiff of the same centered surrogate to within $10^{-5}$--$10^{-10}$. On TPU v6e-8, a four-configuration Pfam screen completes end-to-end, and a promoted balanced $R=2$ run sustains roughly $8.5$ examples per second through a three-hour budget, reaching step $1437$. Held-out Pfam test shards improve reconstruction from $5.57$ to $2.05$ and sparse CE from $5.53$ to $5.30$ relative to step $0$, with CE logged diagnostically rather than optimized directly; target-barycenter alignment metrics do not materially improve, and a deterministic diagonal reference remains stronger on those metrics.
>
---
#### [replaced 041] Large Language Models Unpack Complex Political Opinions through Target-Stance Extraction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决政治意见分析中目标与立场识别的问题。通过构建数据集并测试LLMs，验证其在复杂政治观点提取中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.23531](https://arxiv.org/pdf/2603.23531)**

> **作者:** Özgür Togay; Javier Garcia-Bernardo; Florian Kunneman; Anastasia Giachanou
>
> **摘要:** Political polarization emerges from a complex interplay of beliefs about policies, figures, and issues. However, most computational analyses reduce discourse to coarse partisan labels, overlooking how these beliefs interact. This is especially evident in online political conversations, which are often nuanced and cover a wide range of subjects, making it difficult to automatically identify the target of discussion and the opinion expressed toward them. In this study, we investigate whether Large Language Models (LLMs) can address this challenge through Target-Stance Extraction (TSE), a recent natural language processing task that combines target identification and stance detection, enabling more granular analysis of political opinions. For this, we construct a dataset of 1,084 Reddit posts from r/NeutralPolitics, covering 138 distinct political targets and evaluate a range of proprietary and open-source LLMs using zero-shot, few-shot, and context-augmented prompting strategies. Our results show that the best models perform comparably to highly trained human annotators and remain robust on challenging posts with low inter-annotator agreement. These findings demonstrate that LLMs can extract complex political opinions with minimal supervision, offering a scalable tool for computational social science and political text analysis.
>
---
#### [replaced 042] JoyAI-Image: Awaking Spatial Intelligence in Unified Multimodal Understanding and Generation
- **分类: cs.GR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出JoyAI-Image，一个统一的多模态基础模型，解决视觉理解、文本生成图像和指令编辑问题。通过结合增强空间感知的多模态大语言模型和扩散Transformer，提升几何推理与可控生成能力。**

- **链接: [https://arxiv.org/pdf/2605.04128](https://arxiv.org/pdf/2605.04128)**

> **作者:** Lin Song; Wenbo Li; Guoqing Ma; Wei Tang; Bo Wang; Yuan Zhang; Yijun Yang; Yicheng Xiao; Jianhui Liu; Yanbing Zhang; Guohui Zhang; Wenhu Zhang; Hang Xu; Nan Jiang; Xin Han; Haoze Sun; Maoquan Zhang; Haoyang Huang; Nan Duan
>
> **备注:** Code: this https URL
>
> **摘要:** We present JoyAI-Image, a unified multimodal foundation model for visual understanding, text-to-image generation, and instruction-guided image editing. JoyAI-Image couples a spatially enhanced Multimodal Large Language Model (MLLM) with a Multimodal Diffusion Transformer (MMDiT), allowing perception and generation to interact through a shared multimodal interface. Around this architecture, we build a scalable training recipe that combines unified instruction tuning, long-text rendering supervision, spatially grounded data, and both general and spatial editing signals. This design gives the model broad multimodal capability while strengthening geometry-aware reasoning and controllable visual synthesis. Experiments across understanding, generation, long-text rendering, and editing benchmarks show that JoyAI-Image achieves state-of-the-art or highly competitive performance. More importantly, the bidirectional loop between enhanced understanding, controllable spatial editing, and novel-view-assisted reasoning enables the model to move beyond general visual competence toward stronger spatial intelligence. These results suggest a promising path for unified visual models in downstream applications such as vision-language-action systems and world models.
>
---
#### [replaced 043] You Are What You Say: Exploiting Linguistic Content for VoicePrivacy Attacks
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音隐私攻击任务，研究语言内容对说话人识别的影响。通过BERT模型评估语音匿名化系统的隐私安全性，发现文本相似性影响攻击效果，并提出数据集改进建议。**

- **链接: [https://arxiv.org/pdf/2506.09521](https://arxiv.org/pdf/2506.09521)**

> **作者:** Ünal Ege Gaznepoglu; Anna Leschanowsky; Ahmad Aloradi; Prachi Singh; Daniel Tenbrinck; Emanuël A. P. Habets; Nils Peters
>
> **备注:** 5 pages, 6 figures, 1 table, accepted at INTERSPEECH 2025 update reason: change to the acknowledgements
>
> **摘要:** Speaker anonymization systems hide the identity of speakers while preserving other information such as linguistic content and emotions. To evaluate their privacy benefits, attacks in the form of automatic speaker verification (ASV) systems are employed. In this study, we assess the impact of intra-speaker linguistic content similarity in the attacker training and evaluation datasets, by adapting BERT, a language model, as an ASV system. On the VoicePrivacy Attacker Challenge datasets, our method achieves a mean equal error rate (EER) of 35%, with certain speakers attaining EERs as low as 2%, based solely on the textual content of their utterances. Our explainability study reveals that the system decisions are linked to semantically similar keywords within utterances, stemming from how LibriSpeech is curated. Our study suggests reworking the VoicePrivacy datasets to ensure a fair and unbiased evaluation and challenge the reliance on global EER for privacy evaluations.
>
---
#### [replaced 044] The Generation-Recognition Asymmetry: Six Dimensions of a Fundamental Divide in Formal Language Theory
- **分类: cs.CL; cs.AI; cs.CC; cs.FL**

- **简介: 该论文探讨形式语言理论中生成与识别的不对称性，分析六种维度差异，解决生成与解析操作不对称的问题，提出其在自然语言处理中的意义。**

- **链接: [https://arxiv.org/pdf/2603.10139](https://arxiv.org/pdf/2603.10139)**

> **作者:** Romain Peyrichou
>
> **备注:** Submitted to Information and Computation. 32 pages, 6 figures, 4 tables
>
> **摘要:** Every formal grammar defines a language and can in principle be used in three ways: to generate strings (production), to recognize them (parsing), or -- given only examples -- to infer the grammar itself (grammar induction). Generation and recognition are extensionally equivalent -- they characterize the same set -- but operationally asymmetric in multiple independent ways. Inference is a qualitatively harder problem: it does not have access to a known grammar. Despite the centrality of this triad to compiler design, natural language processing, and formal language theory, no survey has treated it as a unified, multidimensional phenomenon. We identify six dimensions along which generation and recognition diverge: computational complexity, ambiguity, directionality, information availability, grammar inference, and temporality. We show that the common characterization "generation is easy, parsing is hard" is misleading: unconstrained generation is trivial, but generation under constraints can be NP-hard. The real asymmetry is that parsing is always constrained (the input is given) while generation need not be. Two of these dimensions -- directionality and temporality -- have not previously been identified as dimensions of the generation-recognition asymmetry. We connect the temporal dimension to the surprisal framework of Hale (2001) and Levy (2008), arguing that surprisal formalizes the temporal asymmetry between a generator (surprisal = 0) and a parser that predicts under uncertainty (surprisal > 0). We review bidirectional systems in NLP and observe that bidirectionality has been available for fifty years yet has not transferred to most domain-specific applications. We conclude with a discussion of large language models, which architecturally unify generation and recognition while operationally preserving the asymmetry.
>
---
#### [replaced 045] Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Universal Reasoner（UniR），解决冻结LLM增强推理能力的问题。通过模块化设计，无需重新训练即可提升推理性能，支持多任务组合与跨领域泛化。**

- **链接: [https://arxiv.org/pdf/2505.19075](https://arxiv.org/pdf/2505.19075)**

> **作者:** Jaemin Kim; Hangeol Chang; Hyunmin Hwang; Choonghan Kim; Jong Chul Ye
>
> **备注:** ICML 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically require retraining for each LLM backbone due to architectural dependencies. To address these challenges, we propose Universal Reasoner (UniR)-a modular, composable, and plug-and-play reasoning module that can be used with larger frozen LLMs to provide specialized reasoning capabilities with a shared or aligned token space. Specifically, UniR decomposes the reward into a standalone reasoning module trained in a decoupled manner using verifiable rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR is combined with frozen LLMs at inference time by simply adding its output logits to those of the backbone. This additive structure enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Furthermore, UniR demonstrates weak-to-strong generalization, where reasoning modules trained on smaller models effectively guide much larger LLMs in the same model family, and generalize across domains such as in vision language models and medical reasoning. Experiments on mathematical reasoning and machine translation show that UniR surpasses existing fine-tuning methods. Code is open-sourced at this https URL.
>
---
#### [replaced 046] EvalMORAAL: Interpretable Chain-of-Thought and LLM-as-Judge Evaluation for Moral Alignment in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI道德对齐评估任务，旨在检测大模型与全球价值观的匹配度。通过构建框架EvalMORAAL，结合多种评分方法和模型评审，分析20个模型在不同地区的对齐情况。**

- **链接: [https://arxiv.org/pdf/2510.05942](https://arxiv.org/pdf/2510.05942)**

> **作者:** Hadi Mohammadi; Anastasia Giachanou; Robert A. Bagheri
>
> **备注:** Accepted as a poster at *SEM 2026
>
> **摘要:** We present EvalMORAAL, a transparent chain-of-thought (CoT) framework that uses two scoring methods (log-probabilities and direct ratings) plus a model-as-judge peer review to evaluate moral alignment in 20 large language models. We assess models on the World Values Survey (55 countries, 19 topics) and the PEW Global Attitudes Survey (39 countries, 8 topics). With EvalMORAAL, top models align closely with survey responses (Pearson's $r \approx 0.90$ on WVS). Yet we find a clear regional difference: Western regions average $r=0.82$ while non-Western regions average $r=0.61$ (a 0.21 absolute gap), indicating a persistent regional alignment gap. Our framework adds three parts: (1) two scoring methods for all models to enable fair comparison, (2) a structured CoT protocol with self-consistency checks, and (3) a model-as-judge peer review that flags 348 conflicts using a data-driven threshold. Peer agreement relates to WVS survey alignment ($r=0.74$, $p<.001$; PEW $r=0.39$, n.s.), supporting automated quality checks. These results show real progress toward culture-aware AI while highlighting open challenges for use across regions.
>
---
#### [replaced 047] General Preference Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型对齐问题。针对在线RL与偏好优化的分离，提出GPRL方法，通过多维偏好建模提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18721](https://arxiv.org/pdf/2605.18721)**

> **作者:** Muhammad Umer; Muhammad Ahmed Mohsin; Ahsan Bilal; Arslan Chaudhry; Andreas Haupt; Sanmi Koyejo; Emily Fox; John M. Cioffi
>
> **摘要:** Post-training has split large language model (LLM) alignment into two largely disconnected tracks. Online reinforcement learning (RL) with verifiable rewards drives emergent reasoning on math and code but depends on a programmatic verifier that cannot reach open-ended tasks, while preference optimization handles open-ended generation yet forgoes the continuous exploration that powers online RL. Closing this gap requires a verifier for open-ended quality, but a scalar reward model is the wrong shape for the job. Quality is multi-dimensional, and any scalar score is an incomplete proxy that lets online RL collapse onto whichever axis the score is most sensitive to. We turn instead to the General Preference Model (GPM), which embeds responses into $k$ skew-symmetric subspaces and represents preference as a structured, intransitivity-aware comparison. Building on this, we propose General Preference Reinforcement Learning (GPRL), which carries the $k$-way structure through to the policy update. GPRL computes per-dimension group-relative advantages, normalizes each on its own scale so no axis can dominate, and aggregates them with context-dependent eigenvalues. The same structure powers a closed-loop drift monitor that detects single-axis exploitation and corrects it on the fly by reweighting dimensions and tightening the trust region. Starting from $\texttt{Llama-3-8B-Instruct}$, GPRL reaches a length-controlled win rate of $56.51\%$ on AlpacaEval~2.0 while also outperforming SimPO and SPPO on Arena-Hard, MT-Bench, and WildBench by resisting reward hacking across extended training runs.
>
---
#### [replaced 048] Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches
- **分类: cs.SE; cs.CL**

- **简介: 本文属于代码生成任务，旨在解决仓库级代码生成问题。通过综述检索增强生成方法，分析其在仓库级上下文中的应用与挑战。**

- **链接: [https://arxiv.org/pdf/2510.04905](https://arxiv.org/pdf/2510.04905)**

> **作者:** Yicheng Tao; Yuante Li; Yao Qin; Yepang Liu
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved automated code generation. While existing approaches have achieved strong performance at the function and file levels, real-world software engineering requires reasoning over entire repositories, including cross-file dependencies, evolving execution environments, and global semantic consistency. This challenge has led to the emergence of Repository-Level Code Generation (RLCG), where models must retrieve, organize, and utilize repository-scale context to generate coherent and executable code changes. To address these challenges, Retrieval-Augmented Generation (RAG) has become an increasingly important paradigm for repository-level code intelligence. In this survey, we present a comprehensive review of Retrieval-Augmented Code Generation (RACG), with a particular focus on repository-level approaches. Rather than viewing RACG as a static ``retrieve-then-generate'' pipeline, we characterize it as a coupled and evolving process involving context construction, retrieval optimization, generation, and environment interaction. We organize existing methods through a unified analytical framework spanning retrieval substrate, control regime, and evaluation setting. Based on this framework, we systematically examine retrieval strategies, graph-based and non-graph-based retrieval paradigms, training-driven optimizations, and autonomous agent architectures. We further summarize widely used datasets, benchmarks, and system configurations, and discuss key challenges including scalability, reliability, efficiency, and the necessity boundary between RACG and long-context LLMs. Through this survey, we aim to provide a structured understanding of the rapidly evolving RACG landscape and highlight promising directions for future AI-powered software engineering research.
>
---
#### [replaced 049] Flow Map Language Models: One-step Language Modeling via Continuous Denoising
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于连续流的语言模型，解决离散扩散模型在少步生成中质量下降的问题。通过连续流映射实现高效生成，提升速度与质量。**

- **链接: [https://arxiv.org/pdf/2602.16813](https://arxiv.org/pdf/2602.16813)**

> **作者:** Chanhyuk Lee; Jaehoon Yoo; Manan Agarwal; Sheel Shah; Jerry Huang; Aditi Raghunathan; Seunghoon Hong; Nicholas M. Boffi; Jinwoo Kim
>
> **备注:** 58 pages, 40 figures
>
> **摘要:** Language models based on discrete diffusion have attracted widespread interest for their potential to provide faster generation than autoregressive models. Despite their promise, these models typically produce samples whose quality sharply degrades in the few-step regime, preventing a dramatic speedup in practice. Here, we show that language models based on continuous flows over one-hot token embeddings can outperform discrete diffusion in both quality and speed. Importantly, our continuous formulation defines a unique flow map that can be learned directly for efficient few-step inference, a structure we show is unavailable to discrete methods. In this setting, we show that both the flow and its associated flow map can be learned with simple cross-entropy objectives that respect the simplex geometry of the data, and we identify three distinct choices for flow map distillation whose performance we compare in practice. Using these insights, we build a flow language model (FLM), a continuous flow that matches state-of-the-art discrete diffusion baselines on the One Billion Words (LM1B) and OpenWebText (OWT) datasets. We then distill FLM into a flow map language model (FMLM), whose one-step generation exceeds the 8-step quality of recent few-step discrete diffusion language models. Our work challenges the widely-held hypothesis that discrete noising processes are necessary for generative modeling over discrete modalities and paves the way toward accelerated language modeling at scale. Code is available at this https URL.
>
---
#### [replaced 050] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本分类任务中，对比提取式自解释与人类理由的可靠性。分析自解释是否提供合理解释，评估其与人类标注的一致性及忠实度。**

- **链接: [https://arxiv.org/pdf/2410.03296](https://arxiv.org/pdf/2410.03296)**

> **作者:** Stephanie Brandl; Oliver Eberle
>
> **备注:** accepted to the Trustworthy NLP Workshop, co-located with ACL 2026
>
> **摘要:** Instruction-tuned LLMs are able to provide \textit{an} explanation about their output to users by generating self-explanations, without requiring the application of complex interpretability techniques. In this paper, we analyse whether this ability results in a \textit{good} explanation. We evaluate self-explanations in the form of input rationales with respect to their plausibility to humans. We study three text classification tasks: sentiment classification, forced labour detection and claim verification. We include Danish and Italian translations of the sentiment classification task and compare self-explanations to human annotations. For this, we collected human rationale annotations for Climate-Fever, a claim verification dataset. We furthermore evaluate the faithfulness of human and self-explanation rationales with respect to correct model predictions, and extend the study by incorporating post-hoc attribution-based explanations. We analyse four open-weight LLMs and find that alignment between self-explanations and human rationales highly depends on text length and task complexity. Nevertheless, self-explanations yield faithful subsets of token-level rationales, whereas post-hoc attribution methods tend to emphasize structural and formatting tokens, reflecting fundamentally different explanation strategies.
>
---
#### [replaced 051] END: Early Noise Dropping for Efficient and Effective Context Denoising
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决输入噪声干扰模型输出质量的问题。提出END方法，在早期阶段剔除噪声，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2502.18915](https://arxiv.org/pdf/2502.18915)**

> **作者:** Hongye Jin; Pei Chen; Jingfeng Yang; Zhengyang Wang; Fangran Mo; Jinghan Zhang; Meng Jiang; Yifan Gao; Binxuan Huang; Xinyang Zhang; Zheng Li; Tianyi Liu; Huasheng Li; Bing Yin
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, they are often distracted by irrelevant or noisy context in input sequences that degrades output quality. This problem affects both long- and short-context scenarios, such as retrieval-augmented generation, table question-answering, and in-context learning. We reveal that LLMs can implicitly identify whether input sequences contain useful information at early layers, prior to token generation. Leveraging this insight, we introduce Early Noise Dropping (\textsc{END}), a novel approach to mitigate this issue without requiring fine-tuning the LLMs. \textsc{END} segments input sequences into chunks and employs a linear prober on the early layers of LLMs to differentiate between informative and noisy chunks. By discarding noisy chunks early in the process, \textsc{END} preserves critical information, reduces distraction, and lowers computational overhead. Extensive experiments demonstrate that \textsc{END} significantly improves both performance and efficiency across different LLMs on multiple evaluation datasets. Furthermore, by investigating LLMs' implicit understanding to the input with the prober, this work also deepens understanding of how LLMs do reasoning with contexts internally.
>
---
#### [replaced 052] FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频问答任务，旨在解决细粒度人类行为理解问题。提出FineBench基准和FineAgent框架，提升模型对复杂人类动作的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.19846](https://arxiv.org/pdf/2605.19846)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Hung-Ting Su; Winston H. Hsu
>
> **备注:** CVPR'26 (Workshop on Video Large Language Models). Project Page: this https URL
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable capabilities in general video understanding, yet they often struggle with the fine-grained comprehension crucial for real-world applications requiring nuanced interpretation of human actions and interactions. While some recent human-centric benchmarks evaluate aspects of model behaviour such as fairness/ethics, emotion perception, and broader human-centric metrics, they do not combine long-form videos, very dense QA coverage, and frame-level spatial/temporal grounding at scale. To bridge this gap, we introduce FineBench, a human-centric video question answering (VQA) benchmark specifically designed to assess fine-grained understanding. FineBench comprises 199,420 multiple-choice QA pairs densely annotated across 64 long-form videos (15 minutes each), focusing on detailed person movement, person interaction, and object manipulation, including compositional actions. Our extensive evaluation reveals that while proprietary models like GPT-5 achieve respectable performance, current open-source VLMs significantly underperform, struggling particularly with spatial reasoning in multi-person scenes and distinguishing subtle differences in human movements and interactions. To address these identified weaknesses, we propose FineAgent, a modular framework that enhances VLMs by leveraging a Localizer and a Descriptor. Experiments show that FineAgent consistently improves the performance of various open VLMs on FineBench. FineBench provides a rigorous testbed for future research into fine-grained human-centric video understanding, while FineAgent offers a practical approach to enhance such reasoning in current VLMs. Project page and code at this https URL.
>
---
#### [replaced 053] DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于药学问答任务，旨在提升大语言模型在药学领域的准确性。通过构建DrugRAG框架，整合外部药物知识，显著提高了模型表现。**

- **链接: [https://arxiv.org/pdf/2512.14896](https://arxiv.org/pdf/2512.14896)**

> **作者:** Houman Kazemzadeh; Kiarash Mokhtari Dizaji; Seyed Reza Tavakoli; Farbod Davoodi; MohammadReza KarimiNejad; Parham Abed Azad; Fatemeh Latifi; Ali Sabzi; Armin Khosravi; Siavash Ahmadi; Babak Khalaj; Mohammad Hossein Rohban; Glolamali Aminian; Zohreh Amoozgar; Tahereh Javaheri
>
> **备注:** 14 pages, 2 figures, 2 tables. The revised version includes McNemar's paired statistical analysis, Wilson confidence intervals, expanded methodological clarifications, a revised discussion of evidence retrieval, improved reproducibility details, and updated limitations
>
> **摘要:** In our study, we evaluated large language model (LLM) performance on pharmacy licensure-style question-answering tasks and developed an external knowledge integration method to improve accuracy. We benchmarked ten LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset, measuring baseline accuracy without modification. Baseline performance ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores, while smaller open-source models showed substantially lower performance. We then developed DrugRAG, a three-step retrieval-augmented generation (RAG) pipeline that retrieves structured, evidence-based drug information and augments model prompts with contextual pharmacological evidence, operating externally and requiring no changes to model architecture or parameters. DrugRAG increased accuracy across all five evaluated models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61.0% to 71%, Llama 3.1 8B: 46% to 67%). McNemar analyses demonstrated statistically significant paired improvements primarily in smaller and mid-sized open-source models. These findings demonstrate that integrating structured external drug knowledge via DrugRAG can improve LLM performance on pharmacy-focused question-answering tasks without modifying the underlying models, providing a practical pipeline for enhancing evidence-based pharmacy-focused AI applications.
>
---
#### [replaced 054] AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理与社会科学研究的交叉任务，旨在解决传统调查问卷无法全面捕捉公众意见变化的问题。通过构建基于大语言模型的框架，预测缺失的调查回答，提升调查数据的时效性和完整性。**

- **链接: [https://arxiv.org/pdf/2305.09620](https://arxiv.org/pdf/2305.09620)**

> **作者:** Junsol Kim; Byungkyu Lee
>
> **摘要:** Nationally representative surveys track public opinion, yet they ask only a limited set of questions each year, limiting its potential to capture historical changes. To fill this gap, we develop a large language model (LLM)-based framework for predicting missing responses in repeated cross-sectional surveys by incorporating embeddings for questions, respondents, and survey periods. We introduce two new applications of LLMs to survey research: retrodiction (predicting year-level missing opinions) and unasked opinion prediction (predicting entirely missing opinions). Using data from the 1972-2021 General Social Surveys, our LLM-based models perform strongly in retrodicting masked GSS opinions through cross-validation and public opinions measured by other organizations in years when the GSS did not ask them. These capabilities enable us to recover missing trends and pinpoint when public attitudes changed, such as the rising support for same-sex marriage. However, performance remains modest for unasked opinion prediction. We show when our models outperform established benchmarks, examine which opinions and and respondents are more predictable, and evaluate whether our approach reduces LLMs' tendency to homogenize predicted responses. Our study demonstrates that LLMs and surveys can mutually enhance each other: LLMs broaden survey potential, while surveys calibrate LLMs for simulating human opinions.
>
---
#### [replaced 055] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FLAIR方法，用于全双工对话系统中的隐式推理，解决语音交互中同时进行思考与响应的问题。通过连续推理提升对话质量。**

- **链接: [https://arxiv.org/pdf/2603.17837](https://arxiv.org/pdf/2603.17837)**

> **作者:** Donghang Wu; Tianyu Zhang; Yuxin Li; Hexin Liu; Chen Chen; Eng Siong Chng; Yoshua Bengio
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
>
---
#### [replaced 056] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BudgetMem框架，解决LLM代理在运行时内存管理的效率与成本平衡问题。通过预算层级路由实现性能与成本的显式控制。**

- **链接: [https://arxiv.org/pdf/2602.06025](https://arxiv.org/pdf/2602.06025)**

> **作者:** Haozhen Zhang; Haodong Yue; Tao Feng; Quanyu Long; Jianzhu Bao; Bowen Jin; Weizhi Zhang; Xiao Li; Jiaxuan You; Chengwei Qin; Wenya Wang
>
> **备注:** Accepted by ICML 2026. Code is available at this https URL
>
> **摘要:** Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes.
>
---
#### [replaced 057] EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments
- **分类: cs.CL**

- **简介: 该论文针对长对话问答任务，解决资源受限环境下KV缓存过大的问题。提出EpiCache框架，通过分块预填和情景压缩，有效管理缓存，提升准确率并降低内存占用。**

- **链接: [https://arxiv.org/pdf/2509.17396](https://arxiv.org/pdf/2509.17396)**

> **作者:** Minsoo Kim; Arnav Kundu; Han-Byul Kim; Richa Dixit; Minsik Cho
>
> **备注:** ICML 2026
>
> **摘要:** Modern large language models (LLMs) extend context lengths to millions of tokens, enabling coherent, personalized responses grounded in long conversational history. However, the Key-Value (KV) cache grows linearly with the extended dialogue history, causing the model's memory footprint to quickly exceed device limits. While recent KV cache compression methods attempt to reduce memory usage, most apply cache eviction after processing the entire context, incurring unbounded peak memory usage. Additionally, query-dependent eviction narrows the cache semantics to a single query, leading to failure cases in multi-turn conversations. In this paper, we introduce EpiCache, a training-free KV cache management framework for long conversational question answering (LongConvQA) under fixed memory budgets. EpiCache bounds cache growth through block-wise prefill and preserves topic-relevant context via episodic KV compression, which clusters conversation history into coherent episodes and performs episode-specific KV cache eviction. Across three LongConvQA benchmarks (LongMemEval, Realtalk, and LoCoMo), EpiCache improves accuracy by up to 30%, achieves near full-cache accuracy under 4-6x compression, and reduces latency and peak memory by up to 2.4x and 3.7x, respectively.
>
---
#### [replaced 058] Automatically Learning Construction Injury Precursors from Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在从施工事故报告中自动学习伤害前兆。通过比较不同模型，识别具有预测性的文本模式，以提升安全理解与预防能力。**

- **链接: [https://arxiv.org/pdf/1907.11769](https://arxiv.org/pdf/1907.11769)**

> **作者:** Henrietta Baker; Matthew R. Hallowell; Antoine J.-P. Tixier
>
> **备注:** Added author contributions and journal reference, updated corresponding author
>
> **摘要:** In light of the increasing availability of digitally recorded safety reports in the construction industry, it is important to develop methods to exploit these data to improve our understanding of safety incidents and ability to learn from them. In this study, we compare several approaches to automatically learn injury precursors from raw construction accident reports. More precisely, we experiment with two state-of-the-art deep learning architectures for Natural Language Processing (NLP), Convolutional Neural Networks (CNN) and Hierarchical Attention Networks (HAN), and with the established Term Frequency - Inverse Document Frequency representation (TF-IDF) + Support Vector Machine (SVM) approach. For each model, we provide a method to identify (after training) the textual patterns that are, on average, the most predictive of each safety outcome. We show that among those pieces of text, valid injury precursors can be found. The proposed methods can also be used by the user to visualize and understand the models' predictions.
>
---
#### [replaced 059] Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?
- **分类: cs.CL**

- **简介: 该论文属于人工智能与社会认知交叉研究，探讨LLM在权力不对称对话中是否表现出类似人类的社会认知效应，通过模拟不同身份对话分析语言协调、称谓使用等行为。**

- **链接: [https://arxiv.org/pdf/2605.17694](https://arxiv.org/pdf/2605.17694)**

> **作者:** Anvesh Rao Vijjini; Sagar Manjunath; Snigdha Chaturvedi
>
> **备注:** ACL 2026 (main)
>
> **摘要:** Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) language coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio-cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.
>
---
#### [replaced 060] Towards the Anonymization of the Language Modeling
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于隐私保护任务，旨在解决语言模型泄露敏感信息的问题。通过提出MLM和CLM方法，提升模型的匿名性与实用性。**

- **链接: [https://arxiv.org/pdf/2501.02407](https://arxiv.org/pdf/2501.02407)**

> **作者:** Antoine Boutet; Lucas Magnana; Juliette Sénéchal
>
> **摘要:** Rapid advances in Natural Language Processing (NLP) have revolutionized many fields, including healthcare. However, these advances raise significant privacy concerns, especially when pre-trained models fine-tuned and specialized on sensitive data can memorize and then expose and regurgitate personal information. This paper presents a privacy-preserving language modeling approach to address the problem of language models anonymization, and thus promote their sharing. Specifically, we propose both a Masking Language Modeling (MLM) methodology to specialize a BERT-like language model, and a Causal Language Modeling (CLM) methodology to specialize a GPT-like model that avoids the model from memorizing direct and indirect identifying information present in the training data. We have comprehensively evaluated our approaches using a medical dataset and compared them against different baselines. Our results indicate that by avoiding memorizing both direct and indirect identifiers during model specialization, our masking and causal language modeling schemes offer a good tradeoff for maintaining high privacy while retaining high utility.
>
---
#### [replaced 061] Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在分析代理记忆系统的架构与局限性，解决评估不准确和系统性能问题，通过分类与实证分析提出改进方向。**

- **链接: [https://arxiv.org/pdf/2602.19320](https://arxiv.org/pdf/2602.19320)**

> **作者:** Dongming Jiang; Yi Li; Songtao Wei; Jinxin Yang; Ayushi Kishore; Alysa Zhao; Dingyi Kang; Xu Hu; Feng Chen; Qiannan Li; Bingzhe Li
>
> **摘要:** Agentic memory systems enable large language model (LLM) agents to maintain state across long interactions, supporting long-horizon reasoning and personalization beyond fixed context windows. Despite rapid architectural development, the empirical foundations of these systems remain fragile: existing benchmarks are often underscaled, evaluation metrics are misaligned with semantic utility, performance varies significantly across backbone models, and system-level costs are frequently overlooked. This survey presents a structured analysis of agentic memory from both architectural and system perspectives. We first introduce a concise taxonomy of MAG systems based on four memory structures. Then, we analyze key pain points limiting current systems, including benchmark saturation effects, metric validity and judge sensitivity, backbone-dependent accuracy, and the latency and throughput overhead introduced by memory maintenance. By connecting the memory structure to empirical limitations, this survey clarifies why current agentic memory systems often underperform their theoretical promise and outlines directions for more reliable evaluation and scalable system design.
>
---
#### [replaced 062] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design
- **分类: q-bio.QM; cs.CL**

- **简介: 该论文提出AFD-Instruction数据集，解决抗体理解与设计问题，通过功能标注提升LLM在抗体相关任务的表现。**

- **链接: [https://arxiv.org/pdf/2602.04916](https://arxiv.org/pdf/2602.04916)**

> **作者:** Ling Luo; Wenbin Jiang; Hongyuan Chang; Xinkang Wang; Xushi Zhang; Yueting Xiong; Mengsha Tong; Rongshan Yu
>
> **摘要:** Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
>
---
#### [replaced 063] Enhancing Speech Large Language Models through Reinforced Behavior Alignment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，解决SpeechLM在指令遵循上的性能不足问题。通过引入RBA框架，利用强化学习提升模型能力，取得优异效果。**

- **链接: [https://arxiv.org/pdf/2509.03526](https://arxiv.org/pdf/2509.03526)**

> **作者:** Yansong Liu; Jiateng Li; Yuan Liu
>
> **摘要:** The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data.
>
---
#### [replaced 064] Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音转录与说话人辨识任务，旨在降低法语临床对话的识别错误率。通过多轮LLM后处理提升准确性，实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00086](https://arxiv.org/pdf/2603.00086)**

> **作者:** Ambre Marie; Thomas Bertin; Guillaume Dardenne; Gwenolé Quellec
>
> **摘要:** Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p<0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment, pending validation on larger corpora.
>
---
#### [replaced 065] ZeroUnlearn: Few-Shot Knowledge Unlearning in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识去遗忘任务，旨在解决大语言模型中敏感信息留存问题。通过模型编辑实现精准知识重映射，提出ZeroUnlearn框架，高效去除敏感内容并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18879](https://arxiv.org/pdf/2605.18879)**

> **作者:** Yujie Lin; Chengyi Yang; Zhishang Xiang; Yiping Song; Jinsong Su
>
> **摘要:** Large language models inevitably retain sensitive information, defined as inputs that may induce harmful generations, due to training on massive web corpora, raising concerns for privacy and safety. Existing machine unlearning methods primarily rely on retraining or aggressive fine-tuning, which are either computationally expensive or prone to degrading related knowledge and overall model utility. In this work, we reformulate machine unlearning as a precise knowledge re-mapping problem via model editing. We propose ZeroUnlearn, a few-shot unlearning framework. It overwrites sensitive inputs by mapping them to a neutral target state and removing their original representations. ZeroUnlearn enforces representational orthogonality through a multiplicative parameter update with a closed-form solution, enabling efficient and targeted unlearning. We further extend ZeroUnlearn to a gradient-based variant for multi-sample unlearning. Experiments demonstrate that our approach outperforms existing baselines while preserving general model utility. Our code is available at the github: this https URL.
>
---
#### [replaced 066] How Open Must Language Models be to Enable Reliable Scientific Inference?
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨开放性对语言模型科学推理的影响，属于AI伦理与可靠性研究任务。解决封闭模型阻碍科学推断的问题，分析其影响并提出改进措施。**

- **链接: [https://arxiv.org/pdf/2603.26539](https://arxiv.org/pdf/2603.26539)**

> **作者:** James A. Michaelov; Catherine Arnett; Tyler A. Chang; Pamela D. Rivière; Samuel M. Taylor; Cameron R. Jones; Sean Trott; Roger P. Levy; Benjamin K. Bergen; Micah Altman
>
> **摘要:** How does the extent to which a model is open or closed impact the scientific inferences that can be drawn from research that involves it? In this paper, we analyze how restrictions on information about model construction and deployment threaten reliable inference. We argue that current closed models are generally ill-suited for scientific purposes, with some notable exceptions, and discuss ways in which the issues they present to reliable inference can be resolved or mitigated. We recommend that when models are used in research, potential threats to inference should be systematically identified along with the steps taken to mitigate them, and that specific justifications for model selection should be provided.
>
---

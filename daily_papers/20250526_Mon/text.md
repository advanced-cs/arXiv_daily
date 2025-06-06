# 自然语言处理 cs.CL

- **最新发布 244 篇**

- **更新 117 篇**

## 最新发布

#### [new 001] Large Language Models Implicitly Learn to See and Hear Just By Reading
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于多模态理解任务，探索文本LLM是否能通过预训练直接处理图像/音频。研究发现仅用文本训练的自回归LLM可内在学习跨模态理解能力，输入图像块或音频波形即可输出分类结果，在FSD-50K、CIFAR-10等数据集验证，证明文本模型内部形成通用感知模块，无需针对新任务从头训练。**

- **链接: [http://arxiv.org/pdf/2505.17091v1](http://arxiv.org/pdf/2505.17091v1)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 4 tables. Under Review WASPAA 2025
>
> **摘要:** This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time.
>
---
#### [new 002] CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents
- **分类: cs.CL**

- **简介: 该论文提出CReSt基准，评估检索增强生成（RAG）系统在复杂推理、结构化文档理解、精准引用及可靠拒答等任务中的综合能力，解决现有评测无法全面覆盖实际场景的问题。通过2245个人类标注的英韩双语案例，揭示先进大模型在这些维度上的不足，推动鲁棒RAG系统研发。**

- **链接: [http://arxiv.org/pdf/2505.17503v1](http://arxiv.org/pdf/2505.17503v1)**

> **作者:** Minsoo Khang; Sangjun Park; Teakgyu Hong; Dawoon Jung
>
> **摘要:** Large Language Models (LLMs) have made substantial progress in recent years, yet evaluating their capabilities in practical Retrieval-Augmented Generation (RAG) scenarios remains challenging. In practical applications, LLMs must demonstrate complex reasoning, refuse to answer appropriately, provide precise citations, and effectively understand document layout. These capabilities are crucial for advanced task handling, uncertainty awareness, maintaining reliability, and structural understanding. While some of the prior works address these aspects individually, there is a need for a unified framework that evaluates them collectively in practical RAG scenarios. To address this, we present CReSt (A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents), a benchmark designed to assess these key dimensions holistically. CReSt comprises 2,245 human-annotated examples in English and Korean, designed to capture practical RAG scenarios that require complex reasoning over structured documents. It also introduces a tailored evaluation methodology to comprehensively assess model performance in these critical areas. Our evaluation shows that even advanced LLMs struggle to perform consistently across these dimensions, underscoring key areas for improvement. We release CReSt to support further research and the development of more robust RAG systems. The dataset and code are available at: https://github.com/UpstageAI/CReSt.
>
---
#### [new 003] Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长链推理（Long-CoT）指令调优数据选择任务。针对大规模数据集训练成本高且缺乏有效选择策略的问题，提出Select2Reason框架，通过难度量化和推理轨迹长度加权筛选高价值样本。实验表明，仅用10%数据可达到甚至超越全量数据微调效果。**

- **链接: [http://arxiv.org/pdf/2505.17266v1](http://arxiv.org/pdf/2505.17266v1)**

> **作者:** Cehao Yang; Xueyuan Lin; Chengjin Xu; Xuhui Jiang; Xiaojun Wu; Honghao Liu; Hui Xiong; Jian Guo
>
> **摘要:** A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost.
>
---
#### [new 004] CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对放射报告生成评估任务，解决现有指标无法准确衡量临床正确性及受类别不平衡影响的问题。提出CRG Score，通过关注参考报告中明确的临床异常、结合分布平衡罚则及结构化标签，实现更公平、鲁棒的临床对齐评估。**

- **链接: [http://arxiv.org/pdf/2505.17167v1](http://arxiv.org/pdf/2505.17167v1)**

> **作者:** Ibrahim Ethem Hamamci; Sezgin Er; Suprosanna Shit; Hadrien Reynaud; Bernhard Kainz; Bjoern Menze
>
> **摘要:** Evaluating long-context radiology report generation is challenging. NLG metrics fail to capture clinical correctness, while LLM-based metrics often lack generalizability. Clinical accuracy metrics are more relevant but are sensitive to class imbalance, frequently favoring trivial predictions. We propose the CRG Score, a distribution-aware and adaptable metric that evaluates only clinically relevant abnormalities explicitly described in reference reports. CRG supports both binary and structured labels (e.g., type, location) and can be paired with any LLM for feature extraction. By balancing penalties based on label distribution, it enables fairer, more robust evaluation and serves as a clinically aligned reward function.
>
---
#### [new 005] Low-Resource NMT: A Case Study on the Written and Spoken Languages in Hong Kong
- **分类: cs.CL**

- **简介: 该论文研究低资源神经机器翻译任务，解决中文与粤语书面语互译数据稀缺问题。通过收集28K现有平行句及从维基百科提取72K相似句对构建数据集，提出基于Transformer的翻译系统，在6/8测试集上超越百度翻译，有效捕捉语言转换。**

- **链接: [http://arxiv.org/pdf/2505.17816v1](http://arxiv.org/pdf/2505.17816v1)**

> **作者:** Hei Yi Mak; Tan Lee
>
> **备注:** Proceedings of the 2021 5th International Conference on Natural Language Processing and Information Retrieval
>
> **摘要:** The majority of inhabitants in Hong Kong are able to read and write in standard Chinese but use Cantonese as the primary spoken language in daily life. Spoken Cantonese can be transcribed into Chinese characters, which constitute the so-called written Cantonese. Written Cantonese exhibits significant lexical and grammatical differences from standard written Chinese. The rise of written Cantonese is increasingly evident in the cyber world. The growing interaction between Mandarin speakers and Cantonese speakers is leading to a clear demand for automatic translation between Chinese and Cantonese. This paper describes a transformer-based neural machine translation (NMT) system for written-Chinese-to-written-Cantonese translation. Given that parallel text data of Chinese and Cantonese are extremely scarce, a major focus of this study is on the effort of preparing good amount of training data for NMT. In addition to collecting 28K parallel sentences from previous linguistic studies and scattered internet resources, we devise an effective approach to obtaining 72K parallel sentences by automatically extracting pairs of semantically similar sentences from parallel articles on Chinese Wikipedia and Cantonese Wikipedia. We show that leveraging highly similar sentence pairs mined from Wikipedia improves translation performance in all test sets. Our system outperforms Baidu Fanyi's Chinese-to-Cantonese translation on 6 out of 8 test sets in BLEU scores. Translation examples reveal that our system is able to capture important linguistic transformations between standard Chinese and spoken Cantonese.
>
---
#### [new 006] NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NeSyGeo框架，通过神经符号方法生成多模态几何推理数据，解决现有方法在数据多样性和数值泛化上的不足。其构建了基于符号语言的生成系统，合成几何符号序列并映射为图文数据，生成含推理路径的QA对，提升多模态模型几何推理能力。**

- **链接: [http://arxiv.org/pdf/2505.17121v1](http://arxiv.org/pdf/2505.17121v1)**

> **作者:** Weiming Wu; Zi-kang Wang; Jin Ye; Zhi Zhou; Yu-Feng Li; Lan-Zhe Guo
>
> **摘要:** Obtaining large-scale, high-quality data with reasoning paths is crucial for improving the geometric reasoning capabilities of multi-modal large language models (MLLMs). However, existing data generation methods, whether based on predefined templates or constrained symbolic provers, inevitably face diversity and numerical generalization limitations. To address these limitations, we propose NeSyGeo, a novel neuro-symbolic framework for generating geometric reasoning data. First, we propose a domain-specific language grounded in the entity-relation-constraint paradigm to comprehensively represent all components of plane geometry, along with generative actions defined within this symbolic space. We then design a symbolic-visual-text pipeline that synthesizes symbolic sequences, maps them to corresponding visual and textual representations, and generates diverse question-answer (Q&A) pairs using large language models (LLMs). To the best of our knowledge, we are the first to propose a neuro-symbolic approach in generating multimodal reasoning data. Based on this framework, we construct NeSyGeo-CoT and NeSyGeo-Caption datasets, containing 100k samples, and release a new benchmark NeSyGeo-Test for evaluating geometric reasoning abilities in MLLMs. Experiments demonstrate that the proposal significantly and consistently improves the performance of multiple MLLMs under both reinforcement and supervised fine-tuning. With only 4k samples and two epochs of reinforcement fine-tuning, base models achieve improvements of up to +15.8% on MathVision, +8.4% on MathVerse, and +7.3% on GeoQA. Notably, a 4B model can be improved to outperform an 8B model from the same series on geometric reasoning tasks.
>
---
#### [new 007] P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于学术论文转海报自动生成任务，解决现有方法在保留科学细节、视觉文本整合及缺乏标准化评估的问题。提出P2P框架，通过三代理（视觉处理、内容生成、组装）及检查模块生成高质量HTML海报，并构建含3万示例的数据集P2PInstruct和评估基准P2PEval，推动领域发展。**

- **链接: [http://arxiv.org/pdf/2505.17104v1](http://arxiv.org/pdf/2505.17104v1)**

> **作者:** Tao Sun; Enhao Pan; Zhengkai Yang; Kaixin Sui; Jiajun Shi; Xianfu Cheng; Tongliang Li; Wenhao Huang; Ge Zhang; Jian Yang; Zhoujun Li
>
> **摘要:** Academic posters are vital for scholarly communication, yet their manual creation is time-consuming. However, automated academic poster generation faces significant challenges in preserving intricate scientific details and achieving effective visual-textual integration. Existing approaches often struggle with semantic richness and structural nuances, and lack standardized benchmarks for evaluating generated academic posters comprehensively. To address these limitations, we introduce P2P, the first flexible, LLM-based multi-agent framework that generates high-quality, HTML-rendered academic posters directly from research papers, demonstrating strong potential for practical applications. P2P employs three specialized agents-for visual element processing, content generation, and final poster assembly-each integrated with dedicated checker modules to enable iterative refinement and ensure output quality. To foster advancements and rigorous evaluation in this domain, we construct and release P2PInstruct, the first large-scale instruction dataset comprising over 30,000 high-quality examples tailored for the academic paper-to-poster generation task. Furthermore, we establish P2PEval, a comprehensive benchmark featuring 121 paper-poster pairs and a dual evaluation methodology (Universal and Fine-Grained) that leverages LLM-as-a-Judge and detailed, human-annotated checklists. Our contributions aim to streamline research dissemination and provide the community with robust tools for developing and evaluating next-generation poster generation systems.
>
---
#### [new 008] Emerging categories in scientific explanations
- **分类: cs.CL**

- **简介: 该论文属于科学解释分类任务，旨在解决机器学习解释数据集缺乏大规模人类生成解释的问题。工作包括从生物领域文献提取解释性句子，构建6类/3类标注数据集（3类Krippendorf Alpha为0.667），并评估分类一致性。**

- **链接: [http://arxiv.org/pdf/2505.17832v1](http://arxiv.org/pdf/2505.17832v1)**

> **作者:** Giacomo Magnifico; Eduard Barbu
>
> **备注:** Accepted at the 3rd TRR 318 Conference: Contextualizing Explanations (ContEx25), as a two-pager abstract. Will be published at BiUP (Bielefeld University Press) at a later date
>
> **摘要:** Clear and effective explanations are essential for human understanding and knowledge dissemination. The scope of scientific research aiming to understand the essence of explanations has recently expanded from the social sciences to machine learning and artificial intelligence. Explanations for machine learning decisions must be impactful and human-like, and there is a lack of large-scale datasets focusing on human-like and human-generated explanations. This work aims to provide such a dataset by: extracting sentences that indicate explanations from scientific literature among various sources in the biotechnology and biophysics topic domains (e.g. PubMed's PMC Open Access subset); providing a multi-class notation derived inductively from the data; evaluating annotator consensus on the emerging categories. The sentences are organized in an openly-available dataset, with two different classifications (6-class and 3-class category annotation), and the 3-class notation achieves a 0.667 Krippendorf Alpha value.
>
---
#### [new 009] Cultural Value Alignment in Large Language Models: A Prompt-based Analysis of Schwartz Values in Gemini, ChatGPT, and DeepSeek
- **分类: cs.CL**

- **简介: 该研究分析大型语言模型（LLM）的文化价值观对齐，探讨其是否反映文化偏见而非普适伦理。通过对比Gemini、ChatGPT和中文模型DeepSeek在Schwartz价值观框架中的表现，发现DeepSeek更重视集体主义的自我超越价值，而弱化个人主义的自我增强价值。研究提出多角度推理等方法，推动AI公平与多元道德框架构建。**

- **链接: [http://arxiv.org/pdf/2505.17112v1](http://arxiv.org/pdf/2505.17112v1)**

> **作者:** Robin Segerer
>
> **备注:** 15 pages, 1 table, 1 figure
>
> **摘要:** This study examines cultural value alignment in large language models (LLMs) by analyzing how Gemini, ChatGPT, and DeepSeek prioritize values from Schwartz's value framework. Using the 40-item Portrait Values Questionnaire, we assessed whether DeepSeek, trained on Chinese-language data, exhibits distinct value preferences compared to Western models. Results of a Bayesian ordinal regression model show that self-transcendence values (e.g., benevolence, universalism) were highly prioritized across all models, reflecting a general LLM tendency to emphasize prosocial values. However, DeepSeek uniquely downplayed self-enhancement values (e.g., power, achievement) compared to ChatGPT and Gemini, aligning with collectivist cultural tendencies. These findings suggest that LLMs reflect culturally situated biases rather than a universal ethical framework. To address value asymmetries in LLMs, we propose multi-perspective reasoning, self-reflective feedback, and dynamic contextualization. This study contributes to discussions on AI fairness, cultural neutrality, and the need for pluralistic AI alignment frameworks that integrate diverse moral perspectives.
>
---
#### [new 010] Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文评估大语言模型知识注入的可迁移性，解决模型知识更新效果差异问题。通过对比问答、翻译等任务发现，理解型任务知识保留率显著更高（48% vs 17-20%），但跨场景应用效果下降，提示需重视任务选择与深度认知参与。**

- **链接: [http://arxiv.org/pdf/2505.17140v1](http://arxiv.org/pdf/2505.17140v1)**

> **作者:** Essa Jan; Moiz Ali; Muhammad Saram Hassan; Fareed Zaffar; Yasir Zaki
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** As the knowledge of large language models (LLMs) becomes outdated over time, there is a growing need for efficient methods to update them, especially when injecting proprietary information. Our study reveals that comprehension-intensive fine-tuning tasks (e.g., question answering and blanks) achieve substantially higher knowledge retention rates (48%) compared to mapping-oriented tasks like translation (17%) or text-to-JSON conversion (20%), despite exposure to identical factual content. We demonstrate that this pattern persists across model architectures and follows scaling laws, with larger models showing improved retention across all task types. However, all models exhibit significant performance drops when applying injected knowledge in broader contexts, suggesting limited semantic integration. These findings show the importance of task selection in updating LLM knowledge, showing that effective knowledge injection relies not just on data exposure but on the depth of cognitive engagement during fine-tuning.
>
---
#### [new 011] Improving endpoint detection in end-to-end streaming ASR for conversational speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦端点检测（EP）在端到端流式ASR中的优化，旨在解决转导模型（T-ASR）因延迟输出导致的EP错误（如截断结果或延迟响应）。提出在词尾添加结束标记并引入延迟惩罚，结合辅助网络的帧级语音活动检测改进EP，实验基于Switchboard语料库验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.17070v1](http://arxiv.org/pdf/2505.17070v1)**

> **作者:** Anandh C; Karthik Pandia Durai; Jeena Prakash; Manickavela Arumugam; Kadri Hacioglu; S. Pavankumar Dubagunta; Andreas Stolcke; Shankar Venkatesan; Aravind Ganapathiraju
>
> **备注:** Submitted to Interspeech 2024
>
> **摘要:** ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method.
>
---
#### [new 012] Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究LLM在招聘中的性别与位置偏见。通过让22个模型对比男女名字的简历（资质相同），发现均偏好女性，添加性别字段或代词强化此倾向；首位置候选人更易被选。实验揭示LLM决策存在系统性偏差，警示其在高风险场景的应用。**

- **链接: [http://arxiv.org/pdf/2505.17049v1](http://arxiv.org/pdf/2505.17049v1)**

> **作者:** David Rozado
>
> **摘要:** This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning.
>
---
#### [new 013] Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决高压缩率导致的几何失真对模型能力评估的误导问题。研究发现"压缩破解"现象：高压缩使词向量空间出现各向异性，牺牲均匀性以虚假提升压缩率。团队提出结合几何失真分析的三指标自评估方法，与模型性能达0.9以上相关性，优于原有指标。**

- **链接: [http://arxiv.org/pdf/2505.17793v1](http://arxiv.org/pdf/2505.17793v1)**

> **作者:** Jianxiang Zang; Meiling Ning; Yongda Wei; Shihan Dou; Jiazheng Zhang; Nijia Mo; Binhong Li; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recently, the concept of ``compression as intelligence'' has provided a novel informatics metric perspective for language models (LMs), emphasizing that highly structured representations signify the intelligence level of LMs. However, from a geometric standpoint, the word representation space of highly compressed LMs tends to degenerate into a highly anisotropic state, which hinders the LM's ability to comprehend instructions and directly impacts its performance. We found this compression-anisotropy synchronicity is essentially the ``Compression Hacking'' in LM representations, where noise-dominated directions tend to create the illusion of high compression rates by sacrificing spatial uniformity. Based on this, we propose three refined compression metrics by incorporating geometric distortion analysis and integrate them into a self-evaluation pipeline. The refined metrics exhibit strong alignment with the LM's comprehensive capabilities, achieving Spearman correlation coefficients above 0.9, significantly outperforming both the original compression and other internal structure-based metrics. This confirms that compression hacking substantially enhances the informatics interpretation of LMs by incorporating geometric distortion of representations.
>
---
#### [new 014] Measuring diversity of synthetic prompts and data generated with fine-grained persona prompting
- **分类: cs.CL**

- **简介: 该论文属于评估合成数据多样性的研究任务，旨在解决细粒度人物角色提示是否提升生成数据多样性的疑问。通过对比实验，发现合成提示比人类编写数据多样性更低，细粒度人物描述虽能通过大模型略微提升多样性，但其细节并未显著增强效果。**

- **链接: [http://arxiv.org/pdf/2505.17390v1](http://arxiv.org/pdf/2505.17390v1)**

> **作者:** Gauri Kambhatla; Chantal Shaib; Venkata Govindarajan
>
> **摘要:** Fine-grained personas have recently been used for generating 'diverse' synthetic data for pre-training and supervised fine-tuning of Large Language Models (LLMs). In this work, we measure the diversity of persona-driven synthetically generated prompts and responses with a suite of lexical diversity and redundancy metrics. Firstly, we find that synthetic prompts/instructions are significantly less diverse than human-written ones. Next, we sample responses from LLMs of different sizes with fine-grained and coarse persona descriptions to investigate how much fine-grained detail in persona descriptions contribute to generated text diversity. We find that while persona-prompting does improve lexical diversity (especially with larger models), fine-grained detail in personas doesn't increase diversity noticeably.
>
---
#### [new 015] LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决现有基于结果的强化学习方法忽视中间推理步骤正确性的问题。提出LeTS框架，通过融合过程奖励（优化中间推理步骤）与结果奖励（优化最终输出），提升大模型的推理能力，实验验证其有效性和普适性。**

- **链接: [http://arxiv.org/pdf/2505.17447v1](http://arxiv.org/pdf/2505.17447v1)**

> **作者:** Qi Zhang; Shouqing Yang; Lirong Gao; Hao Chen; Xiaomeng Hu; Jinglei Chen; Jiexiang Wang; Sheng Guo; Bo Zheng; Haobo Wang; Junbo Zhao
>
> **备注:** preprint, under review
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in reasoning with the emergence of reasoning models like OpenAI-o1 and DeepSeek-R1. Recent research focuses on integrating reasoning capabilities into the realm of retrieval-augmented generation (RAG) via outcome-supervised reinforcement learning (RL) approaches, while the correctness of intermediate think-and-search steps is usually neglected. To address this issue, we design a process-level reward module to mitigate the unawareness of intermediate reasoning steps in outcome-level supervision without additional annotation. Grounded on this, we propose Learning to Think-and-Search (LeTS), a novel framework that hybridizes stepwise process reward and outcome-based reward to current RL methods for RAG. Extensive experiments demonstrate the generalization and inference efficiency of LeTS across various RAG benchmarks. In addition, these results reveal the potential of process- and outcome-level reward hybridization in boosting LLMs' reasoning ability via RL under other scenarios. The code will be released soon.
>
---
#### [new 016] Are LLMs Ready for English Standardized Tests? A Benchmarking and Elicitation Perspective
- **分类: cs.CL; cs.AI**

- **简介: 论文评估LLMs在英语标准化考试（EST）中的能力。任务：测试LLMs处理EST问题的潜力。问题：其准确性和多模态处理能力是否足够。工作：构建ESTBOOK基准（含5种测试、29题型、10576题），评估准确性和效率，提出分解框架分析推理步骤，为改进智能教育系统提供策略。**

- **链接: [http://arxiv.org/pdf/2505.17056v1](http://arxiv.org/pdf/2505.17056v1)**

> **作者:** Luoxi Tang; Tharunya Sundar; Shuai Yang; Ankita Patra; Manohar Chippada; Giqi Zhao; Yi Li; Riteng Zhang; Tunan Zhao; Ting Yang; Yuqiao Meng; Weicheng Ma; Zhaohan Xi
>
> **摘要:** AI is transforming education by enabling powerful tools that enhance learning experiences. Among recent advancements, large language models (LLMs) hold particular promise for revolutionizing how learners interact with educational content. In this work, we investigate the potential of LLMs to support standardized test preparation by focusing on English Standardized Tests (ESTs). Specifically, we assess their ability to generate accurate and contextually appropriate solutions across a diverse set of EST question types. We introduce ESTBOOK, a comprehensive benchmark designed to evaluate the capabilities of LLMs in solving EST questions. ESTBOOK aggregates five widely recognized tests, encompassing 29 question types and over 10,576 questions across multiple modalities, including text, images, audio, tables, and mathematical symbols. Using ESTBOOK, we systematically evaluate both the accuracy and inference efficiency of LLMs. Additionally, we propose a breakdown analysis framework that decomposes complex EST questions into task-specific solution steps. This framework allows us to isolate and assess LLM performance at each stage of the reasoning process. Evaluation findings offer insights into the capability of LLMs in educational contexts and point toward targeted strategies for improving their reliability as intelligent tutoring systems.
>
---
#### [new 017] QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于长上下文推理模型任务，旨在解决现有大模型在长文本处理中训练效率低和优化不稳定的问题。提出QwenLong-L1框架，通过渐进式上下文扩展、课程分阶段强化学习及难度感知采样策略，提升长文本推理能力，在7个基准测试中性能领先。**

- **链接: [http://arxiv.org/pdf/2505.17667v1](http://arxiv.org/pdf/2505.17667v1)**

> **作者:** Fanqi Wan; Weizhou Shen; Shengyi Liao; Yingcheng Shi; Chenliang Li; Ziyi Yang; Ji Zhang; Fei Huang; Jingren Zhou; Ming Yan
>
> **备注:** Technical Report
>
> **摘要:** Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QwenLong-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QwenLong-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments.
>
---
#### [new 018] RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language
- **分类: cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于多模态问答任务，解决模态间干扰导致模型误判的问题。提出RAVEN模型，通过查询引导的跨模态门控（QuART）动态分配模态token相关性分数，结合三阶段训练策略，提升问答准确性与抗干扰能力，并发布AVS-QA数据集，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17114v1](http://arxiv.org/pdf/2505.17114v1)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning -- each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio--Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks -- including egocentric and exocentric tasks -- show that RAVEN achieves up to 14.5\% and 8.0\% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4\% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23\%. Our code and dataset are available at https://github.com/BASHLab/RAVEN.
>
---
#### [new 019] Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对长链推理中的循环推理问题（因相邻token表示差异小），提出Shift-FFN方法，通过动态放大相邻token差异提升推理准确性，实验显示其结合LoRA可减少循环并优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.17153v1](http://arxiv.org/pdf/2505.17153v1)**

> **作者:** Yao Xu; Mingyu Xu; Fangyu Lei; Wangtao Sun; Xiangrong Zeng; Bingning Wang; Guang Liu; Shizhu He; Jun Zhao; Kang Liu
>
> **摘要:** Recently, models such as OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable performance on complex reasoning tasks through Long Chain-of-Thought (Long-CoT) reasoning. Although distilling this capability into student models significantly enhances their performance, this paper finds that fine-tuning LLMs with full parameters or LoRA with a low rank on long CoT data often leads to Cyclical Reasoning, where models repeatedly reiterate previous inference steps until the maximum length limit. Further analysis reveals that smaller differences in representations between adjacent tokens correlates with a higher tendency toward Cyclical Reasoning. To mitigate this issue, this paper proposes Shift Feedforward Networks (Shift-FFN), a novel approach that edits the current token's representation with the previous one before inputting it to FFN. This architecture dynamically amplifies the representation differences between adjacent tokens. Extensive experiments on multiple mathematical reasoning tasks demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of Cyclical Reasoning across various data sizes compared to full fine-tuning and standard LoRA. Our data and code are available at https://anonymous.4open.science/r/Shift-FFN
>
---
#### [new 020] Stereotype Detection in Natural Language Processing
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于NLP中的刻板印象检测任务，旨在通过分析多学科定义与文献（6000+篇），探究检测方法及挑战，提出其作为偏见早期监测工具的社会价值，呼吁多语言与交叉性研究。**

- **链接: [http://arxiv.org/pdf/2505.17642v1](http://arxiv.org/pdf/2505.17642v1)**

> **作者:** Alessandra Teresa Cignarella; Anastasia Giachanou; Els Lefever
>
> **摘要:** Stereotypes influence social perceptions and can escalate into discrimination and violence. While NLP research has extensively addressed gender bias and hate speech, stereotype detection remains an emerging field with significant societal implications. In this work is presented a survey of existing research, analyzing definitions from psychology, sociology, and philosophy. A semi-automatic literature review was performed by using Semantic Scholar. We retrieved and filtered over 6,000 papers (in the year range 2000-2025), identifying key trends, methodologies, challenges and future directions. The findings emphasize stereotype detection as a potential early-monitoring tool to prevent bias escalation and the rise of hate speech. Conclusions highlight the need for a broader, multilingual, and intersectional approach in NLP studies.
>
---
#### [new 021] Synthetic Data RL: Task Definition Is All You Need
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Synthetic Data RL框架，通过任务定义生成合成数据进行强化学习微调，解决RL依赖大量人工标注数据的问题。方法包括生成问题-答案对、动态调整问题难度并选择训练样本，在多个基准测试中显著提升性能，大幅减少人工数据需求。**

- **链接: [http://arxiv.org/pdf/2505.17063v1](http://arxiv.org/pdf/2505.17063v1)**

> **作者:** Yiduo Guo; Zhen Guo; Chuanwei Huang; Zi-Ang Wang; Zekai Zhang; Haofei Yu; Huishuai Zhang; Yikang Shen
>
> **摘要:** Reinforcement learning (RL) is a powerful way to adapt foundation models to specialized tasks, but its reliance on large-scale human-labeled data limits broad adoption. We introduce Synthetic Data RL, a simple and general framework that reinforcement fine-tunes models using only synthetic data generated from a task definition. Our method first generates question and answer pairs from the task definition and retrieved documents, then adapts the difficulty of the question based on model solvability, and selects questions using the average pass rate of the model across samples for RL training. On Qwen-2.5-7B, our method achieves a 29.2% absolute improvement over the base model on GSM8K (+2.9 pp vs. instruction-tuned, +6.6 pp vs. Self-Instruct), 8.7% on MATH, 13.1% on GPQA (+7.0 pp vs. SynthLLM), 8.9% on MedQA, 17.7% on CQA (law) and 13.7% on CFA (finance). It surpasses supervised fine-tuning under the same data budget and nearly matches RL with full human data across datasets (e.g., +17.2 pp on GSM8K). Adding 100 human demonstrations improves the performance of GSM8K only by 0.4 pp, showing a limited added value. By reducing human data annotation, Synthetic Data RL enables scalable and efficient RL-based model adaptation. Code and demos are available at https://github.com/gydpku/Data_Synthesis_RL/.
>
---
#### [new 022] Discriminating Form and Meaning in Multilingual Models with Minimal-Pair ABX Tasks
- **分类: cs.CL**

- **简介: 该论文提出基于ABX任务的无监督评估方法，分析多语言模型如何区分语言形式（如语言身份）与语义内容。通过零样本检测表征差异，发现语言辨别随训练减弱且集中于浅层，语义表征则在深层强化。对比探针任务验证有效性，为多语言模型表征提供轻量分析框架。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17747v1](http://arxiv.org/pdf/2505.17747v1)**

> **作者:** Maureen de Seyssel; Jie Chi; Skyler Seto; Maartje ter Hoeve; Masha Fedzechkina; Natalie Schluter
>
> **摘要:** We introduce a set of training-free ABX-style discrimination tasks to evaluate how multilingual language models represent language identity (form) and semantic content (meaning). Inspired from speech processing, these zero-shot tasks measure whether minimal differences in representation can be reliably detected. This offers a flexible and interpretable alternative to probing. Applied to XLM-R (Conneau et al, 2020) across pretraining checkpoints and layers, we find that language discrimination declines over training and becomes concentrated in lower layers, while meaning discrimination strengthens over time and stabilizes in deeper layers. We then explore probing tasks, showing some alignment between our metrics and linguistic learning performance. Our results position ABX tasks as a lightweight framework for analyzing the structure of multilingual representations.
>
---
#### [new 023] Contrastive Distillation of Emotion Knowledge from LLMs for Zero-Shot Emotion Recognition
- **分类: cs.CL**

- **简介: 该论文属于零样本情绪识别任务，解决传统模型依赖固定标签无法泛化及LLM设备部署难题。提出对比蒸馏框架，利用GPT-4生成情感描述作为监督信号，将LLM知识迁移到轻量模型，实现跨数据集/标签空间的零样本预测，模型规模缩小万倍且性能接近GPT-4。**

- **链接: [http://arxiv.org/pdf/2505.18040v1](http://arxiv.org/pdf/2505.18040v1)**

> **作者:** Minxue Niu; Emily Mower Provost
>
> **摘要:** The ability to handle various emotion labels without dedicated training is crucial for building adaptable Emotion Recognition (ER) systems. Conventional ER models rely on training using fixed label sets and struggle to generalize beyond them. On the other hand, Large Language Models (LLMs) have shown strong zero-shot ER performance across diverse label spaces, but their scale limits their use on edge devices. In this work, we propose a contrastive distillation framework that transfers rich emotional knowledge from LLMs into a compact model without the use of human annotations. We use GPT-4 to generate descriptive emotion annotations, offering rich supervision beyond fixed label sets. By aligning text samples with emotion descriptors in a shared embedding space, our method enables zero-shot prediction on different emotion classes, granularity, and label schema. The distilled model is effective across multiple datasets and label spaces, outperforming strong baselines of similar size and approaching GPT-4's zero-shot performance, while being over 10,000 times smaller.
>
---
#### [new 024] MIDB: Multilingual Instruction Data Booster for Enhancing Multilingual Instruction Synthesis
- **分类: cs.CL**

- **简介: 该论文属于多语言指令数据质量提升任务，针对机器翻译导致的多语言指令合成数据内容错误、翻译缺陷及本地化不足问题，提出MIDB模型。其通过16种语言的3.6万条专家修订数据训练，修正缺陷并增强本地化，显著提升多语言LLM的指令理解和文化认知能力。**

- **链接: [http://arxiv.org/pdf/2505.17671v1](http://arxiv.org/pdf/2505.17671v1)**

> **作者:** Yilun Liu; Chunguang Zhao; Xinhua Yang; Hongyong Zeng; Shimin Tao; Weibin Meng; Minggui He; Chang Su; Yan Yu; Hongxia Ma; Li Zhang; Daimeng Wei; Hao Yang
>
> **摘要:** Despite doubts on data quality, instruction synthesis has been widely applied into instruction tuning (IT) of LLMs as an economic and rapid alternative. Recent endeavors focus on improving data quality for synthesized instruction pairs in English and have facilitated IT of English-centric LLMs. However, data quality issues in multilingual synthesized instruction pairs are even more severe, since the common synthesizing practice is to translate English synthesized data into other languages using machine translation (MT). Besides the known content errors in these English synthesized data, multilingual synthesized instruction data are further exposed to defects introduced by MT and face insufficient localization of the target languages. In this paper, we propose MIDB, a Multilingual Instruction Data Booster to automatically address the quality issues in multilingual synthesized data. MIDB is trained on around 36.8k revision examples across 16 languages by human linguistic experts, thereby can boost the low-quality data by addressing content errors and MT defects, and improving localization in these synthesized data. Both automatic and human evaluation indicate that not only MIDB steadily improved instruction data quality in 16 languages, but also the instruction-following and cultural-understanding abilities of multilingual LLMs fine-tuned on MIDB-boosted data were significantly enhanced.
>
---
#### [new 025] MDIT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出评估大模型隐性偏见的新任务，解决现有研究忽视隐性毒性问题。构建含31万问题的MDIT-Bench基准，通过多阶段生成方法创建数据集，量化模型在三难度层级的毒性差距。实验显示13个主流模型在隐性毒性识别上表现显著下降，揭示其隐藏风险。**

- **链接: [http://arxiv.org/pdf/2505.17144v1](http://arxiv.org/pdf/2505.17144v1)**

> **作者:** Bohan Jin; Shuhan Qi; Kehai Chen; Xinyi Guo; Xuan Wang
>
> **备注:** Findings of ACL 2025
>
> **摘要:** The widespread use of Large Multimodal Models (LMMs) has raised concerns about model toxicity. However, current research mainly focuses on explicit toxicity, with less attention to some more implicit toxicity regarding prejudice and discrimination. To address this limitation, we introduce a subtler type of toxicity named dual-implicit toxicity and a novel toxicity benchmark termed MDIT-Bench: Multimodal Dual-Implicit Toxicity Benchmark. Specifically, we first create the MDIT-Dataset with dual-implicit toxicity using the proposed Multi-stage Human-in-loop In-context Generation method. Based on this dataset, we construct the MDIT-Bench, a benchmark for evaluating the sensitivity of models to dual-implicit toxicity, with 317,638 questions covering 12 categories, 23 subcategories, and 780 topics. MDIT-Bench includes three difficulty levels, and we propose a metric to measure the toxicity gap exhibited by the model across them. In the experiment, we conducted MDIT-Bench on 13 prominent LMMs, and the results show that these LMMs cannot handle dual-implicit toxicity effectively. The model's performance drops significantly in hard level, revealing that these LMMs still contain a significant amount of hidden but activatable toxicity. Data are available at https://github.com/nuo1nuo/MDIT-Bench.
>
---
#### [new 026] How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception
- **分类: cs.CL**

- **简介: 该论文研究LLM在实体事实问答中知识边界感知问题，发现知识流行度（实体及关系出现频率）与模型表现正相关。通过量化三类流行度指标，提出利用其校准置信度提升预测精度（5.24%），并探索无外部语料的流行度估计方法。**

- **链接: [http://arxiv.org/pdf/2505.17537v1](http://arxiv.org/pdf/2505.17537v1)**

> **作者:** Shiyu Ni; Keping Bi; Jiafeng Guo; Xueqi Cheng
>
> **摘要:** Large language models (LLMs) often fail to recognize their knowledge boundaries, producing confident yet incorrect answers. In this paper, we investigate how knowledge popularity affects LLMs' ability to perceive their knowledge boundaries. Focusing on entity-centric factual question answering (QA), we quantify knowledge popularity from three perspectives: the popularity of entities in the question, the popularity of entities in the answer, and relation popularity, defined as their co-occurrence frequency. Experiments on three representative datasets containing knowledge with varying popularity show that LLMs exhibit better QA performance, higher confidence, and more accurate perception on more popular knowledge, with relation popularity having the strongest correlation. Cause knowledge popularity shows strong correlation with LLMs' QA performance, we propose to leverage these signals for confidence calibration. This improves the accuracy of answer correctness prediction by an average of 5.24% across all models and datasets. Furthermore, we explore prompting LLMs to estimate popularity without external corpora, which yields a viable alternative.
>
---
#### [new 027] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估多模态模型主动风险意识任务，旨在解决AI系统被动响应无法及时识别风险的问题。提出Proactive Safety Bench（PaSBench），包含416个多模态场景，评估36个模型，揭示其推理不稳定导致性能不足，推动安全AI发展。**

- **链接: [http://arxiv.org/pdf/2505.17455v1](http://arxiv.org/pdf/2505.17455v1)**

> **作者:** Youliang Yuan; Wenxiang Jiao; Yuejin Xie; Chihao Shen; Menghan Tian; Wenxuan Wang; Jen-tse Huang; Pinjia He
>
> **备注:** Work in progress
>
> **摘要:** Human safety awareness gaps often prevent the timely recognition of everyday risks. In solving this problem, a proactive safety artificial intelligence (AI) system would work better than a reactive one. Instead of just reacting to users' questions, it would actively watch people's behavior and their environment to detect potential dangers in advance. Our Proactive Safety Bench (PaSBench) evaluates this capability through 416 multimodal scenarios (128 image sequences, 288 text logs) spanning 5 safety-critical domains. Evaluation of 36 advanced models reveals fundamental limitations: Top performers like Gemini-2.5-pro achieve 71% image and 64% text accuracy, but miss 45-55% risks in repeated trials. Through failure analysis, we identify unstable proactive reasoning rather than knowledge deficits as the primary limitation. This work establishes (1) a proactive safety benchmark, (2) systematic evidence of model limitations, and (3) critical directions for developing reliable protective AI. We believe our dataset and findings can promote the development of safer AI assistants that actively prevent harm rather than merely respond to requests. Our dataset can be found at https://huggingface.co/datasets/Youliang/PaSBench.
>
---
#### [new 028] Understanding How Value Neurons Shape the Generation of Specified Values in LLMs
- **分类: cs.CL**

- **简介: 该论文属于LLMs价值对齐的可解释性研究，旨在解决其内部价值表示不透明的问题。提出ValueLocate框架，构建基于Schwartz理论的ValueInsight数据集，通过计算神经元激活差异定位价值关键神经元，并验证操纵这些神经元可改变模型价值取向，建立神经机制与价值观的因果关系。**

- **链接: [http://arxiv.org/pdf/2505.17712v1](http://arxiv.org/pdf/2505.17712v1)**

> **作者:** Yi Su; Jiayi Zhang; Shu Yang; Xinhai Wang; Lijie Hu; Di Wang
>
> **摘要:** Rapid integration of large language models (LLMs) into societal applications has intensified concerns about their alignment with universal ethical principles, as their internal value representations remain opaque despite behavioral alignment advancements. Current approaches struggle to systematically interpret how values are encoded in neural architectures, limited by datasets that prioritize superficial judgments over mechanistic analysis. We introduce ValueLocate, a mechanistic interpretability framework grounded in the Schwartz Values Survey, to address this gap. Our method first constructs ValueInsight, a dataset that operationalizes four dimensions of universal value through behavioral contexts in the real world. Leveraging this dataset, we develop a neuron identification method that calculates activation differences between opposing value aspects, enabling precise localization of value-critical neurons without relying on computationally intensive attribution methods. Our proposed validation method demonstrates that targeted manipulation of these neurons effectively alters model value orientations, establishing causal relationships between neurons and value representations. This work advances the foundation for value alignment by bridging psychological value frameworks with neuron analysis in LLMs.
>
---
#### [new 029] Discovering Forbidden Topics in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出"拒绝发现"任务，旨在识别语言模型回避的主题。通过LLM-crawler方法利用token预填充技术，在Tulu-3-8B、Claude-Haiku及Llama系列模型中检测到审查模式，揭示部分模型存在记忆政治对齐响应的"思想抑制"现象，强调需通过此类方法检测AI系统的偏见与对齐缺陷。**

- **链接: [http://arxiv.org/pdf/2505.17441v1](http://arxiv.org/pdf/2505.17441v1)**

> **作者:** Can Rager; Chris Wendler; Rohit Gandikota; David Bau
>
> **摘要:** Refusal discovery is the task of identifying the full set of topics that a language model refuses to discuss. We introduce this new problem setting and develop a refusal discovery method, LLM-crawler, that uses token prefilling to find forbidden topics. We benchmark the LLM-crawler on Tulu-3-8B, an open-source model with public safety tuning data. Our crawler manages to retrieve 31 out of 36 topics within a budget of 1000 prompts. Next, we scale the crawl to a frontier model using the prefilling option of Claude-Haiku. Finally, we crawl three widely used open-weight models: Llama-3.3-70B and two of its variants finetuned for reasoning: DeepSeek-R1-70B and Perplexity-R1-1776-70B. DeepSeek-R1-70B reveals patterns consistent with censorship tuning: The model exhibits "thought suppression" behavior that indicates memorization of CCP-aligned responses. Although Perplexity-R1-1776-70B is robust to censorship, LLM-crawler elicits CCP-aligned refusals answers in the quantized model. Our findings highlight the critical need for refusal discovery methods to detect biases, boundaries, and alignment failures of AI systems.
>
---
#### [new 030] Reinforcing Question Answering Agents with Minimalist Policy Gradient Optimization
- **分类: cs.CL**

- **简介: 该论文针对大型语言模型（LLMs）在多跳问答任务中因知识不足和推理限制导致的幻觉问题，提出Mujica系统（通过分解问题为子图并结合检索推理）及MyGO优化方法（以最大似然估计替代传统策略梯度，简化训练）。旨在提升复杂QA性能，实现高效稳定训练。**

- **链接: [http://arxiv.org/pdf/2505.17086v1](http://arxiv.org/pdf/2505.17086v1)**

> **作者:** Yihong Wu; Liheng Ma; Muzhi Li; Jiaming Zhou; Jianye Hao; Ho-fung Leung; Irwin King; Yingxue Zhang; Jian-Yun Nie
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable versatility, due to the lack of factual knowledge, their application to Question Answering (QA) tasks remains hindered by hallucination. While Retrieval-Augmented Generation mitigates these issues by integrating external knowledge, existing approaches rely heavily on in-context learning, whose performance is constrained by the fundamental reasoning capabilities of LLMs. In this paper, we propose Mujica, a Multi-hop Joint Intelligence for Complex Question Answering, comprising a planner that decomposes questions into a directed acyclic graph of subquestions and a worker that resolves questions via retrieval and reasoning. Additionally, we introduce MyGO (Minimalist policy Gradient Optimization), a novel reinforcement learning method that replaces traditional policy gradient updates with Maximum Likelihood Estimation (MLE) by sampling trajectories from an asymptotically optimal policy. MyGO eliminates the need for gradient rescaling and reference models, ensuring stable and efficient training. Empirical results across multiple datasets demonstrate the effectiveness of Mujica-MyGO in enhancing multi-hop QA performance for various LLMs, offering a scalable and resource-efficient solution for complex QA tasks.
>
---
#### [new 031] ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.0**

- **简介: 该论文提出ConciseRL方法，通过无超参数的简洁性评分引导强化学习，优化推理模型效率。针对推理过程冗余导致计算浪费和错误问题，利用LLM作为动态评判者提供奖励信号，生成更简洁准确的推理路径。实验显示其在MATH和TheoremQA数据集显著减少计算量并提升准确率，消融实验验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.17250v1](http://arxiv.org/pdf/2505.17250v1)**

> **作者:** Razvan-Gabriel Dumitru; Darius Peteleaza; Vikas Yadav; Liangming Pan
>
> **备注:** 25 pages, 18 figures, and 6 tables
>
> **摘要:** Large language models excel at complex tasks by breaking down problems into structured reasoning steps. However, reasoning traces often extend beyond reaching a correct answer, causing wasted computation, reduced readability, and hallucinations. To address this, we introduce a novel hyperparameter-free conciseness score used as a reward signal within a reinforcement learning framework to guide models toward generating correct and concise reasoning traces. This score is evaluated by a large language model acting as a judge, enabling dynamic, context-aware feedback beyond simple token length. Our method achieves state-of-the-art efficiency-accuracy trade-offs on the MATH dataset, reducing token usage by up to 31x on simple problems while improving accuracy by 7%, and on the hardest problems, it outperforms full reasoning by +7.5% accuracy with up to 3.6x fewer tokens. On TheoremQA, our method improves accuracy by +2.2% using 12.5x fewer tokens. We also conduct ablation studies on the judge model, reward composition, and problem difficulty, showing that our method dynamically adapts reasoning length based on problem difficulty and benefits significantly from stronger judges. The code, model weights, and datasets are open-sourced at https://github.com/RazvanDu/ConciseRL.
>
---
#### [new 032] GemMaroc: Unlocking Darija Proficiency in LLMs with Minimal Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GemMaroc模型，通过少量高质量指令数据优化，提升开源LLMs的摩洛哥阿拉伯语（Darija）能力，解决现有模型对Darija支持不足的问题。研究将三组英文指令集翻译并精简，加入数学/科学提示，采用LoRA微调策略，在Gemma基座模型上实现Darija推理性能提升（如DarijaMMLU达61.6），同时保持跨语言推理能力，训练效率高（48GPU小时），推动绿色AI发展。**

- **链接: [http://arxiv.org/pdf/2505.17082v1](http://arxiv.org/pdf/2505.17082v1)**

> **作者:** Abderrahman Skiredj; Ferdaous Azhari; Houdaifa Atou; Nouamane Tazi; Ismail Berrada
>
> **摘要:** Open-source large language models (LLMs) still marginalise Moroccan Arabic (Darija), forcing practitioners either to bolt on heavyweight Arabic adapters or to sacrifice the very reasoning skills that make LLMs useful. We show that a rigorously quality-over-quantity alignment strategy can surface fluent Darija while safeguarding the backbone s cross-lingual reasoning at a sliver of the usual compute. We translate three compact instruction suites LIMA 1 K, DEITA 6 K and TULU 50 K into Darija, preserve 20 of the English originals, and add mathematics, coding and scientific prompts. A LoRA-tuned Gemma 3-4B trained on 5 K mixed instructions lifts DarijaMMLU from 32.8 to 42.7 ; adding the reasoning-dense TULU portion pushes it to 47.5 with no English regression. Scaling the identical recipe to Gemma 3-27B produces GemMaroc-27B, which matches Atlas-Chat on DarijaMMLU (61.6 ) and leaps ahead on Darija commonsense, scoring 60.5 on HellaSwag versus Atlas-Chat s 48.4 . Crucially, GemMaroc retains Gemma-27B s strong maths and general-reasoning ability, showing only minimal movement on GSM8K and English benchmarks. The entire model is trained in just 48 GPU.h, underscoring a Green AI pathway to inclusive, sustainable language technology. We release code, data and checkpoints to spur Darija-centric applications in education, public services and everyday digital interaction.
>
---
#### [new 033] Assessing the Quality of AI-Generated Clinical Notes: A Validated Evaluation of a Large Language Model Scribe
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估AI生成临床记录质量，解决缺乏评估方法的问题。通过盲测比较大型语言模型（LLM）与专家撰写的笔记，采用PDQI9量表评估97份病历，发现AI笔记质量接近人类（4.20 vs 4.25，p=0.04），验证PDQI9作为有效评估工具。**

- **链接: [http://arxiv.org/pdf/2505.17047v1](http://arxiv.org/pdf/2505.17047v1)**

> **作者:** Erin Palm; Astrit Manikantan; Mark E. Pepin; Herprit Mahal; Srikanth Subramanya Belwadi
>
> **备注:** 15 pages, 5 tables, 1 figure. Submitted for peer review 05/15/2025
>
> **摘要:** In medical practices across the United States, physicians have begun implementing generative artificial intelligence (AI) tools to perform the function of scribes in order to reduce the burden of documenting clinical encounters. Despite their widespread use, no established methods exist to gauge the quality of AI scribes. To address this gap, we developed a blinded study comparing the relative performance of large language model (LLM) generated clinical notes with those from field experts based on audio-recorded clinical encounters. Quantitative metrics from the Physician Documentation Quality Instrument (PDQI9) provided a framework to measure note quality, which we adapted to assess relative performance of AI generated notes. Clinical experts spanning 5 medical specialties used the PDQI9 tool to evaluate specialist-drafted Gold notes and LLM authored Ambient notes. Two evaluators from each specialty scored notes drafted from a total of 97 patient visits. We found uniformly high inter rater agreement (RWG greater than 0.7) between evaluators in general medicine, orthopedics, and obstetrics and gynecology, and moderate (RWG 0.5 to 0.7) to high inter rater agreement in pediatrics and cardiology. We found a modest yet significant difference in the overall note quality, wherein Gold notes achieved a score of 4.25 out of 5 and Ambient notes scored 4.20 out of 5 (p = 0.04). Our findings support the use of the PDQI9 instrument as a practical method to gauge the quality of LLM authored notes, as compared to human-authored notes.
>
---
#### [new 034] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM
- **分类: cs.CL**

- **简介: 该论文属于多模态视频时间理解任务，旨在解决现有模型难以有效融合音频、视觉和语音信息的问题。提出TriSense三模态LLM，通过Query-Based Connector自适应调整模态权重，并构建TriSense-2M数据集（200万样本），提升视频时空分析性能。**

- **链接: [http://arxiv.org/pdf/2505.18110v1](http://arxiv.org/pdf/2505.18110v1)**

> **作者:** Zinuo Li; Xian Zhang; Yongxin Guo; Mohammed Bennamoun; Farid Boussaid; Girish Dwivedi; Luqi Gong; Qiuhong Ke
>
> **摘要:** Humans naturally understand moments in a video by integrating visual and auditory cues. For example, localizing a scene in the video like "A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding" requires simultaneous processing of visual, audio, and speech signals. However, existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding. To address this, we present TriSense, a triple-modality large language model designed for holistic video temporal understanding through the integration of visual, audio, and speech modalities. Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs. To support TriSense's multimodal capabilities, we introduce TriSense-2M, a high-quality dataset of over 2 million curated samples generated via an automated pipeline powered by fine-tuned LLMs. TriSense-2M includes long-form videos and diverse modality combinations, facilitating broad generalization. Extensive experiments across multiple benchmarks demonstrate the effectiveness of TriSense and its potential to advance multimodal video analysis. Code and dataset will be publicly released.
>
---
#### [new 035] TRACE for Tracking the Emergence of Semantic Representations in Transformers
- **分类: cs.CL**

- **简介: 该论文提出TRACE框架，分析Transformer模型训练中语义表示的相变机制。针对现有研究忽视语言结构形成的问题，结合几何、信息及语言信号，利用可控合成语料库追踪相变过程，揭示几何突变与语法语义能力同步提升的规律，阐明模型架构对优化的影响，推动语言抽象生成的可解释性研究。**

- **链接: [http://arxiv.org/pdf/2505.17998v1](http://arxiv.org/pdf/2505.17998v1)**

> **作者:** Nura Aljaafari; Danilo S. Carvalho; André Freitas
>
> **摘要:** Modern transformer models exhibit phase transitions during training, distinct shifts from memorisation to abstraction, but the mechanisms underlying these transitions remain poorly understood. Prior work has often focused on endpoint representations or isolated signals like curvature or mutual information, typically in symbolic or arithmetic domains, overlooking the emergence of linguistic structure. We introduce TRACE (Tracking Representation Abstraction and Compositional Emergence), a diagnostic framework combining geometric, informational, and linguistic signals to detect phase transitions in Transformer-based LMs. TRACE leverages a frame-semantic data generation method, ABSynth, that produces annotated synthetic corpora with controllable complexity, lexical distributions, and structural entropy, while being fully annotated with linguistic categories, enabling precise analysis of abstraction emergence. Experiments reveal that (i) phase transitions align with clear intersections between curvature collapse and dimension stabilisation; (ii) these geometric shifts coincide with emerging syntactic and semantic accuracy; (iii) abstraction patterns persist across architectural variants, with components like feedforward networks affecting optimisation stability rather than fundamentally altering trajectories. This work advances our understanding of how linguistic abstractions emerge in LMs, offering insights into model interpretability, training efficiency, and compositional generalisation that could inform more principled approaches to LM development.
>
---
#### [new 036] WiNGPT-3.0 Technical Report
- **分类: cs.CL**

- **简介: 该论文属于医疗大模型开发任务，旨在解决现有LLMs在结构化医疗推理及临床部署中的不足。团队通过多阶段训练（含监督微调与强化学习），结合长链推理数据集和奖励模型，优化WiNGPT-3.0的医疗诊断与临床推理能力，在MedCalc/MedQA等任务中提升准确率，验证小数据强化学习对医疗模型部署的可行性。**

- **链接: [http://arxiv.org/pdf/2505.17387v1](http://arxiv.org/pdf/2505.17387v1)**

> **作者:** Boqin Zhuang; Chenxiao Song; Huitong Lu; Jiacheng Qiao; Mingqian Liu; Mingxing Yu; Ping Hong; Rui Li; Xiaoxia Song; Xiangjun Xu; Xu Chen; Yaoyao Ma; Yujie Gao
>
> **摘要:** Current Large Language Models (LLMs) exhibit significant limitations, notably in structured, interpretable, and verifiable medical reasoning, alongside practical deployment challenges related to computational resources and data privacy. This report focused on the development of WiNGPT-3.0, the 32-billion parameter LLMs, engineered with the objective of enhancing its capacity for medical reasoning and exploring its potential for effective integration within healthcare IT infrastructures. The broader aim is to advance towards clinically applicable models. The approach involved a multi-stage training pipeline tailored for general, medical, and clinical reasoning. This pipeline incorporated supervised fine-tuning (SFT) and reinforcement learning (RL), leveraging curated Long Chain-of-Thought (CoT) datasets, auxiliary reward models, and an evidence-based diagnostic chain simulation. WiNGPT-3.0 demonstrated strong performance: specific model variants achieved scores of 66.6 on MedCalc and 87.1 on MedQA-USMLE. Furthermore, targeted training improved performance on a clinical reasoning task from a baseline score of 58.1 to 62.5. These findings suggest that reinforcement learning, even when applied with a limited dataset of only a few thousand examples, can enhance medical reasoning accuracy. Crucially, this demonstration of RL's efficacy with limited data and computation paves the way for more trustworthy and practically deployable LLMs within clinical workflows and health information infrastructures.
>
---
#### [new 037] VLM-KG: Multimodal Radiology Knowledge Graph Generation
- **分类: cs.CL; cs.CV; cs.IR; cs.LG**

- **简介: 该论文提出VLM-KG框架，属于多模态医学影像知识图谱生成任务。针对现有方法仅利用文本报告、忽略影像信息且难以处理长文本的问题，通过融合视觉-语言模型与影像-报告数据，构建首个多模态放射学知识图谱生成方案，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2505.17042v1](http://arxiv.org/pdf/2505.17042v1)**

> **作者:** Abdullah Abdullah; Seong Tae Kim
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable success in natural language generation, excelling at instruction following and structured output generation. Knowledge graphs play a crucial role in radiology, serving as valuable sources of factual information and enhancing various downstream tasks. However, generating radiology-specific knowledge graphs presents significant challenges due to the specialized language of radiology reports and the limited availability of domain-specific data. Existing solutions are predominantly unimodal, meaning they generate knowledge graphs only from radiology reports while excluding radiographic images. Additionally, they struggle with long-form radiology data due to limited context length. To address these limitations, we propose a novel multimodal VLM-based framework for knowledge graph generation in radiology. Our approach outperforms previous methods and introduces the first multimodal solution for radiology knowledge graph generation.
>
---
#### [new 038] Counting Cycles with Deepseek
- **分类: cs.CL**

- **简介: 该论文属于AI辅助数学问题解决任务，旨在通过结合AI与人类策略，开发循环计数统计量的高效计算公式（CEEF）。针对该难题，团队提出新方法，利用DeepSeek-R1等AI工具，在人类指导下完成复杂组合计算，推导出通用新公式，验证了AI在结构化引导下的数学创新能力。**

- **链接: [http://arxiv.org/pdf/2505.17964v1](http://arxiv.org/pdf/2505.17964v1)**

> **作者:** Jiashun Jin; Tracy Ke; Bingcheng Sui; Zhenggang Wang
>
> **摘要:** Despite recent progress, AI still struggles on advanced mathematics. We consider a difficult open problem: How to derive a Computationally Efficient Equivalent Form (CEEF) for the cycle count statistic? The CEEF problem does not have known general solutions, and requires delicate combinatorics and tedious calculations. Such a task is hard to accomplish by humans but is an ideal example where AI can be very helpful. We solve the problem by combining a novel approach we propose and the powerful coding skills of AI. Our results use delicate graph theory and contain new formulas for general cases that have not been discovered before. We find that, while AI is unable to solve the problem all by itself, it is able to solve it if we provide it with a clear strategy, a step-by-step guidance and carefully written prompts. For simplicity, we focus our study on DeepSeek-R1 but we also investigate other AI approaches.
>
---
#### [new 039] Semi-Clairvoyant Scheduling of Speculative Decoding Requests to Minimize LLM Inference Latency
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对LLM推理延迟优化任务，解决投机解码中请求调度效率低的问题。现有方法仅依赖输出长度估算执行时间，忽略token接受率影响。提出LAPS-SD算法，通过动态维护多优先队列及预emption，在接受率稳定后精准调度，实验显示降低39%延迟。**

- **链接: [http://arxiv.org/pdf/2505.17074v1](http://arxiv.org/pdf/2505.17074v1)**

> **作者:** Ruixiao Li; Fahao Chen; Peng Li
>
> **摘要:** Speculative decoding accelerates Large Language Model (LLM) inference by employing a small speculative model (SSM) to generate multiple candidate tokens and verify them using the LLM in parallel. This technique has been widely integrated into LLM inference serving systems. However, inference requests typically exhibit uncertain execution time, which poses a significant challenge of efficiently scheduling requests in these systems. Existing work estimates execution time based solely on predicted output length, which could be inaccurate because execution time depends on both output length and token acceptance rate of verification by the LLM. In this paper, we propose a semi-clairvoyant request scheduling algorithm called Least-Attained/Perceived-Service for Speculative Decoding (LAPS-SD). Given a number of inference requests, LAPS-SD can effectively minimize average inference latency by adaptively scheduling requests according to their features during decoding. When the token acceptance rate is dynamic and execution time is difficult to estimate, LAPS-SD maintains multiple priority queues and allows request execution preemption across different queues. Once the token acceptance rate becomes stable, LAPS-SD can accurately estimate the execution time and schedule requests accordingly. Extensive experiments show that LAPS-SD reduces inference latency by approximately 39\% compared to state-of-the-art scheduling methods.
>
---
#### [new 040] When can isotropy help adapt LLMs' next word prediction to numerical domains?
- **分类: cs.CL**

- **简介: 该论文研究LLMs在数值领域适应性，解决其预测可靠性问题。通过分析嵌入空间的各向同性，提出基于log-linear模型的理论框架，揭示LLM需具备平移不变性结构以优化数值预测，并实验验证数据与架构对各向同性的影响。**

- **链接: [http://arxiv.org/pdf/2505.17135v1](http://arxiv.org/pdf/2505.17135v1)**

> **作者:** Rashed Shelim; Shengzhe Xu; Walid Saad; Naren Ramakrishnan
>
> **摘要:** Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
>
---
#### [new 041] Mutarjim: Advancing Bidirectional Arabic-English Translation with a Small Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mutarjim，基于Kuwain-1.5B的小型双向翻译模型，通过优化训练和优质语料，在阿拉伯-英语翻译中超越大模型，降低计算成本。同时创建Tarjama-25新基准，解决现有数据集领域窄、句子短、英语偏见等问题，模型在英译阿任务达SOTA，超越GPT-4o mini。**

- **链接: [http://arxiv.org/pdf/2505.17894v1](http://arxiv.org/pdf/2505.17894v1)**

> **作者:** Khalil Hennara; Muhammad Hreden; Mohamed Motaism Hamed; Zeina Aldallal; Sara Chrouf; Safwan AlModhayan
>
> **摘要:** We introduce Mutarjim, a compact yet powerful language model for bidirectional Arabic-English translation. While large-scale LLMs have shown impressive progress in natural language processing tasks, including machine translation, smaller models. Leveraging this insight, we developed Mutarjim based on Kuwain-1.5B , a language model tailored for both Arabic and English. Despite its modest size, Mutarjim outperforms much larger models on several established benchmarks, achieved through an optimized two-phase training approach and a carefully curated, high-quality training corpus.. Experimental results show that Mutarjim rivals models up to 20 times larger while significantly reducing computational costs and training requirements. We also introduce Tarjama-25, a new benchmark designed to overcome limitations in existing Arabic-English benchmarking datasets, such as domain narrowness, short sentence lengths, and English-source bias. Tarjama-25 comprises 5,000 expert-reviewed sentence pairs and spans a wide range of domains, offering a more comprehensive and balanced evaluation framework. Notably, Mutarjim achieves state-of-the-art performance on the English-to-Arabic task in Tarjama-25, surpassing even significantly larger and proprietary models like GPT-4o mini. We publicly release Tarjama-25 to support future research and advance the evaluation of Arabic-English translation systems.
>
---
#### [new 042] METHOD: Modular Efficient Transformer for Health Outcome Discovery
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出METHOD模型，针对电子健康记录（EHR）中不规则时间序列、复杂临床关系等挑战，设计患者感知注意力机制、自适应滑动窗口及U-Net架构，提升重症预测精度与长序列处理效率，在MIMIC-IV数据集上优于现有模型，任务为医疗结果预测。**

- **链接: [http://arxiv.org/pdf/2505.17054v1](http://arxiv.org/pdf/2505.17054v1)**

> **作者:** Linglong Qian; Zina Ibrahim
>
> **备注:** 23 pages
>
> **摘要:** Recent advances in transformer architectures have revolutionised natural language processing, but their application to healthcare domains presents unique challenges. Patient timelines are characterised by irregular sampling, variable temporal dependencies, and complex contextual relationships that differ substantially from traditional language tasks. This paper introduces \METHOD~(Modular Efficient Transformer for Health Outcome Discovery), a novel transformer architecture specifically designed to address the challenges of clinical sequence modelling in electronic health records. \METHOD~integrates three key innovations: (1) a patient-aware attention mechanism that prevents information leakage whilst enabling efficient batch processing; (2) an adaptive sliding window attention scheme that captures multi-scale temporal dependencies; and (3) a U-Net inspired architecture with dynamic skip connections for effective long sequence processing. Evaluations on the MIMIC-IV database demonstrate that \METHOD~consistently outperforms the state-of-the-art \ETHOS~model, particularly in predicting high-severity cases that require urgent clinical intervention. \METHOD~exhibits stable performance across varying inference lengths, a crucial feature for clinical deployment where patient histories vary significantly in length. Analysis of learned embeddings reveals that \METHOD~better preserves clinical hierarchies and relationships between medical concepts. These results suggest that \METHOD~represents a significant advancement in transformer architectures optimised for healthcare applications, providing more accurate and clinically relevant predictions whilst maintaining computational efficiency.
>
---
#### [new 043] FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain
- **分类: cs.CL**

- **简介: 该论文提出FinRAGBench-V，填补金融领域多模态RAG忽视视觉信息的空白。针对现有研究依赖文本数据导致分析盲区，构建含中英双语语料及七类QA数据的基准，开发RGenCite基线模型并设计自动评估方法，提升多模态RAG系统开发。**

- **链接: [http://arxiv.org/pdf/2505.17471v1](http://arxiv.org/pdf/2505.17471v1)**

> **作者:** Suifeng Zhao; Zhuoran Jin; Sujian Li; Jun Gao
>
> **摘要:** Retrieval-Augmented Generation (RAG) plays a vital role in the financial domain, powering applications such as real-time market analysis, trend forecasting, and interest rate computation. However, most existing RAG research in finance focuses predominantly on textual data, overlooking the rich visual content in financial documents, resulting in the loss of key analytical insights. To bridge this gap, we present FinRAGBench-V, a comprehensive visual RAG benchmark tailored for finance which effectively integrates multimodal data and provides visual citation to ensure traceability. It includes a bilingual retrieval corpus with 60,780 Chinese and 51,219 English pages, along with a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types and seven question categories. Moreover, we introduce RGenCite, an RAG baseline that seamlessly integrates visual citation with generation. Furthermore, we propose an automatic citation evaluation method to systematically assess the visual citation capabilities of Multimodal Large Language Models (MLLMs). Extensive experiments on RGenCite underscore the challenging nature of FinRAGBench-V, providing valuable insights for the development of multimodal RAG systems in finance.
>
---
#### [new 044] Graph-Linguistic Fusion: Using Language Models for Wikidata Vandalism Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Graph-Linguistic Fusion方法，用于Wikidata破坏检测。针对其结构化三元组与多语言文本混合编辑的复杂性，通过Graph2Text将编辑统一为文本表示，结合多语言模型实现统一检测，提升覆盖与维护效率，实验显示优于现有系统并开源代码与数据集。**

- **链接: [http://arxiv.org/pdf/2505.18136v1](http://arxiv.org/pdf/2505.18136v1)**

> **作者:** Mykola Trokhymovych; Lydia Pintscher; Ricardo Baeza-Yates; Diego Saez-Trumper
>
> **摘要:** We introduce a next-generation vandalism detection system for Wikidata, one of the largest open-source structured knowledge bases on the Web. Wikidata is highly complex: its items incorporate an ever-expanding universe of factual triples and multilingual texts. While edits can alter both structured and textual content, our approach converts all edits into a single space using a method we call Graph2Text. This allows for evaluating all content changes for potential vandalism using a single multilingual language model. This unified approach improves coverage and simplifies maintenance. Experiments demonstrate that our solution outperforms the current production system. Additionally, we are releasing the code under an open license along with a large dataset of various human-generated knowledge alterations, enabling further research.
>
---
#### [new 045] ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework
- **分类: cs.CL**

- **简介: 该论文提出ManuSearch框架，通过透明模块化多代理系统（规划、搜索、阅读代理）民主化LLMs的深度搜索，解决封闭系统不透明问题。构建ORION基准评估，实验显示其超越开源基线和闭源系统，并开源代码数据。属于开放领域深度推理任务，旨在推动可复现的LLMs研究。**

- **链接: [http://arxiv.org/pdf/2505.18105v1](http://arxiv.org/pdf/2505.18105v1)**

> **作者:** Lisheng Huang; Yichen Liu; Jinhao Jiang; Rongxiang Zhang; Jiahao Yan; Junyi Li; Wayne Xin Zhao
>
> **备注:** LLM, Complex Search Benchmark
>
> **摘要:** Recent advances in web-augmented large language models (LLMs) have exhibited strong performance in complex reasoning tasks, yet these capabilities are mostly locked in proprietary systems with opaque architectures. In this work, we propose \textbf{ManuSearch}, a transparent and modular multi-agent framework designed to democratize deep search for LLMs. ManuSearch decomposes the search and reasoning process into three collaborative agents: (1) a solution planning agent that iteratively formulates sub-queries, (2) an Internet search agent that retrieves relevant documents via real-time web search, and (3) a structured webpage reading agent that extracts key evidence from raw web content. To rigorously evaluate deep reasoning abilities, we introduce \textbf{ORION}, a challenging benchmark focused on open-web reasoning over long-tail entities, covering both English and Chinese. Experimental results show that ManuSearch substantially outperforms prior open-source baselines and even surpasses leading closed-source systems. Our work paves the way for reproducible, extensible research in open deep search systems. We release the data and code in https://github.com/RUCAIBox/ManuSearch
>
---
#### [new 046] Curriculum Guided Reinforcement Learning for Efficient Multi Hop Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对多跳问答任务，解决现有RAG方法冗余查询、探索不足及效率低的问题。提出EVO-RAG框架，采用课程指导强化学习，结合七因素动态奖励模型，优化查询策略，提升准确率（+4.6%）并减少15%检索深度。**

- **链接: [http://arxiv.org/pdf/2505.17391v1](http://arxiv.org/pdf/2505.17391v1)**

> **作者:** Yuelyu Ji; Rui Meng; Zhuochun Li; Daqing He
>
> **摘要:** Retrieval-augmented generation (RAG) grounds large language models (LLMs) in up-to-date external evidence, yet existing multi-hop RAG pipelines still issue redundant subqueries, explore too shallowly, or wander through overly long search chains. We introduce EVO-RAG, a curriculum-guided reinforcement learning framework that evolves a query-rewriting agent from broad early-stage exploration to concise late-stage refinement. EVO-RAG couples a seven-factor, step-level reward vector (covering relevance, redundancy, efficiency, and answer correctness) with a time-varying scheduler that reweights these signals as the episode unfolds. The agent is trained with Direct Preference Optimization over a multi-head reward model, enabling it to learn when to search, backtrack, answer, or refuse. Across four multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle), EVO-RAG boosts Exact Match by up to 4.6 points over strong RAG baselines while trimming average retrieval depth by 15 %. Ablation studies confirm the complementary roles of curriculum staging and dynamic reward scheduling. EVO-RAG thus offers a general recipe for building reliable, cost-effective multi-hop RAG systems.
>
---
#### [new 047] Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文提出无训练框架Hydra，解决LLM在跨源知识增强推理中的多跳、多实体及多源验证问题。通过融合知识图谱结构与文本语义，结合三重验证机制（源可信度、跨模态印证、路径对齐），提升LLM推理精度。实验显示其性能超现有方法20.3%，并使小模型接近GPT-4水平。**

- **链接: [http://arxiv.org/pdf/2505.17464v1](http://arxiv.org/pdf/2505.17464v1)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present Hydra, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. Hydra handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, Hydra uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, Hydra fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that Hydra achieves overall state-of-the-art results on all benchmarks with GPT-3.5, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, Hydra enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo.
>
---
#### [new 048] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出MoD方法，通过评估大视觉语言模型（LVLMs）对图像token注意力的一致性，动态采用互补或对比策略调整解码，减少模型幻觉问题，在多基准测试中效果显著。**

- **链接: [http://arxiv.org/pdf/2505.17061v1](http://arxiv.org/pdf/2505.17061v1)**

> **作者:** Xinlong Chen; Yuanxing Zhang; Qiang Liu; Junfei Wu; Fuzheng Zhang; Tieniu Tan
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model's attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model's attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. The code is available at https://github.com/xlchen0205/MoD.
>
---
#### [new 049] FB-RAG: Improving RAG with Forward and Backward Lookup
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FB-RAG框架，改进检索增强生成（RAG）系统。针对复杂查询中上下文规模与信息相关性平衡难题，结合反向（与查询重叠）和正向（与候选答案重叠）检索策略，精准选取相关片段。实验显示其性能优于基线方法且降低延迟，适用于信息检索任务。**

- **链接: [http://arxiv.org/pdf/2505.17206v1](http://arxiv.org/pdf/2505.17206v1)**

> **作者:** Kushal Chawla; Alfy Samuel; Anoop Kumar; Daben Liu
>
> **摘要:** The performance of Retrieval Augmented Generation (RAG) systems relies heavily on the retriever quality and the size of the retrieved context. A large enough context ensures that the relevant information is present in the input context for the LLM, but also incorporates irrelevant content that has been shown to confuse the models. On the other hand, a smaller context reduces the irrelevant information, but it often comes at the risk of losing important information necessary to answer the input question. This duality is especially challenging to manage for complex queries that contain little information to retrieve the relevant chunks from the full context. To address this, we present a novel framework, called FB-RAG, which enhances the RAG pipeline by relying on a combination of backward lookup (overlap with the query) and forward lookup (overlap with candidate reasons and answers) to retrieve specific context chunks that are the most relevant for answering the input query. Our evaluations on 9 datasets from two leading benchmarks show that FB-RAG consistently outperforms RAG and Long Context baselines developed recently for these benchmarks. We further show that FB-RAG can improve performance while reducing latency. We perform qualitative analysis of the strengths and shortcomings of our approach, providing specific insights to guide future work.
>
---
#### [new 050] The Staircase of Ethics: Probing LLM Value Priorities through Multi-Step Induction to Complex Moral Dilemmas
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于LLM伦理推理评估任务，解决现有单步评估无法捕捉模型应对动态道德困境能力的问题。提出首个含3302个多阶段道德困境的数据集MMDs，分析9种LLM发现其价值观（如关怀与公平）随情境复杂性动态调整，呼吁采用动态评估方法以推动人机价值对齐。**

- **链接: [http://arxiv.org/pdf/2505.18154v1](http://arxiv.org/pdf/2505.18154v1)**

> **作者:** Ya Wu; Qiang Sheng; Danding Wang; Guang Yang; Yifan Sun; Zhengjia Wang; Yuyan Bu; Juan Cao
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Ethical decision-making is a critical aspect of human judgment, and the growing use of LLMs in decision-support systems necessitates a rigorous evaluation of their moral reasoning capabilities. However, existing assessments primarily rely on single-step evaluations, failing to capture how models adapt to evolving ethical challenges. Addressing this gap, we introduce the Multi-step Moral Dilemmas (MMDs), the first dataset specifically constructed to evaluate the evolving moral judgments of LLMs across 3,302 five-stage dilemmas. This framework enables a fine-grained, dynamic analysis of how LLMs adjust their moral reasoning across escalating dilemmas. Our evaluation of nine widely used LLMs reveals that their value preferences shift significantly as dilemmas progress, indicating that models recalibrate moral judgments based on scenario complexity. Furthermore, pairwise value comparisons demonstrate that while LLMs often prioritize the value of care, this value can sometimes be superseded by fairness in certain contexts, highlighting the dynamic and context-dependent nature of LLM ethical reasoning. Our findings call for a shift toward dynamic, context-aware evaluation paradigms, paving the way for more human-aligned and value-sensitive development of LLMs.
>
---
#### [new 051] EXECUTE: A Multilingual Benchmark for LLM Token Understanding
- **分类: cs.CL**

- **简介: 该论文提出多语言基准EXECUTE，扩展CUTE基准至不同书写系统语言，评估LLMs的字符级理解能力。发现非英语语言的问题并非均在字符层面，部分语言存在词级处理缺陷或无明显问题，并分析中日韩等语言的子字符理解表现。**

- **链接: [http://arxiv.org/pdf/2505.17784v1](http://arxiv.org/pdf/2505.17784v1)**

> **作者:** Lukas Edman; Helmut Schmid; Alexander Fraser
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** The CUTE benchmark showed that LLMs struggle with character understanding in English. We extend it to more languages with diverse scripts and writing systems, introducing EXECUTE. Our simplified framework allows easy expansion to any language. Tests across multiple LLMs reveal that challenges in other languages are not always on the character level as in English. Some languages show word-level processing issues, some show no issues at all. We also examine sub-character tasks in Chinese, Japanese, and Korean to assess LLMs' understanding of character components.
>
---
#### [new 052] EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EarthSE基准，评估大语言模型在地球科学探索中的能力，解决现有基准缺乏专业性、全面性和开放探索评估的问题。构建Earth-Iron（广度）、Earth-Silver（深度）和Earth-Gold（多轮对话）三类数据集，覆盖多学科任务，实验显示现有模型存在明显不足，数据已开源。**

- **链接: [http://arxiv.org/pdf/2505.17139v1](http://arxiv.org/pdf/2505.17139v1)**

> **作者:** Wanghan Xu; Xiangyu Zhao; Yuhao Zhou; Xiaoyu Yue; Ben Fei; Fenghua Ling; Wenlong Zhang; Lei Bai
>
> **摘要:** Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on https://huggingface.co/ai-earth .
>
---
#### [new 053] Investigating Affect Mining Techniques for Annotation Sample Selection in the Creation of Finnish Affective Speech Corpus
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感语音语料库构建任务，旨在解决芬兰语缺乏自然情感表达语料的问题。研究通过结合声学、跨语言情感及文本情感特征的挖掘技术，从三个大型语料库中选取1.2万条样本进行情感标注（唤醒度与效价），对比随机采样方法，优化样本多样性，并提出情感语料库建设的采样策略。**

- **链接: [http://arxiv.org/pdf/2505.17833v1](http://arxiv.org/pdf/2505.17833v1)**

> **作者:** Kalle Lahtinen; Einari Vaaras; Liisa Mustanoja; Okko Räsänen
>
> **备注:** Accepted for publication at Interspeech 2025, Rotterdam, The Netherlands
>
> **摘要:** Study of affect in speech requires suitable data, as emotional expression and perception vary across languages. Until now, no corpus has existed for natural expression of affect in spontaneous Finnish, existing data being acted or from a very specific communicative setting. This paper presents the first such corpus, created by annotating 12,000 utterances for emotional arousal and valence, sampled from three large-scale Finnish speech corpora. To ensure diverse affective expression, sample selection was conducted with an affect mining approach combining acoustic, cross-linguistic speech emotion, and text sentiment features. We compare this method to random sampling in terms of annotation diversity, and conduct post-hoc analyses to identify sampling choices that would have maximized the diversity. As an outcome, the work introduces a spontaneous Finnish affective speech corpus and informs sampling strategies for affective speech corpus creation in other languages or domains.
>
---
#### [new 054] Multi-Modality Expansion and Retention for LLMs through Parameter Merging and Decoupling
- **分类: cs.CL**

- **简介: 该论文属于多模态扩展与模型保留任务，针对传统多模态LLMs需资源密集的从头微调且易遗忘原有能力的问题，提出MMER方法：通过合并现有模型参数并生成二进制掩码解耦模态相关参数，实现零训练扩展新模态同时保留99%原性能，并缓解灾难性遗忘。**

- **链接: [http://arxiv.org/pdf/2505.17110v1](http://arxiv.org/pdf/2505.17110v1)**

> **作者:** Junlin Li; Guodong DU; Jing Li; Sim Kuan Goh; Wenya Wang; Yequan Wang; Fangming Liu; Ho-Kin Tang; Saleh Alharbi; Daojing He; Min Zhang
>
> **摘要:** Fine-tuning Large Language Models (LLMs) with multimodal encoders on modality-specific data expands the modalities that LLMs can handle, leading to the formation of Multimodal LLMs (MLLMs). However, this paradigm heavily relies on resource-intensive and inflexible fine-tuning from scratch with new multimodal data. In this paper, we propose MMER (Multi-modality Expansion and Retention), a training-free approach that integrates existing MLLMs for effective multimodal expansion while retaining their original performance. Specifically, MMER reuses MLLMs' multimodal encoders while merging their LLM parameters. By comparing original and merged LLM parameters, MMER generates binary masks to approximately separate LLM parameters for each modality. These decoupled parameters can independently process modality-specific inputs, reducing parameter conflicts and preserving original MLLMs' fidelity. MMER can also mitigate catastrophic forgetting by applying a similar process to MLLMs fine-tuned on new tasks. Extensive experiments show significant improvements over baselines, proving that MMER effectively expands LLMs' multimodal capabilities while retaining 99% of the original performance, and also markedly mitigates catastrophic forgetting.
>
---
#### [new 055] BanglaByT5: Byte-Level Modelling for Bangla
- **分类: cs.CL**

- **简介: 该论文提出BanglaByT5，针对传统分词器在形态复杂语言孟加拉语中的不足，采用字节级建模，基于ByT5架构预训练14GB语料库。通过零样本及监督评估，在生成和分类任务中超越多语言模型，验证了字节级模型在资源受限场景的适用性。任务为孟加拉语NLP建模，解决传统分词器的表征缺陷，工作包括模型构建、训练及性能验证。**

- **链接: [http://arxiv.org/pdf/2505.17102v1](http://arxiv.org/pdf/2505.17102v1)**

> **作者:** Pramit Bhattacharyya; Arnab Bhattacharya
>
> **摘要:** Large language models (LLMs) have achieved remarkable success across various natural language processing tasks. However, most LLM models use traditional tokenizers like BPE and SentencePiece, which fail to capture the finer nuances of a morphologically rich language like Bangla (Bengali). In this work, we introduce BanglaByT5, the first byte-level encoder-decoder model explicitly tailored for Bangla. Built upon a small variant of Googles ByT5 architecture, BanglaByT5 is pre-trained on a 14GB curated corpus combining high-quality literary and newspaper articles. Through zeroshot and supervised evaluations across generative and classification tasks, BanglaByT5 demonstrates competitive performance, surpassing several multilingual and larger models. Our findings highlight the efficacy of byte-level modelling for morphologically rich languages and highlight BanglaByT5 potential as a lightweight yet powerful tool for Bangla NLP, particularly in both resource-constrained and scalable environments.
>
---
#### [new 056] Stepwise Reasoning Checkpoint Analysis: A Test Time Scaling Method to Enhance LLMs' Reasoning
- **分类: cs.CL**

- **简介: 该论文提出Stepwise Reasoning Checkpoint Analysis（SRCA），针对LLMs测试时间扩展中的路径同质化和中间结果利用率低问题，在推理步骤间设置检查点，通过答案聚类搜索和检查点候选增强策略，提升数学推理准确性。**

- **链接: [http://arxiv.org/pdf/2505.17829v1](http://arxiv.org/pdf/2505.17829v1)**

> **作者:** Zezhong Wang; Xingshan Zeng; Weiwen Liu; Yufei Wang; Liangyou Li; Yasheng Wang; Lifeng Shang; Xin Jiang; Qun Liu; Kam-Fai Wong
>
> **摘要:** Mathematical reasoning through Chain-of-Thought (CoT) has emerged as a powerful capability of Large Language Models (LLMs), which can be further enhanced through Test-Time Scaling (TTS) methods like Beam Search and DVTS. However, these methods, despite improving accuracy by allocating more computational resources during inference, often suffer from path homogenization and inefficient use of intermediate results. To address these limitations, we propose Stepwise Reasoning Checkpoint Analysis (SRCA), a framework that introduces checkpoints between reasoning steps. It incorporates two key strategies: (1) Answer-Clustered Search, which groups reasoning paths by their intermediate checkpoint answers to maintain diversity while ensuring quality, and (2) Checkpoint Candidate Augmentation, which leverages all intermediate answers for final decision-making. Our approach effectively reduces path homogenization and creates a fault-tolerant mechanism by utilizing high-quality intermediate results. Experimental results show that SRCA improves reasoning accuracy compared to existing TTS methods across various mathematical datasets.
>
---
#### [new 057] Harry Potter is Still Here! Probing Knowledge Leakage in Targeted Unlearned Large Language Models via Automated Adversarial Prompting
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于模型卸载评估任务，针对现有标准无法检测对抗条件下的知识泄露问题，提出LURK框架：通过自动化生成对抗性后缀提示，探测未学习模型中残留的哈利波特领域知识，揭示“成功卸载”模型仍可能泄露特定信息，为评估卸载算法鲁棒性提供更严格的工具。**

- **链接: [http://arxiv.org/pdf/2505.17160v1](http://arxiv.org/pdf/2505.17160v1)**

> **作者:** Bang Trinh Tran To; Thai Le
>
> **摘要:** This work presents LURK (Latent UnleaRned Knowledge), a novel framework that probes for hidden retained knowledge in unlearned LLMs through adversarial suffix prompting. LURK automatically generates adversarial prompt suffixes designed to elicit residual knowledge about the Harry Potter domain, a commonly used benchmark for unlearning. Our experiments reveal that even models deemed successfully unlearned can leak idiosyncratic information under targeted adversarial conditions, highlighting critical limitations of current unlearning evaluation standards. By uncovering latent knowledge through indirect probing, LURK offers a more rigorous and diagnostic tool for assessing the robustness of unlearning algorithms. All code will be publicly available.
>
---
#### [new 058] Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation
- **分类: cs.CL**

- **简介: 该论文属于EEG-to-text生成任务，旨在解决生成文本与脑信号语义不匹配的问题。针对EEG与文本信息容量差异，提出GLIM模型，通过学习可解释的EEG表征提升语义关联性，并采用EEG-文本检索和零样本分类等评估方法，验证了生成结果的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.17099v1](http://arxiv.org/pdf/2505.17099v1)**

> **作者:** Xiaozhao Liu; Dinggang Shen; Xihui Liu
>
> **备注:** Code, checkpoint and text samples available at https://github.com/justin-xzliu/GLIM
>
> **摘要:** Pretrained generative models have opened new frontiers in brain decoding by enabling the synthesis of realistic texts and images from non-invasive brain recordings. However, the reliability of such outputs remains questionable--whether they truly reflect semantic activation in the brain, or are merely hallucinated by the powerful generative models. In this paper, we focus on EEG-to-text decoding and address its hallucination issue through the lens of posterior collapse. Acknowledging the underlying mismatch in information capacity between EEG and text, we reframe the decoding task as semantic summarization of core meanings rather than previously verbatim reconstruction of stimulus texts. To this end, we propose the Generative Language Inspection Model (GLIM), which emphasizes learning informative and interpretable EEG representations to improve semantic grounding under heterogeneous and small-scale data conditions. Experiments on the public ZuCo dataset demonstrate that GLIM consistently generates fluent, EEG-grounded sentences without teacher forcing. Moreover, it supports more robust evaluation beyond text similarity, through EEG-text retrieval and zero-shot semantic classification across sentiment categories, relation types, and corpus topics. Together, our architecture and evaluation protocols lay the foundation for reliable and scalable benchmarking in generative brain decoding.
>
---
#### [new 059] MARCO: Meta-Reflection with Cross-Referencing for Code Reasoning
- **分类: cs.CL**

- **简介: 该论文属于代码推理任务，旨在解决现有LLMs静态推理无法累积提升的问题。提出MARCO框架，通过元反思积累知识、交叉引用借鉴其他代理经验，使模型动态进化，实现实时自我改进，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.17481v1](http://arxiv.org/pdf/2505.17481v1)**

> **作者:** Yusheng Zhao; Xiao Luo; Weizhi Zhang; Wei Ju; Zhiping Xiao; Philip S. Yu; Ming Zhang
>
> **摘要:** The ability to reason is one of the most fundamental capabilities of large language models (LLMs), enabling a wide range of downstream tasks through sophisticated problem-solving. A critical aspect of this is code reasoning, which involves logical reasoning with formal languages (i.e., programming code). In this paper, we enhance this capability of LLMs by exploring the following question: how can an LLM agent become progressively smarter in code reasoning with each solution it proposes, thereby achieving substantial cumulative improvement? Most existing research takes a static perspective, focusing on isolated problem-solving using frozen LLMs. In contrast, we adopt a cognitive-evolving perspective and propose a novel framework named Meta-Reflection with Cross-Referencing (MARCO) that enables the LLM to evolve dynamically during inference through self-improvement. From the perspective of human cognitive development, we leverage both knowledge accumulation and lesson sharing. In particular, to accumulate knowledge during problem-solving, we propose meta-reflection that reflects on the reasoning paths of the current problem to obtain knowledge and experience for future consideration. Moreover, to effectively utilize the lessons from other agents, we propose cross-referencing that incorporates the solution and feedback from other agents into the current problem-solving process. We conduct experiments across various datasets in code reasoning, and the results demonstrate the effectiveness of MARCO.
>
---
#### [new 060] Frankentext: Stitching random text fragments into long-form narratives
- **分类: cs.CL**

- **简介: 该论文提出Frankentext任务，要求LLM在90%内容直接复制人类文本的约束下生成连贯长文，解决高复制率下的可控生成与内容整合难题。通过迭代修改草稿维持复制比例，评估生成内容质量、相关性及仿人性。实验显示Gemini-2.5-Pro表现优异，但存在可检测缺陷，推动混合作者检测研究与人机协作分析。**

- **链接: [http://arxiv.org/pdf/2505.18128v1](http://arxiv.org/pdf/2505.18128v1)**

> **作者:** Chau Minh Pham; Jenna Russell; Dzung Pham; Mohit Iyyer
>
> **摘要:** We introduce Frankentexts, a new type of long-form narratives produced by LLMs under the extreme constraint that most tokens (e.g., 90%) must be copied verbatim from human writings. This task presents a challenging test of controllable generation, requiring models to satisfy a writing prompt, integrate disparate text fragments, and still produce a coherent narrative. To generate Frankentexts, we instruct the model to produce a draft by selecting and combining human-written passages, then iteratively revise the draft while maintaining a user-specified copy ratio. We evaluate the resulting Frankentexts along three axes: writing quality, instruction adherence, and detectability. Gemini-2.5-Pro performs surprisingly well on this task: 81% of its Frankentexts are coherent and 100% relevant to the prompt. Notably, up to 59% of these outputs are misclassified as human-written by detectors like Pangram, revealing limitations in AI text detectors. Human annotators can sometimes identify Frankentexts through their abrupt tone shifts and inconsistent grammar between segments, especially in longer generations. Beyond presenting a challenging generation task, Frankentexts invite discussion on building effective detectors for this new grey zone of authorship, provide training data for mixed authorship detection, and serve as a sandbox for studying human-AI co-writing processes.
>
---
#### [new 061] GIM: Improved Interpretability for Large Language Models
- **分类: cs.CL; cs.LG; 68T07; I.2.0; I.2.7**

- **简介: 该论文属于大语言模型（LLM）可解释性研究，针对self-repair现象导致传统方法低估组件重要性问题，提出GIM方法修正注意力机制中softmax重新分配的干扰，通过改进反向传播提升特征归因忠实度，实验验证其效果优于现有方法，促进LLM安全与优化。**

- **链接: [http://arxiv.org/pdf/2505.17630v1](http://arxiv.org/pdf/2505.17630v1)**

> **作者:** Joakim Edin; Róbert Csordás; Tuukka Ruotsalo; Zhengxuan Wu; Maria Maistro; Jing Huang; Lars Maaløe
>
> **摘要:** Ensuring faithful interpretability in large language models is imperative for trustworthy and reliable AI. A key obstacle is self-repair, a phenomenon where networks compensate for reduced signal in one component by amplifying others, masking the true importance of the ablated component. While prior work attributes self-repair to layer normalization and back-up components that compensate for ablated components, we identify a novel form occurring within the attention mechanism, where softmax redistribution conceals the influence of important attention scores. This leads traditional ablation and gradient-based methods to underestimate the significance of all components contributing to these attention scores. We introduce Gradient Interaction Modifications (GIM), a technique that accounts for self-repair during backpropagation. Extensive experiments across multiple large language models (Gemma 2B/9B, LLAMA 1B/3B/8B, Qwen 1.5B/3B) and diverse tasks demonstrate that GIM significantly improves faithfulness over existing circuit identification and feature attribution methods. Our work is a significant step toward better understanding the inner mechanisms of LLMs, which is crucial for improving them and ensuring their safety. Our code is available at https://github.com/JoakimEdin/gim.
>
---
#### [new 062] Relative Bias: A Comparative Framework for Quantifying Bias in LLMs
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 论文提出Relative Bias框架，通过Embedding Transformation分析和LLM-as-a-Judge方法，量化LLMs在特定领域的偏差差异，经案例研究和统计检验验证，提供系统、可扩展的比较评估方案。任务为模型偏差量化，解决多模型间偏差系统评估难题。**

- **链接: [http://arxiv.org/pdf/2505.17131v1](http://arxiv.org/pdf/2505.17131v1)**

> **作者:** Alireza Arbabi; Florian Kerschbaum
>
> **摘要:** The growing deployment of large language models (LLMs) has amplified concerns regarding their inherent biases, raising critical questions about their fairness, safety, and societal impact. However, quantifying LLM bias remains a fundamental challenge, complicated by the ambiguity of what "bias" entails. This challenge grows as new models emerge rapidly and gain widespread use, while introducing potential biases that have not been systematically assessed. In this paper, we propose the Relative Bias framework, a method designed to assess how an LLM's behavior deviates from other LLMs within a specified target domain. We introduce two complementary methodologies: (1) Embedding Transformation analysis, which captures relative bias patterns through sentence representations over the embedding space, and (2) LLM-as-a-Judge, which employs a language model to evaluate outputs comparatively. Applying our framework to several case studies on bias and alignment scenarios following by statistical tests for validation, we find strong alignment between the two scoring methods, offering a systematic, scalable, and statistically grounded approach for comparative bias analysis in LLMs.
>
---
#### [new 063] Conformal Language Model Reasoning with Coherent Factuality
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于语言模型推理任务，解决现有事实性评估无法处理推理步骤间依赖的问题。提出"连贯事实性"概念，基于可推导图和分块符合预测方法，确保推理步骤的上下文正确性。在数学数据集上验证，实现90%事实性同时保留80%原始主张。**

- **链接: [http://arxiv.org/pdf/2505.17126v1](http://arxiv.org/pdf/2505.17126v1)**

> **作者:** Maxon Rubin-Toles; Maya Gambhir; Keshav Ramji; Aaron Roth; Surbhi Goel
>
> **摘要:** Language models are increasingly being used in important decision pipelines, so ensuring the correctness of their outputs is crucial. Recent work has proposed evaluating the "factuality" of claims decomposed from a language model generation and applying conformal prediction techniques to filter out those claims that are not factual. This can be effective for tasks such as information retrieval, where constituent claims may be evaluated in isolation for factuality, but is not appropriate for reasoning tasks, as steps of a logical argument can be evaluated for correctness only within the context of the claims that precede them. To capture this, we define "coherent factuality" and develop a conformal-prediction-based method to guarantee coherent factuality for language model outputs. Our approach applies split conformal prediction to subgraphs within a "deducibility" graph" that represents the steps of a reasoning problem. We evaluate our method on mathematical reasoning problems from the MATH and FELM datasets and find that our algorithm consistently produces correct and substantiated orderings of claims, achieving coherent factuality across target coverage levels. Moreover, we achieve 90% factuality on our stricter definition while retaining 80% or more of the original claims, highlighting the utility of our deducibility-graph-guided approach.
>
---
#### [new 064] A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit
- **分类: cs.CL; cs.AI**

- **简介: 该论文开发基于LLM的动机访谈聊天机器人，旨在通过自动化疗法帮助吸烟者戒烟。解决的问题包括自动化咨询的有效性及对治疗标准的遵循。研究构建了结合MI疗法的聊天机器人，经临床专家验证，测试显示用户戒烟信心提升，系统98%符合MI标准，但同理心评分略低于人类咨询师。**

- **链接: [http://arxiv.org/pdf/2505.17362v1](http://arxiv.org/pdf/2505.17362v1)**

> **作者:** Zafarullah Mahmood; Soliman Ali; Jiading Zhu; Mohamed Abdelwahab; Michelle Yu Collins; Sihan Chen; Yi Cheng Zhao; Jodi Wolff; Osnat Melamed; Nadia Minian; Marta Maslej; Carolynne Cooper; Matt Ratto; Peter Selby; Jonathan Rose
>
> **备注:** To be published in the Findings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Vienna, Austria, 2025
>
> **摘要:** The conversational capabilities of Large Language Models (LLMs) suggest that they may be able to perform as automated talk therapists. It is crucial to know if these systems would be effective and adhere to known standards. We present a counsellor chatbot that focuses on motivating tobacco smokers to quit smoking. It uses a state-of-the-art LLM and a widely applied therapeutic approach called Motivational Interviewing (MI), and was evolved in collaboration with clinician-scientists with expertise in MI. We also describe and validate an automated assessment of both the chatbot's adherence to MI and client responses. The chatbot was tested on 106 participants, and their confidence that they could succeed in quitting smoking was measured before the conversation and one week later. Participants' confidence increased by an average of 1.7 on a 0-10 scale. The automated assessment of the chatbot showed adherence to MI standards in 98% of utterances, higher than human counsellors. The chatbot scored well on a participant-reported metric of perceived empathy but lower than typical human counsellors. Furthermore, participants' language indicated a good level of motivation to change, a key goal in MI. These results suggest that the automation of talk therapy with a modern LLM has promise.
>
---
#### [new 065] Enhancing Large Vision-Language Models with Layout Modality for Table Question Answering on Japanese Annual Securities Reports
- **分类: cs.CL; cs.CV; 68T50; I.2**

- **简介: 该论文针对日文证券报告表格问答任务，提出通过融合表格文本与布局信息增强视觉语言模型，解决现有模型在解析表格字符及空间关系上的不足，实验表明该方法有效提升复杂布局下的理解能力。**

- **链接: [http://arxiv.org/pdf/2505.17625v1](http://arxiv.org/pdf/2505.17625v1)**

> **作者:** Hayato Aida; Kosuke Takahashi; Takahiro Omi
>
> **备注:** Accepted at IIAI AAI 2025, the 3rd International Conference on Computational and Data Sciences in Economics and Finance
>
> **摘要:** With recent advancements in Large Language Models (LLMs) and growing interest in retrieval-augmented generation (RAG), the ability to understand table structures has become increasingly important. This is especially critical in financial domains such as securities reports, where highly accurate question answering (QA) over tables is required. However, tables exist in various formats-including HTML, images, and plain text-making it difficult to preserve and extract structural information. Therefore, multimodal LLMs are essential for robust and general-purpose table understanding. Despite their promise, current Large Vision-Language Models (LVLMs), which are major representatives of multimodal LLMs, still face challenges in accurately understanding characters and their spatial relationships within documents. In this study, we propose a method to enhance LVLM-based table understanding by incorporating in-table textual content and layout features. Experimental results demonstrate that these auxiliary modalities significantly improve performance, enabling robust interpretation of complex document layouts without relying on explicitly structured input formats.
>
---
#### [new 066] SemSketches-2021: experimenting with the machine processing of the pilot semantic sketches corpus
- **分类: cs.CL**

- **简介: 该论文属于SemSketches-2021共享任务，聚焦语义草图语料库的机器处理。旨在解决语义理解与自动标注问题，构建开放语料库并组织竞赛，要求参与者将合适上下文匹配至对应草图，同时开发相关处理工具。**

- **链接: [http://arxiv.org/pdf/2505.17704v1](http://arxiv.org/pdf/2505.17704v1)**

> **作者:** Maria Ponomareva; Maria Petrova; Julia Detkova; Oleg Serikov; Maria Yarova
>
> **摘要:** The paper deals with elaborating different approaches to the machine processing of semantic sketches. It presents the pilot open corpus of semantic sketches. Different aspects of creating the sketches are discussed, as well as the tasks that the sketches can help to solve. Special attention is paid to the creation of the machine processing tools for the corpus. For this purpose, the SemSketches-2021 Shared Task was organized. The participants were given the anonymous sketches and a set of contexts containing the necessary predicates. During the Task, one had to assign the proper contexts to the corresponding sketches.
>
---
#### [new 067] DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DialogXpert，针对LLM在目标驱动对话中的短视解码和高规划成本问题，利用冻结LLM生成候选动作，结合Q网络与BERT嵌入选择最优回复，并融入情感追踪，实现实时高效对话规划，在多任务中成功率超94%。属于对话系统任务，解决LLM的主动交互不足。**

- **链接: [http://arxiv.org/pdf/2505.17795v1](http://arxiv.org/pdf/2505.17795v1)**

> **作者:** Tazeek Bin Abdur Rakib; Ambuj Mehrish; Lay-Ki Soon; Wern Han Lim; Soujanya Poria
>
> **摘要:** Large-language-model (LLM) agents excel at reactive dialogue but struggle with proactive, goal-driven interactions due to myopic decoding and costly planning. We introduce DialogXpert, which leverages a frozen LLM to propose a small, high-quality set of candidate actions per turn and employs a compact Q-network over fixed BERT embeddings trained via temporal-difference learning to select optimal moves within this reduced space. By tracking the user's emotions, DialogXpert tailors each decision to advance the task while nurturing a genuine, empathetic connection. Across negotiation, emotional support, and tutoring benchmarks, DialogXpert drives conversations to under $3$ turns with success rates exceeding 94\% and, with a larger LLM prior, pushes success above 97\% while markedly improving negotiation outcomes. This framework delivers real-time, strategic, and emotionally intelligent dialogue planning at scale. Code available at https://github.com/declare-lab/dialogxpert/
>
---
#### [new 068] Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于认知衰退检测任务，旨在通过分析语音助手的长期语音指令数据，非侵入性替代传统临床评估。提出Cog-TiPRO框架，结合LLM迭代优化提示提取语言特征、HuBERT声学分析及时间建模，检测MCI准确率达73.8%，显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.17137v1](http://arxiv.org/pdf/2505.17137v1)**

> **作者:** Kristin Qi; Youxiang Zhu; Caroline Summerour; John A. Batsis; Xiaohui Liang
>
> **备注:** Submitted to the IEEE GlobeCom 2025
>
> **摘要:** Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
>
---
#### [new 069] Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出无训练激活控制技术，通过放大LLM末层关键激活并插入"等待"标记，增强长链推理能力，减少对昂贵训练的依赖；同时开发参数高效微调方法，仅训练少量模块即提升推理性能。任务为优化LLM的长链思维，解决现有方法依赖大量数据/算力的问题。**

- **链接: [http://arxiv.org/pdf/2505.17697v1](http://arxiv.org/pdf/2505.17697v1)**

> **作者:** Zekai Zhao; Qi Liu; Kun Zhou; Zihan Liu; Yifei Shao; Zhiting Hu; Biwei Huang
>
> **摘要:** Despite the remarkable reasoning performance, eliciting the long chain-of-thought (CoT) ability in large language models (LLMs) typically requires costly reinforcement learning or supervised fine-tuning on high-quality distilled data. We investigate the internal mechanisms behind this capability and show that a small set of high-impact activations in the last few layers largely governs long-form reasoning attributes, such as output length and self-reflection. By simply amplifying these activations and inserting "wait" tokens, we can invoke the long CoT ability without any training, resulting in significantly increased self-reflection rates and accuracy. Moreover, we find that the activation dynamics follow predictable trajectories, with a sharp rise after special tokens and a subsequent exponential decay. Building on these insights, we introduce a general training-free activation control technique. It leverages a few contrastive examples to identify key activations, and employs simple analytic functions to modulate their values at inference time to elicit long CoTs. Extensive experiments confirm the effectiveness of our method in efficiently eliciting long CoT reasoning in LLMs and improving their performance. Additionally, we propose a parameter-efficient fine-tuning method that trains only a last-layer activation amplification module and a few LoRA layers, outperforming full LoRA fine-tuning on reasoning benchmarks with significantly fewer parameters. Our code and data are publicly released.
>
---
#### [new 070] Not All Tokens Are What You Need In Thinking
- **分类: cs.CL**

- **简介: 该论文属于优化大模型推理效率任务，针对现有模型过度思考生成冗余token的问题，提出CTS框架通过条件重要性评分筛选关键token压缩思维链，实验显示其减少13-42%训练token的同时提升或保持推理性能。**

- **链接: [http://arxiv.org/pdf/2505.17827v1](http://arxiv.org/pdf/2505.17827v1)**

> **作者:** Hang Yuan; Bin Yu; Haotian Li; Shijun Yang; Christina Dan Wang; Zhou Yu; Xueyin Xu; Weizhen Qi; Kai Chen
>
> **备注:** 11 pages, 7 figures and 3 tables
>
> **摘要:** Modern reasoning models, such as OpenAI's o1 and DeepSeek-R1, exhibit impressive problem-solving capabilities but suffer from critical inefficiencies: high inference latency, excessive computational resource consumption, and a tendency toward overthinking -- generating verbose chains of thought (CoT) laden with redundant tokens that contribute minimally to the final answer. To address these issues, we propose Conditional Token Selection (CTS), a token-level compression framework with a flexible and variable compression ratio that identifies and preserves only the most essential tokens in CoT. CTS evaluates each token's contribution to deriving correct answers using conditional importance scoring, then trains models on compressed CoT. Extensive experiments demonstrate that CTS effectively compresses long CoT while maintaining strong reasoning performance. Notably, on the GPQA benchmark, Qwen2.5-14B-Instruct trained with CTS achieves a 9.1% accuracy improvement with 13.2% fewer reasoning tokens (13% training token reduction). Further reducing training tokens by 42% incurs only a marginal 5% accuracy drop while yielding a 75.8% reduction in reasoning tokens, highlighting the prevalence of redundancy in existing CoT.
>
---
#### [new 071] Handling Symbolic Language in Student Texts: A Comparative Study of NLP Embedding Models
- **分类: cs.CL; cs.AI; physics.ed-ph**

- **简介: 该论文属于NLP模型比较任务，旨在解决科学文本中符号表达（如公式）处理难题。通过评估多种嵌入模型在物理学科学生文本中的表现（相似度分析和机器学习集成），发现GPT-text-embedding-3-large最优但优势有限，强调模型选择需综合性能、成本及合规性等因素。**

- **链接: [http://arxiv.org/pdf/2505.17950v1](http://arxiv.org/pdf/2505.17950v1)**

> **作者:** Tom Bleckmann; Paul Tschisgale
>
> **摘要:** Recent advancements in Natural Language Processing (NLP) have facilitated the analysis of student-generated language products in learning analytics (LA), particularly through the use of NLP embedding models. Yet when it comes to science-related language, symbolic expressions such as equations and formulas introduce challenges that current embedding models struggle to address. Existing studies and applications often either overlook these challenges or remove symbolic expressions altogether, potentially leading to biased findings and diminished performance of LA applications. This study therefore explores how contemporary embedding models differ in their capability to process and interpret science-related symbolic expressions. To this end, various embedding models are evaluated using physics-specific symbolic expressions drawn from authentic student responses, with performance assessed via two approaches: similarity-based analyses and integration into a machine learning pipeline. Our findings reveal significant differences in model performance, with OpenAI's GPT-text-embedding-3-large outperforming all other examined models, though its advantage over other models was moderate rather than decisive. Beyond performance, additional factors such as cost, regulatory compliance, and model transparency are discussed as key considerations for model selection. Overall, this study underscores the importance for LA researchers and practitioners of carefully selecting NLP embedding models when working with science-related language products that include symbolic expressions.
>
---
#### [new 072] Large Language Models Do Multi-Label Classification Differently
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）在多标签分类任务（尤其主观任务）中的行为差异。发现LLMs在生成时每步抑制多余标签，模型规模增大及微调导致分布熵降低但标签排序优化。提出分布对齐任务及零样本/监督方法提升模型输出与标注者分布的匹配度及预测效果。**

- **链接: [http://arxiv.org/pdf/2505.17510v1](http://arxiv.org/pdf/2505.17510v1)**

> **作者:** Marcus Ma; Georgios Chochlakis; Niyantha Maruthu Pandiyan; Jesse Thomason; Shrikanth Narayanan
>
> **备注:** 18 pages, 11 figures, 6 tables
>
> **摘要:** Multi-label classification is prevalent in real-world settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive LLMs perform multi-label classification, with a focus on subjective tasks, by analyzing the output distributions of the models in each generation step. We find that their predictive behavior reflects the multiple steps in the underlying language modeling required to generate all relevant labels as they tend to suppress all but one label at each step. We further observe that as model scale increases, their token distributions exhibit lower entropy, yet the internal ranking of the labels improves. Finetuning methods such as supervised finetuning and reinforcement learning amplify this phenomenon. To further study this issue, we introduce the task of distribution alignment for multi-label settings: aligning LLM-derived label distributions with empirical distributions estimated from annotator responses in subjective tasks. We propose both zero-shot and supervised methods which improve both alignment and predictive performance over existing approaches.
>
---
#### [new 073] Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对Agentic RAG系统中的过度/不足搜索问题，提出β-GRPO方法。通过强化学习与信心阈值减少模型不确定性，优化搜索决策。实验显示其在7个QA数据集上提升4%准确率，有效提升系统效率与可靠性。**

- **链接: [http://arxiv.org/pdf/2505.17281v1](http://arxiv.org/pdf/2505.17281v1)**

> **作者:** Peilin Wu; Mian Zhang; Xinlu Zhang; Xinya Du; Zhiyu Zoey Chen
>
> **摘要:** Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score.
>
---
#### [new 074] ExeSQL: Self-Taught Text-to-SQL Models with Execution-Driven Bootstrapping for SQL Dialects
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文提出ExeSQL框架，解决跨SQL方言文本到SQL生成中的数据不足与执行验证缺失问题。通过执行驱动的迭代生成、过滤及偏好训练，模型自适应不同方言。实验显示在PostgreSQL、MySQL、Oracle上分别提升15.2%、10.38%、4.49%。**

- **链接: [http://arxiv.org/pdf/2505.17231v1](http://arxiv.org/pdf/2505.17231v1)**

> **作者:** Jipeng Zhang; Haolin Yang; Kehao Miao; Ruiyuan Zhang; Renjie Pi; Jiahui Gao; Xiaofang Zhou
>
> **摘要:** Recent text-to-SQL models have achieved strong performance, but their effectiveness remains largely confined to SQLite due to dataset limitations. However, real-world applications require SQL generation across multiple dialects with varying syntax and specialized features, which remains a challenge for current models. The main obstacle in building a dialect-aware model lies in acquiring high-quality dialect-specific data. Data generated purely through static prompting - without validating SQLs via execution - tends to be noisy and unreliable. Moreover, the lack of real execution environments in the training loop prevents models from grounding their predictions in executable semantics, limiting generalization despite surface-level improvements from data filtering. This work introduces ExeSQL, a text-to-SQL framework with execution-driven, agentic bootstrapping. The method consists of iterative query generation, execution-based filtering (e.g., rejection sampling), and preference-based training, enabling the model to adapt to new SQL dialects through verifiable, feedback-guided learning. Experiments show that ExeSQL bridges the dialect gap in text-to-SQL, achieving average improvements of 15.2%, 10.38%, and 4.49% over GPT-4o on PostgreSQL, MySQL, and Oracle, respectively, across multiple datasets of varying difficulty.
>
---
#### [new 075] Informatics for Food Processing
- **分类: cs.CL; cs.AI; cs.CY; cs.DB; cs.LG**

- **简介: 论文提出基于AI的食品加工评估方法，解决传统分类（如NOVA）主观性及数据缺失问题。采用FoodProX模型分析营养数据生成加工水平评分，结合BERT处理文本数据，并利用Open Food Facts数据库验证多模态AI分类，推动公共健康研究。**

- **链接: [http://arxiv.org/pdf/2505.17087v1](http://arxiv.org/pdf/2505.17087v1)**

> **作者:** Gordana Ispirova; Michael Sebek; Giulia Menichetti
>
> **摘要:** This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research.
>
---
#### [new 076] UNJOIN: Enhancing Multi-Table Text-to-SQL Generation via Schema Simplification
- **分类: cs.CL**

- **简介: 该论文属于多表Text-to-SQL任务，解决复杂数据库中表/列检索、JOIN/UNION生成及泛化难题。提出UNJOIN框架，分两阶段：首阶段将多表列名合并为单表表示（表名前缀），简化检索；次阶段生成SQL并映射回原模式。实验显示其效果达SOTA，且无需数据或微调。**

- **链接: [http://arxiv.org/pdf/2505.18122v1](http://arxiv.org/pdf/2505.18122v1)**

> **作者:** Poojah Ganesan; Rajat Aayush Jha; Dan Roth; Vivek Gupta
>
> **摘要:** Recent advances in large language models (LLMs) have greatly improved Text-to-SQL performance for single-table queries. But, it remains challenging in multi-table databases due to complex schema and relational operations. Existing methods often struggle with retrieving the right tables and columns, generating accurate JOINs and UNIONs, and generalizing across diverse schemas. To address these issues, we introduce UNJOIN, a two-stage framework that decouples the retrieval of schema elements from SQL logic generation. In the first stage, we merge the column names of all tables in the database into a single-table representation by prefixing each column with its table name. This allows the model to focus purely on accurate retrieval without being distracted by the need to write complex SQL logic. In the second stage, the SQL query is generated on this simplified schema and mapped back to the original schema by reconstructing JOINs, UNIONs, and relational logic. Evaluations on SPIDER and BIRD datasets show that UNJOIN matches or exceeds the state-of-the-art baselines. UNJOIN uses only schema information, which does not require data access or fine-tuning, making it scalable and adaptable across databases.
>
---
#### [new 077] Comparative Evaluation of Prompting and Fine-Tuning for Applying Large Language Models to Grid-Structured Geospatial Data
- **分类: cs.CL; cs.ET**

- **简介: 该论文比较了提示工程与微调方法在LLMs处理网格结构地理空间数据中的应用，旨在解决模型在结构化时空推理中的有效性问题。通过对比基础模型的结构化提示与用户交互数据微调模型的性能，揭示了零次提示的局限性及微调方法的优势。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17116v1](http://arxiv.org/pdf/2505.17116v1)**

> **作者:** Akash Dhruv; Yangxinyu Xie; Jordan Branham; Tanwi Mallick
>
> **摘要:** This paper presents a comparative study of large language models (LLMs) in interpreting grid-structured geospatial data. We evaluate the performance of a base model through structured prompting and contrast it with a fine-tuned variant trained on a dataset of user-assistant interactions. Our results highlight the strengths and limitations of zero-shot prompting and demonstrate the benefits of fine-tuning for structured geospatial and temporal reasoning.
>
---
#### [new 078] SLearnLLM: A Self-Learning Framework for Efficient Domain-Specific Adaptation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SLearnLLM框架，解决领域适配中全量SFT数据低效问题。通过模型自评估筛选未知QA对微调，减少冗余训练。实验显示其在医疗/农业领域显著缩短训练时间且效果与全量训练相当。**

- **链接: [http://arxiv.org/pdf/2505.17470v1](http://arxiv.org/pdf/2505.17470v1)**

> **作者:** Xiang Liu; Zhaoxiang Liu; Peng Wang; Kohou Wang; Huan Hu; Kai Wang; Shiguo Lian
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** When using supervised fine-tuning (SFT) to adapt large language models (LLMs) to specific domains, a significant challenge arises: should we use the entire SFT dataset for fine-tuning? Common practice often involves fine-tuning directly on the entire dataset due to limited information on the LLM's past training data. However, if the SFT dataset largely overlaps with the model's existing knowledge, the performance gains are minimal, leading to wasted computational resources. Identifying the unknown knowledge within the SFT dataset and using it to fine-tune the model could substantially improve the training efficiency. To address this challenge, we propose a self-learning framework for LLMs inspired by human learning pattern. This framework takes a fine-tuning (SFT) dataset in a specific domain as input. First, the LLMs answer the questions in the SFT dataset. The LLMs then objectively grade the responses and filter out the incorrectly answered QA pairs. Finally, we fine-tune the LLMs based on this filtered QA set. Experimental results in the fields of agriculture and medicine demonstrate that our method substantially reduces training time while achieving comparable improvements to those attained with full dataset fine-tuning. By concentrating on the unknown knowledge within the SFT dataset, our approach enhances the efficiency of fine-tuning LLMs.
>
---
#### [new 079] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **简介: 该论文对比LLMs与人类在概念压缩与语义保真间的权衡。任务：揭示二者认知差异。问题：LLMs是否平衡压缩效率与语义精度。方法：基于信息瓶颈理论，分析模型嵌入与人类分类数据。发现：LLMs倾向过度压缩，而人类优先语境细节，指导开发更人性化的LLM表示机制。**

- **链接: [http://arxiv.org/pdf/2505.17117v1](http://arxiv.org/pdf/2505.17117v1)**

> **作者:** Chen Shani; Dan Jurafsky; Yann LeCun; Ravid Shwartz-Ziv
>
> **摘要:** Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
>
---
#### [new 080] An approach to identify the most semantically informative deep representations of text and images
- **分类: cs.CL; cs.LG; physics.comp-ph**

- **简介: 该论文研究跨模态语义信息在深度模型中的表示机制。任务是识别文本与图像间最具语义关联的深层表示。通过分析LLMs和视觉Transformer，量化跨语言文本、图文对的语义层信息分布，发现大模型更具普适性，并揭示语义信息的长距关联、因果不对称及跨模态信息传递差异。**

- **链接: [http://arxiv.org/pdf/2505.17101v1](http://arxiv.org/pdf/2505.17101v1)**

> **作者:** Santiago Acevedo; Andrea Mascaretti; Riccardo Rende; Matéo Mahaut; Marco Baroni; Alessandro Laio
>
> **摘要:** Deep neural networks are known to develop similar representations for semantically related data, even when they belong to different domains, such as an image and its description, or the same text in different languages. We present a method for quantitatively investigating this phenomenon by measuring the relative information content of the representations of semantically related data and probing how it is encoded into multiple tokens of large language models (LLMs) and vision transformers. Looking first at how LLMs process pairs of translated sentences, we identify inner ``semantic'' layers containing the most language-transferable information. We find moreover that, on these layers, a larger LLM (DeepSeek-V3) extracts significantly more general information than a smaller one (Llama3.1-8B). Semantic information is spread across many tokens and it is characterized by long-distance correlations between tokens and by a causal left-to-right (i.e., past-future) asymmetry. We also identify layers encoding semantic information within visual transformers. We show that caption representations in the semantic layers of LLMs predict visual representations of the corresponding images. We observe significant and model-dependent information asymmetries between image and text representations.
>
---
#### [new 081] The Pilot Corpus of the English Semantic Sketches
- **分类: cs.CL**

- **简介: 该论文属于语料库语言学任务，旨在构建英语动词的语义草图试点语料库（英俄配对），解决跨语言语义对比分析中的差异问题。通过创建语义草图对、分析相似语义下英俄动词的表达差异，并探讨构建方法及错误类型，揭示语义草图的 linguistics 性质。**

- **链接: [http://arxiv.org/pdf/2505.17733v1](http://arxiv.org/pdf/2505.17733v1)**

> **作者:** Maria Petrova; Maria Ponomareva; Alexandra Ivoylova
>
> **摘要:** The paper is devoted to the creation of the semantic sketches for English verbs. The pilot corpus consists of the English-Russian sketch pairs and is aimed to show what kind of contrastive studies the sketches help to conduct. Special attention is paid to the cross-language differences between the sketches with similar semantics. Moreover, we discuss the process of building a semantic sketch, and analyse the mistakes that could give insight to the linguistic nature of sketches.
>
---
#### [new 082] SweEval: Do LLMs Really Swear? A Safety Benchmark for Testing Limits for Enterprise Use
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; I.2.7; I.2.6**

- **简介: 该论文提出SweEval基准，评估LLMs在企业场景中抵御不当指令的能力。针对企业应用中需避免生成冒犯性语言的问题，通过设计含明确脏话指令的正式/非正式场景测试模型是否合规响应，并公开数据集促进伦理AI研究。**

- **链接: [http://arxiv.org/pdf/2505.17332v1](http://arxiv.org/pdf/2505.17332v1)**

> **作者:** Hitesh Laxmichand Patel; Amit Agarwal; Arion Das; Bhargava Kumar; Srikant Panda; Priyaranjan Pattnayak; Taki Hasan Rafi; Tejaswini Kumar; Dong-Kyu Chae
>
> **备注:** Published in the Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2025), Industry Track, pages 558-582
>
> **摘要:** Enterprise customers are increasingly adopting Large Language Models (LLMs) for critical communication tasks, such as drafting emails, crafting sales pitches, and composing casual messages. Deploying such models across different regions requires them to understand diverse cultural and linguistic contexts and generate safe and respectful responses. For enterprise applications, it is crucial to mitigate reputational risks, maintain trust, and ensure compliance by effectively identifying and handling unsafe or offensive language. To address this, we introduce SweEval, a benchmark simulating real-world scenarios with variations in tone (positive or negative) and context (formal or informal). The prompts explicitly instruct the model to include specific swear words while completing the task. This benchmark evaluates whether LLMs comply with or resist such inappropriate instructions and assesses their alignment with ethical frameworks, cultural nuances, and language comprehension capabilities. In order to advance research in building ethically aligned AI systems for enterprise use and beyond, we release the dataset and code: https://github.com/amitbcp/multilingual_profanity.
>
---
#### [new 083] DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inference with Markov Decision Policies
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出DASH框架，针对LLM推理延迟高的问题，通过输入感知的动态层跳越技术优化计算路径。基于MDP决策模型实现逐token计算选择，结合补偿机制和异步执行策略，在保证性能前提下加速推理，解决大模型部署效率瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.17420v1](http://arxiv.org/pdf/2505.17420v1)**

> **作者:** Ning Yang; Fangxin Liu; Junjie Wang; Tao Yang; Kan Liu; Haibing Guan; Li Jiang
>
> **备注:** 8 pages,5 figures
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance across a wide range of NLP tasks. However, their substantial inference cost poses a major barrier to real-world deployment, especially in latency-sensitive scenarios. To address this challenge, we propose \textbf{DASH}, an adaptive layer-skipping framework that dynamically selects computation paths conditioned on input characteristics. We model the skipping process as a Markov Decision Process (MDP), enabling fine-grained token-level decisions based on intermediate representations. To mitigate potential performance degradation caused by skipping, we introduce a lightweight compensation mechanism that injects differential rewards into the decision process. Furthermore, we design an asynchronous execution strategy that overlaps layer computation with policy evaluation to minimize runtime overhead. Experiments on multiple LLM architectures and NLP benchmarks show that our method achieves significant inference acceleration while maintaining competitive task performance, outperforming existing methods.
>
---
#### [new 084] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector
- **分类: cs.CL**

- **简介: 该论文提出Reasoning-based Bias Detector（RBD），旨在解决LLM作为评估器时的固有偏差问题。通过外部插件检测偏差并生成推理反馈引导模型自我修正，无需修改评估器本身。工作包括构建偏差数据集、开发RBD训练pipeline及验证其在多种LLM和偏差类型上的有效性，实验显示其显著提升评估准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2505.17100v1](http://arxiv.org/pdf/2505.17100v1)**

> **作者:** Haoyan Yang; Runxue Bao; Cao Xiao; Jun Ma; Parminder Bhatia; Shangqian Gao; Taha Kass-Hout
>
> **摘要:** LLM-as-a-Judge has emerged as a promising tool for automatically evaluating generated outputs, but its reliability is often undermined by potential biases in judgment. Existing efforts to mitigate these biases face key limitations: in-context learning-based methods fail to address rooted biases due to the evaluator's limited capacity for self-reflection, whereas fine-tuning is not applicable to all evaluator types, especially closed-source models. To address this challenge, we introduce the Reasoning-based Bias Detector (RBD), which is a plug-in module that identifies biased evaluations and generates structured reasoning to guide evaluator self-correction. Rather than modifying the evaluator itself, RBD operates externally and engages in an iterative process of bias detection and feedback-driven revision. To support its development, we design a complete pipeline consisting of biased dataset construction, supervision collection, distilled reasoning-based fine-tuning of RBD, and integration with LLM evaluators. We fine-tune four sizes of RBD models, ranging from 1.5B to 14B, and observe consistent performance improvements across all scales. Experimental results on 4 bias types--verbosity, position, bandwagon, and sentiment--evaluated using 8 LLM evaluators demonstrate RBD's strong effectiveness. For example, the RBD-8B model improves evaluation accuracy by an average of 18.5% and consistency by 10.9%, and surpasses prompting-based baselines and fine-tuned judges by 12.8% and 17.2%, respectively. These results highlight RBD's effectiveness and scalability. Additional experiments further demonstrate its strong generalization across biases and domains, as well as its efficiency.
>
---
#### [new 085] Fast Quiet-STaR: Thinking Without Thought Tokens
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于提升大模型推理效率的任务，针对Quiet STaR推理耗时高的问题，提出Fast Quiet-STaR框架：通过渐进式减少思考标记的课程学习，使模型内化简洁推理过程；进一步用强化学习适配标准NTP任务，消除显式生成思考标记的需求。实验显示其在相同延迟下提升准确率5.7%-9%。**

- **链接: [http://arxiv.org/pdf/2505.17746v1](http://arxiv.org/pdf/2505.17746v1)**

> **作者:** Wei Huang; Yizhe Xiong; Xin Ye; Zhijie Deng; Hui Chen; Zijia Lin; Guiguang Ding
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance across a range of natural language processing tasks. However, recent advances demonstrate that further gains particularly in complex reasoning tasks require more than merely scaling up model sizes or training data. One promising direction is to enable models to think during the reasoning process. Recently, Quiet STaR significantly improves reasoning by generating token-level thought traces, but incurs substantial inference overhead. In this work, we propose Fast Quiet STaR, a more efficient reasoning framework that preserves the benefits of token-level reasoning while reducing computational cost. Our method introduces a curriculum learning based training strategy that gradually reduces the number of thought tokens, enabling the model to internalize more abstract and concise reasoning processes. We further extend this approach to the standard Next Token Prediction (NTP) setting through reinforcement learning-based fine-tuning, resulting in Fast Quiet-STaR NTP, which eliminates the need for explicit thought token generation during inference. Experiments on four benchmark datasets with Mistral 7B and Qwen2.5 7B demonstrate that Fast Quiet-STaR consistently outperforms Quiet-STaR in terms of average accuracy under the same inference time budget. Notably, Fast Quiet-STaR NTP achieves an average accuracy improvement of 9\% on Mistral 7B and 5.7\% on Qwen2.5 7B, while maintaining the same inference latency. Our code will be available at https://github.com/huangwei200012/Fast-Quiet-STaR.
>
---
#### [new 086] Bayesian Optimization for Enhanced Language Models: Optimizing Acquisition Functions
- **分类: cs.CL; cs.AI**

- **简介: 论文属于大型语言模型微调的超参数优化任务。针对现有贝叶斯优化（BO）未适配获取函数敏感性的缺陷，提出Bilevel-BO-SWA，融合EI和UCB获取函数的双层优化策略，内层减小训练损失，外层优化验证指标。实验显示RoBERTa-base在GLUE任务上提升2.7%。**

- **链接: [http://arxiv.org/pdf/2505.17151v1](http://arxiv.org/pdf/2505.17151v1)**

> **作者:** Zishuo Bao; Yibo Liu; Changyutao Qiu
>
> **备注:** 12 pages, 3 figures, 2 tables
>
> **摘要:** With the rise of different language model architecture, fine-tuning is becoming even more important for down stream tasks Model gets messy, finding proper hyperparameters for fine-tuning. Although BO has been tried for hyperparameter tuning, most of the existing methods are oblivious to the fact that BO relies on careful choices of acquisition functions, which are essential components of BO that guide how much to explore versus exploit during the optimization process; Different acquisition functions have different levels of sensitivity towards training loss and validation performance; existing methods often just apply an acquisition function no matter if the training and validation performance are sensitive to the acquisition function or not. This work introduces{Bilevel - BO - SWA}, a model fusion approach coupled with a bilevel BO strategy to improve the fine - tunning of large language models. Our work on mixture of acquisition functions like EI and UCB into nested opt loops, where inner loop perform minimization of training loss while outer loops optimized w.r.t. val metric. Experiments on GLUE tasks using RoBERTA - base show that when using EI and UCB, there is an improvement in generalization, and fine - tuning can be improved by up to 2.7%.
>
---
#### [new 087] MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出实验引导的假设排序任务，旨在利用模拟实验反馈优化科学假设优先级。针对传统方法无法结合实验结果的问题，团队构建基于领域知识的模拟器并开发聚类排序方法，在124个化学实验数据上验证了优于基线的效果。**

- **链接: [http://arxiv.org/pdf/2505.17873v1](http://arxiv.org/pdf/2505.17873v1)**

> **作者:** Wanhao Liu; Zonglin Yang; Jue Wang; Lidong Bing; Di Zhang; Dongzhan Zhou; Yuqiang Li; Houqiang Li; Erik Cambria; Wanli Ouyang
>
> **摘要:** Hypothesis ranking is a crucial component of automated scientific discovery, particularly in natural sciences where wet-lab experiments are costly and throughput-limited. Existing approaches focus on pre-experiment ranking, relying solely on large language model's internal reasoning without incorporating empirical outcomes from experiments. We introduce the task of experiment-guided ranking, which aims to prioritize candidate hypotheses based on the results of previously tested ones. However, developing such strategies is challenging due to the impracticality of repeatedly conducting real experiments in natural science domains. To address this, we propose a simulator grounded in three domain-informed assumptions, modeling hypothesis performance as a function of similarity to a known ground truth hypothesis, perturbed by noise. We curate a dataset of 124 chemistry hypotheses with experimentally reported outcomes to validate the simulator. Building on this simulator, we develop a pseudo experiment-guided ranking method that clusters hypotheses by shared functional characteristics and prioritizes candidates based on insights derived from simulated experimental feedback. Experiments show that our method outperforms pre-experiment baselines and strong ablations.
>
---
#### [new 088] Training with Pseudo-Code for Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出通过在指令微调数据中加入伪代码表达的指令及响应，提升大语言模型的指令遵循能力。针对模型在简单组合任务中表现不佳且伪代码编写费时的问题，方法通过融合伪代码训练，在11个基准测试中使指令任务提升3-19%，同时保持数学和常识推理能力。属于模型微调任务，解决指令执行准确性问题。**

- **链接: [http://arxiv.org/pdf/2505.18011v1](http://arxiv.org/pdf/2505.18011v1)**

> **作者:** Prince Kumar; Rudra Murthy; Riyaz Bhat; Danish Contractor
>
> **备注:** Under Review
>
> **摘要:** Despite the rapid progress in the capabilities of Large Language Models (LLMs), they continue to have difficulty following relatively simple, unambiguous instructions, especially when compositions are involved. In this paper, we take inspiration from recent work that suggests that models may follow instructions better when they are expressed in pseudo-code. However, writing pseudo-code programs can be tedious and using few-shot demonstrations to craft code representations for use in inference can be unnatural for non-expert users of LLMs. To overcome these limitations, we propose fine-tuning LLMs with instruction-tuning data that additionally includes instructions re-expressed in pseudo-code along with the final response. We evaluate models trained using our method on $11$ publicly available benchmarks comprising of tasks related to instruction-following, mathematics, and common-sense reasoning. We conduct rigorous experiments with $5$ different models and find that not only do models follow instructions better when trained with pseudo-code, they also retain their capabilities on the other tasks related to mathematical and common sense reasoning. Specifically, we observe a relative gain of $3$--$19$% on instruction-following benchmark, and an average gain of upto 14% across all tasks.
>
---
#### [new 089] Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）错误检测任务，针对LLMs重复生成相同错误回答的"自我一致错误"问题，提出其定义并评估现有检测方法失效原因。发现该类错误随模型规模增长更顽固，进而提出跨模型探测方法，通过融合外部验证模型的隐藏层信息提升检测效果。**

- **链接: [http://arxiv.org/pdf/2505.17656v1](http://arxiv.org/pdf/2505.17656v1)**

> **作者:** Hexiang Tan; Fei Sun; Sha Liu; Du Su; Qi Cao; Xin Chen; Jingang Wang; Xunliang Cai; Yuanzhuo Wang; Huawei Shen; Xueqi Cheng
>
> **备注:** Underreview in EMNLP25
>
> **摘要:** As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
>
---
#### [new 090] QRA++: Quantified Reproducibility Assessment for Common Types of Results in Natural Language Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于NLP领域可重复性评估任务，针对现有复现研究因评估标准不统一导致结论难比较的问题，提出QRA++方法。其通过多粒度连续值评估、标准化指标及实验相似性建模，量化可重复性并分析其影响因素。**

- **链接: [http://arxiv.org/pdf/2505.17043v1](http://arxiv.org/pdf/2505.17043v1)**

> **作者:** Anya Belz
>
> **摘要:** Reproduction studies reported in NLP provide individual data points which in combination indicate worryingly low levels of reproducibility in the field. Because each reproduction study reports quantitative conclusions based on its own, often not explicitly stated, criteria for reproduction success/failure, the conclusions drawn are hard to interpret, compare, and learn from. In this paper, we present QRA++, a quantitative approach to reproducibility assessment that (i) produces continuous-valued degree of reproducibility assessments at three levels of granularity; (ii) utilises reproducibility measures that are directly comparable across different studies; and (iii) grounds expectations about degree of reproducibility in degree of similarity between experiments. QRA++ enables more informative reproducibility assessments to be conducted, and conclusions to be drawn about what causes reproducibility to be better/poorer. We illustrate this by applying QRA++ to three example sets of comparable experiments, revealing clear evidence that degree of reproducibility depends on similarity of experiment properties, but also system type and evaluation method.
>
---
#### [new 091] A new classification system of beer categories and styles based on large-scale data mining and self-organizing maps of beer recipes
- **分类: cs.CL**

- **简介: 该论文提出基于大数据挖掘与自组织映射（SOMs）的啤酒分类系统，解决传统感官分类主观性问题。通过分析62,121份啤酒配方的原料、发酵参数等数据，识别出4个主要超类群，揭示冷/热发酵风格差异，为 brewing和研究提供客观、可扩展的分类工具。**

- **链接: [http://arxiv.org/pdf/2505.17039v1](http://arxiv.org/pdf/2505.17039v1)**

> **作者:** Diego Bonatto
>
> **备注:** 46 pages, 8 figures, 1 table
>
> **摘要:** A data-driven quantitative approach was used to develop a novel classification system for beer categories and styles. Sixty-two thousand one hundred twenty-one beer recipes were mined and analyzed, considering ingredient profiles, fermentation parameters, and recipe vital statistics. Statistical analyses combined with self-organizing maps (SOMs) identified four major superclusters that showed distinctive malt and hop usage patterns, style characteristics, and historical brewing traditions. Cold fermented styles showed a conservative grain and hop composition, whereas hot fermented beers exhibited high heterogeneity, reflecting regional preferences and innovation. This new taxonomy offers a reproducible and objective framework beyond traditional sensory-based classifications, providing brewers, researchers, and educators with a scalable tool for recipe analysis and beer development. The findings in this work provide an understanding of beer diversity and open avenues for linking ingredient usage with fermentation profiles and flavor outcomes.
>
---
#### [new 092] What's in a prompt? Language models encode literary style in prompt embeddings
- **分类: cs.CL**

- **简介: 该论文研究语言模型如何通过嵌入编码文学风格。任务是分析深层表征如何捕捉文本的抽象风格而非具体语义。发现短文本在潜在空间的分布受作者风格影响显著，同一作者作品嵌入更相似，揭示模型能压缩风格特征，可用于作者ship鉴定与文学分析。**

- **链接: [http://arxiv.org/pdf/2505.17071v1](http://arxiv.org/pdf/2505.17071v1)**

> **作者:** Raphaël Sarfati; Haley Moller; Toni J. B. Liu; Nicolas Boullé; Christopher Earls
>
> **摘要:** Large language models use high-dimensional latent spaces to encode and process textual information. Much work has investigated how the conceptual content of words translates into geometrical relationships between their vector representations. Fewer studies analyze how the cumulative information of an entire prompt becomes condensed into individual embeddings under the action of transformer layers. We use literary pieces to show that information about intangible, rather than factual, aspects of the prompt are contained in deep representations. We observe that short excerpts (10 - 100 tokens) from different novels separate in the latent space independently from what next-token prediction they converge towards. Ensembles from books from the same authors are much more entangled than across authors, suggesting that embeddings encode stylistic features. This geometry of style may have applications for authorship attribution and literary analysis, but most importantly reveals the sophistication of information processing and compression accomplished by language models.
>
---
#### [new 093] Refusal Direction is Universal Across Safety-Aligned Languages
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLM）跨语言拒绝机制的普遍性。针对多语言安全需理解拒绝行为转移性的问题，使用多语言数据集PolyRefuse在14种语言中验证，发现英语提取的拒绝方向向量可无缝迁移至其他语言，无需微调即可绕过拒绝响应。揭示嵌入空间中拒绝向量的平行性机制，为改进多语种安全防御提供理论支持。**

- **链接: [http://arxiv.org/pdf/2505.17306v1](http://arxiv.org/pdf/2505.17306v1)**

> **作者:** Xinpeng Wang; Mingyang Wang; Yihong Liu; Hinrich Schütze; Barbara Plank
>
> **摘要:** Refusal mechanisms in large language models (LLMs) are essential for ensuring safety. Recent research has revealed that refusal behavior can be mediated by a single direction in activation space, enabling targeted interventions to bypass refusals. While this is primarily demonstrated in an English-centric context, appropriate refusal behavior is important for any language, but poorly understood. In this paper, we investigate the refusal behavior in LLMs across 14 languages using PolyRefuse, a multilingual safety dataset created by translating malicious and benign English prompts into these languages. We uncover the surprising cross-lingual universality of the refusal direction: a vector extracted from English can bypass refusals in other languages with near-perfect effectiveness, without any additional fine-tuning. Even more remarkably, refusal directions derived from any safety-aligned language transfer seamlessly to others. We attribute this transferability to the parallelism of refusal vectors across languages in the embedding space and identify the underlying mechanism behind cross-lingual jailbreaks. These findings provide actionable insights for building more robust multilingual safety defenses and pave the way for a deeper mechanistic understanding of cross-lingual vulnerabilities in LLMs.
>
---
#### [new 094] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出L-MTP方法，旨在提升大语言模型的推理效率与性能。针对传统逐字预测（NTP）因顺序处理导致的上下文覆盖不足与效率低下问题，其通过跳跃机制预测非连续token，增强长程依赖并优化解码策略，实验验证了其有效。**

- **链接: [http://arxiv.org/pdf/2505.17505v1](http://arxiv.org/pdf/2505.17505v1)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Weixiang Zhao; Manyi Zhang; Xianzhi Yu; Xiu Su; Shuo Yang; See-Kiong Ng; Tat-Seng Chua
>
> **摘要:** Large language models (LLMs) have achieved notable progress. Despite their success, next-token prediction (NTP), the dominant method for LLM training and inference, is constrained in both contextual coverage and inference efficiency due to its inherently sequential process. To overcome these challenges, we propose leap multi-token prediction~(L-MTP), an innovative token prediction method that extends the capabilities of multi-token prediction (MTP) by introducing a leap-based mechanism. Unlike conventional MTP, which generates multiple tokens at adjacent positions, L-MTP strategically skips over intermediate tokens, predicting non-sequential ones in a single forward pass. This structured leap not only enhances the model's ability to capture long-range dependencies but also enables a decoding strategy specially optimized for non-sequential leap token generation, effectively accelerating inference. We theoretically demonstrate the benefit of L-MTP in improving inference efficiency. Experiments across diverse benchmarks validate its merit in boosting both LLM performance and inference speed. The source code will be publicly available.
>
---
#### [new 095] FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow
- **分类: cs.CL**

- **简介: 该论文提出FullFront基准，评估多模态大模型在前端工程全流程（设计、视觉理解、代码生成）中的能力。针对现有基准任务单一、数据质量不足的问题，其构建了涵盖三阶段任务的评测体系，并采用两阶段处理生成标准化网页数据。实验显示模型在布局理解、图像处理及交互实现上存在显著局限，与人类表现差距明显。**

- **链接: [http://arxiv.org/pdf/2505.17399v1](http://arxiv.org/pdf/2505.17399v1)**

> **作者:** Haoyu Sun; Huichen Will Wang; Jiawei Gu; Linjie Li; Yu Cheng
>
> **摘要:** Front-end engineering involves a complex workflow where engineers conceptualize designs, translate them into code, and iteratively refine the implementation. While recent benchmarks primarily focus on converting visual designs to code, we present FullFront, a benchmark designed to evaluate Multimodal Large Language Models (MLLMs) \textbf{across the full front-end development pipeline}. FullFront assesses three fundamental tasks that map directly to the front-end engineering pipeline: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase). Unlike existing benchmarks that use either scraped websites with bloated code or oversimplified LLM-generated HTML, FullFront employs a novel, two-stage process to transform real-world webpages into clean, standardized HTML while maintaining diverse visual designs and avoiding copyright issues. Extensive testing of state-of-the-art MLLMs reveals significant limitations in page perception, code generation (particularly for image handling and layout), and interaction implementation. Our results quantitatively demonstrate performance disparities across models and tasks, and highlight a substantial gap between current MLLM capabilities and human expert performance in front-end engineering. The FullFront benchmark and code are available in https://github.com/Mikivishy/FullFront.
>
---
#### [new 096] Bridging Electronic Health Records and Clinical Texts: Contrastive Learning for Enhanced Clinical Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于临床预测任务，旨在解决传统模型因结构化电子健康记录（EHR）语义信息不足，在如30天再入院预测等需深度理解的任务中表现受限的问题。提出深度多模态对比学习框架，通过联合建模EHR与非结构化出院总结，对齐二者潜在表示，微调后提升预测精度（如AUROC提高4.1%）。**

- **链接: [http://arxiv.org/pdf/2505.17643v1](http://arxiv.org/pdf/2505.17643v1)**

> **作者:** Sara Ketabi; Dhanesh Ramachandram
>
> **摘要:** Conventional machine learning models, particularly tree-based approaches, have demonstrated promising performance across various clinical prediction tasks using electronic health record (EHR) data. Despite their strengths, these models struggle with tasks that require deeper contextual understanding, such as predicting 30-day hospital readmission. This can be primarily due to the limited semantic information available in structured EHR data. To address this limitation, we propose a deep multimodal contrastive learning (CL) framework that aligns the latent representations of structured EHR data with unstructured discharge summary notes. It works by pulling together paired EHR and text embeddings while pushing apart unpaired ones. Fine-tuning the pretrained EHR encoder extracted from this framework significantly boosts downstream task performance, e.g., a 4.1% AUROC enhancement over XGBoost for 30-day readmission prediction. Such results demonstrate the effect of integrating domain knowledge from clinical notes into EHR-based pipelines, enabling more accurate and context-aware clinical decision support systems.
>
---
#### [new 097] From Compression to Expansion: A Layerwise Analysis of In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLMs）的in-context learning（ICL）机制，分析其内部表征如何随层深度变化。通过统计几何方法，发现“层间压缩-扩展”现象：早期层压缩输入示例的任务信息，后期层扩展结合查询生成预测。提出偏差-方差分解，理论分析注意力机制作用，揭示模型规模与示例数对ICL性能的影响。**

- **链接: [http://arxiv.org/pdf/2505.17322v1](http://arxiv.org/pdf/2505.17322v1)**

> **作者:** Jiachen Jiang; Yuxin Dong; Jinxin Zhou; Zhihui Zhu
>
> **摘要:** In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks without weight updates by learning from demonstration sequences. While ICL shows strong empirical performance, its internal representational mechanisms are not yet well understood. In this work, we conduct a statistical geometric analysis of ICL representations to investigate how task-specific information is captured across layers. Our analysis reveals an intriguing phenomenon, which we term *Layerwise Compression-Expansion*: early layers progressively produce compact and discriminative representations that encode task information from the input demonstrations, while later layers expand these representations to incorporate the query and generate the prediction. This phenomenon is observed consistently across diverse tasks and a range of contemporary LLM architectures. We demonstrate that it has important implications for ICL performance -- improving with model size and the number of demonstrations -- and for robustness in the presence of noisy examples. To further understand the effect of the compact task representation, we propose a bias-variance decomposition and provide a theoretical analysis showing how attention mechanisms contribute to reducing both variance and bias, thereby enhancing performance as the number of demonstrations increases. Our findings reveal an intriguing layerwise dynamic in ICL, highlight how structured representations emerge within LLMs, and showcase that analyzing internal representations can facilitate a deeper understanding of model behavior.
>
---
#### [new 098] LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LongMagpie框架，解决长上下文指令数据稀缺问题。通过引导预训练LLM自动生成文档-查询对及响应，无需人工标注，实现大规模、多样化的长指令数据合成。实验显示其在长任务中表现优异，同时保持短任务性能。**

- **链接: [http://arxiv.org/pdf/2505.17134v1](http://arxiv.org/pdf/2505.17134v1)**

> **作者:** Chaochen Gao; Xing Wu; Zijia Lin; Debing Zhang; Songlin Hu
>
> **摘要:** High-quality long-context instruction data is essential for aligning long-context large language models (LLMs). Despite the public release of models like Qwen and Llama, their long-context instruction data remains proprietary. Human annotation is costly and challenging, while template-based synthesis methods limit scale, diversity, and quality. We introduce LongMagpie, a self-synthesis framework that automatically generates large-scale long-context instruction data. Our key insight is that aligned long-context LLMs, when presented with a document followed by special tokens preceding a user turn, auto-regressively generate contextually relevant queries. By harvesting these document-query pairs and the model's responses, LongMagpie produces high-quality instructions without human effort. Experiments on HELMET, RULER, and Longbench v2 demonstrate that LongMagpie achieves leading performance on long-context tasks while maintaining competitive performance on short-context tasks, establishing it as a simple and effective approach for open, diverse, and scalable long-context instruction data synthesis.
>
---
#### [new 099] SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SpecEdge框架，针对大语言模型（LLMs）服务成本高、资源消耗大的问题，通过边缘设备与服务器协同处理（投机解码、主动调度），提升吞吐量并降低延迟，成本效益提高1.91倍，服务器吞吐量提升2.22倍，交互延迟减少11.24%。**

- **链接: [http://arxiv.org/pdf/2505.17052v1](http://arxiv.org/pdf/2505.17052v1)**

> **作者:** Jinwoo Park; Seunggeun Cho; Dongsu Han
>
> **摘要:** Large language models (LLMs) power many modern applications, but serving them at scale remains costly and resource-intensive. Current server-centric systems overlook consumer-grade GPUs at the edge. We introduce SpecEdge, an edge-assisted inference framework that splits LLM workloads between edge and server GPUs using a speculative decoding scheme, exchanging only token outputs over the network. SpecEdge employs proactive edge drafting to overlap edge token creation with server verification and pipeline-aware scheduling that interleaves multiple user requests to increase server-side throughput. Experiments show SpecEdge enhances overall cost efficiency by 1.91x through achieving 2.22x server throughput, and reduces inter token latency by 11.24% compared to a server-only baseline, introducing a scalable, cost-effective paradigm for LLM serving.
>
---
#### [new 100] Language models can learn implicit multi-hop reasoning, but only if they have lots of training data
- **分类: cs.CL**

- **简介: 该论文研究语言模型隐式多跳推理能力，探究其解决k-hop任务所需数据与模型深度的关系。发现数据量随k指数增长，层数随k线性增长，并通过理论解释深度必要性，课程学习可部分缓解数据需求。**

- **链接: [http://arxiv.org/pdf/2505.17923v1](http://arxiv.org/pdf/2505.17923v1)**

> **作者:** Yuekun Yao; Yupei Du; Dawei Zhu; Michael Hahn; Alexander Koller
>
> **摘要:** Implicit reasoning is the ability of a language model to solve multi-hop reasoning tasks in a single forward pass, without chain of thought. We investigate this capability using GPT2-style language models trained from scratch on controlled $k$-hop reasoning datasets ($k = 2, 3, 4$). We show that while such models can indeed learn implicit $k$-hop reasoning, the required training data grows exponentially in $k$, and the required number of transformer layers grows linearly in $k$. We offer a theoretical explanation for why this depth growth is necessary. We further find that the data requirement can be mitigated, but not eliminated, through curriculum learning.
>
---
#### [new 101] GPT Editors, Not Authors: The Stylistic Footprint of LLMs in Academic Preprints
- **分类: cs.CL; cs.IT; cs.LG; math.IT; 68U99; I.2.7**

- **简介: 论文探究LLMs在学术论文中的应用方式（编辑vs生成）。通过分析arXiv论文的风格分段，结合PELT阈值和贝叶斯分类器，发现LLM修改均匀分布，降低幻觉风险，属风格检测任务。**

- **链接: [http://arxiv.org/pdf/2505.17327v1](http://arxiv.org/pdf/2505.17327v1)**

> **作者:** Soren DeHaan; Yuanze Liu; Johan Bollen; Sa'ul A. Blanco
>
> **备注:** 13 pages
>
> **摘要:** The proliferation of Large Language Models (LLMs) in late 2022 has impacted academic writing, threatening credibility, and causing institutional uncertainty. We seek to determine the degree to which LLMs are used to generate critical text as opposed to being used for editing, such as checking for grammar errors or inappropriate phrasing. In our study, we analyze arXiv papers for stylistic segmentation, which we measure by varying a PELT threshold against a Bayesian classifier trained on GPT-regenerated text. We find that LLM-attributed language is not predictive of stylistic segmentation, suggesting that when authors use LLMs, they do so uniformly, reducing the risk of hallucinations being introduced into academic preprints.
>
---
#### [new 102] CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CaseReportBench数据集，用于评估LLM在罕见病（如代谢障碍）临床病例报告的密集信息抽取任务。针对现有模型未充分挖掘病例报告价值的问题，构建专家标注数据集，测试不同模型与提示策略（如类别特定提示），发现开源模型Qwen2.5-7B表现优于GPT-4o，但LLM在识别阴性诊断信息方面仍需改进，推动医疗NLP应用。**

- **链接: [http://arxiv.org/pdf/2505.17265v1](http://arxiv.org/pdf/2505.17265v1)**

> **作者:** Xiao Yu Cindy Zhang; Carlos R. Ferreira; Francis Rossignol; Raymond T. Ng; Wyeth Wasserman; Jian Zhu
>
> **摘要:** Rare diseases, including Inborn Errors of Metabolism (IEM), pose significant diagnostic challenges. Case reports serve as key but computationally underutilized resources to inform diagnosis. Clinical dense information extraction refers to organizing medical information into structured predefined categories. Large Language Models (LLMs) may enable scalable information extraction from case reports but are rarely evaluated for this task. We introduce CaseReportBench, an expert-annotated dataset for dense information extraction of case reports, focusing on IEMs. Using this dataset, we assess various models and prompting strategies, introducing novel approaches such as category-specific prompting and subheading-filtered data integration. Zero-shot chain-of-thought prompting offers little advantage over standard zero-shot prompting. Category-specific prompting improves alignment with the benchmark. The open-source model Qwen2.5-7B outperforms GPT-4o for this task. Our clinician evaluations show that LLMs can extract clinically relevant details from case reports, supporting rare disease diagnosis and management. We also highlight areas for improvement, such as LLMs' limitations in recognizing negative findings important for differential diagnosis. This work advances LLM-driven clinical natural language processing and paves the way for scalable medical AI applications.
>
---
#### [new 103] Predictively Combatting Toxicity in Health-related Online Discussions through Machine Learning
- **分类: cs.CL; cs.LG; cs.SI**

- **简介: 该论文提出基于协同过滤的机器学习模型，预测用户在健康类在线讨论（如Reddit的COVID话题）中的潜在毒性互动，替代传统的事后检测方法。通过预测用户与子社区的毒性风险（>80%准确率），实现提前干预以阻止有害对话，解决现有过滤措施的反效果问题。**

- **链接: [http://arxiv.org/pdf/2505.17068v1](http://arxiv.org/pdf/2505.17068v1)**

> **作者:** Jorge Paz-Ruza; Amparo Alonso-Betanzos; Bertha Guijarro-Berdiñas; Carlos Eiras-Franco
>
> **备注:** IJCNN 2025
>
> **摘要:** In health-related topics, user toxicity in online discussions frequently becomes a source of social conflict or promotion of dangerous, unscientific behaviour; common approaches for battling it include different forms of detection, flagging and/or removal of existing toxic comments, which is often counterproductive for platforms and users alike. In this work, we propose the alternative of combatting user toxicity predictively, anticipating where a user could interact toxically in health-related online discussions. Applying a Collaborative Filtering-based Machine Learning methodology, we predict the toxicity in COVID-related conversations between any user and subcommunity of Reddit, surpassing 80% predictive performance in relevant metrics, and allowing us to prevent the pairing of conflicting users and subcommunities.
>
---
#### [new 104] Embedding-to-Prefix: Parameter-Efficient Personalization for Pre-Trained Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大型语言模型个性化任务，旨在解决利用用户特征高效定制生成内容的问题。现有方法依赖成本高的微调或复杂提示，作者提出Embedding-to-Prefix（E2P），通过将用户嵌入投影为前缀软令牌注入模型隐藏层，在冻结主干模型前提下实现高效个性化，实验验证其在对话及内容生成场景中效果显著且计算开销小。**

- **链接: [http://arxiv.org/pdf/2505.17051v1](http://arxiv.org/pdf/2505.17051v1)**

> **作者:** Bernd Huber; Ghazal Fazelnia; Andreas Damianou; Sebastian Peleato; Max Lefarov; Praveen Ravichandran; Marco De Nadai; Mounia Lalmas-Roellke; Paul N. Bennett
>
> **摘要:** Large language models (LLMs) excel at generating contextually relevant content. However, tailoring these outputs to individual users for effective personalization is a significant challenge. While rich user-specific information often exists as pre-existing user representations, such as embeddings learned from preferences or behaviors, current methods to leverage these for LLM personalization typically require costly fine-tuning or token-heavy prompting. We propose Embedding-to-Prefix (E2P), a parameter-efficient method that injects pre-computed context embeddings into an LLM's hidden representation space through a learned projection to a single soft token prefix. This enables effective personalization while keeping the backbone model frozen and avoiding expensive adaptation techniques. We evaluate E2P across two public datasets and in a production setting: dialogue personalization on Persona-Chat, contextual headline generation on PENS, and large-scale personalization for music and podcast consumption. Results show that E2P preserves contextual signals and achieves strong performance with minimal computational overhead, offering a scalable, efficient solution for contextualizing generative AI systems.
>
---
#### [new 105] Reasoning Meets Personalization: Unleashing the Potential of Large Reasoning Model for Personalized Generation
- **分类: cs.CL**

- **简介: 该论文属于个性化生成任务，旨在解决大型推理模型（LRMs）在个性化场景中的性能不足问题。发现其存在发散思维、响应格式错位及检索信息利用低效三大局限，提出Reinforced Reasoning框架，通过分层推理模板、过程干预及交叉引用机制优化输出，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.17571v1](http://arxiv.org/pdf/2505.17571v1)**

> **作者:** Sichun Luo; Guanzhi Deng; Jian Xu; Xiaojie Zhang; Hanxu Hou; Linqi Song
>
> **摘要:** Personalization is a critical task in modern intelligent systems, with applications spanning diverse domains, including interactions with large language models (LLMs). Recent advances in reasoning capabilities have significantly enhanced LLMs, enabling unprecedented performance in tasks such as mathematics and coding. However, their potential for personalization tasks remains underexplored. In this paper, we present the first systematic evaluation of large reasoning models (LRMs) for personalization tasks. Surprisingly, despite generating more tokens, LRMs do not consistently outperform general-purpose LLMs, especially in retrieval-intensive scenarios where their advantages diminish. Our analysis identifies three key limitations: divergent thinking, misalignment of response formats, and ineffective use of retrieved information. To address these challenges, we propose Reinforced Reasoning for Personalization (\model), a novel framework that incorporates a hierarchical reasoning thought template to guide LRMs in generating structured outputs. Additionally, we introduce a reasoning process intervention method to enforce adherence to designed reasoning patterns, enhancing alignment. We also propose a cross-referencing mechanism to ensure consistency. Extensive experiments demonstrate that our approach significantly outperforms existing techniques.
>
---
#### [new 106] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态in-context learning（ICL）任务，旨在解决复杂任务中输入序列质量敏感及模型推理机制不明确的问题。提出TACO模型，通过任务映射分析演示序列的局部/全局关系，并动态配置上下文序列，实现序列构建与任务推理的协同优化，实验显示其性能优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.17098v1](http://arxiv.org/pdf/2505.17098v1)**

> **作者:** Yanshu Li; Tian Yun; Jianjiang Yang; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** 29 pages, 11 figures, 19 tables. arXiv admin note: substantial text overlap with arXiv:2503.04839
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input in-context sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures in-context sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [new 107] Social preferences with unstable interactive reasoning: Large language models in economic trust games
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大型语言模型（LLMs）在经济信任游戏中的社会偏好与互动推理。通过对比ChatGPT-4、Claude和Bard的决策行为，探讨其如何在信任与自利间平衡。发现LLMs自发展现信任和互惠，但多轮互动中推理不稳定；角色设定显著影响其行为，无私模式下ChatGPT-4表现超人类，自私模式则降低。任务为分析LLMs社交决策机制，解决其与人类差异及角色影响问题。**

- **链接: [http://arxiv.org/pdf/2505.17053v1](http://arxiv.org/pdf/2505.17053v1)**

> **作者:** Ou Jiamin; Eikmans Emile; Buskens Vincent; Pankowska Paulina; Shan Yuli
>
> **备注:** 19 pages, 2 figures, 2 tables
>
> **摘要:** While large language models (LLMs) have demonstrated remarkable capabilities in understanding human languages, this study explores how they translate this understanding into social exchange contexts that capture certain essences of real world human interactions. Three LLMs - ChatGPT-4, Claude, and Bard - were placed in economic trust games where players balance self-interest with trust and reciprocity, making decisions that reveal their social preferences and interactive reasoning abilities. Our study shows that LLMs deviate from pure self-interest and exhibit trust and reciprocity even without being prompted to adopt a specific persona. In the simplest one-shot interaction, LLMs emulated how human players place trust at the beginning of such a game. Larger human-machine divergences emerged in scenarios involving trust repayment or multi-round interactions, where decisions were influenced by both social preferences and interactive reasoning. LLMs responses varied significantly when prompted to adopt personas like selfish or unselfish players, with the impact outweighing differences between models or game types. Response of ChatGPT-4, in an unselfish or neutral persona, resembled the highest trust and reciprocity, surpassing humans, Claude, and Bard. Claude and Bard displayed trust and reciprocity levels that sometimes exceeded and sometimes fell below human choices. When given selfish personas, all LLMs showed lower trust and reciprocity than humans. Interactive reasoning to the actions of counterparts or changing game mechanics appeared to be random rather than stable, reproducible characteristics in the response of LLMs, though some improvements were observed when ChatGPT-4 responded in selfish or unselfish personas.
>
---
#### [new 108] Decoding Rarity: Large Language Models in the Diagnosis of Rare Diseases
- **分类: cs.CL; cs.LG**

- **简介: 该论文综述了大语言模型（LLMs）在罕见病诊断中的应用，旨在探索其如何通过分析文本及多模态数据提升诊断效率。研究整合了LLMs在医学信息提取、患者交互及诊断支持中的应用，讨论了数据隐私与模型透明性等挑战，并实验了多模型与问卷结合的诊断方法，展望了多模态平台的未来潜力。**

- **链接: [http://arxiv.org/pdf/2505.17065v1](http://arxiv.org/pdf/2505.17065v1)**

> **作者:** Valentina Carbonari; Pierangelo Veltri; Pietro Hiram Guzzi
>
> **摘要:** Recent advances in artificial intelligence, particularly large language models LLMs, have shown promising capabilities in transforming rare disease research. This survey paper explores the integration of LLMs in the analysis of rare diseases, highlighting significant strides and pivotal studies that leverage textual data to uncover insights and patterns critical for diagnosis, treatment, and patient care. While current research predominantly employs textual data, the potential for multimodal data integration combining genetic, imaging, and electronic health records stands as a promising frontier. We review foundational papers that demonstrate the application of LLMs in identifying and extracting relevant medical information, simulating intelligent conversational agents for patient interaction, and enabling the formulation of accurate and timely diagnoses. Furthermore, this paper discusses the challenges and ethical considerations inherent in deploying LLMs, including data privacy, model transparency, and the need for robust, inclusive data sets. As part of this exploration, we present a section on experimentation that utilizes multiple LLMs alongside structured questionnaires, specifically designed for diagnostic purposes in the context of different diseases. We conclude with future perspectives on the evolution of LLMs towards truly multimodal platforms, which would integrate diverse data types to provide a more comprehensive understanding of rare diseases, ultimately fostering better outcomes in clinical settings.
>
---
#### [new 109] Unveil Multi-Picture Descriptions for Multilingual Mild Cognitive Impairment Detection via Contrastive Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对多语言多图片场景下的轻度认知障碍（MCI）检测任务，提出结合监督对比学习、图像模态及PoE策略的框架，解决现有方法在多模态分析和跨语言泛化上的不足，提升检测性能（UAR+7.1%，F1+2.9%）。**

- **链接: [http://arxiv.org/pdf/2505.17067v1](http://arxiv.org/pdf/2505.17067v1)**

> **作者:** Kristin Qi; Jiali Cheng; Youxiang Zhu; Hadi Amiri; Xiaohui Liang
>
> **备注:** Submitted to the IEEE GlobeCom 2025
>
> **摘要:** Detecting Mild Cognitive Impairment from picture descriptions is critical yet challenging, especially in multilingual and multiple picture settings. Prior work has primarily focused on English speakers describing a single picture (e.g., the 'Cookie Theft'). The TAUKDIAL-2024 challenge expands this scope by introducing multilingual speakers and multiple pictures, which presents new challenges in analyzing picture-dependent content. To address these challenges, we propose a framework with three components: (1) enhancing discriminative representation learning via supervised contrastive learning, (2) involving image modality rather than relying solely on speech and text modalities, and (3) applying a Product of Experts (PoE) strategy to mitigate spurious correlations and overfitting. Our framework improves MCI detection performance, achieving a +7.1% increase in Unweighted Average Recall (UAR) (from 68.1% to 75.2%) and a +2.9% increase in F1 score (from 80.6% to 83.5%) compared to the text unimodal baseline. Notably, the contrastive learning component yields greater gains for the text modality compared to speech. These results highlight our framework's effectiveness in multilingual and multi-picture MCI detection.
>
---
#### [new 110] Are LLMs reliable? An exploration of the reliability of large language models in clinical note generation
- **分类: cs.CL**

- **简介: 该论文评估了12种LLMs在临床笔记生成中的可靠性，解决其响应变异性与医疗数据隐私合规问题。通过测试模型生成笔记的一致性、语义稳定性和准确性，发现Meta的Llama 70B和Mistral小模型表现最佳，建议部署开源小模型以提升医疗文档效率并符合隐私要求。**

- **链接: [http://arxiv.org/pdf/2505.17095v1](http://arxiv.org/pdf/2505.17095v1)**

> **作者:** Kristine Ann M. Carandang; Jasper Meynard P. Araña; Ethan Robert A. Casin; Christopher P. Monterola; Daniel Stanley Y. Tan; Jesus Felix B. Valenzuela; Christian M. Alis
>
> **摘要:** Due to the legal and ethical responsibilities of healthcare providers (HCPs) for accurate documentation and protection of patient data privacy, the natural variability in the responses of large language models (LLMs) presents challenges for incorporating clinical note generation (CNG) systems, driven by LLMs, into real-world clinical processes. The complexity is further amplified by the detailed nature of texts in CNG. To enhance the confidence of HCPs in tools powered by LLMs, this study evaluates the reliability of 12 open-weight and proprietary LLMs from Anthropic, Meta, Mistral, and OpenAI in CNG in terms of their ability to generate notes that are string equivalent (consistency rate), have the same meaning (semantic consistency) and are correct (semantic similarity), across several iterations using the same prompt. The results show that (1) LLMs from all model families are stable, such that their responses are semantically consistent despite being written in various ways, and (2) most of the LLMs generated notes close to the corresponding notes made by experts. Overall, Meta's Llama 70B was the most reliable, followed by Mistral's Small model. With these findings, we recommend the local deployment of these relatively smaller open-weight models for CNG to ensure compliance with data privacy regulations, as well as to improve the efficiency of HCPs in clinical documentation.
>
---
#### [new 111] Personalizing Student-Agent Interactions Using Log-Contextualized Retrieval Augmented Generation (RAG)
- **分类: cs.CL**

- **简介: 该论文属于教育人工智能领域，旨在解决传统RAG模型在STEM+C教育场景中因学生对话与知识库语义关联弱导致的回复不准确问题。提出LC-RAG方法，通过融合环境日志增强检索，使智能代理Copa在协作建模中提供精准个性化指导，提升学生批判性思维支持。**

- **链接: [http://arxiv.org/pdf/2505.17238v1](http://arxiv.org/pdf/2505.17238v1)**

> **作者:** Clayton Cohn; Surya Rayala; Caitlin Snyder; Joyce Fonteles; Shruti Jain; Naveeduddin Mohammed; Umesh Timalsina; Sarah K. Burriss; Ashwin T S; Namrata Srivastava; Menton Deweese; Angela Eeds; Gautam Biswas
>
> **备注:** Submitted to the International Conference on Artificial Intelligence in Education (AIED) Workshop on Epistemics and Decision-Making in AI-Supported Education
>
> **摘要:** Collaborative dialogue offers rich insights into students' learning and critical thinking. This is essential for adapting pedagogical agents to students' learning and problem-solving skills in STEM+C settings. While large language models (LLMs) facilitate dynamic pedagogical interactions, potential hallucinations can undermine confidence, trust, and instructional value. Retrieval-augmented generation (RAG) grounds LLM outputs in curated knowledge, but its effectiveness depends on clear semantic links between user input and a knowledge base, which are often weak in student dialogue. We propose log-contextualized RAG (LC-RAG), which enhances RAG retrieval by incorporating environment logs to contextualize collaborative discourse. Our findings show that LC-RAG improves retrieval over a discourse-only baseline and allows our collaborative peer agent, Copa, to deliver relevant, personalized guidance that supports students' critical thinking and epistemic decision-making in a collaborative computational modeling environment, XYZ.
>
---
#### [new 112] T$^2$: An Adaptive Test-Time Scaling Strategy for Contextual Question Answering
- **分类: cs.CL**

- **简介: 该论文属于上下文问答（CQA）任务，针对现有方法不区分问题复杂度导致效率低的问题，提出T²框架。其通过分解问题结构、生成策略示例、评估筛选最优策略，动态调整推理深度，提升准确率并减少25.2%计算开销。**

- **链接: [http://arxiv.org/pdf/2505.17427v1](http://arxiv.org/pdf/2505.17427v1)**

> **作者:** Zhengyi Zhao; Shubo Zhang; Zezhong Wang; Huimin Wang; Yutian Zhao; Bin Liang; Yefeng Zheng; Binyang Li; Kam-Fai Wong; Xian Wu
>
> **摘要:** Recent advances in Large Language Models (LLMs) have demonstrated remarkable performance in Contextual Question Answering (CQA). However, prior approaches typically employ elaborate reasoning strategies regardless of question complexity, leading to low adaptability. Recent efficient test-time scaling methods introduce budget constraints or early stop mechanisms to avoid overthinking for straightforward questions. But they add human bias to the reasoning process and fail to leverage models' inherent reasoning capabilities. To address these limitations, we present T$^2$: Think-to-Think, a novel framework that dynamically adapts reasoning depth based on question complexity. T$^2$ leverages the insight that if an LLM can effectively solve similar questions using specific reasoning strategies, it can apply the same strategy to the original question. This insight enables to adoption of concise reasoning for straightforward questions while maintaining detailed analysis for complex problems. T$^2$ works through four key steps: decomposing questions into structural elements, generating similar examples with candidate reasoning strategies, evaluating these strategies against multiple criteria, and applying the most appropriate strategy to the original question. Experimental evaluation across seven diverse CQA benchmarks demonstrates that T$^2$ not only achieves higher accuracy than baseline methods but also reduces computational overhead by up to 25.2\%.
>
---
#### [new 113] Scale-invariant Attention
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于长上下文注意力机制研究任务，旨在解决LLM从短上下文训练推广到长上下文推理的泛化问题。提出尺度不变总注意力与稀疏性条件，通过位置依赖的注意力logits变换满足条件，实验验证其在零样本长上下文推理中有效降低验证损失并提升检索性能。**

- **链接: [http://arxiv.org/pdf/2505.17083v1](http://arxiv.org/pdf/2505.17083v1)**

> **作者:** Ben Anson; Xi Wang; Laurence Aitchison
>
> **备注:** Preprint
>
> **摘要:** One persistent challenge in LLM research is the development of attention mechanisms that are able to generalise from training on shorter contexts to inference on longer contexts. We propose two conditions that we expect all effective long context attention mechanisms to have: scale-invariant total attention, and scale-invariant attention sparsity. Under a Gaussian assumption, we show that a simple position-dependent transformation of the attention logits is sufficient for these conditions to hold. Experimentally we find that the resulting scale-invariant attention scheme gives considerable benefits in terms of validation loss when zero-shot generalising from training on short contexts to validation on longer contexts, and is effective at long-context retrieval.
>
---
#### [new 114] RRTL: Red Teaming Reasoning Large Language Models in Tool Learning
- **分类: cs.CL**

- **简介: 该论文属于红队测试任务，旨在评估推理大语言模型（RLLMs）在工具学习中的安全性。针对RLLMs隐藏工具使用、未提示风险等潜在漏洞，提出RRTL方法，包含识别欺骗威胁和强制调用工具的CoT提示策略，并建立基准测试。实验发现RLLMs安全性能存在差异，且存在多语言安全漏洞，为提升其安全性提供依据。**

- **链接: [http://arxiv.org/pdf/2505.17106v1](http://arxiv.org/pdf/2505.17106v1)**

> **作者:** Yifei Liu; Yu Cui; Haibin Zhang
>
> **摘要:** While tool learning significantly enhances the capabilities of large language models (LLMs), it also introduces substantial security risks. Prior research has revealed various vulnerabilities in traditional LLMs during tool learning. However, the safety of newly emerging reasoning LLMs (RLLMs), such as DeepSeek-R1, in the context of tool learning remains underexplored. To bridge this gap, we propose RRTL, a red teaming approach specifically designed to evaluate RLLMs in tool learning. It integrates two novel strategies: (1) the identification of deceptive threats, which evaluates the model's behavior in concealing the usage of unsafe tools and their potential risks; and (2) the use of Chain-of-Thought (CoT) prompting to force tool invocation. Our approach also includes a benchmark for traditional LLMs. We conduct a comprehensive evaluation on seven mainstream RLLMs and uncover three key findings: (1) RLLMs generally achieve stronger safety performance than traditional LLMs, yet substantial safety disparities persist across models; (2) RLLMs can pose serious deceptive risks by frequently failing to disclose tool usage and to warn users of potential tool output risks; (3) CoT prompting reveals multi-lingual safety vulnerabilities in RLLMs. Our work provides important insights into enhancing the security of RLLMs in tool learning.
>
---
#### [new 115] The Real Barrier to LLM Agent Usability is Agentic ROI
- **分类: cs.CL**

- **简介: 该论文属于AI应用优化任务，旨在解决LLM代理在大众市场低采用率问题。提出Agentic ROI框架，分析信息质量、时间和成本三大关键因素，提出分阶段优化路径（先提升信息质量，再降低时间和成本），呼吁从模型性能优化转向实用价值评估，以提升LLM代理的可扩展性与实用性。**

- **链接: [http://arxiv.org/pdf/2505.17767v1](http://arxiv.org/pdf/2505.17767v1)**

> **作者:** Weiwen Liu; Jiarui Qin; Xu Huang; Xingshan Zeng; Yunjia Xi; Jianghao Lin; Chuhan Wu; Yasheng Wang; Lifeng Shang; Ruiming Tang; Defu Lian; Yong Yu; Weinan Zhang
>
> **摘要:** Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. Despite the widespread application in specialized, high-effort tasks like coding and scientific research, we highlight a critical usability gap in high-demand, mass-market applications. This position paper argues that the limited real-world adoption of LLM agents stems not only from gaps in model capabilities, but also from a fundamental tradeoff between the value an agent can provide and the costs incurred during real-world use. Hence, we call for a shift from solely optimizing model performance to a broader, utility-driven perspective: evaluating agents through the lens of the overall agentic return on investment (Agent ROI). By identifying key factors that determine Agentic ROI--information quality, agent time, and cost--we posit a zigzag development trajectory in optimizing agentic ROI: first scaling up to improve the information quality, then scaling down to minimize the time and cost. We outline the roadmap across different development stages to bridge the current usability gaps, aiming to make LLM agents truly scalable, accessible, and effective in real-world contexts.
>
---
#### [new 116] Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出一种无害数据后门攻击方法，用于越狱大型语言模型。针对现有攻击易被检测且隐蔽性差的问题，通过无害QA对训练模型过拟合良性前缀，触发后利用模型自身能力生成有害响应，并优化触发器。实验显示其在LLaMA-3和Qwen等模型上有效，成功率超85%。**

- **链接: [http://arxiv.org/pdf/2505.17601v1](http://arxiv.org/pdf/2505.17601v1)**

> **作者:** Jiawei Kong; Hao Fang; Xiaochen Yang; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Yaowei Wang; Min Zhang
>
> **摘要:** Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.
>
---
#### [new 117] ReasoningShield: Content Safety Detection over Reasoning Traces of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ReasoningShield，解决大推理模型推理轨迹中隐藏安全风险检测问题。现有工具对QA对有效但无法检测推理过程风险。作者定义QT moderation任务，构建含8000+样本的高质数据集，开发基于1B/3B模型的轻量检测器，F1超0.92，且能处理传统QA对，资源公开。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17244v1](http://arxiv.org/pdf/2505.17244v1)**

> **作者:** Changyi Li; Jiayi Wang; Xudong Pan; Geng Hong; Min Yang
>
> **摘要:** Large Reasoning Models (LRMs) are transforming the AI landscape with advanced reasoning capabilities. While the generated reasoning traces enhance model transparency, they can still contain unsafe content, even when the final answer appears safe. Existing moderation tools, primarily designed for question-answer (QA) pairs, are empirically ineffective at detecting hidden risks embedded in reasoning traces. After identifying the key challenges, we formally define the question-thought (QT) moderation task and propose ReasoningShield, the first safety detection model tailored to identify potential risks in the reasoning trace before reaching the final answer. To construct the model, we synthesize a high-quality reasoning safety detection dataset comprising over 8,000 question-thought pairs spanning ten risk categories and three safety levels. Our dataset construction process incorporates a comprehensive human-AI collaborative annotation pipeline, which achieves over 93% annotation accuracy while significantly reducing human costs. On a diverse set of in-distribution and out-of-distribution benchmarks, ReasoningShield outperforms mainstream content safety moderation models in identifying risks within reasoning traces, with an average F1 score exceeding 0.92. Notably, despite being trained on our QT dataset only, ReasoningShield also demonstrates competitive performance in detecting unsafe question-answer pairs on traditional benchmarks, rivaling baselines trained on 10 times larger datasets and base models, which strongly validates the quality of our dataset. Furthermore, ReasoningShield is built upon compact 1B/3B base models to facilitate lightweight deployment and provides human-friendly risk analysis by default. To foster future research, we publicly release all the resources.
>
---
#### [new 118] Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自动事实核查任务，旨在解决RAG模型在处理不同可信度来源的冲突证据时的可靠性问题。研究提出CONFACT数据集，系统评估现有RAG方法的不足，并通过整合媒体背景信息到检索与生成阶段提升冲突证据的分辨能力。**

- **链接: [http://arxiv.org/pdf/2505.17762v1](http://arxiv.org/pdf/2505.17762v1)**

> **作者:** Ziyu Ge; Yuhao Wu; Daniel Wai Kit Chin; Roy Ka-Wei Lee; Rui Cao
>
> **备注:** Camera-ready for IJCAI 2025, AI and Social Good
>
> **摘要:** Large Language Models (LLMs) augmented with retrieval mechanisms have demonstrated significant potential in fact-checking tasks by integrating external knowledge. However, their reliability decreases when confronted with conflicting evidence from sources of varying credibility. This paper presents the first systematic evaluation of Retrieval-Augmented Generation (RAG) models for fact-checking in the presence of conflicting evidence. To support this study, we introduce \textbf{CONFACT} (\textbf{Con}flicting Evidence for \textbf{Fact}-Checking) (Dataset available at https://github.com/zoeyyes/CONFACT), a novel dataset comprising questions paired with conflicting information from various sources. Extensive experiments reveal critical vulnerabilities in state-of-the-art RAG methods, particularly in resolving conflicts stemming from differences in media source credibility. To address these challenges, we investigate strategies to integrate media background information into both the retrieval and generation stages. Our results show that effectively incorporating source credibility significantly enhances the ability of RAG models to resolve conflicting evidence and improve fact-checking performance.
>
---
#### [new 119] Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化偏好推断任务，旨在解决大语言模型（LLMs）难以从用户行为数据中捕捉多样偏好问题。提出AlignXplore模型，结合合成数据冷启动训练与在线强化学习，通过扩展归纳推理链系统推断用户偏好，实验显示其在跨领域和格式上较基线提升11.05%，并揭示类人推理模式的生成机制。**

- **链接: [http://arxiv.org/pdf/2505.18071v1](http://arxiv.org/pdf/2505.18071v1)**

> **作者:** Jia-Nan Li; Jian Guan; Wei Wu; Rui Yan
>
> **摘要:** Large language models (LLMs) have demonstrated significant success in complex reasoning tasks such as math and coding. In contrast to these tasks where deductive reasoning predominates, inductive reasoning\textemdash the ability to derive general rules from incomplete evidence, remains underexplored. This paper investigates extended inductive reasoning in LLMs through the lens of personalized preference inference, a critical challenge in LLM alignment where current approaches struggle to capture diverse user preferences. The task demands strong inductive reasoning capabilities as user preferences are typically embedded implicitly across various interaction forms, requiring models to synthesize consistent preference patterns from scattered signals. We propose \textsc{AlignXplore}, a model that leverages extended reasoning chains to enable systematic preference inference from behavioral signals in users' interaction histories. We develop \textsc{AlignXplore} by combining cold-start training based on synthetic data with subsequent online reinforcement learning. Through extensive experiments, we demonstrate that \textsc{AlignXplore} achieves substantial improvements over the backbone model by an average of 11.05\% on in-domain and out-of-domain benchmarks, while maintaining strong generalization ability across different input formats and downstream models. Further analyses establish best practices for preference inference learning through systematic comparison of reward modeling strategies, while revealing the emergence of human-like inductive reasoning patterns during training.
>
---
#### [new 120] PersonaBOT: Bringing Customer Personas to Life with LLMs and RAG
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于客户画像生成与聊天机器人集成任务，旨在解决传统定性方法开发客户画像效率低、不可扩展的问题。研究通过LLM结合RAG技术，利用Few-Shot和CoT方法生成合成客户画像，经评估后将其融入知识库，提升聊天机器人决策支持能力，最终使系统准确率提升并获用户认可。**

- **链接: [http://arxiv.org/pdf/2505.17156v1](http://arxiv.org/pdf/2505.17156v1)**

> **作者:** Muhammed Rizwan; Lars Carlsson; Mohammad Loni
>
> **摘要:** The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar's test. In the final step, the chatbot's knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts.
>
---
#### [new 121] A Position Paper on the Automatic Generation of Machine Learning Leaderboards
- **分类: cs.CL; stat.ME**

- **简介: 该论文属于自动生成机器学习排行榜（ALG）任务，旨在解决现有方法因问题设定差异导致的可比性差和应用局限。提出统一框架标准化ALG定义，制定公平可复现的评估指南，并建议扩展结果覆盖与元数据以推动领域发展。**

- **链接: [http://arxiv.org/pdf/2505.17465v1](http://arxiv.org/pdf/2505.17465v1)**

> **作者:** Roelien C Timmer; Yufang Hou; Stephen Wan
>
> **摘要:** An important task in machine learning (ML) research is comparing prior work, which is often performed via ML leaderboards: a tabular overview of experiments with comparable conditions (e.g., same task, dataset, and metric). However, the growing volume of literature creates challenges in creating and maintaining these leaderboards. To ease this burden, researchers have developed methods to extract leaderboard entries from research papers for automated leaderboard curation. Yet, prior work varies in problem framing, complicating comparisons and limiting real-world applicability. In this position paper, we present the first overview of Automatic Leaderboard Generation (ALG) research, identifying fundamental differences in assumptions, scope, and output formats. We propose an ALG unified conceptual framework to standardise how the ALG task is defined. We offer ALG benchmarking guidelines, including recommendations for datasets and metrics that promote fair, reproducible evaluation. Lastly, we outline challenges and new directions for ALG, such as, advocating for broader coverage by including all reported results and richer metadata.
>
---
#### [new 122] Assessing GPT's Bias Towards Stigmatized Social Groups: An Intersectional Case Study on Nationality Prejudice and Psychophobia
- **分类: cs.CL**

- **简介: 该论文评估GPT系列模型对国籍与精神障碍群体的交叉偏见，通过结构化提问测试其对美籍/朝籍及精神障碍者的反应差异，发现对朝鲜裔群体存在更显著的负面偏见（尤其叠加精神障碍时），呼吁改进LLM的交叉身份理解能力。**

- **链接: [http://arxiv.org/pdf/2505.17045v1](http://arxiv.org/pdf/2505.17045v1)**

> **作者:** Afifah Kashif; Heer Patel
>
> **摘要:** Recent studies have separately highlighted significant biases within foundational large language models (LLMs) against certain nationalities and stigmatized social groups. This research investigates the ethical implications of these biases intersecting with outputs of widely-used GPT-3.5/4/4o LLMS. Through structured prompt series, we evaluate model responses to several scenarios involving American and North Korean nationalities with various mental disabilities. Findings reveal significant discrepancies in empathy levels with North Koreans facing greater negative bias, particularly when mental disability is also a factor. This underscores the need for improvements in LLMs designed with a nuanced understanding of intersectional identity.
>
---
#### [new 123] EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EVADE，首个针对电商规避内容检测的中文多模态基准，包含2833文本和13961图像样本，设计双任务评估模型推理能力，测试26个主流模型，揭示其性能差距，推动内容审核系统改进。**

- **链接: [http://arxiv.org/pdf/2505.17654v1](http://arxiv.org/pdf/2505.17654v1)**

> **作者:** Ancheng Xu; Zhihao Yang; Jingpeng Li; Guanghu Yuan; Longze Chen; Liang Yan; Jiehui Zhou; Zhen Qin; Hengyun Chang; Hamid Alinejad-Rokny; Bo Zheng; Min Yang
>
> **摘要:** E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.
>
---
#### [new 124] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation
- **分类: cs.CL**

- **简介: 该论文提出MTR-Bench，用于多轮推理评估，解决现有评测侧重单轮且缺乏数据与自动评估的问题。构建含4类40任务3600实例的基准，支持多轮交互与全自动评估，实验显示先进模型表现不足，为交互AI研究提供方向。**

- **链接: [http://arxiv.org/pdf/2505.17123v1](http://arxiv.org/pdf/2505.17123v1)**

> **作者:** Xiaoyuan Li; Keqin Bao; Yubo Ma; Moxin Li; Wenjie Wang; Rui Men; Yichang Zhang; Fuli Feng; Dayiheng Liu; Junyang Lin
>
> **备注:** Under Review
>
> **摘要:** Recent advances in Large Language Models (LLMs) have shown promising results in complex reasoning tasks. However, current evaluations predominantly focus on single-turn reasoning scenarios, leaving interactive tasks largely unexplored. We attribute it to the absence of comprehensive datasets and scalable automatic evaluation protocols. To fill these gaps, we present MTR-Bench for LLMs' Multi-Turn Reasoning evaluation. Comprising 4 classes, 40 tasks, and 3600 instances, MTR-Bench covers diverse reasoning capabilities, fine-grained difficulty granularity, and necessitates multi-turn interactions with the environments. Moreover, MTR-Bench features fully-automated framework spanning both dataset constructions and model evaluations, which enables scalable assessment without human interventions. Extensive experiments reveal that even the cutting-edge reasoning models fall short of multi-turn, interactive reasoning tasks. And the further analysis upon these results brings valuable insights for future research in interactive AI systems.
>
---
#### [new 125] Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于改进大语言模型（LLM）推理任务，旨在解决长思维链计算成本高且未必提升性能的问题。研究发现短思维链更准确（最高提升34.5%），提出short-m@k方法：并行生成k个链，取前m个结果投票，减少计算（最多省40% token）且更快（提速33%）。实验表明该方法在低算力下表现更优，并验证短链训练更有效，挑战了长链必优的假设。**

- **链接: [http://arxiv.org/pdf/2505.17813v1](http://arxiv.org/pdf/2505.17813v1)**

> **作者:** Michael Hassid; Gabriel Synnaeve; Yossi Adi; Roy Schwartz
>
> **备注:** Preprint. Under review
>
> **摘要:** Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). Inspired by our results, we finetune an LLM using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results.
>
---
#### [new 126] Development and Validation of Engagement and Rapport Scales for Evaluating User Experience in Multimodal Dialogue Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态对话系统用户体验评估任务，旨在开发量化用户参与度与情感联结的量表，解决外语学习场景下人机交互质量的客观评价问题。研究设计基于教育与心理学理论，通过74名日语使用者与真人教师及AI对话系统的对比实验，验证量表的信效度及区分能力，结果显示量表能有效捕捉人类与AI交互体验的差异。**

- **链接: [http://arxiv.org/pdf/2505.17075v1](http://arxiv.org/pdf/2505.17075v1)**

> **作者:** Fuma Kurata; Mao Saeki; Masaki Eguchi; Shungo Suzuki; Hiroaki Takatsu; Yoichi Matsuyama
>
> **摘要:** This study aimed to develop and validate two scales of engagement and rapport to evaluate the user experience quality with multimodal dialogue systems in the context of foreign language learning. The scales were designed based on theories of engagement in educational psychology, social psychology, and second language acquisition.Seventy-four Japanese learners of English completed roleplay and discussion tasks with trained human tutors and a dialog agent. After each dialogic task was completed, they responded to the scales of engagement and rapport. The validity and reliability of the scales were investigated through two analyses. We first conducted analysis of Cronbach's alpha coefficient and a series of confirmatory factor analyses to test the structural validity of the scales and the reliability of our designed items. We then compared the scores of engagement and rapport between the dialogue with human tutors and the one with a dialogue agent. The results revealed that our scales succeeded in capturing the difference in the dialogue experience quality between the human interlocutors and the dialogue agent from multiple perspectives.
>
---
#### [new 127] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）评估器中非传递偏好问题，提出基于图论的分析框架，量化非传递性并设计ELSPR过滤策略，通过去除矛盾数据提升模型一致性。实验显示过滤后模型非传递性降低13.78%，与人类评估更接近。**

- **链接: [http://arxiv.org/pdf/2505.17691v1](http://arxiv.org/pdf/2505.17691v1)**

> **作者:** Yan Yu; Yilun Liu; Minggui He; Shimin Tao; Weibin Meng; Xinhua Yang; Li Zhang; Hongxia Ma; Chang Su; Hao Yang; Fuliang Li
>
> **摘要:** Large language models (LLMs) are widely used as evaluators for open-ended tasks, while previous research has emphasized biases in LLM evaluations, the issue of non-transitivity in pairwise comparisons remains unresolved: non-transitive preferences for pairwise comparisons, where evaluators prefer A over B, B over C, but C over A. Our results suggest that low-quality training data may reduce the transitivity of preferences generated by the Evaluator LLM. To address this, We propose a graph-theoretic framework to analyze and mitigate this problem by modeling pairwise preferences as tournament graphs. We quantify non-transitivity and introduce directed graph structural entropy to measure the overall clarity of preferences. Our analysis reveals significant non-transitivity in advanced Evaluator LLMs (with Qwen2.5-Max exhibiting 67.96%), as well as high entropy values (0.8095 for Qwen2.5-Max), reflecting low overall clarity of preferences. To address this issue, we designed a filtering strategy, ELSPR, to eliminate preference data that induces non-transitivity, retaining only consistent and transitive preference data for model fine-tuning. Experiments demonstrate that models fine-tuned with filtered data reduce non-transitivity by 13.78% (from 64.28% to 50.50%), decrease structural entropy by 0.0879 (from 0.8113 to 0.7234), and align more closely with human evaluators (human agreement rate improves by 0.6% and Spearman correlation increases by 0.01).
>
---
#### [new 128] Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SDForger框架，利用LLM生成高质量多变量时间序列。针对传统方法数据需求高、计算复杂的问题，其将时间序列转为表格嵌入并编码为文本，通过少量样本微调LLM，生成保留统计特性和时序动态的合成数据，优于现有方法并支持多模态整合。**

- **链接: [http://arxiv.org/pdf/2505.17103v1](http://arxiv.org/pdf/2505.17103v1)**

> **作者:** Cécile Rousseau; Tobia Boschi; Giandomenico Cornacchia; Dhaval Salwala; Alessandra Pascale; Juan Bernabe Moreno
>
> **摘要:** SDForger is a flexible and efficient framework for generating high-quality multivariate time series using LLMs. Leveraging a compact data representation, SDForger provides synthetic time series generation from a few samples and low-computation fine-tuning of any autoregressive LLM. Specifically, the framework transforms univariate and multivariate signals into tabular embeddings, which are then encoded into text and used to fine-tune the LLM. At inference, new textual embeddings are sampled and decoded into synthetic time series that retain the original data's statistical properties and temporal dynamics. Across a diverse range of datasets, SDForger outperforms existing generative models in many scenarios, both in similarity-based evaluations and downstream forecasting tasks. By enabling textual conditioning in the generation process, SDForger paves the way for multimodal modeling and the streamlined integration of time series with textual information. SDForger source code will be open-sourced soon.
>
---
#### [new 129] Beyond Distillation: Pushing the Limits of Medical LLM Reasoning with Minimalist Rule-Based RL
- **分类: cs.CL; cs.AI**

- **简介: 论文提出AlphaMed，通过极简规则强化学习提升医学LLM推理能力，无需依赖蒸馏CoT数据或监督微调。解决传统方法对昂贵链式思考数据的依赖问题，在六项医学QA基准中达SOTA，超越大型模型。分析数据量、多样性及难度对推理的影响，强调数据信息量的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.17952v1](http://arxiv.org/pdf/2505.17952v1)**

> **作者:** Che Liu; Haozhe Wang; Jiazhen Pan; Zhongwei Wan; Yong Dai; Fangzhen Lin; Wenjia Bai; Daniel Rueckert; Rossella Arcucci
>
> **备注:** Under Review
>
> **摘要:** Improving performance on complex tasks and enabling interpretable decision making in large language models (LLMs), especially for clinical applications, requires effective reasoning. Yet this remains challenging without supervised fine-tuning (SFT) on costly chain-of-thought (CoT) data distilled from closed-source models (e.g., GPT-4o). In this work, we present AlphaMed, the first medical LLM to show that reasoning capability can emerge purely through reinforcement learning (RL), using minimalist rule-based rewards on public multiple-choice QA datasets, without relying on SFT or distilled CoT data. AlphaMed achieves state-of-the-art results on six medical QA benchmarks, outperforming models trained with conventional SFT+RL pipelines. On challenging benchmarks (e.g., MedXpert), AlphaMed even surpasses larger or closed-source models such as DeepSeek-V3-671B and Claude-3.5-Sonnet. To understand the factors behind this success, we conduct a comprehensive data-centric analysis guided by three questions: (i) Can minimalist rule-based RL incentivize reasoning without distilled CoT supervision? (ii) How do dataset quantity and diversity impact reasoning? (iii) How does question difficulty shape the emergence and generalization of reasoning? Our findings show that dataset informativeness is a key driver of reasoning performance, and that minimalist RL on informative, multiple-choice QA data is effective at inducing reasoning without CoT supervision. We also observe divergent trends across benchmarks, underscoring limitations in current evaluation and the need for more challenging, reasoning-oriented medical QA benchmarks.
>
---
#### [new 130] Language Matters: How Do Multilingual Input and Reasoning Paths Affect Large Reasoning Models?
- **分类: cs.CL**

- **简介: 该论文研究多语言大型推理模型（LRMs）的推理语言选择及其影响。发现模型处理多语言输入时倾向默认使用高资源语言（如英语），导致低资源语言性能下降；强制同语言推理则使推理任务变差但文化任务提升。通过跨任务评估揭示模型语言偏见，推动开发更公平的多语言模型。**

- **链接: [http://arxiv.org/pdf/2505.17407v1](http://arxiv.org/pdf/2505.17407v1)**

> **作者:** Zhi Rui Tam; Cheng-Kuang Wu; Yu Ying Chiu; Chieh-Yen Lin; Yun-Nung Chen; Hung-yi Lee
>
> **摘要:** Large reasoning models (LRMs) have demonstrated impressive performance across a range of reasoning tasks, yet little is known about their internal reasoning processes in multilingual settings. We begin with a critical question: {\it In which language do these models reason when solving problems presented in different languages?} Our findings reveal that, despite multilingual training, LRMs tend to default to reasoning in high-resource languages (e.g., English) at test time, regardless of the input language. When constrained to reason in the same language as the input, model performance declines, especially for low-resource languages. In contrast, reasoning in high-resource languages generally preserves performance. We conduct extensive evaluations across reasoning-intensive tasks (MMMLU, MATH-500) and non-reasoning benchmarks (CulturalBench, LMSYS-toxic), showing that the effect of language choice varies by task type: input-language reasoning degrades performance on reasoning tasks but benefits cultural tasks, while safety evaluations exhibit language-specific behavior. By exposing these linguistic biases in LRMs, our work highlights a critical step toward developing more equitable models that serve users across diverse linguistic backgrounds.
>
---
#### [new 131] First Finish Search: Efficient Test-Time Scaling in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决现有测试时间缩放（TTS）方法计算资源高、延迟大的问题。提出无训练的First Finish Search（FFS），并行生成多个样本，优先返回最先完成的，提升推理效率与准确率，在多个模型和数据集上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.18149v1](http://arxiv.org/pdf/2505.18149v1)**

> **作者:** Aradhye Agarwal; Ayan Sengupta; Tanmoy Chakraborty
>
> **摘要:** Test-time scaling (TTS), which involves dynamic allocation of compute during inference, offers a promising way to improve reasoning in large language models. While existing TTS methods work well, they often rely on long decoding paths or require a large number of samples to be generated, increasing the token usage and inference latency. We observe the surprising fact that for reasoning tasks, shorter traces are much more likely to be correct than longer ones. Motivated by this, we introduce First Finish Search (FFS), a training-free parallel decoding strategy that launches $n$ independent samples and returns as soon as any one completes. We evaluate FFS alongside simple decoding, beam search, majority voting, and budget forcing on four reasoning models (DeepSeek-R1, R1-Distill-Qwen-32B, QwQ-32B and Phi-4-Reasoning-Plus) and across four datasets (AIME24, AIME25-I, AIME25-II and GPQA Diamond). With DeepSeek-R1, FFS achieves $82.23\%$ accuracy on the AIME datasets, a $15\%$ improvement over DeepSeek-R1's standalone accuracy, nearly matching OpenAI's o4-mini performance. Our theoretical analysis explains why stopping at the shortest trace is likely to yield a correct answer and identifies the conditions under which early stopping may be suboptimal. The elegance and simplicity of FFS demonstrate that straightforward TTS strategies can perform remarkably well, revealing the untapped potential of simple approaches at inference time.
>
---
#### [new 132] SELF: Self-Extend the Context Length With Logistic Growth Function
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于长文本处理任务，解决大语言模型在超长上下文中的性能下降问题。提出SELF方法，通过逻辑增长函数动态调整分组大小，优化位置编码。实验显示其在LEval和LongBench任务中较LongLM提升5.4%-12%。**

- **链接: [http://arxiv.org/pdf/2505.17296v1](http://arxiv.org/pdf/2505.17296v1)**

> **作者:** Phat Thanh Dang; Saahil Thoppay; Wang Yang; Qifan Wang; Vipin Chaudhary; Xiaotian Han
>
> **备注:** 11 pages, 5 figures, 3 tables
>
> **摘要:** Large language models suffer issues when operated on long contexts that are larger than their training context length due to the standard position encoding for tokens in the attention layer. Tokens a long distance apart will rarely have an effect on each other and long prompts yield unexpected results. To solve this problem, we propose SELF (Self-Extend the Context Length With Logistic Growth Function): a solution of grouping consecutive tokens at varying group sizes using a logistic capacity equation combined with a constant group size at smaller relative distances. Our model had an increase in performance of up to 12% compared to the LongLM extension method in LEval (specifically on the Qwen model). On summarization related tasks in LongBench, our model performed up to 6.4% better than LongLM (specifically on the Llama-2-7b model). On reading comprehension tasks from LEval, our model performed up to 5.4% better than the LongLM. Our code is available at https://github.com/alexeipc/SELF-LLM.
>
---
#### [new 133] Shallow Preference Signals: Large Language Model Aligns Even Better with Truncated Data?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对齐任务，发现人类偏好信号主要集中在响应早期token（"浅层信号"）。通过截断数据集（如保留前40%）训练模型，性能优于完整数据，提出解码策略优化对齐与效率，指出需全面考虑完整响应以提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2505.17122v1](http://arxiv.org/pdf/2505.17122v1)**

> **作者:** Xuan Qi; Jiahao Qiu; Xinzhe Juan; Yue Wu; Mengdi Wang
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Aligning large language models (LLMs) with human preferences remains a key challenge in AI. Preference-based optimization methods, such as Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on human-annotated datasets to improve alignment. In this work, we identify a crucial property of the existing learning method: the distinguishing signal obtained in preferred responses is often concentrated in the early tokens. We refer to this as shallow preference signals. To explore this property, we systematically truncate preference datasets at various points and train both reward models and DPO models on the truncated data. Surprisingly, models trained on truncated datasets, retaining only the first half or fewer tokens, achieve comparable or even superior performance to those trained on full datasets. For example, a reward model trained on the Skywork-Reward-Preference-80K-v0.2 dataset outperforms the full dataset when trained on a 40\% truncated dataset. This pattern is consistent across multiple datasets, suggesting the widespread presence of shallow preference signals. We further investigate the distribution of the reward signal through decoding strategies. We consider two simple decoding strategies motivated by the shallow reward signal observation, namely Length Control Decoding and KL Threshold Control Decoding, which leverage shallow preference signals to optimize the trade-off between alignment and computational efficiency. The performance is even better, which again validates our hypothesis. The phenomenon of shallow preference signals highlights potential issues in LLM alignment: existing alignment methods often focus on aligning only the initial tokens of responses, rather than considering the full response. This could lead to discrepancies with real-world human preferences, resulting in suboptimal alignment performance.
>
---
#### [new 134] Exploring the Effect of Segmentation and Vocabulary Size on Speech Tokenization for Speech Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究语音分词中分段宽度和词汇规模对语音语言模型（SLMs）的影响，旨在优化离散表示以提升模型性能。通过对比固定/可变分段及不同聚类规模，发现中等粗分段与较大集群能提升零样本语音理解效果，最佳模型减少50%训练数据与70%训练时间，强调多token融合的重要性。**

- **链接: [http://arxiv.org/pdf/2505.17446v1](http://arxiv.org/pdf/2505.17446v1)**

> **作者:** Shunsuke Kando; Yusuke Miyao; Shinnosuke Takamichi
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** The purpose of speech tokenization is to transform a speech signal into a sequence of discrete representations, serving as the foundation for speech language models (SLMs). While speech tokenization has many options, their effect on the performance of SLMs remains unclear. This paper investigates two key aspects of speech tokenization: the segmentation width and the cluster size of discrete units. First, we segment speech signals into fixed/variable widths and pooled representations. We then train K-means models in multiple cluster sizes. Through the evaluation on zero-shot spoken language understanding benchmarks, we find the positive effect of moderately coarse segmentation and bigger cluster size. Notably, among the best-performing models, the most efficient one achieves a 50% reduction in training data and a 70% decrease in training runtime. Our analysis highlights the importance of combining multiple tokens to enhance fine-grained spoken language understanding.
>
---
#### [new 135] Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs
- **分类: cs.CL**

- **简介: 该论文提出基准"Fann or Flop"，评估LLMs对阿拉伯诗歌的跨时代、跨体裁理解能力，填补其在深层语义、隐喻、韵律及文化背景解读方面的研究空白。通过评测显示多数模型表现不佳，开源资源推动阿拉伯语言模型改进。**

- **链接: [http://arxiv.org/pdf/2505.18152v1](http://arxiv.org/pdf/2505.18152v1)**

> **作者:** Wafa Alghallabi; Ritesh Thawkar; Sara Ghaboura; Ketan More; Omkar Thawakar; Hisham Cholakkal; Salman Khan; Rao Muhammad Anwer
>
> **备注:** Github:https://github.com/mbzuai-oryx/FannOrFlop, Dataset:https://huggingface.co/datasets/omkarthawakar/FannOrFlop
>
> **摘要:** Arabic poetry stands as one of the most sophisticated and culturally embedded forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce `Fann or Flop`, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in twelve historical eras, covering 21 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM is in understanding classical Arabic through the Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release `Fann or Flop` along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: https://github.com/mbzuai-oryx/FannOrFlop.
>
---
#### [new 136] After Retrieval, Before Generation: Enhancing the Trustworthiness of Large Language Models in RAG
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于RAG（检索增强生成）任务，旨在解决大语言模型在平衡内部参数知识与外部检索知识时的可信度问题，尤其在两者冲突或不可靠时缺乏统一策略。研究构建了包含36,266个问题的TRD数据集，分析现有方法局限，提出BRIDGE框架，通过自适应加权（soft bias）和决策树机制动态选择最优响应策略，实验显示其准确率提升5-15%，实现多场景均衡优化。**

- **链接: [http://arxiv.org/pdf/2505.17118v1](http://arxiv.org/pdf/2505.17118v1)**

> **作者:** Xinbang Dai; Huikang Hu; Yuncheng Hua; Jiaqi Li; Yongrui Chen; Rihui Jin; Nan Hu; Guilin Qi
>
> **备注:** 24 pages, 8 figures
>
> **摘要:** Retrieval-augmented generation (RAG) systems face critical challenges in balancing internal (parametric) and external (retrieved) knowledge, especially when these sources conflict or are unreliable. To analyze these scenarios comprehensively, we construct the Trustworthiness Response Dataset (TRD) with 36,266 questions spanning four RAG settings. We reveal that existing approaches address isolated scenarios-prioritizing one knowledge source, naively merging both, or refusing answers-but lack a unified framework to handle different real-world conditions simultaneously. Therefore, we propose the BRIDGE framework, which dynamically determines a comprehensive response strategy of large language models (LLMs). BRIDGE leverages an adaptive weighting mechanism named soft bias to guide knowledge collection, followed by a Maximum Soft-bias Decision Tree to evaluate knowledge and select optimal response strategies (trust internal/external knowledge, or refuse). Experiments show BRIDGE outperforms baselines by 5-15% in accuracy while maintaining balanced performance across all scenarios. Our work provides an effective solution for LLMs' trustworthy responses in real-world RAG applications.
>
---
#### [new 137] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型参数专业化对知识存储的影响，旨在解决如何优化参数利用以提升模型效率。通过分析20个开源模型，发现先进模型的MLP参数更专业化（专注特定知识类型），实验验证这种分布提升知识利用效率，并通过因果训练确认其关键作用。**

- **链接: [http://arxiv.org/pdf/2505.17260v1](http://arxiv.org/pdf/2505.17260v1)**

> **作者:** Yihuai Hong; Yiran Zhao; Wei Tang; Yang Deng; Yu Rong; Wenxuan Zhang
>
> **摘要:** Over time, a growing wave of large language models from various series has been introduced to the community. Researchers are striving to maximize the performance of language models with constrained parameter sizes. However, from a microscopic perspective, there has been limited research on how to better store knowledge in model parameters, particularly within MLPs, to enable more effective utilization of this knowledge by the model. In this work, we analyze twenty publicly available open-source large language models to investigate the relationship between their strong performance and the way knowledge is stored in their corresponding MLP parameters. Our findings reveal that as language models become more advanced and demonstrate stronger knowledge capabilities, their parameters exhibit increased specialization. Specifically, parameters in the MLPs tend to be more focused on encoding similar types of knowledge. We experimentally validate that this specialized distribution of knowledge contributes to improving the efficiency of knowledge utilization in these models. Furthermore, by conducting causal training experiments, we confirm that this specialized knowledge distribution plays a critical role in improving the model's efficiency in leveraging stored knowledge.
>
---
#### [new 138] Not Minds, but Signs: Reframing LLMs through Semiotics
- **分类: cs.CL**

- **简介: 论文从符号学视角重新定义LLMs，挑战其作为认知系统的主流定位，提出其核心功能是通过概率关联重组语言符号参与文化意义生成，而非具备思维。通过理论分析与案例，强调其作为符号工具在创造、教育等领域的应用，构建严谨伦理框架，凸显其文化参与而非模拟人类心智。**

- **链接: [http://arxiv.org/pdf/2505.17080v1](http://arxiv.org/pdf/2505.17080v1)**

> **作者:** Davide Picca
>
> **摘要:** This paper challenges the prevailing tendency to frame Large Language Models (LLMs) as cognitive systems, arguing instead for a semiotic perspective that situates these models within the broader dynamics of sign manipulation and meaning-making. Rather than assuming that LLMs understand language or simulate human thought, we propose that their primary function is to recombine, recontextualize, and circulate linguistic forms based on probabilistic associations. By shifting from a cognitivist to a semiotic framework, we avoid anthropomorphism and gain a more precise understanding of how LLMs participate in cultural processes, not by thinking, but by generating texts that invite interpretation. Through theoretical analysis and practical examples, the paper demonstrates how LLMs function as semiotic agents whose outputs can be treated as interpretive acts, open to contextual negotiation and critical reflection. We explore applications in literature, philosophy, education, and cultural production, emphasizing how LLMs can serve as tools for creativity, dialogue, and critical inquiry. The semiotic paradigm foregrounds the situated, contingent, and socially embedded nature of meaning, offering a more rigorous and ethically aware framework for studying and using LLMs. Ultimately, this approach reframes LLMs as technological participants in an ongoing ecology of signs. They do not possess minds, but they alter how we read, write, and make meaning, compelling us to reconsider the foundations of language, interpretation, and the role of artificial systems in the production of knowledge.
>
---
#### [new 139] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally
- **分类: cs.CL; cs.AI; cs.CY; q-fin.CP; q-fin.GN**

- **简介: 该论文提出统一框架解析全球央行沟通，构建全球最大WCB数据集（25国28年，38万句），定义立场检测、时间分类和不确定性估计任务，测试多种模型验证跨银行训练更优，提升政策解读准确性，减少误读对弱势群体的负面影响。**

- **链接: [http://arxiv.org/pdf/2505.17048v1](http://arxiv.org/pdf/2505.17048v1)**

> **作者:** Agam Shah; Siddhant Sukhani; Huzaifa Pardawala; Saketh Budideti; Riya Bhadani; Rudra Gopal; Siddhartha Somani; Michael Galarnyk; Soungmin Lee; Arnav Hiray; Akshar Ravichandran; Eric Kim; Pranav Aluru; Joshua Zhang; Sebastian Jaskowski; Veer Guda; Meghaj Tarte; Liqin Ye; Spencer Gosden; Rutwik Routu; Rachel Yuh; Sloka Chava; Sahasra Chava; Dylan Patrick Kelly; Aiden Chiang; Harsit Mittal; Sudheer Chava
>
> **摘要:** Central banks around the world play a crucial role in maintaining economic stability. Deciphering policy implications in their communications is essential, especially as misinterpretations can disproportionately impact vulnerable populations. To address this, we introduce the World Central Banks (WCB) dataset, the most comprehensive monetary policy corpus to date, comprising over 380k sentences from 25 central banks across diverse geographic regions, spanning 28 years of historical data. After uniformly sampling 1k sentences per bank (25k total) across all available years, we annotate and review each sentence using dual annotators, disagreement resolutions, and secondary expert reviews. We define three tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation, with each sentence annotated for all three. We benchmark seven Pretrained Language Models (PLMs) and nine Large Language Models (LLMs) (Zero-Shot, Few-Shot, and with annotation guide) on these tasks, running 15,075 benchmarking experiments. We find that a model trained on aggregated data across banks significantly surpasses a model trained on an individual bank's data, confirming the principle "the whole is greater than the sum of its parts." Additionally, rigorous human evaluations, error analyses, and predictive tasks validate our framework's economic utility. Our artifacts are accessible through the HuggingFace and GitHub under the CC-BY-NC-SA 4.0 license.
>
---
#### [new 140] AI-Augmented LLMs Achieve Therapist-Level Responses in Motivational Interviewing
- **分类: cs.CL; H.1.2; I.2.7**

- **简介: 该论文提出优化LLMs在动机访谈（MI）中的治疗能力，解决其临床沟通的不足。通过开发用户感知质量（UPQ）评估框架，识别17个MI行为指标，改进GPT-4提示工程，提升其反思和同理心，但复杂情感处理仍受限。**

- **链接: [http://arxiv.org/pdf/2505.17380v1](http://arxiv.org/pdf/2505.17380v1)**

> **作者:** Yinghui Huang; Yuxuan Jiang; Hui Liu; Yixin Cai; Weiqing Li; Xiangen Hu
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Large language models (LLMs) like GPT-4 show potential for scaling motivational interviewing (MI) in addiction care, but require systematic evaluation of therapeutic capabilities. We present a computational framework assessing user-perceived quality (UPQ) through expected and unexpected MI behaviors. Analyzing human therapist and GPT-4 MI sessions via human-AI collaboration, we developed predictive models integrating deep learning and explainable AI to identify 17 MI-consistent (MICO) and MI-inconsistent (MIIN) behavioral metrics. A customized chain-of-thought prompt improved GPT-4's MI performance, reducing inappropriate advice while enhancing reflections and empathy. Although GPT-4 remained marginally inferior to therapists overall, it demonstrated superior advice management capabilities. The model achieved measurable quality improvements through prompt engineering, yet showed limitations in addressing complex emotional nuances. This framework establishes a pathway for optimizing LLM-based therapeutic tools through targeted behavioral metric analysis and human-AI co-evaluation. Findings highlight both the scalability potential and current constraints of LLMs in clinical communication applications.
>
---
#### [new 141] Medalyze: Lightweight Medical Report Summarization Application Using FLAN-T5-Large
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Medalyze，基于FLAN-T5-Large模型的医疗文本处理应用，解决医疗报告复杂术语和理解难题。通过三模型架构实现报告摘要、对话健康问题提取及关键问题识别，部署跨平台系统，实验显示其领域任务性能超GPT-4，提供轻量、隐私保护的医疗信息解决方案。**

- **链接: [http://arxiv.org/pdf/2505.17059v1](http://arxiv.org/pdf/2505.17059v1)**

> **作者:** Van-Tinh Nguyen; Hoang-Duong Pham; Thanh-Hai To; Cong-Tuan Hung Do; Thi-Thu-Trang Dong; Vu-Trung Duong Le; Van-Phuc Hoang
>
> **备注:** 12 pages, 8 figures. Submitted to IEEE Access for review. Preliminary version posted for early dissemination and feedback
>
> **摘要:** Understanding medical texts presents significant challenges due to complex terminology and context-specific language. This paper introduces Medalyze, an AI-powered application designed to enhance the comprehension of medical texts using three specialized FLAN-T5-Large models. These models are fine-tuned for (1) summarizing medical reports, (2) extracting health issues from patient-doctor conversations, and (3) identifying the key question in a passage. Medalyze is deployed across a web and mobile platform with real-time inference, leveraging scalable API and YugabyteDB. Experimental evaluations demonstrate the system's superior summarization performance over GPT-4 in domain-specific tasks, based on metrics like BLEU, ROUGE-L, BERTScore, and SpaCy Similarity. Medalyze provides a practical, privacy-preserving, and lightweight solution for improving information accessibility in healthcare.
>
---
#### [new 142] Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型幻觉检测任务，旨在解决高质量幻觉文本难以识别的问题。提出基于课程学习的DPO方法，利用合成幻觉样本作为负面训练数据，通过逐步提升样本难度优化模型，实现稳定学习与性能提升。**

- **链接: [http://arxiv.org/pdf/2505.17558v1](http://arxiv.org/pdf/2505.17558v1)**

> **作者:** Shrey Pandit; Ashwin Vinod; Liu Leqi; Ying Ding
>
> **备注:** Code and dataset are available at https://teachingwithlies.github.io/
>
> **摘要:** Aligning large language models (LLMs) to accurately detect hallucinations remains a significant challenge due to the sophisticated nature of hallucinated text. Recognizing that hallucinated samples typically exhibit higher deceptive quality than traditional negative samples, we use these carefully engineered hallucinations as negative examples in the DPO alignment procedure. Our method incorporates a curriculum learning strategy, gradually transitioning the training from easier samples, identified based on the greatest reduction in probability scores from independent fact checking models, to progressively harder ones. This structured difficulty scaling ensures stable and incremental learning. Experimental evaluation demonstrates that our HaluCheck models, trained with curriculum DPO approach and high quality negative samples, significantly improves model performance across various metrics, achieving improvements of upto 24% on difficult benchmarks like MedHallu and HaluEval. Additionally, HaluCheck models demonstrate robustness in zero-shot settings, significantly outperforming larger state-of-the-art models across various benchmarks.
>
---
#### [new 143] Language models should be subject to repeatable, open, domain-contextualized hallucination benchmarking
- **分类: cs.CL**

- **简介: 该论文提出语言模型需通过可重复、开放且领域相关的幻觉基准测试，解决现有评估方法缺乏科学性和实用性的不足。工作包括构建幻觉分类法及案例研究，证明专家参与数据创建对确保评估效度的重要性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17345v1](http://arxiv.org/pdf/2505.17345v1)**

> **作者:** Justin D. Norman; Michael U. Rivera; D. Alex Hughes
>
> **备注:** 9 pages
>
> **摘要:** Plausible, but inaccurate, tokens in model-generated text are widely believed to be pervasive and problematic for the responsible adoption of language models. Despite this concern, there is little scientific work that attempts to measure the prevalence of language model hallucination in a comprehensive way. In this paper, we argue that language models should be evaluated using repeatable, open, and domain-contextualized hallucination benchmarking. We present a taxonomy of hallucinations alongside a case study that demonstrates that when experts are absent from the early stages of data creation, the resulting hallucination metrics lack validity and practical utility.
>
---
#### [new 144] Benchmarking Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文基准测试了VITS和SBV2JE模型在日语角色语音合成任务中的表现，解决音调敏感与风格多样化的挑战。通过三个角色数据集评估自然度、可懂度及说话人一致性，结果显示SBV2JE在自然度（MOS 4.37近似人类水平）和WER上优于VITS，但计算需求更高。**

- **链接: [http://arxiv.org/pdf/2505.17320v1](http://arxiv.org/pdf/2505.17320v1)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **摘要:** Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper benchmarks two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands.
>
---
#### [new 145] Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI伦理领域，旨在缓解大语言模型（LLMs）中的性别偏见问题。研究提出通过生成结构相同但主角性别不同的故事情境，对比模型的道德判断，引导其形成平衡中立的输出，并利用Direct Preference Optimization（DPO）优化模型。实验表明该方法有效降低偏见且保持模型性能。**

- **链接: [http://arxiv.org/pdf/2505.17217v1](http://arxiv.org/pdf/2505.17217v1)**

> **作者:** Kangda Wei; Hasnat Md Abdullah; Ruihong Huang
>
> **摘要:** Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data.
>
---
#### [new 146] Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究长上下文问答任务中的"针在草堆"问题，探讨金标上下文长度对LLMs的影响。发现短金标上下文显著降低模型性能并加剧位置敏感性，实验涵盖多领域及七种模型，强调需设计更鲁棒的LLM系统。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18148v1](http://arxiv.org/pdf/2505.18148v1)**

> **作者:** Owen Bianchi; Mathew J. Koretsky; Maya Willey; Chelsea X. Alvarado; Tanay Nayak; Adi Asija; Nicole Kuznetsov; Mike A. Nalls; Faraz Faghri; Daniel Khashabi
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) face significant challenges with needle-in-a-haystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size has received little attention. We address this gap by systematically studying how variations in gold context length impact LLM performance on long-context question answering tasks. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This pattern holds across three diverse domains (general knowledge, biomedical reasoning, and mathematical reasoning) and seven state-of-the-art LLMs of various sizes and architectures. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems.
>
---
#### [new 147] DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于领域专用问答任务，针对现有RAG框架在异构数据整合与推理一致性上的不足，提出DO-RAG框架。其通过多级知识图构建与语义向量检索融合，结合动态知识图与链式思维架构，提升事实准确性与检索精度，实验显示其在数据库/电气领域表现显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.17058v1](http://arxiv.org/pdf/2505.17058v1)**

> **作者:** David Osei Opoku; Ming Sheng; Yong Zhang
>
> **备注:** 6 pages, 5 figures;
>
> **摘要:** Domain-specific QA systems require not just generative fluency but high factual accuracy grounded in structured expert knowledge. While recent Retrieval-Augmented Generation (RAG) frameworks improve context recall, they struggle with integrating heterogeneous data and maintaining reasoning consistency. To address these challenges, we propose DO-RAG, a scalable and customizable hybrid QA framework that integrates multi-level knowledge graph construction with semantic vector retrieval. Our system employs a novel agentic chain-of-thought architecture to extract structured relationships from unstructured, multimodal documents, constructing dynamic knowledge graphs that enhance retrieval precision. At query time, DO-RAG fuses graph and vector retrieval results to generate context-aware responses, followed by hallucination mitigation via grounded refinement. Experimental evaluations in the database and electrical domains show near-perfect recall and over 94% answer relevancy, with DO-RAG outperforming baseline frameworks by up to 33.38%. By combining traceability, adaptability, and performance efficiency, DO-RAG offers a reliable foundation for multi-domain, high-precision QA at scale.
>
---
#### [new 148] Mechanistic Interpretability of GPT-like Models on Summarization Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型机制可解释性研究，针对GPT-like模型在摘要任务中的内部运作机制分析不足的问题。通过对比预训练与微调模型，量化注意力模式及激活变化，定位关键中间层（2、3、5层）和注意力头（62%熵降低），揭示模型聚焦信息选择的机制。基于此设计针对性LoRA适配方案，提升性能并减少训练时间，填补了摘要任务解释性研究的空白。**

- **链接: [http://arxiv.org/pdf/2505.17073v1](http://arxiv.org/pdf/2505.17073v1)**

> **作者:** Anurag Mishra
>
> **备注:** 8 pages (6 content + 2 references/appendix), 6 figures, 2 tables; under review for the ACL 2025 Student Research Workshop
>
> **摘要:** Mechanistic interpretability research seeks to reveal the inner workings of large language models, yet most work focuses on classification or generative tasks rather than summarization. This paper presents an interpretability framework for analyzing how GPT-like models adapt to summarization tasks. We conduct differential analysis between pre-trained and fine-tuned models, quantifying changes in attention patterns and internal activations. By identifying specific layers and attention heads that undergo significant transformation, we locate the "summarization circuit" within the model architecture. Our findings reveal that middle layers (particularly 2, 3, and 5) exhibit the most dramatic changes, with 62% of attention heads showing decreased entropy, indicating a shift toward focused information selection. We demonstrate that targeted LoRA adaptation of these identified circuits achieves significant performance improvement over standard LoRA fine-tuning while requiring fewer training epochs. This work bridges the gap between black-box evaluation and mechanistic understanding, providing insights into how neural networks perform information selection and compression during summarization.
>
---
#### [new 149] GreekBarBench: A Challenging Benchmark for Free-Text Legal Reasoning and Citations
- **分类: cs.CL**

- **简介: 该论文提出GreekBarBench基准，评估LLM在希腊法律考试场景下的文本推理与引用能力。针对法律领域自由文本评价难题，设计三维评分系统及LLM裁判方法，并构建元评估基准优化评分标准。实验显示顶尖模型接近但未超越人类专家95分位水平。**

- **链接: [http://arxiv.org/pdf/2505.17267v1](http://arxiv.org/pdf/2505.17267v1)**

> **作者:** Odysseas S. Chlapanis; Dimitrios Galanis; Nikolaos Aletras; Ion Androutsopoulos
>
> **备注:** 19 pages, 17 figures, submitted to May ARR
>
> **摘要:** We introduce GreekBarBench, a benchmark that evaluates LLMs on legal questions across five different legal areas from the Greek Bar exams, requiring citations to statutory articles and case facts. To tackle the challenges of free-text evaluation, we propose a three-dimensional scoring system combined with an LLM-as-a-judge approach. We also develop a meta-evaluation benchmark to assess the correlation between LLM-judges and human expert evaluations, revealing that simple, span-based rubrics improve their alignment. Our systematic evaluation of 13 proprietary and open-weight LLMs shows that even though the best models outperform average expert scores, they fall short of the 95th percentile of experts.
>
---
#### [new 150] Enhancing Mathematics Learning for Hard-of-Hearing Students Through Real-Time Palestinian Sign Language Recognition: A New Dataset
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文提出基于Vision Transformer的巴勒斯坦手语数学手势识别系统，解决听障学生数学教育资源匮乏问题。通过创建含41类数学手势的定制数据集并优化模型，实现97.59%的识别精度，为开发AI辅助教育工具奠定基础，促进包容性数字教育。**

- **链接: [http://arxiv.org/pdf/2505.17055v1](http://arxiv.org/pdf/2505.17055v1)**

> **作者:** Fidaa khandaqji; Huthaifa I. Ashqar; Abdelrahem Atawnih
>
> **摘要:** The study aims to enhance mathematics education accessibility for hard-of-hearing students by developing an accurate Palestinian sign language PSL recognition system using advanced artificial intelligence techniques. Due to the scarcity of digital resources for PSL, a custom dataset comprising 41 mathematical gesture classes was created, and recorded by PSL experts to ensure linguistic accuracy and domain specificity. To leverage state-of-the-art-computer vision techniques, a Vision Transformer ViTModel was fine-tuned for gesture classification. The model achieved an accuracy of 97.59%, demonstrating its effectiveness in recognizing mathematical signs with high precision and reliability. This study highlights the role of deep learning in developing intelligent educational tools that bridge the learning gap for hard-of-hearing students by providing AI-driven interactive solutions to enhance mathematical comprehension. This work represents a significant step toward innovative and inclusive frosting digital integration in specialized learning environments. The dataset is hosted on Hugging Face at https://huggingface.co/datasets/fidaakh/STEM_data.
>
---
#### [new 151] Trust Me, I Can Handle It: Self-Generated Adversarial Scenario Extrapolation for Robust Language Models
- **分类: cs.CL**

- **简介: 该论文提出Adversarial Scenario Extrapolation（ASE）框架，通过自动生成对抗场景并利用推理链提升大语言模型的鲁棒性，解决现有防御方法单一、用户体验差的问题。ASE在推理时引导模型自主分析潜在风险并制定防御策略，实验证明其显著降低攻击成功率和毒性，减少直接拒绝率，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17089v1](http://arxiv.org/pdf/2505.17089v1)**

> **作者:** Md Rafi Ur Rashid; Vishnu Asutosh Dasu; Ye Wang; Gang Tan; Shagufta Mehnaz
>
> **备注:** 26 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.
>
---
#### [new 152] Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦瑞典语语音识别任务，针对其作为中等资源语言在多语种模型中的表现不足问题，通过微调Whisper模型并利用大规模多样化语料库训练，使最佳模型在多个基准测试中WER较OpenAI原版降低47%。**

- **链接: [http://arxiv.org/pdf/2505.17538v1](http://arxiv.org/pdf/2505.17538v1)**

> **作者:** Leonora Vesterbacka; Faton Rekathati; Robin Kurtz; Justyna Sikora; Agnes Toftgård
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This work presents a suite of fine-tuned Whisper models for Swedish, trained on a dataset of unprecedented size and variability for this mid-resourced language. As languages of smaller sizes are often underrepresented in multilingual training datasets, substantial improvements in performance can be achieved by fine-tuning existing multilingual models, as shown in this work. This work reports an overall improvement across model sizes compared to OpenAI's Whisper evaluated on Swedish. Most notably, we report an average 47% reduction in WER comparing our best performing model to OpenAI's whisper-large-v3, in evaluations across FLEURS, Common Voice, and NST.
>
---
#### [new 153] Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; 68T10; I.2.7**

- **简介: 该论文属于语音处理任务，研究帧率对语音分词器的影响，旨在解决不同语言下优化帧率选择的问题。通过对比汉语和英语在不同帧率下的语音识别效果，分析帧率与音位密度、语言声学特征的关联，提出帧率选择的优化策略。（98字）**

- **链接: [http://arxiv.org/pdf/2505.17076v1](http://arxiv.org/pdf/2505.17076v1)**

> **作者:** Haoyang Zhang; Hexin Liu; Xiangyu Zhang; Qiquan Zhang; Yuchen Hu; Junqi Zhao; Fei Tian; Xuerui Yang; Eng Siong Chng
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
>
---
#### [new 154] Tuning Language Models for Robust Prediction of Diverse User Behaviors
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于用户行为预测任务，解决大语言模型在长尾行为预测中过拟合常见行为的问题。提出BehaviorLM方法，通过两阶段微调：首阶段优化常见行为并保留通用知识，次阶段用平衡数据提升罕见行为预测，实验验证其对锚行为和尾部行为均有效。**

- **链接: [http://arxiv.org/pdf/2505.17682v1](http://arxiv.org/pdf/2505.17682v1)**

> **作者:** Fanjin Meng; Jingtao Ding; Jiahui Gong; Chen Yang; Hong Chen; Zuojian Wang; Haisheng Lu; Yong Li
>
> **摘要:** Predicting user behavior is essential for intelligent assistant services, yet deep learning models often struggle to capture long-tailed behaviors. Large language models (LLMs), with their pretraining on vast corpora containing rich behavioral knowledge, offer promise. However, existing fine-tuning approaches tend to overfit to frequent ``anchor'' behaviors, reducing their ability to predict less common ``tail'' behaviors. In this paper, we introduce BehaviorLM, a progressive fine-tuning approach that addresses this issue. In the first stage, LLMs are fine-tuned on anchor behaviors while preserving general behavioral knowledge. In the second stage, fine-tuning uses a balanced subset of all behaviors based on sample difficulty to improve tail behavior predictions without sacrificing anchor performance. Experimental results on two real-world datasets demonstrate that BehaviorLM robustly predicts both anchor and tail behaviors and effectively leverages LLM behavioral knowledge to master tail behavior prediction with few-shot examples.
>
---
#### [new 155] SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SALMONN-omni，首个独立全双工语音LLM，解决现有模块化系统误差累积及codec方法性能下降问题。通过动态思维机制实现听/说状态切换，无需音频编解码器，在问答、对话等场景超越开源模型30%，并优化回声消除、上下文中断等复杂交互。**

- **链接: [http://arxiv.org/pdf/2505.17060v1](http://arxiv.org/pdf/2505.17060v1)**

> **作者:** Wenyi Yu; Siyin Wang; Xiaoyu Yang; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Guangzhi Sun; Lu Lu; Yuxuan Wang; Chao Zhang
>
> **摘要:** In order to enable fluid and natural human-machine speech interaction, existing full-duplex conversational systems often adopt modular architectures with auxiliary components such as voice activity detectors, interrupters, conversation state predictors, or multiple LLMs. These systems, however, suffer from error accumulation across modules and struggle with key challenges such as context-dependent barge-in and echo cancellation. Recent approaches, most notably Moshi, simplify the pipeline by injecting audio codecs into the token space of a single LLM. However, such methods still incur significant performance degradation when operating on the speech rather than text modality. In this paper, we introduce SALMONN-omni, the first single, standalone full-duplex speech LLM that operates without audio codecs in its token space. It features a novel dynamic thinking mechanism within the LLM backbone, enabling the model to learn when to transition between speaking and listening states. Experiments on widely used benchmarks for spoken question answering and open-domain dialogue show that SALMONN-omni achieves at least 30\% relative performance improvement over existing open-source full-duplex models and performs highly competitively to half-duplex and turn-based systems, despite using substantially less training data. Moreover, SALMONN-omni demonstrates strong performance in complex conversational scenarios, including turn-taking, backchanneling, echo cancellation and context-dependent barge-in, with further improvements achieved through reinforcement learning. Some demo conversations between user and SALMONN-omni are provided in the following repository https://github.com/bytedance/SALMONN.
>
---
#### [new 156] Explaining Sources of Uncertainty in Automated Fact-Checking
- **分类: cs.CL**

- **简介: 该论文属于自动事实核查任务，解决模型因证据冲突导致的不确定性解释不足问题。提出CLUE框架，通过无监督识别文本片段间的冲突/一致性关系，并生成自然语言解释，提升人类-AI协作中的可信度。方法无需微调，实验显示其解释更准确、实用。**

- **链接: [http://arxiv.org/pdf/2505.17855v1](http://arxiv.org/pdf/2505.17855v1)**

> **作者:** Jingyi Sun; Greta Warren; Irina Shklovski; Isabelle Augenstein
>
> **摘要:** Understanding sources of a model's uncertainty regarding its predictions is crucial for effective human-AI collaboration. Prior work proposes using numerical uncertainty or hedges ("I'm not sure, but ..."), which do not explain uncertainty that arises from conflicting evidence, leaving users unable to resolve disagreements or rely on the output. We introduce CLUE (Conflict-and-Agreement-aware Language-model Uncertainty Explanations), the first framework to generate natural language explanations of model uncertainty by (i) identifying relationships between spans of text that expose claim-evidence or inter-evidence conflicts and agreements that drive the model's predictive uncertainty in an unsupervised way, and (ii) generating explanations via prompting and attention steering that verbalize these critical interactions. Across three language models and two fact-checking datasets, we show that CLUE produces explanations that are more faithful to the model's uncertainty and more consistent with fact-checking decisions than prompting for uncertainty explanations without span-interaction guidance. Human evaluators judge our explanations to be more helpful, more informative, less redundant, and more logically consistent with the input than this baseline. CLUE requires no fine-tuning or architectural changes, making it plug-and-play for any white-box language model. By explicitly linking uncertainty to evidence conflicts, it offers practical support for fact-checking and generalises readily to other tasks that require reasoning over complex information.
>
---
#### [new 157] Next Token Perception Score: Analytical Assessment of your LLM Perception Skills
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Next Token Perception Score（NTPS），评估LLM感知能力。针对自回归预训练特征与下游感知任务不一致问题，通过量化特征子空间重叠，发现NTPS与任务性能强相关，且LoRA微调可提升NTPS，有效预测模型优化潜力，提供理论与实践工具。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17169v1](http://arxiv.org/pdf/2505.17169v1)**

> **作者:** Yu-Ang Cheng; Leyang Hu; Hai Huang; Randall Balestriero
>
> **摘要:** Autoregressive pretraining has become the de facto paradigm for learning general-purpose representations in large language models (LLMs). However, linear probe performance across downstream perception tasks shows substantial variability, suggesting that features optimized for next-token prediction do not consistently transfer well to downstream perception tasks. We demonstrate that representations learned via autoregression capture features that may lie outside the subspaces most informative for perception. To quantify the (mis)alignment between autoregressive pretraining and downstream perception, we introduce the Next Token Perception Score (NTPS)-a score derived under a linear setting that measures the overlap between autoregressive and perception feature subspaces. This metric can be easily computed in closed form from pretrained representations and labeled data, and is proven to both upper- and lower-bound the excess loss. Empirically, we show that NTPS correlates strongly with linear probe accuracy across 12 diverse NLP datasets and eight pretrained models ranging from 270M to 8B parameters, confirming its utility as a measure of alignment. Furthermore, we show that NTPS increases following low-rank adaptation (LoRA) fine-tuning, especially in large models, suggesting that LoRA aligning representations to perception tasks enhances subspace overlap and thus improves downstream performance. More importantly, we find that NTPS reliably predicts the additional accuracy gains attained by LoRA finetuning thereby providing a lightweight prescreening tool for LoRA adaptation. Our results offer both theoretical insights and practical tools for analytically assessing LLM perception skills.
>
---
#### [new 158] Humans Hallucinate Too: Language Models Identify and Correct Subjective Annotation Errors With Label-in-a-Haystack Prompts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理标注纠错任务，解决主观标注（如情感/道德判断）中区分合理差异与错误的问题。提出Label-in-a-Haystack框架，利用LLM重新预测标签并替换与原始标注冲突的结果，提升标注质量。**

- **链接: [http://arxiv.org/pdf/2505.17222v1](http://arxiv.org/pdf/2505.17222v1)**

> **作者:** Georgios Chochlakis; Peter Wu; Arjun Bedi; Marcus Ma; Kristina Lerman; Shrikanth Narayanan
>
> **备注:** 17 pages, 16 figures, 9 tables
>
> **摘要:** Modeling complex subjective tasks in Natural Language Processing, such as recognizing emotion and morality, is considerably challenging due to significant variation in human annotations. This variation often reflects reasonable differences in semantic interpretations rather than mere noise, necessitating methods to distinguish between legitimate subjectivity and error. We address this challenge by exploring label verification in these contexts using Large Language Models (LLMs). First, we propose a simple In-Context Learning binary filtering baseline that estimates the reasonableness of a document-label pair. We then introduce the Label-in-a-Haystack setting: the query and its label(s) are included in the demonstrations shown to LLMs, which are prompted to predict the label(s) again, while receiving task-specific instructions (e.g., emotion recognition) rather than label copying. We show how the failure to copy the label(s) to the output of the LLM are task-relevant and informative. Building on this, we propose the Label-in-a-Haystack Rectification (LiaHR) framework for subjective label correction: when the model outputs diverge from the reference gold labels, we assign the generated labels to the example instead of discarding it. This approach can be integrated into annotation pipelines to enhance signal-to-noise ratios. Comprehensive analyses, human evaluations, and ecological validity studies verify the utility of LiaHR for label correction. Code is available at https://github.com/gchochla/LiaHR.
>
---
#### [new 159] GloSS over Toxicity: Understanding and Mitigating Toxicity in LLMs via Global Toxic Subspace
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLMs）毒性缓解任务，旨在解决现有方法对毒性来源（FFN层局部子空间）建模不足的问题。提出GloSS方法，通过识别并移除FFN参数中的全局有毒子空间，在保持模型性能的同时达到最优去毒效果，无需大量数据或重新训练。**

- **链接: [http://arxiv.org/pdf/2505.17078v1](http://arxiv.org/pdf/2505.17078v1)**

> **作者:** Zenghao Duan; Zhiyi Yin; Zhichao Shi; Liang Pang; Shaoling Jing; Jiayi Wu; Yu Yan; Huawei Shen; Xueqi Cheng
>
> **摘要:** This paper investigates the underlying mechanisms of toxicity generation in Large Language Models (LLMs) and proposes an effective detoxification approach. Prior work typically considers the Feed-Forward Network (FFN) as the main source of toxicity, representing toxic regions as a set of toxic vectors or layer-wise subspaces. However, our in-depth analysis reveals that the global toxic subspace offers a more effective and comprehensive representation of toxic region within the model. Building on this insight, we propose GloSS (Global Toxic Subspace Suppression), a lightweight, four-stage method that mitigates toxicity by identifying and removing the global toxic subspace from the parameters of FFN. Experiments across a range of LLMs show that GloSS achieves state-of-the-art detoxification performance while preserving the models general capabilities, without requiring large-scale data or model retraining.
>
---
#### [new 160] keepitsimple at SemEval-2025 Task 3: LLM-Uncertainty based Approach for Multilingual Hallucination Span Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对SemEval-2025任务3（多语言幻觉片段检测），提出基于LLM不确定性的检测方法。通过分析黑盒模型随机采样响应的变异性，利用熵值衡量文本片段的不一致性，识别幻觉内容。无需额外训练，通过超参数调优和误差分析优化性能。**

- **链接: [http://arxiv.org/pdf/2505.17485v1](http://arxiv.org/pdf/2505.17485v1)**

> **作者:** Saketh Reddy Vemula; Parameswari Krishnamurthy
>
> **摘要:** Identification of hallucination spans in black-box language model generated text is essential for applications in the real world. A recent attempt at this direction is SemEval-2025 Task 3, Mu-SHROOM-a Multilingual Shared Task on Hallucinations and Related Observable Over-generation Errors. In this work, we present our solution to this problem, which capitalizes on the variability of stochastically-sampled responses in order to identify hallucinated spans. Our hypothesis is that if a language model is certain of a fact, its sampled responses will be uniform, while hallucinated facts will yield different and conflicting results. We measure this divergence through entropy-based analysis, allowing for accurate identification of hallucinated segments. Our method is not dependent on additional training and hence is cost-effective and adaptable. In addition, we conduct extensive hyperparameter tuning and perform error analysis, giving us crucial insights into model behavior.
>
---
#### [new 161] QwenLong-CPRS: Towards $\infty$-LLMs with Dynamic Context Optimization
- **分类: cs.CL**

- **简介: 该论文提出QwenLong-CPRS框架，解决长序列处理中计算开销大和性能下降问题。通过动态上下文压缩（自然语言引导、双向推理层、token评估及窗口并行推理），实现高效压缩与性能提升，兼容主流LLMs，在多个基准测试中超越现有方法，建立新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18092v1](http://arxiv.org/pdf/2505.18092v1)**

> **作者:** Weizhou Shen; Chenliang Li; Fanqi Wan; Shengyi Liao; Shaopeng Lai; Bo Zhang; Yingcheng Shi; Yuning Wu; Gang Fu; Zhansheng Li; Bin Yang; Ji Zhang; Fei Huang; Jingren Zhou; Ming Yan
>
> **摘要:** This technical report presents QwenLong-CPRS, a context compression framework designed for explicit long-context optimization, addressing prohibitive computation overhead during the prefill stage and the "lost in the middle" performance degradation of large language models (LLMs) during long sequence processing. Implemented through a novel dynamic context optimization mechanism, QwenLong-CPRS enables multi-granularity context compression guided by natural language instructions, achieving both efficiency gains and improved performance. Evolved from the Qwen architecture series, QwenLong-CPRS introduces four key innovations: (1) Natural language-guided dynamic optimization, (2) Bidirectional reasoning layers for enhanced boundary awareness, (3) Token critic mechanisms with language modeling heads, and (4) Window-parallel inference. Comprehensive evaluations across five benchmarks (4K-2M word contexts) demonstrate QwenLong-CPRS's threefold effectiveness: (1) Consistent superiority over other context management methods like RAG and sparse attention in both accuracy and efficiency. (2) Architecture-agnostic integration with all flagship LLMs, including GPT-4o, Gemini2.0-pro, Claude3.7-sonnet, DeepSeek-v3, and Qwen2.5-max, achieves 21.59$\times$ context compression alongside 19.15-point average performance gains; (3) Deployed with Qwen2.5-32B-Instruct, QwenLong-CPRS surpasses leading proprietary LLMs by 4.85 and 10.88 points on Ruler-128K and InfiniteBench, establishing new SOTA performance.
>
---
#### [new 162] Prompt Engineering: How Prompt Vocabulary affects Domain Knowledge
- **分类: cs.CL**

- **简介: 该论文研究提示词词汇具体性对领域专用大语言模型性能的影响，通过系统替换词汇的具体程度测试四个模型在STEM、医学和法律任务中的表现，发现存在最佳具体范围，为优化领域提示设计提供依据。**

- **链接: [http://arxiv.org/pdf/2505.17037v1](http://arxiv.org/pdf/2505.17037v1)**

> **作者:** Dimitri Schreiter
>
> **摘要:** Prompt engineering has emerged as a critical component in optimizing large language models (LLMs) for domain-specific tasks. However, the role of prompt specificity, especially in domains like STEM (physics, chemistry, biology, computer science and mathematics), medicine, and law, remains underexplored. This thesis addresses the problem of whether increasing the specificity of vocabulary in prompts improves LLM performance in domain-specific question-answering and reasoning tasks. We developed a synonymization framework to systematically substitute nouns, verbs, and adjectives with varying specificity levels, measuring the impact on four LLMs: Llama-3.1-70B-Instruct, Granite-13B-Instruct-V2, Flan-T5-XL, and Mistral-Large 2, across datasets in STEM, law, and medicine. Our results reveal that while generally increasing the specificity of prompts does not have a significant impact, there appears to be a specificity range, across all considered models, where the LLM performs the best. Identifying this optimal specificity range offers a key insight for prompt design, suggesting that manipulating prompts within this range could maximize LLM performance and lead to more efficient applications in specialized domains.
>
---
#### [new 163] MathEDU: Towards Adaptive Feedback for Student Mathematical Problem-Solving
- **分类: cs.CL**

- **简介: 该论文提出MathEDU，利用LLMs为学生数学解题提供自适应反馈，解决在线教育中个性化反馈不足的问题。构建包含学生解答和教师反馈的MathEDU数据集，在两种场景（历史记录/冷启动）评估模型，结果表明模型能有效识别正确性，但生成详细反馈仍有挑战。**

- **链接: [http://arxiv.org/pdf/2505.18056v1](http://arxiv.org/pdf/2505.18056v1)**

> **作者:** Wei-Ling Hsu; Yu-Chien Tang; An-Zi Yen
>
> **备注:** Pre-print
>
> **摘要:** Online learning enhances educational accessibility, offering students the flexibility to learn anytime, anywhere. However, a key limitation is the lack of immediate, personalized feedback, particularly in helping students correct errors in math problem-solving. Several studies have investigated the applications of large language models (LLMs) in educational contexts. In this paper, we explore the capabilities of LLMs to assess students' math problem-solving processes and provide adaptive feedback. The MathEDU dataset is introduced, comprising authentic student solutions annotated with teacher feedback. We evaluate the model's ability to support personalized learning in two scenarios: one where the model has access to students' prior answer histories, and another simulating a cold-start context. Experimental results show that the fine-tuned model performs well in identifying correctness. However, the model still faces challenges in generating detailed feedback for pedagogical purposes.
>
---
#### [new 164] PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models
- **分类: cs.CL**

- **简介: 该论文针对表格问答（TQA）任务，提出PPT框架解决缺乏自生成数据提升模型的问题。通过分解推理链为离散状态并进行偏好学习，仅用8000个对比样本提升模型性能，优于传统方法且推理效率更高。**

- **链接: [http://arxiv.org/pdf/2505.17565v1](http://arxiv.org/pdf/2505.17565v1)**

> **作者:** Wei Zhou; Mohsen Mesgar; Heike Adel; Annemarie Friedrich
>
> **摘要:** Improving large language models (LLMs) with self-generated data has demonstrated success in tasks such as mathematical reasoning and code generation. Yet, no exploration has been made on table question answering (TQA), where a system answers questions based on tabular data. Addressing this gap is crucial for TQA, as effective self-improvement can boost performance without requiring costly or manually annotated data. In this work, we propose PPT, a Process-based Preference learning framework for TQA. It decomposes reasoning chains into discrete states, assigns scores to each state, and samples contrastive steps for preference learning. Experimental results show that PPT effectively improves TQA models by up to 5% on in-domain datasets and 2.4% on out-of-domain datasets, with only 8,000 preference pairs. Furthermore, the resulting models achieve competitive results compared to more complex and larger state-of-the-art TQA systems, while being five times more efficient during inference.
>
---
#### [new 165] Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods
- **分类: cs.CL**

- **简介: 该论文提出"模型免疫"方法，属于AI事实对齐任务，旨在减少生成模型因训练数据虚假信息导致的错误输出。通过用标注虚假数据微调模型（类似疫苗原理），使其主动增强识别和拒斥误导性内容的能力，同时保持事实准确性，首次将事实核查的虚假数据作为监督信号，并设计伦理管控机制。**

- **链接: [http://arxiv.org/pdf/2505.17870v1](http://arxiv.org/pdf/2505.17870v1)**

> **作者:** Shaina Raza; Rizwan Qureshi; Marcelo Lotif; Aman Chadha; Deval Pandya; Christos Emmanouilidis
>
> **摘要:** Generative AI models often learn and reproduce false information present in their training corpora. This position paper argues that, analogous to biological immunization, where controlled exposure to a weakened pathogen builds immunity, AI models should be fine tuned on small, quarantined sets of explicitly labeled falsehoods as a "vaccine" against misinformation. These curated false examples are periodically injected during finetuning, strengthening the model ability to recognize and reject misleading claims while preserving accuracy on truthful inputs. An illustrative case study shows that immunized models generate substantially less misinformation than baselines. To our knowledge, this is the first training framework that treats fact checked falsehoods themselves as a supervised vaccine, rather than relying on input perturbations or generic human feedback signals, to harden models against future misinformation. We also outline ethical safeguards and governance controls to ensure the safe use of false data. Model immunization offers a proactive paradigm for aligning AI systems with factuality.
>
---
#### [new 166] Signals from the Floods: AI-Driven Disaster Analysis through Multi-Source Data Fusion
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于灾害分析任务，旨在通过融合多源数据提升灾害响应效率。针对社交媒体碎片化与结构化文档信息整合难题，研究结合LDA主题模型与LLM，分析5.5万条推文及1450份咨询文档，提出"相关性指数"方法过滤噪音，优化应急决策与长期韧性规划。**

- **链接: [http://arxiv.org/pdf/2505.17038v1](http://arxiv.org/pdf/2505.17038v1)**

> **作者:** Xian Gong; Paul X. McCarthy; Lin Tian; Marian-Andrei Rizoiu
>
> **摘要:** Massive and diverse web data are increasingly vital for government disaster response, as demonstrated by the 2022 floods in New South Wales (NSW), Australia. This study examines how X (formerly Twitter) and public inquiry submissions provide insights into public behaviour during crises. We analyse more than 55,000 flood-related tweets and 1,450 submissions to identify behavioural patterns during extreme weather events. While social media posts are short and fragmented, inquiry submissions are detailed, multi-page documents offering structured insights. Our methodology integrates Latent Dirichlet Allocation (LDA) for topic modelling with Large Language Models (LLMs) to enhance semantic understanding. LDA reveals distinct opinions and geographic patterns, while LLMs improve filtering by identifying flood-relevant tweets using public submissions as a reference. This Relevance Index method reduces noise and prioritizes actionable content, improving situational awareness for emergency responders. By combining these complementary data streams, our approach introduces a novel AI-driven method to refine crisis-related social media content, improve real-time disaster response, and inform long-term resilience planning.
>
---
#### [new 167] Systematic Evaluation of Machine-Generated Reasoning and PHQ-9 Labeling for Depression Detection Using Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属抑郁检测任务，利用大型语言模型（LLM）评估机器生成推理及PHQ-9症状标注的可靠性。针对LLM检测潜在弱点和质量控制不足问题，系统分析其推理过程，设计指令策略、对比提示，并结合人类标注评估。通过监督微调和直接偏好优化（DPO）优化模型，DPO显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.17119v1](http://arxiv.org/pdf/2505.17119v1)**

> **作者:** Zongru Shao; Xin Wang; Zhanyang Liu; Chenhan Wang; K. P. Subbalakshmi
>
> **备注:** 8 pages without references
>
> **摘要:** Recent research leverages large language models (LLMs) for early mental health detection, such as depression, often optimized with machine-generated data. However, their detection may be subject to unknown weaknesses. Meanwhile, quality control has not been applied to these generated corpora besides limited human verifications. Our goal is to systematically evaluate LLM reasoning and reveal potential weaknesses. To this end, we first provide a systematic evaluation of the reasoning over machine-generated detection and interpretation. Then we use the models' reasoning abilities to explore mitigation strategies for enhanced performance. Specifically, we do the following: A. Design an LLM instruction strategy that allows for systematic analysis of the detection by breaking down the task into several subtasks. B. Design contrastive few-shot and chain-of-thought prompts by selecting typical positive and negative examples of detection reasoning. C. Perform human annotation for the subtasks identified in the first step and evaluate the performance. D. Identify human-preferred detection with desired logical reasoning from the few-shot generation and use them to explore different optimization strategies. We conducted extensive comparisons on the DepTweet dataset across the following subtasks: 1. identifying whether the speaker is describing their own depression; 2. accurately detecting the presence of PHQ-9 symptoms, and 3. finally, detecting depression. Human verification of statistical outliers shows that LLMs demonstrate greater accuracy in analyzing and detecting explicit language of depression as opposed to implicit expressions of depression. Two optimization methods are used for performance enhancement and reduction of the statistic bias: supervised fine-tuning (SFT) and direct preference optimization (DPO). Notably, the DPO approach achieves significant performance improvement.
>
---
#### [new 168] Multimodal Conversation Structure Understanding
- **分类: cs.CL**

- **简介: 该论文属于多模态对话结构理解任务，旨在解决LLMs在多方对话中对角色（发言者、被称呼者、旁听者）和线程（话语链接、聚类）的分析不足。研究构建了含4398条标注的数据集，评估多个模型发现：视听模型表现最佳但匿名时下降，参与人数多显著降低性能，声学特征与表现正相关，为改进多模态LLM对话推理奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.17536v1](http://arxiv.org/pdf/2505.17536v1)**

> **作者:** Kent K. Chang; Mackenzie Hanh Cramer; Anna Ho; Ti Ti Nguyen; Yilin Yuan; David Bamman
>
> **摘要:** Conversations are usually structured by roles -- who is speaking, who's being addressed, and who's listening -- and unfold in threads that break with changes in speaker floor or topical focus. While large language models (LLMs) have shown incredible capabilities in dialogue and reasoning, their ability to understand fine-grained conversational structure, especially in multi-modal, multi-party settings, remains underexplored. To address this gap, we introduce a suite of tasks focused on conversational role attribution (speaker, addressees, side-participants) and conversation threading (utterance linking and clustering), drawing on conversation analysis and sociolinguistics. To support those tasks, we present a human annotated dataset of 4,398 annotations for speakers and reply-to relationship, 5,755 addressees, and 3,142 side-participants. We evaluate popular audio-visual LLMs and vision-language models on our dataset, and our experimental results suggest that multimodal conversational structure understanding remains challenging. The most performant audio-visual LLM outperforms all vision-language models across all metrics, especially in speaker and addressee recognition. However, its performance drops significantly when conversation participants are anonymized. The number of conversation participants in a clip is the strongest negative predictor of role-attribution performance, while acoustic clarity (measured by pitch and spectral centroid) and detected face coverage yield positive associations. We hope this work lays the groundwork for future evaluation and development of multimodal LLMs that can reason more effectively about conversation structure.
>
---
#### [new 169] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于LLM动态心智理论评估任务，旨在解决现有基准忽视人类心理状态时序演变的问题。提出DynToM基准，通过四步框架生成超1.1万动态社交场景及问题，测试十种模型发现其性能较人类低44.7%，尤其在追踪心理状态变化时表现薄弱。**

- **链接: [http://arxiv.org/pdf/2505.17663v1](http://arxiv.org/pdf/2505.17663v1)**

> **作者:** Yang Xiao; Jiashuo Wang; Qiancheng Xu; Changhe Song; Chunpu Xu; Yi Cheng; Wenjie Li; Pengfei Liu
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states.
>
---
#### [new 170] Large Language Models for Predictive Analysis: How Far Are They?
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估大型语言模型（LLMs）在预测分析中的能力，解决其应用缺乏系统评测的问题。提出PredictiQ基准，整合8领域44个真实数据集的1130个预测查询，测试12种LLMs，发现现有模型在该任务上仍具显著挑战。**

- **链接: [http://arxiv.org/pdf/2505.17149v1](http://arxiv.org/pdf/2505.17149v1)**

> **作者:** Qin Chen; Yuanyi Ren; Xiaojun Ma; Yuyang Shi
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Predictive analysis is a cornerstone of modern decision-making, with applications in various domains. Large Language Models (LLMs) have emerged as powerful tools in enabling nuanced, knowledge-intensive conversations, thus aiding in complex decision-making tasks. With the burgeoning expectation to harness LLMs for predictive analysis, there is an urgent need to systematically assess their capability in this domain. However, there is a lack of relevant evaluations in existing studies. To bridge this gap, we introduce the \textbf{PredictiQ} benchmark, which integrates 1130 sophisticated predictive analysis queries originating from 44 real-world datasets of 8 diverse fields. We design an evaluation protocol considering text analysis, code generation, and their alignment. Twelve renowned LLMs are evaluated, offering insights into their practical use in predictive analysis. Generally, we believe that existing LLMs still face considerable challenges in conducting predictive analysis. See \href{https://github.com/Cqkkkkkk/PredictiQ}{Github}.
>
---
#### [new 171] Distilling LLM Agent into Small Models with Retrieval and Code Tools
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Agent Distillation框架，通过蒸馏LLM代理的行为（含检索和代码工具使用）至小模型，解决其在复杂推理、稀有事实和精确计算任务中的幻觉问题。方法包括优化教师轨迹的"first-thought prefix"和增强推理鲁棒性的"self-consistent action"，实验显示0.5B等小模型性能可媲美更大模型。**

- **链接: [http://arxiv.org/pdf/2505.17612v1](http://arxiv.org/pdf/2505.17612v1)**

> **作者:** Minki Kang; Jongwon Jeong; Seanie Lee; Jaewoong Cho; Sung Ju Hwang
>
> **备注:** preprint, v1
>
> **摘要:** Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at https://github.com/Nardien/agent-distillation.
>
---
#### [new 172] Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions, and Improve with Training
- **分类: cs.CL**

- **简介: 该论文属于AI可解释性任务，旨在解决LLMs决策过程不透明的问题。研究通过微调GPT-4o等模型，在复杂决策任务（如选择公寓、贷款）中训练其量化描述自身属性权重（如自然光 vs 安静环境），验证模型可准确报告内部决策机制，并证明该能力可通过训练提升且泛化至其他任务。**

- **链接: [http://arxiv.org/pdf/2505.17120v1](http://arxiv.org/pdf/2505.17120v1)**

> **作者:** Dillon Plunkett; Adam Morris; Keerthi Reddy; Jorge Morales
>
> **摘要:** We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to introspect and explain their own functioning. Here, we show that i) contemporary LLMs are capable of providing accurate, quantitative descriptions of their own internal processes during certain kinds of decision-making, ii) that it is possible to improve these capabilities through training, and iii) that this training generalizes to at least some degree. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomly-generated, quantitative preferences about how to weigh different attributes during decision-making (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain what they are doing as they make other complex decisions, not just decisions they have learned to make via fine-tuning. This work is a step towards training LLMs to accurately and broadly report on their own internal processes -- a possibility that would yield substantial benefits for interpretability, control, and safety.
>
---
#### [new 173] Conversations: Love Them, Hate Them, Steer Them
- **分类: cs.CL**

- **简介: 该论文属于对话系统优化任务，旨在解决大语言模型（LLMs）情感表达不够细腻的问题。研究通过激活工程方法，利用归因修补定位关键干预点，并基于对比文本对的激活差异生成情感向量，引导LLaMA模型增强情感细微性（如提升积极情绪、增加第一人称使用），实现可控且可解释的情感属性调整。**

- **链接: [http://arxiv.org/pdf/2505.17413v1](http://arxiv.org/pdf/2505.17413v1)**

> **作者:** Niranjan Chebrolu; Gerard Christopher Yeo; Kokil Jaidka
>
> **备注:** 11 pages, 8 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable method for controlling specific emotional attributes in LLMs, contributing to developing more aligned and empathetic conversational AI.
>
---
#### [new 174] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web
- **分类: cs.CL**

- **简介: 该论文提出AVerImaTeC数据集，用于自动验证图文声明任务。针对现有数据集依赖合成数据、缺乏证据标注等问题，构建了1297条真实图文声明，每条含基于网络证据的QA对标注，并通过规范化、时间约束等方法解决事实核查挑战，最终建立基准实验及评估方法。**

- **链接: [http://arxiv.org/pdf/2505.17978v1](http://arxiv.org/pdf/2505.17978v1)**

> **作者:** Rui Cao; Zifeng Ding; Zhijiang Guo; Michael Schlichtkrull; Andreas Vlachos
>
> **摘要:** Textual claims are often accompanied by images to enhance their credibility and spread on social media, but this also raises concerns about the spread of misinformation. Existing datasets for automated verification of image-text claims remain limited, as they often consist of synthetic claims and lack evidence annotations to capture the reasoning behind the verdict. In this work, we introduce AVerImaTeC, a dataset consisting of 1,297 real-world image-text claims. Each claim is annotated with question-answer (QA) pairs containing evidence from the web, reflecting a decomposed reasoning regarding the verdict. We mitigate common challenges in fact-checking datasets such as contextual dependence, temporal leakage, and evidence insufficiency, via claim normalization, temporally constrained evidence annotation, and a two-stage sufficiency check. We assess the consistency of the annotation in AVerImaTeC via inter-annotator studies, achieving a $\kappa=0.742$ on verdicts and $74.7\%$ consistency on QA pairs. We also propose a novel evaluation method for evidence retrieval and conduct extensive experiments to establish baselines for verifying image-text claims using open-web evidence.
>
---
#### [new 175] Towards Robust Evaluation of STEM Education: Leveraging MLLMs in Project-Based Learning
- **分类: cs.CL; cs.AI; cs.CE; cs.CY; cs.MM**

- **简介: 该论文提出PBLBench基准，解决现有STEM教育中MLLMs评估不足的问题，通过专家验证的多模态任务测试模型，揭示先进模型表现欠佳，推动AI工具优化以助力教学。**

- **链接: [http://arxiv.org/pdf/2505.17050v1](http://arxiv.org/pdf/2505.17050v1)**

> **作者:** Yanhao Jia; Xinyi Wu; Qinglin Zhang; Yiran Qin; Luwei Xiao; Shuai Zhao
>
> **摘要:** Project-Based Learning (PBL) involves a variety of highly correlated multimodal data, making it a vital educational approach within STEM disciplines. With the rapid development of multimodal large language models (MLLMs), researchers have begun exploring their potential to enhance tasks such as information retrieval, knowledge comprehension, and data generation in educational settings. However, existing benchmarks fall short in providing both a free-form output structure and a rigorous human expert validation process, limiting their effectiveness in evaluating real-world educational tasks. Additionally, few methods have developed automated pipelines to assist with the complex responsibilities of teachers leveraging MLLMs, largely due to model hallucination and instability, which lead to unreliable implementation. To address this gap, we introduce PBLBench, a novel benchmark designed to evaluate complex reasoning grounded in domain-specific knowledge and long-context understanding, thereby challenging models with tasks that closely resemble those handled by human experts. To establish reliable ground truth, we adopt the Analytic Hierarchy Process (AHP), utilizing expert-driven pairwise comparisons to derive structured and weighted evaluation criteria. We assess the performance of 15 leading MLLMs/LLMs using PBLBench and demonstrate that even the most advanced models achieve only 59% rank accuracy, underscoring the significant challenges presented by this benchmark. We believe PBLBench will serve as a catalyst for the development of more capable AI agents, ultimately aiming to alleviate teacher workload and enhance educational productivity.
>
---
#### [new 176] Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究优化基于大语言模型（LLM）的具身智能体效率。针对其多轮交互中易陷入重复或冗余计算的问题，提出内在（生成时加入退出指令）与外在（任务完成验证）两种早退策略，并设计冗余步骤减少与进度退化指标评估效果。实验显示，4种LLM在5个环境中的效率显著提升，性能轻微下降，且结合强智能体辅助可进一步提升表现。**

- **链接: [http://arxiv.org/pdf/2505.17616v1](http://arxiv.org/pdf/2505.17616v1)**

> **作者:** Qingyu Lu; Liang Ding; Siyi Cao; Xuebo Liu; Kanjian Zhang; Jinxia Zhang; Dacheng Tao
>
> **备注:** Under Review
>
> **摘要:** Agents powered by large language models (LLMs) have demonstrated strong planning and decision-making capabilities in complex embodied environments. However, such agents often suffer from inefficiencies in multi-turn interactions, frequently trapped in repetitive loops or issuing ineffective commands, leading to redundant computational overhead. Instead of relying solely on learning from trajectories, we take a first step toward exploring the early-exit behavior for LLM-based agents. We propose two complementary approaches: 1. an $\textbf{intrinsic}$ method that injects exit instructions during generation, and 2. an $\textbf{extrinsic}$ method that verifies task completion to determine when to halt an agent's trial. To evaluate early-exit mechanisms, we introduce two metrics: one measures the reduction of $\textbf{redundant steps}$ as a positive effect, and the other evaluates $\textbf{progress degradation}$ as a negative effect. Experiments with 4 different LLMs across 5 embodied environments show significant efficiency improvements, with only minor drops in agent performance. We also validate a practical strategy where a stronger agent assists after an early-exit agent, achieving better performance with the same total steps. We will release our code to support further research.
>
---
#### [new 177] Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 论文研究端到端口语语言模型训练中的灾难性遗忘问题，评估模型合并、LoRA缩放折扣及经验回放策略，发现后者效果最佳，结合其他方法进一步提升，为更稳健的训练提供见解。**

- **链接: [http://arxiv.org/pdf/2505.17496v1](http://arxiv.org/pdf/2505.17496v1)**

> **作者:** Chi-Yuan Hsiao; Ke-Han Lu; Kai-Wei Chang; Chih-Kai Yang; Wei-Chih Chen; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies-model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines.
>
---
#### [new 178] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）在复杂交互任务（如谈判、工具使用）中规划能力不足的问题，提出一种基于离线目标条件强化学习的方法。通过训练轻量级价值函数预测行动结果，指导LLM评估多路径决策，解决传统RL微调的高成本与可扩展性限制，在多类任务中优于现有RL和提示方法。**

- **链接: [http://arxiv.org/pdf/2505.18098v1](http://arxiv.org/pdf/2505.18098v1)**

> **作者:** Joey Hong; Anca Dragan; Sergey Levine
>
> **备注:** 18 pages, 4 figures, 2 tables
>
> **摘要:** Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability.
>
---
#### [new 179] Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations
- **分类: cs.CL; cs.AI; I.2**

- **简介: 该论文评估大语言模型（LLMs）在地理空间推理中的能力，解决其处理几何及拓扑空间关系的局限。通过几何嵌入、提示工程和日常语言三种方法，测试GPT-3.5、GPT-4等模型在空间关系识别任务中的表现，发现GPT-4在提示下准确率超0.66，并提出添加地理上下文可提升效果，为开发地理基础模型提供改进方向。**

- **链接: [http://arxiv.org/pdf/2505.17136v1](http://arxiv.org/pdf/2505.17136v1)**

> **作者:** Yuhan Ji; Song Gao; Ying Nie; Ivan Majić; Krzysztof Janowicz
>
> **备注:** 33 pages, 13 figures, IJGIS GeoFM Special Issue
>
> **摘要:** Applying AI foundation models directly to geospatial datasets remains challenging due to their limited ability to represent and reason with geographical entities, specifically vector-based geometries and natural language descriptions of complex spatial relations. To address these issues, we investigate the extent to which a well-known-text (WKT) representation of geometries and their spatial relations (e.g., topological predicates) are preserved during spatial reasoning when the geospatial vector data are passed to large language models (LLMs) including GPT-3.5-turbo, GPT-4, and DeepSeek-R1-14B. Our workflow employs three distinct approaches to complete the spatial reasoning tasks for comparison, i.e., geometry embedding-based, prompt engineering-based, and everyday language-based evaluation. Our experiment results demonstrate that both the embedding-based and prompt engineering-based approaches to geospatial question-answering tasks with GPT models can achieve an accuracy of over 0.6 on average for the identification of topological spatial relations between two geometries. Among the evaluated models, GPT-4 with few-shot prompting achieved the highest performance with over 0.66 accuracy on topological spatial relation inference. Additionally, GPT-based reasoner is capable of properly comprehending inverse topological spatial relations and including an LLM-generated geometry can enhance the effectiveness for geographic entity retrieval. GPT-4 also exhibits the ability to translate certain vernacular descriptions about places into formal topological relations, and adding the geometry-type or place-type context in prompts may improve inference accuracy, but it varies by instance. The performance of these spatial reasoning tasks offers valuable insights for the refinement of LLMs with geographical knowledge towards the development of geo-foundation models capable of geospatial reasoning.
>
---
#### [new 180] Robustifying Vision-Language Models via Dynamic Token Reweighting
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）安全防御任务，旨在解决对抗性视觉文本交互（jailbreak攻击）突破安全限制的问题。提出DTR方法，通过动态调整视觉标记权重优化KV缓存，减少对抗输入影响，同时保持模型性能与效率，无需额外安全数据或图像转文本转换。**

- **链接: [http://arxiv.org/pdf/2505.17132v1](http://arxiv.org/pdf/2505.17132v1)**

> **作者:** Tanqiu Jiang; Jiacheng Liang; Rongyi Zhu; Jiawei Zhou; Fenglong Ma; Ting Wang
>
> **摘要:** Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)
>
---
#### [new 181] Mitigating Cyber Risk in the Age of Open-Weight LLMs: Policy Gaps and Technical Realities
- **分类: cs.CR; cs.CL**

- **简介: 该论文分析开放权重AI模型（如DeepSeek-R1）带来的网络安全威胁（如加速恶意软件开发、增强社会工程攻击），评估现有法规（如欧盟AI法案）的政策缺口，提出通过控制高风险能力、优化政策解释、推动防御AI技术及国际协作，平衡安全与技术开放。**

- **链接: [http://arxiv.org/pdf/2505.17109v1](http://arxiv.org/pdf/2505.17109v1)**

> **作者:** Alfonso de Gregorio
>
> **备注:** 8 pages, no figures
>
> **摘要:** Open-weight general-purpose AI (GPAI) models offer significant benefits but also introduce substantial cybersecurity risks, as demonstrated by the offensive capabilities of models like DeepSeek-R1 in evaluations such as MITRE's OCCULT. These publicly available models empower a wider range of actors to automate and scale cyberattacks, challenging traditional defence paradigms and regulatory approaches. This paper analyzes the specific threats -- including accelerated malware development and enhanced social engineering -- magnified by open-weight AI release. We critically assess current regulations, notably the EU AI Act and the GPAI Code of Practice, identifying significant gaps stemming from the loss of control inherent in open distribution, which renders many standard security mitigations ineffective. We propose a path forward focusing on evaluating and controlling specific high-risk capabilities rather than entire models, advocating for pragmatic policy interpretations for open-weight systems, promoting defensive AI innovation, and fostering international collaboration on standards and cyber threat intelligence (CTI) sharing to ensure security without unduly stifling open technological progress.
>
---
#### [new 182] Surfacing Semantic Orthogonality Across Model Safety Benchmarks: A Multi-Dimensional Analysis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文通过UMAP和k-means聚类分析五个AI安全基准，发现六个主要危害类别及数据集间的语义差异，解决安全评估覆盖不全问题，提出量化框架以优化未来数据集开发，提升AI危害评估全面性。**

- **链接: [http://arxiv.org/pdf/2505.17636v1](http://arxiv.org/pdf/2505.17636v1)**

> **作者:** Jonathan Bennion; Shaona Ghosh; Mantek Singh; Nouha Dziri
>
> **备注:** 6th International Conference on Advanced Natural Language Processing (AdNLP 2025), May 17 ~ 18, 2025, Zurich, Switzerland
>
> **摘要:** Various AI safety datasets have been developed to measure LLMs against evolving interpretations of harm. Our evaluation of five recently published open-source safety benchmarks reveals distinct semantic clusters using UMAP dimensionality reduction and kmeans clustering (silhouette score: 0.470). We identify six primary harm categories with varying benchmark representation. GretelAI, for example, focuses heavily on privacy concerns, while WildGuardMix emphasizes self-harm scenarios. Significant differences in prompt length distribution suggests confounds to data collection and interpretations of harm as well as offer possible context. Our analysis quantifies benchmark orthogonality among AI benchmarks, allowing for transparency in coverage gaps despite topical similarities. Our quantitative framework for analyzing semantic orthogonality across safety benchmarks enables more targeted development of datasets that comprehensively address the evolving landscape of harms in AI use, however that is defined in the future.
>
---
#### [new 183] DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于理论心理（ToM）推理任务，旨在解决小规模语言模型在复杂社会推理中的不足。提出DEL-ToM框架，利用动态认识逻辑将ToM任务分解为结构化信念更新序列，并训练Process Belief Model（PBM）验证推理步骤，通过推理时计算扩展提升模型性能，无需重新训练。**

- **链接: [http://arxiv.org/pdf/2505.17348v1](http://arxiv.org/pdf/2505.17348v1)**

> **作者:** Yuheng Wu; Jianwen Xie; Denghui Zhang; Zhaozhuo Xu
>
> **摘要:** Theory-of-Mind (ToM) tasks pose a unique challenge for small language models (SLMs) with limited scale, which often lack the capacity to perform deep social reasoning. In this work, we propose DEL-ToM, a framework that improves ToM reasoning through inference-time scaling rather than architectural changes. Our approach decomposes ToM tasks into a sequence of belief updates grounded in Dynamic Epistemic Logic (DEL), enabling structured and transparent reasoning. We train a verifier, called the Process Belief Model (PBM), to score each belief update step using labels generated automatically via a DEL simulator. During inference, candidate belief traces generated by a language model are evaluated by the PBM, and the highest-scoring trace is selected. This allows SLMs to emulate more deliberate reasoning by allocating additional compute at test time. Experiments across multiple model scales and benchmarks show that DEL-ToM consistently improves performance, demonstrating that verifiable belief supervision can significantly enhance ToM abilities of SLMs without retraining.
>
---
#### [new 184] An End-to-End Approach for Child Reading Assessment in the Xhosa Language
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出针对南非祖鲁语儿童阅读评估的端到端方法，旨在解决低资源语言中儿童语音识别数据不足问题。构建包含EGRA项目的语音数据集，通过wav2vec 2.0、HuBERT和Whisper模型验证，发现数据量平衡及多类训练对模型性能关键。**

- **链接: [http://arxiv.org/pdf/2505.17371v1](http://arxiv.org/pdf/2505.17371v1)**

> **作者:** Sergio Chevtchenko; Nikhil Navas; Rafaella Vale; Franco Ubaudi; Sipumelele Lucwaba; Cally Ardington; Soheil Afshar; Mark Antoniou; Saeed Afshar
>
> **备注:** Paper accepted on AIED 2025 containing 14 pages, 6 figures and 4 tables
>
> **摘要:** Child literacy is a strong predictor of life outcomes at the subsequent stages of an individual's life. This points to a need for targeted interventions in vulnerable low and middle income populations to help bridge the gap between literacy levels in these regions and high income ones. In this effort, reading assessments provide an important tool to measure the effectiveness of these programs and AI can be a reliable and economical tool to support educators with this task. Developing accurate automatic reading assessment systems for child speech in low-resource languages poses significant challenges due to limited data and the unique acoustic properties of children's voices. This study focuses on Xhosa, a language spoken in South Africa, to advance child speech recognition capabilities. We present a novel dataset composed of child speech samples in Xhosa. The dataset is available upon request and contains ten words and letters, which are part of the Early Grade Reading Assessment (EGRA) system. Each recording is labeled with an online and cost-effective approach by multiple markers and a subsample is validated by an independent EGRA reviewer. This dataset is evaluated with three fine-tuned state-of-the-art end-to-end models: wav2vec 2.0, HuBERT, and Whisper. The results indicate that the performance of these models can be significantly influenced by the amount and balancing of the available training data, which is fundamental for cost-effective large dataset collection. Furthermore, our experiments indicate that the wav2vec 2.0 performance is improved by training on multiple classes at a time, even when the number of available samples is constrained.
>
---
#### [new 185] Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态模型视觉-语言对齐任务，旨在解决现有投影器对视觉 token 与语义词的对齐不足问题。通过分析投影器对视觉信息的压缩机制，提出"多语义对齐假设"及"patch 对齐训练"方法，增强视觉-语言细粒度对齐，提升图像描述生成、目标定位等任务性能。**

- **链接: [http://arxiv.org/pdf/2505.17316v1](http://arxiv.org/pdf/2505.17316v1)**

> **作者:** Jiachen Jiang; Jinxin Zhou; Bo Peng; Xia Ning; Zhihui Zhu
>
> **摘要:** Achieving better alignment between vision embeddings and Large Language Models (LLMs) is crucial for enhancing the abilities of Multimodal LLMs (MLLMs), particularly for recent models that rely on powerful pretrained vision encoders and LLMs. A common approach to connect the pretrained vision encoder and LLM is through a projector applied after the vision encoder. However, the projector is often trained to enable the LLM to generate captions, and hence the mechanism by which LLMs understand each vision token remains unclear. In this work, we first investigate the role of the projector in compressing vision embeddings and aligning them with word embeddings. We show that the projector significantly compresses visual information, removing redundant details while preserving essential elements necessary for the LLM to understand visual content. We then examine patch-level alignment -- the alignment between each vision patch and its corresponding semantic words -- and propose a *multi-semantic alignment hypothesis*. Our analysis indicates that the projector trained by caption loss improves patch-level alignment but only to a limited extent, resulting in weak and coarse alignment. To address this issue, we propose *patch-aligned training* to efficiently enhance patch-level alignment. Our experiments show that patch-aligned training (1) achieves stronger compression capability and improved patch-level alignment, enabling the MLLM to generate higher-quality captions, (2) improves the MLLM's performance by 16% on referring expression grounding tasks, 4% on question-answering tasks, and 3% on modern instruction-following benchmarks when using the same supervised fine-tuning (SFT) setting. The proposed method can be easily extended to other multimodal models.
>
---
#### [new 186] Probe by Gaming: A Game-based Benchmark for Assessing Conceptual Knowledge in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CK-Arena，基于多智能体互动游戏的基准，评估LLMs在动态环境中的概念边界推理能力。解决现有测试侧重事实记忆而忽视概念理解的问题，通过模拟互动场景测试模型对概念关联的描述、区分与推断能力，实验表明模型表现与参数规模无严格关联。**

- **链接: [http://arxiv.org/pdf/2505.17512v1](http://arxiv.org/pdf/2505.17512v1)**

> **作者:** Shuhang Xu; Weijian Deng; Yixuan Zhou; Fangwei Zhong
>
> **备注:** 9 pages
>
> **摘要:** Concepts represent generalized abstractions that enable humans to categorize and reason efficiently, yet it is unclear to what extent Large Language Models (LLMs) comprehend these semantic relationships. Existing benchmarks typically focus on factual recall and isolated tasks, failing to evaluate the ability of LLMs to understand conceptual boundaries. To address this gap, we introduce CK-Arena, a multi-agent interaction game built upon the Undercover game, designed to evaluate the capacity of LLMs to reason with concepts in interactive settings. CK-Arena challenges models to describe, differentiate, and infer conceptual boundaries based on partial information, encouraging models to explore commonalities and distinctions between closely related concepts. By simulating real-world interaction, CK-Arena provides a scalable and realistic benchmark for assessing conceptual reasoning in dynamic environments. Experimental results show that LLMs' understanding of conceptual knowledge varies significantly across different categories and is not strictly aligned with parameter size or general model capabilities. The data and code are available at the project homepage: https://ck-arena.site.
>
---
#### [new 187] FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG; I.2.7; I.5.4; I.7**

- **简介: 该论文提出FS-DAG模型，针对小样本视觉丰富文档理解任务，解决跨领域适应、数据稀缺及OCR错误等问题。通过模块化设计融合领域与模态专用backbone，在信息提取中实现高效收敛与高性能，参数仅90M。**

- **链接: [http://arxiv.org/pdf/2505.17330v1](http://arxiv.org/pdf/2505.17330v1)**

> **作者:** Amit Agarwal; Srikant Panda; Kulbhushan Pachauri
>
> **备注:** Published in the Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Industry Track, pages 100-114
>
> **摘要:** In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : https://github.com/oracle-samples/fs-dag
>
---
#### [new 188] One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出ArrAttack方法，针对防御型LLMs生成鲁棒jailbreak提示。任务为突破安全防御机制，解决现有攻击策略对新防御失效的问题。通过构建通用鲁棒性评估模型，自动优化攻击提示，实现跨模型（如GPT-4、Claude-3）的高效黑盒/白盒攻击，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17598v1](http://arxiv.org/pdf/2505.17598v1)**

> **作者:** Linbao Li; Yannan Liu; Daojing He; Yu Li
>
> **摘要:** Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at https://github.com/LLBao/ArrAttack.
>
---
#### [new 189] Understanding Gated Neurons in Transformers from Their Input-Output Functionality
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于Transformer模型可解释性研究，旨在分析神经元输入与输出的交互作用。针对现有研究忽视输入输出关联的问题，提出通过计算神经元输入与输出权重的余弦相似性，发现早期层以增强概念表征的神经元为主，后期层转向削弱作用，解释为事实回忆的分阶段机制，补充了传统激活依赖分析的不足。**

- **链接: [http://arxiv.org/pdf/2505.17936v1](http://arxiv.org/pdf/2505.17936v1)**

> **作者:** Sebastian Gerstner; Hinrich Schütze
>
> **备注:** 31 pages, 22 figures
>
> **摘要:** Interpretability researchers have attempted to understand MLP neurons of language models based on both the contexts in which they activate and their output weight vectors. They have paid little attention to a complementary aspect: the interactions between input and output. For example, when neurons detect a direction in the input, they might add much the same direction to the residual stream ("enrichment neurons") or reduce its presence ("depletion neurons"). We address this aspect by examining the cosine similarity between input and output weights of a neuron. We apply our method to 12 models and find that enrichment neurons dominate in early-middle layers whereas later layers tend more towards depletion. To explain this finding, we argue that enrichment neurons are largely responsible for enriching concept representations, one of the first steps of factual recall. Our input-output perspective is a complement to activation-dependent analyses and to approaches that treat input and output separately.
>
---
#### [new 190] One RL to See Them All: Visual Triple Unified Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出V-Triune系统，通过统一强化学习框架使视觉语言模型（VLMs）同时学习视觉推理与感知任务。针对RL在目标检测等感知任务中应用不足的问题，设计三组件架构（数据格式、奖励计算、指标监测）及动态IoU奖励，训练出Orsta模型，在MEGA-Bench等数据集显著提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2505.18129v1](http://arxiv.org/pdf/2505.18129v1)**

> **作者:** Yan Ma; Linge Du; Xuyang Shen; Shaoxiang Chen; Pengfei Li; Qibing Ren; Lizhuang Ma; Yuchao Dai; Pengfei Liu; Junjie Yan
>
> **备注:** Technical Report
>
> **摘要:** Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, the use of RL beyond reasoning tasks remains largely unexplored, especially for perceptionintensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises triple complementary components: Sample-Level Data Formatting (to unify diverse task inputs), Verifier-Level Reward Computation (to deliver custom rewards via specialized verifiers) , and Source-Level Metric Monitoring (to diagnose problems at the data-source level). We further introduce a novel Dynamic IoU reward, which provides adaptive, progressive, and definite feedback for perception tasks handled by V-Triune. Our approach is instantiated within off-the-shelf RL training framework using open-source 7B and 32B backbone models. The resulting model, dubbed Orsta (One RL to See Them All), demonstrates consistent improvements across both reasoning and perception tasks. This broad capability is significantly shaped by its training on a diverse dataset, constructed around four representative visual reasoning tasks (Math, Puzzle, Chart, and Science) and four visual perception tasks (Grounding, Detection, Counting, and OCR). Subsequently, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to an impressive +14.1 across its various 7B and 32B model variants, with performance benefits extending to a wide range of downstream tasks. These results highlight the effectiveness and scalability of our unified RL approach for VLMs. The V-Triune system, along with the Orsta models, is publicly available at https://github.com/MiniMax-AI.
>
---
#### [new 191] MMMG: a Comprehensive and Reliable Evaluation Suite for Multitask Multimodal Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 论文提出MMMG基准，解决多模态生成自动评估与人类评价不一致的问题。涵盖图像、音频等4种模态组合的49项任务（含29新任务），通过模型与程序结合实现可靠评估，与人类评价一致率达94.3%。测试24个模型显示，现有模型在多模态推理和音频生成上仍有较大提升空间。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17613v1](http://arxiv.org/pdf/2505.17613v1)**

> **作者:** Jihan Yao; Yushi Hu; Yujie Yi; Bin Han; Shangbin Feng; Guang Yang; Bingbing Wen; Ranjay Krishna; Lucy Lu Wang; Yulia Tsvetkov; Noah A. Smith; Banghua Zhu
>
> **摘要:** Automatically evaluating multimodal generation presents a significant challenge, as automated metrics often struggle to align reliably with human evaluation, especially for complex tasks that involve multiple modalities. To address this, we present MMMG, a comprehensive and human-aligned benchmark for multimodal generation across 4 modality combinations (image, audio, interleaved text and image, interleaved text and audio), with a focus on tasks that present significant challenges for generation models, while still enabling reliable automatic evaluation through a combination of models and programs. MMMG encompasses 49 tasks (including 29 newly developed ones), each with a carefully designed evaluation pipeline, and 937 instructions to systematically assess reasoning, controllability, and other key capabilities of multimodal generation models. Extensive validation demonstrates that MMMG is highly aligned with human evaluation, achieving an average agreement of 94.3%. Benchmarking results on 24 multimodal generation models reveal that even though the state-of-the-art model, GPT Image, achieves 78.3% accuracy for image generation, it falls short on multimodal reasoning and interleaved generation. Furthermore, results suggest considerable headroom for improvement in audio generation, highlighting an important direction for future research.
>
---
#### [new 192] Exploring EFL Secondary Students' AI-generated Text Editing While Composition Writing
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 该研究通过混合方法探究EFL中学生在写作中对AI生成文本的编辑策略，分析29名香港学生的屏幕录像，识别15类编辑行为及四种规划- drafting模式，揭示其与AI互动的复杂认知过程，挑战被动使用假设，为AI写作教学提供依据。**

- **链接: [http://arxiv.org/pdf/2505.17041v1](http://arxiv.org/pdf/2505.17041v1)**

> **作者:** David James Woo; Yangyang Yu; Kai Guo
>
> **备注:** 31 pages, 16 figures
>
> **摘要:** Generative Artificial Intelligence is transforming how English as a foreign language students write. Still, little is known about how students manipulate text generated by generative AI during the writing process. This study investigates how EFL secondary school students integrate and modify AI-generated text when completing an expository writing task. The study employed an exploratory mixed-methods design. Screen recordings were collected from 29 Hong Kong secondary school students who attended an AI-assisted writing workshop and recorded their screens while using generative AI to write an article. Content analysis with hierarchical coding and thematic analysis with a multiple case study approach were adopted to analyze the recordings. 15 types of AI-generated text edits across seven categories were identified from the recordings. Notably, AI-initiated edits from iOS and Google Docs emerged as unanticipated sources of AI-generated text. A thematic analysis revealed four patterns of students' editing behaviors based on planning and drafting direction: planning with top-down drafting and revising; top-down drafting and revising without planning; planning with bottom-up drafting and revising; and bottom-up drafting and revising without planning. Network graphs illustrate cases of each pattern, demonstrating that students' interactions with AI-generated text involve more complex cognitive processes than simple text insertion. The findings challenge assumptions about students' passive, simplistic use of generative AI tools and have implications for developing explicit instructional approaches to teaching AI-generated text editing strategies in the AFL writing pedagogy.
>
---
#### [new 193] GSDFuse: Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis
- **分类: cs.CR; cs.AI; cs.CL; 68P30, 68T07; I.2.7**

- **简介: 该论文属于社交媒体隐写分析任务，旨在解决恶意信息隐藏检测中的认知不一致识别与多维弱信号聚合难题。提出GSDFuse方法，通过多模态特征工程提取多元信号、数据增强缓解稀疏性、自适应融合弱信号及判别式嵌入学习，提升复杂对话中的隐写检测性能，实验验证其效果达当前最优。**

- **链接: [http://arxiv.org/pdf/2505.17085v1](http://arxiv.org/pdf/2505.17085v1)**

> **作者:** Kaibo Huang; Zipei Zhang; Yukun Wei; TianXin Zhang; Zhongliang Yang; Linna Zhou
>
> **摘要:** The ubiquity of social media platforms facilitates malicious linguistic steganography, posing significant security risks. Steganalysis is profoundly hindered by the challenge of identifying subtle cognitive inconsistencies arising from textual fragmentation and complex dialogue structures, and the difficulty in achieving robust aggregation of multi-dimensional weak signals, especially given extreme steganographic sparsity and sophisticated steganography. These core detection difficulties are compounded by significant data imbalance. This paper introduces GSDFuse, a novel method designed to systematically overcome these obstacles. GSDFuse employs a holistic approach, synergistically integrating hierarchical multi-modal feature engineering to capture diverse signals, strategic data augmentation to address sparsity, adaptive evidence fusion to intelligently aggregate weak signals, and discriminative embedding learning to enhance sensitivity to subtle inconsistencies. Experiments on social media datasets demonstrate GSDFuse's state-of-the-art (SOTA) performance in identifying sophisticated steganography within complex dialogue environments. The source code for GSDFuse is available at https://github.com/NebulaEmmaZh/GSDFuse.
>
---
#### [new 194] TrimR: Verifier-based Training-Free Thinking Compression for Efficient Test-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型推理效率优化任务，旨在解决测试时扩展Chain-of-Thought（CoT）产生的冗余计算问题。提出TrimR框架，利用轻量验证器动态检测并裁剪大模型的冗余中间思考，无需微调，提升推理效率。实验显示其在多任务上将推理时间减少70%且精度影响微小。**

- **链接: [http://arxiv.org/pdf/2505.17155v1](http://arxiv.org/pdf/2505.17155v1)**

> **作者:** Weizhe Lin; Xing Li; Zhiyuan Yang; Xiaojin Fu; Hui-Ling Zhen; Yaoyuan Wang; Xianzhi Yu; Wulong Liu; Xiaosong Li; Mingxuan Yuan
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate exceptional capability in tackling complex mathematical, logical, and coding tasks by leveraging extended Chain-of-Thought (CoT) reasoning. Test-time scaling methods, such as prolonging CoT with explicit token-level exploration, can push LRMs' accuracy boundaries, but they incur significant decoding overhead. A key inefficiency source is LRMs often generate redundant thinking CoTs, which demonstrate clear structured overthinking and underthinking patterns. Inspired by human cognitive reasoning processes and numerical optimization theories, we propose TrimR, a verifier-based, training-free, efficient framework for dynamic CoT compression to trim reasoning and enhance test-time scaling, explicitly tailored for production-level deployment. Our method employs a lightweight, pretrained, instruction-tuned verifier to detect and truncate redundant intermediate thoughts of LRMs without any LRM or verifier fine-tuning. We present both the core algorithm and asynchronous online system engineered for high-throughput industrial applications. Empirical evaluations on Ascend NPUs and vLLM show that our framework delivers substantial gains in inference efficiency under large-batch workloads. In particular, on the four MATH500, AIME24, AIME25, and GPQA benchmarks, the reasoning runtime of Pangu-R-38B, QwQ-32B, and DeepSeek-R1-Distill-Qwen-32B is improved by up to 70% with negligible impact on accuracy.
>
---
#### [new 195] Gaming Tool Preferences in Agentic LLMs
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 论文研究代理LLMs工具选择协议的漏洞，发现工具描述的细微修改可显著影响LLM的选择（如GPT-4.1和Qwen2.5使用率提升10倍）。通过对比不同模型及编辑策略，揭示现有机制脆弱性，强调需建立更可靠的工具选择基础。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18135v1](http://arxiv.org/pdf/2505.18135v1)**

> **作者:** Kazem Faghih; Wenxiao Wang; Yize Cheng; Siddhant Bharti; Gaurang Sriramanan; Sriram Balasubramanian; Parsa Hosseini; Soheil Feizi
>
> **摘要:** Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 10 different models. These phenomenons, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources.
>
---
#### [new 196] CHAOS: Chart Analysis with Outlier Samples
- **分类: cs.CV; cs.CL**

- **简介: 论文提出CHAOS基准，评估多模态大语言模型处理带噪声图表的鲁棒性，解决其在异常图表中表现不佳的问题。设计5类文本、10类视觉扰动及三难度等级，测试13种模型，分析ChartQA和Chart-to-Text任务，揭示模型弱点并指导研究。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17235v1](http://arxiv.org/pdf/2505.17235v1)**

> **作者:** Omar Moured; Yufan Chen; Ruiping Liu; Simon Reiß; Philip Torr; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Data and code are publicly available at: http://huggingface.co/datasets/omoured/CHAOS
>
> **摘要:** Charts play a critical role in data analysis and visualization, yet real-world applications often present charts with challenging or noisy features. However, "outlier charts" pose a substantial challenge even for Multimodal Large Language Models (MLLMs), which can struggle to interpret perturbed charts. In this work, we introduce CHAOS (CHart Analysis with Outlier Samples), a robustness benchmark to systematically evaluate MLLMs against chart perturbations. CHAOS encompasses five types of textual and ten types of visual perturbations, each presented at three levels of severity (easy, mid, hard) inspired by the study result of human evaluation. The benchmark includes 13 state-of-the-art MLLMs divided into three groups (i.e., general-, document-, and chart-specific models) according to the training scope and data. Comprehensive analysis involves two downstream tasks (ChartQA and Chart-to-Text). Extensive experiments and case studies highlight critical insights into robustness of models across chart perturbations, aiming to guide future research in chart understanding domain. Data and code are publicly available at: http://huggingface.co/datasets/omoured/CHAOS.
>
---
#### [new 197] OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OCR-Reasoning基准，评估多模态大模型在文本密集图像推理中的能力。针对现有方法缺乏系统评测的问题，构建含1,069个标注样本的基准，覆盖6类推理能力及18项任务，并首次标注推理过程。实验显示现有模型准确率不足50%，凸显该任务挑战性。**

- **链接: [http://arxiv.org/pdf/2505.17163v1](http://arxiv.org/pdf/2505.17163v1)**

> **作者:** Mingxin Huang; Yongxin Shi; Dezhi Peng; Songxuan Lai; Zecheng Xie; Lianwen Jin
>
> **摘要:** Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. The benchmark comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Furthermore, unlike other text-rich image understanding benchmarks that only annotate the final answers, OCR-Reasoning also annotates the reasoning process simultaneously. With the annotated reasoning process and the final answers, OCR-Reasoning evaluates not only the final answers generated by models but also their reasoning processes, enabling a holistic analysis of their problem-solving abilities. Leveraging this benchmark, we conducted a comprehensive evaluation of state-of-the-art MLLMs. Our results demonstrate the limitations of existing methodologies. Notably, even state-of-the-art MLLMs exhibit substantial difficulties, with none achieving accuracy surpassing 50\% across OCR-Reasoning, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at https://github.com/SCUT-DLVCLab/OCR-Reasoning.
>
---
#### [new 198] ECHO-LLaMA: Efficient Caching for High-Performance LLaMA Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ECHO-LLaMA，针对LLaMA训练中KV缓存计算开销大的问题，通过跨层共享KV缓存机制降低计算复杂度，提升训练与推理效率。实验显示其训练吞吐量提升77%，计算利用率提高16%，损失降低14%，实现高效低成本的模型训练。**

- **链接: [http://arxiv.org/pdf/2505.17331v1](http://arxiv.org/pdf/2505.17331v1)**

> **作者:** Maryam Dialameh; Rezaul Karim; Hossein Rajabzadeh; Omar Mohamed Awad; Hyock Ju Kwon; Boxing Chen; Walid Ahmed; Yang Liu
>
> **摘要:** This paper introduces ECHO-LLaMA, an efficient LLaMA architecture designed to improve both the training speed and inference throughput of LLaMA architectures while maintaining its learning capacity. ECHO-LLaMA transforms LLaMA models into shared KV caching across certain layers, significantly reducing KV computational complexity while maintaining or improving language performance. Experimental results demonstrate that ECHO-LLaMA achieves up to 77\% higher token-per-second throughput during training, up to 16\% higher Model FLOPs Utilization (MFU), and up to 14\% lower loss when trained on an equal number of tokens. Furthermore, on the 1.1B model, ECHO-LLaMA delivers approximately 7\% higher test-time throughput compared to the baseline. By introducing a computationally efficient adaptation mechanism, ECHO-LLaMA offers a scalable and cost-effective solution for pretraining and finetuning large language models, enabling faster and more resource-efficient training without compromising performance.
>
---
#### [new 199] VideoGameBench: Can Vision-Language Models complete popular video games?
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出VideoGameBench，评估视觉语言模型（VLMs）在实时游戏任务中的能力，解决其在感知、空间导航等人类强项任务上的不足。通过设计包含10款90年代游戏的基准（3款隐藏），要求模型仅用视觉输入和目标描述完成游戏，发现前沿模型因推理延迟表现差（Gemini 2.5 Pro仅完成0.48%），故推出暂停等待的Lite版本（1.6%）。旨在推动VLMs相关研究。**

- **链接: [http://arxiv.org/pdf/2505.18134v1](http://arxiv.org/pdf/2505.18134v1)**

> **作者:** Alex L. Zhang; Thomas L. Griffiths; Karthik R. Narasimhan; Ofir Press
>
> **备注:** 9 pages, 33 pages including supplementary
>
> **摘要:** Vision-language models (VLMs) have achieved strong results on coding and math benchmarks that are challenging for humans, yet their ability to perform tasks that come naturally to humans--such as perception, spatial navigation, and memory management--remains understudied. Real video games are crafted to be intuitive for humans to learn and master by leveraging innate inductive biases, making them an ideal testbed for evaluating such capabilities in VLMs. To this end, we introduce VideoGameBench, a benchmark consisting of 10 popular video games from the 1990s that VLMs directly interact with in real-time. VideoGameBench challenges models to complete entire games with access to only raw visual inputs and a high-level description of objectives and controls, a significant departure from existing setups that rely on game-specific scaffolding and auxiliary information. We keep three of the games secret to encourage solutions that generalize to unseen environments. Our experiments show that frontier vision-language models struggle to progress beyond the beginning of each game. We find inference latency to be a major limitation of frontier models in the real-time setting; therefore, we introduce VideoGameBench Lite, a setting where the game pauses while waiting for the LM's next action. The best performing model, Gemini 2.5 Pro, completes only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite. We hope that the formalization of the human skills mentioned above into this benchmark motivates progress in these research directions.
>
---
#### [new 200] Data Mixing Can Induce Phase Transitions in Knowledge Acquisition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究数据混合对LLMs知识获取的影响。解决知识从高密度数据集的混合训练中出现相变而非平滑扩展的问题。通过合成数据实验发现模型规模/混合比例临界点引发突变，提出容量分配理论并揭示幂律关系，表明混合策略需根据模型大小调整。**

- **链接: [http://arxiv.org/pdf/2505.18091v1](http://arxiv.org/pdf/2505.18091v1)**

> **作者:** Xinran Gu; Kaifeng Lyu; Jiazheng Li; Jingzhao Zhang
>
> **摘要:** Large Language Models (LLMs) are typically trained on data mixtures: most data come from web scrapes, while a small portion is curated from high-quality sources with dense domain-specific knowledge. In this paper, we show that when training LLMs on such data mixtures, knowledge acquisition from knowledge-dense datasets, unlike training exclusively on knowledge-dense data (arXiv:2404.05405), does not always follow a smooth scaling law but can exhibit phase transitions with respect to the mixing ratio and model size. Through controlled experiments on a synthetic biography dataset mixed with web-scraped data, we demonstrate that: (1) as we increase the model size to a critical value, the model suddenly transitions from memorizing very few to most of the biographies; (2) below a critical mixing ratio, the model memorizes almost nothing even with extensive training, but beyond this threshold, it rapidly memorizes more biographies. We attribute these phase transitions to a capacity allocation phenomenon: a model with bounded capacity must act like a knapsack problem solver to minimize the overall test loss, and the optimal allocation across datasets can change discontinuously as the model size or mixing ratio varies. We formalize this intuition in an information-theoretic framework and reveal that these phase transitions are predictable, with the critical mixing ratio following a power-law relationship with the model size. Our findings highlight a concrete case where a good mixing recipe for large models may not be optimal for small models, and vice versa.
>
---
#### [new 201] Towards Practical Defect-Focused Automated Code Review
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出缺陷导向的自动化代码审查方法，针对现有技术忽略上下文、缺陷检测及工作流整合的问题，提出代码切片提取上下文、多角色LLM提升缺陷识别、过滤机制降误报及交互优化，实现在大规模C++代码中的显著效果提升，框架具备语言扩展性。**

- **链接: [http://arxiv.org/pdf/2505.17928v1](http://arxiv.org/pdf/2505.17928v1)**

> **作者:** Junyi Lu; Lili Jiang; Xiaojia Li; Jianbing Fang; Fengjun Zhang; Li Yang; Chun Zuo
>
> **备注:** Accepted to Forty-Second International Conference on Machine Learning (ICML 2025)
>
> **摘要:** The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: 1) capturing relevant context, 2) improving key bug inclusion (KBI), 3) reducing false alarm rates (FAR), and 4) integrating human workflows. To tackle these, we propose 1) code slicing algorithms for context extraction, 2) a multi-role LLM framework for KBI, 3) a filtering mechanism for FAR reduction, and 4) a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2x improvement over standard LLMs and a 10x gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages language-agnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability.
>
---
#### [new 202] Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective
- **分类: cs.LG; cs.CL**

- **简介: 论文从理论角度分析VAPO框架的局限性，探讨其在长链推理任务中的价值函数近似、自适应优势估计及token优化等问题，指出假设挑战，旨在提升强化学习推理代理的鲁棒性和泛化能力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17997v1](http://arxiv.org/pdf/2505.17997v1)**

> **作者:** Jintian Shao; Yiming Cheng; Hongyi Huang; Beiwen Zhang; Zhiyu Wu; You Shan; Mingkai Zheng
>
> **摘要:** The VAPO framework has demonstrated significant empirical success in enhancing the efficiency and reliability of reinforcement learning for long chain-of-thought (CoT) reasoning tasks with large language models (LLMs). By systematically addressing challenges such as value model bias, heterogeneous sequence lengths, and sparse reward signals, VAPO achieves state-of-the-art performance. While its practical benefits are evident, a deeper theoretical understanding of its underlying mechanisms and potential limitations is crucial for guiding future advancements. This paper aims to initiate such a discussion by exploring VAPO from a theoretical perspective, highlighting areas where its assumptions might be challenged and where further investigation could yield more robust and generalizable reasoning agents. We delve into the intricacies of value function approximation in complex reasoning spaces, the optimality of adaptive advantage estimation, the impact of token-level optimization, and the enduring challenges of exploration and generalization.
>
---
#### [new 203] CAMA: Enhancing Multimodal In-Context Learning with Context-Aware Modulated Attention
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态in-context学习（ICL）的不稳定性问题，分析了标准注意力机制的三大缺陷，提出无需训练的CAMA方法，通过调制注意力logits优化模型内部机制，在多个模型和基准上验证了其有效性，属多模态ICL优化任务。**

- **链接: [http://arxiv.org/pdf/2505.17097v1](http://arxiv.org/pdf/2505.17097v1)**

> **作者:** Yanshu Li; JianJiang Yang; Bozheng Li; Ruixiang Tang
>
> **备注:** 10 pages, 2 figures, 6 tables
>
> **摘要:** Multimodal in-context learning (ICL) enables large vision-language models (LVLMs) to efficiently adapt to novel tasks, supporting a wide array of real-world applications. However, multimodal ICL remains unstable, and current research largely focuses on optimizing sequence configuration while overlooking the internal mechanisms of LVLMs. In this work, we first provide a theoretical analysis of attentional dynamics in multimodal ICL and identify three core limitations of standard attention that ICL impair performance. To address these challenges, we propose Context-Aware Modulated Attention (CAMA), a simple yet effective plug-and-play method for directly calibrating LVLM attention logits. CAMA is training-free and can be seamlessly applied to various open-source LVLMs. We evaluate CAMA on four LVLMs across six benchmarks, demonstrating its effectiveness and generality. CAMA opens new opportunities for deeper exploration and targeted utilization of LVLM attention dynamics to advance multimodal reasoning.
>
---
#### [new 204] Speechless: Speech Instruction Training Without Speech for Low Resource Languages
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出Speechless方法，针对低资源语言语音助手训练中缺乏语音指令数据及TTS模型的问题，通过在语义表示层合成数据并对其与Whisper编码器对齐，使大模型仅用文本指令微调即可处理语音指令，简化低资源语言语音系统开发。**

- **链接: [http://arxiv.org/pdf/2505.17417v1](http://arxiv.org/pdf/2505.17417v1)**

> **作者:** Alan Dao; Dinh Bach Vu; Huy Hoang Ha; Tuan Le Duc Anh; Shreyas Gopal; Yue Heng Yeo; Warren Keng Hoong Low; Eng Siong Chng; Jia Qi Yip
>
> **备注:** This paper was accepted by INTERSPEECH 2025
>
> **摘要:** The rapid growth of voice assistants powered by large language models (LLM) has highlighted a need for speech instruction data to train these systems. Despite the abundance of speech recognition data, there is a notable scarcity of speech instruction data, which is essential for fine-tuning models to understand and execute spoken commands. Generating high-quality synthetic speech requires a good text-to-speech (TTS) model, which may not be available to low resource languages. Our novel approach addresses this challenge by halting synthesis at the semantic representation level, bypassing the need for TTS. We achieve this by aligning synthetic semantic representations with the pre-trained Whisper encoder, enabling an LLM to be fine-tuned on text instructions while maintaining the ability to understand spoken instructions during inference. This simplified training process is a promising approach to building voice assistant for low-resource languages.
>
---
#### [new 205] Value-Guided Search for Efficient Chain-of-Thought Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出一种无需细粒度步骤定义的长上下文推理价值引导搜索（VGS）方法，解决现有过程奖励模型（PRMs）依赖复杂步骤划分的问题。通过训练15亿参数价值模型并结合块级搜索与加权投票，在数学竞赛任务中以更低计算成本达到与先进模型相当的准确率，开源数据及代码。**

- **链接: [http://arxiv.org/pdf/2505.17373v1](http://arxiv.org/pdf/2505.17373v1)**

> **作者:** Kaiwen Wang; Jin Peng Zhou; Jonathan Chang; Zhaolin Gao; Nathan Kallus; Kianté Brantley; Wen Sun
>
> **摘要:** In this paper, we propose a simple and efficient method for value model training on long-context reasoning traces. Compared to existing process reward models (PRMs), our method does not require a fine-grained notion of "step," which is difficult to define for long-context reasoning models. By collecting a dataset of 2.5 million reasoning traces, we train a 1.5B token-level value model and apply it to DeepSeek models for improved performance with test-time compute scaling. We find that block-wise value-guided search (VGS) with a final weighted majority vote achieves better test-time scaling than standard methods such as majority voting or best-of-n. With an inference budget of 64 generations, VGS with DeepSeek-R1-Distill-1.5B achieves an average accuracy of 45.7% across four competition math benchmarks (AIME 2024 & 2025, HMMT Feb 2024 & 2025), reaching parity with o3-mini-medium. Moreover, VGS significantly reduces the inference FLOPs required to achieve the same performance of majority voting. Our dataset, model and codebase are open-sourced.
>
---
#### [new 206] Controlled Agentic Planning & Reasoning for Mechanism Synthesis
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出基于双代理LLM的机制合成方法，解决从自然语言到几何/动态设计的自动化推理问题。通过语言与符号层面的闭环推理生成模拟代码，结合符号回归优化，实现平面机构高效合成。提出新基准MSynth，验证模型组件有效性并揭示大模型在符号推理中的优势。**

- **链接: [http://arxiv.org/pdf/2505.17607v1](http://arxiv.org/pdf/2505.17607v1)**

> **作者:** João Pedro Gandarela; Thiago Rios; Stefan Menzel; André Freitas
>
> **备注:** 24 pages, 16 figures
>
> **摘要:** This work presents a dual-agent Large Language Model (LLM)-based reasoning method for mechanism synthesis, capable of reasoning at both linguistic and symbolic levels to generate geometrical and dynamic outcomes. The model consists of a composition of well-defined functions that, starting from a natural language specification, references abstract properties through supporting equations, generates and parametrizes simulation code, and elicits feedback anchor points using symbolic regression and distance functions. This process closes an actionable refinement loop at the linguistic and symbolic layers. The approach is shown to be both effective and convergent in the context of planar mechanisms. Additionally, we introduce MSynth, a novel benchmark for planar mechanism synthesis, and perform a comprehensive analysis of the impact of the model components. We further demonstrate that symbolic regression prompts unlock mechanistic insights only when applied to sufficiently large architectures.
>
---
#### [new 207] Large language model as user daily behavior data generator: balancing population diversity and individual personality
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文提出BehaviorGen框架，利用大语言模型生成兼顾群体多样性和个体特征的合成用户行为数据，解决真实数据依赖带来的隐私与可用性问题。通过模拟用户行为进行数据增强与替换，在移动和手机使用预测任务中提升18.9%，验证了隐私保护下合成数据增强行为建模的潜力。**

- **链接: [http://arxiv.org/pdf/2505.17615v1](http://arxiv.org/pdf/2505.17615v1)**

> **作者:** Haoxin Li; Jingtao Ding; Jiahui Gong; Yong Li
>
> **备注:** 14 pages, 7 figures, 4 tables
>
> **摘要:** Predicting human daily behavior is challenging due to the complexity of routine patterns and short-term fluctuations. While data-driven models have improved behavior prediction by leveraging empirical data from various platforms and devices, the reliance on sensitive, large-scale user data raises privacy concerns and limits data availability. Synthetic data generation has emerged as a promising solution, though existing methods are often limited to specific applications. In this work, we introduce BehaviorGen, a framework that uses large language models (LLMs) to generate high-quality synthetic behavior data. By simulating user behavior based on profiles and real events, BehaviorGen supports data augmentation and replacement in behavior prediction models. We evaluate its performance in scenarios such as pertaining augmentation, fine-tuning replacement, and fine-tuning augmentation, achieving significant improvements in human mobility and smartphone usage predictions, with gains of up to 18.9%. Our results demonstrate the potential of BehaviorGen to enhance user behavior modeling through flexible and privacy-preserving synthetic data generation.
>
---
#### [new 208] COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型计算效率优化任务，旨在解决FFNN层计算冗余问题。提出COUNTDOWN方法，通过线性组合分析下投影矩阵的全局稀疏性，设计M/D两种策略动态移除非必要权重，实现计算量减少90%且性能损失低（D版最优），并提供无预测器的M版方案。**

- **链接: [http://arxiv.org/pdf/2505.17701v1](http://arxiv.org/pdf/2505.17701v1)**

> **作者:** Jaewon Cheon; Pilsung Kang
>
> **摘要:** The growing size of large language models has created significant computational inefficiencies. To address this challenge, sparse activation methods selectively deactivates non-essential parameters during inference, reducing computational costs in FFNN layers. While existing methods focus on non-linear gating mechanisms, we hypothesize that the sparsity of the FFNN layer lies globally in the form of a linear combination over its internal down projection matrix. Based on this insight, we propose two methods: M-COUNTDOWN, leveraging indirect coefficients, and D-COUNTDOWN, utilizing direct coefficients of the linear combination. Experimental results demonstrate that D-COUNTDOWN can omit 90% of computations with performance loss as low as 5.5% ideally, while M-COUNTDOWN provides a predictor-free solution with up to 29.4% better performance preservation compared to existing methods. Our specialized kernel implementations effectively realize these theoretical gains into substantial real-world acceleration.
>
---
#### [new 209] Chart-to-Experience: Benchmarking Multimodal LLMs for Predicting Experiential Impact of Charts
- **分类: cs.HC; cs.CL**

- **简介: 该论文评估多模态大语言模型（MLLMs）预测图表感知与情感影响的能力，解决其性能验证不足的问题。构建含36图表的基准数据集，通过众包评估7个体验因素，测试模型在直接预测与成对比较任务中的表现，发现模型个体评估敏感度低于人类，但成对比较更可靠。**

- **链接: [http://arxiv.org/pdf/2505.17374v1](http://arxiv.org/pdf/2505.17374v1)**

> **作者:** Seon Gyeom Kim; Jae Young Choi; Ryan Rossi; Eunyee Koh; Tak Yeon Lee
>
> **备注:** This paper has been accepted to IEEE PacificVis 2025
>
> **摘要:** The field of Multimodal Large Language Models (MLLMs) has made remarkable progress in visual understanding tasks, presenting a vast opportunity to predict the perceptual and emotional impact of charts. However, it also raises concerns, as many applications of LLMs are based on overgeneralized assumptions from a few examples, lacking sufficient validation of their performance and effectiveness. We introduce Chart-to-Experience, a benchmark dataset comprising 36 charts, evaluated by crowdsourced workers for their impact on seven experiential factors. Using the dataset as ground truth, we evaluated capabilities of state-of-the-art MLLMs on two tasks: direct prediction and pairwise comparison of charts. Our findings imply that MLLMs are not as sensitive as human evaluators when assessing individual charts, but are accurate and reliable in pairwise comparisons.
>
---
#### [new 210] Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究长上下文能力对模型推理性能的影响。针对现有模型推理局限可能源于长上下文能力不足的问题，作者通过对比相同架构但长上下文能力不同的模型发现：增强长上下文能力后，监督微调模型在推理任务（包括短输入）中表现更优。任务属模型优化，工作为实验验证长上下文能力的普适性提升作用。**

- **链接: [http://arxiv.org/pdf/2505.17315v1](http://arxiv.org/pdf/2505.17315v1)**

> **作者:** Wang Yang; Zirui Liu; Hongye Jin; Qingyu Yin; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Recent language models exhibit strong reasoning capabilities, yet the influence of long-context capacity on reasoning remains underexplored. In this work, we hypothesize that current limitations in reasoning stem, in part, from insufficient long-context capacity, motivated by empirical observations such as (1) higher context window length often leads to stronger reasoning performance, and (2) failed reasoning cases resemble failed long-context cases. To test this hypothesis, we examine whether enhancing a model's long-context ability before Supervised Fine-Tuning (SFT) leads to improved reasoning performance. Specifically, we compared models with identical architectures and fine-tuning data but varying levels of long-context capacity. Our results reveal a consistent trend: models with stronger long-context capacity achieve significantly higher accuracy on reasoning benchmarks after SFT. Notably, these gains persist even on tasks with short input lengths, indicating that long-context training offers generalizable benefits for reasoning performance. These findings suggest that long-context modeling is not just essential for processing lengthy inputs, but also serves as a critical foundation for reasoning. We advocate for treating long-context capacity as a first-class objective in the design of future language models.
>
---
#### [new 211] ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于LLM可解释性任务，旨在解决现有特征交互分析方法计算效率低的问题。提出ProxySPEX算法，利用特征交互的层次性，通过拟合梯度提升树并提取关键交互，减少10倍推理开销，提升重构精度。应用于数据归因和机制解释，识别更有效的特征/注意力头交互。**

- **链接: [http://arxiv.org/pdf/2505.17495v1](http://arxiv.org/pdf/2505.17495v1)**

> **作者:** Landon Butler; Abhineet Agarwal; Justin Singh Kang; Yigit Efe Erginbas; Bin Yu; Kannan Ramchandran
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX identifies features that influence model output over 20% more than those selected by marginal approaches. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. ProxySPEX identifies interactions that enable more aggressive pruning of heads than marginal approaches.
>
---
#### [new 212] What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection
- **分类: cs.LG; cs.CL; cs.SD; eess.AS; 53-04**

- **简介: 该论文研究深度伪造语音检测中的语言敏感性，通过文本级对抗攻击揭示现有系统（含商业产品）对语言变异的脆弱性。发现轻微语言扰动使检测准确率骤降（如某商业系统从100%降至32%），分析显示语言复杂度和音频嵌入相似度是关键因素，提出需结合语言特征设计更鲁棒的检测系统。**

- **链接: [http://arxiv.org/pdf/2505.17513v1](http://arxiv.org/pdf/2505.17513v1)**

> **作者:** Binh Nguyen; Shuji Shi; Ryan Ofman; Thai Le
>
> **备注:** 15 pages, 2 fogures
>
> **摘要:** Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.
>
---
#### [new 213] PPO-BR: Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习策略优化任务，针对PPO算法静态信任区域导致探索不足与收敛不稳定的问题，提出PPO-BR方法。通过融合熵驱动的探索扩张（高不确定性状态扩展信任区域）和奖励引导的收敛收缩（后期稳定收缩），实现动态自适应调整，提升收敛速度与稳定性，在多个基准测试中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.17714v1](http://arxiv.org/pdf/2505.17714v1)**

> **作者:** Ben Rahman
>
> **备注:** This manuscript builds upon an earlier version posted to TechRxiv. This arXiv version includes an updated comparison with GRPO (Group Relative Policy Optimization)
>
> **摘要:** Despite Proximal Policy Optimization (PPO) dominating policy gradient methods -- from robotic control to game AI -- its static trust region forces a brittle trade-off: aggressive clipping stifles early exploration, while late-stage updates destabilize convergence. PPO-BR establishes a new paradigm in adaptive RL by fusing exploration and convergence signals into a single bounded trust region -- a theoretically grounded innovation that outperforms five SOTA baselines with less than 2% overhead. This work bridges a critical gap in phase-aware learning, enabling real-world deployment in safety-critical systems like robotic surgery within a single adaptive mechanism. PPO-BR achieves 29.1% faster convergence by combining: (1) entropy-driven expansion (epsilon up) for exploration in high-uncertainty states, and (2) reward-guided contraction (epsilon down) for convergence stability. On six diverse benchmarks (MuJoCo, Atari, sparse-reward), PPO-BR achieves 29.1% faster convergence (p < 0.001), 2.3x lower reward variance than PPO, and less than 1.8% runtime overhead with only five lines of code change. PPO-BR's simplicity and theoretical guarantees make it ready-to-deploy in safety-critical domains -- from surgical robotics to autonomous drones. In contrast to recent methods such as Group Relative Policy Optimization (GRPO), PPO-BR offers a unified entropy-reward mechanism applicable to both language models and general reinforcement learning environments.
>
---
#### [new 214] How Can I Publish My LLM Benchmark Without Giving the True Answers Away?
- **分类: cs.LG; cs.AI; cs.CL; stat.ME**

- **简介: 该论文提出一种发布LLM基准测试的新方法，通过为每题生成多个正确答案并随机选一作为标准答案，既避免泄露真实答案，又通过贝叶斯准确率上限检测模型数据污染。任务为解决基准公开导致的模型过拟合与数据泄露问题，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.18102v1](http://arxiv.org/pdf/2505.18102v1)**

> **作者:** Takashi Ishida; Thanawat Lodkaew; Ikko Yamane
>
> **摘要:** Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
>
---
#### [new 215] Zebra-Llama: Towards Extremely Efficient Hybrid Models
- **分类: cs.LG; cs.CL**

- **简介: 论文提出Zebra-Llama混合模型，结合SSM与MLA架构，通过改进知识蒸馏方法，利用预训练Transformer提升效率。解决LLM部署成本高、推理低效问题，其1B/3B/8B模型仅需少量训练数据（7-11B tokens），KV缓存缩减至2-4%，保持高精度，性能超同类模型（如吞吐量提升2.6-3.8倍）。**

- **链接: [http://arxiv.org/pdf/2505.17272v1](http://arxiv.org/pdf/2505.17272v1)**

> **作者:** Mingyu Yang; Mehdi Rezagholizadeh; Guihong Li; Vikram Appia; Emad Barsoum
>
> **摘要:** With the growing demand for deploying large language models (LLMs) across diverse applications, improving their inference efficiency is crucial for sustainable and democratized access. However, retraining LLMs to meet new user-specific requirements is prohibitively expensive and environmentally unsustainable. In this work, we propose a practical and scalable alternative: composing efficient hybrid language models from existing pre-trained models. Our approach, Zebra-Llama, introduces a family of 1B, 3B, and 8B hybrid models by combining State Space Models (SSMs) and Multi-head Latent Attention (MLA) layers, using a refined initialization and post-training pipeline to efficiently transfer knowledge from pre-trained Transformers. Zebra-Llama achieves Transformer-level accuracy with near-SSM efficiency using only 7-11B training tokens (compared to trillions of tokens required for pre-training) and an 8B teacher. Moreover, Zebra-Llama dramatically reduces KV cache size -down to 3.9%, 2%, and 2.73% of the original for the 1B, 3B, and 8B variants, respectively-while preserving 100%, 100%, and >97% of average zero-shot performance on LM Harness tasks. Compared to models like MambaInLLaMA, X-EcoMLA, Minitron, and Llamba, Zebra-Llama consistently delivers competitive or superior accuracy while using significantly fewer tokens, smaller teachers, and vastly reduced KV cache memory. Notably, Zebra-Llama-8B surpasses Minitron-8B in few-shot accuracy by 7% while using 8x fewer training tokens, over 12x smaller KV cache, and a smaller teacher (8B vs. 15B). It also achieves 2.6x-3.8x higher throughput (tokens/s) than MambaInLlama up to a 32k context length. We will release code and model checkpoints upon acceptance.
>
---
#### [new 216] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）推理的在线强化学习任务，旨在解决KL正则化策略梯度方法的设计与优化问题。提出系统性框架RPG，推导了前向/反向KL正则化下的策略梯度及损失函数，支持归一化/非归一化策略分布，并通过实验验证了方法在训练稳定性与性能上的优势。**

- **链接: [http://arxiv.org/pdf/2505.17508v1](http://arxiv.org/pdf/2505.17508v1)**

> **作者:** Yifan Zhang; Yifeng Liu; Huizhuo Yuan; Yang Yuan; Quanquan Gu; Andrew C Yao
>
> **备注:** 53 pages, 17 figures
>
> **摘要:** Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). Despite the widespread use of Kullback-Leibler (KL) regularization in policy gradient algorithms to stabilize training, the systematic exploration of how different KL divergence formulations can be estimated and integrated into surrogate loss functions for online reinforcement learning (RL) presents a nuanced and systematically explorable design space. In this paper, we propose regularized policy gradient (RPG), a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online RL setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. We conduct extensive experiments on RL for LLM reasoning using these methods, showing improved or competitive results in terms of training stability and performance compared to strong baselines such as GRPO, REINFORCE++, and DAPO. The code is available at https://github.com/complex-reasoning/RPG.
>
---
#### [new 217] From Reasoning to Generalization: Knowledge-Augmented LLMs for ARC Benchmark
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对LLM在抽象推理与泛化能力上的不足，以ARC基准为任务，提出KAAR方法。通过分层注入知识先验并结合RSPC求解器，逐步提升LLM推理能力，实验显示其显著提高模型性能。**

- **链接: [http://arxiv.org/pdf/2505.17482v1](http://arxiv.org/pdf/2505.17482v1)**

> **作者:** Chao Lei; Nir Lipovetzky; Krista A. Ehinger; Yanchuan Chang
>
> **摘要:** Recent reasoning-oriented LLMs have demonstrated strong performance on challenging tasks such as mathematics and science examinations. However, core cognitive faculties of human intelligence, such as abstract reasoning and generalization, remain underexplored. To address this, we evaluate recent reasoning-oriented LLMs on the Abstraction and Reasoning Corpus (ARC) benchmark, which explicitly demands both faculties. We formulate ARC as a program synthesis task and propose nine candidate solvers. Experimental results show that repeated-sampling planning-aided code generation (RSPC) achieves the highest test accuracy and demonstrates consistent generalization across most LLMs. To further improve performance, we introduce an ARC solver, Knowledge Augmentation for Abstract Reasoning (KAAR), which encodes core knowledge priors within an ontology that classifies priors into three hierarchical levels based on their dependencies. KAAR progressively expands LLM reasoning capacity by gradually augmenting priors at each level, and invokes RSPC to generate candidate solutions after each augmentation stage. This stage-wise reasoning reduces interference from irrelevant priors and improves LLM performance. Empirical results show that KAAR maintains strong generalization and consistently outperforms non-augmented RSPC across all evaluated LLMs, achieving around 5% absolute gains and up to 64.52% relative improvement. Despite these achievements, ARC remains a challenging benchmark for reasoning-oriented LLMs, highlighting future avenues of progress in LLMs.
>
---
#### [new 218] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型量化任务，针对现有均匀量化参数初始化依赖次优Min-Max策略的问题，提出NeUQI方法优化初始化参数，提升量化后性能。NeUQI可与现有方法结合，实验显示其优于现有方法，结合轻量蒸馏甚至超越资源密集的PV-tuning。**

- **链接: [http://arxiv.org/pdf/2505.17595v1](http://arxiv.org/pdf/2505.17595v1)**

> **作者:** Li Lin; Xinyu Hu; Xiaojun Wan
>
> **备注:** 9 pages, under review
>
> **摘要:** Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored for its efficiency and ease of deployment since uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on $\geq 2$-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they primarily focus on quantization methodologies, while the initialization of quantization parameters is underexplored and still relies on the suboptimal Min-Max strategies. In this work, we propose NeUQI, a method devoted to efficiently determining near-optimal initial parameters for uniform quantization. NeUQI is orthogonal to prior quantization methodologies and can seamlessly integrate with them. The experiments with the LLaMA and Qwen families on various tasks demonstrate that our NeUQI consistently outperforms existing methods. Furthermore, when combined with a lightweight distillation strategy, NeUQI can achieve superior performance to PV-tuning, a much more resource-intensive approach.
>
---
#### [new 219] Bridging Supervised Learning and Reinforcement Learning in Math Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Negative-aware Fine-Tuning（NFT），在数学推理任务中结合监督学习与强化学习，解决传统监督学习无法利用错误反馈自我改进的问题。通过建模负样本策略优化模型，NFT无需外部教师信号，显著超越SL基线并媲美RL算法，证明SL与RL在二元反馈系统中的内在关联。**

- **链接: [http://arxiv.org/pdf/2505.18116v1](http://arxiv.org/pdf/2505.18116v1)**

> **作者:** Huayu Chen; Kaiwen Zheng; Qinsheng Zhang; Ganqu Cui; Yin Cui; Haotian Ye; Tsung-Yi Lin; Ming-Yu Liu; Jun Zhu; Haoxiang Wang
>
> **摘要:** Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling self-improvement through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) -- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an implicit negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like Rejection sampling Fine-Tuning, matching or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they originate from entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems.
>
---
#### [new 220] Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于因果推理任务，旨在解决LLMs区分因果与相关关系泛化能力不足的问题。提出通过构建结构化知识图谱编码相关前提，引导模型系统推理。实验显示该方法使Qwen3-32B的F1值从32.71提升至48.26，验证了结构化思维对因果推理的提升效果。**

- **链接: [http://arxiv.org/pdf/2505.18034v1](http://arxiv.org/pdf/2505.18034v1)**

> **作者:** Wentao Sun; Joao Paulo Nogueira; Alonso Silva
>
> **摘要:** Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks.
>
---
#### [new 221] Attention with Trained Embeddings Provably Selects Important Tokens
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于NLP理论分析任务，研究softmax attention模型中token embeddings如何通过训练选择重要token。通过梯度下降分析证明：embeddings按token频率对齐输出向量，训练后softmax选择预测性token，使分类间隔最大化。实验验证了理论在IMDB和Yelp数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2505.17282v1](http://arxiv.org/pdf/2505.17282v1)**

> **作者:** Diyuan Wu; Aleksandr Shevchenko; Samet Oymak; Marco Mondelli
>
> **摘要:** Token embeddings play a crucial role in language modeling but, despite this practical relevance, their theoretical understanding remains limited. Our paper addresses the gap by characterizing the structure of embeddings obtained via gradient descent. Specifically, we consider a one-layer softmax attention model with a linear head for binary classification, i.e., $\texttt{Softmax}( p^\top E_X^\top ) E_X v = \frac{ \sum_{i=1}^T \exp(p^\top E_{x_i}) E_{x_i}^\top v}{\sum_{j=1}^T \exp(p^\top E_{x_{j}}) }$, where $E_X = [ E_{x_1} , \dots, E_{x_T} ]^\top$ contains the embeddings of the input sequence, $p$ is the embedding of the $\mathrm{\langle cls \rangle}$ token and $v$ the output vector. First, we show that, already after a single step of gradient training with the logistic loss, the embeddings $E_X$ capture the importance of tokens in the dataset by aligning with the output vector $v$ proportionally to the frequency with which the corresponding tokens appear in the dataset. Then, after training $p$ via gradient flow until convergence, the softmax selects the important tokens in the sentence (i.e., those that are predictive of the label), and the resulting $\mathrm{\langle cls \rangle}$ embedding maximizes the margin for such a selection. Experiments on real-world datasets (IMDB, Yelp) exhibit a phenomenology close to that unveiled by our theory.
>
---
#### [new 222] Voicing Personas: Rewriting Persona Descriptions into Style Prompts for Controllable Text-to-Speech
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于可控文本到语音合成任务，旨在将人物描述转化为可调节语音风格的提示，解决语音属性（如音调、情感、语速）精细控制问题。提出两种改写策略将通用人物描述转为语音导向提示，并分析LLM改写中的性别偏见，实验验证了方法提升合成语音质量。**

- **链接: [http://arxiv.org/pdf/2505.17093v1](http://arxiv.org/pdf/2505.17093v1)**

> **作者:** Yejin Lee; Jaehoon Kang; Kyuhong Shim
>
> **摘要:** In this paper, we propose a novel framework to control voice style in prompt-based, controllable text-to-speech systems by leveraging textual personas as voice style prompts. We present two persona rewriting strategies to transform generic persona descriptions into speech-oriented prompts, enabling fine-grained manipulation of prosodic attributes such as pitch, emotion, and speaking rate. Experimental results demonstrate that our methods enhance the naturalness, clarity, and consistency of synthesized speech. Finally, we analyze implicit social biases introduced by LLM-based rewriting, with a focus on gender. We underscore voice style as a crucial factor for persona-driven AI dialogue systems.
>
---
#### [new 223] OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OrionBench基准，旨在提升视觉语言模型在信息图表中检测图表和人类可识别对象（如图标）的准确性。针对现有模型在信息图元素定位上的不足，构建含26,250张真实/78,750张合成信息图及690万标注的数据集，并通过改进VLM推理方案、模型对比及布局检测应用验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.17473v1](http://arxiv.org/pdf/2505.17473v1)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [new 224] LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于ASR后处理的生成式错误纠正任务，针对罕见词纠正效果差及过度修正问题，提出合成数据增强罕见词训练数据，并整合ASR的N-best假设与语音上下文以减少过度修正，提升多语言纠错效果。**

- **链接: [http://arxiv.org/pdf/2505.17410v1](http://arxiv.org/pdf/2505.17410v1)**

> **作者:** Natsuo Yamashita; Masaaki Yamamoto; Hiroaki Kokubo; Yohei Kawaguchi
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets.
>
---
#### [new 225] Debiasing CLIP: Interpreting and Correcting Bias in Attention Heads
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于CLIP模型去偏任务，旨在解决其学习虚假关联（如背景/性别偏差）的问题。提出LTC框架，通过对比机制识别并消除视觉Transformer中的虚假注意力头，同时强化任务相关头，提升分类性能。实验显示最差群体准确率提升超50%。**

- **链接: [http://arxiv.org/pdf/2505.17425v1](http://arxiv.org/pdf/2505.17425v1)**

> **作者:** Wei Jie Yeo; Rui Mao; Moloud Abdar; Erik Cambria; Ranjan Satapathy
>
> **备注:** Under review
>
> **摘要:** Multimodal models like CLIP have gained significant attention due to their remarkable zero-shot performance across various tasks. However, studies have revealed that CLIP can inadvertently learn spurious associations between target variables and confounding factors. To address this, we introduce \textsc{Locate-Then-Correct} (LTC), a contrastive framework that identifies spurious attention heads in Vision Transformers via mechanistic insights and mitigates them through targeted ablation. Furthermore, LTC identifies salient, task-relevant attention heads, enabling the integration of discriminative features through orthogonal projection to improve classification performance. We evaluate LTC on benchmarks with inherent background and gender biases, achieving over a $>50\%$ gain in worst-group accuracy compared to non-training post-hoc baselines. Additionally, we visualize the representation of selected heads and find that the presented interpretation corroborates our contrastive mechanism for identifying both spurious and salient attention heads. Code available at https://github.com/wj210/CLIP_LTC.
>
---
#### [new 226] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出HoloLLM模型，属于多模态语言引导的人类感知与推理任务。针对视觉主导模型在现实场景中受遮挡、光照或隐私限制的问题，整合LiDAR、红外等多传感器数据。通过设计UMIP模块解决传感器数据与文本对齐难题，并构建协作数据 pipeline 提升标注质量，在新基准测试中提升30%感知精度。**

- **链接: [http://arxiv.org/pdf/2505.17645v1](http://arxiv.org/pdf/2505.17645v1)**

> **作者:** Chuhao Zhou; Jianfei Yang
>
> **备注:** 18 pages, 13 figures, 6 tables
>
> **摘要:** Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence.
>
---
#### [new 227] Self-Training Large Language Models with Confident Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型（LLM）的自训练任务，旨在解决现有方法仅通过最终答案的置信度（多数投票）进行自训练而忽略推理路径质量的问题。提出CORE-PO方法，利用推理级置信度筛选优质推理路径，并通过策略优化提升模型在分布内及分布外任务的推理准确率。**

- **链接: [http://arxiv.org/pdf/2505.17454v1](http://arxiv.org/pdf/2505.17454v1)**

> **作者:** Hyosoon Jang; Yunhui Jang; Sungjae Lee; Jungseul Ok; Sungsoo Ahn
>
> **摘要:** Large language models (LLMs) have shown impressive performance by generating reasoning paths before final answers, but learning such a reasoning path requires costly human supervision. To address this issue, recent studies have explored self-training methods that improve reasoning capabilities using pseudo-labels generated by the LLMs themselves. Among these, confidence-based self-training fine-tunes LLMs to prefer reasoning paths with high-confidence answers, where confidence is estimated via majority voting. However, such methods exclusively focus on the quality of the final answer and may ignore the quality of the reasoning paths, as even an incorrect reasoning path leads to a correct answer by chance. Instead, we advocate the use of reasoning-level confidence to identify high-quality reasoning paths for self-training, supported by our empirical observations. We then propose a new self-training method, CORE-PO, that fine-tunes LLMs to prefer high-COnfidence REasoning paths through Policy Optimization. Our experiments show that CORE-PO improves the accuracy of outputs on four in-distribution and two out-of-distribution benchmarks, compared to existing self-training methods.
>
---
#### [new 228] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于统一多模态理解和生成任务，旨在解决现有模型难以同时优化这两项能力的问题。提出CoRL框架，通过协同强化学习实现联合优化与任务细化，提升文本到图像生成和多模态理解性能。**

- **链接: [http://arxiv.org/pdf/2505.17534v1](http://arxiv.org/pdf/2505.17534v1)**

> **作者:** Jingjing Jiang; Chongjie Si; Jun Luo; Hanwang Zhang; Chao Ma
>
> **摘要:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce \textbf{CoRL}, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, \textbf{ULM-R1}, achieves average improvements of \textbf{7%} on three text-to-image generation datasets and \textbf{23%} on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs.
>
---
#### [new 229] CHART-6: Human-Centered Evaluation of Data Visualization Understanding in Vision-Language Models
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文评估视觉语言模型对数据可视化理解的类人能力。针对现有模型与人类评估标准脱节的问题，使用六个人类设计的测评任务对比八种模型与人类的表现，发现模型平均得分更低且错误模式显著不同，揭示了模型在认知模拟上的不足。任务属模型与人类行为对比，旨在改进数据可视化推理的AI系统开发。**

- **链接: [http://arxiv.org/pdf/2505.17202v1](http://arxiv.org/pdf/2505.17202v1)**

> **作者:** Arnav Verma; Kushin Mukherjee; Christopher Potts; Elisa Kreiss; Judith E. Fan
>
> **摘要:** Data visualizations are powerful tools for communicating patterns in quantitative data. Yet understanding any data visualization is no small feat -- succeeding requires jointly making sense of visual, numerical, and linguistic inputs arranged in a conventionalized format one has previously learned to parse. Recently developed vision-language models are, in principle, promising candidates for developing computational models of these cognitive operations. However, it is currently unclear to what degree these models emulate human behavior on tasks that involve reasoning about data visualizations. This gap reflects limitations in prior work that has evaluated data visualization understanding in artificial systems using measures that differ from those typically used to assess these abilities in humans. Here we evaluated eight vision-language models on six data visualization literacy assessments designed for humans and compared model responses to those of human participants. We found that these models performed worse than human participants on average, and this performance gap persisted even when using relatively lenient criteria to assess model performance. Moreover, while relative performance across items was somewhat correlated between models and humans, all models produced patterns of errors that were reliably distinct from those produced by human participants. Taken together, these findings suggest significant opportunities for further development of artificial systems that might serve as useful models of how humans reason about data visualizations. All code and data needed to reproduce these results are available at: https://osf.io/e25mu/?view_only=399daff5a14d4b16b09473cf19043f18.
>
---
#### [new 230] Chain-of-Lure: A Synthetic Narrative-Driven Approach to Compromise Large Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究大模型安全漏洞，提出Chain-of-Lure方法，通过攻击模型生成叙事诱饵诱导受害模型推理，利用辅助模型优化诱饵绕过安全机制，并用毒性评分替代拒绝关键词评估攻击效果，揭示模型可被利用攻击他人，提出防御策略优化安全机制。**

- **链接: [http://arxiv.org/pdf/2505.17519v1](http://arxiv.org/pdf/2505.17519v1)**

> **作者:** Wenhan Chang; Tianqing Zhu; Yu Zhao; Shuangyong Song; Ping Xiong; Wanlei Zhou; Yongxiang Li
>
> **备注:** 25 pages, 4 figures
>
> **摘要:** In the era of rapid generative AI development, interactions between humans and large language models face significant misusing risks. Previous research has primarily focused on black-box scenarios using human-guided prompts and white-box scenarios leveraging gradient-based LLM generation methods, neglecting the possibility that LLMs can act not only as victim models, but also as attacker models to harm other models. We proposes a novel jailbreaking method inspired by the Chain-of-Thought mechanism, where the attacker model uses mission transfer to conceal harmful user intent in dialogue and generates chained narrative lures to stimulate the reasoning capabilities of victim models, leading to successful jailbreaking. To enhance the attack success rate, we introduce a helper model that performs random narrative optimization on the narrative lures during multi-turn dialogues while ensuring alignment with the original intent, enabling the optimized lures to bypass the safety barriers of victim models effectively. Our experiments reveal that models with weaker safety mechanisms exhibit stronger attack capabilities, demonstrating that models can not only be exploited, but also help harm others. By incorporating toxicity scores, we employ third-party models to evaluate the harmfulness of victim models' responses to jailbreaking attempts. The study shows that using refusal keywords as an evaluation metric for attack success rates is significantly flawed because it does not assess whether the responses guide harmful questions, while toxicity scores measure the harm of generated content with more precision and its alignment with harmful questions. Our approach demonstrates outstanding performance, uncovering latent vulnerabilities in LLMs and providing data-driven feedback to optimize LLM safety mechanisms. We also discuss two defensive strategies to offer guidance on improving defense mechanisms.
>
---
#### [new 231] Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文评估大型语言模型（LLM）作为AI科学家逆向工程黑箱系统的能力，研究其通过被动观察与主动干预学习的差异。实验在程序、形式语言和数学方程领域展开，发现主动查询显著提升性能，帮助LLM克服假设先验知识和遗漏观察的缺陷，为优化其科学发现能力提供指导。**

- **链接: [http://arxiv.org/pdf/2505.17968v1](http://arxiv.org/pdf/2505.17968v1)**

> **作者:** Jiayi Geng; Howard Chen; Dilip Arumugam; Thomas L. Griffiths
>
> **备注:** 30 pages
>
> **摘要:** Using AI to create autonomous researchers has the potential to accelerate scientific discovery. A prerequisite for this vision is understanding how well an AI model can identify the underlying structure of a black-box system from its behavior. In this paper, we explore how well a large language model (LLM) learns to identify a black-box function from passively observed versus actively collected data. We investigate the reverse-engineering capabilities of LLMs across three distinct types of black-box systems, each chosen to represent different problem domains where future autonomous AI researchers may have considerable impact: Program, Formal Language, and Math Equation. Through extensive experiments, we show that LLMs fail to extract information from observations, reaching a performance plateau that falls short of the ideal of Bayesian inference. However, we demonstrate that prompting LLMs to not only observe but also intervene -- actively querying the black-box with specific inputs to observe the resulting output -- improves performance by allowing LLMs to test edge cases and refine their beliefs. By providing the intervention data from one LLM to another, we show that this improvement is partly a result of engaging in the process of generating effective interventions, paralleling results in the literature on human learning. Further analysis reveals that engaging in intervention can help LLMs escape from two common failure modes: overcomplication, where the LLM falsely assumes prior knowledge about the black-box, and overlooking, where the LLM fails to incorporate observations. These insights provide practical guidance for helping LLMs more effectively reverse-engineer black-box systems, supporting their use in making new discoveries.
>
---
#### [new 232] Generalizing Large Language Model Usability Across Resource-Constrained
- **分类: cs.LG; cs.CL**

- **简介: 该论文聚焦提升大语言模型在资源受限场景下的泛化能力，解决其依赖昂贵微调、泛化性差的问题。工作包括：提出文本中心多模态对齐框架，支持动态模态适配；设计对抗性提示增强鲁棒性；开发推理优化策略（提示搜索、不确定性量化）；构建低资源领域（如Verilog代码生成）的合成数据与逻辑模型。**

- **链接: [http://arxiv.org/pdf/2505.17040v1](http://arxiv.org/pdf/2505.17040v1)**

> **作者:** Yun-Da Tsai
>
> **备注:** Doctoral disstertation
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, and recent efforts have sought to extend their capabilities to multimodal domains and resource-constrained environments. However, existing approaches often rely on costly supervised fine-tuning or assume fixed training conditions, limiting their generalization when facing unseen modalities, limited data, or restricted compute resources. This dissertation presents a systematic study toward generalizing LLM usability under real-world constraints. First, it introduces a robust text-centric alignment framework that enables LLMs to seamlessly integrate diverse modalities-including text, images, tables, and any modalities - via natural language interfaces. This approach supports in-context adaptation to unseen or dynamically changing modalities without requiring retraining. To enhance robustness against noisy and missing modalities, an adversarial prompting technique is proposed, generating semantically challenging perturbations at the prompt level to stress-test model reliability. Beyond multimodal setting, the dissertation investigates inference-time optimization strategies for LLMs, leveraging prompt search and uncertainty quantification to improve performance without additional model training. This perspective offers an efficient alternative to scaling model parameters or retraining from scratch. Additionally, the work addresses low-resource domains such as Verilog code generation by designing correct-by-construction synthetic data pipelines and logic-enhanced reasoning models, achieving state-of-the-art performance with minimal data. Together, these contributions form a unified effort to enhance the adaptability, scalability, and efficiency of large language models under practical constraints.
>
---
#### [new 233] T2I-Eval-R1: Reinforcement Learning-Driven Reasoning for Interpretable Text-to-Image Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出T2I-Eval-R1，一种基于强化学习的文本到图像生成评估框架。针对现有监督微调方法依赖昂贵/有偏的标注数据的问题，其通过粗粒度评分训练开源多模态模型，结合组相对策略优化与连续奖励设计，使模型输出分数及可解释推理链，提升评估与人类一致性的准确性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2505.17897v1](http://arxiv.org/pdf/2505.17897v1)**

> **作者:** Zi-Ao Ma; Tian Lan; Rong-Cheng Tu; Shu-Hang Liu; Heyan Huang; Zhijing Wu; Chen Xu; Xian-Ling Mao
>
> **摘要:** The rapid progress in diffusion-based text-to-image (T2I) generation has created an urgent need for interpretable automatic evaluation methods that can assess the quality of generated images, therefore reducing the human annotation burden. To reduce the prohibitive cost of relying on commercial models for large-scale evaluation, and to improve the reasoning capabilities of open-source models, recent research has explored supervised fine-tuning (SFT) of multimodal large language models (MLLMs) as dedicated T2I evaluators. However, SFT approaches typically rely on high-quality critique datasets, which are either generated by proprietary LLMs-with potential issues of bias and inconsistency-or annotated by humans at high cost, limiting their scalability and generalization. To address these limitations, we propose T2I-Eval-R1, a novel reinforcement learning framework that trains open-source MLLMs using only coarse-grained quality scores, thereby avoiding the need for annotating high-quality interpretable evaluation rationale. Our approach integrates Group Relative Policy Optimization (GRPO) into the instruction-tuning process, enabling models to generate both scalar scores and interpretable reasoning chains with only easy accessible annotated judgment scores or preferences. Furthermore, we introduce a continuous reward formulation that encourages score diversity and provides stable optimization signals, leading to more robust and discriminative evaluation behavior. Experimental results on three established T2I meta-evaluation benchmarks demonstrate that T2I-Eval-R1 achieves significantly higher alignment with human assessments and offers more accurate interpretable score rationales compared to strong baseline methods.
>
---
#### [new 234] From Weak Labels to Strong Results: Utilizing 5,000 Hours of Noisy Classroom Transcripts with Minimal Accurate Data
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于课堂环境下的自动语音识别（ASR）任务，解决如何利用大量弱标注（噪声）文本与少量高质量数据提升低资源场景性能的问题。提出Weakly Supervised Pretraining（WSP）方法：先用弱数据预训练模型，再用精数据微调，实验显示其优于其他方法，有效应对低成本标注与稀缺优质数据的矛盾。**

- **链接: [http://arxiv.org/pdf/2505.17088v1](http://arxiv.org/pdf/2505.17088v1)**

> **作者:** Ahmed Adel Attia; Dorottya Demszky; Jing Liu; Carol Espy-Wilson
>
> **摘要:** Recent progress in speech recognition has relied on models trained on vast amounts of labeled data. However, classroom Automatic Speech Recognition (ASR) faces the real-world challenge of abundant weak transcripts paired with only a small amount of accurate, gold-standard data. In such low-resource settings, high transcription costs make re-transcription impractical. To address this, we ask: what is the best approach when abundant inexpensive weak transcripts coexist with limited gold-standard data, as is the case for classroom speech data? We propose Weakly Supervised Pretraining (WSP), a two-step process where models are first pretrained on weak transcripts in a supervised manner, and then fine-tuned on accurate data. Our results, based on both synthetic and real weak transcripts, show that WSP outperforms alternative methods, establishing it as an effective training methodology for low-resource ASR in real-world scenarios.
>
---
#### [new 235] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Deep Video Discovery(DVD)代理，针对长视频理解中LLMs处理信息密集视频的局限，采用自主代理搜索策略，通过多粒度视频数据库工具自主规划、选择参数并迭代优化推理，显著提升LVBench等数据集表现。**

- **链接: [http://arxiv.org/pdf/2505.18079v1](http://arxiv.org/pdf/2505.18079v1)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later.
>
---
#### [new 236] CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，针对MoE模型在异构数据中专家功能冗余、容量利用率低的问题，提出CoMoE方法。通过对比激活与未激活专家的表征差异，提升专家专业化和模块化，实验表明其有效增强MoE性能。**

- **链接: [http://arxiv.org/pdf/2505.17553v1](http://arxiv.org/pdf/2505.17553v1)**

> **作者:** Jinyuan Feng; Chaopeng Wei; Tenghai Qiu; Tianyi Hu; Zhiqiang Pu
>
> **摘要:** In parameter-efficient fine-tuning, mixture-of-experts (MoE), which involves specializing functionalities into different experts and sparsely activating them appropriately, has been widely adopted as a promising approach to trade-off between model capacity and computation overhead. However, current MoE variants fall short on heterogeneous datasets, ignoring the fact that experts may learn similar knowledge, resulting in the underutilization of MoE's capacity. In this paper, we propose Contrastive Representation for MoE (CoMoE), a novel method to promote modularization and specialization in MoE, where the experts are trained along with a contrastive objective by sampling from activated and inactivated experts in top-k routing. We demonstrate that such a contrastive objective recovers the mutual-information gap between inputs and the two types of experts. Experiments on several benchmarks and in multi-task settings demonstrate that CoMoE can consistently enhance MoE's capacity and promote modularization among the experts.
>
---
#### [new 237] ProgRM: Build Better GUI Agents with Progress Rewards
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于GUI代理训练优化任务，旨在解决现有Outcome Reward Model（ORM）反馈粗放、过度惩罚失败轨迹有效步骤的问题。提出Progress Reward Model（ProgRM），通过预测任务进度提供密集中间奖励，并设计LCS基自标注算法标注关键步骤，实验表明其优于ORM和商业LLM。**

- **链接: [http://arxiv.org/pdf/2505.18121v1](http://arxiv.org/pdf/2505.18121v1)**

> **作者:** Danyang Zhang; Situo Zhang; Ziyue Yang; Zichen Zhu; Zihan Zhao; Ruisheng Cao; Lu Chen; Kai Yu
>
> **摘要:** LLM-based (Large Language Model) GUI (Graphical User Interface) agents can potentially reshape our daily lives significantly. However, current LLM-based GUI agents suffer from the scarcity of high-quality training data owing to the difficulties of trajectory collection and reward annotation. Existing works have been exploring LLMs to collect trajectories for imitation learning or to offer reward signals for online RL training. However, the Outcome Reward Model (ORM) used in existing works cannot provide finegrained feedback and can over-penalize the valuable steps in finally failed trajectories. To this end, we propose Progress Reward Model (ProgRM) to provide dense informative intermediate rewards by predicting a task completion progress for each step in online training. To handle the challenge of progress reward label annotation, we further design an efficient LCS-based (Longest Common Subsequence) self-annotation algorithm to discover the key steps in trajectories and assign progress labels accordingly. ProgRM is evaluated with extensive experiments and analyses. Actors trained with ProgRM outperform leading proprietary LLMs and ORM-trained actors, illustrating the effectiveness of ProgRM. The codes for experiments will be made publicly available upon acceptance.
>
---
#### [new 238] Diagnosing Vision Language Models' Perception by Leveraging Human Methods for Color Vision Deficiencies
- **分类: cs.CV; cs.CL**

- **简介: 该论文评估视觉语言模型（LVLMs）处理色觉差异的能力，解决其能否模拟色觉缺陷者（CVD）感知的问题。通过Ishihara色盲测试发现，LVLMs可解释CVD但无法模拟其图像感知，强调需开发更具色彩感知包容性的多模态系统。**

- **链接: [http://arxiv.org/pdf/2505.17461v1](http://arxiv.org/pdf/2505.17461v1)**

> **作者:** Kazuki Hayashi; Shintaro Ozaki; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **摘要:** Large-scale Vision Language Models (LVLMs) are increasingly being applied to a wide range of real-world multimodal applications, involving complex visual and linguistic reasoning. As these models become more integrated into practical use, they are expected to handle complex aspects of human interaction. Among these, color perception is a fundamental yet highly variable aspect of visual understanding. It differs across individuals due to biological factors such as Color Vision Deficiencies (CVDs), as well as differences in culture and language. Despite its importance, perceptual diversity has received limited attention. In our study, we evaluate LVLMs' ability to account for individual level perceptual variation using the Ishihara Test, a widely used method for detecting CVDs. Our results show that LVLMs can explain CVDs in natural language, but they cannot simulate how people with CVDs perceive color in image based tasks. These findings highlight the need for multimodal systems that can account for color perceptual diversity and support broader discussions on perceptual inclusiveness and fairness in multimodal AI.
>
---
#### [new 239] Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models
- **分类: cs.LG; cs.CL; cs.DC**

- **简介: 该论文提出Trinity-RFT框架，解决大语言模型强化微调（RFT）中模式多样、效率不足的问题。通过解耦设计统一同步/异步、on-policy/off-policy及online/offline模式，整合高效交互与优化数据管道，提供通用灵活的RFT平台，支持多场景应用与算法探索。**

- **链接: [http://arxiv.org/pdf/2505.17826v1](http://arxiv.org/pdf/2505.17826v1)**

> **作者:** Xuchen Pan; Yanxi Chen; Yushuo Chen; Yuchang Sun; Daoyuan Chen; Wenhao Zhang; Yuexiang Xie; Yilun Huang; Yilei Zhang; Dawei Gao; Yaliang Li; Bolin Ding; Jingren Zhou
>
> **备注:** This technical report will be continuously updated as the codebase evolves. GitHub: https://github.com/modelscope/Trinity-RFT
>
> **摘要:** Trinity-RFT is a general-purpose, flexible and scalable framework designed for reinforcement fine-tuning (RFT) of large language models. It is built with a decoupled design, consisting of (1) an RFT-core that unifies and generalizes synchronous/asynchronous, on-policy/off-policy, and online/offline modes of RFT, (2) seamless integration for agent-environment interaction with high efficiency and robustness, and (3) systematic data pipelines optimized for RFT. Trinity-RFT can be easily adapted for diverse application scenarios, and serves as a unified platform for exploring advanced reinforcement learning paradigms. This technical report outlines the vision, features, design and implementations of Trinity-RFT, accompanied by extensive examples demonstrating the utility and user-friendliness of the proposed framework.
>
---
#### [new 240] PD$^3$: A Project Duplication Detection Framework via Adapted Multi-Agent Debate
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PD³框架，针对项目重复检测任务，解决传统方法依赖浅层文本比对或单一模型导致的专家洞察不足及理解深度不够问题。通过多智能体辩论机制模拟专家论证，结合定量定性分析提升检测效果，实验显示优于现有方法7.43%-8.00%，并部署平台实现千万级成本节省。**

- **链接: [http://arxiv.org/pdf/2505.17492v1](http://arxiv.org/pdf/2505.17492v1)**

> **作者:** Dezheng Bao; Yueci Yang; Xin Chen; Zhengxuan Jiang; Zeguo Fei; Daoze Zhang; Xuanwen Huang; Junru Chen; Chutian Yu; Xiang Yuan; Yang Yang
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** Project duplication detection is critical for project quality assessment, as it improves resource utilization efficiency by preventing investing in newly proposed project that have already been studied. It requires the ability to understand high-level semantics and generate constructive and valuable feedback. Existing detection methods rely on basic word- or sentence-level comparison or solely apply large language models, lacking valuable insights for experts and in-depth comprehension of project content and review criteria. To tackle this issue, we propose PD$^3$, a Project Duplication Detection framework via adapted multi-agent Debate. Inspired by real-world expert debates, it employs a fair competition format to guide multi-agent debate to retrieve relevant projects. For feedback, it incorporates both qualitative and quantitative analysis to improve its practicality. Over 800 real-world power project data spanning more than 20 specialized fields are used to evaluate the framework, demonstrating that our method outperforms existing approaches by 7.43% and 8.00% in two downstream tasks. Furthermore, we establish an online platform, Review Dingdang, to assist power experts, saving 5.73 million USD in initial detection on more than 100 newly proposed projects.
>
---
#### [new 241] Safety Alignment Can Be Not Superficial With Explicit Safety Signals
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大型语言模型（LLM）安全对齐任务，针对现有方法表面化、易受攻击的问题，提出显式引入安全二分类任务并整合至注意力与解码策略，使模型在生成每一步评估安全，以低开销提升对抗攻击鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.17072v1](http://arxiv.org/pdf/2505.17072v1)**

> **作者:** Jianwei Li; Jung-Eng Kim
>
> **备注:** ICML 2025
>
> **摘要:** Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems.
>
---
#### [new 242] PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PatientSim，解决现有医患模拟器无法反映临床患者多样性的不足。通过整合真实医疗数据和四维度（性格、语言能力、病史回忆、认知混淆）生成37种患者人设，评估LLM并经临床验证，提供开源平台用于医疗对话系统训练与评估。**

- **链接: [http://arxiv.org/pdf/2505.17818v1](http://arxiv.org/pdf/2505.17818v1)**

> **作者:** Daeun Kyung; Hyunseung Chung; Seongsu Bae; Jiho Kim; Jae Ho Sohn; Taerim Kim; Soo Kyung Kim; Edward Choi
>
> **备注:** 9 pages for main text, 4 pages for references, 27 pages for supplementary materials
>
> **摘要:** Doctor-patient consultations require multi-turn, context-aware communication tailored to diverse patient personas. Training or evaluating doctor LLMs in such settings requires realistic patient interaction systems. However, existing simulators often fail to reflect the full range of personas seen in clinical practice. To address this, we introduce PatientSim, a patient simulator that generates realistic and diverse patient personas for clinical scenarios, grounded in medical expertise. PatientSim operates using: 1) clinical profiles, including symptoms and medical history, derived from real-world data in the MIMIC-ED and MIMIC-IV datasets, and 2) personas defined by four axes: personality, language proficiency, medical history recall level, and cognitive confusion level, resulting in 37 unique combinations. We evaluated eight LLMs for factual accuracy and persona consistency. The top-performing open-source model, Llama 3.3, was validated by four clinicians to confirm the robustness of our framework. As an open-source, customizable platform, PatientSim provides a reproducible and scalable solution that can be customized for specific training needs. Offering a privacy-compliant environment, it serves as a robust testbed for evaluating medical dialogue systems across diverse patient presentations and shows promise as an educational tool for healthcare.
>
---
#### [new 243] TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations
- **分类: cs.LG; cs.CL**

- **简介: 论文提出TabSTAR模型，针对含文本特征的表格学习任务，解决现有方法静态文本表示目标无关导致效果差的问题。通过动态解冻预训练文本编码器并输入目标标记，学习任务特定嵌入，实现跨数据集迁移，在分类任务中达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.18125v1](http://arxiv.org/pdf/2505.18125v1)**

> **作者:** Alan Arazi; Eilam Shapira; Roi Reichart
>
> **摘要:** While deep learning has achieved remarkable success across many domains, it has historically underperformed on tabular learning tasks, which remain dominated by gradient boosting decision trees (GBDTs). However, recent advancements are paving the way for Tabular Foundation Models, which can leverage real-world knowledge and generalize across diverse datasets, particularly when the data contains free-text. Although incorporating language model capabilities into tabular tasks has been explored, most existing methods utilize static, target-agnostic textual representations, limiting their effectiveness. We introduce TabSTAR: a Foundation Tabular Model with Semantically Target-Aware Representations. TabSTAR is designed to enable transfer learning on tabular data with textual features, with an architecture free of dataset-specific parameters. It unfreezes a pretrained text encoder and takes as input target tokens, which provide the model with the context needed to learn task-specific embeddings. TabSTAR achieves state-of-the-art performance for both medium- and large-sized datasets across known benchmarks of classification tasks with text features, and its pretraining phase exhibits scaling laws in the number of datasets, offering a pathway for further performance improvements.
>
---
#### [new 244] Reward Model Overoptimisation in Iterated RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习对齐任务，研究迭代RLHF中奖励模型过度优化导致策略不泛化的问题。通过系统分析奖励模型训练数据传递、奖励函数选择及策略初始化等设计，发现过度优化随迭代减少但收益递减，提出稳健初始化策略以提升RLHF流水线的稳定性与泛化性。**

- **链接: [http://arxiv.org/pdf/2505.18126v1](http://arxiv.org/pdf/2505.18126v1)**

> **作者:** Lorenz Wolf; Robert Kirk; Mirco Musolesi
>
> **备注:** 20 pages, 17 figures, 5 tables
>
> **摘要:** Reinforcement learning from human feedback (RLHF) is a widely used method for aligning large language models with human preferences. However, RLHF often suffers from reward model overoptimisation, in which models overfit to the reward function, resulting in non-generalisable policies that exploit the idiosyncrasies and peculiarities of the reward function. A common mitigation is iterated RLHF, in which reward models are repeatedly retrained with updated human feedback and policies are re-optimised. Despite its increasing adoption, the dynamics of overoptimisation in this setting remain poorly understood. In this work, we present the first comprehensive study of overoptimisation in iterated RLHF. We systematically analyse key design choices - how reward model training data is transferred across iterations, which reward function is used for optimisation, and how policies are initialised. Using the controlled AlpacaFarm benchmark, we observe that overoptimisation tends to decrease over successive iterations, as reward models increasingly approximate ground-truth preferences. However, performance gains diminish over time, and while reinitialising from the base policy is robust, it limits optimisation flexibility. Other initialisation strategies often fail to recover from early overoptimisation. These findings offer actionable insights for building more stable and generalisable RLHF pipelines.
>
---
## 更新

#### [replaced 001] Phare: A Safety Probe for Large Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.11365v3](http://arxiv.org/pdf/2505.11365v3)**

> **作者:** Pierre Le Jeune; Benoît Malézieux; Weixuan Xiao; Matteo Dora
>
> **摘要:** Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems.
>
---
#### [replaced 002] Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.17262v2](http://arxiv.org/pdf/2502.17262v2)**

> **作者:** Chengyin Xu; Kaiyuan Chen; Xiao Li; Ke Shen; Chenggang Li
>
> **备注:** 19 pages,6 figures
>
> **摘要:** The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for efficient resource allocation. This is challenged by: 1) the emergence phenomenon, where metrics become meaningful only after extensive training, hindering prediction by smaller models; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby establishing a more stable and predictable support subset through the exclusion of tasks exhibiting non-emergent behavior or irregular scaling. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.36% average prediction error across eight key LLM benchmarks, offering actionable insights for resource allocation and training monitoring of LLMs pretraining.
>
---
#### [replaced 003] Enhancing Low-Resource Language and Instruction Following Capabilities of Audio Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.10999v2](http://arxiv.org/pdf/2409.10999v2)**

> **作者:** Potsawee Manakul; Guangzhi Sun; Warit Sirichotedumrong; Kasima Tharnpipitchai; Kunat Pipatanakul
>
> **备注:** Interspeech 2025
>
> **摘要:** Audio language models process audio inputs using textual prompts for tasks like speech recognition and audio captioning. Although built on multilingual pre-trained components, most are trained primarily on English, limiting their usability for other languages. This paper evaluates audio language models on Thai, a low-resource language, and finds that they lack emergent cross-lingual abilities despite their multilingual foundations. To address this, we explore data mixtures that optimize audio language models for both a target language and English while integrating audio comprehension and speech instruction-following into a unified model. Our experiments provide insights into improving instruction-following in low-resource languages by balancing language-specific and multilingual training data. The proposed model, Typhoon-Audio, significantly outperforms existing open-source models and achieves performance comparable to state-of-the-art Gemini-1.5-Pro in both English and Thai.
>
---
#### [replaced 004] MedPlan:A Two-Stage RAG-Based System for Personalized Medical Plan Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17900v2](http://arxiv.org/pdf/2503.17900v2)**

> **作者:** Hsin-Ling Hsu; Cong-Tinh Dao; Luning Wang; Zitao Shuai; Thao Nguyen Minh Phan; Jun-En Ding; Chun-Chieh Liao; Pengfei Hu; Xiaoxue Han; Chih-Ho Hsu; Dongsheng Luo; Wen-Chih Peng; Feng Liu; Fang-Ming Hung; Chenwei Wu
>
> **摘要:** Despite recent success in applying large language models (LLMs) to electronic health records (EHR), most systems focus primarily on assessment rather than treatment planning. We identify three critical limitations in current approaches: they generate treatment plans in a single pass rather than following the sequential reasoning process used by clinicians; they rarely incorporate patient-specific historical context; and they fail to effectively distinguish between subjective and objective clinical information. Motivated by the SOAP methodology (Subjective, Objective, Assessment, Plan), we introduce \ours{}, a novel framework that structures LLM reasoning to align with real-life clinician workflows. Our approach employs a two-stage architecture that first generates a clinical assessment based on patient symptoms and objective data, then formulates a structured treatment plan informed by this assessment and enriched with patient-specific information through retrieval-augmented generation. Comprehensive evaluation demonstrates that our method significantly outperforms baseline approaches in both assessment accuracy and treatment plan quality.
>
---
#### [replaced 005] Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14645v2](http://arxiv.org/pdf/2502.14645v2)**

> **作者:** Yuchen Wu; Liang Ding; Li Shen; Dacheng Tao
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings.
>
---
#### [replaced 006] The Quantum LLM: Modeling Semantic Spaces with Quantum Principles
- **分类: cs.AI; cs.CL; quant-ph**

- **链接: [http://arxiv.org/pdf/2504.13202v2](http://arxiv.org/pdf/2504.13202v2)**

> **作者:** Timo Aukusti Laine
>
> **备注:** 16 pages, 6 figures. Some corrections
>
> **摘要:** In the previous article, we presented a quantum-inspired framework for modeling semantic representation and processing in Large Language Models (LLMs), drawing upon mathematical tools and conceptual analogies from quantum mechanics to offer a new perspective on these complex systems. In this paper, we clarify the core assumptions of this model, providing a detailed exposition of six key principles that govern semantic representation, interaction, and dynamics within LLMs. The goal is to justify that a quantum-inspired framework is a valid approach to studying semantic spaces. This framework offers valuable insights into their information processing and response generation, and we further discuss the potential of leveraging quantum computing to develop significantly more powerful and efficient LLMs based on these principles.
>
---
#### [replaced 007] Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14418v2](http://arxiv.org/pdf/2505.14418v2)**

> **作者:** Pengzhou Cheng; Haowen Hu; Zheng Wu; Zongru Wu; Tianjie Ju; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** 25 pages, 10 figures, 12 Tables
>
> **摘要:** Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}.
>
---
#### [replaced 008] Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11228v2](http://arxiv.org/pdf/2502.11228v2)**

> **作者:** Mohammad Reza Rezaei; Adji Bousso Dieng
>
> **备注:** A RAG pipeline that accounts for both diversity and answer quality and that can be used with any LLM backbone to solve complex multi-hop question-answering tasks
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) for domain-specific question-answering (QA) tasks by leveraging external knowledge sources. However, traditional RAG systems primarily focus on relevance-based retrieval and often struggle with redundancy, especially when reasoning requires connecting information from multiple sources. This paper introduces Vendi-RAG, a framework based on an iterative process that jointly optimizes retrieval diversity and answer quality. This joint optimization leads to significantly higher accuracy for multi-hop QA tasks. Vendi-RAG leverages the Vendi Score (VS), a flexible similarity-based diversity metric, to promote semantic diversity in document retrieval. It then uses an LLM judge that evaluates candidate answers, generated after a reasoning step, and outputs a score that the retriever uses to balance relevance and diversity among the retrieved documents during each iteration. Experiments on three challenging datasets -- HotpotQA, MuSiQue, and 2WikiMultiHopQA -- demonstrate Vendi-RAG's effectiveness in multi-hop reasoning tasks. The framework achieves significant accuracy improvements over traditional single-step and multi-step RAG approaches, with accuracy increases reaching up to +4.2% on HotpotQA, +4.1% on 2WikiMultiHopQA, and +1.3% on MuSiQue compared to Adaptive-RAG, the current best baseline. The benefits of Vendi-RAG are even more pronounced as the number of retrieved documents increases. Finally, we evaluated Vendi-RAG across different LLM backbones, including GPT-3.5, GPT-4, and GPT-4o-mini, and observed consistent improvements, demonstrating that the framework's advantages are model-agnostic.
>
---
#### [replaced 009] Enhancing Robustness in Large Language Models: Prompting for Mitigating the Impact of Irrelevant Information
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10615v2](http://arxiv.org/pdf/2408.10615v2)**

> **作者:** Ming Jiang; Tingting Huang; Biao Guo; Yao Lu; Feng Zhang
>
> **摘要:** In recent years, Large language models (LLMs) have garnered significant attention due to their superior performance in complex reasoning tasks. However, recent studies may diminish their reasoning capabilities markedly when problem descriptions contain irrelevant information, even with the use of advanced prompting techniques. To further investigate this issue, a dataset of primary school mathematics problems containing irrelevant information, named GSMIR, was constructed. Testing prominent LLMs and prompting techniques on this dataset revealed that while LLMs can identify irrelevant information, they do not effectively mitigate the interference it causes once identified. A novel automatic construction method, ATF, which enhances the ability of LLMs to identify and self-mitigate the influence of irrelevant information, is proposed to address this shortcoming. This method operates in two steps: first, analysis of irrelevant information, followed by its filtering. The ATF method, as demonstrated by experimental results, significantly improves the reasoning performance of LLMs and prompting techniques, even in the presence of irrelevant information on the GSMIR dataset.
>
---
#### [replaced 010] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models
- **分类: cs.CL; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2407.21077v3](http://arxiv.org/pdf/2407.21077v3)**

> **作者:** Somshubra Majumdar; Vahid Noroozi; Mehrzad Samadi; Sean Narenthiran; Aleksander Ficek; Wasi Uddin Ahmad; Jocelyn Huang; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to be presented in ACL 2025
>
> **摘要:** Large Language Models (LLMs) require high quality instruction data for effective alignment, particularly in code generation tasks where expert curated datasets are expensive to produce. We present Genetic-Instruct, a scalable algorithm for synthesizing large-scale, high quality coding instructions using evolutionary principles. Starting from a small set of seed instructions, Genetic-Instruct generates diverse and challenging instruction-code pairs by leveraging an Instructor-LLM for generation, a Coder-LLM for code synthesis, and a Judge-LLM for automatic quality evaluation. Our proposed approach is highly parallelizable and effective even with a small seed data and weaker generator models. We generated more than 7.5 million coding instructions with the proposed approach. Then we evaluated it by fine-tuning LLMs with the synthetic samples and demonstrated a significant improvement in their code generation capability compared to the other synthetic generation approaches and publicly available datasets. Our results highlight the efficiency, scalability, and generalizability of the Genetic-Instruct framework.
>
---
#### [replaced 011] Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.16023v2](http://arxiv.org/pdf/2505.16023v2)**

> **作者:** Sheshera Mysore; Debarati Das; Hancheng Cao; Bahareh Sarrafzadeh
>
> **备注:** Pre-print under-review
>
> **摘要:** As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large-scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human-AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users' intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.
>
---
#### [replaced 012] WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18511v2](http://arxiv.org/pdf/2501.18511v2)**

> **作者:** Benjamin Feuer; Chinmay Hegde
>
> **备注:** ICML 2025
>
> **摘要:** Language model (LLM) post-training, from DPO to distillation, can refine behaviors and unlock new skills, but the open science supporting these post-training techniques is still in its infancy. One limiting factor has been the difficulty of conducting large-scale comparative analyses of synthetic data generating models and LLM judges. To close this gap, we introduce WILDCHAT-50M, the largest public chat dataset to date. We extend the existing WildChat dataset to include responses not only from GPT, but from over 50 different open-weight models, ranging in size from 0.5B to 104B parameters. We conduct an extensive comparative analysis and demonstrate the potential of this dataset by creating RE-WILD, our own public SFT mix, which outperforms the recent Tulu-3 SFT mixture from Allen AI with only 40% as many samples. Our dataset, samples and code are available at https://github.com/penfever/wildchat-50m.
>
---
#### [replaced 013] Explaining Black-box Model Predictions via Two-level Nested Feature Attributions with Consistency Property
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2405.14522v2](http://arxiv.org/pdf/2405.14522v2)**

> **作者:** Yuya Yoshikawa; Masanari Kimura; Ryotaro Shimizu; Yuki Saito
>
> **备注:** This manuscript is an extended version of our paper accepted at IJCAI2025, with detailed proofs and additional experimental results
>
> **摘要:** Techniques that explain the predictions of black-box machine learning models are crucial to make the models transparent, thereby increasing trust in AI systems. The input features to the models often have a nested structure that consists of high- and low-level features, and each high-level feature is decomposed into multiple low-level features. For such inputs, both high-level feature attributions (HiFAs) and low-level feature attributions (LoFAs) are important for better understanding the model's decision. In this paper, we propose a model-agnostic local explanation method that effectively exploits the nested structure of the input to estimate the two-level feature attributions simultaneously. A key idea of the proposed method is to introduce the consistency property that should exist between the HiFAs and LoFAs, thereby bridging the separate optimization problems for estimating them. Thanks to this consistency property, the proposed method can produce HiFAs and LoFAs that are both faithful to the black-box models and consistent with each other, using a smaller number of queries to the models. In experiments on image classification in multiple instance learning and text classification using language models, we demonstrate that the HiFAs and LoFAs estimated by the proposed method are accurate, faithful to the behaviors of the black-box models, and provide consistent explanations.
>
---
#### [replaced 014] PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17041v2](http://arxiv.org/pdf/2502.17041v2)**

> **作者:** Haoran Li; Wenbin Hu; Huihao Jing; Yulin Chen; Qi Hu; Sirui Han; Tianshu Chu; Peizhao Hu; Yangqiu Song
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Recent advancements in generative large language models (LLMs) have enabled wider applicability, accessibility, and flexibility. However, their reliability and trustworthiness are still in doubt, especially for concerns regarding individuals' data privacy. Great efforts have been made on privacy by building various evaluation benchmarks to study LLMs' privacy awareness and robustness from their generated outputs to their hidden representations. Unfortunately, most of these works adopt a narrow formulation of privacy and only investigate personally identifiable information (PII). In this paper, we follow the merit of the Contextual Integrity (CI) theory, which posits that privacy evaluation should not only cover the transmitted attributes but also encompass the whole relevant social context through private information flows. We present PrivaCI-Bench, a comprehensive contextual privacy evaluation benchmark targeted at legal compliance to cover well-annotated privacy and safety regulations, real court cases, privacy policies, and synthetic data built from the official toolkit to study LLMs' privacy and safety compliance. We evaluate the latest LLMs, including the recent reasoner models QwQ-32B and Deepseek R1. Our experimental results suggest that though LLMs can effectively capture key CI parameters inside a given context, they still require further advancements for privacy compliance.
>
---
#### [replaced 015] Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11614v2](http://arxiv.org/pdf/2502.11614v2)**

> **作者:** Yuxia Wang; Rui Xing; Jonibek Mansurov; Giovanni Puccetti; Zhuohan Xie; Minh Ngoc Ta; Jiahui Geng; Jinyan Su; Mervat Abassy; Saad El Dine Ahmed; Kareem Elozeiri; Nurkhan Laiyk; Maiya Goloburda; Tarek Mahmoud; Raj Vardhan Tomar; Alexander Aziz; Ryuto Koike; Masahiro Kaneko; Artem Shelmanov; Ekaterina Artemova; Vladislav Mikhailov; Akim Tsvigun; Alham Fikri Aji; Nizar Habash; Iryna Gurevych; Preslav Nakov
>
> **摘要:** Prior studies have shown that distinguishing text generated by large language models (LLMs) from human-written one is highly challenging, and often no better than random guessing. To verify the generalizability of this finding across languages and domains, we perform an extensive case study to identify the upper bound of human detection accuracy. Across 16 datasets covering 9 languages and 9 domains, 19 annotators achieved an average detection accuracy of 87.6\%, thus challenging previous conclusions. We find that major gaps between human and machine text lie in concreteness, cultural nuances, and diversity. Prompting by explicitly explaining the distinctions in the prompts can partially bridge the gaps in over 50\% of the cases. However, we also find that humans do not always prefer human-written text, particularly when they cannot clearly identify its source.
>
---
#### [replaced 016] Power-Law Decay Loss for Large Language Model Finetuning: Focusing on Information Sparsity to Enhance Generation Quality
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16900v2](http://arxiv.org/pdf/2505.16900v2)**

> **作者:** Jintian Shao; Yiming Cheng; Hongyi Huang; Jiayi Wu; Beiwen Zhang; Zhiyu Wu; You Shan; Mingkai Zheng
>
> **摘要:** During the finetuning stage of text generation tasks, standard cross-entropy loss treats all tokens equally. This can lead models to overemphasize high-frequency, low-information tokens, neglecting lower-frequency tokens crucial for specificity and informativeness in generated content. This paper introduces a novel loss function, Power-Law Decay Loss (PDL), specifically designed to optimize the finetuning process for text generation. The core motivation for PDL stems from observations in information theory and linguistics: the informativeness of a token is often inversely proportional to its frequency of occurrence. PDL re-weights the contribution of each token in the standard cross-entropy loss based on its frequency in the training corpus, following a power-law decay. Specifically, the weights for high-frequency tokens are reduced, while low-frequency, information-dense tokens are assigned higher weights. This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the quality, diversity, and informativeness of the generated text. We theoretically elaborate on the motivation and construction of PDL and discuss its potential applications and advantages across various text generation finetuning tasks, such as abstractive summarization, dialogue systems, and style transfer.
>
---
#### [replaced 017] MediaSpin: Exploring Media Bias Through Fine-Grained Analysis of News Headlines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.02271v2](http://arxiv.org/pdf/2412.02271v2)**

> **作者:** Preetika Verma; Kokil Jaidka
>
> **备注:** 8 pages, 3 figures, 8 tables
>
> **摘要:** The editability of online news content has become a significant factor in shaping public perception, as social media platforms introduce new affordances for dynamic and adaptive news framing. Edits to news headlines can refocus audience attention, add or remove emotional language, and shift the framing of events in subtle yet impactful ways. What types of media bias are editorialized in and out of news headlines, and how can they be systematically identified? This study introduces the MediaSpin dataset, the first to characterize the bias in how prominent news outlets editorialize news headlines after publication. The dataset includes 78,910 pairs of headlines annotated with 13 distinct types of media bias, using human-supervised LLM labeling. We discuss the linguistic insights it affords and show its applications for bias prediction and user behavior analysis.
>
---
#### [replaced 018] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16529v2](http://arxiv.org/pdf/2502.16529v2)**

> **作者:** Deokhyung Kang; Jeonghun Cho; Yejin Jeon; Sunbin Jang; Minsub Lee; Jawoon Cho; Gary Geunbae Lee
>
> **备注:** Accepted at ACL 2025 (Main, long paper)
>
> **摘要:** Visual programming languages (VPLs) allow users to create programs through graphical interfaces, which results in easier accessibility and their widespread usage in various domains. To further enhance this accessibility, recent research has focused on generating VPL code from user instructions using large language models (LLMs). Specifically, by employing prompting-based methods, these studies have shown promising results. Nevertheless, such approaches can be less effective for industrial VPLs such as Ladder Diagram (LD). LD is a pivotal language used in industrial automation processes and involves extensive domain-specific configurations, which are difficult to capture in a single prompt. In this work, we demonstrate that training-based methods outperform prompting-based methods for LD generation accuracy, even with smaller backbone models. Building on these findings, we propose a two-stage training strategy to further enhance VPL generation. First, we employ retrieval-augmented fine-tuning to leverage the repetitive use of subroutines commonly seen in industrial VPLs. Second, we apply direct preference optimization (DPO) to further guide the model toward accurate outputs, using systematically generated preference pairs through graph editing operations. Extensive experiments on real-world LD data demonstrate that our approach improves program-level accuracy by over 10% compared to supervised fine-tuning, which highlights its potential to advance industrial automation.
>
---
#### [replaced 019] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.09121v2](http://arxiv.org/pdf/2407.09121v2)**

> **作者:** Youliang Yuan; Wenxiang Jiao; Wenxuan Wang; Jen-tse Huang; Jiahao Xu; Tian Liang; Pinjia He; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses baseline methods in defending against attacks.
>
---
#### [replaced 020] Temporal Dynamics of Emotion and Cognition in Human Translation: Integrating the Task Segment Framework and the HOF Taxonomy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.03111v4](http://arxiv.org/pdf/2405.03111v4)**

> **作者:** Michael Carl
>
> **备注:** Paper was split & published as: --- Carl, M. (2025) Temporal Dynamics of Emotion and Cognition in Human Translation: Integrating the Task Segment Framework and the HOF Taxonomy. Digital Studies in Language and Literature, DeGruyter --- Carl, M. (2025) Tracing the Temporal Dynamics of Emotion and Cognition in Behavioral Translation Data. Translation Spaces. John Benjamins Publishing Company
>
> **摘要:** The article develops a generative model of the human translating mind, grounded in empirical translation process data. It posits that three embedded processing layers unfold concurrently in the human mind, and their traces are detectable in behavioral data: sequences of routinized/automated processes are observable in fluent translation production, cognitive/reflective thoughts lead to longer keystroke pauses, while affective/emotional states may be identified through characteristic typing and gazing patterns. Utilizing data from the CRITT Translation Process Research Database (TPR-DB), the article illustrates how the temporal structure of keystroke and gaze data can be related to the three assumed hidden mental processing strata. The article relates this embedded generative model to various theoretical frameworks, dual-process theories and Robinson's (2023) ideosomatic theory of translation, opening exciting new theoretical horizons for Cognitive Translation Studies, grounded in empirical data and evaluation.
>
---
#### [replaced 021] TrustRAG: Enhancing Robustness and Trustworthiness in Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.00879v3](http://arxiv.org/pdf/2501.00879v3)**

> **作者:** Huichi Zhou; Kin-Hei Lee; Zhonghao Zhan; Yue Chen; Zhenhao Li; Zhaoyang Wang; Hamed Haddadi; Emine Yilmaz
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. These systems, however, remain susceptible to corpus poisoning attacks, which can severely impair the performance of LLMs. To address this challenge, we propose TrustRAG, a robust framework that systematically filters malicious and irrelevant content before it is retrieved for generation. Our approach employs a two-stage defense mechanism. The first stage implements a cluster filtering strategy to detect potential attack patterns. The second stage employs a self-assessment process that harnesses the internal capabilities of LLMs to detect malicious documents and resolve inconsistencies. TrustRAG provides a plug-and-play, training-free module that integrates seamlessly with any open- or closed-source language model. Extensive experiments demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance.
>
---
#### [replaced 022] ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14238v2](http://arxiv.org/pdf/2505.14238v2)**

> **作者:** Raghav Singhal; Kaustubh Ponkshe; Rohit Vartak; Praneeth Vepakomma
>
> **备注:** Raghav Singhal, Kaustubh Ponkshe, and Rohit Vartak contributed equally to this work
>
> **摘要:** Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA's expressive capacity and validate its advantages through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: https://github.com/CERT-Lab/abba.
>
---
#### [replaced 023] Playpen: An Environment for Exploring Learning Through Conversational Interaction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.08590v2](http://arxiv.org/pdf/2504.08590v2)**

> **作者:** Nicola Horst; Davide Mazzaccara; Antonia Schmidt; Michael Sullivan; Filippo Momentè; Luca Franceschetti; Philipp Sadler; Sherzod Hakimov; Alberto Testoni; Raffaella Bernardi; Raquel Fernández; Alexander Koller; Oliver Lemon; David Schlangen; Mario Giulianelli; Alessandro Suglia
>
> **备注:** Source code: https://github.com/lm-playpen/playpen Please send correspodence to: lm-playschool@googlegroups.com
>
> **摘要:** Interaction between learner and feedback-giver has come into focus recently for post-training of Large Language Models (LLMs), through the use of reward models that judge the appropriateness of a model's response. In this paper, we investigate whether Dialogue Games -- goal-directed and rule-governed activities driven predominantly by verbal actions -- can also serve as a source of feedback signals for learning. We introduce Playpen, an environment for off- and online learning through Dialogue Game self-play, and investigate a representative set of post-training methods: supervised fine-tuning; direct alignment (DPO); and reinforcement learning with GRPO. We experiment with post-training a small LLM (Llama-3.1-8B-Instruct), evaluating performance on unseen instances of training games as well as unseen games, and on standard benchmarks. We find that imitation learning through SFT improves performance on unseen instances, but negatively impacts other skills, while interactive learning with GRPO shows balanced improvements without loss of skills. We release the framework and the baseline training setups to foster research in the promising new direction of learning in (synthetic) interaction.
>
---
#### [replaced 024] Beyond One-Size-Fits-All Pruning via Evolutionary Metric Search for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10735v2](http://arxiv.org/pdf/2502.10735v2)**

> **作者:** Shuqi Liu; Bowei He; Han Wu; Linqi Song
>
> **摘要:** Post-training pruning has emerged as a crucial optimization technique as large language models (LLMs) continue to grow rapidly. However, the significant variations in weight distributions across different LLMs make fixed pruning strategies inadequate for multiple models. In this paper, we introduce \textbf{\textsc{OptiShear}}, an efficient evolutionary optimization framework for adaptive LLM pruning. Our framework features two key innovations: an effective search space built on our Meta pruning metric to handle diverse weight distributions, and a model-wise reconstruction error for rapid evaluation during search trials. We employ Non-dominated Sorting Genetic Algorithm III (NSGA-III) to optimize both pruning metrics and layerwise sparsity ratios. Through extensive evaluation on LLaMA-1/2/3 and Mistral models (7B-70B) across multiple benchmarks, we demonstrate that our adaptive pruning metrics consistently outperform existing methods. Additionally, our discovered layerwise sparsity ratios enhance the effectiveness of other pruning metrics. The framework exhibits strong cross-task and cross-model generalizability, providing a cost-effective solution for model compression.
>
---
#### [replaced 025] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12594v2](http://arxiv.org/pdf/2502.12594v2)**

> **作者:** Bowei He; Lihao Yin; Hui-Ling Zhen; Xiaokun Zhang; Mingxuan Yuan; Chen Ma
>
> **摘要:** Model pruning is an effective approach for compressing large language models (LLMs). However, this process often leads to significant degradation of model capabilities. While post-training techniques such as instruction tuning are commonly employed to recover model performance, existing methods often overlook the uneven deterioration of model capabilities and incur high computational costs. Moreover, some irrelevant instructions may also introduce negative effects to model capacity recovery. To address these challenges, we propose the \textbf{P}ost-training d\textbf{A}ta \textbf{S}election method for \textbf{E}fficient pruned large language model \textbf{R}ecovery (\textbf{PASER}). PASER aims to identify instructions to recover the most compromised model capacities with a certain data budget. Our approach first applies manifold learning and spectral clustering to group recovery instructions in the semantic space, revealing capability-specific instruction sets. Then, the data budget is adaptively allocated across clusters by the degree of corresponding model capability degradation. In each cluster, we prioritize data samples that lead to the most decline of model performance. To mitigate potential negative tuning effects, we also detect and filter out conflicting or irrelevant recovery data. Extensive experiments demonstrate that PASER significantly outperforms conventional baselines, effectively recovering the general capabilities of pruned LLMs while utilizing merely 4\%-20\% of the original post-training data. We provide the anonymous code repository in \href{https://anonymous.4open.science/r/PASER-E606}{Link}.
>
---
#### [replaced 026] Long Sequence Modeling with Attention Tensorization: From Sequence to Tensor Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20926v2](http://arxiv.org/pdf/2410.20926v2)**

> **作者:** Aosong Feng; Rex Ying; Leandros Tassiulas
>
> **摘要:** As the demand for processing extended textual data grows, the ability to handle long-range dependencies and maintain computational efficiency is more critical than ever. One of the key issues for long-sequence modeling using attention-based model is the mismatch between the limited-range modeling power of full attention and the long-range token dependency in the input sequence. In this work, we propose to scale up the attention receptive field by tensorizing long input sequences into compact tensor representations followed by attention on each transformed dimension. The resulting Tensorized Attention can be adopted as efficient transformer backbones to extend input context length with improved memory and time efficiency. We show that the proposed attention tensorization encodes token dependencies as a multi-hop attention process, and is equivalent to Kronecker decomposition of full attention. Extensive experiments show that tensorized attention can be used to adapt pretrained LLMs with improved efficiency. Notably, Llama-8B with tensorization is trained under 32,768 context length and can steadily extrapolate to 128k length during inference with $11\times$ speedup, compared to full attention with FlashAttention-2.
>
---
#### [replaced 027] ConceptCarve: Dynamic Realization of Evidence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07228v2](http://arxiv.org/pdf/2504.07228v2)**

> **作者:** Eylon Caplan; Dan Goldwasser
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Finding evidence for human opinion and behavior at scale is a challenging task, often requiring an understanding of sophisticated thought patterns among vast online communities found on social media. For example, studying how gun ownership is related to the perception of Freedom, requires a retrieval system that can operate at scale over social media posts, while dealing with two key challenges: (1) identifying abstract concept instances, (2) which can be instantiated differently across different communities. To address these, we introduce ConceptCarve, an evidence retrieval framework that utilizes traditional retrievers and LLMs to dynamically characterize the search space during retrieval. Our experiments show that ConceptCarve surpasses traditional retrieval systems in finding evidence within a social media community. It also produces an interpretable representation of the evidence for that community, which we use to qualitatively analyze complex thought patterns that manifest differently across the communities.
>
---
#### [replaced 028] None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12896v4](http://arxiv.org/pdf/2502.12896v4)**

> **作者:** Eva Sánchez Salido; Julio Gonzalo; Guillermo Marco
>
> **摘要:** In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
>
---
#### [replaced 029] Fundamental Limitations on Subquadratic Alternatives to Transformers
- **分类: cs.LG; cs.CC; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.04271v2](http://arxiv.org/pdf/2410.04271v2)**

> **作者:** Josh Alman; Hantao Yu
>
> **摘要:** The Transformer architecture is widely deployed in many popular and impactful Large Language Models. At its core is the attention mechanism for calculating correlations between pairs of tokens. Performing an attention computation takes quadratic time in the input size, and had become the time bottleneck for transformer operations. In order to circumvent this, researchers have used a variety of approaches, including designing heuristic algorithms for performing attention computations faster, and proposing alternatives to the attention mechanism which can be computed more quickly. For instance, state space models such as Mamba were designed to replace attention with an almost linear time alternative. In this paper, we prove that any such approach cannot perform important tasks that Transformer is able to perform (assuming a popular conjecture from fine-grained complexity theory). We focus on document similarity tasks, where one is given as input many documents and would like to find a pair which is (approximately) the most similar. We prove that Transformer is able to perform this task, and we prove that this task cannot be performed in truly subquadratic time by any algorithm. Thus, any model which can be evaluated in subquadratic time - whether because of subquadratic-time heuristics for attention, faster attention replacements like Mamba, or any other reason - cannot perform this task. In other words, in order to perform tasks that (implicitly or explicitly) involve document similarity, one may as well use Transformer and cannot avoid its quadratic running time.
>
---
#### [replaced 030] FBQuant: FeedBack Quantization for Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.16385v2](http://arxiv.org/pdf/2501.16385v2)**

> **作者:** Yijiang Liu; Hengyu Fang; Liulu He; Rongyu Zhang; Yichuan Bai; Yuan Du; Li Du
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** Deploying Large Language Models (LLMs) on edge devices is increasingly important, as it eliminates reliance on network connections, reduces expensive API calls, and enhances user privacy. However, on-device deployment is challenging due to the limited computational resources of edge devices. In particular, the key bottleneck stems from memory bandwidth constraints related to weight loading. Weight-only quantization effectively reduces memory access, yet often induces significant accuracy degradation. Recent efforts to incorporate sub-branches have shown promise for mitigating quantization errors, but these methods either lack robust optimization strategies or rely on suboptimal objectives. To address these gaps, we propose FeedBack Quantization (FBQuant), a novel approach inspired by negative feedback mechanisms in automatic control. FBQuant inherently ensures that the reconstructed weights remain bounded by the quantization process, thereby reducing the risk of overfitting. To further offset the additional latency introduced by sub-branches, we develop an efficient CUDA kernel that decreases 60% of extra inference time. Comprehensive experiments demonstrate the efficiency and effectiveness of FBQuant across various LLMs. Notably, for 3-bit Llama2-7B, FBQuant improves zero-shot accuracy by 1.2%.
>
---
#### [replaced 031] A Retrieval-Based Approach to Medical Procedure Matching in Romanian
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20556v2](http://arxiv.org/pdf/2503.20556v2)**

> **作者:** Andrei Niculae; Adrian Cosma; Emilian Radoi
>
> **备注:** Accepted at BIONLP 2025 and Shared Tasks, ACL 2025
>
> **摘要:** Accurately mapping medical procedure names from healthcare providers to standardized terminology used by insurance companies is a crucial yet complex task. Inconsistencies in naming conventions lead to missclasified procedures, causing administrative inefficiencies and insurance claim problems in private healthcare settings. Many companies still use human resources for manual mapping, while there is a clear opportunity for automation. This paper proposes a retrieval-based architecture leveraging sentence embeddings for medical name matching in the Romanian healthcare system. This challenge is significantly more difficult in underrepresented languages such as Romanian, where existing pretrained language models lack domain-specific adaptation to medical text. We evaluate multiple embedding models, including Romanian, multilingual, and medical-domain-specific representations, to identify the most effective solution for this task. Our findings contribute to the broader field of medical NLP for low-resource languages such as Romanian.
>
---
#### [replaced 032] Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17086v3](http://arxiv.org/pdf/2502.17086v3)**

> **作者:** Hyungyu Shin; Jingyu Tang; Yoonjoo Lee; Nayoung Kim; Hyunseung Lim; Ji Yong Cho; Hwajung Hong; Moontae Lee; Juho Kim
>
> **摘要:** Peer review underpins scientific progress, but it is increasingly strained by reviewer shortages and growing workloads. Large Language Models (LLMs) can automatically draft reviews now, but determining whether LLM-generated reviews are trustworthy requires systematic evaluation. Researchers have evaluated LLM reviews at either surface-level (e.g., BLEU and ROUGE) or content-level (e.g., specificity and factual accuracy). Yet it remains uncertain whether LLM-generated reviews attend to the same critical facets that human experts weigh -- the strengths and weaknesses that ultimately drive an accept-or-reject decision. We introduce a focus-level evaluation framework that operationalizes the focus as a normalized distribution of attention across predefined facets in paper reviews. Based on the framework, we developed an automatic focus-level evaluation pipeline based on two sets of facets: target (e.g., problem, method, and experiment) and aspect (e.g., validity, clarity, and novelty), leveraging 676 paper reviews (https://figshare.com/s/d5adf26c802527dd0f62) from OpenReview that consists of 3,657 strengths and weaknesses identified from human experts. The comparison of focus distributions between LLMs and human experts showed that the off-the-shelf LLMs consistently have a more biased focus towards examining technical validity while significantly overlooking novelty assessment when criticizing papers.
>
---
#### [replaced 033] Offset Unlearning for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.11045v2](http://arxiv.org/pdf/2404.11045v2)**

> **作者:** James Y. Huang; Wenxuan Zhou; Fei Wang; Fred Morstatter; Sheng Zhang; Hoifung Poon; Muhao Chen
>
> **备注:** Published in TMLR. https://openreview.net/pdf?id=A4RLpHPXCu
>
> **摘要:** Despite the strong capabilities of Large Language Models (LLMs) to acquire knowledge from their training corpora, the memorization of sensitive information in the corpora such as copyrighted, biased, and private content has led to ethical and legal concerns. In response to these challenges, unlearning has emerged as a potential remedy for LLMs affected by problematic training data. However, previous unlearning techniques are either not applicable to black-box LLMs due to required access to model internal weights, or violate data protection principles by retaining sensitive data for inference-time correction. We propose {\delta}-Unlearning, an offset unlearning framework for black-box LLMs. Instead of tuning the black-box LLM itself, {\delta}-Unlearning learns the logit offset needed for unlearning by contrasting the logits from a pair of smaller models. Experiments demonstrate that {\delta}- Unlearning can effectively unlearn target data while maintaining similar or even stronger performance on general out-of-forget-scope tasks. {\delta}-Unlearning also effectively incorporates different unlearning algorithms, making our approach a versatile solution to adapting various existing unlearning algorithms to black-box LLMs.
>
---
#### [replaced 034] Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00022v2](http://arxiv.org/pdf/2505.00022v2)**

> **作者:** Thomas F Burns; Letitia Parcalabescu; Stephan Wäldchen; Michael Barlow; Gregor Ziegltrum; Volker Stampa; Bastian Harren; Björn Deiseroth
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a large-scale German pre-training dataset which draws from: (1) Common Crawl web data, (2) FineWeb2, and (3) synthetically-generated data conditioned on actual, organic web data. We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokenizer-free hierarchical autoregressive transformer (HAT). A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets.
>
---
#### [replaced 035] StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.05804v2](http://arxiv.org/pdf/2504.05804v2)**

> **作者:** Yiming Tang; Yi Fan; Chenxiao Yu; Tiankai Yang; Yue Zhao; Xiyang Hu
>
> **摘要:** The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.
>
---
#### [replaced 036] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.19133v4](http://arxiv.org/pdf/2410.19133v4)**

> **作者:** Lester James V. Miranda; Yizhong Wang; Yanai Elazar; Sachin Kumar; Valentina Pyatkin; Faeze Brahman; Noah A. Smith; Hannaneh Hajishirzi; Pradeep Dasigi
>
> **备注:** Code in https://github.com/allenai/hybrid-preferences, MultiPref dataset in https://huggingface.co/datasets/allenai/multipref, Updated related work and acknowledgments
>
> **摘要:** Learning from human feedback has enabled the alignment of language models (LMs) with human preferences. However, collecting human preferences is expensive and time-consuming, with highly variable annotation quality. An appealing alternative is to distill preferences from LMs as a source of synthetic annotations, offering a cost-effective and scalable alternative, albeit susceptible to other biases and errors. In this work, we introduce HyPER, a Hybrid Preference routER that defers an annotation to either humans or LMs, achieving better annotation quality while reducing the cost of human-only annotation. We formulate this as an optimization problem: given a preference dataset and an evaluation metric, we (1) train a performance prediction model (PPM) to predict a reward model's (RM) performance on an arbitrary combination of human and LM annotations and (2) employ a routing strategy that selects a combination that maximizes the predicted performance. We train the PPM on MultiPref, a new preference dataset with 10k instances paired with humans and LM labels. We show that the selected hybrid mixture of synthetic and direct human preferences using HyPER achieves better RM performance compared to using either one exclusively by 7-13% on RewardBench and generalizes across unseen preference datasets and other base models. We also observe the same trend in other benchmarks using Best-of-N reranking, where the hybrid mix has 2-3% better performance. Finally, we analyze features from HyPER and find that prompts with moderate safety concerns or complexity benefit the most from human feedback.
>
---
#### [replaced 037] Investigating Language Preference of Multilingual RAG Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11175v3](http://arxiv.org/pdf/2502.11175v3)**

> **作者:** Jeonghyun Park; Hwanhee Lee
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) systems enhance language models by integrating external multilingual information to produce context-aware responses. However, mRAG systems struggle with retrieving relevant information due to linguistic variations between queries and documents, generating inconsistent responses when multilingual sources conflict. In this work, we systematically investigate language preferences in both retrieval and generation of mRAG through a series of experiments. Our analysis indicates that retrievers tend to prefer high-resource and query languages, yet this preference does not consistently improve generation performance. Moreover, we observe that generators prefer the query language or Latin scripts, leading to inconsistent outputs. To overcome these issues, we propose Dual Knowledge Multilingual RAG (DKM-RAG), a simple yet effective framework that fuses translated multilingual passages with complementary model knowledge. Empirical results demonstrate that DKM-RAG mitigates language preference in generation and enhances performance across diverse linguistic settings.
>
---
#### [replaced 038] Task Arithmetic for Language Expansion in Speech Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11274v2](http://arxiv.org/pdf/2409.11274v2)**

> **作者:** Yao-Fei Cheng; Hayato Futami; Yosuke Kashiwagi; Emiru Tsunoo; Wen Shen Teo; Siddhant Arora; Shinji Watanabe
>
> **摘要:** Recent progress in large language models (LLMs) has gained interest in speech-text multimodal foundation models, achieving strong performance on instruction-tuned speech translation (ST). However, expanding language pairs is costly due to re-training on combined new and previous datasets. To address this, we aim to build a one-to-many ST system from existing one-to-one ST systems using task arithmetic without re-training. Direct application of task arithmetic in ST leads to language confusion; therefore, we introduce an augmented task arithmetic method incorporating a language control model to ensure correct target language generation. Our experiments on MuST-C and CoVoST-2 show BLEU score improvements of up to 4.66 and 4.92, with COMET gains of 8.87 and 11.83. In addition, we demonstrate our framework can extend to language pairs lacking paired ST training data or pre-trained ST models by synthesizing ST models based on existing machine translation (MT) and ST models via task analogies.
>
---
#### [replaced 039] From Lists to Emojis: How Format Bias Affects Model Alignment
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.11704v2](http://arxiv.org/pdf/2409.11704v2)**

> **作者:** Xuanchang Zhang; Wei Xiong; Lichang Chen; Tianyi Zhou; Heng Huang; Tong Zhang
>
> **备注:** Working in progress
>
> **摘要:** In this paper, we study format biases in reinforcement learning from human feedback (RLHF). We observe that many widely-used preference models, including human evaluators, GPT-4, and top-ranking models on the RewardBench benchmark, exhibit strong biases towards specific format patterns, such as lists, links, bold text, and emojis. Furthermore, large language models (LLMs) can exploit these biases to achieve higher rankings on popular benchmarks like AlpacaEval and LMSYS Chatbot Arena. One notable example of this is verbosity bias, where current preference models favor longer responses that appear more comprehensive, even when their quality is equal to or lower than shorter, competing responses. However, format biases beyond verbosity remain largely underexplored in the literature. In this work, we extend the study of biases in preference learning beyond the commonly recognized length bias, offering a comprehensive analysis of a wider range of format biases. Additionally, we show that with a small amount of biased data (less than 1%), we can inject significant bias into the reward model. Moreover, these format biases can also be easily exploited by downstream alignment algorithms, such as best-of-n sampling and online iterative DPO, as it is usually easier to manipulate the format than to improve the quality of responses. Our findings emphasize the need to disentangle format and content both for designing alignment algorithms and evaluating models.
>
---
#### [replaced 040] Compositional Causal Reasoning Evaluation in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04556v3](http://arxiv.org/pdf/2503.04556v3)**

> **作者:** Jacqueline R. M. A. Maasch; Alihan Hüyük; Xinnuo Xu; Aditya V. Nori; Javier Gonzalez
>
> **摘要:** Causal reasoning and compositional reasoning are two core aspirations in AI. Measuring the extent of these behaviors requires principled evaluation methods. We explore a unified perspective that considers both behaviors simultaneously, termed compositional causal reasoning (CCR): the ability to infer how causal measures compose and, equivalently, how causal quantities propagate through graphs. We instantiate a framework for the systematic evaluation of CCR for the average treatment effect and the probability of necessity and sufficiency. As proof of concept, we demonstrate CCR evaluation for language models in the LLama, Phi, and GPT families. On a math word problem, our framework revealed a range of taxonomically distinct error patterns. CCR errors increased with the complexity of causal paths for all models except o1.
>
---
#### [replaced 041] SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16637v2](http://arxiv.org/pdf/2505.16637v2)**

> **作者:** Wenjie Yang; Mao Zheng; Mingyang Song; Zheng Li
>
> **摘要:** Large language models (LLMs) have recently demonstrated remarkable capabilities in machine translation (MT). However, most advanced MT-specific LLMs heavily rely on external supervision signals during training, such as human-annotated reference data or trained reward models (RMs), which are often expensive to obtain and challenging to scale. To overcome this limitation, we propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for MT that is reference-free, fully online, and relies solely on self-judging rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs, e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR with external supervision from COMET, our strongest model, SSR-X-Zero-7B, achieves state-of-the-art performance in English $\leftrightarrow$ Chinese translation, surpassing all existing open-source models under 72B parameters and even outperforming closed-source models, e.g., GPT-4o and Gemini 1.5 Pro. Our analysis highlights the effectiveness of the self-rewarding mechanism compared to the external LLM-as-a-judge approach in MT and demonstrates its complementary benefits when combined with trained RMs. Our findings provide valuable insight into the potential of self-improving RL methods. We have publicly released our code, data and models.
>
---
#### [replaced 042] What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.16592v2](http://arxiv.org/pdf/2505.16592v2)**

> **作者:** Shijia Zhou; Siyao Peng; Simon Luebke; Jörg Haßler; Mario Haim; Saif M. Mohammad; Barbara Plank
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Media framing refers to the emphasis on specific aspects of perceived reality to shape how an issue is defined and understood. Its primary purpose is to shape public perceptions often in alignment with the authors' opinions and stances. However, the interaction between stance and media frame remains largely unexplored. In this work, we apply an interdisciplinary approach to conceptualize and computationally explore this interaction with internet memes on climate change. We curate CLIMATEMEMES, the first dataset of climate-change memes annotated with both stance and media frames, inspired by research in communication science. CLIMATEMEMES includes 1,184 memes sourced from 47 subreddits, enabling analysis of frame prominence over time and communities, and sheds light on the framing preferences of different stance holders. We propose two meme understanding tasks: stance detection and media frame detection. We evaluate LLaVA-NeXT and Molmo in various setups, and report the corresponding results on their LLM backbone. Human captions consistently enhance performance. Synthetic captions and human-corrected OCR also help occasionally. Our findings highlight that VLMs perform well on stance, but struggle on frames, where LLMs outperform VLMs. Finally, we analyze VLMs' limitations in handling nuanced frames and stance expressions on climate change internet memes.
>
---
#### [replaced 043] GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01406v2](http://arxiv.org/pdf/2502.01406v2)**

> **作者:** Jonathan Drechsel; Steffen Herbold
>
> **摘要:** AI systems frequently exhibit and amplify social biases, including gender bias, leading to harmful consequences in critical areas. This study introduces a novel encoder-decoder approach that leverages model gradients to learn a single monosemantic feature neuron encoding gender information. We show that our method can be used to debias transformer-based language models, while maintaining other capabilities. We demonstrate the effectiveness of our approach across various model architectures and highlight its potential for broader applications.
>
---
#### [replaced 044] 1bit-Merging: Dynamic Quantized Merging for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10743v2](http://arxiv.org/pdf/2502.10743v2)**

> **作者:** Shuqi Liu; Yuxuan Yao; Bowei He; Zehua Liu; Xiongwei Han; Mingxuan Yuan; Han Wu; Linqi Song
>
> **摘要:** Recent advances in large language models have led to specialized models excelling in specific domains, creating a need for efficient model merging techniques. While traditional merging approaches combine parameters into a single static model, they often compromise task-specific performance. However, task-specific routing methods maintain accuracy but introduce substantial storage overhead. We present \texttt{1bit}-Merging, a novel framework that integrates task-specific routing with 1-bit quantized task vectors to balance performance and storage efficiency. Our approach leverages the observation that different task-specific models store knowledge in distinct layers-chat models primarily in attention layers and math/code models in MLP layers, enabling targeted compression strategies. Through extensive experiments with LLaMA2 and Mistral model families across chat, mathematical reasoning, and code generation tasks, we demonstrate that 1bit-Merging achieves comparable or superior performance to existing methods while significantly reducing storage requirements. Our framework offers a practical solution for combining specialized models while maintaining their individual strengths and addressing the storage challenges of current approaches.
>
---
#### [replaced 045] Optimizing Large Language Model Training Using FP4 Quantization
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.17116v2](http://arxiv.org/pdf/2501.17116v2)**

> **作者:** Ruizhe Wang; Yeyun Gong; Xiao Liu; Guoshuai Zhao; Ziyue Yang; Baining Guo; Zhengjun Zha; Peng Cheng
>
> **摘要:** The growing computational demands of training large language models (LLMs) necessitate more efficient methods. Quantized training presents a promising solution by enabling low-bit arithmetic operations to reduce these costs. While FP8 precision has demonstrated feasibility, leveraging FP4 remains a challenge due to significant quantization errors and limited representational capacity. This work introduces the first FP4 training framework for LLMs, addressing these challenges with two key innovations: a differentiable quantization estimator for precise weight updates and an outlier clamping and compensation strategy to prevent activation collapse. To ensure stability, the framework integrates a mixed-precision training scheme and vector-wise quantization. Experimental results demonstrate that our FP4 framework achieves accuracy comparable to BF16 and FP8, with minimal degradation, scaling effectively to 13B-parameter LLMs trained on up to 100B tokens. With the emergence of next-generation hardware supporting FP4, our framework sets a foundation for efficient ultra-low precision training.
>
---
#### [replaced 046] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.15957v2](http://arxiv.org/pdf/2505.15957v2)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** Project Website: https://github.com/ckyang1124/LALM-Evaluation-Survey
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [replaced 047] Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16552v2](http://arxiv.org/pdf/2505.16552v2)**

> **作者:** Wenhui Tan; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Ruihua Song
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
>
---
#### [replaced 048] When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15400v2](http://arxiv.org/pdf/2505.15400v2)**

> **作者:** Xiaoyun Zhang; Jingqing Ruan; Xing Ma; Yawen Zhu; Haodong Zhao; Hao Li; Jiansong Chen; Ke Zeng; Xunliang Cai
>
> **摘要:** Large reasoning models (LRMs) achieve remarkable performance via long reasoning chains, but often incur excessive computational overhead due to redundant reasoning, especially on simple tasks. In this work, we systematically quantify the upper bounds of LRMs under both Long-Thinking and No-Thinking modes, and uncover the phenomenon of "Internal Self-Recovery Mechanism" where models implicitly supplement reasoning during answer generation. Building on this insight, we propose Adaptive Self-Recovery Reasoning (ASRR), a framework that suppresses unnecessary reasoning and enables implicit recovery. By introducing accuracy-aware length reward regulation, ASRR adaptively allocates reasoning effort according to problem difficulty, achieving high efficiency with negligible performance sacrifice. Experiments across multiple benchmarks and models show that, compared with GRPO, ASRR reduces reasoning budget by up to 32.5% (1.5B) and 25.7% (7B) with minimal accuracy loss (1.2% and 0.6% pass@1), and significantly boosts harmless rates on safety benchmarks (up to +21.7%). Our results highlight the potential of ASRR for enabling efficient, adaptive, and safer reasoning in LRMs.
>
---
#### [replaced 049] Chain-of-Model Learning for Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11820v2](http://arxiv.org/pdf/2505.11820v2)**

> **作者:** Kaitao Song; Xiaohua Wang; Xu Tan; Huiqiang Jiang; Chengruidong Zhang; Yongliang Shen; Cen LU; Zihao Li; Zifan Song; Caihua Shan; Yansen Wang; Kan Ren; Xiaoqing Zheng; Tao Qin; Yuqing Yang; Dongsheng Li; Lili Qiu
>
> **摘要:** In this paper, we propose a novel learning paradigm, termed Chain-of-Model (CoM), which incorporates the causal relationship into the hidden states of each layer as a chain style, thereby introducing great scaling efficiency in model training and inference flexibility in deployment. We introduce the concept of Chain-of-Representation (CoR), which formulates the hidden states at each layer as a combination of multiple sub-representations (i.e., chains) at the hidden dimension level. In each layer, each chain from the output representations can only view all of its preceding chains in the input representations. Consequently, the model built upon CoM framework can progressively scale up the model size by increasing the chains based on the previous models (i.e., chains), and offer multiple sub-models at varying sizes for elastic inference by using different chain numbers. Based on this principle, we devise Chain-of-Language-Model (CoLM), which incorporates the idea of CoM into each layer of Transformer architecture. Based on CoLM, we further introduce CoLM-Air by introducing a KV sharing mechanism, that computes all keys and values within the first chain and then shares across all chains. This design demonstrates additional extensibility, such as enabling seamless LM switching, prefilling acceleration and so on. Experimental results demonstrate our CoLM family can achieve comparable performance to the standard Transformer, while simultaneously enabling greater flexiblity, such as progressive scaling to improve training efficiency and offer multiple varying model sizes for elastic inference, paving a a new way toward building language models. Our code will be released in the future at: https://github.com/microsoft/CoLM.
>
---
#### [replaced 050] Can Large Language Models Invent Algorithms to Improve Themselves?: Algorithm Discovery for Recursive Self-Improvement through Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.15639v4](http://arxiv.org/pdf/2410.15639v4)**

> **作者:** Yoichi Ishibashi; Taro Yano; Masafumi Oyamada
>
> **备注:** Accepted at NAACL 2025 (main)
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable capabilities, yet their improvement methods remain fundamentally constrained by human design. We present Self-Developing, a framework that enables LLMs to autonomously discover, implement, and refine their own improvement algorithms. Our approach employs an iterative cycle where a seed model generates algorithmic candidates as executable code, evaluates their effectiveness, and uses Direct Preference Optimization to recursively improve increasingly sophisticated improvement strategies. We demonstrate this framework through model merging, a practical technique for combining specialized models. Self-Developing successfully discovered novel merging algorithms that outperform existing human-designed algorithms. On mathematical reasoning benchmarks, the autonomously discovered algorithms improve the seed model's GSM8k performance by 6\% and exceed human-designed approaches like Task Arithmetic by 4.3\%. Remarkably, these algorithms exhibit strong generalization, achieving 7.4\% gains on out-of-domain models without re-optimization. Our findings demonstrate that LLMs can transcend their training to invent genuinely novel optimization techniques. This capability represents a crucial step toward a new era where LLMs not only solve problems but autonomously develop the methodologies for their own advancement.
>
---
#### [replaced 051] TAD-Bench: A Comprehensive Benchmark for Embedding-Based Text Anomaly Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.11960v2](http://arxiv.org/pdf/2501.11960v2)**

> **作者:** Yang Cao; Sikun Yang; Chen Li; Haolong Xiang; Lianyong Qi; Bo Liu; Rongsheng Li; Ming Liu
>
> **摘要:** Text anomaly detection is crucial for identifying spam, misinformation, and offensive language in natural language processing tasks. Despite the growing adoption of embedding-based methods, their effectiveness and generalizability across diverse application scenarios remain under-explored. To address this, we present TAD-Bench, a comprehensive benchmark designed to systematically evaluate embedding-based approaches for text anomaly detection. TAD-Bench integrates multiple datasets spanning different domains, combining state-of-the-art embeddings from large language models with a variety of anomaly detection algorithms. Through extensive experiments, we analyze the interplay between embeddings and detection methods, uncovering their strengths, weaknesses, and applicability to different tasks. These findings offer new perspectives on building more robust, efficient, and generalizable anomaly detection systems for real-world applications.
>
---
#### [replaced 052] TrendFact: A Benchmark for Explainable Hotspot Perception in Fact-Checking with Natural Language Explanation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.15135v3](http://arxiv.org/pdf/2410.15135v3)**

> **作者:** Xiaocheng Zhang; Xi Wang; Yifei Lu; Jianing Wang; Zhuangzhuang Ye; Mengjiao Bao; Peng Yan; Xiaohong Su
>
> **摘要:** Although fact verification remains fundamental, explanation generation serves as a critical enabler for trustworthy fact-checking systems by producing interpretable rationales and facilitating comprehensive verification processes. However, current benchmarks have limitations that include the lack of impact assessment, insufficient high-quality explanatory annotations, and an English-centric bias. To address these, we introduce TrendFact, the first hotspot perception fact-checking benchmark that comprehensively evaluates fact verification, evidence retrieval, and explanation generation tasks. TrendFact consists of 7,643 carefully curated samples sourced from trending platforms and professional fact-checking datasets, as well as an evidence library of 66,217 entries with publication dates. We further propose two metrics, ECS and HCPI, to complement existing benchmarks by evaluating the system's explanation consistency and hotspot perception capability, respectively. Experimental results show that current fact-checking systems, including advanced RLMs such as DeepSeek-R1, face significant limitations when evaluated on TrendFact, highlighting the real-world challenges posed by it. To enhance the fact-checking capabilities of reasoning large language models (RLMs), we propose FactISR, which integrates dynamic evidence augmentation, evidence triangulation, and an iterative self-reflection mechanism. Accordingly, FactISR effectively improves RLM performance, offering new insights for explainable and complex fact-checking.
>
---
#### [replaced 053] ThinkLess: A Training-Free Inference-Efficient Method for Reducing Reasoning Redundancy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15684v2](http://arxiv.org/pdf/2505.15684v2)**

> **作者:** Gengyang Li; Yifeng Gao; Yuming Li; Yunfang Wu
>
> **摘要:** While Chain-of-Thought (CoT) prompting improves reasoning in large language models (LLMs), the excessive length of reasoning tokens increases latency and KV cache memory usage, and may even truncate final answers under context limits. We propose ThinkLess, an inference-efficient framework that terminates reasoning generation early and maintains output quality without modifying the model. Atttention analysis reveals that answer tokens focus minimally on earlier reasoning steps and primarily attend to the reasoning terminator token, due to information migration under causal masking. Building on this insight, ThinkLess inserts the terminator token at earlier positions to skip redundant reasoning while preserving the underlying knowledge transfer. To prevent format discruption casued by early termination, ThinkLess employs a lightweight post-regulation mechanism, relying on the model's natural instruction-following ability to produce well-structured answers. Without fine-tuning or auxiliary data, ThinkLess achieves comparable accuracy to full-length CoT decoding while greatly reducing decoding time and memory consumption.
>
---
#### [replaced 054] Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.14479v3](http://arxiv.org/pdf/2406.14479v3)**

> **作者:** Jiachen Jiang; Jinxin Zhou; Zhihui Zhu
>
> **摘要:** Analyzing the similarity of internal representations has been an important technique for understanding the behavior of deep neural networks. Most existing methods for analyzing the similarity between representations of high dimensions, such as those based on Centered Kernel Alignment (CKA), rely on statistical properties of the representations for a set of data points. In this paper, we focus on transformer models and study the similarity of representations between the hidden layers of individual transformers. In this context, we show that a simple sample-wise cosine similarity metric is capable of capturing the similarity and aligns with the complicated CKA. Our experimental results on common transformers reveal that representations across layers are positively correlated, with similarity increasing when layers get closer. We provide a theoretical justification for this phenomenon under the geodesic curve assumption for the learned transformer. We then show that an increase in representation similarity implies an increase in predicted probability when directly applying the last-layer classifier to any hidden layer representation. We then propose an aligned training method to improve the effectiveness of shallow layer by enhancing the similarity between internal representations, with trained models that enjoy the following properties: (1) more early saturation events, (2) layer-wise accuracies monotonically increase and reveal the minimal depth needed for the given task, (3) when served as multi-exit models, they achieve on-par performance with standard multi-exit architectures which consist of additional classifiers designed for early exiting in shallow layers. To our knowledge, our work is the first to show that one common classifier is sufficient for multi-exit models. We conduct experiments on both vision and NLP tasks to demonstrate the performance of the proposed aligned training.
>
---
#### [replaced 055] Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02623v3](http://arxiv.org/pdf/2503.02623v3)**

> **作者:** Paul Stangel; David Bani-Harouni; Chantal Pellegrini; Ege Özsoy; Kamilia Zaripova; Matthias Keicher; Nassir Navab
>
> **摘要:** A safe and trustworthy use of Large Language Models (LLMs) requires an accurate expression of confidence in their answers. We propose a novel Reinforcement Learning approach that allows to directly fine-tune LLMs to express calibrated confidence estimates alongside their answers to factual questions. Our method optimizes a reward based on the logarithmic scoring rule, explicitly penalizing both over- and under-confidence. This encourages the model to align its confidence estimates with the actual predictive accuracy. The optimal policy under our reward design would result in perfectly calibrated confidence expressions. Unlike prior approaches that decouple confidence estimation from response generation, our method integrates confidence calibration seamlessly into the generative process of the LLM. Empirically, we demonstrate that models trained with our approach exhibit substantially improved calibration and generalize to unseen tasks without further fine-tuning, suggesting the emergence of general confidence awareness. We provide our training and evaluation code in the supplementary and will make it publicly available upon acceptance.
>
---
#### [replaced 056] ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.15968v2](http://arxiv.org/pdf/2406.15968v2)**

> **作者:** Roy Xie; Junlin Wang; Ruomin Huang; Minxing Zhang; Rong Ge; Jian Pei; Neil Zhenqiang Gong; Bhuwan Dhingra
>
> **备注:** Accepted to EMNLP 2024 Main Conference
>
> **摘要:** The rapid scaling of large language models (LLMs) has raised concerns about the transparency and fair use of the data used in their pretraining. Detecting such content is challenging due to the scale of the data and limited exposure of each instance during training. We propose ReCaLL (Relative Conditional Log-Likelihood), a novel membership inference attack (MIA) to detect LLMs' pretraining data by leveraging their conditional language modeling capabilities. ReCaLL examines the relative change in conditional log-likelihoods when prefixing target data points with non-member context. Our empirical findings show that conditioning member data on non-member prefixes induces a larger decrease in log-likelihood compared to non-member data. We conduct comprehensive experiments and show that ReCaLL achieves state-of-the-art performance on the WikiMIA dataset, even with random and synthetic prefixes, and can be further improved using an ensemble approach. Moreover, we conduct an in-depth analysis of LLMs' behavior with different membership contexts, providing insights into how LLMs leverage membership information for effective inference at both the sequence and token level.
>
---
#### [replaced 057] Multi-modal Retrieval Augmented Multi-modal Generation: Datasets, Evaluation Metrics and Strong Baselines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16365v4](http://arxiv.org/pdf/2411.16365v4)**

> **作者:** Zi-Ao Ma; Tian Lan; Rong-Cheng Tu; Yong Hu; Yu-Shi Zhu; Tong Zhang; Heyan Huang; Zhijing Wu; Xian-Ling Mao
>
> **摘要:** We present a systematic investigation of Multi-modal Retrieval Augmented Multi-modal Generation (M$^2$RAG), a novel task that enables foundation models to process multi-modal web content and generate multi-modal responses, which exhibits better information density and readability. Despite its potential impact, M$^2$RAG remains understudied, lacking comprehensive analysis and high-quality data resources. To address this gap, we establish a comprehensive benchmark through a rigorous data curation pipeline, and employ text-modal metrics and multi-modal metrics based on foundation models for evaluation. We further propose several strategies for foundation models to process M$^2$RAG task effectively and construct a training set by filtering high-quality samples using our designed metrics. Our extensive experiments demonstrate the reliability of our proposed metrics, a landscape of model performance within our designed strategies, and show that our fine-tuned 7B-8B models outperform the GPT-4o model and approach the state-of-the-art OpenAI o3-mini. Additionally, we perform fine-grained analyses across diverse domains and validate the effectiveness of our designs in data curation pipeline. All resources, including codes, datasets, and model weights, will be publicly released.
>
---
#### [replaced 058] HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14311v2](http://arxiv.org/pdf/2505.14311v2)**

> **作者:** Shamsuddeen Hassan Muhammad; Ibrahim Said Ahmad; Idris Abdulmumin; Falalu Ibrahim Lawan; Babangida Sani; Sukairaj Hafiz Imam; Yusuf Aliyu; Sani Abdullahi Sani; Ali Usman Umar; Tajuddeen Gwadabe; Kenneth Church; Vukosi Marivate
>
> **摘要:** Hausa Natural Language Processing (NLP) has gained increasing attention in recent years, yet remains understudied as a low-resource language despite having over 120 million first-language (L1) and 80 million second-language (L2) speakers worldwide. While significant advances have been made in high-resource languages, Hausa NLP faces persistent challenges, including limited open-source datasets and inadequate model representation. This paper presents an overview of the current state of Hausa NLP, systematically examining existing resources, research contributions, and gaps across fundamental NLP tasks: text classification, machine translation, named entity recognition, speech recognition, and question answering. We introduce HausaNLP (https://catalog.hausanlp.org), a curated catalog that aggregates datasets, tools, and research works to enhance accessibility and drive further development. Furthermore, we discuss challenges in integrating Hausa into large language models (LLMs), addressing issues of suboptimal tokenization and dialectal variation. Finally, we propose strategic research directions emphasizing dataset expansion, improved language modeling approaches, and strengthened community collaboration to advance Hausa NLP. Our work provides both a foundation for accelerating Hausa NLP progress and valuable insights for broader multilingual NLP research.
>
---
#### [replaced 059] Do Retrieval-Augmented Language Models Adapt to Varying User Needs?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19779v2](http://arxiv.org/pdf/2502.19779v2)**

> **作者:** Peilin Wu; Xinlu Zhang; Wenhao Yu; Xingyu Liu; Xinya Du; Zhiyu Zoey Chen
>
> **备注:** Updated the motivation, data selection and creation process, and terminology of some keywords for better writing. The updates are mianly in introduction and experiment section
>
> **摘要:** Recent advancements in Retrieval-Augmented Language Models (RALMs) have demonstrated their efficacy in knowledge-intensive tasks. However, existing evaluation benchmarks often assume a single optimal approach to leveraging retrieved information, failing to account for varying user needs. This paper introduces a novel evaluation framework that systematically assesses RALMs under three user need cases-Context-Exclusive, Context-First, and Memory-First-across three distinct context settings: Context Matching, Knowledge Conflict, and Information Irrelevant. By varying both user instructions and the nature of retrieved information, our approach captures the complexities of real-world applications where models must adapt to diverse user requirements. Through extensive experiments on multiple QA datasets, including HotpotQA, DisentQA, and our newly constructed synthetic URAQ dataset, we find that restricting memory usage improves robustness in adversarial retrieval conditions but decreases peak performance with ideal retrieval results and model family dominates behavioral differences. Our findings highlight the necessity of user-centric evaluations in the development of retrieval-augmented systems and provide insights into optimizing model performance across varied retrieval contexts. We will release our code and URAQ dataset upon acceptance of the paper.
>
---
#### [replaced 060] URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04686v5](http://arxiv.org/pdf/2501.04686v5)**

> **作者:** Ruilin Luo; Zhuofan Zheng; Yifan Wang; Xinzhe Ni; Zicheng Lin; Songtao Jiang; Yiyao Yu; Chufan Shi; Ruihang Chu; Jin Zeng; Yujiu Yang
>
> **备注:** Update version. Project url: https://ursa-math.github.io
>
> **摘要:** Process Reward Models (PRMs) have shown promise in enhancing the mathematical reasoning capabilities of Large Language Models (LLMs) through Test-Time Scaling (TTS). However, their integration into multimodal reasoning remains largely unexplored. In this work, we take the first step toward unlocking the potential of PRMs in multimodal mathematical reasoning. We identify three key challenges: (1) the scarcity of high-quality reasoning data constrains the capabilities of foundation Multimodal Large Language Models (MLLMs), which imposes further limitations on the upper bounds of TTS and reinforcement learning (RL); (2) a lack of automated methods for process labeling within multimodal contexts persists; (3) the employment of process rewards in unimodal RL faces issues like reward hacking, which may extend to multimodal scenarios. To address these issues, we introduce URSA, a three-stage Unfolding multimodal Process-Supervision Aided training framework. We first construct MMathCoT-1M, a high-quality large-scale multimodal Chain-of-Thought (CoT) reasoning dataset, to build a stronger math reasoning foundation MLLM, URSA-8B. Subsequently, we go through an automatic process to synthesize process supervision data, which emphasizes both logical correctness and perceptual consistency. We introduce DualMath-1.1M to facilitate the training of URSA-8B-RM. Finally, we propose Process-Supervised Group-Relative-Policy-Optimization (PS-GRPO), pioneering a multimodal PRM-aided online RL method that outperforms vanilla GRPO. With PS-GRPO application, URSA-8B-PS-GRPO outperforms Gemma3-12B and GPT-4o by 8.4% and 2.7% on average across 6 benchmarks. Code, data and checkpoint can be found at https://github.com/URSA-MATH.
>
---
#### [replaced 061] HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12908v4](http://arxiv.org/pdf/2503.12908v4)**

> **作者:** Xinyan Jiang; Hang Ye; Yongxin Zhu; Xiaoying Zheng; Zikang Chen; Jun Gong
>
> **备注:** Accepted by ACL2025 findings
>
> **摘要:** Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks.
>
---
#### [replaced 062] Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14359v2](http://arxiv.org/pdf/2502.14359v2)**

> **作者:** Filippo Momentè; Alessandro Suglia; Mario Giulianelli; Ambra Ferrari; Alexander Koller; Oliver Lemon; David Schlangen; Raquel Fernández; Raffaella Bernardi
>
> **摘要:** We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
>
---
#### [replaced 063] Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01926v2](http://arxiv.org/pdf/2502.01926v2)**

> **作者:** Angelina Wang; Michelle Phan; Daniel E. Ho; Sanmi Koyejo
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Algorithmic fairness has conventionally adopted the mathematically convenient perspective of racial color-blindness (i.e., difference unaware treatment). However, we contend that in a range of important settings, group difference awareness matters. For example, differentiating between groups may be necessary in legal contexts (e.g., the U.S. compulsory draft applies to men but not women) and harm assessments (e.g., referring to girls as ``terrorists'' may be less harmful than referring to Muslim people as such). Thus, in contrast to most fairness work, we study fairness through the perspective of treating people differently -- when it is contextually appropriate to. We first introduce an important distinction between descriptive (fact-based), normative (value-based), and correlation (association-based) benchmarks. This distinction is significant because each category requires separate interpretation and mitigation tailored to its specific characteristics. Then, we present a benchmark suite composed of eight different scenarios for a total of 16k questions that enables us to assess difference awareness. Finally, we show results across ten models that demonstrate difference awareness is a distinct dimension to fairness where existing bias mitigation strategies may backfire.
>
---
#### [replaced 064] Diagnosing our datasets: How does my language model learn clinical information?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15024v2](http://arxiv.org/pdf/2505.15024v2)**

> **作者:** Furong Jia; David Sontag; Monica Agrawal
>
> **摘要:** Large language models (LLMs) have performed well across various clinical natural language processing tasks, despite not being directly trained on electronic health record (EHR) data. In this work, we examine how popular open-source LLMs learn clinical information from large mined corpora through two crucial but understudied lenses: (1) their interpretation of clinical jargon, a foundational ability for understanding real-world clinical notes, and (2) their responses to unsupported medical claims. For both use cases, we investigate the frequency of relevant clinical information in their corresponding pretraining corpora, the relationship between pretraining data composition and model outputs, and the sources underlying this data. To isolate clinical jargon understanding, we evaluate LLMs on a new dataset MedLingo. Unsurprisingly, we find that the frequency of clinical jargon mentions across major pretraining corpora correlates with model performance. However, jargon frequently appearing in clinical notes often rarely appears in pretraining corpora, revealing a mismatch between available data and real-world usage. Similarly, we find that a non-negligible portion of documents support disputed claims that can then be parroted by models. Finally, we classified and analyzed the types of online sources in which clinical jargon and unsupported medical claims appear, with implications for future dataset composition.
>
---
#### [replaced 065] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05528v2](http://arxiv.org/pdf/2505.05528v2)**

> **作者:** Hanxun Huang; Sarah Erfani; Yige Li; Xingjun Ma; James Bailey
>
> **备注:** ICML 2025
>
> **摘要:** As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.
>
---
#### [replaced 066] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03303v2](http://arxiv.org/pdf/2503.03303v2)**

> **作者:** Yi-Fan Lu; Xian-Ling Mao; Tian Lan; Tong Zhang; Yu-Shi Zhu; Heyan Huang
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Automatic evaluation for Open Domain Event Detection (ODED) is a highly challenging task, because ODED is characterized by a vast diversity of un-constrained output labels from various domains. Nearly all existing evaluation methods for ODED usually first construct evaluation benchmarks with limited labels and domain coverage, and then evaluate ODED methods using metrics based on token-level label matching rules. However, this kind of evaluation framework faces two issues: (1) The limited evaluation benchmarks lack representatives of the real world, making it difficult to accurately reflect the performance of various ODED methods in real-world scenarios; (2) Evaluation metrics based on token-level matching rules fail to capture semantic similarity between predictions and golden labels. To address these two problems above, we propose a scalable and reliable Semantic-level Evaluation framework for Open domain Event detection (SEOE) by constructing a more representative evaluation benchmark and introducing a semantic evaluation metric. Specifically, our proposed framework first constructs a scalable evaluation benchmark that currently includes 564 event types covering 7 major domains, with a cost-effective supplementary annotation strategy to ensure the benchmark's representativeness. The strategy also allows for the supplement of new event types and domains in the future. Then, the proposed SEOE leverages large language models (LLMs) as automatic evaluation agents to compute a semantic F1-score, incorporating fine-grained definitions of semantically similar labels to enhance the reliability of the evaluation. Extensive experiments validate the representatives of the benchmark and the reliability of the semantic evaluation metric. Existing ODED methods are thoroughly evaluated, and the error patterns of predictions are analyzed, revealing several insightful findings.
>
---
#### [replaced 067] System Message Generation for User Preferences using Open-Source Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11330v2](http://arxiv.org/pdf/2502.11330v2)**

> **作者:** Minbyul Jeong; Jungho Cho; Minsoo Khang; Dawoon Jung; Teakgyu Hong
>
> **摘要:** System messages play a crucial role in interactions with large language models (LLMs), often serving as prompts to initiate conversations. Through system messages, users can assign specific roles, perform intended tasks, incorporate background information, and specify various output formats and communication styles. Despite such versatility, publicly available datasets often lack system messages and are subject to strict license constraints in industrial applications. Moreover, manually annotating system messages that align with user instructions is resource-intensive. In light of these challenges, we introduce SysGen, a pipeline for generating system messages that better align assistant responses with user instructions using existing supervised fine-tuning datasets that lack system messages. Training open-source models on SysGen data yields substantial improvements in both single-turn (Multifacet) and multi-turn (SysBench) conversation benchmarks. Notably, our method shows strong gains in shorter conversations, suggesting that it enhances early-stage interaction effectiveness. Our qualitative analysis further emphasizes the value of diverse and structured system messages in improving LLM adaptability across varied user scenarios.
>
---
#### [replaced 068] GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12842v2](http://arxiv.org/pdf/2505.12842v2)**

> **作者:** Zheng Wu; Pengzhou Cheng; Zongru Wu; Lingzhong Dong; Zhuosheng Zhang
>
> **摘要:** Graphical user interface (GUI) agents have recently emerged as an intriguing paradigm for human-computer interaction, capable of automatically executing user instructions to operate intelligent terminal devices. However, when encountering out-of-distribution (OOD) instructions that violate environmental constraints or exceed the current capabilities of agents, GUI agents may suffer task breakdowns or even pose security threats. Therefore, effective OOD detection for GUI agents is essential. Traditional OOD detection methods perform suboptimally in this domain due to the complex embedding space and evolving GUI environments. In this work, we observe that the in-distribution input semantic space of GUI agents exhibits a clustering pattern with respect to the distance from the centroid. Based on the finding, we propose GEM, a novel method based on fitting a Gaussian mixture model over input embedding distances extracted from the GUI Agent that reflect its capability boundary. Evaluated on eight datasets spanning smartphones, computers, and web browsers, our method achieves an average accuracy improvement of 23.70\% over the best-performing baseline. Analysis verifies the generalization ability of our method through experiments on nine different backbones. The codes are available at https://github.com/Wuzheng02/GEM-OODforGUIagents.
>
---
#### [replaced 069] Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.18533v2](http://arxiv.org/pdf/2501.18533v2)**

> **作者:** Yi Ding; Lijun Li; Bing Cao; Jing Shao
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.
>
---
#### [replaced 070] VeriFastScore: Speeding up long-form factuality evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16973v2](http://arxiv.org/pdf/2505.16973v2)**

> **作者:** Rishanth Rajendhran; Amir Zadeh; Matthew Sarte; Chuan Li; Mohit Iyyer
>
> **摘要:** Metrics like FactScore and VeriScore that evaluate long-form factuality operate by decomposing an input response into atomic claims and then individually verifying each claim. While effective and interpretable, these methods incur numerous LLM calls and can take upwards of 100 seconds to evaluate a single response, limiting their practicality in large-scale evaluation and training scenarios. To address this, we propose VeriFastScore, which leverages synthetic data to fine-tune Llama3.1 8B for simultaneously extracting and verifying all verifiable claims within a given text based on evidence from Google Search. We show that this task cannot be solved via few-shot prompting with closed LLMs due to its complexity: the model receives ~4K tokens of evidence on average and needs to concurrently decompose claims, judge their verifiability, and verify them against noisy evidence. However, our fine-tuned VeriFastScore model demonstrates strong correlation with the original VeriScore pipeline at both the example level (r=0.80) and system level (r=0.94) while achieving an overall speedup of 6.6x (9.9x excluding evidence retrieval) over VeriScore. To facilitate future factuality research, we publicly release our VeriFastScore model and synthetic datasets.
>
---
#### [replaced 071] LoRE-Merging: Exploring Low-Rank Estimation For Large Language Model Merging
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.10749v2](http://arxiv.org/pdf/2502.10749v2)**

> **作者:** Zehua Liu; Han Wu; Yuxuan Yao; Ruifeng She; Xiongwei Han; Tao Zhong; Mingxuan Yuan
>
> **摘要:** While most current approaches rely on further training techniques, such as fine-tuning or reinforcement learning, to enhance model capacities, model merging stands out for its ability of improving models without requiring any additional training. In this paper, we propose a unified framework for model merging based on low-rank estimation of task vectors without the need for access to the base model, named \textsc{LoRE-Merging}. Our approach is motivated by the observation that task vectors from fine-tuned models frequently exhibit a limited number of dominant singular values, making low-rank estimations less prone to interference. We implement the method by formulating the merging problem as an optimization problem. Extensive empirical experiments demonstrate the effectiveness of our framework in mitigating interference and preserving task-specific information, thereby advancing the state-of-the-art performance in model merging techniques.
>
---
#### [replaced 072] EpiCoder: Encompassing Diversity and Complexity in Code Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.04694v2](http://arxiv.org/pdf/2501.04694v2)**

> **作者:** Yaoxiang Wang; Haoling Li; Xin Zhang; Jie Wu; Xiao Liu; Wenxiang Hu; Zhongxin Guo; Yangyu Huang; Ying Xin; Yujiu Yang; Jinsong Su; Qi Chen; Scarlett Li
>
> **备注:** ICML 2025
>
> **摘要:** Existing methods for code generation use code snippets as seed data, restricting the complexity and diversity of the synthesized data. In this paper, we introduce a novel feature tree-based synthesis framework, which revolves around hierarchical code features derived from high-level abstractions of code. The feature tree is constructed from raw data and refined iteratively to increase the quantity and diversity of the extracted features, which captures and recognizes more complex patterns and relationships within the code. By adjusting the depth and breadth of the sampled subtrees, our framework provides precise control over the complexity of the generated code, enabling functionalities that range from function-level operations to multi-file scenarios. We fine-tuned widely-used base models to obtain EpiCoder series, achieving state-of-the-art performance on multiple benchmarks at both the function and file levels. In particular, empirical evidence indicates that our approach shows significant potential in the synthesizing of repository-level code data. Our code and data are publicly available at https://github.com/microsoft/EpiCoder.
>
---
#### [replaced 073] Rank, Chunk and Expand: Lineage-Oriented Reasoning for Taxonomy Expansion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13282v3](http://arxiv.org/pdf/2505.13282v3)**

> **作者:** Sahil Mishra; Kumar Arjun; Tanmoy Chakraborty
>
> **备注:** Accepted in the Findings of ACL 2025
>
> **摘要:** Taxonomies are hierarchical knowledge graphs crucial for recommendation systems, and web applications. As data grows, expanding taxonomies is essential, but existing methods face key challenges: (1) discriminative models struggle with representation limits and generalization, while (2) generative methods either process all candidates at once, introducing noise and exceeding context limits, or discard relevant entities by selecting noisy candidates. We propose LORex ($\textbf{L}$ineage-$\textbf{O}$riented $\textbf{Re}$asoning for Taxonomy E$\textbf{x}$pansion), a plug-and-play framework that combines discriminative ranking and generative reasoning for efficient taxonomy expansion. Unlike prior methods, LORex ranks and chunks candidate terms into batches, filtering noise and iteratively refining selections by reasoning candidates' hierarchy to ensure contextual efficiency. Extensive experiments across four benchmarks and twelve baselines show that LORex improves accuracy by 12% and Wu & Palmer similarity by 5% over state-of-the-art methods.
>
---
#### [replaced 074] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05066v3](http://arxiv.org/pdf/2503.05066v3)**

> **作者:** Shwai He; Weilin Cai; Jiayi Huang; Ang Li
>
> **摘要:** The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation to balance performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where underloaded experts complete computations early but must wait for overloaded experts, leading to global delays. We define this phenomenon as the \textbf{\textit{Straggler Effect}}, as the most burdened experts dictate the overall inference latency. To address this, we first propose \textit{\textbf{Capacity-Aware Token Drop}}, which enforces expert capacity limits by discarding excess tokens from overloaded experts, effectively reducing load imbalance with minimal performance impact (e.g., $30\%$ speedup with only $0.9\%$ degradation on OLMoE). Next, given the presence of low-load experts remaining well below the capacity threshold, we introduce \textit{\textbf{Capacity-Aware Expanded Drop}}, which allows tokens to include additional local experts in their candidate set before enforcing strict local capacity constraints, thereby improving load balance and enhancing the utilization of underused experts. Extensive experiments on both language and multimodal MoE models demonstrate the effectiveness of our approach, yielding substantial gains in expert utilization, model performance, and inference efficiency, e.g., applying Expanded Drop to Mixtral-8$\times$7B-Instruct yields a {0.2\%} average performance improvement and a {1.85$\times$} inference speedup.
>
---
#### [replaced 075] ICA-RAG: Information Completeness Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14614v4](http://arxiv.org/pdf/2502.14614v4)**

> **作者:** Mingyi Jia; Zhihao Jia; Junwen Duan; Yan Song; Jianxin Wang
>
> **摘要:** Retrieval-Augmented Large Language Models~(LLMs), which integrate external knowledge, have shown remarkable performance in medical domains, including clinical diagnosis. However, existing RAG methods often struggle to tailor retrieval strategies to diagnostic difficulty and input sample informativeness. This limitation leads to excessive and often unnecessary retrieval, impairing computational efficiency and increasing the risk of introducing noise that can degrade diagnostic accuracy. To address this, we propose ICA-RAG (\textbf{I}nformation \textbf{C}ompleteness Guided \textbf{A}daptive \textbf{R}etrieval-\textbf{A}ugmented \textbf{G}eneration), a novel framework for enhancing RAG reliability in disease diagnosis. ICA-RAG utilizes an adaptive control module to assess the necessity of retrieval based on the input's information completeness. By optimizing retrieval and incorporating knowledge filtering, ICA-RAG better aligns retrieval operations with clinical requirements. Experiments on three Chinese electronic medical record datasets demonstrate that ICA-RAG significantly outperforms baseline methods, highlighting its effectiveness in clinical diagnosis.
>
---
#### [replaced 076] EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16160v2](http://arxiv.org/pdf/2505.16160v2)**

> **作者:** Bin Xu; Yu Bai; Huashan Sun; Yiguan Lin; Siming Liu; Xinyue Liang; Yaolin Li; Yang Gao; Heyan Huang
>
> **摘要:** As large language models continue to advance, their application in educational contexts remains underexplored and under-optimized. In this paper, we address this gap by introducing the first diverse benchmark tailored for educational scenarios, incorporating synthetic data containing 9 major scenarios and over 4,000 distinct educational contexts. To enable comprehensive assessment, we propose a set of multi-dimensional evaluation metrics that cover 12 critical aspects relevant to both teachers and students. We further apply human annotation to ensure the effectiveness of the model-generated evaluation responses. Additionally, we succeed to train a relatively small-scale model on our constructed dataset and demonstrate that it can achieve performance comparable to state-of-the-art large models (e.g., Deepseek V3, Qwen Max) on the test set. Overall, this work provides a practical foundation for the development and evaluation of education-oriented language models. Code and data are released at https://github.com/ybai-nlp/EduBench.
>
---
#### [replaced 077] Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01796v2](http://arxiv.org/pdf/2407.01796v2)**

> **作者:** Sirui Xia; Xintao Wang; Jiaqing Liang; Yifei Zhang; Weikang Zhou; Jiaji Deng; Fei Yu; Yanghua Xiao
>
> **备注:** Accepted to NAACL 2025 Findings
>
> **摘要:** Retrieval-Augmented Generation (RAG) has been widely adopted to enhance Large Language Models (LLMs) in knowledge-intensive tasks. To enhance credibility and verifiability in RAG systems, Attributed Text Generation (ATG) is proposed, which provides citations to retrieval knowledge in LLM-generated responses. Prior methods mainly adopt coarse-grained attributions, with passage-level or paragraph-level references or citations, which fall short in verifiability. This paper proposes ReClaim (Refer & Claim), a fine-grained ATG method that alternates the generation of references and answers step by step. Different from previous coarse-grained attribution, ReClaim provides sentence-level citations in long-form question-answering tasks. With extensive experiments, we verify the effectiveness of ReClaim in extensive settings, achieving a citation accuracy rate of 90%.
>
---
#### [replaced 078] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12345v2](http://arxiv.org/pdf/2505.12345v2)**

> **作者:** Qizhou Chen; Dakan Wang; Taolin Zhang; Zaoming Yan; Chengsong You; Chengyu Wang; Xiaofeng He
>
> **备注:** UniEdit Dataset: https://huggingface.co/datasets/qizhou/UniEdit Code: https://github.com/qizhou000/UniEdit
>
> **摘要:** Model editing aims to enhance the accuracy and reliability of large language models (LLMs) by efficiently adjusting their internal parameters. Currently, most LLM editing datasets are confined to narrow knowledge domains and cover a limited range of editing evaluation. They often overlook the broad scope of editing demands and the diversity of ripple effects resulting from edits. In this context, we introduce UniEdit, a unified benchmark for LLM editing grounded in open-domain knowledge. First, we construct editing samples by selecting entities from 25 common domains across five major categories, utilizing the extensive triple knowledge available in open-domain knowledge graphs to ensure comprehensive coverage of the knowledge domains. To address the issues of generality and locality in editing, we design an Neighborhood Multi-hop Chain Sampling (NMCS) algorithm to sample subgraphs based on a given knowledge piece to entail comprehensive ripple effects to evaluate. Finally, we employ proprietary LLMs to convert the sampled knowledge subgraphs into natural language text, guaranteeing grammatical accuracy and syntactical diversity. Extensive statistical analysis confirms the scale, comprehensiveness, and diversity of our UniEdit benchmark. We conduct comprehensive experiments across multiple LLMs and editors, analyzing their performance to highlight strengths and weaknesses in editing across open knowledge domains and various evaluation criteria, thereby offering valuable insights for future research endeavors.
>
---
#### [replaced 079] HoT: Highlighted Chain of Thought for Referencing Supporting Facts from Inputs
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.02003v3](http://arxiv.org/pdf/2503.02003v3)**

> **作者:** Tin Nguyen; Logan Bolton; Mohammad Reza Taesiri; Anh Totti Nguyen
>
> **摘要:** An Achilles heel of Large Language Models (LLMs) is their tendency to hallucinate non-factual statements. A response mixed of factual and non-factual statements poses a challenge for humans to verify and accurately base their decisions on. To combat this problem, we propose Highlighted Chain-of-Thought Prompting (HoT), a technique for prompting LLMs to generate responses with XML tags that ground facts to those provided in the query. That is, given an input question, LLMs would first re-format the question to add XML tags highlighting key facts, and then, generate a response with highlights over the facts referenced from the input. Interestingly, in few-shot settings, HoT outperforms vanilla chain of thought prompting (CoT) on a wide range of 17 tasks from arithmetic, reading comprehension to logical reasoning. When asking humans to verify LLM responses, highlights help time-limited participants to more accurately and efficiently recognize when LLMs are correct. Yet, surprisingly, when LLMs are wrong, HoTs tend to make users believe that an answer is correct.
>
---
#### [replaced 080] DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11456v2](http://arxiv.org/pdf/2504.11456v2)**

> **作者:** Zhiwei He; Tian Liang; Jiahao Xu; Qiuzhi Liu; Xingyu Chen; Yue Wang; Linfeng Song; Dian Yu; Zhenwen Liang; Wenxuan Wang; Zhuosheng Zhang; Rui Wang; Zhaopeng Tu; Haitao Mi; Dong Yu
>
> **备注:** WIP
>
> **摘要:** Reinforcement learning (RL) with large language models shows promise in complex reasoning. However, its progress is hindered by the lack of large-scale training data that is sufficiently challenging, contamination-free and verifiable. To this end, we introduce DeepMath-103K, a large-scale mathematical dataset designed with high difficulty (primarily levels 5-9), rigorous decontamination against numerous benchmarks, and verifiable answers for rule-based RL reward. It further includes three distinct R1 solutions adaptable for diverse training paradigms such as supervised fine-tuning (SFT). Spanning a wide range of mathematical topics, DeepMath-103K fosters the development of generalizable and advancing reasoning. Notably, models trained on DeepMath-103K achieve state-of-the-art results on challenging mathematical benchmarks and demonstrate generalization beyond math such as biology, physics and chemistry, underscoring its broad efficacy. Data: https://huggingface.co/datasets/zwhe99/DeepMath-103K.
>
---
#### [replaced 081] Provably Correct Automata Embeddings for Optimal Automata-Conditioned Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2503.05042v2](http://arxiv.org/pdf/2503.05042v2)**

> **作者:** Beyazit Yalcinkaya; Niklas Lauffer; Marcell Vazquez-Chanlatte; Sanjit A. Seshia
>
> **摘要:** Automata-conditioned reinforcement learning (RL) has given promising results for learning multi-task policies capable of performing temporally extended objectives given at runtime, done by pretraining and freezing automata embeddings prior to training the downstream policy. However, no theoretical guarantees were given. This work provides a theoretical framework for the automata-conditioned RL problem and shows that it is probably approximately correct learnable. We then present a technique for learning provably correct automata embeddings, guaranteeing optimal multi-task policy learning. Our experimental evaluation confirms these theoretical results.
>
---
#### [replaced 082] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11803v2](http://arxiv.org/pdf/2412.11803v2)**

> **作者:** Boyang Xue; Fei Mi; Qi Zhu; Hongru Wang; Rui Wang; Sheng Wang; Erxin Yu; Xuming Hu; Kam-Fai Wong
>
> **备注:** Accepted in ACL2025 Main Conference
>
> **摘要:** Despite demonstrating impressive capabilities, Large Language Models (LLMs) still often struggle to accurately express the factual knowledge they possess, especially in cases where the LLMs' knowledge boundaries are ambiguous. To improve LLMs' factual expressions, we propose the UAlign framework, which leverages Uncertainty estimations to represent knowledge boundaries, and then explicitly incorporates these representations as input features into prompts for LLMs to Align with factual knowledge. First, we prepare the dataset on knowledge question-answering (QA) samples by calculating two uncertainty estimations, including confidence score and semantic entropy, to represent the knowledge boundaries for LLMs. Subsequently, using the prepared dataset, we train a reward model that incorporates uncertainty estimations and then employ the Proximal Policy Optimization (PPO) algorithm for factuality alignment on LLMs. Experimental results indicate that, by integrating uncertainty representations in LLM alignment, the proposed UAlign can significantly enhance the LLMs' capacities to confidently answer known questions and refuse unknown questions on both in-domain and out-of-domain tasks, showing reliability improvements and good generalizability over various prompt- and training-based baselines.
>
---
#### [replaced 083] Mitigate Position Bias in Large Language Models via Scaling a Single Dimension
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.02536v3](http://arxiv.org/pdf/2406.02536v3)**

> **作者:** Yijiong Yu; Huiqiang Jiang; Xufang Luo; Qianhui Wu; Chin-Yew Lin; Dongsheng Li; Yuqing Yang; Yongfeng Huang; Lili Qiu
>
> **备注:** Accepted at Findings of ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly applied in various real-world scenarios due to their excellent generalization capabilities and robust generative abilities. However, they exhibit position bias, also known as "lost in the middle", a phenomenon that is especially pronounced in long-context scenarios, which indicates the placement of the key information in different positions of a prompt can significantly affect accuracy. This paper first explores the micro-level manifestations of position bias, concluding that attention weights are a micro-level expression of position bias. It further identifies that, in addition to position embeddings, causal attention mask also contributes to position bias by creating position-specific hidden states. Based on these insights, we propose a method to mitigate position bias by scaling this positional hidden states. Experiments on the NaturalQuestions Multi-document QA, KV retrieval, LongBench and timeline reorder tasks, using various models including RoPE models, context windowextended models, and Alibi models, demonstrate the effectiveness and generalizability of our approach. Our method can improve performance by up to 15.2% by modifying just one dimension of hidden states. Our code is available at https://aka.ms/PositionalHidden.
>
---
#### [replaced 084] Information Gain-Guided Causal Intervention for Autonomous Debiasing Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12898v2](http://arxiv.org/pdf/2504.12898v2)**

> **作者:** Zhouhao Sun; Xiao Ding; Li Du; Yunpeng Xu; Yixuan Ma; Yang Zhao; Bing Qin; Ting Liu
>
> **摘要:** Despite significant progress, recent studies indicate that current large language models (LLMs) may still capture dataset biases and utilize them during inference, leading to the poor generalizability of LLMs. However, due to the diversity of dataset biases and the insufficient nature of bias suppression based on in-context learning, the effectiveness of previous prior knowledge-based debiasing methods and in-context learning based automatic debiasing methods is limited. To address these challenges, we explore the combination of causal mechanisms with information theory and propose an information gain-guided causal intervention debiasing (ICD) framework. To eliminate biases within the instruction-tuning dataset, it is essential to ensure that these biases do not provide any additional information to predict the answers, i.e., the information gain of these biases for predicting the answers needs to be 0. Under this guidance, this framework utilizes a causal intervention-based data rewriting method to automatically and autonomously balance the distribution of instruction-tuning dataset for reducing the information gain. Subsequently, it employs a standard supervised fine-tuning process to train LLMs on the debiased dataset. Experimental results show that ICD can effectively debias LLM to improve its generalizability across different tasks.
>
---
#### [replaced 085] Small Language Models in the Real World: Insights from Industrial Text Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16078v2](http://arxiv.org/pdf/2505.16078v2)**

> **作者:** Lujun Li; Lama Sleem; Niccolo' Gentile; Geoffrey Nichil; Radu State
>
> **摘要:** With the emergence of ChatGPT, Transformer models have significantly advanced text classification and related tasks. Decoder-only models such as Llama exhibit strong performance and flexibility, yet they suffer from inefficiency on inference due to token-by-token generation, and their effectiveness in text classification tasks heavily depends on prompt quality. Moreover, their substantial GPU resource requirements often limit widespread adoption. Thus, the question of whether smaller language models are capable of effectively handling text classification tasks emerges as a topic of significant interest. However, the selection of appropriate models and methodologies remains largely underexplored. In this paper, we conduct a comprehensive evaluation of prompt engineering and supervised fine-tuning methods for transformer-based text classification. Specifically, we focus on practical industrial scenarios, including email classification, legal document categorization, and the classification of extremely long academic texts. We examine the strengths and limitations of smaller models, with particular attention to both their performance and their efficiency in Video Random-Access Memory (VRAM) utilization, thereby providing valuable insights for the local deployment and application of compact models in industrial settings.
>
---
#### [replaced 086] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12158v2](http://arxiv.org/pdf/2505.12158v2)**

> **作者:** Elisa Bassignana; Amanda Cercas Curry; Dirk Hovy
>
> **备注:** Accepted at ACL Main 2025
>
> **摘要:** Socioeconomic status (SES) fundamentally influences how people interact with each other and more recently, with digital technologies like Large Language Models (LLMs). While previous research has highlighted the interaction between SES and language technology, it was limited by reliance on proxy metrics and synthetic data. We survey 1,000 individuals from diverse socioeconomic backgrounds about their use of language technologies and generative AI, and collect 6,482 prompts from their previous interactions with LLMs. We find systematic differences across SES groups in language technology usage (i.e., frequency, performed tasks), interaction styles, and topics. Higher SES entails a higher level of abstraction, convey requests more concisely, and topics like 'inclusivity' and 'travel'. Lower SES correlates with higher anthropomorphization of LLMs (using ''hello'' and ''thank you'') and more concrete language. Our findings suggest that while generative language technologies are becoming more accessible to everyone, socioeconomic linguistic differences still stratify their use to exacerbate the digital divide. These differences underscore the importance of considering SES in developing language technologies to accommodate varying linguistic needs rooted in socioeconomic factors and limit the AI Gap across SES groups.
>
---
#### [replaced 087] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.13237v2](http://arxiv.org/pdf/2505.13237v2)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025. Project page: https://github.com/ckyang1124/SAKURA
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [replaced 088] Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16241v2](http://arxiv.org/pdf/2505.16241v2)**

> **作者:** Viet-Anh Nguyen; Shiqian Zhao; Gia Dao; Runyi Hu; Yi Xie; Luu Anh Tuan
>
> **摘要:** Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.
>
---
#### [replaced 089] Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16315v2](http://arxiv.org/pdf/2505.16315v2)**

> **作者:** Xiaoxue Cheng; Junyi Li; Zhenduo Zhang; Xinyu Tang; Wayne Xin Zhao; Xinyu Kong; Zhiqiang Zhang
>
> **备注:** work in progress
>
> **摘要:** Large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks, but often suffer from overthinking, generating redundant content regardless of task difficulty. Inspired by the dual process theory in cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a reinforcement learning framework that enables LRMs to achieve efficient reasoning through adaptive cognitive allocation and dynamic system switch. ACPO incorporates two key components: (1) introducing system-aware reasoning tokens to explicitly represent the thinking modes thereby making the model's cognitive process transparent, and (2) integrating online difficulty estimation and token length budget to guide adaptive system switch and reasoning during reinforcement learning. To this end, we propose a two-stage training strategy. The first stage begins with supervised fine-tuning to cold start the model, enabling it to generate reasoning paths with explicit thinking modes. In the second stage, we apply ACPO to further enhance adaptive system switch for difficulty-aware reasoning. Experimental results demonstrate that ACPO effectively reduces redundant reasoning while adaptively adjusting cognitive allocation based on task complexity, achieving efficient hybrid reasoning.
>
---
#### [replaced 090] Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.06346v2](http://arxiv.org/pdf/2501.06346v2)**

> **作者:** Jannik Brinkmann; Chris Wendler; Christian Bartelt; Aaron Mueller
>
> **摘要:** Human bilinguals often use similar brain regions to process multiple languages, depending on when they learned their second language and their proficiency. In large language models (LLMs), how are multiple languages learned and encoded? In this work, we explore the extent to which LLMs share representations of morphsyntactic concepts such as grammatical number, gender, and tense across languages. We train sparse autoencoders on Llama-3-8B and Aya-23-8B, and demonstrate that abstract grammatical concepts are often encoded in feature directions shared across many languages. We use causal interventions to verify the multilingual nature of these representations; specifically, we show that ablating only multilingual features decreases classifier performance to near-chance across languages. We then use these features to precisely modify model behavior in a machine translation task; this demonstrates both the generality and selectivity of these feature's roles in the network. Our findings suggest that even models trained predominantly on English data can develop robust, cross-lingual abstractions of morphosyntactic concepts.
>
---
#### [replaced 091] Mastering Board Games by External and Internal Planning with Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.12119v3](http://arxiv.org/pdf/2412.12119v3)**

> **作者:** John Schultz; Jakub Adamek; Matej Jusup; Marc Lanctot; Michael Kaisers; Sarah Perrin; Daniel Hennes; Jeremy Shar; Cannada Lewis; Anian Ruoss; Tom Zahavy; Petar Veličković; Laurel Prince; Satinder Singh; Eric Malmi; Nenad Tomašev
>
> **备注:** 70 pages, 10 figures
>
> **摘要:** Advancing planning and reasoning capabilities of Large Language Models (LLMs) is one of the key prerequisites towards unlocking their potential for performing reliably in complex and impactful domains. In this paper, we aim to demonstrate this across board games (Chess, Fischer Random / Chess960, Connect Four, and Hex), and we show that search-based planning can yield significant improvements in LLM game-playing strength. We introduce, compare and contrast two major approaches: In external search, the model guides Monte Carlo Tree Search (MCTS) rollouts and evaluations without calls to an external game engine, and in internal search, the model is trained to generate in-context a linearized tree of search and a resulting final choice. Both build on a language model pre-trained on relevant domain knowledge, reliably capturing the transition and value functions in the respective environments, with minimal hallucinations. We evaluate our LLM search implementations against game-specific state-of-the-art engines, showcasing substantial improvements in strength over the base model, and reaching Grandmaster-level performance in chess while operating closer to the human search budget. Our proposed approach, combining search with domain knowledge, is not specific to board games, hinting at more general future applications.
>
---
#### [replaced 092] ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.10808v2](http://arxiv.org/pdf/2405.10808v2)**

> **作者:** Markus Bayer; Justin Lutz; Christian Reuter
>
> **备注:** 20 pages, 10 figures, 7 tables
>
> **摘要:** Active learning is designed to minimize annotation efforts by prioritizing instances that most enhance learning. However, many active learning strategies struggle with a `cold-start' problem, needing substantial initial data to be effective. This limitation reduces their utility in the increasingly relevant few-shot scenarios, where the instance selection has a substantial impact. To address this, we introduce ActiveLLM, a novel active learning approach that leverages Large Language Models such as GPT-4, o1, Llama 3, or Mistral Large for selecting instances. We demonstrate that ActiveLLM significantly enhances the classification performance of BERT classifiers in few-shot scenarios, outperforming traditional active learning methods as well as improving the few-shot learning methods ADAPET, PERFECT, and SetFit. Additionally, ActiveLLM can be extended to non-few-shot scenarios, allowing for iterative selections. In this way, ActiveLLM can even help other active learning strategies to overcome their cold-start problem. Our results suggest that ActiveLLM offers a promising solution for improving model performance across various learning setups.
>
---
#### [replaced 093] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11799v3](http://arxiv.org/pdf/2502.11799v3)**

> **作者:** Peiying Yu; Guoxin Chen; Jingjing Wang
>
> **备注:** ACL 2025 Main
>
> **摘要:** Despite the remarkable capabilities of large language models (LLMs) in various reasoning tasks, they still struggle with table reasoning tasks, particularly in maintaining consistency throughout multi-step reasoning processes. While existing approaches have explored various decomposition strategies, they often lack effective mechanisms to identify and correct errors in intermediate reasoning steps, leading to cascading error propagation. To address these issues, we propose Table-Critic, a novel multi-agent framework that facilitates collaborative criticism and iterative refinement of the reasoning process until convergence to correct solutions. Our framework consists of four specialized agents: a Judge for error identification, a Critic for comprehensive critiques, a Refiner for process improvement, and a Curator for pattern distillation. To effectively deal with diverse and unpredictable error types, we introduce a self-evolving template tree that systematically accumulates critique knowledge through experience-driven learning and guides future reflections. Extensive experiments have demonstrated that Table-Critic achieves substantial improvements over existing methods, achieving superior accuracy and error correction rates while maintaining computational efficiency and lower solution degradation rate.
>
---
#### [replaced 094] Cross-lingual Human-Preference Alignment for Neural Machine Translation with Direct Quality Optimization
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2409.17673v2](http://arxiv.org/pdf/2409.17673v2)**

> **作者:** Kaden Uhlig; Joern Wuebker; Raphael Reinauer; John DeNero
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) and derivative techniques like Direct Preference Optimization (DPO) are task-alignment algorithms used to repurpose general, foundational models for specific tasks. We show that applying task-alignment to neural machine translation (NMT) addresses an existing task--data mismatch in NMT, leading to improvements across all languages of a multilingual model, even when task-alignment is only applied to a subset of those languages. We do so by introducing Direct Quality Optimization (DQO), a variant of DPO leveraging a pre-trained translation quality estimation model as a proxy for human preferences, and verify the improvements with both automatic metrics and human evaluation.
>
---
#### [replaced 095] InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15872v2](http://arxiv.org/pdf/2505.15872v2)**

> **作者:** Yunjia Xi; Jianghao Lin; Menghui Zhu; Yongzhao Xiao; Zhuoying Ou; Jiaqi Liu; Tong Wan; Bo Chen; Weiwen Liu; Yasheng Wang; Ruiming Tang; Weinan Zhang; Yong Yu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding responses with retrieved information. As an emerging paradigm, Agentic RAG further enhances this process by introducing autonomous LLM agents into the information seeking process. However, existing benchmarks fall short in evaluating such systems, as they are confined to a static retrieval environment with a fixed, limited corpus} and simple queries that fail to elicit agentic behavior. Moreover, their evaluation protocols assess information seeking effectiveness by pre-defined gold sets of documents, making them unsuitable for the open-ended and dynamic nature of real-world web environments. To bridge this gap, we present InfoDeepSeek, a new benchmark with challenging questions designed for assessing agentic information seeking in real-world, dynamic web environments. We propose a systematic methodology for constructing challenging queries satisfying the criteria of determinacy, difficulty, and diversity. Based on this, we develop the first evaluation framework tailored to dynamic agentic information seeking, including fine-grained metrics about the accuracy, utility, and compactness of information seeking outcomes. Through extensive experiments across LLMs, search engines, and question types, InfoDeepSeek reveals nuanced agent behaviors and offers actionable insights for future research.
>
---
#### [replaced 096] SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog
- **分类: cs.CL; cs.AI; cs.DL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.07199v3](http://arxiv.org/pdf/2504.07199v3)**

> **作者:** Jennifer D'Souza; Sameer Sadruddin; Holger Israel; Mathias Begoin; Diana Slawig
>
> **备注:** 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
>
> **摘要:** We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
>
---
#### [replaced 097] Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning
- **分类: cs.CR; cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.10440v2](http://arxiv.org/pdf/2502.10440v2)**

> **作者:** Junfeng Guo; Yiming Li; Ruibo Chen; Yihan Wu; Chenxi Liu; Yanshuo Chen; Heng Huang
>
> **备注:** The first two authors contributed equally to this work. 25 pages
>
> **摘要:** Large language models (LLMs) are increasingly integrated into real-world personalized applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning or backdoor attacks. However, these methods require altering the LLM's results of verification samples, inevitably making these watermarks susceptible to anomaly detection and even introducing new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct yet benign verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) Generating CoTs: For each verification question, we generate two `innocent' CoTs, including a target CoT for building watermark behaviors; (2) Optimizing Watermark Phrases and Target CoTs: Inspired by our theoretical analysis, we optimize them to minimize retrieval errors under the \emph{black-box} and \emph{text-only} setting of suspicious LLM, ensuring that only watermarked verification queries can retrieve their correspondingly target CoTs contained in the knowledge base; (3) Ownership Verification: We exploit a pairwise Wilcoxon test to verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases and its resistance to adaptive attacks.
>
---
#### [replaced 098] Position IDs Matter: An Enhanced Position Layout for Efficient Context Compression in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.14364v3](http://arxiv.org/pdf/2409.14364v3)**

> **作者:** Runsong Zhao; Xin Liu; Xinyu Liu; Pengcheng Huang; Chunyang Xiao; Tong Xiao; Jingbo Zhu
>
> **摘要:** Using special tokens (e.g., gist, memory, or compressed tokens) to compress context information is a common practice for large language models (LLMs). However, existing approaches often neglect that position encodings inherently induce local inductive biases in models, causing the compression process to ignore holistic contextual dependencies. We propose Enhanced Position Layout (EPL), a simple yet effective method that improves the context compression capability of LLMs by only adjusting position IDs, the numerical identifiers that specify token positions. EPL minimizes the distance between context tokens and their corresponding special tokens and at the same time maintains the sequence order in position IDs between context tokens, special tokens, and the subsequent tokens. Integrating EPL into our best performing context compression model results in 1.9 ROUGE-1 F1 improvement on out-of-domain question answering datasets in average. When extended to multimodal scenarios, EPL brings an average accuracy gain of 2.6 to vision compression LLMs.
>
---
#### [replaced 099] Enhancing Unsupervised Sentence Embeddings via Knowledge-Driven Data Augmentation and Gaussian-Decayed Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.12887v3](http://arxiv.org/pdf/2409.12887v3)**

> **作者:** Peichao Lai; Zhengfeng Zhang; Wentao Zhang; Fangcheng Fu; Bin Cui
>
> **摘要:** Recently, using large language models (LLMs) for data augmentation has led to considerable improvements in unsupervised sentence embedding models. However, existing methods encounter two primary challenges: limited data diversity and high data noise. Current approaches often neglect fine-grained knowledge, such as entities and quantities, leading to insufficient diversity. Besides, unsupervised data frequently lacks discriminative information, and the generated synthetic samples may introduce noise. In this paper, we propose a pipeline-based data augmentation method via LLMs and introduce the Gaussian-decayed gradient-assisted Contrastive Sentence Embedding (GCSE) model to enhance unsupervised sentence embeddings. To tackle the issue of low data diversity, our pipeline utilizes knowledge graphs (KGs) to extract entities and quantities, enabling LLMs to generate more diverse samples. To address high data noise, the GCSE model uses a Gaussian-decayed function to limit the impact of false hard negative samples, enhancing the model's discriminative capability. Experimental results show that our approach achieves state-of-the-art performance in semantic textual similarity (STS) tasks, using fewer data samples and smaller LLMs, demonstrating its efficiency and robustness across various models.
>
---
#### [replaced 100] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19614v2](http://arxiv.org/pdf/2502.19614v2)**

> **作者:** Sungduk Yu; Man Luo; Avinash Madusu; Vasudev Lal; Phillip Howard
>
> **摘要:** Peer review is a critical process for ensuring the integrity of published scientific research. Confidence in this process is predicated on the assumption that experts in the relevant domain give careful consideration to the merits of manuscripts which are submitted for publication. With the recent rapid advancements in large language models (LLMs), a new risk to the peer review process is that negligent reviewers will rely on LLMs to perform the often time consuming process of reviewing a paper. However, there is a lack of existing resources for benchmarking the detectability of AI text in the domain of peer review. To address this deficiency, we introduce a comprehensive dataset containing a total of 788,984 AI-written peer reviews paired with corresponding human reviews, covering 8 years of papers submitted to each of two leading AI research conferences (ICLR and NeurIPS). We use this new resource to evaluate the ability of 18 existing AI text detection algorithms to distinguish between peer reviews fully written by humans and different state-of-the-art LLMs. Additionally, we explore a context-aware detection method called Anchor, which leverages manuscript content to detect AI-generated reviews, and analyze the sensitivity of detection models to LLM-assisted editing of human-written text. Our work reveals the difficulty of identifying AI-generated text at the individual peer review level, highlighting the urgent need for new tools and methods to detect this unethical use of generative AI. Our dataset is publicly available at: https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark.
>
---
#### [replaced 101] SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15594v2](http://arxiv.org/pdf/2502.15594v2)**

> **作者:** Jiaqi Wu; Chen Chen; Chunyan Hou; Xiaojie Yuan
>
> **摘要:** With the widespread real-world deployment of large language models (LLMs), ensuring their behavior complies with safety standards has become crucial. Jailbreak attacks exploit vulnerabilities in LLMs to induce undesirable behavior, posing a significant threat to LLM safety. Previous defenses often fail to achieve both effectiveness and efficiency simultaneously. Defenses from a representation perspective offer new insights, but existing interventions cannot dynamically adjust representations based on the harmfulness of the queries. To address this limitation, we propose SafeIntervention (SafeInt), a novel defense method that shields LLMs from jailbreak attacks through safety-aware representation intervention. Built on our analysis of the representations of jailbreak samples, the core idea of SafeInt is to relocate jailbreak-related representations into the rejection region. This is achieved by intervening in the representation distributions of jailbreak samples to align them with those of unsafe samples. We conduct comprehensive experiments covering six jailbreak attacks, two jailbreak datasets, and two utility benchmarks. Experimental results demonstrate that SafeInt outperforms all baselines in defending LLMs against jailbreak attacks while largely maintaining utility. Additionally, we evaluate SafeInt against adaptive attacks and verify its effectiveness in mitigating real-time attacks.
>
---
#### [replaced 102] LLäMmlein: Compact and Competitive German-Only Language Models from Scratch
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11171v3](http://arxiv.org/pdf/2411.11171v3)**

> **作者:** Jan Pfister; Julia Wunderle; Andreas Hotho
>
> **备注:** camera ready; https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/llammlein/
>
> **摘要:** We create two German-only decoder models, LL\"aMmlein 120M and 1B, transparently from scratch and publish them, along with the training data, for the German NLP research community to use. The model training involved several key steps, including extensive data preprocessing, the creation of a custom German tokenizer, the training itself, as well as the evaluation of the final models on various benchmarks. Throughout the training process, multiple checkpoints were saved and analyzed using the SuperGLEBer benchmark to monitor the models' learning dynamics. Compared to state-of-the-art models on the SuperGLEBer benchmark, both LL\"aMmlein models performed competitively, consistently matching or surpassing models with similar parameter sizes. The results show that the models' quality scales with size as expected, but performance improvements on some tasks plateaued early, offering valuable insights into resource allocation for future model development.
>
---
#### [replaced 103] Personality Editing for Language Models through Relevant Knowledge Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11789v2](http://arxiv.org/pdf/2502.11789v2)**

> **作者:** Seojin Hwang; Yumin Kim; Byeongjeong Kim; Donghoon Shin; Hwanhee Lee
>
> **备注:** 19 pages, 3 figures, 24 tables
>
> **摘要:** Large Language Models (LLMs) play a vital role in applications like conversational agents and content creation, where controlling a model's personality is crucial for maintaining tone, consistency, and engagement. However, traditional prompt-based techniques for controlling personality often fall short, as they do not effectively mitigate the model's inherent biases. In this paper, we introduce a novel method PALETTE that enhances personality control through knowledge editing. By generating adjustment queries inspired by psychological assessments, our approach systematically adjusts responses to personality-related queries similar to modifying factual knowledge, thereby achieving controlled shifts in personality traits. Experimental results from both automatic and human evaluations demonstrate that our method enables more stable and well-balanced personality control in LLMs.
>
---
#### [replaced 104] SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11274v2](http://arxiv.org/pdf/2505.11274v2)**

> **作者:** Zheng Li; Qingxiu Dong; Jingyuan Ma; Di Zhang; Zhifang Sui
>
> **摘要:** Recently, large reasoning models demonstrate exceptional performance on various tasks. However, reasoning models inefficiently over-process both trivial and complex queries, leading to resource waste and prolonged user latency. To address this challenge, we propose SelfBudgeter - a self-adaptive controllable reasoning strategy for efficient reasoning. Our approach adopts a dual-phase training paradigm: first, the model learns to pre-estimate the reasoning cost based on the difficulty of the query. Then, we introduce budget-guided GPRO for reinforcement learning, which effectively maintains accuracy while reducing output length. SelfBudgeter allows users to anticipate generation time and make informed decisions about continuing or interrupting the process. Furthermore, our method enables direct manipulation of reasoning length via pre-filling token budget. Experimental results demonstrate that SelfBudgeter can rationally allocate budgets according to problem complexity, achieving up to 74.47% response length compression on the MATH benchmark while maintaining nearly undiminished accuracy.
>
---
#### [replaced 105] Cognitive Debiasing Large Language Models for Decision-Making
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04141v3](http://arxiv.org/pdf/2504.04141v3)**

> **作者:** Yougang Lyu; Shijie Ren; Yue Feng; Zihan Wang; Zhumin Chen; Zhaochun Ren; Maarten de Rijke
>
> **摘要:** Large language models (LLMs) have shown potential in supporting decision-making applications, particularly as personal assistants in the financial, healthcare, and legal domains. While prompt engineering strategies have enhanced the capabilities of LLMs in decision-making, cognitive biases inherent to LLMs present significant challenges. Cognitive biases are systematic patterns of deviation from norms or rationality in decision-making that can lead to the production of inaccurate outputs. Existing cognitive bias mitigation strategies assume that input prompts only contain one type of cognitive bias, limiting their effectiveness in more challenging scenarios involving multiple cognitive biases. To fill this gap, we propose a cognitive debiasing approach, self-adaptive cognitive debiasing (SACD), that enhances the reliability of LLMs by iteratively refining prompts. Our method follows three sequential steps -- bias determination, bias analysis, and cognitive debiasing -- to iteratively mitigate potential cognitive biases in prompts. Experimental results on finance, healthcare, and legal decision-making tasks, using both closed-source and open-source LLMs, demonstrate that the proposed SACD method outperforms both advanced prompt engineering methods and existing cognitive debiasing techniques in average accuracy under single-bias and multi-bias settings.
>
---
#### [replaced 106] Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.16975v2](http://arxiv.org/pdf/2501.16975v2)**

> **作者:** Hongzhi Huang; Defa Zhu; Banggu Wu; Yutao Zeng; Ya Wang; Qiyang Min; Xun Zhou
>
> **备注:** accepted by ICML2025
>
> **摘要:** Tokenization is a fundamental component of large language models (LLMs), yet its influence on model scaling and performance is not fully explored. In this paper, we introduce Over-Tokenized Transformers, a novel framework that decouples input and output vocabularies to improve language modeling performance. Specifically, our approach scales up input vocabularies to leverage multi-gram tokens. Through extensive experiments, we uncover a log-linear relationship between input vocabulary size and training loss, demonstrating that larger input vocabularies consistently enhance model performance, regardless of model size. Using a large input vocabulary, we achieve performance comparable to double-sized baselines with no additional cost. Our findings highlight the importance of tokenization in scaling laws and provide practical insight for tokenizer design, paving the way for more efficient and powerful LLMs.
>
---
#### [replaced 107] An Annotated Corpus of Arabic Tweets for Hate Speech Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11969v2](http://arxiv.org/pdf/2505.11969v2)**

> **作者:** Wajdi Zaghouani; Md. Rafiul Biswas
>
> **摘要:** Identifying hate speech content in the Arabic language is challenging due to the rich quality of dialectal variations. This study introduces a multilabel hate speech dataset in the Arabic language. We have collected 10000 Arabic tweets and annotated each tweet, whether it contains offensive content or not. If a text contains offensive content, we further classify it into different hate speech targets such as religion, gender, politics, ethnicity, origin, and others. A text can contain either single or multiple targets. Multiple annotators are involved in the data annotation task. We calculated the inter-annotator agreement, which was reported to be 0.86 for offensive content and 0.71 for multiple hate speech targets. Finally, we evaluated the data annotation task by employing a different transformers-based model in which AraBERTv2 outperformed with a micro-F1 score of 0.7865 and an accuracy of 0.786.
>
---
#### [replaced 108] Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20641v2](http://arxiv.org/pdf/2503.20641v2)**

> **作者:** Han Wu; Yuxuan Yao; Shuqi Liu; Zehua Liu; Xiaojin Fu; Xiongwei Han; Xing Li; Hui-Ling Zhen; Tao Zhong; Mingxuan Yuan
>
> **备注:** Technical report
>
> **摘要:** The transition from System 1 to System 2 reasoning in large language models (LLMs) has marked significant advancements in handling complex tasks through deliberate, iterative thinking. However, this progress often comes at the cost of efficiency, as models tend to overthink, generating redundant reasoning steps without proportional improvements in output quality. Long-to-Short (L2S) reasoning has emerged as a promising solution to this challenge, aiming to balance reasoning depth with practical efficiency. While existing approaches, such as supervised fine-tuning (SFT), reinforcement learning (RL), and prompt engineering, have shown potential, they are either computationally expensive or unstable. Model merging, on the other hand, offers a cost-effective and robust alternative by integrating the quick-thinking capabilities of System 1 models with the methodical reasoning of System 2 models. In this work, we present a comprehensive empirical study on model merging for L2S reasoning, exploring diverse methodologies, including task-vector-based, SVD-based, and activation-informed merging. Our experiments reveal that model merging can reduce average response length by up to 55% while preserving or even improving baseline performance. We also identify a strong correlation between model scale and merging efficacy with extensive evaluations on 1.5B/7B/14B/32B models. Furthermore, we investigate the merged model's ability to self-critique and self-correct, as well as its adaptive response length based on task complexity. Our findings highlight model merging as a highly efficient and effective paradigm for L2S reasoning, offering a practical solution to the overthinking problem while maintaining the robustness of System 2 reasoning. This work can be found on Github https://github.com/hahahawu/Long-to-Short-via-Model-Merging.
>
---
#### [replaced 109] Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.00869v3](http://arxiv.org/pdf/2407.00869v3)**

> **作者:** Yue Zhou; Henry Peng Zou; Barbara Di Eugenio; Yang Zhang
>
> **备注:** Accepted to the main conference of EMNLP 2024
>
> **摘要:** We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.
>
---
#### [replaced 110] Boosting Long-Context Management via Query-Guided Activation Refilling
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.12486v3](http://arxiv.org/pdf/2412.12486v3)**

> **作者:** Hongjin Qian; Zheng Liu; Peitian Zhang; Zhicheng Dou; Defu Lian
>
> **备注:** ACL25 Main Conference
>
> **摘要:** Processing long contexts poses a significant challenge for large language models (LLMs) due to their inherent context-window limitations and the computational burden of extensive key-value (KV) activations, which severely impact efficiency. For information-seeking tasks, full context perception is often unnecessary, as a query's information needs can dynamically range from localized details to a global perspective, depending on its complexity. However, existing methods struggle to adapt effectively to these dynamic information needs. In the paper, we propose a method for processing long-context information-seeking tasks via query-guided Activation Refilling (ACRE). ACRE constructs a Bi-layer KV Cache for long contexts, where the layer-1 (L1) cache compactly captures global information, and the layer-2 (L2) cache provides detailed and localized information. ACRE establishes a proxying relationship between the two caches, allowing the input query to attend to the L1 cache and dynamically refill it with relevant entries from the L2 cache. This mechanism integrates global understanding with query-specific local details, thus improving answer decoding. Experiments on a variety of long-context information-seeking datasets demonstrate ACRE's effectiveness, achieving improvements in both performance and efficiency.
>
---
#### [replaced 111] CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.0**

- **链接: [http://arxiv.org/pdf/2502.08923v2](http://arxiv.org/pdf/2502.08923v2)**

> **作者:** Razvan-Gabriel Dumitru; Minglai Yang; Vikas Yadav; Mihai Surdeanu
>
> **备注:** 33 pages, 18 figures, 19 tables
>
> **摘要:** We introduce CopySpec, a simple yet effective technique to tackle the inefficiencies LLMs face when generating responses that closely resemble previous outputs or responses that can be verbatim extracted from context. CopySpec identifies repeated sequences in the model's chat history or context and speculates that the same tokens will follow, enabling seamless copying without compromising output quality and without requiring additional GPU memory. To evaluate the effectiveness of our approach, we conducted experiments using seven LLMs and five datasets: MT-Bench, CNN/DM, GSM8K, HumanEval, and our newly created dataset, MT-Redundant. MT-Redundant, introduced in this paper, transforms the second turn of MT-Bench into a request for variations of the first turn's answer, simulating real-world scenarios where users request modifications to prior responses. Our results demonstrate significant speed-ups: up to 2.35x on CNN/DM, 3.08x on the second turn of select MT-Redundant categories, and 2.66x on the third turn of GSM8K's self-correction tasks. Importantly, we show that CopySpec integrates seamlessly with speculative decoding, yielding an average 49% additional speed-up over speculative decoding for the second turn of MT-Redundant across all eight categories. While LLMs, even with speculative decoding, suffer from slower inference as context size grows, CopySpec leverages larger contexts to accelerate inference, making it a faster complementary solution. Our code and dataset are publicly available at https://github.com/RazvanDu/CopySpec.
>
---
#### [replaced 112] SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19306v3](http://arxiv.org/pdf/2501.19306v3)**

> **作者:** Jiefeng Chen; Jie Ren; Xinyun Chen; Chengrun Yang; Ruoxi Sun; Jinsung Yoon; Sercan Ö Arık
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have created new opportunities to enhance performance on complex reasoning tasks by leveraging test-time computation. However, existing parallel scaling methods, such as repeated sampling or reward model scoring, often suffer from premature convergence and high costs due to task-specific reward model training, while sequential methods like SELF-REFINE cannot effectively leverage increased compute. This paper introduces Self-Enhanced Test-Time Scaling (SETS), a new approach that overcomes these limitations by strategically combining parallel and sequential techniques. SETS exploits the inherent self-verification and self-correction capabilities of LLMs, unifying sampling, verification, and correction within a single framework. This innovative design facilitates efficient and scalable test-time computation for enhanced performance on complex tasks. Our comprehensive experimental results on challenging benchmarks spanning planning, reasoning, math, and coding demonstrate that SETS achieves significant performance improvements and more advantageous test-time scaling behavior than the alternatives.
>
---
#### [replaced 113] MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13610v3](http://arxiv.org/pdf/2410.13610v3)**

> **作者:** Yakun Zhu; Shaohang Wei; Xu Wang; Kui Xue; Xiaofan Zhang; Shaoting Zhang
>
> **备注:** NAACL 2025 main conference. Code and Dataset available at [https://github.com/shzyk/MENTI](https://github.com/shzyk/MENTI)
>
> **摘要:** Integrating tools into Large Language Models (LLMs) has facilitated the widespread application. Despite this, in specialized downstream task contexts, reliance solely on tools is insufficient to fully address the complexities of the real world. This particularly restricts the effective deployment of LLMs in fields such as medicine. In this paper, we focus on the downstream tasks of medical calculators, which use standardized tests to assess an individual's health status. We introduce MeNTi, a universal agent architecture for LLMs. MeNTi integrates a specialized medical toolkit and employs meta-tool and nested calling mechanisms to enhance LLM tool utilization. Specifically, it achieves flexible tool selection and nested tool calling to address practical issues faced in intricate medical scenarios, including calculator selection, slot filling, and unit conversion. To assess the capabilities of LLMs for quantitative assessment throughout the clinical process of calculator scenarios, we introduce CalcQA. This benchmark requires LLMs to use medical calculators to perform calculations and assess patient health status. CalcQA is constructed by professional physicians and includes 100 case-calculator pairs, complemented by a toolkit of 281 medical tools. The experimental results demonstrate significant performance improvements with our framework. This research paves new directions for applying LLMs in demanding scenarios of medicine.
>
---
#### [replaced 114] Hogwild! Inference: Parallel LLM Generation via Concurrent Attention
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06261v3](http://arxiv.org/pdf/2504.06261v3)**

> **作者:** Gleb Rodionov; Roman Garipov; Alina Shutova; George Yakushev; Erik Schultheis; Vage Egiazarian; Anton Sinitsin; Denis Kuznedelev; Dan Alistarh
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the LLM instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's memory in the concurrent KV cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's memory. Hogwild! Inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
>
---
#### [replaced 115] KCIF: Knowledge-Conditioned Instruction Following
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12972v3](http://arxiv.org/pdf/2410.12972v3)**

> **作者:** Rudra Murthy; Praveen Venkateswaran; Prince Kumar; Danish Contractor
>
> **备注:** Under Review
>
> **摘要:** LLM evaluation benchmarks have traditionally separated the testing of knowledge/reasoning capabilities from instruction following. In this work, we study the interaction between knowledge and instruction following, and observe that LLMs struggle to follow simple answer modifying instructions, and are also distracted by instructions that should have no bearing on the original knowledge task answer. We leverage existing multiple-choice answer based knowledge benchmarks and apply a set of simple instructions which include manipulating text (eg.: change case), numeric quantities (eg.: increase value, change formatting), operate on lists (eg.: sort answer candidates) and distractor instructions (eg.: change case of numeric answers). We evaluate models at varying parameter sizes (1B-405B) from different model families and find that, surprisingly, all models report a significant drop in performance on such simple task compositions. While large-sized and frontier models report performance drops of 40-50%, in small and medium sized models the drop is severe (sometimes exceeding 80%). Our results highlight a limitation in the traditional separation of knowledge/reasoning and instruction following, and suggest that joint-study of these capabilities are important. We release our benchmark dataset, evaluation framework code, and results for future work.
>
---
#### [replaced 116] QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.07147v2](http://arxiv.org/pdf/2310.07147v2)**

> **作者:** Zhikai Li; Xiaoxuan Liu; Banghua Zhu; Zhen Dong; Qingyi Gu; Kurt Keutzer
>
> **摘要:** Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pretrained models on downstream datasets provides further significant performance gains; however, this process typically requires a large number of expensive, high-end GPUs. Although there have been efforts focused on parameter-efficient fine-tuning, they cannot fully unlock the powerful potential of full-parameter fine-tuning. In this paper, we propose QFT, a Quantized Full-parameter Tuning framework for LLMs that quantizes and stores all training states, including weights, gradients, and optimizer states, in INT8 format to reduce training memory, thereby enabling full-parameter fine-tuning on existing GPUs at an affordable cost. To ensure training performance, we make two key efforts: i) for quantized gradients and optimizer states, we theoretically prove that the Lion optimizer, with its property of consistent update magnitudes, is highly robust to quantization; ii) and for quantized weights, we employ the hybrid feature quantizer, which identifies and protects a small subset of sparse critical features while quantizing the remaining dense features, thus ensuring accurate weight updates without FP32 backups. Moreover, to support backpropagation in the integer context, we develop a stack-based gradient flow scheme with O(1) complexity, forming a unified integer training pipeline. As a result, QFT reduces the model state memory to 21% of the standard solution while achieving comparable performance, e.g., tuning a LLaMA-7B model requires only <30GB of memory, making it feasible on a single A6000 GPU.
>
---
#### [replaced 117] Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19557v3](http://arxiv.org/pdf/2411.19557v3)**

> **作者:** Kaustubh Ponkshe; Raghav Singhal; Eduard Gorbunov; Alexey Tumanov; Samuel Horvath; Praneeth Vepakomma
>
> **备注:** Kaustubh Ponkshe and Raghav Singhal contributed equally to this work
>
> **摘要:** Low-rank adapters have become standard for efficiently fine-tuning large language models (LLMs), but they often fall short of achieving the performance of full fine-tuning. We propose a method, LoRA Silver Bullet or LoRA-SB, that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy. We theoretically demonstrate that the architecture of LoRA-XS, which inserts a learnable (r x r) matrix between B and A while keeping other matrices fixed, provides the precise conditions needed for this approximation. We leverage its constrained update space to achieve optimal scaling for high-rank gradient updates while removing the need for hyperparameter tuning. We prove that our initialization offers an optimal low-rank approximation of the initial gradient and preserves update directions throughout training. Extensive experiments across mathematical reasoning, commonsense reasoning, and language understanding tasks demonstrate that our approach exceeds the performance of standard LoRA while using \textbf{27-90} times fewer learnable parameters, and comprehensively outperforms LoRA-XS. Our findings establish that it is possible to simulate full fine-tuning in low-rank subspaces, and achieve significant efficiency gains without sacrificing performance. Our code is publicly available at https://github.com/RaghavSinghal10/lora-sb.
>
---

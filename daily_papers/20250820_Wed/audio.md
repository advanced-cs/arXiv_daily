# 音频 cs.SD;  eess.SP

- **最新发布 6 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Evaluating Identity Leakage in Speaker De-Identification Systems
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.14012v1](http://arxiv.org/pdf/2508.14012v1)**

> **作者:** Seungmin Seo; Oleg Aulov; Afzal Godil; Kevin Mangold
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speaker de-identification aims to conceal a speaker's identity while preserving intelligibility of the underlying speech. We introduce a benchmark that quantifies residual identity leakage with three complementary error rates: equal error rate, cumulative match characteristic hit rate, and embedding-space similarity measured via canonical correlation analysis and Procrustes analysis. Evaluation results reveal that all state-of-the-art speaker de-identification systems leak identity information. The highest performing system in our evaluation performs only slightly better than random guessing, while the lowest performing system achieves a 45% hit rate within the top 50 candidates based on CMC. These findings highlight persistent privacy risks in current speaker de-identification technologies.
>
---
#### [new 002] DegDiT: Controllable Audio Generation with Dynamic Event Graph Guided Diffusion Transformer
- **分类: cs.SD; cs.AI**

- **简介: 论文提出DegDiT框架，解决文本到音频生成中时间定位精度、词汇开放性和效率的平衡问题。通过动态事件图引导扩散Transformer，实现可控音频生成，并引入数据筛选与偏好优化提升性能。**

- **链接: [http://arxiv.org/pdf/2508.13786v1](http://arxiv.org/pdf/2508.13786v1)**

> **作者:** Yisu Liu; Chenxing Li; Wanqian Zhang; Wenfu Wang; Meng Yu; Ruibo Fu; Zheng Lin; Weiping Wang; Dong Yu
>
> **摘要:** Controllable text-to-audio generation aims to synthesize audio from textual descriptions while satisfying user-specified constraints, including event types, temporal sequences, and onset and offset timestamps. This enables precise control over both the content and temporal structure of the generated audio. Despite recent progress, existing methods still face inherent trade-offs among accurate temporal localization, open-vocabulary scalability, and practical efficiency. To address these challenges, we propose DegDiT, a novel dynamic event graph-guided diffusion transformer framework for open-vocabulary controllable audio generation. DegDiT encodes the events in the description as structured dynamic graphs. The nodes in each graph are designed to represent three aspects: semantic features, temporal attributes, and inter-event connections. A graph transformer is employed to integrate these nodes and produce contextualized event embeddings that serve as guidance for the diffusion model. To ensure high-quality and diverse training data, we introduce a quality-balanced data selection pipeline that combines hierarchical event annotation with multi-criteria quality scoring, resulting in a curated dataset with semantic diversity. Furthermore, we present consensus preference optimization, facilitating audio generation through consensus among multiple reward signals. Extensive experiments on AudioCondition, DESED, and AudioTime datasets demonstrate that DegDiT achieves state-of-the-art performances across a variety of objective and subjective evaluation metrics.
>
---
#### [new 003] Is Transfer Learning Necessary for Violin Transcription?
- **分类: cs.SD; eess.AS**

- **简介: 论文研究弦乐自动转录任务，针对小样本数据难题，比较从头训练与迁移学习的效果。结果表明，基于30小时弦乐数据从头训练的模型性能优于迁移钢琴预训练模型，说明高质量乐器特定数据比预训练更关键。**

- **链接: [http://arxiv.org/pdf/2508.13516v1](http://arxiv.org/pdf/2508.13516v1)**

> **作者:** Yueh-Po Peng; Ting-Kang Wang; Li Su; Vincent K. M. Cheung
>
> **摘要:** Automatic music transcription (AMT) has achieved remarkable progress for instruments such as the piano, largely due to the availability of large-scale, high-quality datasets. In contrast, violin AMT remains underexplored due to limited annotated data. A common approach is to fine-tune pretrained models for other downstream tasks, but the effectiveness of such transfer remains unclear in the presence of timbral and articulatory differences. In this work, we investigate whether training from scratch on a medium-scale violin dataset can match the performance of fine-tuned piano-pretrained models. We adopt a piano transcription architecture without modification and train it on the MOSA dataset, which contains about 30 hours of aligned violin recordings. Our experiments on URMP and Bach10 show that models trained from scratch achieved competitive or even superior performance compared to fine-tuned counterparts. These findings suggest that strong violin AMT is possible without relying on pretrained piano representations, highlighting the importance of instrument-specific data collection and augmentation strategies.
>
---
#### [new 004] Leveraging Mamba with Full-Face Vision for Audio-Visual Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.13624v1](http://arxiv.org/pdf/2508.13624v1)**

> **作者:** Rong Chao; Wenze Ren; You-Jin Li; Kuo-Hsuan Hung; Sung-Feng Huang; Szu-Wei Fu; Wen-Huang Cheng; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025 Workshop
>
> **摘要:** Recent Mamba-based models have shown promise in speech enhancement by efficiently modeling long-range temporal dependencies. However, models like Speech Enhancement Mamba (SEMamba) remain limited to single-speaker scenarios and struggle in complex multi-speaker environments such as the cocktail party problem. To overcome this, we introduce AVSEMamba, an audio-visual speech enhancement model that integrates full-face visual cues with a Mamba-based temporal backbone. By leveraging spatiotemporal visual information, AVSEMamba enables more accurate extraction of target speech in challenging conditions. Evaluated on the AVSEC-4 Challenge development and blind test sets, AVSEMamba outperforms other monaural baselines in speech intelligibility (STOI), perceptual quality (PESQ), and non-intrusive quality (UTMOS), and achieves \textbf{1st place} on the monaural leaderboard.
>
---
#### [new 005] MMAU-Pro: A Challenging and Comprehensive Benchmark for Holistic Evaluation of Audio General Intelligence
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MMAU-Pro基准，用于全面评估AI模型的音频通用智能。针对现有评测不足，构建包含5305个实例的多模态数据集，涵盖49项技能与复杂推理任务，揭示当前模型在音频理解上的显著局限。**

- **链接: [http://arxiv.org/pdf/2508.13992v1](http://arxiv.org/pdf/2508.13992v1)**

> **作者:** Sonal Kumar; Šimon Sedláček; Vaibhavi Lokegaonkar; Fernando López; Wenyi Yu; Nishit Anand; Hyeonggon Ryu; Lichang Chen; Maxim Plička; Miroslav Hlaváček; William Fineas Ellingwood; Sathvik Udupa; Siyuan Hou; Allison Ferner; Sara Barahona; Cecilia Bolaños; Satish Rahi; Laura Herrera-Alarcón; Satvik Dixit; Siddhi Patil; Soham Deshmukh; Lasha Koroshinadze; Yao Liu; Leibny Paola Garcia Perera; Eleni Zanou; Themos Stafylakis; Joon Son Chung; David Harwath; Chao Zhang; Dinesh Manocha; Alicia Lozano-Diez; Santosh Kesiraju; Sreyan Ghosh; Ramani Duraiswami
>
> **摘要:** Audio comprehension-including speech, non-speech sounds, and music-is essential for achieving human-level intelligence. Consequently, AI agents must demonstrate holistic audio understanding to qualify as generally intelligent. However, evaluating auditory intelligence comprehensively remains challenging. To address this gap, we introduce MMAU-Pro, the most comprehensive and rigorously curated benchmark for assessing audio intelligence in AI systems. MMAU-Pro contains 5,305 instances, where each instance has one or more audios paired with human expert-generated question-answer pairs, spanning speech, sound, music, and their combinations. Unlike existing benchmarks, MMAU-Pro evaluates auditory intelligence across 49 unique skills and multiple complex dimensions, including long-form audio comprehension, spatial audio reasoning, multi-audio understanding, among others. All questions are meticulously designed to require deliberate multi-hop reasoning, including both multiple-choice and open-ended response formats. Importantly, audio data is sourced directly ``from the wild" rather than from existing datasets with known distributions. We evaluate 22 leading open-source and proprietary multimodal AI models, revealing significant limitations: even state-of-the-art models such as Gemini 2.5 Flash and Audio Flamingo 3 achieve only 59.2% and 51.7% accuracy, respectively, approaching random performance in multiple categories. Our extensive analysis highlights specific shortcomings and provides novel insights, offering actionable perspectives for the community to enhance future AI systems' progression toward audio general intelligence. The benchmark and code is available at https://sonalkum.github.io/mmau-pro.
>
---
#### [new 006] End-to-End Audio-Visual Learning for Cochlear Implant Sound Coding in Noisy Environments
- **分类: eess.AS; cs.AI; cs.SD; eess.IV**

- **简介: 论文提出AVSE-ECS系统，通过端到端深度学习融合音频-视觉语音增强模块，提升耳蜗植入器在噪声环境下的语音编码性能，解决嘈杂环境中言语理解困难的问题。**

- **链接: [http://arxiv.org/pdf/2508.13576v1](http://arxiv.org/pdf/2508.13576v1)**

> **作者:** Meng-Ping Lin; Enoch Hsin-Ho Huang; Shao-Yi Chien; Yu Tsao
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The cochlear implant (CI) is a remarkable biomedical device that successfully enables individuals with severe-to-profound hearing loss to perceive sound by converting speech into electrical stimulation signals. Despite advancements in the performance of recent CI systems, speech comprehension in noisy or reverberant conditions remains a challenge. Recent and ongoing developments in deep learning reveal promising opportunities for enhancing CI sound coding capabilities, not only through replicating traditional signal processing methods with neural networks, but also through integrating visual cues as auxiliary data for multimodal speech processing. Therefore, this paper introduces a novel noise-suppressing CI system, AVSE-ECS, which utilizes an audio-visual speech enhancement (AVSE) model as a pre-processing module for the deep-learning-based ElectrodeNet-CS (ECS) sound coding strategy. Specifically, a joint training approach is applied to model AVSE-ECS, an end-to-end CI system. Experimental results indicate that the proposed method outperforms the previous ECS strategy in noisy conditions, with improved objective speech intelligibility scores. The methods and findings in this study demonstrate the feasibility and potential of using deep learning to integrate the AVSE module into an end-to-end CI system
>
---
## 更新

#### [replaced 001] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12332v4](http://arxiv.org/pdf/2505.12332v4)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.
>
---
#### [replaced 002] Multi-Sampling-Frequency Naturalness MOS Prediction Using Self-Supervised Learning Model with Sampling-Frequency-Independent Layer
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.14647v2](http://arxiv.org/pdf/2507.14647v2)**

> **作者:** Go Nishikawa; Wataru Nakata; Yuki Saito; Kanami Imamura; Hiroshi Saruwatari; Tomohiko Nakamura
>
> **备注:** 4 pages, 2 figures; Accepted to ASRU 2025 Challenge track
>
> **摘要:** We introduce our submission to the AudioMOS Challenge (AMC) 2025 Track 3: mean opinion score (MOS) prediction for speech with multiple sampling frequencies (SFs). Our submitted model integrates an SF-independent (SFI) convolutional layer into a self-supervised learning (SSL) model to achieve SFI speech feature extraction for MOS prediction. We present some strategies to improve the MOS prediction performance of our model: distilling knowledge from a pretrained non-SFI-SSL model and pretraining with a large-scale MOS dataset. Our submission to the AMC 2025 Track 3 ranked the first in one evaluation metric and the fourth in the final ranking. We also report the results of our ablation study to investigate essential factors of our model.
>
---
#### [replaced 003] Less is More: Data Curation Matters in Scaling Speech Enhancement
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.23859v2](http://arxiv.org/pdf/2506.23859v2)**

> **作者:** Chenda Li; Wangyou Zhang; Wei Wang; Robin Scheibler; Kohei Saijo; Samuele Cornell; Yihui Fu; Marvin Sach; Zhaoheng Ni; Anurag Kumar; Tim Fingscheidt; Shinji Watanabe; Yanmin Qian
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** The vast majority of modern speech enhancement systems rely on data-driven neural network models. Conventionally, larger datasets are presumed to yield superior model performance, an observation empirically validated across numerous tasks in other domains. However, recent studies reveal diminishing returns when scaling speech enhancement data. We focus on a critical factor: prevalent quality issues in ``clean'' training labels within large-scale datasets. This work re-examines this phenomenon and demonstrates that, within large-scale training sets, prioritizing high-quality training data is more important than merely expanding the data volume. Experimental findings suggest that models trained on a carefully curated subset of 700 hours can outperform models trained on the 2,500-hour full dataset. This outcome highlights the crucial role of data curation in scaling speech enhancement systems effectively.
>
---
#### [replaced 004] Adaptation and Optimization of Automatic Speech Recognition (ASR) for the Maritime Domain in the Field of VHF Communication
- **分类: cs.SD; cs.AI; cs.HC; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2306.00614v2](http://arxiv.org/pdf/2306.00614v2)**

> **作者:** Emin Cagatay Nakilcioglu; Maximilian Reimann; Ole John
>
> **摘要:** This paper introduces a multilingual automatic speech recognizer (ASR) for maritime radio communi-cation that automatically converts received VHF radio signals into text. The challenges of maritime radio communication are described at first, and the deep learning architecture of marFM consisting of audio processing techniques and machine learning algorithms is presented. Subsequently, maritime radio data of interest is analyzed and then used to evaluate the transcription performance of our ASR model for various maritime radio data.
>
---
#### [replaced 005] What Matters for Bioacoustic Encoding
- **分类: cs.SD; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11845v2](http://arxiv.org/pdf/2508.11845v2)**

> **作者:** Marius Miron; David Robinson; Milad Alizadeh; Ellen Gilsenan-McMahon; Gagan Narula; Emmanuel Chemla; Maddie Cusimano; Felix Effenberger; Masato Hagiwara; Benjamin Hoffman; Sara Keen; Diane Kim; Jane Lawton; Jen-Yu Liu; Aza Raskin; Olivier Pietquin; Matthieu Geist
>
> **摘要:** Bioacoustics, the study of sounds produced by living organisms, plays a vital role in conservation, biodiversity monitoring, and behavioral studies. Many tasks in this field, such as species, individual, and behavior classification and detection, are well-suited to machine learning. However, they often suffer from limited annotated data, highlighting the need for a general-purpose bioacoustic encoder capable of extracting useful representations for diverse downstream tasks. Such encoders have been proposed before, but are often limited in scope due to a focus on a narrow range of species (typically birds), and a reliance on a single model architecture or training paradigm. Moreover, they are usually evaluated on a small set of tasks and datasets. In this work, we present a large-scale empirical study that covers aspects of bioacoustics that are relevant to research but have previously been scarcely considered: training data diversity and scale, model architectures and training recipes, and the breadth of evaluation tasks and datasets. We obtain encoders that are state-of-the-art on the existing and proposed benchmarks. We also identify what matters for training these encoders, such that this work can be extended when more data are available or better architectures are proposed. Specifically, across 26 datasets with tasks including species classification, detection, individual ID, and vocal repertoire discovery, we find self-supervised pre-training followed by supervised post-training on a mixed bioacoustics + general-audio corpus yields the strongest in- and out-of-distribution performance. We show the importance of data diversity in both stages. To support ongoing research and application, we will release the model checkpoints.
>
---
#### [replaced 006] Can Masked Autoencoders Also Listen to Birds?
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.12880v4](http://arxiv.org/pdf/2504.12880v4)**

> **作者:** Lukas Rauch; René Heinrich; Ilyass Moummad; Alexis Joly; Bernhard Sick; Christoph Scholz
>
> **备注:** accepted @TMLR: https://openreview.net/forum?id=GIBWR0Xo2J
>
> **摘要:** Masked Autoencoders (MAEs) learn rich semantic representations in audio classification through an efficient self-supervised reconstruction task. However, general-purpose models fail to generalize well when applied directly to fine-grained audio domains. Specifically, bird-sound classification requires distinguishing subtle inter-species differences and managing high intra-species acoustic variability, revealing the performance limitations of general-domain Audio-MAEs. This work demonstrates that bridging this domain gap domain gap requires full-pipeline adaptation, not just domain-specific pretraining data. We systematically revisit and adapt the pretraining recipe, fine-tuning methods, and frozen feature utilization to bird sounds using BirdSet, a large-scale bioacoustic dataset comparable to AudioSet. Our resulting Bird-MAE achieves new state-of-the-art results in BirdSet's multi-label classification benchmark. Additionally, we introduce the parameter-efficient prototypical probing, enhancing the utility of frozen MAE representations and closely approaching fine-tuning performance in low-resource settings. Bird-MAE's prototypical probes outperform linear probing by up to 37 percentage points in mean average precision and narrow the gap to fine-tuning across BirdSet downstream tasks. Bird-MAE also demonstrates robust few-shot capabilities with prototypical probing in our newly established few-shot benchmark on BirdSet, highlighting the potential of tailored self-supervised learning pipelines for fine-grained audio domains.
>
---
#### [replaced 007] AxLSTMs: learning self-supervised audio representations with xLSTMs
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.16568v4](http://arxiv.org/pdf/2408.16568v4)**

> **作者:** Sarthak Yadav; Sergios Theodoridis; Zheng-Hua Tan
>
> **备注:** INTERSPEECH 2025
>
> **摘要:** While the transformer has emerged as the eminent neural architecture, several independent lines of research have emerged to address its limitations. Recurrent neural approaches have observed a lot of renewed interest, including the extended long short-term memory (xLSTM) architecture, which reinvigorates the original LSTM. However, while xLSTMs have shown competitive performance compared to the transformer, their viability for learning self-supervised general-purpose audio representations has not been evaluated. This work proposes Audio xLSTM (AxLSTM), an approach for learning audio representations from masked spectrogram patches in a self-supervised setting. Pretrained on the AudioSet dataset, the proposed AxLSTM models outperform comparable self-supervised audio spectrogram transformer (SSAST) baselines by up to 25% in relative performance across a set of ten diverse downstream tasks while having up to 45% fewer parameters.
>
---
#### [replaced 008] Speech Enhancement based on cascaded two flows
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.06842v3](http://arxiv.org/pdf/2508.06842v3)**

> **作者:** Seonggyu Lee; Sein Cheong; Sangwook Han; Kihyuk Kim; Jong Won Shin
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech enhancement (SE) based on diffusion probabilistic models has exhibited impressive performance, while requiring a relatively high number of function evaluations (NFE). Recently, SE based on flow matching has been proposed, which showed competitive performance with a small NFE. Early approaches adopted the noisy speech as the only conditioning variable. There have been other approaches which utilize speech enhanced with a predictive model as another conditioning variable and to sample an initial value, but they require a separate predictive model on top of the generative SE model. In this work, we propose to employ an identical model based on flow matching for both SE and generating enhanced speech used as an initial starting point and a conditioning variable. Experimental results showed that the proposed method required the same or fewer NFEs even with two cascaded generative methods while achieving equivalent or better performances to the previous baselines.
>
---

# Glow: Generative Flow with Invertible 1×1 Convolutions

* Diederik P. Kingma*, Prafulla Dhariwal
OpenAI, San Francisco Equal contribution.
Submitted on 9 July 2018

## Abstract

フローベースの生成モデル（Dinh et al。、2014）は、正確な対数尤度の正確性、正確な潜在変数推論の取り扱いやすさ、および訓練と合成の両方の並列性のために概念的に魅力的である。

 In this paper we propose Glow, a simple type of generative flow using an invertible 1×1 convolution. Using our method we demonstrate a significant improvement in log-likelihood on standard benchmarks. Perhaps most strikingly, we demonstrate that a generative model optimized towards the plain log-likelihood objective is capable of efficient realistic-looking synthesis and manipulation of large images.

 本論文では、可逆な1×1畳み込みを用いた単純なタイプの生成フローであるGlowを提案する。 我々の方法を使用して、標準的なベンチマークでの対数尤度の大幅な改善を実証する。 おそらく最も衝撃的なことになると思いますが、我々は、平易な対数尤度目的として最適化された生成モデルが、効率的な現実的な合成と大規模画像の操作が可能であることを実証する。
 

  The code for our model is available at
  
  私たちのモデルは以下から利用可能である。
   https://github.com/openai/glow.

![画像](https://arxiv-sanity-sanity-production.s3.amazonaws.com/render-output/387521/figures/selected_samples/composite.png)




Two major unsolved problems in the field of machine learning are (1) data-efficiency: the ability to learn from few datapoints, like humans; and (2) generalization: robustness to changes of the task or its context. 

機械学習の分野における2つの主な未解決の問題は以下の２点である。（1）データ効率：人間のような少数のデータポイントから学ぶ能力; （2）生成：タスクまたはそのコンテキストの変更に対する堅牢性。

AI systems, for example, often do not work at all when given inputs that are different from their training distribution. 

例えば、AIシステムは、訓練分布とは異なる入力が与えられた場合には、しばしば全く機能しない。

A promise of generative models, a major branch of machine learning, is to overcome these limitations by: (1) learning realistic world models, potentially allowing agents to plan in a world model before actual interaction with the world, and (2) learning meaningful features of the input while requiring little or no human supervision or labeling. 

機械学習の主要な流れの一つである生成モデルの願望は、以下の２つによって、限界を克服することです。

（1）現実世界のモデルを学習し、エージェントが世界と実際に交流する前に世界モデルで計画できるようにする
（2）人の教示やラベリングをほとんどまたは全く必要とせずに入力の意味のある特徴を学習すること。

Since such features can be learned from large unlabeled datasets and are not necessarily task-specific, downstream solutions based on those features could potentially be more robust and more data efficient. In this paper we work towards this ultimate vision, in addition to intermediate applications, by aiming to improve upon the state-of-the-art of generative models.

このような機能は、ラベルのない大きなデータセットから学習でき、必ずしもタスク固有ではないため、これらの機能に基づいたダウンストリームソリューションは、より堅牢でデータ効率が向上する可能性があります。 

本稿では、この究極のビジョンに向けた、中間的なマイルストン、アプリケーションを加え、生成モデルのState-of-the-artを改善することを目指して、取り組みを行います。

Generative modeling is generally concerned with the extremely challenging task of modeling all dependencies within very high-dimensional input data, usually specified in the form of a full joint probability distribution.

生成モデルは、一般に、非常に高次元の入力データ内のすべての依存関係をモデリングするという非常に困難な作業となります。通常は、完全結合確率分布の形で指定されます。

Since such joint models potentially capture all patterns that are present in the data, the applications of accurate generative models are near endless. 

このような結合モデルは、データに存在するすべてのパターンを取得する可能性があるため、正確な生成モデルのアプリケーションはほぼ無限に近くなってしまいます。

Immediate applications are as diverse as speech synthesis, text analysis, semi-supervised learning and model-based control; see Section 4 for references.

すぐに使用できるアプリケーションは、音声合成、テキスト解析、半教師付き学習、モデルベースの制御など多岐にわたっています。参照のためにセクション4を見てください。

The discipline of generative modeling has experienced enormous leaps in capabilities in recent years, mostly with likelihood-based methods (Graves, 2013; Kingma and Welling, 2013, 2018; Dinh et al., 2014; van den Oord et al., 2016a) and generative adversarial networks (GANs) (Goodfellow et al., 2014) (see Section  4). Likelihood-based methods can be divided into three categories.

近年、生成モデルの訓練は大きな飛躍を遂げています。主に尤度ベースの方法（Graves、2013; Kingma and Welling、2013,2018; Dinh et al。、2014; van den Oord et al。、2016a）敵対的生成ネットワーク（Goodfellow et al。、2014） です。（セクション4を参照）。尤度ベースの方法は、3つのカテゴリに分けることができる。


### 1.自動回帰モデル（Hochreiter and Schmidhuber、1997; Graves、2013; van den Oordら、2016a、b; Van Den Oordら、2016）。

1. Autoregressive models (Hochreiter and Schmidhuber, 1997; Graves, 2013; van den Oord et al., 2016a, b; Van Den Oord et al., 2016). 

この手法は単純さがメリットですが、合成には限られた並列性が限られる欠点があります。合成の計算量がデータの次元に比例するためです。このことは大規模画像やビデオでは特に面倒です。

Those have the advantage of simplicity, but have as disadvantage that synthesis has limited parallelizability, since the computational length of synthesis is proportional to the dimensionality of the data; this is especially troublesome for large images or video.

Variational autoencoders (VAEs)  (Kingma and Welling, 2013, 2018), which optimize a lower bound on the log-likelihood of the data. Variational autoencoders have the advantage of parallelizability of training and synthesis, but can be comparatively challenging to optimize (Kingma et al., 2016).

データの対数尤度（log-likelihood）の下限を最適化するVariation autoencoders（VAE）（Kingma and Welling、2013、2018）バリエーションオートエンコーダーはトレーニングと合成の並列化の利点がありますが、最適化するのは比較的難しい。（Kingma et al。、2016）

Flow-based generative models, first described in NICE (Dinh et al., 2014) and extended in RealNVP (Dinh et al., 2016). We explain the key ideas behind this class of model in the following sections.


フローベースの生成モデル、最初にNICE（Dinh et al。、2014）、RealNVP（Dinh et al。、2016）で拡張されています。次のセクションでは、このクラスのモデルの背後にある重要なアイデアについて説明します。

Flow-based generative models have so far gained little attention in the research community compared to GANs (Goodfellow et al., 2014) and VAEs (Kingma and Welling, 2013). Some of the merits of flow-based generative models include:

フローベースの生成モデルは、GAN（Goodfellow et al。、2014）およびVAE（Kingma and Welling、2013）と比較して、研究コミュニティでこれまでのところほとんど注目されていない。 フローベースの生成モデルのメリットの一部は次のとおりです。

Exact latent-variable inference and log-likelihood evaluation. In VAEs, one is able to infer only approximately the value of the latent variables that correspond to a datapoint.

正確な潜在変数推論と対数尤度評価。VAEでは、データポイントに対応する潜在変数の値のおおよそを推論することができる。GANには潜在変数を推論するエンコーダが全くありません。可逆生成モデルでは近似なしで正確に行うことができる。 これは正確な推論につながるだけでなく、データの下限の代わりにデータの正確な対数尤度の最適化も可能にします。

GAN’s have no encoder at all to infer the latents. In reversible generative models, this can be done exactly without approximation. Not only does this lead to accurate inference, it also enables optimization of the exact log-likelihood of the data, instead of a lower bound of it.



Efficient inference and efficient synthesis. Autoregressive models, such as the PixelCNN (van den Oord et al., 2016b), are also reversible, however synthesis from such models is difficult to parallelize, and typically inefficient on parallel hardware. Flow-based generative models like Glow (and RealNVP) are efficient to parallelize for both inference and synthesis.

    Useful latent space for downstream tasks. The hidden layers of autoregressive models have unknown marginal distributions, making it much more difficult to perform valid manipulation of data. In GANs, datapoints can usually not be directly represented in a latent space, as they have no encoder and might not have full support over the data distribution.  (Grover et al., 2018). This is not the case for reversible generative models and VAEs, which allow for various applications such as interpolations between datapoints and meaningful modifications of existing datapoints.

    Significant potential for memory savings. Computing gradients in reversible neural networks requires an amount of memory that is constant instead of linear in their depth, as explained in the RevNet paper (Gomez et al., 2017).

In this paper we propose a new a generative flow coined Glow, with various new elements as described in Section 3. In Section 5, we compare our model quantitatively with previous flows, and in Section  6, we study the qualitative aspects of our model on high-resolution datasets.

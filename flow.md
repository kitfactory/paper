
# Variational Inference with Normalizing Flows  

## 概要 

近似的な事後分布の選択は、変分推論の中核問題の1つです。変分推論のほとんどの応用は、平均場または他の単純な構造近似に焦点を当て、効率的な推論を可能にするために、単純近似の群を使用する。この制限は、変分法を使用して行われた推論の質に重大な影響を与えます。我々は、柔軟で、任意に複雑でスケーラブルな近似事後分布を指定するための新しいアプローチを導入する。我々の近似は、正規化フローによって構築された分布であり、所望のレベルの複雑さが達成されるまで、一連の可逆的変換を適用することによって、単純な初期密度をより複雑なものに変換する。私たちは、フローを正規化するこのビューを使用して、有限かつ微小なフローのカテゴリを開発し、豊富な後方近似を構築するためのアプローチの統一されたビューを提供します。我々は、償却された変分法のスケーラビリティと組み合わされた真の事後条件とより良く一致する事後条件を有するという理論上の利点が、変分推論の性能および適用性を明確に改善することを実証する。

## 1. 導入

確率論的モデリングをますます拡大するデータセットのますます複雑な問題に拡張する手段として、変分推論に新たな関心が高まっています。



There has been a great deal of renewed interest in variational inference as a means of scaling probabilistic modeling to increasingly complex problems on increasingly larger data sets.
 > 


 > 


 


変分推論は現在、テキストの大規模な話題モデル（Hoffman et al。、2013）の中核に位置し、半教師付き分類（Kingma et al。、2014）の最先端技術を提供し、モデルを駆動する （Gregor et al。、2014、2015; Rezende et al。、2014; Kingma＆Welling、2014）、多くの物理的および化学的システムを理解するための標準的ツールとなっています。

Variational inference now lies at the core of large-scale topic models of text (Hoffman et al., 2013), provides the state-of-the-art in semi-supervised classification (Kingma et al., 2014), drives the models that currently produce the most realistic generative models of images (Gregor et al., 2014, 2015; Rezende et al., 2014; Kingma & Welling, 2014), and are a default tool for the understanding of many physical and chemical systems.


これらの成功および継続的な進歩にもかかわらず、統計的推論のためのデフォルト方法として、変分推論には、その力を制限し、より広い採用を妨げる様々な方法の多くの欠点があります。その制限の1つであり、事後分布の近似の選択です。本稿ではこの問題を取り上げます。


Despite these successes and ongoing advances, there are a number of disadvantages of variational methods that limit their power and hamper their wider adoption as a default method for statistical inference.
 > 
これらの成功および継続的な進歩にもかかわらず、統計的推論のためのデフォルト方法として、彼らの力を制限し、それらのより広い採用を妨げる様々な方法の多くの欠点がある。

It is one of these limitations, the choice of posterior approximation, that we address in this paper.
 > 



変分推論では、困難な事後分布を既知の確率分布のクラスで近似する必要があり、その上で真の事後確率に対する最善の近似を探索する。

Variational inference requires that intractable posterior distributions be approximated by a class of known probability distributions, over which we search for the best approximation to the true posterior.

使用される近似のクラスはしばしば限定されており、例えば、平均場近似は、真の事後分布に似た解が決してないことを意味する。

The class of approximations used is often limited, e.g., mean-field approximations, implying that no solution is ever able to resemble the true posterior distribution. 
 

これは、MCMCのような他の推論法とは異なり、漸近体制においてさえ、真の事後分布を回復することができないという点で、変分法に対する広く提起された反対である。
 
This is a widely raised objection to variational methods, in that unlike other inferential methods such as MCMC, even in the asymptotic regime we are unable recover the true posterior distribution.

より豊かでより忠実な後部近似がより良い性能をもたらすという多くの証拠がある。 例えば、平均場近似を利用するシグモイド・ビリーフネットワークと比較すると、深い自己回帰ネットワークは、パフォーマンスの明確な改善をもたらす自己回帰依存構造を持つ後方近似を使用します（Mnih＆Gregor、2014）。

There is much evidence that richer, more faithful posterior approximations do result in better performance. For example, when compared to sigmoid belief networks that make use of mean-field approximations, deep auto-regressive networks use a posterior approximation with an auto-regressive dependency structure that provides a clear improvement in performance (Mnih & Gregor, 2014). 

限られた事後近似の有害な影響を説明する大きな証拠もあります。 Turner＆Sahani（2011）は、よく経験される2つの問題の解説をしています。 第1は、事後分布の分散の過小評価の広く観察されている問題であり、選択された事後近似に基づいて予測が不十分で信頼性の低い決定をもたらす可能性がある。

There is also a large body of evidence that describes the detrimental effect of limited posterior approximations. Turner & Sahani (2011) provide an exposition of two commonly experienced problems. The first is the widely-observed problem of under-estimation of the variance of the posterior distribution, which can result in poor predictions and unreliable decisions based on the chosen posterior approximation.


第2に、後方近似の限られた容量は、任意のモデルパラメータのMAP推定値に偏りをもたらす可能性がある（これは、例えば、時系列モデルの場合である）。

 The second is that the limited capacity of the posterior approximation can also result in biases in the MAP estimates of any model parameters (and this is the case e.g., in time-series models).


近似的な事後分布の内のいくつかの基本的な依存形態を組み込んだ構造化平均場近似に基づいて、豊富な事後分布近似のための多数の提案が探求されてきた。 Jaakkola＆Jordan（1998）が開発したもののように、混合モデルとして近似的な事後確率を指定することも可能である。 Jordan et al。（1999）; Gershman et al。 （2012）。

A number of proposals for rich posterior approximations have been explored, typically based on structured mean-field approximations that incorporate some basic form of dependency within the approximate posterior. Another potentially powerful alternative would be to specify the approximate posterior as a mixture model, such as those developed by Jaakkola & Jordan (1998); Jordan et al.
(1999); Gershman et al. (2012).

しかし、混合アプローチは、変分推論の潜在的なスケーラビリティを制限する。なぜなら、パラメータ更新ごとに各混合成分に対する対数尤度およびその勾配の評価を必要とするからである。


 But the mixture approach limits the potential scalability of variational inference since it requires evaluation of the log-likelihood and its gradients for each mixture component per parameter update, which is typically computationally expensive.

本稿では、変分推論のための近似事後分布を指定するための新しい手法を提案する。

This paper presents a new approach for specifying approximate posterior distributions for variational inference. 

私たちは第二章で、償却された変分推論と効率的なモンテカルロ勾配推定に基づいた、グラフィカルなモデル推論の現行ベストプラクティスを評価することから始めます。それから、以下の貢献を行います。

We begin by reviewing the current best practice for inference in general directed graphical models, based on amortized variational inference and efficient Monte Carlo gradient estimation, in section 2. We then make the following contributions:

我々は、一連の可逆写像を通して確率密度を変換することによって複雑な分布を構築するためのツール、正規化フローを用いて近似事後分布の仕様を提案する（第3章）。

We propose the specification of approximate posterior distributions using normalizing flows, a tool for constructing complex distributions by transforming a probability density through a series of invertible mappings (sect. 3). 

正規化フローの推論は、厳密に変更された変分下限と、線形時間の複雑さを伴う項を追加するだけの追加の項を提供する（セクション4）。

Inference with normalizing flows provides a tighter, modified variational lower bound with additional terms that only add terms with linear time complexity (sect 4).

正規化フローは、変分推論の引用された限界の1つを克服する、漸近的な体系で真の事後分布を回復することができる事後分布近似クラスを指定することができるような、微小な流れが可能となることを示す。

We show that normalizing flows admit infinitesimal flows that allow us to specify a class of posterior approximations that in the asymptotic regime is able to recover the true posterior distribution, overcoming one oft-quoted limitation of variational inference.


特別なタイプの正規化フローの適用として、事後分布近似を改善するための関連するアプローチの統合ビューを提示する（第5章）。

We present a unified view of related approaches for improved posterior approximation as the application of special types of normalizing flows (sect 5).

我々は、一般的な正規化フローが、事後分布近似のための他の競合するアプローチよりも系統的に優れていることを実験的に示す。

We show experimentally that the use of general normalizing flows systematically outperforms other competing approaches for posterior approximation.

# 2.償却変分推定 Amortized Variational Inference

推論を実行するには、確率モデルの限界尤度を用いて推論することが必要です、モデル内の不足している変数、または潜在変数の周縁化を必要とする。

To perform inference it is sufficient to reason using the marginal likelihood of a probabilistic model, and requires the marginalization of any missing or latent variables in the model.

この統合は一般的に困難であり、代わりに限界尤度の下限を最適化します。 観測値x、積分しなければならない潜在変数 z、モデルパラメータ θを持つ一般的な確率モデルを考えてみよう。

This integration is typically intractable, and instead, we optimize a lower bound on the marginal likelihood.

この統合は一般的に困難であり、代わりに、限界尤度の下限を最適化します。

 Consider a general probabilistic model with observations x, latent variables z over which we must integrate, and model parameters θ.

観測値x、積分しなければならない潜在変数z、モデルパラメータθを持つ一般的な確率モデルを考えてみよう。

We introduce an approximate posterior distribution for the latent variables qϕ(z|x) and follow the variational principle Jordan et al. (1999) to obtain a bound on the marginal likelihood:

潜在変数qφ（z|x）の近似事後分布を、Jordanらの変分原理に従う （1999年）、限界尤度の限界を求める。

![1_3](img/flow_1.png)

最終的な方程式を得るためにジェンセンの不等式を使う。pθ（x | z）は尤度関数であり、p（z）は潜在変数に対する事前確率である。

where we used Jensen’s inequality to obtain the final equation, 
pθ(x|z) is a likelihood function and 
p(z) is a prior over the latent variables. 

この定式化をパラメータθに対する事後推論に容易に拡張することができるが、潜在変数のみの推論に焦点を当てる。

We can easily extend this formulation to posterior inference over the parameters 
θ, but we will focus on inference over the latent variables only.

この境界はしばしば負の自由エネルギーFまたは変分下限（ELBO）と呼ばれる。

 This bound is often referred to as the negative free energy F or as the evidence lower bound (ELBO).
 
これは2つの項から成り立ちます。最初のものは、近似的な事後分布と事前分布（正規化子として機能する）との間のKLダイバージェンスであり、2つ目は再構成誤差です。

It consists of two terms: the first is the KL divergence between the approximate posterior and the prior distribution (which acts as a regularizer), and the second is a reconstruction error. 

この下限（3）は、モデルのパラメータθとφの両方を最適化するための統一された目的関数と、変分近似をそれぞれ提供する。

This bound (3) provides a unified objective function for optimization of both the parameters θ and ϕ of the model and variational approximation,　respectively.

Current best practice in variational inference performs this optimization using mini-batches and stochastic gradient descent, which is what allows variational inference to be scaled to problems with very large data sets. There are two problems that must be addressed to successfully use the variational approach: 1) efficient computation of the derivatives of the expected log-likelihood


, and 2) choosing the richest, computationally-feasible approximate posterior distribution 
q
(
⋅
)
. 


第2の問題はこの論文の焦点です。 最初の問題に対処するために、モンテカルロ勾配推定と推論ネットワークの2つのツールを使用しています。これらを一緒に使用すると、償却された変分推論と呼ばれます。

The second problem is the focus of this paper. To address the first problem, we make use of two tools: Monte Carlo gradient estimation and inference networks, which when used together is what we refer to as amortized variational inference.







2.1. 確率的逆伝播


長年の変分推論の研究の大部分は、期待される対数尤度の勾配を計算する方法について行われてきた。

The bulk of research in variational inference over the years has been on ways in which to compute the gradient of the expected log-likelihood 

![flow_2_1](img/flow_2_1.png)

従来、局所的な変分法（Bishop、2006）に頼っていたのではないが、一般にMonte Carlo近似（解析的に知られていなければ、結合のKL項を含む）を用いてこのような期待を常に計算する。

Whereas we would have previously resorted to local variational methods (Bishop, 2006), in general we now always compute such expectations using Monte Carlo approximations (including the KL term in the bound, if it is not analytically known).


この形式は二重確率的推定（Titsias＆Lazaro-Gredilla、2014）と命名されている。ミニバッチからの1つの確率的な源を、モンテカルロ近似からもう一つの近似からもう一つの源を持つためである。

 This forms what has been aptly named doubly-stochastic estimation (Titsias & Lazaro-Gredilla, 2014), since we have one source of stochasticity from the mini-batch and a second from the Monte Carlo approximation of the expectation.


連続潜在変数を持つモデルに焦点を当てる。我々が取るアプローチは、確率的逆伝播と呼ばれるモンテカルロ近似（Rezende et al。、2014）と組み合わされた、期待の非中心的再パラメータ化を用いて必要な勾配を計算する（Papaspiliopoulos et al。、2003; Williams、1992）。

We focus on models with continuous latent variables, and the approach we take computes the required gradients using a non-centered reparameterization of the expectation (Papaspiliopoulos et al., 2003; Williams, 1992), combined with Monte Carlo approximation — referred to as stochastic backpropagation (Rezende et al., 2014). 

このアプローチは、確率的勾配変分ベイズ（SGVB）（Kingma＆Welling、2014）またはアファイン変分推論（Challis＆Barber、2012）とも呼ばれています。

 This approach has also been referred to or as stochastic gradient variational Bayes (SGVB) (Kingma & Welling, 2014) or as affine variational inference (Challis & Barber, 2012).

 確率的な逆伝播には2つのステップが含まれます。






### 再パラメータ化。

 既知の分布と微分可能な変換（位置 - スケール変換や累積分布関数など）などにより潜在変数を再パラメータ化します。

 Reparameterization. We reparameterize the latent variable in terms of a known base distribution and a differentiable transformation (such as a location-scale transformation or cumulative distribution function).

例えば、qφ（z）がガウス分布N（z |μ、σ2）、φ= {μ、σ2}である場合、標準の正規分布を基底分布とする位置 - スケール変換は、次のようにzを再パラメータ化することが可能です。 

For example, if qϕ(z) is a Gaussian distribution N(z|μ,σ2), with ϕ={μ,σ2}, then the location-scale transformation using the standard Normal as a base distribution allows us to reparameterize z. as:


![](img/flow_2_2.png)


### モンテカルロを使ったバックプロパゲーション
ベースとなる分布から引き出す、モンテカルロ近似を用いて、変分分布のパラメータφに関するを差分を取る（逆伝播する）ことができます。

Backpropagation with Monte Carlo. 
We can now differentiate (backpropagation) with reference to the parameters ϕ of the variational distribution using a Monte Carlo approximation with draws from the base distribution:

 ![](img/flow_2_3.png)


モンテカルロ制御変量（MCCV）推定量に基づく多くの汎用目的のアプローチは、確率的な逆伝播の代替として存在し、連続的または離散的な潜在変数を伴う勾配計算を可能にするWilliams（1992）; Mnih＆Gregor（2014）; Ranganath et al。 （2013）。 Wingate＆Weber（2013）。

A number of general purpose approaches based on Monte Carlo control variate (MCCV) estimators exist as an alternative to stochastic backpropagation, and allow for gradient computation with latent variables that may be continuous or discrete Williams (1992); Mnih & Gregor (2014); Ranganath et al. (2013); Wingate & Weber (2013). 

確率的逆伝播の重要な利点は、連続潜在変数を持つモデルの場合、競合する推定子の中で最も小さな分散を持つことです。

An important advantage of stochastic backpropagation is that, for models with continuous latent variables, it has the lowest variance among competing estimators.

## 2.2.推論ネットワーク

第2の重要なプラクティスは、近似的な事後分布
qφ（・）は、認識モデルまたは推論ネットワークを用いて表される。 （Rezende et al。、2014; Dayan、2000; Gershman＆Goodman、2014; Kingma＆Welling、2014）。

A second important practice is that the approximate posterior distribution 
qϕ(⋅) is represented using a recognition model or inference network (Rezende et al., 2014; Dayan, 2000; Gershman & Goodman, 2014; Kingma & Welling, 2014). 


推論ネットワークは、観測から潜在変数への逆マップを学習するモデルです。 推論ネットワークを使用して、データポイントごとの変分パラメータを計算する必要はなくなりますが、訓練とテストの両方で推論に有効なグローバル変分パラメータφを計算することがあります。

An inference network is a model that learns an inverse map from observations to latent variables. Using an inference network, we avoid the need to compute per data point variational parameters, but can instead compute a set of global variational parameters ϕ valid for inference at both training and test time.


This allows us to amortize the cost of inference by generalizing between the posterior estimates for all latent variables through the parameters of the inference network. The simplest inference models that we can use are diagonal Gaussian densities.

これにより、推論ネットワークのパラメータを介してすべての潜在変数の事後推定値を一般化することにより、推論のコストを償却することができます。 我々が使用できる最も単純な推論モデルは、対角ガウス密度である。

 qϕ(z|x)=N(z|μϕ(%$x$),diag(σ2ϕ(x))),


the mean function μϕ(x) and the standard-deviation function σϕ(x) are specified using deep neural networks.

ディープニューラルネットワークを用いて平均関数μφ（x）と標準偏差関数σφ（x）を特定する。





---
